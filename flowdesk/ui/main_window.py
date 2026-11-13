"""The workspace window: ribbon, group pane and the sample / population tree."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QSettings, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDrag, QFont, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..core.gates import BooleanGate
from ..core.model import Group, Population, Sample, StatNode, Workspace
from ..core.serialization import EXTENSION, data_dir_for, load_workspace, save_workspace
from ..core.stats import compute_statistic, format_value
from .compensation import CompensationWindow, CompensationWizard
from .controller import POPULATION_MIME, WorkspaceController
from .dialogs import AddStatisticDialog, BooleanGateDialog, ColumnsDialog, DerivedParameterDialog, KeywordsDialog, PreferencesDialog, confirm, show_about
from .graph_window import GraphWindow
from .group_dialog import GroupBooleanGateDialog, GroupDialog, group_font, role_label, set_group_live
from .icons import gate_icon_name, icon, pixmap
from .layout_editor import LayoutEditorWindow
from .ribbon import Ribbon
from .table_editor import TableEditorWindow

SAMPLES_MIME = "application/x-flowdesk-samples"

# Worker threads still running when the user chose "Quit anyway". Qt aborts the
# process when a running QThread is destroyed, so they are detached from the
# closing window and kept referenced here; flowdesk.app.main leaves through
# os._exit while any of them still runs.
_ABANDONED_THREADS: list = []


def abandoned_threads() -> list:
    """The detached worker threads that are still running."""
    out = []
    for t in _ABANDONED_THREADS:
        try:
            if t.isRunning():
                out.append(t)
        except RuntimeError:  # C++ object already gone
            pass
    return out


def _abandon_threads(threads) -> None:
    for t in threads:
        _ABANDONED_THREADS.append(t)  # keep the wrapper (and so the C++ thread) alive
        t.setParent(None)


def _workspace_load_warnings(ws) -> list[str]:
    """What load_workspace could not restore (computed parameters whose data
    folder or files are missing ...), one message per problem."""
    found = [str(w) for w in (getattr(ws, "load_warnings", None) or [])]
    for s in getattr(ws, "samples", []):  # samples add more once their events are read
        for w in getattr(s, "load_warnings", None) or []:
            msg = f"{s.name}: {w}"
            if msg not in found:
                found.append(msg)
    return found

from .style import modifier_html as _modifier_html  # noqa: E402

_MOD = _modifier_html()

QUICK_START_HTML = f"""
<h2>{APP_NAME} Quick Start</h2>
<p>If you have analysed cytometry data before, you already know this workflow.</p>
<h3>1. Load samples</h3>
<p><b>Add Samples</b> / <b>Add Folder</b>, or drop FCS files onto the window. Files whose name
contains <i>comp</i> or <i>unstained</i> land in the <span style="color:#bc1900"><b>Compensation</b></span> group automatically.
The spillover matrix stored in each file is applied (parameters named <b>Comp-&hellip;</b>).</p>
<h3>2. Gate</h3>
<p>Double-click a sample to open its <b>graph window</b>. Pick parameters with the axis buttons below and left of the plot
(or <b>{_MOD} + arrow keys</b>), choose the plot type, and draw with the tools:
Rectangle (R), Polygon (P), Ellipse (E), Quadrant (Q), Range (H), Bisector (B), Freehand (F), Zoom (Z), Select (V).
Name the population when asked. <b>Double-click inside a gate</b> to drill down. Drag gates and their handles to adjust;
<b>Delete</b> removes, <b>{_MOD}+Z</b> undoes. The <b>T</b> buttons customise the axis scale (biexponential width basis etc.).</p>
<h3>3. Apply to the group</h3>
<p>Drag a population from the sample list onto a group in the group pane (or <b>Apply to Group</b>) to copy it to every sample
of that group. Gates drawn in a group's own graph window (double-click the group) apply to all members at once; new samples that
join the group receive the gates too. With a group selected, its <b>bold row</b> heads the sample list: below it is the group's
gating tree with the mean over its samples; rename (F2), delete, drag or double-click those rows to act on every member.
FCS files dropped onto a group join it.</p>
<p>Dragging a gate onto another population of the same sample <b>moves</b> it there; hold <b>Alt/Option</b> to copy instead.
Dragging a sample onto another copies its whole gating tree; a statistic row dropped on a population adds that statistic.</p>
<h3>4. Statistics, Boolean gates, keywords</h3>
<p><b>Add Statistic&hellip;</b> adds Median / Mean / CV / percentile rows under populations; <b>AND / OR / NOT</b> create Boolean gates.
<b>Edit Columns&hellip;</b> shows keywords such as $DATE or $CYT as columns; <b>Create Group&hellip;</b> can group samples automatically by keyword.</p>
<h3>5. Layouts and tables</h3>
<p>Open the <b>Layout Editor</b> and drag populations onto the page (drop one onto an existing plot to overlay). Export PNG / PDF / SVG,
or a <b>batch PDF</b> with one page per sample. The <b>Table Editor</b> collects statistics for every sample of a group and exports CSV.</p>
<h3>6. Compensation</h3>
<p><b>Compensation</b> shows and edits matrices and applies them to groups. The <b>Compensation Wizard</b> computes a matrix from
the single-stain controls of the Compensation group.</p>
<h3>Files</h3>
<p>Workspaces are saved as <b>.fdw</b>. FlowJo&trade; workspaces (<b>.wsp</b>) can be opened directly and exported again.
Populations can be exported as FCS or CSV; <b>Concatenate Samples&hellip;</b> joins several samples into one file with a SampleID parameter.</p>
"""

ROLE_SAMPLE = Qt.ItemDataRole.UserRole + 1
ROLE_POP = Qt.ItemDataRole.UserRole + 2
ROLE_STAT = Qt.ItemDataRole.UserRole + 3
ROLE_GROUP = Qt.ItemDataRole.UserRole + 4  # group row / group-template population rows

PLACEHOLDER = "\u2026"  # shown instead of numbers a sample has not computed yet


def _sample_suffixes() -> tuple[str, ...]:
    """File suffixes that are added as samples (the controller may list more than FCS)."""
    from . import controller as controller_module

    suffixes = getattr(controller_module, "SAMPLE_SUFFIXES", None) or (".fcs",)
    return tuple(s.lower() for s in suffixes)


def _parse_population_payload(md) -> list[dict]:
    """The entries of a POPULATION_MIME payload (a dict or a list of dicts)."""
    try:
        payload = json.loads(bytes(md.data(POPULATION_MIME).data()).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return []
    if isinstance(payload, dict):
        payload = [payload]
    return [e for e in payload if isinstance(e, dict)] if isinstance(payload, list) else []


def _stat_spec(st: StatNode, path: str) -> dict:
    return {"path": path, "statistic": st.statistic, "param": st.param, "percentile": st.percentile}


def _mean_sd(values) -> tuple[float, float, int]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not vals:
        return float("nan"), float("nan"), 0
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mean, sd, len(vals)


def group_population_summary(members: list[Sample], path: str) -> dict:
    """Mean / SD over the member samples of a population's % of parent and
    count (the Statistic and #Cells of a group-template row)."""
    pops = [p for p in (s.root.find(path) for s in members) if p is not None and p.parent is not None]
    freq = _mean_sd(p.freq_of_parent for p in pops)
    count = _mean_sd(p.count for p in pops)
    return {"freq": freq[0], "freq_sd": freq[1], "count": count[0], "count_sd": count[1], "n": freq[2], "found": len(pops)}


def group_statistic_summary(members: list[Sample], path: str, st: StatNode) -> tuple[float, float, int]:
    """Mean, SD and number of members of a statistic row of the group template."""
    values = []
    for s in members:
        pop = s.root.find(path)
        if pop is not None:
            values.append(compute_statistic(st.statistic, pop, st.param, st.percentile))
    return _mean_sd(values)


@dataclass
class TreeDrop:
    """What a drop onto the sample tree carried and where it landed."""

    target_pop: Population  # a sample population or a group-template population
    target_sample: Sample | None = None
    target_group: Group | None = None
    pops: list = field(default_factory=list)  # [(Sample, Population)], sample roots included
    group_pops: list = field(default_factory=list)  # [(Group, template Population)]
    stats: list = field(default_factory=list)  # [(Sample | Group, Population, stat dict)]
    copy: bool = False  # Alt / Option held: copy instead of move


# ------------------------------------------------------------------ group pane
class GroupPane(QTableWidget):
    openGroup = Signal(object)
    populationsDropped = Signal(object, list)  # group, [(sample | group, population)]
    samplesDropped = Signal(object, list)  # group, [Sample]
    statsDropped = Signal(object, list)  # group, [(population, stat dict)]
    filesDropped = Signal(object, list)  # group (None: not on a group row), [path]
    contextMenu = Signal(object, QPoint)
    groupRenamed = Signal(object, str)  # group, new name (F2 in the name column)
    deleteRequested = Signal(object)  # group (Delete / Backspace)

    def __init__(self, controller: WorkspaceController, parent=None):
        super().__init__(0, 3, parent)
        self.controller = controller
        self.setHorizontalHeaderLabels(["Group", "Samples", "Role / criteria"])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(0, 220)
        self.setColumnWidth(1, 70)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # F2 renames (double-click opens the group graph, as in FlowJo)
        self.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.itemSelectionChanged.connect(self._selection_changed)
        self.itemChanged.connect(self._item_changed)
        self.itemDoubleClicked.connect(lambda item: self.openGroup.emit(self.group_at(item.row())))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self.contextMenu.emit(self.group_at(self.rowAt(pos.y())), self.viewport().mapToGlobal(pos)))

    def group_at(self, row: int) -> Group | None:
        if row < 0 or row >= self.rowCount():
            return None
        return self.item(row, 0).data(Qt.ItemDataRole.UserRole)

    def rebuild(self):
        ws = self.controller.workspace
        self.blockSignals(True)
        # take the old items out first: Python owns them then, so a wrapper still
        # referenced (the item being renamed) never points at a freed item
        taken = [self.takeItem(r, c) for r in range(self.rowCount()) for c in range(self.columnCount())]
        self.setRowCount(len(ws.groups))
        not_editable = ~Qt.ItemFlag.ItemIsEditable
        for row, g in enumerate(ws.groups):
            name = QTableWidgetItem(g.name)
            name.setData(Qt.ItemDataRole.UserRole, g)
            name.setIcon(icon("group"))
            name.setFont(group_font(g))
            name.setForeground(QBrush(QColor(g.color)))
            if g.role == "all":
                name.setFlags(name.flags() & not_editable)
            else:
                name.setToolTip(f"{g.name}: F2 renames, Delete removes the group")
            count = QTableWidgetItem(str(len(ws.members(g))))
            count.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            count.setFlags(count.flags() & not_editable)
            if g.role == "all":
                role = "all samples"
            else:
                if g.criteria:
                    role = " ".join(f"{c.connector.lower() if i else ''} {c.keyword} {c.function.lower()} “{c.value}”".strip() for i, c in enumerate(g.criteria))
                    if not g.live:
                        role += " (frozen)"
                    if g.sample_ids:
                        role += f" (+{len(g.sample_ids)} manual)"
                else:
                    role = f"{len(g.sample_ids)} samples added manually"
                if g.role in ("test", "control", "compensation"):
                    role = f"{role_label(g.role)} · {role}"
            role_item = QTableWidgetItem(role)
            role_item.setFlags(role_item.flags() & not_editable)
            self.setItem(row, 0, name)
            self.setItem(row, 1, count)
            self.setItem(row, 2, role_item)
            if g is self.controller.current_group:
                self.selectRow(row)
        self.blockSignals(False)
        del taken

    def _selection_changed(self):
        rows = {i.row() for i in self.selectedItems()}
        if rows:
            g = self.group_at(min(rows))
            if g is not None:
                self.controller.set_current_group(g)

    def current_group_row(self) -> Group | None:
        rows = {i.row() for i in self.selectedItems()}
        return self.group_at(min(rows)) if rows else None

    def _item_changed(self, item: QTableWidgetItem):
        if self.signalsBlocked() or item.column() != 0:
            return
        g = item.data(Qt.ItemDataRole.UserRole)
        if g is None:
            return
        name = item.text().strip()
        if name and name != g.name:
            self.groupRenamed.emit(g, name)
        elif name != g.name:
            self.blockSignals(True)
            item.setText(g.name)
            self.blockSignals(False)

    def rename_current(self) -> bool:
        """Start editing the current group's name in place (F2)."""
        g = self.current_group_row()
        if g is None or g.role == "all":
            return False
        row = next((r for r in range(self.rowCount()) if self.group_at(r) is g), -1)
        if row < 0:
            return False
        self.setCurrentCell(row, 0)
        self.editItem(self.item(row, 0))
        return True

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.state() != QAbstractItemView.State.EditingState:
            g = self.current_group_row()
            if g is not None and g.role != "all":
                self.deleteRequested.emit(g)
                event.accept()
                return
        super().keyPressEvent(event)

    # drag & drop
    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasFormat(POPULATION_MIME) or md.hasFormat(SAMPLES_MIME) or md.hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        row = self.rowAt(event.position().toPoint().y())
        g = self.group_at(row)
        md = event.mimeData()
        if md.hasUrls():  # files: join the group they land on (or just the workspace)
            if g is not None:
                self.selectRow(row)
            event.acceptProposedAction()
            return
        if g is None:
            event.ignore()
            return
        if md.hasFormat(SAMPLES_MIME) and not md.hasFormat(POPULATION_MIME) and g.role == "all":
            event.ignore()
            return
        self.selectRow(row)
        event.acceptProposedAction()

    def dropEvent(self, event):
        row = self.rowAt(event.position().toPoint().y())
        g = self.group_at(row)
        md = event.mimeData()
        if md.hasUrls():
            paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
            if paths:
                self.filesDropped.emit(g, paths)
            event.acceptProposedAction()
            return
        if g is None:
            event.ignore()
            return
        ws = self.controller.workspace
        handled = False
        if md.hasFormat(SAMPLES_MIME) and g.role != "all":
            ids = json.loads(bytes(md.data(SAMPLES_MIME).data()).decode("utf-8"))
            samples = [s for s in (ws.sample_by_id(int(i)) for i in ids) if s is not None]
            if samples:
                self.samplesDropped.emit(g, samples)
            handled = True
        if md.hasFormat(POPULATION_MIME):
            items, stats = [], []
            for entry in _parse_population_payload(md):
                owner = None
                if entry.get("group_id") is not None:
                    owner = next((x for x in ws.groups if x.id == entry.get("group_id")), None)
                    root = owner.root if owner is not None else None
                else:
                    try:
                        owner = ws.sample_by_id(int(entry.get("sample_id", -1)))
                    except (TypeError, ValueError):
                        owner = None
                    root = owner.root if owner is not None else None
                if root is None:
                    continue
                for path in entry.get("paths", []):
                    pop = root.find(path)
                    if pop is not None and pop.parent is not None:
                        items.append((owner, pop))
                for spec in entry.get("stats", []) or []:
                    pop = root.find(spec.get("path", ""))
                    if pop is not None:
                        stats.append((pop, spec))
            if items:
                self.populationsDropped.emit(g, items)
            if stats:
                self.statsDropped.emit(g, stats)
            handled = True
        if handled:
            event.acceptProposedAction()
        else:
            event.ignore()


# ----------------------------------------------------------------- sample tree
class SampleTree(QTreeWidget):
    openRequested = Signal(object, object)  # sample, population
    groupOpenRequested = Signal(object, object)  # group, template population (None: the group row)
    filesDropped = Signal(list)
    filesDroppedOnGroup = Signal(object, list)  # group, [path]
    treeDropped = Signal(object)  # TreeDrop
    contextMenu = Signal(object, QPoint)  # item, global pos
    renamed = Signal(object, str)  # (sample|population), new name
    groupItemRenamed = Signal(object, object, str)  # group, template population (root: the group), new name
    deleteRequested = Signal()  # Delete / Backspace on the selection
    rebuilt = Signal()  # after rebuild(): extensions re-apply filters, sort indicator, display names

    # True: rows of samples whose events are not loaded show "…" instead of
    # loading the file for the counts (a background loader fills them in and
    # rebuilds the tree)
    defer_unloaded = False

    def __init__(self, controller: WorkspaceController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.keyword_columns: list[str] = []
        self.setColumnCount(3)
        self.setHeaderLabels(["Name", "Statistic", "#Cells"])
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setStretchLastSection(False)
        self.header().setMinimumSectionSize(60)
        self.setColumnWidth(1, 150)
        self.setColumnWidth(2, 100)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        self.setIconSize(QSize(18, 18))
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: self.contextMenu.emit(self.itemAt(pos), self.viewport().mapToGlobal(pos)))
        self.itemDoubleClicked.connect(self._double_clicked)
        self.itemSelectionChanged.connect(self._selection_changed)
        self.itemExpanded.connect(lambda item: self._set_expanded(item, True))
        self.itemCollapsed.connect(lambda item: self._set_expanded(item, False))
        self.itemChanged.connect(self._item_changed)
        self._building = False

    def _clear_items(self):
        """Remove every row. clear() frees the items behind the back of Python
        wrappers that are still referenced (a drop target, a renamed row ...),
        which later crashes PySide; taking them out hands them to Python, which
        frees them once nothing refers to them any more."""
        taken = self.invisibleRootItem().takeChildren()
        del taken

    def _computable(self, sample: Sample) -> bool:
        return not self.defer_unloaded or sample.loaded

    # ---- building
    def rebuild(self):
        self._building = True
        selected = {(s.id, p.path) for s, p in self.controller.selected}
        selected_groups = {(g.id, p.path) for g, p in self.selected_group_rows()}
        self._clear_items()
        ws = self.controller.workspace
        group = self.controller.current_group
        self.setColumnCount(3 + len(self.keyword_columns))
        self.setHeaderLabels(["Name", "Statistic", "#Cells"] + list(self.keyword_columns))
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, self.columnCount()):
            self.header().setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            if self.columnWidth(col) < 60:
                self.setColumnWidth(col, 120)
        decimals = int(self.controller.preferences.get("freq_decimals"))
        members = ws.members(group)
        if group.role != "all":
            self._add_group_row(group, members, decimals)
        for sample in members:
            computable = self._computable(sample)
            events = f"{sample.n_events:,}" if (computable or sample.n_events) else PLACEHOLDER
            item = QTreeWidgetItem([sample.name, "", events])
            item.setIcon(0, icon("sample" if sample.load_error is None else "sample-missing"))
            item.setData(0, ROLE_SAMPLE, sample)
            item.setData(0, ROLE_POP, sample.root)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled)
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            comp = sample.comp.name if (sample.comp is not None and sample.comp_enabled and not sample.comp.is_identity) else ""
            if comp:
                item.setText(1, comp)
                item.setForeground(1, QBrush(QColor("#6b7280")))
            if sample.load_error:
                item.setToolTip(0, f"File not found: {sample.path}")
                item.setForeground(0, QBrush(QColor("#b91c1c")))
            for k, key in enumerate(self.keyword_columns):
                item.setText(3 + k, sample.keywords.get(key, ""))
            self.addTopLevelItem(item)
            self._add_children(item, sample, sample.root, decimals, computable)
            item.setExpanded(sample.root.expanded)
            if (sample.id, "") in selected:
                item.setSelected(True)
        self._restore_selection(selected, selected_groups)
        self._building = False
        self._selection_changed()
        self.rebuilt.emit()

    def _add_children(self, parent_item: QTreeWidgetItem, sample: Sample, pop: Population, decimals: int, computable: bool = True):
        for st in pop.stats:
            text = format_value(st.statistic, compute_statistic(st.statistic, pop, st.param, st.percentile), decimals) if computable else PLACEHOLDER
            row = QTreeWidgetItem([st.label, text, ""])
            row.setIcon(0, icon("statistic"))
            row.setData(0, ROLE_SAMPLE, sample)
            row.setData(0, ROLE_POP, pop)
            row.setData(0, ROLE_STAT, st)
            row.setForeground(0, QBrush(QColor("#374151")))
            row.setFlags((row.flags() | Qt.ItemFlag.ItemIsDragEnabled) & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsDropEnabled)
            parent_item.addChild(row)
        for child in pop.children:
            if computable:
                freq = child.freq_of_parent
                row = QTreeWidgetItem([child.name, format_value("Freq. of Parent", freq, decimals) + " %" if freq == freq else "", f"{child.count:,}"])
            else:
                row = QTreeWidgetItem([child.name, PLACEHOLDER, PLACEHOLDER])
            row.setIcon(0, icon(gate_icon_name(child.gate)))
            row.setData(0, ROLE_SAMPLE, sample)
            row.setData(0, ROLE_POP, child)
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled)
            row.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if child.gate is not None and child.gate.kind == "boolean":
                row.setToolTip(0, child.gate.describe())
            elif child.gate is not None:
                row.setToolTip(0, f"{child.gate.kind} gate on {', '.join(child.gate.dims)}")
            parent_item.addChild(row)
            self._add_children(row, sample, child, decimals, computable)
            row.setExpanded(child.expanded)

    def _add_group_row(self, group: Group, members: list[Sample], decimals: int):
        """The bold group row above the members; its children are the group's
        gating template (group.root) with the mean over the members."""
        n = len(members)
        item = QTreeWidgetItem([group.name, f"{n} sample{'s' if n != 1 else ''}", ""])
        item.setIcon(0, icon("group"))
        item.setData(0, ROLE_GROUP, group)
        item.setData(0, ROLE_POP, group.root)
        item.setFont(0, group_font(group, item.font(0), force_bold=True))
        item.setForeground(0, QBrush(QColor(group.color)))
        item.setForeground(1, QBrush(QColor("#6b7280")))
        item.setToolTip(0, f"Group “{group.name}” ({role_label(group.role)}): gates and statistics below apply to every member sample; the numbers are the mean over the members")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled)
        self.addTopLevelItem(item)
        ready = [s for s in members if self._computable(s)]
        self._add_group_children(item, group, group.root, ready, len(ready) < len(members), decimals)
        item.setExpanded(getattr(group, "expanded", True))

    def _add_group_children(self, parent_item: QTreeWidgetItem, group: Group, tpop: Population, members: list[Sample], partial: bool, decimals: int):
        grey = QBrush(QColor("#6b7280"))
        for st in tpop.stats:
            mean, sd, n = group_statistic_summary(members, tpop.path, st)
            text = format_value(st.statistic, mean, decimals) if n else (PLACEHOLDER if partial else "")
            row = QTreeWidgetItem([st.label, text, ""])
            row.setIcon(0, icon("statistic"))
            row.setData(0, ROLE_GROUP, group)
            row.setData(0, ROLE_POP, tpop)
            row.setData(0, ROLE_STAT, st)
            row.setForeground(0, QBrush(QColor("#374151")))
            row.setForeground(1, grey)
            if n:
                row.setToolTip(1, f"Mean over {n} sample(s): {format_value(st.statistic, mean, decimals)} (SD {format_value(st.statistic, sd, decimals)})")
            row.setFlags((row.flags() | Qt.ItemFlag.ItemIsDragEnabled) & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsDropEnabled)
            parent_item.addChild(row)
        for child in tpop.children:
            summary = group_population_summary(members, child.path)
            if summary["n"]:
                freq = format_value("Freq. of Parent", summary["freq"], decimals) + " %"
                count = f"{summary['count']:,.0f}"
            else:
                freq = PLACEHOLDER if partial else ""
                count = PLACEHOLDER if partial else ""
            row = QTreeWidgetItem([child.name, freq, count])
            row.setIcon(0, icon(gate_icon_name(child.gate)))
            row.setData(0, ROLE_GROUP, group)
            row.setData(0, ROLE_POP, child)
            row.setForeground(1, grey)
            row.setForeground(2, grey)
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled)
            row.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tip = f"Group gate: applies to every sample of “{group.name}”"
            if child.gate is not None and child.gate.kind == "boolean":
                tip += f"\n{child.gate.describe()}"
            elif child.gate is not None:
                tip += f"\n{child.gate.kind} gate on {', '.join(child.gate.dims)}"
            row.setToolTip(0, tip)
            if summary["n"]:
                tail = f" ({summary['found'] - summary['n']} without events)" if summary["found"] > summary["n"] else ""
                row.setToolTip(1, f"Mean % of parent over {summary['n']} sample(s): {freq} (SD {format_value('Freq. of Parent', summary['freq_sd'], decimals)} %){tail}")
                row.setToolTip(2, f"Mean count over {summary['n']} sample(s) (SD {summary['count_sd']:,.0f})")
            parent_item.addChild(row)
            self._add_group_children(row, group, child, members, partial, decimals)
            row.setExpanded(child.expanded)

    def _restore_selection(self, selected: set, selected_groups: set | None = None):
        if not selected and not selected_groups:
            return
        it = QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            s = item.data(0, ROLE_SAMPLE)
            g = item.data(0, ROLE_GROUP)
            p = item.data(0, ROLE_POP)
            if p is not None and item.data(0, ROLE_STAT) is None:
                if s is not None and (s.id, p.path) in selected:
                    item.setSelected(True)
                elif g is not None and selected_groups and (g.id, p.path) in selected_groups:
                    item.setSelected(True)
            it += 1

    def _set_expanded(self, item: QTreeWidgetItem, on: bool):
        if self._building:
            return
        pop = item.data(0, ROLE_POP)
        if pop is not None and item.data(0, ROLE_STAT) is None:
            g = item.data(0, ROLE_GROUP)
            if g is not None and pop is g.root:
                g.expanded = on
            else:
                pop.expanded = on

    def _item_changed(self, item: QTreeWidgetItem, column: int):
        if self._building or column != 0:
            return
        pop = item.data(0, ROLE_POP)
        if item.data(0, ROLE_STAT) is not None or pop is None:
            return
        new_name = item.text(0).strip()
        group = item.data(0, ROLE_GROUP)
        if group is not None:
            current = group.name if pop.parent is None else pop.name
            if new_name and new_name != current:
                self.groupItemRenamed.emit(group, pop, new_name)
            elif not new_name:
                item.setText(0, current)
            return
        sample = item.data(0, ROLE_SAMPLE)
        target = sample if pop.parent is None else pop
        current = sample.name if pop.parent is None else pop.name
        if new_name and new_name != current:
            self.renamed.emit(target, new_name)
        elif not new_name:
            item.setText(0, current)

    # ---- selection / activation
    def selected_rows(self) -> list[tuple[Sample, Population]]:
        rows = []
        for item in self.selectedItems():
            s = item.data(0, ROLE_SAMPLE)
            p = item.data(0, ROLE_POP)
            if s is not None and p is not None and item.data(0, ROLE_STAT) is None:
                rows.append((s, p))
        return rows

    def selected_group_rows(self) -> list[tuple[Group, Population]]:
        """Selected group row (group, group.root) and group-template population rows."""
        rows = []
        for item in self.selectedItems():
            g = item.data(0, ROLE_GROUP)
            p = item.data(0, ROLE_POP)
            if g is not None and p is not None and item.data(0, ROLE_STAT) is None:
                rows.append((g, p))
        return rows

    def selected_stats(self) -> list[tuple[Population, StatNode]]:
        """Selected statistic rows of samples (not of the group template)."""
        out = []
        for item in self.selectedItems():
            st = item.data(0, ROLE_STAT)
            if st is not None and item.data(0, ROLE_GROUP) is None:
                out.append((item.data(0, ROLE_POP), st))
        return out

    def selected_group_stats(self) -> list[tuple[Group, Population, StatNode]]:
        out = []
        for item in self.selectedItems():
            st = item.data(0, ROLE_STAT)
            g = item.data(0, ROLE_GROUP)
            if st is not None and g is not None:
                out.append((g, item.data(0, ROLE_POP), st))
        return out

    def _selection_changed(self):
        if self._building:
            return
        groups = list(dict.fromkeys(g for g, _ in self.selected_group_rows()))
        self.controller.set_selection(self.selected_rows(), groups)

    def _double_clicked(self, item: QTreeWidgetItem, column: int):
        g = item.data(0, ROLE_GROUP)
        p = item.data(0, ROLE_POP)
        if g is not None and p is not None and item.data(0, ROLE_STAT) is None:
            self.groupOpenRequested.emit(g, p if p.parent is not None else None)
            return
        s = item.data(0, ROLE_SAMPLE)
        if s is not None and p is not None and item.data(0, ROLE_STAT) is None:
            self.openRequested.emit(s, p)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.state() != QAbstractItemView.State.EditingState:
            grows = self.selected_group_rows()
            rows = self.selected_rows()
            if rows:
                self.openRequested.emit(*rows[0])
                return
            if grows:
                g, p = grows[0]
                self.groupOpenRequested.emit(g, p if p.parent is not None else None)
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.state() != QAbstractItemView.State.EditingState:
            # reached only when the window's Delete action did not take the key
            # (Backspace on macOS, or an action that was disabled)
            if self.selectedItems():
                self.deleteRequested.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def expand_all_items(self, expand: bool):
        it = QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            pop = item.data(0, ROLE_POP)
            if pop is not None and item.data(0, ROLE_STAT) is None:
                g = item.data(0, ROLE_GROUP)
                if g is not None and pop is g.root:
                    g.expanded = expand
                else:
                    pop.expanded = expand
            it += 1
        self.expandAll() if expand else self.collapseAll()

    # ---- drag & drop
    def mimeTypes(self):
        return [POPULATION_MIME, SAMPLES_MIME, "text/plain"]

    def mimeData(self, items):
        """POPULATION_MIME: one entry per sample {"sample_id", "paths"} (a sample
        row has path "", dropping it copies the whole tree), plus "stats"
        [{path, statistic, param, percentile}] for dragged statistic rows.
        Group-template rows give {"group_id", "sample_id" (first member, so
        layout / table editors can show them), "paths", "stats"}."""
        ws = self.controller.workspace
        entries: dict[tuple, dict] = {}
        texts = []
        sample_ids: list[int] = []

        def entry_for(key, base):
            if key not in entries:
                entries[key] = base
            return entries[key]

        for item in items:
            s = item.data(0, ROLE_SAMPLE)
            g = item.data(0, ROLE_GROUP)
            p = item.data(0, ROLE_POP)
            st = item.data(0, ROLE_STAT)
            if p is None:
                continue
            if g is not None:
                members = ws.members(g)
                e = entry_for(("g", g.id), {"group_id": g.id, "sample_id": members[0].id if members else None, "paths": []})
                label = g.name
            elif s is not None:
                e = entry_for(("s", s.id), {"sample_id": s.id, "paths": []})
                label = s.name
                if st is None and p.parent is None and s.id not in sample_ids:
                    sample_ids.append(s.id)
            else:
                continue
            if st is not None:
                e.setdefault("stats", []).append(_stat_spec(st, p.path))
                texts.append(f"{label}{p.path} {st.label}")
            else:
                e["paths"].append(p.path)
                texts.append(f"{label}{p.path}")
        md = QMimeData()
        payload = list(entries.values())
        if payload:
            md.setData(POPULATION_MIME, QByteArray(json.dumps(payload[0] if len(payload) == 1 else payload).encode("utf-8")))
        if sample_ids:
            md.setData(SAMPLES_MIME, QByteArray(json.dumps(sorted(sample_ids)).encode("utf-8")))
        md.setText("\n".join(texts))
        return md

    def startDrag(self, actions):
        items = self.selectedItems()
        md = self.mimeData(items)
        if md is None or not md.formats():
            return
        drag = QDrag(self)
        drag.setMimeData(md)
        first = next((i for i in items if i.data(0, ROLE_POP) is not None), None)
        if first is not None:
            pop = first.data(0, ROLE_POP)
            if first.data(0, ROLE_STAT) is not None:
                name = "statistic"
            elif first.data(0, ROLE_GROUP) is not None and pop.parent is None:
                name = "group"
            else:
                name = gate_icon_name(pop.gate) if pop.parent is not None else "sample"
            drag.setPixmap(pixmap(name, 24))
        # within a sample a plain drag moves a gate; Alt / Option copies
        drag.exec(Qt.DropAction.CopyAction | Qt.DropAction.MoveAction, Qt.DropAction.CopyAction)

    @staticmethod
    def _copy_modifier(event) -> bool:
        return bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasUrls() or md.hasFormat(POPULATION_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            event.acceptProposedAction()
            return
        if md.hasFormat(POPULATION_MIME):
            item = self.itemAt(event.position().toPoint())
            if item is not None and item.data(0, ROLE_STAT) is None and item.data(0, ROLE_POP) is not None:
                drop = self._describe_drop(md, item, self._copy_modifier(event))
                move = drop is not None and self._is_move(drop)
                action = Qt.DropAction.MoveAction if move else Qt.DropAction.CopyAction
                if event.possibleActions() & action:
                    event.setDropAction(action)
                    event.accept()
                else:
                    event.acceptProposedAction()
                return
        event.ignore()

    @staticmethod
    def _is_move(drop: TreeDrop) -> bool:
        if drop.copy:
            return False
        if drop.target_group is not None:
            return any(g is drop.target_group for g, _ in drop.group_pops)
        return any(s is drop.target_sample and p.parent is not None for s, p in drop.pops)

    def _describe_drop(self, md, item: QTreeWidgetItem, copy: bool) -> TreeDrop | None:
        target_pop = item.data(0, ROLE_POP)
        if target_pop is None:
            return None
        drop = TreeDrop(target_pop=target_pop, target_sample=item.data(0, ROLE_SAMPLE), target_group=item.data(0, ROLE_GROUP), copy=copy)
        ws = self.controller.workspace
        for entry in _parse_population_payload(md):
            if entry.get("group_id") is not None:
                g = next((x for x in ws.groups if x.id == entry.get("group_id")), None)
                if g is None:
                    continue
                for path in entry.get("paths", []):
                    pop = g.root.find(path)
                    if pop is not None:
                        drop.group_pops.append((g, pop))
                for spec in entry.get("stats", []) or []:
                    pop = g.root.find(spec.get("path", ""))
                    if pop is not None:
                        drop.stats.append((g, pop, spec))
                continue
            try:
                sample = ws.sample_by_id(int(entry.get("sample_id", -1)))
            except (TypeError, ValueError):
                sample = None
            if sample is None:
                continue
            for path in entry.get("paths", []):
                pop = sample.root.find(path)
                if pop is not None:
                    drop.pops.append((sample, pop))
            for spec in entry.get("stats", []) or []:
                pop = sample.root.find(spec.get("path", ""))
                if pop is not None:
                    drop.stats.append((sample, pop, spec))
        return drop

    def dropEvent(self, event):
        md = event.mimeData()
        item = self.itemAt(event.position().toPoint())
        if md.hasUrls():
            paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
            group = item.data(0, ROLE_GROUP) if item is not None else None
            if paths:
                if group is not None:
                    self.filesDroppedOnGroup.emit(group, paths)
                else:
                    self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        if md.hasFormat(POPULATION_MIME):
            if item is None or item.data(0, ROLE_STAT) is not None or item.data(0, ROLE_POP) is None:
                event.ignore()
                return
            drop = self._describe_drop(md, item, self._copy_modifier(event))
            if drop is not None and (drop.pops or drop.group_pops or drop.stats):
                move = self._is_move(drop)
                self.treeDropped.emit(drop)
                action = Qt.DropAction.MoveAction if move else Qt.DropAction.CopyAction
                if event.possibleActions() & action:
                    event.setDropAction(action)
                    event.accept()
                    return
            event.acceptProposedAction()
            return
        event.ignore()


from PySide6.QtWidgets import QTreeWidgetItemIterator  # noqa: E402


# ----------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self, controller: WorkspaceController | None = None):
        super().__init__()
        self.controller = controller or WorkspaceController()
        c = self.controller
        self.setWindowIcon(icon("app"))
        self.resize(1280, 800)
        self.graph_windows: dict[tuple, GraphWindow] = {}
        self.layout_editor: LayoutEditorWindow | None = None
        self.table_editor: TableEditorWindow | None = None
        self.compensation_window: CompensationWindow | None = None
        # extension hooks: callables(menu, item) / (menu, group) run before the
        # sample-tree / group-pane context menu opens, so modules can add entries
        self.tree_context_hooks: list = []
        self.group_context_hooks: list = []
        # other top-level windows (title, window) listed in the Window menu
        self.aux_windows: list = []
        self._stat_clipboard: list[dict] = []  # copied statistic rows

        self.menus: dict[str, QMenu] = {}
        self.ribbon_pages: dict = {}
        self._build_actions()
        self._build_menus()
        self._build_ribbon()
        self._build_central()
        self.statusBar().showMessage("Ready")
        self.setAcceptDrops(True)
        self._install_extensions()
        ld = self.loader
        # with a background loader a tree rebuild never reads event data on the
        # GUI thread: unloaded samples show "…" until their counts are ready
        self.tree.defer_unloaded = ld is not None
        if ld is not None and hasattr(ld, "countsReady"):
            ld.countsReady.connect(lambda _samples: self._refresh_tree())
        if ld is not None and hasattr(ld, "finished"):
            # a job can end without filling anything (its samples were loaded meanwhile, for
            # instance by a graph window): refresh once more so no "\u2026" stays behind
            ld.finished.connect(lambda *_args: self._refresh_tree())

        c.open_graph = self.open_graph
        c.open_group_graph = self.open_group_graph
        c.open_layout_editor = self.show_layout_editor
        c.open_table_editor = self.show_table_editor
        c.open_compensation = self.show_compensation
        c.add_to_layout = self._add_to_layout
        c.add_to_table = self._add_to_table

        c.workspaceReplaced.connect(self._workspace_replaced)
        c.samplesChanged.connect(self._refresh_all)
        c.gatingChanged.connect(lambda _sid: self._refresh_tree())
        c.groupsChanged.connect(self._refresh_all)
        c.compensationChanged.connect(lambda _sid: self._refresh_tree())
        c.scalesChanged.connect(lambda _sid: None)
        c.currentGroupChanged.connect(lambda _g: self._refresh_tree())
        # opened / imported workspaces, added samples and a group switch load
        # the shown samples in the background
        c.workspaceReplaced.connect(self._load_shown)
        c.samplesChanged.connect(self._load_shown)
        c.currentGroupChanged.connect(self._load_shown)
        c.selectionChanged.connect(self._update_action_states)
        c.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        c.modifiedChanged.connect(lambda _m: self._update_title())
        c.preferencesChanged.connect(self._refresh_tree)
        # edits made in other windows (layouts, tables, matrices, settings) add
        # undo steps too: keep Undo / Redo enabled and labelled for them
        for sig in (c.layoutsChanged, c.tablesChanged, c.settingsChanged, c.modifiedChanged, c.scalesChanged, c.compensationChanged):
            sig.connect(lambda *_a: self._sync_undo_actions())
        app = QApplication.instance()
        if app is not None:
            app.focusWindowChanged.connect(self._focus_window_changed)
        self._restore_geometry()
        self._workspace_replaced()

    # ------------------------------------------------------------------ actions
    def _act(self, text: str, icon_name: str | None, slot, shortcut=None, tip: str | None = None, checkable=False) -> QAction:
        act = QAction(icon(icon_name) if icon_name else None, text, self) if icon_name else QAction(text, self)
        if shortcut:
            act.setShortcut(shortcut)
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        act.setCheckable(checkable)
        if slot is not None:
            act.triggered.connect(slot)
        return act

    def _build_actions(self):
        c = self.controller
        self.act_new = self._act("New Workspace", "new", self.new_workspace, QKeySequence.StandardKey.New, "Start a new, empty workspace")
        self.act_open = self._act("Open…", "open", self.open_workspace, QKeySequence.StandardKey.Open, "Open a FlowDesk workspace (.fdw) or import a FlowJo workspace (.wsp)")
        self.act_save = self._act("Save", "save", self.save_workspace, QKeySequence.StandardKey.Save, "Save the workspace")
        self.act_save_as = self._act("Save As…", "save-as", self.save_workspace_as, QKeySequence.StandardKey.SaveAs, "Save the workspace under a new name")
        self.act_import_wsp = self._act("Import FlowJo Workspace…", "import", self.import_wsp, None, "Import gates, groups and transforms from a FlowJo .wsp file")
        self.act_export_wsp = self._act("Export FlowJo Workspace…", "export", self.export_wsp, None, "Write the workspace as a FlowJo-style .wsp file")
        self.act_quit = self._act("Quit", None, self.close, QKeySequence.StandardKey.Quit)
        self.act_add_samples = self._act("Add Samples…", "add-samples", self.add_samples, QKeySequence("Ctrl+Shift+A"), "Add FCS files to the workspace")
        self.act_add_folder = self._act("Add Folder…", "add-folder", self.add_folder, None, "Add every FCS file inside a folder")
        self.act_remove_samples = self._act("Remove Samples", "remove", self.remove_samples, None, "Remove the selected samples from the workspace")
        self.act_graph = self._act("Graph Window", "graph", self.open_selected_graph, QKeySequence("Ctrl+G"), "Open a graph window for the selected population")
        self.act_layout = self._act("Layout Editor", "layout-editor", self.show_layout_editor, QKeySequence("Ctrl+L"), "Arrange plots on pages and export them")
        self.act_table = self._act("Table Editor", "table-editor", self.show_table_editor, QKeySequence("Ctrl+T"), "Define statistics tables and export them")
        self.act_comp = self._act("Compensation", "compensation", self.show_compensation, QKeySequence("Ctrl+M"), "View, edit and apply compensation matrices")
        self.act_wizard = self._act("Compensation Wizard…", "wizard", self.run_wizard, None, "Compute a spillover matrix from single-stain controls")
        self.act_create_group = self._act("Create Group…", "group", self.create_group, None, "Create a group of samples that share gates")
        self.act_edit_group = self._act("Edit Group…", "rename", self.edit_group, None, "Edit the current group")
        self.act_delete_group = self._act("Delete Group", "delete", self.delete_group, None, "Delete the current group")
        self.act_add_stat = self._act("Add Statistic…", "statistic", self.add_statistic, None, "Add a statistic below the selected populations")
        self.act_and = self._act("AND Gate", "boolean-and", lambda: self.create_boolean("and"), None, "Create a Boolean AND of selected populations")
        self.act_or = self._act("OR Gate", "boolean-or", lambda: self.create_boolean("or"), None, "Create a Boolean OR of selected populations")
        self.act_not = self._act("NOT Gate", "boolean-not", lambda: self.create_boolean("not"), None, "Create a Boolean NOT of the selected population")
        self.act_apply_group = self._act("Apply to Group", "group", self.apply_to_group, None, "Copy the selected gates to every sample of the current group")
        self.act_remove_group = self._act("Remove from Group", "minus", self.remove_from_group, None, "Remove the selected gates from every sample of the current group")
        self.act_keywords = self._act("Edit Keywords…", "keyword", self.edit_keywords, None, "View and edit the keywords of the selected sample")
        self.act_columns = self._act("Edit Columns…", "table", self.edit_columns, None, "Choose keyword columns to show in the sample list")
        self.act_expand = self._act("Expand All", "expand", lambda: self.tree.expand_all_items(True))
        self.act_collapse = self._act("Collapse All", "collapse", lambda: self.tree.expand_all_items(False))
        self.act_undo = self._act("Undo", "undo", c.undo, QKeySequence.StandardKey.Undo)
        self.act_redo = self._act("Redo", "redo", c.redo, QKeySequence.StandardKey.Redo)
        # one undo history for every window: the shortcuts work in the Layout /
        # Table Editor, Compensation and Plate Editor windows too
        self.act_undo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_redo.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        self.act_cut = self._act("Cut", "cut", self.cut_populations, QKeySequence.StandardKey.Cut)
        self.act_copy = self._act("Copy", "copy", self.copy_populations, QKeySequence.StandardKey.Copy)
        self.act_paste = self._act("Paste", "paste", self.paste_populations, QKeySequence.StandardKey.Paste)
        self.act_delete = self._act("Delete", "delete", self.delete_selected, QKeySequence.StandardKey.Delete)
        self.act_rename = self._act("Rename", "rename", self.rename_selected, QKeySequence("F2"))
        self.act_select_all = self._act("Select All", None, self.tree_select_all, QKeySequence.StandardKey.SelectAll)
        self.act_prefs = self._act("Preferences…", "preferences", self.show_preferences, QKeySequence.StandardKey.Preferences)
        self.act_about = self._act(f"About {APP_NAME}", "about", lambda: show_about(self))
        self.act_derive = self._act("Derive Parameter…", "derive", self.derive_parameter, None, "Create a derived parameter (ratio, product, log) for the selected samples")
        self.act_export_events = self._act("Export Events (CSV)…", "export", self.export_events, None, "Export the events of the selected population to a CSV file")
        self.act_export_fcs = self._act("Export Population as FCS…", "export", self.export_fcs, None, "Write the events of the selected populations to new FCS files")
        self.act_concat = self._act("Concatenate Samples…", "add-samples", self.concatenate_samples, None, "Combine the selected samples (or populations) into one FCS file with a SampleID parameter")
        self.act_quickstart = self._act("Quick Start", "help", self.show_quickstart, QKeySequence.StandardKey.HelpContents)
        self.act_add_to_layout = self._act("Add to Layout", "layout-editor", lambda: self._add_selected_to_layout(), None, "Add plots of the selected populations to the Layout Editor")
        self.act_add_to_table = self._act("Add to Table", "table-editor", lambda: self._add_to_table(self.controller.selected_populations() + [p for _, p in self.tree.selected_group_rows()]), None, "Add the selected populations as table columns")
        self.act_uncompensate = self._act("Remove Compensation", "minus", self.remove_compensation, None, "Use raw, uncompensated data for the selected samples")
        self.act_acq_comp = self._act("Use Acquisition Matrix", "compensation", self.use_acquisition_comp, None, "Use the spillover matrix stored in the FCS file")
        self.act_live_group = self._act("Live Membership", None, None, None, "Membership of the current keyword group follows its criteria; untick to freeze it", checkable=True)
        self.act_live_group.triggered.connect(lambda on: self.set_group_live(self.controller.current_group, on))
        for a in (self.act_new, self.act_open, self.act_save, self.act_save_as, self.act_quit, self.act_add_samples, self.act_graph, self.act_layout, self.act_table, self.act_comp, self.act_undo, self.act_redo, self.act_cut, self.act_copy, self.act_paste, self.act_delete, self.act_rename, self.act_select_all, self.act_prefs):
            self.addAction(a)
        # short labels for the ribbon buttons (menus keep the full text)
        for act, short in (
            (self.act_import_wsp, "Import FlowJo .wsp"),
            (self.act_export_wsp, "Export FlowJo .wsp"),
            (self.act_export_events, "Events CSV"),
            (self.act_export_fcs, "Population FCS"),
            (self.act_acq_comp, "Acquisition Matrix"),
            (self.act_uncompensate, "Uncompensate"),
            (self.act_concat, "Concatenate"),
            (self.act_derive, "Derive Parameter"),
            (self.act_wizard, "Compensation Wizard"),
            (self.act_create_group, "Create Group"),
            (self.act_add_stat, "Add Statistic"),
            (self.act_add_samples, "Add Samples"),
            (self.act_add_folder, "Add Folder"),
            (self.act_remove_samples, "Remove"),
            (self.act_apply_group, "Apply to Group"),
            (self.act_remove_group, "Remove from Group"),
            (self.act_keywords, "Keywords"),
            (self.act_columns, "Columns"),
        ):
            act.setIconText(short)

    # Optional feature modules add their own ribbon bands / menu entries through
    # an `install(main_window)` function; a missing or broken module must never
    # prevent the workspace window from opening.
    EXTENSION_MODULES = (
        "flowdesk.ui.experiment",
        "flowdesk.ui.plate_editor",
        "flowdesk.ui.tools_analysis",
        "flowdesk.ui.tools_biology",
        "flowdesk.ui.tools_qc",
        "flowdesk.ui.compensation",
        "flowdesk.ui.graph_extras",
        "flowdesk.ui.cytometer",
        # optional providers the handlers below delegate to when installed
        "flowdesk.ui.population_tools",
        "flowdesk.ui.export_tools",
        # background loading of event data and counts (status-bar progress)
        "flowdesk.ui.loader",
        # sample-list commands: find, header sort, move, rename, group by keyword
        "flowdesk.ui.workspace_tools",
    )
    # more files than this are added through the loader (headers read in the
    # background with progress and cancel)
    LOADER_THRESHOLD = 50

    @property
    def loader(self):
        """The background SampleLoader extension, or None."""
        return getattr(self, "extensions", {}).get("flowdesk.ui.loader")

    def _load_shown(self, *_args):
        """Load the event data and counts of the samples the tree shows in the
        background; the tree shows "…" for them until countsReady. Queued for
        the event loop: the request often comes from inside a loader callback
        (samples added when their headers were read), which must not re-enter
        the loader."""
        if self.loader is None or getattr(self, "_load_pending", False):
            return
        self._load_pending = True
        QTimer.singleShot(0, self._load_shown_now)

    def _load_shown_now(self):
        self._load_pending = False
        ld = self.loader
        if ld is None:
            return
        try:
            ld.load(self.controller.workspace.members(self.controller.current_group))
        except Exception:  # noqa: BLE001 - loading is an optimisation, never fatal
            import traceback

            traceback.print_exc()

    def _install_extensions(self):
        import importlib

        self.extensions: dict[str, object] = {}
        self.extension_errors: dict[str, str] = {}  # module -> why it is not available
        for modname in self.EXTENSION_MODULES:
            try:
                module = importlib.import_module(modname)
            except ModuleNotFoundError as exc:
                if exc.name != modname:  # the module exists but one of its imports is missing
                    self._extension_failed(modname, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - SyntaxError, AttributeError, NameError ... at import
                self._extension_failed(modname, exc)
                continue
            install = getattr(module, "install", None)
            if install is None:
                continue
            try:
                self.extensions[modname] = install(self)
            except Exception as exc:  # noqa: BLE001 - defensive
                self._extension_failed(modname, exc)
        if self.extension_errors:
            names = ", ".join(m.rsplit(".", 1)[-1] for m in self.extension_errors)
            self.statusBar().showMessage(f"Some tools could not be loaded ({names}); see the console for details", 15000)

    def _extension_failed(self, modname: str, exc: BaseException):
        import sys
        import traceback

        print(f"FlowDesk: the optional module {modname} could not be loaded:", file=sys.stderr)
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        self.extension_errors[modname] = f"{type(exc).__name__}: {exc}"

    def _build_menus(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        self.menus["file"] = file_menu
        for a in (self.act_new, self.act_open):
            file_menu.addAction(a)
        self.recent_menu = file_menu.addMenu("Open Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        file_menu.addSeparator()
        for a in (self.act_save, self.act_save_as):
            file_menu.addAction(a)
        file_menu.addSeparator()
        for a in (self.act_import_wsp, self.act_export_wsp, self.act_export_events, self.act_export_fcs):
            file_menu.addAction(a)
        file_menu.addSeparator()
        for a in (self.act_add_samples, self.act_add_folder, self.act_remove_samples):
            file_menu.addAction(a)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)
        edit = mb.addMenu("&Edit")
        self.menus["edit"] = edit
        for a in (self.act_undo, self.act_redo):
            edit.addAction(a)
        edit.addSeparator()
        for a in (self.act_cut, self.act_copy, self.act_paste, self.act_delete, self.act_rename, self.act_select_all):
            edit.addAction(a)
        edit.addSeparator()
        edit.addAction(self.act_prefs)
        ws_menu = mb.addMenu("&Workspace")
        self.menus["workspace"] = ws_menu
        for a in (self.act_graph, self.act_create_group, self.act_edit_group, self.act_delete_group, self.act_live_group, self.act_add_stat, self.act_and, self.act_or, self.act_not, self.act_apply_group, self.act_remove_group, self.act_keywords, self.act_columns, self.act_expand, self.act_collapse):
            ws_menu.addAction(a)
        tools = mb.addMenu("&Tools")
        self.menus["tools"] = tools
        for a in (self.act_layout, self.act_table, self.act_comp, self.act_wizard, self.act_derive, self.act_concat, self.act_uncompensate, self.act_acq_comp, self.act_add_to_layout, self.act_add_to_table):
            tools.addAction(a)
        self.window_menu = mb.addMenu("&Window")
        self.window_menu.aboutToShow.connect(self._fill_window_menu)
        self.menus["window"] = self.window_menu
        help_menu = mb.addMenu("&Help")
        self.menus["help"] = help_menu
        help_menu.addAction(self.act_quickstart)
        help_menu.addAction(self.act_about)

    def _build_ribbon(self):
        self.ribbon = Ribbon()
        page = self.ribbon.add_page(APP_NAME)
        self.ribbon_pages["flowdesk"] = page
        band = page.add_band("Workspace")
        band.add_large(self.act_new)
        band.add_large(self.act_open)
        band.add_large(self.act_save)
        band.add_small(self.act_save_as)
        band.add_small(self.act_import_wsp)
        band.add_small(self.act_export_wsp)
        band = page.add_band("Samples")
        band.add_large(self.act_add_samples)
        band.add_large(self.act_add_folder)
        band.add_large(self.act_remove_samples)
        band = page.add_band("Application")
        band.add_large(self.act_prefs)
        band.add_large(self.act_about)

        page = self.ribbon.add_page("Workspace")
        self.ribbon_pages["workspace"] = page
        band = page.add_band("Navigate")
        band.add_large(self.act_graph)
        band.add_large(self.act_layout)
        band.add_large(self.act_table)
        band.add_large(self.act_comp)
        band = page.add_band("Populations")
        band.add_large(self.act_create_group)
        band.add_large(self.act_add_stat)
        boolean_menu = QMenu(self)
        boolean_menu.addAction(self.act_and)
        boolean_menu.addAction(self.act_or)
        boolean_menu.addAction(self.act_not)
        self.act_boolean = self._act("Boolean Gate", "gate-boolean", lambda: self.create_boolean("and"), None, "Combine populations with AND / OR / NOT")
        band.add_large(self.act_boolean, boolean_menu)
        band.add_small(self.act_apply_group)
        band.add_small(self.act_remove_group)
        band.add_small(self.act_edit_group)
        band = page.add_band("Keywords")
        band.add_small(self.act_keywords)
        band.add_small(self.act_columns)
        band.add_small(self.act_delete_group)

        page = self.ribbon.add_page("Edit")
        self.ribbon_pages["edit"] = page
        band = page.add_band("History")
        band.add_large(self.act_undo)
        band.add_large(self.act_redo)
        band = page.add_band("Clipboard")
        band.add_large(self.act_paste)
        band.add_small(self.act_cut)
        band.add_small(self.act_copy)
        band.add_small(self.act_delete)
        band = page.add_band("Selection")
        band.add_large(self.act_rename)
        band.add_small(self.act_select_all)
        band.add_small(self.act_expand)
        band.add_small(self.act_collapse)

        page = self.ribbon.add_page("Tools")
        self.ribbon_pages["tools"] = page
        band = page.add_band("Compensation")
        band.add_large(self.act_wizard)
        band.add_large(self.act_comp)
        band.add_small(self.act_acq_comp)
        band.add_small(self.act_uncompensate)
        band = page.add_band("Samples")
        band.add_small(self.act_derive)
        band.add_small(self.act_concat)
        band = page.add_band("Export")
        band.add_small(self.act_add_to_layout)
        band.add_small(self.act_add_to_table)
        band.add_small(self.act_export_fcs)
        band.add_small(self.act_export_events)
        band.add_small(self.act_export_wsp)

        page = self.ribbon.add_page("Configure")
        self.ribbon_pages["configure"] = page
        band = page.add_band("Settings")
        band.add_large(self.act_prefs)
        band.add_large(self.act_columns)
        band = page.add_band("Help")
        band.add_large(self.act_quickstart)
        band.add_large(self.act_about)

    def _build_central(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.ribbon)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setContentsMargins(6, 6, 6, 6)
        self.group_pane = GroupPane(self.controller)
        self.group_pane.openGroup.connect(lambda g: self.open_group_graph(g, None) if g is not None else None)
        self.group_pane.populationsDropped.connect(self._populations_dropped_on_group)
        self.group_pane.samplesDropped.connect(lambda g, samples: self.controller.add_samples_to_group(g, samples))
        self.group_pane.statsDropped.connect(self._stats_dropped_on_group)
        self.group_pane.filesDropped.connect(self._files_dropped_on_group)
        self.group_pane.groupRenamed.connect(self.rename_group)
        self.group_pane.deleteRequested.connect(self._delete_group_confirmed)
        self.group_pane.contextMenu.connect(self._group_context_menu)
        self.tree = SampleTree(self.controller)
        self.tree.openRequested.connect(self.open_graph)
        self.tree.groupOpenRequested.connect(self.open_group_graph)
        self.tree.filesDropped.connect(self._files_dropped_on_tree)
        self.tree.filesDroppedOnGroup.connect(self._files_dropped_on_group)
        self.tree.treeDropped.connect(self._tree_dropped)
        self.tree.contextMenu.connect(self._tree_context_menu)
        self.tree.renamed.connect(self._renamed)
        self.tree.groupItemRenamed.connect(self._group_item_renamed)
        self.tree.deleteRequested.connect(self.delete_selected)
        self.splitter.addWidget(self.group_pane)
        self.splitter.addWidget(self.tree)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([130, 600])
        layout.addWidget(self.splitter, 1)
        self.setCentralWidget(central)
        self.count_label = QLabel()
        self.statusBar().addPermanentWidget(self.count_label)

    # --------------------------------------------------------------- refresh
    def _workspace_replaced(self):
        ws = self.controller.workspace
        self.tree.keyword_columns = list(ws.settings.get("columns", []))
        for w in list(self.graph_windows.values()):
            w.close()
        self._refresh_all()
        self._update_title()
        if self.layout_editor is not None:
            self.layout_editor.refresh()

    def _refresh_all(self):
        self.group_pane.rebuild()
        self._refresh_tree()

    def _refresh_tree(self):
        self.tree.rebuild()
        self._update_action_states()
        ws = self.controller.workspace
        g = self.controller.current_group
        n = len(ws.members(g))
        self.count_label.setText(f"{n} sample{'s' if n != 1 else ''} in “{g.name}” · {len(ws.samples)} total")

    def _update_title(self):
        ws = self.controller.workspace
        self.setWindowTitle(f"{APP_NAME} – {ws.name}[*]")
        self.setWindowModified(ws.modified)

    def _update_action_states(self):
        c = self.controller
        rows = c.selected
        pops = [p for _, p in rows if p.parent is not None]
        samples = c.selected_samples()
        has_rows = bool(rows)
        grows = self.tree.selected_group_rows()  # group row / group-template populations
        gpops = [p for _, p in grows if p.parent is not None]
        stats = self.tree.selected_stats()
        gstats = self.tree.selected_group_stats()
        self.act_graph.setEnabled(has_rows or bool(grows))
        self.act_remove_samples.setEnabled(bool(samples))
        self.act_add_stat.setEnabled(has_rows or bool(grows))
        n_bool = len(pops) or len(gpops)
        self.act_and.setEnabled(n_bool >= 1)
        self.act_or.setEnabled(n_bool >= 1)
        self.act_not.setEnabled(n_bool >= 1)
        if hasattr(self, "act_boolean"):
            self.act_boolean.setEnabled(n_bool >= 1)
        self.act_apply_group.setEnabled(bool(pops))
        self.act_remove_group.setEnabled(bool(pops) or bool(gpops))
        self.act_copy.setEnabled(bool(pops) or bool(gpops) or bool(stats) or bool(gstats))
        self.act_cut.setEnabled(bool(pops) or bool(gpops))
        self.act_paste.setEnabled(bool(c.clipboard or self._stat_clipboard) and (has_rows or bool(grows)))
        self.act_delete.setEnabled(bool(pops) or bool(stats) or bool(samples) or bool(gpops) or bool(gstats))
        self.act_rename.setEnabled(len(rows) + len(grows) == 1)
        self.act_keywords.setEnabled(len(samples) == 1)
        self.act_export_events.setEnabled(len(rows) == 1)
        self.act_export_fcs.setEnabled(has_rows)
        self.act_concat.setEnabled(len(rows) >= 2)
        self.act_add_to_layout.setEnabled(has_rows or bool(gpops))
        self.act_add_to_table.setEnabled(has_rows or bool(grows))
        self.act_derive.setEnabled(bool(samples))
        self.act_uncompensate.setEnabled(bool(samples))
        self.act_acq_comp.setEnabled(bool(samples))
        self._sync_undo_actions()
        g = c.current_group
        self.act_delete_group.setEnabled(g.role != "all")
        self.act_edit_group.setEnabled(g.role != "all")
        self.act_live_group.setEnabled(g.role != "all" and bool(g.criteria))
        self.act_live_group.setChecked(bool(g.live))

    def _sync_undo_actions(self):
        c = self.controller
        self.act_undo.setEnabled(c.can_undo)
        self.act_redo.setEnabled(c.can_redo)
        self.act_undo.setText(f"Undo {c.undo_label()}" if c.can_undo else "Undo")
        redo_label = c._redo[-1][0] if c.can_redo and getattr(c, "_redo", None) else ""
        self.act_redo.setText(f"Redo {redo_label}" if redo_label else "Redo")

    _UNDO_KEYS = (QKeySequence.StandardKey.Undo, QKeySequence.StandardKey.Redo)

    def _own_undo_actions(self, win) -> bool:
        """True when `win` has Undo / Redo shortcuts of its own (graph windows):
        the application-wide ones would make the key ambiguous there."""
        keys = {k.toString() for std in self._UNDO_KEYS for k in QKeySequence.keyBindings(std)}
        for a in win.findChildren(QAction):
            if a is self.act_undo or a is self.act_redo:
                continue
            if any(s.toString() in keys for s in a.shortcuts()):
                return True
        return False

    def _focus_window_changed(self, _window=None):
        """Undo / Redo answer the keyboard in every window except those that
        bring their own (Qt ignores a key two active shortcuts claim)."""
        try:
            active = QApplication.activeWindow()
            ctx = Qt.ShortcutContext.ApplicationShortcut
            if active is not None and active is not self and active.isWindow() and self._own_undo_actions(active):
                ctx = Qt.ShortcutContext.WindowShortcut
            self.act_undo.setShortcutContext(ctx)
            self.act_redo.setShortcutContext(ctx)
        except RuntimeError:  # window being destroyed
            pass

    # ------------------------------------------------------------ file menu
    def _fill_recent(self):
        self.recent_menu.clear()
        recent = self.controller.preferences.get("recent_files", [])
        if not recent:
            self.recent_menu.addAction("(empty)").setEnabled(False)
            return
        for p in recent:
            act = self.recent_menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(lambda checked=False, path=p: self._open_path(path))

    def register_window(self, win, title: str | None = None):
        """List another top-level window in the Window menu (while it is open)."""
        self.aux_windows = [(t, w) for t, w in self.aux_windows if w is not win]
        self.aux_windows.append((title, win))

    def open_windows(self) -> list[tuple[str, object]]:
        """(title, window) of every open window, as the Window menu lists them:
        the workspace, the graph windows, then the editors."""
        def alive(w):
            try:
                return w is not None and w.isVisible()
            except RuntimeError:  # deleted on the C++ side
                return False

        out: list[tuple[str, object]] = [("Workspace", self)]
        for w in list(self.graph_windows.values()):
            if alive(w):
                out.append((w.windowTitle() or "Graph Window", w))
        editors = [("Layout Editor", self.layout_editor), ("Table Editor", self.table_editor), ("Compensation", self.compensation_window)]
        plate = getattr(self, "extensions", {}).get("flowdesk.ui.plate_editor")
        editors.append(("Plate Editor", getattr(plate, "editor", None)))
        editors.extend((t or "", w) for t, w in self.aux_windows)
        seen = {id(w) for _, w in out}
        for name, w in editors:
            if alive(w) and id(w) not in seen:
                seen.add(id(w))
                title = w.windowTitle() or name
                out.append((title if name.lower() in title.lower() or not name else f"{name} – {title}", w))
        return out

    @staticmethod
    def _activate(win):
        win.show()
        if win.isMinimized():
            win.showNormal()
        win.raise_()
        win.activateWindow()

    def _fill_window_menu(self):
        self.window_menu.clear()
        active = QApplication.activeWindow()
        windows = self.open_windows()
        n_graphs = sum(1 for _, w in windows if isinstance(w, GraphWindow))
        for k, (title, w) in enumerate(windows):
            if k == 1 or (k > 1 and isinstance(windows[k - 1][1], GraphWindow) and not isinstance(w, GraphWindow)):
                self.window_menu.addSeparator()
            a = self.window_menu.addAction(title)
            a.setCheckable(True)
            a.setChecked(w is active)
            a.triggered.connect(lambda checked=False, win=w: self._activate(win))
        if n_graphs:
            self.window_menu.addSeparator()
            close_all = self.window_menu.addAction("Close All Graph Windows")
            close_all.triggered.connect(lambda: [w.close() for w in list(self.graph_windows.values())])

    def maybe_save(self) -> bool:
        ws = self.controller.workspace
        if not ws.modified:
            return True
        res = QMessageBox.question(self, "Save changes?", f"Save changes to “{ws.name}” before closing?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if res == QMessageBox.StandardButton.Save:
            return self.save_workspace()
        return res == QMessageBox.StandardButton.Discard

    def new_workspace(self):
        if not self.maybe_save():
            return
        self.controller.set_workspace(Workspace())

    def open_workspace(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Workspace", "", f"Workspaces and templates (*{EXTENSION} *.wsp *.fdwt *.wspt);;FlowDesk workspace (*{EXTENSION});;FlowJo workspace (*.wsp);;Templates (*.fdwt *.wspt);;All files (*)")
        if path:
            self._open_path(path)

    def _open_path(self, path: str):
        p = Path(path)
        if p.suffix.lower() == ".mtx":
            self.show_compensation().import_matrix(str(p))
            return
        if p.suffix.lower() == ".wsp":
            self._import_wsp_path(p)
            return
        if p.suffix.lower() in (".fdwt", ".wspt"):
            self._apply_template_path(p)
            return
        if not self.maybe_save():
            return
        try:
            ws = load_workspace(p)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", f"Could not open {p.name}:\n{exc}")
            return
        self.controller.preferences.add_recent(str(p))
        self.controller.set_workspace(ws)
        missing = [s.name for s in ws.samples if s.load_error]
        if missing:
            QMessageBox.warning(self, "Missing files", "These sample files were not found:\n" + "\n".join(missing[:15]) + ("\n…" if len(missing) > 15 else ""))
        warnings = _workspace_load_warnings(ws)
        if warnings and self._ask_locate_computed_data(p, warnings):
            self._locate_computed_data(p)

    @staticmethod
    def _bullet_list(lines: list[str], limit: int = 15) -> str:
        return "\n".join(f"• {w}" for w in lines[:limit]) + (f"\n… and {len(lines) - limit} more" if len(lines) > limit else "")

    def _ask_locate_computed_data(self, path: Path, warnings: list[str]) -> bool:
        """Tell the user which computed parameters were not restored; True when
        they want to locate the data folder."""
        folder = data_dir_for(path).name
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Computed parameters not loaded")
        box.setText(f"Some computed parameters (t-SNE, UMAP, clusters …) of “{path.name}” could not be loaded.")
        box.setInformativeText(
            f"They are stored in the folder “{folder}” next to the workspace file, which may have been renamed, "
            "moved or not copied along. Populations gated on them show no events until the folder is found.\n\n"
            + self._bullet_list(warnings)
        )
        locate = box.addButton("Locate Folder…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Continue Without Them", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(locate)
        box.exec()
        return box.clickedButton() is locate

    def _load_with_data_dir(self, path: Path, data_dir: Path) -> Workspace:
        """Read the workspace at `path` taking its computed arrays from `data_dir`."""
        import inspect

        try:
            takes_dir = "data_dir" in inspect.signature(load_workspace).parameters
        except (TypeError, ValueError):
            takes_dir = False
        if takes_dir:
            return load_workspace(path, data_dir=data_dir)
        from ..core.serialization import workspace_from_dict

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("format") != "flowdesk-workspace":
            raise ValueError("not a FlowDesk workspace file")
        ws = workspace_from_dict(data, path.parent, data_dir)
        ws.path = path
        return ws

    def _locate_computed_data(self, path: Path) -> bool:
        """Let the user pick the .fdw.data folder of `path` and re-open the
        workspace with the arrays from there. Returns True when it was re-opened."""
        expected = data_dir_for(path).name
        folder = QFileDialog.getExistingDirectory(self, f"Locate the folder “{expected}”", str(path.parent))
        if not folder:
            return False
        try:
            ws = self._load_with_data_dir(path, Path(folder))
        except Exception as exc:  # noqa: BLE001 - reported to the user
            QMessageBox.critical(self, "Open failed", f"Could not open {path.name}:\n{exc}")
            return False
        self.controller.set_workspace(ws)
        # saving writes the arrays to <name>.fdw.data next to the workspace again
        self.controller.mark_modified()
        remaining = _workspace_load_warnings(ws)
        if remaining:
            QMessageBox.warning(self, "Computed parameters not loaded", f"Still not loaded from “{Path(folder).name}”:\n" + self._bullet_list(remaining))
        else:
            self.statusBar().showMessage(f"Computed parameters loaded from {Path(folder).name}; save to store them next to {path.name}", 8000)
        return True

    def save_workspace(self) -> bool:
        ws = self.controller.workspace
        if ws.path is None:
            return self.save_workspace_as()
        try:
            ws.settings["columns"] = list(self.tree.keyword_columns)
            save_workspace(ws, ws.path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return False
        self.controller.preferences.add_recent(str(ws.path))
        self.controller.mark_saved()
        self.statusBar().showMessage(f"Saved {ws.path.name}", 4000)
        return True

    def save_workspace_as(self) -> bool:
        ws = self.controller.workspace
        start = str(ws.path) if ws.path else str(Path.home() / f"{ws.name}{EXTENSION}")
        path, _ = QFileDialog.getSaveFileName(self, "Save Workspace As", start, f"FlowDesk workspace (*{EXTENSION})")
        if not path:
            return False
        if not path.lower().endswith(EXTENSION):
            path += EXTENSION
        ws.path = Path(path)
        return self.save_workspace()

    def import_wsp(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import FlowJo Workspace", "", "FlowJo workspace (*.wsp);;All files (*)")
        if path:
            self._import_wsp_path(Path(path))

    def _import_wsp_path(self, path: Path):
        if not self.maybe_save():
            return
        from ..core.wsp_import import import_wsp

        try:
            ws, report = import_wsp(path, search_dirs=[path.parent])
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", f"Could not import {path.name}:\n{exc}")
            return
        self.controller.preferences.add_recent(str(path))
        self.controller.set_workspace(ws)
        self.controller.mark_modified()
        missing = [s for s in ws.samples if not s.path.exists()]
        if missing:
            res = QMessageBox.question(self, "Locate sample files", f"{len(missing)} sample file(s) were not found next to the workspace. Choose a folder to search?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if res == QMessageBox.StandardButton.Yes:
                folder = QFileDialog.getExistingDirectory(self, "Folder containing the FCS files")
                if folder:
                    self._relocate_samples(missing, Path(folder))
        warnings = [w for w in report.warnings if "not found" not in w.lower()]
        self.statusBar().showMessage(f"Imported {len(ws.samples)} samples, {report.populations if hasattr(report, 'populations') else ''} populations from {path.name}", 6000)
        if warnings:
            QMessageBox.information(self, "Import notes", "\n".join(warnings[:20]) + ("\n…" if len(warnings) > 20 else ""))

    def _apply_template_path(self, path: Path):
        """Apply a workspace template (.fdwt, or a FlowJo .wspt/.wsp used as one)
        to the samples of the current workspace."""
        from ..core.templates import apply_template, load_template

        try:
            template = load_template(path)
        except Exception as exc:
            QMessageBox.critical(self, "Template", f"Could not read {path.name}:\n{exc}")
            return
        ws = self.controller.workspace
        if not ws.samples:
            QMessageBox.information(self, "Template", "Add samples first, then apply the template to them.")
            return
        if not confirm(self, "Apply template", f"Apply the groups, gates, scales, layouts and tables of “{path.name}” to the {len(ws.samples)} sample(s) in this workspace?"):
            return
        c = self.controller
        c.begin_change("apply template", kind="all")
        report = apply_template(ws, template)
        c.mark_modified()
        c.groupsChanged.emit()
        c.gatingChanged.emit(None)
        c.scalesChanged.emit(None)
        c.compensationChanged.emit(None)
        c.layoutsChanged.emit()
        c.tablesChanged.emit()
        self._load_shown()
        self.statusBar().showMessage(f"Template applied: {report['groups']} groups, {report['samples']} sample(s) gated, {report['layouts']} layouts, {report['tables']} tables", 6000)

    def _relocate_samples(self, samples: list[Sample], folder: Path):
        found = 0
        candidates = {p.name: p for p in folder.rglob("*.fcs")}
        for s in samples:
            p = candidates.get(s.path.name)
            if p is not None:
                s.path = p
                s.load_error = None
                s.load_header()
                found += 1
        self.controller.samplesChanged.emit()
        self.statusBar().showMessage(f"Located {found} of {len(samples)} files", 5000)

    def export_wsp(self):
        from ..core.wsp_export import export_wsp

        ws = self.controller.workspace
        start = str(ws.path.with_suffix(".wsp")) if ws.path else str(Path.home() / f"{ws.name}.wsp")
        path, _ = QFileDialog.getSaveFileName(self, "Export FlowJo Workspace", start, "FlowJo workspace (*.wsp)")
        if not path:
            return
        try:
            warnings = export_wsp(ws, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {Path(path).name}", 4000)
        if warnings:
            QMessageBox.information(self, "Export notes", "\n".join(warnings[:20]) + ("\n…" if len(warnings) > 20 else ""))

    def _delegate(self, module: str, name: str) -> bool:
        """Run `name` of an installed provider extension instead of the built-in
        handler; False when no provider offers it."""
        ext = getattr(self, "extensions", {}).get(module)
        handler = getattr(ext, name, None)
        if not callable(handler):
            return False
        handler()
        return True

    def export_events(self):
        if self._delegate("flowdesk.ui.export_tools", "export_events"):
            return
        rows = self.controller.selected
        if len(rows) != 1:
            return
        sample, pop = rows[0]
        if not sample.ensure_loaded():
            QMessageBox.warning(self, "Export", f"File not found: {sample.path}")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Events", str(Path.home() / f"{sample.path.stem}_{pop.display_name}.csv"), "CSV (*.csv)")
        if not path:
            return
        names = sample.parameter_names()
        mask = pop.mask()
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(names)
            columns = [sample.data(n)[mask] for n in names]
            for i in range(int(mask.sum())):
                writer.writerow([f"{col[i]:.6g}" for col in columns])
        self.statusBar().showMessage(f"Exported {int(mask.sum()):,} events", 4000)

    def export_fcs(self):
        if self._delegate("flowdesk.ui.export_tools", "export_fcs"):
            return
        rows = self.controller.selected
        if not rows:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the exported FCS files")
        if not folder:
            return
        compensated = False
        if any(s._comp_active() is not None for s, _ in rows):
            compensated = confirm(self, "Export FCS", "Include compensated parameters (Comp-…) in the exported files?")
        written = 0
        for sample, pop in rows:
            name = f"{sample.path.stem}" + (f"_{pop.name}" if pop.parent is not None else "_export") + ".fcs"
            safe = "".join(ch if ch not in '/\\:*?"<>|' else "_" for ch in name)
            try:
                sample.export_population_fcs(pop, Path(folder) / safe, compensated)
                written += 1
            except Exception as exc:
                QMessageBox.warning(self, "Export FCS", f"{sample.name}: {exc}")
        self.statusBar().showMessage(f"Wrote {written} FCS file(s) to {folder}", 5000)

    # ------------------------------------------------------------- samples
    def add_samples(self):
        patterns = " ".join(f"*{s} *{s.upper()}" for s in _sample_suffixes())
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Samples", "", f"Sample files ({patterns});;All files (*)")
        if paths:
            self.add_sample_paths(paths)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add all FCS files in folder")
        if folder:
            self.add_sample_paths(self._folder_samples(folder))

    @staticmethod
    def _folder_samples(folder) -> list[str]:
        """Sample files inside `folder` and its sub-folders, sorted (any case of the suffix)."""
        suffixes = _sample_suffixes()
        return [str(p) for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in suffixes and p.is_file()]

    def add_sample_paths(self, paths: list, group: Group | None = None) -> list[Sample]:
        """Add sample files to the workspace (and to `group` unless it is All
        Samples, in the same undo step). More than LOADER_THRESHOLD files go
        through the background loader: the samples arrive when their headers
        are read and an empty list is returned now."""
        paths = [str(p) for p in paths]
        if not paths:
            return []
        ld = self.loader
        if ld is not None and len(paths) > self.LOADER_THRESHOLD and hasattr(ld, "add_paths"):
            ld.add_paths(paths, lambda ready: self._add_samples_now(ready, group))
            return []
        return self._add_samples_now(paths, group)

    def _add_samples_now(self, paths: list, group: Group | None = None) -> list[Sample]:
        c = self.controller
        ws = c.workspace
        target = group if (group is not None and group.role != "all" and any(g is group for g in ws.groups)) else None
        if target is None:
            return c.add_samples(paths)
        with c.undo_group("add samples", "samples"):
            samples = list(c.add_samples(paths))
            # files already in the workspace join the group as well
            for p in paths:
                s = ws.sample_by_path(p)
                if s is not None and s not in samples:
                    samples.append(s)
            joining = [s for s in samples if s.id not in target.sample_ids]
            if joining:
                c.add_samples_to_group(target, joining)
        if samples:
            self.statusBar().showMessage(f"Added {len(samples)} sample(s) to “{target.name}”", 5000)
        return samples

    def _files_dropped(self, paths: list[str], group: Group | None = None) -> list[Sample]:
        """Add dropped sample files and folders (and open a dropped workspace).
        With a group other than All Samples the samples also join that group,
        in the same undo step. Returns the samples added now (none while the
        loader reads a large drop in the background)."""
        suffixes = _sample_suffixes()
        files = [p for p in paths if p.lower().endswith(suffixes) and not Path(p).is_dir()]
        for folder in (p for p in paths if Path(p).is_dir()):
            files.extend(self._folder_samples(folder))
        others = [p for p in paths if p.lower().endswith((EXTENSION, ".wsp", ".fdwt", ".wspt", ".mtx"))]
        samples = self.add_sample_paths(list(dict.fromkeys(files)), group) if files else []
        if others:
            self._open_path(others[0])
        return samples

    def _files_dropped_on_group(self, group: Group | None, paths: list[str]):
        self._files_dropped(paths, group)

    def _files_dropped_on_tree(self, paths: list[str]):
        # files dropped into the sample list join the group it shows (FlowJo)
        g = self.controller.current_group
        self._files_dropped(paths, g if g.role != "all" else None)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self._files_dropped(paths)

    def remove_samples(self):
        samples = self.controller.selected_samples()
        if not samples:
            return
        if confirm(self, "Remove samples", f"Remove {len(samples)} sample(s) from the workspace?"):
            self.controller.remove_samples(samples)

    def _renamed(self, target, name: str):
        if isinstance(target, Sample):
            self.controller.rename_sample(target, name)
        else:
            self.controller.rename_population(target, name, None)

    # -------------------------------------------------------------- windows
    def open_graph(self, sample: Sample, population: Population):
        key = (sample.id, population.id, None)
        win = self.graph_windows.get(key)
        if win is None:
            win = GraphWindow(self.controller, sample, population, parent=None)
            win.closed.connect(lambda w, k=key: self.graph_windows.pop(k, None))
            self.graph_windows[key] = win
            self._place_window(win)
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def open_group_graph(self, group: Group, population: Population | None):
        members = self.controller.workspace.members(group)
        if not members:
            QMessageBox.information(self, "Group graph", f"Group “{group.name}” has no samples.")
            return None
        sample = members[0]
        pop = sample.root if population is None else (sample.root.find(population.path) or sample.root)
        # one window per group population, whichever member it shows
        key = (group.id, pop.path)
        win = self.graph_windows.get(key)
        if win is None:
            win = GraphWindow(self.controller, sample, pop, group=group, parent=None)
            win.closed.connect(lambda w, k=key: self.graph_windows.pop(k, None))
            self.graph_windows[key] = win
            self._place_window(win)
        win.show()
        win.raise_()
        win.activateWindow()
        return win

    def _place_window(self, win):
        """Cascade new graph windows from the workspace window's top-right
        corner, kept (frame included) inside the available area of its screen."""
        n = len(self.graph_windows)
        step = 24 * (n % 8)
        pos = self.pos() + QPoint(self.width() - 200 + step, 60 + step)
        screen = self.screen() or QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen is None:
            win.move(pos)
            return
        avail = screen.availableGeometry()
        # frame margins: the window's own once shown, else the workspace window's
        ref = win if win.isVisible() else self
        dl = ref.geometry().left() - ref.frameGeometry().left()
        dt = ref.geometry().top() - ref.frameGeometry().top()
        dr = ref.frameGeometry().right() - ref.geometry().right()
        db = ref.frameGeometry().bottom() - ref.geometry().bottom()
        w = min(win.width(), max(200, avail.width() - dl - dr))
        h = min(win.height(), max(200, avail.height() - dt - db))
        if (w, h) != (win.width(), win.height()):
            win.resize(w, h)
        fw, fh = w + dl + dr, h + dt + db
        x, y = pos.x(), pos.y()  # frame top-left (move() places the frame)
        if x + fw > avail.right() + 1:
            # no room right of the workspace: cascade from the screen's right edge
            x = avail.right() + 1 - fw - step
        if y + fh > avail.bottom() + 1:
            y = avail.top() + 30 + step
            if y + fh > avail.bottom() + 1:
                y = avail.bottom() + 1 - fh
        x = max(avail.left(), min(x, avail.right() + 1 - fw))
        y = max(avail.top(), min(y, avail.bottom() + 1 - fh))
        win.move(QPoint(x, y))

    def open_selected_graph(self):
        rows = self.controller.selected[:6]
        for s, p in rows:
            self.open_graph(s, p)
        for g, p in self.tree.selected_group_rows()[: max(0, 6 - len(rows))]:
            self.open_group_graph(g, p if p.parent is not None else None)

    def show_layout_editor(self):
        if self.layout_editor is None:
            self.layout_editor = LayoutEditorWindow(self.controller)
        self.layout_editor.show()
        self.layout_editor.raise_()
        self.layout_editor.activateWindow()
        return self.layout_editor

    def show_table_editor(self):
        if self.table_editor is None:
            self.table_editor = TableEditorWindow(self.controller)
        self.table_editor.show()
        self.table_editor.raise_()
        self.table_editor.activateWindow()
        return self.table_editor

    def show_compensation(self):
        if self.compensation_window is None:
            self.compensation_window = CompensationWindow(self.controller)
        self.compensation_window.show()
        self.compensation_window.raise_()
        self.compensation_window.activateWindow()
        return self.compensation_window

    def run_wizard(self):
        wiz = CompensationWizard(self.controller, self)
        wiz.exec()

    def _add_to_layout(self, sample: Sample, population: Population, overlays=None):
        editor = self.show_layout_editor()
        editor.add_population(sample, population, overlays=overlays)

    def _add_selected_to_layout(self):
        for s, p in self.controller.selected:
            self._add_to_layout(s, p)
        for g, p in self._selected_group_pops():  # group gates: shown on the first member
            members = self.controller.workspace.members(g)
            pop = members[0].root.find(p.path) if members else None
            if pop is not None:
                self._add_to_layout(members[0], pop)

    def _add_to_table(self, populations: list[Population]):
        editor = self.show_table_editor()
        paths = [p.path for p in populations]
        if paths:
            editor.add_columns_for_paths(paths)

    # -------------------------------------------------------------- groups
    def _keyword_names(self) -> list[str]:
        names: dict[str, None] = {}
        for s in self.controller.workspace.samples:
            for k in s.keywords:
                names.setdefault(k, None)
        return sorted(names, key=lambda k: (not k.startswith("$"), k))

    def create_group(self):
        ws = self.controller.workspace
        dlg = GroupDialog(ws.samples, self._keyword_names(), None, self)
        selected = self.controller.selected_samples()
        if selected:
            for i in range(dlg.sample_list.count()):
                it = dlg.sample_list.item(i)
                it.setSelected(it.data(Qt.ItemDataRole.UserRole) in {s.id for s in selected})
        if dlg.exec():
            name, color, ids, crits = dlg.values()
            extras = dlg.extra_values()
            samples = [s for s in ws.samples if s.id in ids]
            c = self.controller
            with c.undo_group("create group"):
                g = c.create_group(name, samples, crits, color)
                dlg.apply_extras(g, ws)  # role, style; frozen: keeps the samples matching now
            c.groupsChanged.emit()
            c.set_current_group(g)
            if extras["role"] == "compensation":
                self.statusBar().showMessage(f"“{g.name}” holds the compensation controls", 5000)

    def edit_group(self):
        g = self.controller.current_group
        if g.role == "all":
            return
        ws = self.controller.workspace
        dlg = GroupDialog(ws.samples, self._keyword_names(), g, self)
        if dlg.exec():
            name, color, ids, crits = dlg.values()
            self.controller.begin_change("edit group")
            g.color = color
            g.criteria = crits
            g.sample_ids = set(ids)
            dlg.apply_extras(g, ws)
            self.controller.rename_group(g, name)
            for s in ws.members(g):
                ws.apply_template(g, s)
            self.controller.mark_modified()
            self.controller.groupsChanged.emit()
            self.controller.gatingChanged.emit(None)

    def rename_group(self, group: Group, name: str) -> bool:
        """Rename a group in place (F2 in the group pane or on the group row)."""
        name = name.strip()
        c = self.controller
        ws = c.workspace
        if group.role == "all" or not name or name == group.name or ws.group_by_name(name) is not None:
            if name and name != group.name and ws.group_by_name(name) is not None:
                self.statusBar().showMessage(f"A group named “{name}” already exists", 5000)
            QTimer.singleShot(0, self._refresh_all)  # put the old name back
            return False
        c.begin_change("rename group")
        c.rename_group(group, name)
        c.gatingChanged.emit(None)
        return True

    def set_group_live(self, group: Group, live: bool):
        """Freeze a keyword group's membership (live off) or make it follow its criteria again."""
        c = self.controller
        if group is None or group.role == "all" or bool(group.live) == bool(live):
            self._update_action_states()
            return
        c.begin_change("live membership" if live else "freeze group")
        set_group_live(c.workspace, group, live)
        if live:
            for s in c.workspace.members(group):
                c.workspace.apply_template(group, s)
        c.mark_modified()
        c.groupsChanged.emit()
        c.gatingChanged.emit(None)

    def delete_group(self):
        self._delete_group_confirmed(self.controller.current_group)

    def _delete_group_confirmed(self, g: Group | None):
        if g is None or g.role == "all":
            return
        if confirm(self, "Delete group", f"Delete group “{g.name}”? Samples and their gates are kept."):
            self.controller.delete_group(g)

    def _exec_menu(self, menu: QMenu, pos: QPoint):
        menu.exec(pos)

    def _run_hooks(self, hooks: list, menu: QMenu, subject):
        import traceback

        for hook in list(hooks):
            try:
                hook(menu, subject)
            except Exception:  # noqa: BLE001 - a broken extension must not block the menu
                traceback.print_exc()

    def _group_context_menu(self, group: Group | None, pos: QPoint):
        menu = QMenu(self)
        menu.addAction(self.act_create_group)
        if group is not None:
            self.controller.set_current_group(group)
            self.group_pane.rebuild()
            act = menu.addAction(icon("graph"), f"Open group graph “{group.name}”")
            act.triggered.connect(lambda: self.open_group_graph(group, None))
            if group.role != "all":
                menu.addAction(self.act_edit_group)
                rename = menu.addAction(icon("rename"), "Rename")
                rename.triggered.connect(self.group_pane.rename_current)
                menu.addAction(self.act_delete_group)
                if group.criteria:
                    menu.addAction(self.act_live_group)
                samples = self.controller.selected_samples()
                if samples:
                    a = menu.addAction(icon("plus"), f"Add {len(samples)} selected sample(s) to group")
                    a.triggered.connect(lambda: self.controller.add_samples_to_group(group, samples))
                    a = menu.addAction(icon("minus"), "Remove selected sample(s) from group")
                    a.triggered.connect(lambda: self.controller.remove_samples_from_group(group, samples))
            pops = [p for _, p in self.controller.selected if p.parent is not None]
            if pops:
                a = menu.addAction(icon("group"), f"Apply {len(pops)} selected gate(s) to this group")
                a.triggered.connect(lambda: self.controller.copy_populations_to_group(pops, group))
        self._run_hooks(self.group_context_hooks, menu, group)
        self._exec_menu(menu, pos)

    def _populations_dropped_on_group(self, group: Group, items: list[tuple[Sample, Population]]):
        pops = [p for _, p in items]
        n = self.controller.copy_populations_to_group(pops, group)
        self.statusBar().showMessage(f"Applied {len(pops)} gate(s) to {n} sample(s) of “{group.name}”", 5000)

    def _stats_dropped_on_group(self, group: Group, items: list):
        """Statistic rows dropped on a group: add them to the population with the
        same path in every member (and the group template)."""
        c = self.controller
        if not items:
            return
        with c.undo_group("add statistic"):
            for pop, spec in items:
                if pop.parent is None:  # an Ungated statistic: every member's root
                    targets = [s.root for s in c.workspace.members(group)]
                    c.add_statistic(targets, spec.get("statistic"), spec.get("param"), spec.get("percentile"), None)
                else:
                    c.add_statistic([pop], spec.get("statistic"), spec.get("param"), spec.get("percentile"), group)

    def apply_to_group(self):
        pops = [p for _, p in self.controller.selected if p.parent is not None]
        if pops:
            g = self.controller.current_group
            n = self.controller.copy_populations_to_group(pops, g)
            self.statusBar().showMessage(f"Applied {len(pops)} gate(s) to {n} sample(s) of “{g.name}”", 5000)

    def remove_from_group(self):
        pops = [p for _, p in self.controller.selected if p.parent is not None]
        if pops and confirm(self, "Remove from group", f"Remove {len(pops)} gate(s) from every sample of “{self.controller.current_group.name}”?"):
            self.controller.delete_populations(pops, self.controller.current_group)

    # ----------------------------------------------------------- populations
    @staticmethod
    def _unique_child_name(parent: Population, name: str, ignore: Population | None = None) -> str:
        base, k, out = name, 2, name
        while any(c.name == out and c is not ignore for c in parent.children):
            out = f"{base} {k}"
            k += 1
        return out

    @staticmethod
    def _movable(pops: list[Population], target: Population) -> list[Population]:
        """The populations of `pops` that can move under `target`: not the target,
        not an ancestor of it, not already its child, not inside another moved one."""
        ancestors = {id(a) for a in target.ancestors()} | {id(target)}
        out = [p for p in pops if p.parent is not None and id(p) not in ancestors and p.parent is not target]
        ids = {id(p) for p in out}
        return [p for p in out if not any(id(a) in ids for a in p.ancestors())]

    @staticmethod
    def _rewrite_refs(root: Population, renames: dict[str, str]):
        """Point Boolean gates at the new paths of moved populations."""
        if not renames:
            return
        for pop in root.walk(False):
            gate = pop.gate
            if isinstance(gate, BooleanGate):
                refs = []
                for ref in gate.refs:
                    for old, new in renames.items():
                        if ref == old or ref.startswith(old + "/"):
                            ref = new + ref[len(old):]
                            break
                    refs.append(ref)
                gate.refs = refs

    def move_populations(self, sample: Sample, pops: list[Population], target: Population) -> int:
        """Re-parent gates of one sample under `target` (FlowJo: drag a gate onto
        another population of the same sample). One undo step; Boolean gates and
        layout plots that pointed at the old paths follow."""
        pops = self._movable([p for p in pops if p.sample is sample], target)
        if not pops:
            return 0
        c = self.controller
        ws = c.workspace
        olds = [p.path for p in pops]

        def follows(path: str) -> bool:
            return any(path == o or path.startswith(o + "/") for o in olds)

        layout_refs = [
            (item, None) for lay in ws.layouts for item in getattr(lay, "items", []) if getattr(item, "sample_id", None) == sample.id and follows(getattr(item, "pop_path", "") or "")
        ] + [
            (item, ov) for lay in ws.layouts for item in getattr(lay, "items", []) for ov in getattr(item, "overlays", []) or [] if isinstance(ov, dict) and ov.get("sample_id") == sample.id and follows(ov.get("pop_path", "") or "")
        ]
        c.begin_change("move gate", ("layouts",) if layout_refs else None)
        renames: dict[str, str] = {}
        for pop in pops:
            old = pop.path
            pop.parent.remove_child(pop)
            pop.name = self._unique_child_name(target, pop.name)
            target.add_child(pop)
            renames[old] = pop.path
        self._rewrite_refs(sample.root, renames)
        for item, ov in layout_refs:
            path = ov.get("pop_path", "") if ov is not None else item.pop_path
            for old, new in renames.items():
                if path == old or path.startswith(old + "/"):
                    path = new + path[len(old):]
                    break
            if ov is not None:
                ov["pop_path"] = path
            else:
                item.pop_path = path
        sample.root.invalidate()
        c.mark_modified()
        c.gatingChanged.emit(sample.id)
        if layout_refs:
            c.layoutsChanged.emit()
        self.statusBar().showMessage(f"Moved {len(pops)} gate(s) under “{target.display_name}”", 4000)
        return len(pops)

    def move_group_populations(self, group: Group, pops: list[Population], target: Population) -> int:
        """Re-parent group-template populations under another template node; every
        member sample's gate moves the same way (one undo step)."""
        pops = self._movable(pops, target)
        if not pops:
            return 0
        c = self.controller
        members = c.workspace.members(group)
        moves = [(p.path, target.path, self._unique_child_name(target, p.name)) for p in pops]
        c.begin_change("move gate")
        for root in [group.root] + [s.root for s in members]:
            renames = {}
            for old, dest_path, name in moves:
                node = root.find(old)
                dest = root.find(dest_path)
                if node is None or dest is None or node.parent is None or id(dest) in {id(x) for x in node.walk()}:
                    continue
                node.parent.remove_child(node)
                node.name = self._unique_child_name(dest, name)
                dest.add_child(node)
                renames[old] = node.path
            self._rewrite_refs(root, renames)
            root.invalidate()
        c.mark_modified()
        c.gatingChanged.emit(None)
        c.groupsChanged.emit()
        return len(pops)

    def _tree_dropped(self, drop: TreeDrop):
        """A drop on the sample tree: a plain drag inside one sample (or inside
        the group tree) moves gates, Alt / Option copies; drops on other samples
        copy gates (a sample row copies its whole tree) and statistic rows; drops
        on the group tree apply to every member."""
        c = self.controller
        target = drop.target_pop
        ops = []
        stats = [(p, spec) for _owner, p, spec in drop.stats]
        if drop.target_group is not None:
            g = drop.target_group
            moves = [p for gg, p in drop.group_pops if gg is g and not drop.copy and p.parent is not None]
            copies = [p for gg, p in drop.group_pops if not (gg is g and not drop.copy) and p.parent is not None and p is not target]
            for s, p in drop.pops:
                if p.parent is not None:
                    copies.append(p)
                else:  # a sample row: its whole tree
                    copies.extend(p.children)
            if moves:
                ops.append(lambda: self.move_group_populations(g, moves, target))
            if copies:
                if target.parent is None:
                    ops.append(lambda: c.copy_populations_to_group(copies, g))
                else:
                    ops.append(lambda: c.paste_populations(copies, target, g))
            for p, spec in stats:
                if target.parent is None:
                    members = [s.root for s in c.workspace.members(g)] + [g.root]
                    ops.append(lambda spec=spec, members=members: c.add_statistic(members, spec.get("statistic"), spec.get("param"), spec.get("percentile"), None))
                else:
                    ops.append(lambda spec=spec: c.add_statistic([target], spec.get("statistic"), spec.get("param"), spec.get("percentile"), g))
        else:
            sample = drop.target_sample
            same = [p for s, p in drop.pops if s is sample and p.parent is not None]
            moves = [] if drop.copy else same
            copies = [p for p in same if p is not target] if drop.copy else []
            for s, p in drop.pops:
                if s is sample:
                    continue
                if p.parent is not None:
                    copies.append(p)
                else:  # a whole sample row onto another sample: copy its entire tree
                    copies.extend(p.children)
                    for st in p.stats:
                        stats.append((p, _stat_spec(st, "")))
            copies.extend(p for _g, p in drop.group_pops if p.parent is not None)
            if moves:
                ops.append(lambda: self.move_populations(sample, moves, target))
            if copies:
                ops.append(lambda: c.paste_populations(copies, target, None))
            for p, spec in stats:
                ops.append(lambda spec=spec: c.add_statistic([target], spec.get("statistic"), spec.get("param"), spec.get("percentile"), None))
        if not ops:
            return
        if len(ops) == 1:
            ops[0]()
            return
        with c.undo_group("drop", None):
            for op in ops:
                op()

    def _group_item_renamed(self, group: Group, pop: Population, name: str):
        """F2 on the group tree: the group row renames the group, a template
        population renames that gate in every member sample."""
        if pop.parent is None:
            self.rename_group(group, name)
            return
        name = name.strip()
        if not name or pop.parent.child_by_name(name) is not None:
            QTimer.singleShot(0, self._refresh_tree)
            return
        self.controller.rename_population(pop, name, group)

    def _selected_group_pops(self) -> list[tuple[Group, Population]]:
        return [(g, p) for g, p in self.tree.selected_group_rows() if p.parent is not None]

    def add_statistic(self):
        rows = self.controller.selected
        grows = self.tree.selected_group_rows()
        if not rows and not grows:
            return
        c = self.controller
        if rows:
            sample = rows[0][0]
        else:
            members = c.workspace.members(grows[0][0])
            sample = members[0] if members else None
        params = sample.parameter_names() if sample is not None else c.workspace.all_parameter_names(grows[0][0])
        labels = {p: sample.param_label(p) for p in params} if sample is not None else {p: p for p in params}
        dlg = AddStatisticDialog(params, labels, grows[0][0] if grows and not rows else c.current_group, self)
        if dlg.exec():
            stat, param, pct, to_group = dlg.values()
            with c.undo_group("add statistic"):
                if rows:
                    c.add_statistic([p for _, p in rows], stat, param, pct, c.current_group if to_group else None)
                for g, p in grows:  # group rows always apply to the members
                    if p.parent is None:
                        c.add_statistic([s.root for s in c.workspace.members(g)] + [g.root], stat, param, pct, None)
                    else:
                        c.add_statistic([p], stat, param, pct, g)

    def create_boolean(self, op: str):
        c = self.controller
        rows = c.selected
        pops = [p for _, p in rows if p.parent is not None]
        gpops = self._selected_group_pops()
        if pops:
            sample = rows[0][0]
            root = sample.root
            parent = pops[0].parent
            group = c.current_group if c.current_group.role != "all" or len(c.workspace.samples) > 1 else None
            checked = bool(c.preferences.get("apply_to_group_default")) or c.current_group.role != "all"
            locked = False
            selected = pops
        elif gpops:
            group = gpops[0][0]
            root = group.root
            parent = gpops[0][1].parent
            checked, locked = True, True  # built in the group tree: always for the whole group
            selected = [p for _, p in gpops]
        else:
            return
        candidates = list(root.walk(False))
        dlg = GroupBooleanGateDialog(candidates, op, group, checked, self, locked=locked)
        dlg.check(selected)
        if dlg.exec():
            op, refs, name = dlg.values()
            if not refs or (op == "not" and len(refs) != 1) or (op != "not" and len(refs) < 2):
                QMessageBox.information(self, "Boolean gate", "Select one population for NOT, or at least two for AND / OR.")
                return
            apply_group = group if dlg.apply_to_group() else None
            c.create_boolean(parent, op, refs, name or f"{op.upper()} gate", apply_group)

    def copy_populations(self):
        pops = [p for _, p in self.controller.selected if p.parent is not None] + [p for _, p in self._selected_group_pops()]
        stats = [_stat_spec(st, p.path) for p, st in self.tree.selected_stats()] + [_stat_spec(st, p.path) for _g, p, st in self.tree.selected_group_stats()]
        if pops:
            self.controller.clipboard = [p.clone(deep=True) for p in pops]
            self._stat_clipboard = []
            self.statusBar().showMessage(f"Copied {len(pops)} gate(s)", 3000)
            self._update_action_states()
        elif stats:
            self._stat_clipboard = stats
            self.controller.clipboard = []
            self.statusBar().showMessage(f"Copied {len(stats)} statistic(s)", 3000)
            self._update_action_states()

    def cut_populations(self):
        pops = [p for _, p in self.controller.selected if p.parent is not None]
        gpops = self._selected_group_pops()
        if pops or gpops:
            self.copy_populations()
            c = self.controller
            with c.undo_group("cut gates"):
                if pops:
                    c.delete_populations(pops, None)
                for g in dict.fromkeys(g for g, _ in gpops):
                    c.delete_populations([p for gg, p in gpops if gg is g], g)

    def paste_populations(self):
        c = self.controller
        rows = c.selected
        grows = self.tree.selected_group_rows()
        if not (rows or grows) or not (c.clipboard or self._stat_clipboard):
            return
        targets = list(dict.fromkeys(p for _, p in rows))
        with c.undo_group("paste gates" if c.clipboard else "add statistic"):
            for target in targets:
                if c.clipboard:
                    c.paste_populations(c.clipboard, target, None)
                for spec in self._stat_clipboard:
                    c.add_statistic([target], spec.get("statistic"), spec.get("param"), spec.get("percentile"), None)
            for g, target in grows:
                if c.clipboard:
                    if target.parent is None:
                        c.copy_populations_to_group(c.clipboard, g)
                    else:
                        c.paste_populations(c.clipboard, target, g)
                for spec in self._stat_clipboard:
                    if target.parent is None:
                        c.add_statistic([s.root for s in c.workspace.members(g)] + [g.root], spec.get("statistic"), spec.get("param"), spec.get("percentile"), None)
                    else:
                        c.add_statistic([target], spec.get("statistic"), spec.get("param"), spec.get("percentile"), g)

    def remove_group_statistics(self, items: list[tuple[Group, Population, StatNode]]):
        """Remove statistic rows of the group template from it and every member."""
        if not items:
            return
        c = self.controller
        c.begin_change("remove statistic")
        for g, tpop, st in items:
            key = (st.statistic, st.param, st.percentile)
            for root in [g.root] + [s.root for s in c.workspace.members(g)]:
                pop = root.find(tpop.path)
                if pop is not None:
                    pop.stats = [x for x in pop.stats if (x.statistic, x.param, x.percentile) != key]
        c.mark_modified()
        c.gatingChanged.emit(None)
        c.groupsChanged.emit()

    def delete_selected(self):
        if self.group_pane.hasFocus() and self.group_pane.state() != QAbstractItemView.State.EditingState:
            self._delete_group_confirmed(self.group_pane.current_group_row())
            return
        c = self.controller
        stats = self.tree.selected_stats()
        gstats = self.tree.selected_group_stats()
        gpops = self._selected_group_pops()
        for pop, st in stats:
            c.remove_statistic(pop, st)
        if gstats:
            self.remove_group_statistics(gstats)
        if gpops:
            names = ", ".join(f"“{p.name}”" for _, p in gpops[:3]) + ("…" if len(gpops) > 3 else "")
            groups = list(dict.fromkeys(g for g, _ in gpops))
            if confirm(self, "Delete group gates", f"Delete {names} from every sample of “{groups[0].name}”?"):
                with c.undo_group("delete gate"):
                    for g in groups:
                        c.delete_populations([p for gg, p in gpops if gg is g], g)
        pops = [p for _, p in c.selected if p.parent is not None]
        if not pops:
            samples = c.selected_samples()
            if samples and not stats and not gstats and not gpops:
                self.remove_samples()
            return
        n_children = sum(sum(1 for _ in p.walk(False)) for p in pops)
        text = f"Delete {len(pops)} gate(s)" + (f" and their {n_children} sub-population(s)" if n_children else "") + "?"
        if confirm(self, "Delete gates", text):
            c.delete_populations(pops, None)

    def rename_selected(self):
        if self.group_pane.hasFocus():
            self.group_pane.rename_current()
            return
        items = self.tree.selectedItems()
        if len(items) == 1 and items[0].data(0, ROLE_STAT) is None:
            self.tree.editItem(items[0], 0)

    def tree_select_all(self):
        self.tree.selectAll()

    def edit_keywords(self):
        samples = self.controller.selected_samples()
        if len(samples) != 1:
            return
        s = samples[0]
        dlg = KeywordsDialog(s, self)
        if dlg.exec():
            ws = self.controller.workspace
            self.controller.begin_change("edit keywords", kind="samples")
            s.keywords = dlg.values()
            s.refresh_channels_from_keywords()
            # a sample that now matches a live keyword group receives its gates
            for g in ws.groups:
                if g.criteria and g.role != "all" and g.contains(s):
                    ws.apply_template(g, s)
            self.controller.mark_modified()
            self.controller.samplesChanged.emit()
            self.controller.groupsChanged.emit()
            self.controller.gatingChanged.emit(s.id)

    def concatenate_samples(self):
        from ..core.model import concatenate_samples

        if self._delegate("flowdesk.ui.export_tools", "concatenate_samples"):
            return
        rows = self.controller.selected
        if len(rows) < 2:
            QMessageBox.information(self, "Concatenate", "Select at least two samples or populations.")
            return
        pops = [p for _, p in rows]
        samples = [s for s, _ in rows]
        path, _ = QFileDialog.getSaveFileName(self, "Concatenated FCS file", str(Path.home() / "concatenated.fcs"), "FCS (*.fcs)")
        if not path:
            return
        if not path.lower().endswith(".fcs"):
            path += ".fcs"
        limit, ok = QInputDialog.getInt(self, "Concatenate", "Maximum events per sample (0 = all):", 0, 0, 10_000_000, 1000)
        if not ok:
            return
        try:
            n = concatenate_samples(samples, pops, path, limit or None)
        except Exception as exc:
            QMessageBox.critical(self, "Concatenate failed", str(exc))
            return
        self.controller.add_samples([path])
        self.statusBar().showMessage(f"Wrote {n:,} events to {Path(path).name} and added it to the workspace", 6000)

    def show_quickstart(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("Quick Start")
        dlg.resize(640, 560)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(QUICK_START_HTML)
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def edit_columns(self):
        dlg = ColumnsDialog(self._keyword_names(), self.tree.keyword_columns, self)
        if dlg.exec():
            self.tree.keyword_columns = dlg.values()
            self.controller.workspace.settings["columns"] = list(self.tree.keyword_columns)
            self.controller.mark_modified()
            self._refresh_tree()

    def derive_parameter(self):
        if self._delegate("flowdesk.ui.population_tools", "derive_parameter"):
            return
        samples = self.controller.selected_samples() or self.controller.workspace.samples
        if not samples:
            return
        params = samples[0].parameter_names()
        dlg = DerivedParameterDialog(params, self)
        if dlg.exec():
            name, spec = dlg.values()
            self.controller.begin_change("derive parameter")
            for s in samples:
                s.add_derived(name, spec)
            self.controller.mark_modified()
            self.controller.scalesChanged.emit(None)
            self.controller.gatingChanged.emit(None)

    def remove_compensation(self):
        samples = self.controller.selected_samples()
        if samples:
            self.controller.set_compensation(samples, None, False)

    def use_acquisition_comp(self):
        from ..core.compensation import SpilloverMatrix

        samples = self.controller.selected_samples()
        found = []
        for s in samples:
            if s.ensure_loaded():
                m = SpilloverMatrix.from_fcs(s.fcs)
                if m is not None:
                    found.append((s, m))
        if not found:
            return
        self.controller.begin_change("use acquisition matrix")
        changed = []
        for s, m in found:
            s.comp_cleared = False
            s.set_compensation(m, True)
            changed.append(s)
        if changed:
            self.controller.mark_modified()
            self.controller.compensationChanged.emit(None)
            self.controller.gatingChanged.emit(None)

    def show_preferences(self):
        dlg = PreferencesDialog(self.controller.preferences, self)
        if dlg.exec():
            self.controller.preferences.update(dlg.values())
            self.controller.preferencesChanged.emit()

    # --------------------------------------------------------- context menu
    def _tree_context_menu(self, item, pos: QPoint):
        menu = QMenu(self)
        if item is None:
            menu.addAction(self.act_add_samples)
            menu.addAction(self.act_add_folder)
            self._exec_menu(menu, pos)
            return
        if not item.isSelected():
            self.tree.clearSelection()
            item.setSelected(True)
        sample = item.data(0, ROLE_SAMPLE)
        group = item.data(0, ROLE_GROUP)
        pop = item.data(0, ROLE_POP)
        st = item.data(0, ROLE_STAT)
        if group is not None:
            self._group_tree_context_menu(menu, item, group, pop, st)
            self._run_hooks(self.tree_context_hooks, menu, item)
            self._exec_menu(menu, pos)
            return
        if st is not None:
            act = menu.addAction(icon("delete"), "Remove statistic")
            act.triggered.connect(lambda: self.controller.remove_statistic(pop, st))
            menu.addAction(self.act_copy)
            self._run_hooks(self.tree_context_hooks, menu, item)
            self._exec_menu(menu, pos)
            return
        menu.addAction(self.act_graph)
        menu.addSeparator()
        if pop.parent is None:
            menu.addAction(self.act_keywords)
            menu.addAction(self.act_remove_samples)
            comp_menu = menu.addMenu(icon("compensation"), "Compensation")
            comp_menu.addAction(self.act_acq_comp)
            comp_menu.addAction(self.act_uncompensate)
            for name, matrix in self.controller.workspace.matrices.items():
                a = comp_menu.addAction(f"Use “{name}”")
                a.triggered.connect(lambda checked=False, m=matrix: self.controller.set_compensation(self.controller.selected_samples(), m, True))
            groups = [g for g in self.controller.workspace.groups if g.role != "all"]
            if groups:
                gm = menu.addMenu(icon("group"), "Add to group")
                for g in groups:
                    a = gm.addAction(g.name)
                    a.triggered.connect(lambda checked=False, grp=g: self.controller.add_samples_to_group(grp, self.controller.selected_samples()))
            menu.addSeparator()
        else:
            menu.addAction(self.act_rename)
            menu.addAction(self.act_delete)
            menu.addAction(self.act_cut)
            menu.addAction(self.act_copy)
            menu.addAction(self.act_apply_group)
            menu.addAction(self.act_remove_group)
            menu.addSeparator()
            bm = menu.addMenu(icon("gate-boolean"), "Boolean gate")
            bm.addAction(self.act_and)
            bm.addAction(self.act_or)
            bm.addAction(self.act_not)
        menu.addAction(self.act_paste)
        menu.addAction(self.act_add_stat)
        menu.addSeparator()
        menu.addAction(self.act_add_to_layout)
        menu.addAction(self.act_add_to_table)
        menu.addAction(self.act_export_fcs)
        menu.addAction(self.act_export_events)
        self._run_hooks(self.tree_context_hooks, menu, item)
        self._exec_menu(menu, pos)

    def _group_tree_context_menu(self, menu: QMenu, item, group: Group, pop: Population, st):
        """Context menu of the group row and its template rows."""
        if st is not None:
            act = menu.addAction(icon("delete"), "Remove statistic from the group")
            act.triggered.connect(lambda: self.remove_group_statistics([(group, pop, st)]))
            menu.addAction(self.act_copy)
            return
        act = menu.addAction(icon("graph"), f"Open group graph “{group.name}”")
        act.triggered.connect(lambda: self.open_group_graph(group, pop if pop.parent is not None else None))
        menu.addSeparator()
        menu.addAction(self.act_rename)
        if pop.parent is None:
            menu.addAction(self.act_edit_group)
            menu.addAction(self.act_delete_group)
            if group.criteria:
                menu.addAction(self.act_live_group)
        else:
            menu.addAction(self.act_delete)
            menu.addAction(self.act_cut)
            menu.addAction(self.act_copy)
            bm = menu.addMenu(icon("gate-boolean"), "Boolean gate")
            bm.addAction(self.act_and)
            bm.addAction(self.act_or)
            bm.addAction(self.act_not)
        menu.addAction(self.act_paste)
        menu.addAction(self.act_add_stat)
        menu.addSeparator()
        if pop.parent is not None:
            menu.addAction(self.act_add_to_layout)
        menu.addAction(self.act_add_to_table)

    # ------------------------------------------------------------- closing
    def _restore_geometry(self):
        settings = QSettings("FlowDesk", "FlowDesk")
        geo = settings.value("main/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        sizes = settings.value("main/splitter")
        if sizes:
            try:
                self.splitter.setSizes([int(v) for v in sizes])
            except (TypeError, ValueError):
                pass

    # how long closing waits for running analyses to stop before asking
    WORKER_STOP_TIMEOUT_MS = 2000

    def stop_background_work(self, timeout_ms: int | None = None) -> list:
        """Cancel the analyses running in worker threads (every extension's
        stop_all(), then any QThread still running below this window) and wait
        up to `timeout_ms` for them to stop. Returns the threads still running."""
        import time
        import traceback

        timeout_ms = self.WORKER_STOP_TIMEOUT_MS if timeout_ms is None else timeout_ms
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        for ext in list(getattr(self, "extensions", {}).values()):
            stop = getattr(ext, "stop_all", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001 - closing must go on
                    traceback.print_exc()
        running = [t for t in self.findChildren(QThread) if t.isRunning()]
        for t in running:
            if hasattr(t, "cancelled"):
                t.cancelled = True  # the analysis workers' cancellation flag
            t.requestInterruption()
            t.quit()
        for t in running:
            t.wait(max(0, int((deadline - time.monotonic()) * 1000)))
        return [t for t in running if t.isRunning()]

    def closeEvent(self, event):
        if not self.maybe_save():
            event.ignore()
            return
        still_running = self.stop_background_work()
        if still_running:
            res = QMessageBox.question(self, "Quit", "An analysis is still running. Quit anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if res != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            # destroying a running QThread aborts the process: keep them alive past the window
            _abandon_threads(still_running)
        settings = QSettings("FlowDesk", "FlowDesk")
        settings.setValue("main/geometry", self.saveGeometry())
        settings.setValue("main/splitter", self.splitter.sizes())
        for w in list(self.graph_windows.values()):
            w.close()
        for w in (self.layout_editor, self.table_editor, self.compensation_window):
            if w is not None:
                w.close()
        QApplication.instance().quit()
        event.accept()
