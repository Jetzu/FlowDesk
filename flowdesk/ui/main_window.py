"""The workspace window: ribbon, group pane and the sample / population tree."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PySide6.QtCore import QByteArray, QMimeData, QPoint, QSettings, QSize, Qt, QUrl, Signal
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
from ..core.serialization import EXTENSION, load_workspace, save_workspace
from ..core.stats import compute_statistic, format_value
from .compensation import CompensationWindow, CompensationWizard
from .controller import POPULATION_MIME, WorkspaceController
from .dialogs import AddStatisticDialog, BooleanGateDialog, ColumnsDialog, DerivedParameterDialog, GroupDialog, KeywordsDialog, PreferencesDialog, confirm, show_about
from .graph_window import GraphWindow
from .icons import gate_icon_name, icon, pixmap
from .layout_editor import LayoutEditorWindow
from .ribbon import Ribbon
from .table_editor import TableEditorWindow

SAMPLES_MIME = "application/x-flowdesk-samples"

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
join the group receive the gates too.</p>
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


# ------------------------------------------------------------------ group pane
class GroupPane(QTableWidget):
    openGroup = Signal(object)
    populationsDropped = Signal(object, list)  # group, [(sample, population)]
    samplesDropped = Signal(object, list)  # group, [Sample]
    contextMenu = Signal(object, QPoint)

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
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAcceptDrops(True)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.itemSelectionChanged.connect(self._selection_changed)
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
        self.setRowCount(len(ws.groups))
        for row, g in enumerate(ws.groups):
            name = QTableWidgetItem(g.name)
            name.setData(Qt.ItemDataRole.UserRole, g)
            name.setIcon(icon("group"))
            font = QFont()
            font.setBold(g.bold or g.role == "all")
            name.setFont(font)
            name.setForeground(QBrush(QColor(g.color)))
            count = QTableWidgetItem(str(len(ws.members(g))))
            count.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if g.role == "all":
                role = "all samples"
            elif g.criteria:
                role = " ".join(f"{c.connector.lower() if i else ''} {c.keyword} {c.function.lower()} “{c.value}”".strip() for i, c in enumerate(g.criteria))
                if g.sample_ids:
                    role += f" (+{len(g.sample_ids)} manual)"
            else:
                role = f"{len(g.sample_ids)} samples added manually"
            self.setItem(row, 0, name)
            self.setItem(row, 1, count)
            self.setItem(row, 2, QTableWidgetItem(role))
            if g is self.controller.current_group:
                self.selectRow(row)
        self.blockSignals(False)

    def _selection_changed(self):
        rows = {i.row() for i in self.selectedItems()}
        if rows:
            g = self.group_at(min(rows))
            if g is not None:
                self.controller.set_current_group(g)

    # drag & drop
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(POPULATION_MIME) or event.mimeData().hasFormat(SAMPLES_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        row = self.rowAt(event.position().toPoint().y())
        g = self.group_at(row)
        md = event.mimeData()
        if g is None:
            event.ignore()
            return
        if md.hasFormat(SAMPLES_MIME) and g.role == "all":
            event.ignore()
            return
        self.selectRow(row)
        event.acceptProposedAction()

    def dropEvent(self, event):
        row = self.rowAt(event.position().toPoint().y())
        g = self.group_at(row)
        md = event.mimeData()
        if g is None:
            event.ignore()
            return
        ws = self.controller.workspace
        if md.hasFormat(POPULATION_MIME):
            payload = json.loads(bytes(md.data(POPULATION_MIME).data()).decode("utf-8"))
            if isinstance(payload, dict):
                payload = [payload]
            items = []
            for entry in payload:
                sample = ws.sample_by_id(int(entry.get("sample_id", -1)))
                if sample is None:
                    continue
                for path in entry.get("paths", []):
                    pop = sample.root.find(path)
                    if pop is not None and pop.parent is not None:
                        items.append((sample, pop))
            if items:
                self.populationsDropped.emit(g, items)
            event.acceptProposedAction()
            return
        if md.hasFormat(SAMPLES_MIME):
            ids = json.loads(bytes(md.data(SAMPLES_MIME).data()).decode("utf-8"))
            samples = [s for s in (ws.sample_by_id(int(i)) for i in ids) if s is not None]
            if samples:
                self.samplesDropped.emit(g, samples)
            event.acceptProposedAction()
            return
        event.ignore()


# ----------------------------------------------------------------- sample tree
class SampleTree(QTreeWidget):
    openRequested = Signal(object, object)  # sample, population
    filesDropped = Signal(list)
    populationsDroppedOn = Signal(object, object, list)  # target sample, target population, [(sample, pop)]
    contextMenu = Signal(object, QPoint)  # item, global pos
    renamed = Signal(object, str)  # (sample|population), new name

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

    # ---- building
    def rebuild(self):
        self._building = True
        selected = {(s.id, p.path) for s, p in self.controller.selected}
        self.clear()
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
        for sample in ws.members(group):
            item = QTreeWidgetItem([sample.name, "", f"{sample.n_events:,}"])
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
            self._add_children(item, sample, sample.root, decimals)
            item.setExpanded(sample.root.expanded)
            if (sample.id, "") in selected:
                item.setSelected(True)
        self._restore_selection(selected)
        self._building = False
        self._selection_changed()

    def _add_children(self, parent_item: QTreeWidgetItem, sample: Sample, pop: Population, decimals: int):
        for st in pop.stats:
            value = compute_statistic(st.statistic, pop, st.param, st.percentile)
            row = QTreeWidgetItem([st.label, format_value(st.statistic, value, decimals), ""])
            row.setIcon(0, icon("statistic"))
            row.setData(0, ROLE_SAMPLE, sample)
            row.setData(0, ROLE_POP, pop)
            row.setData(0, ROLE_STAT, st)
            row.setForeground(0, QBrush(QColor("#374151")))
            row.setFlags((row.flags() | Qt.ItemFlag.ItemIsDragEnabled) & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsDropEnabled)
            parent_item.addChild(row)
        for child in pop.children:
            freq = child.freq_of_parent
            row = QTreeWidgetItem([child.name, format_value("Freq. of Parent", freq, decimals) + " %" if freq == freq else "", f"{child.count:,}"])
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
            self._add_children(row, sample, child, decimals)
            row.setExpanded(child.expanded)

    def _restore_selection(self, selected: set):
        if not selected:
            return
        it = QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            s = item.data(0, ROLE_SAMPLE)
            p = item.data(0, ROLE_POP)
            if s is not None and p is not None and item.data(0, ROLE_STAT) is None and (s.id, p.path) in selected:
                item.setSelected(True)
            it += 1

    def _set_expanded(self, item: QTreeWidgetItem, on: bool):
        if self._building:
            return
        pop = item.data(0, ROLE_POP)
        if pop is not None and item.data(0, ROLE_STAT) is None:
            pop.expanded = on

    def _item_changed(self, item: QTreeWidgetItem, column: int):
        if self._building or column != 0:
            return
        pop = item.data(0, ROLE_POP)
        sample = item.data(0, ROLE_SAMPLE)
        if item.data(0, ROLE_STAT) is not None or pop is None:
            return
        new_name = item.text(0).strip()
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

    def selected_stats(self) -> list[tuple[Population, StatNode]]:
        out = []
        for item in self.selectedItems():
            st = item.data(0, ROLE_STAT)
            if st is not None:
                out.append((item.data(0, ROLE_POP), st))
        return out

    def _selection_changed(self):
        if self._building:
            return
        self.controller.set_selection(self.selected_rows())

    def _double_clicked(self, item: QTreeWidgetItem, column: int):
        s = item.data(0, ROLE_SAMPLE)
        p = item.data(0, ROLE_POP)
        if s is not None and p is not None and item.data(0, ROLE_STAT) is None:
            self.openRequested.emit(s, p)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            rows = self.selected_rows()
            if rows:
                self.openRequested.emit(*rows[0])
                return
        super().keyPressEvent(event)

    def expand_all_items(self, expand: bool):
        it = QTreeWidgetItemIterator(self)
        while it.value():
            item = it.value()
            pop = item.data(0, ROLE_POP)
            if pop is not None and item.data(0, ROLE_STAT) is None:
                pop.expanded = expand
            it += 1
        self.expandAll() if expand else self.collapseAll()

    # ---- drag & drop
    def mimeTypes(self):
        return [POPULATION_MIME, SAMPLES_MIME, "text/plain"]

    def mimeData(self, items):
        rows = []
        for item in items:
            s = item.data(0, ROLE_SAMPLE)
            p = item.data(0, ROLE_POP)
            if s is not None and p is not None and item.data(0, ROLE_STAT) is None:
                rows.append((s, p))
        md = QMimeData()
        by_sample: dict[int, list[str]] = {}
        for s, p in rows:
            by_sample.setdefault(s.id, []).append(p.path)
        payload = [{"sample_id": sid, "paths": paths} for sid, paths in by_sample.items()]
        md.setData(POPULATION_MIME, QByteArray(json.dumps(payload[0] if len(payload) == 1 else payload).encode("utf-8")))
        sample_ids = sorted({s.id for s, p in rows if p.parent is None})
        if sample_ids:
            md.setData(SAMPLES_MIME, QByteArray(json.dumps(sample_ids).encode("utf-8")))
        md.setText("\n".join(f"{s.name}{p.path}" for s, p in rows))
        return md

    def startDrag(self, actions):
        items = self.selectedItems()
        md = self.mimeData(items)
        if md is None:
            return
        drag = QDrag(self)
        drag.setMimeData(md)
        first = next((i for i in items if i.data(0, ROLE_POP) is not None), None)
        if first is not None:
            drag.setPixmap(pixmap(gate_icon_name(first.data(0, ROLE_POP).gate) if first.data(0, ROLE_POP).parent is not None else "sample", 24))
        drag.exec(Qt.DropAction.CopyAction)

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
            if item is not None and item.data(0, ROLE_STAT) is None:
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        md = event.mimeData()
        if md.hasUrls():
            paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
            if paths:
                self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        if md.hasFormat(POPULATION_MIME):
            item = self.itemAt(event.position().toPoint())
            if item is None or item.data(0, ROLE_STAT) is not None:
                event.ignore()
                return
            target_sample = item.data(0, ROLE_SAMPLE)
            target_pop = item.data(0, ROLE_POP)
            payload = json.loads(bytes(md.data(POPULATION_MIME).data()).decode("utf-8"))
            if isinstance(payload, dict):
                payload = [payload]
            ws = self.controller.workspace
            dropped = []
            for entry in payload:
                sample = ws.sample_by_id(int(entry.get("sample_id", -1)))
                if sample is None:
                    continue
                for path in entry.get("paths", []):
                    pop = sample.root.find(path)
                    if pop is not None and pop.parent is not None:
                        dropped.append((sample, pop))
            if dropped:
                self.populationsDroppedOn.emit(target_sample, target_pop, dropped)
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

        self.menus: dict[str, QMenu] = {}
        self.ribbon_pages: dict = {}
        self._build_actions()
        self._build_menus()
        self._build_ribbon()
        self._build_central()
        self.statusBar().showMessage("Ready")
        self.setAcceptDrops(True)
        self._install_extensions()

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
        c.selectionChanged.connect(self._update_action_states)
        c.statusMessage.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        c.modifiedChanged.connect(lambda _m: self._update_title())
        c.preferencesChanged.connect(self._refresh_tree)
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
        self.act_add_to_table = self._act("Add to Table", "table-editor", lambda: self._add_to_table(self.controller.selected_populations()), None, "Add the selected populations as table columns")
        self.act_uncompensate = self._act("Remove Compensation", "minus", self.remove_compensation, None, "Use raw, uncompensated data for the selected samples")
        self.act_acq_comp = self._act("Use Acquisition Matrix", "compensation", self.use_acquisition_comp, None, "Use the spillover matrix stored in the FCS file")
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
        "flowdesk.ui.graph_extras",
        "flowdesk.ui.cytometer",
    )

    def _install_extensions(self):
        import importlib
        import traceback

        self.extensions: dict[str, object] = {}
        for modname in self.EXTENSION_MODULES:
            try:
                module = importlib.import_module(modname)
            except ImportError:
                continue
            install = getattr(module, "install", None)
            if install is None:
                continue
            try:
                self.extensions[modname] = install(self)
            except Exception:  # pragma: no cover - defensive
                traceback.print_exc()

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
        for a in (self.act_graph, self.act_create_group, self.act_edit_group, self.act_delete_group, self.act_add_stat, self.act_and, self.act_or, self.act_not, self.act_apply_group, self.act_remove_group, self.act_keywords, self.act_columns, self.act_expand, self.act_collapse):
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
        self.group_pane.contextMenu.connect(self._group_context_menu)
        self.tree = SampleTree(self.controller)
        self.tree.openRequested.connect(self.open_graph)
        self.tree.filesDropped.connect(self._files_dropped)
        self.tree.populationsDroppedOn.connect(self._populations_dropped_on_tree)
        self.tree.contextMenu.connect(self._tree_context_menu)
        self.tree.renamed.connect(self._renamed)
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
        self.act_graph.setEnabled(has_rows)
        self.act_remove_samples.setEnabled(bool(samples))
        self.act_add_stat.setEnabled(has_rows)
        self.act_and.setEnabled(len(pops) >= 1)
        self.act_or.setEnabled(len(pops) >= 1)
        self.act_not.setEnabled(len(pops) >= 1)
        if hasattr(self, "act_boolean"):
            self.act_boolean.setEnabled(len(pops) >= 1)
        self.act_apply_group.setEnabled(bool(pops))
        self.act_remove_group.setEnabled(bool(pops))
        self.act_copy.setEnabled(bool(pops))
        self.act_cut.setEnabled(bool(pops))
        self.act_paste.setEnabled(bool(c.clipboard) and has_rows)
        self.act_delete.setEnabled(bool(pops) or bool(self.tree.selected_stats()))
        self.act_rename.setEnabled(len(rows) == 1)
        self.act_keywords.setEnabled(len(samples) == 1)
        self.act_export_events.setEnabled(len(rows) == 1)
        self.act_export_fcs.setEnabled(has_rows)
        self.act_concat.setEnabled(len(rows) >= 2)
        self.act_add_to_layout.setEnabled(has_rows)
        self.act_add_to_table.setEnabled(has_rows)
        self.act_derive.setEnabled(bool(samples))
        self.act_uncompensate.setEnabled(bool(samples))
        self.act_acq_comp.setEnabled(bool(samples))
        self.act_undo.setEnabled(c.can_undo)
        self.act_redo.setEnabled(c.can_redo)
        self.act_undo.setText(f"Undo {c.undo_label()}" if c.can_undo else "Undo")
        g = c.current_group
        self.act_delete_group.setEnabled(g.role != "all")
        self.act_edit_group.setEnabled(g.role != "all")

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

    def _fill_window_menu(self):
        self.window_menu.clear()
        act = self.window_menu.addAction("Workspace")
        act.triggered.connect(lambda: (self.show(), self.raise_(), self.activateWindow()))
        self.window_menu.addSeparator()
        for w in list(self.graph_windows.values()):
            a = self.window_menu.addAction(w.windowTitle())
            a.triggered.connect(lambda checked=False, win=w: (win.show(), win.raise_(), win.activateWindow()))
        if self.graph_windows:
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
        c.begin_change("apply template")
        report = apply_template(ws, template)
        c.mark_modified()
        c.groupsChanged.emit()
        c.gatingChanged.emit(None)
        c.scalesChanged.emit(None)
        c.compensationChanged.emit(None)
        c.layoutsChanged.emit()
        c.tablesChanged.emit()
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
            export_wsp(ws, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {Path(path).name}", 4000)

    def export_events(self):
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
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Samples", "", "FCS files (*.fcs *.FCS);;All files (*)")
        if paths:
            self.controller.add_samples(paths)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Add all FCS files in folder")
        if folder:
            self.controller.add_folder(folder)

    def _files_dropped(self, paths: list[str]):
        fcs = [p for p in paths if p.lower().endswith(".fcs")]
        folders = [p for p in paths if Path(p).is_dir()]
        others = [p for p in paths if p.lower().endswith((EXTENSION, ".wsp", ".fdwt", ".wspt"))]
        if fcs:
            self.controller.add_samples(fcs)
        for f in folders:
            self.controller.add_folder(f)
        if others:
            self._open_path(others[0])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self._files_dropped([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])

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
        key = (sample.id, pop.id, group.id)
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
        n = len(self.graph_windows)
        pos = self.pos() + QPoint(self.width() - 200 + 24 * (n % 8), 60 + 24 * (n % 8))
        win.move(pos)

    def open_selected_graph(self):
        for s, p in self.controller.selected[:6]:
            self.open_graph(s, p)

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

    def _add_to_layout(self, sample: Sample, population: Population):
        editor = self.show_layout_editor()
        editor.add_population(sample, population)

    def _add_selected_to_layout(self):
        for s, p in self.controller.selected:
            self._add_to_layout(s, p)

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
            samples = [s for s in ws.samples if s.id in ids]
            g = self.controller.create_group(name, samples, crits, color)
            self.controller.set_current_group(g)

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
            self.controller.rename_group(g, name)
            for s in ws.members(g):
                ws.apply_template(g, s)
            self.controller.mark_modified()
            self.controller.groupsChanged.emit()
            self.controller.gatingChanged.emit(None)

    def delete_group(self):
        g = self.controller.current_group
        if g.role == "all":
            return
        if confirm(self, "Delete group", f"Delete group “{g.name}”? Samples and their gates are kept."):
            self.controller.delete_group(g)

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
                menu.addAction(self.act_delete_group)
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
        menu.exec(pos)

    def _populations_dropped_on_group(self, group: Group, items: list[tuple[Sample, Population]]):
        pops = [p for _, p in items]
        n = self.controller.copy_populations_to_group(pops, group)
        self.statusBar().showMessage(f"Applied {len(pops)} gate(s) to {n} sample(s) of “{group.name}”", 5000)

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
    def _populations_dropped_on_tree(self, target_sample: Sample, target_pop: Population, dropped: list[tuple[Sample, Population]]):
        pops = [p for s, p in dropped if not (s is target_sample and p is target_pop)]
        if pops:
            self.controller.paste_populations(pops, target_pop, None)

    def add_statistic(self):
        rows = self.controller.selected
        if not rows:
            return
        sample = rows[0][0]
        params = sample.parameter_names()
        labels = {p: sample.param_label(p) for p in params}
        dlg = AddStatisticDialog(params, labels, self.controller.current_group, self)
        if dlg.exec():
            stat, param, pct, to_group = dlg.values()
            self.controller.add_statistic([p for _, p in rows], stat, param, pct, self.controller.current_group if to_group else None)

    def create_boolean(self, op: str):
        rows = self.controller.selected
        pops = [p for _, p in rows if p.parent is not None]
        if not pops:
            return
        sample = rows[0][0]
        parent = pops[0].parent
        candidates = [p for p in sample.root.walk(False) if p.gate is None or p.gate.kind != "boolean" or True]
        dlg = BooleanGateDialog(candidates, op, self)
        dlg.check(pops)
        if dlg.exec():
            op, refs, name = dlg.values()
            if not refs or (op == "not" and len(refs) != 1) or (op != "not" and len(refs) < 2):
                QMessageBox.information(self, "Boolean gate", "Select one population for NOT, or at least two for AND / OR.")
                return
            apply_group = self.controller.current_group if self.controller.preferences.get("apply_to_group_default") else None
            self.controller.create_boolean(parent, op, refs, name or f"{op.upper()} gate", apply_group)

    def copy_populations(self):
        pops = [p for _, p in self.controller.selected if p.parent is not None]
        if pops:
            self.controller.clipboard = [p.clone(deep=True) for p in pops]
            self.statusBar().showMessage(f"Copied {len(pops)} gate(s)", 3000)
            self._update_action_states()

    def cut_populations(self):
        pops = [p for _, p in self.controller.selected if p.parent is not None]
        if pops:
            self.copy_populations()
            self.controller.delete_populations(pops, None)

    def paste_populations(self):
        rows = self.controller.selected
        if not rows or not self.controller.clipboard:
            return
        targets = list(dict.fromkeys(p for _, p in rows))
        for target in targets:
            self.controller.paste_populations(self.controller.clipboard, target, None)

    def delete_selected(self):
        stats = self.tree.selected_stats()
        for pop, st in stats:
            self.controller.remove_statistic(pop, st)
        pops = [p for _, p in self.controller.selected if p.parent is not None]
        if not pops:
            samples = self.controller.selected_samples()
            if samples and not stats:
                self.remove_samples()
            return
        n_children = sum(sum(1 for _ in p.walk(False)) for p in pops)
        text = f"Delete {len(pops)} gate(s)" + (f" and their {n_children} sub-population(s)" if n_children else "") + "?"
        if confirm(self, "Delete gates", text):
            self.controller.delete_populations(pops, None)

    def rename_selected(self):
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
            s.keywords = dlg.values()
            s.refresh_channels_from_keywords()
            self.controller.mark_modified()
            self.controller.samplesChanged.emit()
            self.controller.groupsChanged.emit()
            self.controller.gatingChanged.emit(s.id)

    def concatenate_samples(self):
        from ..core.model import concatenate_samples

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
        changed = []
        for s in samples:
            s.comp_cleared = False
            if s.ensure_loaded():
                m = SpilloverMatrix.from_fcs(s.fcs)
                if m is not None:
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
            menu.exec(pos)
            return
        if not item.isSelected():
            self.tree.clearSelection()
            item.setSelected(True)
        sample = item.data(0, ROLE_SAMPLE)
        pop = item.data(0, ROLE_POP)
        st = item.data(0, ROLE_STAT)
        if st is not None:
            act = menu.addAction(icon("delete"), "Remove statistic")
            act.triggered.connect(lambda: self.controller.remove_statistic(pop, st))
            menu.exec(pos)
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
        menu.exec(pos)

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

    def closeEvent(self, event):
        if not self.maybe_save():
            event.ignore()
            return
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
