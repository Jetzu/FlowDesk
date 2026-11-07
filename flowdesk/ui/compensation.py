"""Compensation window (matrix editor + sample status) and the compensation wizard.

The window lists the spillover matrices of the workspace together with the
acquisition-defined matrices found in the samples' $SPILLOVER keywords, shows
the selected matrix as an editable grid of spill percentages and lists which
matrix every sample uses. The wizard computes a new matrix from single-stain
controls (FlowJo-style "Compensation Wizard").
"""
from __future__ import annotations

import csv
import datetime as _dt
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from ..core.compensation import ACQUISITION_DEFINED, SpilloverMatrix, estimate_spillover
from ..core.fcs import parse_spillover
from .icons import icon

UNSTAINED = "Unstained"
NOT_USED = "— not used —"

_SCATTER_RE = re.compile(r"^(FSC|SSC|TIME|EVENT|WIDTH|HEIGHT)", re.IGNORECASE)


# ---------------------------------------------------------------- helpers
def fluorescence_params(sample) -> list[str]:
    """Fluorescence parameters of a sample: every channel except scatter, time
    and pulse width/height channels; area ("-A") channels are preferred."""
    names = list(sample.channel_names)
    fluo = []
    for n in names:
        up = n.upper()
        if _SCATTER_RE.match(up) or "TIME" in up:
            continue
        if up.endswith("-W") or up.endswith("-H"):
            continue
        fluo.append(n)
    area = [n for n in fluo if n.upper().endswith("-A")]
    return area or fluo


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text.lower()).strip()


def _token_pattern(token: str) -> re.Pattern:
    parts = [re.escape(p) for p in _norm(token).split(" ") if p]
    body = r"[\s_\-]?".join(parts)
    # word boundaries; a token followed by "-<alnum>" (e.g. "PE" in "PE-Cy7")
    # does not count as a match.
    return re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])(?!-[a-z0-9])", re.IGNORECASE)


def _param_tokens(param: str, stain: str | None) -> list[str]:
    tokens = {param}
    base = re.sub(r"-[AHW]$", "", param, flags=re.IGNORECASE)
    if base:
        tokens.add(base)
    if stain and stain.strip() and stain.strip() != param:
        tokens.add(stain.strip())
    return [t for t in tokens if len(t) >= 2]


def guess_control_assignment(sample_names: list[str], params: list[str], stains: dict[str, str]) -> dict[str, str]:
    """Guess which control each sample is: sample name -> "Unstained" | param.

    Samples that cannot be assigned are left out."""
    candidates: list[tuple[re.Pattern, int, str]] = []
    for p in params:
        for tok in _param_tokens(p, (stains or {}).get(p)):
            candidates.append((_token_pattern(tok), len(_norm(tok)), p))
    out: dict[str, str] = {}
    for name in sample_names:
        stem = re.sub(r"\.fcs$", "", name, flags=re.IGNORECASE)
        if "unstain" in stem.lower() or re.search(r"(?<![a-z])(blank|no[\s_\-]?stain)(?![a-z])", stem, re.IGNORECASE):
            out[name] = UNSTAINED
            continue
        best: tuple[int, str] | None = None
        for pat, length, p in candidates:
            if pat.search(stem) and (best is None or length > best[0]):
                best = (length, p)
        if best is not None:
            out[name] = best[1]
    return out


def compute_spillover(assignment: dict, use_unstained: bool = True, name: str = "Computed") -> SpilloverMatrix:
    """Compute a spillover matrix from control samples.

    assignment: Sample -> "Unstained" | parameter name (other values ignored).
    The matrix parameters are the assigned parameters, in channel order of the
    first control."""
    singles: dict[str, object] = {}
    unstained = None
    for sample, role in assignment.items():
        if role == UNSTAINED:
            if unstained is None:
                unstained = sample
        elif role and role != NOT_USED and role not in singles:
            singles[role] = sample
    if not singles:
        raise ValueError("No single-stain controls assigned")
    first = next(iter(singles.values()))
    order = {n: i for i, n in enumerate(first.channel_names)}
    params = sorted(singles, key=lambda p: order.get(p, len(order)))

    def columns(sample):
        if not sample.ensure_loaded() or sample.fcs is None or sample.fcs.events is None:
            raise ValueError(f"Cannot load {sample.name}")
        idx = [sample.fcs.channel_index(p) for p in params]
        return np.asarray(sample.fcs.events[:, idx], dtype=np.float64)

    controls = {p: columns(s) for p, s in singles.items()}
    unst = columns(unstained) if (use_unstained and unstained is not None) else None
    return estimate_spillover(controls, params, unst, name)


def _unique_name(existing, base: str) -> str:
    name, k = base, 2
    while name in existing:
        name = f"{base} {k}"
        k += 1
    return name


def _tint(value: float) -> QColor:
    v = min(abs(float(value)), 0.5) / 0.5
    if abs(value) < 1e-9:
        return QColor("#ffffff")
    light, strong = (234, 242, 251), (40, 100, 190)
    c = [round(a + (b - a) * v) for a, b in zip(light, strong)]
    return QColor(*c)


def fill_matrix_grid(table: QTableWidget, matrix: SpilloverMatrix | None, editable: bool = True):
    """Show a matrix as percentages (row = source fluorochrome, column = detector)."""
    blocked = table.blockSignals(True)
    try:
        if matrix is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        n = len(matrix.params)
        if table.rowCount() != n or table.columnCount() != n:
            table.setRowCount(n)
            table.setColumnCount(n)
        table.setHorizontalHeaderLabels(list(matrix.params))
        table.setVerticalHeaderLabels(list(matrix.params))
        for i in range(n):
            for j in range(n):
                v = float(matrix.matrix[i, j])
                item = table.item(i, j)
                if item is None:
                    item = QTableWidgetItem()
                    table.setItem(i, j, item)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if i == j:
                    item.setText("100.00")
                    item.setBackground(QBrush(QColor("#e4e4e4")))
                    item.setForeground(QBrush(QColor("#555555")))
                else:
                    item.setText(f"{v * 100:.2f}")
                    bg = _tint(v)
                    item.setBackground(QBrush(bg))
                    item.setForeground(QBrush(QColor("#ffffff" if bg.lightness() < 130 else "#000000")))
                    if editable:
                        flags |= Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
    finally:
        table.blockSignals(blocked)


def read_matrix_csv(path: str | Path) -> SpilloverMatrix:
    """Read a matrix CSV: header row of detector names (optionally preceded by
    an empty corner cell) followed by rows of values (fractions or percent)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read()
    try:
        dialect = csv.Sniffer().sniff(sample.splitlines()[0] if sample else ",", delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(sample.splitlines(), dialect) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("The file does not contain a matrix")
    header = [c.strip() for c in rows[0]]
    has_corner = header[0] == "" or len(header) == len(rows[1]) and not _is_number(rows[1][0])
    params = header[1:] if has_corner else header
    params = [p for p in params if p]
    n = len(params)
    values = []
    for r in rows[1 : n + 1]:
        cells = r[1:] if has_corner or not _is_number(r[0]) else r
        values.append([float(c.strip().rstrip("%")) for c in cells[:n]])
    m = np.array(values, dtype=float)
    if m.shape != (n, n):
        raise ValueError(f"Expected a {n}x{n} matrix, got {m.shape}")
    if np.allclose(np.diag(m), 100.0):
        m = m / 100.0
    return SpilloverMatrix(Path(path).stem, params, m)


def write_matrix_csv(matrix: SpilloverMatrix, path: str | Path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + list(matrix.params))
        for p, row in zip(matrix.params, matrix.matrix):
            w.writerow([p] + [f"{v:.6g}" for v in row])


def _is_number(text: str) -> bool:
    try:
        float(text.strip().rstrip("%"))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- window
class _Entry:
    __slots__ = ("label", "matrix", "stored")

    def __init__(self, label: str, matrix: SpilloverMatrix, stored: bool):
        self.label = label
        self.matrix = matrix
        self.stored = stored


class CompensationWindow(QMainWindow):
    """Matrix list, matrix editor and per-sample compensation status."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Compensation")
        self.setWindowIcon(icon("compensation"))
        self.resize(1000, 680)
        self._entries: list[_Entry] = []
        self._current_label: str | None = None
        self._wizard: CompensationWizard | None = None

        self._build_toolbar()

        self.matrix_list = QListWidget()
        self.matrix_list.setMinimumWidth(200)
        self.matrix_list.currentRowChanged.connect(self._on_matrix_selected)
        self.matrix_list.itemDoubleClicked.connect(lambda _i: self.rename_matrix())

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold;")
        self.grid = QTableWidget()
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.grid.cellChanged.connect(self._on_cell_changed)
        hint = QLabel("Rows: source fluorochrome • Columns: detector • values are % spillover")
        hint.setStyleSheet("color: #666;")
        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.addWidget(self.title_label)
        cl.addWidget(self.grid, 1)
        cl.addWidget(hint)

        self.status_table = QTableWidget(0, 3)
        self.status_table.setHorizontalHeaderLabels(["Sample", "Matrix in use", "Enabled"])
        self.status_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.status_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.status_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.status_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.status_table.cellDoubleClicked.connect(self._on_status_double_clicked)
        self.status_table.itemChanged.connect(self._on_status_item_changed)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.matrix_list)
        top.addWidget(centre)
        top.setStretchFactor(1, 1)
        top.setSizes([240, 760])
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.status_table)
        main.setSizes([430, 250])
        self.setCentralWidget(main)

        controller.compensationChanged.connect(lambda *_: self.refresh())
        controller.samplesChanged.connect(self.refresh)
        controller.workspaceReplaced.connect(self.refresh)
        controller.groupsChanged.connect(self._update_actions)
        controller.selectionChanged.connect(self._update_actions)
        self.refresh()

    # ------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        tb = QToolBar("Compensation")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(tb)

        def act(text, ic, slot, tip=""):
            a = QAction(icon(ic), text, self)
            a.triggered.connect(lambda _checked=False: slot())
            if tip:
                a.setToolTip(tip)
            return a

        self.act_new = act("New", "plus", self.new_matrix, "New identity matrix for the selected sample's parameters")
        self.act_duplicate = act("Duplicate", "copy", self.duplicate_matrix)
        self.act_rename = act("Rename", "rename", self.rename_matrix)
        self.act_delete = act("Delete", "delete", self.delete_matrix)
        self.act_import = act("Import CSV", "import", self.import_csv)
        self.act_export = act("Export CSV", "export", self.export_csv)
        self.act_wizard = act("Compensation Wizard…", "wizard", self.open_wizard)
        for a in (self.act_new, self.act_duplicate, self.act_rename, self.act_delete):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_import)
        tb.addAction(self.act_export)
        tb.addSeparator()
        tb.addAction(self.act_wizard)
        tb.addSeparator()

        self.act_apply_selected = act("Apply to selected samples", "check", self.apply_to_selected)
        self.act_apply_group = act("Apply to group…", "group", self.apply_to_group_dialog)
        self.act_apply_all = act("Apply to all samples", "play", self.apply_to_all)
        self.act_remove = act("Remove compensation from selected samples", "minus", self.remove_from_selected)
        menu = QMenu(self)
        for a in (self.act_apply_selected, self.act_apply_group, self.act_apply_all):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(self.act_remove)
        self.apply_button = QToolButton()
        self.apply_button.setText("Apply to…")
        self.apply_button.setIcon(icon("play"))
        self.apply_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.apply_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.apply_button.setMenu(menu)
        self.apply_button.clicked.connect(self.apply_to_selected)
        tb.addWidget(self.apply_button)

    def _update_actions(self):
        entry = self.current_entry()
        has = entry is not None
        self.act_duplicate.setEnabled(has)
        self.act_rename.setEnabled(has and entry.stored)
        self.act_delete.setEnabled(has and entry.stored)
        self.act_export.setEnabled(has)
        self.act_apply_selected.setEnabled(has)
        self.act_apply_group.setEnabled(has)
        self.act_apply_all.setEnabled(has)
        self.apply_button.setEnabled(True)

    # ------------------------------------------------------------ model
    @property
    def workspace(self):
        return self.controller.workspace

    def available_matrices(self) -> list[tuple[str, SpilloverMatrix, bool]]:
        """(label, matrix, stored_in_workspace) for every matrix in the list."""
        out: list[tuple[str, SpilloverMatrix, bool]] = []
        for name, m in self.workspace.matrices.items():
            out.append((name, m, True))
        acq: list[tuple[SpilloverMatrix, str]] = []
        for s in self.workspace.samples:
            found = []
            spill = parse_spillover(s.keywords) if s.keywords else None
            if spill is not None:
                found.append(SpilloverMatrix(ACQUISITION_DEFINED, spill[0], spill[1]))
            if s.comp is not None and s.comp.name == ACQUISITION_DEFINED:
                found.append(s.comp)
            for m in found:
                if not any(m == other for other, _ in acq):
                    acq.append((m, s.name))
        stored_names = set(self.workspace.matrices)
        for m, sname in acq:
            label = ACQUISITION_DEFINED if len(acq) == 1 else f"{ACQUISITION_DEFINED} ({sname})"
            label = _unique_name(stored_names, label) if label in stored_names else label
            out.append((label, m.copy(label), False))
        return out

    def matrix_labels(self) -> list[str]:
        return [e.label for e in self._entries]

    def current_entry(self) -> _Entry | None:
        row = self.matrix_list.currentRow()
        return self._entries[row] if 0 <= row < len(self._entries) else None

    def current_matrix(self) -> SpilloverMatrix | None:
        e = self.current_entry()
        return e.matrix if e is not None else None

    def select_matrix(self, label: str) -> bool:
        for i, e in enumerate(self._entries):
            if e.label == label:
                self.matrix_list.setCurrentRow(i)
                return True
        return False

    # ------------------------------------------------------------ refresh
    def refresh(self):
        self._entries = [_Entry(l, m, st) for l, m, st in self.available_matrices()]
        keep = self._current_label
        blocked = self.matrix_list.blockSignals(True)
        self.matrix_list.clear()
        row = -1
        for i, e in enumerate(self._entries):
            item = QListWidgetItem(icon("compensation" if e.stored else "sample"), e.label)
            n = len(e.matrix.params)
            item.setToolTip(f"{n}×{n}: {', '.join(e.matrix.params)}" + ("" if e.stored else "\n(read from the FCS $SPILLOVER keyword)"))
            if not e.stored:
                f = item.font()
                f.setItalic(True)
                item.setFont(f)
            self.matrix_list.addItem(item)
            if e.label == keep:
                row = i
        if row < 0 and self._entries:
            row = 0
        self.matrix_list.setCurrentRow(row)
        self.matrix_list.blockSignals(blocked)
        self._show_current()
        self._refresh_status()
        self._update_actions()

    def _show_current(self):
        e = self.current_entry()
        self._current_label = e.label if e else None
        if e is None:
            self.title_label.setText("No matrix — create one or run the Compensation Wizard")
        else:
            kind = "workspace matrix" if e.stored else "acquisition-defined, edits create a copy"
            self.title_label.setText(f"{e.label}  ({kind})")
        fill_matrix_grid(self.grid, e.matrix if e else None, editable=True)

    def _on_matrix_selected(self, _row: int):
        self._show_current()
        self._update_actions()

    def _refresh_status(self):
        t = self.status_table
        blocked = t.blockSignals(True)
        samples = self.workspace.samples
        t.setRowCount(len(samples))
        for r, s in enumerate(samples):
            it = QTableWidgetItem(icon("sample"), s.name)
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            t.setItem(r, 0, it)
            t.setItem(r, 1, QTableWidgetItem(s.comp.name if s.comp is not None else "none"))
            en = QTableWidgetItem()
            en.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            en.setCheckState(Qt.CheckState.Checked if s.comp_enabled and s.comp is not None else Qt.CheckState.Unchecked)
            t.setItem(r, 2, en)
        t.blockSignals(blocked)

    def _on_status_double_clicked(self, row: int, _col: int):
        if not (0 <= row < len(self.workspace.samples)):
            return
        comp = self.workspace.samples[row].comp
        if comp is None:
            return
        for i, e in enumerate(self._entries):
            if e.matrix.name == comp.name and e.matrix == comp:
                self.matrix_list.setCurrentRow(i)
                return
        for i, e in enumerate(self._entries):
            if e.matrix == comp:
                self.matrix_list.setCurrentRow(i)
                return

    def _on_status_item_changed(self, item: QTableWidgetItem):
        if item.column() != 2:
            return
        row = item.row()
        if not (0 <= row < len(self.workspace.samples)):
            return
        s = self.workspace.samples[row]
        enabled = item.checkState() == Qt.CheckState.Checked
        if s.comp is None:
            self._refresh_status()
            return
        if enabled != s.comp_enabled:
            self.controller.set_compensation([s], s.comp, enabled)

    # ------------------------------------------------------------ editing
    def set_cell(self, row: int, col: int, percent: float):
        """Set one spill value (in percent) of the current matrix."""
        e = self.current_entry()
        if e is None or row == col:
            return
        n = len(e.matrix.params)
        if not (0 <= row < n and 0 <= col < n):
            return
        with self.controller.undo_group('edit matrix', kind="matrices"):
            if e.stored:
                m = e.matrix.copy()
            else:
                m = e.matrix.copy(_unique_name(self.workspace.matrices, f"{e.label} (edited)"))
            m.matrix[row, col] = percent / 100.0
            self._current_label = m.name
            users = [s for s in self.workspace.samples if e.stored and s.comp is not None and s.comp.name == m.name]
            self.controller.store_matrix(m)
            # samples using a workspace matrix follow its edits
            for enabled in (True, False):
                group = [s for s in users if s.comp_enabled == enabled]
                if group:
                    self.controller.set_compensation(group, m, enabled)
            if self._current_label != (self.current_entry().label if self.current_entry() else None):
                self.select_matrix(m.name)

    def _on_cell_changed(self, row: int, col: int):
        item = self.grid.item(row, col)
        e = self.current_entry()
        if item is None or e is None:
            return
        text = item.text().strip().rstrip("%").replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            fill_matrix_grid(self.grid, e.matrix)
            return
        if abs(value / 100.0 - e.matrix.matrix[row, col]) < 1e-12:
            fill_matrix_grid(self.grid, e.matrix)
            return
        self.set_cell(row, col, value)

    # ------------------------------------------------------------ matrix actions
    def _reference_sample(self):
        sel = self.controller.selected_samples()
        if sel:
            return sel[0]
        return self.workspace.samples[0] if self.workspace.samples else None

    def new_matrix(self) -> SpilloverMatrix | None:
        s = self._reference_sample()
        if s is None:
            QMessageBox.information(self, "Compensation", "Add samples to the workspace first.")
            return None
        params = fluorescence_params(s)
        if not params:
            QMessageBox.information(self, "Compensation", f"{s.name} has no fluorescence parameters.")
            return None
        m = SpilloverMatrix.identity(params, _unique_name(self.workspace.matrices, "Matrix"))
        self._current_label = m.name
        self.controller.store_matrix(m)
        self.select_matrix(m.name)
        return m

    def duplicate_matrix(self) -> SpilloverMatrix | None:
        e = self.current_entry()
        if e is None:
            return None
        m = e.matrix.copy(_unique_name(self.workspace.matrices, f"{e.label} copy"))
        self._current_label = m.name
        self.controller.store_matrix(m)
        self.select_matrix(m.name)
        return m

    def rename_matrix(self, new_name: str | None = None):
        e = self.current_entry()
        if e is None or not e.stored:
            return
        if new_name is None:
            new_name, ok = QInputDialog.getText(self, "Rename Matrix", "Name:", QLineEdit.EchoMode.Normal, e.label)
            if not ok:
                return
        new_name = new_name.strip()
        if not new_name or new_name == e.label:
            return
        if new_name in self.workspace.matrices:
            QMessageBox.warning(self, "Rename Matrix", f"A matrix named “{new_name}” already exists.")
            return
        with self.controller.undo_group('rename matrix', kind="matrices"):
            old = e.label
            m = e.matrix.copy(new_name)
            users = [s for s in self.workspace.samples if s.comp is not None and s.comp.name == old]
            self._current_label = new_name
            self.controller.remove_matrix(old)
            self.controller.store_matrix(m)
            for enabled in (True, False):
                group = [s for s in users if s.comp_enabled == enabled]
                if group:
                    self.controller.set_compensation(group, m, enabled)
            self.select_matrix(new_name)

    def delete_matrix(self, confirm: bool = True):
        e = self.current_entry()
        if e is None or not e.stored:
            return
        if confirm:
            res = QMessageBox.question(self, "Delete Matrix", f"Delete the matrix “{e.label}”?\nSamples using it keep their copy.")
            if res != QMessageBox.StandardButton.Yes:
                return
        self._current_label = None
        self.controller.remove_matrix(e.label)

    def import_csv(self, path: str | None = None) -> SpilloverMatrix | None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import Matrix", "", "CSV files (*.csv *.txt);;All files (*)")
            if not path:
                return None
        try:
            m = read_matrix_csv(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import Matrix", f"Could not read the matrix:\n{exc}")
            return None
        m.name = _unique_name(self.workspace.matrices, m.name)
        self._current_label = m.name
        self.controller.store_matrix(m)
        self.select_matrix(m.name)
        return m

    def export_csv(self, path: str | None = None):
        m = self.current_matrix()
        if m is None:
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Matrix", f"{m.name}.csv", "CSV files (*.csv)")
            if not path:
                return
        try:
            write_matrix_csv(m, path)
        except OSError as exc:
            QMessageBox.warning(self, "Export Matrix", f"Could not write the file:\n{exc}")

    def open_wizard(self):
        self._wizard = CompensationWizard(self.controller, self)
        self._wizard.setModal(False)
        self._wizard.show()
        return self._wizard

    # ------------------------------------------------------------ applying
    def apply_to_samples(self, samples, matrix: SpilloverMatrix | None = None, enabled: bool = True):
        if matrix is None:
            matrix = self.current_matrix()
        if matrix is None or not samples:
            return
        self.controller.set_compensation(list(samples), matrix, enabled)

    def apply_to_selected(self):
        samples = self.controller.selected_samples()
        if not samples:
            rows = sorted({i.row() for i in self.status_table.selectedItems()})
            samples = [self.workspace.samples[r] for r in rows if r < len(self.workspace.samples)]
        if not samples:
            QMessageBox.information(self, "Compensation", "Select samples in the workspace (or in the table below) first.")
            return
        self.apply_to_samples(samples)

    def apply_to_group(self, group):
        self.apply_to_samples(self.workspace.members(group))

    def apply_to_group_dialog(self):
        groups = self.workspace.groups
        names = [g.name for g in groups]
        cur = self.controller.current_group
        idx = names.index(cur.name) if cur is not None and cur.name in names else 0
        name, ok = QInputDialog.getItem(self, "Apply to Group", "Group:", names, idx, False)
        if ok and name:
            g = self.workspace.group_by_name(name)
            if g is not None:
                self.apply_to_group(g)

    def apply_to_all(self):
        self.apply_to_samples(self.workspace.samples)

    def remove_from_selected(self):
        samples = self.controller.selected_samples()
        if not samples:
            rows = sorted({i.row() for i in self.status_table.selectedItems()})
            samples = [self.workspace.samples[r] for r in rows if r < len(self.workspace.samples)]
        if samples:
            self.controller.set_compensation(samples, None)

    def closeEvent(self, event):
        event.ignore()
        self.hide()


# ---------------------------------------------------------------- wizard
class _ControlsPage(QWizardPage):
    def __init__(self, wizard: "CompensationWizard"):
        super().__init__()
        self.wiz = wizard
        self.setTitle("Controls")
        self.setSubTitle("Assign each compensation control to the fluorochrome it is stained with.")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Sample", "Events", "Control for"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.universal_negative = QCheckBox("Use unstained control as the universal negative")
        self.info = QLabel()
        self.info.setStyleSheet("color: #666;")
        lay = QVBoxLayout(self)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.universal_negative)
        lay.addWidget(self.info)
        self.combos: list[QComboBox] = []

    def populate(self):
        samples = self.wiz.control_samples
        params = self.wiz.params
        stains: dict[str, str] = {}
        for s in samples:
            for ch in s.channels:
                if ch.name in params and ch.stain and ch.name not in stains:
                    stains[ch.name] = ch.stain
        guess = guess_control_assignment([s.name for s in samples], params, stains)
        self.table.setRowCount(len(samples))
        self.combos = []
        for r, s in enumerate(samples):
            self.table.setItem(r, 0, QTableWidgetItem(icon("sample"), s.name))
            ev = QTableWidgetItem(f"{s.n_events:,}")
            ev.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 1, ev)
            combo = QComboBox()
            combo.addItem(NOT_USED)
            combo.addItem(UNSTAINED)
            for p in params:
                label = f"{p} :: {stains[p]}" if p in stains and stains[p] != p else p
                combo.addItem(label, p)
            role = guess.get(s.name)
            if role == UNSTAINED:
                combo.setCurrentIndex(1)
            elif role in params:
                combo.setCurrentIndex(2 + params.index(role))
            combo.currentIndexChanged.connect(lambda _i: self._changed())
            self.table.setCellWidget(r, 2, combo)
            self.combos.append(combo)
        self.universal_negative.setChecked(any(v == UNSTAINED for v in guess.values()))
        if not self.wiz.used_comp_group:
            self.info.setText("The Compensation group is empty: showing all samples.")
        self._changed()

    def roles(self) -> list[str]:
        out = []
        for c in self.combos:
            i = c.currentIndex()
            out.append(NOT_USED if i <= 0 else UNSTAINED if i == 1 else c.itemData(i))
        return out

    def _changed(self):
        roles = self.roles()
        singles = [r for r in roles if r not in (NOT_USED, UNSTAINED)]
        dup = sorted({r for r in singles if singles.count(r) > 1})
        text = f"{len(set(singles))} single-stain control(s)"
        if UNSTAINED in roles:
            text += ", unstained control assigned"
        if dup:
            text += f". Several controls for {', '.join(dup)}: the first one is used."
        self.info.setText(text if self.wiz.used_comp_group else "The Compensation group is empty: showing all samples. " + text)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return any(r not in (NOT_USED, UNSTAINED) for r in self.roles())


class _MatrixPage(QWizardPage):
    def __init__(self, wizard: "CompensationWizard"):
        super().__init__()
        self.wiz = wizard
        self.setTitle("Matrix")
        self.setSubTitle("The spillover matrix computed from the controls (values in %).")
        self.grid = QTableWidget()
        self.grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.name_edit = QLineEdit()
        self.apply_check = QCheckBox("Apply to the group")
        self.apply_check.setChecked(True)
        self.group_combo = QComboBox()
        self.apply_check.toggled.connect(self.group_combo.setEnabled)
        self.name_edit.textChanged.connect(lambda _t: self.completeChanged.emit())
        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        row = QHBoxLayout()
        row.addWidget(self.apply_check)
        row.addWidget(self.group_combo, 1)
        lay = QVBoxLayout(self)
        lay.addWidget(self.grid, 1)
        lay.addWidget(self.message)
        lay.addLayout(form)
        lay.addLayout(row)

    def initializePage(self):
        ws = self.wiz.controller.workspace
        if not self.name_edit.text():
            self.name_edit.setText(_unique_name(ws.matrices, f"Computed {_dt.date.today().isoformat()}"))
        self.group_combo.clear()
        for g in ws.groups:
            self.group_combo.addItem(icon("group"), g.name)
        all_idx = next((i for i, g in enumerate(ws.groups) if g.role == "all"), 0)
        self.group_combo.setCurrentIndex(all_idx)
        self.wiz.compute()
        fill_matrix_grid(self.grid, self.wiz.matrix, editable=False)
        self.message.setText(self.wiz.message)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self.wiz.matrix is not None and bool(self.name_edit.text().strip())


class CompensationWizard(QWizard):
    """Compute a spillover matrix from single-stain controls."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Compensation Wizard")
        self.setWindowIcon(icon("wizard"))
        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.resize(720, 520)
        ws = controller.workspace
        comp_group = next((g for g in ws.groups if g.role == "compensation"), None)
        members = ws.members(comp_group) if comp_group is not None else []
        self.used_comp_group = bool(members)
        self.control_samples = members or list(ws.samples)
        params: list[str] = []
        for s in self.control_samples:
            for p in fluorescence_params(s):
                if p not in params:
                    params.append(p)
        self.params = params
        self.matrix: SpilloverMatrix | None = None
        self.message = ""
        self.controls_page = _ControlsPage(self)
        self.matrix_page = _MatrixPage(self)
        self.addPage(self.controls_page)
        self.addPage(self.matrix_page)
        self.controls_page.populate()

    def assignment(self) -> dict:
        """Sample -> "Unstained" | parameter for the assigned rows."""
        return {s: r for s, r in zip(self.control_samples, self.controls_page.roles()) if r != NOT_USED}

    def compute(self) -> SpilloverMatrix | None:
        name = self.matrix_page.name_edit.text().strip() or "Computed"
        try:
            self.matrix = compute_spillover(self.assignment(), self.controls_page.universal_negative.isChecked(), name)
        except (ValueError, KeyError) as exc:
            self.matrix = None
            self.message = f"Could not compute the matrix: {exc}"
            return None
        missing = [p for p in self.params if p not in self.matrix.params]
        self.message = f"{len(self.matrix.params)}×{len(self.matrix.params)} matrix computed."
        if missing:
            self.message += f" No control for: {', '.join(missing)} (not compensated)."
        return self.matrix

    def accept(self):
        if self.matrix is None:
            self.compute()
        if self.matrix is None:
            return
        ws = self.controller.workspace
        name = self.matrix_page.name_edit.text().strip() or self.matrix.name
        if name in ws.matrices:
            res = QMessageBox.question(self, "Compensation Wizard", f"Replace the existing matrix “{name}”?")
            if res != QMessageBox.StandardButton.Yes:
                return
        m = self.matrix.copy(name)
        # one undo step: storing the matrix and applying it to the group
        with self.controller.undo_group("compensation wizard", kind="matrices"):
            self.controller.store_matrix(m)
            if self.matrix_page.apply_check.isChecked():
                g = ws.group_by_name(self.matrix_page.group_combo.currentText())
                if g is not None:
                    members = ws.members(g)
                    if members:
                        self.controller.set_compensation(members, m, True)
        super().accept()
