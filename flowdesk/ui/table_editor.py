"""Table Editor: define tables of statistics (one row per sample, one column
per population statistic or keyword), compute them for a group and export
them as CSV or copy them as tab-separated text."""
from __future__ import annotations

import csv
import io
import json

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..core.fcs import FCSError
from ..core.layouts import TableColumn, TableDef
from ..core.model import Group, Workspace
from ..core.stats import NEEDS_PARAM, STATISTICS, compute_statistic, format_value, statistic_label
from .controller import POPULATION_MIME
from .icons import icon
from .layout_editor import SectionUndoRecorder

UNGATED = "Ungated"


# ---------------------------------------------------------------- pure logic
def resolve_group(workspace: Workspace, tabledef: TableDef, group: Group | None = None) -> Group:
    if group is not None:
        return group
    g = workspace.group_by_name(tabledef.group_name)
    return g if g is not None else workspace.groups[0]


def cell_value(sample, col: TableColumn, decimals: int = 1) -> str:
    if col.keyword:
        value = sample.keywords.get(col.keyword)
        if value is None and col.keyword in ("$FIL", "FILENAME", "Name"):
            value = sample.name
        return "" if value is None else str(value).strip()
    pop = sample.root.find(col.pop_path) if col.pop_path else sample.root
    if pop is None:
        return ""
    try:
        # a parameter the sample lacks (a computed tSNE/cluster parameter of
        # other samples) gives NaN -> blank; an unreadable file or a stale
        # computed array must not break the whole table either
        value = compute_statistic(col.statistic, pop, col.param, col.percentile)
    except (ValueError, KeyError, IndexError, FCSError):
        return ""
    return format_value(col.statistic, value, decimals)


def build_table(workspace: Workspace, tabledef: TableDef, group: Group | None = None, decimals: int = 1) -> tuple[list[str], list[list[str]]]:
    """Compute a table: headers ("Sample" + one per column) and one row per
    member sample of the group."""
    group = resolve_group(workspace, tabledef, group)
    headers = ["Sample"] + [c.header for c in tabledef.columns]
    rows = []
    for sample in workspace.members(group):
        rows.append([sample.name] + [cell_value(sample, c, decimals) for c in tabledef.columns])
    return headers, rows


def to_delimited(headers: list[str], rows: list[list[str]], delimiter: str = ",") -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


def write_csv(path: str, headers: list[str], rows: list[list[str]]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def keyword_names(workspace: Workspace, group: Group | None) -> list[str]:
    seen: dict[str, None] = {}
    samples = workspace.members(group) if group is not None else workspace.samples
    for s in samples:
        for k in s.keywords:
            seen.setdefault(k, None)
    # user keywords first, then $-keywords, each alphabetical
    return sorted(seen, key=lambda k: (k.startswith("$"), k.lower()))


def _unique_name(existing: list[str], base: str) -> str:
    if base not in existing:
        return base
    k = 2
    while f"{base} {k}" in existing:
        k += 1
    return f"{base} {k}"


# ------------------------------------------------------------------- dialogs
class ColumnDialog(QDialog):
    """Add / edit a statistic column."""

    def __init__(self, controller, group: Group | None, column: TableColumn | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Column" if column is not None else "Add Column")
        self.controller = controller
        ws = controller.workspace
        form = QFormLayout()

        self.pop_combo = QComboBox()
        self.pop_combo.addItem(UNGATED, "")
        for p in ws.population_paths(group):
            self.pop_combo.addItem(p.lstrip("/").replace("/", " / "), p)
        self.pop_combo.setMinimumWidth(280)
        form.addRow("Population:", self.pop_combo)

        self.stat_combo = QComboBox()
        self.stat_combo.addItems(STATISTICS)
        form.addRow("Statistic:", self.stat_combo)

        self.param_combo = QComboBox()
        for n in ws.all_parameter_names(group):
            label = n
            for s in ws.members(group) if group is not None else ws.samples:
                if n in s.parameter_names():
                    label = s.param_label(n)
                    break
            self.param_combo.addItem(label, n)
        form.addRow("Parameter:", self.param_combo)

        self.pct_spin = QDoubleSpinBox()
        self.pct_spin.setRange(0.0, 100.0)
        self.pct_spin.setDecimals(1)
        self.pct_spin.setValue(50.0)
        form.addRow("Percentile:", self.pct_spin)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("(automatic)")
        form.addRow("Label:", self.label_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

        self._column = column
        if column is not None:
            i = self.pop_combo.findData(column.pop_path)
            if i < 0:
                self.pop_combo.addItem(column.pop_path.lstrip("/").replace("/", " / "), column.pop_path)
                i = self.pop_combo.count() - 1
            self.pop_combo.setCurrentIndex(i)
            self.stat_combo.setCurrentText(column.statistic)
            if column.param:
                j = self.param_combo.findData(column.param)
                if j < 0:
                    self.param_combo.addItem(column.param, column.param)
                    j = self.param_combo.count() - 1
                self.param_combo.setCurrentIndex(j)
            if column.percentile is not None:
                self.pct_spin.setValue(float(column.percentile))
            self.label_edit.setText(column.label)
        else:
            self.stat_combo.setCurrentText("Freq. of Parent")
        self.stat_combo.currentTextChanged.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self):
        stat = self.stat_combo.currentText()
        self.param_combo.setEnabled(stat in NEEDS_PARAM)
        self.pct_spin.setEnabled(stat == "Percentile")

    def column(self) -> TableColumn:
        stat = self.stat_combo.currentText()
        col = self._column if self._column is not None else TableColumn()
        col.pop_path = self.pop_combo.currentData() or ""
        col.statistic = stat
        col.param = self.param_combo.currentData() if stat in NEEDS_PARAM else None
        col.percentile = self.pct_spin.value() if stat == "Percentile" else None
        col.keyword = None
        col.label = self.label_edit.text().strip()
        return col


class KeywordDialog(QDialog):
    """Pick one or more keywords to add as columns."""

    def __init__(self, names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Keyword Column")
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter keywords…")
        self.filter_edit.setClearButtonEnabled(True)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for n in names:
            self.list.addItem(QListWidgetItem(icon("keyword"), n))
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        self.filter_edit.textChanged.connect(self._filter)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.filter_edit)
        lay.addWidget(self.list)
        lay.addWidget(buttons)
        self.resize(320, 420)

    def _filter(self, text: str):
        t = text.lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(t not in item.text().lower())

    def selected_keywords(self) -> list[str]:
        return [i.text() for i in self.list.selectedItems() if not i.isHidden()]


# --------------------------------------------------------------- result view
def _fill_table_widget(widget: QTableWidget, headers: list[str], rows: list[list[str]]):
    widget.clear()
    widget.setColumnCount(len(headers))
    widget.setRowCount(len(rows))
    widget.setHorizontalHeaderLabels(headers)
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            item = QTableWidgetItem(text)
            item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            if c > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            widget.setItem(r, c, item)
    widget.resizeColumnsToContents()


class TableResultWindow(QMainWindow):
    """Shows a computed table with Export CSV / Copy / Close."""

    def __init__(self, headers: list[str], rows: list[list[str]], title: str = "Table", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.headers = list(headers)
        self.rows = [list(r) for r in rows]
        self._title = title

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(8, 8, 8, 8)
        self.info = QLabel()
        lay.addWidget(self.info)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        lay.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.export_button = QPushButton(icon("export"), "Export CSV…")
        self.copy_button = QPushButton(icon("copy"), "Copy")
        self.close_button = QPushButton("Close")
        self.export_button.clicked.connect(self.export_dialog)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.close_button.clicked.connect(self.close)
        for b in (self.export_button, self.copy_button, self.close_button):
            buttons.addWidget(b)
        lay.addLayout(buttons)
        self.setCentralWidget(central)
        copy_act = QAction(self)
        copy_act.setShortcut(QKeySequence.StandardKey.Copy)
        copy_act.triggered.connect(self.copy_to_clipboard)
        self.addAction(copy_act)
        self.set_data(headers, rows)
        self.resize(760, 420)

    def set_data(self, headers: list[str], rows: list[list[str]]):
        self.headers = list(headers)
        self.rows = [list(r) for r in rows]
        _fill_table_widget(self.table, self.headers, self.rows)
        self.info.setText(f"{len(self.rows)} sample(s), {max(len(self.headers) - 1, 0)} column(s)")

    def tsv(self) -> str:
        return to_delimited(self.headers, self.rows, "\t")

    def copy_to_clipboard(self):
        QGuiApplication.clipboard().setText(self.tsv())

    def export_csv(self, path: str):
        write_csv(path, self.headers, self.rows)

    def export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Table", f"{self._title}.csv", "CSV files (*.csv);;All files (*)")
        if path:
            try:
                self.export_csv(path)
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))


# --------------------------------------------------------- column definitions
class ColumnTable(QTableWidget):
    """The table definition view; accepts populations dragged from the workspace."""

    HEADERS = ["Population", "Statistic", "Parameter", "Label"]

    def __init__(self, editor: "TableEditorWindow"):
        super().__init__(0, len(self.HEADERS))
        self.editor = editor
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed | QAbstractItemView.EditTrigger.SelectedClicked)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setDropIndicatorShown(True)

    def _accepts(self, event) -> bool:
        return event.mimeData().hasFormat(POPULATION_MIME)

    def dragEnterEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if not self._accepts(event):
            super().dropEvent(event)
            return
        paths = parse_population_mime(bytes(event.mimeData().data(POPULATION_MIME)))
        if paths:
            self.editor.add_columns_for_paths(paths)
        event.acceptProposedAction()


def parse_population_mime(payload: bytes) -> list[str]:
    """Paths from a POPULATION_MIME payload ({"sample_id", "paths"} or a list of those)."""
    try:
        data = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return []
    items = data if isinstance(data, list) else [data]
    paths: list[str] = []
    for d in items:
        if isinstance(d, dict):
            for p in d.get("paths", []) or []:
                if isinstance(p, str) and p not in paths:
                    paths.append(p)
        elif isinstance(d, str) and d not in paths:
            paths.append(d)
    return paths


# --------------------------------------------------------------- the editor
class TableEditorWindow(QMainWindow):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Table Editor")
        self.setWindowIcon(icon("table-editor"))
        self._updating = False
        self._current_id: str | None = None
        self.result_windows: list[TableResultWindow] = []

        self._build_actions()
        self._build_ui()
        self.undo_recorder = SectionUndoRecorder(controller, "tables", "edit table")

        c = controller
        c.tablesChanged.connect(self.refresh)
        c.workspaceReplaced.connect(self._on_workspace_replaced)
        c.groupsChanged.connect(self._refresh_groups)
        c.samplesChanged.connect(self._refresh_groups)

        self.ensure_default_table()
        self.refresh()
        self.resize(900, 520)

    # ---------------------------------------------------------------- ui
    def _act(self, icon_name: str, text: str, slot, tip: str = "") -> QAction:
        a = QAction(icon(icon_name), text, self)
        a.setToolTip(tip or text)
        a.triggered.connect(slot)
        return a

    def _build_actions(self):
        self.act_new = self._act("new", "New Table", self.new_table)
        self.act_rename = self._act("rename", "Rename", self.rename_table)
        self.act_duplicate = self._act("copy", "Duplicate", self.duplicate_table)
        self.act_delete = self._act("delete", "Delete Table", self.delete_table)
        self.act_add_col = self._act("plus", "Add Column", self.add_column_dialog, "Add a statistic column")
        self.act_add_kw = self._act("keyword", "Add Keyword", self.add_keyword_dialog, "Add keyword column(s)")
        self.act_edit_col = self._act("statistic", "Edit Column", self.edit_column_dialog)
        self.act_remove_col = self._act("minus", "Remove Column", self.remove_columns)
        self.act_remove_col.setShortcut(QKeySequence.StandardKey.Delete)
        self.act_up = self._act("up", "Move Up", lambda: self.move_columns(-1))
        self.act_down = self._act("down", "Move Down", lambda: self.move_columns(1))
        self.act_create = self._act("play", "Create Table", self.create_table, "Compute the table for the chosen group")
        self.act_export = self._act("export", "Export…", self.export_dialog, "Compute and export as CSV")
        self.act_copy = self._act("copy", "Copy", self.copy_to_clipboard, "Compute and copy as tab-separated text")

    def _build_ui(self):
        tb = QToolBar("Table Editor")
        tb.setMovable(False)
        tb.setIconSize(QSize(24, 24))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        for a in (self.act_new, self.act_rename, self.act_duplicate, self.act_delete):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_add_col, self.act_add_kw, self.act_edit_col, self.act_remove_col, self.act_up, self.act_down):
            tb.addAction(a)
        tb.addSeparator()
        group_box = QWidget()
        gl = QVBoxLayout(group_box)
        gl.setContentsMargins(6, 0, 6, 0)
        gl.setSpacing(2)
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(150)
        self.group_combo.setToolTip("Group whose samples become the rows of the table")
        self.group_combo.currentIndexChanged.connect(self._on_group_changed)
        gl.addWidget(self.group_combo)
        glabel = QLabel("Group")
        glabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        gl.addWidget(glabel)
        tb.addWidget(group_box)
        tb.addAction(self.act_create)
        btn = tb.widgetForAction(self.act_create)
        if btn is not None:
            btn.setIconSize(QSize(32, 32))
            f = btn.font()
            f.setBold(True)
            btn.setFont(f)
        tb.addSeparator()
        tb.addAction(self.act_export)
        tb.addAction(self.act_copy)
        self.toolbar = tb
        self.addToolBar(tb)

        self.table_list = QListWidget()
        self.table_list.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.table_list.currentItemChanged.connect(self._on_table_selected)
        self.table_list.itemChanged.connect(self._on_table_renamed)

        self.columns = ColumnTable(self)
        self.columns.itemDoubleClicked.connect(self._on_column_double_clicked)
        self.columns.itemChanged.connect(self._on_column_item_changed)
        self.columns.itemSelectionChanged.connect(self._update_actions)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Tables"))
        ll.addWidget(self.table_list)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.hint = QLabel("Drag populations from the workspace here, or use Add Column.")
        self.hint.setEnabled(False)
        rl.addWidget(self.hint)
        rl.addWidget(self.columns)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([200, 700])
        self.setCentralWidget(split)
        self.statusBar()

    # --------------------------------------------------------------- state
    @property
    def workspace(self) -> Workspace:
        return self.controller.workspace

    def current_table(self) -> TableDef | None:
        tables = self.workspace.tables
        for t in tables:
            if t.id == self._current_id:
                return t
        return tables[0] if tables else None

    def set_current_table(self, tabledef: TableDef):
        self._current_id = tabledef.id
        self.refresh()

    def current_group(self) -> Group:
        t = self.current_table()
        if t is not None:
            g = self.workspace.group_by_name(t.group_name)
            if g is not None:
                return g
        return self.controller.current_group

    def ensure_default_table(self):
        if not self.workspace.tables:
            t = TableDef(name="Table", group_name=self.controller.current_group.name)
            self.workspace.tables.append(t)
            self._current_id = t.id
            self.undo_recorder.reset()  # the default table is the baseline, not an undoable edit
            self.controller.notify_tables()

    def showEvent(self, event):
        self.ensure_default_table()
        super().showEvent(event)

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def _on_workspace_replaced(self):
        self.undo_recorder.reset()
        self._current_id = None
        for w in self.result_windows:
            w.close()
        self.result_windows.clear()
        if self.isVisible():
            self.ensure_default_table()
        self.refresh()

    # ------------------------------------------------------------- refresh
    def refresh(self):
        # after our own edits the baseline is already current; after undo /
        # redo or a change made elsewhere the model is the truth now
        self.undo_recorder.reset()
        self._updating = True
        try:
            tables = self.workspace.tables
            cur = self.current_table()
            self._current_id = cur.id if cur is not None else None
            self.table_list.clear()
            for t in tables:
                item = QListWidgetItem(icon("table-editor"), t.name)
                item.setData(Qt.ItemDataRole.UserRole, t.id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.table_list.addItem(item)
                if cur is not None and t.id == cur.id:
                    self.table_list.setCurrentItem(item)
        finally:
            self._updating = False
        self._refresh_groups()
        self._refresh_columns()

    def _refresh_groups(self):
        self._updating = True
        try:
            t = self.current_table()
            wanted = t.group_name if t is not None else self.controller.current_group.name
            self.group_combo.clear()
            for g in self.workspace.groups:
                n = len(self.workspace.members(g))
                self.group_combo.addItem(icon("group"), f"{g.name} ({n})", g.name)
            i = self.group_combo.findData(wanted)
            if i < 0:
                i = self.group_combo.findData(self.controller.current_group.name)
            self.group_combo.setCurrentIndex(max(i, 0))
        finally:
            self._updating = False

    def _refresh_columns(self, select_rows: list[int] | None = None):
        if select_rows is None:
            select_rows = sorted({i.row() for i in self.columns.selectedIndexes()})
        self._updating = True
        try:
            t = self.current_table()
            cols = t.columns if t is not None else []
            self.columns.setRowCount(len(cols))
            ro = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            for r, col in enumerate(cols):
                if col.keyword:
                    cells = [("Keyword", "keyword"), (col.keyword, None), ("", None)]
                else:
                    pop = col.pop_path.lstrip("/").replace("/", " / ") if col.pop_path else UNGATED
                    stat = statistic_label(col.statistic, None, col.percentile)
                    cells = [(pop, "statistic"), (stat, None), (col.param or "", None)]
                for c, (text, ic) in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setFlags(ro)
                    if ic:
                        item.setIcon(icon(ic))
                    self.columns.setItem(r, c, item)
                label = QTableWidgetItem(col.label)
                label.setFlags(ro | Qt.ItemFlag.ItemIsEditable)
                label.setToolTip(f"Header: {col.header}")
                self.columns.setItem(r, 3, label)
            self.columns.clearSelection()
            for r in select_rows:
                if 0 <= r < len(cols):
                    self._add_row_selection(r)
            self.hint.setVisible(not cols)
        finally:
            self._updating = False
        self._update_actions()

    def _add_row_selection(self, row: int):
        from PySide6.QtCore import QItemSelectionModel

        idx = self.columns.model().index(row, 0)
        self.columns.selectionModel().select(idx, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def _update_actions(self):
        has_table = self.current_table() is not None
        rows = self.selected_rows()
        for a in (self.act_rename, self.act_duplicate, self.act_delete, self.act_add_col, self.act_add_kw, self.act_create, self.act_export, self.act_copy):
            a.setEnabled(has_table)
        self.act_edit_col.setEnabled(len(rows) == 1)
        self.act_remove_col.setEnabled(bool(rows))
        self.act_up.setEnabled(bool(rows))
        self.act_down.setEnabled(bool(rows))

    def selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.columns.selectedIndexes()})

    def _commit(self, select_rows: list[int] | None = None):
        self.undo_recorder.record()  # one undo step per edit
        self.controller.notify_tables()  # triggers refresh()
        if select_rows is not None:
            self._refresh_columns(select_rows)

    # ------------------------------------------------------------- tables
    def _on_table_selected(self, item, _prev=None):
        if self._updating or item is None:
            return
        self._current_id = item.data(Qt.ItemDataRole.UserRole)
        self._refresh_groups()
        self._refresh_columns([])

    def _on_table_renamed(self, item: QListWidgetItem):
        if self._updating:
            return
        tid = item.data(Qt.ItemDataRole.UserRole)
        t = next((x for x in self.workspace.tables if x.id == tid), None)
        name = item.text().strip()
        if t is None or not name or name == t.name:
            self.refresh()
            return
        t.name = name
        self._commit()

    def new_table(self) -> TableDef:
        name = _unique_name([t.name for t in self.workspace.tables], "Table")
        t = TableDef(name=name, group_name=self.controller.current_group.name)
        self.workspace.tables.append(t)
        self._current_id = t.id
        self._commit()
        return t

    def rename_table(self):
        item = self.table_list.currentItem()
        if item is not None:
            self.table_list.editItem(item)

    def duplicate_table(self) -> TableDef | None:
        t = self.current_table()
        if t is None:
            return None
        d = TableDef.from_dict(t.to_dict())
        from ..core.gates import new_id

        d.id = new_id()
        for c in d.columns:
            c.id = new_id()
        d.name = _unique_name([x.name for x in self.workspace.tables], f"{t.name} copy")
        self.workspace.tables.insert(self.workspace.tables.index(t) + 1, d)
        self._current_id = d.id
        self._commit()
        return d

    def delete_table(self):
        t = self.current_table()
        if t is None:
            return
        i = self.workspace.tables.index(t)
        self.workspace.tables.remove(t)
        rest = self.workspace.tables
        self._current_id = rest[min(i, len(rest) - 1)].id if rest else None
        self._commit()

    def _on_group_changed(self, _index: int):
        if self._updating:
            return
        t = self.current_table()
        name = self.group_combo.currentData()
        if t is None or not name or name == t.group_name:
            return
        t.group_name = name
        self._commit()

    # ------------------------------------------------------------ columns
    def _ensure_table(self) -> TableDef:
        t = self.current_table()
        if t is None:
            t = TableDef(name="Table", group_name=self.controller.current_group.name)
            self.workspace.tables.append(t)
            self._current_id = t.id
        return t

    def add_columns_for_paths(self, paths: list[str], statistic: str = "Freq. of Parent", param: str | None = None) -> list[TableColumn]:
        """Append one column per population path (Count for "" = Ungated)."""
        t = self._ensure_table()
        added = []
        for p in paths:
            p = p or ""
            if p and not p.startswith("/"):
                p = "/" + p
            stat = statistic
            if p == "" and statistic in ("Freq. of Parent", "Freq. of Grandparent"):
                stat = "Count"
            col = TableColumn(pop_path=p, statistic=stat, param=param if stat in NEEDS_PARAM else None)
            t.columns.append(col)
            added.append(col)
        if added:
            n = len(t.columns)
            self._commit(list(range(n - len(added), n)))
            self.statusBar().showMessage(f"Added {len(added)} column(s) to {t.name}", 3000)
        return added

    def add_column_dialog(self):
        t = self._ensure_table()
        dlg = ColumnDialog(self.controller, self.current_group(), None, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            t.columns.append(dlg.column())
            self._commit([len(t.columns) - 1])

    def add_keyword_dialog(self):
        t = self._ensure_table()
        names = keyword_names(self.workspace, self.current_group())
        dlg = KeywordDialog(names, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.add_keyword_columns(dlg.selected_keywords())

    def add_keyword_columns(self, keywords: list[str]) -> list[TableColumn]:
        t = self._ensure_table()
        added = [TableColumn(keyword=k) for k in keywords if k]
        if added:
            t.columns.extend(added)
            n = len(t.columns)
            self._commit(list(range(n - len(added), n)))
        return added

    def _on_column_double_clicked(self, item: QTableWidgetItem):
        if item.column() == 3:
            return  # label is edited in place
        self.edit_column_dialog(item.row())

    def edit_column_dialog(self, row: int | None = None):
        t = self.current_table()
        if t is None:
            return
        if row is None or isinstance(row, bool):
            rows = self.selected_rows()
            if len(rows) != 1:
                return
            row = rows[0]
        if not 0 <= row < len(t.columns):
            return
        col = t.columns[row]
        if col.keyword:
            names = keyword_names(self.workspace, self.current_group())
            dlg = KeywordDialog(names, self)
            dlg.setWindowTitle("Edit Keyword Column")
            dlg.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            for i in range(dlg.list.count()):
                if dlg.list.item(i).text() == col.keyword:
                    dlg.list.setCurrentRow(i)
            if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_keywords():
                col.keyword = dlg.selected_keywords()[0]
                self._commit([row])
            return
        dlg = ColumnDialog(self.controller, self.current_group(), col, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            dlg.column()
            self._commit([row])

    def _on_column_item_changed(self, item: QTableWidgetItem):
        if self._updating or item.column() != 3:
            return
        t = self.current_table()
        if t is None or not 0 <= item.row() < len(t.columns):
            return
        t.columns[item.row()].label = item.text().strip()
        self._commit([item.row()])

    def remove_columns(self):
        t = self.current_table()
        rows = self.selected_rows()
        if t is None or not rows:
            return
        for r in reversed(rows):
            if 0 <= r < len(t.columns):
                del t.columns[r]
        self._commit([min(rows[0], len(t.columns) - 1)] if t.columns else [])

    def move_columns(self, delta: int):
        t = self.current_table()
        rows = self.selected_rows()
        if t is None or not rows:
            return
        cols = t.columns
        if (delta < 0 and rows[0] == 0) or (delta > 0 and rows[-1] == len(cols) - 1):
            return
        order = rows if delta < 0 else list(reversed(rows))
        for r in order:
            cols[r], cols[r + delta] = cols[r + delta], cols[r]
        self._commit([r + delta for r in rows])

    # ------------------------------------------------------------- output
    def build(self) -> tuple[list[str], list[list[str]]]:
        t = self.current_table()
        if t is None:
            return ["Sample"], []
        decimals = int(self.controller.preferences.get("freq_decimals", 1) or 1)
        return build_table(self.workspace, t, self.current_group(), decimals)

    def create_table(self) -> TableResultWindow | None:
        t = self.current_table()
        if t is None:
            return None
        headers, rows = self.build()
        title = f"{t.name} — {self.current_group().name}"
        win = TableResultWindow(headers, rows, title, self)
        win.setWindowFlag(Qt.WindowType.Window, True)
        self.result_windows = [w for w in self.result_windows if w.isVisible()]
        self.result_windows.append(win)
        win.show()
        win.raise_()
        return win

    def export_csv(self, path: str):
        headers, rows = self.build()
        write_csv(path, headers, rows)

    def export_dialog(self):
        t = self.current_table()
        if t is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Table", f"{t.name}.csv", "CSV files (*.csv);;All files (*)")
        if path:
            try:
                self.export_csv(path)
                self.statusBar().showMessage(f"Exported {path}", 4000)
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def copy_to_clipboard(self):
        headers, rows = self.build()
        QGuiApplication.clipboard().setText(to_delimited(headers, rows, "\t"))
        self.statusBar().showMessage(f"Copied {len(rows)} row(s)", 3000)
