"""Dialogs shared by the workspace and graph windows."""
from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__
from ..core.model import CRITERION_FUNCTIONS, PLOT_TYPES, Criterion, GraphSettings, Group, Population, Sample
from ..core.stats import NEEDS_PARAM, STATISTICS
from ..core.transforms import SCALE_KINDS, ArcsinhScale, LinearScale, LogScale, LogicleScale, Scale
from .renderer import RenderOptions, render_plot

WIDTH_BASIS_PRESETS = [-10, -31.6, -100, -316, -1000, -3162, -10000]


# -------------------------------------------------------------------- gate name
class GateNameDialog(QDialog):
    def __init__(self, default_name: str, group: Group | None = None, apply_default: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Name the population")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(default_name)
        self.name_edit.selectAll()
        form.addRow("Name:", self.name_edit)
        layout.addLayout(form)
        self.apply_check = QCheckBox(f"Apply to all samples in group “{group.name}”" if group is not None else "Apply to group")
        self.apply_check.setChecked(bool(apply_default) and group is not None)
        self.apply_check.setEnabled(group is not None)
        layout.addWidget(self.apply_check)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.name_edit.setFocus()

    def values(self) -> tuple[str, bool]:
        return self.name_edit.text().strip(), self.apply_check.isChecked()


# ------------------------------------------------------------------ axis scale
def preview_pixmap(sample: Sample, param: str, scale: Scale, width: int, height: int) -> QPixmap:
    old = sample.scales.get(param)
    sample.scales[param] = scale
    sample._display_cache.pop(param, None)
    try:
        opts = RenderOptions(show_gates=False, number_font_size=8, label_font_size=9, smoothing=1, histogram_bins=200, font_dpi=72.0)
        result = render_plot(sample, sample.root, GraphSettings(param, None, "histogram"), width, height, opts)
        return QPixmap.fromImage(result.image)
    finally:
        if old is None:
            sample.scales.pop(param, None)
        else:
            sample.scales[param] = old
        sample._display_cache.pop(param, None)


class ScaleDialog(QDialog):
    """Customize the axis scale of one parameter (the "T" button)."""

    def __init__(self, sample: Sample, param: str, scale: Scale, group: Group | None = None, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.param = param
        self.setWindowTitle(f"Customize Axis – {param}")
        self.resize(560, 420)
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        layout.addLayout(left, 0)
        self.preview = QLabel()
        self.preview.setMinimumSize(300, 260)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)

        form = QFormLayout()
        self.kind = QComboBox()
        for key, label in SCALE_KINDS:
            self.kind.addItem(label, key)
        form.addRow("Scale:", self.kind)
        left.addLayout(form)

        self.stack = QStackedWidget()
        left.addWidget(self.stack)
        # linear
        w = QWidget()
        f = QFormLayout(w)
        self.lin_min = _spin(-1e9, 1e9, 0, decimals=4)
        self.lin_max = _spin(-1e9, 1e9, 262144, decimals=4)
        f.addRow("Minimum:", self.lin_min)
        f.addRow("Maximum:", self.lin_max)
        self.stack.addWidget(w)
        # logicle / biex
        w = QWidget()
        f = QFormLayout(w)
        self.biex_pos = _spin(1.0, 8.0, 4.5, 2)
        self.biex_width = QComboBox()
        self.biex_width.setEditable(True)
        for v in WIDTH_BASIS_PRESETS:
            self.biex_width.addItem(f"{v:g}")
        self.biex_neg = _spin(0.0, 4.0, 0.0, 2)
        self.biex_max = _spin(1.0, 1e9, 262144)
        f.addRow("Positive decades:", self.biex_pos)
        f.addRow("Width basis:", self.biex_width)
        f.addRow("Extra negative decades:", self.biex_neg)
        f.addRow("Maximum:", self.biex_max)
        self.stack.addWidget(w)
        # log
        w = QWidget()
        f = QFormLayout(w)
        self.log_min = _spin(1e-6, 1e9, 1)
        self.log_max = _spin(1e-6, 1e9, 262144)
        f.addRow("Minimum (>0):", self.log_min)
        f.addRow("Maximum:", self.log_max)
        self.stack.addWidget(w)
        # arcsinh
        w = QWidget()
        f = QFormLayout(w)
        self.as_m = _spin(1.0, 8.0, 4.5, 2)
        self.as_a = _spin(0.0, 4.0, 0.0, 2)
        self.as_t = _spin(1.0, 1e9, 262144)
        f.addRow("Positive decades:", self.as_m)
        f.addRow("Negative decades:", self.as_a)
        f.addRow("Maximum:", self.as_t)
        self.stack.addWidget(w)

        self.apply_group = QCheckBox(f"Apply to all samples in “{group.name}”" if group is not None else "Apply to this sample only")
        self.apply_group.setChecked(group is not None)
        self.apply_group.setEnabled(group is not None)
        left.addWidget(self.apply_group)
        self.apply_all_params = QCheckBox("Apply to all fluorescence parameters")
        left.addWidget(self.apply_all_params)
        left.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        left.addWidget(buttons)

        self._load(scale)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        for wdg in (self.lin_min, self.lin_max, self.biex_pos, self.biex_neg, self.biex_max, self.log_min, self.log_max, self.as_m, self.as_a, self.as_t):
            wdg.valueChanged.connect(self._schedule_preview)
        self.biex_width.currentTextChanged.connect(self._schedule_preview)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._update_preview)
        self._update_preview()

    def _load(self, scale: Scale):
        kind = scale.kind
        idx = max(0, [k for k, _ in SCALE_KINDS].index(kind) if kind in [k for k, _ in SCALE_KINDS] else 0)
        self.kind.setCurrentIndex(idx)
        self.stack.setCurrentIndex(idx)
        if isinstance(scale, LinearScale):
            self.lin_min.setValue(scale.min)
            self.lin_max.setValue(scale.max)
        elif isinstance(scale, LogicleScale):
            self.biex_pos.setValue(scale.M)
            self.biex_width.setCurrentText(f"{scale.width_basis:.4g}")
            self.biex_neg.setValue(scale.A)
            self.biex_max.setValue(scale.T)
        elif isinstance(scale, LogScale):
            self.log_min.setValue(scale.min)
            self.log_max.setValue(scale.max)
        elif isinstance(scale, ArcsinhScale):
            self.as_m.setValue(scale.M)
            self.as_a.setValue(scale.A)
            self.as_t.setValue(scale.T)
        ch = self.sample.base_channel(self.param)
        top = ch.range if ch is not None and ch.range > 1 else 262144.0
        if not isinstance(scale, LinearScale):
            self.lin_max.setValue(top)
        if not isinstance(scale, LogicleScale):
            self.biex_max.setValue(top)
            self.biex_pos.setValue(max(3.0, math.log10(top) - 1.0))
            self.biex_width.setCurrentText("-100")
        if not isinstance(scale, LogScale):
            self.log_max.setValue(top)
        if not isinstance(scale, ArcsinhScale):
            self.as_t.setValue(top)

    def _kind_changed(self, idx: int):
        self.stack.setCurrentIndex(idx)
        self._schedule_preview()

    def _schedule_preview(self, *_):
        self._timer.start()

    def scale(self) -> Scale:
        kind = self.kind.currentData()
        if kind == "linear":
            return LinearScale(self.lin_min.value(), self.lin_max.value())
        if kind == "log":
            return LogScale(self.log_min.value(), self.log_max.value())
        if kind == "arcsinh":
            return ArcsinhScale(self.as_t.value(), self.as_m.value(), self.as_a.value())
        try:
            wb = float(self.biex_width.currentText())
        except ValueError:
            wb = -100.0
        if wb >= 0:
            wb = -abs(wb) if wb else -10.0
        return LogicleScale.from_biex(self.biex_max.value(), self.biex_pos.value(), self.biex_neg.value(), wb)

    def _update_preview(self):
        try:
            pm = preview_pixmap(self.sample, self.param, self.scale(), max(300, self.preview.width()), max(240, self.preview.height()))
            self.preview.setPixmap(pm)
        except Exception as exc:  # pragma: no cover - preview must never crash the dialog
            self.preview.setText(str(exc))

    def apply_to_group(self) -> bool:
        return self.apply_group.isChecked()

    def apply_to_all_params(self) -> bool:
        return self.apply_all_params.isChecked()


def _spin(lo, hi, value, decimals=0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(value)
    s.setSingleStep(0.1 if decimals else max(1.0, abs(value) / 10 if value else 1.0))
    s.setKeyboardTracking(False)
    return s


# -------------------------------------------------------------------- statistic
class AddStatisticDialog(QDialog):
    def __init__(self, params: list[str], labels: dict[str, str] | None = None, group: Group | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Statistic")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.stat = QComboBox()
        self.stat.addItems(STATISTICS)
        self.stat.setCurrentText("Median")
        self.param = QComboBox()
        for p in params:
            self.param.addItem((labels or {}).get(p, p), p)
        self.percentile = QDoubleSpinBox()
        self.percentile.setRange(0, 100)
        self.percentile.setValue(95)
        form.addRow("Statistic:", self.stat)
        form.addRow("Parameter:", self.param)
        form.addRow("Percentile:", self.percentile)
        layout.addLayout(form)
        self.apply_group = QCheckBox(f"Add to all samples in “{group.name}”" if group else "Add to group")
        self.apply_group.setChecked(group is not None)
        self.apply_group.setEnabled(group is not None)
        layout.addWidget(self.apply_group)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.stat.currentTextChanged.connect(self._update)
        self._update()

    def _update(self):
        name = self.stat.currentText()
        self.param.setEnabled(name in NEEDS_PARAM)
        self.percentile.setEnabled(name == "Percentile")

    def values(self) -> tuple[str, str | None, float | None, bool]:
        name = self.stat.currentText()
        param = self.param.currentData() if name in NEEDS_PARAM else None
        pct = self.percentile.value() if name == "Percentile" else None
        return name, param, pct, self.apply_group.isChecked()


# ---------------------------------------------------------------- boolean gate
class BooleanGateDialog(QDialog):
    def __init__(self, candidates: list[Population], op: str = "and", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Boolean Gate")
        self.resize(420, 420)
        layout = QVBoxLayout(self)
        ops = QHBoxLayout()
        self.op_and = QRadioButton("AND")
        self.op_or = QRadioButton("OR")
        self.op_not = QRadioButton("NOT")
        {"and": self.op_and, "or": self.op_or, "not": self.op_not}[op].setChecked(True)
        for b in (self.op_and, self.op_or, self.op_not):
            ops.addWidget(b)
            b.toggled.connect(self._update_name)
        ops.addStretch(1)
        layout.addLayout(ops)
        layout.addWidget(QLabel("Populations:"))
        self.list = QListWidget()
        for pop in candidates:
            item = QListWidgetItem(pop.path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, pop)
            self.list.addItem(item)
        self.list.itemChanged.connect(self._update_name)
        layout.addWidget(self.list, 1)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def check(self, pops: list[Population]):
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) in pops:
                item.setCheckState(Qt.CheckState.Checked)
        self._update_name()

    def op(self) -> str:
        return "or" if self.op_or.isChecked() else "not" if self.op_not.isChecked() else "and"

    def selected(self) -> list[Population]:
        out = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _update_name(self, *_):
        names = [p.name for p in self.selected()]
        op = self.op()
        if op == "not":
            text = f"NOT {names[0]}" if names else ""
        else:
            text = f" {op.upper()} ".join(names)
        self.name_edit.setText(text)

    def values(self) -> tuple[str, list[str], str]:
        return self.op(), [p.path for p in self.selected()], self.name_edit.text().strip()


# ----------------------------------------------------------------------- group
class GroupDialog(QDialog):
    def __init__(self, samples: list[Sample], keywords: list[str], group: Group | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Group" if group else "Create Group")
        self.resize(520, 480)
        self._color = group.color if group else "#000000"
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(group.name if group else "Group")
        form.addRow("Name:", self.name_edit)
        color_row = QHBoxLayout()
        self.color_button = QPushButton()
        self._update_color_button()
        self.color_button.clicked.connect(self._pick_color)
        color_row.addWidget(self.color_button)
        color_row.addStretch(1)
        form.addRow("Color:", color_row)
        layout.addLayout(form)

        box = QGroupBox("Samples (manual membership)")
        bl = QVBoxLayout(box)
        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for s in samples:
            item = QListWidgetItem(s.name)
            item.setData(Qt.ItemDataRole.UserRole, s.id)
            self.sample_list.addItem(item)
            if group is not None and s.id in group.sample_ids:
                item.setSelected(True)
        bl.addWidget(self.sample_list)
        layout.addWidget(box, 1)

        crit_box = QGroupBox("Automatic membership by keyword (live)")
        cl = QVBoxLayout(crit_box)
        self.crit_table = QTableWidget(0, 4)
        self.crit_table.setHorizontalHeaderLabels(["Connector", "Keyword", "Function", "Value"])
        self.crit_table.horizontalHeader().setStretchLastSection(True)
        cl.addWidget(self.crit_table)
        self._keywords = keywords
        if group is not None:
            for c in group.criteria:
                self._add_criterion_row(c)
        btns = QHBoxLayout()
        add = QPushButton("Add criterion")
        add.clicked.connect(lambda: self._add_criterion_row(Criterion("$FIL", "Contains", "")))
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove_row)
        btns.addWidget(add)
        btns.addWidget(rem)
        btns.addStretch(1)
        cl.addLayout(btns)
        layout.addWidget(crit_box, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_color_button(self):
        self.color_button.setText(self._color)
        self.color_button.setStyleSheet(f"QPushButton {{ color: {self._color}; font-weight: bold; }}")

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self, "Group color")
        if c.isValid():
            self._color = c.name()
            self._update_color_button()

    def _add_criterion_row(self, crit: Criterion):
        row = self.crit_table.rowCount()
        self.crit_table.insertRow(row)
        conn = QComboBox()
        conn.addItems(["And", "Or"])
        conn.setCurrentText(crit.connector if crit.connector in ("And", "Or") else "And")
        kw = QComboBox()
        kw.setEditable(True)
        kw.addItems(self._keywords)
        kw.setCurrentText(crit.keyword)
        fn = QComboBox()
        fn.addItems(CRITERION_FUNCTIONS)
        fn.setCurrentText(crit.function if crit.function in CRITERION_FUNCTIONS else "Contains")
        self.crit_table.setCellWidget(row, 0, conn)
        self.crit_table.setCellWidget(row, 1, kw)
        self.crit_table.setCellWidget(row, 2, fn)
        self.crit_table.setItem(row, 3, QTableWidgetItem(crit.value))

    def _remove_row(self):
        row = self.crit_table.currentRow()
        if row >= 0:
            self.crit_table.removeRow(row)

    def values(self) -> tuple[str, str, list[int], list[Criterion]]:
        ids = [it.data(Qt.ItemDataRole.UserRole) for it in self.sample_list.selectedItems()]
        crits = []
        for row in range(self.crit_table.rowCount()):
            conn = self.crit_table.cellWidget(row, 0).currentText()
            kw = self.crit_table.cellWidget(row, 1).currentText().strip()
            fn = self.crit_table.cellWidget(row, 2).currentText()
            item = self.crit_table.item(row, 3)
            val = item.text().strip() if item else ""
            if kw and val:
                crits.append(Criterion(kw, fn, val, conn))
        return self.name_edit.text().strip() or "Group", self._color, ids, crits


# --------------------------------------------------------------------- keywords
class KeywordsDialog(QDialog):
    def __init__(self, sample: Sample, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.setWindowTitle(f"Keywords – {sample.name}")
        self.resize(560, 520)
        layout = QVBoxLayout(self)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter keywords…")
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Keyword", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        for key in sorted(sample.keywords, key=lambda k: (not k.startswith("$"), k)):
            self._add_row(key, sample.keywords[key], editable=not key.startswith("$"))
        btns = QHBoxLayout()
        add = QPushButton("Add keyword")
        add.clicked.connect(lambda: self._add_row("NEW_KEYWORD", "", editable=True, select=True))
        btns.addWidget(add)
        btns.addStretch(1)
        layout.addLayout(btns)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_row(self, key: str, value: str, editable: bool, select: bool = False):
        row = self.table.rowCount()
        self.table.insertRow(row)
        k = QTableWidgetItem(key)
        v = QTableWidgetItem(value)
        if not editable:
            k.setFlags(k.flags() & ~Qt.ItemFlag.ItemIsEditable)
            k.setForeground(QColor("#555555"))
        self.table.setItem(row, 0, k)
        self.table.setItem(row, 1, v)
        if select:
            self.table.setCurrentItem(k)
            self.table.editItem(k)

    def _apply_filter(self, text: str):
        text = text.lower()
        for row in range(self.table.rowCount()):
            k = self.table.item(row, 0).text().lower()
            v = self.table.item(row, 1).text().lower()
            self.table.setRowHidden(row, bool(text) and text not in k and text not in v)

    def values(self) -> dict[str, str]:
        out = {}
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text().strip()
            if key:
                out[key] = self.table.item(row, 1).text()
        return out


# ---------------------------------------------------------- derived parameter
class DerivedParameterDialog(QDialog):
    OPS = [("ratio", "A / B"), ("product", "A × B"), ("sum", "A + B"), ("difference", "A − B"), ("log10", "log10(A)")]

    def __init__(self, params: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Derive Parameter")
        layout = QFormLayout(self)
        self.name_edit = QLineEdit("Ratio")
        self.op = QComboBox()
        for key, label in self.OPS:
            self.op.addItem(label, key)
        self.a = QComboBox()
        self.a.addItems(params)
        self.b = QComboBox()
        self.b.addItems(params)
        if len(params) > 1:
            self.b.setCurrentIndex(1)
        self.max_edit = _spin(1e-6, 1e9, 10, 3)
        layout.addRow("Name:", self.name_edit)
        layout.addRow("Formula:", self.op)
        layout.addRow("A:", self.a)
        layout.addRow("B:", self.b)
        layout.addRow("Axis maximum:", self.max_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        self.op.currentIndexChanged.connect(lambda: self.b.setEnabled(self.op.currentData() != "log10"))

    def values(self) -> tuple[str, dict]:
        spec = {"op": self.op.currentData(), "a": self.a.currentText(), "b": self.b.currentText() if self.op.currentData() != "log10" else None, "min": 0.0, "max": self.max_edit.value()}
        return self.name_edit.text().strip() or "Derived", spec


# --------------------------------------------------------------------- overlay
class OverlayDialog(QDialog):
    """Pick populations (from any sample of the group) to overlay on a plot."""

    def __init__(self, samples: list[Sample], current: list[tuple[Sample, Population]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Overlay populations")
        self.resize(460, 520)
        layout = QVBoxLayout(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Population", "Events"])
        self.tree.setColumnWidth(0, 300)
        layout.addWidget(self.tree, 1)
        current_ids = {(s.id, p.id) for s, p in current}
        for s in samples:
            top = QTreeWidgetItem([s.name, str(s.n_events)])
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            top.setCheckState(0, Qt.CheckState.Checked if (s.id, s.root.id) in current_ids else Qt.CheckState.Unchecked)
            top.setData(0, Qt.ItemDataRole.UserRole, (s, s.root))
            self.tree.addTopLevelItem(top)
            self._add_children(top, s, s.root, current_ids)
            top.setExpanded(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_children(self, item, sample, pop, current_ids):
        for c in pop.children:
            child = QTreeWidgetItem([c.name, str(c.count)])
            child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            child.setCheckState(0, Qt.CheckState.Checked if (sample.id, c.id) in current_ids else Qt.CheckState.Unchecked)
            child.setData(0, Qt.ItemDataRole.UserRole, (sample, c))
            item.addChild(child)
            self._add_children(child, sample, c, current_ids)
            child.setExpanded(True)

    def values(self) -> list[tuple[Sample, Population]]:
        out = []

        def walk(item):
            if item.checkState(0) == Qt.CheckState.Checked:
                out.append(item.data(0, Qt.ItemDataRole.UserRole))
            for i in range(item.childCount()):
                walk(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            walk(self.tree.topLevelItem(i))
        return out


# --------------------------------------------------------------------- columns
class ColumnsDialog(QDialog):
    """Choose keyword columns shown in the workspace sample list."""

    def __init__(self, keywords: list[str], selected: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Columns")
        self.resize(360, 480)
        layout = QVBoxLayout(self)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter…")
        layout.addWidget(self.filter)
        self.list = QListWidget()
        for k in keywords:
            item = QListWidgetItem(k)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if k in selected else Qt.CheckState.Unchecked)
            self.list.addItem(item)
        layout.addWidget(self.list, 1)
        self.filter.textChanged.connect(self._filter)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _filter(self, text):
        text = text.lower()
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def values(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count()) if self.list.item(i).checkState() == Qt.CheckState.Checked]


# ----------------------------------------------------------------- preferences
class PreferencesDialog(QDialog):
    def __init__(self, prefs, parent=None):
        super().__init__(parent)
        self.prefs = prefs
        self.setWindowTitle("Preferences")
        self.resize(520, 440)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        # graphs
        w = QWidget()
        f = QFormLayout(w)
        self.plot_type = QComboBox()
        for key, label in PLOT_TYPES:
            self.plot_type.addItem(label, key)
        self.plot_type.setCurrentIndex(max(0, [k for k, _ in PLOT_TYPES].index(prefs.get("plot_type")) if prefs.get("plot_type") in [k for k, _ in PLOT_TYPES] else 0))
        self.dots = QComboBox()
        for n in (10000, 25000, 50000, 100000, 250000, 1000000):
            self.dots.addItem(f"{n:,}", n)
        self.dots.setCurrentIndex(max(0, [10000, 25000, 50000, 100000, 250000, 1000000].index(int(prefs.get("dots_to_draw"))) if int(prefs.get("dots_to_draw")) in (10000, 25000, 50000, 100000, 250000, 1000000) else 2))
        self.dot_size = QSpinBox()
        self.dot_size.setRange(1, 4)
        self.dot_size.setValue(int(prefs.get("dot_size")))
        self.smoothing = QComboBox()
        self.smoothing.addItems(["None", "Low", "Medium", "High"])
        self.smoothing.setCurrentIndex(min(3, max(0, int(prefs.get("smoothing")))))
        self.contour = QComboBox()
        for v in (2.0, 5.0, 10.0, 20.0):
            self.contour.addItem(f"{v:g}%", v)
        self.contour.setCurrentIndex(max(0, [2.0, 5.0, 10.0, 20.0].index(float(prefs.get("contour_level"))) if float(prefs.get("contour_level")) in (2.0, 5.0, 10.0, 20.0) else 1))
        self.outliers = QCheckBox("Show outliers on contour plots")
        self.outliers.setChecked(bool(prefs.get("show_outliers")))
        self.hist_mode = QComboBox()
        self.hist_mode.addItem("Count", "count")
        self.hist_mode.addItem("Normalized to mode (% of max)", "mode")
        self.hist_mode.setCurrentIndex(1 if prefs.get("histogram_ymode") == "mode" else 0)
        self.hist_bins = QComboBox()
        for n in (64, 128, 256, 512, 1024):
            self.hist_bins.addItem(str(n), n)
        self.hist_bins.setCurrentText(str(int(prefs.get("histogram_bins"))))
        f.addRow("Default plot type:", self.plot_type)
        f.addRow("Events to draw:", self.dots)
        f.addRow("Dot size:", self.dot_size)
        f.addRow("Smoothing:", self.smoothing)
        f.addRow("Contour level:", self.contour)
        f.addRow("", self.outliers)
        f.addRow("Histogram y-axis:", self.hist_mode)
        f.addRow("Histogram resolution:", self.hist_bins)
        tabs.addTab(w, "Graphs")

        # gates
        w = QWidget()
        f = QFormLayout(w)
        self._gate_color = prefs.get("gate_color")
        self._sel_color = prefs.get("selected_gate_color")
        self.gate_color_btn = QPushButton(self._gate_color)
        self.gate_color_btn.clicked.connect(lambda: self._pick("_gate_color", self.gate_color_btn))
        self.sel_color_btn = QPushButton(self._sel_color)
        self.sel_color_btn.clicked.connect(lambda: self._pick("_sel_color", self.sel_color_btn))
        self.gate_width = QDoubleSpinBox()
        self.gate_width.setRange(0.5, 4.0)
        self.gate_width.setSingleStep(0.5)
        self.gate_width.setValue(float(prefs.get("gate_line_width")))
        self.show_names = QCheckBox("Show gate names on plots")
        self.show_names.setChecked(bool(prefs.get("show_gate_names")))
        self.show_freqs = QCheckBox("Show frequencies on plots")
        self.show_freqs.setChecked(bool(prefs.get("show_gate_freqs")))
        self.freq_decimals = QSpinBox()
        self.freq_decimals.setRange(0, 3)
        self.freq_decimals.setValue(int(prefs.get("freq_decimals")))
        self.apply_group = QCheckBox("New gates are applied to the current group by default")
        self.apply_group.setChecked(bool(prefs.get("apply_to_group_default")))
        self.polygon_mode = QComboBox()
        self.polygon_mode.addItem("Exact (every event against the polygon)", "exact")
        self.polygon_mode.addItem("FlowJo-compatible (bitmap)", "bitmap")
        self.polygon_mode.setCurrentIndex(1 if prefs.get("polygon_gating") == "bitmap" else 0)
        self.polygon_mode.setToolTip("FlowJo evaluates polygon gates on a binned display grid; this mode mimics it so that counts match FlowJo more closely.")
        f.addRow("Polygon gates:", self.polygon_mode)
        f.addRow("Gate color:", self.gate_color_btn)
        f.addRow("Selected gate color:", self.sel_color_btn)
        f.addRow("Gate line width:", self.gate_width)
        f.addRow("", self.show_names)
        f.addRow("", self.show_freqs)
        f.addRow("Frequency decimals:", self.freq_decimals)
        f.addRow("", self.apply_group)
        tabs.addTab(w, "Gates")

        # fonts & axes
        w = QWidget()
        f = QFormLayout(w)
        self.label_size = QSpinBox()
        self.label_size.setRange(6, 24)
        self.label_size.setValue(int(prefs.get("label_font_size")))
        self.number_size = QSpinBox()
        self.number_size.setRange(6, 20)
        self.number_size.setValue(int(prefs.get("number_font_size")))
        self.axis_style = QComboBox()
        self.axis_style.addItem("Name :: Stain", "name_stain")
        self.axis_style.addItem("Name only", "name")
        self.axis_style.addItem("Stain only", "stain")
        self.axis_style.setCurrentIndex(max(0, ["name_stain", "name", "stain"].index(prefs.get("axis_label_style")) if prefs.get("axis_label_style") in ("name_stain", "name", "stain") else 0))
        self.width_basis = QComboBox()
        for v in WIDTH_BASIS_PRESETS:
            self.width_basis.addItem(f"{v:g}", float(v))
        self.width_basis.setCurrentText(f"{float(prefs.get('default_width_basis')):g}")
        f.addRow("Axis label font size:", self.label_size)
        f.addRow("Tick number font size:", self.number_size)
        f.addRow("Axis labels:", self.axis_style)
        f.addRow("Default width basis:", self.width_basis)
        tabs.addTab(w, "Axes && Fonts")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick(self, attr, button):
        c = QColorDialog.getColor(QColor(getattr(self, attr)), self, "Color")
        if c.isValid():
            setattr(self, attr, c.name())
            button.setText(c.name())

    def values(self) -> dict:
        return {
            "plot_type": self.plot_type.currentData(),
            "dots_to_draw": int(self.dots.currentData()),
            "dot_size": self.dot_size.value(),
            "smoothing": self.smoothing.currentIndex(),
            "contour_level": float(self.contour.currentData()),
            "show_outliers": self.outliers.isChecked(),
            "histogram_ymode": self.hist_mode.currentData(),
            "histogram_bins": int(self.hist_bins.currentData()),
            "gate_color": self._gate_color,
            "selected_gate_color": self._sel_color,
            "gate_line_width": self.gate_width.value(),
            "show_gate_names": self.show_names.isChecked(),
            "show_gate_freqs": self.show_freqs.isChecked(),
            "freq_decimals": self.freq_decimals.value(),
            "apply_to_group_default": self.apply_group.isChecked(),
            "polygon_gating": self.polygon_mode.currentData(),
            "label_font_size": self.label_size.value(),
            "number_font_size": self.number_size.value(),
            "axis_label_style": self.axis_style.currentData(),
            "default_width_basis": float(self.width_basis.currentData()),
        }


# ----------------------------------------------------------------------- about
def show_about(parent=None):
    QMessageBox.about(
        parent,
        f"About {APP_NAME}",
        f"<h3>{APP_NAME} {__version__}</h3>"
        "<p>A desktop workspace for flow cytometry analysis: gating, compensation, layouts and tables.</p>"
        "<p>Reads FCS 2.0/3.0/3.1 files and FlowJo&trade; .wsp workspace files. "
        "Implemented independently from the public FCS, Gating-ML and logicle specifications; it contains no code from, and was not "
        "derived from, any other product.</p>"
        "<p>FlowDesk is not affiliated with, endorsed by or sponsored by Becton, Dickinson and Company (BD) or the makers of any "
        "other flow cytometry software. FlowJo and other product names are trademarks of their respective owners and are named "
        "only to describe file compatibility. Provided as is, without warranty; see LEGAL.md.</p>",
    )


def confirm(parent, title: str, text: str) -> bool:
    return QMessageBox.question(parent, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes) == QMessageBox.StandardButton.Yes
