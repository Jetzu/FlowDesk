"""Plate Editor: samples laid out on a 96- or 384-well plate.

Wells are assigned from a keyword ($WELLID, WELL ID, ...) or from a pattern on
the sample name ("Specimen_001_A1.fcs" -> A1); unassigned samples can be put
on a well by hand (pick the sample in the side list, click a well). Each well is
colour-coded by a statistic of a chosen population. Clicking a well selects the
sample in the workspace, double-clicking opens its graph window.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, QObject, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QGuiApplication, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core.model import Sample
from ..core.stats import NEEDS_PARAM, STATISTICS, compute_statistic, format_value, statistic_label
from .icons import icon

PLATE_FORMATS = {96: (8, 12), 384: (16, 24)}
WELL_KEYWORDS = ["$WELLID", "WELL ID", "WELLID", "Well", "$SMNO"]
AUTO_SOURCE = "Auto"
NAME_SOURCE = "Sample name (pattern)"
DEFAULT_NAME_PATTERN = r"([A-P])(\d{1,2})"
ROW_LETTERS = "ABCDEFGHIJKLMNOP"
# tolerant built-in well pattern: a row letter not preceded by another letter,
# an optional separator and a 1-2 digit column not followed by more digits
WELL_RE = re.compile(r"(?<![A-Za-z])([A-P])[-_ ]?0*(\d{1,2})(?!\d)")
# blue -> teal -> yellow -> orange -> red
COLOR_STOPS = [(0.0, (49, 54, 149)), (0.25, (69, 117, 180)), (0.5, (171, 217, 233)), (0.62, (254, 224, 144)), (0.8, (244, 109, 67)), (1.0, (165, 0, 38))]
EMPTY_COLOR = QColor("#f3f4f6")
NO_VALUE_COLOR = QColor("#d1d5db")


# =============================================================== pure helpers
def well_name(row: int, col: int) -> str:
    return f"{ROW_LETTERS[row]}{col + 1}"


def parse_well(text: str, rows: int = 8, cols: int = 12, pattern: str | None = None) -> tuple[int, int] | None:
    """Find a well ID such as "A1", "B02" or "H-12" in text. With `pattern`
    the first two groups are (row letter, column number); the last match that
    fits on the plate wins (file names usually end with the well)."""
    if not text:
        return None
    text = str(text)
    if text.lower().endswith(".fcs"):
        text = text[:-4]
    try:
        rx = re.compile(pattern) if pattern else WELL_RE
    except re.error:
        return None
    found = None
    for m in rx.finditer(text):
        if len(m.groups()) < 2:
            continue
        letter, number = m.group(1), m.group(2)
        if not letter or not number or not number.isdigit():
            continue
        r = ROW_LETTERS.find(letter.upper()[0])
        c = int(number) - 1
        if 0 <= r < rows and 0 <= c < cols:
            found = (r, c)
    return found


def sample_well(sample: Sample, source: str, rows: int, cols: int, pattern: str = DEFAULT_NAME_PATTERN) -> tuple[int, int] | None:
    if source == NAME_SOURCE:
        # the default pattern uses the tolerant built-in expression (it will
        # not read "e1" out of "Plate1_A1"); a custom one is used as given
        if not pattern or pattern == DEFAULT_NAME_PATTERN:
            return parse_well(sample.name, rows, cols)
        return parse_well(sample.name, rows, cols, pattern)
    if source == AUTO_SOURCE:
        for key in WELL_KEYWORDS:
            well = parse_well(sample.keywords.get(key, "").strip().upper(), rows, cols)
            if well is not None:
                return well
        return sample_well(sample, NAME_SOURCE, rows, cols, pattern)
    return parse_well(sample.keywords.get(source, "").strip().upper(), rows, cols)


def assign_wells(samples: list[Sample], source: str, rows: int, cols: int, pattern: str = DEFAULT_NAME_PATTERN, manual: dict[int, tuple[int, int]] | None = None):
    """Returns ({(row, col): sample}, [unassigned samples]). Manual assignments
    (sample id -> well) take precedence; a second sample on an occupied well is
    left unassigned."""
    wells: dict[tuple[int, int], Sample] = {}
    unassigned: list[Sample] = []
    manual = manual or {}
    for s in samples:
        w = manual.get(s.id)
        if w is not None and 0 <= w[0] < rows and 0 <= w[1] < cols and w not in wells:
            wells[w] = s
    placed = set(id(s) for s in wells.values())
    for s in samples:
        if id(s) in placed:
            continue
        w = sample_well(s, source, rows, cols, pattern)
        if w is None or w in wells:
            unassigned.append(s)
        else:
            wells[w] = s
    return wells, unassigned


def color_for(t: float) -> QColor:
    t = min(max(float(t), 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(COLOR_STOPS, COLOR_STOPS[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return QColor(*(int(round(a + (b - a) * f)) for a, b in zip(c0, c1)))
    return QColor(*COLOR_STOPS[-1][1])


def value_range(values: dict) -> tuple[float, float] | None:
    finite = [v for v in values.values() if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return min(finite), max(finite)


# ======================================================================= views
class PlateWidget(QWidget):
    """Paints the plate grid; emits well clicks."""

    wellClicked = Signal(int, int, object)  # row, col, keyboard modifiers
    wellDoubleClicked = Signal(int, int)

    HEADER = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows, self.cols = PLATE_FORMATS[96]
        self.wells: dict[tuple[int, int], Sample] = {}
        self.values: dict[tuple[int, int], float] = {}
        self.selected: set[tuple[int, int]] = set()
        self.vrange: tuple[float, float] | None = None
        self.stat_name = ""
        self.setMinimumSize(520, 340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(760, 500)

    def _cell(self) -> tuple[float, float]:
        w = (self.width() - self.HEADER - 4) / self.cols
        h = (self.height() - self.HEADER - 4) / self.rows
        return max(w, 4.0), max(h, 4.0)

    def well_rect(self, row: int, col: int) -> QRectF:
        w, h = self._cell()
        return QRectF(self.HEADER + col * w + 1, self.HEADER + row * h + 1, w - 2, h - 2)

    def well_at(self, pos) -> tuple[int, int] | None:
        w, h = self._cell()
        x, y = pos.x() - self.HEADER, pos.y() - self.HEADER
        if x < 0 or y < 0:
            return None
        c, r = int(x // w), int(y // h)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def color_of(self, well: tuple[int, int]) -> QColor:
        if well not in self.wells:
            return EMPTY_COLOR
        v = self.values.get(well)
        if v is None or not math.isfinite(v) or self.vrange is None:
            return NO_VALUE_COLOR
        lo, hi = self.vrange
        return color_for(0.5 if hi <= lo else (v - lo) / (hi - lo))

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#ffffff"))
        w, h = self._cell()
        head_font = QFont(self.font())
        head_font.setBold(True)
        head_font.setPointSizeF(max(6.0, min(10.0, h * 0.35)))
        p.setFont(head_font)
        p.setPen(QColor("#374151"))
        for c in range(self.cols):
            p.drawText(QRectF(self.HEADER + c * w, 0, w, self.HEADER), Qt.AlignmentFlag.AlignCenter, str(c + 1))
        for r in range(self.rows):
            p.drawText(QRectF(0, self.HEADER + r * h, self.HEADER, h), Qt.AlignmentFlag.AlignCenter, ROW_LETTERS[r])
        name_font = QFont(self.font())
        name_font.setPointSizeF(max(5.0, min(8.5, h * 0.22)))
        fm = QFontMetrics(name_font)
        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.well_rect(r, c)
                well = (r, c)
                fill = self.color_of(well)
                p.setBrush(fill)
                sel = well in self.selected
                p.setPen(QPen(QColor("#d81b1b") if sel else QColor("#9ca3af"), 2.5 if sel else 0.8))
                radius = min(rect.width(), rect.height()) * 0.18
                p.drawRoundedRect(rect, radius, radius)
                sample = self.wells.get(well)
                if sample is None:
                    continue
                dark = fill.lightnessF() < 0.55
                p.setPen(QColor("#ffffff") if dark else QColor("#111827"))
                p.setFont(name_font)
                inner = rect.adjusted(2, 1, -2, -1)
                label = Path(sample.name).stem
                text = fm.elidedText(label, Qt.TextElideMode.ElideMiddle, int(inner.width()))
                v = self.values.get(well)
                if v is not None and math.isfinite(v) and inner.height() > 2.3 * fm.height():
                    p.drawText(inner, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)
                    p.drawText(inner, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, format_value(self.stat_name, v))
                else:
                    p.drawText(inner, Qt.AlignmentFlag.AlignCenter, text)
        p.end()

    def mousePressEvent(self, event):
        well = self.well_at(event.position())
        if well is not None and event.button() == Qt.MouseButton.LeftButton:
            self.wellClicked.emit(well[0], well[1], event.modifiers())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        well = self.well_at(event.position())
        if well is not None:
            self.wellDoubleClicked.emit(well[0], well[1])

    def mouseMoveEvent(self, event):
        well = self.well_at(event.position())
        if well is None:
            self.setToolTip("")
        else:
            s = self.wells.get(well)
            tip = well_name(*well) + (f": {s.name}" if s else " (empty)")
            v = self.values.get(well)
            if s is not None and v is not None and math.isfinite(v):
                tip += f"\n{self.stat_name}: {format_value(self.stat_name, v)}"
            self.setToolTip(tip)
        super().mouseMoveEvent(event)


class LegendWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vrange: tuple[float, float] | None = None
        self.stat_name = ""
        self.label = ""
        self.setFixedHeight(34)

    def set_range(self, vrange, stat_name: str, label: str = ""):
        self.vrange = vrange
        self.stat_name = stat_name
        self.label = label or stat_name
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().window())
        bar = QRectF(90, 4, max(self.width() - 180, 40), 14)
        grad = QLinearGradient(QPointF(bar.left(), 0), QPointF(bar.right(), 0))
        for t, rgb in COLOR_STOPS:
            grad.setColorAt(t, QColor(*rgb))
        p.setPen(QColor("#6b7280"))
        p.setBrush(grad)
        p.drawRect(bar)
        p.setPen(self.palette().windowText().color())
        if self.vrange is None:
            lo_t, hi_t = "–", "–"
        else:
            lo_t, hi_t = format_value(self.stat_name, self.vrange[0]), format_value(self.stat_name, self.vrange[1])
        p.drawText(QRectF(0, 0, 86, 22), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, lo_t)
        p.drawText(QRectF(bar.right() + 4, 0, 86, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, hi_t)
        p.drawText(QRectF(bar.left(), 18, bar.width(), 16), Qt.AlignmentFlag.AlignCenter, self.label)
        p.end()


class PlateEditorWindow(QMainWindow):
    def __init__(self, controller, main_window=None, parent=None):
        super().__init__(parent)
        if parent is not None:
            self.setWindowFlag(Qt.WindowType.Window, True)
        self.controller = controller
        self.main_window = main_window
        self.setWindowTitle("Plate Editor")
        self.setWindowIcon(icon("grid"))
        self.resize(1040, 640)
        self.wells: dict[tuple[int, int], Sample] = {}
        self.unassigned: list[Sample] = []
        self.values: dict[tuple[int, int], float] = {}
        self._updating = False

        central = QWidget()
        outer = QVBoxLayout(central)
        top = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItem("96 wells (8 × 12)", 96)
        self.format_combo.addItem("384 wells (16 × 24)", 384)
        self.source_combo = QComboBox()
        self.pattern_edit = QLineEdit(DEFAULT_NAME_PATTERN)
        self.pattern_edit.setToolTip("Regular expression on the sample name: group 1 = row letter, group 2 = column")
        self.pattern_edit.setMaximumWidth(170)
        for w, label in ((self.format_combo, "Plate:"), (self.source_combo, "Wells from:"), (self.pattern_edit, "Pattern:")):
            top.addWidget(QLabel(label))
            top.addWidget(w)
        top.addStretch(1)
        outer.addLayout(top)
        stat_row = QHBoxLayout()
        self.pop_combo = QComboBox()
        self.pop_combo.setMinimumWidth(180)
        self.stat_combo = QComboBox()
        self.stat_combo.addItems(STATISTICS)
        self.param_combo = QComboBox()
        self.param_combo.setMinimumWidth(140)
        for w, label in ((self.pop_combo, "Population:"), (self.stat_combo, "Statistic:"), (self.param_combo, "Parameter:")):
            stat_row.addWidget(QLabel(label))
            stat_row.addWidget(w)
        stat_row.addStretch(1)
        outer.addLayout(stat_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.plate = PlateWidget()
        ll.addWidget(self.plate, 1)
        self.legend = LegendWidget()
        ll.addWidget(self.legend)
        split.addWidget(left)
        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("Unassigned samples:"))
        self.unassigned_list = QListWidget()
        sl.addWidget(self.unassigned_list, 1)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Assign:"))
        self.assign_combo = QComboBox()
        arow.addWidget(self.assign_combo, 1)
        sl.addLayout(arow)
        sl.addWidget(QLabel("Pick a sample, then click a well."))
        self.unassign_btn = QPushButton("Unassign Selected Wells")
        sl.addWidget(self.unassign_btn)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setSizes([800, 220])
        outer.addWidget(split, 1)

        btns = QHBoxLayout()
        self.export_btn = QPushButton(icon("export"), "Export CSV…")
        self.copy_btn = QPushButton(icon("copy"), "Copy Image")
        self.group_btn = QPushButton(icon("group"), "Create Group from Selected Wells…")
        self.status = QLabel()
        btns.addWidget(self.status, 1)
        for b in (self.export_btn, self.copy_btn, self.group_btn):
            btns.addWidget(b)
        outer.addLayout(btns)
        self.setCentralWidget(central)

        self.format_combo.currentIndexChanged.connect(lambda *_: self._layout_changed())
        self.source_combo.currentIndexChanged.connect(lambda *_: self._layout_changed())
        self.pattern_edit.editingFinished.connect(self._layout_changed)
        self.pop_combo.currentIndexChanged.connect(lambda *_: self._stat_changed())
        self.stat_combo.currentIndexChanged.connect(lambda *_: self._stat_changed())
        self.param_combo.currentIndexChanged.connect(lambda *_: self._stat_changed())
        self.unassigned_list.currentRowChanged.connect(self._list_row_changed)
        self.assign_combo.currentIndexChanged.connect(self._assign_combo_changed)
        self.plate.wellClicked.connect(self.click_well)
        self.plate.wellDoubleClicked.connect(self.open_well)
        self.unassign_btn.clicked.connect(self.unassign_selected)
        self.export_btn.clicked.connect(lambda: self.export_csv())
        self.copy_btn.clicked.connect(self.copy_image)
        self.group_btn.clicked.connect(lambda: self.create_group_from_selection())

        c = controller
        c.samplesChanged.connect(self.refresh)
        c.gatingChanged.connect(lambda _sid: self.refresh())
        c.workspaceReplaced.connect(self._workspace_replaced)
        c.currentGroupChanged.connect(lambda _g: self.refresh())
        c.groupsChanged.connect(self.refresh)
        self._load_settings()
        self.refresh()

    # ----------------------------------------------------------- persistence
    def _settings(self) -> dict:
        return self.controller.workspace.settings.setdefault("plate_editor", {})

    def _load_settings(self):
        st = self.controller.workspace.settings.get("plate_editor", {})
        self._updating = True
        idx = self.format_combo.findData(int(st.get("format", 96)))
        self.format_combo.setCurrentIndex(max(idx, 0))
        self.pattern_edit.setText(st.get("pattern", DEFAULT_NAME_PATTERN))
        self._pending_source = st.get("source", AUTO_SOURCE)
        self._updating = False

    @property
    def manual(self) -> dict[int, tuple[int, int]]:
        raw = self.controller.workspace.settings.get("plate_editor", {}).get("manual", {})
        rows, cols = self.dims
        out = {}
        for sid, wname in raw.items():
            w = parse_well(wname, rows, cols)
            if w is not None:
                out[int(sid)] = w
        return out

    def _workspace_replaced(self):
        self._load_settings()
        self.refresh()

    # ---------------------------------------------------------------- state
    @property
    def dims(self) -> tuple[int, int]:
        return PLATE_FORMATS[int(self.format_combo.currentData() or 96)]

    def set_format(self, n_wells: int):
        self.format_combo.setCurrentIndex(self.format_combo.findData(n_wells))

    def set_source(self, source: str):
        idx = self.source_combo.findText(source)
        if idx < 0:
            self.source_combo.addItem(source)
            idx = self.source_combo.count() - 1
        self.source_combo.setCurrentIndex(idx)

    def set_statistic(self, population_path: str, statistic: str, param: str | None = None):
        self._updating = True
        i = self.pop_combo.findData(population_path)
        if i >= 0:
            self.pop_combo.setCurrentIndex(i)
        self.stat_combo.setCurrentText(statistic)
        if param:
            j = self.param_combo.findText(param)
            if j >= 0:
                self.param_combo.setCurrentIndex(j)
        self._updating = False
        self._stat_changed()

    def _layout_changed(self):
        if self._updating:
            return
        st = self._settings()
        st["format"] = int(self.format_combo.currentData() or 96)
        st["source"] = self.source_combo.currentText()
        st["pattern"] = self.pattern_edit.text()
        self.plate.selected.clear()
        self.refresh()

    def _stat_changed(self):
        if self._updating:
            return
        self.param_combo.setEnabled(self.stat_combo.currentText() in NEEDS_PARAM)
        self.compute_values()
        self.plate.update()

    def _fill_combo(self, combo: QComboBox, items: list[tuple[str, object]], keep):
        combo.blockSignals(True)
        combo.clear()
        for text, data in items:
            combo.addItem(text, data)
        idx = combo.findData(keep) if keep is not None else -1
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def refresh(self):
        ws = self.controller.workspace
        self._updating = True
        # well source: defaults + any keyword that looks like a well
        keep_source = getattr(self, "_pending_source", None) or self.source_combo.currentText() or AUTO_SOURCE
        self._pending_source = None
        keys: dict[str, None] = {}
        for s in ws.samples:
            for k in s.keywords:
                if "WELL" in k.upper() and k not in WELL_KEYWORDS:
                    keys.setdefault(k, None)
        sources = [AUTO_SOURCE] + WELL_KEYWORDS + sorted(keys) + [NAME_SOURCE]
        if keep_source not in sources:
            sources.append(keep_source)
        self._fill_combo(self.source_combo, [(s, s) for s in sources], keep_source)
        # statistic sources
        group = self.controller.current_group
        paths = ws.population_paths(group) if group in ws.groups else ws.population_paths()
        self._fill_combo(self.pop_combo, [("Ungated", "")] + [(p.strip("/"), p) for p in paths], self.pop_combo.currentData())
        keep_param = self.param_combo.currentText()
        self.param_combo.blockSignals(True)
        self.param_combo.clear()
        self.param_combo.addItems(ws.all_parameter_names())
        if keep_param:
            j = self.param_combo.findText(keep_param)
            if j >= 0:
                self.param_combo.setCurrentIndex(j)
        self.param_combo.blockSignals(False)
        self.param_combo.setEnabled(self.stat_combo.currentText() in NEEDS_PARAM)
        self._updating = False
        self.assign()
        self.compute_values()
        self.plate.update()

    def assign(self):
        rows, cols = self.dims
        self.plate.rows, self.plate.cols = rows, cols
        self.wells, self.unassigned = assign_wells(self.controller.workspace.samples, self.source_combo.currentText() or AUTO_SOURCE, rows, cols, self.pattern_edit.text().strip(), self.manual)
        self.plate.wells = self.wells
        self.plate.selected &= set(self.wells)
        keep = self.assign_combo.currentData()
        self.unassigned_list.blockSignals(True)
        self.unassigned_list.clear()
        for s in self.unassigned:
            item = QListWidgetItem(icon("sample"), s.name)
            item.setData(Qt.ItemDataRole.UserRole, s.id)
            self.unassigned_list.addItem(item)
        self.unassigned_list.blockSignals(False)
        self._fill_combo(self.assign_combo, [("(none)", None)] + [(s.name, s.id) for s in self.unassigned], keep)
        self.status.setText(f"{len(self.wells)} samples on the plate · {len(self.unassigned)} unassigned")

    def _list_row_changed(self, row: int):
        if 0 <= row < len(self.unassigned):
            self.assign_combo.setCurrentIndex(row + 1)

    def _assign_combo_changed(self, idx: int):
        row = idx - 1
        self.unassigned_list.blockSignals(True)
        self.unassigned_list.setCurrentRow(row if row >= 0 else -1)
        self.unassigned_list.blockSignals(False)

    def statistic_spec(self) -> tuple[str, str, str | None]:
        stat = self.stat_combo.currentText()
        param = self.param_combo.currentText() if stat in NEEDS_PARAM else None
        return (self.pop_combo.currentData() or "", stat, param)

    def compute_values(self) -> dict[tuple[int, int], float]:
        path, stat, param = self.statistic_spec()
        values: dict[tuple[int, int], float] = {}
        for well, s in self.wells.items():
            pop = s.root if not path else s.root.find(path)
            if pop is None:
                values[well] = float("nan")
                continue
            try:
                values[well] = compute_statistic(stat, pop, param, 50.0 if stat == "Percentile" else None)
            except (ValueError, KeyError):
                values[well] = float("nan")
        self.values = values
        vr = value_range(values)
        label = statistic_label(stat, param, 50.0 if stat == "Percentile" else None)
        if path:
            label = f"{path.strip('/')} · {label}"
        self.plate.values = values
        self.plate.vrange = vr
        self.plate.stat_name = stat
        self.legend.set_range(vr, stat, label)
        self._legend_label = label
        return values

    def well_color(self, row: int, col: int) -> QColor:
        return self.plate.color_of((row, col))

    # -------------------------------------------------------------- actions
    def click_well(self, row: int, col: int, modifiers=Qt.KeyboardModifier.NoModifier):
        well = (row, col)
        pending = self.assign_combo.currentData()
        if pending is not None and well not in self.wells:
            sample = self.controller.workspace.sample_by_id(pending)
            if sample is not None:
                self.assign_sample(sample, row, col)
                return
        try:
            mods = Qt.KeyboardModifier(modifiers)
        except (TypeError, ValueError):
            mods = Qt.KeyboardModifier.NoModifier
        additive = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ShiftModifier))
        if additive:
            if well in self.plate.selected:
                self.plate.selected.discard(well)
            elif well in self.wells:
                self.plate.selected.add(well)
        else:
            self.plate.selected = {well} if well in self.wells else set()
        self.plate.update()
        sample = self.wells.get(well)
        if sample is not None:
            self.select_in_workspace([self.wells[w] for w in sorted(self.plate.selected)] if additive else [sample])

    def assign_sample(self, sample: Sample, row: int, col: int):
        manual = self._settings().setdefault("manual", {})
        manual[str(sample.id)] = well_name(row, col)
        self.controller.mark_modified()
        self.assign_combo.setCurrentIndex(0)
        self.assign()
        self.compute_values()
        self.plate.update()

    def unassign_selected(self):
        manual = self._settings().setdefault("manual", {})
        for w in list(self.plate.selected):
            s = self.wells.get(w)
            if s is not None:
                manual.pop(str(s.id), None)
        self.plate.selected.clear()
        self.assign()
        self.compute_values()
        self.plate.update()

    def select_in_workspace(self, samples: list[Sample]):
        tree = getattr(self.main_window, "tree", None) if self.main_window is not None else None
        if tree is None:
            self.controller.set_selection([(s, s.root) for s in samples])
            return
        from .main_window import ROLE_SAMPLE

        def find_items():
            out = []
            for i in range(tree.topLevelItemCount()):
                item = tree.topLevelItem(i)
                if item.data(0, ROLE_SAMPLE) in samples:
                    out.append(item)
            return out

        items = find_items()
        if len(items) < len(samples):
            ws = self.controller.workspace
            all_group = next((g for g in ws.groups if g.role == "all"), ws.groups[0])
            if self.controller.current_group is not all_group:
                self.controller.set_current_group(all_group)
                items = find_items()
        if not items:
            return
        tree.clearSelection()
        tree.setCurrentItem(items[0], 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        for item in items:
            item.setSelected(True)
        tree.scrollToItem(items[0])

    def open_well(self, row: int, col: int):
        sample = self.wells.get((row, col))
        if sample is not None and self.controller.open_graph is not None:
            return self.controller.open_graph(sample, sample.root)
        return None

    def plate_table(self) -> list[list[str]]:
        rows, cols = self.dims
        out = [["Sample"] + [str(c + 1) for c in range(cols)]]
        for r in range(rows):
            out.append([ROW_LETTERS[r]] + [self.wells[(r, c)].name if (r, c) in self.wells else "" for c in range(cols)])
        out.append([])
        out.append([getattr(self, "_legend_label", self.stat_combo.currentText())] + [str(c + 1) for c in range(cols)])
        stat = self.stat_combo.currentText()
        for r in range(rows):
            line = [ROW_LETTERS[r]]
            for c in range(cols):
                v = self.values.get((r, c))
                line.append(format_value(stat, v) if (r, c) in self.wells and v is not None else "")
            out.append(line)
        return out

    def export_csv(self, path: str | None = None) -> Path | None:
        if path is None:
            start = str(Path.home() / f"{self.controller.workspace.name}_plate.csv")
            path, _ = QFileDialog.getSaveFileName(self, "Export Plate", start, "CSV (*.csv)")
            if not path:
                return None
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(self.plate_table())
        self.controller.statusMessage.emit(f"Plate exported to {Path(path).name}")
        return Path(path)

    def grab_image(self):
        return self.plate.grab()

    def copy_image(self):
        QGuiApplication.clipboard().setPixmap(self.grab_image())

    def selected_samples(self) -> list[Sample]:
        return [self.wells[w] for w in sorted(self.plate.selected) if w in self.wells]

    def create_group_from_selection(self, name: str | None = None):
        samples = self.selected_samples()
        if not samples:
            return None
        if name is None:
            wells = ", ".join(well_name(*w) for w in sorted(self.plate.selected)[:6])
            name, ok = QInputDialog.getText(self, "Create Group", "Group name:", text=f"Wells {wells}")
            if not ok or not name.strip():
                return None
        return self.controller.create_group(name.strip(), samples)

    def closeEvent(self, event):
        event.ignore()
        self.hide()


# ================================================================== extension
class PlateEditorExtension(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.editor: PlateEditorWindow | None = None
        self.action = QAction(icon("grid"), "Plate Editor", window)
        self.action.setStatusTip("Show the samples on a 96/384-well plate, colour-coded by a statistic")
        self.action.setToolTip(self.action.statusTip())
        self.action.triggered.connect(lambda _checked=False: self.show_editor())
        exp = getattr(window, "extensions", {}).get("flowdesk.ui.experiment")
        band = getattr(exp, "band", None)
        self.own_band = band is None
        pages = getattr(window, "ribbon_pages", {})
        if band is None and "workspace" in pages:
            band = pages["workspace"].add_band("Experiment")
        if band is not None:
            band.add_large(self.action)
        self.band = band
        menus = getattr(window, "menus", {})
        if "workspace" in menus:
            menus["workspace"].addAction(self.action)

    def show_editor(self) -> PlateEditorWindow:
        if self.editor is None:
            self.editor = PlateEditorWindow(self.window.controller, self.window, self.window)
        else:
            self.editor.refresh()
        self.editor.show()
        self.editor.raise_()
        self.editor.activateWindow()
        return self.editor


def install(window) -> PlateEditorExtension:
    return PlateEditorExtension(window)
