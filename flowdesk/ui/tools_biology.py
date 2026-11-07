"""Biology tools (Cell Cycle, Proliferation, Kinetics) as a main-window extension.

`install(window)` adds a "Biology" band to the Tools ribbon page and the same
entries to the Tools menu. Every computation lives in `flowdesk.core.biology`;
the dialogs only collect settings, plot the result and show/export tables.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import biology as bio
from .icons import icon

PHASE_COLORS = {
    "G1": QColor("#1f6fd1"),
    "S": QColor("#2e9e44"),
    "G2/M": QColor("#d1352b"),
    "Debris": QColor("#8c6d31"),
    "Aggregates": QColor("#8e44ad"),
    "Model": QColor("#111111"),
}
CURVE_ORDER = ("G1", "S", "G2/M", "Debris", "Aggregates", "Model")
GEN_COLORS = ["#d1352b", "#e67e22", "#c9a400", "#2e9e44", "#16a3a3", "#1f6fd1", "#6a3fc1", "#b03a9a", "#7f5539", "#555555"]


# =============================================================================
# plot widget
# =============================================================================
def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    out = []
    v = start
    while v <= hi + 1e-9 * step:
        out.append(v)
        v += step
    return out


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.3g}"  # 1.5e+05, not a second "2e+05"
    return f"{v:g}"


class BiologyPlot(QWidget):
    """Histogram bars with fitted curves, or a line plot with a quartile band."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 260)
        self.mode = "empty"
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.lo = None
        self.hi = None
        self.curves: list[tuple[str, np.ndarray, QColor]] = []
        self.xlabel = ""
        self.ylabel = ""
        self.markers: list[float] = []

    def clear(self):
        self.mode = "empty"
        self.curves = []
        self.update()

    def set_histogram(self, centers, counts, curves=(), xlabel: str = "", ylabel: str = "Count"):
        self.mode = "hist"
        self.x = np.asarray(centers, dtype=float)
        self.y = np.asarray(counts, dtype=float)
        self.curves = [(name, np.asarray(c, dtype=float), QColor(col)) for name, c, col in curves]
        self.xlabel, self.ylabel = xlabel, ylabel
        self.update()

    def set_lines(self, x, y, lo=None, hi=None, xlabel: str = "", ylabel: str = ""):
        self.mode = "line"
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.lo = None if lo is None else np.asarray(lo, dtype=float)
        self.hi = None if hi is None else np.asarray(hi, dtype=float)
        self.curves = []
        self.xlabel, self.ylabel = xlabel, ylabel
        self.update()

    # ------------------------------------------------------------------ paint
    def _ranges(self):
        xs = self.x[np.isfinite(self.x)]
        if xs.size == 0:
            return None
        x0, x1 = float(xs.min()), float(xs.max())
        if self.mode == "hist" and self.x.size > 1:
            w = (x1 - x0) / max(self.x.size - 1, 1)
            x0, x1 = x0 - w / 2, x1 + w / 2
        ys = [self.y]
        if self.mode == "hist":
            ys += [c for _, c, _ in self.curves]
            y0 = 0.0
        else:
            ys += [a for a in (self.lo, self.hi) if a is not None]
            allv = np.concatenate([a[np.isfinite(a)] for a in ys]) if ys else np.zeros(0)
            y0 = float(allv.min()) if allv.size else 0.0
        allv = np.concatenate([a[np.isfinite(a)] for a in ys])
        y1 = float(allv.max()) if allv.size else 1.0
        if self.mode == "line":
            pad = (y1 - y0) * 0.08 or abs(y1) * 0.1 or 1.0
            y0, y1 = y0 - pad, y1 + pad
        else:
            y1 = y1 * 1.05 or 1.0
        if x1 <= x0:
            x1 = x0 + 1.0
        return x0, x1, y0, y1

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.85))
        p.setFont(font)
        rng = self._ranges() if self.mode != "empty" else None
        if rng is None:
            p.setPen(QColor("#888888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Run the analysis to see the plot")
            p.end()
            return
        x0, x1, y0, y1 = rng
        area = QRectF(56, 12, max(self.width() - 70, 10), max(self.height() - 50, 10))

        def px(x):
            return area.left() + (x - x0) / (x1 - x0) * area.width()

        def py(y):
            return area.bottom() - (y - y0) / (y1 - y0) * area.height()

        # axes and ticks
        p.setPen(QPen(QColor("#444444"), 1))
        p.drawRect(area)
        for t in _nice_ticks(x0, x1):
            xx = px(t)
            p.drawLine(QPointF(xx, area.bottom()), QPointF(xx, area.bottom() + 4))
            p.drawText(QRectF(xx - 30, area.bottom() + 4, 60, 14), Qt.AlignmentFlag.AlignHCenter, _fmt_tick(t))
        for t in _nice_ticks(y0, y1):
            yy = py(t)
            p.drawLine(QPointF(area.left() - 4, yy), QPointF(area.left(), yy))
            p.drawText(QRectF(0, yy - 7, area.left() - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _fmt_tick(t))
        p.drawText(QRectF(area.left(), self.height() - 16, area.width(), 14), Qt.AlignmentFlag.AlignHCenter, self.xlabel)
        p.save()
        p.translate(10, area.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-area.height() / 2, -8, area.height(), 14), Qt.AlignmentFlag.AlignHCenter, self.ylabel)
        p.restore()
        p.setClipRect(area)
        if self.mode == "hist":
            n = self.x.size
            w = (x1 - x0) / max(n, 1)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#c8ccd4"))
            for xc, yc in zip(self.x, self.y):
                if yc > 0:
                    p.drawRect(QRectF(px(xc - w / 2), py(yc), max(px(xc + w / 2) - px(xc - w / 2), 0.5), py(0) - py(yc)))
            p.setBrush(Qt.BrushStyle.NoBrush)
            for name, curve, col in self.curves:
                pen = QPen(col, 2.0 if name == "Model" else 1.6)
                if name == "Model":
                    pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                path = QPainterPath()
                for i, (xc, yc) in enumerate(zip(self.x, curve)):
                    pt = QPointF(px(xc), py(yc))
                    path.moveTo(pt) if i == 0 else path.lineTo(pt)
                p.drawPath(path)
        else:
            if self.lo is not None and self.hi is not None:
                band = QPainterPath()
                ok = np.isfinite(self.x) & np.isfinite(self.lo) & np.isfinite(self.hi)
                xs, los, his = self.x[ok], self.lo[ok], self.hi[ok]
                if xs.size:
                    band.moveTo(px(xs[0]), py(his[0]))
                    for a, b in zip(xs[1:], his[1:]):
                        band.lineTo(px(a), py(b))
                    for a, b in zip(xs[::-1], los[::-1]):
                        band.lineTo(px(a), py(b))
                    band.closeSubpath()
                    p.fillPath(band, QColor(31, 111, 209, 50))
            p.setPen(QPen(QColor("#1f6fd1"), 2))
            path = QPainterPath()
            first = True
            for xc, yc in zip(self.x, self.y):
                if not (math.isfinite(xc) and math.isfinite(yc)):
                    first = True
                    continue
                pt = QPointF(px(xc), py(yc))
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
                p.drawEllipse(pt, 2.5, 2.5)
            p.drawPath(path)
        p.setClipping(False)
        # legend
        if self.curves:
            y = area.top() + 4
            for name, _, col in self.curves:
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(area.right() - 92, y + 6), QPointF(area.right() - 76, y + 6))
                p.setPen(QColor("#222222"))
                p.drawText(QRectF(area.right() - 72, y, 70, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
                y += 13
        p.end()


# =============================================================================
# dialogs
# =============================================================================
def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return ""
        if abs(v) >= 1e5:
            return f"{v:.0f}"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.3f}" if abs(v) < 1 and v != 0 else f"{v:.2f}"
    return "" if v is None else str(v)


class BiologyDialog(QDialog):
    """Shared layout: population / parameter choice, options, run buttons,
    a plot and a results table with Copy and Export CSV."""

    TITLE = "Biology"
    RUN_LABEL = "Fit"

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle(self.TITLE)
        self.resize(900, 640)
        self.result = None
        self.header: list[str] = []
        self.rows: list[list] = []

        self.sample_combo = QComboBox()
        self.pop_combo = QComboBox()
        self.form = QFormLayout()
        self.form.addRow("Sample:", self.sample_combo)
        self.form.addRow("Population:", self.pop_combo)
        self._build_options(self.form)

        self.run_button = QPushButton(icon("play"), self.RUN_LABEL)
        self.group_button = QPushButton(icon("table"), "Apply to all samples in group")
        self.copy_button = QPushButton(icon("copy"), "Copy")
        self.export_button = QPushButton(icon("export"), "Export CSV…")
        self.status = QLabel("")
        self.status.setWordWrap(True)

        settings = QGroupBox("Settings")
        sl = QVBoxLayout(settings)
        sl.addLayout(self.form)
        sl.addWidget(self.run_button)
        sl.addWidget(self.group_button)
        sl.addStretch(1)
        sl.addWidget(self.status)

        self.plot = BiologyPlot()
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(settings)
        top.addWidget(self.plot)
        top.setStretchFactor(1, 1)

        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.table)
        br = QHBoxLayout()
        br.addStretch(1)
        br.addWidget(self.copy_button)
        br.addWidget(self.export_button)
        bl.addLayout(br)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self.sample_combo.currentIndexChanged.connect(self._sample_changed)
        self.run_button.clicked.connect(lambda: self.run())
        self.group_button.clicked.connect(lambda: self.apply_to_group())
        self.copy_button.clicked.connect(self.copy_results)
        self.export_button.clicked.connect(lambda: self.export_csv())
        controller.samplesChanged.connect(self.refresh)
        controller.workspaceReplaced.connect(self.refresh)
        controller.gatingChanged.connect(lambda *_: self._refresh_populations())
        self.refresh()

    # ----------------------------------------------------------- subclass API
    def _build_options(self, form: QFormLayout):
        raise NotImplementedError

    def _refresh_params(self, sample):
        raise NotImplementedError

    def compute(self, population):
        raise NotImplementedError

    def show_result(self, result):
        raise NotImplementedError

    def single_table(self, sample, population, result) -> tuple[list[str], list[list]]:
        header = self.group_header()
        return header, [self.group_row(sample, population, result)]

    def group_header(self) -> list[str]:
        raise NotImplementedError

    def group_row(self, sample, population, result) -> list:
        raise NotImplementedError

    def group_error_row(self, sample, population, message: str) -> list:
        header = self.group_header()
        row = [sample.name, population.display_name if population else ""] + [""] * (len(header) - 2)
        row[-1] = message
        return row

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _fill_params(combo: QComboBox, sample, default: str | None, filter_fn=None):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        names = sample.parameter_names() if sample is not None else []
        for n in names:
            if filter_fn is None or filter_fn(n):
                combo.addItem(sample.param_label(n), n)
        idx = combo.findData(current) if current else -1
        if idx < 0 and default is not None:
            idx = combo.findData(default)
        combo.setCurrentIndex(max(idx, 0) if combo.count() else -1)
        combo.blockSignals(False)

    @staticmethod
    def _param(combo: QComboBox) -> str | None:
        d = combo.currentData()
        return d if d is not None else (combo.currentText() or None)

    def set_param(self, combo: QComboBox, name: str):
        idx = combo.findData(name)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def refresh(self):
        ws = self.controller.workspace
        selected = self.controller.selected[0] if self.controller.selected else None
        current = self.current_sample()
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        for s in ws.samples:
            self.sample_combo.addItem(s.name, s.id)
        target = selected[0] if selected is not None else current
        idx = self.sample_combo.findData(target.id) if target is not None else -1
        self.sample_combo.setCurrentIndex(max(idx, 0) if self.sample_combo.count() else -1)
        self.sample_combo.blockSignals(False)
        self._sample_changed(pop_path=selected[1].path if selected is not None else None)

    def current_sample(self):
        sid = self.sample_combo.currentData()
        return self.controller.workspace.sample_by_id(sid) if sid is not None else None

    def _refresh_populations(self, pop_path: str | None = None):
        sample = self.current_sample()
        if pop_path is None:
            pop_path = self.pop_combo.currentData()
        self.pop_combo.blockSignals(True)
        self.pop_combo.clear()
        if sample is not None:
            self.pop_combo.addItem("Ungated", "")
            for pop in sample.populations():
                self.pop_combo.addItem("  " * (pop.depth - 1) + pop.name, pop.path)
        idx = self.pop_combo.findData(pop_path) if pop_path is not None else -1
        self.pop_combo.setCurrentIndex(max(idx, 0) if self.pop_combo.count() else -1)
        self.pop_combo.blockSignals(False)

    def _sample_changed(self, *_, pop_path: str | None = None):
        sample = self.current_sample()
        if sample is not None:
            sample.ensure_loaded()
        self._refresh_populations(pop_path)
        self._refresh_params(sample)

    def set_population(self, sample, population):
        idx = self.sample_combo.findData(sample.id)
        if idx >= 0:
            self.sample_combo.setCurrentIndex(idx)
        self._refresh_populations(population.path)

    def current_population(self):
        sample = self.current_sample()
        if sample is None:
            return None
        path = self.pop_combo.currentData() or ""
        return sample.root.find(path)

    def _set_table(self, header: list[str], rows: list[list]):
        self.header, self.rows = list(header), [list(r) for r in rows]
        self.table.clear()
        self.table.setColumnCount(len(header))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(header)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                item = QTableWidgetItem(_fmt(v))
                if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    # ----------------------------------------------------------------- actions
    def run(self):
        """Analyse the chosen population; returns the core result (or None)."""
        pop = self.current_population()
        if pop is None:
            self.status.setText("No population selected.")
            return None
        try:
            result = self.compute(pop)
        except (ValueError, KeyError, RuntimeError) as exc:
            self.status.setText(f"Could not analyse {pop.display_name}: {exc}")
            self.result = None
            self.plot.clear()
            return None
        self.result = result
        self.show_result(result)
        header, rows = self.single_table(pop.sample, pop, result)
        self._set_table(header, rows)
        self.status.setText(f"{pop.sample.name} / {pop.display_name}: {self.describe(result)}")
        return result

    def describe(self, result) -> str:
        return ""

    def apply_to_group(self) -> list[list]:
        """Run the same settings on the chosen population path of every sample
        in the current group; one table row per sample."""
        pop = self.current_population()
        path = pop.path if pop is not None else ""
        group = self.controller.current_group
        members = self.controller.workspace.members(group)
        header = self.group_header()
        rows: list[list] = []
        failed = 0
        for sample in members:
            if not sample.ensure_loaded():
                rows.append(self.group_error_row(sample, None, sample.load_error or "not loaded"))
                failed += 1
                continue
            target = sample.root.find(path)
            if target is None:
                rows.append(self.group_error_row(sample, None, f"no population {path}"))
                failed += 1
                continue
            try:
                result = self.compute(target)
            except (ValueError, KeyError, RuntimeError) as exc:
                rows.append(self.group_error_row(sample, target, str(exc)))
                failed += 1
                continue
            rows.append(self.group_row(sample, target, result))
        self._set_table(header, rows)
        msg = f"{len(rows) - failed} of {len(members)} samples in {group.name} analysed"
        self.status.setText(msg + (f" ({failed} failed)" if failed else ""))
        return rows

    def tsv(self) -> str:
        lines = ["\t".join(self.header)]
        for r in self.rows:
            lines.append("\t".join(_fmt(v) for v in r))
        return "\n".join(lines)

    def copy_results(self):
        QGuiApplication.clipboard().setText(self.tsv())

    def export_csv(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Results", f"{self.TITLE.replace(' ', '_')}.csv", "CSV files (*.csv)")
            if not path:
                return None
        bio.write_csv(path, self.header, self.rows)
        return str(path)


class CellCycleDialog(BiologyDialog):
    TITLE = "Cell Cycle"

    def _build_options(self, form):
        self.param_combo = QComboBox()
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(32, 1024)
        self.bins_spin.setValue(256)
        self.g1_spin = QDoubleSpinBox()
        self.g1_spin.setRange(0, 1e9)
        self.g1_spin.setDecimals(1)
        self.g1_spin.setSpecialValueText("Auto")
        self.g1_spin.setToolTip("Initial G1 peak position (0 = find the highest peak)")
        self.model_combo = QComboBox()
        for key, label in bio.CELL_CYCLE_MODELS.items():
            self.model_combo.addItem(label, key)
        self.model_combo.setItemData(0, "S phase as a Gaussian-broadened plateau, all components fitted together", Qt.ItemDataRole.ToolTipRole)
        self.model_combo.setItemData(1, "G1 and G2/M fitted on their outer halves, S phase by subtraction (no S-phase shape assumed)", Qt.ItemDataRole.ToolTipRole)
        self.debris_check = QCheckBox("Model debris (decreasing exponential below G1)")
        self.aggregates_check = QCheckBox("Model aggregates (G1+G1, G1+G2, G2+G2 doublets)")
        form.addRow("DNA parameter:", self.param_combo)
        form.addRow("Model:", self.model_combo)
        form.addRow("", self.debris_check)
        form.addRow("", self.aggregates_check)
        form.addRow("Bins:", self.bins_spin)
        form.addRow("G1 guess:", self.g1_spin)

    def set_model(self, model: str, debris: bool | None = None, aggregates: bool | None = None):
        idx = self.model_combo.findData(bio.cell_cycle_model_key(model))
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        if debris is not None:
            self.debris_check.setChecked(bool(debris))
        if aggregates is not None:
            self.aggregates_check.setChecked(bool(aggregates))

    def _refresh_params(self, sample):
        self._fill_params(self.param_combo, sample, bio.default_dna_parameter(sample) if sample else None)

    def compute(self, population):
        param = self._param(self.param_combo)
        if not param:
            raise ValueError("no parameter")
        g1 = self.g1_spin.value() or None
        return bio.cell_cycle_for_population(
            population, param, bins=self.bins_spin.value(), g1_guess=g1, model=self.model_combo.currentData(),
            debris=self.debris_check.isChecked(), aggregates=self.aggregates_check.isChecked(),
        )

    def show_result(self, r):
        curves = [(k, r.curves[k], PHASE_COLORS[k]) for k in CURVE_ORDER if k in r.curves]
        self.plot.set_histogram(r.centers, r.counts, curves, xlabel=self._param(self.param_combo) or "")

    def describe(self, r):
        text = f"{r.model_name}: G1 {r.percent_g1:.1f}%  S {r.percent_s:.1f}%  G2/M {r.percent_g2m:.1f}%"
        if math.isfinite(r.percent_debris):
            text += f"  (debris {r.percent_debris:.1f}%"
            text += f", aggregates {r.percent_aggregates:.1f}%)" if math.isfinite(r.percent_aggregates) else ")"
        elif math.isfinite(r.percent_aggregates):
            text += f"  (aggregates {r.percent_aggregates:.1f}%)"
        return text

    def group_header(self):
        return ["Sample", "Population", "Parameter", "%G1", "%S", "%G2/M", "G1 Mean", "G2 Mean", "G1 CV", "G2 CV", "G2/G1", "RMS", "Events", "%Debris", "%Aggregates", "Model", "Note"]

    def group_row(self, sample, population, r):
        s = r.summary()
        values = [s.get(k, float("nan")) for k in self.group_header()[3:-2]]
        return [sample.name, population.display_name, self._param(self.param_combo)] + values + [r.model_name, "" if r.success else r.message]


class ProliferationDialog(BiologyDialog):
    TITLE = "Proliferation"

    def _build_options(self, form):
        self.param_combo = QComboBox()
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(32, 1024)
        self.bins_spin.setValue(256)
        self.gen_spin = QSpinBox()
        self.gen_spin.setRange(2, 16)
        self.gen_spin.setValue(10)
        self.log_check = QCheckBox("Fit on the display (log/logicle) scale")
        self.log_check.setChecked(True)
        self.log_check.setToolTip("Checked: fit the sample's display-transformed values; unchecked: fit log10 of the raw values")
        # generation 0 (undivided): automatic, a typed intensity, or the median
        # of an undivided control population
        self.g0_mode = QComboBox()
        for label, key in self.G0_MODES:
            self.g0_mode.addItem(label, key)
        self.g0_spin = QDoubleSpinBox()
        self.g0_spin.setRange(0.0, 1e9)
        self.g0_spin.setDecimals(1)
        self.g0_spin.setValue(0.0)
        self.g0_spin.setToolTip("Generation-0 (undivided) peak intensity in data units")
        self.control_sample_combo = QComboBox()
        self.control_pop_combo = QComboBox()
        self.control_sample_combo.setToolTip("An undivided (unstimulated) control stained with the same dye")
        self.fix_g0_check = QCheckBox("Hold generation 0 at this position")
        self.fix_g0_check.setChecked(True)
        self.fix_g0_check.setToolTip("Unchecked: the position is only where the search for generation 0 starts")
        self.g0_info = QLabel("")
        form.addRow("Dye parameter:", self.param_combo)
        form.addRow("Bins:", self.bins_spin)
        form.addRow("Max generations:", self.gen_spin)
        form.addRow("", self.log_check)
        form.addRow("Generation 0:", self.g0_mode)
        form.addRow("Gen 0 intensity:", self.g0_spin)
        form.addRow("Control sample:", self.control_sample_combo)
        form.addRow("Control population:", self.control_pop_combo)
        form.addRow("", self.fix_g0_check)
        form.addRow("", self.g0_info)
        self.g0_mode.currentIndexChanged.connect(lambda *_: self._g0_mode_changed())
        self.control_sample_combo.currentIndexChanged.connect(lambda *_: self._refresh_control_populations())
        self.controller.gatingChanged.connect(lambda *_: self._refresh_control_populations())
        self._g0_mode_changed()

    G0_MODES = [("Auto (brightest peak)", "auto"), ("Intensity", "value"), ("Undivided control", "control")]

    def _g0_mode_changed(self):
        mode = self.g0_mode.currentData()
        self.form.setRowVisible(self.g0_spin, mode == "value")
        self.form.setRowVisible(self.control_sample_combo, mode == "control")
        self.form.setRowVisible(self.control_pop_combo, mode == "control")
        self.form.setRowVisible(self.fix_g0_check, mode != "auto")
        self.g0_info.setText("")

    def refresh(self):
        super().refresh()
        self._refresh_control_samples()

    def _refresh_control_samples(self):
        keep = self.control_sample_combo.currentData()
        self.control_sample_combo.blockSignals(True)
        self.control_sample_combo.clear()
        for s in self.controller.workspace.samples:
            self.control_sample_combo.addItem(s.name, s.id)
        idx = self.control_sample_combo.findData(keep) if keep is not None else -1
        self.control_sample_combo.setCurrentIndex(max(idx, 0) if self.control_sample_combo.count() else -1)
        self.control_sample_combo.blockSignals(False)
        self._refresh_control_populations()

    def _control_sample(self):
        sid = self.control_sample_combo.currentData()
        return self.controller.workspace.sample_by_id(sid) if sid is not None else None

    def _refresh_control_populations(self, pop_path: str | None = None):
        sample = self._control_sample()
        if pop_path is None:
            pop_path = self.control_pop_combo.currentData()
        self.control_pop_combo.blockSignals(True)
        self.control_pop_combo.clear()
        if sample is not None:
            sample.ensure_loaded()
            self.control_pop_combo.addItem("Ungated", "")
            for pop in sample.populations():
                self.control_pop_combo.addItem("  " * (pop.depth - 1) + pop.name, pop.path)
        idx = self.control_pop_combo.findData(pop_path) if pop_path is not None else -1
        self.control_pop_combo.setCurrentIndex(max(idx, 0) if self.control_pop_combo.count() else -1)
        self.control_pop_combo.blockSignals(False)

    def set_g0_auto(self):
        self.g0_mode.setCurrentIndex(self.g0_mode.findData("auto"))

    def set_g0_value(self, value: float, fix: bool = True):
        self.g0_mode.setCurrentIndex(self.g0_mode.findData("value"))
        self.g0_spin.setValue(float(value))
        self.fix_g0_check.setChecked(fix)

    def set_g0_control(self, sample, pop_path: str = "", fix: bool = True):
        """Take generation 0 from the median of an undivided control population."""
        self.g0_mode.setCurrentIndex(self.g0_mode.findData("control"))
        idx = self.control_sample_combo.findData(sample.id)
        if idx >= 0:
            self.control_sample_combo.setCurrentIndex(idx)
        self._refresh_control_populations(pop_path)
        self.fix_g0_check.setChecked(fix)

    def g0_setting(self, param: str) -> tuple[float | None, bool]:
        """(generation-0 position in data units or None for automatic, hold it fixed)."""
        mode = self.g0_mode.currentData()
        fix = self.fix_g0_check.isChecked()
        if mode == "value":
            v = self.g0_spin.value()
            return (v if v > 0 else None), fix
        if mode == "control":
            sample = self._control_sample()
            if sample is None or not sample.ensure_loaded():
                raise ValueError("choose an undivided control sample")
            pop = sample.root.find(self.control_pop_combo.currentData() or "")
            if pop is None:
                raise ValueError("choose an undivided control population")
            g0 = bio.control_g0(pop, param)
            self.g0_info.setText(f"Control median: {_fmt(g0)}")
            return g0, fix
        return None, False

    def _refresh_params(self, sample):
        self._fill_params(self.param_combo, sample, bio.default_dye_parameter(sample) if sample else None)

    def compute(self, population):
        param = self._param(self.param_combo)
        if not param:
            raise ValueError("no parameter")
        g0, fix = self.g0_setting(param)
        return bio.proliferation_for_population(
            population, param, bins=self.bins_spin.value(), max_generations=self.gen_spin.value(),
            display=self.log_check.isChecked(), g0=g0, fix_g0=fix and g0 is not None,
        )

    def show_result(self, r):
        curves = []
        for k in range(r.n_generations):
            curves.append((f"Gen {k}", r.curves[f"Gen {k}"], GEN_COLORS[k % len(GEN_COLORS)]))
        curves.append(("Model", r.curves["Model"], PHASE_COLORS["Model"]))
        name = self._param(self.param_combo) or ""
        self.plot.set_histogram(r.centers, r.counts, curves, xlabel=name if not r.log else f"log10 {name}")

    def describe(self, r):
        return f"{r.n_generations} generations, division index {r.division_index:.2f}, {r.percent_divided:.1f}% divided"

    def group_header(self):
        gens = [f"%Gen {k}" for k in range(self.gen_spin.value())]
        return ["Sample", "Population", "Parameter"] + gens + ["Division Index", "Proliferation Index", "%Divided", "Expansion Index", "Replication Index", "Gen 0 Peak", "RMS", "Events", "Note"]

    def group_row(self, sample, population, r):
        s = r.summary()
        gen0_peak = r.generation0
        if self.log_check.isChecked():
            # report the generation-0 peak in data units
            gen0_peak = population.sample.scale(self._param(self.param_combo)).inverse_scalar(float(r.positions[0]))
        s["Gen 0 Peak"] = gen0_peak
        return [sample.name, population.display_name, self._param(self.param_combo)] + [s.get(k, 0.0 if k.startswith("%Gen") else float("nan")) for k in self.group_header()[3:-1]] + [""]


class KineticsDialog(BiologyDialog):
    TITLE = "Kinetics"
    RUN_LABEL = "Run"
    STATISTICS = [("Median", "median"), ("Mean", "mean"), ("% Positive", "percent_positive")]
    BINNINGS = [("Equal count", "equal_count"), ("Equal width", "equal_width")]

    def _build_options(self, form):
        self.time_combo = QComboBox()
        self.param_combo = QComboBox()
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(2, 500)
        self.bins_spin.setValue(20)
        self.stat_combo = QComboBox()
        for label, key in self.STATISTICS:
            self.stat_combo.addItem(label, key)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-1e12, 1e12)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(1000.0)
        self.binning_combo = QComboBox()
        for label, key in self.BINNINGS:
            self.binning_combo.addItem(label, key)
        form.addRow("Time parameter:", self.time_combo)
        form.addRow("Parameter:", self.param_combo)
        form.addRow("Time bins:", self.bins_spin)
        form.addRow("Binning:", self.binning_combo)
        form.addRow("Statistic:", self.stat_combo)
        form.addRow("Positive above:", self.threshold_spin)
        self.stat_combo.currentIndexChanged.connect(self._stat_changed)
        self._stat_changed()

    def _stat_changed(self, *_):
        self.threshold_spin.setEnabled(self.stat_combo.currentData() == "percent_positive")

    def _refresh_params(self, sample):
        self._fill_params(self.time_combo, sample, bio.default_time_parameter(sample) if sample else None)
        default = None
        if sample is not None:
            default = next((n for n in sample.parameter_names() if bio._is_fluorescence(n)), None)
        self._fill_params(self.param_combo, sample, default)

    def compute(self, population):
        tparam, param = self._param(self.time_combo), self._param(self.param_combo)
        if not tparam or not param:
            raise ValueError("choose a time parameter and a parameter")
        stat = self.stat_combo.currentData()
        return bio.kinetics_for_population(
            population, tparam, param, n_bins=self.bins_spin.value(), statistic=stat,
            threshold=self.threshold_spin.value() if stat == "percent_positive" else None,
            binning=self.binning_combo.currentData(),
        )

    def _stat_label(self) -> str:
        return f"{self.stat_combo.currentText()} {self._param(self.param_combo) or ''}".strip()

    def show_result(self, r):
        self.plot.set_lines(r.centers, r.values, None if r.statistic == "percent_positive" else r.lo, None if r.statistic == "percent_positive" else r.hi,
                            xlabel=self._param(self.time_combo) or "Time", ylabel=self._stat_label())

    def describe(self, r):
        return f"overall {self._stat_label()} = {_fmt(r.overall)}, slope {_fmt(r.slope)} per time unit"

    def single_table(self, sample, population, r):
        header = list(bio.KINETICS_HEADER)
        header[2] = self._stat_label()
        return header, bio.kinetics_table(r)

    def group_header(self):
        return ["Sample", "Population", "Parameter", "Statistic", "Events", "Bins", "Overall", "Min Bin", "Max Bin", "Range % of Overall", "Slope / time", "Note"]

    def group_row(self, sample, population, r):
        s = r.summary()
        return [sample.name, population.display_name, self._param(self.param_combo), self.stat_combo.currentText()] + [s[k] for k in self.group_header()[4:-1]] + [""]


# =============================================================================
# extension hook
# =============================================================================
class BiologyTools:
    """What `install` returns: the actions, the ribbon band and open dialogs."""

    DIALOGS = {"cell_cycle": CellCycleDialog, "proliferation": ProliferationDialog, "kinetics": KineticsDialog}

    def __init__(self, window):
        self.window = window
        self.controller = window.controller
        self.dialogs: dict[str, BiologyDialog] = {}
        self.actions: dict[str, QAction] = {}
        specs = [
            ("cell_cycle", "Cell Cycle…", "histogram", "Fit G1 / S / G2-M fractions to a DNA-content histogram"),
            ("proliferation", "Proliferation…", "overlay", "Model dye-dilution generations (CFSE, CellTrace) and proliferation indices"),
            ("kinetics", "Kinetics…", "graph", "Plot a statistic of a parameter against acquisition time"),
        ]
        for key, text, ic, tip in specs:
            act = QAction(icon(ic), text, window)
            act.setToolTip(tip)
            act.setStatusTip(tip)
            act.triggered.connect(lambda _=False, k=key: self.open(k))
            self.actions[key] = act
        self.band = None
        page = getattr(window, "ribbon_pages", {}).get("tools")
        if page is not None:
            self.band = page.add_band("Biology")
            for act in self.actions.values():
                self.band.add_large(act)
        menu = getattr(window, "menus", {}).get("tools")
        self.menu = menu
        if menu is not None:
            menu.addSeparator()
            for act in self.actions.values():
                menu.addAction(act)

    def open(self, key: str) -> BiologyDialog:
        dlg = self.dialogs.get(key)
        if dlg is None:
            dlg = self.DIALOGS[key](self.controller, self.window)
            self.dialogs[key] = dlg
        else:
            dlg.refresh()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg


def install(window) -> BiologyTools:
    return BiologyTools(window)
