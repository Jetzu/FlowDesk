"""Quality-control tools (Data Cleaning, MESF Calibration) as a main-window extension.

`install(window)` adds a "Quality Control" band to the Tools ribbon page and
the same entries to the Tools menu. All computations live in
`flowdesk.core.qc` and `flowdesk.core.calibration`; the dialogs collect the
settings, plot the result, show / export tables and create the populations
or calibrated parameters through the controller.

The per-sample loops ("Apply to all samples in group", "Apply to samples…")
run in a worker thread with a progress dialog (as in `tools_analysis`): the
worker loads the files and computes, the GUI thread then changes the
workspace inside one undo step. `apply_to_group()` / `apply()` remain the
synchronous versions of the same code; `start_group()` / `start_apply()` are
what the buttons call.
"""
from __future__ import annotations

import math
import weakref
from contextlib import contextmanager, nullcontext

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import calibration as cal
from ..core import qc
from .icons import icon

TRACE_COLORS = ["#1f6fd1", "#d1352b", "#2e9e44", "#e67e22", "#6a3fc1", "#16a3a3", "#b03a9a", "#7f5539", "#c9a400", "#555555"]
FLAG_FILL = QColor(209, 53, 43, 46)
RATE_STRIP = QColor("#d1352b")
SIGNAL_STRIP = QColor("#e67e22")


# =============================================================================
# small helpers
# =============================================================================
def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    v = math.ceil(lo / step) * step
    out = []
    while v <= hi + 1e-9 * step:
        out.append(v)
        v += step
    return out


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-3:
        return f"{v:.0e}"
    return f"{v:g}"


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


def _write_csv(path, header: list, rows: list[list]):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(v, float) and not math.isfinite(v)) else v for v in r])


def _begin(controller, label: str, kind=None):
    """controller.begin_change with an undo kind when the controller takes one."""
    if kind is None:
        controller.begin_change(label)
        return
    try:
        controller.begin_change(label, kind)
    except TypeError:
        controller.begin_change(label)


@contextmanager
def _undo_step(controller, label: str, kind=None):
    """One undo step for the whole block (controller.undo_group), falling back
    to a single begin_change for controllers without undo groups."""
    group = getattr(controller, "undo_group", None)
    if group is None:
        _begin(controller, label, kind)
        yield
        return
    with group(label, kind):
        yield


class Cancelled(Exception):
    """Raised by a worker function when the user pressed Cancel."""


def _check_cancel(should_cancel):
    if should_cancel is not None and should_cancel():
        raise Cancelled()


class _Worker(QThread):
    """Runs `fn(progress=..., should_cancel=...)` in its own thread.

    As tools_analysis._Worker: a QThread subclass owned by the dialog, so Qt
    deletes it with the dialog. `fn` must not touch widgets; its result is
    handed to the GUI thread through `done_result` (a queued connection).
    """

    done_result = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancelled = False

    def run(self):
        try:
            result = self.fn(progress=self.progress.emit, should_cancel=lambda: self.cancelled)
        except Cancelled:
            self.failed.emit("")
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.done_result.emit(result)


class _Axes:
    """Maps data coordinates into a plot rectangle and draws frame + ticks."""

    def __init__(self, area: QRectF, x0: float, x1: float, y0: float, y1: float):
        self.area = area
        self.x0, self.x1 = x0, x1 if x1 > x0 else x0 + 1.0
        self.y0, self.y1 = y0, y1 if y1 > y0 else y0 + 1.0

    def px(self, x: float) -> float:
        return self.area.left() + (x - self.x0) / (self.x1 - self.x0) * self.area.width()

    def py(self, y: float) -> float:
        return self.area.bottom() - (y - self.y0) / (self.y1 - self.y0) * self.area.height()

    def draw(self, p: QPainter, xlabel: str = "", ylabel: str = "", x_ticks: bool = True):
        a = self.area
        p.setPen(QPen(QColor("#444444"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(a)
        if x_ticks:
            for t in _nice_ticks(self.x0, self.x1):
                xx = self.px(t)
                p.drawLine(QPointF(xx, a.bottom()), QPointF(xx, a.bottom() + 4))
                p.drawText(QRectF(xx - 30, a.bottom() + 4, 60, 14), Qt.AlignmentFlag.AlignHCenter, _fmt_tick(t))
            if xlabel:
                p.drawText(QRectF(a.left(), a.bottom() + 17, a.width(), 14), Qt.AlignmentFlag.AlignHCenter, xlabel)
        for t in _nice_ticks(self.y0, self.y1, 4):
            yy = self.py(t)
            p.drawLine(QPointF(a.left() - 4, yy), QPointF(a.left(), yy))
            p.drawText(QRectF(a.left() - 58, yy - 7, 52, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _fmt_tick(t))
        if ylabel:
            p.save()
            p.translate(a.left() - 50, a.center().y())
            p.rotate(-90)
            p.drawText(QRectF(-a.height() / 2, -12, a.height(), 14), Qt.AlignmentFlag.AlignHCenter, ylabel)
            p.restore()


def _small_font(widget: QWidget) -> QFont:
    font = QFont(widget.font())
    font.setPointSizeF(max(7.0, font.pointSizeF() * 0.85))
    return font


# =============================================================================
# plots
# =============================================================================
class QCPlot(QWidget):
    """Flow rate against time (top) and per-parameter bin medians (bottom),
    flagged bins shaded; strips mark which check flagged a bin."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.result: qc.QCResult | None = None
        self.time_label = "Time"
        self.max_traces = 8

    def clear(self):
        self.result = None
        self.update()

    def set_result(self, result: qc.QCResult, time_label: str = "Time"):
        self.result = result
        self.time_label = time_label
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        p.setFont(_small_font(self))
        r = self.result
        if r is None or r.n_bins < 1:
            p.setPen(QColor("#888888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Run the checks to see flow rate and signal over time")
            p.end()
            return
        left, right = 66, 12
        w = max(self.width() - left - right, 20)
        h_total = max(self.height() - 12 - 40, 40)
        h_top = h_total * 0.45
        top_area = QRectF(left, 10, w, h_top)
        bot_area = QRectF(left, 10 + h_top + 10, w, h_total - h_top - 10)
        edges = r.edges
        x0, x1 = float(edges[0]), float(edges[-1])
        rates = r.rates
        rmax = float(np.nanmax(rates)) if rates.size else 1.0
        top = _Axes(top_area, x0, x1, 0.0, (rmax * 1.1) or 1.0)
        meds = list(r.medians.items())[: self.max_traces]
        vals = np.concatenate([m[np.isfinite(m)] for _, m in meds]) if meds else np.zeros(0)
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            pad = (hi - lo) * 0.1 or 0.05
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = 0.0, 1.0
        bot = _Axes(bot_area, x0, x1, lo, hi)
        # shading of flagged bins over both panels
        rate_flags = r.bin_flags.get("rate")
        sig_flags = np.zeros(r.n_bins, dtype=bool)
        for k, f in r.bin_flags.items():
            if k.startswith("signal:"):
                sig_flags |= f
        for i in range(r.n_bins):
            if not r.flagged_bins[i]:
                continue
            xa, xb = top.px(edges[i]), top.px(edges[i + 1])
            for area in (top_area, bot_area):
                p.fillRect(QRectF(xa, area.top(), max(xb - xa, 1.0), area.height()), FLAG_FILL)
            if rate_flags is not None and rate_flags[i]:
                p.fillRect(QRectF(xa, top_area.top(), max(xb - xa, 1.0), 4), RATE_STRIP)
            if sig_flags[i]:
                p.fillRect(QRectF(xa, bot_area.top(), max(xb - xa, 1.0), 4), SIGNAL_STRIP)
        top.draw(p, "", "Events / time unit", x_ticks=False)
        # x tick labels under the top panel are omitted; the panels share the axis
        p.setPen(QPen(QColor("#1f6fd1"), 1.6))
        path = QPainterPath()
        for i in range(r.n_bins):
            ya = top.py(float(rates[i]))
            pa, pb = QPointF(top.px(edges[i]), ya), QPointF(top.px(edges[i + 1]), ya)
            if i == 0:
                path.moveTo(pa)
            else:
                path.lineTo(pa)
            path.lineTo(pb)
        p.save()
        p.setClipRect(top_area)
        p.drawPath(path)
        if math.isfinite(r.rate_center):
            pen = QPen(QColor("#333333"), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            yc = top.py(r.rate_center)
            p.drawLine(QPointF(top_area.left(), yc), QPointF(top_area.right(), yc))
            if math.isfinite(r.rate_threshold):
                pen.setStyle(Qt.PenStyle.DotLine)
                p.setPen(pen)
                for yv in (r.rate_center - r.rate_threshold, r.rate_center + r.rate_threshold):
                    yy = top.py(yv)
                    p.drawLine(QPointF(top_area.left(), yy), QPointF(top_area.right(), yy))
        p.restore()
        bot.draw(p, self.time_label, "Median (display)")
        centers = 0.5 * (edges[:-1] + edges[1:])
        p.save()
        p.setClipRect(bot_area)
        for k, (name, med) in enumerate(meds):
            p.setPen(QPen(QColor(TRACE_COLORS[k % len(TRACE_COLORS)]), 1.4))
            path = QPainterPath()
            first = True
            for xc, yc in zip(centers, med):
                if not math.isfinite(yc):
                    first = True
                    continue
                pt = QPointF(bot.px(float(xc)), bot.py(float(yc)))
                if first:
                    path.moveTo(pt)
                    first = False
                else:
                    path.lineTo(pt)
            p.drawPath(path)
        p.restore()
        # legend
        y = bot_area.top() + 6
        for k, (name, _) in enumerate(meds):
            col = QColor(TRACE_COLORS[k % len(TRACE_COLORS)])
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(bot_area.right() - 96, y + 6), QPointF(bot_area.right() - 82, y + 6))
            p.setPen(QColor("#222222"))
            p.drawText(QRectF(bot_area.right() - 78, y, 76, 13), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
            y += 13
        if len(r.medians) > len(meds):
            p.drawText(QRectF(bot_area.right() - 78, y, 76, 13), Qt.AlignmentFlag.AlignLeft, f"+{len(r.medians) - len(meds)} more")
        p.end()


class CalibrationPlot(QWidget):
    """Bead histogram with the detected peaks (left) and the log-log fit (right)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(460, 260)
        self.centers = np.zeros(0)
        self.counts = np.zeros(0)
        self.peaks: list[cal.BeadPeak] = []
        self.xlabel = ""
        self.fit: cal.CalibrationFit | None = None
        self.points: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # intensity, mesf, used

    def clear(self):
        self.centers = np.zeros(0)
        self.peaks = []
        self.fit = None
        self.points = None
        self.update()

    def set_histogram(self, centers, counts, peaks, xlabel: str = ""):
        self.centers = np.asarray(centers, dtype=float)
        self.counts = np.asarray(counts, dtype=float)
        self.peaks = list(peaks)
        self.xlabel = xlabel
        self.update()

    def set_fit(self, fit: cal.CalibrationFit | None, intensities=None, mesf=None, used=None):
        self.fit = fit
        if intensities is not None and mesf is not None:
            n = len(intensities)
            self.points = (np.asarray(intensities, dtype=float), np.asarray(mesf, dtype=float), np.asarray(used if used is not None else [True] * n, dtype=bool))
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        p.setFont(_small_font(self))
        if self.centers.size == 0 and self.fit is None:
            p.setPen(QColor("#888888"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Find the bead peaks to see the histogram")
            p.end()
            return
        h = max(self.height() - 12 - 36, 40)
        total_w = max(self.width() - 20, 60)
        w_hist = total_w * 0.58 - 60
        hist_area = QRectF(60, 10, max(w_hist, 20), h)
        fit_area = QRectF(hist_area.right() + 66, 10, max(total_w - hist_area.width() - 66 - 60 + 10, 20), h)
        if self.centers.size:
            ymax = float(self.counts.max()) * 1.08 or 1.0
            step = self.centers[1] - self.centers[0] if self.centers.size > 1 else 1.0
            ax = _Axes(hist_area, float(self.centers[0] - step / 2), float(self.centers[-1] + step / 2), 0.0, ymax)
            p.save()
            p.setClipRect(hist_area)
            for k, pk in enumerate(self.peaks):
                xa, xb = ax.px(pk.lo), ax.px(pk.hi)
                p.fillRect(QRectF(xa, hist_area.top(), max(xb - xa, 1), hist_area.height()), QColor(31, 111, 209, 22 if k % 2 else 40))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#8a93a3"))
            for xc, yc in zip(self.centers, self.counts):
                if yc > 0:
                    p.drawRect(QRectF(ax.px(xc - step / 2), ax.py(yc), max(ax.px(xc + step / 2) - ax.px(xc - step / 2), 0.6), ax.py(0) - ax.py(yc)))
            p.restore()
            ax.draw(p, self.xlabel, "Count")
            pen = QPen(QColor("#d1352b"), 1.2)
            pen.setStyle(Qt.PenStyle.DashLine)
            for k, pk in enumerate(self.peaks, start=1):
                xx = ax.px(pk.position)
                p.setPen(pen)
                p.drawLine(QPointF(xx, hist_area.top() + 14), QPointF(xx, hist_area.bottom()))
                p.setPen(QColor("#d1352b"))
                p.drawText(QRectF(xx - 12, hist_area.top() + 1, 24, 13), Qt.AlignmentFlag.AlignHCenter, str(k))
        # fit panel
        pts = self.points
        if pts is not None:
            x, y, used = pts
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        else:
            ok = np.zeros(0, dtype=bool)
        if ok.any() or self.fit is not None:
            lx = np.log10(x[ok]) if ok.any() else np.zeros(0)
            ly = np.log10(y[ok]) if ok.any() else np.zeros(0)
            if self.fit is not None and lx.size == 0:
                lx = np.log10(np.asarray([v for v in self.fit.intensities if v > 0] or [1.0, 10.0]))
                ly = self.fit.slope * lx + self.fit.intercept
            xa0, xa1 = float(lx.min()), float(lx.max())
            ya0, ya1 = float(ly.min()), float(ly.max())
            padx = (xa1 - xa0) * 0.1 or 0.5
            pady = (ya1 - ya0) * 0.1 or 0.5
            ax2 = _Axes(fit_area, xa0 - padx, xa1 + padx, ya0 - pady, ya1 + pady)
            ax2.draw(p, "log10 intensity", f"log10 {self.fit.units if self.fit else cal.UNITS}")
            p.save()
            p.setClipRect(fit_area)
            if self.fit is not None:
                p.setPen(QPen(QColor("#111111"), 1.6))
                xs = (ax2.x0, ax2.x1)
                p.drawLine(QPointF(ax2.px(xs[0]), ax2.py(self.fit.slope * xs[0] + self.fit.intercept)),
                           QPointF(ax2.px(xs[1]), ax2.py(self.fit.slope * xs[1] + self.fit.intercept)))
            if pts is not None:
                for xi, yi, ui in zip(x[ok], y[ok], used[ok]):
                    c = QPointF(ax2.px(math.log10(xi)), ax2.py(math.log10(yi)))
                    p.setPen(QPen(QColor("#1f6fd1"), 1.4))
                    p.setBrush(QColor("#1f6fd1") if ui else Qt.BrushStyle.NoBrush)
                    p.drawEllipse(c, 3.5, 3.5)
            p.restore()
            if self.fit is not None:
                p.setPen(QColor("#222222"))
                p.drawText(QRectF(fit_area.left() + 6, fit_area.top() + 4, fit_area.width() - 12, 28), Qt.AlignmentFlag.AlignLeft,
                           f"slope {self.fit.slope:.4f}\nr² {self.fit.r2:.5f}")
        p.end()


# =============================================================================
# dialog base
# =============================================================================
class _ToolDialog(QDialog):
    """Sample / population combos that follow the tree selection, a results
    table with Copy and Export CSV."""

    TITLE = "Tool"

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle(self.TITLE)
        self.header: list[str] = []
        self.rows: list[list] = []
        self.sample_combo = QComboBox()
        self.pop_combo = QComboBox()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.copy_button = QPushButton(icon("copy"), "Copy")
        self.export_button = QPushButton(icon("export"), "Export CSV…")
        self.copy_button.clicked.connect(self.copy_results)
        self.export_button.clicked.connect(lambda: self.export_csv())
        self._worker: _Worker | None = None
        self._progress: QProgressDialog | None = None
        self._job_apply = None
        self._job_buttons: tuple = ()

    # ------------------------------------------------------- threaded jobs
    def busy(self) -> bool:
        """A threaded job (group / multi-sample apply) is running."""
        return self._worker is not None

    def _start_job(self, title: str, total: int, compute, apply, buttons=()) -> bool:
        """Run `compute(progress, should_cancel)` in a worker thread with a
        progress dialog ("sample i of n", Cancel stops after the current
        sample and changes nothing), then `apply(result)` in the GUI thread."""
        if self._worker is not None:
            return False
        worker = _Worker(compute, self)
        self._worker = worker
        self._job_apply = apply
        self._job_buttons = tuple(buttons)
        for b in self._job_buttons:
            b.setEnabled(False)
        worker.done_result.connect(self._job_finished)
        worker.failed.connect(self._job_failed)
        worker.progress.connect(self._job_progress)
        prog = QProgressDialog("Starting…", "Cancel", 0, max(1, int(total)), self)
        prog.setWindowTitle(title)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setAutoClose(False)
        prog.setAutoReset(False)
        prog.setValue(0)
        prog.canceled.connect(self._job_cancel)
        prog.show()
        self._progress = prog
        worker.start()
        return True

    def _job_progress(self, value: int, text: str):
        prog = self._progress
        if prog is None:
            return
        prog.setLabelText(text)
        # a modal progress dialog processes events in setValue: the job may
        # finish (and _end_job run) inside it, so nothing may follow here
        prog.setValue(min(int(value), prog.maximum()))

    def _job_cancel(self):
        if self._worker is not None:
            self._worker.cancelled = True
            if self._progress is not None:
                self._progress.setLabelText("Cancelling after the current sample…")

    def _end_job(self) -> bool:
        """Stop waiting for the worker; returns whether it was cancelled."""
        worker = self._worker
        cancelled = bool(worker is not None and worker.cancelled)
        if worker is not None:
            # the worker emits its result just before run() returns
            if worker.isRunning():
                worker.wait(10000)
            worker.deleteLater()
        self._worker = None
        prog, self._progress = self._progress, None
        if prog is not None:  # closed, deleted with the dialog (as in tools_analysis)
            prog.canceled.disconnect(self._job_cancel)
            prog.close()
        for b in self._job_buttons:
            b.setEnabled(True)
        self._job_buttons = ()
        return cancelled

    def _job_finished(self, result):
        apply, self._job_apply = self._job_apply, None
        if self._end_job():
            self.status.setText("Cancelled; nothing was changed.")
            return
        if apply is not None:
            apply(result)

    def _job_failed(self, message: str):
        self._job_apply = None
        self._end_job()
        self.status.setText(f"Failed: {message}" if message else "Cancelled; nothing was changed.")

    def done(self, r):
        worker = self._worker
        if worker is not None:
            worker.cancelled = True
            self._job_apply = None
            worker.wait(30000)
            self._end_job()
        super().done(r)

    def _connect_controller(self):
        self.sample_combo.currentIndexChanged.connect(self._sample_changed)
        self.controller.samplesChanged.connect(self._on_samples_changed)
        self.controller.workspaceReplaced.connect(self._on_workspace_replaced)
        self.controller.gatingChanged.connect(self._on_gating_changed)

    def _on_samples_changed(self, *_):
        self.refresh(use_selection=False)

    def _on_workspace_replaced(self, *_):
        self.refresh(use_selection=True)

    def _on_gating_changed(self, *_):
        self._refresh_populations()

    def _table_panel(self) -> QWidget:
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.table)
        br = QHBoxLayout()
        br.addStretch(1)
        br.addWidget(self.copy_button)
        br.addWidget(self.export_button)
        bl.addLayout(br)
        return bottom

    # ---------------------------------------------------------------- combos
    def current_sample(self):
        sid = self.sample_combo.currentData()
        return self.controller.workspace.sample_by_id(sid) if sid is not None else None

    def current_population(self):
        sample = self.current_sample()
        if sample is None:
            return None
        return sample.root.find(self.pop_combo.currentData() or "")

    def refresh(self, use_selection: bool = True):
        ws = self.controller.workspace
        selected = self.controller.selected[0] if (use_selection and self.controller.selected) else None
        current = self.current_sample()
        pop_path = selected[1].path if selected is not None else self.pop_combo.currentData()
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        for s in ws.samples:
            self.sample_combo.addItem(s.name, s.id)
        target = selected[0] if selected is not None else current
        idx = self.sample_combo.findData(target.id) if target is not None else -1
        self.sample_combo.setCurrentIndex(max(idx, 0) if self.sample_combo.count() else -1)
        self.sample_combo.blockSignals(False)
        self._sample_changed(pop_path=pop_path)

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

    def _refresh_params(self, sample):
        pass

    def set_population(self, sample, population):
        idx = self.sample_combo.findData(sample.id)
        if idx >= 0:
            self.sample_combo.setCurrentIndex(idx)
        self._refresh_populations(population.path)

    @staticmethod
    def _fill_params(combo: QComboBox, sample, default: str | None, names=None, extra: list[tuple[str, object]] = ()):
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for label, data in extra:
            combo.addItem(label, data)
        names = names if names is not None else (sample.parameter_names() if sample is not None else [])
        for n in names:
            combo.addItem(sample.param_label(n), n)
        idx = combo.findData(current) if current is not None else -1
        if idx < 0 and default is not None:
            idx = combo.findData(default)
        combo.setCurrentIndex(max(idx, 0) if combo.count() else -1)
        combo.blockSignals(False)

    @staticmethod
    def set_param(combo: QComboBox, name):
        idx = combo.findData(name)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ----------------------------------------------------------------- table
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
        _write_csv(path, self.header, self.rows)
        return str(path)


def _spin(lo: float, hi: float, value: float, decimals: int = 1, step: float = 0.5, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


# =============================================================================
# data cleaning
# =============================================================================
def _run_qc(population, request: dict) -> qc.QCResult:
    """The checks of a DataCleaningDialog.request() on one population (no widgets)."""
    settings = dict(request["settings"])
    if not settings.get("checks"):
        raise ValueError("no check selected")
    return qc.qc_for_population(population, request["time_param"], list(request["params"]), **settings)


class DataCleaningDialog(_ToolDialog):
    """FlowAI / PeacoQC-style data cleaning of one population over time."""

    TITLE = "Data Cleaning"
    MODES = [("Flag parameter (QC_flag gate)", "flag"), ("Time gates only", "time")]

    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        self.resize(1040, 720)
        self.result: qc.QCResult | None = None
        # what self.result was computed from: weak references to the sample and
        # its event data, the population path and the request (settings)
        self._result_key: tuple | None = None
        self._result_stale = False  # scales / compensation changed since
        self._param_choice: set[str] | None = None  # parameters the user ticked

        self.time_combo = QComboBox()
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(10, 2000)
        self.bins_spin.setValue(100)
        self.rate_check = QCheckBox("Flow rate")
        self.rate_check.setChecked(True)
        self.rate_z = _spin(0.5, 20, 3.0)
        self.rate_min = _spin(0, 100, 10.0, 0, 5, " %")
        self.rate_min.setToolTip("Smallest rate change (relative to the median rate) that counts")
        self.signal_check = QCheckBox("Signal stability")
        self.signal_check.setChecked(True)
        self.signal_z = _spin(0.5, 20, 3.0)
        self.signal_min = _spin(0, 50, 2.0, 1, 0.5, " %")
        self.signal_min.setToolTip("Smallest shift of a bin median, in percent of the display axis")
        self.range_check = QCheckBox("Dynamic range")
        self.range_check.setChecked(True)
        self.pile_spin = QSpinBox()
        self.pile_spin.setRange(1, 100000)
        self.pile_spin.setValue(3)
        self.pile_spin.setToolTip("Events sitting exactly on the data maximum count as saturated from this many on")
        self.low_check = QCheckBox("also minimum")
        self.low_check.setToolTip("Also remove events piled up on the data minimum (clipped)")
        self.cp_check = QCheckBox("Change-point segments")
        self.cp_check.setChecked(True)
        self.cp_check.setToolTip("Flag whole stretches whose level shifted (catches long shifted segments)")
        self.param_list = QListWidget()
        self.param_list.setMinimumHeight(90)
        self.param_list.itemChanged.connect(self._param_ticked)

        form = QFormLayout()
        form.addRow("Sample:", self.sample_combo)
        form.addRow("Population:", self.pop_combo)
        form.addRow("Time parameter:", self.time_combo)
        form.addRow("Time bins:", self.bins_spin)

        def row(*widgets):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            for x in widgets:
                h.addWidget(x) if not isinstance(x, str) else h.addWidget(QLabel(x))
            h.addStretch(1)
            return w

        form.addRow(row(self.rate_check))
        form.addRow("   z / min change:", row(self.rate_z, self.rate_min))
        form.addRow(row(self.signal_check))
        form.addRow("   z / min shift:", row(self.signal_z, self.signal_min))
        form.addRow(row(self.range_check, self.low_check))
        form.addRow("   min pile-up:", self.pile_spin)
        form.addRow(row(self.cp_check))
        form.addRow("Parameters:", self.param_list)

        self.run_button = QPushButton(icon("play"), "Run")
        self.name_edit = QLineEdit(qc.DEFAULT_NAME)
        self.mode_combo = QComboBox()
        for label, key in self.MODES:
            self.mode_combo.addItem(label, key)
        self.create_button = QPushButton(icon("check"), "Create QC population")
        self.group_button = QPushButton(icon("table"), "Apply to all samples in group")
        out_form = QFormLayout()
        out_form.addRow("Name:", self.name_edit)
        out_form.addRow("Mode:", self.mode_combo)

        settings = QGroupBox("Checks")
        sl = QVBoxLayout(settings)
        sl.addLayout(form)
        sl.addWidget(self.run_button)
        sl.addLayout(out_form)
        sl.addWidget(self.create_button)
        sl.addWidget(self.group_button)
        sl.addWidget(self.status)
        sl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings)
        scroll.setMinimumWidth(330)

        self.plot = QCPlot()
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(scroll)
        top.addWidget(self.plot)
        top.setStretchFactor(1, 1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self._table_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self.run_button.clicked.connect(lambda: self.run())
        self.create_button.clicked.connect(lambda: self.apply())
        self.group_button.clicked.connect(lambda: self.start_group())
        for chk, widgets in ((self.rate_check, (self.rate_z, self.rate_min)), (self.signal_check, (self.signal_z, self.signal_min)), (self.range_check, (self.pile_spin, self.low_check))):
            chk.toggled.connect(lambda on, ws=widgets: [w.setEnabled(on) for w in ws])
        self.time_combo.currentIndexChanged.connect(self._time_changed)
        self._connect_controller()
        self.controller.scalesChanged.connect(self._on_data_changed)
        self.controller.compensationChanged.connect(self._on_data_changed)
        self.refresh(use_selection=True)

    # ---------------------------------------------------------- stale results
    def _result_sample(self):
        key = self._result_key
        return key[0]() if key is not None else None

    def _drop_result(self):
        """Forget the last result (it belongs to data that is gone)."""
        self.result = None
        self._result_key = None
        self._result_stale = False
        self.plot.clear()
        self._set_table([], [])
        self.status.setText("")

    def _on_workspace_replaced(self, *_):
        # sample ids restart in every workspace: a result of the old one must
        # never be matched to a sample of the new one
        self._drop_result()
        super()._on_workspace_replaced()

    def _on_samples_changed(self, *_):
        s = self._result_sample()
        if self.result is not None and (s is None or not any(x is s for x in self.controller.workspace.samples)):
            self._drop_result()
        super()._on_samples_changed()

    def _on_data_changed(self, sample_id=None):
        """Scales / compensation changed: the result no longer matches the
        sample's values; "Create" recomputes it (the plot stays until then)."""
        s = self._result_sample()
        if self._result_key is not None and (s is None or sample_id is None or sample_id == s.id):
            self._result_stale = True

    def _remember(self, result: qc.QCResult, population, request: dict):
        sample = population.sample
        self.result = result
        self._result_key = (weakref.ref(sample), weakref.ref(sample.fcs) if sample.fcs is not None else None, population.path, request)
        self._result_stale = False

    def _result_matches(self, population, request: dict) -> bool:
        """self.result was computed on this very sample object (not just the
        same id), on its current events and population membership, with the
        current settings."""
        r, key = self.result, self._result_key
        if r is None or key is None or self._result_stale:
            return False
        sref, fref, path, req = key
        sample = population.sample
        if sample is None or sref() is not sample or path != population.path or req != request:
            return False
        if (fref() if fref is not None else None) is not sample.fcs:
            return False  # the data were reloaded
        mask = population.mask()
        return r.event_index is not None and mask.size == sample.n_events and np.array_equal(np.flatnonzero(mask), r.event_index)

    # ------------------------------------------------------------ parameters
    def _refresh_params(self, sample):
        default_t = qc.default_time_parameter(sample) if sample is not None else None
        names = sample.parameter_names() if sample is not None else []
        self._fill_params(self.time_combo, sample, default_t, names=names, extra=[("(event order)", None)])
        if sample is not None and self.time_combo.currentData() is None and default_t is not None:
            self.set_param(self.time_combo, default_t)
        self._fill_param_list(sample)
        self._time_changed()

    def _fill_param_list(self, sample):
        self.param_list.blockSignals(True)
        self.param_list.clear()
        if sample is not None:
            tp = self.time_param()
            defaults = set(qc.default_qc_parameters(sample, tp))
            chosen = self._param_choice
            for n in sample.parameter_names():
                if n == tp or (n in sample.computed and n.startswith(qc.FLAG_PARAM)):
                    continue
                item = QListWidgetItem(sample.param_label(n))
                item.setData(Qt.ItemDataRole.UserRole, n)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                on = (n in chosen) if chosen is not None else (n in defaults)
                item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                self.param_list.addItem(item)
        self.param_list.blockSignals(False)

    def _param_ticked(self, *_):
        self._param_choice = set(self.selected_params())

    def _time_changed(self, *_):
        has_time = self.time_param() is not None
        self.rate_check.setEnabled(has_time)
        if self.mode_combo.count():
            item = self.mode_combo.model().item(1)
            if item is not None:
                item.setEnabled(has_time)
            if not has_time:
                self.mode_combo.setCurrentIndex(0)

    def time_param(self) -> str | None:
        return self.time_combo.currentData()

    def selected_params(self) -> list[str]:
        out = []
        for i in range(self.param_list.count()):
            it = self.param_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def set_params(self, names: list[str]):
        self._param_choice = set(names)
        self.param_list.blockSignals(True)
        for i in range(self.param_list.count()):
            it = self.param_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in self._param_choice else Qt.CheckState.Unchecked)
        self.param_list.blockSignals(False)

    def checks(self) -> tuple[str, ...]:
        out = []
        if self.rate_check.isChecked() and self.time_param() is not None:
            out.append("rate")
        if self.signal_check.isChecked():
            out.append("signal")
        if self.range_check.isChecked():
            out.append("range")
        return tuple(out)

    def settings(self) -> dict:
        return {
            "checks": self.checks(),
            "n_bins": self.bins_spin.value(),
            "z": self.signal_z.value(),
            "rate_z": self.rate_z.value(),
            "signal_z": self.signal_z.value(),
            "min_rate_change": self.rate_min.value() / 100.0,
            "min_shift": self.signal_min.value() / 100.0,
            "changepoints": self.cp_check.isChecked(),
            "min_pile": self.pile_spin.value(),
            "check_low": self.low_check.isChecked(),
        }

    def mode(self) -> str:
        return self.mode_combo.currentData() or "flag"

    def set_mode(self, mode: str):
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    # ---------------------------------------------------------------- actions
    def request(self) -> dict:
        """What the checks run with: time parameter, parameters and settings."""
        return {"time_param": self.time_param(), "params": list(self.selected_params()), "settings": self.settings()}

    def compute(self, population, request: dict | None = None) -> qc.QCResult:
        return _run_qc(population, request if request is not None else self.request())

    def run(self) -> qc.QCResult | None:
        """Run the checks on the chosen population; returns the result (or None)."""
        pop = self.current_population()
        if pop is None:
            self.status.setText("No population selected.")
            return None
        request = self.request()
        try:
            result = self.compute(pop, request)
        except (ValueError, KeyError, RuntimeError) as exc:
            self.result = None
            self._result_key = None
            self.plot.clear()
            self.status.setText(f"Could not check {pop.display_name}: {exc}")
            return None
        self._remember(result, pop, request)
        self.plot.set_result(result, self.time_param() or "Event order")
        self._set_table(qc.TABLE_HEADER, qc.result_table(result))
        self.status.setText(f"{pop.sample.name} / {pop.display_name}: {self.describe(result)}")
        return result

    @staticmethod
    def describe(r: qc.QCResult) -> str:
        parts = [f"{qc.CHECK_LABELS[k]} {r.percent_removed.get(k, 0.0):.1f}%" for k in r.checks]
        parts = [f"{qc.CHECK_LABELS[k]} not tested" if r.bins_tested(k) == 0 else p for k, p in zip(r.checks, parts)]
        text = f"{r.n_good} of {r.n_events} events kept ({r.percent_removed.get('total', 0.0):.1f}% removed: {', '.join(parts)})"
        if r.notes:
            text += "; " + "; ".join(f"{k}: {v}" for k, v in r.notes.items())
        return text

    def _result_for(self, pop) -> qc.QCResult:
        """The last result when it was computed on this population with the
        current settings, otherwise a fresh one."""
        request = self.request()
        if self._result_matches(pop, request):
            return self.result
        r = self.compute(pop, request)
        self._remember(r, pop, request)
        return r

    def apply(self, name: str | None = None, mode: str | None = None):
        """Create the QC population under the chosen population (runs the
        checks first when needed). Returns the created populations."""
        pop = self.current_population()
        if pop is None:
            self.status.setText("No population selected.")
            return None
        name = (name or self.name_edit.text() or qc.DEFAULT_NAME).strip()
        mode = mode or self.mode()
        try:
            result = self._result_for(pop)
            if result is not self.result or self.plot.result is not result:
                self.plot.set_result(result, self.time_param() or "Event order")
                self._set_table(qc.TABLE_HEADER, qc.result_table(result))
            if mode == "time" and not result.time_param:
                raise ValueError("time gates need a time parameter")
            if mode == "time" and not result.intervals:
                raise ValueError("no clean time interval")
            c = self.controller
            c.begin_change("data cleaning")
            created = qc.apply_qc(pop, result, name, mode)
        except (ValueError, KeyError, RuntimeError) as exc:
            self.status.setText(f"Could not create {name}: {exc}")
            return None
        c.mark_modified()
        self._notify(pop.sample.id)
        self.status.setText(f"Created {created[0].name} below {pop.display_name}: {created[0].count} of {result.n_events} events" + (f" ({len(created) - 1} time intervals)" if len(created) > 1 else ""))
        return created

    create_population = apply

    def _notify(self, sample_id):
        c = self.controller
        c.samplesChanged.emit()  # a new computed parameter
        c.gatingChanged.emit(sample_id)
        c.statusMessage.emit("Data cleaning population created")

    def _group_plan(self, name: str | None, mode: str | None) -> dict:
        pop = self.current_population()
        group = self.controller.current_group
        return {
            "current": pop,
            "path": pop.path if pop is not None else "",
            "name": (name or self.name_edit.text() or qc.DEFAULT_NAME).strip(),
            "mode": mode or self.mode(),
            "group": group,
            "members": list(self.controller.workspace.members(group)),
            "request": self.request(),
        }

    @staticmethod
    def _compute_group(plan: dict, progress=None, should_cancel=None) -> list[tuple]:
        """Load every member and run the checks on its population (no widgets,
        no workspace changes: runs in the worker thread). Returns
        (sample, population or None, result or None, error) per member."""
        members, path, mode = plan["members"], plan["path"], plan["mode"]
        out: list[tuple] = []
        for i, sample in enumerate(members):
            _check_cancel(should_cancel)
            if progress is not None:
                progress(i, f"Sample {i + 1} of {len(members)}: {sample.name}")
            if not sample.ensure_loaded():
                out.append((sample, None, None, sample.load_error or "not loaded"))
                continue
            target = sample.root.find(path)
            if target is None:
                out.append((sample, None, None, f"no population {path}"))
                continue
            try:
                result = _run_qc(target, plan["request"])
                if mode == "time" and not result.time_param:
                    raise ValueError("time gates need a time parameter")
                if mode == "time" and not result.intervals:
                    raise ValueError("no clean time interval")
            except (ValueError, KeyError, RuntimeError) as exc:
                out.append((sample, target, None, str(exc)))
                continue
            out.append((sample, target, result, ""))
        _check_cancel(should_cancel)
        if progress is not None:
            progress(len(members), "Creating populations…")
        return out

    def _apply_group(self, plan: dict, outcomes: list[tuple]) -> list[list]:
        """Create the QC populations from computed results (GUI thread, one undo step)."""
        name, mode, pop = plan["name"], plan["mode"], plan["current"]
        rows: list[list] = []
        failed = 0
        created_any = False
        blank = [""] * 8
        step = _undo_step(self.controller, "data cleaning (group)") if any(o[2] is not None for o in outcomes) else nullcontext()
        with step:
            for sample, target, result, error in outcomes:
                if result is None:
                    rows.append([sample.name, target.display_name if target is not None else "", *blank, error])
                    failed += 1
                    continue
                try:
                    created = qc.apply_qc(target, result, name, mode)
                except (ValueError, KeyError, RuntimeError) as exc:
                    rows.append([sample.name, target.display_name, *blank, str(exc)])
                    failed += 1
                    continue
                created_any = True
                if pop is not None and sample is pop.sample:
                    self._remember(result, target, plan["request"])
                    self.plot.set_result(result, plan["request"]["time_param"] or "Event order")
                rows.append(qc.group_row(sample.name, target.display_name, result, f"{created[0].name}: {created[0].count} events"))
        if created_any:
            self.controller.mark_modified()
            self._notify(None)
        self._set_table(qc.GROUP_HEADER, rows)
        msg = f"{name} created on {len(rows) - failed} of {len(plan['members'])} samples in {plan['group'].name}"
        self.status.setText(msg + (f" ({failed} failed)" if failed else ""))
        return rows

    def apply_to_group(self, name: str | None = None, mode: str | None = None) -> list[list]:
        """Run the checks on the chosen population path of every sample in the
        current group and create the QC population on each (one undo step).
        Synchronous; the button runs the same in a worker thread (start_group)."""
        plan = self._group_plan(name, mode)
        return self._apply_group(plan, self._compute_group(plan))

    def start_group(self, name: str | None = None, mode: str | None = None) -> bool:
        """apply_to_group in a worker thread with a progress dialog; the
        populations are created when it finishes (nothing on Cancel). Returns
        False when a job is already running."""
        if self.busy():
            return False
        plan = self._group_plan(name, mode)
        if not plan["members"]:
            self.apply_to_group(name, mode)
            return True
        return self._start_job(
            self.TITLE,
            len(plan["members"]),
            lambda progress, should_cancel: self._compute_group(plan, progress, should_cancel),
            lambda outcomes: self._apply_group(plan, outcomes),
            (self.run_button, self.create_button, self.group_button),
        )


# =============================================================================
# MESF calibration
# =============================================================================
class CalibrationDialog(_ToolDialog):
    """Find bead peaks, enter the kit's MESF values, fit and create
    "<param> MESF" parameters on samples."""

    TITLE = "Calibrate (MESF)"
    COL_MEDIAN = 2
    COL_MESF = 5
    COL_USED = 6

    def __init__(self, controller, parent=None):
        super().__init__(controller, parent)
        self.resize(1000, 700)
        self.peaks: list[cal.BeadPeak] = []
        self.fit_result: cal.CalibrationFit | None = None
        self._filling = False

        self.param_combo = QComboBox()
        self.peaks_spin = QSpinBox()
        self.peaks_spin.setRange(1, 20)
        self.peaks_spin.setValue(6)
        self.blank_check = QCheckBox("Exclude blank (lowest peak)")
        self.units_edit = QLineEdit(cal.UNITS)
        self.units_edit.setToolTip("Unit name of the calibrated parameter (MESF, ERF, ABC…)")
        self.r2_label = QLabel("r² –")
        self.fit_label = QLabel("")
        self.fit_label.setWordWrap(True)
        self.target_combo = QComboBox()

        form = QFormLayout()
        form.addRow("Beads sample:", self.sample_combo)
        form.addRow("Population:", self.pop_combo)
        form.addRow("Parameter:", self.param_combo)
        form.addRow("Number of peaks:", self.peaks_spin)
        form.addRow("", self.blank_check)
        form.addRow("Units:", self.units_edit)

        self.run_button = QPushButton(icon("histogram"), "Find Peaks")
        self.fit_button = QPushButton(icon("graph"), "Fit")
        self.apply_button = QPushButton(icon("derive"), "Apply to samples…")
        self.save_button = QPushButton(icon("save"), "Save calibration…")
        self.load_button = QPushButton(icon("open"), "Load calibration…")
        settings = QGroupBox("Calibration")
        sl = QVBoxLayout(settings)
        sl.addLayout(form)
        sl.addWidget(self.run_button)
        sl.addWidget(self.fit_button)
        sl.addWidget(self.r2_label)
        sl.addWidget(self.fit_label)
        tf = QFormLayout()
        tf.addRow("Apply to:", self.target_combo)
        sl.addLayout(tf)
        sl.addWidget(self.apply_button)
        hb = QHBoxLayout()
        hb.addWidget(self.save_button)
        hb.addWidget(self.load_button)
        sl.addLayout(hb)
        sl.addWidget(self.status)
        sl.addStretch(1)

        self.plot = CalibrationPlot()
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(settings)
        top.addWidget(self.plot)
        top.setStretchFactor(1, 1)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(self._table_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.itemChanged.connect(self._table_edited)
        self.run_button.clicked.connect(lambda: self.run())
        self.fit_button.clicked.connect(lambda: self.fit())
        self.apply_button.clicked.connect(lambda: self.start_apply())
        self.save_button.clicked.connect(lambda: self.save_calibration())
        self.load_button.clicked.connect(lambda: self.load_calibration())
        self._connect_controller()
        self.controller.groupsChanged.connect(self._fill_targets)
        self._fill_targets()
        self.refresh(use_selection=True)

    # ------------------------------------------------------------- settings
    def _refresh_params(self, sample):
        default = None
        if sample is not None:
            names = [n for n in sample.parameter_names() if not n.endswith(f" {self.units()}") and n not in sample.computed]
            default = next((n for n in names if not any(n.upper().startswith(p) for p in ("FSC", "SSC", "TIME", "HDR-T"))), names[0] if names else None)
            self._fill_params(self.param_combo, sample, default, names=names)
        else:
            self._fill_params(self.param_combo, None, None, names=[])

    def _fill_targets(self, *_):
        current = self.target_combo.currentData()
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("Current group", "current")
        for g in self.controller.workspace.groups:
            self.target_combo.addItem(g.name, g.name)
        idx = self.target_combo.findData(current) if current is not None else -1
        self.target_combo.setCurrentIndex(max(idx, 0))
        self.target_combo.blockSignals(False)

    def param(self) -> str | None:
        return self.param_combo.currentData()

    def units(self) -> str:
        return (self.units_edit.text() if hasattr(self, "units_edit") else cal.UNITS).strip() or cal.UNITS

    # ----------------------------------------------------------------- table
    def _fill_table(self, intensities, mesf, used=None, peaks=None):
        self._filling = True
        n = len(intensities)
        header = list(cal.PEAK_HEADER)
        header[self.COL_MESF] = self.units()
        self.table.clear()
        self.table.setColumnCount(len(header))
        self.table.setRowCount(n)
        self.table.setHorizontalHeaderLabels(header)
        for r in range(n):
            pk = peaks[r] if peaks is not None and r < len(peaks) else None
            cells = [str(r + 1), _fmt(pk.position) if pk else "", _fmt(float(intensities[r])), _fmt(pk.cv) if pk else "", str(pk.count) if pk else "",
                     "" if mesf is None or r >= len(mesf) or mesf[r] is None or not math.isfinite(float(mesf[r])) else f"{float(mesf[r]):g}"]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c not in (self.COL_MEDIAN, self.COL_MESF):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
            chk = QTableWidgetItem("")
            chk.setFlags((chk.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable)
            on = True if used is None or r >= len(used) else bool(used[r])
            chk.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.table.setItem(r, self.COL_USED, chk)
        self.table.resizeColumnsToContents()
        self._filling = False
        self._sync_rows()

    def _sync_rows(self):
        """Mirror the table into header/rows for Copy / Export CSV."""
        header = [self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        rows = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(self.table.columnCount()):
                it = self.table.item(r, c)
                if c == self.COL_USED:
                    row.append(bool(it is not None and it.checkState() == Qt.CheckState.Checked))
                    continue
                text = it.text() if it is not None else ""
                try:
                    row.append(float(text) if c != 0 else int(text))
                except ValueError:
                    row.append(text)
            rows.append(row)
        self.header, self.rows = header, rows

    def _table_edited(self, *_):
        if not self._filling:
            self._sync_rows()

    def _column(self, col: int) -> list[float]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, col)
            try:
                out.append(float(it.text().replace(",", "")) if it is not None and it.text().strip() else float("nan"))
            except ValueError:
                out.append(float("nan"))
        return out

    def intensities(self) -> list[float]:
        return self._column(self.COL_MEDIAN)

    def mesf_values(self) -> list[float]:
        return self._column(self.COL_MESF)

    def used(self) -> list[bool]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, self.COL_USED)
            out.append(it is None or it.checkState() == Qt.CheckState.Checked)
        return out

    def set_mesf_values(self, values):
        """Enter the kit's values (one per peak, ascending; None = unknown)."""
        self._filling = True
        for r, v in enumerate(values):
            if r >= self.table.rowCount():
                break
            it = self.table.item(r, self.COL_MESF)
            if it is None:
                it = QTableWidgetItem()
                self.table.setItem(r, self.COL_MESF, it)
            it.setText("" if v is None else f"{float(v):g}")
        self._filling = False
        self._sync_rows()

    def set_used(self, flags):
        self._filling = True
        for r, on in enumerate(flags):
            it = self.table.item(r, self.COL_USED)
            if it is not None:
                it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._filling = False
        self._sync_rows()

    # --------------------------------------------------------------- actions
    def run(self) -> list[cal.BeadPeak] | None:
        """Find the bead peaks; fills the table (intensity = peak medians)."""
        pop = self.current_population()
        param = self.param()
        if pop is None or not param:
            self.status.setText("Choose a beads sample, population and parameter.")
            return None
        from ..core.transforms import LinearScale

        sample = pop.sample
        try:
            peaks = cal.peaks_for_population(pop, param, self.peaks_spin.value())
            values = np.asarray(sample.data(param), dtype=float)[pop.mask()]
            scale = sample.scale(param)
            scale = None if isinstance(scale, LinearScale) else scale
            centers, counts, _sm, kind = cal.bead_histogram(values, True, scale)
        except (ValueError, KeyError, RuntimeError) as exc:
            self.peaks = []
            self.plot.clear()
            self.status.setText(f"Could not find peaks: {exc}")
            return None
        old_mesf = self.mesf_values() if self.table.rowCount() == len(peaks) else None
        stored = cal.stored_calibration(self.controller.workspace, param)
        if (old_mesf is None or not any(math.isfinite(v) for v in old_mesf)) and stored is not None and len(stored.mesf) == len(peaks):
            old_mesf = list(stored.mesf)
        self.peaks = peaks
        used = [not (self.blank_check.isChecked() and k == 0) for k in range(len(peaks))]
        self._fill_table([p.median for p in peaks], old_mesf, used, peaks)
        self.plot.set_histogram(centers, counts, peaks, f"{param} ({'display' if kind == 'display' else kind})")
        self.plot.set_fit(None, [p.median for p in peaks], old_mesf or [float("nan")] * len(peaks), used)
        self.fit_result = None
        self.r2_label.setText("r² –")
        found = f"{len(peaks)} of {self.peaks_spin.value()} peaks found" if len(peaks) < self.peaks_spin.value() else f"{len(peaks)} peaks found"
        self.status.setText(f"{sample.name}: {found}; enter the kit's {self.units()} values and press Fit.")
        return peaks

    def fit(self) -> cal.CalibrationFit | None:
        """Fit log10(MESF) against log10(peak median) over the used peaks."""
        x, y, used = self.intensities(), self.mesf_values(), self.used()
        if not x:
            self.status.setText("Find the bead peaks (or load a calibration) first.")
            return None
        sample = self.current_sample()
        try:
            fit = cal.fit_calibration(x, y, exclude_blank=self.blank_check.isChecked(), use=used, param=self.param() or "", units=self.units(), source=sample.name if sample else "")
        except ValueError as exc:
            self.status.setText(f"Could not fit: {exc}")
            return None
        self.fit_result = fit
        self.plot.set_fit(fit, x, y, fit.used)
        self.r2_label.setText(f"r² {fit.r2:.5f}")
        self.fit_label.setText(f"log10 {fit.units} = {fit.slope:.4f} · log10 I {'+' if fit.intercept >= 0 else '−'} {abs(fit.intercept):.4f}")
        self.status.setText(f"Fit on {sum(fit.used)} peaks, r² {fit.r2:.5f}.")
        return fit

    def target_samples(self, target=None) -> list:
        ws = self.controller.workspace
        if target is None:
            target = self.target_combo.currentData() or "current"
        if isinstance(target, (list, tuple)):
            return list(target)
        if target == "all":
            return list(ws.samples)
        if target == "current":
            return ws.members(self.controller.current_group)
        if not isinstance(target, str):  # a Group
            return ws.members(target)
        group = ws.group_by_name(target)
        return ws.members(group) if group is not None else []

    def _apply_plan(self, target) -> dict | None:
        fit = self.fit_result or self.fit()
        if fit is None:
            return None
        return {"fit": fit, "samples": list(self.target_samples(target))}

    @staticmethod
    def _prepare_samples(plan: dict, progress=None, should_cancel=None) -> list[tuple]:
        """Load every target sample and its source values (no widgets, no
        workspace changes: runs in the worker thread). Returns (sample, ok)."""
        fit, samples = plan["fit"], plan["samples"]
        out: list[tuple] = []
        for i, s in enumerate(samples):
            _check_cancel(should_cancel)
            if progress is not None:
                progress(i, f"Sample {i + 1} of {len(samples)}: {s.name}")
            ok = s.ensure_loaded() and fit.param in s.parameter_names()
            if ok:
                try:
                    s.data(fit.param)  # compensation of the source parameter, off the GUI thread
                except (KeyError, ValueError):
                    pass  # apply_calibration reports it
            out.append((s, ok))
        _check_cancel(should_cancel)
        if progress is not None:
            progress(len(samples), "Creating calibrated parameters…")
        return out

    def _apply_prepared(self, plan: dict, prepared: list[tuple]) -> list[tuple]:
        """Create the calibrated parameters (GUI thread, one undo step)."""
        fit = plan["fit"]
        done: list[tuple] = []
        skipped: list[str] = []
        ready = any(ok for _, ok in prepared)
        with (_undo_step(self.controller, f"calibrate {fit.param}", ("gating", "settings")) if ready else nullcontext()):
            for s, ok in prepared:
                if not ok:
                    skipped.append(s.name)
                    continue
                try:
                    name = cal.apply_calibration(s, fit)
                except (ValueError, KeyError) as exc:
                    skipped.append(f"{s.name} ({exc})")
                    continue
                done.append((s, name))
            if ready:
                cal.store_calibration(self.controller.workspace, fit)
        if ready:
            c = self.controller
            c.mark_modified()
            c.samplesChanged.emit()
            c.scalesChanged.emit(None)
            c.gatingChanged.emit(None)
            c.statusMessage.emit(f"Created {cal.calibrated_name(fit.param, fit.units)} on {len(done)} sample(s)")
        msg = f"Created {cal.calibrated_name(fit.param, fit.units)} on {len(done)} of {len(plan['samples'])} sample(s)"
        if skipped:
            msg += f"; skipped {', '.join(skipped[:5])}{'…' if len(skipped) > 5 else ''}"
        self.status.setText(msg)
        return done

    def apply(self, target=None) -> list[tuple]:
        """Create "<param> MESF" on the target samples ("current" group, "all",
        a group name / Group, or a list of samples). Returns (sample, name) pairs.
        Synchronous; the button runs the same in a worker thread (start_apply)."""
        plan = self._apply_plan(target)
        if plan is None:
            return []
        return self._apply_prepared(plan, self._prepare_samples(plan))

    def start_apply(self, target=None) -> bool:
        """apply() in a worker thread with a progress dialog (the files are
        loaded there); the parameters are created when it finishes (nothing on
        Cancel). Returns False when nothing was started."""
        if self.busy():
            return False
        plan = self._apply_plan(target)
        if plan is None:
            return False
        if not plan["samples"]:
            self._apply_prepared(plan, [])
            return True
        return self._start_job(
            self.TITLE,
            len(plan["samples"]),
            lambda progress, should_cancel: self._prepare_samples(plan, progress, should_cancel),
            lambda prepared: self._apply_prepared(plan, prepared),
            (self.run_button, self.fit_button, self.apply_button),
        )

    def save_calibration(self, path: str | None = None) -> str | None:
        fit = self.fit_result or self.fit()
        if fit is None:
            return None
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Calibration", f"{fit.param or 'calibration'}_{fit.units}.json", "Calibration (*.json)")
            if not path:
                return None
        cal.save_calibration(path, fit)
        self.status.setText(f"Saved calibration to {path}")
        return str(path)

    def load_calibration(self, path: str | None = None) -> cal.CalibrationFit | None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Load Calibration", "", "Calibration (*.json);;All files (*)")
            if not path:
                return None
        try:
            fit = cal.load_calibration(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText(f"Could not load calibration: {exc}")
            return None
        if fit.param:
            self.set_param(self.param_combo, fit.param)
        self.units_edit.setText(fit.units)
        self.peaks = []
        self._fill_table(fit.intensities, fit.mesf, fit.used or None)
        self.fit_result = fit
        self.plot.set_histogram(np.zeros(0), np.zeros(0), [], "")
        self.plot.set_fit(fit, fit.intensities, fit.mesf, fit.used or None)
        self.r2_label.setText(f"r² {fit.r2:.5f}")
        self.fit_label.setText(f"log10 {fit.units} = {fit.slope:.4f} · log10 I {'+' if fit.intercept >= 0 else '−'} {abs(fit.intercept):.4f}")
        self.status.setText(f"Loaded calibration for {fit.param or '?'} from {path}")
        return fit


# =============================================================================
# extension hook
# =============================================================================
class QCTools:
    """What `install` returns: the actions, the ribbon band and open dialogs."""

    DIALOGS = {"cleaning": DataCleaningDialog, "calibration": CalibrationDialog}

    def __init__(self, window):
        self.window = window
        self.controller = window.controller
        self.dialogs: dict[str, QDialog] = {}
        self.actions: dict[str, QAction] = {}
        specs = [
            ("cleaning", "Data Cleaning…", "check", "Flag unstable flow rate, signal shifts and saturated events (FlowAI / PeacoQC-style) and gate the clean events"),
            ("calibration", "Calibrate (MESF)…", "wizard", "Fit bead peaks to MESF values and add calibrated parameters"),
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
            self.band = page.add_band("Quality Control")
            for act in self.actions.values():
                self.band.add_large(act)
        menu = getattr(window, "menus", {}).get("tools")
        self.menu = menu
        if menu is not None:
            menu.addSeparator()
            for act in self.actions.values():
                menu.addAction(act)

    def open(self, key: str) -> QDialog:
        dlg = self.dialogs.get(key)
        if dlg is None:
            dlg = self.DIALOGS[key](self.controller, self.window)
            self.dialogs[key] = dlg
        else:
            dlg.refresh(use_selection=True)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg


def install(window) -> QCTools:
    return QCTools(window)
