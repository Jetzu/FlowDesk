"""Plot rendering: turns a population into a QImage.

Used by the graph window, the layout editor and the tree thumbnails so that
every plot in the program looks the same.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF
from scipy.ndimage import gaussian_filter, gaussian_filter1d

try:
    import contourpy
except ImportError:  # pragma: no cover
    contourpy = None

from ..core.model import GraphSettings, Population, Sample
from ..core.transforms import Scale

# ------------------------------------------------------------------ colour maps
PSEUDOCOLOR_STOPS = [
    (0.00, (18, 18, 150)),
    (0.12, (0, 48, 255)),
    (0.30, (0, 170, 255)),
    (0.45, (0, 220, 180)),
    (0.58, (40, 230, 40)),
    (0.72, (210, 235, 0)),
    (0.85, (255, 170, 0)),
    (1.00, (225, 10, 10)),
]
GRAY_STOPS = [(0.0, (200, 200, 200)), (0.6, (110, 110, 110)), (1.0, (10, 10, 10))]
OVERLAY_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def build_lut(stops) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], dtype=float)
    t = np.linspace(0, 1, 256)
    for c in range(3):
        lut[:, c] = np.clip(np.interp(t, xs, cols[:, c]), 0, 255).astype(np.uint8)
    return lut


PSEUDO_LUT = build_lut(PSEUDOCOLOR_STOPS)
GRAY_LUT = build_lut(GRAY_STOPS)


# --------------------------------------------------------------------- options
@dataclass
class RenderOptions:
    plot_type: str = "pseudocolor"
    dots_to_draw: int = 50000
    dot_size: int = 1
    show_outliers: bool = True
    smoothing: int = 1
    contour_level: float = 5.0
    gate_color: str = "#000000"
    gate_line_width: float = 1.0
    selected_gate_color: str = "#d81b1b"
    label_font_size: int = 11
    number_font_size: int = 9
    show_gate_names: bool = True
    show_gate_freqs: bool = True
    axis_label_style: str = "name_stain"
    histogram_bins: int = 256
    histogram_ymode: str = "count"
    colormap: str = "pseudocolor"
    background: str = "#ffffff"
    show_gates: bool = True
    show_axes: bool = True
    freq_decimals: int = 1
    title: str | None = None
    overlay_colors: list[str] = field(default_factory=lambda: list(OVERLAY_COLORS))
    histogram_fill: str = "#d7e3f4"
    histogram_line: str = "#000000"
    view: tuple = (0.0, 1.0, 0.0, 1.0)  # zoom window in display fractions (x0, x1, y0, y1)
    show_gate_labels: bool = True
    # Font sizes are given in points. They are converted to *pixel* sizes as if
    # the target had `font_dpi` dots per inch, so a plot looks the same on every
    # platform and in vector exports (a QImage reports 72 dpi on macOS but 96 on
    # Windows / Linux, which made point-sized text a third larger there).
    # 96 matches what the layout editor assumes; graph windows use 72 (1 pt = 1 px).
    font_dpi: float = 96.0
    show_legend: bool = True  # legend box when overlays are drawn
    # legend layout, kept in the graph's options so it survives and reaches layouts: top-left corner in plot fractions
    # (None = top right), font size in points (0 = the number font), a title, labels renamed by the user (original
    # label -> text) and whether labels carry the sample name
    legend_pos: tuple | None = None
    legend_font_size: float = 0.0
    legend_title: str = ""
    legend_labels: dict = field(default_factory=dict)
    legend_show_samples: bool = True
    histogram_offset: bool = False  # stack overlaid histograms with a vertical offset
    histogram_line_width: float = 1.2

    @classmethod
    def from_prefs(cls, prefs, graph: GraphSettings | None = None, **overrides) -> "RenderOptions":
        values = {}
        for f in cls.__dataclass_fields__:
            if f in ("title", "overlay_colors", "view", "font_dpi") or f.startswith("legend_"):
                continue
            try:
                v = prefs.get(f) if hasattr(prefs, "get") else None
            except Exception:
                v = None
            if v is not None:
                values[f] = v
        if graph is not None:
            values["plot_type"] = graph.plot_type or values.get("plot_type", "pseudocolor")
            for k, v in graph.options.items():
                if k in cls.__dataclass_fields__:
                    values[k] = v
        values.update(overrides)
        opts = cls()
        for k, v in values.items():
            current = getattr(opts, k)
            if isinstance(current, (list, tuple, type(None))) or v is None:
                setattr(opts, k, v)
            else:
                try:
                    setattr(opts, k, type(current)(v))
                except (TypeError, ValueError):
                    setattr(opts, k, v)
        return opts


# ---------------------------------------------------------------------- results
@dataclass
class GateShape:
    population: Population
    kind: str  # rect | polygon | ellipse | range | quadrant
    points: list[tuple[float, float]]  # image pixel coordinates
    bbox: QRectF
    label_rect: QRectF
    label: str
    transposed: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class RenderResult:
    image: QImage
    plot_rect: QRectF
    x: str | None
    y: str | None
    x_scale: Scale | None
    y_scale: Scale | None
    gates: list[GateShape]
    n_events: int
    histogram: bool = False
    error: str | None = None
    view: tuple = (0.0, 1.0, 0.0, 1.0)
    legend_rect: QRectF | None = None  # overlay legend box (image pixels), when drawn
    legend_entries: list = field(default_factory=list)  # [(label, colour name)]
    legend_keys: list = field(default_factory=list)  # original label of each entry (key of RenderOptions.legend_labels)
    legend_rows: list = field(default_factory=list)  # QRectF of each entry's row (image pixels)

    # ---- coordinate helpers (image pixels <-> display fractions)
    def to_pixel(self, fx: float, fy: float) -> QPointF:
        r = self.plot_rect
        x0, x1, y0, y1 = self.view
        vx = (fx - x0) / (x1 - x0)
        vy = (fy - y0) / (y1 - y0)
        return QPointF(r.left() + vx * r.width(), r.bottom() - vy * r.height())

    def to_fraction(self, px: float, py: float) -> tuple[float, float]:
        r = self.plot_rect
        if r.width() <= 0 or r.height() <= 0:
            return 0.0, 0.0
        x0, x1, y0, y1 = self.view
        vx = (px - r.left()) / r.width()
        vy = (r.bottom() - py) / r.height()
        return x0 + vx * (x1 - x0), y0 + vy * (y1 - y0)


# --------------------------------------------------------------------- helpers
def axis_title(sample: Sample, name: str, style: str) -> str:
    stain = sample.param_stain(name)
    if style == "name":
        return name
    if style == "stain":
        return stain or name
    return f"{name} :: {stain or name}"


def has_parameter(sample: Sample, name: str) -> bool:
    """Whether `sample` can provide values for `name` (FCS channel or stain,
    compensated, derived or computed parameter). Computed parameters exist only
    on the samples a tool ran on, so a plot or an overlay may ask a sample for
    a parameter it does not have."""
    if not name:
        return False
    if name in sample.computed or name in sample.derived or name in sample.channel_names:
        return True
    if name in sample.parameter_names():
        return True
    return any(c.stain == name for c in sample.channels)


def format_freq(value: float, decimals: int = 1) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{decimals}f}"


def plot_font(size_pt: float, options: "RenderOptions | None" = None) -> QFont:
    """A QFont of `size_pt` points expressed as a pixel size (see RenderOptions.font_dpi)."""
    dpi = float(getattr(options, "font_dpi", 96.0) or 96.0)
    font = QFont()
    font.setPixelSize(max(1, int(round(float(size_pt) * dpi / 72.0))))
    return font


def _subsample(idx: np.ndarray, limit: int) -> np.ndarray:
    if limit and idx.size > limit:
        rng = np.random.default_rng(1234)
        return np.sort(rng.choice(idx, size=limit, replace=False))
    return idx


def density_grid(xd: np.ndarray, yd: np.ndarray, res: int, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (smoothed grid [res x res, row = y], raw counts, bx, by)."""
    bx = np.clip((xd * (res - 1) + 0.5).astype(np.int32), 0, res - 1)
    by = np.clip((yd * (res - 1) + 0.5).astype(np.int32), 0, res - 1)
    counts = np.bincount(by * res + bx, minlength=res * res).reshape(res, res).astype(np.float64)
    grid = gaussian_filter(counts, sigma=sigma) if sigma > 0 else counts
    return grid, counts, bx, by


def probability_levels(grid: np.ndarray, counts: np.ndarray, step_percent: float, max_levels: int = 24) -> list[float]:
    """Density thresholds so that successive contours enclose step% fewer events."""
    total = counts.sum()
    if total <= 0:
        return []
    order = np.argsort(grid.ravel())[::-1]
    cum = np.cumsum(counts.ravel()[order]) / total
    dens_sorted = grid.ravel()[order]
    step = max(step_percent, 0.5) / 100.0
    levels = []
    q = 1.0 - step
    while q > step / 2 and len(levels) < max_levels:
        k = int(np.searchsorted(cum, q))
        k = min(k, dens_sorted.size - 1)
        t = float(dens_sorted[k])
        if t > 0 and (not levels or t > levels[-1] * 1.0001):
            levels.append(t)
        q -= step
    return levels


# ---------------------------------------------------------------------- render
def render_plot(
    sample: Sample,
    population: Population,
    graph: GraphSettings,
    width: int,
    height: int,
    options: RenderOptions,
    overlays: list[tuple[Sample, Population]] | None = None,
    selected: Population | None = None,
    dpr: float = 1.0,
    painter: QPainter | None = None,
) -> RenderResult:
    """Render a plot into a new QImage (default) or onto `painter` (any paint
    device: SVG generator, PDF writer, printer), in a `width` x `height`
    logical-pixel frame. `dpr` sets the resolution of the rasterised events."""
    width = max(int(width), 60)
    height = max(int(height), 60)
    dpr = max(float(dpr), 1.0)
    own = painter is None
    if own:
        image = QImage(int(width * dpr), int(height * dpr), QImage.Format.Format_RGB32)
        image.setDevicePixelRatio(dpr)
        image.fill(QColor(options.background))
        painter = QPainter(image)
    else:
        image = QImage()
        painter.save()
        painter.fillRect(QRectF(0, 0, width, height), QColor(options.background))
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    try:
        return _paint_plot(painter, image, sample, population, graph, width, height, options, overlays, selected, dpr)
    finally:
        if own:
            painter.end()
        else:
            painter.restore()


def _paint_plot(painter: QPainter, image: QImage, sample, population, graph, width: int, height: int, options: RenderOptions, overlays, selected, dpr: float) -> RenderResult:

    x = graph.x
    y = graph.y
    histogram = graph.plot_type == "histogram" or y is None
    if histogram:
        y = None

    label_font = plot_font(options.label_font_size, options)
    number_font = plot_font(options.number_font_size, options)
    fm_num = QFontMetricsF(number_font)
    fm_lab = QFontMetricsF(label_font)

    # ---- plot rectangle
    top = 8.0 + (fm_lab.height() + 4 if options.title else 0)
    bottom = fm_num.height() + fm_lab.height() + 10 if options.show_axes else 6
    left = (fm_num.horizontalAdvance("250K") + fm_lab.height() + 14) if options.show_axes else 6
    right = 12.0
    plot_rect = QRectF(left, top, width - left - right, height - top - bottom)
    if plot_rect.width() < 20 or plot_rect.height() < 20:
        plot_rect = QRectF(2, 2, width - 4, height - 4)

    view = tuple(options.view) if options.view else (0.0, 1.0, 0.0, 1.0)
    if view[1] - view[0] < 1e-6 or view[3] - view[2] < 1e-6:
        view = (0.0, 1.0, 0.0, 1.0)
    result = RenderResult(image, plot_rect, x, y, None, None, [], 0, histogram, view=view)
    zoomed = view != (0.0, 1.0, 0.0, 1.0)

    def apply_view_x(arr):
        return (arr - view[0]) / (view[1] - view[0])

    def apply_view_y(arr):
        return (arr - view[2]) / (view[3] - view[2])

    if not sample.ensure_loaded():
        _draw_message(painter, plot_rect, f"File not found\n{sample.name}", label_font)
        result.error = sample.load_error or "file not found"
        return result
    if not x:
        x, y2 = sample.default_axes()
        if not histogram:
            y = y2
        result.x, result.y = x, y
    absent = [n for n in (x, y) if n and not has_parameter(sample, n)]
    if absent:
        # e.g. a tSNE plot of a group batch over a sample tSNE was not run on;
        # checked before scale() so no default scale is cached for it
        names = " / ".join(dict.fromkeys(absent))
        _draw_message(painter, plot_rect, f"{names} not available\nfor {sample.name}", label_font)
        result.error = f"parameter not found: {names}"
        return result
    try:
        x_scale = sample.scale(x)
        xd_all = sample.display(x)
        y_scale = sample.scale(y) if y else None
        yd_all = sample.display(y) if y else None
    except KeyError as exc:
        _draw_message(painter, plot_rect, f"Parameter not found: {exc}", label_font)
        result.error = str(exc)
        return result
    result.x_scale, result.y_scale = x_scale, y_scale

    mask = population.mask()
    idx = np.flatnonzero(mask)
    result.n_events = int(idx.size)
    # computed parameters may be NaN for events the tool did not process
    finite = np.isfinite(xd_all[idx])
    if yd_all is not None:
        finite &= np.isfinite(yd_all[idx])
    if not finite.all():
        idx = idx[finite]

    pw, ph = int(plot_rect.width()), int(plot_rect.height())
    painter.setClipRect(plot_rect)
    if histogram:
        hx = xd_all[idx]
        if zoomed:
            hx = apply_view_x(hx)
            hx = hx[(hx >= 0) & (hx <= 1)]
        _draw_histogram(painter, plot_rect, hx, options, overlays, x, result)
    else:
        draw_idx = _subsample(idx, int(options.dots_to_draw))
        xd = xd_all[draw_idx]
        yd = yd_all[draw_idx]
        if zoomed:
            xd = apply_view_x(xd)
            yd = apply_view_y(yd)
            keep = (xd >= 0) & (xd <= 1) & (yd >= 0) & (yd <= 1)
            xd, yd = xd[keep], yd[keep]
        ptype = options.plot_type
        bw, bh = int(pw * dpr), int(ph * dpr)
        if ptype in ("pseudocolor", "dot", "density"):
            buf = _render_dots(xd, yd, bw, bh, options, ptype, dpr)
            _blit(painter, plot_rect, buf, dpr)
        elif ptype in ("contour", "zebra"):
            _render_contours(painter, plot_rect, xd, yd, options, ptype == "zebra")
        else:
            buf = _render_dots(xd, yd, bw, bh, options, "pseudocolor", dpr)
            _blit(painter, plot_rect, buf, dpr)
        if overlays:
            for k, (osample, opop) in enumerate(overlays):
                if not osample.ensure_loaded() or not (has_parameter(osample, x) and has_parameter(osample, y)):
                    continue
                try:
                    oidx = _subsample(np.flatnonzero(opop.mask()), int(options.dots_to_draw))
                    ox = osample.display(x)[oidx]
                    oy = osample.display(y)[oidx]
                except KeyError:
                    continue
                ok = np.isfinite(ox) & np.isfinite(oy)  # computed parameters: NaN = not computed
                if not ok.all():
                    ox, oy = ox[ok], oy[ok]
                if zoomed:
                    ox, oy = apply_view_x(ox), apply_view_y(oy)
                color = QColor(options.overlay_colors[k % len(options.overlay_colors)])
                painter.setPen(QPen(color, max(1.5, options.dot_size)))
                pts = [QPointF(plot_rect.left() + fx * pw, plot_rect.bottom() - fy * ph) for fx, fy in zip(ox.tolist(), oy.tolist())]
                painter.drawPoints(QPolygonF(pts))
    painter.setClipping(False)

    # ---- frame and axes
    painter.setPen(QPen(QColor("#000000"), 1.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(plot_rect)
    if options.show_axes:
        _draw_axes(painter, plot_rect, x_scale, y_scale, sample, x, y, options, number_font, label_font, result)
    if options.title:
        painter.setFont(label_font)
        painter.setPen(QColor("#000000"))
        painter.drawText(QRectF(0, 2, width, fm_lab.height() + 2), Qt.AlignmentFlag.AlignCenter, options.title)

    # ---- gates
    if options.show_gates:
        draw_gates(painter, result, population, sample, options, selected)
    # ---- overlay legend
    if overlays and options.show_legend:
        series = [(sample, population, options.histogram_line if histogram else "#7a7f87")]
        series += [(osample, opop, options.overlay_colors[k % len(options.overlay_colors)]) for k, (osample, opop) in enumerate(overlays)]
        keys, entries = [], []
        custom = options.legend_labels or {}
        for ssample, spop, colour in series:
            key = f"{ssample.name} / {spop.display_name}"
            label = key if options.legend_show_samples else spop.display_name
            absent = [n for n in (x, y) if n and not has_parameter(ssample, n)]
            if absent:
                label += f" (no {' / '.join(absent)})"
            keys.append(key)
            entries.append((str(custom.get(key, label)), colour))
        result.legend_entries = entries
        result.legend_keys = keys
        font = plot_font(options.legend_font_size, options) if options.legend_font_size and options.legend_font_size > 0 else number_font
        result.legend_rect, result.legend_rows = _draw_legend(painter, plot_rect, entries, font, options.legend_pos, options.legend_title)
    return result


def _draw_legend(painter: QPainter, plot: QRectF, entries: list[tuple[str, str]], font: QFont, pos: tuple | None = None,
                 title: str = "") -> tuple[QRectF, list[QRectF]]:
    """Colour swatch + label per series in a box: in the top-right plot corner, or with its top-left corner at `pos`
    (fractions of the plot rectangle), always kept inside the plot. Returns the box and each entry's row."""
    fm = QFontMetricsF(font)
    sw = max(8.0, fm.height() * 0.7)
    pad = 4.0
    avail = max(40.0, plot.width() * 0.62 - sw - 3 * pad)
    labels = [fm.elidedText(t, Qt.TextElideMode.ElideMiddle, avail) for t, _ in entries]
    row_h = fm.height()
    title_font = QFont(font)
    title_font.setBold(True)
    tfm = QFontMetricsF(title_font)
    title = fm.elidedText(title, Qt.TextElideMode.ElideRight, avail + sw + pad) if title else ""
    text_w = max([fm.horizontalAdvance(t) for t in labels] + [tfm.horizontalAdvance(title) - sw - pad if title else 0.0])
    title_h = tfm.height() if title else 0.0
    w = pad * 3 + sw + text_w
    h = pad * 2 + title_h + row_h * len(entries)
    if pos is not None and len(pos) == 2:
        left = plot.left() + float(pos[0]) * plot.width()
        top = plot.top() + float(pos[1]) * plot.height()
    else:
        left, top = plot.right() - w - 4, plot.top() + 4
    left = min(max(left, plot.left() + 1), plot.right() - w - 1)
    top = min(max(top, plot.top() + 1), plot.bottom() - h - 1)
    box = QRectF(left, top, w, h)
    painter.save()
    painter.setPen(QPen(QColor("#9aa0a6"), 0.8))
    painter.setBrush(QColor(255, 255, 255, 225))
    painter.drawRoundedRect(box, 2.5, 2.5)
    if title:
        painter.setFont(title_font)
        painter.setPen(QColor("#000000"))
        painter.drawText(QRectF(box.left() + pad, box.top() + pad, w - 2 * pad, title_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
    painter.setFont(font)
    rows = []
    for i, (label, (_, colour)) in enumerate(zip(labels, entries)):
        y = box.top() + pad + title_h + i * row_h
        painter.setPen(QPen(QColor(colour).darker(115), 0.8))
        painter.setBrush(QColor(colour))
        painter.drawRect(QRectF(box.left() + pad, y + (row_h - sw) / 2, sw, sw))
        painter.setPen(QColor("#000000"))
        painter.drawText(QRectF(box.left() + 2 * pad + sw, y, text_w + 2, row_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        rows.append(QRectF(box.left(), y, w, row_h))
    painter.restore()
    return box, rows


def _draw_message(painter: QPainter, rect: QRectF, text: str, font: QFont):
    painter.setPen(QPen(QColor("#000000"), 1.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(rect)
    painter.setFont(font)
    painter.setPen(QColor("#888888"))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)


def _blit(painter: QPainter, rect: QRectF, buf: np.ndarray, dpr: float = 1.0):
    h, w, _ = buf.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = buf[..., 2]
    rgba[..., 1] = buf[..., 1]
    rgba[..., 2] = buf[..., 0]
    rgba[..., 3] = 255
    img = QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGB32)
    img.setDevicePixelRatio(dpr)
    painter.drawImage(QPointF(rect.left(), rect.top()), img)


def _render_dots(xd: np.ndarray, yd: np.ndarray, pw: int, ph: int, options: RenderOptions, ptype: str, dpr: float = 1.0) -> np.ndarray:
    bg = QColor(options.background)
    buf = np.empty((ph, pw, 3), dtype=np.uint8)
    buf[..., 0] = bg.red()
    buf[..., 1] = bg.green()
    buf[..., 2] = bg.blue()
    n = xd.size
    if n == 0:
        return buf
    px = np.clip((xd * (pw - 1) + 0.5).astype(np.int32), 0, pw - 1)
    py = np.clip(((1.0 - yd) * (ph - 1) + 0.5).astype(np.int32), 0, ph - 1)
    if ptype == "dot":
        colors = np.zeros((n, 3), dtype=np.uint8)
        order = np.arange(n)
    else:
        res = int(min(256, max(48, min(pw, ph) // 2)))
        sigma = options.smoothing * res / 128.0
        grid, _counts, bx, by = density_grid(xd, yd, res, sigma)
        dens = grid[by, bx]
        order = np.argsort(dens, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n) / max(n - 1, 1)
        cidx = (ranks * 255).astype(np.uint8)
        lut = GRAY_LUT if (ptype == "density" or options.colormap == "gray") else PSEUDO_LUT
        colors = lut[cidx]
    px, py, colors = px[order], py[order], colors[order]
    size = max(1, int(round(int(options.dot_size) * dpr)))
    offsets = [(0, 0)] if size == 1 else [(dx, dy) for dx in range(-(size // 2), size - size // 2) for dy in range(-(size // 2), size - size // 2)]
    for dx, dy in offsets:
        qx = np.clip(px + dx, 0, pw - 1)
        qy = np.clip(py + dy, 0, ph - 1)
        buf[qy, qx] = colors
    return buf


def _render_contours(painter: QPainter, rect: QRectF, xd: np.ndarray, yd: np.ndarray, options: RenderOptions, zebra: bool):
    n = xd.size
    if n == 0:
        return
    res = 128
    sigma = 1.6 + options.smoothing * 0.8
    grid, counts, bx, by = density_grid(xd, yd, res, sigma)
    levels = probability_levels(grid, counts, options.contour_level)
    pw, ph = rect.width(), rect.height()

    def gx(i):
        return rect.left() + (i / (res - 1)) * pw

    def gy(j):
        return rect.bottom() - (j / (res - 1)) * ph

    if zebra and levels:
        # fill each bin by its level index
        level_idx = np.searchsorted(np.array(levels), grid, side="right")  # 0 = below the lowest contour
        cidx = (level_idx / max(len(levels), 1) * 255).astype(np.uint8)
        lut = PSEUDO_LUT
        rgb = lut[cidx]
        rgb[level_idx == 0] = (QColor(options.background).red(), QColor(options.background).green(), QColor(options.background).blue())
        rgb = rgb[::-1]  # row 0 = top
        rgba = np.empty((res, res, 4), dtype=np.uint8)
        rgba[..., 0] = rgb[..., 2]
        rgba[..., 1] = rgb[..., 1]
        rgba[..., 2] = rgb[..., 0]
        rgba[..., 3] = 255
        img = QImage(rgba.data, res, res, res * 4, QImage.Format.Format_RGB32)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(rect, img)
        painter.restore()

    # outliers
    if options.show_outliers and levels:
        dens = grid[by, bx]
        out = dens < levels[0]
        if out.any():
            painter.setPen(QPen(QColor("#000000"), 1.0))
            pts = [QPointF(rect.left() + fx * pw, rect.bottom() - fy * ph) for fx, fy in zip(xd[out].tolist(), yd[out].tolist())]
            painter.drawPoints(QPolygonF(pts))

    pen = QPen(QColor("#1a3f8f" if not zebra else "#000000"), 1.0)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if contourpy is not None:
        gen = contourpy.contour_generator(z=grid, line_type=contourpy.LineType.Separate)
        for lv in levels:
            for line in gen.lines(lv):
                if len(line) < 2:
                    continue
                poly = QPolygonF([QPointF(gx(px_), gy(py_)) for px_, py_ in line])
                painter.drawPolyline(poly)
    else:  # crude fallback: draw bins above each level as outlines
        for lv in levels:
            mask = grid >= lv
            ys, xs = np.nonzero(mask)
            pts = [QPointF(gx(i), gy(j)) for j, i in zip(ys.tolist(), xs.tolist())]
            painter.drawPoints(QPolygonF(pts))


def _draw_histogram(painter: QPainter, rect: QRectF, xd: np.ndarray, options: RenderOptions, overlays, x: str, result: RenderResult):
    bins = max(32, int(options.histogram_bins))
    series: list[tuple[np.ndarray, QColor, bool]] = []
    counts, _ = np.histogram(np.clip(xd, 0, 1), bins=bins, range=(0.0, 1.0))
    counts = counts.astype(float)
    if options.smoothing > 0:
        counts = gaussian_filter1d(counts, sigma=options.smoothing)
    series.append((counts, QColor(options.histogram_line), True))
    if overlays:
        for k, (osample, opop) in enumerate(overlays):
            if not osample.ensure_loaded() or not has_parameter(osample, x):
                continue
            try:
                od = osample.display(x)[opop.mask()]
            except KeyError:
                continue
            od = od[np.isfinite(od)]
            if result.view != (0.0, 1.0, 0.0, 1.0):
                od = (od - result.view[0]) / (result.view[1] - result.view[0])
            oc = np.histogram(np.clip(od, 0, 1), bins=bins, range=(0.0, 1.0))[0].astype(float)
            if options.smoothing > 0:
                oc = gaussian_filter1d(oc, sigma=options.smoothing)
            series.append((oc, QColor(options.overlay_colors[k % len(options.overlay_colors)]), False))
    normalize = options.histogram_ymode == "mode" or (overlays and len(series) > 1)
    ymax = 1.0
    if normalize:
        series = [(c / c.max() * 100.0 if c.max() > 0 else c, col, fill) for c, col, fill in series]
        ymax = 100.0
    else:
        ymax = max((c.max() for c, _, _ in series), default=1.0)
        ymax = ymax * 1.05 if ymax > 0 else 1.0
    result.extra_ymax = ymax  # type: ignore[attr-defined]
    result.extra_normalized = normalize  # type: ignore[attr-defined]
    pw, ph = rect.width(), rect.height()
    n = len(series)
    offset = 0.0
    if options.histogram_offset and n > 1:
        offset = min(0.22, 0.6 / (n - 1))  # fraction of the plot height between baselines
    height_frac = 1.0 - offset * (n - 1)
    result.extra_offset = offset  # type: ignore[attr-defined]
    result.extra_height_frac = height_frac  # type: ignore[attr-defined]
    lw = max(0.3, float(options.histogram_line_width))
    # back to front so the lower (earlier) curves stay visible when staggered
    order = list(range(n))[::-1] if offset else list(range(n))
    for k in order:
        c, col, fill = series[k]
        base = rect.bottom() - k * offset * ph
        path = QPainterPath()
        path.moveTo(rect.left(), base)
        for i, v in enumerate(c):
            path.lineTo(rect.left() + (i + 0.5) / bins * pw, base - (v / ymax) * height_frac * ph)
        path.lineTo(rect.right(), base)
        if fill or offset:
            painter.setPen(Qt.PenStyle.NoPen)
            fc = QColor(options.histogram_fill) if fill else QColor(col)
            if not fill:
                fc.setAlpha(40)
            painter.setBrush(QBrush(fc))
            painter.drawPath(path)
        painter.setPen(QPen(col, lw))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)


def _draw_axes(painter, rect: QRectF, x_scale, y_scale, sample, x, y, options, number_font, label_font, result: RenderResult):
    painter.setFont(number_font)
    fm = QFontMetricsF(number_font)
    black = QPen(QColor("#000000"), 1.0)
    painter.setPen(black)
    major, minor = 6.0, 3.0
    # x axis
    last_right = -1e9
    for t in x_scale.ticks():
        px = result.to_pixel(t.pos, 0.0).x()
        if px < rect.left() - 0.5 or px > rect.right() + 0.5:
            continue
        ln = major if t.major else minor
        painter.drawLine(QPointF(px, rect.bottom()), QPointF(px, rect.bottom() + ln))
        if t.major and t.label:
            w = fm.horizontalAdvance(t.label)
            if px - w / 2 > last_right + 3:
                painter.drawText(QRectF(px - w / 2 - 2, rect.bottom() + major + 1, w + 4, fm.height()), Qt.AlignmentFlag.AlignCenter, t.label)
                last_right = px + w / 2
    painter.setFont(label_font)
    fml = QFontMetricsF(label_font)
    xtitle = fml.elidedText(axis_title(sample, x, options.axis_label_style), Qt.TextElideMode.ElideMiddle, rect.width() + 20)
    painter.drawText(QRectF(rect.left() - 10, rect.bottom() + major + fm.height() + 1, rect.width() + 20, fml.height() + 2), Qt.AlignmentFlag.AlignCenter, xtitle)
    # y axis
    painter.setFont(number_font)
    if y_scale is not None:
        ticks = y_scale.ticks()
        ytitle = axis_title(sample, y, options.axis_label_style)
    else:
        ymax = getattr(result, "extra_ymax", 1.0)
        normalized = getattr(result, "extra_normalized", False)
        from ..core.transforms import LinearScale

        lin = LinearScale(0.0, ymax if ymax > 0 else 1.0)
        ticks = [t for t in lin.ticks() if t.pos <= 1.0]
        ytitle = "% of Max" if normalized else "Count"
    hist_frac = float(getattr(result, "extra_height_frac", 1.0))
    last_top = 1e9
    for t in ticks:
        py = result.to_pixel(0.0, t.pos).y() if y_scale is not None else rect.bottom() - t.pos * hist_frac * rect.height()
        if py < rect.top() - 0.5 or py > rect.bottom() + 0.5:
            continue
        ln = major if t.major else minor
        painter.drawLine(QPointF(rect.left() - ln, py), QPointF(rect.left(), py))
        if t.major and t.label:
            h = fm.height()
            if py + h / 2 < last_top - 2:
                painter.drawText(QRectF(0, py - h / 2, rect.left() - major - 2, h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, t.label)
                last_top = py - h / 2
    painter.save()
    painter.setFont(label_font)
    painter.translate(fml.height() * 0.9, rect.center().y())
    painter.rotate(-90)
    ytitle = fml.elidedText(ytitle, Qt.TextElideMode.ElideMiddle, rect.height() + 20)
    painter.drawText(QRectF(-rect.height() / 2 - 10, -fml.height() / 2, rect.height() + 20, fml.height()), Qt.AlignmentFlag.AlignCenter, ytitle)
    painter.restore()


# ----------------------------------------------------------------------- gates
def _gate_label(pop: Population, options: RenderOptions) -> str:
    parts = []
    if options.show_gate_names:
        parts.append(pop.name)
    if options.show_gate_freqs:
        parts.append(format_freq(pop.freq_of_parent, options.freq_decimals))
    return "\n".join(parts)


def draw_gates(painter: QPainter, result: RenderResult, population: Population, sample: Sample, options: RenderOptions, selected: Population | None = None):
    """Draw the gates of `population`'s children onto a rendered plot and fill
    result.gates with their shapes (used for hit testing)."""
    x, y = result.x, result.y
    if x is None or result.x_scale is None:
        return
    rect = result.plot_rect
    label_font = plot_font(options.label_font_size, options)
    painter.setFont(label_font)
    fm = QFontMetricsF(label_font)
    result.gates = []
    placed_labels: list[QRectF] = []
    quadrant_groups: dict[str, list[tuple[Population, GateShape]]] = {}
    for child in population.children:
        gate = child.gate
        if gate is None or gate.kind == "boolean":
            continue
        dims = gate.dims
        transposed = False
        if len(dims) == 2:
            if y is None:
                continue
            if dims[0] == x and dims[1] == y:
                pass
            elif dims[0] == y and dims[1] == x:
                transposed = True
            else:
                continue
        elif len(dims) == 1:
            if dims[0] != x and dims[0] != y:
                continue
        else:
            continue
        color = QColor(options.selected_gate_color if child is selected else (gate.color or options.gate_color))
        pen = QPen(color, max(0.8, gate.line_width * options.gate_line_width))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        shape = _gate_shape(result, sample, child, transposed)
        if shape is None:
            continue
        if shape.kind == "quadrant":
            quadrant_groups.setdefault(gate.quadrant[0], []).append((child, shape))
            continue
        # A gate may reach beyond the axis (open-ended ranges, vertices outside the plotted range). It keeps that extent in
        # data space; only its drawing stops at the axes.
        painter.save()
        painter.setClipRect(rect)
        tint = float(getattr(gate, "tint", 0.0) or 0.0)
        if tint > 0:
            _fill_shape(painter, shape, QColor(gate.color or options.gate_color), tint)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        _draw_shape(painter, shape)
        painter.restore()
        if getattr(gate, "locked", False) and rect.contains(QPointF(shape.bbox.right(), shape.bbox.top())):
            _draw_padlock(painter, QPointF(shape.bbox.right(), shape.bbox.top()), color)
            painter.setPen(pen)
        label = _gate_label(child, options) if options.show_gate_labels else ""
        shape.label = label
        shape.label_rect = _place_label(painter, fm, shape, label, rect, gate, result, placed_labels)
        if not shape.label_rect.isNull():
            placed_labels.append(shape.label_rect)
        result.gates.append(shape)
    for key, items in quadrant_groups.items():
        _draw_quadrants(painter, fm, result, items, options, selected)


def _gate_shape(result: RenderResult, sample: Sample, pop: Population, transposed: bool) -> GateShape | None:
    gate = pop.gate
    rect = result.plot_rect
    xs, ys = result.x_scale, result.y_scale
    if gate.kind == "rectangle":
        if gate.quadrant and len(gate.dims) == 2 and ys is not None:
            return GateShape(pop, "quadrant", [], QRectF(), QRectF(), "", transposed)
        if len(gate.dims) == 1:
            sc = xs if gate.dims[0] == result.x else ys
            horizontal = gate.dims[0] == result.x
            lo = 0.0 if gate.lo[0] is None else float(sc.forward_scalar(gate.lo[0]))
            hi = 1.0 if gate.hi[0] is None else float(sc.forward_scalar(gate.hi[0]))
            lo, hi = sorted((max(0.0, lo), min(1.0, hi)))
            if horizontal:
                p0 = result.to_pixel(lo, 0)
                p1 = result.to_pixel(hi, 1)
                bbox = QRectF(QPointF(p0.x(), p1.y()), QPointF(p1.x(), p0.y()))
                pts = [(p0.x(), rect.top() + 14), (p1.x(), rect.top() + 14)]
            else:
                p0 = result.to_pixel(0, lo)
                p1 = result.to_pixel(1, hi)
                bbox = QRectF(QPointF(p0.x(), p1.y()), QPointF(p1.x(), p0.y()))
                pts = [(rect.right() - 14, p0.y()), (rect.right() - 14, p1.y())]
            return GateShape(pop, "range", pts, bbox, QRectF(), "", not horizontal, {"histogram": result.histogram})
        dx, dy = (1, 0) if transposed else (0, 1)
        x0 = 0.0 if gate.lo[dx] is None else float(xs.forward_scalar(gate.lo[dx]))
        x1 = 1.0 if gate.hi[dx] is None else float(xs.forward_scalar(gate.hi[dx]))
        y0 = 0.0 if gate.lo[dy] is None else float(ys.forward_scalar(gate.lo[dy]))
        y1 = 1.0 if gate.hi[dy] is None else float(ys.forward_scalar(gate.hi[dy]))
        p0 = result.to_pixel(x0, y0)
        p1 = result.to_pixel(x1, y1)
        bbox = QRectF(QPointF(min(p0.x(), p1.x()), min(p0.y(), p1.y())), QPointF(max(p0.x(), p1.x()), max(p0.y(), p1.y())))
        pts = [(bbox.left(), bbox.top()), (bbox.right(), bbox.top()), (bbox.right(), bbox.bottom()), (bbox.left(), bbox.bottom())]
        return GateShape(pop, "rect", pts, bbox, QRectF(), "", transposed)
    if gate.kind == "polygon":
        vx = np.array([v[0] for v in gate.vertices], dtype=float)
        vy = np.array([v[1] for v in gate.vertices], dtype=float)
        if transposed:
            vx, vy = vy, vx
        fx = xs.forward(vx)
        fy = ys.forward(vy)
        pts = [(p.x(), p.y()) for p in (result.to_pixel(float(a), float(b)) for a, b in zip(fx, fy))]
        if not pts:
            return None
        xsx = [p[0] for p in pts]
        ysy = [p[1] for p in pts]
        bbox = QRectF(QPointF(min(xsx), min(ysy)), QPointF(max(xsx), max(ysy)))
        return GateShape(pop, "polygon", pts, bbox, QRectF(), "", transposed)
    if gate.kind == "ellipse":
        c, pa, pb = gate.center, gate.a_end, gate.b_end
        if transposed:
            c, pa, pb = (c[1], c[0]), (pa[1], pa[0]), (pb[1], pb[0])
        pts = []
        for px_, py_ in (c, pa, pb):
            pt = result.to_pixel(float(xs.forward_scalar(px_)), float(ys.forward_scalar(py_)))
            pts.append((pt.x(), pt.y()))
        cx, cy = pts[0]
        ux, uy = pts[1][0] - cx, pts[1][1] - cy
        vx_, vy_ = pts[2][0] - cx, pts[2][1] - cy
        outline = []
        for k in range(72):
            t = 2 * math.pi * k / 72
            outline.append((cx + ux * math.cos(t) + vx_ * math.sin(t), cy + uy * math.cos(t) + vy_ * math.sin(t)))
        xsx = [p[0] for p in outline]
        ysy = [p[1] for p in outline]
        bbox = QRectF(QPointF(min(xsx), min(ysy)), QPointF(max(xsx), max(ysy)))
        return GateShape(pop, "ellipse", pts, bbox, QRectF(), "", transposed, {"outline": outline})
    return None


def _shape_path(shape: GateShape) -> QPainterPath:
    path = QPainterPath()
    if shape.kind in ("rect", "range"):
        path.addRect(shape.bbox)
    elif shape.kind == "polygon":
        path.addPolygon(QPolygonF([QPointF(a, b) for a, b in shape.points]))
        path.closeSubpath()
    elif shape.kind == "ellipse":
        path.addPolygon(QPolygonF([QPointF(a, b) for a, b in shape.extra["outline"]]))
        path.closeSubpath()
    return path


def _fill_shape(painter: QPainter, shape: GateShape, color: QColor, tint: float):
    """Translucent fill under a gate outline (the gate's tint)."""
    fill = QColor(color)
    fill.setAlphaF(min(max(tint, 0.0), 1.0))
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(fill)
    painter.drawPath(_shape_path(shape))
    painter.restore()


def _draw_padlock(painter: QPainter, anchor: QPointF, color: QColor):
    """A small padlock glyph just above-right of `anchor` (locked gates)."""
    painter.save()
    w, h = 7.0, 5.5
    x, y = anchor.x() + 2.0, anchor.y() - h - 2.0
    painter.setPen(QPen(color, 1.1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawArc(QRectF(x + 1.2, y - 3.6, w - 2.4, 6.0), 0, 180 * 16)
    painter.setBrush(color)
    painter.drawRoundedRect(QRectF(x, y, w, h), 1.0, 1.0)
    painter.restore()


def _draw_shape(painter: QPainter, shape: GateShape):
    if shape.kind == "rect":
        painter.drawRect(shape.bbox)
    elif shape.kind == "polygon":
        painter.drawPolygon(QPolygonF([QPointF(a, b) for a, b in shape.points]))
    elif shape.kind == "ellipse":
        painter.drawPolygon(QPolygonF([QPointF(a, b) for a, b in shape.extra["outline"]]))
    elif shape.kind == "range":
        (x0, y0), (x1, y1) = shape.points
        if not shape.transposed:
            top = y0
            painter.drawLine(QPointF(x0, shape.bbox.top()), QPointF(x0, shape.bbox.bottom()))
            painter.drawLine(QPointF(x1, shape.bbox.top()), QPointF(x1, shape.bbox.bottom()))
            painter.drawLine(QPointF(x0, top), QPointF(x1, top))
        else:
            left = x0
            painter.drawLine(QPointF(shape.bbox.left(), y0), QPointF(shape.bbox.right(), y0))
            painter.drawLine(QPointF(shape.bbox.left(), y1), QPointF(shape.bbox.right(), y1))
            painter.drawLine(QPointF(left, y0), QPointF(left, y1))


def _place_label(painter: QPainter, fm: QFontMetricsF, shape: GateShape, label: str, plot: QRectF, gate, result: RenderResult, placed: list[QRectF] | None = None) -> QRectF:
    if not label:
        return QRectF()
    lines = label.split("\n")
    w = max(fm.horizontalAdvance(l) for l in lines) + 4
    h = fm.height() * len(lines)
    if w > plot.width() - 6:
        # a label wider than the plot: shorten the text to fit
        avail = max(20.0, plot.width() - 10)
        lines = [fm.elidedText(l, Qt.TextElideMode.ElideRight, avail) for l in lines]
        label = "\n".join(lines)
        w = max(fm.horizontalAdvance(l) for l in lines) + 4
    placed = placed or []
    if gate.label_pos is not None:
        p = result.to_pixel(gate.label_pos[0], gate.label_pos[1])
        rect = QRectF(p.x() - w / 2, p.y() - h / 2, w, h)
    else:
        b = shape.bbox.intersected(plot)  # the visible part of a gate that reaches beyond the axes
        if b.isEmpty():
            b = shape.bbox
        if shape.kind == "range":
            if not shape.transposed:
                rect = QRectF(b.center().x() - w / 2, b.top() + 16, w, h)
            else:
                rect = QRectF(b.right() - w - 18, b.center().y() - h / 2, w, h)
        else:
            # preferred spots, like a hand-placed label: right of the gate,
            # above it, left of it, then inside its top-left corner
            candidates = [
                QRectF(b.right() + 3, b.top() - h / 2, w, h),
                QRectF(b.center().x() - w / 2, b.top() - h - 2, w, h),
                QRectF(b.left() - w - 3, b.top() - h / 2, w, h),
                QRectF(b.left() + 3, b.top() + 2, w, h),
            ]
            inner = plot.adjusted(1, 1, -1, -1)

            def fits(r: QRectF) -> bool:
                return inner.contains(r) and not any(r.intersects(p) for p in placed)

            rect = next((c for c in candidates if fits(c)), None)
            if rect is None:
                rect = candidates[0]
                if rect.right() > plot.right() - 2:
                    rect.moveLeft(b.right() - w - 3)
                if rect.left() < plot.left() + 1:
                    rect.moveLeft(plot.left() + 1)
                if rect.top() < plot.top() + 1:
                    rect.moveTop(plot.top() + 1)
                if rect.bottom() > plot.bottom() - 1:
                    rect.moveBottom(plot.bottom() - 1)
        # avoid piling labels of overlapping gates on top of each other
        if placed:
            for _ in range(8):
                clash = next((p for p in placed if p.intersects(rect)), None)
                if clash is None:
                    break
                rect.moveTop(clash.bottom() + 2)
                if rect.bottom() > plot.bottom() - 1:
                    rect.moveTop(min(p.top() for p in placed) - rect.height() - 2)
                    break
    painter.save()
    painter.setPen(QColor("#000000"))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
    painter.restore()
    return rect


def _draw_quadrants(painter: QPainter, fm: QFontMetricsF, result: RenderResult, items: list[tuple[Population, GateShape]], options: RenderOptions, selected):
    rect = result.plot_rect
    xs, ys = result.x_scale, result.y_scale
    # find the shared centre from any gate with both bounds
    cx_val = cy_val = None
    for pop, shape in items:
        g = pop.gate
        dx, dy = (1, 0) if shape.transposed else (0, 1)
        for v in (g.lo[dx], g.hi[dx]):
            if v is not None:
                cx_val = v
        for v in (g.lo[dy], g.hi[dy]):
            if v is not None:
                cy_val = v
    if cx_val is None or cy_val is None:
        return
    fx = float(xs.forward_scalar(cx_val))
    fy = float(ys.forward_scalar(cy_val))
    centre = result.to_pixel(fx, fy)
    any_selected = any(pop is selected for pop, _ in items)
    pen = QPen(QColor(options.selected_gate_color if any_selected else options.gate_color), options.gate_line_width)
    painter.setPen(pen)
    painter.drawLine(QPointF(centre.x(), rect.top()), QPointF(centre.x(), rect.bottom()))
    painter.drawLine(QPointF(rect.left(), centre.y()), QPointF(rect.right(), centre.y()))
    if any(getattr(pop.gate, "locked", False) for pop, _ in items):
        _draw_padlock(painter, centre, pen.color())
    painter.setPen(QColor("#000000"))
    for pop, shape in items:
        q = pop.gate.quadrant[1]
        label = _gate_label(pop, options) if options.show_gate_labels else ""
        lines = label.split("\n") if label else []
        w = max((fm.horizontalAdvance(l) for l in lines), default=0) + 4
        h = fm.height() * max(len(lines), 1)
        avail = (centre.x() - rect.left()) if q in (1, 4) else (rect.right() - centre.x())
        if label and options.show_gate_names and w > avail - 6:
            short = pop.name.split(":")[0].strip() if ":" in pop.name else pop.name[:6]
            lines = [short] + ([format_freq(pop.freq_of_parent, options.freq_decimals)] if options.show_gate_freqs else [])
            label = "\n".join(lines)
            w = max((fm.horizontalAdvance(l) for l in lines), default=0) + 4
            h = fm.height() * max(len(lines), 1)
        if q == 1:
            lr = QRectF(rect.left() + 3, rect.top() + 2, w, h)
            bbox = QRectF(rect.left(), rect.top(), centre.x() - rect.left(), centre.y() - rect.top())
        elif q == 2:
            lr = QRectF(rect.right() - w - 3, rect.top() + 2, w, h)
            bbox = QRectF(centre.x(), rect.top(), rect.right() - centre.x(), centre.y() - rect.top())
        elif q == 3:
            lr = QRectF(rect.right() - w - 3, rect.bottom() - h - 2, w, h)
            bbox = QRectF(centre.x(), centre.y(), rect.right() - centre.x(), rect.bottom() - centre.y())
        else:
            lr = QRectF(rect.left() + 3, rect.bottom() - h - 2, w, h)
            bbox = QRectF(rect.left(), centre.y(), centre.x() - rect.left(), rect.bottom() - centre.y())
        if label:
            painter.drawText(lr, Qt.AlignmentFlag.AlignCenter, label)
        shape.points = [(centre.x(), centre.y())]
        shape.bbox = bbox
        shape.label_rect = lr
        shape.label = label
        shape.extra = {"centre": (centre.x(), centre.y()), "index": q}
        result.gates.append(shape)
