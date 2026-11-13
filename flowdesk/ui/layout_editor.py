"""Layout Editor: arrange plots of populations and free text on printable
pages, overlay populations by dropping them onto a plot (or from a chooser),
align / distribute / size objects, copy and paste them between layouts, and
export pages (PNG / JPEG / TIFF / BMP at a chosen resolution, PDF, SVG,
PowerPoint), copy them to the clipboard as images, print them, or batch a
layout over the samples of a group (one page per sample: PDF, PowerPoint,
separate image files, printer or new layouts) behind a cancellable progress
dialog.

Text boxes and plot annotations may contain keyword / field / statistic
tokens such as {$FIL}, {Sample} or {Median FSC-A}; they are resolved per
page (see flowdesk.core.layout_text).

Coordinates: one scene unit is one page point (1/72 inch). The persisted model
is flowdesk.core.layouts.Layout / LayoutItem; every edit made on the canvas is
written back into those objects and announced through
controller.notify_layouts() (coalesced with a short timer while dragging).
Every such announcement also records an undo step ("edit layout"): the
notifications of one interaction (a drag, a resize, a held arrow key) on the
same items are one step, which ends on mouse release / focus out, see
SectionUndoRecorder and LayoutEditorWindow.schedule_notify().

Extension points for other packages: LayoutItem.props / Layout.props (free
JSON settings), LayoutEditorWindow.register_item_kind() (custom page objects)
and LayoutEditorWindow.register_text_expander() (custom text tokens).
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QMarginsF, QMimeData, QPointF, QRectF, QSettings, QSizeF, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QCursor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QKeySequence,
    QPageLayout,
    QPageSize,
    QAbstractTextDocumentLayout,
    QPainter,
    QPdfWriter,
    QPen,
    QIcon,
    QPixmap,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFontDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import layout_text
from ..core.gates import new_id
from ..core.layout_text import TextContext, annotation_label, annotation_lines, expand_text, safe_filename, stat_token
from ..core.layouts import CUSTOM_PAGE, DEFAULT_MARGIN, PAGE_SIZES, Layout, LayoutItem
from ..core.model import PLOT_TYPES, GraphSettings, Group, Population, Sample
from ..core.pptx_export import PptxPicture, PptxSlide, PptxText, write_pptx
from ..core.stats import NEEDS_PARAM, STATISTICS
from .controller import POPULATION_MIME
from .icons import icon
from .renderer import RenderOptions, render_plot
from .ribbon import Ribbon

MARGIN = DEFAULT_MARGIN  # default page margin guide (points); a layout's own is Layout.margin
GRID = 8.0  # default snap grid (points)
SPACING = 16.0  # gap between dropped / arranged items
DEFAULT_PLOT = 200.0
CAPTION_FONT = 8.0  # caption font (points)
CAPTION_LINE = 10.0
CAPTION_H = 3 * CAPTION_LINE + 4.0
TITLE_H = 14.0
HANDLE = 6.0
MIN_SIZE = 40.0
HEADER_FONT = 12.0
SCENE_SCALE = 2.0  # device scale factor for on-screen plot images
HISTOGRAM_ENTRY = "— histogram —"
FONT_SIZES = [6, 8, 9, 10, 11, 12, 14, 16, 18, 24, 36, 48]
IMAGE_DPI = 96.0  # logical dpi of a QImage, used by render_plot for its fonts
ITEMS_MIME = "application/x-flowdesk-layout-items"
DPI_CHOICES = [72, 96, 150, 300, 600]
# raster formats: file suffix -> (Qt writer name, label)
IMAGE_FORMATS = {
    "png": ("PNG", "PNG image"),
    "jpg": ("JPEG", "JPEG image"),
    "tif": ("TIFF", "TIFF image"),
    "bmp": ("BMP", "BMP image"),
}
FORMAT_ALIASES = {"jpeg": "jpg", "tiff": "tif"}
TEXT_ALIGN = {
    "left": Qt.AlignmentFlag.AlignLeft,
    "center": Qt.AlignmentFlag.AlignHCenter,
    "right": Qt.AlignmentFlag.AlignRight,
    "justify": Qt.AlignmentFlag.AlignJustify,
}
# per-plot display options offered by the Layout Plot Options dialog:
# (key, label, choices) - the keys are RenderOptions fields stored in
# item.graph.options, like a graph window stores them
DISPLAY_TOGGLES = [
    ("show_gates", "Show gates", True),
    ("show_gate_labels", "Show gate labels", True),
    ("show_gate_names", "Show gate names", True),
    ("show_gate_freqs", "Show gate frequencies", True),
    ("show_outliers", "Show outliers (contour plots)", True),
]
DISPLAY_CHOICES = [
    ("smoothing", "Smoothing", [("None", 0), ("Low", 1), ("Medium", 2), ("High", 3)]),
    ("dot_size", "Dot size", [("1 px", 1), ("2 px", 2), ("3 px", 3), ("4 px", 4)]),
    ("dots_to_draw", "Events to draw", [("10,000", 10000), ("25,000", 25000), ("50,000", 50000), ("100,000", 100000), ("All", 10_000_000)]),
    ("contour_level", "Contour level", [("2%", 2.0), ("5%", 5.0), ("10%", 10.0), ("20%", 20.0)]),
    ("histogram_ymode", "Histogram y-axis", [("Count", "count"), ("% of Max (normalized to mode)", "mode")]),
    ("histogram_line_width", "Histogram line width", [("Thin (0.8)", 0.8), ("Normal (1.2)", 1.2), ("Medium (2)", 2.0), ("Thick (3)", 3.0)]),
    ("colormap", "Colour map", [("Pseudocolor", "pseudocolor"), ("Grayscale", "gray")]),
    ("background", "Background", [("White", "#ffffff"), ("Light grey", "#f2f2f2"), ("Black", "#000000")]),
]
DISPLAY_KEYS = [k for k, _, _ in DISPLAY_TOGGLES] + [k for k, _, _ in DISPLAY_CHOICES] + ["font_family"]
# item.props keys for fonts of a plot item (points)
PLOT_FONT_KEYS = ("caption_font_size", "title_font_size", "axis_label_size", "axis_number_size")


# ------------------------------------------------------------------ helpers
def _mime_payload(mime):
    if mime is None or not mime.hasFormat(POPULATION_MIME):
        return None
    try:
        return json.loads(bytes(mime.data(POPULATION_MIME)).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def parse_population_entries(mime) -> list[dict]:
    """Decode a POPULATION_MIME payload (one {"sample_id", "paths"} entry or a
    list of them; group-template entries carry "group_id", dragged statistic
    rows "stats") -> a list of normalised entries."""
    data = _mime_payload(mime)
    if data is None:
        return []
    out = []
    for d in data if isinstance(data, list) else [data]:
        if not isinstance(d, dict):
            continue
        paths = d.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        out.append(
            {
                "sample_id": d.get("sample_id"),
                "group_id": d.get("group_id"),
                "paths": [str(p) for p in paths],
                "stats": [dict(st) for st in d.get("stats") or [] if isinstance(st, dict)],
            }
        )
    return out


def parse_population_mime(mime) -> tuple[int | None, list[str]]:
    """Decode a POPULATION_MIME payload -> (sample_id, paths) of its first
    entry (see parse_population_entries for multi-sample payloads)."""
    entries = parse_population_entries(mime)
    if not entries:
        return None, []
    return entries[0]["sample_id"], entries[0]["paths"]


def caption_font_size(item: LayoutItem) -> float:
    try:
        return max(4.0, float(item.props.get("caption_font_size", CAPTION_FONT)))
    except (TypeError, ValueError):
        return CAPTION_FONT


def caption_line_height(item: LayoutItem) -> float:
    return max(CAPTION_LINE, caption_font_size(item) * 1.25)


def item_annotations(item: LayoutItem) -> list[dict]:
    ann = item.props.get("annotations") if isinstance(item.props, dict) else None
    return [a for a in ann if isinstance(a, dict)] if isinstance(ann, list) else []


def caption_line_count(item: LayoutItem) -> int:
    return (3 if item.show_caption else 0) + len(item_annotations(item))


def caption_height(item: LayoutItem) -> float:
    n = caption_line_count(item)
    if not n:
        return 0.0
    if n == 3 and caption_line_height(item) == CAPTION_LINE:
        return CAPTION_H
    return n * caption_line_height(item) + 4.0


def title_font_size(item: LayoutItem) -> float:
    try:
        return max(4.0, float(item.props.get("title_font_size", CAPTION_FONT + 2)))
    except (TypeError, ValueError):
        return CAPTION_FONT + 2


def title_height(item: LayoutItem) -> float:
    if not item.show_title:
        return 0.0
    return max(TITLE_H, title_font_size(item) * 1.4)


def plot_rect(item: LayoutItem) -> QRectF:
    """The plot image area in item-local coordinates."""
    top = title_height(item)
    h = max(20.0, item.h - top - caption_height(item))
    return QRectF(0.0, top, max(20.0, item.w), h)


def _px_font(size: float, bold: bool = False) -> QFont:
    f = QFont()
    f.setPixelSize(max(1, int(round(size))))
    f.setBold(bold)
    return f


def text_format(item: LayoutItem) -> dict:
    """Formatting of a text item (stored in item.props)."""
    p = item.props if isinstance(item.props, dict) else {}
    align = str(p.get("align", "left")).lower()
    return {
        "bold": bool(p.get("bold", False)),
        "italic": bool(p.get("italic", False)),
        "underline": bool(p.get("underline", False)),
        "color": str(p.get("color", "#000000") or "#000000"),
        "align": align if align in TEXT_ALIGN else "left",
        "font_family": str(p.get("font_family", "") or ""),
    }


def text_font(item: LayoutItem) -> QFont:
    fmt = text_format(item)
    f = QFont(fmt["font_family"]) if fmt["font_family"] else QFont()
    f.setPixelSize(max(1, int(round(item.font_size))))
    f.setBold(fmt["bold"])
    f.setItalic(fmt["italic"])
    f.setUnderline(fmt["underline"])
    return f


def _snap(v: float, step: float = GRID) -> float:
    step = step if step and step > 0 else GRID
    return round(v / step) * step


def _copy_item(item: LayoutItem) -> LayoutItem:
    d = item.to_dict()
    d["id"] = new_id()
    return LayoutItem.from_dict(d)


def _reading_order(items: list[LayoutItem]) -> list[LayoutItem]:
    return sorted(items, key=lambda i: (round(i.y / 24.0), i.x))


def image_bytes(image: QImage, fmt: str = "PNG") -> bytes:
    buf = QByteArray()
    dev = QBuffer(buf)
    dev.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(dev, fmt)
    dev.close()
    return bytes(buf.data())


def _set_dpi(image: QImage, dpi: float):
    dpm = int(round(float(dpi) / 0.0254))
    image.setDotsPerMeterX(dpm)
    image.setDotsPerMeterY(dpm)


def normalise_format(fmt: str | None, path=None) -> str:
    """'PNG' / 'jpeg' / '.tiff' / None (from the path suffix) -> a key of
    IMAGE_FORMATS or 'svg' / 'pdf'."""
    if not fmt and path is not None:
        fmt = Path(str(path)).suffix
    f = (fmt or "png").lower().lstrip(".")
    f = FORMAT_ALIASES.get(f, f)
    return f if f in IMAGE_FORMATS or f in ("svg", "pdf") else "png"


class LayoutDefaults:
    """Defaults for new layouts (page, orientation, margin, plot size, grid,
    caption font), persisted in QSettings under "layout_defaults/". Tests (or
    an embedding application) can swap `store` for a plain dict."""

    KEYS = {
        "page": "A4",
        "landscape": False,
        "margin": DEFAULT_MARGIN,
        "plot_size": DEFAULT_PLOT,
        "grid": False,
        "grid_step": GRID,
        "caption_font_size": CAPTION_FONT,
    }

    def __init__(self, store=None):
        self.store = store

    def _settings(self):
        return QSettings("FlowDesk", "FlowDesk")

    def get(self, key: str):
        default = self.KEYS[key]
        if self.store is not None:
            value = self.store.get(key, default)
        else:
            value = self._settings().value(f"layout_defaults/{key}", default)
        try:
            if isinstance(default, bool):
                return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
            if isinstance(default, float):
                return float(value)
            return str(value)
        except (TypeError, ValueError):
            return default

    def set(self, key: str, value):
        if key not in self.KEYS:
            return
        if self.store is not None:
            self.store[key] = value
        else:
            self._settings().setValue(f"layout_defaults/{key}", value)

    def as_dict(self) -> dict:
        return {k: self.get(k) for k in self.KEYS}


# --------------------------------------------------------------------- undo
class SectionUndoRecorder:
    """Undo steps for an editor that changes one workspace section
    ("layouts" or "tables") in place and announces the change afterwards.

    The editors write into the model objects directly (a drag moves the
    LayoutItem as it goes), so there is no single place to call
    controller.begin_change() *before* a change. The recorder keeps the
    section as it was after the last recorded change (the baseline); record()
    pushes one undo step that brings back that baseline plus the present
    gating state (unchanged by an editor).

    Coalescing: a step recorded with coalesce=True opens a burst; later
    coalescing records with the same `key` (the editor's name for what is
    being edited, e.g. a set of (item id, property) pairs) merge into it while
    it is still the top undo step, until end_burst() (mouse release, focus
    out), reset() or a record that does not merge closes the burst. So one
    drag is one undo step however long it takes, and two edits are two steps
    however quickly they follow each other."""

    def __init__(self, controller, section: str, label: str):
        self.controller = controller
        self.section = section
        self.label = label
        self._baseline: list[dict] | None = None
        self._pushed: dict | None = None
        self._burst_open = False
        self._burst_key = None
        self.reset()

    def _state(self) -> list[dict]:
        return [o.to_dict() for o in getattr(self.controller.workspace, self.section, [])]

    def reset(self):
        """Take the present state as the one the next step returns to (after
        undo / redo, a replaced workspace or a change made elsewhere)."""
        self._baseline = self._state()
        self.end_burst()

    def end_burst(self):
        """The interaction that coalescing records belonged to is over: the
        next change is a step of its own."""
        self._burst_open = False
        self._burst_key = None

    def record(self, coalesce: bool = False, key=None) -> bool:
        """Call after a change, before announcing it. Returns whether a new
        undo step was pushed."""
        c = self.controller
        current = self._state()
        before, self._baseline = self._baseline, current
        if before is None or before == current:
            return False
        if getattr(c, "_undo_group_depth", 0):
            # inside an open undo group: the group's step must bring back the
            # section as it was before this change. begin_change() would widen
            # the step from the present (already edited) state, so the
            # baseline is written into the step instead.
            from ..core.serialization import snapshot_kind

            top = c.undo_top_snapshot()
            had = top is not None and self.section in snapshot_kind(top)
            c.begin_change(self.label, kind=self.section)
            top = c.undo_top_snapshot()
            if not had and top is not None:
                top[self.section] = before
            self.end_burst()
            return False
        if (
            coalesce
            and self._burst_open
            and self._pushed is not None
            and c.undo_top_snapshot() is self._pushed
            and key == self._burst_key
        ):
            return False  # same burst: the open step already returns to before it
        from ..core.serialization import gating_snapshot

        snap = gating_snapshot(c.workspace, self.section)
        snap[self.section] = before
        c.push_undo(self.label, snap)
        self._pushed = snap
        self._burst_open, self._burst_key = coalesce, (key if coalesce else None)
        return True


# -------------------------------------------------------------- plot images
class PlotImageCache:
    """LRU cache of rendered plot images, keyed by everything that affects the
    rendering (data version, sample, population, graph, overlays, size)."""

    def __init__(self, limit: int = 300):
        self.limit = limit
        self._images: OrderedDict = OrderedDict()
        self.version = 0
        self.renders = 0  # number of actual render_plot calls (for tests)

    def invalidate(self):
        self.version += 1
        self._images.clear()

    def get(self, key):
        img = self._images.get(key)
        if img is not None:
            self._images.move_to_end(key)
        return img

    def put(self, key, image: QImage):
        self._images[key] = image
        self._images.move_to_end(key)
        while len(self._images) > self.limit:
            self._images.popitem(last=False)


# ---------------------------------------------------------------- dialogs
class PageSetupDialog(QDialog):
    """Paper size (or a custom size), orientation and margins of a layout,
    optionally saved as the default for new layouts."""

    UNITS = {"pt": 1.0, "mm": 72.0 / 25.4, "in": 72.0}

    def __init__(self, layout: Layout, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Page Setup")
        form = QFormLayout(self)
        self.page_combo = QComboBox()
        self.page_combo.addItems(list(PAGE_SIZES.keys()) + [CUSTOM_PAGE])
        current = layout.page if (layout.page in PAGE_SIZES or layout.page == CUSTOM_PAGE) else "A4"
        self.page_combo.setCurrentText(current)
        form.addRow("Page size:", self.page_combo)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(list(self.UNITS))
        self.unit_combo.setCurrentText("mm")
        form.addRow("Units:", self.unit_combo)
        w, h = layout.page_size
        if layout.landscape:
            w, h = h, w
        self.width_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        self.margin_spin = QDoubleSpinBox()
        for sp in (self.width_spin, self.height_spin, self.margin_spin):
            sp.setDecimals(1)
            sp.setRange(0.0, 5000.0)
        size_row = QHBoxLayout()
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self.height_spin)
        form.addRow("Custom size (W × H):", size_row)
        form.addRow("Margins:", self.margin_spin)
        self.portrait = QRadioButton("Portrait")
        self.landscape = QRadioButton("Landscape")
        (self.landscape if layout.landscape else self.portrait).setChecked(True)
        row = QHBoxLayout()
        row.addWidget(self.portrait)
        row.addWidget(self.landscape)
        form.addRow("Orientation:", row)
        self.default_check = QCheckBox("Use as the default for new layouts")
        form.addRow(self.default_check)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self._pt = {"w": w, "h": h, "margin": layout.margin}
        self._unit = "mm"
        self._show_values()
        self.unit_combo.currentTextChanged.connect(self._unit_changed)
        self.page_combo.currentTextChanged.connect(self._page_changed)
        self._page_changed(self.page_combo.currentText())

    def _factor(self, unit: str | None = None) -> float:
        return self.UNITS.get(unit or self._unit, 1.0)

    def _show_values(self):
        f = self._factor()
        self.width_spin.setValue(self._pt["w"] / f)
        self.height_spin.setValue(self._pt["h"] / f)
        self.margin_spin.setValue(self._pt["margin"] / f)

    def _read_values(self):
        f = self._factor()
        self._pt = {"w": self.width_spin.value() * f, "h": self.height_spin.value() * f, "margin": self.margin_spin.value() * f}

    def _unit_changed(self, unit: str):
        self._read_values()
        self._unit = unit
        self._show_values()

    def _page_changed(self, page: str):
        custom = page == CUSTOM_PAGE
        self.width_spin.setEnabled(custom)
        self.height_spin.setEnabled(custom)
        if not custom and page in PAGE_SIZES:
            self._read_values()
            self._pt["w"], self._pt["h"] = PAGE_SIZES[page]
            self._show_values()

    def result_values(self) -> dict:
        self._read_values()
        return {
            "page": self.page_combo.currentText(),
            "landscape": self.landscape.isChecked(),
            "margin": round(self._pt["margin"], 2),
            "custom_size": (round(self._pt["w"], 2), round(self._pt["h"], 2)) if self.page_combo.currentText() == CUSTOM_PAGE else None,
            "make_default": self.default_check.isChecked(),
        }

    def values(self) -> tuple[str, bool]:
        return self.page_combo.currentText(), self.landscape.isChecked()


class StatisticDialog(QDialog):
    """Pick a population statistic (or a keyword) for an annotation / text."""

    def __init__(self, editor: "LayoutEditorWindow", sample: Sample | None = None, pop_path: str | None = None, parent=None, keyword_mode: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Insert Keyword" if keyword_mode else "Insert Statistic")
        ws = editor.workspace
        form = QFormLayout(self)
        self.keyword_mode = keyword_mode
        if keyword_mode:
            self.keyword_combo = QComboBox()
            self.keyword_combo.setEditable(True)
            for name in layout_text.BUILTIN_FIELDS:
                self.keyword_combo.addItem(f"{name} (field)", name)
            samples = [sample] if sample is not None else ws.samples
            seen: dict[str, None] = {}
            for s in samples:
                for k in getattr(s, "keywords", {}) or {}:
                    seen.setdefault(str(k), None)
            for k in sorted(seen, key=lambda k: (k.startswith("$"), k.lower())):
                self.keyword_combo.addItem(k, k)
            form.addRow("Keyword / field:", self.keyword_combo)
        else:
            self.pop_combo = QComboBox()
            self.pop_combo.addItem("(this plot's population)", None)
            self.pop_combo.addItem("Ungated", "")
            paths = [p.path for p in sample.root.walk(False)] if sample is not None else ws.population_paths(None)
            for path in paths:
                self.pop_combo.addItem(path.lstrip("/").replace("/", " / "), path)
            if pop_path is not None:
                i = self.pop_combo.findData(pop_path)
                if i >= 0:
                    self.pop_combo.setCurrentIndex(i)
            form.addRow("Population:", self.pop_combo)
            self.stat_combo = QComboBox()
            self.stat_combo.addItems(STATISTICS)
            self.stat_combo.setCurrentText("Freq. of Parent")
            form.addRow("Statistic:", self.stat_combo)
            self.param_combo = QComboBox()
            names = sample.parameter_names() if sample is not None else ws.all_parameter_names(None)
            for n in names:
                label = sample.param_label(n) if sample is not None and hasattr(sample, "param_label") else n
                self.param_combo.addItem(label, n)
            form.addRow("Parameter:", self.param_combo)
            self.pct_spin = QDoubleSpinBox()
            self.pct_spin.setRange(0.0, 100.0)
            self.pct_spin.setDecimals(1)
            self.pct_spin.setValue(50.0)
            form.addRow("Percentile:", self.pct_spin)
            self.stat_combo.currentTextChanged.connect(self._stat_changed)
            self._stat_changed(self.stat_combo.currentText())
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _stat_changed(self, name: str):
        self.param_combo.setEnabled(name in NEEDS_PARAM)
        self.pct_spin.setEnabled(name == "Percentile")

    def entry(self) -> dict:
        """An annotation entry: {"statistic", "param", "percentile", "pop_path"} or {"keyword"}."""
        if self.keyword_mode:
            data = self.keyword_combo.currentData()
            text = self.keyword_combo.currentText().strip()
            if data is None or (text and text != self.keyword_combo.itemText(self.keyword_combo.currentIndex())):
                data = text
            return {"keyword": str(data)}
        stat = self.stat_combo.currentText()
        e = {"statistic": stat}
        if stat in NEEDS_PARAM:
            e["param"] = self.param_combo.currentData() or self.param_combo.currentText()
        if stat == "Percentile":
            e["percentile"] = float(self.pct_spin.value())
        pop = self.pop_combo.currentData()
        if pop is not None:
            e["pop_path"] = pop
        return e

    def token(self) -> str:
        e = self.entry()
        if "keyword" in e:
            return layout_text.keyword_token(e["keyword"])
        return stat_token(e["statistic"], e.get("param"), e.get("percentile"), e.get("pop_path"))


class PlotOptionsDialog(QDialog):
    """Edit a plot item (FlowJo's graph definition for a layout plot): the
    sample and population it shows, plot type and axes, caption / title,
    display options (gates, smoothing, dot size, contour level, events,
    colour map, background, fonts), overlays and annotations."""

    def __init__(self, editor: "LayoutEditorWindow", item: LayoutItem, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layout Plot Options")
        self.item = item
        self.editor = editor
        self.overlays = [dict(o) for o in item.overlays]
        self.annotations = [dict(a) for a in item_annotations(item)]
        self._reset_display = False
        ws = editor.workspace
        sample = ws.sample_by_id(item.sample_id) if item.sample_id is not None else None
        params = sample.parameter_names() if sample is not None else []
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        # -- graph
        page = QWidget()
        form = QFormLayout(page)
        self.sample_combo = QComboBox()
        for s in ws.samples:
            self.sample_combo.addItem(s.name, s.id)
        if sample is None:
            self.sample_combo.insertItem(0, "(missing sample)", item.sample_id)
            self.sample_combo.setCurrentIndex(0)
        else:
            self.sample_combo.setCurrentIndex(max(0, self.sample_combo.findData(sample.id)))
        form.addRow("Sample:", self.sample_combo)
        self.pop_combo = QComboBox()
        form.addRow("Population:", self.pop_combo)
        self._fill_populations(sample, item.pop_path)
        self.type_combo = QComboBox()
        for key, label in PLOT_TYPES:
            self.type_combo.addItem(label, key)
        idx = self.type_combo.findData(item.graph.plot_type)
        self.type_combo.setCurrentIndex(max(0, idx))
        form.addRow("Plot type:", self.type_combo)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self._fill_axes(params, item.graph.x, item.graph.y if item.graph.plot_type != "histogram" else None)
        form.addRow("X axis:", self.x_combo)
        form.addRow("Y axis:", self.y_combo)
        self.caption_check = QCheckBox("Show caption (sample, population, count)")
        self.caption_check.setChecked(item.show_caption)
        form.addRow(self.caption_check)
        self.title_check = QCheckBox("Show title")
        self.title_check.setChecked(item.show_title)
        form.addRow(self.title_check)
        self.tabs.addTab(page, "Graph")

        # -- display: stored in item.graph.options like a graph window stores
        # them in the population's graph settings
        effective = RenderOptions.from_prefs(editor.controller.preferences, item.graph)
        self._prefs_effective = RenderOptions.from_prefs(editor.controller.preferences, None)
        page = QWidget()
        form = QFormLayout(page)
        self.display_checks: dict[str, QCheckBox] = {}
        for key, label, default in DISPLAY_TOGGLES:
            cb = QCheckBox(label)
            cb.setChecked(bool(getattr(effective, key, default)))
            form.addRow(cb)
            self.display_checks[key] = cb
        self.display_combos: dict[str, QComboBox] = {}
        for key, label, choices in DISPLAY_CHOICES:
            combo = QComboBox()
            for text, value in choices:
                combo.addItem(text, value)
            self._select_value(combo, getattr(effective, key, None))
            form.addRow(f"{label}:", combo)
            self.display_combos[key] = combo
        self.font_combo = QFontComboBox()
        family = item.graph.options.get("font_family") or ""
        self.font_combo.setEditable(True)
        if family:
            self.font_combo.setCurrentFont(QFont(family))
        else:
            self.font_combo.setCurrentIndex(-1)
            self.font_combo.setEditText("")
        form.addRow("Plot font:", self.font_combo)
        reset = QPushButton("Reset to Preferences")
        reset.setToolTip("Use the global Preferences for every display option of this plot")
        reset.clicked.connect(self._reset_to_prefs)
        form.addRow(reset)
        self._initial_display = self._display_values()
        self.tabs.addTab(page, "Display")

        # -- overlays
        page = QWidget()
        form = QFormLayout(page)
        self.legend_check = QCheckBox("Show overlay legend")
        self.legend_check.setChecked(bool(effective.show_legend))
        form.addRow(self.legend_check)
        self.offset_check = QCheckBox("Offset (staggered) overlaid histograms")
        self.offset_check.setChecked(bool(effective.histogram_offset))
        form.addRow(self.offset_check)
        self.overlay_list = QListWidget()
        self.overlay_list.setMinimumHeight(90)
        form.addRow("Overlays:", self.overlay_list)
        row = QHBoxLayout()
        add = QPushButton("Add…")
        add.setToolTip("Choose populations of any sample of the group to overlay")
        add.clicked.connect(self._choose_overlays)
        row.addWidget(add)
        add_all = QPushButton("Add All Samples of Group")
        add_all.setToolTip("Overlay this population of every sample in the batch group")
        add_all.clicked.connect(self._add_group_overlays)
        row.addWidget(add_all)
        form.addRow(row)
        self._fill_overlays()
        self.tabs.addTab(page, "Overlays")

        # -- annotate
        page = QWidget()
        form = QFormLayout(page)
        form.addRow(QLabel("Statistics and keywords listed under the plot (per page when batching):"))
        self.annotation_list = QListWidget()
        self.annotation_list.setMinimumHeight(90)
        form.addRow(self.annotation_list)
        row = QHBoxLayout()
        for text, slot in (("Add Statistic…", self._add_statistic), ("Add Keyword…", self._add_keyword), ("Remove", self._remove_annotation)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        form.addRow(row)
        self._fill_annotations()
        self.tabs.addTab(page, "Annotate")

        # -- fonts (points; 0 = automatic)
        page = QWidget()
        form = QFormLayout(page)
        self.font_spins: dict[str, QDoubleSpinBox] = {}
        for key, label in (
            ("caption_font_size", "Caption font"),
            ("title_font_size", "Title font"),
            ("axis_label_size", "Axis label font"),
            ("axis_number_size", "Axis number font"),
        ):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 48.0)
            sp.setDecimals(1)
            sp.setSpecialValueText("Automatic")
            try:
                sp.setValue(float(item.props.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                sp.setValue(0.0)
            form.addRow(f"{label}:", sp)
            self.font_spins[key] = sp
        self.tabs.addTab(page, "Fonts")

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)
        self.type_combo.currentIndexChanged.connect(self._type_changed)
        self.sample_combo.currentIndexChanged.connect(self._sample_changed)

    # -- helpers
    @staticmethod
    def _select_value(combo: QComboBox, value):
        for i in range(combo.count()):
            d = combo.itemData(i)
            try:
                same = d == value or (isinstance(d, (int, float)) and isinstance(value, (int, float)) and abs(float(d) - float(value)) < 1e-9)
            except TypeError:
                same = False
            if same:
                combo.setCurrentIndex(i)
                return
        if value is not None:
            combo.addItem(str(value), value)
            combo.setCurrentIndex(combo.count() - 1)

    def selected_sample(self) -> Sample | None:
        sid = self.sample_combo.currentData()
        return self.editor.workspace.sample_by_id(sid) if sid is not None else None

    def _fill_populations(self, sample: Sample | None, current: str, keep_missing: bool = True):
        self.pop_combo.blockSignals(True)
        self.pop_combo.clear()
        self.pop_combo.addItem("Ungated", "")
        if sample is not None:
            for p in sample.root.walk(False):
                self.pop_combo.addItem("    " * (p.depth - 1) + p.name, p.path)
        i = self.pop_combo.findData(current or "")
        if i < 0 and current and keep_missing:
            self.pop_combo.addItem(f"{current} (missing)", current)
            i = self.pop_combo.count() - 1
        self.pop_combo.setCurrentIndex(max(0, i))
        self.pop_combo.blockSignals(False)

    def _fill_axes(self, params: list[str], x: str | None, y: str | None):
        self.x_combo.clear()
        self.x_combo.addItems(params)
        if x and x not in params:
            self.x_combo.addItem(x)
        if x:
            self.x_combo.setCurrentText(x)
        self.y_combo.clear()
        self.y_combo.addItem(HISTOGRAM_ENTRY)
        self.y_combo.addItems(params)
        if y and y not in params:
            self.y_combo.addItem(y)
        if y:
            self.y_combo.setCurrentText(y)
        else:
            self.y_combo.setCurrentIndex(0)

    def _sample_changed(self):
        sample = self.selected_sample()
        self._fill_populations(sample, self.pop_combo.currentData() or "", keep_missing=False)  # absent there: Ungated
        params = sample.parameter_names() if sample is not None else []
        y = None if self.y_combo.currentIndex() == 0 else self.y_combo.currentText()
        self._fill_axes(params, self.x_combo.currentText(), y)

    def _type_changed(self):
        if self.type_combo.currentData() == "histogram":
            self.y_combo.setCurrentIndex(0)
        elif self.y_combo.currentIndex() == 0 and self.y_combo.count() > 2:
            self.y_combo.setCurrentIndex(2 if self.y_combo.count() > 2 else 1)

    def _display_values(self) -> dict:
        out = {k: cb.isChecked() for k, cb in self.display_checks.items()}
        out.update({k: c.currentData() for k, c in self.display_combos.items()})
        out["font_family"] = self.font_combo.currentText().strip()
        return out

    def _reset_to_prefs(self):
        eff = self._prefs_effective
        for key, cb in self.display_checks.items():
            cb.setChecked(bool(getattr(eff, key, True)))
        for key, combo in self.display_combos.items():
            self._select_value(combo, getattr(eff, key, None))
        self.font_combo.setCurrentIndex(-1)
        self.font_combo.setEditText("")
        self._reset_display = True
        self._initial_display = self._display_values()

    def changed_display_options(self) -> dict:
        """Display options the user changed in the dialog."""
        now = self._display_values()
        return {k: v for k, v in now.items() if v != self._initial_display.get(k)}

    def _fill_overlays(self):
        self.overlay_list.clear()
        ws = self.editor.workspace
        for k, ov in enumerate(self.overlays):
            s = ws.sample_by_id(ov.get("sample_id"))
            label = f"{s.name if s else '(missing sample)'} — {ov.get('pop_path') or 'Ungated'}"
            li = QListWidgetItem()
            self.overlay_list.addItem(li)
            w = QWidget()
            row = QHBoxLayout(w)
            row.setContentsMargins(4, 0, 4, 0)
            row.addWidget(QLabel(label), 1)
            btn = QToolButton()
            btn.setIcon(icon("delete"))
            btn.setToolTip("Remove overlay")
            btn.clicked.connect(lambda _=False, k=k: self._remove_overlay(k))
            row.addWidget(btn)
            li.setSizeHint(w.sizeHint())
            self.overlay_list.setItemWidget(li, w)

    def _remove_overlay(self, k: int):
        if 0 <= k < len(self.overlays):
            del self.overlays[k]
            self._fill_overlays()

    def _base(self) -> tuple[int | None, str]:
        return self.sample_combo.currentData(), self.pop_combo.currentData() or ""

    def add_overlay_entries(self, entries: list[dict]) -> int:
        base = self._base()
        n = 0
        for e in entries:
            entry = {"sample_id": e.get("sample_id"), "pop_path": e.get("pop_path") or ""}
            if (entry["sample_id"], entry["pop_path"]) == base or entry in self.overlays:
                continue
            self.overlays.append(entry)
            n += 1
        self._fill_overlays()
        return n

    def _choose_overlays(self):
        from .dialogs import OverlayDialog

        ws = self.editor.workspace
        samples = self.editor.overlay_samples(self.item)
        current = []
        for ov in self.overlays:
            s = ws.sample_by_id(ov.get("sample_id"))
            p = (s.root.find(ov.get("pop_path") or "") if ov.get("pop_path") else s.root) if s is not None else None
            if p is not None:
                current.append((s, p))
        dlg = OverlayDialog(samples, current, self)
        if dlg.exec():
            self.overlays = []
            self.add_overlay_entries([{"sample_id": s.id, "pop_path": p.path} for s, p in dlg.values()])

    def _add_group_overlays(self):
        sid, path = self._base()
        entries = []
        for s in self.editor.overlay_samples(self.item):
            if (path and s.root.find(path) is None) or s.id == sid:
                continue
            entries.append({"sample_id": s.id, "pop_path": path})
        self.add_overlay_entries(entries)

    def _fill_annotations(self):
        self.annotation_list.clear()
        for a in self.annotations:
            self.annotation_list.addItem(annotation_label(a))

    def _add_statistic(self):
        dlg = StatisticDialog(self.editor, self.selected_sample(), None, self)
        if dlg.exec():
            self.annotations.append(dlg.entry())
            self._fill_annotations()

    def _add_keyword(self):
        dlg = StatisticDialog(self.editor, self.selected_sample(), None, self, keyword_mode=True)
        if dlg.exec():
            self.annotations.append(dlg.entry())
            self._fill_annotations()

    def _remove_annotation(self):
        row = self.annotation_list.currentRow()
        if 0 <= row < len(self.annotations):
            del self.annotations[row]
            self._fill_annotations()

    # -- result
    def apply_display_options(self, item: LayoutItem, changed: dict | None = None, reset: bool | None = None):
        """Write the changed display options (and fonts) into `item`; also
        used for the other selected plots of a multi-selection."""
        changed = self.changed_display_options() if changed is None else changed
        reset = self._reset_display if reset is None else reset
        opts = item.graph.options
        if reset:
            for k in DISPLAY_KEYS:
                opts.pop(k, None)
        for k, v in changed.items():
            if k == "font_family" and not v:
                opts.pop(k, None)
            else:
                opts[k] = v
        for key, sp in self.font_spins.items():
            v = float(sp.value())
            if v > 0:
                item.props[key] = v
            else:
                item.props.pop(key, None)

    def apply(self):
        it = self.item
        sid, path = self._base()
        if sid is not None:
            it.sample_id = sid
        it.pop_path = path
        ptype = self.type_combo.currentData()
        y = None if self.y_combo.currentIndex() == 0 else self.y_combo.currentText()
        if y is None:
            ptype = "histogram"
        elif ptype == "histogram":
            ptype = "pseudocolor"
        it.graph.plot_type = ptype
        it.graph.x = self.x_combo.currentText() or it.graph.x
        it.graph.y = y if y is not None else it.graph.y
        if ptype == "histogram":
            it.graph.y = None
        it.show_caption = self.caption_check.isChecked()
        it.show_title = self.title_check.isChecked()
        it.graph.options["show_legend"] = self.legend_check.isChecked()
        it.graph.options["histogram_offset"] = self.offset_check.isChecked()
        self.apply_display_options(it)
        it.overlays = [o for o in self.overlays if (o.get("sample_id"), o.get("pop_path") or "") != (it.sample_id, it.pop_path)]
        if self.annotations:
            it.props["annotations"] = [dict(a) for a in self.annotations]
        else:
            it.props.pop("annotations", None)


class BatchDialog(QDialog):
    """FlowJo's layout Batch dialog: what to iterate over (the samples of a
    group, or one page per keyword value), the page header and the output
    (one PDF, one PowerPoint, separate image files, the printer or new
    layouts)."""

    OUTPUTS = [
        ("pdf", "PDF document (one page per sample)"),
        ("pptx", "PowerPoint presentation (one slide per sample)"),
        ("images", "Separate image files per sample"),
        ("printer", "Printer"),
        ("layouts", "New layouts (one per sample)"),
    ]

    def __init__(self, editor: "LayoutEditorWindow", layout: Layout, output: str | None = None, parent=None):
        super().__init__(parent or editor)
        self.setWindowTitle("Batch")
        ws = editor.workspace
        form = QFormLayout(self)
        self.group_combo = QComboBox()
        self.group_combo.addItems([g.name for g in ws.groups])
        group = layout.batch_group or editor.group_combo.currentText()
        if group:
            self.group_combo.setCurrentText(group)
        form.addRow("Group:", self.group_combo)
        self.by_group = QRadioButton("Each sample of the group")
        self.by_keyword = QRadioButton("Each value of keyword:")
        (self.by_keyword if layout.props.get("batch_by") == "keyword" else self.by_group).setChecked(True)
        self.keyword_combo = QComboBox()
        self.keyword_combo.setEditable(True)
        seen: dict[str, None] = {}
        for s in ws.samples:
            for k in getattr(s, "keywords", {}) or {}:
                seen.setdefault(str(k), None)
        self.keyword_combo.addItems(sorted(seen, key=lambda k: (k.startswith("$"), k.lower())))
        if layout.props.get("batch_keyword"):
            self.keyword_combo.setCurrentText(str(layout.props["batch_keyword"]))
        row = QHBoxLayout()
        row.addWidget(self.by_keyword)
        row.addWidget(self.keyword_combo, 1)
        form.addRow("Iterate:", self.by_group)
        form.addRow("", row)
        self.header_edit = QLineEdit(str(layout.props.get("batch_header", "")))
        self.header_edit.setPlaceholderText("{Sample}   (tokens: {$FIL}, {Group}, {Page}/{Pages}, {Median FSC-A} …)")
        form.addRow("Page header:", self.header_edit)
        self.output_combo = QComboBox()
        for key, label in self.OUTPUTS:
            self.output_combo.addItem(label, key)
        if output:
            self.output_combo.setCurrentIndex(max(0, self.output_combo.findData(output)))
        form.addRow("Output:", self.output_combo)
        self.format_combo = QComboBox()
        for key, (_, label) in IMAGE_FORMATS.items():
            self.format_combo.addItem(label, key)
        self.format_combo.addItem("SVG image", "svg")
        self.format_combo.addItem("PDF document", "pdf")
        form.addRow("Image format:", self.format_combo)
        self.dpi_combo = QComboBox()
        for d in DPI_CHOICES:
            self.dpi_combo.addItem(f"{d} dpi", d)
        self.dpi_combo.setCurrentIndex(DPI_CHOICES.index(300))
        form.addRow("Resolution:", self.dpi_combo)
        self.page_image_check = QCheckBox("PowerPoint: each page as one picture (instead of editable objects)")
        form.addRow(self.page_image_check)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.output_combo.currentIndexChanged.connect(self._output_changed)
        self._output_changed()

    def _output_changed(self):
        out = self.output_combo.currentData()
        self.format_combo.setEnabled(out == "images")
        self.dpi_combo.setEnabled(out == "images")
        self.page_image_check.setEnabled(out == "pptx")

    def values(self) -> dict:
        return {
            "group": self.group_combo.currentText(),
            "by": "keyword" if self.by_keyword.isChecked() else "group",
            "keyword": self.keyword_combo.currentText().strip() or None,
            "header": self.header_edit.text().strip(),
            "output": self.output_combo.currentData(),
            "format": self.format_combo.currentData(),
            "dpi": self.dpi_combo.currentData(),
            "page_image": self.page_image_check.isChecked(),
        }


# ---------------------------------------------------------------- scene items
class PageItem(QGraphicsItem):
    """The white page with its margin guide and optional grid."""

    def __init__(self, editor: "LayoutEditorWindow"):
        super().__init__()
        self.editor = editor
        self.setZValue(-1e6)

    def boundingRect(self) -> QRectF:
        w, h = self.editor.page_size()
        return QRectF(-4, -4, w + 10, h + 10)

    def paint(self, painter: QPainter, option, widget=None):
        w, h = self.editor.page_size()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 50))
        painter.drawRect(QRectF(4, 4, w, h))
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#9aa0a6"), 0))
        painter.drawRect(QRectF(0, 0, w, h))
        if self.editor.show_grid:
            step = max(2.0, self.editor.grid_step)
            painter.setPen(QPen(QColor("#eef0f3"), 0))
            x = step
            while x < w:
                painter.drawLine(QPointF(x, 0), QPointF(x, h))
                x += step
            y = step
            while y < h:
                painter.drawLine(QPointF(0, y), QPointF(w, y))
                y += step
        m = self.editor.page_margin()
        if m > 0:
            pen = QPen(QColor("#c9d7ef"), 0, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(m, m, max(0.0, w - 2 * m), max(0.0, h - 2 * m)))


HANDLE_NAMES = ["tl", "t", "tr", "r", "br", "b", "bl", "l"]
HANDLE_CURSORS = {
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
}


class PlotGraphicsItem(QGraphicsItem):
    """A plot of one population (plus overlays), with caption and title."""

    def __init__(self, editor: "LayoutEditorWindow", item: LayoutItem):
        super().__init__()
        self.editor = editor
        self.item = item
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setPos(item.x, item.y)
        self._resize_handle: str | None = None
        self._resize_start: tuple[QRectF, QPointF] | None = None
        self.drop_highlight = False

    # -- geometry
    def boundingRect(self) -> QRectF:
        return QRectF(-HANDLE, -HANDLE, self.item.w + 2 * HANDLE, self.item.h + 2 * HANDLE)

    def box(self) -> QRectF:
        return QRectF(0, 0, self.item.w, self.item.h)

    def sync_from_model(self):
        self.prepareGeometryChange()
        self.setPos(self.item.x, self.item.y)
        self.update()

    def handle_rects(self) -> dict[str, QRectF]:
        w, h = self.item.w, self.item.h
        pts = {
            "tl": (0, 0), "t": (w / 2, 0), "tr": (w, 0), "r": (w, h / 2),
            "br": (w, h), "b": (w / 2, h), "bl": (0, h), "l": (0, h / 2),
        }
        return {k: QRectF(x - HANDLE / 2, y - HANDLE / 2, HANDLE, HANDLE) for k, (x, y) in pts.items()}

    def handle_at(self, pos: QPointF) -> str | None:
        if not self.isSelected():
            return None
        for k, r in self.handle_rects().items():
            if r.adjusted(-2, -2, 2, 2).contains(pos):
                return k
        return None

    # -- painting
    def paint(self, painter: QPainter, option, widget=None):
        self.editor.paint_item(painter, self.item, None, SCENE_SCALE, allow_stale=self._resize_handle is not None)
        if self.drop_highlight:
            painter.setPen(QPen(QColor("#2f6fd6"), 2.0))
            painter.setBrush(QColor(47, 111, 214, 30))
            painter.drawRect(self.box())
        if self.isSelected():
            painter.setPen(QPen(QColor("#2f6fd6"), 0, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.box())
            painter.setPen(QPen(QColor("#2f6fd6"), 0))
            painter.setBrush(QColor("#ffffff"))
            for r in self.handle_rects().values():
                painter.drawRect(r)

    # -- interaction
    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.editor.snap_to_grid:
            p = QPointF(value)
            return QPointF(self.editor.snap(p.x()), self.editor.snap(p.y()))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            p = self.pos()
            if (p.x(), p.y()) != (self.item.x, self.item.y):
                self.item.x, self.item.y = p.x(), p.y()
                self.editor.schedule_notify((self.item.id, "geometry"))
        return super().itemChange(change, value)

    def hoverMoveEvent(self, event):
        h = self.handle_at(event.pos())
        if h:
            self.setCursor(QCursor(HANDLE_CURSORS[h]))
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        h = self.handle_at(event.pos()) if event.button() == Qt.MouseButton.LeftButton else None
        if h:
            self._resize_handle = h
            self._resize_start = (QRectF(self.item.x, self.item.y, self.item.w, self.item.h), event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_handle and self._resize_start:
            keep = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.resize_to(self._resize_handle, self._resize_start[0], event.scenePos() - self._resize_start[1], keep)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resize_handle:
            self._resize_handle = None
            self._resize_start = None
            self.update()
            self.editor.schedule_notify((self.item.id, "geometry"))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resize_to(self, handle: str, start: QRectF, delta: QPointF, keep_aspect: bool = False):
        left, top, right, bottom = start.left(), start.top(), start.right(), start.bottom()
        dx, dy = delta.x(), delta.y()
        if "l" in handle:
            left += dx
        if "r" in handle:
            right += dx
        if "t" in handle:
            top += dy
        if "b" in handle:
            bottom += dy
        if self.editor.snap_to_grid:
            left, right, top, bottom = (self.editor.snap(v) for v in (left, right, top, bottom))
        w = max(MIN_SIZE, right - left)
        h = max(MIN_SIZE, bottom - top)
        if keep_aspect and start.height() > 0:
            ratio = start.width() / start.height()
            if handle in ("t", "b"):
                w = h * ratio
            elif handle in ("l", "r"):
                h = w / ratio
            elif w / h > ratio:
                h = w / ratio
            else:
                w = h * ratio
        x = start.right() - w if "l" in handle else start.left()
        y = start.bottom() - h if "t" in handle else start.top()
        self.prepareGeometryChange()
        self.item.w, self.item.h = w, h
        self.item.x, self.item.y = x, y
        self.setPos(x, y)
        self.update()

    def mouseDoubleClickEvent(self, event):
        self.editor.edit_item(self.item)
        event.accept()

    def contextMenuEvent(self, event):
        if not self.isSelected():
            self.scene().clearSelection()
            self.setSelected(True)
        self.editor.item_context_menu(self, event.screenPos())
        event.accept()


class TextGraphicsItem(QGraphicsTextItem):
    """Free text on the page; double-click to edit inline. Tokens such as
    {$FIL} or {Median FSC-A} are shown resolved (for the bound / first plot's
    sample) and as typed while editing."""

    def __init__(self, editor: "LayoutEditorWindow", item: LayoutItem):
        super().__init__(editor.display_text(item))
        self.editor = editor
        self.item = item
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.apply_font()
        self.setPos(item.x, item.y)

    def apply_font(self):
        fmt = text_format(self.item)
        self.setFont(text_font(self.item))
        self.setDefaultTextColor(QColor(fmt["color"]))
        self.setTextWidth(self.item.w if self.item.w > 0 else -1)
        doc = self.document()
        opt = doc.defaultTextOption()
        opt.setAlignment(TEXT_ALIGN[fmt["align"]])
        doc.setDefaultTextOption(opt)
        self.item.h = self.boundingRect().height()

    def shown_text(self) -> str:
        return self.item.text if self.editing else self.editor.display_text(self.item)

    def sync_from_model(self):
        shown = self.shown_text()
        if self.toPlainText() != shown:
            self.setPlainText(shown)
        self.apply_font()
        self.setPos(self.item.x, self.item.y)

    @property
    def editing(self) -> bool:
        return self.textInteractionFlags() != Qt.TextInteractionFlag.NoTextInteraction

    def start_editing(self):
        if self.toPlainText() != self.item.text:
            self.setPlainText(self.item.text)  # edit the template, not its values
            self.apply_font()
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        cursor = self.textCursor()
        cursor.select(cursor.SelectionType.Document)
        self.setTextCursor(cursor)

    def finish_editing(self):
        if getattr(self, "_finishing", False):
            return  # dropping the focusable flag below sends a focus-out that lands here again
        self._finishing = True
        try:
            self._finish_editing()
        finally:
            self._finishing = False

    def _finish_editing(self):
        text = self.toPlainText()  # the edited template, before anything else touches the document
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        old_h = self.item.h
        changed = text != self.item.text
        if changed:
            self.item.text = text
        shown = self.editor.display_text(self.item)
        if shown != text:
            self.setPlainText(shown)
            self.apply_font()
        h = self.boundingRect().height()
        if changed or h != old_h:
            self.item.h = h
            self.editor.schedule_notify((self.item.id, "text"))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.editor.snap_to_grid:
            p = QPointF(value)
            return QPointF(self.editor.snap(p.x()), self.editor.snap(p.y()))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            p = self.pos()
            if (p.x(), p.y()) != (self.item.x, self.item.y):
                self.item.x, self.item.y = p.x(), p.y()
                self.editor.schedule_notify((self.item.id, "geometry"))
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if not self.editing:
            self.start_editing()
        super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self.editing:
            self.finish_editing()
            self.editor.end_edit_burst()  # the text edit is one undo step of its own

    def keyPressEvent(self, event):
        if self.editing and event.key() == Qt.Key.Key_Escape:
            self.clearFocus()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        if not self.isSelected():
            self.scene().clearSelection()
            self.setSelected(True)
        self.editor.item_context_menu(self, event.screenPos())
        event.accept()


# ---------------------------------------------------------------------- view
ARROW_KEYS = (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down)


class LayoutView(QGraphicsView):
    def __init__(self, editor: "LayoutEditorWindow", scene: QGraphicsScene):
        super().__init__(scene)
        self.editor = editor
        self.setAcceptDrops(True)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setBackgroundBrush(QColor("#8a8f96"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._highlight: PlotGraphicsItem | None = None

    def _set_highlight(self, item):
        if item is self._highlight:
            return
        if self._highlight is not None:
            self._highlight.drop_highlight = False
            self._highlight.update()
        self._highlight = item
        if item is not None:
            item.drop_highlight = True
            item.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(POPULATION_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(POPULATION_MIME):
            pos = self.mapToScene(event.position().toPoint())
            self._set_highlight(self.editor.plot_item_at(pos))
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._set_highlight(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._set_highlight(None)
        mime = event.mimeData()
        if not mime.hasFormat(POPULATION_MIME):
            event.ignore()
            return
        pos = self.mapToScene(event.position().toPoint())
        self.editor.drop_mime(mime, pos)
        event.acceptProposedAction()

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None or item is self.editor.page_item:
            self.editor.page_context_menu(event.globalPos(), self.mapToScene(event.pos()))
            event.accept()
            return
        super().contextMenuEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.editor.zoom_by(factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        self.editor.end_edit_burst()  # a new interaction starts a new undo step
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.editor.end_edit_burst()  # the drag / resize is over: one undo step

    def keyReleaseEvent(self, event):
        super().keyReleaseEvent(event)
        if not event.isAutoRepeat() and event.key() in ARROW_KEYS:
            self.editor.end_edit_burst()

    STANDARD_KEYS = (
        (QKeySequence.StandardKey.Undo, "undo"),
        (QKeySequence.StandardKey.Redo, "redo"),
        (QKeySequence.StandardKey.Copy, "copy"),
        (QKeySequence.StandardKey.Cut, "cut"),
        (QKeySequence.StandardKey.Paste, "paste"),
        (QKeySequence.StandardKey.SelectAll, "select_all"),
        (QKeySequence.StandardKey.Close, "close"),
    )

    def keyPressEvent(self, event):
        focus = self.scene().focusItem()
        editing = isinstance(focus, TextGraphicsItem) and focus.editing
        if not editing:
            # the editor window's actions carry these shortcuts; this covers a
            # key press that reaches the view without the window being active
            for key, name in self.STANDARD_KEYS:
                if event.matches(key):
                    self.editor.trigger_command(name)
                    event.accept()
                    return
        if not editing and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.editor.delete_selected()
            event.accept()
            return
        if not editing and event.key() in ARROW_KEYS:
            if not event.isAutoRepeat():
                self.editor.end_edit_burst()  # each press is a step; a held key is one
            step = self.editor.grid_step if self.editor.snap_to_grid else (10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0)
            dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(event.key(), 0.0)
            dy = {Qt.Key.Key_Up: -step, Qt.Key.Key_Down: step}.get(event.key(), 0.0)
            for it in self.scene().selectedItems():
                it.moveBy(dx, dy)
            event.accept()
            return
        super().keyPressEvent(event)


# -------------------------------------------------------------------- window
class LayoutEditorWindow(QMainWindow):
    # custom page objects registered by other packages:
    # kind -> {"graphics_factory", "painter", "dialog"}
    _item_kinds: dict[str, dict] = {}
    defaults = LayoutDefaults()

    @classmethod
    def register_item_kind(cls, kind: str, graphics_factory=None, painter=None, dialog=None):
        """Register a page object kind (LayoutItem.kind). `painter(editor,
        qpainter, item, sample_override, scale)` paints it in item-local
        points (scene, export, print, batch); `graphics_factory(editor, item)`
        may return its own QGraphicsItem (default: a movable / resizable box
        that calls `painter`); `dialog(editor, item, parent) -> bool` edits it
        (double-click / Edit…) and returns whether it changed."""
        if kind in ("plot", "text"):
            raise ValueError(f"{kind!r} is a built-in item kind")
        cls._item_kinds[kind] = {"graphics_factory": graphics_factory, "painter": painter, "dialog": dialog}

    @classmethod
    def register_text_expander(cls, fn):
        """Add a resolver for custom text tokens, see core.layout_text."""
        layout_text.register_text_expander(fn)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Layout Editor")
        self.setWindowIcon(icon("layout-editor"))
        self.resize(1100, 800)
        self.cache = PlotImageCache()
        self.render_legends: dict[str, list] = {}
        d = self.defaults.as_dict()
        self.show_grid = bool(d["grid"])
        self.snap_to_grid = bool(d["grid"])
        self.grid_step = float(d["grid_step"]) if float(d["grid_step"]) > 0 else GRID
        self._layout: Layout | None = None
        self._suppress = False
        self._scene_items: dict[str, QGraphicsItem] = {}
        self._notify_timer = QTimer(self)
        self._notify_timer.setSingleShot(True)
        self._notify_timer.setInterval(150)
        self._notify_timer.timeout.connect(lambda: self._emit_layouts(coalesce=True))
        self._pending_keys: set = set()  # what the pending notification covers
        self.undo_recorder = SectionUndoRecorder(controller, "layouts", "edit layout")
        self._last_copy: dict | None = None  # items copied by this editor (clipboard fallback)

        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor("#8a8f96"))
        self.scene.selectionChanged.connect(lambda: self._update_text_actions() if hasattr(self, "acts") else None)
        self.view = LayoutView(self, self.scene)
        self.page_item = PageItem(self)
        self.scene.addItem(self.page_item)

        self.layout_list = QListWidget()
        self.layout_list.setMinimumWidth(150)
        self.layout_list.currentRowChanged.connect(self._list_row_changed)
        self.layout_list.itemDoubleClicked.connect(lambda _: self.rename_layout())
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.addWidget(QLabel("Layouts"))
        lv.addWidget(self.layout_list, 1)
        lv.addWidget(QLabel("Batch group:"))
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(120)
        self.group_combo.setToolTip("The group whose samples a batch iterates over (one page per sample)")
        self.group_combo.currentTextChanged.connect(self._batch_group_changed)
        lv.addWidget(self.group_combo)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([180, 920])
        self._build_actions()
        self._build_menus()
        self._build_ribbon()
        central = QWidget()
        cv = QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.ribbon)
        cv.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.statusBar()

        c = controller
        c.gatingChanged.connect(lambda *_: self.refresh(rebuild=False))
        c.scalesChanged.connect(lambda *_: self.refresh(rebuild=False))
        c.compensationChanged.connect(lambda *_: self.refresh(rebuild=False))
        c.preferencesChanged.connect(lambda *_: self.refresh(rebuild=False))
        c.samplesChanged.connect(lambda *_: self.refresh(rebuild=False))
        c.layoutsChanged.connect(self._on_layouts_changed)
        c.workspaceReplaced.connect(self._on_workspace_replaced)
        c.groupsChanged.connect(self._fill_groups)
        self._sync_list()
        self._fill_groups()

    # ------------------------------------------------------------ actions
    def _new_action(self, name: str, icon_name: str | None, text: str, slot, shortcut=None, tip: str | None = None, checkable: bool = False) -> QAction:
        a = QAction(icon(icon_name) if icon_name else QIcon(), text, self)
        a.setToolTip(tip or text.replace("…", ""))
        if shortcut is not None:
            if isinstance(shortcut, QKeySequence.StandardKey):
                a.setShortcuts(shortcut)  # every platform binding
            elif isinstance(shortcut, (list, tuple)):
                seqs: list[QKeySequence] = []
                for k in shortcut:
                    for seq in QKeySequence.keyBindings(k) if isinstance(k, QKeySequence.StandardKey) else [QKeySequence(k)]:
                        if not seq.isEmpty() and seq not in seqs:
                            seqs.append(seq)
                a.setShortcuts(seqs)
            else:
                a.setShortcut(QKeySequence(shortcut))
            # scoped to the editor window (an unparented top-level window)
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        a.setCheckable(checkable)
        if checkable:
            a.toggled.connect(lambda on: slot(on))
        else:
            a.triggered.connect(lambda _=False: slot())
        self.addAction(a)
        self.acts[name] = a
        return a

    def _build_actions(self):
        self.acts: dict[str, QAction] = {}
        A = self._new_action
        SK = QKeySequence.StandardKey
        # layouts
        A("new_layout", "new", "New Layout", self.new_layout)
        A("rename_layout", "rename", "Rename Layout…", self.rename_layout)
        A("duplicate_layout", "copy", "Duplicate Layout", self.duplicate_layout)
        A("delete_layout", "delete", "Delete Layout", self.delete_layout)
        # edit
        A("undo", "undo", "Undo", self.undo, SK.Undo)
        A("redo", "redo", "Redo", self.redo, [SK.Redo, "Ctrl+Y", "Ctrl+Shift+Z"])
        A("cut", "cut", "Cut", self.cut_selected, SK.Cut)
        A("copy", "copy", "Copy", self.copy_selected, SK.Copy, "Copy the selected objects (or the page) — also as an image for other applications")
        A("paste", "paste", "Paste", self.paste_items, SK.Paste)
        A("select_all", "check", "Select All", self.select_all, SK.SelectAll)
        A("duplicate", "plus", "Duplicate", self.duplicate_selected, "Ctrl+D")
        A("delete", "minus", "Delete", self.delete_selected)
        A("copy_page_image", "image", "Copy Page as Image", self.copy_page_image, "Ctrl+Shift+C")
        A("close", None, "Close Window", self.close, [SK.Close, "Ctrl+W"])
        # insert
        A("add_text", "text", "Add Text", self.add_text)
        A("insert_keyword", "keyword", "Insert Keyword / Field…", self.insert_keyword_dialog)
        A("insert_statistic", "statistic", "Insert Statistic…", self.insert_statistic_dialog)
        A("overlay", "overlay", "Overlay…", self.overlay_dialog, tip="Add overlays to the selected plot from a chooser")
        A("overlay_group", "group", "Overlay All Samples of Group", self.overlay_group_selected, tip="Overlay the plot's population of every sample in the batch group")
        A("edit_item", "rename", "Edit…", self.edit_selected, tip="Layout Plot Options / edit the selected object")
        A("open_graph", "graph", "Open Graph Window", self.open_graph_selected, tip="Open the selected plot's population in a graph window to adjust gates")
        # arrange
        A("align_left", "left", "Align Left", self.align_left)
        A("align_right", "right", "Align Right", self.align_right)
        A("align_top", "up", "Align Top", self.align_top)
        A("align_bottom", "down", "Align Bottom", self.align_bottom)
        A("align_hcenter", "left", "Align Centres Horizontally", self.align_hcenter, tip="Line up the horizontal centres (one above the other)")
        A("align_vcenter", "up", "Align Centres Vertically", self.align_vcenter, tip="Line up the vertical centres (side by side)")
        A("distribute_h", "right", "Distribute Horizontally", self.distribute_horizontally, tip="Equal horizontal gaps between the selected objects")
        A("distribute_v", "down", "Distribute Vertically", self.distribute_vertically, tip="Equal vertical gaps between the selected objects")
        A("same_width", "right", "Same Width", self.same_width)
        A("same_height", "down", "Same Height", self.same_height)
        A("same_size", "check", "Same Size", self.same_size)
        A("bring_front", "up", "Bring to Front", lambda: self.bring_to_front(self.selected_layout_items(), True), "Ctrl+Shift+]")
        A("send_back", "down", "Send to Back", lambda: self.bring_to_front(self.selected_layout_items(), False), "Ctrl+Shift+[")
        for n in (2, 3, 4):
            A(f"grid_{n}", "grid", f"Arrange in {n} Columns", lambda n=n: self.arrange_grid(n))
        # text
        A("bold", None, "Bold", lambda on: self.toggle_text_format("bold"), "Ctrl+B", checkable=True)
        A("italic", None, "Italic", lambda on: self.toggle_text_format("italic"), "Ctrl+I", checkable=True)
        A("underline", None, "Underline", lambda on: self.toggle_text_format("underline"), "Ctrl+U", checkable=True)
        for key, label in (("left", "Align Text Left"), ("center", "Centre Text"), ("right", "Align Text Right")):
            A(f"text_{key}", {"left": "left", "center": "up", "right": "right"}[key], label, lambda key=key: self.set_text_format(align=key))
        A("text_color", None, "Text Colour…", self.text_color_dialog)
        A("text_font", None, "Font…", self.text_font_dialog)
        # page / view
        A("page_setup", "page", "Page Setup…", self.page_setup)
        A("zoom_in", "zoom", "Zoom In", lambda: self.zoom_by(1.25), SK.ZoomIn)
        A("zoom_out", "minus", "Zoom Out", lambda: self.zoom_by(0.8), SK.ZoomOut)
        A("fit_page", "page", "Fit Page", self.fit_page)
        self.grid_action = A("grid", "grid", "Show Grid / Snap to Grid", self.set_grid, checkable=True)
        self.grid_action.blockSignals(True)
        self.grid_action.setChecked(self.show_grid)
        self.grid_action.blockSignals(False)
        A("layout_defaults", "preferences", "Save as Defaults for New Layouts", self.save_layout_defaults)
        # export / batch
        A("export", "export", "Export…", self.export_dialog)
        A("export_pptx", "export", "Export to PowerPoint…", self.export_pptx_dialog)
        A("print", "print", "Print…", self.print_layout, SK.Print)
        A("batch", "play", "Batch…", self.batch_dialog, tip="Iterate the layout over the samples of a group: PDF, PowerPoint, image files, printer or new layouts")
        A("batch_pdf", "play", "Create Batch PDF…", self.batch_pdf_dialog)
        A("batch_pptx", "play", "Create Batch PowerPoint…", lambda: self.batch_dialog("pptx"))
        A("batch_images", "image", "Batch Image Files per Sample…", lambda: self.batch_dialog("images"))

    def _build_menus(self):
        mb = self.menuBar()
        a = self.acts
        m = mb.addMenu("&File")
        for k in ("new_layout", "duplicate_layout", "rename_layout", "delete_layout", None, "page_setup", "layout_defaults", None,
                  "export", "export_pptx", "copy_page_image", "print", None, "batch", "batch_pdf", "batch_pptx", "batch_images", None, "close"):
            m.addSeparator() if k is None else m.addAction(a[k])
        m = mb.addMenu("&Edit")
        self.edit_menu = m
        for k in ("undo", "redo", None, "cut", "copy", "paste", "duplicate", "delete", None, "select_all"):
            m.addSeparator() if k is None else m.addAction(a[k])
        m.aboutToShow.connect(self._update_undo_actions)
        m = mb.addMenu("&Object")
        for k in ("add_text", "insert_keyword", "insert_statistic", None, "edit_item", "overlay", "overlay_group", "open_graph"):
            m.addSeparator() if k is None else m.addAction(a[k])
        self.arrange_menu = self._arrange_menu(mb.addMenu("&Arrange"))
        m = mb.addMenu("&Text")
        for k in ("bold", "italic", "underline", None, "text_left", "text_center", "text_right", None, "text_font", "text_color"):
            m.addSeparator() if k is None else m.addAction(a[k])
        m.addMenu(self._font_size_menu(m))
        m.aboutToShow.connect(self._update_text_actions)
        m = mb.addMenu("&View")
        for k in ("zoom_in", "zoom_out", "fit_page", None, "grid"):
            m.addSeparator() if k is None else m.addAction(a[k])

    def _arrange_menu(self, m: QMenu) -> QMenu:
        a = self.acts
        align = m.addMenu(icon("left"), "Align")
        for k in ("align_left", "align_hcenter", "align_right", None, "align_top", "align_vcenter", "align_bottom"):
            align.addSeparator() if k is None else align.addAction(a[k])
        dist = m.addMenu(icon("right"), "Distribute")
        dist.addAction(a["distribute_h"])
        dist.addAction(a["distribute_v"])
        size = m.addMenu(icon("check"), "Size")
        for k in ("same_width", "same_height", "same_size"):
            size.addAction(a[k])
        m.addSeparator()
        m.addAction(a["bring_front"])
        m.addAction(a["send_back"])
        m.addSeparator()
        grid = m.addMenu(icon("grid"), "Arrange Plots in a Grid")
        for n in (2, 3, 4):
            grid.addAction(a[f"grid_{n}"])
        return m

    def _font_size_menu(self, parent) -> QMenu:
        sizes = QMenu("Font Size", parent)
        for sz in FONT_SIZES:
            sizes.addAction(f"{sz} pt", lambda sz=sz: self.set_font_size(self.selected_layout_items(), sz))
        return sizes

    def _build_ribbon(self):
        a = self.acts
        self.ribbon = Ribbon()
        page = self.ribbon.add_page("Layout")
        band = page.add_band("Layouts")
        band.add_large(a["new_layout"])
        band.add_small(a["duplicate_layout"])
        band.add_small(a["rename_layout"])
        band.add_small(a["delete_layout"])
        band = page.add_band("Edit")
        band.add_large(a["undo"])
        band.add_large(a["redo"])
        for k in ("cut", "copy", "paste", "select_all", "duplicate", "delete"):
            band.add_small(a[k])
        band = page.add_band("Insert")
        band.add_large(a["add_text"])
        band.add_small(a["insert_keyword"])
        band.add_small(a["insert_statistic"])
        band.add_small(a["overlay"])
        band = page.add_band("Plot")
        band.add_large(a["edit_item"])
        band.add_small(a["open_graph"])
        band.add_small(a["overlay_group"])
        band = page.add_band("Page")
        band.add_large(a["page_setup"])
        for k in ("grid", "zoom_in", "zoom_out", "fit_page"):
            band.add_small(a[k])

        page = self.ribbon.add_page("Arrange")
        band = page.add_band("Align")
        for k in ("align_left", "align_hcenter", "align_right", "align_top", "align_vcenter", "align_bottom"):
            band.add_small(a[k])
        band = page.add_band("Distribute")
        band.add_small(a["distribute_h"])
        band.add_small(a["distribute_v"])
        band = page.add_band("Size")
        for k in ("same_width", "same_height", "same_size"):
            band.add_small(a[k])
        band = page.add_band("Order")
        band.add_small(a["bring_front"])
        band.add_small(a["send_back"])
        band = page.add_band("Grid")
        grid_menu = QMenu(self)
        for n in (2, 3, 4):
            grid_menu.addAction(a[f"grid_{n}"])
        band.add_large(a["grid_2"], grid_menu)

        page = self.ribbon.add_page("Text")
        band = page.add_band("Font")
        for k in ("bold", "italic", "underline", "text_font", "text_color"):
            band.add_small(a[k])
        band = page.add_band("Paragraph")
        for k in ("text_left", "text_center", "text_right"):
            band.add_small(a[k])
        band = page.add_band("Fields")
        band.add_small(a["insert_keyword"])
        band.add_small(a["insert_statistic"])

        page = self.ribbon.add_page("Export")
        band = page.add_band("Export")
        band.add_large(a["export"])
        band.add_large(a["export_pptx"])
        band.add_small(a["copy_page_image"])
        band.add_small(a["print"])
        band = page.add_band("Batch")
        band.add_large(a["batch"])
        for k in ("batch_pdf", "batch_pptx", "batch_images"):
            band.add_small(a[k])

    def trigger_command(self, name: str):
        """Run an editor command by action name (keyboard fallbacks)."""
        act = self.acts.get(name)
        if act is not None and act.isEnabled():
            act.trigger()

    def _update_undo_actions(self):
        c = self.controller
        self.acts["undo"].setText(f"Undo {c.undo_label()}".strip() if c.can_undo else "Undo")
        self.acts["redo"].setText("Redo")

    def _update_text_actions(self):
        try:
            texts = [i for i in self.selected_layout_items() if i.kind == "text"]
        except RuntimeError:  # the scene is being destroyed
            return
        fmt = text_format(texts[0]) if texts else {"bold": False, "italic": False, "underline": False}
        for k in ("bold", "italic", "underline"):
            act = self.acts[k]
            act.blockSignals(True)
            act.setChecked(bool(fmt[k]))
            act.blockSignals(False)

    # ----------------------------------------------------------- model glue
    @property
    def workspace(self):
        return self.controller.workspace

    def page_size(self, layout: Layout | None = None) -> tuple[float, float]:
        layout = layout or self._layout
        return layout.page_size if layout is not None else PAGE_SIZES["A4"]

    def page_margin(self, layout: Layout | None = None) -> float:
        layout = layout or self._layout
        return layout.margin if layout is not None else DEFAULT_MARGIN

    def snap(self, v: float) -> float:
        return _snap(v, self.grid_step)

    def new_layout_object(self, name: str) -> Layout:
        """A Layout with the user's defaults for new layouts."""
        d = self.defaults.as_dict()
        page = d["page"] if (d["page"] in PAGE_SIZES) else "A4"
        lay = Layout(name, page=page, landscape=bool(d["landscape"]))
        if abs(float(d["margin"]) - DEFAULT_MARGIN) > 1e-9:
            lay.props["margin"] = float(d["margin"])
        return lay

    def default_plot_size(self) -> float:
        try:
            return max(MIN_SIZE, float(self.defaults.get("plot_size")))
        except (TypeError, ValueError):
            return DEFAULT_PLOT

    def schedule_notify(self, key=None):
        """Announce an in-place edit shortly (one notification per timer
        period while dragging). `key` names what was edited, e.g. (item id,
        "geometry"): the notifications of one interaction with the same keys
        are one undo step, which end_edit_burst() closes."""
        self._pending_keys.add(key)
        self._notify_timer.start()

    def flush_notify(self):
        if self._notify_timer.isActive():
            self._notify_timer.stop()
            self._emit_layouts(coalesce=True)

    def end_edit_burst(self):
        """The interaction is over (mouse released, text edit finished):
        announce what is pending and make the next edit an undo step of its
        own."""
        self.flush_notify()
        self.undo_recorder.end_burst()

    def _notify_now(self):
        self._notify_timer.stop()
        self._emit_layouts()

    def _emit_layouts(self, coalesce: bool = False):
        # one undo step per edit; the timer-driven notifications of one drag
        # (same items) merge into one
        key = frozenset(self._pending_keys) if coalesce else None
        self._pending_keys.clear()
        self.undo_recorder.record(coalesce, key)
        self._suppress = True
        try:
            self.controller.notify_layouts()
        finally:
            self._suppress = False

    def _on_layouts_changed(self):
        if self._suppress:
            self._sync_list()
            return
        # undo / redo or a change made elsewhere: the model is the truth now
        self._notify_timer.stop()
        self._pending_keys.clear()
        self.undo_recorder.reset()
        self.refresh()

    def _on_workspace_replaced(self):
        self._notify_timer.stop()
        self._pending_keys.clear()
        self.undo_recorder.reset()
        self._layout = None
        self.cache.invalidate()
        self._fill_groups()
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_layout()
        QTimer.singleShot(0, self.fit_page)

    def closeEvent(self, event):
        self.flush_notify()
        focus = self.scene.focusItem()
        if isinstance(focus, TextGraphicsItem) and focus.editing:
            focus.finish_editing()
            self.flush_notify()
        self.end_edit_burst()
        event.ignore()
        self.hide()

    def _ensure_layout(self) -> Layout:
        ws = self.workspace
        if not ws.layouts:
            ws.layouts.append(self.new_layout_object("Layout"))
            self._layout = ws.layouts[0]
            self.undo_recorder.reset()  # the default layout is the baseline, not an undoable edit
            self._notify_now()
            self._sync_list()
            self._rebuild_scene()
        if self._layout is None or self._layout not in ws.layouts:
            self.set_current_layout(ws.layouts[0])
        return self._layout

    def current_layout(self) -> Layout:
        return self._ensure_layout()

    def set_current_layout(self, layout: Layout):
        self.flush_notify()
        self._layout = layout
        self._sync_list()
        self._fill_groups()
        self._rebuild_scene()

    def _sync_list(self):
        layouts = self.workspace.layouts
        self.layout_list.blockSignals(True)
        self.layout_list.clear()
        for lay in layouts:
            li = QListWidgetItem(icon("layout-editor"), lay.name)
            self.layout_list.addItem(li)
        if self._layout in layouts:
            self.layout_list.setCurrentRow(layouts.index(self._layout))
        self.layout_list.blockSignals(False)

    def _list_row_changed(self, row: int):
        layouts = self.workspace.layouts
        if 0 <= row < len(layouts) and layouts[row] is not self._layout:
            self.set_current_layout(layouts[row])

    def _fill_groups(self):
        combo = self.group_combo
        current = None
        if self._layout is not None and self._layout.batch_group:
            current = self._layout.batch_group
        elif self.controller.current_group is not None:
            current = self.controller.current_group.name
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([g.name for g in self.workspace.groups])
        if current:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _batch_group_changed(self, name: str):
        if self._layout is not None and name and self._layout.batch_group != name:
            self._layout.batch_group = name
            self.schedule_notify((self._layout.id, "batch_group"))

    # --------------------------------------------------------------- scene
    def _rebuild_scene(self):
        for it in list(self._scene_items.values()):
            self.scene.removeItem(it)
        self._scene_items = {}
        self.page_item.prepareGeometryChange()
        layout = self._layout
        if layout is not None:
            for k, item in enumerate(layout.items):
                self._add_scene_item(item, k)
            w, h = layout.page_size
            self.scene.setSceneRect(QRectF(-60, -60, w + 120, h + 120))
        self.page_item.update()

    def _add_scene_item(self, item: LayoutItem, z: int | None = None) -> QGraphicsItem:
        gi = None
        if item.kind == "text":
            gi = TextGraphicsItem(self, item)
        elif item.kind in self._item_kinds and self._item_kinds[item.kind].get("graphics_factory") is not None:
            gi = self._item_kinds[item.kind]["graphics_factory"](self, item)
        if gi is None:
            gi = PlotGraphicsItem(self, item)  # plots and registered kinds without a factory
        gi.setZValue(z if z is not None else len(self._scene_items))
        self.scene.addItem(gi)
        self._scene_items[item.id] = gi
        return gi

    def scene_item(self, item: LayoutItem) -> QGraphicsItem | None:
        return self._scene_items.get(item.id)

    def _restack(self):
        if self._layout is None:
            return
        for k, item in enumerate(self._layout.items):
            gi = self._scene_items.get(item.id)
            if gi is not None:
                gi.setZValue(k)

    def selected_layout_items(self) -> list[LayoutItem]:
        sel = [gi.item for gi in self.scene.selectedItems() if isinstance(getattr(gi, "item", None), LayoutItem) and self._scene_items.get(gi.item.id) is gi]
        order = {it.id: k for k, it in enumerate(self._layout.items)} if self._layout else {}
        return sorted(sel, key=lambda i: order.get(i.id, 0))

    def plot_item_at(self, pos: QPointF) -> PlotGraphicsItem | None:
        for gi in self.scene.items(pos):
            if isinstance(gi, PlotGraphicsItem) and gi.item.kind == "plot" and gi.box().contains(gi.mapFromScene(pos)):
                return gi
        return None

    def refresh(self, rebuild: bool = True):
        """Re-render every plot. With rebuild, also re-read the layout list and
        recreate the scene items from the model (after external changes)."""
        self.cache.invalidate()
        if rebuild:
            layouts = self.workspace.layouts
            if self._layout not in layouts:
                self._layout = layouts[0] if layouts else None
            self._sync_list()
            self._rebuild_scene()
        else:
            for gi in self._scene_items.values():
                if isinstance(gi, TextGraphicsItem) and not gi.editing:
                    gi.sync_from_model()  # statistic / keyword tokens may show new values
                gi.update()
        self.page_item.update()

    # ------------------------------------------------------------ resolving
    def resolve(self, item: LayoutItem, sample_override: Sample | None = None) -> tuple[Sample | None, Population | None]:
        sample = sample_override
        if sample is None and item.sample_id is not None:
            sample = self.workspace.sample_by_id(item.sample_id)
        if sample is None:
            return None, None
        pop = sample.root.find(item.pop_path) if item.pop_path else sample.root
        return sample, pop

    def resolve_overlays(self, item: LayoutItem, sample_override: Sample | None = None) -> list[tuple[Sample, Population]]:
        out = []
        for ov in item.overlays:
            sid = ov.get("sample_id")
            s = sample_override if (sample_override is not None and sid == item.sample_id) else self.workspace.sample_by_id(sid)
            if s is None:
                continue
            path = ov.get("pop_path") or ""
            p = s.root.find(path) if path else s.root
            if p is not None:
                out.append((s, p))
        return out

    def render_options(self, graph: GraphSettings, scale: float, item: LayoutItem | None = None) -> RenderOptions:
        f = 72.0 * scale / IMAGE_DPI
        label_pt, number_pt = 7.5, 6.0
        if item is not None:
            try:
                label_pt = float(item.props.get("axis_label_size") or label_pt)
                number_pt = float(item.props.get("axis_number_size") or number_pt)
            except (TypeError, ValueError):
                pass
        return RenderOptions.from_prefs(
            self.controller.preferences,
            graph,
            title=None,
            show_axes=True,
            label_font_size=max(4, int(round(label_pt * f))),
            number_font_size=max(4, int(round(number_pt * f))),
        )

    def plot_image(self, item: LayoutItem, sample: Sample, pop: Population, overlays, w_pt: float, h_pt: float, scale: float, allow_stale: bool = False) -> QImage:
        w_px = max(60, int(math.ceil(w_pt * scale)))
        h_px = max(60, int(math.ceil(h_pt * scale)))
        ident = (
            self.cache.version,
            id(sample),
            sample.id,
            item.pop_path,
            json.dumps(item.graph.to_dict(), sort_keys=True, default=str),
            tuple((id(s), p.path) for s, p in overlays),
            scale,
            item.props.get("axis_label_size"),
            item.props.get("axis_number_size"),
        )
        key = ident + (w_px, h_px)
        img = self.cache.get(key)
        if img is not None:
            return img
        if allow_stale:
            stale = getattr(item, "_last_image", None)
            if stale is not None and stale[0] == ident:
                return stale[1]
        graph = item.graph.copy()
        if not graph.x:
            graph.x, y = sample.default_axes()
            if graph.plot_type != "histogram":
                graph.y = graph.y or y
        result = render_plot(sample, pop, graph, w_px, h_px, self.render_options(graph, scale, item), overlays=overlays or None)
        self.cache.renders += 1
        self.render_legends[item.id] = list(result.legend_entries)  # what the overlay legend showed (tests, tooltips)
        img = result.image
        self.cache.put(key, img)
        if scale == SCENE_SCALE:
            item._last_image = (ident, img)  # stale image shown while resizing
        return img

    # ------------------------------------------------------------- painting
    def text_context(self, item: LayoutItem | None, sample_override: Sample | None = None, layout: Layout | None = None,
                     page: int = 1, pages: int = 1) -> TextContext:
        """What tokens of a text item / annotation resolve against: the batch
        sample, else the item's own sample, else the first plot's sample on
        the page."""
        layout = layout or self._layout
        sample = sample_override
        if sample is None and item is not None and item.sample_id is not None:
            sample = self.workspace.sample_by_id(item.sample_id)
        if sample is None and layout is not None:
            for it in layout.items:
                if it.kind == "plot" and it.sample_id is not None:
                    sample = self.workspace.sample_by_id(it.sample_id)
                    if sample is not None:
                        break
        pop = None
        if sample is not None and item is not None and item.pop_path:
            pop = sample.root.find(item.pop_path)
        group = ""
        if layout is not None and layout.batch_group:
            group = layout.batch_group
        elif getattr(self.controller, "current_group", None) is not None:
            group = self.controller.current_group.name
        decimals = 1
        try:
            decimals = int(self.controller.preferences.get("freq_decimals", 1))
        except (TypeError, ValueError):
            pass
        return TextContext(
            sample=sample, population=pop, workspace=self.workspace, layout_name=layout.name if layout is not None else "",
            group_name=group, page=page, pages=pages, decimals=decimals, layout=layout,
        )

    def display_text(self, item: LayoutItem, sample_override: Sample | None = None, layout: Layout | None = None, page: int = 1, pages: int = 1) -> str:
        if not layout_text.has_tokens(item.text) and "{{" not in item.text and "}}" not in item.text:
            return item.text
        return expand_text(item.text, self.text_context(item, sample_override, layout, page, pages))

    def plot_title(self, item: LayoutItem, pop: Population | None) -> str:
        if item.props.get("title"):
            return str(item.props["title"])
        return pop.display_name if pop is not None else (item.pop_path.rsplit("/", 1)[-1] or "Ungated")

    def caption_lines(self, item: LayoutItem, sample: Sample | None, pop: Population | None, ctx: TextContext | None = None) -> list[str]:
        lines = []
        if item.show_caption:
            lines = [
                sample.name if sample is not None else "(missing sample)",
                (pop.path or pop.display_name) if pop is not None else (item.pop_path or "Ungated"),
                f"{pop.count:,}" if pop is not None else "",
            ]
        ann = item_annotations(item)
        if ann:
            ctx = self.text_context(None) if ctx is None else ctx
            ctx = TextContext(**{**ctx.__dict__, "sample": sample, "population": pop})
            lines += annotation_lines(ann, ctx)
        return lines

    def paint_item(self, painter: QPainter, item: LayoutItem, sample_override: Sample | None, scale: float, allow_stale: bool = False,
                   ctx: TextContext | None = None):
        """Paint any page object in item-local coordinates (points)."""
        if item.kind == "text":
            self.paint_text(painter, item, sample_override, ctx)
            return
        kind = self._item_kinds.get(item.kind)
        if kind is not None and kind.get("painter") is not None:
            painter.save()
            try:
                kind["painter"](self, painter, item, sample_override, scale)
            finally:
                painter.restore()
            return
        self.paint_plot(painter, item, sample_override, scale, allow_stale, ctx=ctx)

    def paint_plot(self, painter: QPainter, item: LayoutItem, sample_override: Sample | None, scale: float, allow_stale: bool = False,
                   draw_text: bool = True, ctx: TextContext | None = None):
        """Paint a plot item in item-local coordinates (points). With
        draw_text False only the plot area is painted (PowerPoint export
        writes title and caption as text boxes)."""
        sample, pop = self.resolve(item, sample_override)
        rect = plot_rect(item)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        if sample is None or pop is None:
            painter.setPen(QPen(QColor("#000000"), 0.75))
            painter.setBrush(QColor("#ffffff"))
            painter.drawRect(rect)
            painter.setPen(QColor("#888888"))
            painter.setFont(_px_font(10))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "no sample" if sample is None else "no population")
        else:
            overlays = self.resolve_overlays(item, sample_override)
            img = self.plot_image(item, sample, pop, overlays, rect.width(), rect.height(), scale, allow_stale)
            painter.drawImage(rect, img)
        if draw_text:
            family = item.graph.options.get("font_family") or ""
            if item.show_title:
                tf = _px_font(title_font_size(item), bold=True)
                if family:
                    tf.setFamily(family)
                painter.setFont(tf)
                painter.setPen(QColor("#000000"))
                painter.drawText(QRectF(0, 0, item.w, title_height(item)), Qt.AlignmentFlag.AlignCenter, self.plot_title(item, pop))
            lines = self.caption_lines(item, sample, pop, ctx)
            if lines:
                font = _px_font(caption_font_size(item))
                if family:
                    font.setFamily(family)
                painter.setFont(font)
                painter.setPen(QColor("#000000"))
                fm = QFontMetricsF(font)
                lh = caption_line_height(item)
                y = rect.bottom() + 2
                for line in lines:
                    text = fm.elidedText(line, Qt.TextElideMode.ElideMiddle, item.w)
                    painter.drawText(QRectF(0, y, item.w, lh), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, text)
                    y += lh
        painter.restore()

    def text_document(self, item: LayoutItem, text: str) -> QTextDocument:
        fmt = text_format(item)
        doc = QTextDocument()
        doc.setDocumentMargin(4.0)  # what QGraphicsTextItem uses, so export matches the scene
        doc.setDefaultFont(text_font(item))
        opt = doc.defaultTextOption()
        opt.setAlignment(TEXT_ALIGN[fmt["align"]])
        doc.setDefaultTextOption(opt)
        doc.setPlainText(text)
        if item.w > 0:
            doc.setTextWidth(item.w)
        return doc

    def paint_text(self, painter: QPainter, item: LayoutItem, sample_override: Sample | None = None, ctx: TextContext | None = None):
        layout = ctx.layout if ctx is not None else None
        page, pages = (ctx.page, ctx.pages) if ctx is not None else (1, 1)
        text = self.display_text(item, sample_override, layout, page, pages)
        doc = self.text_document(item, text)
        painter.save()
        pc = QAbstractTextDocumentLayout.PaintContext()
        pc.palette.setColor(pc.palette.ColorRole.Text, QColor(text_format(item)["color"]))
        doc.documentLayout().draw(painter, pc)
        painter.restore()

    def paint_page(self, painter: QPainter, layout: Layout, sample_override: Sample | None = None, scale: float = 2.0, header: str | None = None,
                   page: int = 1, pages: int = 1, items: list[LayoutItem] | None = None):
        """Paint a whole page in page-point coordinates."""
        w, h = layout.page_size
        m = layout.margin
        ctx = self.text_context(None, sample_override, layout, page, pages)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(QRectF(0, 0, w, h), QColor("#ffffff"))
        if header:
            painter.setFont(_px_font(HEADER_FONT, bold=True))
            painter.setPen(QColor("#000000"))
            top = max(MARGIN, m)
            painter.drawText(QRectF(max(8.0, m), 8, w - 2 * max(8.0, m), top - 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, header)
        for item in layout.items if items is None else items:
            painter.save()
            painter.translate(item.x, item.y)
            self.paint_item(painter, item, sample_override, scale, ctx=ctx)
            painter.restore()
        painter.restore()

    def page_pixmap(self, layout: Layout | None = None, sample_override: Sample | None = None, scale: float = 2.0, header: str | None = None,
                    page: int = 1, pages: int = 1) -> QPixmap:
        layout = layout or self.current_layout()
        self._commit_text_edit()
        w, h = layout.page_size
        pm = QPixmap(int(round(w * scale)), int(round(h * scale)))
        pm.fill(QColor("#ffffff"))
        p = QPainter(pm)
        p.scale(scale, scale)
        if header is None and sample_override is not None:
            header = self.batch_header(layout, sample_override, page, pages)
        self.paint_page(p, layout, sample_override, scale, header=header, page=page, pages=pages)
        p.end()
        return pm

    def page_image(self, layout: Layout | None = None, sample_override: Sample | None = None, dpi: float = 144.0, header: str | None = None,
                   page: int = 1, pages: int = 1) -> QImage:
        """The rendered page as an image at `dpi` (dots per inch recorded)."""
        img = self.page_pixmap(layout, sample_override, float(dpi) / 72.0, header, page, pages).toImage()
        _set_dpi(img, dpi)
        return img

    def items_image(self, items: list[LayoutItem], scale: float = 2.0, sample_override: Sample | None = None) -> QImage:
        """The given objects (their bounding box) as an image, for the clipboard."""
        layout = self.current_layout()
        if not items:
            return self.page_pixmap(layout, sample_override, scale).toImage()
        self._commit_text_edit()
        r = QRectF()
        for it in items:
            r = r.united(QRectF(it.x, it.y, max(1.0, it.w), max(1.0, it.h)))
        r.adjust(-4, -4, 4, 4)
        img = QImage(max(1, int(math.ceil(r.width() * scale))), max(1, int(math.ceil(r.height() * scale))), QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor("#ffffff"))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.scale(scale, scale)
        p.translate(-r.left(), -r.top())
        order = {i.id: k for k, i in enumerate(layout.items)}
        ctx = self.text_context(None, sample_override, layout)
        for it in sorted(items, key=lambda i: order.get(i.id, 0)):
            p.save()
            p.translate(it.x, it.y)
            self.paint_item(p, it, sample_override, scale, ctx=ctx)
            p.restore()
        p.end()
        _set_dpi(img, 72.0 * scale)
        return img

    def _commit_text_edit(self):
        focus = self.scene.focusItem()
        if isinstance(focus, TextGraphicsItem) and focus.editing:
            focus.finish_editing()

    # ------------------------------------------------------------- export
    def _batch_samples(self, batch_group) -> list[Sample]:
        if batch_group is None:
            return []
        group = batch_group if isinstance(batch_group, Group) else self.workspace.group_by_name(str(batch_group))
        return self.workspace.members(group) if group is not None else []

    def batch_pages(self, batch_group=None, keyword: str | None = None) -> list[Sample]:
        """The samples a batch iterates over: every member of the group, or
        (iterating by keyword value) the first member with each distinct value
        of `keyword`, ordered by that value."""
        samples = self._batch_samples(batch_group)
        if not keyword or not samples:
            return samples
        from .experiment import natural_key

        first: dict[str, Sample] = {}
        for s in samples:
            v = layout_text._keyword_value(s, keyword)
            first.setdefault("" if v is None else v, s)
        return [first[v] for v in sorted(first, key=natural_key)]

    def batch_header(self, layout: Layout, sample: Sample | None, page: int = 1, pages: int = 1, template: str | None = None) -> str:
        """The page header of a batch page (Layout.props["batch_header"],
        default the sample name; tokens allowed)."""
        if sample is None:
            return ""
        if template is None:
            template = layout.props.get("batch_header")
        if not template:
            keyword = layout.props.get("batch_keyword") if layout.props.get("batch_by") == "keyword" else None
            template = f"{keyword}: {{{keyword}}}" if keyword else "{Sample}"
        return expand_text(str(template), self.text_context(None, sample, layout, page, pages))

    def _pages(self, batch_group, keyword: str | None) -> list[Sample | None]:
        samples = self.batch_pages(batch_group, keyword) if batch_group is not None else []
        return list(samples) if samples else [None]

    @staticmethod
    def _step(progress, done: int, text: str = "") -> bool:
        if progress is None:
            return True
        try:
            return progress(done, text) is not False
        except TypeError:
            return progress(done) is not False

    def export_png(self, path, scale: float = 2.0, dpi: float | None = None) -> bool:
        return self.export_image(path, "png", dpi if dpi is not None else 72.0 * scale)

    def export_image(self, path, fmt: str | None = None, dpi: float = 144.0, sample_override: Sample | None = None, header: str | None = None,
                     page: int = 1, pages: int = 1, quality: int = 95) -> bool:
        """Write the current page as PNG / JPEG / TIFF / BMP at `dpi` (or SVG
        / PDF by format)."""
        fmt = normalise_format(fmt, path)
        if fmt == "svg":
            return self.export_svg(path, sample_override, header=header)
        if fmt == "pdf":
            return self._write_pdf(path, self.current_layout(), [sample_override], header_template=header) == 1
        img = self.page_image(self.current_layout(), sample_override, dpi, header, page, pages)
        writer, _ = IMAGE_FORMATS[fmt]
        if writer in ("JPEG", "BMP"):
            img = img.convertToFormat(QImage.Format.Format_RGB32)
            _set_dpi(img, dpi)
        return img.save(str(path), writer, quality if writer == "JPEG" else -1)

    def _write_pdf(self, path, layout: Layout, pages: list[Sample | None], progress=None, header_template: str | None = None) -> int:
        self._commit_text_edit()
        w, h = layout.page_size
        writer = QPdfWriter(str(path))
        writer.setResolution(300)
        writer.setCreator("FlowDesk")
        writer.setTitle(layout.name)
        size = QPageSize(QSizeF(w, h), QPageSize.Unit.Point, layout.name, QPageSize.SizeMatchPolicy.ExactMatch)
        writer.setPageLayout(QPageLayout(size, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0)))
        painter = QPainter(writer)
        factor = writer.resolution() / 72.0
        n = len(pages)
        cancelled = False
        for k, sample in enumerate(pages):
            if not self._step(progress, k, f"Page {k + 1} of {n}"):
                cancelled = True
                break
            if k:
                writer.newPage()
            painter.save()
            painter.scale(factor, factor)
            header = self.batch_header(layout, sample, k + 1, n, header_template) if sample is not None else None
            self.paint_page(painter, layout, sample, 3.0, header=header, page=k + 1, pages=n)
            painter.restore()
        painter.end()
        del writer
        if cancelled:
            try:
                os.remove(str(path))
            except OSError:
                pass
            return 0
        self._step(progress, n, "Done")
        return n

    def export_pdf(self, path, batch_group=None, progress=None, keyword: str | None = None, header: str | None = None) -> int:
        """Write the current layout as PDF. With batch_group (Group or group
        name) one page per member sample is written (per distinct value of
        `keyword` when given). `progress(done, text)` returning False cancels
        (the partial file is removed and 0 returned). Returns the page count."""
        layout = self.current_layout()
        return self._write_pdf(path, layout, self._pages(batch_group, keyword), progress, header)

    def export_svg(self, path, sample_override: Sample | None = None, header: str | None = None) -> bool:
        try:
            from PySide6.QtSvg import QSvgGenerator
        except ImportError:
            return False
        layout = self.current_layout()
        self._commit_text_edit()
        w, h = layout.page_size
        gen = QSvgGenerator()
        gen.setFileName(str(path))
        gen.setSize(QSizeF(w, h).toSize())
        gen.setViewBox(QRectF(0, 0, w, h))
        gen.setResolution(72)
        gen.setTitle(layout.name)
        p = QPainter(gen)
        if header is None and sample_override is not None:
            header = self.batch_header(layout, sample_override)
        self.paint_page(p, layout, sample_override, 2.0, header=header)
        p.end()
        return True

    def export_images(self, folder, batch_group, fmt: str = "png", dpi: float = 300.0, progress=None, keyword: str | None = None,
                      header: str | None = None) -> list[str]:
        """One file per sample of the batch group ("<layout> - <sample>.<ext>")
        in `folder`: PNG / JPEG / TIFF / BMP at `dpi`, SVG or PDF. Stops when
        `progress` returns False. Returns the written paths."""
        layout = self.current_layout()
        fmt = normalise_format(fmt)
        folder = Path(str(folder))
        folder.mkdir(parents=True, exist_ok=True)
        pages = self._pages(batch_group, keyword)
        n = len(pages)
        written: list[str] = []
        used: set[str] = set()
        for k, sample in enumerate(pages):
            if not self._step(progress, k, f"Page {k + 1} of {n}"):
                break
            stem = safe_filename(f"{layout.name} - {sample.name}" if sample is not None else layout.name)
            name, j = stem, 2
            while name.lower() in used:
                name = f"{stem} ({j})"
                j += 1
            used.add(name.lower())
            path = folder / f"{name}.{fmt}"
            hdr = self.batch_header(layout, sample, k + 1, n, header) if sample is not None else None
            if fmt == "pdf":
                ok = self._write_pdf(path, layout, [sample], header_template=header) == 1
            elif fmt == "svg":
                ok = self.export_svg(path, sample, header=hdr)
            else:
                ok = self.export_image(path, fmt, dpi, sample, hdr, k + 1, n)
            if ok:
                written.append(str(path))
        else:
            self._step(progress, n, "Done")
        return written

    # -- PowerPoint
    def _plot_area_png(self, item: LayoutItem, sample_override: Sample | None, scale: float) -> bytes:
        rect = plot_rect(item)
        img = QImage(max(1, int(math.ceil(rect.width() * scale))), max(1, int(math.ceil(rect.height() * scale))), QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QColor("#ffffff"))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.scale(scale, scale)
        p.translate(-rect.left(), -rect.top())
        self.paint_plot(p, item, sample_override, scale, draw_text=False)
        p.end()
        _set_dpi(img, 72.0 * scale)
        return image_bytes(img, "PNG")

    def pptx_slide(self, layout: Layout, sample: Sample | None = None, page: int = 1, pages: int = 1, header: str | None = None,
                   as_page_image: bool = False, scale: float = 2.0) -> PptxSlide:
        """One slide for one page: every plot as a PNG rendered at `scale`
        with its title and caption as text boxes, text items as formatted text
        boxes (tokens resolved for `sample`), other objects as pictures. With
        as_page_image the whole page is a single picture."""
        w, h = layout.page_size
        m = layout.margin
        slide = PptxSlide(title=header or layout.name)
        if as_page_image:
            img = self.page_image(layout, sample, 72.0 * scale, header, page, pages)
            slide.shapes.append(PptxPicture(image_bytes(img, "PNG"), 0, 0, w, h, "Page", layout.name))
            return slide
        if header:
            top = max(MARGIN, m)
            slide.shapes.append(PptxText(header, max(8.0, m), 8, w - 2 * max(8.0, m), top - 12, HEADER_FONT, bold=True, name="Header"))
        ctx = self.text_context(None, sample, layout, page, pages)
        for item in layout.items:
            if item.kind == "text":
                fmt = text_format(item)
                slide.shapes.append(
                    PptxText(
                        self.display_text(item, sample, layout, page, pages), item.x + 4, item.y + 4, max(1.0, item.w - 8), max(1.0, item.h - 8),
                        item.font_size, fmt["bold"], fmt["italic"], fmt["underline"], fmt["color"], fmt["align"], fmt["font_family"], "Text",
                    )
                )
            elif item.kind == "plot" or item.kind not in self._item_kinds:
                s, pop = self.resolve(item, sample)
                rect = plot_rect(item)
                family = item.graph.options.get("font_family") or ""
                if item.show_title:
                    slide.shapes.append(
                        PptxText(self.plot_title(item, pop), item.x, item.y, item.w, title_height(item), title_font_size(item), bold=True,
                                 align="center", font_family=family, name="Title")
                    )
                descr = f"{s.name if s else ''} {item.pop_path or 'Ungated'}".strip()
                slide.shapes.append(PptxPicture(self._plot_area_png(item, sample, scale), item.x + rect.x(), item.y + rect.y(), rect.width(), rect.height(), "Plot", descr))
                lines = self.caption_lines(item, s, pop, ctx)
                if lines:
                    slide.shapes.append(
                        PptxText("\n".join(lines), item.x, item.y + rect.bottom() + 2, item.w, caption_height(item), caption_font_size(item),
                                 align="center", font_family=family, name="Caption")
                    )
            else:
                img = self.items_image([item], scale, sample)
                slide.shapes.append(PptxPicture(image_bytes(img, "PNG"), item.x - 4, item.y - 4, item.w + 8, item.h + 8, item.kind.title()))
        return slide

    def export_pptx(self, path, batch_group=None, progress=None, keyword: str | None = None, header: str | None = None,
                    as_page_image: bool = False, scale: float = 2.0) -> int:
        """Write the current layout as a PowerPoint file: one slide, or with
        batch_group one slide per sample (per distinct `keyword` value).
        `progress(done, text)` returning False cancels (no file is written,
        0 returned). Returns the slide count."""
        layout = self.current_layout()
        self._commit_text_edit()
        pages = self._pages(batch_group, keyword)
        n = len(pages)
        slides = []
        for k, sample in enumerate(pages):
            if not self._step(progress, k, f"Slide {k + 1} of {n}"):
                return 0
            hdr = self.batch_header(layout, sample, k + 1, n, header) if sample is not None else None
            slides.append(self.pptx_slide(layout, sample, k + 1, n, hdr, as_page_image, scale))
        count = write_pptx(path, slides, layout.page_size, title=layout.name)
        self._step(progress, n, "Done")
        return count

    # -- new layouts / printer
    def batch_to_layouts(self, batch_group, keyword: str | None = None, progress=None, header: str | None = None) -> list[Layout]:
        """FlowJo's "batch to new layouts": one copy of the current layout per
        sample with every plot (and token text) pointed at that sample. One
        undo step."""
        layout = self.current_layout()
        self._commit_text_edit()
        self.flush_notify()
        pages = [s for s in self._pages(batch_group, keyword) if s is not None]
        n = len(pages)
        created: list[Layout] = []
        names = {l.name for l in self.workspace.layouts}
        insert_at = self.workspace.layouts.index(layout) + 1
        for k, sample in enumerate(pages):
            if not self._step(progress, k, f"Layout {k + 1} of {n}"):
                break
            d = layout.to_dict()
            d["id"] = new_id()
            base = f"{layout.name} - {sample.name}"
            name, j = base, 2
            while name in names:
                name = f"{base} {j}"
                j += 1
            names.add(name)
            d["name"] = name
            d["batch_group"] = layout.batch_group
            for it in d["items"]:
                it["id"] = new_id()
                old = it.get("sample_id")
                if it.get("kind") == "text":
                    it["sample_id"] = sample.id
                    continue
                it["sample_id"] = sample.id
                for ov in it.get("overlays", []):
                    if ov.get("sample_id") == old:
                        ov["sample_id"] = sample.id
            new = Layout.from_dict(d)
            hdr = self.batch_header(layout, sample, k + 1, n, header)
            if hdr:
                m = layout.margin
                new.items.append(LayoutItem(kind="text", x=max(8.0, m), y=4.0, w=layout.page_size[0] - 2 * max(8.0, m), h=24.0, text=hdr,
                                            font_size=int(HEADER_FONT), show_caption=False, sample_id=sample.id, props={"bold": True}))
            self.workspace.layouts.insert(insert_at + k, new)
            created.append(new)
        if created:
            self._notify_now()
            self._sync_list()
        self._step(progress, n, "Done")
        return created

    def print_pages(self, printer, batch_group=None, keyword: str | None = None, progress=None, header: str | None = None) -> int:
        layout = self.current_layout()
        self._commit_text_edit()
        pages = self._pages(batch_group, keyword)
        n = len(pages)
        painter = QPainter(printer)
        rect = printer.pageLayout().paintRectPixels(printer.resolution())
        w, h = layout.page_size
        f = min(rect.width() / w, rect.height() / h)
        done = 0
        for k, sample in enumerate(pages):
            if not self._step(progress, k, f"Page {k + 1} of {n}"):
                break
            if k:
                printer.newPage()
            painter.save()
            painter.scale(f, f)
            hdr = self.batch_header(layout, sample, k + 1, n, header) if sample is not None else None
            self.paint_page(painter, layout, sample, 3.0, header=hdr, page=k + 1, pages=n)
            painter.restore()
            done += 1
        painter.end()
        return done

    def print_layout(self, batch_group=None, keyword: str | None = None, header: str | None = None):
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter

        layout = self.current_layout()
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPageOrientation(QPageLayout.Orientation.Landscape if layout.landscape else QPageLayout.Orientation.Portrait)
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if batch_group is None:
            self.print_pages(printer)
            return
        n = len(self._pages(batch_group, keyword))
        self._with_progress("Printing", n, lambda step: self.print_pages(printer, batch_group, keyword, step, header))

    def _with_progress(self, title: str, total: int, fn):
        from .loader import run_with_progress

        return run_with_progress(self, title, total, fn)

    # -- dialogs
    def _ask_dpi(self, default: int = 300) -> int | None:
        items = [f"{d} dpi" for d in DPI_CHOICES]
        current = DPI_CHOICES.index(default) if default in DPI_CHOICES else 0
        text, ok = QInputDialog.getItem(self, "Export Resolution", "Resolution:", items, current, False)
        if not ok:
            return None
        try:
            return int(text.split()[0])
        except (ValueError, IndexError):
            return default

    EXPORT_FILTERS = [
        ("png", "PNG image (*.png)"),
        ("jpg", "JPEG image (*.jpg *.jpeg)"),
        ("tif", "TIFF image (*.tif *.tiff)"),
        ("bmp", "BMP image (*.bmp)"),
        ("pdf", "PDF document (*.pdf)"),
        ("svg", "SVG image (*.svg)"),
        ("pptx", "PowerPoint presentation (*.pptx)"),
    ]

    def export_dialog(self):
        layout = self.current_layout()
        flt_text = ";;".join(f for _, f in self.EXPORT_FILTERS)
        path, flt = QFileDialog.getSaveFileName(self, "Export Layout", layout.name, flt_text)
        if not path:
            return
        ext = Path(path).suffix.lower().lstrip(".")
        ext = FORMAT_ALIASES.get(ext, ext)
        known = [k for k, _ in self.EXPORT_FILTERS]
        if ext not in known:
            ext = next((k for k, f in self.EXPORT_FILTERS if f == flt), "png")
            path += f".{ext}"
        self.export_to(path, ext)

    def export_to(self, path, ext: str, dpi: int | None = None) -> bool:
        if ext == "pptx":
            ok = self.export_pptx(path) > 0
        elif ext == "pdf":
            ok = self.export_pdf(path) > 0
        elif ext == "svg":
            ok = self.export_svg(path)
            if not ok:
                QMessageBox.warning(self, "Export", "SVG export is not available (QtSvg missing).")
                return False
        else:
            if dpi is None:
                dpi = self._ask_dpi(300)
                if dpi is None:
                    return False
            ok = self.export_image(path, ext, dpi)
        self.statusBar().showMessage(f"Exported {path}" if ok else f"Could not write {path}", 5000)
        return ok

    def export_pptx_dialog(self):
        layout = self.current_layout()
        path, _ = QFileDialog.getSaveFileName(self, "Export to PowerPoint", f"{layout.name}.pptx", "PowerPoint presentation (*.pptx)")
        if not path:
            return
        if not path.lower().endswith(".pptx"):
            path += ".pptx"
        self.export_to(path, "pptx")

    def copy_page_image(self, dpi: float = 144.0):
        """Copy the whole page as an image (paste into PowerPoint / Word)."""
        img = self.page_image(self.current_layout(), None, dpi)
        QGuiApplication.clipboard().setImage(img)
        self.statusBar().showMessage("Layout page copied to the clipboard", 3000)
        return img

    def _default_batch_group(self) -> Group | None:
        ws = self.workspace
        return ws.group_by_name(self.group_combo.currentText()) or (ws.groups[0] if ws.groups else None)

    def batch_pdf_dialog(self):
        layout = self.current_layout()
        group = self._default_batch_group()
        if group is None or not self.workspace.members(group):
            QMessageBox.information(self, "Batch", f"The group “{group.name if group else ''}” has no samples.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Create Batch PDF", f"{layout.name} - {group.name}.pdf", "PDF document (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        keyword = layout.props.get("batch_keyword") if layout.props.get("batch_by") == "keyword" else None
        n = len(self._pages(group, keyword))
        pages = self._with_progress("Create Batch PDF", n, lambda step: self.export_pdf(path, group, step, keyword))
        self.statusBar().showMessage(f"Wrote {pages} page(s) to {path}" if pages else "Batch cancelled", 5000)

    def batch_dialog(self, output: str | None = None):
        """FlowJo's Batch dialog: iterate the layout over a group (or by
        keyword value) with a header template, to PDF, PowerPoint, separate
        image files, the printer or new layouts."""
        layout = self.current_layout()
        dlg = BatchDialog(self, layout, output)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.run_batch(dlg.values())

    def run_batch(self, opts: dict, path=None):
        """Run a batch described by BatchDialog.values(); `path` skips the
        file / folder picker. Stores the iteration settings in the layout."""
        layout = self.current_layout()
        group = self.workspace.group_by_name(opts.get("group") or "") or self._default_batch_group()
        keyword = opts.get("keyword") if opts.get("by") == "keyword" else None
        header = opts.get("header") or None
        props_before = dict(layout.props)
        layout.props["batch_by"] = opts.get("by", "group")
        if keyword:
            layout.props["batch_keyword"] = keyword
        if header:
            layout.props["batch_header"] = header
        else:
            layout.props.pop("batch_header", None)
        if group is not None and layout.batch_group != group.name:
            layout.batch_group = group.name
            self._fill_groups()
        if layout.props != props_before:
            self._notify_now()
        if group is None or not self.workspace.members(group):
            QMessageBox.information(self, "Batch", "The batch group has no samples.")
            return None
        out = opts.get("output", "pdf")
        n = len(self._pages(group, keyword))
        stem = safe_filename(f"{layout.name} - {group.name}")
        if out in ("pdf", "pptx"):
            if path is None:
                ext = out
                label = "PDF document (*.pdf)" if out == "pdf" else "PowerPoint presentation (*.pptx)"
                path, _ = QFileDialog.getSaveFileName(self, "Batch", f"{stem}.{ext}", label)
                if not path:
                    return None
                if not path.lower().endswith(f".{ext}"):
                    path += f".{ext}"
            if out == "pdf":
                result = self._with_progress("Batch PDF", n, lambda step: self.export_pdf(path, group, step, keyword, header))
            else:
                result = self._with_progress("Batch PowerPoint", n, lambda step: self.export_pptx(path, group, step, keyword, header, bool(opts.get("page_image"))))
            self.statusBar().showMessage(f"Wrote {result} page(s) to {path}" if result else "Batch cancelled", 5000)
            return result
        if out == "images":
            if path is None:
                path = QFileDialog.getExistingDirectory(self, "Folder for the image files")
                if not path:
                    return None
            fmt = opts.get("format", "png")
            dpi = float(opts.get("dpi", 300))
            files = self._with_progress("Batch Images", n, lambda step: self.export_images(path, group, fmt, dpi, step, keyword, header))
            self.statusBar().showMessage(f"Wrote {len(files)} file(s) to {path}", 5000)
            return files
        if out == "layouts":
            created = self._with_progress("Batch to Layouts", n, lambda step: self.batch_to_layouts(group, keyword, step, header))
            self.statusBar().showMessage(f"Created {len(created)} layout(s)", 5000)
            return created
        if out == "printer":
            self.print_layout(group, keyword, header)
        return None

    # ------------------------------------------------------------ layouts
    def new_layout(self, name: str | None = None) -> Layout:
        names = {l.name for l in self.workspace.layouts}
        base = name or "Layout"
        name, k = base, 2
        while name in names:
            name = f"{base} {k}"
            k += 1
        lay = self.new_layout_object(name)
        self.workspace.layouts.append(lay)
        self._notify_now()
        self.set_current_layout(lay)
        return lay

    def rename_layout(self, name: str | None = None):
        lay = self.current_layout()
        if name is None:
            name, ok = QInputDialog.getText(self, "Rename Layout", "Name:", text=lay.name)
            if not ok:
                return
        name = name.strip()
        if name and name != lay.name:
            lay.name = name
            self._notify_now()
            self._sync_list()

    def duplicate_layout(self) -> Layout:
        lay = self.current_layout()
        d = lay.to_dict()
        d["id"] = new_id()
        d["name"] = f"{lay.name} copy"
        for it in d["items"]:
            it["id"] = new_id()
        copy = Layout.from_dict(d)
        self.workspace.layouts.insert(self.workspace.layouts.index(lay) + 1, copy)
        self._notify_now()
        self.set_current_layout(copy)
        return copy

    def delete_layout(self, confirm: bool = True):
        lay = self.current_layout()
        if confirm and lay.items:
            ans = QMessageBox.question(self, "Delete Layout", f"Delete the layout “{lay.name}”?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        layouts = self.workspace.layouts
        idx = layouts.index(lay)
        layouts.remove(lay)
        if not layouts:
            layouts.append(self.new_layout_object("Layout"))
        self._layout = layouts[min(idx, len(layouts) - 1)]
        self._notify_now()
        self._sync_list()
        self._rebuild_scene()

    def page_setup(self):
        lay = self.current_layout()
        dlg = PageSetupDialog(lay, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            v = dlg.result_values()
            self.set_page(v["page"], v["landscape"], margin=v["margin"], custom_size=v["custom_size"])
            if v["make_default"]:
                self.save_layout_defaults()

    def set_page(self, page: str, landscape: bool, margin: float | None = None, custom_size: tuple[float, float] | None = None):
        """Paper size (a PAGE_SIZES name, or "Custom" with custom_size in
        points, portrait width × height), orientation and margin (points).
        One undo step."""
        lay = self.current_layout()
        if page == CUSTOM_PAGE and custom_size is None and lay.page != CUSTOM_PAGE:
            page = "A4"
        lay.page = page if (page in PAGE_SIZES or page == CUSTOM_PAGE) else "A4"
        if lay.page == CUSTOM_PAGE and custom_size is not None:
            cw, ch = (max(36.0, float(v)) for v in custom_size)
            lay.props["custom_w"], lay.props["custom_h"] = cw, ch
        elif lay.page != CUSTOM_PAGE:
            lay.props.pop("custom_w", None)
            lay.props.pop("custom_h", None)
        lay.landscape = bool(landscape)
        if margin is not None:
            m = max(0.0, float(margin))
            if abs(m - DEFAULT_MARGIN) < 1e-9:
                lay.props.pop("margin", None)
            else:
                lay.props["margin"] = m
        self._notify_now()
        self._rebuild_scene()
        self.fit_page()

    def save_layout_defaults(self, layout: Layout | None = None):
        """Use the layout's page setup, the grid and the default plot size /
        caption font for new layouts (persisted per user)."""
        lay = layout or self.current_layout()
        d = self.defaults
        d.set("page", lay.page if lay.page in PAGE_SIZES else "A4")
        d.set("landscape", bool(lay.landscape))
        d.set("margin", float(lay.margin))
        d.set("grid", bool(self.show_grid))
        d.set("grid_step", float(self.grid_step))
        plots = [i for i in self.selected_layout_items() if i.kind == "plot"]
        if plots:
            d.set("plot_size", float(plots[0].w))
            d.set("caption_font_size", float(caption_font_size(plots[0])))
        self.statusBar().showMessage("Saved as the defaults for new layouts", 4000)

    # -------------------------------------------------------------- items
    def default_graph(self, sample: Sample, population: Population) -> GraphSettings:
        g = population.graph.copy()
        if not g.x:
            g.plot_type = self.controller.preferences.get("plot_type", "pseudocolor") or "pseudocolor"
            x, y = sample.default_axes()
            g.x = x
            g.y = None if g.plot_type == "histogram" else (g.y or y)
        if g.plot_type == "histogram":
            g.y = None
        return g

    def _free_position(self, w: float, h: float) -> QPointF:
        lay = self.current_layout()
        pw, ph = lay.page_size
        m = lay.margin
        y = m
        while y + h <= ph - m + 1:
            x = m
            while x + w <= pw - m + 1:
                r = QRectF(x, y, w, h)
                if not any(r.intersects(QRectF(i.x, i.y, i.w, i.h)) for i in lay.items):
                    return QPointF(x, y)
                x += w + SPACING
            y += h + SPACING
        return QPointF(m, m)

    def _make_plot_item(self, sample: Sample, population: Population, x: float, y: float) -> LayoutItem:
        item = LayoutItem(kind="plot", sample_id=sample.id, pop_path=population.path, graph=self.default_graph(sample, population))
        cf = float(self.defaults.get("caption_font_size"))
        if abs(cf - CAPTION_FONT) > 1e-9:
            item.props["caption_font_size"] = cf
        size = self.default_plot_size()
        item.w = size
        item.h = size + caption_height(item) + title_height(item)
        item.x, item.y = (self.snap(x), self.snap(y)) if self.snap_to_grid else (x, y)
        return item

    @staticmethod
    def _overlay_entry(ov) -> dict | None:
        if isinstance(ov, dict):
            return {"sample_id": ov.get("sample_id"), "pop_path": ov.get("pop_path") or ""}
        try:
            s, p = ov
            return {"sample_id": s.id, "pop_path": p.path}
        except (TypeError, ValueError, AttributeError):
            return None

    def add_population(self, sample: Sample, population: Population, pos: QPointF | None = None, overlays=None) -> LayoutItem:
        """Add a plot of `population` (axes, plot type and display options
        from its graph settings). `overlays` - [{"sample_id", "pop_path"}] or
        [(Sample, Population)] as a graph window holds them - are kept, so
        "Add to Layout" from a graph window brings its overlays along."""
        lay = self.current_layout()
        size = self.default_plot_size()
        probe = LayoutItem()
        h = size + caption_height(probe)
        if pos is None:
            pos = self._free_position(size, h)
        item = self._make_plot_item(sample, population, pos.x(), pos.y())
        for ov in overlays or []:
            entry = self._overlay_entry(ov)
            if entry is None or (entry["sample_id"], entry["pop_path"]) == (item.sample_id, item.pop_path) or entry in item.overlays:
                continue
            item.overlays.append(entry)
        lay.items.append(item)
        self._add_scene_item(item, len(lay.items) - 1)
        self._notify_now()
        return item

    def _group_by_id(self, gid) -> Group | None:
        return next((g for g in self.workspace.groups if g.id == gid), None)

    def _entry_samples(self, entry: dict) -> list[Sample]:
        """Samples a dropped entry stands for: a group-template entry means
        every member of the group."""
        if entry.get("group_id") is not None:
            g = self._group_by_id(entry["group_id"])
            if g is not None:
                members = self.workspace.members(g)
                if members:
                    return members
        s = self.workspace.sample_by_id(entry.get("sample_id")) if entry.get("sample_id") is not None else None
        return [s] if s is not None else []

    def drop_mime(self, mime, scene_pos: QPointF) -> list[LayoutItem]:
        """Handle a POPULATION_MIME drop at scene_pos. Dropping on a plot adds
        the populations as overlays of that plot - one per sample for a
        multi-sample selection, every member for a group population - and
        dropped statistics as annotations (returns [that item]); dropping
        elsewhere creates one plot per population (on the first member for a
        group) and a live statistic text box per dropped statistic."""
        entries = parse_population_entries(mime)
        resolved = []  # (entry, samples)
        for e in entries:
            samples = self._entry_samples(e)
            if samples and (e["paths"] or e["stats"]):
                resolved.append((e, samples))
        if not resolved:
            return []
        lay = self.current_layout()
        target = self.plot_item_at(scene_pos)
        if target is not None:
            item = target.item
            changed = False
            for e, samples in resolved:
                for path in e["paths"]:
                    for s in samples:
                        p = s.root.find(path) if path else s.root
                        if p is None or (s.id == item.sample_id and p.path == item.pop_path):
                            continue
                        entry = {"sample_id": s.id, "pop_path": p.path}
                        if entry not in item.overlays:
                            item.overlays.append(entry)
                            changed = True
                for st in e["stats"]:
                    ann = {"statistic": st.get("statistic", "Freq. of Parent")}
                    if st.get("param"):
                        ann["param"] = st["param"]
                    if st.get("percentile") is not None:
                        ann["percentile"] = st["percentile"]
                    if (st.get("path") or "") != item.pop_path:
                        ann["pop_path"] = st.get("path") or ""
                    anns = item_annotations(item)
                    if ann not in anns:
                        before = caption_height(item)
                        item.props["annotations"] = anns + [ann]
                        item.h += caption_height(item) - before  # the plot keeps its size
                        changed = True
            if changed:
                target.sync_from_model()
                target.update()
                self._notify_now()
            return [item]
        pw, _ = lay.page_size
        m = lay.margin
        x0 = max(0.0, scene_pos.x())
        x, y = x0, max(0.0, scene_pos.y())
        created: list[LayoutItem] = []
        row_h = 0.0

        def place(item: LayoutItem):
            nonlocal x, y, row_h
            item.x, item.y = (self.snap(x), self.snap(y)) if self.snap_to_grid else (x, y)
            if created and item.x + item.w > pw - m + 1:
                x = m if x0 + item.w > pw - m else x0
                y += row_h + SPACING
                row_h = 0.0
                item.x, item.y = (self.snap(x), self.snap(y)) if self.snap_to_grid else (x, y)
            lay.items.append(item)
            self._add_scene_item(item, len(lay.items) - 1)
            created.append(item)
            row_h = max(row_h, item.h)
            x = item.x + item.w + SPACING

        for e, samples in resolved:
            sample = samples[0]
            for path in e["paths"]:
                p = sample.root.find(path) if path else sample.root
                if p is not None:
                    place(self._make_plot_item(sample, p, x, y))
            for st in e["stats"]:
                path = st.get("path") or ""
                p = sample.root.find(path) if path else sample.root
                name = p.display_name if p is not None else (path.rsplit("/", 1)[-1] or "Ungated")
                label = layout_text.statistic_label(st.get("statistic", "Count"), st.get("param"), st.get("percentile"))
                token = stat_token(st.get("statistic", "Count"), st.get("param"), st.get("percentile"))
                item = LayoutItem(kind="text", w=200, h=24, text=f"{name} {label}: {token}", font_size=11, show_caption=False,
                                  sample_id=sample.id, pop_path=path)
                place(item)
        if not created:
            return []
        self.scene.clearSelection()
        for it in created:
            self._scene_items[it.id].setSelected(True)
        self._notify_now()
        return created

    def add_text(self, text: str = "Text", pos: QPointF | None = None, edit: bool = True) -> LayoutItem:
        lay = self.current_layout()
        if pos is None:
            pos = self._free_position(200, 24)
        item = LayoutItem(kind="text", x=pos.x(), y=pos.y(), w=200, h=24, text=text, font_size=14, show_caption=False)
        lay.items.append(item)
        gi = self._add_scene_item(item, len(lay.items) - 1)
        self.scene.clearSelection()
        gi.setSelected(True)
        self._notify_now()
        if edit and self.isVisible():
            self.view.setFocus()
            gi.start_editing()
        return item

    def remove_items(self, items: list[LayoutItem]):
        lay = self.current_layout()
        ids = {i.id for i in items}
        if not ids:
            return
        lay.items = [i for i in lay.items if i.id not in ids]
        for iid in ids:
            gi = self._scene_items.pop(iid, None)
            if gi is not None:
                self.scene.removeItem(gi)
        self._notify_now()

    def delete_selected(self):
        self.remove_items(self.selected_layout_items())

    def duplicate_items(self, items: list[LayoutItem]) -> list[LayoutItem]:
        lay = self.current_layout()
        out = []
        for it in items:
            c = _copy_item(it)
            c.x += SPACING
            c.y += SPACING
            lay.items.append(c)
            self._add_scene_item(c, len(lay.items) - 1)
            out.append(c)
        if out:
            self.scene.clearSelection()
            for c in out:
                self._scene_items[c.id].setSelected(True)
            self._notify_now()
        return out

    def duplicate_selected(self) -> list[LayoutItem]:
        return self.duplicate_items(self.selected_layout_items())

    def bring_to_front(self, items: list[LayoutItem], front: bool = True):
        lay = self.current_layout()
        ids = {i.id for i in items}
        moved = [i for i in lay.items if i.id in ids]
        rest = [i for i in lay.items if i.id not in ids]
        lay.items = rest + moved if front else moved + rest
        self._restack()
        self._notify_now()

    def _sync_items(self, items: list[LayoutItem]):
        for it in items:
            gi = self._scene_items.get(it.id)
            if gi is not None:
                gi.sync_from_model()

    def arrange_grid(self, columns: int = 2):
        """Lay out the plot items in reading order on a grid with equal cell
        size and spacing inside the page margins."""
        lay = self.current_layout()
        columns = max(1, int(columns))
        plots = _reading_order([i for i in lay.items if i.kind == "plot"])
        if not plots:
            return
        pw, _ = lay.page_size
        m = lay.margin
        avail = pw - 2 * m
        cell = (avail - (columns - 1) * SPACING) / columns
        y = m
        for r in range(0, len(plots), columns):
            row = plots[r : r + columns]
            row_h = 0.0
            for c, it in enumerate(row):
                it.w = cell
                it.h = cell + caption_height(it) + title_height(it)
                it.x = m + c * (cell + SPACING)
                it.y = y
                row_h = max(row_h, it.h)
            y += row_h + SPACING
        self._sync_items(plots)
        self._notify_now()

    def _targets(self) -> list[LayoutItem]:
        return self.selected_layout_items()

    def _commit_geometry(self, items: list[LayoutItem]):
        """Show moved / resized items and record the change (one undo step)."""
        self.end_edit_burst()
        self._sync_items(items)
        self._notify_now()

    def _align(self, fn) -> bool:
        items = self._targets()
        if len(items) < 2:
            return False
        fn(items)
        self._commit_geometry(items)
        return True

    def align_left(self):
        return self._align(lambda items: [setattr(i, "x", min(j.x for j in items)) for i in items])

    def align_right(self):
        def fn(items):
            right = max(j.x + j.w for j in items)
            for i in items:
                i.x = right - i.w
        return self._align(fn)

    def align_top(self):
        return self._align(lambda items: [setattr(i, "y", min(j.y for j in items)) for i in items])

    def align_bottom(self):
        def fn(items):
            bottom = max(j.y + j.h for j in items)
            for i in items:
                i.y = bottom - i.h
        return self._align(fn)

    def align_hcenter(self):
        """Centres on one vertical line (the middle of the selection)."""
        def fn(items):
            cx = (min(j.x for j in items) + max(j.x + j.w for j in items)) / 2.0
            for i in items:
                i.x = cx - i.w / 2.0
        return self._align(fn)

    def align_vcenter(self):
        """Centres on one horizontal line (the middle of the selection)."""
        def fn(items):
            cy = (min(j.y for j in items) + max(j.y + j.h for j in items)) / 2.0
            for i in items:
                i.y = cy - i.h / 2.0
        return self._align(fn)

    def _distribute(self, horizontal: bool) -> bool:
        items = self._targets()
        if len(items) < 3:
            return False
        pos, size = ("x", "w") if horizontal else ("y", "h")
        order = sorted(items, key=lambda i: (getattr(i, pos), getattr(i, pos) + getattr(i, size) / 2.0))
        start = getattr(order[0], pos)
        end = max(getattr(i, pos) + getattr(i, size) for i in order)
        gap = (end - start - sum(getattr(i, size) for i in order)) / (len(order) - 1)
        v = start
        for i in order:
            setattr(i, pos, v)
            v += getattr(i, size) + gap
        self._commit_geometry(items)
        return True

    def distribute_horizontally(self):
        """Equal horizontal gaps; the left-most and right-most edges stay."""
        return self._distribute(True)

    def distribute_vertically(self):
        """Equal vertical gaps; the top-most and bottom-most edges stay."""
        return self._distribute(False)

    def _match_size(self, width: bool, height: bool) -> bool:
        items = self._targets()
        if len(items) < 2:
            return False
        ref = items[0]
        if not width and ref.kind == "text":
            ref = next((i for i in items if i.kind != "text"), ref)
        for i in items:
            if i is ref:
                continue
            if width:
                i.w = ref.w
            if height and i.kind != "text":  # a text box is as high as its text
                i.h = ref.h
        self._commit_geometry(items)
        return True

    def same_width(self):
        return self._match_size(True, False)

    def same_height(self):
        return self._match_size(False, True)

    def same_size(self):
        items = [i for i in self._targets() if i.kind == "plot"]
        if len(items) > 1:
            ref = items[0]
            for i in items[1:]:
                i.w, i.h = ref.w, ref.h
            texts = [i for i in self._targets() if i.kind == "text"]
            for t in texts:
                t.w = ref.w
            self._commit_geometry(items + texts)
            return True
        return self._match_size(True, True)

    # -------------------------------------------------------- clipboard / undo
    def undo(self):
        self._commit_text_edit()
        self.end_edit_burst()
        self.controller.undo()

    def redo(self):
        self._commit_text_edit()
        self.end_edit_burst()
        self.controller.redo()

    def select_all(self):
        for gi in self._scene_items.values():
            gi.setSelected(True)
        return list(self.current_layout().items)

    _shared_clipboard: dict | None = None  # copies of every editor window (fallback when the system clipboard drops them)

    def copy_items(self, items: list[LayoutItem]) -> QMimeData:
        """Put `items` on the clipboard: as layout objects (paste into any
        layout) and as an image (paste into PowerPoint / Word)."""
        payload = {"items": [i.to_dict() for i in items], "layout": self._layout.id if self._layout else None}
        md = QMimeData()
        md.setData(ITEMS_MIME, QByteArray(json.dumps(payload).encode("utf-8")))
        img = self.items_image(items, 2.0)
        md.setImageData(img)
        md.setText("\n".join(i.text if i.kind == "text" else f"{i.kind}: {i.pop_path or 'Ungated'}" for i in items))
        LayoutEditorWindow._shared_clipboard = payload
        self._last_copy = payload
        QGuiApplication.clipboard().setMimeData(md)
        return md

    def copy_selected(self):
        items = self.selected_layout_items()
        if not items:
            return self.copy_page_image()
        self.copy_items(items)
        self.statusBar().showMessage(f"Copied {len(items)} object(s)", 3000)
        return items

    def cut_selected(self):
        items = self.selected_layout_items()
        if items:
            self.copy_items(items)
            self.remove_items(items)
        return items

    def _clipboard_payload(self) -> dict | None:
        cb = QGuiApplication.clipboard()
        md = cb.mimeData()
        if md is not None and md.hasFormat(ITEMS_MIME):
            try:
                return json.loads(bytes(md.data(ITEMS_MIME)).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
        if LayoutEditorWindow._shared_clipboard is not None and cb.ownsClipboard():
            return LayoutEditorWindow._shared_clipboard
        return None

    def paste_items(self, payload: dict | None = None) -> list[LayoutItem]:
        """Paste copied layout objects into the current layout (new ids,
        shifted when they would land exactly on the originals). One undo
        step; the pasted objects are selected."""
        payload = payload or self._clipboard_payload()
        if not payload or not payload.get("items"):
            return []
        lay = self.current_layout()
        self._commit_text_edit()
        self.end_edit_burst()
        new = []
        for d in payload["items"]:
            d = dict(d)
            d["id"] = new_id()
            new.append(LayoutItem.from_dict(d))
        taken = {(round(i.x, 3), round(i.y, 3)) for i in lay.items}
        shift = 0.0
        while any((round(i.x + shift, 3), round(i.y + shift, 3)) in taken for i in new) and shift < 50 * SPACING:
            shift += SPACING
        for it in new:
            it.x += shift
            it.y += shift
            lay.items.append(it)
            self._add_scene_item(it, len(lay.items) - 1)
        self.scene.clearSelection()
        for it in new:
            self._scene_items[it.id].setSelected(True)
        self._notify_now()
        self.statusBar().showMessage(f"Pasted {len(new)} object(s)", 3000)
        return new

    # -------------------------------------------------------------- editing
    def edit_item(self, item: LayoutItem) -> bool:
        if item.kind == "text":
            gi = self._scene_items.get(item.id)
            if isinstance(gi, TextGraphicsItem):
                gi.start_editing()
            return False
        kind = self._item_kinds.get(item.kind)
        if kind is not None:
            if kind.get("dialog") is None:
                return False
            if kind["dialog"](self, item, self):
                self._sync_items([item])
                self._notify_now()
                return True
            return False
        return self.edit_plot(item)

    def edit_selected(self):
        items = self.selected_layout_items()
        if items:
            return self.edit_item(items[0])
        return False

    def edit_plot(self, item: LayoutItem, others: list[LayoutItem] | None = None) -> bool:
        """Layout Plot Options for `item`; the display options changed there
        also go to `others` (default: the other selected plots). One undo
        step."""
        if others is None:
            others = [i for i in self.selected_layout_items() if i.kind == "plot" and i is not item]
        dlg = PlotOptionsDialog(self, item, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        return self.apply_plot_dialog(dlg, others)

    def apply_plot_dialog(self, dlg: PlotOptionsDialog, others: list[LayoutItem] | None = None) -> bool:
        item = dlg.item
        changed = dlg.changed_display_options()
        reset = dlg._reset_display
        touched = [item]
        old_extra = caption_height(item) + title_height(item)
        dlg.apply()
        new_extra = caption_height(item) + title_height(item)
        item.h = max(MIN_SIZE, item.h + new_extra - old_extra)
        for o in others or []:
            before = caption_height(o) + title_height(o)
            dlg.apply_display_options(o, changed, reset)
            o.h = max(MIN_SIZE, o.h + caption_height(o) + title_height(o) - before)
            touched.append(o)
        self.end_edit_burst()
        self._sync_items(touched)
        self._notify_now()
        return True

    def set_font_size(self, items: list[LayoutItem], size: int):
        for it in items:
            if it.kind == "text":
                it.font_size = int(size)
        self._sync_items(items)
        self._notify_now()

    def set_text_format(self, items: list[LayoutItem] | None = None, **fmt) -> list[LayoutItem]:
        """Bold / italic / underline / color / align / font_family of text
        items (stored in item.props). One undo step."""
        texts = [i for i in (self.selected_layout_items() if items is None else items) if i.kind == "text"]
        if not texts:
            return []
        for it in texts:
            for k, v in fmt.items():
                if k in ("bold", "italic", "underline"):
                    v = bool(v)
                    if v:
                        it.props[k] = True
                    else:
                        it.props.pop(k, None)
                elif k == "align":
                    if v in ("left", None):
                        it.props.pop("align", None)
                    elif v in TEXT_ALIGN:
                        it.props["align"] = v
                elif k == "color":
                    c = QColor(v) if v else QColor("#000000")
                    if c.isValid() and c.name() != "#000000":
                        it.props["color"] = c.name()
                    else:
                        it.props.pop("color", None)
                elif k == "font_family":
                    if v:
                        it.props["font_family"] = str(v)
                    else:
                        it.props.pop("font_family", None)
                elif k == "font_size" and v:
                    it.font_size = int(v)
        self.end_edit_burst()
        self._sync_items(texts)
        self._notify_now()
        return texts

    def toggle_text_format(self, key: str) -> list[LayoutItem]:
        """Bold / italic / underline on or off for the selected text boxes
        (the opposite of the first one's state, as word processors do)."""
        texts = [i for i in self.selected_layout_items() if i.kind == "text"]
        if not texts:
            self._update_text_actions()
            return []
        out = self.set_text_format(texts, **{key: not text_format(texts[0])[key]})
        self._update_text_actions()
        return out

    def text_color_dialog(self):
        texts = [i for i in self.selected_layout_items() if i.kind == "text"]
        if not texts:
            return
        c = QColorDialog.getColor(QColor(text_format(texts[0])["color"]), self, "Text Colour")
        if c.isValid():
            self.set_text_format(texts, color=c.name())

    def text_font_dialog(self):
        texts = [i for i in self.selected_layout_items() if i.kind == "text"]
        if not texts:
            return
        f = text_font(texts[0])
        f.setPointSizeF(float(texts[0].font_size))
        ok, font = QFontDialog.getFont(f, self, "Font")
        if isinstance(ok, QFont):  # PySide returns (ok, font) or (font, ok) depending on the version
            ok, font = font, ok
        if ok:
            size = font.pointSizeF() if font.pointSizeF() > 0 else font.pixelSize()
            self.set_text_format(texts, font_family=font.family(), bold=font.bold(), italic=font.italic(),
                                 underline=font.underline(), font_size=int(round(size)) if size and size > 0 else None)

    def insert_token(self, token: str) -> LayoutItem | None:
        """Insert a keyword / statistic token: at the cursor of the text being
        edited, at the end of the selected text box, or as a new text box."""
        focus = self.scene.focusItem()
        if isinstance(focus, TextGraphicsItem) and focus.editing:
            focus.textCursor().insertText(token)
            return focus.item
        texts = [i for i in self.selected_layout_items() if i.kind == "text"]
        if texts:
            it = texts[0]
            it.text = f"{it.text} {token}".strip() if it.text else token
            self._sync_items([it])
            self.end_edit_burst()
            self._notify_now()
            return it
        plots = [i for i in self.selected_layout_items() if i.kind == "plot"]
        item = self.add_text(token, edit=False)
        if plots:
            item.sample_id, item.pop_path = plots[0].sample_id, plots[0].pop_path
            self._sync_items([item])
            self._notify_now()
        return item

    def add_annotation(self, items: list[LayoutItem], entry: dict) -> int:
        n = 0
        for it in items:
            if it.kind != "plot":
                continue
            anns = item_annotations(it)
            if entry in anns:
                continue
            before = caption_height(it)
            it.props["annotations"] = anns + [dict(entry)]
            it.h += caption_height(it) - before
            n += 1
        if n:
            self.end_edit_burst()
            self._sync_items(items)
            self._notify_now()
        return n

    def _dialog_sample(self) -> tuple[Sample | None, str | None]:
        sel = self.selected_layout_items()
        it = sel[0] if sel else None
        ctx = self.text_context(it)
        return ctx.sample, (it.pop_path if it is not None else None)

    def insert_keyword_dialog(self):
        sample, _ = self._dialog_sample()
        dlg = StatisticDialog(self, sample, None, self, keyword_mode=True)
        if dlg.exec():
            plots = [i for i in self.selected_layout_items() if i.kind == "plot"]
            if plots and not any(i.kind == "text" for i in self.selected_layout_items()):
                self.add_annotation(plots, dlg.entry())
            else:
                self.insert_token(dlg.token())

    def insert_statistic_dialog(self):
        sample, path = self._dialog_sample()
        dlg = StatisticDialog(self, sample, None, self)
        if dlg.exec():
            plots = [i for i in self.selected_layout_items() if i.kind == "plot"]
            if plots and not any(i.kind == "text" for i in self.selected_layout_items()):
                self.add_annotation(plots, dlg.entry())
            else:
                self.insert_token(dlg.token())

    # ------------------------------------------------------------- overlays
    def overlay_samples(self, item: LayoutItem | None = None) -> list[Sample]:
        """Samples offered for overlays: the members of the layout's batch
        group (or the current group), with the plot's own sample first."""
        ws = self.workspace
        group = self._default_batch_group()
        samples = list(ws.members(group)) if group is not None else list(ws.samples)
        own = ws.sample_by_id(item.sample_id) if item is not None and item.sample_id is not None else None
        if own is not None and own not in samples:
            samples.insert(0, own)
        return samples

    def add_overlays(self, item: LayoutItem, entries) -> int:
        n = 0
        for ov in entries:
            entry = self._overlay_entry(ov)
            if entry is None or (entry["sample_id"], entry["pop_path"]) == (item.sample_id, item.pop_path) or entry in item.overlays:
                continue
            item.overlays.append(entry)
            n += 1
        return n

    def overlay_group(self, items: list[LayoutItem]) -> int:
        """Overlay each plot's population of every sample in the batch group."""
        n = 0
        for it in items:
            if it.kind != "plot":
                continue
            entries = []
            for s in self.overlay_samples(it):
                if s.id == it.sample_id or (it.pop_path and s.root.find(it.pop_path) is None):
                    continue
                entries.append({"sample_id": s.id, "pop_path": it.pop_path})
            n += self.add_overlays(it, entries)
        if n:
            self.end_edit_burst()
            self._sync_items(items)
            self._notify_now()
        return n

    def overlay_group_selected(self):
        return self.overlay_group(self.selected_layout_items())

    def overlay_dialog(self):
        from .dialogs import OverlayDialog

        plots = [i for i in self.selected_layout_items() if i.kind == "plot"]
        if not plots:
            self.statusBar().showMessage("Select a plot to overlay populations on", 3000)
            return
        item = plots[0]
        current = []
        for ov in item.overlays:
            s = self.workspace.sample_by_id(ov.get("sample_id"))
            p = (s.root.find(ov.get("pop_path")) if ov.get("pop_path") else s.root) if s is not None else None
            if p is not None:
                current.append((s, p))
        dlg = OverlayDialog(self.overlay_samples(item), current, self)
        if dlg.exec():
            item.overlays = []
            self.add_overlays(item, dlg.values())
            self.end_edit_burst()
            self._sync_items([item])
            self._notify_now()

    def open_graph(self, item: LayoutItem):
        """Open the plot's population in a graph window (to adjust gates)."""
        sample, pop = self.resolve(item)
        opener = getattr(self.controller, "open_graph", None)
        if sample is None or pop is None or opener is None:
            return None
        return opener(sample, pop)

    def open_graph_selected(self):
        plots = [i for i in self.selected_layout_items() if i.kind == "plot"]
        return self.open_graph(plots[0]) if plots else None

    # ---------------------------------------------------------- context menu
    def item_context_menu(self, gi, screen_pos):
        items = self.selected_layout_items() or [gi.item]
        a = self.acts
        menu = QMenu(self)
        item = gi.item
        if item.kind == "text":
            menu.addAction(icon("rename"), "Edit Text", gi.start_editing)
            sizes = menu.addMenu("Font Size")
            grp = QActionGroup(sizes)
            for sz in FONT_SIZES:
                act = sizes.addAction(f"{sz} pt", lambda sz=sz: self.set_font_size(items, sz))
                act.setCheckable(True)
                act.setChecked(sz == item.font_size)
                grp.addAction(act)
            fmt = text_format(item)
            for key, label in (("bold", "Bold"), ("italic", "Italic"), ("underline", "Underline")):
                act = menu.addAction(label, lambda key=key, on=not fmt[key]: self.set_text_format(items, **{key: on}))
                act.setCheckable(True)
                act.setChecked(fmt[key])
            align = menu.addMenu("Alignment")
            for key, label in (("left", "Left"), ("center", "Centre"), ("right", "Right")):
                act = align.addAction(label, lambda key=key: self.set_text_format(items, align=key))
                act.setCheckable(True)
                act.setChecked(fmt["align"] == key)
            menu.addAction(a["text_font"])
            menu.addAction(a["text_color"])
            menu.addAction(a["insert_keyword"])
            menu.addAction(a["insert_statistic"])
        elif item.kind == "plot":
            menu.addAction(icon("rename"), "Edit…", lambda: self.edit_plot(item, [i for i in items if i.kind == "plot" and i is not item]))
            menu.addAction(a["open_graph"])
            menu.addAction(a["overlay"])
            menu.addAction(a["overlay_group"])
            menu.addAction(icon("statistic"), "Annotate with Statistic…", self.insert_statistic_dialog)
            menu.addAction(icon("keyword"), "Annotate with Keyword…", self.insert_keyword_dialog)
        else:
            menu.addAction(icon("rename"), "Edit…", lambda: self.edit_item(item))
        menu.addSeparator()
        menu.addAction(a["cut"])
        menu.addAction(a["copy"])
        menu.addAction(a["paste"])
        menu.addAction(icon("copy"), "Duplicate", lambda: self.duplicate_items(items))
        menu.addAction(icon("delete"), "Remove", lambda: self.remove_items(items))
        menu.addSeparator()
        if len(items) > 1:
            self._arrange_menu(menu.addMenu("Arrange"))
        menu.addAction(icon("up"), "Bring to Front", lambda: self.bring_to_front(items, True))
        menu.addAction(icon("down"), "Send to Back", lambda: self.bring_to_front(items, False))
        menu.exec(screen_pos)

    def page_context_menu(self, screen_pos, scene_pos: QPointF | None = None):
        a = self.acts
        menu = QMenu(self)
        menu.addAction(a["paste"])
        menu.addAction(a["select_all"])
        menu.addSeparator()
        menu.addAction(icon("text"), "Add Text Here", lambda: self.add_text(pos=scene_pos))
        menu.addAction(a["insert_keyword"])
        menu.addAction(a["insert_statistic"])
        menu.addSeparator()
        menu.addAction(a["page_setup"])
        menu.addAction(a["copy_page_image"])
        menu.exec(screen_pos)

    # ----------------------------------------------------------------- view
    def zoom_by(self, factor: float):
        current = self.view.transform().m11()
        target = min(8.0, max(0.1, current * factor))
        self.view.scale(target / current, target / current)

    def fit_page(self):
        w, h = self.page_size()
        self.view.fitInView(QRectF(-12, -12, w + 24, h + 24), Qt.AspectRatioMode.KeepAspectRatio)

    def set_grid(self, on: bool):
        self.show_grid = bool(on)
        self.snap_to_grid = bool(on)
        if self.grid_action.isChecked() != bool(on):
            self.grid_action.setChecked(bool(on))
        self.page_item.update()
