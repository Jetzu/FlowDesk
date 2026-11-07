"""Layout Editor: arrange plots of populations and free text on printable
pages, overlay populations by dropping them onto a plot, and export pages
(PNG / PDF / SVG), print them, or produce a batch PDF with one page per
sample of a group.

Coordinates: one scene unit is one page point (1/72 inch). The persisted model
is flowdesk.core.layouts.Layout / LayoutItem; every edit made on the canvas is
written back into those objects and announced through
controller.notify_layouts() (coalesced with a short timer while dragging).
Every such announcement also records an undo step ("edit layout"), see
SectionUndoRecorder.
"""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict

from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSizeF, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetricsF,
    QImage,
    QKeySequence,
    QPageLayout,
    QPageSize,
    QPainter,
    QPdfWriter,
    QPen,
    QPixmap,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QRadioButton,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.gates import new_id
from ..core.layouts import PAGE_SIZES, Layout, LayoutItem
from ..core.model import PLOT_TYPES, GraphSettings, Group, Population, Sample
from .controller import POPULATION_MIME
from .icons import icon
from .renderer import RenderOptions, render_plot

MARGIN = 36.0  # page margin guide (points)
GRID = 8.0  # snap grid (points)
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


# ------------------------------------------------------------------ helpers
def parse_population_mime(mime) -> tuple[int | None, list[str]]:
    """Decode a POPULATION_MIME payload -> (sample_id, paths)."""
    if mime is None or not mime.hasFormat(POPULATION_MIME):
        return None, []
    try:
        data = json.loads(bytes(mime.data(POPULATION_MIME)).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, []
    sid = data.get("sample_id")
    paths = data.get("paths") or []
    if isinstance(paths, str):
        paths = [paths]
    return sid, [str(p) for p in paths]


def caption_height(item: LayoutItem) -> float:
    return CAPTION_H if item.show_caption else 0.0


def title_height(item: LayoutItem) -> float:
    return TITLE_H if item.show_title else 0.0


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


def _snap(v: float) -> float:
    return round(v / GRID) * GRID


def _copy_item(item: LayoutItem) -> LayoutItem:
    d = item.to_dict()
    d["id"] = new_id()
    return LayoutItem.from_dict(d)


def _reading_order(items: list[LayoutItem]) -> list[LayoutItem]:
    return sorted(items, key=lambda i: (round(i.y / 24.0), i.x))


# --------------------------------------------------------------------- undo
class SectionUndoRecorder:
    """Undo steps for an editor that changes one workspace section
    ("layouts" or "tables") in place and announces the change afterwards.

    The editors write into the model objects directly (a drag moves the
    LayoutItem as it goes), so there is no single place to call
    controller.begin_change() *before* a change. The recorder keeps the
    section as it was after the last recorded change (the baseline); record()
    pushes one undo step that brings back that baseline plus the present
    gating state (unchanged by an editor). Steps recorded with coalesce=True
    within COALESCE_SECONDS of the previous coalescing step of the same
    recorder merge into it, so one drag is one undo step."""

    COALESCE_SECONDS = 1.0

    def __init__(self, controller, section: str, label: str):
        self.controller = controller
        self.section = section
        self.label = label
        self._baseline: list[dict] | None = None
        self._pushed: dict | None = None
        self._pushed_coalescing = False
        self._time = 0.0
        self.reset()

    def _state(self) -> list[dict]:
        return [o.to_dict() for o in getattr(self.controller.workspace, self.section, [])]

    def reset(self):
        """Take the present state as the one the next step returns to (after
        undo / redo, a replaced workspace or a change made elsewhere)."""
        self._baseline = self._state()

    def record(self, coalesce: bool = False) -> bool:
        """Call after a change, before announcing it. Returns whether a new
        undo step was pushed."""
        c = self.controller
        current = self._state()
        before, self._baseline = self._baseline, current
        if before is None or before == current:
            return False
        if getattr(c, "_undo_group_depth", 0):
            # inside an open undo group: only make sure it covers the section
            c.begin_change(self.label, kind=self.section)
            return False
        now = time.monotonic()
        if (
            coalesce
            and self._pushed_coalescing
            and self._pushed is not None
            and c.undo_top_snapshot() is self._pushed
            and now - self._time < self.COALESCE_SECONDS
        ):
            self._time = now  # same burst: the open step already returns to before it
            return False
        from ..core.serialization import gating_snapshot

        snap = gating_snapshot(c.workspace, self.section)
        snap[self.section] = before
        c.push_undo(self.label, snap)
        self._pushed, self._pushed_coalescing, self._time = snap, coalesce, now
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
    def __init__(self, layout: Layout, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Page Setup")
        form = QFormLayout(self)
        self.page_combo = QComboBox()
        self.page_combo.addItems(list(PAGE_SIZES.keys()))
        self.page_combo.setCurrentText(layout.page if layout.page in PAGE_SIZES else "A4")
        form.addRow("Page size:", self.page_combo)
        self.portrait = QRadioButton("Portrait")
        self.landscape = QRadioButton("Landscape")
        (self.landscape if layout.landscape else self.portrait).setChecked(True)
        row = QHBoxLayout()
        row.addWidget(self.portrait)
        row.addWidget(self.landscape)
        form.addRow("Orientation:", row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> tuple[str, bool]:
        return self.page_combo.currentText(), self.landscape.isChecked()


class PlotOptionsDialog(QDialog):
    """Edit a plot item: plot type, axes, caption/title and overlays."""

    def __init__(self, editor: "LayoutEditorWindow", item: LayoutItem, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Layout Plot Options")
        self.item = item
        self.editor = editor
        self.overlays = [dict(o) for o in item.overlays]
        sample = editor.workspace.sample_by_id(item.sample_id) if item.sample_id is not None else None
        params = sample.parameter_names() if sample is not None else []
        form = QFormLayout(self)
        self.type_combo = QComboBox()
        for key, label in PLOT_TYPES:
            self.type_combo.addItem(label, key)
        idx = self.type_combo.findData(item.graph.plot_type)
        self.type_combo.setCurrentIndex(max(0, idx))
        form.addRow("Plot type:", self.type_combo)
        self.x_combo = QComboBox()
        self.x_combo.addItems(params)
        if item.graph.x and item.graph.x not in params:
            self.x_combo.addItem(item.graph.x)
        if item.graph.x:
            self.x_combo.setCurrentText(item.graph.x)
        form.addRow("X axis:", self.x_combo)
        self.y_combo = QComboBox()
        self.y_combo.addItem(HISTOGRAM_ENTRY)
        self.y_combo.addItems(params)
        if item.graph.y and item.graph.y not in params:
            self.y_combo.addItem(item.graph.y)
        if item.graph.y and item.graph.plot_type != "histogram":
            self.y_combo.setCurrentText(item.graph.y)
        else:
            self.y_combo.setCurrentIndex(0)
        form.addRow("Y axis:", self.y_combo)
        self.caption_check = QCheckBox("Show caption (sample, population, count)")
        self.caption_check.setChecked(item.show_caption)
        form.addRow(self.caption_check)
        self.title_check = QCheckBox("Show title")
        self.title_check.setChecked(item.show_title)
        form.addRow(self.title_check)
        # overlay render options; stored in item.graph.options like a graph
        # window stores them in the population's graph settings
        effective = RenderOptions.from_prefs(editor.controller.preferences, item.graph)
        self.legend_check = QCheckBox("Show overlay legend")
        self.legend_check.setChecked(bool(effective.show_legend))
        form.addRow(self.legend_check)
        self.offset_check = QCheckBox("Offset (staggered) overlaid histograms")
        self.offset_check.setChecked(bool(effective.histogram_offset))
        form.addRow(self.offset_check)
        self.overlay_list = QListWidget()
        self.overlay_list.setMinimumHeight(90)
        form.addRow("Overlays:", self.overlay_list)
        self._fill_overlays()
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.type_combo.currentIndexChanged.connect(self._type_changed)

    def _type_changed(self):
        if self.type_combo.currentData() == "histogram":
            self.y_combo.setCurrentIndex(0)
        elif self.y_combo.currentIndex() == 0 and self.y_combo.count() > 2:
            self.y_combo.setCurrentIndex(2 if self.y_combo.count() > 2 else 1)

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

    def apply(self):
        it = self.item
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
        it.overlays = self.overlays


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
            painter.setPen(QPen(QColor("#eef0f3"), 0))
            x = GRID
            while x < w:
                painter.drawLine(QPointF(x, 0), QPointF(x, h))
                x += GRID
            y = GRID
            while y < h:
                painter.drawLine(QPointF(0, y), QPointF(w, y))
                y += GRID
        pen = QPen(QColor("#c9d7ef"), 0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(MARGIN, MARGIN, w - 2 * MARGIN, h - 2 * MARGIN))


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
        self.editor.paint_plot(painter, self.item, None, SCENE_SCALE, allow_stale=self._resize_handle is not None)
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
            return QPointF(_snap(p.x()), _snap(p.y()))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            p = self.pos()
            if (p.x(), p.y()) != (self.item.x, self.item.y):
                self.item.x, self.item.y = p.x(), p.y()
                self.editor.schedule_notify()
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
            self.editor.schedule_notify()
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
            left, right, top, bottom = (_snap(v) for v in (left, right, top, bottom))
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
        self.editor.edit_plot(self.item)
        event.accept()

    def contextMenuEvent(self, event):
        if not self.isSelected():
            self.scene().clearSelection()
            self.setSelected(True)
        self.editor.item_context_menu(self, event.screenPos())
        event.accept()


class TextGraphicsItem(QGraphicsTextItem):
    """Free text on the page; double-click to edit inline."""

    def __init__(self, editor: "LayoutEditorWindow", item: LayoutItem):
        super().__init__(item.text)
        self.editor = editor
        self.item = item
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setDefaultTextColor(QColor("#000000"))
        self.apply_font()
        self.setPos(item.x, item.y)

    def apply_font(self):
        self.setFont(_px_font(self.item.font_size))
        self.setTextWidth(self.item.w if self.item.w > 0 else -1)
        self.item.h = self.boundingRect().height()

    def sync_from_model(self):
        if self.toPlainText() != self.item.text:
            self.setPlainText(self.item.text)
        self.apply_font()
        self.setPos(self.item.x, self.item.y)

    @property
    def editing(self) -> bool:
        return self.textInteractionFlags() != Qt.TextInteractionFlag.NoTextInteraction

    def start_editing(self):
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        cursor = self.textCursor()
        cursor.select(cursor.SelectionType.Document)
        self.setTextCursor(cursor)

    def finish_editing(self):
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        text = self.toPlainText()
        h = self.boundingRect().height()
        if text != self.item.text or h != self.item.h:
            self.item.text = text
            self.item.h = h
            self.editor.schedule_notify()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.editor.snap_to_grid:
            p = QPointF(value)
            return QPointF(_snap(p.x()), _snap(p.y()))
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            p = self.pos()
            if (p.x(), p.y()) != (self.item.x, self.item.y):
                self.item.x, self.item.y = p.x(), p.y()
                self.editor.schedule_notify()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if not self.editing:
            self.start_editing()
        super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self.editing:
            self.finish_editing()

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

    def wheelEvent(self, event):
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.editor.zoom_by(factor)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        focus = self.scene().focusItem()
        editing = isinstance(focus, TextGraphicsItem) and focus.editing
        if not editing and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.editor.delete_selected()
            event.accept()
            return
        if not editing and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = GRID if self.editor.snap_to_grid else (10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0)
            dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(event.key(), 0.0)
            dy = {Qt.Key.Key_Up: -step, Qt.Key.Key_Down: step}.get(event.key(), 0.0)
            for it in self.scene().selectedItems():
                it.moveBy(dx, dy)
            event.accept()
            return
        super().keyPressEvent(event)


# -------------------------------------------------------------------- window
class LayoutEditorWindow(QMainWindow):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Layout Editor")
        self.setWindowIcon(icon("layout-editor"))
        self.resize(1100, 800)
        self.cache = PlotImageCache()
        self.show_grid = False
        self.snap_to_grid = False
        self._layout: Layout | None = None
        self._suppress = False
        self._scene_items: dict[str, QGraphicsItem] = {}
        self._notify_timer = QTimer(self)
        self._notify_timer.setSingleShot(True)
        self._notify_timer.setInterval(150)
        self._notify_timer.timeout.connect(lambda: self._emit_layouts(coalesce=True))
        self.undo_recorder = SectionUndoRecorder(controller, "layouts", "edit layout")

        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor("#8a8f96"))
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
        lv.addWidget(self.layout_list)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([180, 920])
        self.setCentralWidget(splitter)
        self._build_toolbar()
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

    # ------------------------------------------------------------ toolbar
    def _action(self, tb: QToolBar, icon_name: str, text: str, slot, tip: str | None = None, shortcut=None) -> QAction:
        a = QAction(icon(icon_name), text, self)
        a.setToolTip(tip or text)
        if shortcut is not None:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(lambda _=False: slot())
        tb.addAction(a)
        return a

    def _build_toolbar(self):
        tb = QToolBar("Layout")
        tb.setObjectName("layoutToolbar")
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(tb)
        self._action(tb, "new", "New Layout", self.new_layout)
        self._action(tb, "rename", "Rename Layout", self.rename_layout)
        self._action(tb, "copy", "Duplicate Layout", self.duplicate_layout)
        self._action(tb, "delete", "Delete Layout", self.delete_layout)
        tb.addSeparator()
        self._action(tb, "text", "Add Text", self.add_text)
        self._action(tb, "minus", "Delete Item(s)", self.delete_selected)
        self._action(tb, "plus", "Duplicate Item", self.duplicate_selected, shortcut="Ctrl+D")
        tb.addSeparator()
        grid_btn = QToolButton()
        grid_btn.setIcon(icon("grid"))
        grid_btn.setToolTip("Arrange plots in a grid")
        grid_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(grid_btn)
        for n in (2, 3, 4):
            menu.addAction(f"{n} columns", lambda n=n: self.arrange_grid(n))
        grid_btn.setMenu(menu)
        tb.addWidget(grid_btn)
        self._action(tb, "left", "Align Left", self.align_left)
        self._action(tb, "up", "Align Top", self.align_top)
        self._action(tb, "check", "Same Size", self.same_size)
        tb.addSeparator()
        self._action(tb, "page", "Page Setup…", self.page_setup)
        self._action(tb, "zoom", "Zoom In", lambda: self.zoom_by(1.25), shortcut=QKeySequence.StandardKey.ZoomIn)
        self._action(tb, "minus", "Zoom Out", lambda: self.zoom_by(0.8), shortcut=QKeySequence.StandardKey.ZoomOut)
        self._action(tb, "page", "Fit Page", self.fit_page)
        self.grid_action = QAction(icon("grid"), "Show Grid / Snap to Grid", self)
        self.grid_action.setCheckable(True)
        self.grid_action.toggled.connect(self.set_grid)
        tb.addAction(self.grid_action)
        tb.addSeparator()
        self._action(tb, "export", "Export…", self.export_dialog)
        self._action(tb, "print", "Print…", self.print_layout, shortcut=QKeySequence.StandardKey.Print)
        tb.addSeparator()
        tb.addWidget(QLabel(" Batch: "))
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(120)
        self.group_combo.currentTextChanged.connect(self._batch_group_changed)
        tb.addWidget(self.group_combo)
        self._action(tb, "play", "Create batch PDF…", self.batch_pdf_dialog)

    # ----------------------------------------------------------- model glue
    @property
    def workspace(self):
        return self.controller.workspace

    def page_size(self, layout: Layout | None = None) -> tuple[float, float]:
        layout = layout or self._layout
        return layout.page_size if layout is not None else PAGE_SIZES["A4"]

    def schedule_notify(self):
        self._notify_timer.start()

    def flush_notify(self):
        if self._notify_timer.isActive():
            self._notify_timer.stop()
            self._emit_layouts(coalesce=True)

    def _notify_now(self):
        self._notify_timer.stop()
        self._emit_layouts()

    def _emit_layouts(self, coalesce: bool = False):
        # one undo step per edit; drags (timer-driven) merge into one
        self.undo_recorder.record(coalesce)
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
        self.undo_recorder.reset()
        self.refresh()

    def _on_workspace_replaced(self):
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
        event.ignore()
        self.hide()

    def _ensure_layout(self) -> Layout:
        ws = self.workspace
        if not ws.layouts:
            ws.layouts.append(Layout("Layout"))
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
            self.schedule_notify()

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
        gi = TextGraphicsItem(self, item) if item.kind == "text" else PlotGraphicsItem(self, item)
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
        sel = [gi.item for gi in self.scene.selectedItems() if isinstance(gi, (PlotGraphicsItem, TextGraphicsItem))]
        order = {it.id: k for k, it in enumerate(self._layout.items)} if self._layout else {}
        return sorted(sel, key=lambda i: order.get(i.id, 0))

    def plot_item_at(self, pos: QPointF) -> PlotGraphicsItem | None:
        for gi in self.scene.items(pos):
            if isinstance(gi, PlotGraphicsItem) and gi.box().contains(gi.mapFromScene(pos)):
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

    def render_options(self, graph: GraphSettings, scale: float) -> RenderOptions:
        f = 72.0 * scale / IMAGE_DPI
        return RenderOptions.from_prefs(
            self.controller.preferences,
            graph,
            title=None,
            show_axes=True,
            label_font_size=max(4, int(round(7.5 * f))),
            number_font_size=max(4, int(round(6.0 * f))),
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
        result = render_plot(sample, pop, graph, w_px, h_px, self.render_options(graph, scale), overlays=overlays or None)
        self.cache.renders += 1
        img = result.image
        self.cache.put(key, img)
        if scale == SCENE_SCALE:
            item._last_image = (ident, img)  # stale image shown while resizing
        return img

    # ------------------------------------------------------------- painting
    def paint_plot(self, painter: QPainter, item: LayoutItem, sample_override: Sample | None, scale: float, allow_stale: bool = False):
        """Paint a plot item in item-local coordinates (points)."""
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
        font = _px_font(CAPTION_FONT)
        if item.show_title:
            painter.setFont(_px_font(CAPTION_FONT + 2, bold=True))
            painter.setPen(QColor("#000000"))
            title = pop.display_name if pop is not None else (item.pop_path.rsplit("/", 1)[-1] or "Ungated")
            painter.drawText(QRectF(0, 0, item.w, TITLE_H), Qt.AlignmentFlag.AlignCenter, title)
        if item.show_caption:
            painter.setFont(font)
            painter.setPen(QColor("#000000"))
            fm = QFontMetricsF(font)
            lines = [
                sample.name if sample is not None else "(missing sample)",
                (pop.path or pop.display_name) if pop is not None else (item.pop_path or "Ungated"),
                f"{pop.count:,}" if pop is not None else "",
            ]
            y = rect.bottom() + 2
            for line in lines:
                text = fm.elidedText(line, Qt.TextElideMode.ElideMiddle, item.w)
                painter.drawText(QRectF(0, y, item.w, CAPTION_LINE), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, text)
                y += CAPTION_LINE
        painter.restore()

    def paint_text(self, painter: QPainter, item: LayoutItem):
        doc = QTextDocument()
        doc.setDefaultFont(_px_font(item.font_size))
        doc.setPlainText(item.text)
        if item.w > 0:
            doc.setTextWidth(item.w)
        painter.save()
        doc.drawContents(painter)
        painter.restore()

    def paint_page(self, painter: QPainter, layout: Layout, sample_override: Sample | None = None, scale: float = 2.0, header: str | None = None):
        """Paint a whole page in page-point coordinates."""
        w, h = layout.page_size
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(QRectF(0, 0, w, h), QColor("#ffffff"))
        if header:
            painter.setFont(_px_font(HEADER_FONT, bold=True))
            painter.setPen(QColor("#000000"))
            painter.drawText(QRectF(MARGIN, 8, w - 2 * MARGIN, MARGIN - 12), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, header)
        for item in layout.items:
            painter.save()
            painter.translate(item.x, item.y)
            if item.kind == "text":
                self.paint_text(painter, item)
            else:
                self.paint_plot(painter, item, sample_override, scale)
            painter.restore()
        painter.restore()

    def page_pixmap(self, layout: Layout | None = None, sample_override: Sample | None = None, scale: float = 2.0) -> QPixmap:
        layout = layout or self.current_layout()
        self._commit_text_edit()
        w, h = layout.page_size
        pm = QPixmap(int(round(w * scale)), int(round(h * scale)))
        pm.fill(QColor("#ffffff"))
        p = QPainter(pm)
        p.scale(scale, scale)
        self.paint_page(p, layout, sample_override, scale, header=sample_override.name if sample_override is not None else None)
        p.end()
        return pm

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

    def export_png(self, path, scale: float = 2.0) -> bool:
        return self.page_pixmap(self.current_layout(), None, scale).save(str(path), "PNG")

    def export_pdf(self, path, batch_group=None) -> int:
        """Write the current layout as PDF. With batch_group (Group or group
        name) one page per member sample is written. Returns the page count."""
        layout = self.current_layout()
        self._commit_text_edit()
        w, h = layout.page_size
        writer = QPdfWriter(str(path))
        writer.setResolution(300)
        writer.setCreator("FlowDesk")
        writer.setTitle(layout.name)
        size = QPageSize(QSizeF(w, h), QPageSize.Unit.Point, layout.name, QPageSize.SizeMatchPolicy.ExactMatch)
        writer.setPageLayout(QPageLayout(size, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0)))
        samples = self._batch_samples(batch_group)
        pages: list[Sample | None] = list(samples) if samples else [None]
        painter = QPainter(writer)
        factor = writer.resolution() / 72.0
        for k, sample in enumerate(pages):
            if k:
                writer.newPage()
            painter.save()
            painter.scale(factor, factor)
            self.paint_page(painter, layout, sample, 3.0, header=sample.name if sample is not None else None)
            painter.restore()
        painter.end()
        return len(pages)

    def export_svg(self, path) -> bool:
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
        self.paint_page(p, layout, None, 2.0)
        p.end()
        return True

    def export_dialog(self):
        layout = self.current_layout()
        path, flt = QFileDialog.getSaveFileName(self, "Export Layout", layout.name, "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)")
        if not path:
            return
        lower = path.lower()
        if not lower.endswith((".png", ".pdf", ".svg")):
            ext = ".pdf" if "PDF" in flt else ".svg" if "SVG" in flt else ".png"
            path += ext
            lower = path.lower()
        if lower.endswith(".pdf"):
            self.export_pdf(path)
        elif lower.endswith(".svg"):
            if not self.export_svg(path):
                QMessageBox.warning(self, "Export", "SVG export is not available (QtSvg missing).")
                return
        else:
            self.export_png(path)
        self.statusBar().showMessage(f"Exported {path}", 5000)

    def batch_pdf_dialog(self):
        layout = self.current_layout()
        group = self.workspace.group_by_name(self.group_combo.currentText()) or self.workspace.groups[0]
        if not self.workspace.members(group):
            QMessageBox.information(self, "Batch", f"The group “{group.name}” has no samples.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Create Batch PDF", f"{layout.name} - {group.name}.pdf", "PDF document (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        n = self.export_pdf(path, batch_group=group)
        self.statusBar().showMessage(f"Wrote {n} page(s) to {path}", 5000)

    def print_layout(self):
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter

        layout = self.current_layout()
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setPageOrientation(QPageLayout.Orientation.Landscape if layout.landscape else QPageLayout.Orientation.Portrait)
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._commit_text_edit()
        painter = QPainter(printer)
        rect = printer.pageLayout().paintRectPixels(printer.resolution())
        w, h = layout.page_size
        f = min(rect.width() / w, rect.height() / h)
        painter.scale(f, f)
        self.paint_page(painter, layout, None, 3.0)
        painter.end()

    # ------------------------------------------------------------ layouts
    def new_layout(self, name: str | None = None) -> Layout:
        names = {l.name for l in self.workspace.layouts}
        base = name or "Layout"
        name, k = base, 2
        while name in names:
            name = f"{base} {k}"
            k += 1
        lay = Layout(name)
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
            layouts.append(Layout("Layout"))
        self._layout = layouts[min(idx, len(layouts) - 1)]
        self._notify_now()
        self._sync_list()
        self._rebuild_scene()

    def page_setup(self):
        lay = self.current_layout()
        dlg = PageSetupDialog(lay, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            page, landscape = dlg.values()
            self.set_page(page, landscape)

    def set_page(self, page: str, landscape: bool):
        lay = self.current_layout()
        lay.page = page if page in PAGE_SIZES else "A4"
        lay.landscape = bool(landscape)
        self._notify_now()
        self._rebuild_scene()
        self.fit_page()

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
        y = MARGIN
        while y + h <= ph - MARGIN + 1:
            x = MARGIN
            while x + w <= pw - MARGIN + 1:
                r = QRectF(x, y, w, h)
                if not any(r.intersects(QRectF(i.x, i.y, i.w, i.h)) for i in lay.items):
                    return QPointF(x, y)
                x += w + SPACING
            y += h + SPACING
        return QPointF(MARGIN, MARGIN)

    def _make_plot_item(self, sample: Sample, population: Population, x: float, y: float) -> LayoutItem:
        item = LayoutItem(kind="plot", sample_id=sample.id, pop_path=population.path, graph=self.default_graph(sample, population))
        item.w = DEFAULT_PLOT
        item.h = DEFAULT_PLOT + caption_height(item) + title_height(item)
        item.x, item.y = (_snap(x), _snap(y)) if self.snap_to_grid else (x, y)
        return item

    def add_population(self, sample: Sample, population: Population, pos: QPointF | None = None) -> LayoutItem:
        lay = self.current_layout()
        probe = LayoutItem()
        h = DEFAULT_PLOT + caption_height(probe)
        if pos is None:
            pos = self._free_position(DEFAULT_PLOT, h)
        item = self._make_plot_item(sample, population, pos.x(), pos.y())
        lay.items.append(item)
        self._add_scene_item(item, len(lay.items) - 1)
        self._notify_now()
        return item

    def drop_mime(self, mime, scene_pos: QPointF) -> list[LayoutItem]:
        """Handle a POPULATION_MIME drop at scene_pos. Dropping on a plot adds
        the populations as overlays of that plot (returns [that item]);
        dropping elsewhere creates one plot per population."""
        sid, paths = parse_population_mime(mime)
        sample = self.workspace.sample_by_id(sid) if sid is not None else None
        if sample is None or not paths:
            return []
        pops = []
        for path in paths:
            p = sample.root.find(path) if path else sample.root
            if p is not None:
                pops.append(p)
        if not pops:
            return []
        lay = self.current_layout()
        target = self.plot_item_at(scene_pos)
        if target is not None:
            item = target.item
            for p in pops:
                if sample.id == item.sample_id and p.path == item.pop_path:
                    continue
                entry = {"sample_id": sample.id, "pop_path": p.path}
                if entry not in item.overlays:
                    item.overlays.append(entry)
            target.update()
            self._notify_now()
            return [item]
        pw, _ = lay.page_size
        x0 = max(0.0, scene_pos.x())
        x, y = x0, max(0.0, scene_pos.y())
        created = []
        row_h = 0.0
        for p in pops:
            item = self._make_plot_item(sample, p, x, y)
            if created and item.x + item.w > pw - MARGIN + 1:
                x = MARGIN if x0 + item.w > pw - MARGIN else x0
                y += row_h + SPACING
                row_h = 0.0
                item.x, item.y = (_snap(x), _snap(y)) if self.snap_to_grid else (x, y)
            lay.items.append(item)
            self._add_scene_item(item, len(lay.items) - 1)
            created.append(item)
            row_h = max(row_h, item.h)
            x = item.x + item.w + SPACING
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
        avail = pw - 2 * MARGIN
        cell = (avail - (columns - 1) * SPACING) / columns
        y = MARGIN
        for r in range(0, len(plots), columns):
            row = plots[r : r + columns]
            row_h = 0.0
            for c, it in enumerate(row):
                it.w = cell
                it.h = cell + caption_height(it) + title_height(it)
                it.x = MARGIN + c * (cell + SPACING)
                it.y = y
                row_h = max(row_h, it.h)
            y += row_h + SPACING
        self._sync_items(plots)
        self._notify_now()

    def _targets(self) -> list[LayoutItem]:
        return self.selected_layout_items()

    def align_left(self):
        items = self._targets()
        if len(items) > 1:
            x = min(i.x for i in items)
            for i in items:
                i.x = x
            self._sync_items(items)
            self._notify_now()

    def align_top(self):
        items = self._targets()
        if len(items) > 1:
            y = min(i.y for i in items)
            for i in items:
                i.y = y
            self._sync_items(items)
            self._notify_now()

    def same_size(self):
        items = [i for i in self._targets() if i.kind == "plot"]
        if len(items) > 1:
            ref = items[0]
            for i in items[1:]:
                i.w, i.h = ref.w, ref.h
            self._sync_items(items)
            self._notify_now()

    # -------------------------------------------------------------- editing
    def edit_plot(self, item: LayoutItem) -> bool:
        dlg = PlotOptionsDialog(self, item, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        old_extra = caption_height(item) + title_height(item)
        dlg.apply()
        new_extra = caption_height(item) + title_height(item)
        item.h = max(MIN_SIZE, item.h + new_extra - old_extra)
        self._sync_items([item])
        self._notify_now()
        return True

    def set_font_size(self, items: list[LayoutItem], size: int):
        for it in items:
            if it.kind == "text":
                it.font_size = int(size)
        self._sync_items(items)
        self._notify_now()

    def item_context_menu(self, gi, screen_pos):
        items = self.selected_layout_items() or [gi.item]
        menu = QMenu(self)
        if isinstance(gi, PlotGraphicsItem):
            menu.addAction(icon("rename"), "Edit…", lambda: self.edit_plot(gi.item))
        else:
            menu.addAction(icon("rename"), "Edit Text", gi.start_editing)
            sizes = menu.addMenu("Font Size")
            grp = QActionGroup(sizes)
            for s in FONT_SIZES:
                a = sizes.addAction(f"{s} pt", lambda s=s: self.set_font_size(items, s))
                a.setCheckable(True)
                a.setChecked(s == gi.item.font_size)
                grp.addAction(a)
        menu.addAction(icon("copy"), "Duplicate", lambda: self.duplicate_items(items))
        menu.addAction(icon("delete"), "Remove", lambda: self.remove_items(items))
        menu.addSeparator()
        menu.addAction(icon("up"), "Bring to Front", lambda: self.bring_to_front(items, True))
        menu.addAction(icon("down"), "Send to Back", lambda: self.bring_to_front(items, False))
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
