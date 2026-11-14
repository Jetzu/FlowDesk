"""Interactive plot: draws a population and lets the user create and edit gates."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QColorDialog, QInputDialog, QMenu, QMessageBox, QWidget

from ..core.gates import EllipseGate, PolygonGate, RectangleGate, gate_from_dict, make_quadrant_gates
from ..core.model import Group, Population, Sample
from ..core.transforms import drag_edge, rescale
from .dialogs import GateNameDialog
from .renderer import GateShape, RenderOptions, RenderResult, draw_gates, render_plot
from .style import PLOT_FONT_DPI

TOOLS = ("pointer", "rect", "polygon", "ellipse", "quadrant", "range", "bisector", "freehand", "zoom")
HANDLE = 4.0
HIT = 6.0
AXIS_STRIP = 40.0  # px below / left of the plot where dragging changes the axis range


def _dist(a: QPointF, b: QPointF) -> float:
    return math.hypot(a.x() - b.x(), a.y() - b.y())


def _dist_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay, bx, by, px, py = a.x(), a.y(), b.x(), b.y(), p.x(), p.y()
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify_path(points: list[QPointF], tolerance: float) -> list[QPointF]:
    """Ramer-Douglas-Peucker simplification."""
    if len(points) < 3:
        return points

    def rdp(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        idx, dmax = 0, 0.0
        for i in range(1, len(pts) - 1):
            d = _dist_to_segment(pts[i], a, b)
            if d > dmax:
                idx, dmax = i, d
        if dmax > tolerance:
            left = rdp(pts[: idx + 1])
            right = rdp(pts[idx:])
            return left[:-1] + right
        return [a, b]

    return rdp(points)


class PlotWidget(QWidget):
    selectionChanged = Signal(object)  # Population | None
    populationActivated = Signal(object)  # Population (double-clicked gate)
    graphChanged = Signal()  # axes / plot type changed
    statusText = Signal(str)
    toolDone = Signal()  # a creation tool finished (window may switch back to pointer)
    contextMenuRequested = Signal(object, object)  # Population | None, QPoint (global)
    zoomChanged = Signal()
    renameRequested = Signal(object)  # Population (double-clicked gate label)
    axisEditRequested = Signal(str)  # "x" / "y": double-click on an axis opens the transform dialog

    def __init__(self, controller, sample: Sample, population: Population, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.sample = sample
        self.population = population
        self.group: Group | None = None  # when set, the widget shows a *group* graph
        self.graph = population.graph
        self._ensure_axes()
        self.tool = "pointer"
        self.selected: Population | None = None
        self.overlays: list[tuple[Sample, Population]] = []
        self.view = (0.0, 1.0, 0.0, 1.0)
        self._base: QImage | None = None
        self._result: RenderResult | None = None
        self._dirty = True
        self._drag: dict | None = None
        self._poly: list[QPointF] = []
        self._free: list[QPointF] = []
        self._rubber: QRectF | None = None
        self._axis_drag: dict | None = None
        self._legend_drag: dict | None = None
        self._mouse = QPointF()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(240, 240)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        c = controller
        c.gatingChanged.connect(self._on_data_changed)
        c.scalesChanged.connect(self._on_data_changed)
        c.compensationChanged.connect(self._on_data_changed)
        c.preferencesChanged.connect(lambda: self.invalidate())

    # ---------------------------------------------------------------- state
    def _ensure_axes(self):
        names = self.sample.parameter_names()
        if not names:
            return
        if not self.graph.x or self.graph.x not in names:
            x, y = self.sample.default_axes()
            self.graph.x = x
            if self.graph.y is None and self.graph.plot_type != "histogram":
                self.graph.y = y
        if self.graph.y is not None and self.graph.y not in names:
            self.graph.y = None if self.graph.plot_type == "histogram" else self.sample.default_axes()[1]
        if self.graph.plot_type == "histogram":
            self.graph.y = None
        elif self.graph.y is None:
            self.graph.plot_type = "histogram"

    def set_population(self, population: Population):
        self.population = population
        self.graph = population.graph
        self.selected = None
        self.view = (0.0, 1.0, 0.0, 1.0)
        self._ensure_axes()
        self._cancel_creation()
        self.invalidate()
        self.selectionChanged.emit(None)

    def set_axes(self, x: str | None = None, y: str | None = "keep", plot_type: str | None = None):
        if x is not None:
            self.graph.x = x
        if y != "keep":
            self.graph.y = y
        if plot_type is not None:
            self.graph.plot_type = plot_type
            if plot_type == "histogram":
                self.graph.y = None
            elif self.graph.y is None:
                self.graph.y = self.sample.default_axes()[1] or self.graph.x
        elif self.graph.y is None:
            self.graph.plot_type = "histogram"
        elif self.graph.plot_type == "histogram":
            self.graph.plot_type = self.controller.preferences.get("plot_type") or "pseudocolor"
        self.view = (0.0, 1.0, 0.0, 1.0)
        self.controller.mark_modified()
        self.invalidate()
        self.graphChanged.emit()

    def set_tool(self, tool: str):
        if tool not in TOOLS:
            tool = "pointer"
        self._cancel_creation()
        self.tool = tool
        self.setCursor(Qt.CursorShape.ArrowCursor if tool == "pointer" else Qt.CursorShape.CrossCursor)
        self.update()

    def set_selected(self, pop: Population | None):
        if pop is not self.selected:
            self.selected = pop
            self.update()
            self.selectionChanged.emit(pop)

    def set_overlays(self, overlays: list[tuple[Sample, Population]]):
        self.overlays = list(overlays)
        self.invalidate()

    def reset_zoom(self):
        self.view = (0.0, 1.0, 0.0, 1.0)
        self.invalidate()
        self.zoomChanged.emit()

    def zoom_by(self, factor: float):
        """Zoom around the centre of the current view (factor < 1 zooms in)."""
        x0, x1, y0, y1 = self.view

        def span(a, b):
            c = (a + b) / 2
            half = min(0.5, max(0.005, (b - a) * factor / 2))
            lo, hi = c - half, c + half
            if lo < 0:
                lo, hi = 0.0, hi - lo
            if hi > 1:
                lo, hi = lo - (hi - 1), 1.0
            return max(0.0, lo), min(1.0, hi)

        nx0, nx1 = span(x0, x1)
        histogram = self.graph.plot_type == "histogram" or self.graph.y is None
        ny0, ny1 = (0.0, 1.0) if histogram else span(y0, y1)
        self.view = (nx0, nx1, ny0, ny1)
        self.invalidate()
        self.zoomChanged.emit()

    def options(self) -> RenderOptions:
        return RenderOptions.from_prefs(self.controller.preferences, self.graph, view=self.view, font_dpi=PLOT_FONT_DPI)

    def invalidate(self):
        self._dirty = True
        self.update()

    def _on_data_changed(self, sample_id=None):
        if sample_id is None or sample_id == self.sample.id:
            if self.selected is not None and self.selected.parent is not self.population:
                # the selected population may have been replaced by a group copy
                replacement = self.population.child_by_name(self.selected.name) if self.population is not None else None
                self.selected = replacement
                self.selectionChanged.emit(self.selected)
            self.invalidate()

    # ------------------------------------------------------------- painting
    def result(self) -> RenderResult | None:
        return self._result

    def _render(self):
        dpr = self.devicePixelRatioF()
        self._ensure_axes()
        opts = self.options()
        opts.show_gates = False
        self._result = render_plot(self.sample, self.population, self.graph, self.width(), self.height(), opts, overlays=self.overlays, dpr=dpr)
        self._base = self._result.image
        self._dirty = False

    def paintEvent(self, event):
        if self._dirty or self._base is None or self._base.deviceIndependentSize().toSize() != self.size():
            self._render()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.drawImage(0, 0, self._base)
        res = self._result
        if res is not None and res.error is None:
            opts = self.options()
            if opts.show_gates:
                draw_gates(painter, res, self.population, self.sample, opts, self.selected)
                self._draw_handles(painter, res)
            else:
                res.gates = []
            self._draw_creation(painter)
        painter.end()

    def _shape_for(self, pop: Population) -> GateShape | None:
        if self._result is None:
            return None
        for s in self._result.gates:
            if s.population is pop:
                return s
        return None

    def _handles(self, shape: GateShape) -> list[QPointF]:
        if shape.kind == "rect":
            b = shape.bbox
            return [b.topLeft(), b.topRight(), b.bottomRight(), b.bottomLeft()]
        if shape.kind == "polygon":
            return [QPointF(x, y) for x, y in shape.points]
        if shape.kind == "ellipse":
            c, a, b = (QPointF(*p) for p in shape.points)
            return [a, b, QPointF(2 * c.x() - a.x(), 2 * c.y() - a.y()), QPointF(2 * c.x() - b.x(), 2 * c.y() - b.y())]
        if shape.kind == "range":
            (x0, y0), (x1, y1) = shape.points
            return [QPointF(x0, y0), QPointF(x1, y1)]
        if shape.kind == "quadrant":
            cx, cy = shape.extra["centre"]
            return [QPointF(cx, cy)]
        return []

    def _draw_handles(self, painter: QPainter, res: RenderResult):
        if self.selected is None:
            return
        shape = self._shape_for(self.selected)
        if shape is None:
            return
        locked = bool(getattr(self.selected.gate, "locked", False))
        pen = QPen(QColor(self.controller.preferences.get("selected_gate_color")), 1.0)
        if locked:  # dashed, hollow handles: the gate cannot be edited
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        else:
            painter.setBrush(QBrush(QColor("#ffffff")))
        painter.setPen(pen)
        plot = res.plot_rect.adjusted(-HANDLE, -HANDLE, HANDLE, HANDLE)
        for h in self._handles(shape):
            if plot.contains(h):  # a handle beyond the axis is not drawn (zoom out to reach it)
                painter.drawRect(QRectF(h.x() - HANDLE, h.y() - HANDLE, 2 * HANDLE, 2 * HANDLE))

    def _draw_creation(self, painter: QPainter):
        pen = QPen(QColor("#d81b1b"), 1.2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._rubber is not None:
            if self.tool == "ellipse":
                painter.drawEllipse(self._rubber)
            elif self.tool == "range" and self._result is not None:
                r = self._result.plot_rect
                painter.drawLine(QPointF(self._rubber.left(), r.top()), QPointF(self._rubber.left(), r.bottom()))
                painter.drawLine(QPointF(self._rubber.right(), r.top()), QPointF(self._rubber.right(), r.bottom()))
            else:
                if self.tool == "zoom":
                    painter.setPen(QPen(QColor("#2f6fd6"), 1.0, Qt.PenStyle.DashLine))
                painter.drawRect(self._rubber)
        if self._poly:
            pts = list(self._poly) + [self._mouse]
            painter.drawPolyline(QPolygonF(pts))
            painter.setBrush(QBrush(QColor("#ffffff")))
            for p in self._poly:
                painter.drawRect(QRectF(p.x() - 3, p.y() - 3, 6, 6))
        if self._free:
            painter.drawPolyline(QPolygonF(self._free))
        if self.tool in ("quadrant", "bisector") and self._result is not None and self._result.plot_rect.contains(self._mouse):
            r = self._result.plot_rect
            painter.setPen(QPen(QColor("#d81b1b"), 1.0, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(self._mouse.x(), r.top()), QPointF(self._mouse.x(), r.bottom()))
            if self.tool == "quadrant":
                painter.drawLine(QPointF(r.left(), self._mouse.y()), QPointF(r.right(), self._mouse.y()))

    # ------------------------------------------------------- coordinate maths
    def _frac(self, pos: QPointF) -> tuple[float, float]:
        return self._result.to_fraction(pos.x(), pos.y())

    def _clamp_frac(self, f: float) -> float:
        return min(max(f, 0.0), 1.0)

    def _data_x(self, fx: float) -> float:
        return float(self._result.x_scale.inverse_scalar(self._clamp_frac(fx)))

    def _data_y(self, fy: float) -> float:
        return float(self._result.y_scale.inverse_scalar(self._clamp_frac(fy)))

    def _data_from_pos(self, pos: QPointF) -> tuple[float, float | None]:
        fx, fy = self._frac(pos)
        return self._data_x(fx), (self._data_y(fy) if self._result.y_scale is not None else None)

    def _frac_x_of(self, value: float) -> float:
        return float(self._result.x_scale.forward_scalar(value))

    def _frac_y_of(self, value: float) -> float:
        return float(self._result.y_scale.forward_scalar(value))

    # ---------------------------------------------------------- hit testing
    def _hit(self, pos: QPointF) -> tuple[GateShape, str] | None:
        res = self._result
        if res is None:
            return None
        for shape in reversed(res.gates):
            if shape.label and shape.label_rect.contains(pos):
                return shape, "label"
        if self.selected is not None:
            shape = self._shape_for(self.selected)
            if shape is not None:
                for i, h in enumerate(self._handles(shape)):
                    if _dist(h, pos) <= HIT + 1:
                        return shape, f"handle:{i}"
        candidates = sorted(res.gates, key=lambda s: s.bbox.width() * s.bbox.height())
        for shape in candidates:
            if shape.kind == "quadrant":
                cx, cy = shape.extra["centre"]
                r = res.plot_rect
                if (abs(pos.x() - cx) <= HIT and r.top() <= pos.y() <= r.bottom()) or (abs(pos.y() - cy) <= HIT and r.left() <= pos.x() <= r.right()):
                    return shape, "quad"
                continue
            if shape.kind == "rect":
                if shape.bbox.adjusted(-HIT, -HIT, HIT, HIT).contains(pos):
                    return shape, "body"
            elif shape.kind == "polygon":
                poly = QPolygonF([QPointF(x, y) for x, y in shape.points])
                if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                    return shape, "body"
                for i in range(len(shape.points)):
                    a = QPointF(*shape.points[i])
                    b = QPointF(*shape.points[(i + 1) % len(shape.points)])
                    if _dist_to_segment(pos, a, b) <= HIT:
                        return shape, "body"
            elif shape.kind == "ellipse":
                poly = QPolygonF([QPointF(x, y) for x, y in shape.extra["outline"]])
                if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                    return shape, "body"
                for i in range(len(shape.extra["outline"])):
                    a = QPointF(*shape.extra["outline"][i])
                    b = QPointF(*shape.extra["outline"][(i + 1) % len(shape.extra["outline"])])
                    if _dist_to_segment(pos, a, b) <= HIT:
                        return shape, "body"
            elif shape.kind == "range":
                (x0, y0), (x1, y1) = shape.points
                if not shape.transposed:
                    if abs(pos.x() - x0) <= HIT or abs(pos.x() - x1) <= HIT or (min(x0, x1) <= pos.x() <= max(x0, x1) and abs(pos.y() - y0) <= HIT + 4):
                        return shape, "body"
                else:
                    if abs(pos.y() - y0) <= HIT or abs(pos.y() - y1) <= HIT or (min(y0, y1) <= pos.y() <= max(y0, y1) and abs(pos.x() - x0) <= HIT + 4):
                        return shape, "body"
        return None

    # --------------------------------------------------------------- mouse
    def mousePressEvent(self, event: QMouseEvent):
        self.setFocus()
        pos = event.position()
        self._mouse = pos
        if self._result is None or self._result.error:
            return
        if event.button() == Qt.MouseButton.RightButton:
            row = self._legend_hit(pos)
            if row is not None:
                self._legend_menu(row, event.globalPosition().toPoint())
                return
            hit = self._hit(pos) if self.tool == "pointer" else None
            pop = hit[0].population if hit else None
            if pop is not None:
                self.set_selected(pop)
            self.contextMenuRequested.emit(pop, event.globalPosition().toPoint())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        inside = self._result.plot_rect.contains(pos)
        if self.tool == "pointer" and self._hit(pos) is None:
            if self._legend_hit(pos) is not None:
                self._legend_drag = {"start": pos, "rect": QRectF(self._result.legend_rect), "moved": False}
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            axis = self._axis_hit(pos)
            if axis is not None:
                self._start_axis_drag(axis, pos)
                return
        if self.tool == "pointer":
            hit = self._hit(pos)
            if hit is None:
                self.set_selected(None)
                return
            shape, part = hit
            self.set_selected(shape.population)
            gate = shape.population.gate
            if part != "label" and (getattr(gate, "locked", False) or any(getattr(s.gate, "locked", False) for s in self._quadrant_siblings(shape.population))):
                self.statusText.emit(f"“{shape.population.name}” is locked")
                return
            self._drag = {"part": part, "start": pos, "last": pos, "orig": gate.to_dict(), "shape": shape, "moved": False, "siblings": self._quadrant_siblings(shape.population)}
            return
        if not inside:
            return
        if self.tool in ("rect", "ellipse", "zoom", "range"):
            self._rubber = QRectF(pos, pos)
        elif self.tool == "polygon":
            if self._poly and _dist(pos, self._poly[0]) <= HIT and len(self._poly) >= 3:
                self._finish_polygon()
            else:
                self._poly.append(pos)
                self.update()
        elif self.tool == "freehand":
            self._free = [pos]
        elif self.tool == "quadrant":
            self._create_quadrants(pos)
        elif self.tool == "bisector":
            self._create_bisector(pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        self._mouse = pos
        res = self._result
        if res is None or res.error:
            return
        if self._axis_drag is not None:
            self._move_axis_drag(pos)
            return
        if self._legend_drag is not None:
            self._move_legend_drag(pos)
            return
        if self._drag is not None:
            if not self._drag["moved"]:
                if _dist(pos, self._drag["start"]) < 2:
                    return
                self._drag["moved"] = True
                self.controller.begin_change("edit gate")
            self._apply_drag(pos)
            self._drag["last"] = pos
            self.update()
            return
        if self._rubber is not None:
            self._rubber = QRectF(self._rubber.topLeft(), pos).normalized()
            self.update()
        elif self._free:
            if _dist(pos, self._free[-1]) >= 3:
                self._free.append(pos)
                self.update()
        elif self._poly or self.tool in ("quadrant", "bisector"):
            self.update()
        # status
        if res.plot_rect.contains(pos):
            xv, yv = self._data_from_pos(pos)
            text = f"{res.x} = {xv:,.1f}"
            if yv is not None and res.y:
                text += f"    {res.y} = {yv:,.1f}"
            self.statusText.emit(text)
        if self.tool == "pointer":
            hit = self._hit(pos)
            if hit is None:
                self._hover_cursor(pos)
            elif hit[1].startswith("handle") or hit[1] == "quad":
                self.setCursor(Qt.CursorShape.SizeAllCursor if hit[1] == "quad" else Qt.CursorShape.CrossCursor)
            else:
                self.setCursor(Qt.CursorShape.OpenHandCursor if hit[1] == "body" else Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        if self._axis_drag is not None:
            self._finish_axis_drag()
            return
        if self._legend_drag is not None:
            drag, self._legend_drag = self._legend_drag, None
            if drag["moved"]:
                self.controller.mark_modified()
                self.graphChanged.emit()
            self._hover_cursor(pos)
            return
        if self._drag is not None:
            drag = self._drag
            self._drag = None
            if drag["moved"]:
                pop = drag["shape"].population
                self.controller.gate_changed(pop)
                for sib in drag["siblings"]:
                    sib.invalidate()
            self.update()
            return
        if self._rubber is not None:
            rect = self._rubber.normalized()
            self._rubber = None
            if self.tool == "zoom":
                if rect.width() > 4 and rect.height() > 4:
                    fx0, fy0 = self._frac(rect.bottomLeft())
                    fx1, fy1 = self._frac(rect.topRight())
                    self.view = (max(0.0, fx0), min(1.0, fx1), max(0.0, fy0), min(1.0, fy1))
                    self.invalidate()
                    self.zoomChanged.emit()
            elif self.tool == "rect" and rect.width() > 3 and rect.height() > 3:
                self._create_rect(rect)
            elif self.tool == "ellipse" and rect.width() > 3 and rect.height() > 3:
                self._create_ellipse(rect)
            elif self.tool == "range" and rect.width() > 2:
                self._create_range(rect)
            self.update()
            return
        if self._free:
            pts = simplify_path(self._free, 2.0)
            self._free = []
            if len(pts) >= 3:
                self._create_polygon(pts)
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        pos = event.position()
        if self.tool == "polygon":
            if len(self._poly) >= 3:
                self._finish_polygon()
            return
        if self.tool == "zoom":
            self.reset_zoom()
            return
        if self.tool != "pointer" or self._result is None:
            return
        hit = self._hit(pos)
        if hit is None:
            row = self._legend_hit(pos)
            if row is not None and row >= 0:
                self._rename_legend_entry(row)
                return
            axis = self._axis_hit(pos)
            if axis is not None:
                self._axis_drag = None
                self.axisEditRequested.emit(axis)
            return
        self._drag = None
        if hit[1] == "label":
            self.renameRequested.emit(hit[0].population)
        else:
            self.populationActivated.emit(hit[0].population)

    def leaveEvent(self, event):
        self.statusText.emit("")
        super().leaveEvent(event)

    # ------------------------------------------------------- axis dragging
    def _axis_hit(self, pos: QPointF) -> str | None:
        """"x" / "y" when `pos` is on the tick labels below / left of the plot (a histogram's count axis is not draggable)."""
        res = self._result
        if res is None or res.error or res.x_scale is None:
            return None
        r = res.plot_rect
        if r.left() - 4 <= pos.x() <= r.right() + 4 and r.bottom() < pos.y() <= r.bottom() + AXIS_STRIP:
            return "x"
        if res.y and res.y_scale is not None and not res.histogram and r.top() - 4 <= pos.y() <= r.bottom() + 4 and r.left() - AXIS_STRIP * 1.6 <= pos.x() < r.left():
            return "y"
        return None

    def _axis_frac(self, drag: dict, pos: QPointF) -> tuple[float, float]:
        """(fraction of the full scale, fraction of the visible axis) of `pos`, measured on the plot as it was when the drag began."""
        r, view = drag["rect"], drag["view"]
        if drag["axis"] == "x":
            vis = (pos.x() - r.left()) / max(r.width(), 1.0)
            return view[0] + vis * (view[1] - view[0]), vis
        vis = (r.bottom() - pos.y()) / max(r.height(), 1.0)
        return view[2] + vis * (view[3] - view[2]), vis

    def _start_axis_drag(self, axis: str, pos: QPointF):
        res = self._result
        param = res.x if axis == "x" else res.y
        scale = self.sample.scale(param).copy()
        drag = {"axis": axis, "param": param, "scale": scale, "rect": QRectF(res.plot_rect), "view": tuple(res.view), "moved": False, "start": pos}
        full, vis = self._axis_frac(drag, pos)
        drag["edge"] = drag_edge(scale, vis)
        drag["value"] = float(scale.inverse_scalar(min(max(full, 0.0), 1.0)))
        self._axis_drag = drag
        self.setCursor(self._axis_cursor(axis, drag["edge"], pressed=True))

    def _move_axis_drag(self, pos: QPointF):
        drag = self._axis_drag
        if not drag["moved"]:
            if _dist(pos, drag["start"]) < 3:
                return
            drag["moved"] = True
            self.controller.begin_change("change axis range")
        full, _vis = self._axis_frac(drag, pos)
        new = rescale(drag["scale"], drag["edge"], drag["value"], full)
        drag["new"] = new
        self.sample.set_scale(drag["param"], new)
        self.statusText.emit(f"{drag['param']}: {new.describe()}  (double-click the axis for exact values)")
        self.invalidate()

    def _finish_axis_drag(self):
        drag, self._axis_drag = self._axis_drag, None
        if drag["moved"] and drag.get("new") is not None:
            self.controller.mark_modified()
            self.controller.scalesChanged.emit(self.sample.id)
            self.controller.gatingChanged.emit(self.sample.id)
        self.setCursor(self._axis_cursor(drag["axis"], drag["edge"]))

    @staticmethod
    def _axis_cursor(axis: str, edge: str, pressed: bool = False):
        if edge == "pan":
            return Qt.CursorShape.ClosedHandCursor if pressed else Qt.CursorShape.OpenHandCursor
        return Qt.CursorShape.SizeHorCursor if axis == "x" else Qt.CursorShape.SizeVerCursor

    def _hover_cursor(self, pos: QPointF):
        if self._legend_hit(pos) is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        axis = self._axis_hit(pos)
        if axis is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        res = self._result
        param = res.x if axis == "x" else res.y
        r = res.plot_rect
        vis = (pos.x() - r.left()) / max(r.width(), 1.0) if axis == "x" else (r.bottom() - pos.y()) / max(r.height(), 1.0)
        edge = drag_edge(self.sample.scale(param), vis)
        self.setCursor(self._axis_cursor(axis, edge))
        what = {"min": "the low end", "max": "the high end", "pan": "the whole range"}[edge]
        self.statusText.emit(f"Drag to move {what} of the {axis} axis · double-click for the transform dialog")

    # -------------------------------------------------------------- legend
    def _legend_hit(self, pos: QPointF) -> int | None:
        """None outside the legend; else the entry row under `pos` (-1 = the title or the frame)."""
        res = self._result
        if res is None or res.legend_rect is None or not res.legend_rect.contains(pos):
            return None
        for i, row in enumerate(res.legend_rows):
            if row.contains(pos):
                return i
        return -1

    def _move_legend_drag(self, pos: QPointF):
        drag = self._legend_drag
        if not drag["moved"]:
            if _dist(pos, drag["start"]) < 3:
                return
            drag["moved"] = True
            self.controller.begin_change("move legend")
        r = self._result.plot_rect
        top_left = drag["rect"].topLeft() + (pos - drag["start"])
        fx = (top_left.x() - r.left()) / max(r.width(), 1.0)
        fy = (top_left.y() - r.top()) / max(r.height(), 1.0)
        self.graph.options["legend_pos"] = [round(min(max(fx, 0.0), 1.0), 4), round(min(max(fy, 0.0), 1.0), 4)]
        self.invalidate()

    def _set_legend_option(self, key: str, value, label: str = "edit legend"):
        self.controller.begin_change(label)
        if value is None:
            self.graph.options.pop(key, None)
        else:
            self.graph.options[key] = value
        self.controller.mark_modified()
        self.graphChanged.emit()
        self.invalidate()

    def _rename_legend_entry(self, row: int):
        res = self._result
        if not (0 <= row < len(res.legend_keys)):
            return
        key = res.legend_keys[row]
        text, ok = QInputDialog.getText(self, "Legend entry", f"Label for {key}:", text=res.legend_entries[row][0])
        if not ok:
            return
        labels = dict(self.graph.options.get("legend_labels") or {})
        if text.strip() and text.strip() != key:
            labels[key] = text.strip()
        else:
            labels.pop(key, None)
        self._set_legend_option("legend_labels", labels or None, "rename legend entry")

    def _recolor_legend_entry(self, row: int):
        if row < 1:
            return
        opts = self.options()
        colours = list(opts.overlay_colors)
        k = (row - 1) % len(colours)
        colour = QColorDialog.getColor(QColor(colours[k]), self, "Overlay colour")
        if colour.isValid():
            colours[k] = colour.name()
            self._set_legend_option("overlay_colors", colours, "change overlay colour")

    def _legend_menu(self, row: int, global_pos):
        opts = self.options()
        menu = QMenu(self)
        if row >= 0:
            menu.addAction("Rename Entry…", lambda: self._rename_legend_entry(row))
            if row >= 1:
                menu.addAction("Entry Colour…", lambda: self._recolor_legend_entry(row))
            menu.addSeparator()
        act = menu.addAction("Show Sample Names")
        act.setCheckable(True)
        act.setChecked(bool(opts.legend_show_samples))
        act.toggled.connect(lambda on: self._set_legend_option("legend_show_samples", bool(on)))

        def ask_title():
            text, ok = QInputDialog.getText(self, "Legend title", "Title (empty for none):", text=opts.legend_title or "")
            if ok:
                self._set_legend_option("legend_title", text.strip() or None)

        menu.addAction("Title…", ask_title)
        sizes = menu.addMenu("Font Size")
        current = float(opts.legend_font_size or 0)
        for size in (0, 7, 8, 9, 10, 11, 12, 14, 16, 18):
            a = sizes.addAction("Automatic" if size == 0 else f"{size} pt")
            a.setCheckable(True)
            a.setChecked(abs(current - size) < 1e-6)
            a.triggered.connect(lambda _=False, v=size: self._set_legend_option("legend_font_size", float(v) if v else None))
        menu.addSeparator()
        menu.addAction("Reset Position", lambda: self._set_legend_option("legend_pos", None, "move legend"))
        if opts.legend_labels:
            menu.addAction("Reset Labels", lambda: self._set_legend_option("legend_labels", None))
        menu.addAction("Hide Legend", lambda: self._set_legend_option("show_legend", False))
        menu.exec(global_pos)

    # ----------------------------------------------------------- keyboard
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._poly or self._free or self._rubber is not None:
                self._cancel_creation()
            elif self.tool != "pointer":
                self.toolDone.emit()  # the window switches back to the pointer
            else:
                self.set_selected(None)
            return
        if not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.AltModifier)):
            if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self.zoom_by(0.8)
                return
            if key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
                self.zoom_by(1.25)
                return
            if key == Qt.Key.Key_0:
                self.reset_zoom()
                return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._poly:
                self._poly.pop()
                self.update()
                return
            if self.selected is not None:
                self.delete_selected()
                return
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            names = self.sample.parameter_names()
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and names:
                i = names.index(self.graph.x) if self.graph.x in names else 0
                i = (i + (1 if key == Qt.Key.Key_Right else -1)) % len(names)
                self.set_axes(x=names[i])
                return
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down) and names and self.graph.y is not None:
                i = names.index(self.graph.y) if self.graph.y in names else 0
                i = (i + (1 if key == Qt.Key.Key_Up else -1)) % len(names)
                self.set_axes(y=names[i])
                return
        super().keyPressEvent(event)

    def _cancel_creation(self):
        self._poly = []
        self._free = []
        self._rubber = None
        self._drag = None
        self.update()

    # ------------------------------------------------------- gate editing
    def _quadrant_siblings(self, pop: Population) -> list[Population]:
        g = pop.gate
        if g is None or getattr(g, "quadrant", None) is None or pop.parent is None:
            return []
        return [c for c in pop.parent.children if c is not pop and c.gate is not None and getattr(c.gate, "quadrant", None) and c.gate.quadrant[0] == g.quadrant[0]]

    def _apply_drag(self, pos: QPointF):
        drag = self._drag
        shape: GateShape = drag["shape"]
        pop = shape.population
        res = self._result
        gate = gate_from_dict(drag["orig"])
        part = drag["part"]
        dx = pos.x() - drag["start"].x()
        dy = pos.y() - drag["start"].y()
        dfx = dx / res.plot_rect.width() * (res.view[1] - res.view[0])
        dfy = -dy / res.plot_rect.height() * (res.view[3] - res.view[2])
        fx, fy = self._frac(pos)
        if part == "label":
            start_fx, start_fy = self._frac(QPointF(shape.label_rect.center()))
            gate.label_pos = (start_fx + dfx, start_fy + dfy)
            pop.gate = gate
            return
        if part == "quad":
            xv = self._data_x(fx)
            yv = self._data_y(fy)
            for sib in [pop] + drag["siblings"]:
                g = sib.gate
                t = shape.transposed
                ix, iy = (1, 0) if t else (0, 1)
                if g.lo[ix] is not None:
                    g.lo[ix] = xv
                if g.hi[ix] is not None:
                    g.hi[ix] = xv
                if g.lo[iy] is not None:
                    g.lo[iy] = yv
                if g.hi[iy] is not None:
                    g.hi[iy] = yv
                sib.invalidate()
            return
        transposed = shape.transposed
        if isinstance(gate, RectangleGate):
            if len(gate.dims) == 1:
                horizontal = not transposed
                sc_fwd = self._frac_x_of if horizontal else self._frac_y_of
                sc_inv = self._data_x if horizontal else self._data_y
                delta = dfx if horizontal else dfy
                lo = None if gate.lo[0] is None else sc_fwd(gate.lo[0])
                hi = None if gate.hi[0] is None else sc_fwd(gate.hi[0])
                if part == "body":
                    lo = None if lo is None else lo + delta
                    hi = None if hi is None else hi + delta
                elif part == "handle:0":
                    lo = fx if horizontal else fy
                elif part == "handle:1":
                    hi = fx if horizontal else fy
                if lo is not None and hi is not None and lo > hi:
                    lo, hi = hi, lo
                gate.lo[0] = None if lo is None else sc_inv(lo)
                gate.hi[0] = None if hi is None else sc_inv(hi)
            else:
                ix, iy = (1, 0) if transposed else (0, 1)
                x0 = None if gate.lo[ix] is None else self._frac_x_of(gate.lo[ix])
                x1 = None if gate.hi[ix] is None else self._frac_x_of(gate.hi[ix])
                y0 = None if gate.lo[iy] is None else self._frac_y_of(gate.lo[iy])
                y1 = None if gate.hi[iy] is None else self._frac_y_of(gate.hi[iy])
                if part == "body":
                    x0 = None if x0 is None else x0 + dfx
                    x1 = None if x1 is None else x1 + dfx
                    y0 = None if y0 is None else y0 + dfy
                    y1 = None if y1 is None else y1 + dfy
                elif part.startswith("handle:"):
                    h = int(part.split(":")[1])
                    if h == 0:  # top-left
                        x0, y1 = fx, fy
                    elif h == 1:  # top-right
                        x1, y1 = fx, fy
                    elif h == 2:  # bottom-right
                        x1, y0 = fx, fy
                    else:  # bottom-left
                        x0, y0 = fx, fy
                if x0 is not None and x1 is not None and x0 > x1:
                    x0, x1 = x1, x0
                if y0 is not None and y1 is not None and y0 > y1:
                    y0, y1 = y1, y0
                gate.lo[ix] = None if x0 is None else self._data_x(x0)
                gate.hi[ix] = None if x1 is None else self._data_x(x1)
                gate.lo[iy] = None if y0 is None else self._data_y(y0)
                gate.hi[iy] = None if y1 is None else self._data_y(y1)
        elif isinstance(gate, PolygonGate):
            verts = []
            for i, (vx, vy) in enumerate(gate.vertices):
                if transposed:
                    vx, vy = vy, vx
                fvx, fvy = self._frac_x_of(vx), self._frac_y_of(vy)
                if part == "body":
                    fvx, fvy = fvx + dfx, fvy + dfy
                elif part == f"handle:{i}":
                    fvx, fvy = fx, fy
                nx, ny = self._data_x(fvx), self._data_y(fvy)
                verts.append((ny, nx) if transposed else (nx, ny))
            gate.vertices = verts
        elif isinstance(gate, EllipseGate):
            c, pa, pb = (QPointF(*p) for p in shape.points)  # pixel coords of the original
            if part == "body":
                c, pa, pb = (QPointF(p.x() + dx, p.y() + dy) for p in (c, pa, pb))
            elif part.startswith("handle:"):
                h = int(part.split(":")[1])
                if h in (0, 2):
                    new_a = pos if h == 0 else QPointF(2 * c.x() - pos.x(), 2 * c.y() - pos.y())
                    ux, uy = new_a.x() - c.x(), new_a.y() - c.y()
                    blen = _dist(pb, c)
                    norm = math.hypot(ux, uy) or 1.0
                    pa = new_a
                    pb = QPointF(c.x() - uy / norm * blen, c.y() + ux / norm * blen)
                else:
                    new_b = pos if h == 1 else QPointF(2 * c.x() - pos.x(), 2 * c.y() - pos.y())
                    vx, vy = new_b.x() - c.x(), new_b.y() - c.y()
                    alen = _dist(pa, c)
                    norm = math.hypot(vx, vy) or 1.0
                    pb = new_b
                    pa = QPointF(c.x() + vy / norm * alen, c.y() - vx / norm * alen)
            pts = []
            for p in (c, pa, pb):
                f_x, f_y = self._frac(p)
                xv, yv = self._data_x(f_x), self._data_y(f_y)
                pts.append((yv, xv) if transposed else (xv, yv))
            gate.center, gate.a_end, gate.b_end = pts
        pop.gate = gate
        pop.invalidate()

    def delete_selected(self):
        pop = self.selected
        if pop is None or pop.parent is None:
            return
        n = sum(1 for _ in pop.walk(False))
        if n and not _confirm(self, "Delete gate", f"“{pop.name}” has {n} sub-population(s). Delete them as well?"):
            return
        group = self._group_for_change()
        pops = [pop]
        if pop.gate is not None and getattr(pop.gate, "quadrant", None):
            pops += self._quadrant_siblings(pop)
        self.controller.delete_populations(pops, group)
        self.set_selected(None)

    def _group_for_change(self) -> Group | None:
        return self.group

    # ------------------------------------------------------- gate creation
    def _default_name(self, two_d: bool) -> str:
        xs = self.sample.param_stain(self.graph.x)
        if two_d and self.graph.y:
            ys = self.sample.param_stain(self.graph.y)
            return f"{xs}, {ys} subset"
        return f"{xs} subset"

    def _finish_gate(self, gate, default_name: str, ask: bool = True):
        apply_default = bool(self.controller.preferences.get("apply_to_group_default")) or self.group is not None
        group = self.group or self.controller.current_group
        if ask:
            dlg = GateNameDialog(default_name, group, apply_default, self)
            if not dlg.exec():
                self.toolDone.emit()
                return None
            name, apply = dlg.values()
            name = name or default_name
        else:
            name, apply = default_name, apply_default
        pop = Population(name, gate)
        new_pop = self.controller.add_population(self.population, pop, group if apply else None)
        if new_pop.parent is not self.population:
            new_pop = self.population.child_by_name(name) or new_pop
        self.set_selected(new_pop)
        self.toolDone.emit()
        return new_pop

    def _create_rect(self, rect: QRectF):
        res = self._result
        fx0, fy0 = self._frac(rect.bottomLeft())
        fx1, fy1 = self._frac(rect.topRight())
        if res.y_scale is None:
            gate = RectangleGate([self.graph.x], [self._data_x(fx0)], [self._data_x(fx1)])
            self._finish_gate(gate, self._default_name(False))
            return
        gate = RectangleGate([self.graph.x, self.graph.y], [self._data_x(fx0), self._data_y(fy0)], [self._data_x(fx1), self._data_y(fy1)])
        self._finish_gate(gate, self._default_name(True))

    def _create_range(self, rect: QRectF):
        fx0, _ = self._frac(rect.bottomLeft())
        fx1, _ = self._frac(rect.topRight())
        gate = RectangleGate([self.graph.x], [self._data_x(fx0)], [self._data_x(fx1)])
        self._finish_gate(gate, self._default_name(False))

    def _create_ellipse(self, rect: QRectF):
        if self._result.y_scale is None:
            return
        c = rect.center()
        a = QPointF(rect.right(), c.y())
        b = QPointF(c.x(), rect.top())
        pts = []
        for p in (c, a, b):
            fx, fy = self._frac(p)
            pts.append((self._data_x(fx), self._data_y(fy)))
        gate = EllipseGate([self.graph.x, self.graph.y], pts[0], pts[1], pts[2])
        self._finish_gate(gate, self._default_name(True))

    def _create_polygon(self, pts: list[QPointF]):
        if self._result.y_scale is None:
            return
        verts = []
        for p in pts:
            fx, fy = self._frac(p)
            verts.append((self._data_x(fx), self._data_y(fy)))
        gate = PolygonGate([self.graph.x, self.graph.y], verts)
        self._finish_gate(gate, self._default_name(True))

    def _finish_polygon(self):
        pts = list(self._poly)
        self._poly = []
        self._create_polygon(pts)
        self.update()

    def _create_quadrants(self, pos: QPointF):
        if self._result.y_scale is None:
            return
        fx, fy = self._frac(pos)
        xv, yv = self._data_x(fx), self._data_y(fy)
        group = self.group or self.controller.current_group
        apply = bool(self.controller.preferences.get("apply_to_group_default")) or self.group is not None
        self.controller.begin_change("create quadrants")
        gates = make_quadrant_gates(self.graph.x, self.graph.y, xv, yv, self.sample.param_stain(self.graph.x), self.sample.param_stain(self.graph.y))
        created = []
        for name, gate in gates:
            base = name
            k = 2
            while self.population.child_by_name(name) is not None:
                name = f"{base} {k}"
                k += 1
            created.append(self.controller.add_population(self.population, Population(name, gate), group if apply else None))
        if created:
            self.set_selected(self.population.child_by_name(created[0].name) or created[0])
        self.toolDone.emit()

    def _create_bisector(self, pos: QPointF):
        fx, _ = self._frac(pos)
        xv = self._data_x(fx)
        stain = self.sample.param_stain(self.graph.x)
        group = self.group or self.controller.current_group
        apply = bool(self.controller.preferences.get("apply_to_group_default")) or self.group is not None
        self.controller.begin_change("create bisector")
        for name, lo, hi in ((f"{stain}-", None, xv), (f"{stain}+", xv, None)):
            base = name
            k = 2
            while self.population.child_by_name(name) is not None:
                name = f"{base} {k}"
                k += 1
            self.controller.add_population(self.population, Population(name, RectangleGate([self.graph.x], [lo], [hi])), group if apply else None)
        self.toolDone.emit()


def _confirm(parent, title, text) -> bool:
    return QMessageBox.question(parent, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes) == QMessageBox.StandardButton.Yes
