"""Procedurally drawn icons (original artwork, no external files)."""
from __future__ import annotations

import math
from functools import lru_cache

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QFont

INK = QColor("#3c4043")
BLUE = QColor("#2f6fd6")
LIGHT_BLUE = QColor("#9cc0f5")
GREEN = QColor("#2e9e5b")
ORANGE = QColor("#e8912d")
RED = QColor("#d64545")
PURPLE = QColor("#7c4dd6")
GREY = QColor("#b9bec5")
PAPER = QColor("#ffffff")
YELLOW = QColor("#f2c94c")


def _pen(color=INK, width=1.8):
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _page(p: QPainter, r: QRectF):
    p.setPen(_pen(INK, 1.6))
    p.setBrush(PAPER)
    p.drawRoundedRect(r, 2, 2)


def _tube(p: QPainter, x: float, y: float, w: float, h: float, fill=LIGHT_BLUE):
    path = QPainterPath()
    path.moveTo(x, y)
    path.lineTo(x + w, y)
    path.lineTo(x + w, y + h * 0.7)
    path.arcTo(QRectF(x, y + h * 0.4, w, h * 0.6), 0, -180)
    path.lineTo(x, y)
    p.setPen(_pen(INK, 1.6))
    p.setBrush(fill)
    p.drawPath(path)
    p.setPen(_pen(INK, 1.6))
    p.drawLine(QPointF(x - 2, y), QPointF(x + w + 2, y))


def _scatter(p: QPainter, r: QRectF, color=BLUE, n=18, seed=3):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    rnd = seed
    for i in range(n):
        rnd = (rnd * 1103515245 + 12345) & 0x7FFFFFFF
        fx = (rnd % 1000) / 1000.0
        rnd = (rnd * 1103515245 + 12345) & 0x7FFFFFFF
        fy = (rnd % 1000) / 1000.0
        # cluster near the centre
        fx = 0.5 + (fx - 0.5) * 0.7
        fy = 0.5 + (fy - 0.5) * 0.7
        p.drawEllipse(QPointF(r.left() + fx * r.width(), r.top() + fy * r.height()), 1.4, 1.4)


def _axes(p: QPainter, r: QRectF):
    p.setPen(_pen(INK, 1.6))
    p.drawLine(QPointF(r.left(), r.top()), QPointF(r.left(), r.bottom()))
    p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))


def _draw(name: str, p: QPainter, s: float):
    """Draw icon `name` into a s x s canvas."""
    m = s * 0.12
    r = QRectF(m, m, s - 2 * m, s - 2 * m)
    cx, cy = s / 2, s / 2

    if name == "new":
        _page(p, QRectF(m + 3, m, r.width() - 6, r.height()))
        p.setPen(_pen(GREEN, 2.2))
        p.drawLine(QPointF(cx, cy - 5), QPointF(cx, cy + 5))
        p.drawLine(QPointF(cx - 5, cy), QPointF(cx + 5, cy))
    elif name == "open" or name == "folder":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(YELLOW)
        path = QPainterPath()
        path.moveTo(r.left(), r.top() + 4)
        path.lineTo(r.left() + 9, r.top() + 4)
        path.lineTo(r.left() + 12, r.top() + 7)
        path.lineTo(r.right(), r.top() + 7)
        path.lineTo(r.right(), r.bottom())
        path.lineTo(r.left(), r.bottom())
        path.closeSubpath()
        p.drawPath(path)
        if name == "open":
            p.setBrush(QColor("#f7dd7a"))
            p.drawRect(QRectF(r.left() + 3, r.top() + 11, r.width() - 6, r.height() - 11))
    elif name in ("save", "save-as"):
        p.setPen(_pen(INK, 1.6))
        p.setBrush(BLUE)
        p.drawRoundedRect(r, 2, 2)
        p.setBrush(PAPER)
        p.drawRect(QRectF(r.left() + 5, r.top() + 1, r.width() - 10, 7))
        p.setBrush(QColor("#dde6f7"))
        p.drawRect(QRectF(r.left() + 4, r.bottom() - 9, r.width() - 8, 8))
        if name == "save-as":
            p.setPen(_pen(ORANGE, 2.0))
            p.drawLine(QPointF(r.right() - 6, r.bottom() - 6), QPointF(r.right() + 1, r.bottom() + 1))
    elif name == "add-samples":
        _tube(p, cx - 4, m + 2, 8, s - 2 * m - 2)
        p.setPen(_pen(GREEN, 2.4))
        p.drawLine(QPointF(r.right() - 4, r.top() + 1), QPointF(r.right() - 4, r.top() + 9))
        p.drawLine(QPointF(r.right() - 8, r.top() + 5), QPointF(r.right(), r.top() + 5))
    elif name == "add-folder":
        _draw("folder", p, s)
        p.setPen(_pen(GREEN, 2.4))
        p.drawLine(QPointF(cx, cy + 1), QPointF(cx, cy + 9))
        p.drawLine(QPointF(cx - 4, cy + 5), QPointF(cx + 4, cy + 5))
    elif name == "sample":
        _tube(p, cx - 5, m + 1, 10, s - 2 * m - 1)
    elif name == "sample-missing":
        _tube(p, cx - 5, m + 1, 10, s - 2 * m - 1, fill=GREY)
        p.setPen(_pen(RED, 2.2))
        p.drawLine(QPointF(r.right() - 7, r.bottom() - 7), QPointF(r.right(), r.bottom()))
        p.drawLine(QPointF(r.right(), r.bottom() - 7), QPointF(r.right() - 7, r.bottom()))
    elif name == "remove" or name == "delete":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(GREY)
        p.drawRect(QRectF(r.left() + 4, r.top() + 5, r.width() - 8, r.height() - 5))
        p.drawRect(QRectF(r.left() + 2, r.top() + 3, r.width() - 4, 3))
        p.drawLine(QPointF(cx - 3, r.top() + 1), QPointF(cx + 3, r.top() + 1))
        p.setPen(_pen(PAPER, 1.4))
        for dx in (-3, 0, 3):
            p.drawLine(QPointF(cx + dx, r.top() + 9), QPointF(cx + dx, r.bottom() - 3))
    elif name == "layout-editor":
        _page(p, r)
        for i, (fx, fy) in enumerate(((0.12, 0.12), (0.55, 0.12), (0.12, 0.55))):
            sub = QRectF(r.left() + fx * r.width(), r.top() + fy * r.height(), r.width() * 0.33, r.height() * 0.33)
            _axes(p, sub)
            _scatter(p, sub, [BLUE, GREEN, ORANGE][i], n=8, seed=i + 1)
    elif name == "table-editor" or name == "table":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(PAPER)
        p.drawRect(r)
        p.setBrush(LIGHT_BLUE)
        p.drawRect(QRectF(r.left(), r.top(), r.width(), 5))
        for i in (1, 2, 3):
            y = r.top() + 5 + i * (r.height() - 5) / 4
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        for i in (1, 2):
            x = r.left() + i * r.width() / 3
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
    elif name == "compensation":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(PAPER)
        p.drawRect(r)
        n = 3
        for i in range(n):
            for j in range(n):
                cell = QRectF(r.left() + i * r.width() / n, r.top() + j * r.height() / n, r.width() / n, r.height() / n)
                p.setBrush(BLUE if i == j else (QColor("#e4ecfa") if abs(i - j) == 1 else PAPER))
                p.drawRect(cell)
    elif name == "group":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(QColor("#f4d35e"))
        path = QPainterPath()
        path.moveTo(r.left(), r.top() + 3)
        path.lineTo(r.left() + 8, r.top() + 3)
        path.lineTo(r.left() + 10, r.top() + 6)
        path.lineTo(r.right(), r.top() + 6)
        path.lineTo(r.right(), r.bottom())
        path.lineTo(r.left(), r.bottom())
        path.closeSubpath()
        p.drawPath(path)
        for k in range(3):
            _tube(p, r.left() + 4 + k * 6, r.top() + 9, 4, r.height() - 11, fill=LIGHT_BLUE)
    elif name == "statistic" or name == "stats":
        p.setPen(_pen(BLUE, 2.4))
        path = QPainterPath()
        path.moveTo(r.right() - 2, r.top() + 2)
        path.lineTo(r.left() + 2, r.top() + 2)
        path.lineTo(cx + 1, cy)
        path.lineTo(r.left() + 2, r.bottom() - 2)
        path.lineTo(r.right() - 2, r.bottom() - 2)
        p.drawPath(path)
    elif name in ("boolean-and", "boolean-or", "boolean-not", "gate-boolean"):
        p.setPen(_pen(INK, 1.6))
        p.setBrush(QColor(BLUE.red(), BLUE.green(), BLUE.blue(), 90))
        a = QRectF(r.left(), r.top() + 3, r.width() * 0.62, r.height() - 6)
        b = QRectF(r.right() - r.width() * 0.62, r.top() + 3, r.width() * 0.62, r.height() - 6)
        if name == "boolean-and":
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(a)
            p.drawEllipse(b)
            inter = QPainterPath()
            inter.addEllipse(a)
            inter2 = QPainterPath()
            inter2.addEllipse(b)
            p.setBrush(BLUE)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPath(inter.intersected(inter2))
        elif name == "boolean-or":
            p.setBrush(BLUE)
            p.drawEllipse(a)
            p.drawEllipse(b)
        elif name == "boolean-not":
            p.setBrush(BLUE)
            p.drawRect(r)
            p.setBrush(PAPER)
            p.drawEllipse(QPointF(cx, cy), r.width() * 0.28, r.height() * 0.28)
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(a)
            p.drawEllipse(b)
            p.setPen(_pen(INK, 1.0))
            p.setFont(QFont("Helvetica", int(s * 0.3), QFont.Weight.Bold))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "∧")
    elif name == "keyword":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(QColor("#fff3bf"))
        path = QPainterPath()
        path.moveTo(r.left(), r.top() + 2)
        path.lineTo(r.right() - 6, r.top() + 2)
        path.lineTo(r.right(), cy)
        path.lineTo(r.right() - 6, r.bottom() - 2)
        path.lineTo(r.left(), r.bottom() - 2)
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(INK)
        p.drawEllipse(QPointF(r.left() + 5, cy), 1.6, 1.6)
    elif name == "preferences" or name == "options":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(GREY)
        path = QPainterPath()
        teeth = 8
        for i in range(teeth * 2):
            ang = math.pi * i / teeth
            rad = r.width() / 2 if i % 2 == 0 else r.width() / 2 - 3
            pt = QPointF(cx + rad * math.cos(ang), cy + rad * math.sin(ang))
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(PAPER)
        p.drawEllipse(QPointF(cx, cy), r.width() * 0.18, r.width() * 0.18)
    elif name in ("undo", "redo"):
        p.setPen(_pen(INK, 2.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(r.left() + 3, r.top() + 4, r.width() - 6, r.height() - 4)
        if name == "undo":
            p.drawArc(rect, 20 * 16, 250 * 16)
            tip = QPointF(rect.left() + 1, rect.center().y() + 1)
            p.setBrush(INK)
            p.drawPolygon(QPolygonF([tip, QPointF(tip.x() + 6, tip.y() - 6), QPointF(tip.x() + 7, tip.y() + 2)]))
        else:
            p.drawArc(rect, -90 * 16, 250 * 16)
            tip = QPointF(rect.right() - 1, rect.center().y() + 1)
            p.setBrush(INK)
            p.drawPolygon(QPolygonF([tip, QPointF(tip.x() - 6, tip.y() - 6), QPointF(tip.x() - 7, tip.y() + 2)]))
    elif name == "copy":
        _page(p, QRectF(r.left(), r.top(), r.width() - 6, r.height() - 6))
        _page(p, QRectF(r.left() + 6, r.top() + 6, r.width() - 6, r.height() - 6))
    elif name == "paste":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(QColor("#d9b36c"))
        p.drawRoundedRect(QRectF(r.left() + 1, r.top() + 3, r.width() - 2, r.height() - 3), 2, 2)
        _page(p, QRectF(r.left() + 5, r.top() + 8, r.width() - 10, r.height() - 9))
        p.setBrush(GREY)
        p.drawRect(QRectF(cx - 4, r.top(), 8, 5))
    elif name == "cut":
        p.setPen(_pen(INK, 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(r.left() + 5, r.bottom() - 4), 3.5, 3.5)
        p.drawEllipse(QPointF(r.right() - 5, r.bottom() - 4), 3.5, 3.5)
        p.drawLine(QPointF(r.left() + 7, r.bottom() - 7), QPointF(r.right() - 4, r.top()))
        p.drawLine(QPointF(r.right() - 7, r.bottom() - 7), QPointF(r.left() + 4, r.top()))
    elif name == "rename":
        p.setPen(_pen(INK, 1.8))
        p.drawLine(QPointF(cx, r.top() + 2), QPointF(cx, r.bottom() - 2))
        p.drawLine(QPointF(cx - 4, r.top() + 2), QPointF(cx + 4, r.top() + 2))
        p.drawLine(QPointF(cx - 4, r.bottom() - 2), QPointF(cx + 4, r.bottom() - 2))
        p.setPen(_pen(GREY, 1.4))
        p.drawRect(QRectF(r.left(), r.top() + 6, r.width(), r.height() - 12))
    elif name in ("expand", "collapse"):
        p.setPen(_pen(INK, 1.6))
        for i in range(3):
            y = r.top() + 3 + i * 7
            x = r.left() + (0 if i == 0 else 6)
            p.drawLine(QPointF(x + 6, y), QPointF(r.right(), y))
            p.setBrush(INK)
            if name == "expand":
                p.drawPolygon(QPolygonF([QPointF(x, y - 3), QPointF(x + 4, y), QPointF(x, y + 3)]))
            else:
                p.drawPolygon(QPolygonF([QPointF(x - 1, y - 2), QPointF(x + 5, y - 2), QPointF(x + 2, y + 2)]))
    elif name in ("export", "import"):
        _page(p, QRectF(r.left() + 4, r.top(), r.width() - 8, r.height()))
        p.setPen(_pen(BLUE, 2.4))
        p.setBrush(BLUE)
        if name == "export":
            p.drawLine(QPointF(cx, r.bottom() - 2), QPointF(cx, cy - 2))
            p.drawPolygon(QPolygonF([QPointF(cx - 5, cy), QPointF(cx + 5, cy), QPointF(cx, cy - 6)]))
        else:
            p.drawLine(QPointF(cx, cy - 6), QPointF(cx, r.bottom() - 6))
            p.drawPolygon(QPolygonF([QPointF(cx - 5, r.bottom() - 7), QPointF(cx + 5, r.bottom() - 7), QPointF(cx, r.bottom() - 1)]))
    elif name == "gate-rectangle":
        _axes(p, r)
        _scatter(p, r, n=14)
        p.setPen(_pen(RED, 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(r.left() + 5, r.top() + 4, r.width() * 0.55, r.height() * 0.55))
    elif name == "gate-polygon":
        _axes(p, r)
        _scatter(p, r, n=14)
        p.setPen(_pen(RED, 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        poly = QPolygonF([QPointF(r.left() + 5, r.top() + 10), QPointF(cx, r.top() + 2), QPointF(r.right() - 2, r.top() + 8), QPointF(r.right() - 5, cy + 5), QPointF(cx - 2, r.bottom() - 4)])
        p.drawPolygon(poly)
    elif name == "gate-ellipse":
        _axes(p, r)
        _scatter(p, r, n=14)
        p.setPen(_pen(RED, 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.save()
        p.translate(cx + 1, cy - 1)
        p.rotate(-30)
        p.drawEllipse(QPointF(0, 0), r.width() * 0.4, r.height() * 0.25)
        p.restore()
    elif name == "gate-quadrant":
        _axes(p, r)
        _scatter(p, r, n=14)
        p.setPen(_pen(RED, 1.8))
        p.drawLine(QPointF(cx + 1, r.top()), QPointF(cx + 1, r.bottom()))
        p.drawLine(QPointF(r.left(), cy - 1), QPointF(r.right(), cy - 1))
    elif name == "gate-range":
        _axes(p, r)
        p.setPen(_pen(BLUE, 1.8))
        path = QPainterPath()
        path.moveTo(r.left(), r.bottom())
        path.cubicTo(r.left() + 6, r.bottom(), r.left() + 8, r.top() + 2, cx, r.top() + 2)
        path.cubicTo(cx + 6, r.top() + 2, r.right() - 6, r.bottom(), r.right(), r.bottom())
        p.drawPath(path)
        p.setPen(_pen(RED, 1.8))
        p.drawLine(QPointF(cx - 5, r.top() + 5), QPointF(cx + 7, r.top() + 5))
        p.drawLine(QPointF(cx - 5, r.top() + 2), QPointF(cx - 5, r.top() + 8))
        p.drawLine(QPointF(cx + 7, r.top() + 2), QPointF(cx + 7, r.top() + 8))
    elif name == "gate-bisector":
        _axes(p, r)
        p.setPen(_pen(BLUE, 1.8))
        path = QPainterPath()
        path.moveTo(r.left(), r.bottom())
        path.cubicTo(r.left() + 4, r.bottom(), r.left() + 4, r.top() + 4, r.left() + 9, r.top() + 4)
        path.cubicTo(r.left() + 13, r.top() + 4, cx, r.bottom() - 2, cx + 3, r.bottom() - 2)
        path.cubicTo(r.right() - 6, r.bottom() - 2, r.right() - 6, r.top() + 8, r.right(), r.bottom())
        p.drawPath(path)
        p.setPen(_pen(RED, 1.8))
        p.drawLine(QPointF(cx + 1, r.top()), QPointF(cx + 1, r.bottom()))
    elif name == "gate-freehand":
        _axes(p, r)
        _scatter(p, r, n=12)
        p.setPen(_pen(RED, 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(r.left() + 5, cy)
        path.cubicTo(r.left() + 3, r.top(), cx + 4, r.top() - 2, r.right() - 3, r.top() + 7)
        path.cubicTo(r.right() + 2, cy + 2, cx + 4, r.bottom() - 2, cx - 3, r.bottom() - 5)
        path.cubicTo(r.left() + 9, r.bottom() - 7, r.left() + 7, cy + 4, r.left() + 5, cy)
        p.drawPath(path)
    elif name == "pointer":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(PAPER)
        poly = QPolygonF([QPointF(r.left() + 4, r.top() + 2), QPointF(r.left() + 4, r.bottom() - 4), QPointF(r.left() + 8, r.bottom() - 8), QPointF(r.left() + 11, r.bottom()), QPointF(r.left() + 14, r.bottom() - 2), QPointF(r.left() + 11, r.bottom() - 9), QPointF(r.right() - 3, r.bottom() - 9)])
        p.drawPolygon(poly)
    elif name == "zoom":
        p.setPen(_pen(INK, 2.0))
        p.setBrush(QColor("#e4ecfa"))
        p.drawEllipse(QPointF(cx - 2, cy - 2), r.width() * 0.32, r.width() * 0.32)
        p.drawLine(QPointF(cx + 4, cy + 4), QPointF(r.right(), r.bottom()))
        p.setPen(_pen(INK, 1.6))
        p.drawLine(QPointF(cx - 5, cy - 2), QPointF(cx + 1, cy - 2))
        p.drawLine(QPointF(cx - 2, cy - 5), QPointF(cx - 2, cy + 1))
    elif name == "pseudocolor":
        _axes(p, r)
        for color, scale, n in ((BLUE, 1.0, 16), (GREEN, 0.7, 10), (ORANGE, 0.45, 7), (RED, 0.25, 4)):
            sub = QRectF(r.center().x() - r.width() * scale / 2 + 2, r.center().y() - r.height() * scale / 2, r.width() * scale, r.height() * scale)
            _scatter(p, sub, color, n=n, seed=int(scale * 10))
    elif name == "dot":
        _axes(p, r)
        _scatter(p, r, INK, n=20)
    elif name == "contour":
        _axes(p, r)
        p.setPen(_pen(BLUE, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k in (0.42, 0.3, 0.17):
            p.drawEllipse(QPointF(cx + 2, cy - 1), r.width() * k, r.height() * k * 0.8)
    elif name == "zebra":
        _axes(p, r)
        for k, col in ((0.42, LIGHT_BLUE), (0.3, GREEN), (0.17, ORANGE)):
            p.setPen(_pen(INK, 1.0))
            p.setBrush(col)
            p.drawEllipse(QPointF(cx + 2, cy - 1), r.width() * k, r.height() * k * 0.8)
    elif name == "density":
        _axes(p, r)
        for k, col in ((0.42, QColor("#c8c8c8")), (0.3, QColor("#8a8a8a")), (0.17, QColor("#303030"))):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawEllipse(QPointF(cx + 2, cy - 1), r.width() * k, r.height() * k * 0.8)
    elif name == "histogram":
        _axes(p, r)
        p.setPen(_pen(BLUE, 1.6))
        p.setBrush(LIGHT_BLUE)
        path = QPainterPath()
        path.moveTo(r.left(), r.bottom())
        path.cubicTo(r.left() + 6, r.bottom(), r.left() + 6, r.top() + 2, cx, r.top() + 2)
        path.cubicTo(cx + 6, r.top() + 2, r.right() - 6, r.bottom(), r.right(), r.bottom())
        path.closeSubpath()
        p.drawPath(path)
    elif name == "transform":
        _axes(p, r)
        p.setPen(_pen(BLUE, 1.8))
        path = QPainterPath()
        path.moveTo(r.left(), r.bottom() - 2)
        path.cubicTo(cx - 2, r.bottom() - 2, cx + 1, r.top() + 2, r.right(), r.top() + 2)
        p.drawPath(path)
        p.setPen(_pen(INK, 1.0))
        p.setFont(QFont("Helvetica", int(s * 0.32), QFont.Weight.Bold))
        p.drawText(QRectF(r.left() + 2, r.top(), r.width() * 0.5, r.height() * 0.5), Qt.AlignmentFlag.AlignCenter, "T")
    elif name == "graph":
        _axes(p, r)
        _scatter(p, r, BLUE, n=18)
    elif name == "text":
        p.setPen(_pen(INK, 1.0))
        p.setFont(QFont("Helvetica", int(s * 0.55), QFont.Weight.Bold))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "A")
    elif name == "image":
        _page(p, r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(GREEN)
        p.drawPolygon(QPolygonF([QPointF(r.left() + 2, r.bottom() - 2), QPointF(r.left() + 9, r.top() + 9), QPointF(r.right() - 2, r.bottom() - 2)]))
        p.setBrush(ORANGE)
        p.drawEllipse(QPointF(r.right() - 7, r.top() + 7), 2.5, 2.5)
    elif name == "grid":
        p.setPen(_pen(GREY, 1.2))
        for i in range(4):
            x = r.left() + i * r.width() / 3
            y = r.top() + i * r.height() / 3
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
    elif name == "page":
        _page(p, QRectF(r.left() + 3, r.top(), r.width() - 6, r.height()))
        p.setPen(_pen(GREY, 1.4))
        for i in range(3):
            y = r.top() + 6 + i * 5
            p.drawLine(QPointF(r.left() + 7, y), QPointF(r.right() - 7, y))
    elif name == "print":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(GREY)
        p.drawRoundedRect(QRectF(r.left(), cy - 3, r.width(), r.height() * 0.45), 2, 2)
        _page(p, QRectF(r.left() + 5, r.top(), r.width() - 10, r.height() * 0.4))
        _page(p, QRectF(r.left() + 5, cy + 3, r.width() - 10, r.height() * 0.4))
    elif name == "play":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(GREEN)
        p.drawPolygon(QPolygonF([QPointF(r.left() + 4, r.top() + 1), QPointF(r.right() - 1, cy), QPointF(r.left() + 4, r.bottom() - 1)]))
    elif name == "help" or name == "about":
        p.setPen(_pen(BLUE, 1.8))
        p.setBrush(PAPER)
        p.drawEllipse(QPointF(cx, cy), r.width() / 2, r.height() / 2)
        p.setPen(_pen(BLUE, 1.0))
        p.setFont(QFont("Helvetica", int(s * 0.42), QFont.Weight.Bold))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "?" if name == "help" else "i")
    elif name == "wizard":
        p.setPen(_pen(PURPLE, 2.0))
        p.drawLine(QPointF(r.left() + 2, r.bottom() - 2), QPointF(r.right() - 5, r.top() + 5))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(YELLOW)
        for pt, rad in ((QPointF(r.right() - 4, r.top() + 4), 3.0), (QPointF(r.right() - 10, r.top() + 1), 1.4), (QPointF(r.right() - 1, r.top() + 10), 1.4)):
            p.drawEllipse(pt, rad, rad)
    elif name == "check":
        p.setPen(_pen(GREEN, 2.6))
        p.drawPolyline(QPolygonF([QPointF(r.left() + 2, cy), QPointF(cx - 2, r.bottom() - 3), QPointF(r.right() - 1, r.top() + 3)]))
    elif name == "warning":
        p.setPen(_pen(INK, 1.4))
        p.setBrush(YELLOW)
        p.drawPolygon(QPolygonF([QPointF(cx, r.top()), QPointF(r.right(), r.bottom()), QPointF(r.left(), r.bottom())]))
        p.setPen(_pen(INK, 2.0))
        p.drawLine(QPointF(cx, r.top() + 7), QPointF(cx, r.bottom() - 6))
        p.drawPoint(QPointF(cx, r.bottom() - 3))
    elif name in ("plus", "minus"):
        p.setPen(_pen(INK, 2.4))
        p.drawLine(QPointF(r.left() + 3, cy), QPointF(r.right() - 3, cy))
        if name == "plus":
            p.drawLine(QPointF(cx, r.top() + 3), QPointF(cx, r.bottom() - 3))
    elif name in ("up", "down", "left", "right"):
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(INK)
        if name == "up":
            poly = [QPointF(cx, r.top() + 3), QPointF(r.right() - 3, r.bottom() - 4), QPointF(r.left() + 3, r.bottom() - 4)]
        elif name == "down":
            poly = [QPointF(cx, r.bottom() - 3), QPointF(r.right() - 3, r.top() + 4), QPointF(r.left() + 3, r.top() + 4)]
        elif name == "left":
            poly = [QPointF(r.left() + 3, cy), QPointF(r.right() - 4, r.top() + 3), QPointF(r.right() - 4, r.bottom() - 3)]
        else:
            poly = [QPointF(r.right() - 3, cy), QPointF(r.left() + 4, r.top() + 3), QPointF(r.left() + 4, r.bottom() - 3)]
        p.drawPolygon(QPolygonF(poly))
    elif name == "overlay":
        _axes(p, r)
        for col, dx in ((BLUE, 0), (RED, 5)):
            p.setPen(_pen(col, 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            path.moveTo(r.left(), r.bottom())
            path.cubicTo(r.left() + 6 + dx, r.bottom(), r.left() + 6 + dx, r.top() + 3, cx + dx - 3, r.top() + 3)
            path.cubicTo(cx + 3 + dx, r.top() + 3, r.right() - 6 + dx, r.bottom(), r.right(), r.bottom())
            p.drawPath(path)
    elif name == "ancestry":
        for i in range(3):
            sub = QRectF(r.left() + i * r.width() / 3 + 1, r.top() + 4, r.width() / 3 - 3, r.height() - 8)
            _axes(p, sub)
            _scatter(p, sub, BLUE, n=5, seed=i + 2)
            if i < 2:
                p.setPen(_pen(INK, 1.2))
                p.drawLine(QPointF(sub.right() + 0.5, sub.center().y()), QPointF(sub.right() + 2.5, sub.center().y()))
    elif name == "workspace":
        _page(p, r)
        p.setPen(_pen(INK, 1.4))
        for i in range(3):
            y = r.top() + 5 + i * 6
            x = r.left() + 4 + (0 if i == 0 else 4)
            p.drawLine(QPointF(x, y), QPointF(x + 4, y))
            p.setPen(_pen(BLUE, 1.6))
            p.drawLine(QPointF(x + 6, y), QPointF(r.right() - 4, y))
            p.setPen(_pen(INK, 1.4))
    elif name == "gate-not":
        p.setPen(_pen(INK, 1.6))
        p.setBrush(BLUE)
        p.drawRect(r)
        p.setBrush(PAPER)
        p.drawEllipse(QPointF(cx, cy), r.width() * 0.28, r.height() * 0.28)
    elif name == "derive":
        p.setPen(_pen(INK, 1.0))
        p.setFont(QFont("Helvetica", int(s * 0.36), QFont.Weight.Bold))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, "ƒx")
    elif name == "app":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(BLUE)
        p.drawRoundedRect(QRectF(1, 1, s - 2, s - 2), s * 0.22, s * 0.22)
        inner = QRectF(s * 0.22, s * 0.2, s * 0.6, s * 0.6)
        p.setPen(_pen(PAPER, s * 0.07))
        p.drawLine(QPointF(inner.left(), inner.top()), QPointF(inner.left(), inner.bottom()))
        p.drawLine(QPointF(inner.left(), inner.bottom()), QPointF(inner.right(), inner.bottom()))
        _scatter(p, inner, PAPER, n=14, seed=5)
        p.setPen(_pen(YELLOW, s * 0.07))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(inner.center().x() + 1, inner.center().y() - 1), inner.width() * 0.3, inner.height() * 0.24)
    else:
        p.setPen(_pen(GREY, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)


@lru_cache(maxsize=512)
def pixmap(name: str, size: int = 32) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    _draw(name, p, float(size))
    p.end()
    return pm


@lru_cache(maxsize=256)
def icon(name: str) -> QIcon:
    ic = QIcon()
    for size in (16, 24, 32, 48, 64):
        ic.addPixmap(pixmap(name, size))
    return ic


GATE_ICON_BY_KIND = {
    "rectangle": "gate-rectangle",
    "polygon": "gate-polygon",
    "ellipse": "gate-ellipse",
    "boolean": "gate-boolean",
}


def gate_icon_name(gate) -> str:
    if gate is None:
        return "sample"
    if gate.kind == "rectangle":
        if gate.quadrant:
            return "gate-quadrant"
        if len(gate.dims) == 1:
            return "gate-range"
        return "gate-rectangle"
    if gate.kind == "boolean":
        return {"and": "boolean-and", "or": "boolean-or", "not": "boolean-not"}.get(gate.op, "gate-boolean")
    return GATE_ICON_BY_KIND.get(gate.kind, "gate-rectangle")
