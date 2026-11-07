"""Ellipse / Boolean gates through the UI, and the editor dialogs."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402

from flowdesk.core.gates import EllipseGate, RectangleGate  # noqa: E402
from flowdesk.core.layouts import TableDef  # noqa: E402
from flowdesk.core.model import Population  # noqa: E402
from flowdesk.ui import plot_widget as plot_widget_module  # noqa: E402

FOLDER = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel")
FCS = sorted(FOLDER.glob("*.fcs")) if FOLDER.exists() else []

pytestmark = pytest.mark.skipif(len(FCS) < 2, reason="real test data not available")


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture
def window(app, monkeypatch):
    from flowdesk.ui.main_window import MainWindow

    class FakeNameDialog:
        def __init__(self, default_name, group=None, apply_default=False, parent=None):
            self.default_name = default_name

        def exec(self):
            return True

        def values(self):
            return (self.default_name, False)

    monkeypatch.setattr(plot_widget_module, "GateNameDialog", FakeNameDialog)
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    win.controller.add_samples(FCS[:2])
    app.processEvents()
    yield win
    for gw in list(win.graph_windows.values()):
        gw.close()
    win.controller.workspace.modified = False
    win.hide()


def _mouse(app, widget, typ, pos):
    button = Qt.MouseButton.LeftButton
    ev = QMouseEvent(typ, pos, widget.mapToGlobal(pos.toPoint()), button, button, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(widget, ev)


def _drag(app, widget, a: QPointF, b: QPointF):
    _mouse(app, widget, QEvent.Type.MouseButtonPress, a)
    mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
    _mouse(app, widget, QEvent.Type.MouseMove, mid)
    _mouse(app, widget, QEvent.Type.MouseMove, b)
    _mouse(app, widget, QEvent.Type.MouseButtonRelease, b)
    app.processEvents()


def test_ellipse_gate_drawn_and_edited_with_the_mouse(app, window):
    c = window.controller
    s0 = c.workspace.samples[0]
    gw = window.open_graph(s0, s0.root)
    gw.resize(620, 660)
    app.processEvents()
    gw.plot.set_axes(x="FSC-A", y="SSC-A")
    gw.plot.set_tool("ellipse")
    app.processEvents()
    pw = gw.plot
    pw.repaint()
    res = pw.result()
    _drag(app, pw, res.to_pixel(0.25, 0.1), res.to_pixel(0.6, 0.45))
    pop = next((p for p in s0.root.children if isinstance(p.gate, EllipseGate)), None)
    assert pop is not None, "ellipse gate was not created"
    assert pop.name == "FSC-A, SSC-A subset"  # default name follows the familiar convention
    n0 = pop.count
    assert 0 < n0 < s0.n_events
    assert pw.tool == "pointer" and pw.selected is pop
    # rotate / stretch by dragging the first axis handle
    pw.repaint()
    shape = pw._shape_for(pop)
    handles = pw._handles(shape)
    assert len(handles) == 4
    a_end = handles[0]
    _drag(app, pw, a_end, QPointF(a_end.x() + 40, a_end.y() - 50))
    pop2 = s0.root.find(pop.path)
    assert pop2.count != n0
    assert c.can_undo
    # move the whole ellipse
    pw.repaint()
    shape = pw._shape_for(pop2)
    centre = shape.bbox.center()
    before = pop2.gate.center
    _drag(app, pw, centre, QPointF(centre.x() + 30, centre.y() + 30))
    assert s0.root.find(pop.path).gate.center != before
    # the freehand tool produces a polygon
    gw.plot.set_tool("freehand")
    pw.repaint()
    res = pw.result()
    pts = [res.to_pixel(0.3 + 0.2 * dx, 0.5 + 0.2 * dy) for dx, dy in ((0, 0), (1, 0), (1, 1), (0.5, 1.3), (0, 1))]
    _mouse(app, pw, QEvent.Type.MouseButtonPress, pts[0])
    for p in pts[1:]:
        _mouse(app, pw, QEvent.Type.MouseMove, p)
    _mouse(app, pw, QEvent.Type.MouseButtonRelease, pts[-1])
    app.processEvents()
    assert any(p.gate.kind == "polygon" for p in s0.root.children)
    gw.close()


def test_boolean_gate_dialog_and_creation(app, window):
    from flowdesk.ui.dialogs import BooleanGateDialog

    c = window.controller
    s0 = c.workspace.samples[0]
    a = c.add_population(s0.root, Population("A", RectangleGate(["FSC-A", "SSC-A"], [20000, 0], [90000, None])), None)
    b = c.add_population(s0.root, Population("B", RectangleGate(["FSC-A", "SSC-A"], [60000, 0], [150000, None])), None)
    dlg = BooleanGateDialog([a, b], "or")
    dlg.check([a, b])
    op, refs, name = dlg.values()
    assert op == "or" and refs == ["/A", "/B"] and name == "A OR B"
    pop = c.create_boolean(s0.root, op, refs, name, None)
    union = (a.mask() | b.mask()).sum()
    assert pop.count == union
    dlg.op_and.setChecked(True)
    op, refs, name = dlg.values()
    assert op == "and" and name == "A AND B"
    both = c.create_boolean(s0.root, op, refs, name, None)
    assert both.count == (a.mask() & b.mask()).sum()
    dlg.op_not.setChecked(True)
    dlg.list.item(1).setCheckState(Qt.CheckState.Unchecked)
    op, refs, name = dlg.values()
    assert op == "not" and refs == ["/A"] and name == "NOT A"
    nota = c.create_boolean(s0.root, op, refs, name, None)
    assert nota.count == s0.n_events - a.count
    # the tree shows the boolean rows with their icons and the Statistic column
    app.processEvents()
    names = [window.tree.topLevelItem(0).child(i).text(0) for i in range(window.tree.topLevelItem(0).childCount())]
    assert {"A OR B", "A AND B", "NOT A"} <= set(names)


def test_table_editor_dialogs(app, window):
    from flowdesk.ui.table_editor import ColumnDialog, KeywordDialog, build_table

    c = window.controller
    s0 = c.workspace.samples[0]
    c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 5000], [200000, 150000])), c.current_group)
    te = window.show_table_editor()
    group = c.current_group
    dlg = ColumnDialog(c, group)
    dlg.pop_combo.setCurrentIndex(dlg.pop_combo.findText("Cells") if dlg.pop_combo.findText("Cells") >= 0 else max(0, dlg.pop_combo.count() - 1))
    dlg.stat_combo.setCurrentText("Median")
    dlg.param_combo.setCurrentIndex(dlg.param_combo.findData("FSC-A") if dlg.param_combo.findData("FSC-A") >= 0 else dlg.param_combo.findText("FSC-A"))
    dlg.label_edit.setText("FSC median of cells")
    col = dlg.column()
    assert col.statistic == "Median" and col.pop_path.endswith("Cells") and col.label == "FSC median of cells"
    assert col.param in ("FSC-A", "FSC-A :: FSC-A") or col.param == "FSC-A"
    kd = KeywordDialog(["$CYT", "$DATE"])
    for i in range(kd.list.count()) if hasattr(kd, "list") else []:
        kd.list.item(i).setCheckState(Qt.CheckState.Checked)
    td = te.current_table()
    td.columns.append(col)
    headers, rows = build_table(c.workspace, td, group)
    assert headers[0] == "Sample" and "FSC median of cells" in headers
    assert len(rows) == len(c.workspace.members(group))
    assert all(float(r[headers.index("FSC median of cells")]) > 0 for r in rows)
    te.close()


def test_layout_editor_mouse_drag_moves_item(app, window):
    from PySide6.QtTest import QTest

    c = window.controller
    s0 = c.workspace.samples[0]
    pop = c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 5000], [200000, 150000])), None)
    le = window.show_layout_editor()
    le.resize(1000, 800)
    app.processEvents()
    item = le.add_population(s0, pop)
    gi = le.scene_item(item)
    app.processEvents()
    view = le.view
    start_scene = QPointF(item.x + item.w / 2, item.y + item.h / 2)
    start = view.mapFromScene(start_scene)
    end = start + QPointF(120, 80).toPoint()
    x0, y0 = item.x, item.y
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(view.viewport(), start + QPointF(30, 20).toPoint())
    QTest.mouseMove(view.viewport(), end)
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    app.processEvents()
    le.flush_notify()
    scale = view.transform().m11() or 1.0
    assert item.x > x0 + 60 / scale and item.y > y0 + 40 / scale, (item.x, x0, item.y, y0, scale)
    # the model is what gets saved
    assert le.current_layout().items[0].x == item.x
    le.close()


def test_layout_editor_item_resize_and_move(app, window):
    c = window.controller
    s0 = c.workspace.samples[0]
    pop = c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 5000], [200000, 150000])), None)
    le = window.show_layout_editor()
    item = le.add_population(s0, pop)
    gi = le.scene_item(item)
    assert gi is not None
    start = QRectF(item.x, item.y, item.w, item.h)
    gi.resize_to("br", start, QPointF(60, 40))
    assert item.w == pytest.approx(start.width() + 60, abs=8) and item.h == pytest.approx(start.height() + 40, abs=8)
    gi.setPos(item.x + 100, item.y + 50)
    app.processEvents()
    le.flush_notify()
    assert item.x == pytest.approx(start.x() + 100, abs=8)
    le.close()
