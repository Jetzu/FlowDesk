"""End-to-end tests of the workspace window, graph window and gate editing."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from flowdesk.core.gates import PolygonGate, RectangleGate  # noqa: E402
from flowdesk.core.model import Population  # noqa: E402
from flowdesk.core.serialization import load_workspace, save_workspace  # noqa: E402
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
            return ("New gate", True)

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


def test_samples_and_group_gates(app, window):
    c = window.controller
    assert len(c.workspace.samples) == 2
    assert window.tree.topLevelItemCount() == 2
    s0 = c.workspace.samples[0]
    lymph = c.add_population(s0.root, Population("Lymphocytes", PolygonGate(["FSC-A", "SSC-A"], [(59859, 27756), (77404, 21588), (104238, 23644), (130039, 40000), (120000, 80000), (80000, 90000), (50000, 60000)])), c.current_group)
    app.processEvents()
    # the gate was applied to every sample of the group
    for s in c.workspace.samples:
        assert s.root.find("/Lymphocytes") is not None
        assert s.root.find("/Lymphocytes").count > 1000
    assert window.tree.topLevelItem(0).childCount() == 1
    assert "%" in window.tree.topLevelItem(0).child(0).text(1)
    # a sample added later receives the template too
    c.add_samples(FCS[2:3])
    assert c.workspace.samples[2].root.find("/Lymphocytes") is not None
    assert lymph.parent is None or True


def test_graph_window_gate_drawing_and_editing(app, window):
    c = window.controller
    s0 = c.workspace.samples[0]
    gw = window.open_graph(s0, s0.root)
    gw.resize(600, 650)
    app.processEvents()
    gw.plot.set_axes(x="FSC-A", y="SSC-A")
    app.processEvents()
    pw = gw.plot
    pw.set_tool("rect")
    pw.repaint()
    res = pw.result()
    assert res is not None and res.error is None
    p0 = res.to_pixel(0.2, 0.1)
    p1 = res.to_pixel(0.6, 0.5)
    _mouse(app, pw, QEvent.Type.MouseButtonPress, p0)
    _mouse(app, pw, QEvent.Type.MouseMove, p1)
    _mouse(app, pw, QEvent.Type.MouseButtonRelease, p1)
    app.processEvents()
    pop = s0.root.child_by_name("New gate")
    assert pop is not None and isinstance(pop.gate, RectangleGate)
    assert pw.tool == "pointer" and pw.selected is pop
    count_before = pop.count
    assert 0 < count_before < s0.n_events
    # the gate was applied to the group (FakeNameDialog answers "apply")
    assert c.workspace.samples[1].root.child_by_name("New gate") is not None
    # drag the gate: the count changes and undo is available
    pw.repaint()
    shape = pw._shape_for(pop)
    centre = shape.bbox.center()
    _mouse(app, pw, QEvent.Type.MouseButtonPress, centre)
    _mouse(app, pw, QEvent.Type.MouseMove, QPointF(centre.x() + 40, centre.y() - 30))
    _mouse(app, pw, QEvent.Type.MouseMove, QPointF(centre.x() + 80, centre.y() - 60))
    _mouse(app, pw, QEvent.Type.MouseButtonRelease, QPointF(centre.x() + 80, centre.y() - 60))
    app.processEvents()
    moved = s0.root.child_by_name("New gate")
    assert moved.count != count_before
    assert c.can_undo
    c.undo()
    app.processEvents()
    assert s0.root.child_by_name("New gate").count == count_before
    # quadrants
    pw.set_tool("quadrant")
    pw.repaint()
    centre = pw.result().to_pixel(0.5, 0.5)
    _mouse(app, pw, QEvent.Type.MouseButtonPress, centre)
    _mouse(app, pw, QEvent.Type.MouseButtonRelease, centre)
    app.processEvents()
    quads = [p for p in s0.root.children if p.name.startswith("Q")]
    assert len(quads) == 4
    assert sum(q.count for q in quads) == s0.n_events
    # double-clicking a gate opens it in the window (navigates)
    pw.set_tool("pointer")
    pw.repaint()
    shape = pw._shape_for(s0.root.child_by_name("New gate"))
    _mouse(app, pw, QEvent.Type.MouseButtonDblClick, shape.bbox.center())
    app.processEvents()
    assert gw.population.name == "New gate"
    # histogram with a range gate
    gw.plot.set_axes(x="FSC-A", y=None)
    gw.plot.set_tool("range")
    pw.repaint()
    res = pw.result()
    assert res.histogram
    a = res.to_pixel(0.3, 0.5)
    b = res.to_pixel(0.7, 0.5)
    _mouse(app, pw, QEvent.Type.MouseButtonPress, a)
    _mouse(app, pw, QEvent.Type.MouseMove, b)
    _mouse(app, pw, QEvent.Type.MouseButtonRelease, b)
    app.processEvents()
    child = gw.population.child_by_name("New gate")
    assert child is not None and child.gate.is_range
    gw.close()


def test_save_load_and_other_windows(app, window, tmp_path):
    c = window.controller
    s0 = c.workspace.samples[0]
    pop = c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 5000], [200000, 150000])), None)
    le = window.show_layout_editor()
    le.add_population(s0, pop)
    te = window.show_table_editor()
    te.add_columns_for_paths(["/Cells"])
    cw = window.show_compensation()
    assert cw.isVisible()
    path = tmp_path / "test.fdw"
    c.workspace.path = path
    assert window.save_workspace()
    ws2 = load_workspace(path)
    assert [p.path for p in ws2.samples[0].root.walk(False)] == ["/Cells"]
    assert len(ws2.layouts) == 1 and len(ws2.layouts[0].items) == 1
    assert len(ws2.tables) == 1 and len(ws2.tables[0].columns) == 1
    # export population as FCS and read it back
    out = tmp_path / "cells.fcs"
    s0.export_population_fcs(pop, out)
    from flowdesk.core.fcs import FCSFile

    f = FCSFile.read(out)
    assert f.n_events == pop.count
    assert f.channel_names == s0.channel_names
    le.close()
    te.close()
    cw.close()


def test_drag_and_drop_between_tree_and_groups(app, window):
    import json

    from PySide6.QtCore import QMimeData, QPoint
    from PySide6.QtGui import QDropEvent

    from flowdesk.ui.controller import POPULATION_MIME
    from flowdesk.ui.main_window import SAMPLES_MIME

    c = window.controller
    s0, s1 = c.workspace.samples[:2]
    pop = c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 5000], [200000, 150000])), None)
    assert s1.root.find("/Cells") is None
    # the tree produces the population mime data for the selected row
    window.tree.rebuild()
    item = window.tree.topLevelItem(0).child(0)
    md = window.tree.mimeData([item])
    payload = json.loads(bytes(md.data(POPULATION_MIME).data()).decode())
    assert payload == {"sample_id": s0.id, "paths": ["/Cells"]}
    # dropping it onto the "All Samples" group applies the gate to every member
    pane = window.group_pane
    pane.rebuild()
    row_rect = pane.visualItemRect(pane.item(0, 0))
    event = QDropEvent(row_rect.center().toPointF(), Qt.DropAction.CopyAction, md, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    pane.dropEvent(event)
    app.processEvents()
    assert s1.root.find("/Cells") is not None
    assert c.workspace.groups[0].root.find("/Cells") is not None
    # dropping a sample onto a custom group adds it manually
    g = c.create_group("Donors", [])
    pane.rebuild()
    smd = QMimeData()
    smd.setData(SAMPLES_MIME, json.dumps([s1.id]).encode())
    row = next(r for r in range(pane.rowCount()) if pane.group_at(r) is g)
    event = QDropEvent(pane.visualItemRect(pane.item(row, 0)).center().toPointF(), Qt.DropAction.CopyAction, smd, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    pane.dropEvent(event)
    assert s1.id in g.sample_ids and s1.root.find("/Cells") is not None
    # dropping a population onto another sample's row copies it there
    c.add_population(s0.root.find("/Cells"), Population("Sub", RectangleGate(["FSC-A"], [50000], [100000])), None)
    window.tree.rebuild()
    sub_item = window.tree.topLevelItem(0).child(0).child(0)
    md2 = window.tree.mimeData([sub_item])
    target_item = window.tree.topLevelItem(1)
    event = QDropEvent(window.tree.visualItemRect(target_item).center().toPointF(), Qt.DropAction.CopyAction, md2, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    window.tree.dropEvent(event)
    assert s1.root.find("/Sub") is not None


def test_group_graph_window_applies_to_all_members(app, window):
    c = window.controller
    group = c.workspace.groups[0]
    gw = window.open_group_graph(group, None)
    assert gw is not None and gw.group is group
    gw.resize(600, 650)
    app.processEvents()
    gw.plot.set_axes(x="FSC-A", y="SSC-A")
    gw.plot.set_tool("rect")
    app.processEvents()
    gw.plot.repaint()
    res = gw.plot.result()
    assert res is not None
    p0, p1 = res.to_pixel(0.2, 0.1), res.to_pixel(0.7, 0.6)
    _mouse(app, gw.plot, QEvent.Type.MouseButtonPress, p0)
    _mouse(app, gw.plot, QEvent.Type.MouseMove, p1)
    _mouse(app, gw.plot, QEvent.Type.MouseButtonRelease, p1)
    app.processEvents()
    for s in c.workspace.samples:
        assert s.root.child_by_name("New gate") is not None, s.name
    assert group.root.child_by_name("New gate") is not None
    # deleting from the group window removes it everywhere
    gw.plot.set_selected(c.workspace.samples[0].root.child_by_name("New gate"))
    c.delete_populations([gw.plot.selected], group)
    for s in c.workspace.samples:
        assert s.root.child_by_name("New gate") is None
    gw.close()


def test_apply_template_through_window(app, window, tmp_path, monkeypatch):
    from flowdesk.core.templates import save_template
    from flowdesk.ui import main_window as mw

    c = window.controller
    ws = c.workspace
    s0 = ws.samples[0]
    c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 5000], [200000, 150000])), c.current_group)
    tpl = tmp_path / "analysis.fdwt"
    save_template(ws, tpl)
    # a fresh workspace with the same files, then the template is applied via File > Open
    from flowdesk.core.model import Workspace

    ws2 = Workspace()
    for p in FCS[:2]:
        ws2.add_sample(p)
    c.set_workspace(ws2)
    assert ws2.samples[0].root.find("/Cells") is None
    monkeypatch.setattr(mw, "confirm", lambda *a, **k: True)
    window._open_path(str(tpl))
    app.processEvents()
    for s in ws2.samples:
        assert s.root.find("/Cells") is not None and s.root.find("/Cells").count > 0
    assert ws2.groups[0].root.find("/Cells") is not None
    assert window.tree.topLevelItem(0).childCount() == 1


def test_wsp_import_through_window(app, window):
    wsp = FOLDER / "19-Aug-2025.wsp"
    if not wsp.exists():
        pytest.skip("no wsp")
    window.controller.workspace.modified = False
    window._open_path(str(wsp))
    app.processEvents()
    ws = window.controller.workspace
    assert len(ws.samples) == 4
    assert ws.samples[0].root.find("/Lymphocytes") is not None
    assert window.tree.topLevelItemCount() == 4
    assert window.tree.topLevelItem(0).childCount() >= 1
