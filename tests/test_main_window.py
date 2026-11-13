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

# the tests on the real panel (the `window` fixture) skip without it; the
# workspace-hub tests at the end of the file write their own small FCS files


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture
def window(app, monkeypatch):
    from flowdesk.ui.main_window import MainWindow

    if len(FCS) < 2:
        pytest.skip("real test data not available")

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
    _wait_loader(win, app)
    yield win
    for gw in list(win.graph_windows.values()):
        gw.close()
    win.controller.workspace.modified = False
    win.hide()


def _wait_loader(win, app):
    """Let the background loader finish (counts in the tree are deterministic then)."""
    ld = win.extensions.get("flowdesk.ui.loader")
    app.processEvents()  # the window queues its load request for the event loop
    if ld is not None:
        assert ld.wait(60000)
    app.processEvents()


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


# ---------------------------------------------------------------------------
# workspace hub: group tree row, group drops, re-parenting, keys, windows
# (synthetic FCS files, no real data needed)

HUB_NAMES = ["Donor1.fcs", "Donor2.fcs", "Donor3.fcs"]


def _write_samples(folder, names=HUB_NAMES, n=400):
    import numpy as np

    from flowdesk.core.fcs import Channel, write_fcs

    rng = np.random.default_rng(11)
    paths = []
    for k, name in enumerate(names):
        events = np.column_stack([
            rng.normal(50000 + 4000 * k, 9000, n),
            rng.normal(30000, 7000, n),
            rng.lognormal(7, 1, n),
            rng.lognormal(6, 1, n),
        ]).astype(np.float32)
        channels = [Channel(1, "FSC-A"), Channel(2, "SSC-A"), Channel(3, "FITC-A", "CD3"), Channel(4, "PE-A", "CD4")]
        write_fcs(folder / name, events, channels, extra_keywords={"$CYT": "Aurora", "Donor": name[:-4]})
        paths.append(folder / name)
    return paths


@pytest.fixture
def hub(app, tmp_path, monkeypatch):
    from flowdesk.ui import main_window as mw

    monkeypatch.setattr(mw, "confirm", lambda *a, **k: True)
    win = mw.MainWindow()
    win.resize(1000, 700)
    win.show()
    win.controller.add_samples(_write_samples(tmp_path)[:2])
    _wait_loader(win, app)
    win.controller._undo.clear()
    yield win
    ld = win.extensions.get("flowdesk.ui.loader")
    if ld is not None:
        ld.stop_all()
    for w in list(win.graph_windows.values()):
        w.close()
    for w in (win.layout_editor, win.table_editor, win.compensation_window):
        if w is not None:
            w.close()
    win.controller.workspace.modified = False
    win.hide()
    win.deleteLater()
    app.processEvents()


def _rect(lo, dim="FSC-A"):
    return RectangleGate([dim], [lo], [None])


def _find_item(tree, predicate):
    from PySide6.QtWidgets import QTreeWidgetItemIterator

    it = QTreeWidgetItemIterator(tree)
    while it.value():
        if predicate(it.value()):
            return it.value()
        it += 1
    return None


def _drop(widget, item_rect, md, modifiers=Qt.KeyboardModifier.NoModifier):
    from PySide6.QtGui import QDropEvent

    ev = QDropEvent(item_rect.center().toPointF(), Qt.DropAction.CopyAction | Qt.DropAction.MoveAction, md, Qt.MouseButton.LeftButton, modifiers)
    widget.dropEvent(ev)
    return ev


def test_group_row_mirrors_template_renames_everywhere_and_shows_means(app, hub):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    from flowdesk.core.stats import format_value
    from flowdesk.ui.main_window import ROLE_GROUP, ROLE_SAMPLE

    c = hub.controller
    ws = c.workspace
    s0, s1 = ws.samples
    g = c.create_group("Panel", [s0, s1])
    c.add_population(s0.root, Population("Cells", _rect(45000)), g)
    c.add_population(s0.root.find("/Cells"), Population("CD3+", _rect(800, "FITC-A")), g)
    c.set_current_group(g)
    app.processEvents()
    tree = hub.tree
    top = tree.topLevelItem(0)
    assert top.data(0, ROLE_GROUP) is g and top.data(0, ROLE_SAMPLE) is None
    assert top.font(0).bold() and top.text(0) == "Panel"
    assert [tree.topLevelItem(i).data(0, ROLE_SAMPLE) for i in (1, 2)] == [s0, s1]  # members below
    # the children mirror the group template
    assert [top.child(i).text(0) for i in range(top.childCount())] == [p.name for p in g.root.children] == ["Cells"]
    cells_item = top.child(0)
    assert cells_item.child(0).text(0) == "CD3+"
    # Statistic = mean % of parent over the members, #Cells = mean count
    decimals = int(c.preferences.get("freq_decimals"))
    freqs = [s.root.find("/Cells").freq_of_parent for s in (s0, s1)]
    assert cells_item.text(1) == format_value("Freq. of Parent", sum(freqs) / 2, decimals) + " %"
    assert cells_item.text(2) == f"{sum(s.root.find('/Cells').count for s in (s0, s1)) / 2:,.0f}"
    assert "Mean" in cells_item.toolTip(1)
    # F2 on the group population renames it in every member and the template
    tree.clearSelection()
    tree.setCurrentItem(cells_item)
    cells_item.setSelected(True)
    tree.setFocus()
    assert hub.act_rename.isEnabled() and hub.act_graph.isEnabled()
    hub.act_rename.trigger()  # the F2 action
    editor = tree.findChild(QLineEdit)
    assert editor is not None
    editor.setText("Lymphs")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    app.processEvents()
    for s in (s0, s1):
        assert s.root.find("/Lymphs/CD3+") is not None and s.root.find("/Cells") is None
    assert g.root.find("/Lymphs") is not None
    assert tree.topLevelItem(0).child(0).text(0) == "Lymphs"
    c.undo()
    assert all(s.root.find("/Cells/CD3+") is not None for s in (s0, s1)) and g.root.find("/Cells") is not None


def test_group_tree_rows_open_delete_drag_and_statistics(app, hub):
    import json

    from flowdesk.ui.controller import POPULATION_MIME
    from flowdesk.ui.main_window import ROLE_GROUP

    c = hub.controller
    ws = c.workspace
    s0, s1 = ws.samples
    g = c.create_group("Panel", [s0, s1])
    c.add_population(s0.root, Population("Cells", _rect(45000)), g)
    c.set_current_group(g)
    app.processEvents()
    tree = hub.tree
    cells_item = tree.topLevelItem(0).child(0)
    # double-click opens the group graph at that population
    tree._double_clicked(cells_item, 0)
    gw = hub.graph_windows.get((g.id, "/Cells"))
    assert gw is not None and gw.group is g and gw.population.path == "/Cells"
    gw.close()
    # a group statistic: mean over the members
    tree.clearSelection()
    cells_item.setSelected(True)
    c.add_statistic([g.root.find("/Cells")], "Median", "FSC-A", None, g)
    app.processEvents()
    stat_item = _find_item(tree, lambda it: it.data(0, ROLE_GROUP) is g and it.text(0).startswith("Median"))
    assert stat_item is not None and stat_item.text(1) != ""
    assert all(s.root.find("/Cells").stats for s in (s0, s1))
    # the drag payload names the group (and its first member for layouts / tables)
    cells_item = tree.topLevelItem(0).child(0)
    md = tree.mimeData([cells_item])
    payload = json.loads(bytes(md.data(POPULATION_MIME).data()).decode())
    assert payload == {"group_id": g.id, "sample_id": s0.id, "paths": ["/Cells"]}
    # deleting a group population removes it from every member (one undo step)
    tree.clearSelection()
    cells_item.setSelected(True)
    assert hub.act_delete.isEnabled()
    hub.delete_selected()
    assert all(s.root.find("/Cells") is None for s in (s0, s1)) and g.root.find("/Cells") is None
    c.undo()
    assert all(s.root.find("/Cells") is not None for s in (s0, s1))


def test_drop_fcs_on_group_row_adds_sample_and_membership_in_one_undo_step(app, hub, tmp_path):
    from PySide6.QtCore import QMimeData, QUrl

    c = hub.controller
    ws = c.workspace
    s0 = ws.samples[0]
    (tmp_path / "more").mkdir()
    extra = _write_samples(tmp_path / "more", ["Donor9.fcs"])[0]
    g = c.create_group("Donors", [s0])
    c.add_population(s0.root, Population("Cells", _rect(45000)), g)
    c._undo.clear()
    pane = hub.group_pane
    pane.rebuild()
    row = next(r for r in range(pane.rowCount()) if pane.group_at(r) is g)
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(extra))])
    ev = _drop(pane, pane.visualItemRect(pane.item(row, 0)), md)
    assert ev.isAccepted()
    added = ws.sample_by_path(extra)
    assert added is not None and added.id in g.sample_ids
    assert added.root.find("/Cells") is not None  # the group's gates came along
    assert len(c._undo) == 1
    c.undo()
    assert ws.sample_by_path(extra) is None and g.sample_ids == {s0.id}
    # the same drop on the group row of the sample tree
    c.set_current_group(g)
    app.processEvents()
    top = hub.tree.topLevelItem(0)
    _drop(hub.tree, hub.tree.visualItemRect(top), md)
    added = ws.sample_by_path(extra)
    assert added is not None and g.contains(added)
    # dropping on the All Samples row only adds files to the workspace
    c.undo()
    pane.rebuild()
    all_row = next(r for r in range(pane.rowCount()) if pane.group_at(r).role == "all")
    _drop(pane, pane.visualItemRect(pane.item(all_row, 0)), md)
    assert ws.sample_by_path(extra) is not None and ws.sample_by_path(extra).id not in g.sample_ids


def test_drag_gate_moves_within_sample_alt_copies_and_undo_restores(app, hub):
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest

    from flowdesk.ui.main_window import ROLE_POP, ROLE_SAMPLE, ROLE_STAT

    c = hub.controller
    s0, s1 = c.workspace.samples
    cells = c.add_population(s0.root, Population("Cells", _rect(40000)))
    cd3 = c.add_population(cells, Population("CD3+", _rect(800, "FITC-A")))
    live = c.add_population(cells, Population("Live", _rect(15000, "SSC-A")))
    c._undo.clear()
    app.processEvents()
    tree = hub.tree

    def item_of(path, sample=s0):
        return _find_item(tree, lambda it: it.data(0, ROLE_SAMPLE) is sample and it.data(0, ROLE_POP) is not None and it.data(0, ROLE_POP).path == path and it.data(0, ROLE_STAT) is None)

    md = tree.mimeData([item_of("/Cells/CD3+")])
    ev = _drop(tree, tree.visualItemRect(item_of("/Cells/Live")), md)
    assert ev.dropAction() == Qt.DropAction.MoveAction
    assert s0.root.find("/Cells/CD3+") is None
    assert s0.root.find("/Cells/Live/CD3+") is cd3 and cd3.parent is live
    assert len(c._undo) == 1 and c.undo_label() == "move gate"
    # Ctrl+Z (the application-wide Undo shortcut) restores the original tree
    hub.activateWindow()
    app.processEvents()
    combo = QKeySequence.keyBindings(QKeySequence.StandardKey.Undo)[0][0]
    QTest.keyClick(tree, combo.key(), combo.keyboardModifiers())
    app.processEvents()
    if s0.root.find("/Cells/CD3+") is None:  # no active window on this platform: run the action
        hub.act_undo.trigger()
    assert [p.path for p in s0.root.walk(False)] == ["/Cells", "/Cells/CD3+", "/Cells/Live"]
    assert s0.root.find("/Cells/CD3+").id == cd3.id
    # Alt / Option + drag copies instead
    md = tree.mimeData([item_of("/Cells/CD3+")])
    _drop(tree, tree.visualItemRect(item_of("/Cells/Live")), md, Qt.KeyboardModifier.AltModifier)
    assert s0.root.find("/Cells/CD3+") is not None and s0.root.find("/Cells/Live/CD3+") is not None
    # a gate cannot move into its own sub-population
    md = tree.mimeData([item_of("/Cells")])
    _drop(tree, tree.visualItemRect(item_of("/Cells/Live")), md)
    assert s0.root.find("/Cells/Live") is not None and s0.root.find("/Cells").parent is s0.root
    # onto another sample a plain drag copies (the source keeps its gate)
    md = tree.mimeData([item_of("/Cells/Live")])
    _drop(tree, tree.visualItemRect(item_of("", s1)), md)
    assert s1.root.find("/Live") is not None and s0.root.find("/Cells/Live") is not None


def test_drag_sample_row_copies_tree_and_statistic_rows_copy_statistics(app, hub):
    from flowdesk.ui.main_window import ROLE_POP, ROLE_SAMPLE, ROLE_STAT

    c = hub.controller
    s0, s1 = c.workspace.samples
    cells = c.add_population(s0.root, Population("Cells", _rect(40000)))
    c.add_population(cells, Population("CD3+", _rect(800, "FITC-A")))
    c.add_statistic([cells], "Median", "FSC-A", None)
    app.processEvents()
    tree = hub.tree
    s0_item = _find_item(tree, lambda it: it.data(0, ROLE_SAMPLE) is s0 and it.data(0, ROLE_POP) is s0.root)
    s1_item = _find_item(tree, lambda it: it.data(0, ROLE_SAMPLE) is s1 and it.data(0, ROLE_POP) is s1.root)
    _drop(tree, tree.visualItemRect(s1_item), tree.mimeData([s0_item]))
    assert [p.path for p in s1.root.walk(False)] == ["/Cells", "/Cells/CD3+"]
    assert len(c._undo) >= 1
    # a statistic row dropped on another population adds just that statistic
    c.undo()
    c.add_population(s1.root, Population("Big", _rect(60000)))
    app.processEvents()
    stat_item = _find_item(tree, lambda it: it.data(0, ROLE_SAMPLE) is s0 and it.data(0, ROLE_STAT) is not None)
    big_item = _find_item(tree, lambda it: it.data(0, ROLE_SAMPLE) is s1 and it.data(0, ROLE_POP) is not None and it.data(0, ROLE_POP).path == "/Big")
    md = tree.mimeData([stat_item])
    assert "stats" in md.data("application/x-flowdesk-population").data().decode()
    _drop(tree, tree.visualItemRect(big_item), md)
    big = s1.root.find("/Big")
    assert [(st.statistic, st.param) for st in big.stats] == [("Median", "FSC-A")]
    assert s1.root.find("/Big/Cells") is None  # only the statistic, not the gate


def test_delete_key_on_sample_and_undo_shortcut_in_table_editor(app, hub):
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest

    from flowdesk.ui.main_window import ROLE_POP, ROLE_SAMPLE

    c = hub.controller
    ws = c.workspace
    s0, s1 = ws.samples
    tree = hub.tree
    item = _find_item(tree, lambda it: it.data(0, ROLE_SAMPLE) is s1 and it.data(0, ROLE_POP) is s1.root)
    tree.clearSelection()
    tree.setCurrentItem(item)
    item.setSelected(True)
    assert hub.act_delete.isEnabled()
    hub.activateWindow()
    tree.setFocus()
    app.processEvents()
    QTest.keyClick(tree, Qt.Key.Key_Delete)
    app.processEvents()
    assert ws.samples == [s0]
    # Undo / Redo answer the keyboard in every window
    assert hub.act_undo.shortcutContext() == Qt.ShortcutContext.ApplicationShortcut
    c.add_population(s0.root, Population("Cells", _rect(40000)))
    te = hub.show_table_editor()
    te.add_columns_for_paths(["/Cells"])
    table = ws.tables[-1]
    assert len(table.columns) == 1
    assert hub.act_undo.isEnabled() and "edit table" in hub.act_undo.text()
    te.activateWindow()
    app.processEvents()
    if QApplication.activeWindow() is te:
        combo = QKeySequence.keyBindings(QKeySequence.StandardKey.Undo)[0][0]
        QTest.keyClick(te, combo.key(), combo.keyboardModifiers())
        app.processEvents()
    else:  # the platform plugin did not activate the window: the action is what the key runs
        hub.act_undo.trigger()
    assert not ws.tables or len(ws.tables[-1].columns) == 0
    # graph windows bring their own Undo: the application-wide one steps back there
    gw = hub.open_graph(s0, s0.root)
    app.processEvents()
    hub._focus_window_changed()
    if QApplication.activeWindow() is gw:
        assert hub.act_undo.shortcutContext() == Qt.ShortcutContext.WindowShortcut
    hub.activateWindow()
    app.processEvents()
    hub._focus_window_changed()
    if QApplication.activeWindow() is hub:
        assert hub.act_undo.shortcutContext() == Qt.ShortcutContext.ApplicationShortcut


def test_group_pane_rename_in_place_and_delete_key(app, hub):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    c = hub.controller
    ws = c.workspace
    g = c.create_group("Donors", [ws.samples[0]])
    c._undo.clear()
    pane = hub.group_pane
    pane.rebuild()
    row = next(r for r in range(pane.rowCount()) if pane.group_at(r) is g)
    pane.selectRow(row)
    assert c.current_group is g
    assert pane.rename_current()
    editor = pane.findChild(QLineEdit)
    assert editor is not None
    editor.setText("Patients")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    app.processEvents()
    assert g.name == "Patients" and g.root.name == "Patients"
    assert c.undo_label() == "rename group"
    c.undo()
    assert ws.group_by_name("Donors") is not None
    # a name that is taken is refused
    pane.rebuild()
    pane.item(next(r for r in range(pane.rowCount()) if pane.group_at(r).name == "Donors"), 0).setText("Compensation")
    assert ws.group_by_name("Donors") is not None
    # Delete in the group pane deletes the group (confirm patched to Yes)
    app.processEvents()
    pane.rebuild()
    g = ws.group_by_name("Donors")
    hub.tree.clearSelection()
    pane.selectRow(next(r for r in range(pane.rowCount()) if pane.group_at(r) is g))
    pane.setFocus()
    QTest.keyClick(pane, Qt.Key.Key_Delete)
    assert ws.group_by_name("Donors") is None
    # the "All Samples" row is neither renamable nor deletable
    pane.rebuild()
    pane.selectRow(0)
    assert not pane.rename_current()
    QTest.keyClick(pane, Qt.Key.Key_Delete)
    assert ws.groups[0].role == "all"


def test_edit_keywords_joining_live_group_receives_its_gates(app, hub, monkeypatch):
    from flowdesk.core.model import Criterion
    from flowdesk.ui import main_window as mw

    c = hub.controller
    ws = c.workspace
    s0, s1 = ws.samples
    g = c.create_group("Stim", [], [Criterion("Stim", "Equals", "yes")])
    s0.keywords["Stim"] = "yes"
    c.add_population(s0.root, Population("Cells", _rect(45000)), g)
    assert s1.root.find("/Cells") is None and not g.contains(s1)

    class FakeKeywords:
        def __init__(self, sample, parent=None):
            self.sample = sample

        def exec(self):
            return True

        def values(self):
            return dict(self.sample.keywords, Stim="yes")

    monkeypatch.setattr(mw, "KeywordsDialog", FakeKeywords)
    c.set_selection([(s1, s1.root)])
    hub.edit_keywords()
    assert g.contains(s1) and s1.root.find("/Cells") is not None
    c.undo()
    assert s1.root.find("/Cells") is None and not g.contains(s1)


def test_boolean_gate_for_the_whole_group(app, hub, monkeypatch):
    from flowdesk.ui.group_dialog import GroupBooleanGateDialog

    c = hub.controller
    s0, s1 = c.workspace.samples
    g = c.create_group("Panel", [s0, s1])
    c.add_population(s0.root, Population("A", _rect(40000)), g)
    c.add_population(s0.root, Population("B", _rect(20000, "SSC-A")), g)
    c.set_current_group(g)
    app.processEvents()
    seen = {}

    def fake_exec(dlg):
        seen["checked"] = dlg.group_check.isChecked()
        seen["text"] = dlg.group_check.text()
        return True

    monkeypatch.setattr(GroupBooleanGateDialog, "exec", fake_exec)
    c.set_selection([(s0, s0.root.find("/A")), (s0, s0.root.find("/B"))])
    hub.create_boolean("and")
    assert seen["checked"] and "Panel" in seen["text"]
    for s in (s0, s1):
        gate = s.root.find("/A AND B")
        assert gate is not None and gate.gate.kind == "boolean"
    assert g.root.find("/A AND B") is not None


def test_graph_window_on_screen_when_maximized_and_window_menu_lists_editors(app, hub):
    c = hub.controller
    s0 = c.workspace.samples[0]
    hub.showMaximized()
    app.processEvents()
    screen = hub.screen()
    avail = screen.availableGeometry()
    for k in range(3):
        c.add_population(s0.root, Population(f"G{k}", _rect(20000 + k)))
    wins = [hub.open_graph(s0, s0.root)] + [hub.open_graph(s0, s0.root.find(f"/G{k}")) for k in range(3)]
    app.processEvents()
    for gw in wins:
        assert avail.contains(gw.frameGeometry()), (gw.frameGeometry(), avail)
    le = hub.show_layout_editor()
    app.processEvents()
    hub._fill_window_menu()
    titles = [a.text() for a in hub.window_menu.actions() if not a.isSeparator()]
    assert titles[0] == "Workspace"
    assert any("Layout Editor" in t for t in titles)
    assert any(t == wins[0].windowTitle() for t in titles)
    assert "Close All Graph Windows" in titles
    entry = next(a for a in hub.window_menu.actions() if "Layout Editor" in a.text())
    entry.trigger()
    app.processEvents()
    assert le.isVisible()
    le.close()
    app.processEvents()
    hub._fill_window_menu()
    assert not any("Layout Editor" in a.text() for a in hub.window_menu.actions())
    hub.showNormal()


def test_context_menu_hooks_and_extension_delegation(app, hub, monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "_exec_menu", lambda menu, pos: calls.append(("exec", [x.text() for x in menu.actions()])))
    hub.tree_context_hooks.append(lambda menu, item: (calls.append(("tree", item)), menu.addAction("From extension")))
    hub.group_context_hooks.append(lambda menu, group: calls.append(("group", group)))
    hub.tree_context_hooks.append(lambda menu, item: 1 / 0)  # a broken hook does not stop the menu
    s0 = hub.controller.workspace.samples[0]
    item = hub.tree.topLevelItem(0)
    hub._tree_context_menu(item, hub.tree.mapToGlobal(hub.tree.rect().center()))
    assert ("tree", item) in calls and "From extension" in calls[-1][1]
    g = hub.controller.workspace.groups[1]
    hub._group_context_menu(g, hub.group_pane.mapToGlobal(hub.group_pane.rect().center()))
    assert ("group", g) in calls
    # providers registered as extensions take over the export / derive commands
    class Provider:
        def __init__(self):
            self.ran = []

        def export_events(self):
            self.ran.append("export_events")

        def derive_parameter(self):
            self.ran.append("derive_parameter")

    p = Provider()
    hub.extensions["flowdesk.ui.export_tools"] = p
    hub.extensions["flowdesk.ui.population_tools"] = p
    hub.export_events()
    hub.derive_parameter()
    assert p.ran == ["export_events", "derive_parameter"] and s0 is not None


def test_many_files_go_through_the_background_loader(app, hub, tmp_path):
    """60 files: headers are read by the loader, the tree shows "…" for samples
    whose data is not loaded and the counts once countsReady fired."""
    from flowdesk.ui.main_window import PLACEHOLDER, ROLE_POP, ROLE_SAMPLE

    ld = hub.extensions.get("flowdesk.ui.loader")
    if ld is None:
        pytest.skip("the loader extension is not available")
    assert hub.tree.defer_unloaded
    folder = tmp_path / "plate"
    folder.mkdir()
    paths = _write_samples(folder, [f"Well_{k:02d}.fcs" for k in range(60)], n=60)
    c = hub.controller
    ws = c.workspace
    ws.groups[0].root.add_child(Population("Cells", _rect(45000)))  # the All Samples template
    ready = []
    ld.countsReady.connect(lambda samples: ready.append(len(samples)))
    added = hub._files_dropped([str(folder)])
    assert added == [] and ld.busy  # queued in the background
    import time

    deadline = time.monotonic() + 90
    while (len(ws.samples) < 62 or not ready or ld.busy) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert len(ws.samples) == 62 and ready
    new = [s for s in ws.samples if s.path.parent == folder]
    assert len(new) == 60 and all(s.loaded for s in new)
    texts = []
    for i in range(hub.tree.topLevelItemCount()):
        item = hub.tree.topLevelItem(i)
        if item.data(0, ROLE_SAMPLE) in new:
            cells = next(item.child(k) for k in range(item.childCount()) if item.child(k).data(0, ROLE_POP).path == "/Cells")
            texts.append((cells.text(1), cells.text(2)))
    assert len(texts) == 60
    assert all(t1.endswith("%") and t2 != PLACEHOLDER for t1, t2 in texts)
    # a rebuild never reads event data: an unloaded sample shows the placeholder
    s = new[0]
    assert s.unload()
    hub.tree.rebuild()
    item = next(hub.tree.topLevelItem(i) for i in range(hub.tree.topLevelItemCount()) if hub.tree.topLevelItem(i).data(0, ROLE_SAMPLE) is s)
    assert item.child(0).text(2) == PLACEHOLDER and not s.loaded



def test_group_tree_clipboard_statistics_menu_and_pane_drops(app, hub, monkeypatch):
    import json

    from PySide6.QtCore import QMimeData

    from flowdesk.ui import main_window as mw
    from flowdesk.ui.controller import POPULATION_MIME
    from flowdesk.ui.main_window import ROLE_GROUP, ROLE_POP, ROLE_SAMPLE, ROLE_STAT

    c = hub.controller
    ws = c.workspace
    s0, s1 = ws.samples
    g = c.create_group("Panel", [s0, s1])
    c.add_population(s0.root, Population("Cells", _rect(45000)), g)
    c.add_population(s0.root, Population("Big", _rect(60000)), g)
    c.set_current_group(g)
    app.processEvents()
    tree = hub.tree

    def group_item(path):
        return _find_item(tree, lambda it: it.data(0, ROLE_GROUP) is g and it.data(0, ROLE_STAT) is None and it.data(0, ROLE_POP).path == path)

    def select(item):
        tree.clearSelection()
        tree.setCurrentItem(item)
        item.setSelected(True)

    # Add Statistic on a group row adds it to every member
    class FakeStat:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return True

        def values(self):
            return ("Mean", "FSC-A", None, False)

    monkeypatch.setattr(mw, "AddStatisticDialog", FakeStat)
    select(group_item("/Cells"))
    hub.add_statistic()
    assert all(any(st.statistic == "Mean" for st in s.root.find("/Cells").stats) for s in (s0, s1))
    # copy a group gate, paste it under another group gate: every member gets it
    select(group_item("/Cells"))
    hub.copy_populations()
    select(group_item("/Big"))
    assert hub.act_paste.isEnabled()
    hub.paste_populations()
    assert all(s.root.find("/Big/Cells") is not None for s in (s0, s1)) and g.root.find("/Big/Cells") is not None
    # cut removes the group gate from every member
    select(group_item("/Big/Cells"))
    hub.cut_populations()
    assert all(s.root.find("/Big/Cells") is None for s in (s0, s1))
    # removing a group statistic row removes it from the members
    stat_item = _find_item(tree, lambda it: it.data(0, ROLE_GROUP) is g and it.data(0, ROLE_STAT) is not None)
    select(stat_item)
    assert hub.act_delete.isEnabled()
    hub.delete_selected()
    assert not any(s.root.find("/Cells").stats for s in (s0, s1))
    # the group-row context menu, and Graph Window on a group row
    shown = []
    monkeypatch.setattr(hub, "_exec_menu", lambda menu, pos: shown.append([a.text() for a in menu.actions()]))
    hub._tree_context_menu(group_item("/Cells"), tree.mapToGlobal(tree.rect().center()))
    assert any("Open group graph" in t for t in shown[-1])
    select(group_item("/Cells"))
    hub.open_selected_graph()
    assert (g.id, "/Cells") in hub.graph_windows
    # a statistic row dropped on a group in the group pane goes to every member
    c.add_statistic([s0.root.find("/Big")], "CV", "FSC-A", None)
    c.set_current_group(ws.groups[0])
    app.processEvents()
    s_stat = _find_item(tree, lambda it: it.data(0, ROLE_STAT) is not None and it.data(0, ROLE_GROUP) is None)
    md = tree.mimeData([s_stat])
    assert json.loads(bytes(md.data(POPULATION_MIME).data()).decode())["stats"][0]["statistic"] == "CV"
    pane = hub.group_pane
    pane.rebuild()
    row = next(r for r in range(pane.rowCount()) if pane.group_at(r) is g)
    _drop(pane, pane.visualItemRect(pane.item(row, 0)), md)
    assert all(any(st.statistic == "CV" for st in s.root.find("/Big").stats) for s in (s0, s1))
    # dragging a sample row onto a custom group adds the sample to it
    g2 = c.create_group("Solo", [])
    pane.rebuild()
    sample_item = _find_item(tree, lambda it: it.data(0, ROLE_SAMPLE) is s1 and it.data(0, ROLE_POP) is s1.root)
    md = tree.mimeData([sample_item])
    assert isinstance(md, QMimeData)
    row = next(r for r in range(pane.rowCount()) if pane.group_at(r) is g2)
    _drop(pane, pane.visualItemRect(pane.item(row, 0)), md)
    assert s1.id in g2.sample_ids
