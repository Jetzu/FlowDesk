"""Dragging an axis changes its range (one undo step); the overlay legend can be moved, renamed, resized and titled."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))
import make_testset  # noqa: E402

from flowdesk.core.transforms import ArcsinhScale, LinearScale, LogicleScale, LogScale, rescale  # noqa: E402


# ------------------------------------------------------------------- core math
@pytest.mark.parametrize("scale", [LinearScale(0, 262144), LogScale(1, 262144), LogicleScale(262144, 0.5, 4.5, 0.0), ArcsinhScale(262144, 4.5, 0.5)])
def test_grabbed_value_lands_where_it_is_dropped(scale):
    v = 20000.0
    for f in (0.5, 0.7, 0.9):
        assert rescale(scale, "max", v, f).forward_scalar(v) == pytest.approx(f, abs=0.01)
    if scale.kind in ("linear", "log"):
        assert rescale(scale, "pan", v, 0.3).forward_scalar(v) == pytest.approx(0.3, abs=0.01)


def test_min_edge_keeps_the_top_and_reveals_negatives():
    lin = rescale(LinearScale(0, 262144), "min", 100000, 0.5)
    assert lin.max == 262144 and lin.min == pytest.approx(-62144)
    log = LogicleScale(262144, 0.5, 4.5, 0.0)
    more = rescale(log, "min", -500, 0.1)
    assert more.T == log.T and more.A > log.A and more.forward_scalar(-500) == pytest.approx(0.1, abs=0.01)


def test_dragging_beyond_the_far_end_is_harmless():
    s = LinearScale(0, 1000)
    assert rescale(s, "max", -5, 0.5) == s  # grabbed below the low end: nothing to stretch
    assert rescale(s, "max", 500, 0.0).forward_scalar(500) == pytest.approx(0.02)  # clamped, never a zero-width range


# ------------------------------------------------------------------- widget
@pytest.fixture(scope="module")
def testset(tmp_path_factory):
    from flowdesk.app import create_app

    create_app([])
    out = tmp_path_factory.mktemp("axisdrag")
    make_testset.write_testset(out, seed=7, n=1500)
    return out


@pytest.fixture
def widget(testset):
    from flowdesk.core.serialization import load_workspace
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.plot_widget import PlotWidget

    c = WorkspaceController(load_workspace(testset / "FlowDesk testset.fdw"))
    s = c.workspace.samples[0]
    pop = s.root
    pop.graph.x, pop.graph.y, pop.graph.plot_type = "FSC-A", "SSC-A", "dot"
    w = PlotWidget(c, s, pop)
    w.resize(640, 520)
    w.show()
    w.grab()
    yield w
    w.close()


def _drag(w, a: QPointF, b: QPointF, button=Qt.MouseButton.LeftButton):
    QTest.mousePress(w, button, Qt.KeyboardModifier.NoModifier, a.toPoint())
    QTest.mouseMove(w, ((a + b) / 2).toPoint())
    QTest.mouseMove(w, b.toPoint())
    QTest.mouseRelease(w, button, Qt.KeyboardModifier.NoModifier, b.toPoint())
    w.grab()


def test_left_drag_changes_the_range_keeping_the_low_end_and_undoes(widget):
    w = widget
    r = w._result.plot_rect
    before = w.sample.scale("FSC-A").to_dict()
    y = r.bottom() + 12
    for start in (0.3, 0.8):  # the same behaviour wherever the drag starts
        grab = QPointF(r.left() + start * r.width(), y)
        value = w.sample.scale("FSC-A").inverse_scalar(start)
        drop = QPointF(r.left() + (start + 0.15) * r.width(), y)
        _drag(w, grab, drop)
        after = w.sample.scale("FSC-A")
        assert after.kind == "linear" and after.min == before["min"]
        assert after.forward_scalar(value) == pytest.approx(start + 0.15, abs=0.02)  # the grabbed value followed the mouse
    assert w.sample.scale("FSC-A").max < before["max"]  # dragging towards the top zoomed in
    assert w.controller.can_undo and w.controller.workspace.modified
    w.controller.undo()
    w.controller.undo()
    assert w.sample.scale("FSC-A").to_dict() == before


def test_right_drag_moves_the_whole_range(widget):
    w = widget
    r = w._result.plot_rect
    sc = w.sample.scale("FSC-A")
    span = sc.max - sc.min
    _drag(w, QPointF(r.left() + 0.2 * r.width(), r.bottom() + 12), QPointF(r.left() + 0.4 * r.width(), r.bottom() + 12), Qt.MouseButton.RightButton)
    moved = w.sample.scale("FSC-A")
    assert moved.max - moved.min == pytest.approx(span) and moved.min < 0
    ysc = w.sample.scale("SSC-A")
    yspan = ysc.max - ysc.min
    _drag(w, QPointF(r.left() - 20, r.bottom() - 0.5 * r.height()), QPointF(r.left() - 20, r.bottom() - 0.3 * r.height()), Qt.MouseButton.RightButton)
    ynew = w.sample.scale("SSC-A")
    assert ynew.max - ynew.min == pytest.approx(yspan) and ynew.min > ysc.min  # moved up: the range slides to higher values


def test_right_click_without_drag_still_opens_the_menu(widget):
    got = []
    widget.contextMenuRequested.connect(lambda pop, pos: got.append(pop))
    r = widget._result.plot_rect
    p = QPoint(int(r.center().x()), int(r.bottom() + 12))
    QTest.mousePress(widget, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, p)
    QTest.mouseRelease(widget, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, p)
    assert got == [None]


def test_right_drag_on_a_logicle_axis_shows_more_negatives(widget):
    w = widget
    w.graph.x = "Comp-PE-A" if "Comp-PE-A" in w.sample.parameter_names() else w.sample.parameter_names()[3]
    w.invalidate()
    w.grab()
    sc = w.sample.scale(w.graph.x)
    if sc.kind != "logicle":
        pytest.skip("parameter is not logicle in this test set")
    r = w._result.plot_rect
    _drag(w, QPointF(r.left() + 0.2 * r.width(), r.bottom() + 12), QPointF(r.left() + 0.35 * r.width(), r.bottom() + 12), Qt.MouseButton.RightButton)
    new = w.sample.scale(w.graph.x)
    assert new.T == sc.T and new.A > sc.A


def test_refocus_puts_dragged_axes_back(widget):
    w = widget
    r = w._result.plot_rect
    x0, y0 = w.sample.scale("FSC-A").to_dict(), w.sample.scale("SSC-A").to_dict()
    assert not w.axes_changed()
    _drag(w, QPointF(r.left() + 0.5 * r.width(), r.bottom() + 12), QPointF(r.left() + 0.8 * r.width(), r.bottom() + 12))
    _drag(w, QPointF(r.left() + 0.5 * r.width(), r.bottom() + 12), QPointF(r.left() + 0.9 * r.width(), r.bottom() + 12))  # a second drag
    _drag(w, QPointF(r.left() - 20, r.center().y()), QPointF(r.left() - 20, r.top() + 20), Qt.MouseButton.RightButton)
    w.view = (0.1, 0.6, 0.1, 0.6)
    assert w.axes_changed() and w.sample.scale("FSC-A").to_dict() != x0
    w.refocus()
    assert w.sample.scale("FSC-A").to_dict() == x0 and w.sample.scale("SSC-A").to_dict() == y0  # back past every drag
    assert w.view == (0.0, 1.0, 0.0, 1.0) and not w.axes_changed() and "axis_home" not in w.graph.options
    w.controller.undo()
    assert w.sample.scale("FSC-A").to_dict() != x0  # refocus itself is one undo step


def test_transform_dialog_choice_becomes_the_new_home(widget):
    w = widget
    r = w._result.plot_rect
    _drag(w, QPointF(r.left() + 0.5 * r.width(), r.bottom() + 12), QPointF(r.left() + 0.8 * r.width(), r.bottom() + 12))
    assert w.axes_changed()
    w.forget_axis_home(["FSC-A"])
    assert not w.axes_changed()


def test_graph_window_refocus_button(testset):
    from flowdesk.core.serialization import load_workspace
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.graph_window import GraphWindow

    c = WorkspaceController(load_workspace(testset / "FlowDesk testset.fdw"))
    s = c.workspace.samples[0]
    pop = s.root
    pop.graph.x, pop.graph.y = "FSC-A", "SSC-A"
    gw = GraphWindow(c, s, pop)
    gw.resize(700, 600)
    gw.show()
    gw.plot.grab()
    assert not gw.refocus_button.isEnabled()
    r = gw.plot._result.plot_rect
    _drag(gw.plot, QPointF(r.left() + 0.5 * r.width(), r.bottom() + 12), QPointF(r.left() + 0.8 * r.width(), r.bottom() + 12))
    assert gw.refocus_button.isEnabled()
    gw.refocus_button.click()
    assert not gw.refocus_button.isEnabled()
    c.workspace.modified = False
    gw.close()


def test_axis_strip_hit_rules(widget):
    w = widget
    r = w._result.plot_rect
    assert w._axis_hit(QPointF(r.center().x(), r.bottom() + 10)) == "x"
    assert w._axis_hit(QPointF(r.left() - 10, r.center().y())) == "y"
    assert w._axis_hit(r.center()) is None
    w.set_axes(y=None) if hasattr(w, "set_axes") else None
    w.graph.y = None
    w.invalidate()
    w.grab()
    if w._result.histogram:
        assert w._axis_hit(QPointF(w._result.plot_rect.left() - 10, w._result.plot_rect.center().y())) is None


def test_double_click_on_axis_requests_the_transform_dialog(widget):
    got = []
    widget.axisEditRequested.connect(got.append)
    r = widget._result.plot_rect
    QTest.mouseDClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(int(r.center().x()), int(r.bottom() + 12)))
    assert got == ["x"]


def test_legend_moves_renames_resizes_and_hides_samples(widget, monkeypatch):
    w = widget
    other = w.controller.workspace.samples[1]
    w.overlays = [(other, other.root)]
    w.invalidate()
    w.grab()
    res = w._result
    assert res.legend_rect is not None and len(res.legend_rows) == 2
    box = res.legend_rect
    target = QPointF(res.plot_rect.left() + 40, res.plot_rect.bottom() - 60)
    _drag(w, box.center(), target)
    moved = w._result.legend_rect
    assert w.graph.options.get("legend_pos") and moved.center().x() < box.center().x() - 50
    assert w._result.plot_rect.contains(moved)
    w.controller.undo()
    w.grab()
    assert "legend_pos" not in w.graph.options

    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Unstimulated", True)))
    w._rename_legend_entry(1)
    w.grab()
    assert w._result.legend_entries[1][0] == "Unstimulated"

    small = w._result.legend_rect
    w._set_legend_option("legend_font_size", 16.0)
    w.grab()
    assert w._result.legend_rect.height() > small.height()

    w._set_legend_option("legend_show_samples", False)
    w._set_legend_option("legend_title", "Donors")
    w.grab()
    first = w._result.legend_entries[0][0]
    assert w.sample.name not in first and first == w.population.display_name
    assert len(w._result.legend_rows) == 2 and w._result.legend_rows[0].top() > w._result.legend_rect.top() + 8


def test_legend_settings_reach_the_render_options(testset):
    from flowdesk.core.model import GraphSettings
    from flowdesk.ui.renderer import RenderOptions

    g = GraphSettings("FSC-A", "SSC-A", "dot", {"legend_pos": [0.1, 0.7], "legend_font_size": 12, "legend_title": "T", "legend_labels": {"a": "b"},
                                                "legend_show_samples": False})
    o = RenderOptions.from_prefs({}, g)
    assert o.legend_pos == [0.1, 0.7] and o.legend_font_size == 12.0 and o.legend_title == "T" and o.legend_labels == {"a": "b"}
    assert o.legend_show_samples is False
    assert RenderOptions.from_prefs({"legend_title": "global"}).legend_title == ""  # per graph only, never a global preference


def test_plot_menu_opens_on_release_and_not_after_a_drag(widget, monkeypatch):
    w = widget
    got = []
    w.contextMenuRequested.connect(lambda pop, pos: got.append(pop))
    c = w._result.plot_rect.center().toPoint()
    QTest.mousePress(w, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, c)
    assert got == []  # nothing on press
    QTest.mouseRelease(w, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, c)
    assert got == [None]
    QTest.mousePress(w, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, c)
    QTest.mouseMove(w, c + QPoint(40, 30))
    QTest.mouseRelease(w, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, c + QPoint(40, 30))
    assert got == [None]  # dragged: no menu
    other = w.controller.workspace.samples[1]
    w.overlays = [(other, other.root)]
    w.invalidate()
    w.grab()
    legend = []
    monkeypatch.setattr(w, "_legend_menu", lambda row, pos: legend.append(row))
    lp = w._result.legend_rows[1].center().toPoint()
    QTest.mousePress(w, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, lp)
    assert legend == []
    QTest.mouseRelease(w, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, lp)
    assert legend == [1]
