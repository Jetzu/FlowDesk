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

from flowdesk.core.transforms import ArcsinhScale, LinearScale, LogicleScale, LogScale, drag_edge, rescale  # noqa: E402


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


def test_drag_edge_by_position_and_kind():
    lin, lgc = LinearScale(), LogicleScale()
    assert [drag_edge(lin, f) for f in (0.1, 0.5, 0.9)] == ["min", "pan", "max"]
    assert [drag_edge(lgc, f) for f in (0.1, 0.5, 0.9)] == ["min", "max", "max"]


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


def _drag(w, a: QPointF, b: QPointF):
    QTest.mousePress(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a.toPoint())
    QTest.mouseMove(w, ((a + b) / 2).toPoint())
    QTest.mouseMove(w, b.toPoint())
    QTest.mouseRelease(w, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b.toPoint())
    w.grab()


def test_drag_high_end_of_x_axis_shrinks_the_range_and_undoes(widget):
    w = widget
    r = w._result.plot_rect
    before = w.sample.scale("FSC-A").to_dict()
    y = r.bottom() + 12
    grab = QPointF(r.left() + 0.9 * r.width(), y)
    value = w.sample.scale("FSC-A").inverse_scalar(0.9)
    _drag(w, grab, QPointF(r.right() - 2, y))  # pull the value grabbed at 90 % to the end of the axis
    after = w.sample.scale("FSC-A")
    assert after.kind == "linear" and after.min == before["min"] and after.max < before["max"]
    assert after.forward_scalar(value) == pytest.approx((r.right() - 2 - r.left()) / r.width(), abs=0.02)
    assert w.controller.can_undo and w.controller.workspace.modified
    w.controller.undo()
    assert w.sample.scale("FSC-A").to_dict() == before


def test_drag_middle_pans_and_y_axis_low_end(widget):
    w = widget
    r = w._result.plot_rect
    span = w.sample.scale("FSC-A").max - w.sample.scale("FSC-A").min
    _drag(w, QPointF(r.center().x(), r.bottom() + 12), QPointF(r.center().x() + 80, r.bottom() + 12))
    panned = w.sample.scale("FSC-A")
    assert panned.max - panned.min == pytest.approx(span) and panned.min < 0
    ysc = w.sample.scale("SSC-A").to_dict()
    _drag(w, QPointF(r.left() - 20, r.bottom() - 0.1 * r.height()), QPointF(r.left() - 20, r.bottom() - 0.3 * r.height()))
    new = w.sample.scale("SSC-A")
    assert new.max == ysc["max"] and new.min < ysc["min"]


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
