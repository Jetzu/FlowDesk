"""FlowJo-compatible (bitmap) polygon evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flowdesk.core import gates
from flowdesk.core.gates import POLYGON_SETTINGS, PolygonGate, polygon_bitmap
from flowdesk.core.transforms import LinearScale

FOLDER = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel")


@pytest.fixture
def exact_mode():
    old = dict(POLYGON_SETTINGS)
    POLYGON_SETTINGS["mode"] = "exact"
    yield
    POLYGON_SETTINGS.clear()
    POLYGON_SETTINGS.update(old)


class _Acc:
    def __init__(self, x, y):
        self._d = {"x": x, "y": y}
        self.scale = LinearScale(0.0, 1.0)
        self.n_events = len(x)

    def data(self, name):
        return self._d[name]

    def display(self, name):
        return self._d[name]

    def to_display(self, name, values):
        return np.asarray(values, dtype=float)

    def resolve(self, path):
        return None


def test_bitmap_matches_exact_away_from_edges(exact_mode):
    rng = np.random.default_rng(0)
    x = rng.random(20000)
    y = rng.random(20000)
    acc = _Acc(x, y)
    gate = PolygonGate(["x", "y"], [(0.2, 0.2), (0.8, 0.25), (0.7, 0.8), (0.3, 0.7)])
    exact = gate.evaluate(acc)
    POLYGON_SETTINGS["mode"] = "bitmap"
    bitmap = gate.evaluate(acc)
    # the two agree except within one bin of the outline
    disagree = exact != bitmap
    assert disagree.mean() < 0.02
    mask = polygon_bitmap(gate.vertices, 256, 0.25, 4)
    assert mask.shape == (256, 256) and mask.sum() > 0
    far_inside = (abs(x - 0.5) < 0.1) & (abs(y - 0.5) < 0.1)
    assert bitmap[far_inside].all() and exact[far_inside].all()
    # editing the vertices refreshes the cached bitmap
    gate.vertices = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
    bigger = gate.evaluate(acc)
    assert bigger.sum() > bitmap.sum()


@pytest.mark.skipif(not (FOLDER / "19-Aug-2025.wsp").exists(), reason="real FlowJo workspace not available")
def test_bitmap_mode_is_closer_to_flowjo_counts(exact_mode):
    from flowdesk.core.wsp_import import flowjo_counts, import_wsp

    ws, _ = import_wsp(FOLDER / "19-Aug-2025.wsp", search_dirs=[FOLDER])
    fj = flowjo_counts(FOLDER / "19-Aug-2025.wsp")

    def stats():
        # every gate of every sample: polygon gates and the rectangles below them
        diffs = []
        for s in ws.samples:
            s.root.invalidate()
            for p in s.root.walk(False):
                if (s.id, p.path) in fj and fj[(s.id, p.path)] > 0:
                    diffs.append(100.0 * (p.count - fj[(s.id, p.path)]) / fj[(s.id, p.path)])
        return float(np.sqrt(np.mean(np.square(diffs)))), max(abs(d) for d in diffs), float(np.mean(diffs))

    POLYGON_SETTINGS["mode"] = "exact"
    rms_exact, max_exact, bias_exact = stats()
    POLYGON_SETTINGS["mode"] = "bitmap"
    rms_bitmap, max_bitmap, bias_bitmap = stats()
    assert rms_bitmap < rms_exact
    assert max_bitmap <= max_exact
    assert abs(bias_bitmap) < abs(bias_exact) / 4
    assert rms_bitmap < 0.35 and max_bitmap < 1.3


def test_controller_applies_preference(exact_mode):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from flowdesk.app import create_app

    create_app([])
    from flowdesk.ui.controller import WorkspaceController

    c = WorkspaceController()
    old = c.preferences.get("polygon_gating")
    try:
        c.preferences.set("polygon_gating", "bitmap")
        c.preferencesChanged.emit()
        assert gates.POLYGON_SETTINGS["mode"] == "bitmap"
        c.preferences.set("polygon_gating", "exact")
        c.preferencesChanged.emit()
        assert gates.POLYGON_SETTINGS["mode"] == "exact"
    finally:
        c.preferences.set("polygon_gating", old or "exact")
