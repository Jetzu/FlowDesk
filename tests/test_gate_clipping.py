"""A gate that reaches beyond the plotted axes is drawn only up to the axes; in data space it keeps its full extent."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))
import make_testset  # noqa: E402

from flowdesk.agent.bridgecore import EventLog  # noqa: E402
from flowdesk.agent.ops import Ops  # noqa: E402


@pytest.fixture(scope="module")
def ops(tmp_path_factory):
    from flowdesk.app import create_app
    from flowdesk.core.serialization import load_workspace
    from flowdesk.ui.controller import WorkspaceController

    create_app([])
    out = tmp_path_factory.mktemp("clip")
    make_testset.write_testset(out, seed=7, n=1500)
    c = WorkspaceController(load_workspace(out / "FlowDesk testset.fdw"))
    o = Ops(c, EventLog())
    o.out = out
    return o


def _margins_white(path) -> bool:
    from PySide6.QtGui import QImage

    img = QImage(str(path))
    w, h = img.width(), img.height()
    strips = [(x, y) for x in range(w - 8, w) for y in range(h)] + [(x, y) for x in range(w) for y in range(0, 3)]
    return all(img.pixelColor(x, y).red() > 240 and img.pixelColor(x, y).green() > 240 and img.pixelColor(x, y).blue() > 240 for x, y in strips)


def test_oversized_gate_is_drawn_only_inside_the_axes_but_counts_in_data_space(ops):
    r = ops.call("create_gate", {"name": "Wide", "shape": "polygon", "x": "FSC-A", "y": "SSC-A", "sample": "Donor1_Control",
                                 "vertices": [[100000, 50000], [900000, 50000], [900000, 900000], [100000, 900000]]})
    assert r["counts"][0]["count"] > 100  # the whole region counts, including what lies beyond the axis range
    path = ops.out / "wide.png"
    ops.call("plot", {"sample": "Donor1_Control", "x": "FSC-A", "y": "SSC-A", "return_image": False, "path": str(path)})
    assert _margins_white(path)


def test_open_ended_range_gate_stops_at_the_axis(ops):
    ops.call("create_gate", {"name": "CD4 high", "shape": "range", "x": "CD4", "sample": "Donor1_Control", "x_range": [3000, None]})
    path = ops.out / "range.png"
    ops.call("plot", {"sample": "Donor1_Control", "x": "CD4", "return_image": False, "path": str(path)})
    assert _margins_white(path)
