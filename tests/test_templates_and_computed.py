"""Workspace templates and computed parameters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flowdesk.core.fcs import Channel, write_fcs
from flowdesk.core.gates import RectangleGate
from flowdesk.core.layouts import Layout, LayoutItem, TableColumn, TableDef
from flowdesk.core.model import Population, Workspace
from flowdesk.core.serialization import load_workspace, save_workspace
from flowdesk.core.templates import TEMPLATE_EXTENSION, apply_template, build_template, load_template, save_template
from flowdesk.core.transforms import LogicleScale


def _fcs(path: Path, n: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    events = np.column_stack([rng.normal(50000, 10000, n), rng.normal(30000, 8000, n), rng.lognormal(6, 1.2, n)]).astype(np.float32)
    write_fcs(path, events, [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144), Channel(3, "FITC-A", "CD4", 262144)])
    return path


def test_template_round_trip(tmp_path):
    ws = Workspace()
    s1 = ws.add_sample(_fcs(tmp_path / "a.fcs", 300, 1))
    ws.apply_population_to_group(ws.groups[0], "", Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [40000, 0], [None, None])))
    s1.set_scale("FITC-A", LogicleScale(262144, 0.7, 4.2, 0.1))
    ws.layouts.append(Layout("Figure", [LayoutItem(sample_id=s1.id, pop_path="/Cells")]))
    ws.tables.append(TableDef("Stats", [TableColumn("/Cells", "Count")]))
    ws.settings["experiment"] = {"name": "Exp 1"}
    tpl = build_template(ws)
    assert tpl["groups"][0]["sample_ids"] == []
    assert "FITC-A" in tpl["scales"] and tpl["scales"]["FITC-A"]["kind"] == "logicle"
    path = tmp_path / f"exp{TEMPLATE_EXTENSION}"
    save_template(ws, path)
    loaded = load_template(path)
    # a fresh workspace with new samples receives the analysis
    ws2 = Workspace()
    s2 = ws2.add_sample(_fcs(tmp_path / "b.fcs", 200, 2))
    report = apply_template(ws2, loaded)
    assert s2.root.find("/Cells") is not None and s2.root.find("/Cells").count > 0
    assert s2.scale("FITC-A").kind == "logicle" and s2.scale("FITC-A").W == pytest.approx(0.7)
    assert [l.name for l in ws2.layouts] == ["Figure"] and [t.name for t in ws2.tables] == ["Stats"]
    assert ws2.settings["experiment"]["name"] == "Exp 1"
    assert report["samples"] >= 1
    # a FlowDesk workspace file also works as a template source
    save_workspace(ws, tmp_path / "w.fdw")
    tpl2 = load_template(tmp_path / "w.fdw")
    assert tpl2["groups"][0]["root"]["children"][0]["name"] == "Cells"


def test_computed_parameters_persist(tmp_path):
    ws = Workspace()
    s = ws.add_sample(_fcs(tmp_path / "c.fcs", 500, 3))
    s.ensure_loaded()
    values = np.full(s.n_events, np.nan, dtype=np.float32)
    values[:250] = np.linspace(-5, 5, 250)
    s.add_computed("tSNE_X", values, {"tool": "tSNE"})
    assert "tSNE_X" in s.parameter_names()
    assert s.scale("tSNE_X").kind == "linear" and s.scale("tSNE_X").min < -5 < 5 < s.scale("tSNE_X").max
    pop = Population("left", RectangleGate(["tSNE_X"], [None], [0.0]))
    s.root.add_child(pop)
    assert pop.count == int((values[:250] < 0).sum())  # NaN events never match
    path = tmp_path / "comp.fdw"
    save_workspace(ws, path)
    assert (tmp_path / "comp.fdw.data").is_dir()
    ws2 = load_workspace(path)
    s2 = ws2.samples[0]
    assert "tSNE_X" in s2.computed and np.isnan(s2.computed["tSNE_X"][-1])
    assert s2.root.find("/left").count == pop.count
    s2.remove_computed("tSNE_X")
    assert "tSNE_X" not in s2.parameter_names()


def test_renderer_ignores_nan_events(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from flowdesk.app import create_app

    create_app([])
    from flowdesk.core.model import GraphSettings
    from flowdesk.ui.renderer import RenderOptions, render_plot

    ws = Workspace()
    s = ws.add_sample(_fcs(tmp_path / "d.fcs", 400, 4))
    s.ensure_loaded()
    v = np.full(s.n_events, np.nan, dtype=np.float32)
    v[::2] = np.arange(200, dtype=np.float32)
    s.add_computed("UMAP_1", v)
    res = render_plot(s, s.root, GraphSettings("UMAP_1", "FSC-A", "pseudocolor"), 300, 300, RenderOptions())
    assert res.error is None and res.n_events == s.n_events
