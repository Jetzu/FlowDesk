"""Interoperability of the newer features (computed parameters, cluster
populations, derived parameters, templates, experiment / cytometer settings,
gate tint / lock, overlay render options) with FlowJo export / import, the
layout and table editors, the graph window and workspace files."""
from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from flowdesk.core import analysis as A
from flowdesk.core.fcs import Channel, write_fcs
from flowdesk.core.gates import BooleanGate, PolygonGate, RectangleGate
from flowdesk.core.layouts import Layout, LayoutItem, TableColumn, TableDef
from flowdesk.core.model import GraphSettings, Population, StatNode, Workspace
from flowdesk.core.serialization import load_workspace, save_workspace
from flowdesk.core.templates import apply_template, build_template, load_template, save_template
from flowdesk.core.transforms import LinearScale, LogicleScale
from flowdesk.core.wsp_export import export_wsp
from flowdesk.core.wsp_import import flowjo_counts, import_wsp

REAL_DIR = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel")
REAL_WSP = REAL_DIR / "19-Aug-2025.wsp"
CHANNELS = [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144), Channel(3, "FITC-A", "CD4", 262144), Channel(4, "PE-A", "CD8", 262144)]


# --------------------------------------------------------------------- helpers
def _fcs(path: Path, n: int, seed: int, cytometer: str = "Fortessa") -> Path:
    rng = np.random.default_rng(seed)
    events = np.column_stack(
        [rng.normal(50000, 10000, n), rng.normal(30000, 8000, n), rng.lognormal(6, 1.2, n), rng.lognormal(5, 1.0, n)]
    ).astype(np.float32)
    write_fcs(path, events, CHANNELS, {"$CYT": cytometer})
    return path


def _workspace(tmp_path: Path, partial: bool = True) -> Workspace:
    """Three samples; tSNE and FlowSOM ran on a and c only (on part of the
    events of a when `partial`), b never got them. Cluster populations are
    applied through the group "Treated" (a, c)."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    ws = Workspace()
    a = ws.add_sample(_fcs(data / "a.fcs", 1000, 1))
    ws.add_sample(_fcs(data / "b.fcs", 800, 2))
    c = ws.add_sample(_fcs(data / "c.fcs", 600, 3))
    for s in ws.samples:
        s.ensure_loaded()
    ws.apply_population_to_group(ws.groups[0], "", Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 0], [None, None])))
    treated = ws.add_group("Treated", [a, c])
    rows_a = np.arange(700) if partial else np.arange(a.n_events)
    index = [(a.id, rows_a), (c.id, np.arange(c.n_events))]
    n = A.index_size(index)
    rng = np.random.default_rng(7)
    xy = rng.normal(0, 0.6, (n, 2))
    A.write_back(ws, index, {"tSNE_1": xy[:, 0], "tSNE_2": xy[:, 1]}, {"tool": "tSNE"})
    labels = rng.integers(1, 4, n).astype(float)
    A.write_back(ws, index, {"FlowSOM": labels}, {"tool": "FlowSOM"})
    A.clusters_to_populations(ws, treated, "/Cells", "FlowSOM", labels, "FlowSOM")
    return ws


def _tsne_population(ws: Workspace, sample):
    pop = Population("tSNE island", PolygonGate(["tSNE_1", "tSNE_2"], [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]))
    sample.root.find("/Cells").add_child(pop)
    return pop


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


# ------------------------------------------------------- 1. FlowJo export
def test_export_skips_gates_on_computed_and_derived_parameters(tmp_path):
    ws = _workspace(tmp_path)
    a = ws.samples[0]
    cells = a.root.find("/Cells")
    _tsne_population(ws, a).add_child(Population("big", RectangleGate(["FSC-A"], [40000.0], [None])))
    a.add_derived("CD4/CD8", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
    cells.add_child(Population("ratio high", RectangleGate(["CD4/CD8"], [1.0], [None])))
    cells.add_child(Population("FITC+", RectangleGate(["FITC-A"], [500.0], [None])))
    a.root.add_child(Population("island or FITC+", BooleanGate("or", ["/Cells/tSNE island", "/Cells/FITC+"])))
    a.root.add_child(Population("not FITC+", BooleanGate("not", ["/Cells/FITC+"])))
    cells.stats = [StatNode("Median", "tSNE_1"), StatNode("Median", "FITC-A")]
    cells.graph = GraphSettings("tSNE_1", "tSNE_2", "pseudocolor")
    a.scale("tSNE_1")  # a cached auto-scale must not become a FlowJo transform
    expected = {p.path: p.count for p in a.populations()}

    out = tmp_path / "export.wsp"
    warnings: list[str] = []
    returned = export_wsp(ws, out, warnings=warnings)
    assert returned is warnings and warnings
    text = out.read_text(encoding="utf-8")
    for name in ("tSNE_1", "tSNE_2", "FlowSOM", "CD4/CD8", "tSNE island", "island or FITC+", "ratio high"):
        assert name not in text, name
    joined = "\n".join(warnings)
    assert "computed parameters (tSNE_1, tSNE_2, FlowSOM)" in joined and "derived parameters (CD4/CD8)" in joined
    assert "/Cells/tSNE island: gated on tSNE_1, tSNE_2" in joined and "1 sub-population" in joined
    assert "3 populations (FlowSOM 1, FlowSOM 2, FlowSOM 3) gated on FlowSOM" in joined
    assert "/island or FITC+: boolean gate on the left-out tSNE island" in joined
    assert "statistic Median : tSNE_1" in joined

    ws2, report = import_wsp(out)
    assert not report.warnings, report.warnings
    a2 = ws2.sample_by_id(a.id)
    kept = {p.path: p.count for p in a2.populations()}
    assert set(kept) == {"/Cells", "/Cells/FITC+", "/not FITC+"}
    assert all(kept[p] == expected[p] for p in kept)
    assert [s.statistic + ":" + s.param for s in a2.root.find("/Cells").stats] == ["Median:FITC-A"]
    # the group templates lost their cluster populations as well
    assert ws2.group_by_name("Treated").root.find("/Cells/FlowSOM 1") is None
    assert ws2.group_by_name("Treated").root.find("/Cells") is not None
    # the old call signature still works and returns the same messages
    assert export_wsp(ws, tmp_path / "again.wsp") == warnings


def test_export_without_foreign_parameters_has_no_warnings_and_keeps_tint_lock(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    ws = Workspace()
    s = ws.add_sample(_fcs(data / "s.fcs", 500, 4))
    gate = RectangleGate(["FSC-A", "SSC-A"], [30000, 0], [None, None])
    gate.tint, gate.locked, gate.color = 0.3, True, "#ff0000"
    s.root.add_child(Population("Cells", gate))
    out = tmp_path / "plain.wsp"
    assert export_wsp(ws, out) == []
    el = next(e for e in ET.parse(out).getroot().iter() if e.tag.endswith("RectangleGate"))
    assert el.get("tint") == "#ff0000" and el.get("isTinted") == "1"  # FlowJo's tint is the gate colour
    ws2, report = import_wsp(out)
    assert not report.warnings
    g2 = ws2.samples[0].root.find("/Cells").gate
    assert g2.color == "#ff0000" and ws2.samples[0].root.find("/Cells").count == s.root.find("/Cells").count


def test_real_workspace_export_with_computed_parameter(tmp_path):
    if not REAL_WSP.exists():
        pytest.skip("real FlowJo workspace not available")
    ws, _ = import_wsp(REAL_WSP, search_dirs=[REAL_DIR])
    original = {(s.id, p.path) for s in ws.samples for p in s.populations()}
    s0 = ws.samples[0]
    assert s0.ensure_loaded()
    values = np.where(np.arange(s0.n_events) % 2 == 0, 1.0, 2.0).astype(np.float32)
    s0.add_computed("FlowSOM", values, {"tool": "FlowSOM"})
    A.clusters_to_populations(ws, None, "", "FlowSOM", [1, 2], "Cluster")
    out = tmp_path / "real.wsp"
    warnings = export_wsp(ws, out)
    assert any("Cluster 1, Cluster 2" in w for w in warnings)
    ws2, report = import_wsp(out, search_dirs=[REAL_DIR])
    assert {(s.id, p.path) for s in ws2.samples for p in s.populations()} == original
    assert not [w for w in report.warnings if "not found" not in w]
    stored = flowjo_counts(out)
    assert all(c == -1 or c == ws.sample_by_id(sid).root.find(path).count for (sid, path), c in stored.items() if path)


# ------------------------------------------------------- 2. FlowJo import
def _inject_unknown_dimensions(src: Path, dst: Path):
    """Add FlowJo derived parameters and gates FlowDesk cannot evaluate."""
    gating = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
    dt = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
    tree = ET.parse(src)
    root = tree.getroot()
    sample_el = root.find("SampleList/Sample")
    dps = ET.Element("DerivedParameters")
    ET.SubElement(dps, "DerivedParameter", {"name": "CD4/CD8", "type": "Ratio"})
    ET.SubElement(dps, "DerivedParameter", {"name": "Unused ratio", "type": "Ratio"})
    sample_el.insert(3, dps)
    cells = sample_el.find("SampleNode/Subpopulations/Population")
    sub = cells.find("Subpopulations")
    if sub is None:
        sub = ET.SubElement(cells, "Subpopulations")
    parent_gid = cells.find("Gate").get(f"{{{gating}}}id")

    def population(name: str, gid: str, dim_child: ET.Element) -> ET.Element:
        pop = ET.SubElement(sub, "Population", {"name": name, "count": "17"})
        g = ET.SubElement(pop, "Gate", {f"{{{gating}}}id": gid, f"{{{gating}}}parent_id": parent_gid})
        rect = ET.SubElement(g, f"{{{gating}}}RectangleGate", {"eventsInside": "1"})
        dim = ET.SubElement(rect, f"{{{gating}}}dimension", {f"{{{gating}}}min": "1"})
        dim.append(dim_child)
        return pop

    ratio = population("ratio high", "ID900001", ET.Element(f"{{{dt}}}fcs-dimension", {f"{{{dt}}}name": "CD4/CD8"}))
    ET.SubElement(ET.SubElement(ratio, "Subpopulations"), "Population", {"name": "child"})
    population("new dimension", "ID900002", ET.Element(f"{{{dt}}}new-dimension", {f"{{{dt}}}transformation-ref": "fratio_1"}))
    population("missing channel", "ID900003", ET.Element(f"{{{dt}}}fcs-dimension", {f"{{{dt}}}name": "APC-A"}))
    tree.write(dst, encoding="UTF-8", xml_declaration=True)


def test_import_gates_on_unknown_parameters_become_empty_populations(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    ws = Workspace()
    s = ws.add_sample(_fcs(data / "s.fcs", 500, 5))
    s.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 0], [None, None])))
    base = tmp_path / "base.wsp"
    export_wsp(ws, base)
    odd = tmp_path / "odd.wsp"
    _inject_unknown_dimensions(base, odd)

    ws2, report = import_wsp(odd, search_dirs=[data])
    s2 = ws2.samples[0]
    for path in ("/Cells/ratio high", "/Cells/ratio high/child", "/Cells/new dimension", "/Cells/missing channel"):
        pop = s2.root.find(path)
        assert pop is not None, path
        assert pop.count == 0, path  # never "ungated": that would copy the parent's events
    assert s2.root.find("/Cells").count == s.root.find("/Cells").count
    assert s2.root.find("/Cells/new dimension").gate.dims == ["fratio_1"]
    joined = "\n".join(report.warnings)
    assert "CD4/CD8 is a FlowJo derived parameter (not imported)" in joined
    assert "APC-A is not a parameter of the FCS file" in joined
    assert "'fratio_1' is a transformed (ratio / new) dimension" in joined
    assert "FlowJo derived parameters are not imported: Unused ratio" in joined
    # exporting it again leaves the unusable populations out instead of failing
    warnings = export_wsp(ws2, tmp_path / "re.wsp")
    assert any("ratio high" in w for w in warnings) and any("missing channel" in w for w in warnings)


# ------------------------------------------------------------- 3. templates
def test_template_skips_computed_data_but_keeps_gates(tmp_path):
    ws = _workspace(tmp_path)
    a, b, c = ws.samples
    a.add_derived("CD4/CD8", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
    ws.apply_population_to_group(ws.groups[0], "/Cells", Population("ratio high", RectangleGate(["CD4/CD8"], [1.0], [None])))
    a.set_scale("CD4/CD8", LinearScale(0.0, 5.0))
    a.set_scale("FITC-A", LogicleScale(262144, 0.7, 4.2, 0.1))
    for s in ws.samples:
        s.scale("tSNE_1")  # auto-scales (b: cached for a parameter it lacks)
    ws.settings["experiment"] = {"name": "Exp"}
    ws.settings["cytometers"] = {"Fortessa": {"FITC-A": {"kind": "logicle", "width_basis": -100.0, "decades": 4.5, "neg": 0.0}}}
    ws.settings["plate_editor"] = {"format": 96, "manual": {str(a.id): "A1"}}

    tpl = build_template(ws)
    text = json.dumps(tpl)
    assert "computed" not in text and ".npy" not in text
    assert set(tpl["scales"]) == {"FITC-A", "CD4/CD8"}  # no tSNE_1 / FlowSOM scales
    assert tpl["derived"] == {"CD4/CD8": {"op": "ratio", "a": "FITC-A", "b": "PE-A"}}
    assert tpl["settings"]["plate_editor"] == {"format": 96} and tpl["settings"]["cytometers"]
    assert "manual" in ws.settings["plate_editor"]  # the source workspace is untouched
    path = tmp_path / "t.fdwt"
    save_template(ws, path)

    ws2 = Workspace()
    data2 = tmp_path / "new"
    data2.mkdir()
    x = ws2.add_sample(_fcs(data2 / "x.fcs", 700, 11))
    y = ws2.add_sample(_fcs(data2 / "y.fcs", 500, 12))
    ws2.add_group("Treated", [x])
    report = apply_template(ws2, load_template(path))
    assert report["derived"] == 2
    clusters = [x.root.find(f"/Cells/FlowSOM {k}") for k in (1, 2, 3)]
    assert all(p is not None and p.count == 0 for p in clusters)  # parameter not computed yet
    assert y.root.find("/Cells/FlowSOM 1") is None  # y is not in "Treated"
    assert x.scale("FITC-A").kind == "logicle" and "tSNE_1" not in x.scales
    assert x.root.find("/Cells/ratio high").count == int(((x.data("FITC-A") / x.data("PE-A")) >= 1.0)[x.root.find("/Cells").mask()].sum())
    assert x.scale("CD4/CD8").max == pytest.approx(5.0)
    assert "manual" not in ws2.settings.get("plate_editor", {})
    # once the parameter exists the cluster gates select events
    ids = np.random.default_rng(0).integers(1, 4, x.n_events).astype(np.float32)
    x.add_computed("FlowSOM", ids, {"tool": "FlowSOM"})
    counts = [p.count for p in clusters]
    cells = x.root.find("/Cells").mask()
    assert counts == [int(((ids == k) & cells).sum()) for k in (1, 2, 3)] and sum(counts) == int(cells.sum())


def test_template_recreates_calibrated_parameters(tmp_path):
    from flowdesk.core import calibration as cal

    ws = _workspace(tmp_path)
    a = ws.samples[0]
    fit = cal.fit_calibration([100.0, 1000.0, 10000.0], [500.0, 5000.0, 50000.0], param="FITC-A")
    name = cal.apply_calibration(a, fit, "FITC-A")
    cal.store_calibration(ws, fit, "FITC-A")
    ws.apply_population_to_group(ws.groups[0], "/Cells", Population("MESF high", RectangleGate([name], [5000.0], [None])))
    assert a.root.find("/Cells/MESF high").count > 0
    tpl = build_template(ws)
    assert name not in tpl["scales"] and "computed" not in json.dumps(tpl)
    assert tpl["settings"]["calibrations"]["FITC-A"]["slope"] == pytest.approx(1.0)

    ws2 = Workspace()
    (tmp_path / "new").mkdir()
    x = ws2.add_sample(_fcs(tmp_path / "new" / "x.fcs", 500, 21))
    report = apply_template(ws2, tpl)
    assert report["calibrated"] == 1 and name in x.computed
    pop = x.root.find("/Cells/MESF high")
    expected = int(((x.data("FITC-A") * 5.0 >= 5000.0) & x.root.find("/Cells").mask()).sum())
    assert pop.count == pytest.approx(expected, abs=2) and pop.count > 0


# ------------------------------------------------------ 4. layout / table
def test_layout_plots_on_computed_parameters(app, tmp_path):
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.layout_editor import LayoutEditorWindow
    from flowdesk.ui.renderer import RenderOptions, render_plot

    ws = _workspace(tmp_path)
    a, b, c = ws.samples
    ctl = WorkspaceController(ws)
    editor = LayoutEditorWindow(ctl)
    editor.show()
    app.processEvents()
    try:
        item = editor.add_population(a, a.root.find("/Cells"))
        item.graph = GraphSettings("tSNE_1", "tSNE_2", "pseudocolor", {"show_legend": False})
        item.overlays = [{"sample_id": b.id, "pop_path": "/Cells"}]
        hist = editor.add_population(a, a.root.find("/Cells"))
        hist.graph = GraphSettings("tSNE_1", None, "histogram", {"histogram_offset": True})
        hist.overlays = [{"sample_id": b.id, "pop_path": "/Cells"}, {"sample_id": c.id, "pop_path": "/Cells"}]
        editor.refresh()
        pm = editor.page_pixmap()
        assert not pm.isNull() and pm.width() > 0
        assert editor.export_pdf(tmp_path / "one.pdf") == 1 and (tmp_path / "one.pdf").stat().st_size > 1000
        # batch over All Samples: b has no tSNE -> a message, not an exception
        assert editor.export_pdf(tmp_path / "batch.pdf", batch_group=ws.groups[0]) == 3
        res = render_plot(b, b.root.find("/Cells"), item.graph, 300, 300, RenderOptions())
        assert res.error and "tSNE_1" in res.error
        assert "tSNE_1" not in b.scales  # no default scale cached for a missing parameter
        ok = render_plot(a, a.root.find("/Cells"), item.graph, 300, 300, RenderOptions(), overlays=[(b, b.root.find("/Cells"))])
        assert ok.error is None and ok.n_events == a.root.find("/Cells").count
        assert "(no tSNE_1 / tSNE_2)" in ok.legend_entries[1][0]
        # the render options of a layout plot survive to_dict/from_dict and save/load
        back = LayoutItem.from_dict(json.loads(json.dumps(hist.to_dict())))
        assert back.graph.options == {"histogram_offset": True} and back.overlays == hist.overlays
        save_workspace(ws, tmp_path / "lay.fdw")
        lay = load_workspace(tmp_path / "lay.fdw").layouts[0]
        assert [i.graph.options for i in lay.items] == [{"show_legend": False}, {"histogram_offset": True}]
    finally:
        editor.flush_notify()
        editor.hide()
        editor.deleteLater()


def test_layout_plot_options_dialog_edits_overlay_render_options(app, tmp_path):
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.layout_editor import LayoutEditorWindow, PlotOptionsDialog

    ws = _workspace(tmp_path)
    a = ws.samples[0]
    editor = LayoutEditorWindow(WorkspaceController(ws))
    try:
        item = editor.add_population(a, a.root.find("/Cells"))
        item.graph = GraphSettings("tSNE_1", None, "histogram", {"show_legend": False})
        dlg = PlotOptionsDialog(editor, item, editor)
        assert [dlg.x_combo.itemText(i) for i in range(dlg.x_combo.count())][-3:] == ["tSNE_1", "tSNE_2", "FlowSOM"]
        assert not dlg.legend_check.isChecked() and not dlg.offset_check.isChecked()
        dlg.legend_check.setChecked(True)
        dlg.offset_check.setChecked(True)
        dlg.apply()
        assert item.graph.options == {"show_legend": True, "histogram_offset": True}
        assert item.graph.x == "tSNE_1" and item.graph.plot_type == "histogram"
    finally:
        editor.hide()
        editor.deleteLater()


def test_table_statistics_on_computed_parameters(app, tmp_path):
    from flowdesk.ui.table_editor import build_table

    ws = _workspace(tmp_path, partial=False)
    a, b, c = ws.samples
    td = TableDef(
        "T",
        [
            TableColumn("/Cells", "Median", "tSNE_1"),
            TableColumn("/Cells/FlowSOM 1", "Count"),
            TableColumn("/Cells", "Percentile", "tSNE_2", 95.0),
            TableColumn("/Cells", "Mode", "FlowSOM"),
        ],
    )
    headers, rows = build_table(ws, td)
    by_name = {r[0]: r[1:] for r in rows}
    assert by_name["b.fcs"] == ["", "", "", ""]  # no tSNE / no cluster population: blank, no exception
    med = float(by_name["a.fcs"][0])
    assert med == pytest.approx(float(np.median(a.data("tSNE_1")[a.root.find("/Cells").mask()])), abs=1e-3)
    assert int(by_name["a.fcs"][1]) == a.root.find("/Cells/FlowSOM 1").count


# ------------------------------------------------------------ 5. graph window
def test_graph_window_with_computed_parameters(app, tmp_path, monkeypatch):
    from flowdesk.ui import graph_window as gw_module
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.dialogs import ScaleDialog
    from flowdesk.ui.graph_window import GraphWindow, prepare_scale_dialog

    ws = _workspace(tmp_path)
    a, b, c = ws.samples
    ctl = WorkspaceController(ws)
    gw = GraphWindow(ctl, a, a.root.find("/Cells"))
    gw.show()
    app.processEvents()
    try:
        gw._fill_axis_menu("x")
        labels = [act.text() for act in gw.x_button.menu().actions()]
        assert {"tSNE_1", "tSNE_2", "FlowSOM"} <= set(labels)
        gw.plot.set_axes(x="tSNE_1", y="tSNE_2")
        gw.plot.set_overlays([(b, b.root.find("/Cells")), (c, c.root.find("/Cells"))])
        app.processEvents()
        assert not gw.plot_image().isNull()
        res = gw._render_onto(None, 1.0)
        assert res.error is None and "(no tSNE_1 / tSNE_2)" in res.legend_entries[1][0]
        gw.plot.set_axes(y=None)  # histogram overlay, staggered
        a.root.find("/Cells").graph.options["histogram_offset"] = True
        gw.plot.invalidate()
        assert gw._render_onto(None, 1.0).error is None
        gw.stats_action.setChecked(True)
        gw._refresh_stats()
        assert gw.stats_table.rowCount() >= 3

        # the T dialog keeps the decimals of a computed parameter's linear auto-scale
        auto = a.scale("tSNE_1")
        assert auto.kind == "linear" and auto.max - auto.min < 10
        dlg = ScaleDialog(a, "tSNE_1", auto, None, None)
        prepare_scale_dialog(dlg, auto)
        got = dlg.scale()
        assert got.kind == "linear" and got.min == pytest.approx(auto.min, abs=1e-3) and got.max == pytest.approx(auto.max, abs=1e-3)
        dlg.deleteLater()
        monkeypatch.setattr(ScaleDialog, "exec", lambda self: True)
        assert gw_module.ScaleDialog is ScaleDialog
        gw.plot.set_axes(x="tSNE_1", y="tSNE_2")
        gw._customize_axis("x")
        assert a.scale("tSNE_1").min == pytest.approx(auto.min, abs=1e-3)
        assert a.scale("tSNE_1").max == pytest.approx(auto.max, abs=1e-3)

        # a graph window of a sample without the parameter shows a message
        gw_b = GraphWindow(ctl, b, b.root.find("/Cells"))
        gw_b.plot.set_axes(x="tSNE_1", y="tSNE_2")
        assert "tSNE_1" in (gw_b._render_onto(None, 1.0).error or "")
        gw_b.close()
    finally:
        gw.close()


# ------------------------------------------------------------- 6. save / load
def test_save_load_everything_and_move_with_data_folder(tmp_path):
    ws = _workspace(tmp_path)
    a, b, c = ws.samples
    _tsne_population(ws, a)
    ws.settings["experiment"] = {"name": "Exp 2", "panel": [{"parameter": "FITC-A", "stain": "CD4"}]}
    ws.settings["cytometers"] = {"Fortessa": {"FITC-A": {"kind": "logicle", "width_basis": -100.0, "decades": 4.5, "neg": 0.0}}}
    # a group that came from a template
    tpl = {"format": "flowdesk-template", "groups": [{"name": "From template", "role": "custom", "criteria": [{"keyword": "$FIL", "function": "Contains", "value": "b"}], "root": {"name": "From template", "children": [{"name": "Big", "gate": {"kind": "rectangle", "dims": ["FSC-A"], "lo": [60000], "hi": [None]}, "children": []}]}}]}
    apply_template(ws, tpl, replace_groups=False)
    gate = ws.groups[0].root.find("/Cells").gate
    gate.tint, gate.locked = 0.3, True
    for s in ws.samples:
        g = s.root.find("/Cells").gate
        g.tint, g.locked = 0.15, True
    ws.layouts.append(Layout("Fig", [LayoutItem(sample_id=a.id, pop_path="/Cells", graph=GraphSettings("tSNE_1", "tSNE_2", "dot", {"show_legend": False}))]))
    before = {s.name: {p.path: p.count for p in s.populations()} for s in ws.samples}
    members = {g.name: [s.name for s in ws.members(g)] for g in ws.groups}
    assert before["b.fcs"]["/Big"] > 0 and before["a.fcs"]["/Cells/FlowSOM 1"] > 0

    first = tmp_path / "first"
    first.mkdir()
    save_workspace(ws, first / "w.fdw")
    stored = json.loads((first / "w.fdw").read_text(encoding="utf-8"))
    assert str(first) not in json.dumps([s.get("computed") for s in stored["samples"]])  # data files by name only
    moved = tmp_path / "moved" / "deeper"
    moved.mkdir(parents=True)
    shutil.move(str(first / "w.fdw"), str(moved / "w.fdw"))
    shutil.move(str(first / "w.fdw.data"), str(moved / "w.fdw.data"))
    shutil.rmtree(first)

    ws2 = load_workspace(moved / "w.fdw")
    assert [s.name for s in ws2.samples] == ["a.fcs", "b.fcs", "c.fcs"]
    assert {g.name: [s.name for s in ws2.members(g)] for g in ws2.groups} == members
    a2, b2, c2 = ws2.samples
    assert set(a2.computed) == {"tSNE_1", "tSNE_2", "FlowSOM"} and not b2.computed
    assert np.array_equal(a2.computed["tSNE_1"], a.computed["tSNE_1"], equal_nan=True)
    assert a2.computed_info["FlowSOM"]["tool"] == "FlowSOM"
    assert {s.name: {p.path: p.count for p in s.populations()} for s in ws2.samples} == before
    assert ws2.settings["experiment"]["name"] == "Exp 2" and ws2.settings["cytometers"]["Fortessa"]["FITC-A"]["kind"] == "logicle"
    g2 = a2.root.find("/Cells").gate
    assert g2.tint == pytest.approx(0.15) and g2.locked
    assert ws2.groups[0].root.find("/Cells").gate.locked and ws2.groups[0].root.find("/Cells").gate.tint == pytest.approx(0.3)
    assert ws2.layouts[0].items[0].graph.options == {"show_legend": False}


# --------------------------------------------------- undo of editor changes
def test_layout_edits_are_undo_steps(app, tmp_path):
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.layout_editor import LayoutEditorWindow, PlotGraphicsItem

    ws = _workspace(tmp_path)
    a = ws.samples[0]
    ctl = WorkspaceController(ws)
    editor = LayoutEditorWindow(ctl)
    editor.show()
    app.processEvents()
    try:
        layout = editor.current_layout()
        steps = len(ctl._undo)
        item = editor.add_population(a, a.root.find("/Cells"))
        assert ctl.undo_label() == "edit layout" and len(ctl._undo) == steps + 1
        # a drag (timer-driven notifications) is a single further step
        x0 = item.x
        for dx in (10.0, 20.0, 30.0):
            item.x += dx
            editor.schedule_notify()
            editor.flush_notify()
        assert len(ctl._undo) == steps + 2
        ctl.undo()
        assert layout.items[0].x == pytest.approx(x0)
        ctl.undo()
        assert ws.layouts[0] is layout and layout.items == []
        assert not [gi for gi in editor.scene.items() if isinstance(gi, PlotGraphicsItem)]
        ctl.redo()
        assert [i.id for i in layout.items] == [item.id]
        assert len([gi for gi in editor.scene.items() if isinstance(gi, PlotGraphicsItem)]) == 1
        # gating changed in between is not reverted by undoing a layout edit
        editor.add_population(a, a.root.find("/Cells"))
        ctl.begin_change("rename")
        a.root.find("/Cells").name = "Cells renamed"
        ctl.undo()
        ctl.undo()
        assert len(layout.items) == 1 and a.root.find("/Cells") is not None
    finally:
        editor.flush_notify()
        editor.hide()
        editor.deleteLater()


def test_table_edits_are_undo_steps(app, tmp_path):
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.table_editor import TableEditorWindow

    ws = _workspace(tmp_path)
    ctl = WorkspaceController(ws)
    editor = TableEditorWindow(ctl)
    try:
        assert len(ws.tables) == 1 and not ctl.can_undo  # the default table is not an undoable edit
        editor.add_columns_for_paths(["/Cells"], "Median", "tSNE_1")
        table = ws.tables[0]
        col = table.columns[0]
        col.statistic, col.label = "Mean", "tSNE mean"
        editor._commit([0])
        assert editor.columns.item(0, 3).text() == "tSNE mean"
        ctl.undo()
        assert table.columns[0].statistic == "Median" and table.columns[0].label == ""
        assert editor.columns.item(0, 3).text() == "" and editor.columns.item(0, 1).text().startswith("Median")
        ctl.undo()
        assert table.columns == [] and editor.columns.rowCount() == 0
        ctl.redo()
        ctl.redo()
        assert ws.tables[0].columns[0].label == "tSNE mean" and editor.columns.item(0, 3).text() == "tSNE mean"
    finally:
        editor.hide()
        editor.deleteLater()
