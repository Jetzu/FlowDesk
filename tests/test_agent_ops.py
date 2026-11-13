"""The agent operations (flowdesk.agent.ops) on the generated test set: gating, statistics, plots, analysis and export."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))
import make_testset  # noqa: E402

from flowdesk.agent.bridgecore import EventLog  # noqa: E402
from flowdesk.agent.ops import Ops  # noqa: E402
from flowdesk.agent.ops_base import OpError  # noqa: E402

CD3 = "/Lymphocytes/Single cells/CD3+"


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture(scope="module")
def testset(tmp_path_factory):
    out = tmp_path_factory.mktemp("agent_testset")
    make_testset.write_testset(out, seed=7, n=4000)
    return out


@pytest.fixture
def ops(app, testset):
    from flowdesk.core.serialization import load_workspace
    from flowdesk.ui.controller import WorkspaceController

    c = WorkspaceController(load_workspace(testset / "FlowDesk testset.fdw"))
    c.workspace.modified = False
    return Ops(c, EventLog())


def _comp(sample, param: str) -> np.ndarray:
    return np.asarray(sample.data(param))


# ----------------------------------------------------------------- state & references
def test_ping_and_workspace_summary(ops):
    assert ops.ping()["samples"] == 6
    w = ops.workspace()
    assert [s["name"] for s in w["samples"]][:2] == ["Donor1_Control.fcs", "Donor2_Control.fcs"]
    assert {g["name"] for g in w["groups"]} == {"All Samples", "Compensation", "Control", "Stimulated"}
    assert CD3 in w["populations"] and "Comp-PE-A" in w["parameters"]
    assert w["experiment"]["name"] == "FlowDesk test set"
    assert w["undo"] == {"can_undo": False, "next_undo": "", "can_redo": False}
    assert next(s for s in w["samples"] if s["name"] == "Donor1_Control.fcs")["keywords"]["Condition"] == "Control"


def test_sample_references_are_tolerant_and_errors_name_the_candidates(ops):
    assert ops.sample("donor1 control").name == "Donor1_Control.fcs"
    assert ops.sample("Donor2_Stimulated").name == "Donor2_Stimulated.fcs"
    assert ops.sample(1).id == 1 and ops.sample("#2").id == 2
    with pytest.raises(OpError, match="several samples.*Donor1_Control.*Donor1_Stimulated"):
        ops.sample("Donor1")
    with pytest.raises(OpError, match="no sample 'nope'.*Donor3_Stimulated"):
        ops.sample("nope")


def test_parameter_references_resolve_stains_to_compensated_parameters(ops):
    s = ops.sample("Donor1_Control")
    assert ops.param(s, "CD4") == "Comp-PE-A"
    assert ops.param(s, "cd69") == "Comp-PE-Cy7-A"
    assert ops.param(s, "FSC-A") == "FSC-A"
    assert ops.param(s, "Comp-FITC-A") == "Comp-FITC-A"
    with pytest.raises(OpError, match="no parameter 'CD99'"):
        ops.param(s, "CD99")


def test_sample_info_lists_parameters_with_stain_and_range(ops):
    info = ops.sample_info("Donor1_Control")
    by_name = {p["name"]: p for p in info["parameters"]}
    assert by_name["Comp-PE-A"]["stain"] == "CD4" and by_name["Comp-PE-A"]["scale"] == "logicle"
    assert info["compensation"]["matrix"] == "Acquisition-defined" and info["events"] == 4000


def test_populations_tree_has_counts_and_percentages(ops):
    tree = ops.populations(sample="Donor1_Control")["tree"]
    lym = tree["children"][0]
    assert lym["name"] == "Lymphocytes" and lym["count"] == ops.sample("Donor1_Control").root.find("/Lymphocytes").count
    assert 0 < lym["percent_of_parent"] < 100
    group_tree = ops.populations(group="Stimulated")
    assert group_tree["members"] == ["Donor1_Stimulated.fcs", "Donor2_Stimulated.fcs", "Donor3_Stimulated.fcs"]


# --------------------------------------------------------------------------- gating
def test_range_gate_counts_match_the_raw_data(ops):
    r = ops.create_gate("CD4 high", "range", x="CD4", x_range=[4000, None], sample="Donor1_Control", parent=CD3)
    s = ops.sample("Donor1_Control")
    parent_mask = s.root.find(CD3).mask()
    expected = int((_comp(s, "Comp-PE-A")[parent_mask] >= 4000).sum())
    assert r["counts"][0]["count"] == expected > 0
    assert r["population"] == CD3 + "/CD4 high"
    assert ops.c.can_undo


def test_gate_on_a_group_reaches_every_member_and_undo_removes_it(ops):
    r = ops.create_gate("Cytotoxic", "rectangle", x="CD8", y="CD4", x_range=[4000, None], y_range=[None, 4000], group="Stimulated", parent=CD3)
    assert r["applied_to"] == ["Donor1_Stimulated.fcs", "Donor2_Stimulated.fcs", "Donor3_Stimulated.fcs"]
    assert all(row["count"] > 0 for row in r["counts"])
    assert ops.ws.group_by_name("Stimulated").root.find(CD3 + "/Cytotoxic") is not None
    assert ops.sample("Donor1_Control").root.find(CD3 + "/Cytotoxic") is None  # other group untouched
    ops.undo()
    assert all(ops.sample(n).root.find(CD3 + "/Cytotoxic") is None for n in ("Donor1_Stimulated", "Donor2_Stimulated"))


def test_polygon_ellipse_and_boolean_gates(ops):
    poly = ops.create_gate("Big lymphs", "polygon", x="FSC-A", y="SSC-A", sample="Donor2_Control",
                           vertices=[[60000, 8000], [100000, 8000], [100000, 45000], [60000, 45000]])
    s = ops.sample("Donor2_Control")
    fsc, ssc = _comp(s, "FSC-A"), _comp(s, "SSC-A")
    inside = (fsc > 60000) & (fsc < 100000) & (ssc > 8000) & (ssc < 45000)
    assert abs(poly["counts"][0]["count"] - int(inside.sum())) <= 0.01 * inside.sum()
    ell = ops.create_gate("Round lymphs", "ellipse", x="FSC-A", y="SSC-A", sample="Donor2_Control",
                          center=[80000, 25000], a_end=[100000, 25000], b_end=[80000, 45000])
    assert 0 < ell["counts"][0]["count"] < s.n_events
    a = ops.create_gate("CD69 pos", "range", x="CD69", x_range=[3000, None], sample="Donor2_Control", parent=CD3)
    b = ops.create_gate("CD4 pos", "range", x="CD4", x_range=[4000, None], sample="Donor2_Control", parent=CD3)
    union = ops.create_gate("CD69 or CD4", "boolean", sample="Donor2_Control", parent=CD3, operation="or",
                            refs=[CD3 + "/CD69 pos", CD3 + "/CD4 pos"])
    assert union["counts"][0]["count"] >= max(a["counts"][0]["count"], b["counts"][0]["count"])
    p = s.root.find(CD3)
    both = (_comp(s, "Comp-PE-Cy7-A")[p.mask()] >= 3000) | (_comp(s, "Comp-PE-A")[p.mask()] >= 4000)
    assert union["counts"][0]["count"] == int(both.sum())


def test_edit_gate_changes_counts_and_keeps_children(ops):
    ops.create_gate("Pos", "range", x="CD69", x_range=[6000, None], sample="Donor3_Stimulated", parent=CD3)
    ops.create_gate("Child", "range", x="CD4", x_range=[4000, None], sample="Donor3_Stimulated", parent=CD3 + "/Pos")
    before = ops.sample("Donor3_Stimulated").root.find(CD3 + "/Pos").count
    r = ops.edit_gate(CD3 + "/Pos", sample="Donor3_Stimulated", x_range=[3000, None])
    assert r["counts"][0]["count"] > before
    assert ops.sample("Donor3_Stimulated").root.find(CD3 + "/Pos/Child") is not None


def test_rename_and_delete_population(ops):
    ops.create_gate("Temp", "range", x="CD4", x_range=[4000, None], sample="Donor1_Control", parent=CD3)
    ops.rename_population(CD3 + "/Temp", "Renamed", sample="Donor1_Control")
    assert ops.sample("Donor1_Control").root.find(CD3 + "/Renamed") is not None
    gone = ops.delete_population(CD3 + "/Renamed", sample="Donor1_Control")
    assert gone["deleted"] == [CD3 + "/Renamed"]
    ops.undo()
    assert ops.sample("Donor1_Control").root.find(CD3 + "/Renamed") is not None


def test_gating_mistakes_get_actionable_errors(ops):
    with pytest.raises(OpError, match="say where the gate goes"):
        ops.create_gate("X", "range", x="CD4", x_range=[1, None])
    with pytest.raises(OpError, match="either sample= or group="):
        ops.create_gate("X", "range", x="CD4", x_range=[1, None], sample="Donor1_Control", group="Control")
    with pytest.raises(OpError, match="parent population '/Nope'.*does not exist"):
        ops.create_gate("X", "range", x="CD4", x_range=[1, None], sample="Donor1_Control", parent="/Nope")
    with pytest.raises(OpError, match="already has a child named 'CD3\\+'"):
        ops.create_gate("CD3+", "range", x="CD3", x_range=[1, None], sample="Donor1_Control", parent="/Lymphocytes/Single cells")
    with pytest.raises(OpError, match="x_range"):
        ops.create_gate("X", "range", x="CD4", sample="Donor1_Control", parent=CD3)
    with pytest.raises(OpError, match="at least 3 points"):
        ops.create_gate("X", "polygon", x="FSC-A", y="SSC-A", vertices=[[1, 2]], sample="Donor1_Control")
    with pytest.raises(OpError, match="shape must be"):
        ops.create_gate("X", "triangle", x="FSC-A", sample="Donor1_Control")


# ------------------------------------------------------------------------ statistics
def test_statistics_table_matches_numpy(ops):
    t = ops.statistics(populations=[CD3, CD3 + "/CD69+"], statistics=["Count", "median", "Freq. of Parent"], parameters="CD69",
                       samples=["Donor1_Control", "Donor1_Stimulated"])
    assert t["columns"][0] == "sample" and len(t["rows"]) == 2
    for row in t["rows"]:
        s = ops.sample(row[0])
        mask = s.root.find(CD3).mask()
        assert row[t["columns"].index("%s | Count" % CD3)] == int(mask.sum())
        med = float(np.median(_comp(s, "Comp-PE-Cy7-A")[mask]))
        assert row[t["columns"].index("%s | Median | Comp-PE-Cy7-A" % CD3)] == pytest.approx(med, rel=1e-3)
    pos = t["columns"].index("%s | Freq. of Parent" % (CD3 + "/CD69+"))
    assert t["rows"][1][pos] > 3 * t["rows"][0][pos]  # stimulation raises CD69+ cells


def test_statistics_errors(ops):
    with pytest.raises(OpError, match="need parameters"):
        ops.statistics(populations=[CD3], statistics=["Median"])
    with pytest.raises(OpError, match="unknown statistic 'banana'"):
        ops.statistics(populations=[CD3], statistics=["banana"])


def test_histogram_suggests_the_valley_between_negative_and_positive(ops):
    h = ops.histogram("Donor1_Control", "CD3", "/Lymphocytes")
    assert h["bimodal"] is True and len(h["peaks"]) >= 2
    thr = h["suggested_threshold_raw"]
    assert 400 < thr < 15000
    s = ops.sample("Donor1_Control")
    vals = _comp(s, "Comp-FITC-A")[s.root.find("/Lymphocytes").mask()]
    assert h["percent_above_threshold"] == pytest.approx(100 * float((vals >= thr).mean()), abs=2.0)
    unimodal = ops.histogram("Donor1_Control", "FSC-A", "/Lymphocytes")
    assert unimodal["bimodal"] is False


# --------------------------------------------------------------------------- plots
def test_plot_returns_a_png_and_a_numerical_description(ops, tmp_path):
    r = ops.plot("Donor1_Control", population=CD3, x="CD4", y="CD8", width=480, height=360, path=str(tmp_path / "p.png"))
    assert r["png"][:8] == b"\x89PNG\r\n\x1a\n" and (tmp_path / "p.png").exists()
    assert r["events"] == ops.sample("Donor1_Control").root.find(CD3).count
    assert [a["parameter"] for a in r["axes"]] == ["Comp-PE-A", "Comp-APC-A"]
    assert len(r["density"]["map"]) == 18 and set("".join(r["density"]["map"])) - {" "}
    assert {g["name"] for g in r["gates_on_plot"]} >= {"Q1: CD4- , CD8+", "Q3: CD4+ , CD8-"}
    hist = ops.plot("Donor1_Control", population="/Lymphocytes", x="CD3", return_image=False)
    assert hist["plot_type"] == "histogram" and hist["histogram"]["bimodal"] is True and "png" not in hist
    with pytest.raises(OpError, match="plot_type must be"):
        ops.plot("Donor1_Control", x="CD3", y="CD4", plot_type="sphere")


# ---------------------------------------------------------------- keywords, groups, files
def test_keywords_and_live_groups(ops):
    ops.set_keyword("Donor1_Control", "Batch", "B7")
    assert ops.sample("Donor1_Control").keywords["Batch"] == "B7"
    g = ops.create_group("Batch 7", criteria=[{"keyword": "Batch", "function": "Equals", "value": "B7"}])
    assert g["members"] == ["Donor1_Control.fcs"]
    ops.set_keyword("Donor2_Control", "Batch", "B7")
    assert len(ops.ws.members(ops.ws.group_by_name("Batch 7"))) == 2  # a live keyword group follows the data
    with pytest.raises(OpError, match="function"):
        ops.create_group("Bad", criteria=[{"keyword": "Batch", "function": "Resembles", "value": "x"}])
    with pytest.raises(OpError, match="cannot be deleted"):
        ops.delete_group("All Samples")


def test_unsaved_changes_are_protected_and_save_open_round_trips(ops, tmp_path):
    ops.create_gate("Marker", "range", x="CD4", x_range=[4000, None], sample="Donor1_Control", parent=CD3)
    assert ops.ws.modified
    with pytest.raises(OpError, match="unsaved changes"):
        ops.new_workspace()
    target = tmp_path / "saved.fdw"
    # sample paths are relative to the workspace file, so save next to the data for a faithful reopen
    saved = ops.save_workspace(str(target))
    assert saved["path"] == str(target) and not ops.ws.modified
    reopened = ops.open_workspace(str(target))
    assert reopened["samples"] == 6
    assert ops.sample("Donor1_Control").root.find(CD3 + "/Marker") is not None
    with pytest.raises(OpError, match="no such file"):
        ops.open_workspace(str(tmp_path / "missing.fdw"))


def test_add_and_remove_samples_in_a_new_workspace(ops, testset):
    ops.new_workspace()
    assert ops.ws.samples == []
    added = ops.add_samples(folder=str(testset / "samples"))
    assert len(added["added"]) == 6
    ops.remove_samples("Donor3_Control")
    assert len(ops.ws.samples) == 5
    with pytest.raises(OpError, match="give paths="):
        ops.add_samples()
    with pytest.raises(OpError, match="no such file"):
        ops.add_samples(paths=[str(testset / "nope.fcs")])


# -------------------------------------------------------------------------- analysis
def test_compensation_wizard_recovers_the_true_spillover(ops, testset):
    ops.new_workspace()
    ops.add_samples(folder=str(testset / "compensation controls"))
    r = ops.compensation("compute", method="median")
    m = r["matrix"]
    assert set(m["parameters"]) >= {"FITC-A", "PE-A", "APC-A", "PE-Cy7-A"} and all(c["status"] in ("ok", "weak") for c in r["controls"])
    i, j = m["parameters"].index("FITC-A"), m["parameters"].index("PE-A")
    assert m["spillover_percent"][i][j] == pytest.approx(14.0, abs=3.0)  # the generated FITC -> PE spill
    k = m["parameters"].index("PE-Cy7-A")
    assert m["spillover_percent"][j][k] == pytest.approx(22.0, abs=4.0)
    assert not ops.ws.matrices  # compute changes nothing
    applied = ops.compensation("apply", name="From controls", method="median")
    assert "From controls" in ops.ws.matrices and applied["applied_to"]
    assert ops.compensation("list")["workspace_matrices"][0]["name"] == "From controls"
    with pytest.raises(OpError, match="exists"):
        ops.compensation("apply", name="From controls")


def test_dimension_reduction_and_clustering_create_parameters_and_populations(ops):
    r = ops.dimension_reduction("pca", population=CD3, samples=["Donor1_Stimulated"], parameters=["CD3", "CD4", "CD8", "CD69"], max_events=0)
    assert len(r["created_parameters"]) == 2 and r["created_parameters"][0].startswith("PC")
    s = ops.sample("Donor1_Stimulated")
    assert all(n in s.parameter_names() for n in r["created_parameters"])
    c = ops.cluster("kmeans", population=CD3, samples=["Donor1_Stimulated"], parameters=["CD4", "CD8", "CD69"], clusters=3)
    assert len(c["populations"]) == 3
    parent = s.root.find(CD3)
    assert sum(s.root.find(CD3 + "/" + n).count for n in c["populations"]) == parent.count
    with pytest.raises(OpError, match="method must be"):
        ops.dimension_reduction("spiral")
    with pytest.raises(OpError, match="at least two parameters"):
        ops.cluster("kmeans", population=CD3, samples=["Donor1_Stimulated"], parameters=["CD4"])


def test_cell_cycle_and_proliferation_on_the_extras(ops, testset):
    ops.new_workspace()
    ops.add_samples(folder=str(testset / "extras"))
    cc = ops.cell_cycle("PI-A", samples="DNA content", model="watson", debris=True, aggregates=True)["rows"][0]
    assert "error" not in cc and 50 < cc["percent_g1"] < 70 and 8 < cc["percent_s"] < 25
    pr = ops.proliferation("CTV-A", samples="CellTrace")["rows"][0]
    assert "error" not in pr and pr["percent_divided"] > 60


def test_data_cleaning_flags_the_flow_rate_dip_and_can_create_the_population(ops):
    r = ops.data_cleaning(samples=["Donor1_Stimulated", "Donor3_Stimulated"], create_population=True)
    rows = {x["sample"]: x for x in r["rows"]}
    removed = {k: v["percent_removed"]["total"] for k, v in rows.items()}
    assert removed["Donor3_Stimulated.fcs"] > removed["Donor1_Stimulated.fcs"] + 3  # the generated dip is found
    assert ops.sample("Donor3_Stimulated").root.find("/QC pass") is not None
    assert rows["Donor3_Stimulated.fcs"]["created"] == "/QC pass"


# -------------------------------------------------------------------------- export
def test_export_events_and_statistics(ops, tmp_path):
    from flowdesk.core.fcs import FCSFile

    r = ops.export_events(str(tmp_path / "cd3.fcs"), population=CD3, samples="Donor1_Control")
    assert r["written"][0]["events"] == ops.sample("Donor1_Control").root.find(CD3).count
    assert FCSFile.read(r["written"][0]["file"]).n_events == r["written"][0]["events"]
    many = ops.export_events(str(tmp_path / "all"), population=CD3, group="Stimulated", fmt="csv", parameters=["CD4", "CD8"])
    assert len(many["written"]) == 3 and all(Path(w["file"]).suffix == ".csv" for w in many["written"])
    first = Path(many["written"][0]["file"]).read_text().splitlines()
    assert len(first) == many["written"][0]["events"] + 1 and "PE" in first[0]
    st = ops.export_statistics(str(tmp_path / "stats.csv"), populations=[CD3], statistics=["Count"])
    assert st["rows"] == 6 and (tmp_path / "stats.csv").read_text().splitlines()[0].startswith("sample,")
    wsp = ops.export_flowjo_workspace(str(tmp_path / "out"))
    assert wsp["path"].endswith(".wsp") and Path(wsp["path"]).stat().st_size > 1000


# ---------------------------------------------------------------------- control ops
def test_python_escape_hatch_is_undoable_and_reports_errors(ops):
    r = ops.python("print(len(workspace.samples))\nresult = {'n': len(workspace.samples)}")
    assert r["ok"] and r["stdout"].strip() == "6" and r["result"] == {"n": 6}
    assert not ops.c.can_undo  # reading changes nothing
    ops.python("workspace.samples[0].keywords['Tag'] = 'x'", undoable=True, undo_label="tag it")
    assert ops.c.undo_label() == "tag it" and ops.ws.modified
    bad = ops.python("1/0")
    assert bad["ok"] is False and "ZeroDivisionError" in bad["traceback"]


def test_events_and_log_record_what_the_agent_did(ops):
    ops.create_gate("E", "range", x="CD4", x_range=[4000, None], sample="Donor1_Control", parent=CD3)
    ops.undo()
    ev = ops.events(0)
    kinds = [e["kind"] for e in ev["events"]]
    assert kinds == ["create_gate", "undo"] and ev["seq"] == 2
    assert ops.events(1)["events"][0]["kind"] == "undo"
    assert any("create_gate" in line for line in ops.log(10)["lines"])


def test_dispatcher_rejects_unknown_operations(ops):
    assert "create_gate" in ops.names() and "python" in ops.names()
    with pytest.raises(OpError, match="unknown operation 'launch_missiles'"):
        ops.call("launch_missiles")
    assert ops.call("ping")["samples"] == 6


def test_set_scale_changes_the_axis_for_the_raw_and_compensated_parameter(ops):
    r = ops.set_scale("CD4", "linear", samples="Donor1_Control", minimum=-1000, maximum=60000)
    s = ops.sample("Donor1_Control")
    assert r["parameter"] == "Comp-PE-A" and s.scale("Comp-PE-A").kind == "linear" and s.scale("PE-A").kind == "linear"
    assert s.scale("PE-A").inverse(1.0) == pytest.approx(60000)
    with pytest.raises(OpError, match="kind must be"):
        ops.set_scale("CD4", "wavy")
