import json
import shutil

import numpy as np
import pytest

from flowdesk.core.compensation import ACQUISITION_DEFINED, SpilloverMatrix
from flowdesk.core.gates import BooleanGate, EllipseGate, PolygonGate, RectangleGate, make_quadrant_gates
from flowdesk.core.layouts import Layout, LayoutItem, TableColumn, TableDef
from flowdesk.core.model import (
    Criterion,
    GraphSettings,
    Group,
    Population,
    Sample,
    StatNode,
    Workspace,
    default_compensation_group,
)
from flowdesk.core.serialization import (
    copy_population,
    gating_snapshot,
    load_workspace,
    restore_snapshot,
    save_workspace,
    workspace_to_dict,
)
from flowdesk.core.transforms import LinearScale, LogicleScale, LogScale, default_biex
from tests.conftest import SYNTH_NAMES, SYNTH_STAINS, make_synthetic_events, write_fcs

S2 = np.array([[1.0, 0.2], [0.05, 1.0]])


def make_file(directory, name, seed=1, n=2000, extra_keywords=None):
    ev = make_synthetic_events(n=n, seed=seed)["events"]
    return write_fcs(directory / name, ev, SYNTH_NAMES, SYNTH_STAINS, extra_keywords=extra_keywords)


@pytest.fixture
def sample(synthetic_fcs):
    s = Sample(synthetic_fcs["path"], 1)
    assert s.load_header()
    return s


# ======================================================================= Workspace
def test_add_sample_dedupe(tmp_path):
    p = make_file(tmp_path, "a.fcs")
    q = make_file(tmp_path, "b.fcs")
    ws = Workspace()
    s1 = ws.add_sample(p)
    assert ws.add_sample(p) is s1
    assert ws.add_sample(tmp_path / "." / "a.fcs") is s1
    s2 = ws.add_sample(q)
    assert len(ws.samples) == 2
    assert (s1.id, s2.id) == (1, 2)
    assert s1.keywords["$TOT"] == "2000" and not s1.loaded
    assert s1.n_events == 2000  # from $TOT, before loading data
    assert ws.sample_by_id(2) is s2 and ws.sample_by_id(99) is None
    assert ws.modified
    # explicit ids are kept and skipped by the counter
    s3 = ws.add_sample(make_file(tmp_path, "c.fcs"), sample_id=3)
    assert s3.id == 3
    assert ws.add_sample(make_file(tmp_path, "d.fcs")).id == 4


def test_remove_sample_and_groups(tmp_path):
    ws = Workspace()
    s = ws.add_sample(make_file(tmp_path, "a.fcs"))
    g = ws.add_group("Test", samples=[s])
    assert g.contains(s)
    ws.remove_sample(s)
    assert s.id not in g.sample_ids and ws.samples == []
    assert ws.add_group("Test").name == "Test 2"
    assert ws.add_group("Test").name == "Test 3"
    all_group = ws.group_by_name("All Samples")
    ws.remove_group(all_group)
    assert all_group in ws.groups  # cannot remove "All Samples"
    ws.remove_group(g)
    assert ws.group_by_name("Test") is None
    assert [gr.name for gr in Workspace().groups] == ["All Samples", "Compensation"]


def test_compensation_group_criteria(tmp_path):
    ws = Workspace()
    comp = ws.group_by_name("Compensation")
    assert comp.role == "compensation"
    names = {
        "Compensation Controls_FITC Stained Control.fcs": True,
        "UNSTAINED.fcs": True,
        "unstained cells.fcs": True,
        "COMP_PE.fcs": True,
        "Specimen_001_Tube_003.fcs": False,
        "donor6.fcs": False,
    }
    samples = {n: ws.add_sample(make_file(tmp_path, n)) for n in names}
    for n, expected in names.items():
        assert comp.contains(samples[n]) is expected, n
    assert {s.name for s in ws.members(comp)} == {n for n, e in names.items() if e}
    assert len(ws.members(ws.group_by_name("All Samples"))) == len(names)
    # the $FIL keyword wins over the file name when present
    s_kw = ws.add_sample(make_file(tmp_path, "tube7.fcs", extra_keywords={"$FIL": "Comp tube 7"}))
    s_kw2 = ws.add_sample(make_file(tmp_path, "comp_but_not.fcs", extra_keywords={"$FIL": "specimen"}))
    assert comp.contains(s_kw)
    assert not comp.contains(s_kw2)


def test_criterion_functions():
    kw = {"$FIL": "Donor6_CD4.fcs", "TUBE NAME": "CD4 panel"}
    assert Criterion("$FIL", "Contains", "cd4").matches(kw, "x")
    assert Criterion("$FIL", "Does not contain", "cd8").matches(kw, "x")
    assert Criterion("TUBE NAME", "Equals", " cd4 PANEL ").matches(kw, "x")
    assert Criterion("TUBE NAME", "Not equals", "cd8 panel").matches(kw, "x")
    assert Criterion("$FIL", "Starts with", "donor").matches(kw, "x")
    assert Criterion("$FIL", "Ends with", ".FCS").matches(kw, "x")
    assert not Criterion("$FIL", "Ends with", "cd4").matches(kw, "x")
    assert Criterion("$FIL", "whatever", "donor").matches(kw, "x")  # unknown -> contains
    assert not Criterion("$SRC", "Contains", "a").matches(kw, "x")  # missing keyword -> ""
    assert Criterion("FILENAME", "Contains", "abc").matches({}, "ABC.fcs")
    c = Criterion("$FIL", "Starts with", "d", "Or")
    assert Criterion.from_dict(c.to_dict()) == c


def test_group_criteria_connectors(sample):
    sample.keywords["$FIL"] = "donor6 tcell.fcs"
    g_and = Group("g", criteria=[Criterion("$FIL", "Contains", "donor"), Criterion("$FIL", "Contains", "bcell", "And")])
    g_or = Group("g", criteria=[Criterion("$FIL", "Contains", "donor"), Criterion("$FIL", "Contains", "bcell", "Or")])
    assert not g_and.matches_criteria(sample)
    assert g_or.matches_criteria(sample)
    assert not Group("empty").matches_criteria(sample)
    g_or.live = False
    assert not g_or.contains(sample)
    g_or.sample_ids.add(sample.id)
    assert g_or.contains(sample)
    assert Group("all", role="all").contains(sample)
    comp = default_compensation_group()
    assert [c.connector for c in comp.criteria] == ["And", "Or"]


# ================================================================ group gating
def _donor_ws(tmp_path):
    ws = Workspace()
    g = ws.add_group("Donors", criteria=[Criterion("$FIL", "Contains", "donor")])
    d1 = ws.add_sample(make_file(tmp_path, "donor1.fcs", seed=1))
    d2 = ws.add_sample(make_file(tmp_path, "donor2.fcs", seed=2))
    other = ws.add_sample(make_file(tmp_path, "specimen.fcs", seed=3))
    return ws, g, d1, d2, other


def test_apply_population_to_group(tmp_path):
    ws, g, d1, d2, other = _donor_ws(tmp_path)
    cells = Population("Cells", RectangleGate(["FSC-A"], [50000], [None]))
    cells.stats.append(StatNode("Median", "FITC-A"))
    assert ws.apply_population_to_group(g, "", cells) == 2
    for s in (d1, d2):
        pop = s.find_population("/Cells")
        assert pop is not None and pop.sample is s
        assert pop.gate is not cells.gate and pop.gate.id != cells.gate.id
        assert pop.gate.lo == [50000]
        assert pop.stats[0].statistic == "Median"
    assert other.find_population("/Cells") is None
    assert g.root.find("/Cells") is not None
    # nested population under an existing parent
    fitc = Population("FITC+", RectangleGate(["FITC-A"], [5000], [None]))
    assert ws.apply_population_to_group(g, "/Cells", fitc) == 2
    assert d1.find_population("/Cells/FITC+").path == "/Cells/FITC+"
    # independent masks per sample
    assert d1.find_population("/Cells/FITC+").count != d2.find_population("/Cells/FITC+").count
    # replacing Cells keeps existing sub-populations and the position
    d1.root.add_child(Population("Other"))
    cells2 = Population("Cells", RectangleGate(["FSC-A"], [60000], [None]))
    ws.apply_population_to_group(g, "", cells2)
    new_cells = d1.find_population("/Cells")
    assert new_cells.gate.lo == [60000]
    assert new_cells.child_by_name("FITC+") is not None
    assert [c.name for c in d1.root.children] == ["Cells", "Other"]
    assert d1.find_population("/Cells/FITC+").mask().sum() <= new_cells.count
    # the template got replaced, not duplicated
    assert [c.name for c in g.root.children] == ["Cells"]
    # a missing parent path in a sample is skipped but created in the template
    assert ws.apply_population_to_group(g, "/Nope", Population("X")) == 0
    assert g.root.find("/Nope/X") is not None


def test_group_template_applies_to_new_samples(tmp_path):
    ws, g, d1, d2, other = _donor_ws(tmp_path)
    ws.apply_population_to_group(g, "", Population("Cells", RectangleGate(["FSC-A"], [50000], [None])))
    ws.apply_population_to_group(g, "/Cells", Population("FITC+", RectangleGate(["FITC-A"], [5000], [None])))
    d3 = ws.add_sample(make_file(tmp_path, "Donor3.fcs", seed=4))
    assert d3.find_population("/Cells/FITC+") is not None
    assert d3.find_population("/Cells").gate.lo == [50000]
    assert d3.find_population("/Cells/FITC+").count > 0
    assert ws.add_sample(make_file(tmp_path, "blank.fcs")).root.children == []
    # compensation group works the same way
    comp = ws.group_by_name("Compensation")
    ws.apply_population_to_group(comp, "", Population("Singlets", RectangleGate(["SSC-A"], [None], [100000])))
    c1 = ws.add_sample(make_file(tmp_path, "Comp_FITC.fcs"))
    assert c1.find_population("/Singlets") is not None
    assert d3.find_population("/Singlets") is None


def test_remove_and_rename_population_in_group(tmp_path):
    ws, g, d1, d2, other = _donor_ws(tmp_path)
    ws.apply_population_to_group(g, "", Population("Cells", RectangleGate(["FSC-A"], [50000], [None])))
    ws.apply_population_to_group(g, "/Cells", Population("FITC+", RectangleGate(["FITC-A"], [5000], [None])))
    other.root.add_child(Population("Cells"))
    assert ws.rename_population_in_group(g, "/Cells", "Lymphocytes") == 2
    assert d1.find_population("/Lymphocytes/FITC+") is not None
    assert g.root.find("/Lymphocytes") is not None
    assert other.find_population("/Cells") is not None  # not a member
    assert ws.remove_population_from_group(g, "/Lymphocytes/FITC+") == 2
    assert d2.find_population("/Lymphocytes/FITC+") is None
    assert g.root.find("/Lymphocytes/FITC+") is None
    assert ws.remove_population_from_group(g, "/Missing") == 0


def test_population_paths_union_order(tmp_path):
    ws = Workspace()
    s1 = ws.add_sample(make_file(tmp_path, "a.fcs"))
    s2 = ws.add_sample(make_file(tmp_path, "b.fcs"))
    a = s1.root.add_child(Population("A"))
    a.add_child(Population("B"))
    s2.root.add_child(Population("C"))
    s2.root.add_child(Population("A"))
    assert ws.population_paths() == ["/A", "/A/B", "/C"]
    g = ws.add_group("only b", samples=[s2])
    assert ws.population_paths(g) == ["/C", "/A"]
    s1.add_derived("ratio", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
    names = ws.all_parameter_names()
    assert names[: len(SYNTH_NAMES)] == SYNTH_NAMES and names[-1] == "ratio"
    assert ws.all_parameter_names(g) == SYNTH_NAMES


# ================================================================== Population
def test_population_tree_ops():
    root = Population("root")
    a = root.add_child(Population("A", RectangleGate(["X"], [0], [1])))
    b = a.add_child(Population("B"))
    c = a.add_child(Population("C"), index=0)
    assert [x.name for x in a.children] == ["C", "B"]
    assert root.path == "" and a.path == "/A" and b.path == "/A/B"
    assert root.display_name == "Ungated" and a.display_name == "A"
    assert b.depth == 2 and b.root is root and b.ancestors() == [root, a]
    assert root.find("/A/B") is b and root.find("A/C") is c and root.find("") is root
    assert root.find("/A/X") is None
    assert [p.name for p in root.walk()] == ["root", "A", "C", "B"]
    assert [p.name for p in root.walk(include_self=False)] == ["A", "C", "B"]
    a.remove_child(c)
    assert c.parent is None and a.children == [b]
    assert repr(b) == "<Population /A/B>" and repr(root) == "<Population />"


def test_population_clone():
    a = Population("A", RectangleGate(["X"], [0], [1]))
    a.stats.append(StatNode("Mean", "X"))
    a.graph = GraphSettings("X", "Y", "contour", {"levels": 5})
    a.expanded = False
    a.add_child(Population("B", PolygonGate(["X", "Y"], [(0, 0), (1, 0), (1, 1)])))
    deep = a.clone()
    assert deep.id != a.id and deep.gate.id != a.gate.id
    assert deep.gate.lo == [0] and deep.graph.plot_type == "contour" and deep.graph.options == {"levels": 5}
    assert deep.graph is not a.graph
    assert deep.stats[0].statistic == "Mean" and deep.stats[0] is not a.stats[0]
    assert deep.expanded is False
    assert deep.find("/B") is not None and deep.find("/B").gate is not a.children[0].gate
    assert a.clone(deep=False).children == []
    cp = copy_population(a)
    assert cp.id == a.id and cp.find("/B").gate.id == a.children[0].gate.id


def test_mask_caching_and_invalidate(sample, synthetic_fcs):
    ev = synthetic_fcs["events"]
    gate = RectangleGate(["FSC-A"], [50000], [None])
    cells = sample.root.add_child(Population("Cells", gate))
    fitc = cells.add_child(Population("FITC+", RectangleGate(["FITC-A"], [5000], [None])))
    ungated_child = cells.add_child(Population("Copy"))
    m = cells.mask()
    assert m is cells.mask()  # cached
    np.testing.assert_array_equal(m, ev[:, 0] >= 50000)
    assert ungated_child.count == cells.count
    assert fitc.count == int(((ev[:, 0] >= 50000) & (ev[:, 2] >= 5000)).sum())
    assert sample.root.count == ev.shape[0]
    gate.lo = [80000]
    assert cells.mask() is m  # stale until invalidated
    cells.invalidate()
    assert cells._mask is None and fitc._mask is None
    assert cells.count == int((ev[:, 0] >= 80000).sum())
    assert fitc.count == int(((ev[:, 0] >= 80000) & (ev[:, 2] >= 5000)).sum())
    # a gate on an unknown parameter yields an empty population, not an error
    bad = sample.root.add_child(Population("Bad", RectangleGate(["APC-A"], [0], [None])))
    assert bad.count == 0
    # detached population
    assert Population("lonely").mask().size == 0 and Population("lonely").count == 0


def test_ellipse_and_negated_gates_in_sample(sample, synthetic_fcs):
    """Regression: EllipseGate used to shadow Gate.membership with a static method,
    so Population.mask() raised TypeError for ellipse gates."""
    ev = synthetic_fcs["events"].astype(np.float64)
    for n in ("FSC-A", "SSC-A"):
        sample.set_scale(n, LinearScale(0, 262144))
    e = EllipseGate(["FSC-A", "SSC-A"], (90000, 50000), (120000, 50000), (90000, 70000))
    pop = sample.root.add_child(Population("Ell", e))
    expected = ((ev[:, 0] - 90000) / 30000) ** 2 + ((ev[:, 1] - 50000) / 20000) ** 2 <= 1
    assert pop.count == int(expected.sum())
    assert abs(pop.count - 0.8 * ev.shape[0]) < 0.05 * ev.shape[0]
    e2 = EllipseGate(["FSC-A", "SSC-A"], (90000, 50000), (120000, 50000), (90000, 70000))
    e2.negated = True
    out = sample.root.add_child(Population("Not Ell", e2))
    assert out.count == ev.shape[0] - pop.count
    np.testing.assert_array_equal(out.mask(), ~pop.mask())


def test_boolean_and_quadrant_gates_in_sample(sample, synthetic_fcs):
    ev = synthetic_fcs["events"]
    quads = make_quadrant_gates("FITC-A", "PE-A", 5000, 5000)
    pops = [sample.root.add_child(Population(n, g)) for n, g in quads]
    assert sum(p.count for p in pops) == ev.shape[0]
    q2 = pops[1]
    assert q2.count == int(((ev[:, 2] >= 5000) & (ev[:, 3] >= 5000)).sum())
    either = sample.root.add_child(Population("Either", BooleanGate("or", [pops[0].path, pops[2].path])))
    assert either.count == pops[0].count + pops[2].count
    neither = sample.root.add_child(Population("Not Q2", BooleanGate("not", [q2.path])))
    assert neither.count == ev.shape[0] - q2.count
    assert sample.resolve("/nope") is None


# ====================================================================== Sample
def test_sample_header_comp_and_parameter_names(tmp_path):
    p = make_file(tmp_path, "s.fcs", extra_keywords={"$SPILLOVER": "2,FITC-A,PE-A,1,0.2,0.05,1"})
    s = Sample(p, 1)
    assert s.load_header()
    assert s.comp is not None and s.comp.name == ACQUISITION_DEFINED
    assert s.parameter_names() == ["Comp-FITC-A", "Comp-PE-A"] + SYNTH_NAMES
    s.comp_enabled = False
    assert s.parameter_names() == SYNTH_NAMES
    s.set_compensation(SpilloverMatrix.identity(["FITC-A", "PE-A"]))
    assert s.parameter_names() == SYNTH_NAMES
    # matrix parameters missing from the sample are skipped
    s.set_compensation(SpilloverMatrix("m", ["FITC-A", "APC-A"], np.eye(2) + 0.1))
    assert s.parameter_names() == ["Comp-FITC-A"] + SYNTH_NAMES
    assert s.default_axes() == ("FSC-A", "SSC-A")
    # comp from FCS is also picked up by ensure_loaded when no header was read
    s2 = Sample(p, 2)
    assert s2.ensure_loaded() and s2.comp.params == ["FITC-A", "PE-A"]
    assert s2.keywords["$TOT"] == "2000" and s2.n_events == 2000


def test_sample_load_errors(tmp_path):
    s = Sample(tmp_path / "missing.fcs", 1)
    assert not s.load_header() and s.load_error
    assert not s.ensure_loaded()
    with pytest.raises(KeyError):
        s.data("FSC-A")
    assert s.n_events == 0 and s.default_axes() == (None, None)
    junk = tmp_path / "junk.fcs"
    junk.write_bytes(b"x" * 100)
    s2 = Sample(junk, 2)
    assert not s2.ensure_loaded() and "not an FCS" in s2.load_error


def test_sample_compensated_data(sample, synthetic_fcs):
    ev = synthetic_fcs["events"].astype(np.float64)
    comp = SpilloverMatrix("m", ["FITC-A", "PE-A"], S2)
    sample.set_compensation(comp)
    expected = ev[:, [2, 3]] @ np.linalg.inv(S2)
    np.testing.assert_allclose(sample.data("Comp-FITC-A"), expected[:, 0], rtol=1e-5, atol=0.05)
    np.testing.assert_allclose(sample.data("Comp-PE-A"), expected[:, 1], rtol=1e-5, atol=0.05)
    np.testing.assert_array_equal(sample.data("FITC-A"), synthetic_fcs["events"][:, 2])  # raw untouched
    assert sample.data("Comp-PE-A") is not None and sample._comp_cache[0] is comp
    assert sample.is_compensated_param("Comp-PE-A") and not sample.is_compensated_param("PE-A")
    assert sample.base_channel("Comp-FITC-A").name == "FITC-A"
    assert sample.base_channel("CD4").name == "PE-A"
    assert sample.base_channel("nothing") is None
    assert sample.param_label("Comp-FITC-A") == "Comp-FITC-A :: CD3"
    assert sample.param_label("FSC-A") == "FSC-A"
    assert sample.param_stain("Comp-PE-A") == "CD4" and sample.param_stain("FSC-A") == "FSC-A"
    with pytest.raises(KeyError):
        sample.data("Comp-APC-A")
    # changing the matrix drops the cache
    sample.set_compensation(SpilloverMatrix("m2", ["FITC-A", "PE-A"], np.eye(2)))
    assert sample._comp_cache is None
    sample.set_compensation(SpilloverMatrix("m3", ["FITC-A", "PE-A"], [[1, 0.5], [0, 1]]))
    np.testing.assert_allclose(sample.data("Comp-PE-A"), ev[:, 3] - 0.5 * ev[:, 2], rtol=1e-5, atol=0.05)
    # gating on a compensated parameter
    pop = sample.root.add_child(Population("CPE+", RectangleGate(["Comp-PE-A"], [5000], [None])))
    assert pop.count == int(((ev[:, 3] - 0.5 * ev[:, 2]) >= 5000).sum())


def test_scales_and_display_cache(sample, synthetic_fcs):
    ev = synthetic_fcs["events"]
    assert sample.scale("FSC-A") == LinearScale(0, 262144)
    assert sample.scale("Time") == LinearScale(0, 262144)
    assert sample.scale("FITC-A") == default_biex(262144)
    d1 = sample.display("FITC-A")
    assert d1 is sample.display("FITC-A")
    np.testing.assert_allclose(d1, default_biex(262144).forward(ev[:, 2]), atol=1e-6)
    pop = sample.root.add_child(Population("P", PolygonGate(["FITC-A", "PE-A"], [(100, 100), (1e5, 100), (1e5, 1e5), (100, 1e5)])))
    pop.count
    assert pop._mask is not None
    sample.set_scale("FITC-A", LinearScale(0, 100000))
    assert pop._mask is None  # masks invalidated
    d2 = sample.display("FITC-A")
    assert d2 is not d1
    np.testing.assert_allclose(d2, ev[:, 2] / 100000, rtol=1e-6)
    np.testing.assert_allclose(sample.to_display("FITC-A", [50000]), [0.5])
    # compensated parameters fall back to the base parameter's scale
    sample.set_compensation(SpilloverMatrix("m", ["FITC-A", "PE-A"], S2))
    assert sample.scale("Comp-FITC-A") == LinearScale(0, 100000)
    sample.set_scale("Comp-PE-A", LogScale(1, 1e5))
    assert sample.scale("Comp-PE-A") == LogScale(1, 1e5)
    assert sample.scale("unknown-param") == default_biex(262144)


def test_log_amplified_channel_default_scale(tmp_path):
    raw = np.array([[0, 100], [512, 200], [768, 300]], dtype=np.uint16)
    p = write_fcs(tmp_path / "log.fcs", raw, ["FL1-H", "FSC-H"], datatype="I", log_amplified={0: (4, 1)})
    s = Sample(p, 1)
    s.load_header()
    assert s.scale("FL1-H") == LogScale(1, 1024)
    assert s.scale("FSC-H") == LinearScale(0, 1024)


def test_derived_parameters(sample, synthetic_fcs):
    ev = synthetic_fcs["events"].astype(np.float64)
    sample.add_derived("FITC/PE", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
    sample.add_derived("sum", {"op": "sum", "a": "FITC-A", "b": "PE-A"})
    sample.add_derived("diff", {"op": "difference", "a": "FITC-A", "b": "PE-A"})
    sample.add_derived("prod", {"op": "product", "a": "FSC-A", "b": "SSC-A"})
    sample.add_derived("log", {"op": "log10", "a": "FITC-A"})
    sample.add_derived("copy", {"op": "identity", "a": "SSC-A"})
    assert sample.parameter_names()[-6:] == ["FITC/PE", "sum", "diff", "prod", "log", "copy"]
    np.testing.assert_allclose(sample.data("FITC/PE"), (ev[:, 2] / ev[:, 3]).astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(sample.data("sum"), ev[:, 2] + ev[:, 3], rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(sample.data("diff"), ev[:, 2] - ev[:, 3], rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(sample.data("prod"), ev[:, 0] * ev[:, 1], rtol=1e-5)
    log_ref = np.where(ev[:, 2] > 0, np.log10(np.maximum(ev[:, 2], 1e-30)), 0.0)
    np.testing.assert_allclose(sample.data("log"), log_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sample.data("copy"), ev[:, 1], rtol=1e-6)
    assert sample.data("FITC/PE") is sample.data("FITC/PE")  # cached
    assert sample.scale("FITC/PE") == LinearScale(0, 10)
    assert isinstance(sample.scale("sum"), LogicleScale)
    # division by zero gives 0
    sample.add_derived("zero", {"op": "ratio", "a": "FITC-A", "b": "Time"})
    assert sample.data("zero")[0] == 0.0  # Time[0] == 0
    # derived from compensated values follows the matrix
    sample.set_compensation(SpilloverMatrix("m", ["FITC-A", "PE-A"], [[1, 0.5], [0, 1]]))
    sample.add_derived("cratio", {"op": "ratio", "a": "Comp-PE-A", "b": "FSC-A", "max": 2})
    np.testing.assert_allclose(sample.data("cratio"), (ev[:, 3] - 0.5 * ev[:, 2]) / ev[:, 0], rtol=1e-4, atol=1e-6)
    assert sample.scale("cratio") == LinearScale(0, 2)
    # gating on a derived parameter
    pop = sample.root.add_child(Population("hi ratio", RectangleGate(["FITC/PE"], [1.0], [None])))
    assert pop.count == int(((ev[:, 2] / ev[:, 3]) >= 1.0).sum())


def test_graph_settings_and_statnode():
    g = GraphSettings.from_dict(None)
    assert (g.x, g.y, g.plot_type) == (None, None, "pseudocolor")
    g2 = GraphSettings("A", "B", "histogram", {"smooth": True})
    assert GraphSettings.from_dict(g2.to_dict()) == g2
    st = StatNode("Percentile", "FITC-A", 95)
    assert st.label == "Percentile 95 : FITC-A"
    assert StatNode.from_dict(st.to_dict()) == st


def test_layout_and_table_definitions():
    lay = Layout("L", page="Letter", landscape=True)
    assert lay.page_size == (792.0, 612.0)
    assert Layout(page="nonsense").page_size == (595.0, 842.0)
    col = TableColumn("/Cells/FITC+", "Median", "FITC-A")
    assert col.header == "FITC+ | Median : FITC-A"
    assert TableColumn("", "Count").auto_label == "Ungated | Count"
    assert TableColumn(keyword="$DATE").header == "$DATE"
    assert TableColumn("/A", label="Custom").header == "Custom"
    t = TableDef("T", [col, TableColumn(keyword="$DATE")], "Donors")
    assert TableDef.from_dict(t.to_dict()) == t
    item = LayoutItem(sample_id=1, pop_path="/A", graph=GraphSettings("X", "Y"), overlays=[{"sample_id": 2}])
    lay.items.append(item)
    assert Layout.from_dict(lay.to_dict()) == lay


# =============================================================== serialization
def _rich_workspace(tmp_path, data_dir):
    ws = Workspace()
    ws.settings = {"theme": "dark", "dpi": 300}
    s1 = ws.add_sample(make_file(data_dir, "donor1.fcs", seed=1))
    s2 = ws.add_sample(make_file(data_dir, "Comp_FITC.fcs", seed=2))
    s1.ensure_loaded()
    s1.set_compensation(SpilloverMatrix("mine", ["FITC-A", "PE-A"], S2))
    s1.set_scale("FITC-A", LogicleScale.from_biex(262144, 4.5, 0.5, -250))
    s1.set_scale("SSC-A", LogScale(10, 262144))
    s1.set_scale("FSC-A", LinearScale(-1000, 200000))
    s1.add_derived("ratio", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
    cells = s1.root.add_child(Population("Cells", PolygonGate(["FSC-A", "SSC-A"], [(40000, 20000), (150000, 20000), (150000, 90000), (40000, 90000)])))
    cells.stats.append(StatNode("Median", "Comp-FITC-A"))
    cells.stats.append(StatNode("Percentile", "PE-A", 90.0))
    cells.graph = GraphSettings("FSC-A", "SSC-A", "contour", {"levels": 7})
    cells.expanded = False
    ell = EllipseGate(["FITC-A", "PE-A"], (20000, 50000), (30000, 50000), (20000, 80000))
    ell.negated = True
    ell.color = "#00ff00"
    cells.add_child(Population("NotEll", ell))
    for name, g in make_quadrant_gates("Comp-FITC-A", "Comp-PE-A", 5000, 5000):
        cells.add_child(Population(name, g))
    cells.add_child(Population("Bool", BooleanGate("and", ["/Cells/NotEll", "/Cells/" + cells.children[1].name])))
    donors = ws.add_group("Donors", criteria=[Criterion("$FIL", "Contains", "donor")], color="#123456")
    donors.sample_ids.add(s2.id)
    ws.apply_population_to_group(donors, "", Population("Lymph", RectangleGate(["FSC-A"], [30000], [None])))
    ws.matrices["mine"] = SpilloverMatrix("mine", ["FITC-A", "PE-A"], S2, suffix="*")
    ws.layouts.append(Layout("Page 1", [LayoutItem(sample_id=s1.id, pop_path="/Cells", graph=GraphSettings("FSC-A", "SSC-A", "dot")), LayoutItem(kind="text", text="Hello")], landscape=True, batch_group="Donors"))
    ws.tables.append(TableDef("Freqs", [TableColumn("/Cells", "Freq. of Parent"), TableColumn(keyword="$TOT")], "Donors"))
    return ws, s1, s2


def test_save_load_round_trip(tmp_path):
    ws, s1, s2 = _rich_workspace(tmp_path, tmp_path)
    counts = {p.path: p.count for p in s1.populations()}
    assert counts["/Cells"] > 0
    fdw = tmp_path / "exp.fdw"
    save_workspace(ws, fdw)
    assert ws.path == fdw and not ws.modified and ws.name == "exp"
    assert json.loads(fdw.read_text())["format"] == "flowdesk-workspace"
    ws2 = load_workspace(fdw)
    assert ws2.path == fdw and not ws2.modified
    assert workspace_to_dict(ws2, tmp_path) == workspace_to_dict(ws, tmp_path)
    t1 = ws2.sample_by_id(s1.id)
    assert t1.path == s1.path and t1.name == "donor1.fcs" and not t1.loaded
    assert t1.channel_names == SYNTH_NAMES  # restored from saved keywords
    assert t1.comp == s1.comp and t1.comp.name == "mine"
    assert t1.scales["FITC-A"] == s1.scales["FITC-A"]
    assert t1.scales["FITC-A"].width_basis == pytest.approx(-250)
    assert t1.scales["SSC-A"] == LogScale(10, 262144)
    assert t1.derived == {"ratio": {"op": "ratio", "a": "FITC-A", "b": "PE-A"}}
    cells = t1.find_population("/Cells")
    assert cells.graph.options == {"levels": 7} and not cells.expanded
    assert [s.label for s in cells.stats] == ["Median : Comp-FITC-A", "Percentile 90 : PE-A"]
    assert cells.find("/NotEll").gate.negated and cells.find("/NotEll").gate.color == "#00ff00"
    assert cells.children[1].gate.quadrant == s1.find_population("/Cells").children[1].gate.quadrant
    assert {p.path: p.count for p in t1.populations()} == counts
    donors = ws2.group_by_name("Donors")
    assert donors.color == "#123456" and donors.sample_ids == {s2.id}
    assert donors.criteria[0].value == "donor"
    assert donors.root.find("/Lymph").gate.lo == [30000]
    assert [g.name for g in ws2.groups] == ["All Samples", "Compensation", "Donors"]
    assert ws2.matrices["mine"].suffix == "*"
    assert ws2.layouts[0].landscape and ws2.layouts[0].items[1].text == "Hello"
    assert ws2.layouts[0].items[0].graph.plot_type == "dot"
    assert ws2.tables[0].columns[1].keyword == "$TOT"
    assert ws2.settings == {"theme": "dark", "dpi": 300}
    # group template keeps working after load
    s3 = ws2.add_sample(make_file(tmp_path, "donor9.fcs", seed=9))
    assert s3.find_population("/Lymph") is not None
    assert s3.id not in (s1.id, s2.id)


def test_load_relative_paths(tmp_path):
    proj = tmp_path / "project"
    data = proj / "data"
    data.mkdir(parents=True)
    ws, s1, s2 = _rich_workspace(tmp_path, data)
    save_workspace(ws, proj / "exp.fdw")
    saved = json.loads((proj / "exp.fdw").read_text())
    assert saved["samples"][0]["rel_path"] == "data/donor1.fcs"
    moved = tmp_path / "moved"
    shutil.move(str(proj), str(moved))
    assert not s1.path.exists()
    ws2 = load_workspace(moved / "exp.fdw")
    t1 = ws2.sample_by_id(s1.id)
    assert t1.path.resolve() == (moved / "data" / "donor1.fcs").resolve()
    assert t1.ensure_loaded() and t1.n_events == 2000
    assert t1.find_population("/Cells").count > 0
    # FCS file moved next to the .fdw (different relative path): found by name
    shutil.move(str(moved / "data" / "Comp_FITC.fcs"), str(moved / "Comp_FITC.fcs"))
    ws3 = load_workspace(moved / "exp.fdw")
    t2 = ws3.sample_by_id(s2.id)
    assert t2.path == moved / "Comp_FITC.fcs"
    assert t2.ensure_loaded()


def test_load_rejects_other_json(tmp_path):
    p = tmp_path / "x.fdw"
    p.write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError):
        load_workspace(p)


def test_load_sample_without_keywords_reads_header(tmp_path):
    ws = Workspace()
    ws.add_sample(make_file(tmp_path, "a.fcs"))
    fdw = tmp_path / "w.fdw"
    save_workspace(ws, fdw)
    d = json.loads(fdw.read_text())
    d["samples"][0]["keywords"] = {}
    d["groups"] = [g for g in d["groups"] if g["role"] != "all"]
    fdw.write_text(json.dumps(d))
    ws2 = load_workspace(fdw)
    assert ws2.samples[0].channel_names == SYNTH_NAMES
    assert ws2.groups[0].role == "all"  # re-inserted


def test_gating_snapshot_restore(tmp_path):
    ws, s1, s2 = _rich_workspace(tmp_path, tmp_path)
    before = workspace_to_dict(ws, tmp_path)
    cells_count = s1.find_population("/Cells").count
    snap = gating_snapshot(ws)
    json.dumps({"groups": snap["groups"]})  # data-free and serialisable
    # make a series of edits
    s1.root.add_child(Population("New", RectangleGate(["FSC-A"], [0], [None])))
    s1.find_population("/Cells").remove_child(s1.find_population("/Cells/NotEll"))
    s1.set_scale("FITC-A", LinearScale(0, 1000))
    s1.set_compensation(None)
    s2.root.children.clear()
    ws.apply_population_to_group(ws.group_by_name("Donors"), "", Population("Extra"))
    assert s1.find_population("/New") is not None
    restore_snapshot(ws, snap)
    assert ws.modified
    assert workspace_to_dict(ws, tmp_path) == before
    r1 = ws.sample_by_id(s1.id)
    assert r1.find_population("/New") is None
    assert r1.find_population("/Cells/NotEll") is not None
    assert r1.find_population("/Cells").sample is r1
    assert r1.find_population("/Cells").count == cells_count
    assert r1.comp is not None and r1.comp.name == "mine"
    assert r1.scales["FITC-A"].width_basis == pytest.approx(-250)
    assert ws.group_by_name("Donors").root.find("/Extra") is None
    assert ws.sample_by_id(s2.id).find_population("/Lymph") is not None


def test_restore_snapshot_refreshes_derived_on_compensated_params(sample, synthetic_fcs):
    """Regression: restore_snapshot reset the compensation cache but not the
    derived-parameter cache, so derived values computed from Comp- parameters
    kept the pre-undo compensation."""
    ev = synthetic_fcs["events"].astype(np.float64)
    ws = Workspace()
    ws.samples.append(sample)
    sample.set_compensation(SpilloverMatrix("a", ["FITC-A", "PE-A"], [[1, 0.5], [0, 1]]))
    sample.add_derived("cdiff", {"op": "difference", "a": "Comp-PE-A", "b": "FITC-A"})
    original = ev[:, 3] - 0.5 * ev[:, 2] - ev[:, 2]
    np.testing.assert_allclose(sample.data("cdiff"), original, rtol=1e-4, atol=0.1)
    snap = gating_snapshot(ws)
    sample.set_compensation(SpilloverMatrix("b", ["FITC-A", "PE-A"], np.eye(2) + [[0, 0.1], [0, 0]]))
    assert not np.allclose(sample.data("cdiff"), original, rtol=1e-4, atol=0.1)
    restore_snapshot(ws, snap)
    np.testing.assert_allclose(sample.data("cdiff"), original, rtol=1e-4, atol=0.1)
