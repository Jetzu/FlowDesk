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
    # header only: the time axis falls back to $PnR (not cached) ...
    assert sample.scale("Time") == LinearScale(0, 262144) and "Time" not in sample.scales
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
    # one transform per raw parameter: the raw axis follows the Comp- edit
    assert sample.scale("PE-A") == LogScale(1, 1e5) and "Comp-PE-A" not in dict.keys(sample.scales)
    assert sample.scale("unknown-param") == default_biex(262144)
    # ... and once the events are loaded it covers the acquired span (data 0..1000)
    assert sample.scale("Time") == LinearScale(0, 1020)


def test_log_amplified_channel_default_scale(tmp_path):
    raw = np.array([[0, 100], [512, 200], [768, 300]], dtype=np.uint16)
    p = write_fcs(tmp_path / "log.fcs", raw, ["FL1-H", "FSC-H"], datatype="I", log_amplified={0: (4, 1)})
    s = Sample(p, 1)
    s.load_header()
    # $PnE=4,1 on a 1024-channel ADC decodes to 1 .. 10^4: the axis covers it
    assert s.scale("FL1-H") == LogScale(1, 10000)
    assert s.scale("FSC-H") == LinearScale(0, 1024)
    assert s.data("FL1-H").max() <= 10000 and s.data("FL1-H").min() >= 1


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


def test_identity_matrix_comp_names_alias_raw_channels(tmp_path):
    """FlowJo writes identity matrices for uncompensated samples and still
    gates on "Comp-" names: those must read the raw channel, not count 0."""
    from flowdesk.core.fcs import Channel, write_fcs

    rng = np.random.default_rng(5)
    events = np.column_stack([rng.normal(1000, 100, 500), rng.normal(2000, 100, 500)]).astype(np.float32)
    path = tmp_path / "ident.fcs"
    write_fcs(path, events, [Channel(1, "FITC-A", "", 262144), Channel(2, "PE-A", "", 262144)])
    ws = Workspace()
    s = ws.add_sample(path)
    s.ensure_loaded()
    ident = SpilloverMatrix("acq", ["FITC-A", "PE-A"], np.eye(2))
    s.set_compensation(ident, True)
    assert "Comp-FITC-A" not in s.parameter_names()  # no clutter for identity matrices
    assert np.array_equal(s.data("Comp-FITC-A"), s.data("FITC-A"))
    pop = Population("bright", RectangleGate(["Comp-FITC-A"], [1000.0], [None]))
    s.root.add_child(pop)
    assert pop.count == int((events[:, 0] >= 1000).sum())
    s.set_compensation(ident, False)  # disabled matrix: same alias
    assert np.array_equal(s.data("Comp-PE-A"), s.data("PE-A"))


# ======================================================= computed parameters on disk
def _computed_workspace(tmp_path, n=500):
    """One sample with three computed parameters ("tSNE 1" and "tSNE_1" sanitise
    to the same file name) and a gate on "tSNE 1", saved as exp.fdw."""
    ws = Workspace()
    s = ws.add_sample(make_file(tmp_path, "a.fcs", seed=3, n=n))
    s.ensure_loaded()
    rng = np.random.default_rng(11)
    values = {name: rng.random(n).astype(np.float32) for name in ("tSNE 1", "tSNE_1", "CON")}
    for name, v in values.items():
        s.add_computed(name, v, {"tool": "test"})
    s.root.add_child(Population("hi", RectangleGate(["tSNE 1"], [0.5], [None])))
    fdw = tmp_path / "exp.fdw"
    save_workspace(ws, fdw)
    return ws, s, values, fdw


def test_computed_parameters_round_trip_as_distinct_arrays(tmp_path):
    ws, s, values, fdw = _computed_workspace(tmp_path)
    ws2 = load_workspace(fdw)
    t = ws2.samples[0]
    assert ws2.load_warnings == [] and t.load_warnings == []
    assert list(t.computed) == ["tSNE 1", "tSNE_1", "CON"]
    for name, v in values.items():
        assert np.array_equal(t.computed[name], v)
    assert not np.array_equal(t.computed["tSNE 1"], t.computed["tSNE_1"])
    assert t.root.find("/hi").count == int((values["tSNE 1"] >= 0.5).sum())


@pytest.mark.parametrize("new_n", [300, 800])
def test_replaced_fcs_file_drops_stale_computed_arrays(tmp_path, new_n):
    """Findings 6/7: the FCS file next to the workspace was replaced by one with
    a different event count; the saved arrays cannot belong to it."""
    ws, s, values, fdw = _computed_workspace(tmp_path)
    make_file(tmp_path, "a.fcs", seed=4, n=new_n)
    ws2 = load_workspace(fdw)
    t = ws2.samples[0]
    t.ensure_loaded()
    assert t.n_events == new_n
    assert t.computed == {} and t.computed_info == {}
    assert not {"tSNE 1", "tSNE_1", "CON"} & set(t.parameter_names())
    assert t.root.find("/hi").count == 0  # empty like a gate on a missing parameter, no ValueError
    assert t.root.find("/hi").mask().shape == (new_n,)
    assert len(t.load_warnings) == 3 and all("500" in w and str(new_n) in w for w in t.load_warnings)
    assert any("'tSNE 1'" in w for w in ws2.load_warnings) and all(w.startswith("a.fcs: ") for w in ws2.load_warnings)


def test_ensure_loaded_drops_computed_arrays_that_do_not_fit_the_file(tmp_path):
    """The workspace was opened while the FCS file was away (arrays kept, nothing
    to compare with); the file that turns up later has other events."""
    ws, s, values, fdw = _computed_workspace(tmp_path)
    (tmp_path / "a.fcs").rename(tmp_path / "away.fcs")
    ws2 = load_workspace(fdw)
    t = ws2.samples[0]
    assert list(t.computed) == ["tSNE 1", "tSNE_1", "CON"] and t.load_warnings == []
    make_file(tmp_path, "a.fcs", seed=5, n=650)
    assert t.ensure_loaded()
    assert t.computed == {} and len(t.load_warnings) == 3
    assert "'tSNE 1' was dropped" in t.load_warnings[0] and "650" in t.load_warnings[0]
    assert t.root.find("/hi").count == 0
    # an array that does not fit the events never reaches a gate either
    t.computed["late"] = np.zeros(10, dtype=np.float32)
    late = t.root.add_child(Population("late", RectangleGate(["late"], [None], [1.0])))
    with pytest.raises(KeyError):
        t.data("late")
    assert late.count == 0 and late.mask().shape == (650,)


def test_population_mask_treats_mismatched_computed_values_as_missing(sample):
    sample.ensure_loaded()
    n = sample.n_events
    sample.add_computed("c", np.arange(n, dtype=np.float32))
    pop = sample.root.add_child(Population("c-high", RectangleGate(["c"], [n / 2], [None])))
    assert pop.count == n - int(np.ceil(n / 2))
    # a stale array swapped in behind the model's back (as the loader used to)
    sample.computed["c"] = np.arange(n + 7, dtype=np.float32)
    pop.invalidate()
    assert pop.count == 0
    # other ValueErrors are bugs and still surface
    class Broken(RectangleGate):
        def evaluate(self, acc):
            raise ValueError("boom")

    bad = sample.root.add_child(Population("bad", Broken(["FSC-A"], [0], [None])))
    with pytest.raises(ValueError, match="boom"):
        bad.mask()


def test_truncated_fcs_file_keeps_its_computed_arrays(tmp_path):
    """$TOT promises more events than the DATA segment holds: the reader keeps
    the events present, and the arrays computed for them must survive a reload
    (the old check against the saved $TOT dropped them)."""
    ev = make_synthetic_events(n=400, seed=6)["events"]
    path = write_fcs(tmp_path / "t.fcs", ev, SYNTH_NAMES, SYNTH_STAINS, extra_keywords={"$TOT": "500"})
    ws = Workspace()
    s = ws.add_sample(path)
    s.ensure_loaded()
    assert s.n_events == 400 and s.keywords["$TOT"] == "500"
    v = np.linspace(0, 1, 400, dtype=np.float32)
    s.add_computed("UMAP 1", v)
    save_workspace(ws, tmp_path / "t.fdw")
    ws2 = load_workspace(tmp_path / "t.fdw")
    t = ws2.samples[0]
    t.ensure_loaded()
    assert np.array_equal(t.computed["UMAP 1"], v) and t.load_warnings == [] and ws2.load_warnings == []


def test_missing_or_unreadable_computed_data_is_reported(tmp_path):
    """Finding 16: the .fdw was renamed / copied without its .fdw.data folder."""
    ws, s, values, fdw = _computed_workspace(tmp_path)
    renamed = tmp_path / "exp final.fdw"
    shutil.copy(fdw, renamed)
    ws2 = load_workspace(renamed)
    t = ws2.samples[0]
    assert t.computed == {} and t.load_error is None
    assert len(ws2.load_warnings) == 3
    for name, w in zip(("tSNE 1", "tSNE_1", "CON"), ws2.load_warnings):
        assert w.startswith("a.fcs: ") and repr(name) in w and "exp final.fdw.data" in w and "missing" in w
    assert t.load_warnings == [w[len("a.fcs: "):] for w in ws2.load_warnings]
    # one file deleted, one corrupted: only those two are reported
    data_dir = tmp_path / "exp.fdw.data"
    files = json.loads(fdw.read_text())["samples"][0]["computed"]
    (data_dir / files["tSNE_1"]["file"]).unlink()
    (data_dir / files["CON"]["file"]).write_bytes(b"not an npy file")
    ws3 = load_workspace(fdw)
    t3 = ws3.samples[0]
    assert list(t3.computed) == ["tSNE 1"] and np.array_equal(t3.computed["tSNE 1"], values["tSNE 1"])
    assert len(ws3.load_warnings) == 2
    assert "'tSNE_1'" in ws3.load_warnings[0] and "missing" in ws3.load_warnings[0]
    assert "'CON'" in ws3.load_warnings[1] and "unreadable" in ws3.load_warnings[1]
    # warnings are load-time state, never written to the file
    save_workspace(ws3, tmp_path / "again.fdw")
    assert "load_warnings" not in (tmp_path / "again.fdw").read_text()


def test_save_removes_data_files_nothing_references(tmp_path, monkeypatch):
    """Finding 35: arrays of removed computed parameters do not linger."""
    import flowdesk.core.serialization as ser

    ws, s, values, fdw = _computed_workspace(tmp_path)
    data_dir = tmp_path / "exp.fdw.data"
    names = lambda: sorted(p.name for p in data_dir.iterdir())  # noqa: E731
    assert names() == ["sample1_0_tSNE_1.npy", "sample1_1_tSNE_1.npy", "sample1_2_CON.npy"]
    (data_dir / "notes.txt").write_text("mine")  # not ours: never touched
    s.remove_computed("CON")
    save_workspace(ws, fdw)
    assert names() == ["notes.txt", "sample1_0_tSNE_1.npy", "sample1_1_tSNE_1.npy"]
    # a save that fails before the workspace file is written keeps every file
    s.remove_computed("tSNE_1")

    def failing_dump(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ser.json, "dump", failing_dump)
    with pytest.raises(OSError):
        save_workspace(ws, fdw)
    monkeypatch.undo()
    assert names() == ["notes.txt", "sample1_0_tSNE_1.npy", "sample1_1_tSNE_1.npy"]
    save_workspace(ws, fdw)
    assert names() == ["notes.txt", "sample1_0_tSNE_1.npy"]
    ws2 = load_workspace(fdw)
    assert list(ws2.samples[0].computed) == ["tSNE 1"] and ws2.load_warnings == []
    # once nothing is left the folder goes (here after the foreign file is removed)
    (data_dir / "notes.txt").unlink()
    s.remove_computed("tSNE 1")
    save_workspace(ws, fdw)
    assert not data_dir.exists()


def test_restore_keeps_renamed_population_objects(tmp_path):
    """Finding 10: children are matched by id first, so undoing a rename brings
    back the name on the same objects (name matching only as a fallback)."""
    ws, s1, s2 = _rich_workspace(tmp_path, tmp_path)
    cells = s1.find_population("/Cells")
    kids = list(cells.children)
    lymph = ws.group_by_name("Donors").root.find("/Lymph")
    count = cells.count
    snap = gating_snapshot(ws)
    cells.name = "Renamed"
    lymph.name = "Lymph renamed"
    restore_snapshot(ws, snap)
    assert s1.root.children[0] is cells and cells.name == "Cells" and cells.parent is s1.root
    assert cells.children == kids and all(k.parent is cells for k in kids)
    assert cells.count == count and cells.sample is s1
    assert ws.group_by_name("Donors").root.children[0] is lymph and lymph.name == "Lymph"
    # a child replaced by a clone (new id, same name) is still matched by name
    snap = gating_snapshot(ws)
    clone = cells.clone(deep=True)
    s1.root.remove_child(cells)
    s1.root.add_child(clone, 0)
    restore_snapshot(ws, snap)
    assert s1.root.children[0] is clone and clone.id == cells.id
    # swapped names: ids decide which object gets which name
    a, b = clone.children[0], clone.children[1]
    a_name, b_name = a.name, b.name
    snap = gating_snapshot(ws)
    a.name, b.name = "tmp", a_name
    restore_snapshot(ws, snap)
    assert (a.name, b.name) == (a_name, b_name)


# ============================================ shared transforms, defaults, caches
def test_comp_and_raw_parameters_share_one_transform(tmp_path, synthetic_fcs):
    s = Sample(synthetic_fcs["path"], 1)
    s.load_header()
    s.set_compensation(SpilloverMatrix("m", ["FITC-A", "PE-A"], S2))
    log = LogScale(1, 1e5)
    s.set_scale("Comp-PE-A", log)
    assert s.scale("PE-A") is log and s.scale("Comp-PE-A") is log
    assert not any(k.startswith("Comp-") for k in s.scales)
    assert "Comp-PE-A" in s.scales and s.scales.get("Comp-PE-A") is log  # lookups resolve too
    lin = LinearScale(0, 5e4)
    s.set_scale("PE-A", lin)  # and the other way round
    assert s.scale("Comp-PE-A") is lin
    # direct writes (wsp import, templates) land on the raw key as well
    s.scales["Comp-FITC-A"] = LogScale(10, 1e5)
    assert list(dict.keys(s.scales)).count("FITC-A") == 1 and s.scale("FITC-A") == LogScale(10, 1e5)
    # identity / missing matrix: Comp-X still names X's transform
    s.set_compensation(None)
    assert s.scale("Comp-PE-A") is lin
    # an exported workspace only carries raw keys
    ws = Workspace()
    ws.samples.append(s)
    assert not any(k.startswith("Comp-") for k in workspace_to_dict(ws)["samples"][0]["scales"])


def test_legacy_comp_keys_collapse_on_first_lookup(tmp_path):
    p = make_file(tmp_path, "a.fcs")
    ws = Workspace()
    s = ws.add_sample(p)
    s.set_compensation(SpilloverMatrix("m", ["FITC-A", "PE-A"], S2))
    path = tmp_path / "legacy.fdw"
    save_workspace(ws, path)
    data = json.loads(path.read_text())
    # an old FlowDesk wrote a separate entry for the Comp- axis (and a default for the raw one)
    data["samples"][0]["scales"] = {"PE-A": LinearScale(0, 262144).to_dict(), "Comp-PE-A": LogScale(1, 1e4).to_dict(),
                                    "Comp-FITC-A": LogScale(3, 3e4).to_dict()}
    path.write_text(json.dumps(data))
    ws2 = load_workspace(path)
    s2 = ws2.samples[0]
    assert s2.scale("PE-A") == LogScale(1, 1e4)  # the explicit Comp- entry wins
    assert s2.scale("Comp-FITC-A") == LogScale(3, 3e4) and s2.scale("FITC-A") == LogScale(3, 3e4)
    assert set(s2.scales) == {"PE-A", "FITC-A"}
    # written before the channels are known: kept until they are, then collapsed
    s3 = Sample(p, 3)
    s3.scales = {"Comp-PE-A": LogScale(1, 1e3)}
    assert list(dict.keys(s3.scales)) == ["Comp-PE-A"]
    s3.load_header()
    assert s3.scale("Comp-PE-A") == LogScale(1, 1e3) and list(dict.keys(s3.scales)) == ["PE-A"]
    # a matrix with another prefix: the old spelling collapses on lookup
    s4 = Sample(p, 4)
    s4.load_header()
    s4.scales["<PE-A>"] = LogScale(2, 2e4)
    s4.comp = SpilloverMatrix("m", ["FITC-A", "PE-A"], S2, prefix="<", suffix=">")
    assert s4.scale("<PE-A>") == LogScale(2, 2e4) and s4.scale("PE-A") == LogScale(2, 2e4)
    assert "<PE-A>" not in dict.keys(s4.scales)


def test_time_axis_ranged_to_acquired_data(tmp_path):
    n = 4096
    rng = np.random.default_rng(3)
    ev = np.column_stack([rng.normal(9e4, 8e3, n), rng.normal(5e4, 6e3, n), rng.normal(300, 150, n), np.linspace(0, 8190, n)]).astype(np.float32)
    # BD FACSDiva style: Time $PnR = 262144 but the run spans 0 .. 8190
    p = write_fcs(tmp_path / "diva.fcs", ev, ["FSC-A", "SSC-A", "FITC-A", "Time"], ranges=[262144] * 4)
    s = Sample(p, 1)
    s.load_header()
    assert s.scale("Time") == LinearScale(0, 262144)  # header only: $PnR, not cached
    assert "Time" not in s.scales
    s.ensure_loaded()
    sc = s.scale("Time")
    assert isinstance(sc, LinearScale) and sc.min == 0 and sc.max == pytest.approx(8354, abs=1)
    assert s.scale("FSC-A") == LinearScale(0, 262144)  # scatter keeps $PnR
    # a chosen time scale is kept
    s.set_scale("Time", LinearScale(0, 5000))
    assert s.scale("Time") == LinearScale(0, 5000)


def test_selective_invalidation_on_scale_change(sample, synthetic_fcs):
    s = sample
    s.set_compensation(SpilloverMatrix("m", ["FITC-A", "PE-A"], S2))
    scatter = s.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [50000, 20000], [None, None])))
    cd4 = scatter.add_child(Population("CD4", PolygonGate(["Comp-FITC-A", "Comp-PE-A"], [(100, 1000), (1e5, 1000), (1e5, 1e5), (100, 1e5)])))
    cd4_sub = cd4.add_child(Population("Sub", RectangleGate(["FSC-A"], [60000], [None])))
    pe_range = scatter.add_child(Population("PE range", RectangleGate(["Comp-PE-A"], [5000], [None])))
    fitc_ell = scatter.add_child(Population("FITC", EllipseGate(["FSC-A", "Comp-FITC-A"], (9e4, 2e4), (1.1e5, 2e4), (9e4, 3e4))))
    both = scatter.add_child(Population("Bool", BooleanGate("and", ["/Cells/CD4", "/Cells/FITC"])))
    other_bool = scatter.add_child(Population("Bool2", BooleanGate("or", ["/Cells/PE range"])))
    for p in s.populations():
        p.count
    for n in ("SSC-A", "FSC-A", "PE-A", "Comp-PE-A", "Comp-FITC-A"):
        s.display(n)
    ssc_display = s._display_cache["SSC-A"]
    scatter_mask = scatter._mask
    s.set_scale("PE-A", LogScale(1, 1e5))  # "CD4" (stain of PE-A) via the T button
    assert scatter._mask is scatter_mask and s._display_cache["SSC-A"] is ssc_display
    assert "PE-A" not in s._display_cache and "Comp-PE-A" not in s._display_cache
    assert "Comp-FITC-A" in s._display_cache and "FSC-A" in s._display_cache
    assert cd4._mask is None and cd4_sub._mask is None and both._mask is None  # polygon, its child, boolean on it
    assert pe_range._mask is not None and other_bool._mask is not None  # data-space range on PE: unaffected
    assert fitc_ell._mask is not None
    # an identical scale invalidates nothing
    for p in s.populations():
        p.count
    s.set_scale("Comp-PE-A", LogScale(1, 1e5))
    assert cd4._mask is not None
    _assert_caches_fresh(s)


def _assert_caches_fresh(s):
    """Every cached display / derived value and mask equals a fresh computation."""
    fresh = s.detached_copy()
    for name, cached in s._display_cache.items():
        np.testing.assert_array_equal(cached, fresh.display(name), err_msg=f"stale display cache {name}")
    for name, cached in s._derived_cache.items():
        np.testing.assert_array_equal(cached, fresh.data(name), err_msg=f"stale derived cache {name}")
    for own, other in zip(s.root.walk(), fresh.root.walk()):
        if own._mask is not None:
            np.testing.assert_array_equal(own._mask, other.mask(), err_msg=f"stale mask {own.path}")


def _fill(s):
    for p in s.root.walk():
        p.count
    for n in s.parameter_names():
        try:
            s.display(n)
        except KeyError:
            pass


def test_selective_invalidation_never_leaves_stale_caches(sample, synthetic_fcs):
    s = sample
    s.ensure_loaded()
    n = s.n_events
    s.add_derived("FITC/PE", {"op": "ratio", "a": "Comp-FITC-A", "b": "Comp-PE-A", "kind": "linear", "min": -5, "max": 5})
    s.add_derived("FSCxSSC", {"op": "product", "a": "FSC-A", "b": "SSC-A"})
    s.add_computed("tSNE1", np.linspace(-10, 10, n))
    cells = s.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [50000, 20000], [None, None])))
    cells.add_child(Population("Ratio", PolygonGate(["FITC/PE", "FSC-A"], [(-1, 0), (1, 0), (1, 2e5), (-1, 2e5)])))
    cells.add_child(Population("CompPE", RectangleGate(["Comp-PE-A"], [5000], [None])))
    cells.add_child(Population("Scatter2", PolygonGate(["FSCxSSC", "SSC-A"], [(0, 0), (1e10, 0), (1e10, 2e5), (0, 2e5)])))
    t = cells.add_child(Population("tSNE", RectangleGate(["tSNE1"], [0], [None])))
    t.add_child(Population("tSNE poly", PolygonGate(["tSNE1", "FSC-A"], [(-5, 0), (5, 0), (5, 2e5), (-5, 2e5)])))
    cells.add_child(Population("Any", BooleanGate("or", ["/Cells/Ratio", "/Cells/tSNE"])))
    _fill(s)
    _assert_caches_fresh(s)
    # 1) a transform
    s.set_scale("Comp-FITC-A", LinearScale(-1000, 1e5))
    _assert_caches_fresh(s)
    _fill(s)
    # 2) compensation: compensated parameters and what is derived from them change,
    #    raw scatter (and the product of raw channels) keep their caches
    scatter_mask = cells._mask
    product = s._derived_cache["FSCxSSC"]
    s.set_compensation(SpilloverMatrix("m", ["FITC-A", "PE-A"], S2))
    assert cells._mask is scatter_mask and s._derived_cache["FSCxSSC"] is product
    assert "FITC/PE" not in s._derived_cache and cells.find("Ratio")._mask is None
    _assert_caches_fresh(s)
    _fill(s)
    s.set_compensation(SpilloverMatrix("m2", ["FITC-A", "PE-A"], [[1, 0.5], [0.1, 1]]), enabled=False)
    _assert_caches_fresh(s)
    _fill(s)
    # 3) a computed parameter
    s.add_computed("tSNE1", np.linspace(10, -10, n))
    assert cells._mask is scatter_mask
    _assert_caches_fresh(s)
    _fill(s)
    s.remove_computed("tSNE1")
    _assert_caches_fresh(s)
    assert cells.find("tSNE").count == 0
    _fill(s)
    # 4) derived parameters
    s.add_derived("FSCxSSC", {"op": "sum", "a": "FSC-A", "b": "SSC-A"})
    _assert_caches_fresh(s)
    _fill(s)
    s.remove_derived("FITC/PE")
    assert "FITC/PE" not in s.derived and "FITC/PE" not in s.scales
    _assert_caches_fresh(s)
    assert cells.find("Ratio").count == 0


def test_derived_registry_default_kind_and_remove(sample):
    from flowdesk.core import model

    def double(smp, spec):
        return smp.data(spec["inputs"][0]) * 2

    model.DERIVED_OPS["double"] = double
    try:
        sample.add_derived("2xFSC", {"op": "double", "inputs": ["FSC-A"], "kind": "log", "min": 10, "max": 1e6})
        np.testing.assert_allclose(sample.data("2xFSC"), sample.data("FSC-A") * 2, rtol=1e-6)
        assert sample.scale("2xFSC") == LogScale(10, 1e6)
        sample.add_derived("r", {"op": "ratio", "a": "FITC-A", "b": "PE-A", "kind": "linear", "min": -2, "max": 3})
        assert sample.scale("r") == LinearScale(-2, 3)
        # redefining a derived parameter drops its cached default scale
        sample.add_derived("r", {"op": "ratio", "a": "FITC-A", "b": "PE-A", "kind": "linear", "min": 0, "max": 9})
        assert sample.scale("r") == LinearScale(0, 9)
        sample.remove_derived("2xFSC")
        assert "2xFSC" not in sample.parameter_names() and "2xFSC" not in sample._derived_cache
        sample.remove_derived("does not exist")  # no error
    finally:
        model.DERIVED_OPS.pop("double", None)


def test_compensated_file_does_not_get_its_matrix_again(tmp_path):
    ev = make_synthetic_events(500)["events"]
    spill = {"$SPILLOVER": "2,FITC-A,PE-A,1,0.2,0.05,1"}
    names = ["FSC-A", "SSC-A", "Comp-FITC-A", "Comp-PE-A", "Time"]
    p = write_fcs(tmp_path / "exported.fcs", ev, names, extra_keywords=spill)
    s = Sample(p, 1)
    s.load_header()
    assert s.comp is None
    s.ensure_loaded()
    assert s.comp is None
    q = write_fcs(tmp_path / "flagged.fcs", ev, SYNTH_NAMES, extra_keywords={"$SPILLOVER": "2,FITC-A,PE-A,1,0.2,0.05,1", "FLOWDESK_COMPENSATED": "TRUE"})
    t = Sample(q, 2)
    t.load_header()
    assert t.comp is None
    r = write_fcs(tmp_path / "raw.fcs", ev, SYNTH_NAMES, extra_keywords=spill)
    u = Sample(r, 3)
    u.load_header()
    assert u.comp is not None and u.comp.params == ["FITC-A", "PE-A"]


def test_unload_and_memory_budget(tmp_path):
    ws = Workspace()
    samples = [ws.add_sample(make_file(tmp_path, f"m{i}.fcs", seed=i, n=40000)) for i in range(3)]
    for s in samples:
        s.root.add_child(Population("Cells", RectangleGate(["FSC-A"], [50000], [None])))
    counts = [s.root.children[0].count for s in samples]  # unlimited: all stay
    assert all(s.loaded for s in samples) and ws.memory_budget.released == 0
    assert samples[0].resident_bytes() > 40000 * 5 * 4
    s0 = samples[0]
    s0.root.add_child(Population("P", PolygonGate(["FITC-A", "PE-A"], [(1, 1), (1e5, 1), (1e5, 1e5)])))
    s0.set_scale("FITC-A", LinearScale(0, 1e5))
    s0.root.children[0].count
    assert s0.unload() and s0.fcs is None and s0.root.children[0]._mask is None and not s0._display_cache
    assert s0.scale("FITC-A") == LinearScale(0, 1e5) and s0.keywords and s0.channels  # header and scales stay
    np.testing.assert_array_equal(s0.data("FSC-A"), make_synthetic_events(n=40000, seed=0)["events"][:, 0])  # transparent reload
    assert s0.root.children[0].count == counts[0]
    # a 1 MB cap: loading one sample releases the least recently used one
    for s in samples:
        s.unload()
    ws.memory_budget.limit_mb = 1
    samples[0].data("FSC-A")
    samples[1].data("FSC-A")
    assert samples[0].fcs is None and samples[1].loaded
    assert samples[0].data("FSC-A").shape == (40000,)  # reloaded on demand ...
    assert samples[1].fcs is None  # ... releasing the other one
    # pinned samples (shown in a window) are never released
    ws.memory_budget.pin_provider = lambda: [samples[0]]
    samples[2].data("FSC-A")
    assert samples[0].loaded and samples[2].loaded
    ws.memory_budget.pinned = {samples[2].id}
    ws.memory_budget.pin_provider = None
    samples[1].data("FSC-A")
    assert samples[0].fcs is None and samples[2].loaded and samples[1].loaded
    assert ws.memory_budget.released >= 2
    # a sample without a file behind it cannot be released
    lost = Sample(tmp_path / "gone.fcs", 9)
    assert not lost.unload()


def test_adding_1000_header_only_samples_is_fast(tmp_path):
    import time

    src = make_file(tmp_path, "template.fcs", n=10)
    data = src.read_bytes()
    folder = tmp_path / "many"
    folder.mkdir()
    paths = []
    for i in range(1000):
        q = folder / f"s{i:04d}.fcs"
        q.write_bytes(data)
        paths.append(q)
    ws = Workspace()
    t0 = time.perf_counter()
    for q in paths:
        ws.add_sample(q)
    elapsed = time.perf_counter() - t0
    assert len(ws.samples) == 1000 and elapsed < 1.0, elapsed
    assert ws.add_sample(folder / "." / "s0500.fcs") is ws.samples[500]
    assert len({s.id for s in ws.samples}) == 1000
    # the index follows samples appended directly and relocated paths
    extra = Sample(folder / "x.fcs", 5000)
    ws.samples.append(extra)
    assert ws.sample_by_path(folder / "x.fcs") is extra
    moved = tmp_path / "moved.fcs"
    ws.samples[3].path = moved
    assert ws.sample_by_path(moved) is ws.samples[3] and ws.sample_by_path(paths[3]) is None
    ws.remove_sample(extra)
    assert ws.sample_by_path(folder / "x.fcs") is None


def test_detached_copy_and_adopt(tmp_path):
    p = make_file(tmp_path, "w.fcs", n=3000)
    s = Sample(p, 1)
    s.load_header()
    cells = s.root.add_child(Population("Cells", RectangleGate(["FSC-A"], [50000], [None])))
    pos = cells.add_child(Population("Pos", PolygonGate(["FITC-A", "PE-A"], [(1000, 1000), (1e5, 1000), (1e5, 1e5), (1000, 1e5)])))
    fp = s.state_fingerprint()
    copy = s.detached_copy()
    assert copy.root is not s.root and copy.scales is not s.scales
    copy.ensure_loaded()  # what a worker does
    for node in copy.root.walk():
        node.mask()
    assert s.fcs is None and cells._mask is None  # the original is untouched
    assert s.adopt(copy, fp)
    assert s.loaded and cells._mask is not None and pos._mask is not None
    fresh = Sample(p, 2)
    fresh.load_header()
    fresh.root.add_child(cells.clone())
    assert cells.count == fresh.root.children[0].count
    # something changed meanwhile: the data is taken, the masks are not
    t = Sample(p, 3)
    t.load_header()
    g = t.root.add_child(Population("G", RectangleGate(["FSC-A"], [50000], [None])))
    fp = t.state_fingerprint()
    copy = t.detached_copy()
    copy.ensure_loaded()
    copy.root.children[0].mask()
    g.gate.lo = [80000]
    assert not t.adopt(copy, fp)
    assert t.loaded and g._mask is None
    assert g.count == int((t.data("FSC-A") >= 80000).sum())
