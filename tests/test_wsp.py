"""FlowJo workspace (.wsp) import / export tests."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from flowdesk.core.compensation import SpilloverMatrix
from flowdesk.core.gates import BooleanGate, EllipseGate, PolygonGate, RectangleGate
from flowdesk.core.model import Criterion, Group, Population, Sample, StatNode, Workspace
from flowdesk.core.transforms import ArcsinhScale, LinearScale, LogicleScale, LogScale
from flowdesk.core.wsp_export import export_wsp
from flowdesk.core.wsp_import import flowjo_counts, import_wsp

DESK = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger")
FACS = DESK / "Daten Forschungspraktikum Lukas Gerstung" / "FACS DATA"
SEARCH_DIRS = [FACS / "FACS ASSAY TEST", FACS / "FACS Co-Cultivation"]
REAL_FILES = [
    DESK / "recompensated T cell panel" / "19-Aug-2025.wsp",
    DESK / "FACS Assay Test.wsp",
    DESK / "31-Apr-2025.wsp",
    DESK / "30-Apr-2025.wsp",
    DESK / "12-May-2025.wsp",
    DESK / "Platte.wsp",
]


def _id(p: Path) -> str:
    return p.name


def _xml_samples(path: Path) -> dict[int, str]:
    root = ET.parse(path).getroot()
    out = {}
    for sel in root.find("SampleList"):
        node = sel.find("SampleNode")
        out[int(sel.find("DataSet").get("sampleID"))] = node.get("name")
    return out


# ------------------------------------------------------------------ real files
@pytest.mark.parametrize("wsp", REAL_FILES, ids=_id)
def test_real_structure(wsp: Path):
    if not wsp.exists():
        pytest.skip(f"{wsp} not available")
    ws, report = import_wsp(wsp, SEARCH_DIRS)
    assert ws.path is None
    assert ws.settings["imported_from"] == str(wsp)
    xml_samples = _xml_samples(wsp)
    assert len(ws.samples) == len(xml_samples) == report.samples
    assert {s.id: s.name for s in ws.samples} == xml_samples
    assert ws.groups[0].role == "all"
    counts = flowjo_counts(wsp)
    for s in ws.samples:
        xml_paths = {p for (sid, p) in counts if sid == s.id and p}
        ours = {p.path for p in s.populations()}
        assert ours == xml_paths, s.name


@pytest.mark.parametrize("wsp", REAL_FILES, ids=_id)
def test_real_counts(wsp: Path):
    if not wsp.exists():
        pytest.skip(f"{wsp} not available")
    ws, _ = import_wsp(wsp, SEARCH_DIRS)
    counts = flowjo_counts(wsp)
    loaded = [s for s in ws.samples if s.path.is_file()]
    if not loaded:
        pytest.skip("no FCS files found")
    rows, failures = [], []
    for s in loaded:
        for pop in s.root.walk():
            fj = counts.get((s.id, pop.path))
            if fj is None or fj < 0:
                continue
            ours = pop.count
            diff = abs(ours - fj)
            rel = diff / fj if fj else 0.0
            rows.append(f"{s.id:>3} {pop.path or '/':<70.70} FlowJo {fj:>7} ours {ours:>7} ({100 * rel:+.2f}%)")
            if rel > 0.02 and diff > 5:
                failures.append(rows[-1])
    print(f"\n{wsp.name}:\n" + "\n".join(rows))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("wsp", REAL_FILES[:2], ids=_id)
def test_real_round_trip(wsp: Path, tmp_path: Path):
    if not wsp.exists():
        pytest.skip(f"{wsp} not available")
    ws, _ = import_wsp(wsp, SEARCH_DIRS)
    out = tmp_path / "out.wsp"
    export_wsp(ws, out)
    ws2, _ = import_wsp(out)
    _assert_same(ws, ws2)
    # counts written on export are our own counts for loaded samples
    stored = flowjo_counts(out)
    for s in ws2.samples:
        if s.path.is_file():
            for pop in s.populations():
                c = stored[(s.id, pop.path)]
                if c >= 0:
                    assert pop.count == c


# ------------------------------------------------------------------- helpers
def _gate_equal(a, b):
    assert type(a) is type(b)
    assert a.dims == b.dims
    assert a.negated == b.negated
    if isinstance(a, RectangleGate):
        for x, y in zip(a.lo + a.hi, b.lo + b.hi):
            assert (x is None) == (y is None)
            if x is not None:
                assert x == pytest.approx(y)
    elif isinstance(a, PolygonGate):
        assert np.allclose(a.vertices, b.vertices)
    elif isinstance(a, EllipseGate):
        assert np.allclose(a.center, b.center)
        assert np.allclose(a.a_end, b.a_end)
        assert np.allclose(a.b_end, b.b_end)
    elif isinstance(a, BooleanGate):
        assert a.op == b.op and a.refs == b.refs


def _tree_equal(a: Population, b: Population):
    assert a.name == b.name
    assert (a.gate is None) == (b.gate is None), a.path
    if a.gate is not None:
        _gate_equal(a.gate, b.gate)
    assert [(s.statistic, s.param, s.percentile) for s in a.stats] == [(s.statistic, s.param, s.percentile) for s in b.stats]
    assert a.graph.plot_type == b.graph.plot_type and a.graph.x == b.graph.x
    assert [c.name for c in a.children] == [c.name for c in b.children]
    for ca, cb in zip(a.children, b.children):
        _tree_equal(ca, cb)


def _scale_equal(a, b):
    assert type(a) is type(b)
    da, db = a.to_dict(), b.to_dict()
    for k in da:
        if isinstance(da[k], float):
            assert da[k] == pytest.approx(db[k], rel=1e-6, abs=1e-9), k
        else:
            assert da[k] == db[k]


def _matrix_equal(a: SpilloverMatrix | None, b: SpilloverMatrix | None):
    assert (a is None) == (b is None)
    if a is not None:
        assert a.name == b.name and a.params == b.params and a.prefix == b.prefix and a.suffix == b.suffix
        assert np.allclose(a.matrix, b.matrix)


def _assert_same(ws: Workspace, ws2: Workspace):
    assert [(s.id, s.name, s.path) for s in ws.samples] == [(s.id, s.name, s.path) for s in ws2.samples]
    for s, s2 in zip(ws.samples, ws2.samples):
        _tree_equal(s.root, s2.root)
        assert set(s.scales) == set(s2.scales)
        for k in s.scales:
            _scale_equal(s.scales[k], s2.scales[k])
        _matrix_equal(s.comp if s.comp_enabled else None, s2.comp)
    assert set(ws.matrices) == set(ws2.matrices)
    for k in ws.matrices:
        _matrix_equal(ws.matrices[k], ws2.matrices[k])
    assert [(g.name, g.role) for g in ws.groups] == [(g.name, g.role) for g in ws2.groups]
    for g, g2 in zip(ws.groups, ws2.groups):
        assert {s.id for s in ws.members(g)} == {s.id for s in ws2.members(g2)}
        assert [(c.keyword, c.function, c.value, c.connector) for c in g.criteria] == [(c.keyword, c.function, c.value, c.connector) for c in g2.criteria]
        _tree_equal(g.root, g2.root)


# ------------------------------------------------------------ synthetic data
def _write_fcs(path: Path, names: list[str], events: np.ndarray):
    """Minimal FCS 3.0 writer (float32, little endian) for tests."""
    n, p = events.shape
    kw = {"$BYTEORD": "1,2,3,4", "$DATATYPE": "F", "$MODE": "L", "$NEXTDATA": "0", "$PAR": str(p), "$TOT": str(n)}
    for i, name in enumerate(names, start=1):
        kw.update({f"$P{i}N": name, f"$P{i}B": "32", f"$P{i}E": "0,0", f"$P{i}R": "262144"})
    data = events.astype("<f4").tobytes()

    def text(begin, end):
        k = dict(kw, **{"$BEGINDATA": str(begin), "$ENDDATA": str(end)})
        return ("|" + "|".join(f"{a}|{b}" for a, b in k.items()) + "|").encode()

    t = text(0, 0)
    for _ in range(3):  # offsets change the text length; iterate to a fixed point
        begin = 58 + len(t)
        t = text(begin, begin + len(data) - 1)
    begin = 58 + len(t)
    header = b"FCS3.0    " + f"{58:>8}{57 + len(t):>8}{begin:>8}{begin + len(data) - 1:>8}{0:>8}{0:>8}".encode()
    path.write_bytes(header + t + data)


@pytest.fixture
def synthetic(tmp_path: Path) -> Workspace:
    rng = np.random.default_rng(1)
    n = 5000
    ev = np.column_stack([
        rng.uniform(0, 250000, n),
        rng.uniform(0, 250000, n),
        rng.lognormal(7, 2, n) - 300,
        rng.lognormal(6, 2, n) - 200,
    ])
    names = ["FSC-A", "SSC-A", "FITC-A", "PE-A"]
    fcs = tmp_path / "data dir" / "my sample 10^5 + x.fcs"
    fcs.parent.mkdir()
    _write_fcs(fcs, names, ev)

    ws = Workspace()
    s = Sample(fcs, 7)
    s.load_header()
    s.scales = {
        "FSC-A": LinearScale(-5000, 262144),
        "SSC-A": LogScale(1, 262144),
        "FITC-A": LogicleScale.from_biex(262144, 4.42, 0.1, -63.1),
        "PE-A": ArcsinhScale(262144, 4.5, 0.5),
        "Comp-PE-A": LogicleScale(262144, 0.7, 4.5, 0.2),
    }
    s.comp = SpilloverMatrix("My matrix", ["FITC-A", "PE-A"], np.array([[1.0, 0.15], [0.02, 1.0]]))
    ws.samples.append(s)
    cells = s.root.add_child(Population("Cells", PolygonGate(["FSC-A", "SSC-A"], [(1e4, 1e3), (2e5, 1e3), (2e5, 2e5), (1e4, 2e5)])))
    cells.graph.x, cells.graph.y, cells.graph.plot_type = "Comp-FITC-A", "PE-A", "contour"
    q = cells.add_child(Population("FITC+", RectangleGate(["Comp-FITC-A"], [1000.0], [None])))
    q.graph.x, q.graph.plot_type = "Comp-FITC-A", "histogram"
    pe = cells.add_child(Population("PE+", RectangleGate(["Comp-FITC-A", "PE-A"], [None, 500.0], [1e5, None])))
    neg = RectangleGate(["SSC-A"], [5e4], [1.5e5])
    neg.negated = True
    cells.add_child(Population("Outside", neg))
    cells.add_child(Population("Blob", EllipseGate(["FSC-A", "SSC-A"], (1e5, 1e5), (1.6e5, 1.3e5), (8e4, 1.4e5))))
    cells.add_child(Population("not FITC", BooleanGate("not", ["/Cells/FITC+"])))
    cells.add_child(Population("both", BooleanGate("and", ["/Cells/FITC+", "/Cells/PE+"])))
    s.root.add_child(Population("either", BooleanGate("or", ["/Cells/FITC+", "/Cells/Blob"])))
    pe.stats = [StatNode("Median", "PE-A"), StatNode("Percentile", "Comp-FITC-A", 95.0), StatNode("Freq. of Parent")]
    ws.matrices["Workspace matrix"] = SpilloverMatrix("Workspace matrix", ["FITC-A", "PE-A"], np.array([[1.0, 0.3], [0.1, 1.0]]), "Comp-", "")
    g = ws.group_by_name("All Samples")
    g.root.add_child(Population("Cells", PolygonGate(["FSC-A", "SSC-A"], [(0, 0), (1, 0), (1, 1)])))
    custom = Group("Stained", "custom", [Criterion("$FIL", "Contains", "sample")], "#0000d0", True)
    custom.sample_ids.add(7)
    ws.groups.append(custom)
    return ws


def test_synthetic_round_trip(synthetic: Workspace, tmp_path: Path):
    ws = synthetic
    expected = {p.path: p.count for p in ws.samples[0].root.walk()}
    assert expected["/Cells/both"] > 0 and expected["/Cells/Outside"] > 0
    out = tmp_path / "round trip.wsp"
    export_wsp(ws, out)
    text = out.read_text(encoding="utf-8")
    assert 'xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating"' in text
    assert "<transforms:biex" in text and "<gating:EllipsoidGate" in text and "<NotNode" in text
    assert "%5E10" not in text and "my%20sample%2010%5E5%20+%20x.fcs" in text
    ws2, report = import_wsp(out)
    assert not report.warnings, report.warnings
    _assert_same(ws, ws2)
    got = {p.path: p.count for p in ws2.samples[0].root.walk()}
    assert got == expected
    assert flowjo_counts(out)[(7, "/Cells/both")] == expected["/Cells/both"]


def test_uri_resolution_by_basename(synthetic: Workspace, tmp_path: Path):
    out = tmp_path / "moved.wsp"
    export_wsp(synthetic, out)
    text = out.read_text(encoding="utf-8").replace("/data%20dir/", "/gone%20away/")
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (moved / "moved.wsp").write_text(text, encoding="utf-8")
    ws, report = import_wsp(moved / "moved.wsp", search_dirs=[tmp_path])
    assert ws.samples[0].path == synthetic.samples[0].path
    ws, report = import_wsp(moved / "moved.wsp")
    assert report.samples_missing == 1 and report.warnings
