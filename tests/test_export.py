"""FCS writing: population export and sample concatenation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flowdesk.core.fcs import Channel, FCSFile, write_fcs
from flowdesk.core.gates import RectangleGate
from flowdesk.core.model import Population, Workspace, concatenate_samples


def _make_file(path: Path, n: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    events = np.column_stack([rng.normal(50000, 10000, n), rng.normal(30000, 8000, n), rng.lognormal(6, 1.2, n)]).astype(np.float32)
    channels = [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144), Channel(3, "FITC-A", "CD4", 262144)]
    write_fcs(path, events, channels, {"$CYT": "TestCytometer", "VENDOR_KEY": "a/b"})
    return path


def test_write_and_read_back(tmp_path):
    path = _make_file(tmp_path / "a.fcs", 500, 1)
    f = FCSFile.read(path)
    assert f.version.startswith("FCS3.1")
    assert f.n_events == 500
    assert f.channel_names == ["FSC-A", "SSC-A", "FITC-A"]
    assert f.channels[2].stain == "CD4"
    assert f.keywords["$CYT"] == "TestCytometer"
    assert f.keywords["VENDOR_KEY"] == "a/b"  # escaped delimiter survives
    assert f.keywords["$FIL"] == "a.fcs"
    assert np.isfinite(f.events).all()


def test_population_export_and_concatenate(tmp_path):
    ws = Workspace()
    s1 = ws.add_sample(_make_file(tmp_path / "s1.fcs", 400, 2))
    s2 = ws.add_sample(_make_file(tmp_path / "s2.fcs", 300, 3))
    gate = RectangleGate(["FSC-A", "SSC-A"], [45000, 0], [None, None])
    ws.apply_population_to_group(ws.groups[0], "", Population("Big", gate))
    pop1 = s1.root.find("/Big")
    out = tmp_path / "big.fcs"
    s1.export_population_fcs(pop1, out)
    f = FCSFile.read(out)
    assert f.n_events == pop1.count
    assert (f.events[:, 0] >= 45000).all()
    assert f.keywords["FLOWDESK_POPULATION"] == "/Big"
    cat = tmp_path / "cat.fcs"
    n = concatenate_samples([s1, s2], [s1.root.find("/Big"), s2.root.find("/Big")], cat)
    f = FCSFile.read(cat)
    assert f.n_events == n == pop1.count + s2.root.find("/Big").count
    assert f.channel_names[-1] == "SampleID"
    ids = f.column("SampleID")
    assert set(np.unique(ids).tolist()) == {1.0, 2.0}
    limited = concatenate_samples([s1, s2], None, tmp_path / "lim.fcs", max_events_per_sample=100)
    assert limited == 200
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.fcs"
        write_fcs(bad, np.zeros((10, 2), dtype=np.float32), [Channel(1, "X", "", 1024), Channel(2, "Y", "", 1024)])
        s3 = ws.add_sample(bad)
        concatenate_samples([s1, s3], None, tmp_path / "mixed.fcs")


# ------------------------------------------------------------- core/export.py
from flowdesk.core import export  # noqa: E402
from flowdesk.core.compensation import SpilloverMatrix  # noqa: E402
from flowdesk.core.fcs import parse_spillover  # noqa: E402

PANEL = ["FSC-A", "SSC-A", "FITC-A", "PE-A", "Time"]
STAINS = ["", "", "CD3", "CD4", ""]


def _panel_file(path: Path, n: int, seed: int, names=PANEL, stains=STAINS, keywords=None) -> Path:
    rng = np.random.default_rng(seed)
    cols = {
        "FSC-A": rng.normal(80000, 15000, n), "SSC-A": rng.normal(40000, 9000, n),
        "FITC-A": rng.lognormal(7, 1.0, n), "PE-A": rng.lognormal(8, 1.0, n), "Time": np.linspace(0, 100, n),
        "APC-A": rng.lognormal(6, 1.0, n),
    }
    events = np.column_stack([cols[c] for c in names]).astype(np.float32)
    chans = [Channel(i, c, s, 262144) for i, (c, s) in enumerate(zip(names, stains), start=1)]
    kw = {"$CYT": "TestCytometer", "$P3V": "450", "$P3F": "530/30", "$P4V": "500", "$SPILLOVER": "2,FITC-A,PE-A,1,0.15,0.02,1"}
    kw.update(keywords or {})
    write_fcs(path, events, chans, kw)
    return path


@pytest.fixture
def rich(tmp_path):
    """A compensated sample with a gate, a derived ratio and computed parameters."""
    ws = Workspace()
    s = ws.add_sample(_panel_file(tmp_path / "rich.fcs", 2000, 7))
    assert s.comp is not None and s._comp_active() is not None
    pop = s.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [60000, 0], [None, None])))
    s.add_derived("FITC/PE", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
    s.add_computed("tSNE_X", np.linspace(-40, 55, s.n_events).astype(np.float32), {"min": -50.0, "max": 60.0})
    s.add_computed("FlowSOM", (np.arange(s.n_events) % 7).astype(np.float32))
    return ws, s, pop


def test_select_parameters_modes(rich):
    ws, s, pop = rich
    all_names = export.select_parameters(s, "all")
    assert all_names[:2] == ["Comp-FITC-A", "Comp-PE-A"] and "FITC/PE" in all_names and "tSNE_X" in all_names
    comp = export.select_parameters(s, "compensated", include_derived=False, include_computed=False)
    assert comp == ["Comp-FITC-A", "Comp-PE-A", "FSC-A", "SSC-A", "Time"]
    raw = export.select_parameters(s, "uncompensated", include_computed=False)
    assert raw == PANEL + ["FITC/PE"]
    assert export.select_parameters(s, "selected", ["PE-A", "nope", "tSNE_X"]) == ["PE-A", "tSNE_X"]
    with pytest.raises(ValueError):
        export.select_parameters(s, "bogus")


def test_export_population_chosen_columns_derived_computed_subsample(rich, tmp_path):
    ws, s, pop = rich
    out = tmp_path / "chosen.fcs"
    params = ["FSC-A", "Comp-FITC-A", "FITC/PE", "tSNE_X", "FlowSOM"]
    n = export.export_population(
        s, pop, out, parameters=params, include_computed=True, include_derived=True, scaled=False,
        max_events=300, seed=0, extra_keywords={"STIM": "PMA", "DONOR": "D1"},
    )
    assert n == 300 < pop.count
    f = FCSFile.read(out)
    assert f.channel_names == params and f.n_events == 300
    kw = f.keywords
    assert kw["STIM"] == "PMA" and kw["DONOR"] == "D1" and kw["FLOWDESK_POPULATION"] == "/Cells"
    assert kw["$P5R"] == "7"  # FlowSOM ids 0..6: range from the data
    assert kw["$P4R"] == "61"  # tSNE_X: range from computed_info["max"]
    assert kw["$P4D"].startswith("Linear,-50")
    assert kw["$P2S"] == "CD3" and kw["$P2V"] == "450" and kw["$P2F"] == "530/30"
    # the rows are the seeded random subset of the population, in order
    idx = export.subsample_index(pop.mask(), 300, None, 0)
    np.testing.assert_allclose(f.column("FITC/PE"), s.data("FITC/PE")[idx], rtol=1e-6)
    np.testing.assert_allclose(f.column("Comp-FITC-A"), s.data("Comp-FITC-A")[idx], rtol=1e-5)
    np.testing.assert_array_equal(f.column("tSNE_X"), s.data("tSNE_X")[idx])
    # the data are compensated already: no matrix to apply again
    assert parse_spillover(kw) is None and kw["FLOWDESK_COMPENSATED"] == "TRUE"
    again = Workspace().add_sample(out)
    assert again.comp is None
    names = again.parameter_names()
    assert len(names) == len(set(names)) == 5
    # the same seed gives the same subset
    export.export_population(s, pop, tmp_path / "again.fcs", parameters=params, max_events=300, seed=0)
    np.testing.assert_array_equal(FCSFile.read(tmp_path / "again.fcs").events, f.events)


def test_export_all_parameters_and_raw_matrix(rich, tmp_path):
    ws, s, pop = rich
    out = tmp_path / "all.fcs"
    export.export_population(s, None, out)  # whole sample, every parameter
    f = FCSFile.read(out)
    assert f.channel_names == s.parameter_names() and f.n_events == s.n_events
    assert f.keywords["FLOWDESK_POPULATION"] == "Ungated"
    reopened = Workspace().add_sample(out)
    assert reopened.comp is None and reopened.channel_names.count("Comp-FITC-A") == 1
    # raw channels only: the matrix in use goes with them
    s.comp = SpilloverMatrix("Edited", ["FITC-A", "PE-A"], np.array([[1, 0.3], [0.01, 1]]))
    s.invalidate_display()
    raw = tmp_path / "raw.fcs"
    export.export_population(s, pop, raw, mode="uncompensated", include_derived=False, include_computed=False)
    g = FCSFile.read(raw)
    assert g.channel_names == PANEL
    names, m = parse_spillover(g.keywords)
    assert names == ["FITC-A", "PE-A"]
    np.testing.assert_allclose(m, [[1, 0.3], [0.01, 1]])
    assert "FLOWDESK_COMPENSATED" not in g.keywords
    # only one of the matrix channels written: the matrix shrinks with it
    export.export_population(s, pop, tmp_path / "one.fcs", parameters=["FSC-A", "PE-A"])
    assert parse_spillover(FCSFile.read(tmp_path / "one.fcs").keywords)[0] == ["PE-A"]


def test_export_scale_values_fcs_and_csv(rich, tmp_path):
    ws, s, pop = rich
    out = tmp_path / "scaled.fcs"
    export.export_population(s, pop, out, parameters=["FSC-A", "Comp-PE-A", "tSNE_X"], scaled=True)
    f = FCSFile.read(out)
    mask = pop.mask()
    for name in ("FSC-A", "Comp-PE-A", "tSNE_X"):
        np.testing.assert_allclose(f.column(name), s.display(name)[mask] * 1024, rtol=1e-5, atol=1e-3)
    assert f.keywords["$P2R"] == "1024" and parse_spillover(f.keywords) is None
    assert f.keywords["FLOWDESK_VALUES"].startswith("scale")
    csv_path = tmp_path / "scaled.csv"
    n = export.export_population(s, pop, csv_path, parameters=["FSC-A", "Comp-PE-A"], scaled=True, scale_range=4096)
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    assert rows.shape == (n, 2)
    np.testing.assert_allclose(rows[:, 1], s.display("Comp-PE-A")[mask] * 4096, rtol=1e-5, atol=1e-2)
    assert csv_path.read_text().splitlines()[0] == "FSC-A,Comp-PE-A"


def test_export_csv_options(rich, tmp_path):
    ws, s, pop = rich
    p = tmp_path / "noheader.csv"
    n = export.export_population(s, pop, p, parameters=["FSC-A", "FITC/PE"], header=False, percent=10, seed=3)
    rows = np.loadtxt(p, delimiter=",", ndmin=2)
    assert rows.shape == (n, 2) and n == round(pop.count * 0.1)
    p2 = tmp_path / "labels.csv"
    export.export_population(s, pop, p2, parameters=["FITC-A", "Comp-PE-A"], header_labels=True, max_events=5)
    assert p2.read_text().splitlines()[0] == "FITC-A :: CD3,Comp-PE-A :: CD4"
    with pytest.raises(KeyError):
        export.export_population(s, pop, tmp_path / "x.csv", parameters=["nope"])


def test_format_name_and_batch_export_over_group(tmp_path):
    ws = Workspace()
    for k, well in enumerate(["A1", "B2", "C3"]):
        ws.add_sample(_panel_file(tmp_path / f"s{k}.fcs", 500 + 100 * k, k, keywords={"$WELLID": well}))
    gate = RectangleGate(["FSC-A", "SSC-A"], [60000, 0], [None, None])
    group = ws.add_group("Test", ws.samples)
    ws.apply_population_to_group(group, "", Population("Cells", gate))
    # one member lacks the population
    lonely = ws.add_sample(_panel_file(tmp_path / "lonely.fcs", 100, 9))
    group.sample_ids.add(lonely.id)
    s0 = ws.samples[0]
    assert export.format_name("{sample}_{population}_{$WELLID}", s0, s0.root.find("/Cells")) == "s0_Cells_A1"
    assert export.format_name("{WELLID}-{index}", s0, s0.root, group, 4) == "A1-004"
    assert export.format_name("{name}:{path}", s0, None) == "s0.fcs_Ungated"
    res = export.batch_export(ws, group, "/Cells", tmp_path / "out", pattern="{group}_{$WELLID}_{population}", fmt="fcs", max_events=200, seed=1)
    assert sorted(p.name for p in res.paths) == ["Test_A1_Cells.fcs", "Test_B2_Cells.fcs", "Test_C3_Cells.fcs"]
    assert [s.name for s, _ in res.skipped] == ["lonely.fcs"]
    for sample, path, n in res.written:
        f = FCSFile.read(path)
        assert f.n_events == n == min(200, sample.root.find("/Cells").count)
        assert f.keywords["FLOWDESK_POPULATION"] == "/Cells" and f.keywords["FLOWDESK_SOURCE"] == sample.name
    # CSV batch; existing names are not overwritten
    res2 = export.batch_export(ws, group, ["/Cells", ""], tmp_path / "out", pattern="{group}_{$WELLID}_{population}", fmt="csv")
    assert len(res2.paths) == 7 and all(p.suffix == ".csv" for p in res2.paths)
    res3 = export.batch_export(ws, group, "/Cells", tmp_path / "out", pattern="{group}_{$WELLID}_{population}")
    assert sorted(p.name for p in res3.paths)[0] == "Test_A1_Cells (2).fcs"


def test_concatenate_reordered_subset_keywords_and_computed(tmp_path):
    ws = Workspace()
    a = ws.add_sample(_panel_file(tmp_path / "a.fcs", 400, 1, keywords={"STIM": "PMA", "DOSE": "10"}))
    reordered = ["PE-A", "Time", "FSC-A", "FITC-A", "SSC-A", "APC-A"]
    b = ws.add_sample(_panel_file(tmp_path / "b.fcs", 300, 2, names=reordered, stains=["CD4", "", "", "CD3", "", "CD8"], keywords={"STIM": "unstim", "DOSE": "0"}))
    subset = ["FSC-A", "SSC-A", "PE-A", "FITC-A"]
    c = ws.add_sample(_panel_file(tmp_path / "c.fcs", 200, 3, names=subset, stains=["", "", "CD4", "CD3"], keywords={"STIM": "PMA"}))
    for s in (a, b, c):
        s.add_derived("ratio", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
        s.add_computed("UMAP_1", np.arange(s.n_events, dtype=np.float32))
    pops = [s.root for s in (a, b, c)]
    out = tmp_path / "cat.fcs"
    n = export.concatenate([a, b, c], pops, out, keywords=["STIM", "DOSE"], match="intersection", include_derived=True, include_computed=True)
    assert n == 900
    f = FCSFile.read(out)
    assert f.channel_names == ["FSC-A", "SSC-A", "FITC-A", "PE-A", "ratio", "UMAP_1", "STIM", "DOSE", "SampleID"]
    kw = f.keywords
    assert kw["FLOWDESK_KW_STIM"] == "1=PMA;2=unstim"
    table = export.parse_code_table(kw["FLOWDESK_KW_STIM"])
    stim = f.column("STIM")
    assert [table[v] for v in (stim[0], stim[400], stim[700])] == ["PMA", "unstim", "PMA"]
    dose = f.column("DOSE")
    assert kw["FLOWDESK_KW_DOSE"] == "numeric" and dose[0] == 10 and dose[400] == 0 and np.isnan(dose[700])
    np.testing.assert_array_equal(np.unique(f.column("SampleID")), [1, 2, 3])
    # values were matched by name, not by position
    np.testing.assert_allclose(f.column("PE-A")[400:700], b.data("PE-A"), rtol=1e-6)
    np.testing.assert_allclose(f.column("ratio")[700:], c.data("ratio"), rtol=1e-6)
    np.testing.assert_array_equal(f.column("UMAP_1")[400:700], np.arange(300))
    assert kw["$P3S"] == "CD3" and kw["$P3V"] == "450"
    assert kw["FLOWDESK_CONCATENATED"] == "a.fcs;b.fcs;c.fcs"
    assert "STIM" not in kw  # differs between the samples: not inherited from the first one
    # union keeps every parameter (NaN where a sample has none)
    u = tmp_path / "union.fcs"
    export.concatenate([a, b, c], None, u, match="union", sample_id=False)
    g = FCSFile.read(u)
    assert "APC-A" in g.channel_names and "Time" in g.channel_names and "SampleID" not in g.channel_names
    apc = g.column("APC-A")
    assert np.isnan(apc[:400]).all() and np.isfinite(apc[400:700]).all() and np.isnan(apc[700:]).all()
    with pytest.raises(ValueError, match="different parameters"):
        export.concatenate([a, c], None, tmp_path / "strict.fcs", match="strict")
    # the same matrix in every sample goes with the raw columns
    assert parse_spillover(g.keywords)[0] == ["FITC-A", "PE-A"]


def test_concatenate_compensated_subsample_and_csv(rich, tmp_path):
    ws, s, pop = rich
    other = ws.add_sample(_panel_file(tmp_path / "other.fcs", 800, 11))
    out = tmp_path / "comp_cat.fcs"
    n = export.concatenate([s, other], [pop, other.root], out, include_compensated=True, max_events_per_sample=100, seed=5)
    assert n == 200
    f = FCSFile.read(out)
    assert f.channel_names[:2] == ["Comp-FITC-A", "Comp-PE-A"] and f.channel_names[-1] == "SampleID"
    assert f.keywords["FLOWDESK_COMPENSATED"] == "TRUE" and parse_spillover(f.keywords) is None
    reopened = Workspace().add_sample(out)
    assert reopened.comp is None
    csv_out = tmp_path / "cat.csv"
    assert export.concatenate([s, other], None, csv_out, parameters=["FSC-A", "PE-A"]) == s.n_events + other.n_events
    assert csv_out.read_text().splitlines()[0] == "FSC-A,PE-A,SampleID"


def test_keyword_codes():
    assert export.keyword_codes(["b", "a10", "a2", None]) == ([3.0, 2.0, 1.0, 0.0], "0=;1=a2;2=a10;3=b")
    assert export.keyword_codes(["1.5", "", "3"])[1] == "numeric"
    assert export.parse_code_table("0=;1=a2;2=a10") == {0.0: "", 1.0: "a2", 2.0: "a10"}
    assert export.parse_code_table("numeric") == {}


def test_split_datasets(tmp_path):
    a = _make_file(tmp_path / "a.fcs", 50, 1)
    b = _make_file(tmp_path / "b.fcs", 70, 2)
    raw_a = bytearray(a.read_bytes())
    # point the first data set's $NEXTDATA at the second one (same width)
    key = b"$NEXTDATA" + raw_a[58:59] + b"0" + raw_a[58:59]
    assert key in raw_a
    size = len(raw_a)
    text_end = int(raw_a[18:26])
    # rewrite with a fixed-width value so offsets do not move
    write_fcs(a, FCSFile.read(a).events, FCSFile.read(a).channels, {}, {"$NEXTDATA": f"{0:012d}"})
    size = a.stat().st_size
    write_fcs(a, FCSFile.read(a).events, FCSFile.read(a).channels, {}, {"$NEXTDATA": f"{size:012d}"})
    assert a.stat().st_size == size and text_end
    multi = tmp_path / "multi.fcs"
    multi.write_bytes(a.read_bytes() + b.read_bytes())
    out = export.split_datasets(multi, tmp_path / "split")
    assert [p.name for p in out] == ["multi_dataset1.fcs", "multi_dataset2.fcs"]
    assert [FCSFile.read(p).n_events for p in out] == [50, 70]
    assert export.split_datasets(b) == []
