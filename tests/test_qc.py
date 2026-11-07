"""Data cleaning (flowdesk.core.qc) on synthetic acquisitions."""
import json

import numpy as np
import pytest

from flowdesk.core import qc
from flowdesk.core.fcs import Channel, write_fcs
from flowdesk.core.gates import BooleanGate, RectangleGate
from flowdesk.core.model import Population, Workspace
from flowdesk.core.serialization import load_workspace, save_workspace

T_END = 1000.0
DROP = (250.0, 400.0)  # flow rate drops by 20 %
SHIFT = (650.0, 800.0)  # FITC median shifts


def _times(rng, rate, t_end, drop, drop_factor):
    segs = [(0.0, drop[0], rate), (drop[0], drop[1], rate * drop_factor), (drop[1], t_end, rate)]
    parts = [rng.uniform(a, b, rng.poisson(r * (b - a))) for a, b, r in segs]
    return np.sort(np.concatenate(parts))


def synth_display_run(seed=0, rate=100.0, drop_factor=0.8, shift_by=0.06):
    """Display-space (0..1) values of four parameters over ~100k events with a
    flow-rate drop segment and a FITC median shift in another segment."""
    rng = np.random.default_rng(seed)
    t = _times(rng, rate, T_END, DROP, drop_factor)
    n = t.size
    vals = {name: rng.normal(c, 0.08, n) for name, c in (("FSC-A", 0.40), ("SSC-A", 0.35), ("FITC-A", 0.60), ("PE-A", 0.55))}
    in_drop = (t >= DROP[0]) & (t < DROP[1])
    in_shift = (t >= SHIFT[0]) & (t < SHIFT[1])
    vals["FITC-A"][in_shift] += shift_by
    return t, vals, in_drop, in_shift


def synth_raw_events(seed=0, rate=60.0, drop_factor=0.8, shift_factor=2.5, saturated=0):
    """Raw event matrix FSC-A, SSC-A, FITC-A, PE-A, Time for an FCS file;
    FITC is 2.5x brighter in the SHIFT segment, the rate is 20 % lower in DROP.
    `saturated` events get FSC-A = 262143 (detector maximum)."""
    rng = np.random.default_rng(seed)
    t = _times(rng, rate, T_END, DROP, drop_factor)
    n = t.size
    fsc = rng.normal(80000, 6000, n)
    ssc = rng.normal(40000, 5000, n)
    fitc = rng.lognormal(np.log(5000), 0.3, n)
    in_drop = (t >= DROP[0]) & (t < DROP[1])
    in_shift = (t >= SHIFT[0]) & (t < SHIFT[1])
    fitc[in_shift] *= shift_factor
    pos = rng.random(n) < 0.5
    pe = np.where(pos, rng.lognormal(np.log(20000), 0.3, n), rng.normal(200, 150, n))
    sat = np.zeros(n, dtype=bool)
    if saturated:
        sat[rng.choice(n, saturated, replace=False)] = True
        fsc[sat] = 262143.0
    events = np.column_stack([fsc, ssc, fitc, pe, t]).astype(np.float32)
    return events, in_drop | in_shift, sat


QC_CHANNELS = [Channel(1, "FSC-A"), Channel(2, "SSC-A"), Channel(3, "FITC-A", "CD3"), Channel(4, "PE-A", "CD4"), Channel(5, "Time", range=1024)]


def write_qc_fcs(path, seed=0, **kw):
    events, bad, sat = synth_raw_events(seed, **kw)
    write_fcs(path, events, QC_CHANNELS)
    return path, bad, sat


# ----------------------------------------------------------------------- checks
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_run_flags_both_segments(seed):
    t, vals, in_drop, in_shift = synth_display_run(seed)
    r = qc.qc_summary(t, vals, checks=("rate", "signal"))
    bad = in_drop | in_shift
    keep_good = r.good[~bad].mean()
    removed_bad = (~r.good[bad]).mean()
    assert keep_good >= 0.95
    assert removed_bad >= 0.90
    centers = 0.5 * (r.edges[:-1] + r.edges[1:])
    inside_drop = (r.edges[:-1] >= DROP[0]) & (r.edges[1:] <= DROP[1])
    inside_shift = (r.edges[:-1] >= SHIFT[0]) & (r.edges[1:] <= SHIFT[1])
    assert r.bin_flags["rate"][inside_drop].all()  # the drop is a rate problem
    assert r.bin_flags["signal:FITC-A"][inside_shift].all()  # the shift a signal one
    assert not r.bin_flags["signal:FITC-A"][inside_drop].any()
    assert not r.bin_flags["rate"][inside_shift].any()
    clean = (centers < DROP[0] - 15) | ((centers > DROP[1] + 15) & (centers < SHIFT[0] - 15)) | (centers > SHIFT[1] + 15)
    assert r.flagged_bins[clean].mean() < 0.05
    # per-check percentages and the table
    assert 12 < r.percent_removed["rate"] < 18
    assert 13 < r.percent_removed["signal"] < 19
    assert r.percent_removed["total"] == pytest.approx(100 * (1 - r.good.mean()))
    rows = r.bin_rows()
    assert len(rows) == 100 and sum(row["events"] for row in rows) == t.size
    assert set(rows[0]) == {"bin", "t_start", "t_end", "events", "rate", "flagged_by"}
    flagged_rows = [row for row in rows if row["flagged_by"]]
    assert any("rate" in row["flagged_by"] for row in flagged_rows)
    assert any("signal:FITC-A" in row["flagged_by"] for row in flagged_rows)
    # three clean stretches of time remain
    assert len(r.intervals) == 3
    (a0, a1), (b0, b1), (c0, c1) = r.intervals
    assert a0 == pytest.approx(t.min()) and abs(a1 - DROP[0]) < 15
    assert abs(b0 - DROP[1]) < 15 and abs(b1 - SHIFT[0]) < 15
    assert abs(c0 - SHIFT[1]) < 15 and c1 > t.max()


def test_clean_run_keeps_almost_everything():
    for seed in range(4):
        t, vals, _, _ = synth_display_run(seed, drop_factor=1.0, shift_by=0.0)
        r = qc.qc_summary(t, vals, vals, checks=("rate", "signal", "range"))
        assert r.percent_removed["total"] < 2.0
        assert len(r.intervals) <= 2


def test_flow_rate_check_counts_and_ramp():
    rng = np.random.default_rng(3)
    # rate ramps up during the first 5 % of the run
    t = np.sort(np.concatenate([rng.uniform(0, 50, 1500), rng.uniform(50, 1000, 95000)]))
    rc = qc.flow_rate_check(t, n_bins=100)
    assert rc.counts.sum() == t.size and rc.counts.size == 100
    assert rc.rates[50] == pytest.approx(rc.counts[50] / np.diff(rc.edges)[50])
    assert rc.flags[:5].all() and rc.ramp[:5].all()
    assert rc.flags[6:].mean() < 0.03
    # a short clog (no events for 2 % of the time) is flagged
    t2 = t[(t < 600) | (t > 620)]
    rc2 = qc.flow_rate_check(t2, n_bins=100)
    assert rc2.flags[60:62].all()


def test_signal_check_spike_and_changepoints():
    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0, 100, 50000))
    v = rng.normal(0.5, 0.05, t.size)
    spike = (t >= 20) & (t < 21)
    v[spike] += 0.2
    sc = qc.signal_stability_check(t, {"A": v}, n_bins=100)
    assert sc.flags["A"][20] and sc.flags["A"].sum() <= 2
    assert np.nanmedian(sc.medians["A"]) == pytest.approx(0.5, abs=0.01)
    # a long shifted stretch (45 % of the run) inflates the MAD; the
    # change-point segments still catch it
    v2 = rng.normal(0.5, 0.05, t.size)
    v2[t >= 55] += 0.05
    sc_cp = qc.signal_stability_check(t, {"A": v2}, n_bins=100, changepoints=True)
    assert sc_cp.flags["A"][57:].all() and not sc_cp.flags["A"][:53].any()
    # small drifts below min_shift are tolerated
    v3 = rng.normal(0.5, 0.05, t.size) + 0.01 * (t / 100.0)
    assert not qc.signal_stability_check(t, {"A": v3}).any_flags.any()


def test_change_points_step_series():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 1, 30), rng.normal(0, 1, 30)])
    segs = qc.change_points(x, 1.0, threshold=4.0)
    cuts = [a for a, _ in segs[1:]]
    assert len(segs) == 3 and abs(cuts[0] - 40) <= 1 and abs(cuts[1] - 70) <= 1
    assert qc.change_points(rng.normal(0, 1, 100), 1.0, threshold=4.0) == [(0, 100)]


def test_dynamic_range_check():
    rng = np.random.default_rng(1)
    a = rng.normal(80000, 6000, 10000)
    a[:30] = 262143.0  # saturated at the detector maximum
    b = rng.normal(1000, 300, 10000)
    b[100:140] = b.min() - 50  # clipped at a floor
    c = rng.normal(500, 100, 10000)  # nothing wrong
    dr = qc.dynamic_range_check({"A": a, "B": b, "C": c}, {"A": 262144.0, "B": (None, 262144.0), "C": 262144.0})
    assert dr.flags[:30].all() and dr.counts["A"] == 30
    assert dr.counts["B"] == 0  # the minimum side is off by default
    assert dr.counts["C"] == 0
    dr_low = qc.dynamic_range_check({"B": b}, check_low=True)
    assert dr_low.counts["B"] == 40 and dr_low.flags[100:140].all()
    # pile-up at the data maximum without metadata
    d = rng.normal(100, 10, 1000)
    d[:5] = 150.0
    assert qc.dynamic_range_check({"D": d}).counts["D"] == 5
    # a $PnR below most of the data is ignored as wrong metadata
    e = rng.normal(5000, 100, 1000)
    dr_e = qc.dynamic_range_check({"E": e}, {"E": 1024.0})
    assert dr_e.counts["E"] == 0 and "E" in dr_e.notes
    # explicit lower limit
    f = np.linspace(-10, 10, 100)
    assert qc.dynamic_range_check({"F": f}, {"F": (-9.0, None)}, check_low=True).counts["F"] == 5


def test_qc_time_ranges():
    t = np.arange(10, dtype=float)
    m = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    ranges = qc.qc_time_ranges(m, t)
    assert ranges[:2] == [(0.0, 3.0), (5.0, 7.0)]
    assert ranges[2][0] == 8.0 and ranges[2][1] > 9.0
    # unsorted input gives the same answer
    perm = np.random.default_rng(0).permutation(10)
    assert qc.qc_time_ranges(m[perm], t[perm]) == ranges
    assert qc.qc_time_ranges(np.zeros(10, bool), t) == []
    # membership of the intervals reproduces the mask
    inside = np.zeros(10, dtype=bool)
    for a, b in ranges:
        inside |= (t >= a) & (t < b)
    assert np.array_equal(inside, m)


def test_summary_combines_checks():
    t, vals, in_drop, in_shift = synth_display_run(4)
    raw = {"FSC-A": vals["FSC-A"] * 262144.0}
    raw["FSC-A"][:25] = 262143.0
    r = qc.qc_summary(t, vals, raw, {"FSC-A": 262144.0})
    assert r.checks == ("rate", "signal", "range")
    assert r.range_counts["FSC-A"] == 25 and not r.range_good[:25].any()
    assert np.array_equal(r.good, r.time_good & r.range_good)
    assert r.percent_removed["range"] == pytest.approx(100 * 25 / t.size)
    s = r.summary()
    assert s["events"] == t.size and s["kept"] == r.n_good and s["bins"] == 100
    table = qc.result_table(r)
    assert [row[0] for row in table[:4]] == ["Flow rate", "Signal stability", "Dynamic range", "Total"]
    assert len(table) == 4 + 100 and len(table[0]) == len(qc.TABLE_HEADER)
    json.dumps(r.settings)  # settings are JSON-friendly


# -------------------------------------------------------------- workspace glue
@pytest.fixture
def qc_sample(tmp_path):
    path, bad, sat = write_qc_fcs(tmp_path / "run.fcs", seed=7, saturated=40)
    ws = Workspace()
    s = ws.add_sample(path)
    assert s.ensure_loaded()
    cells = s.root.add_child(Population("Cells", RectangleGate(["SSC-A"], [None], [1e9])))
    return ws, s, cells, bad, sat


def test_default_parameters(qc_sample):
    _, s, _, _, _ = qc_sample
    assert qc.default_time_parameter(s) == "Time"
    assert qc.default_qc_parameters(s, "Time") == ["FSC-A", "SSC-A", "FITC-A", "PE-A"]
    assert qc.is_time_parameter("HDR-T") and qc.is_time_parameter("TIME2") and not qc.is_time_parameter("FSC-A")


def test_qc_for_population_and_flag_population(qc_sample, tmp_path):
    ws, s, cells, bad, sat = qc_sample
    r = qc.qc_for_population(cells, "Time")
    assert r.sample_id == s.id and r.population_path == "/Cells" and r.time_param == "Time"
    assert r.event_index.size == cells.count
    assert r.range_counts["FSC-A"] == 40 and not r.good[sat].any()
    truly_bad = bad | sat
    assert r.good[~truly_bad].mean() >= 0.95 and (~r.good[truly_bad]).mean() >= 0.90
    created = qc.apply_qc(cells, r)
    pop = created[0]
    assert pop.name == qc.DEFAULT_NAME and pop.parent is cells
    assert isinstance(pop.gate, RectangleGate) and pop.gate.dims == [qc.FLAG_PARAM] and pop.gate.lo == [0.5] and pop.gate.hi == [None]
    assert pop.count == r.n_good
    flag = s.data(qc.FLAG_PARAM)
    assert set(np.unique(flag[np.isfinite(flag)])) <= {0.0, 1.0}
    assert s.computed_info[qc.FLAG_PARAM]["population"] == "/Cells"
    # re-running replaces the population (no duplicate) and keeps the flag name
    created2 = qc.apply_qc(cells, r)
    assert [c.name for c in cells.children] == [qc.DEFAULT_NAME] and created2[0].count == r.n_good
    assert [n for n in s.parameter_names() if n.startswith(qc.FLAG_PARAM)] == [qc.FLAG_PARAM]
    # another population gets its own flag parameter
    r_root = qc.qc_for_population(s.root, "Time")
    root_pop = qc.apply_qc(s.root, r_root, "QC all")[0]
    assert root_pop.gate.dims == [f"{qc.FLAG_PARAM}_2"] and pop.count == r.n_good
    # saved and re-opened, the gate and the computed flag come back
    path = tmp_path / "ws.fdw"
    save_workspace(ws, path)
    ws2 = load_workspace(path)
    s2 = ws2.samples[0]
    assert s2.ensure_loaded()
    p2 = s2.find_population("/Cells/QC pass")
    assert p2 is not None and p2.count == r.n_good
    assert s2.computed_info[qc.FLAG_PARAM]["tool"] == "Data Cleaning"


def test_time_gate_population(qc_sample):
    _, s, cells, bad, _ = qc_sample
    r = qc.qc_for_population(cells, "Time", checks=("rate", "signal"))
    assert len(r.intervals) == 3
    created = qc.apply_qc(cells, r, mode="time")
    union, parts = created[0], created[1:]
    assert isinstance(union.gate, BooleanGate) and union.gate.op == "or"
    assert [p.name for p in parts] == ["QC interval 1", "QC interval 2", "QC interval 3"]
    assert all(isinstance(p.gate, RectangleGate) and p.gate.dims == ["Time"] for p in parts)
    assert union.count == int(r.time_good.sum())
    assert sum(p.count for p in parts) == union.count
    # replacing removes the old interval populations
    qc.apply_qc(cells, r, mode="time")
    assert sorted(c.name for c in cells.children) == sorted(["QC pass", "QC interval 1", "QC interval 2", "QC interval 3"])
    # a single interval gives one plain time gate
    r_clean = qc.qc_for_population(cells, "Time", checks=("range",))
    single = qc.apply_qc(cells, r_clean, "Clean", mode="time")
    assert len(single) == 1 and isinstance(single[0].gate, RectangleGate) and single[0].count == cells.count


def test_without_time_parameter(qc_sample):
    _, s, cells, _, _ = qc_sample
    r = qc.qc_for_population(cells, None)
    assert r.checks == ("signal", "range") and "rate" not in r.percent_removed
    with pytest.raises(ValueError):
        qc.apply_qc(cells, r, mode="time")
    assert qc.apply_qc(cells, r)[0].count == r.n_good
    with pytest.raises(ValueError):
        qc.qc_for_population(cells, "NoSuchTime")
