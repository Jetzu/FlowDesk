import numpy as np
import pytest

from flowdesk.core.compensation import (
    ACQUISITION_DEFINED,
    SpilloverMatrix,
    estimate_spillover,
    otsu_threshold,
    split_positive,
)
from flowdesk.core.fcs import FCSFile
from tests.conftest import write_fcs

PARAMS = ["FITC-A", "PE-A", "APC-A"]
S_TRUE = np.array(
    [
        [1.00, 0.15, 0.02],
        [0.05, 1.00, 0.20],
        [0.00, 0.08, 1.00],
    ]
)


def test_apply_equals_observed_times_inverse():
    rng = np.random.default_rng(0)
    events = rng.normal(1000, 300, (500, 5)).astype(np.float32)
    names = ["FSC-A", "FITC-A", "SSC-A", "PE-A", "APC-A"]
    m = SpilloverMatrix("m", PARAMS, S_TRUE)
    out = m.apply(events, names)
    assert out.dtype == np.float32 and out.shape == (500, 3)
    observed = events[:, [1, 3, 4]].astype(np.float64)
    np.testing.assert_allclose(out, observed @ np.linalg.inv(S_TRUE), rtol=1e-5, atol=1e-2)
    # compensating spilled-over signals recovers the true ones
    true = rng.normal(5000, 1000, (200, 3))
    np.testing.assert_allclose(m.apply(true @ S_TRUE, PARAMS), true, rtol=1e-4, atol=0.1)


def test_apply_missing_parameter():
    m = SpilloverMatrix("m", PARAMS, S_TRUE)
    assert m.apply(np.zeros((3, 2)), ["FITC-A", "PE-A"]) is None


def test_singular_matrix_uses_pinv():
    m = SpilloverMatrix("sing", ["A", "B"], [[1, 1], [1, 1]])
    np.testing.assert_allclose(m.inverse(), np.linalg.pinv(np.ones((2, 2))))


def test_names():
    m = SpilloverMatrix("m", PARAMS, S_TRUE)
    assert m.compensated_name("PE-A") == "Comp-PE-A"
    assert m.compensated_names == ["Comp-FITC-A", "Comp-PE-A", "Comp-APC-A"]
    assert m.base_name("Comp-PE-A") == "PE-A"
    assert m.base_name("PE-A") is None
    assert m.base_name("Comp-BV421-A") is None  # not a parameter of the matrix
    m2 = SpilloverMatrix("m", PARAMS, S_TRUE, prefix="", suffix=" [comp]")
    assert m2.compensated_name("FITC-A") == "FITC-A [comp]"
    assert m2.base_name("FITC-A [comp]") == "FITC-A"
    assert m2.base_name("FITC-A") is None
    m3 = SpilloverMatrix("m", PARAMS, S_TRUE, prefix="<", suffix=">")
    assert m3.base_name("<APC-A>") == "APC-A"
    assert m3.base_name("<APC-A") is None


def test_identity_copy_and_dict():
    ident = SpilloverMatrix.identity(PARAMS)
    assert ident.is_identity and ident.name == "Identity"
    m = SpilloverMatrix("m", PARAMS, S_TRUE, suffix="_c")
    assert not m.is_identity
    c = m.copy("other")
    assert c.name == "other" and c == m and c.matrix is not m.matrix
    c.matrix[0, 1] = 0.9
    assert m.matrix[0, 1] == 0.15
    again = SpilloverMatrix.from_dict(m.to_dict())
    assert again == m and again.suffix == "_c" and again.prefix == "Comp-" and again.name == "m"
    assert m != SpilloverMatrix("m", ["FITC-A", "APC-A", "PE-A"], S_TRUE)
    assert m != "m"
    # usable as dict key / set member
    assert len({m, again}) == 2


def test_matrix_reshape_from_flat():
    m = SpilloverMatrix("m", ["A", "B"], [1, 0.1, 0.2, 1])
    np.testing.assert_allclose(m.matrix, [[1, 0.1], [0.2, 1]])


def test_from_fcs(tmp_path):
    p = write_fcs(tmp_path / "s.fcs", np.ones((3, 2), np.float32), ["A", "B"], extra_keywords={"$SPILLOVER": "2,A,B,1,0.3,0.1,1"})
    m = SpilloverMatrix.from_fcs(FCSFile.read(p))
    assert m.name == ACQUISITION_DEFINED
    assert m.params == ["A", "B"]
    np.testing.assert_allclose(m.matrix, [[1, 0.3], [0.1, 1]])
    p2 = write_fcs(tmp_path / "n.fcs", np.ones((3, 2), np.float32), ["A", "B"])
    assert SpilloverMatrix.from_fcs(FCSFile.read(p2)) is None


# ------------------------------------------------------------------ estimation
def test_otsu_threshold_two_clusters():
    rng = np.random.default_rng(1)
    low, high = rng.normal(100, 10, 3000), rng.normal(1000, 50, 1000)
    v = np.concatenate([low, high])
    thr = otsu_threshold(v)
    # every threshold inside the empty gap is equally good; the first one wins
    assert low.max() < thr < high.min()
    assert (v < thr).sum() == 3000
    assert otsu_threshold(np.array([])) == 0.0
    assert otsu_threshold(np.array([np.nan, np.inf])) == 0.0
    # NaNs ignored
    assert otsu_threshold(np.concatenate([v, [np.nan] * 10])) == pytest.approx(thr)
    # overlapping-but-bimodal data: threshold lands between the modes
    mixed = np.concatenate([rng.normal(0, 1, 5000), rng.normal(6, 1, 5000)])
    assert 2.0 < otsu_threshold(mixed) < 4.0


def test_split_positive():
    rng = np.random.default_rng(2)
    neg = rng.normal(0, 100, 2000)
    pos = rng.lognormal(np.log(30000), 0.3, 500)
    mask = split_positive(np.concatenate([neg, pos]))
    assert mask[:2000].sum() == 0
    assert mask[2000:].all()
    # unimodal data falls back to the brightest 10% if otsu leaves too few
    one = np.concatenate([np.full(995, 100.0), np.full(5, 50000.0)])
    m1 = split_positive(one)
    assert m1.sum() >= 5


def _single_stain_controls(seed=0, n=4000, noise=60.0):
    rng = np.random.default_rng(seed)
    k = len(PARAMS)
    controls = {}
    for i, p in enumerate(PARAMS):
        true = rng.normal(0, noise, (n, k))
        positive = rng.random(n) < 0.5
        true[positive, i] += rng.lognormal(np.log(40000), 0.3, positive.sum())
        controls[p] = true @ S_TRUE + rng.normal(0, 20, (n, k))
    unstained = rng.normal(0, noise, (n, k)) @ S_TRUE + rng.normal(0, 20, (n, k))
    return controls, unstained


def test_estimate_spillover_with_unstained():
    controls, unstained = _single_stain_controls()
    m = estimate_spillover(controls, PARAMS, unstained=unstained)
    assert m.name == "Computed"
    assert m.params == PARAMS
    np.testing.assert_allclose(np.diag(m.matrix), 1.0)
    assert np.max(np.abs(m.matrix - S_TRUE)) < 0.02


def test_estimate_spillover_without_unstained():
    controls, _ = _single_stain_controls(seed=5)
    m = estimate_spillover(controls, PARAMS, name="Auto")
    assert m.name == "Auto"
    assert np.max(np.abs(m.matrix - S_TRUE)) < 0.02
    # compensating a control with the estimated matrix removes the spillover
    comp = m.apply(controls["PE-A"], PARAMS)
    pos = controls["PE-A"][:, 1] > 10000
    assert abs(np.median(comp[pos, 2])) < 500  # was ~0.2 * 40000 = 8000 before
    assert np.median(controls["PE-A"][pos, 2]) > 6000


def test_estimate_spillover_missing_or_tiny_controls():
    controls, unstained = _single_stain_controls()
    partial = {"FITC-A": controls["FITC-A"], "PE-A": controls["PE-A"][:10]}
    m = estimate_spillover(partial, PARAMS, unstained=unstained)
    assert np.max(np.abs(m.matrix[0] - S_TRUE[0])) < 0.02
    np.testing.assert_array_equal(m.matrix[1], [0, 1, 0])  # too few events -> identity row
    np.testing.assert_array_equal(m.matrix[2], [0, 0, 1])  # missing -> identity row
    assert np.all(m.matrix >= 0)


# ------------------------------------------------------------------- real file
def test_real_control_spillover(real_comp_control):
    f = FCSFile.read(real_comp_control)
    m = SpilloverMatrix.from_fcs(f)
    assert m is not None and m.name == ACQUISITION_DEFINED
    assert set(m.params) <= set(f.channel_names)
    out = m.apply(f.events, f.channel_names)
    assert out.shape == (5000, len(m.params))
    assert np.all(np.isfinite(out))
    observed = f.events[:, [f.channel_names.index(p) for p in m.params]].astype(np.float64)
    np.testing.assert_allclose(out, observed @ np.linalg.inv(m.matrix), rtol=1e-4, atol=0.5)


# ------------------------------------------------------- gates & diagnostics
from pathlib import Path

from flowdesk.core.compensation import (
    ControlGates,
    diagnostics_warnings,
    group_matrix_name,
    mark_group_compensation,
    matrix_from_fcs_file,
    matrix_from_text,
    matrix_to_text,
    pending_group_compensation,
    read_matrix_mtx,
    refine_spillover,
    residual_spill,
    robust_slope,
    set_group_matrix,
    spillover_spreading,
    write_matrix_mtx,
)
from flowdesk.core.fcs import Channel


def test_estimate_spillover_diagnostics():
    controls, unstained = _single_stain_controls(seed=11)
    m = estimate_spillover(controls, PARAMS, unstained=unstained)
    assert set(m.diagnostics) == set(PARAMS)
    for i, p in enumerate(PARAMS):
        d = m.diagnostics[p]
        assert d.status == "ok" and d.ok and d.used
        assert d.method == "otsu" and d.negative == "unstained"
        assert d.n_events == 4000
        assert 1700 < d.n_pos < 2300  # half of the events are positive
        assert d.n_neg == 4000  # the unstained control
        assert d.pos_median > 30000 and abs(d.neg_median) < 200
        assert d.stain_index > 100
        assert d.pos_medians.shape == (3,) and d.pos_range[0] <= d.pos_median <= d.pos_range[1]
    assert diagnostics_warnings(m.diagnostics) == []
    # internal negatives without an unstained control
    m2 = estimate_spillover(controls, PARAMS)
    assert all(d.negative == "internal" and 1700 < d.n_neg < 2300 for d in m2.diagnostics.values())


def test_gate_overrides_change_the_matrix():
    controls, unstained = _single_stain_controls(seed=12)
    auto = estimate_spillover(controls, PARAMS, unstained=unstained)
    # a positive gate on the dim half of the positives only
    fitc = controls["FITC-A"][:, 0]
    lo, hi = np.percentile(fitc[fitc > 10000], [0, 50])
    gated = estimate_spillover(controls, PARAMS, unstained=unstained, gates={"FITC-A": ControlGates((lo, hi))})
    d = gated.diagnostics["FITC-A"]
    assert d.method == "gates" and d.n_pos < auto.diagnostics["FITC-A"].n_pos
    assert d.pos_median < auto.diagnostics["FITC-A"].pos_median
    # explicit own negatives replace the universal negative for that control
    gated2 = estimate_spillover(controls, PARAMS, unstained=unstained, gates={"FITC-A": ControlGates((lo, hi), (-1e6, 5000))})
    assert gated2.diagnostics["FITC-A"].negative == "internal"
    # a positive gate that also takes in negatives moves the matrix
    moved = estimate_spillover(controls, PARAMS, unstained=unstained, gates={"PE-A": ControlGates.from_threshold(-50.0)})
    assert not np.allclose(moved.matrix[1], auto.matrix[1])
    assert moved.diagnostics["PE-A"].pos_median < auto.diagnostics["PE-A"].pos_median
    # explicit masks
    pos = controls["APC-A"][:, 2] > 10000
    masked = estimate_spillover(controls, PARAMS, masks={"APC-A": (pos, ~pos)})
    assert masked.diagnostics["APC-A"].method == "masks"
    assert masked.diagnostics["APC-A"].n_pos == int(pos.sum())
    np.testing.assert_allclose(masked.matrix[2], auto.matrix[2], atol=0.01)
    # gates survive a settings round trip (open ends as "inf")
    g = ControlGates.from_threshold(123.0)
    assert ControlGates.from_dict(g.to_dict()) == g and ControlGates().is_auto


def test_control_without_positives_is_flagged():
    controls, unstained = _single_stain_controls(seed=13)
    bad = dict(controls)
    bad["PE-A"] = unstained.copy()  # an unstained tube assigned as the PE control
    m = estimate_spillover(bad, PARAMS, unstained=unstained)
    d = m.diagnostics["PE-A"]
    assert d.status == "no positives" and not d.used
    assert "no distinct positive population" in d.message
    np.testing.assert_array_equal(m.matrix[1], [0, 1, 0])  # row left uncompensated
    warnings = diagnostics_warnings(m.diagnostics)
    assert len(warnings) == 1 and warnings[0].startswith("PE-A")
    # too few events / missing controls are reported too
    m2 = estimate_spillover({"FITC-A": controls["FITC-A"][:5]}, PARAMS)
    assert m2.diagnostics["FITC-A"].status == "too few events"
    assert m2.diagnostics["APC-A"].status == "missing"
    # an explicit positive gate holding no events
    m3 = estimate_spillover(controls, PARAMS, gates={"FITC-A": ControlGates((1e7, 2e7))})
    assert m3.diagnostics["FITC-A"].status == "no positives"


def test_weak_control_is_flagged_but_used():
    rng = np.random.default_rng(14)
    n = 4000
    true = rng.normal(0, 400, (n, 3))
    pos = rng.random(n) < 0.5
    true[pos, 0] += 5000  # dim stain: stain index ~6
    controls = {"FITC-A": true @ S_TRUE}
    m = estimate_spillover(controls, PARAMS)
    d = m.diagnostics["FITC-A"]
    assert d.status == "weak" and d.used and 3 < d.stain_index < 10
    assert abs(m.matrix[0, 1] - S_TRUE[0, 1]) < 0.03


# ------------------------------------------------------------- AutoSpill
S_NEG = np.array([[1.0, 0.15, 0.02], [0.05, 1.0, -0.03], [0.0, 0.08, 1.0]])


def _heteroscedastic_controls(seed=3, n=6000):
    rng = np.random.default_rng(seed)
    out = {}
    for i, p in enumerate(PARAMS):
        t = rng.normal(0, 50, (n, 3))
        pos = rng.random(n) < 0.5
        t[pos, i] += rng.lognormal(np.log(40000), 0.4, pos.sum())
        clean = t @ S_NEG
        # detector noise grows with the signal (heteroscedastic positives)
        out[p] = clean + rng.normal(0, 1, clean.shape) * (20 + 0.05 * np.abs(clean))
    unstained = np.random.default_rng(seed + 1).normal(0, 50, (n, 3)) @ S_NEG
    return out, unstained


def test_robust_slope_ignores_outliers():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 50000, 3000)
    y = 0.2 * x + 100 + rng.normal(0, 1, 3000) * (10 + 0.02 * x)
    y[:150] += 30000  # 5 % gross outliers
    assert abs(robust_slope(x, y) - 0.2) < 0.005
    assert abs(robust_slope(x, y, heteroscedastic=False) - 0.2) < 0.01
    assert abs(np.polyfit(x, y, 1)[0] - 0.2) > 0.005  # least squares is pulled away
    assert robust_slope(np.ones(10), np.arange(10.0)) == 0.0


def test_refine_spillover_converges_and_beats_median_difference():
    controls, unstained = _heteroscedastic_controls()
    median = estimate_spillover(controls, PARAMS, unstained=unstained)
    refined = refine_spillover(controls, PARAMS, median, unstained=unstained, tol=1e-4)
    rep = refined.report
    assert rep.converged and rep.iterations >= 1 and rep.history[-1] < 1e-4
    err_median = np.abs(median.matrix - S_NEG).max()
    err_refined = np.abs(refined.matrix - S_NEG).max()
    assert err_refined < 0.006 < err_median  # the median estimate cannot go negative
    assert refined.matrix[1, 2] < -0.02  # the negative coefficient is recovered
    np.testing.assert_allclose(np.diag(refined.matrix), 1.0)
    assert rep.max_residual < 0.002 and rep.rms_residual <= rep.max_residual
    assert "converged" in rep.summary() and len(rep.worst(2)) == 2
    assert set(refined.diagnostics) == set(PARAMS)
    # without a starting matrix (robust regression start) it gets there too
    alone = refine_spillover(controls, PARAMS)
    assert np.abs(alone.matrix - S_NEG).max() < 0.006 and alone.name == "AutoSpill"


def test_refine_keeps_rows_without_controls():
    controls, unstained = _heteroscedastic_controls(seed=4, n=3000)
    partial = {"FITC-A": controls["FITC-A"]}
    start = SpilloverMatrix("s", PARAMS, S_TRUE)
    r = refine_spillover(partial, PARAMS, start, max_iter=5)
    np.testing.assert_allclose(r.matrix[1:], S_TRUE[1:])
    assert abs(r.matrix[0, 1] - S_NEG[0, 1]) < 0.01


# ------------------------------------------------------- spreading matrix
def test_spillover_spreading_known_spread():
    rng = np.random.default_rng(7)
    n, F, s0, k, a = 8000, 40000.0, 50.0, 0.5, 0.2
    pos = rng.random(n) < 0.5
    t0 = np.where(pos, F, 0.0)
    obs0 = t0 + rng.normal(0, 1, n)
    # photon-statistics spreading of the spill into detector B
    obs1 = a * t0 + rng.normal(0, 1, n) * np.sqrt(s0 ** 2 + k ** 2 * t0)
    params = ["A-A", "B-A"]
    m = SpilloverMatrix("m", params, [[1, a], [0, 1]])
    ssm = spillover_spreading({"A-A": np.column_stack([obs0, obs1])}, params, m)
    expected = (np.sqrt(s0 ** 2 + k ** 2 * F) - s0) / np.sqrt(F)
    assert ssm[0, 1] == pytest.approx(expected, rel=0.08)
    assert ssm[0, 0] == 0.0
    assert np.isnan(ssm[1]).all()  # no control for B
    # on the real-shaped synthetic set: finite, non-negative, zero diagonal
    controls, unstained = _single_stain_controls(seed=15)
    est = estimate_spillover(controls, PARAMS, unstained=unstained)
    full = spillover_spreading(controls, PARAMS, est, unstained)
    assert full.shape == (3, 3) and np.all(np.diag(full) == 0) and np.all(full >= 0)


def test_residual_spill_signs():
    controls, unstained = _single_stain_controls(seed=16)
    good = SpilloverMatrix("good", PARAMS, S_TRUE)
    res = residual_spill(controls, PARAMS, good, unstained)
    off = res[~np.eye(3, dtype=bool)]
    assert np.all(np.abs(off) < 0.01)
    under = S_TRUE.copy()
    under[1, 2] = 0.0  # PE -> APC not compensated
    over = S_TRUE.copy()
    over[0, 1] = 0.30  # FITC -> PE over-compensated
    assert residual_spill(controls, PARAMS, SpilloverMatrix("u", PARAMS, under), unstained)[1, 2] > 0.1
    assert residual_spill(controls, PARAMS, SpilloverMatrix("o", PARAMS, over), unstained)[0, 1] < -0.1


# ------------------------------------------------------- channel matching
def test_stain_named_matrix_matches_channels():
    chans = [Channel(1, "FSC-A"), Channel(2, "FL1-A", "CD4 FITC"), Channel(3, "FL2-A", "CD8 PE")]
    m = SpilloverMatrix("s", ["CD4 FITC", "cd8 pe"], [[1, 0.2], [0.05, 1]])
    assert m.resolve_params(chans) == {"CD4 FITC": "FL1-A", "cd8 pe": "FL2-A"}
    renamed = m.for_channels(chans)
    assert renamed.params == ["FL1-A", "FL2-A"] and renamed.name == "s"
    np.testing.assert_allclose(renamed.matrix, m.matrix)
    plain = SpilloverMatrix("p", ["FL1-A", "FL2-A"], np.eye(2))
    assert plain.for_channels(chans) is plain
    missing = SpilloverMatrix("x", ["FL1-A", "FL3-A", "FL4-A"], np.eye(3))
    assert missing.missing_params(chans) == ["FL3-A", "FL4-A"]
    assert missing.for_channels(chans) is None
    assert missing.missing_params(["FL1-A", "FL3-A", "FL4-A"]) == []  # plain names work too


# ------------------------------------------------------------- matrix files
def test_mtx_xml_roundtrip(tmp_path):
    m = SpilloverMatrix("Panel 1", PARAMS, S_NEG, prefix="Comp-", suffix="")
    path = tmp_path / "panel.mtx"
    write_matrix_mtx(m, path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<?xml") and "gatingML" in text and "spilloverMatrix" in text
    assert "http://www.isac-net.org/std/Gating-ML/v2.0/transformations" in text
    back = read_matrix_mtx(path)
    assert back == m and back.name == "Panel 1" and back.prefix == "Comp-"
    # a file whose root is the spilloverMatrix itself
    inner = text[text.index("<transforms:spilloverMatrix"): text.index("</transforms:spilloverMatrix>") + len("</transforms:spilloverMatrix>")]
    ns = ('xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations" '
          'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes" ')
    bare = tmp_path / "bare.mtx"
    bare.write_text(inner.replace("<transforms:spilloverMatrix ", "<transforms:spilloverMatrix " + ns, 1), encoding="utf-8")
    assert read_matrix_mtx(bare) == m
    bad = tmp_path / "bad.mtx"
    bad.write_text("<gatingML/>", encoding="utf-8")
    with pytest.raises(ValueError):
        read_matrix_mtx(bad)


def test_mtx_text_roundtrip(tmp_path):
    m = SpilloverMatrix("t", PARAMS, S_TRUE)
    path = tmp_path / "text.mtx"
    write_matrix_mtx(m, path, fmt="text")
    lines = path.read_text().splitlines()
    assert lines[0].split("\t") == PARAMS  # first row: the parameters
    back = read_matrix_mtx(path)
    assert back == m and back.name == "text"
    # with a title line and row labels
    titled = tmp_path / "titled.txt"
    titled.write_text("My matrix\n" + matrix_to_text(m, "\t"), encoding="utf-8")
    t2 = read_matrix_mtx(titled)
    assert t2 == m and t2.name == "My matrix"


def test_matrix_text_parsing_variants():
    m = SpilloverMatrix("x", ["A", "B"], [[1, 0.125], [0.03, 1]])
    pct = matrix_to_text(m, "\t", percent=True)
    assert "12.5" in pct and matrix_from_text(pct) == m
    assert matrix_from_text(matrix_to_text(m, ",", labels=False)) == m
    european = ";A;B\nA;100;12,5\nB;3;100\n"
    assert matrix_from_text(european) == m
    with pytest.raises(ValueError):
        matrix_from_text("A\tB\n1\t2\t3\n")
    with pytest.raises(ValueError):
        matrix_from_text("")


def test_matrix_from_fcs_file(tmp_path):
    p = write_fcs(tmp_path / "acq.fcs", np.ones((3, 2), np.float32), ["A", "B"], extra_keywords={"$SPILLOVER": "2,A,B,1,0.3,0.1,1"})
    m = matrix_from_fcs_file(p)
    assert m.name == "acq" and m.params == ["A", "B"]
    np.testing.assert_allclose(m.matrix, [[1, 0.3], [0.1, 1]])
    with pytest.raises(ValueError):
        matrix_from_fcs_file(write_fcs(tmp_path / "none.fcs", np.ones((3, 2), np.float32), ["A", "B"]))


# ------------------------------------------------------------ group matrices
def test_group_matrix_records(tmp_path):
    from flowdesk.core.model import Workspace

    ws = Workspace()
    samples = [ws.add_sample(write_fcs(tmp_path / f"s{i}.fcs", np.ones((5, 2), np.float32), ["A", "B"])) for i in range(3)]
    g = ws.add_group("Panel", samples[:1])
    m = SpilloverMatrix("M", ["A", "B"], [[1, 0.1], [0, 1]])
    ws.matrices["M"] = m
    set_group_matrix(ws, g, "M", [samples[0].id])
    assert group_matrix_name(ws, g) == "M"
    assert pending_group_compensation(ws) == []
    g.sample_ids.add(samples[1].id)  # a sample joins later
    pend = pending_group_compensation(ws)
    assert [(x[0], x[1], x[2].name) for x in pend] == [(g, samples[1], "M")]
    mark_group_compensation(ws, g, [samples[1].id])
    assert pending_group_compensation(ws) == []
    # a renamed group keeps its matrix (found by id)
    g.name = "Renamed"
    assert group_matrix_name(ws, g) == "M"
    # plain matrix names: every member lacking the matrix is pending
    ws.settings["group_matrices"] = {"Renamed": "M"}
    assert {x[1] for x in pending_group_compensation(ws)} == {samples[0], samples[1]}
    samples[0].set_compensation(m.copy())
    assert {x[1] for x in pending_group_compensation(ws)} == {samples[1]}
    set_group_matrix(ws, g, None)
    assert group_matrix_name(ws, g) is None and "group_matrices" not in ws.settings


# ---------------------------------------------- real controls vs FACSDiva
DIVA_DIR = Path("/Users/lukasgerstung/Coding/Plotting/FACS-Data/Experiment_20260430")
DIVA_CONTROLS = {
    "Unstained": "Compensation Controls_Unstained Control.fcs",
    "FITC-A": "Compensation Controls_FITC Stained Control.fcs",
    "PE-A": "Compensation Controls_PE Stained Control.fcs",
    "APC-A": "Compensation Controls_APC Stained Control.fcs",
}


def test_real_controls_match_facsdiva_matrix():
    """The specimens of this experiment carry the matrix FACSDiva computed
    from these controls ($SPILLOVER); the wizard's estimate stays within
    1.5 percentage points of it."""
    files = {k: DIVA_DIR / v for k, v in DIVA_CONTROLS.items()}
    specimen = DIVA_DIR / "Specimen_001_Mix_1.fcs"
    if not all(p.exists() for p in [*files.values(), specimen]):
        pytest.skip("real FACSDiva experiment not available")
    fcs = {k: FCSFile.read(p) for k, p in files.items()}
    params = ["FITC-A", "APC-A", "PE-A"]

    def cols(f):
        return f.events[:, [f.channel_index(p) for p in params]].astype(np.float64)

    diva = matrix_from_fcs_file(specimen)
    idx = [diva.params.index(p) for p in params]
    ref = diva.matrix[np.ix_(idx, idx)]
    m = estimate_spillover({p: cols(fcs[p]) for p in params}, params, cols(fcs["Unstained"]))
    assert np.abs(m.matrix - ref).max() < 0.015
    assert all(d.status == "ok" for d in m.diagnostics.values())
    # AutoSpill is a different estimator; it stays close (within 4 points)
    r = refine_spillover({p: cols(fcs[p]) for p in params}, params, m, unstained=cols(fcs["Unstained"]))
    assert r.report.converged and np.abs(r.matrix - ref).max() < 0.04
