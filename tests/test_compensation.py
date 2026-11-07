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
