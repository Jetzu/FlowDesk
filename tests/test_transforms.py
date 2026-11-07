import math

import numpy as np
import pytest

from flowdesk.core.transforms import (
    SCALE_KINDS,
    ArcsinhScale,
    LinearScale,
    LogicleScale,
    LogScale,
    default_biex,
    default_scale_for_channel,
    format_linear,
    power_label,
    scale_from_dict,
)

T = 262144.0


def majors(scale):
    return [t for t in scale.ticks() if t.major]


# ----------------------------------------------------------------------- helpers
def test_power_label_superscripts():
    assert power_label(3) == "10³"
    assert power_label(5) == "10⁵"
    assert power_label(-2) == "10⁻²"
    assert power_label(2, negative=True) == "-10²"


def test_format_linear():
    assert format_linear(0.0, 50000, True) == "0"
    assert format_linear(50000, 50000, True) == "50K"
    assert format_linear(2.5e6, 5e5, True) == "2.5M"
    assert format_linear(1234.0, 100, False) == "1234"
    assert format_linear(0.25, 0.05, False) == "0.25"


# ------------------------------------------------------------------------ linear
def test_linear_forward_inverse():
    s = LinearScale(0, T)
    assert s.forward_scalar(0) == 0.0
    assert s.forward_scalar(T) == 1.0
    assert s.forward_scalar(T / 4) == pytest.approx(0.25)
    x = np.array([-1000.0, 0.0, 12345.0, T])
    np.testing.assert_allclose(s.inverse(s.forward(x)), x)
    assert s.data_min == 0.0 and s.data_max == T


def test_linear_degenerate_range():
    s = LinearScale(5, 5)
    assert s.min == 5 and s.max == 6


def test_linear_ticks_262144():
    s = LinearScale(0, T)
    m = majors(s)
    assert [t.label for t in m] == ["0", "50K", "100K", "150K", "200K", "250K"]
    np.testing.assert_allclose([t.pos for t in m], np.arange(0, 250001, 50000) / T)
    minors = [t for t in s.ticks() if not t.major]
    # minor every 10K, skipping the majors
    assert len(minors) == 21
    assert all(t.label == "" for t in minors)
    assert min(t.pos for t in minors) == pytest.approx(10000 / T)


def test_linear_ticks_small_and_negative():
    assert [t.label for t in majors(LinearScale(0, 1))] == ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    assert [t.label for t in majors(LinearScale(-1000, 5000))] == ["-1000", "0", "1000", "2000", "3000", "4000", "5000"]


def test_linear_describe():
    assert LinearScale(0, T).describe() == "Linear (0 – 262.144K)"


# --------------------------------------------------------------------------- log
def test_log_scale():
    s = LogScale(1, 1e5)
    assert s.decades == pytest.approx(5)
    np.testing.assert_allclose(s.forward(np.array([1, 10, 100, 1e5])), [0, 0.2, 0.4, 1.0])
    # values <= min (including negatives and zero) map to 0
    np.testing.assert_allclose(s.forward(np.array([-5.0, 0.0, 0.5])), 0.0)
    np.testing.assert_allclose(s.inverse(np.array([0.0, 0.6, 1.0])), [1, 1000, 1e5])
    m = majors(s)
    assert [t.label for t in m] == ["10⁰", "10¹", "10²", "10³", "10⁴", "10⁵"]
    assert "0" not in [t.label for t in s.ticks()]
    assert s.describe() == "Log (5.0 decades)"


def test_log_scale_invalid_bounds():
    s = LogScale(-3, 0.5)
    assert s.min == 1.0
    assert s.max == 10.0


# ----------------------------------------------------------------------- logicle
def test_logicle_top_maps_to_one():
    for s in (LogicleScale(), LogicleScale(T=1e6, W=1, M=5, A=1), default_biex(T)):
        assert s.forward_scalar(s.T) == pytest.approx(1.0, abs=1e-9)
        assert s.inverse_scalar(1.0) == pytest.approx(s.T, rel=1e-9)


def test_logicle_zero_position():
    s = LogicleScale(T=T, W=0.5, M=4.5, A=0)
    # zero sits at x1 = (A + W) / (M + A)
    assert s.forward_scalar(0.0) == pytest.approx(0.5 / 4.5, abs=1e-6)
    assert s.inverse_scalar(s.x1) == pytest.approx(0.0, abs=1e-6)


def test_logicle_round_trip_including_negatives():
    s = LogicleScale(T=T, W=0.5, M=4.5, A=0)
    lo = s.data_min
    assert lo < -100
    x = np.concatenate([np.linspace(lo, 1000, 4001), np.geomspace(1000, T, 2000)])
    back = s.inverse(s.forward(x))
    np.testing.assert_allclose(back, x, rtol=1e-4, atol=1e-2)
    # with extra negative decades
    s2 = LogicleScale(T=T, W=1.0, M=4.5, A=1.0)
    x2 = np.linspace(-5000, 5000, 2001)
    assert s2.data_min < -5000
    np.testing.assert_allclose(s2.inverse(s2.forward(x2)), x2, rtol=1e-4, atol=0.5)


def test_logicle_monotonic():
    s = default_biex(T)
    x = np.linspace(s.data_min, T, 50001)
    y = s.forward(x)
    assert np.all(np.diff(y) > 0)
    assert y[0] == pytest.approx(0.0, abs=1e-9)
    # clamps outside the data range
    assert s.forward_scalar(-1e6) == 0.0
    assert s.forward_scalar(1e7) == 1.0


def test_logicle_inverse_clips_input():
    s = LogicleScale()
    assert s.inverse_scalar(2.0) == pytest.approx(T)
    assert s.inverse_scalar(-1.0) == pytest.approx(s.data_min)


def test_logicle_from_biex_width_basis_round_trip():
    for wb in (-10.0, -100.0, -500.0, -1000.0):
        s = LogicleScale.from_biex(T, 4.5, 0.0, wb)
        assert s.width_basis == pytest.approx(wb, rel=1e-9)
        assert s.M == 4.5 and s.A == 0.0
    # W derived from the width basis: W = (M - log10(T / |wb|)) / 2
    s = LogicleScale.from_biex(T, 4.5, 0.0, -100.0)
    assert s.W == pytest.approx((4.5 - math.log10(T / 100)) / 2)


def test_default_biex():
    s = default_biex(T)
    assert s.W == pytest.approx(0.5)
    assert s.M == pytest.approx(math.log10(T) - 1)
    assert s.width_basis == pytest.approx(-100.0)
    small = default_biex(1000)
    assert small.M == 3.0  # pos is clamped to >= 3
    assert default_biex(0).T == T


def test_logicle_w_and_a_clamped():
    assert LogicleScale(W=5, M=4.5).W == 2.25
    assert LogicleScale(W=-1).W == 0.0
    assert LogicleScale(T=-5).T == T
    assert LogicleScale(M=0).M == 4.5
    # A larger than M - 2W gets reduced
    s = LogicleScale(W=2, M=4.5, A=3)
    assert s.A == pytest.approx(0.5)
    # from_biex with a width basis wider than allowed gives W = 0 (not negative)
    s0 = LogicleScale.from_biex(T, 4.5, 0.0, -1.0)
    assert s0.W == 0.0
    assert s0.forward_scalar(T) == pytest.approx(1.0)
    assert s0.data_min == pytest.approx(0.0, abs=1e-6)


def test_logicle_ticks():
    s = default_biex(T)
    m = majors(s)
    labels = [t.label for t in m]
    assert "0" in labels
    for lab in ("10²", "10³", "10⁴", "10⁵"):
        assert lab in labels
    assert "-10²" in labels
    # sorted by position and no two labels closer than the minimum gap
    pos = [t.pos for t in s.ticks()]
    assert pos == sorted(pos)
    mpos = sorted(t.pos for t in m)
    assert min(np.diff(mpos)) >= 0.055
    zero = next(t for t in m if t.label == "0")
    assert zero.pos == pytest.approx(s.forward_scalar(0.0))
    tick_1e3 = next(t for t in m if t.label == "10³")
    assert tick_1e3.pos == pytest.approx(s.forward_scalar(1000.0))
    assert any(not t.major for t in s.ticks())


def test_logicle_describe():
    assert LogicleScale.from_biex(T, 4.5, 0.0, -100).describe() == "Biexponential (pos 4.50, width -100, neg 0.00)"


# ------------------------------------------------------------------------ arcsinh
def test_arcsinh():
    s = ArcsinhScale(T=T, M=4.5, A=0)
    assert s.forward_scalar(0.0) == 0.0
    assert s.forward_scalar(T) == pytest.approx(1.0)
    x = np.linspace(0, T, 1001)
    np.testing.assert_allclose(s.inverse(s.forward(x)), x, rtol=1e-6, atol=1e-6)
    assert s.forward_scalar(-100) == 0.0  # clipped
    assert s.forward_scalar(1e7) == 1.0
    s2 = ArcsinhScale(T=T, M=4.5, A=1)
    assert s2.forward_scalar(0.0) == pytest.approx(1 / 5.5)
    xn = np.linspace(-50, T, 1001)
    np.testing.assert_allclose(s2.inverse(s2.forward(xn)), xn, rtol=1e-6, atol=1e-6)
    labels = [t.label for t in majors(s)]
    assert labels[0] == "0" and "10⁵" in labels
    assert ArcsinhScale(T=-1, M=-1, A=-1).to_dict() == {"kind": "arcsinh", "T": T, "M": 4.5, "A": 0.0}


# -------------------------------------------------------------------- dict round trip
@pytest.mark.parametrize(
    "scale",
    [
        LinearScale(-500, 100000),
        LogScale(10, 1e5),
        LogicleScale(T=1e6, W=0.75, M=5.0, A=0.5),
        default_biex(T),
        ArcsinhScale(T=1e5, M=4, A=0.5),
    ],
)
def test_scale_dict_round_trip(scale):
    d = scale.to_dict()
    again = scale_from_dict(d)
    assert type(again) is type(scale)
    assert again.to_dict() == d
    assert again == scale
    assert hash(again) == hash(scale)
    c = scale.copy()
    assert c == scale and c is not scale
    np.testing.assert_allclose(again.forward(np.array([0.0, 1000.0, 50000.0])), scale.forward(np.array([0.0, 1000.0, 50000.0])))


def test_scale_from_dict_aliases_and_errors():
    s = scale_from_dict({"kind": "biex", "T": T, "pos": 4.5, "neg": 0.0, "width_basis": -250.0})
    assert isinstance(s, LogicleScale)
    assert s.width_basis == pytest.approx(-250.0)
    assert isinstance(scale_from_dict({"kind": "fasinh"}), ArcsinhScale)
    assert scale_from_dict({}) == LinearScale(0, T)
    with pytest.raises(ValueError):
        scale_from_dict({"kind": "weird"})
    assert LinearScale(0, 1) != LogScale(1, 10)
    assert LinearScale(0, 1) != "linear"
    assert [k for k, _ in SCALE_KINDS] == ["linear", "logicle", "log", "arcsinh"]


# ---------------------------------------------------------------- channel defaults
@pytest.mark.parametrize("name", ["FSC-A", "SSC-H", "Time", "HDR-T", "fsc-w", "FITC-W", "Event #", "SSC-B-A"])
def test_default_scale_linear_channels(name):
    s = default_scale_for_channel(name)
    assert isinstance(s, LinearScale)
    assert (s.min, s.max) == (0.0, T)


@pytest.mark.parametrize("name", ["FITC-A", "PE-Cy7-A", "B1-A", "Comp-APC-A"])
def test_default_scale_fluorescence_is_logicle(name):
    s = default_scale_for_channel(name, top=T)
    assert isinstance(s, LogicleScale)
    assert s == default_biex(T)


def test_default_scale_log_amplified():
    s = default_scale_for_channel("FL1", top=1024, log_amplified=True)
    assert isinstance(s, LogScale)
    assert (s.min, s.max) == (1.0, 1024.0)
    s2 = default_scale_for_channel("FL1", top=1e5, log_amplified=True)
    assert (s2.min, s2.max) == (10.0, 1e5)
    # log amplification wins over a linear name
    assert isinstance(default_scale_for_channel("FSC-A", log_amplified=True), LogScale)


def test_default_scale_display_hints():
    assert isinstance(default_scale_for_channel("FITC-A", "LIN"), LinearScale)
    assert isinstance(default_scale_for_channel("FITC-A", "Linear,0,1"), LinearScale)
    assert isinstance(default_scale_for_channel("FSC-A", "LOG"), LogicleScale)
    assert isinstance(default_scale_for_channel("SSC-A", "Logarithmic,4,0.1"), LogicleScale)
    assert default_scale_for_channel("FITC-A", "LIN", top=1000).max == 1000.0
