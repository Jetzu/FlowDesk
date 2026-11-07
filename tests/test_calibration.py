"""MESF bead calibration (flowdesk.core.calibration) on synthetic bead sets."""
import json

import numpy as np
import pytest

from flowdesk.core import calibration as cal
from flowdesk.core.fcs import Channel, write_fcs
from flowdesk.core.model import Workspace
from flowdesk.core.serialization import load_workspace, save_workspace
from flowdesk.core.transforms import LinearScale, LogScale, default_biex

SLOPE, INTERCEPT = 1.05, 0.3  # log10(MESF) = SLOPE * log10(I) + INTERCEPT
KIT_MESF = [800.0, 2500.0, 8000.0, 25000.0, 80000.0, 250000.0]


def peak_intensities(mesf=KIT_MESF, slope=SLOPE, intercept=INTERCEPT):
    return [10 ** ((np.log10(m) - intercept) / slope) for m in mesf]


def synth_beads(n_per_peak=3000, cv=0.05, blank=False, seed=1, mesf=KIT_MESF, slope=SLOPE, intercept=INTERCEPT):
    """Bead intensities: one lognormal peak with the given CV per MESF value
    (plus an unlabelled blank bead near zero), shuffled."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1 + cv ** 2))
    parts = [rng.lognormal(np.log(i), sigma, n_per_peak) for i in peak_intensities(mesf, slope, intercept)]
    if blank:
        parts.append(rng.normal(30.0, 40.0, n_per_peak))
    v = np.concatenate(parts) + rng.normal(0, 5.0, n_per_peak * len(parts))
    rng.shuffle(v)
    return v


def write_bead_fcs(path, seed=1, blank=True):
    v = synth_beads(seed=seed, blank=blank)
    rng = np.random.default_rng(seed + 10)
    events = np.column_stack([rng.normal(60000, 3000, v.size), rng.normal(20000, 2000, v.size), v]).astype(np.float32)
    write_fcs(path, events, [Channel(1, "FSC-A"), Channel(2, "SSC-A"), Channel(3, "FITC-A", "Beads")])
    return path


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_six_peak_fit_recovers_slope_and_intercept(seed):
    v = synth_beads(seed=seed)
    peaks = cal.find_bead_peaks(v, 6, log=True)
    assert len(peaks) == 6
    medians = [p.median for p in peaks]
    assert medians == sorted(medians)
    for p, true in zip(peaks, peak_intensities()):
        assert abs(p.median / true - 1) < 0.01
        assert 4.0 < p.cv < 6.0 and abs(p.count - 3000) < 30
    fit = cal.fit_calibration(medians, KIT_MESF)
    assert abs(fit.slope / SLOPE - 1) < 0.02
    assert abs(fit.intercept / INTERCEPT - 1) < 0.02
    assert fit.r2 > 0.9999 and all(fit.used)


def test_display_space_peaks_and_blank_exclusion():
    v = synth_beads(blank=True, seed=4)
    scale = default_biex(262144.0)
    peaks = cal.find_bead_peaks(v, 7, scale=scale)
    assert len(peaks) == 7
    assert all(0.0 <= p.position <= 1.0 for p in peaks)
    assert abs(peaks[0].median - 30) < 10  # the blank bead
    medians = [p.median for p in peaks]
    # the blank (MESF 0) can never enter a log fit
    fit = cal.fit_calibration(medians, [0.0] + KIT_MESF)
    assert fit.used == [False] + [True] * 6
    assert abs(fit.slope / SLOPE - 1) < 0.02 and abs(fit.intercept / INTERCEPT - 1) < 0.02
    # with a nominal value for the blank, exclude_blank still drops it
    fit2 = cal.fit_calibration(medians, [100.0] + KIT_MESF, exclude_blank=True)
    assert fit2.used[0] is False and abs(fit2.slope / SLOPE - 1) < 0.02
    # an explicit use mask
    fit3 = cal.fit_calibration(medians, [0.0] + KIT_MESF, use=[True, True, True, True, True, True, False])
    assert fit3.used == [False, True, True, True, True, True, False]
    with pytest.raises(ValueError):
        cal.fit_calibration([100.0, 200.0], [0.0, 1000.0])
    with pytest.raises(ValueError):
        cal.fit_calibration([100.0], [1000.0, 2000.0])


def test_fewer_peaks_than_requested_and_errors():
    v = synth_beads(mesf=KIT_MESF[:3], seed=5)
    peaks = cal.find_bead_peaks(v, 6)
    assert len(peaks) == 3
    with pytest.raises(ValueError):
        cal.find_bead_peaks(np.array([1.0, 2.0]), 2)


def test_conversion_roundtrip_and_dict():
    fit = cal.CalibrationFit(SLOPE, INTERCEPT, 1.0, "FITC-A", [1.0, 2.0], [3.0, 4.0], [True, True])
    x = np.array([-500.0, -1.0, 0.0, 1.0, 300.0, 70000.0])
    m = cal.mesf_from_intensity(x, fit)
    assert np.all(np.diff(m) > 0)  # sign-symmetric, monotonic through zero
    assert m[2] == 0.0 and m[0] < 0
    assert m[4] == pytest.approx(10 ** INTERCEPT * 300.0 ** SLOPE)
    assert np.allclose(cal.intensity_from_mesf(m, fit), x)
    assert fit.to_mesf(300.0) == pytest.approx(m[4])
    d = cal.calibration_to_dict(fit)
    json.dumps(d)
    back = cal.calibration_from_dict(d)
    assert back == fit
    with pytest.raises(ValueError):
        cal.calibration_from_dict({"param": "x"})


def test_save_load_file(tmp_path):
    fit = cal.fit_calibration(peak_intensities(), KIT_MESF, param="FITC-A", source="beads.fcs")
    path = cal.save_calibration(tmp_path / "cal.json", fit)
    back = cal.load_calibration(path)
    assert back.slope == pytest.approx(SLOPE) and back.intercept == pytest.approx(INTERCEPT)
    assert back.param == "FITC-A" and back.source == "beads.fcs"
    (tmp_path / "other.json").write_text(json.dumps({"kind": "something-else", "slope": 1, "intercept": 0}))
    with pytest.raises(ValueError):
        cal.load_calibration(tmp_path / "other.json")


def test_apply_to_sample_creates_computed_mesf_parameter(tmp_path):
    ws = Workspace()
    s = ws.add_sample(write_bead_fcs(tmp_path / "beads.fcs"))
    assert s.ensure_loaded()
    peaks = cal.peaks_for_population(s.root, "FITC-A", 7)
    assert len(peaks) == 7
    fit = cal.fit_calibration([p.median for p in peaks], [0.0] + KIT_MESF, param="FITC-A")
    assert abs(fit.slope / SLOPE - 1) < 0.02 and abs(fit.intercept / INTERCEPT - 1) < 0.02
    name = cal.apply_calibration(s, fit)
    assert name == "FITC-A MESF" and name in s.parameter_names() and name in s.computed
    raw = s.data("FITC-A").astype(float)
    mesf = s.data(name).astype(float)
    order = np.argsort(raw)
    assert np.all(np.diff(mesf[order]) >= 0)  # monotonic in the intensity
    assert s.computed_info[name]["calibration"]["slope"] == pytest.approx(fit.slope)
    assert s.computed_info[name]["source"] == "FITC-A"
    # the calibrated parameter keeps a log-type axis
    assert s.scale(name).kind == s.scale("FITC-A").kind == "logicle"
    assert s.scale(name).data_max == pytest.approx(cal.mesf_from_intensity(262144.0, fit), rel=1e-3)
    disp = s.display(name)
    assert 0.0 <= float(np.nanmin(disp)) and float(np.nanmax(disp)) <= 1.0
    # peak medians in MESF reproduce the kit values
    for p, kit in zip(peaks[1:], KIT_MESF):
        assert abs(fit.to_mesf(p.median) / kit - 1) < 0.02
    cal.store_calibration(ws, fit)
    assert cal.stored_calibration(ws, "FITC-A") == fit
    path = tmp_path / "ws.fdw"
    save_workspace(ws, path)
    ws2 = load_workspace(path)
    s2 = ws2.samples[0]
    assert s2.ensure_loaded() and "FITC-A MESF" in s2.parameter_names()
    assert np.allclose(s2.data("FITC-A MESF"), mesf.astype(np.float32))
    assert cal.stored_calibration(ws2, "FITC-A").slope == pytest.approx(fit.slope)
    with pytest.raises(ValueError):
        cal.apply_calibration(s, cal.CalibrationFit(1.0, 0.0, 1.0, "PE-A"))


def test_calibrated_scales_follow_source():
    fit = cal.CalibrationFit(1.0, 1.0, 1.0, "X")
    lin = cal._calibrated_scale(LinearScale(0, 1000), fit)
    assert isinstance(lin, LinearScale) and lin.max == pytest.approx(10000)
    log = cal._calibrated_scale(LogScale(10, 1e5), fit)
    assert isinstance(log, LogScale) and log.min == pytest.approx(100) and log.max == pytest.approx(1e6)
