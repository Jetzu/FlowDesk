"""MESF / ERF bead calibration - Qt-free.

Workflow (FlowJo "Calibrate"): acquire a bead set whose peaks carry known
MESF (molecules of equivalent soluble fluorochrome) values, find the peaks
in the bead sample, fit log10(MESF) = slope * log10(intensity) + intercept
and convert sample intensities to MESF units:

    MESF = 10**intercept * intensity**slope

Non-positive intensities (compensated data spread below zero) are mapped
sign-symmetrically (0 -> 0), so the conversion stays monotonic over all
values. A calibration must increase with the intensity: a fit (or a stored
calibration) with a non-positive slope - MESF values entered in the wrong
order - is rejected with ValueError.

The model's derived parameters cannot express a power law (they offer
ratio / product / sum / difference / log10), so `apply_calibration` stores
the converted values as a computed parameter "<param> MESF" whose
computed_info holds the calibration (it is saved with the workspace), gives
it a log-type scale and records the calibration in
workspace.settings["calibrations"][param].
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UNITS = "MESF"
SETTINGS_KEY = "calibrations"


# =============================================================================
# peak finding
# =============================================================================
@dataclass
class BeadPeak:
    position: float  # peak position on the histogram axis (display / log10 units)
    median: float  # median intensity of the peak's events, data units
    cv: float  # robust CV in percent (1.4826 x MAD / median)
    count: int
    lo: float  # window on the histogram axis
    hi: float


def _transform(values: np.ndarray, log: bool, scale) -> tuple[np.ndarray, str]:
    if scale is not None:
        return np.asarray(scale.forward(values), dtype=float), "display"
    if log:
        pos = values[values > 0]
        floor = float(np.percentile(pos, 0.1)) if pos.size else 1.0
        return np.log10(np.maximum(values, floor)), "log10"
    return values.astype(float), "linear"


def _smooth(h: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return h.astype(float)
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(h.astype(float), k, mode="same")


def bead_histogram(values, log: bool = True, scale=None, bins: int = 512, smooth: float = 1.5):
    """(centers, counts, smoothed counts, axis kind) of the bead histogram."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no events")
    x, kind = _transform(v, log, scale)
    lo, hi = np.percentile(x, [0.05, 99.95])
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.02 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, int(bins) + 1)
    counts, _ = np.histogram(x, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(float), _smooth(counts, smooth), kind


def find_bead_peaks(values, n_peaks: int, log: bool = True, scale=None, bins: int = 512, smooth: float = 1.5, min_prominence: float = 0.01) -> list[BeadPeak]:
    """The `n_peaks` most prominent peaks of a bead sample, sorted ascending.

    The histogram is built in display space when a `scale` is given (what
    the user sees), on log10 intensities when `log` (non-positive values are
    put at the lowest positive percentile) or on linear intensities.  Peaks
    come from scipy.signal.find_peaks on the smoothed histogram (prominence
    >= min_prominence x the highest bin); each peak owns the events between
    the histogram minima that separate it from its neighbours, and its
    median / CV are computed on those events in data units."""
    from scipy.signal import find_peaks

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 10:
        raise ValueError("too few events to find bead peaks")
    n_peaks = max(1, int(n_peaks))
    centers, _counts, sm, _kind = bead_histogram(v, log, scale, bins, smooth)
    x, _ = _transform(v, log, scale)
    padded = np.concatenate([[0.0], sm, [0.0]])  # peaks at the histogram border count too
    idx, props = find_peaks(padded, prominence=max(min_prominence * sm.max(), 1e-12), distance=2)
    idx = idx - 1
    if idx.size == 0:
        raise ValueError("no bead peaks found")
    order = np.argsort(props["prominences"])[::-1][:n_peaks]
    chosen = np.sort(idx[order])
    peaks: list[BeadPeak] = []
    step = centers[1] - centers[0] if centers.size > 1 else 1.0
    # window boundaries: the smoothed-histogram minimum between neighbouring peaks
    cuts = []
    for a, b in zip(chosen[:-1], chosen[1:]):
        j = a + int(np.argmin(sm[a:b + 1]))
        cuts.append(centers[j])
    for k, i in enumerate(chosen):
        pos = float(centers[i])
        left = cuts[k - 1] if k > 0 else None
        right = cuts[k] if k < len(cuts) else None
        # outer sides mirror the inner half-width (no neighbour to stop at)
        half_l = (pos - left) if left is not None else None
        half_r = (right - pos) if right is not None else None
        if left is None:
            left = pos - (half_r if half_r is not None else 10 * step)
        if right is None:
            right = pos + (half_l if half_l is not None else 10 * step)
        sel = (x >= left) & (x < right)
        ev = v[sel]
        if ev.size == 0:
            continue
        med = float(np.median(ev))
        mad = float(np.median(np.abs(ev - med))) * 1.4826
        cv = 100.0 * mad / abs(med) if med != 0 else float("nan")
        peaks.append(BeadPeak(pos, med, cv, int(ev.size), float(left), float(right)))
    return peaks


# =============================================================================
# fit
# =============================================================================
@dataclass
class CalibrationFit:
    slope: float
    intercept: float
    r2: float
    param: str = ""
    intensities: list[float] = field(default_factory=list)  # peak medians
    mesf: list[float] = field(default_factory=list)  # the kit's values
    used: list[bool] = field(default_factory=list)  # peaks that entered the fit
    units: str = UNITS
    source: str = ""  # bead sample name

    def to_mesf(self, values):
        return mesf_from_intensity(values, self)

    def to_intensity(self, values):
        return intensity_from_mesf(values, self)


def fit_calibration(peak_medians, mesf_values, exclude_blank: bool = False, use=None, param: str = "", units: str = UNITS, source: str = "") -> CalibrationFit:
    """Least-squares line through (log10 intensity, log10 MESF).

    Peaks with a non-positive intensity or MESF value (a blank bead with
    MESF 0) never enter the fit; exclude_blank also drops the lowest peak;
    `use` is an optional per-peak bool list. Needs two usable peaks."""
    x = np.asarray(peak_medians, dtype=float)
    y = np.asarray(mesf_values, dtype=float)
    if x.shape != y.shape:
        raise ValueError("need one MESF value per peak")
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if use is not None:
        ok &= np.asarray(use, dtype=bool)
    if exclude_blank and x.size:
        ok[int(np.nanargmin(np.where(np.isfinite(x), x, np.inf)))] = False
    if ok.sum() < 2:
        raise ValueError("need at least two peaks with positive intensity and MESF value")
    lx, ly = np.log10(x[ok]), np.log10(y[ok])
    if np.ptp(lx) <= 0:
        raise ValueError("peak intensities must differ")
    if np.ptp(ly) <= 0:
        raise ValueError(f"{units} values must differ")
    slope, intercept = np.polyfit(lx, ly, 1)
    if not slope > 0:
        raise ValueError(
            f"the {units} values decrease with the peak intensity (slope {slope:.3g}): "
            f"enter them in the order of the peaks, from dim to bright"
        )
    pred = slope * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return CalibrationFit(float(slope), float(intercept), float(r2), param, [float(v) for v in x], [float(v) for v in y], [bool(v) for v in ok], units, source)


def check_calibration(slope, intercept) -> tuple[float, float]:
    """(slope, intercept) as floats; ValueError unless both are finite and the
    slope is positive (the only monotonically increasing calibrations)."""
    try:
        slope, intercept = float(slope), float(intercept)
    except (TypeError, ValueError):
        raise ValueError(f"invalid calibration slope {slope!r} / intercept {intercept!r}") from None
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        raise ValueError(f"calibration slope and intercept must be finite, got {slope} and {intercept}")
    if slope <= 0:
        raise ValueError(f"calibration slope must be positive, got {slope:g}")
    return slope, intercept


def mesf_from_intensity(values, fit: CalibrationFit) -> np.ndarray:
    """10**intercept * |I|**slope with the sign of I (0 stays 0)."""
    slope, intercept = check_calibration(fit.slope, fit.intercept)
    v = np.asarray(values, dtype=float)
    a = np.abs(v)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.sign(v) * (10.0 ** intercept) * np.power(a, slope, where=a > 0, out=np.zeros_like(a))
    return out


def intensity_from_mesf(values, fit: CalibrationFit) -> np.ndarray:
    """The inverse of mesf_from_intensity (0 stays 0)."""
    slope, intercept = check_calibration(fit.slope, fit.intercept)
    m = np.asarray(values, dtype=float)
    a = np.abs(m) / 10.0 ** intercept
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.sign(m) * np.power(a, 1.0 / slope, where=a > 0, out=np.zeros_like(a))
    return out


def calibration_to_dict(fit: CalibrationFit) -> dict:
    return {
        "kind": "mesf-calibration",
        "param": fit.param,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r2": fit.r2,
        "intensities": list(fit.intensities),
        "mesf": list(fit.mesf),
        "used": list(fit.used),
        "units": fit.units,
        "source": fit.source,
    }


def calibration_from_dict(d: dict) -> CalibrationFit:
    if "slope" not in d or "intercept" not in d:
        raise ValueError("not a calibration")
    slope, intercept = check_calibration(d["slope"], d["intercept"])
    return CalibrationFit(
        slope, intercept, float(d.get("r2", float("nan"))), str(d.get("param", "")),
        [float(v) for v in d.get("intensities", [])], [float(v) for v in d.get("mesf", [])],
        [bool(v) for v in d.get("used", [])], str(d.get("units", UNITS)), str(d.get("source", "")),
    )


def save_calibration(path, fit: CalibrationFit):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(calibration_to_dict(fit), fh, indent=1)
    return Path(path)


def load_calibration(path) -> CalibrationFit:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and data.get("kind") not in (None, "mesf-calibration"):
        raise ValueError("not a FlowDesk calibration file")
    return calibration_from_dict(data)


# =============================================================================
# workspace glue
# =============================================================================
def calibrated_name(param: str, units: str = UNITS) -> str:
    return f"{param} {units}"


def peaks_for_population(population, param: str, n_peaks: int, log: bool = True, **kw) -> list[BeadPeak]:
    """Bead peaks of a population, histogrammed in the sample's display space
    (or log10 when that display is linear and `log`)."""
    from .transforms import LinearScale

    sample = population.sample
    if sample is None or not sample.ensure_loaded():
        raise ValueError("sample data is not available")
    values = np.asarray(sample.data(param), dtype=float)[population.mask()]
    scale = sample.scale(param)
    if isinstance(scale, LinearScale):
        scale = None  # beads need a log-like axis; fall back to log10 (or linear)
    elif not log:
        scale = None
    return find_bead_peaks(values, n_peaks, log=log, scale=scale, **kw)


def _calibrated_scale(src, fit: CalibrationFit):
    from .transforms import LinearScale, LogScale, default_biex

    if isinstance(src, LogScale):
        lo = float(mesf_from_intensity(src.min, fit))
        hi = float(mesf_from_intensity(src.max, fit))
        return LogScale(max(lo, 1e-6), max(hi, lo * 10, 10.0))
    if isinstance(src, LinearScale):
        lo = float(mesf_from_intensity(src.min, fit))
        hi = float(mesf_from_intensity(src.max, fit))
        return LinearScale(lo, hi) if hi > lo else LinearScale(0.0, max(hi, 1.0))
    top = float(mesf_from_intensity(getattr(src, "data_max", 262144.0), fit))
    return default_biex(top if math.isfinite(top) and top > 1 else 262144.0)


def apply_calibration(sample, fit: CalibrationFit, param: str | None = None, name: str | None = None) -> str:
    """Add "<param> MESF" (computed parameter, fit stored in computed_info and
    a scale following the source parameter's) to a sample. Returns its name."""
    param = param or fit.param
    if not param:
        raise ValueError("calibration has no parameter")
    check_calibration(fit.slope, fit.intercept)
    if not sample.ensure_loaded():
        raise ValueError(f"{sample.name}: data not available")
    if param not in sample.parameter_names():
        raise ValueError(f"{sample.name} has no parameter {param}")
    name = name or calibrated_name(param, fit.units)
    values = mesf_from_intensity(np.asarray(sample.data(param), dtype=float), fit).astype(np.float32)
    info = {"tool": "MESF calibration", "source": param, "units": fit.units, "calibration": calibration_to_dict(fit)}
    src_scale = sample.scale(param)
    sample.add_computed(name, values, info)
    sample.scales[name] = _calibrated_scale(src_scale, fit)
    sample._display_cache.pop(name, None)
    return name


def store_calibration(workspace, fit: CalibrationFit, param: str | None = None):
    workspace.settings.setdefault(SETTINGS_KEY, {})[param or fit.param] = calibration_to_dict(fit)


def stored_calibration(workspace, param: str) -> CalibrationFit | None:
    d = (workspace.settings.get(SETTINGS_KEY) or {}).get(param)
    return calibration_from_dict(d) if d else None


PEAK_HEADER = ["Peak", "Position", "Median", "rCV %", "Events", "MESF", "Used"]
