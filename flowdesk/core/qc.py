"""Data cleaning (acquisition quality control) - Qt-free.

The checks follow the ideas of the FlowAI / flowClean / PeacoQC plugins that
FlowJo users run before gating:

* **Flow rate** - events per equal-width time bin.  A bin is flagged when its
  rate deviates from the median rate by more than ``z`` robust SDs
  (1.4826 x MAD over the bins) *and* by more than ``min_rate_change`` of the
  median rate.  Leading / trailing bins are flagged while the rate is still
  ramping (deviation above 3/4 of the threshold).  An isolated bin needs
  1.5 x the threshold unless a neighbour deviates the same way (this keeps
  the chance flags of many bins x parameters at z = 3 down).
* **Signal stability** - per time bin and parameter the median of the display
  (0..1) values.  A bin is flagged when any parameter's median deviates from
  the run median by more than ``z`` robust SDs and by more than ``min_shift``
  display units.  For a multimodal parameter whose run median falls between
  two peaks (e.g. a 50 / 50 positive / negative marker) the bin median jumps
  across the gap whenever the positive fraction fluctuates, so statistic
  "auto" (the default) uses the 10 % trimmed mean there instead; the
  statistic used per parameter is reported.
* Both series also go through a simple change-point detector (binary
  segmentation of the bin series on a CUSUM / two-sample t statistic, noise
  estimated from successive differences so level shifts do not inflate it).
  A whole segment is flagged when its level differs from the run median by
  more than ``z`` single-bin noise SDs and ``min_effect``.  This catches long
  shifted stretches that inflate the MAD and escape the per-bin test.
* **Dynamic range** - per event: values at / above the detector maximum
  (``$PnR`` - 1) and values piled up exactly at the data maximum (saturated
  events); optionally also events piled up at the data minimum (clipped),
  as PeacoQC's ``RemoveMargins`` does.

`qc_summary` combines them into a per-event "good" mask, a per-bin table and
the percentage removed by every check; `qc_time_ranges` converts a mask into
clean time intervals.  `qc_for_population` / `apply_qc` connect the checks to
the workspace model (they need no Qt either).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

DEFAULT_NAME = "QC pass"
FLAG_PARAM = "QC_flag"
CHECKS = ("rate", "signal", "range")
CHECK_LABELS = {"rate": "Flow rate", "signal": "Signal stability", "range": "Dynamic range"}
MODES = ("flag", "time")

_MAD_K = 1.4826
_NON_QC_PREFIXES = ("TIME", "EVENT", "INDEX", "SAMPLEID", "HDR-T", "DELTATIME", "SORT", "REGIONCLASSIFICATION", "BEAD")
_TIME_ALIASES = ("HDR-T",)  # MACSQuant


# =============================================================================
# helpers
# =============================================================================
def _robust(x: np.ndarray) -> tuple[float, float]:
    """Median and robust SD (1.4826 x MAD, mean absolute deviation if MAD = 0)."""
    med = float(np.median(x))
    dev = np.abs(x - med)
    s = _MAD_K * float(np.median(dev))
    if s <= 0:
        s = 1.2533 * float(dev.mean())
    return med, s


def _diff_noise(x: np.ndarray) -> float:
    """Noise SD from successive differences - insensitive to level shifts."""
    if x.size < 3:
        return 0.0
    return _MAD_K * float(np.median(np.abs(np.diff(x)))) / math.sqrt(2.0)


def bin_events(time, n_bins: int = 100, edges=None) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width time bins between the first and the last event.

    Returns (edges, bin index per event); events with a non-finite time (or
    outside explicit edges) get index -1."""
    t = np.asarray(time, dtype=float)
    finite = np.isfinite(t)
    idx = np.full(t.shape, -1, dtype=np.intp)
    if edges is not None:
        edges = np.asarray(edges, dtype=float)
        n = edges.size - 1
        k = np.searchsorted(edges, t[finite], side="right") - 1
        k[t[finite] == edges[-1]] = n - 1  # the last edge is inclusive
        ok = (k >= 0) & (k < n)
        sub = np.full(k.shape, -1, dtype=np.intp)
        sub[ok] = k[ok]
        idx[finite] = sub
        return edges, idx
    n = max(1, int(n_bins))
    if not finite.any():
        return np.linspace(0.0, 1.0, n + 1), idx
    t0, t1 = float(t[finite].min()), float(t[finite].max())
    if t1 <= t0:
        t1 = t0 + 1.0
    edges = np.linspace(t0, t1, n + 1)
    idx[finite] = np.clip(((t[finite] - t0) / (t1 - t0) * n).astype(np.intp), 0, n - 1)
    return edges, idx


def change_points(x, sigma: float, threshold: float = 4.0, min_size: int = 2, max_segments: int = 12) -> list[tuple[int, int]]:
    """Binary segmentation of a series into constant-level segments.

    A segment is split at the position maximising the two-sample statistic
    |mean(left) - mean(right)| / (sigma * sqrt(1/k + 1/(n-k))) when that
    maximum exceeds `threshold`. Returns sorted (start, stop) index pairs."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return []
    if sigma <= 0 or not math.isfinite(sigma):
        return [(0, n)]
    min_size = max(1, int(min_size))
    cuts: list[int] = []
    stack = [(0, n)]
    while stack and len(cuts) < max_segments - 1:
        a, b = stack.pop()
        m = b - a
        if m < 2 * min_size:
            continue
        cs = np.cumsum(x[a:b])
        k = np.arange(min_size, m - min_size + 1)
        left = cs[k - 1] / k
        right = (cs[-1] - cs[k - 1]) / (m - k)
        stat = np.abs(left - right) / (sigma * np.sqrt(1.0 / k + 1.0 / (m - k)))
        j = int(np.argmax(stat))
        if stat[j] > threshold:
            cut = a + int(k[j])
            cuts.append(cut)
            stack.append((a, cut))
            stack.append((cut, b))
    bounds = [0] + sorted(cuts) + [n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


@dataclass
class SeriesFlags:
    """Outcome of testing one bin series (rates or medians of one parameter)."""

    flags: np.ndarray
    center: float = float("nan")
    spread: float = float("nan")
    threshold: float = float("nan")
    segments: list[tuple[int, int, float, bool]] = field(default_factory=list)  # start, stop, level, flagged


def flag_series(x, z: float = 3.0, min_effect: float = 0.0, changepoints: bool = True, cp_threshold: float | None = None) -> SeriesFlags:
    """Flag bins of a series that deviate from its median (NaN = untested)."""
    x = np.asarray(x, dtype=float)
    flags = np.zeros(x.shape, dtype=bool)
    valid = np.isfinite(x)
    v = x[valid]
    if v.size < 3:
        return SeriesFlags(flags)
    center, spread = _robust(v)
    spread = max(spread, 1e-12 * max(1.0, abs(center)))
    threshold = max(z * spread, float(min_effect))
    dev = v - center
    out = np.abs(dev) > threshold
    # 100 bins x many parameters at z = 3 would flag a few bins by chance: a
    # single bin must exceed 1.5 x the threshold unless a neighbour deviates
    # the same way by at least half of it (clogs / bubbles are far stronger)
    half = (np.abs(dev) > 0.5 * threshold) & (dev != 0)
    sign = np.sign(dev)
    support = np.zeros(v.size, dtype=bool)
    support[1:] |= half[:-1] & (sign[:-1] == sign[1:])
    support[:-1] |= half[1:] & (sign[1:] == sign[:-1])
    fv = (np.abs(dev) > 1.5 * threshold) | (out & support)
    segments: list[tuple[int, int, float, bool]] = []
    if changepoints and v.size >= 6:
        noise = _diff_noise(v) or spread
        clip = max(6.0 * spread, 3.0 * float(min_effect))
        clipped = np.clip(v, center - clip, center + clip)
        thr = cp_threshold if cp_threshold is not None else max(4.0, z + 1.0)
        segs = change_points(clipped, noise, thr, min_size=2)
        if len(segs) > 1:
            levels = [float(np.median(v[a:b])) for a, b in segs]
            resid = np.concatenate([v[a:b] - lvl for (a, b), lvl in zip(segs, levels)])
            within = _MAD_K * float(np.median(np.abs(resid))) or noise
            # a segment counts when its level is off by more than z noise SDs
            # of a single bin (the noise without the level shifts) - slow
            # drift inside the noise is tolerated - and by min_effect
            need = max(float(min_effect), z * within)
            for (a, b), lvl in zip(segs, levels):
                bad = abs(lvl - center) > need
                if bad:
                    fv[a:b] = True
                segments.append((a, b, lvl, bad))
    # map segment indices (over valid entries) back to bin indices
    pos = np.nonzero(valid)[0]
    segments = [(int(pos[a]), int(pos[b - 1]) + 1, lvl, bad) for a, b, lvl, bad in segments]
    flags[valid] = fv
    return SeriesFlags(flags, center, spread, threshold, segments)


# =============================================================================
# checks
# =============================================================================
@dataclass
class RateCheck:
    edges: np.ndarray
    counts: np.ndarray
    rates: np.ndarray  # events per time unit
    flags: np.ndarray  # per bin
    ramp: np.ndarray  # per bin: flagged as start / end ramp
    center: float
    spread: float
    threshold: float
    event_bins: np.ndarray


def flow_rate_check(time, n_bins: int = 100, z: float = 3.0, min_rate_change: float = 0.1, ramp: bool = True, changepoints: bool = True, edges=None) -> RateCheck:
    """Events per time bin; flags bins whose rate deviates from the median rate
    by more than z robust SDs (and min_rate_change x median), plus the bins at
    the start / end of the acquisition while the rate ramps."""
    edges, idx = bin_events(time, n_bins, edges)
    n = edges.size - 1
    counts = np.bincount(idx[idx >= 0], minlength=n).astype(float)
    width = np.diff(edges)
    rates = np.where(width > 0, counts / np.where(width > 0, width, 1.0), 0.0)
    center = float(np.median(rates)) if n else 0.0
    sf = flag_series(rates, z, min_rate_change * abs(center), changepoints)
    flags = sf.flags.copy()
    ramp_flags = np.zeros(n, dtype=bool)
    if ramp and n >= 4 and math.isfinite(sf.threshold):
        limit = 0.75 * sf.threshold
        max_run = max(1, n // 5)
        for order in (range(n), range(n - 1, -1, -1)):
            run = 0
            for i in order:
                if abs(rates[i] - sf.center) > limit and run < max_run:
                    ramp_flags[i] = True
                    run += 1
                else:
                    break
        flags |= ramp_flags
    return RateCheck(edges, counts, rates, flags, ramp_flags, sf.center, sf.spread, sf.threshold, idx)


@dataclass
class SignalCheck:
    edges: np.ndarray
    medians: dict[str, np.ndarray]  # per parameter, per bin (NaN = too few events)
    flags: dict[str, np.ndarray]  # per parameter, per bin
    any_flags: np.ndarray
    centers: dict[str, float]
    thresholds: dict[str, float]
    event_bins: np.ndarray
    statistics: dict[str, str] = field(default_factory=dict)  # "median" or "trimmed mean"


STATISTICS = ("auto", "median", "mean")
TRIM = 0.1


def location_statistic(values, statistic: str = "auto", gap_ratio: float = 1.0 / 3.0) -> str:
    """"median" or "trimmed mean" for a parameter. "auto" picks the median
    unless the smoothed density at the run median is below `gap_ratio` of the
    highest density, i.e. the median sits in a valley between peaks."""
    if statistic == "median":
        return "median"
    if statistic == "mean":
        return "trimmed mean"
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return "median"
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return "median"
    h, edges = np.histogram(v, bins=128, range=(lo, hi))
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    sm = np.convolve(h.astype(float), k / k.sum(), mode="same")
    j = int(np.clip(np.searchsorted(edges, np.median(v)) - 1, 0, sm.size - 1))
    return "median" if sm[j] >= gap_ratio * sm.max() else "trimmed mean"


def bin_medians(values, idx: np.ndarray, n_bins: int, min_events: int = 20, statistic: str = "median") -> np.ndarray:
    """Median (or 10 % trimmed mean) of `values` in every bin (NaN where
    fewer than min_events)."""
    v = np.asarray(values, dtype=float)
    ok = (idx >= 0) & np.isfinite(v)
    out = np.full(n_bins, np.nan)
    if not ok.any():
        return out
    b = idx[ok]
    order = np.argsort(b, kind="stable")
    bs, vs = b[order], v[ok][order]
    starts = np.searchsorted(bs, np.arange(n_bins), side="left")
    stops = np.searchsorted(bs, np.arange(n_bins), side="right")
    trimmed = statistic in ("trimmed mean", "mean")
    for i in range(n_bins):
        if stops[i] - starts[i] >= max(1, min_events):
            chunk = vs[starts[i]:stops[i]]
            if trimmed:
                cut = int(TRIM * chunk.size)
                chunk = np.partition(chunk, (cut, chunk.size - cut - 1))[cut:chunk.size - cut] if cut else chunk
                out[i] = float(chunk.mean())
            else:
                out[i] = np.median(chunk)
    return out


def signal_stability_check(time, values_by_param: dict, n_bins: int = 100, z: float = 3.0, min_shift: float = 0.02, changepoints: bool = True, min_bin_events: int = 20, edges=None, statistic: str = "auto") -> SignalCheck:
    """Per bin and parameter the median of the display-space (0..1) values;
    a bin is flagged when any parameter's median deviates from the run median
    by more than z robust SDs (and min_shift display units) or lies in a
    change-point segment whose level shifted by more than min_shift."""
    edges, idx = bin_events(time, n_bins, edges)
    n = edges.size - 1
    medians, flags, centers, thresholds, stats = {}, {}, {}, {}, {}
    any_flags = np.zeros(n, dtype=bool)
    for name, values in values_by_param.items():
        stats[name] = location_statistic(values, statistic)
        med = bin_medians(values, idx, n, min_bin_events, stats[name])
        sf = flag_series(med, z, min_shift, changepoints)
        medians[name] = med
        flags[name] = sf.flags
        centers[name] = sf.center
        thresholds[name] = sf.threshold
        any_flags |= sf.flags
    return SignalCheck(edges, medians, flags, any_flags, centers, thresholds, idx, stats)


@dataclass
class RangeCheck:
    flags: np.ndarray  # per event: True = outside the dynamic range
    counts: dict[str, int]  # per parameter flagged events
    limits: dict[str, tuple[float | None, float | None]]
    notes: dict[str, str] = field(default_factory=dict)


def _limits(spec) -> tuple[float | None, float | None]:
    if spec is None:
        return None, None
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return None, float(spec)
    if isinstance(spec, (tuple, list)):
        lo, hi = (list(spec) + [None, None])[:2]
        return (None if lo is None else float(lo)), (None if hi is None else float(hi))
    top = getattr(spec, "data_max", None)  # a transforms.Scale: its top is the detector range
    return None, (float(top) if top is not None and math.isfinite(float(top)) else None)


def dynamic_range_check(values_by_param: dict, scales: dict | None = None, min_pile: int = 3, check_low: bool = False, max_fraction: float = 0.2) -> RangeCheck:
    """Flag events at the detector maximum or minimum (saturated / clipped).

    values_by_param: data-unit values per parameter. scales: per parameter a
    Scale (its top is taken as the detector maximum), a top value ($PnR) or a
    (lo, hi) pair; parameters without one only use pile-up detection.
    An event is flagged when it lies at or above hi - 1 (hi = $PnR), at or
    below lo, or sits with at least `min_pile` events exactly on the data
    maximum (or minimum, when check_low). A limit that would remove more than
    `max_fraction` of the events is taken as wrong metadata and ignored."""
    scales = scales or {}
    flags = None
    counts: dict[str, int] = {}
    used: dict[str, tuple[float | None, float | None]] = {}
    notes: dict[str, str] = {}
    for name, values in values_by_param.items():
        v = np.asarray(values, dtype=float)
        if flags is None:
            flags = np.zeros(v.shape, dtype=bool)
        own = np.zeros(v.shape, dtype=bool)
        finite = np.isfinite(v)
        if not finite.any():
            counts[name] = 0
            continue
        lo, hi = _limits(scales.get(name))
        if hi is not None:
            cut = hi - 1.0 if hi > 2.0 else hi
            over = finite & (v >= cut * (1.0 - 1e-7))
            if over.mean() <= max_fraction:
                own |= over
            else:
                notes[name] = f"upper limit {hi:g} ignored ({100 * over.mean():.0f}% of events above it)"
                hi = None
        if lo is not None and check_low:
            under = finite & (v <= lo)
            if under.mean() <= max_fraction:
                own |= under
            else:
                notes[name] = f"lower limit {lo:g} ignored"
                lo = None
        vmax, vmin = float(v[finite].max()), float(v[finite].min())
        tol = 1e-6 * max(1.0, abs(vmax))
        at_max = finite & (v >= vmax - tol)
        if at_max.sum() >= min_pile:
            own |= at_max
        if check_low:
            tol = 1e-6 * max(1.0, abs(vmin))
            at_min = finite & (v <= vmin + tol)
            if at_min.sum() >= min_pile:
                own |= at_min
        counts[name] = int(own.sum())
        used[name] = (lo, hi)
        flags |= own
    if flags is None:
        flags = np.zeros(0, dtype=bool)
    return RangeCheck(flags, counts, used, notes)


# =============================================================================
# combination
# =============================================================================
@dataclass
class QCResult:
    good: np.ndarray  # per analysed event
    time_good: np.ndarray  # not in a flagged time bin (rate / signal checks)
    range_good: np.ndarray  # inside the dynamic range
    edges: np.ndarray
    counts: np.ndarray
    rates: np.ndarray
    bin_flags: dict[str, np.ndarray]  # "rate", "signal:<param>" -> per-bin bool
    flagged_bins: np.ndarray
    medians: dict[str, np.ndarray]
    percent_removed: dict[str, float]  # "rate", "signal", "range", "total"
    range_counts: dict[str, int]
    intervals: list[tuple[float, float]]
    checks: tuple[str, ...]
    settings: dict
    rate_center: float = float("nan")
    rate_threshold: float = float("nan")
    notes: dict[str, str] = field(default_factory=dict)
    # filled in by qc_for_population
    event_index: np.ndarray | None = None
    population_path: str | None = None
    time_param: str | None = None
    sample_id: int | None = None

    @property
    def n_events(self) -> int:
        return int(self.good.size)

    @property
    def n_good(self) -> int:
        return int(self.good.sum())

    @property
    def n_bins(self) -> int:
        return int(self.edges.size - 1)

    @property
    def params(self) -> list[str]:
        return list(self.medians.keys())

    def flagged_by(self, i: int) -> list[str]:
        return [k for k, f in self.bin_flags.items() if f[i]]

    def bin_rows(self) -> list[dict]:
        """bin (1-based), t_start, t_end, events, rate, flagged_by."""
        rows = []
        for i in range(self.n_bins):
            rows.append({
                "bin": i + 1,
                "t_start": float(self.edges[i]),
                "t_end": float(self.edges[i + 1]),
                "events": int(self.counts[i]),
                "rate": float(self.rates[i]),
                "flagged_by": ", ".join(self.flagged_by(i)),
            })
        return rows

    def summary(self) -> dict:
        return {
            "events": self.n_events,
            "kept": self.n_good,
            "removed": self.n_events - self.n_good,
            "percent_removed": dict(self.percent_removed),
            "flagged_bins": int(self.flagged_bins.sum()),
            "bins": self.n_bins,
            "intervals": len(self.intervals),
        }


def qc_time_ranges(mask, time) -> list[tuple[float, float]]:
    """Clean (t0, t1) intervals: runs of good events in time order. t1 is the
    time of the first rejected event after the run (exclusive bound, as in a
    RectangleGate) or just above the last event (relative margin 1e-6)."""
    t = np.asarray(time, dtype=float)
    m = np.asarray(mask, dtype=bool)
    fin = np.isfinite(t)
    t, m = t[fin], m[fin]
    if t.size == 0 or not m.any():
        return []
    order = np.argsort(t, kind="stable")
    ts, ms = t[order], m[order]
    d = np.diff(np.concatenate([[0], ms.astype(np.int8), [0]]))
    starts = np.nonzero(d == 1)[0]
    stops = np.nonzero(d == -1)[0]
    out = []
    for a, b in zip(starts, stops):
        t0 = float(ts[a])
        # past the last event: a margin that survives float32 event data
        t1 = float(ts[b]) if b < ts.size else float(ts[-1]) + max(abs(float(ts[-1])), 1.0) * 1e-6
        if t1 > t0:
            out.append((t0, t1))
    return out


def qc_summary(time, signal_values: dict | None = None, range_values: dict | None = None, limits: dict | None = None, *,
               checks=CHECKS, n_bins: int = 100, z: float = 3.0, rate_z: float | None = None, signal_z: float | None = None,
               min_rate_change: float = 0.1, min_shift: float = 0.02, changepoints: bool = True, ramp: bool = True,
               min_bin_events: int = 20, min_pile: int = 3, check_low: bool = False, statistic: str = "auto") -> QCResult:
    """Run the selected checks and combine them.

    time: per-event acquisition time. signal_values: display-space values per
    parameter for the signal check. range_values: data-unit values per
    parameter for the dynamic-range check (defaults to signal_values) with
    `limits` per parameter (see dynamic_range_check)."""
    t = np.asarray(time, dtype=float)
    n_ev = t.size
    checks = tuple(c for c in CHECKS if c in tuple(checks))
    signal_values = signal_values or {}
    range_values = range_values if range_values is not None else signal_values
    edges, idx = bin_events(t, n_bins)
    nb = edges.size - 1
    counts = np.bincount(idx[idx >= 0], minlength=nb).astype(float)
    width = np.diff(edges)
    rates = np.where(width > 0, counts / np.where(width > 0, width, 1.0), 0.0)
    bin_flags: dict[str, np.ndarray] = {}
    medians: dict[str, np.ndarray] = {}
    percent: dict[str, float] = {}
    notes: dict[str, str] = {}
    rate_center = rate_threshold = float("nan")
    statistics: dict[str, str] = {}
    binned = idx >= 0

    def events_in(flags_per_bin: np.ndarray) -> np.ndarray:
        out = np.zeros(n_ev, dtype=bool)
        out[binned] = flags_per_bin[idx[binned]]
        return out

    time_bad = np.zeros(n_ev, dtype=bool)
    if "rate" in checks:
        rc = flow_rate_check(t, z=rate_z if rate_z is not None else z, min_rate_change=min_rate_change, ramp=ramp, changepoints=changepoints, edges=edges)
        bin_flags["rate"] = rc.flags
        rate_center, rate_threshold = rc.center, rc.threshold
        bad = events_in(rc.flags)
        percent["rate"] = float(100.0 * bad.mean()) if n_ev else 0.0
        time_bad |= bad
    if "signal" in checks:
        sc = signal_stability_check(t, signal_values, z=signal_z if signal_z is not None else z, min_shift=min_shift, changepoints=changepoints, min_bin_events=min_bin_events, edges=edges, statistic=statistic)
        medians = sc.medians
        statistics = sc.statistics
        for name, f in sc.flags.items():
            bin_flags[f"signal:{name}"] = f
        bad = events_in(sc.any_flags)
        percent["signal"] = float(100.0 * bad.mean()) if n_ev else 0.0
        time_bad |= bad
    elif signal_values:
        # medians are still useful for the plot
        statistics = {name: location_statistic(v, statistic) for name, v in signal_values.items()}
        medians = {name: bin_medians(v, idx, nb, min_bin_events, statistics[name]) for name, v in signal_values.items()}
    range_bad = np.zeros(n_ev, dtype=bool)
    range_counts: dict[str, int] = {}
    if "range" in checks and range_values:
        dr = dynamic_range_check(range_values, limits, min_pile=min_pile, check_low=check_low)
        if dr.flags.size == n_ev:
            range_bad = dr.flags
        range_counts = dr.counts
        notes.update(dr.notes)
        percent["range"] = float(100.0 * range_bad.mean()) if n_ev else 0.0
    flagged_bins = np.zeros(nb, dtype=bool)
    for f in bin_flags.values():
        flagged_bins |= f
    time_good = ~time_bad
    range_good = ~range_bad
    good = time_good & range_good
    percent["total"] = float(100.0 * (1.0 - good.mean())) if n_ev else 0.0
    settings = {
        "checks": list(checks), "n_bins": int(nb), "z": float(z),
        "rate_z": float(rate_z if rate_z is not None else z), "signal_z": float(signal_z if signal_z is not None else z),
        "min_rate_change": float(min_rate_change), "min_shift": float(min_shift), "changepoints": bool(changepoints),
        "ramp": bool(ramp), "min_bin_events": int(min_bin_events), "min_pile": int(min_pile), "check_low": bool(check_low),
        "params": list(signal_values.keys()), "statistic": statistic, "statistics": dict(statistics),
    }
    return QCResult(good, time_good, range_good, edges, counts, rates, bin_flags, flagged_bins, medians, percent, range_counts,
                    qc_time_ranges(time_good, t), checks, settings, rate_center, rate_threshold, notes)


# =============================================================================
# workspace glue (Qt-free)
# =============================================================================
def is_time_parameter(name: str) -> bool:
    upper = name.strip().upper()
    return bool(re.fullmatch(r"\$?TIME\d*", upper)) or upper.startswith("TIME") or upper in _TIME_ALIASES


def default_time_parameter(sample) -> str | None:
    """A parameter called Time / TIME (or starting with it, or MACSQuant's
    HDR-T), else None."""
    names = list(sample.channel_names) + [n for n in sample.parameter_names() if n not in sample.channel_names]
    for n in names:
        if n.strip().upper() == "TIME":
            return n
    for n in names:
        if is_time_parameter(n):
            return n
    return None


def default_qc_parameters(sample, time_param: str | None = None) -> list[str]:
    """Measured area / height channels: no time / event-count parameters and
    no pulse widths (their zero / maximum values are not saturation)."""
    out = []
    for n in sample.channel_names:
        if n == time_param:
            continue
        upper = n.upper()
        if any(upper.startswith(p) for p in _NON_QC_PREFIXES):
            continue
        if upper.endswith("-W") or upper.endswith("WIDTH"):
            continue
        out.append(n)
    return out


def qc_for_population(population, time_param: str | None, params: list[str] | None = None, **settings) -> QCResult:
    """Run the checks on the events of a population.

    Signal medians use the sample's display values, the range check its data
    values against $PnR of the parameter's channel. Without a time parameter
    the events are binned in acquisition order and the rate check is skipped."""
    sample = population.sample
    if sample is None or not sample.ensure_loaded():
        raise ValueError("sample data is not available")
    if time_param is not None and time_param not in sample.parameter_names():
        raise ValueError(f"{sample.name} has no parameter {time_param}")
    mask = population.mask()
    index = np.nonzero(mask)[0]
    if index.size < 10:
        raise ValueError(f"{population.display_name} has too few events ({index.size})")
    params = list(params) if params is not None else default_qc_parameters(sample, time_param)
    params = [p for p in params if p in sample.parameter_names() and p != time_param]
    checks = tuple(settings.pop("checks", CHECKS))
    if time_param is None:
        time = index.astype(float)
        checks = tuple(c for c in checks if c != "rate")
    else:
        time = np.asarray(sample.data(time_param), dtype=float)[index]
    signal = {p: np.asarray(sample.display(p), dtype=float)[index] for p in params}
    raw = {p: np.asarray(sample.data(p), dtype=float)[index] for p in params}
    limits = {}
    for p in params:
        ch = sample.base_channel(p)
        if ch is not None and ch.range and ch.range > 1 and p not in sample.computed and p not in sample.derived:
            limits[p] = (None, float(ch.range))
    result = qc_summary(time, signal, raw, limits, checks=checks, **settings)
    result.event_index = index
    result.population_path = population.path
    result.time_param = time_param
    result.sample_id = sample.id
    return result


def flag_values(result: QCResult, n_events: int) -> np.ndarray:
    """QC_flag values: 1 good, 0 bad, NaN outside the analysed population."""
    out = np.full(n_events, np.nan, dtype=np.float32)
    if result.event_index is not None:
        out[result.event_index] = result.good.astype(np.float32)
    return out


def _json_safe(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = float(v) if math.isfinite(float(v)) else None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = _json_safe(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [float(x) if isinstance(x, np.floating) else x for x in v]
        else:
            out[k] = v
    return out


def _flag_name(sample, base: str, population_path: str) -> str:
    """Reuse the flag parameter of an earlier run on the same population,
    otherwise pick a name no other parameter uses."""
    names = set(sample.parameter_names())
    if base not in names:
        return base
    if base in sample.computed and sample.computed_info.get(base, {}).get("population", "") == population_path:
        return base
    k = 2
    while True:
        cand = f"{base}_{k}"
        if cand not in names:
            return cand
        if cand in sample.computed and sample.computed_info.get(cand, {}).get("population", "") == population_path:
            return cand
        k += 1


def interval_prefix(name: str) -> str:
    return "QC interval" if name == DEFAULT_NAME else f"{name} interval"


def apply_qc(population, result: QCResult, name: str = DEFAULT_NAME, mode: str = "flag", flag_name: str = FLAG_PARAM) -> list:
    """Create the clean-events population below `population`.

    mode "flag": a computed parameter (1 good / 0 bad / NaN outside the
    population) plus a RectangleGate(flag >= 0.5) population.
    mode "time": the union of the clean time intervals - one RectangleGate on
    the time parameter when there is a single interval, otherwise one
    "QC interval k" population per interval and an OR (Boolean) population.
    Time gates cannot express the per-event dynamic-range flags, so that
    mode uses the time-based checks only. An earlier result of the same name
    is replaced. Returns the created populations (the clean one first)."""
    from .gates import BooleanGate, RectangleGate
    from .model import Population

    sample = population.sample
    if sample is None:
        raise ValueError("population has no sample")
    name = (name or DEFAULT_NAME).strip() or DEFAULT_NAME
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    old = population.child_by_name(name)
    kept_children = list(old.children) if old is not None else []
    if old is not None:
        population.remove_child(old)
    prefix = interval_prefix(name)
    pattern = re.compile(re.escape(prefix) + r" \d+$")
    for c in list(population.children):
        if pattern.fullmatch(c.name):
            population.remove_child(c)
    created = []
    if mode == "flag":
        flag = _flag_name(sample, flag_name, population.path)
        info = {
            "tool": "Data Cleaning",
            "population": population.path,
            "time_param": result.time_param,
            "settings": _json_safe(result.settings),
            "percent_removed": _json_safe(result.percent_removed),
            "min": -0.5,
            "max": 1.5,
        }
        sample.add_computed(flag, flag_values(result, sample.n_events), info)
        pop = Population(name, RectangleGate([flag], [0.5], [None]))
        population.add_child(pop)
        created.append(pop)
    else:
        tp = result.time_param
        if not tp:
            raise ValueError("time gates need a time parameter")
        intervals = result.intervals
        if not intervals:
            raise ValueError("no clean time interval")
        if len(intervals) == 1:
            t0, t1 = intervals[0]
            pop = Population(name, RectangleGate([tp], [t0], [t1]))
            population.add_child(pop)
            created.append(pop)
        else:
            parts = []
            for k, (t0, t1) in enumerate(intervals, start=1):
                c = Population(f"{prefix} {k}", RectangleGate([tp], [t0], [t1]))
                population.add_child(c)
                parts.append(c)
            pop = Population(name, BooleanGate("or", [c.path for c in parts]))
            population.add_child(pop)
            created = [pop] + parts
    for c in kept_children:  # sub-gates the user drew below an earlier QC population
        if created[0].child_by_name(c.name) is None:
            created[0].add_child(c)
    sample.root.invalidate()
    return created


# =============================================================================
# tables
# =============================================================================
TABLE_HEADER = ["Bin", "t start", "t end", "Events", "Rate", "Flagged by", "% Removed"]
GROUP_HEADER = ["Sample", "Population", "Events", "Kept", "% Removed", "% Rate", "% Signal", "% Range", "Flagged Bins", "Clean Intervals", "Note"]


def result_table(result: QCResult) -> list[list]:
    """Rows for TABLE_HEADER: one per check (removed events and %), then one per bin."""
    rows: list[list] = []
    n = result.n_events
    for key in result.checks:
        pct = result.percent_removed.get(key, 0.0)
        rows.append([CHECK_LABELS[key], "", "", int(round(pct * n / 100.0)), "", "removed", pct])
    rows.append(["Total", "", "", n - result.n_good, "", "removed", result.percent_removed.get("total", 0.0)])
    for r in result.bin_rows():
        rows.append([r["bin"], r["t_start"], r["t_end"], r["events"], r["rate"], r["flagged_by"], ""])
    return rows


def group_row(sample_name: str, population_name: str, result: QCResult, note: str = "") -> list:
    p = result.percent_removed
    return [sample_name, population_name, result.n_events, result.n_good, p.get("total", 0.0),
            p.get("rate", float("nan")), p.get("signal", float("nan")), p.get("range", float("nan")),
            int(result.flagged_bins.sum()), len(result.intervals), note]
