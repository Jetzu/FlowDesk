"""Biology tools: cell-cycle fitting, dye-dilution proliferation modelling and
kinetics (a statistic as a function of acquisition time).

Everything here is plain numpy/scipy so it can run (and be tested) without Qt.

* `fit_cell_cycle`   -- two models on a linear DNA histogram, both with a G1
                        Gaussian and a G2/M Gaussian at ~2x G1:
                        "dean_jett_fox": S phase is a rectangle between both peaks
                        smoothed by their widths (closed-form error-function
                        plateau), everything fitted jointly;
                        "watson": the Watson pragmatic model (Watson, Chambers &
                        Smith 1987) -- G1 is fitted on its outer (left) half, G2/M
                        on its outer (right) half and the S phase is what is left
                        between them after subtracting G1, G2/M, debris and
                        aggregates, so no S-phase shape is assumed.
                        Optional debris (exponentially decreasing from the left
                        edge) and aggregate components (doublets as the
                        self-convolution of the singlet distribution: G1+G1 at 2x
                        G1, G1+G2 at 3x, G2+G2 at 4x and the S-phase mixtures).
* `fit_proliferation`-- generations as Gaussians whose intensities halve per
                        division, amplitudes by non-negative least squares,
                        generation-0 position and the shared width refined by a
                        grid search followed by a Nelder-Mead polish; the
                        generation-0 position can be given (e.g. the median of an
                        undivided control) and optionally held fixed.
* `kinetics`         -- events binned by time, a statistic per bin.
* `cached_population_fit` -- cell-cycle / proliferation fits of a population
                        behind a small LRU cache, used by the table statistics.
"""
from __future__ import annotations

import csv
import json
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, minimize, minimize_scalar, nnls
from scipy.signal import find_peaks, peak_widths
from scipy.special import ndtr

SQRT2PI = math.sqrt(2.0 * math.pi)
LOG2 = math.log10(2.0)

DNA_KEYWORDS = ("DNA", "PI", "DAPI", "HOECHST", "7AAD", "7-AAD", "DRAQ", "FXCYCLE", "VYBRANT")
DYE_KEYWORDS = ("CFSE", "CTV", "CELLTRACE", "VIOLET", "CTFR", "CTY", "EFLUOR 670", "PKH")
SCATTER_PREFIXES = ("FSC", "SSC", "TIME", "HDR-T", "EVENT", "WIDTH", "INDEXSORT", "SAMPLEID", "DELTATIME", "SORT")


def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQRT2PI)


def _finite(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    return v[np.isfinite(v)]


def write_csv(path, header: list, rows: list[list]):
    """Write a header and rows to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(v, float) and not math.isfinite(v)) else v for v in r])


# =============================================================================
# cell cycle
# =============================================================================
@dataclass
class CellCycleResult:
    percent_g1: float
    percent_s: float
    percent_g2m: float
    g1_mean: float
    g2_mean: float
    g1_cv: float
    g2_cv: float
    g1_sd: float
    g2_sd: float
    centers: np.ndarray
    counts: np.ndarray
    curves: dict[str, np.ndarray]  # "G1", "S", "G2/M", "Model"
    rms: float
    n_events: int
    fit_range: tuple[float, float] = (0.0, 0.0)
    success: bool = True
    message: str = ""
    model: str = "dean_jett_fox"
    # percent of all modelled events; NaN when the component was not modelled.
    # %G1 + %S + %G2/M always sum to 100 (of the cycling singlets).
    percent_debris: float = float("nan")
    percent_aggregates: float = float("nan")
    debris_decay: float = float("nan")  # 1/e length of the debris exponential

    @property
    def g2_g1_ratio(self) -> float:
        return self.g2_mean / self.g1_mean if self.g1_mean else float("nan")

    @property
    def model_name(self) -> str:
        return CELL_CYCLE_MODELS.get(self.model, self.model)

    def summary(self) -> dict[str, float]:
        out = {
            "%G1": self.percent_g1,
            "%S": self.percent_s,
            "%G2/M": self.percent_g2m,
            "G1 Mean": self.g1_mean,
            "G2 Mean": self.g2_mean,
            "G1 CV": self.g1_cv,
            "G2 CV": self.g2_cv,
            "G2/G1": self.g2_g1_ratio,
            "RMS": self.rms,
            "Events": self.n_events,
        }
        if math.isfinite(self.percent_debris):
            out["%Debris"] = self.percent_debris
        if math.isfinite(self.percent_aggregates):
            out["%Aggregates"] = self.percent_aggregates
        return out


CELL_CYCLE_MODELS = {"dean_jett_fox": "Dean-Jett-Fox", "watson": "Watson Pragmatic"}
_MODEL_ALIASES = {
    "dean_jett_fox": "dean_jett_fox", "djf": "dean_jett_fox", "deanjettfox": "dean_jett_fox", "dean": "dean_jett_fox",
    "watson": "watson", "watson_pragmatic": "watson", "pragmatic": "watson",
}


def cell_cycle_model_key(model: str | None) -> str:
    """Normalise a model name ("Watson Pragmatic", "DJF", ...) to its key."""
    key = str(model or "dean_jett_fox").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _MODEL_ALIASES:
        raise ValueError(f"unknown cell-cycle model {model!r}; use one of {sorted(CELL_CYCLE_MODELS)}")
    return _MODEL_ALIASES[key]


def cell_cycle_components(x, n1, ns, n2, mu, s1, s2, ratio, width=1.0):
    """Expected counts per bin of the three phases (bin width `width`)."""
    m2 = ratio * mu
    g1 = width * n1 * _gauss(x, mu, s1)
    g2 = width * n2 * _gauss(x, m2, s2)
    # uniform density on [mu, m2] convolved with a Gaussian whose width follows
    # the G1 sd at the lower edge and the G2 sd at the upper edge
    s = width * ns * (ndtr((x - mu) / s1) - ndtr((x - m2) / s2)) / max(m2 - mu, 1e-12)
    return g1, s, g2


def debris_component(x, nd, decay, x0, width=1.0):
    """Expected debris counts per bin: `nd` events distributed as an
    exponential decreasing from `x0` (the left histogram edge) with 1/e length
    `decay`; exact bin integrals for bins of width `width` centred on `x`."""
    x = np.asarray(x, dtype=float)
    decay = max(float(decay), 1e-12)
    a = np.maximum(x - 0.5 * width - x0, 0.0)
    b = np.maximum(x + 0.5 * width - x0, 0.0)
    return nd * (np.exp(-a / decay) - np.exp(-b / decay))


def aggregate_component(x, na, n1, ns, n2, mu, s1, s2, ratio, width=1.0):
    """Expected doublet counts per bin: `na` aggregates whose DNA content is the
    sum of two randomly paired singlets, i.e. the self-convolution of the
    singlet (G1 + S + G2/M) distribution -- G1+G1 at 2x G1, G1+G2/M at 3x,
    G2/M+G2/M at 4x, plus the S-phase mixtures, with the widths adding in
    quadrature."""
    x = np.asarray(x, dtype=float)
    tot = n1 + ns + n2
    if na <= 0 or tot <= 0 or x.size == 0:
        return np.zeros_like(x)
    step = 0.5 * width
    xmax = float(np.max(x)) + width
    if xmax <= 0:
        return np.zeros_like(x)
    t = np.arange(0.0, xmax + step, step)
    g1, s, g2 = cell_cycle_components(t, n1 / tot, ns / tot, n2 / tot, mu, s1, s2, ratio, width=1.0)
    f = g1 + s + g2
    d = np.convolve(f, f)[: t.size] * step
    return width * na * np.interp(x, t, d, left=0.0, right=0.0)


def _cycle_guesses(centers, counts, g1_guess):
    smooth = gaussian_filter1d(counts.astype(float), 1.5)
    peaks, props = find_peaks(smooth, prominence=max(smooth.max() * 0.02, 1e-9))
    width = centers[1] - centers[0]
    if g1_guess is not None:
        x1 = float(g1_guess)
    elif peaks.size:
        order = peaks[np.argsort(smooth[peaks])[::-1]]
        p1 = order[0]
        x1 = centers[p1]
        # the highest peak may be G2/M when most cells are cycling
        lower = [p for p in order[1:4] if 0.4 * x1 <= centers[p] <= 0.6 * x1 and smooth[p] > 0.15 * smooth[p1]]
        if lower:
            x1 = centers[lower[0]]
    else:
        x1 = float(np.average(centers, weights=counts + 1e-12))
    i1 = int(np.clip(np.searchsorted(centers, x1), 0, centers.size - 1))
    # G2: the highest peak near twice G1 (among the highest peaks)
    x2 = 2.0 * x1
    cands = [p for p in peaks if 1.7 * x1 <= centers[p] <= 2.3 * x1]
    if cands:
        x2 = centers[max(cands, key=lambda p: smooth[p])]
    a, b = max(i1 - 4, 0), min(i1 + 5, centers.size)
    i1 = a + int(np.argmax(smooth[a:b]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = peak_widths(smooth, [i1], rel_height=0.5)[0][0] * width
        s1 = w / 2.3548
    except Exception:
        s1 = 0.0
    if not np.isfinite(s1) or s1 <= 0 or s1 > 0.25 * x1:
        s1 = 0.05 * x1
    s1 = max(s1, width)
    return x1, x2, s1


def fit_cell_cycle(
    values,
    bins: int = 256,
    g1_guess: float | None = None,
    range: tuple[float, float] | None = None,  # noqa: A002
    model: str = "dean_jett_fox",
    debris: bool = False,
    aggregates: bool = False,
) -> CellCycleResult:
    """Fit G1 / S / G2-M fractions to linear DNA-content values.

    model: "dean_jett_fox" (S = Gaussian-smoothed rectangle, joint fit) or
    "watson" (Watson pragmatic: G1 / G2-M Gaussians fitted on their outer
    halves, S by subtraction). debris / aggregates add an exponentially
    decreasing debris component below G1 and a doublet component (the
    self-convolution of the singlet distribution); the fit then covers the
    whole histogram. %G1/%S/%G2M are percentages of the cycling singlets;
    percent_debris / percent_aggregates are percentages of all modelled events.
    """
    model = cell_cycle_model_key(model)
    v = _finite(values)
    n = int(v.size)
    if n < 50:
        raise ValueError("at least 50 events are needed for a cell-cycle fit")
    if range is None:
        lo = max(0.0, float(v.min())) if v.min() >= 0 else float(np.percentile(v, 0.1))
        hi = float(np.percentile(v, 99.9)) * 1.1
    else:
        lo, hi = map(float, range)
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    x0 = float(edges[0])

    x1, x2, s1 = _cycle_guesses(centers, counts, g1_guess)
    ratio0 = float(np.clip(x2 / x1, 1.9, 2.1)) if x1 > 0 else 2.0
    s2 = s1 * ratio0
    fit_lo = lo if debris else max(lo, x1 - 5 * s1)
    fit_hi = hi if aggregates else min(hi, ratio0 * x1 + 5 * s2)
    sel = (centers >= fit_lo) & (centers <= fit_hi)
    xs, ys = centers[sel], counts[sel]
    total = max(ys.sum(), 1.0)
    in1 = np.abs(xs - x1) < 2 * s1
    in2 = np.abs(xs - ratio0 * x1) < 2 * s2
    n1 = max(ys[in1].sum(), 1.0)
    n2 = max(ys[in2].sum(), 0.05 * total)
    extra0: list[float] = []
    extra_lb: list[float] = []
    extra_ub: list[float] = []
    nd0 = na0 = 0.0
    if debris:
        cut = x1 - 3 * s1
        below = xs < cut
        cnt = float(ys[below].sum())
        decay0 = float(np.average(xs[below] - x0, weights=ys[below])) if cnt > 0 else 0.3 * x1
        decay0 = float(np.clip(decay0, width, 1.5 * max(x1, width)))
        nd0 = max(cnt / max(1.0 - math.exp(-max(cut - x0, 0.0) / decay0), 0.05), 1.0)
        extra0 += [nd0, decay0]
        extra_lb += [0.0, 0.5 * width]
        extra_ub += [np.inf, 3.0 * max(x1, width)]
    if aggregates:
        above = xs > ratio0 * x1 + 3 * s2
        na0 = max(float(ys[above].sum()) / 0.7, 1.0)
        extra0 += [na0]
        extra_lb += [0.0]
        extra_ub += [np.inf]
    ns = max(total - n1 - n2 - nd0 - na0, 0.05 * total)
    p0 = np.array([n1, ns, n2, x1, s1, s2, ratio0] + extra0)
    lb = [0, 0, 0, 0.7 * x1, 0.5 * width, 0.5 * width, 1.9] + extra_lb
    ub = [np.inf, np.inf, np.inf, 1.3 * x1, 0.3 * x1, 0.6 * x1, 2.1] + extra_ub
    p0 = np.clip(p0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
    wts = 1.0 / np.sqrt(ys + 1.0)

    def components(x, p):
        g1, s, g2 = cell_cycle_components(x, *p[:7], width=width)
        k = 7
        deb = agg = None
        if debris:
            deb = debris_component(x, p[k], p[k + 1], x0, width)
            k += 2
        if aggregates:
            agg = aggregate_component(x, p[k], *p[:7], width=width)
        return g1, s, g2, deb, agg

    def resid(p):
        parts = components(xs, p)
        return (sum(c for c in parts if c is not None) - ys) * wts

    success, message = True, ""
    try:
        sol = least_squares(resid, p0, bounds=(lb, ub), x_scale="jac", max_nfev=4000)
        p = sol.x
        success, message = bool(sol.success), str(sol.message)
    except Exception as exc:  # pragma: no cover - defensive
        p, success, message = p0, False, str(exc)
    if (debris or aggregates) and success:
        # data-weighted least squares underestimates sparse regions (the
        # debris tail, the doublet peaks) by ~1 count per bin; polish with the
        # Poisson deviance (maximum likelihood) from that solution
        def dev(q):
            return _poisson_deviance_residuals(sum(c for c in components(xs, q) if c is not None), ys)

        try:
            start = np.clip(p, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
            sol2 = least_squares(dev, start, bounds=(lb, ub), x_scale="jac", max_nfev=4000)
            if sol2.success:
                p = sol2.x
        except Exception:  # pragma: no cover - defensive
            pass
    p = np.asarray(p, dtype=float)
    n1, ns, n2, mu, s1, s2, ratio = p[:7]
    nd = float(p[7]) if debris else 0.0
    decay = float(p[8]) if debris else float("nan")
    na = float(p[-1]) if aggregates else 0.0
    g1, s, g2, deb, agg = components(centers, p)
    zeros = np.zeros_like(centers)
    if model == "watson":
        n1, ns, n2, mu, s1, s2, ratio, g1, s, g2, agg, ok, msg = _watson_pragmatic(
            centers, counts, width, (fit_lo, fit_hi), p[:7], s, deb if deb is not None else zeros, na if aggregates else 0.0
        )
        success, message = success and ok, msg or message
    model_curve = g1 + s + g2 + (deb if deb is not None else zeros) + (agg if agg is not None else zeros)
    cycling = n1 + ns + n2
    pct = (100 * n1 / cycling, 100 * ns / cycling, 100 * n2 / cycling) if cycling > 0 else (float("nan"),) * 3
    everything = cycling + nd + na
    rms = float(np.sqrt(np.mean((model_curve[sel] - counts[sel]) ** 2)))
    curves = {"G1": g1, "S": s, "G2/M": g2}
    if deb is not None:
        curves["Debris"] = deb
    if agg is not None:
        curves["Aggregates"] = agg
    curves["Model"] = model_curve
    return CellCycleResult(
        percent_g1=float(pct[0]),
        percent_s=float(pct[1]),
        percent_g2m=float(pct[2]),
        g1_mean=float(mu),
        g2_mean=float(mu * ratio),
        g1_cv=float(100 * s1 / mu),
        g2_cv=float(100 * s2 / (mu * ratio)),
        g1_sd=float(s1),
        g2_sd=float(s2),
        centers=centers,
        counts=counts,
        curves=curves,
        rms=rms,
        n_events=n,
        fit_range=(float(fit_lo), float(fit_hi)),
        success=success,
        message=message,
        model=model,
        percent_debris=float(100 * nd / everything) if debris and everything > 0 else float("nan"),
        percent_aggregates=float(100 * na / everything) if aggregates and everything > 0 else float("nan"),
        debris_decay=decay,
    )


def _poisson_deviance_residuals(model, counts):
    """Signed square roots of the Poisson deviance terms: least squares on
    these is maximum likelihood for histogram counts."""
    m = np.maximum(model, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        ylog = np.where(counts > 0, counts * np.log(counts / m), 0.0)
    return np.sign(counts - m) * np.sqrt(np.maximum(2.0 * (ylog - (counts - m)), 0.0))


def _watson_pragmatic(centers, counts, width, fit_range, p, plateau, deb, na, iterations: int = 4):
    """Watson pragmatic refinement of a joint fit `p` (n1, ns, n2, mu, s1, s2,
    ratio). G1 is refitted on its outer half (from 4 SD below the mean to
    1.5 SD above it) and G2/M on its outer half (1.5 SD below to 4 SD above),
    with debris and aggregates as a fixed background and the S phase under each
    peak as a Gaussian-smoothed step whose height is the S level measured just
    inside that peak (so early- or late-S accumulation is followed). The S
    phase is everything in the cycle region not explained by G1, G2/M, debris
    and aggregates; a few iterations update the S levels and the doublet
    shape."""
    n1, ns, n2, mu, s1, s2, ratio = (float(q) for q in p)
    m2 = mu * ratio
    fit_lo, fit_hi = fit_range
    ok, msg = True, ""

    def agg_of():
        if na <= 0:
            return np.zeros_like(centers)
        return aggregate_component(centers, na, n1, ns, n2, mu, s1, s2, m2 / mu, width)

    def refit(mask, start, lb, ub, bg):
        x, y, b = centers[mask], counts[mask], bg[mask]

        def r(q):
            return _poisson_deviance_residuals(width * q[0] * _gauss(x, q[1], q[2]) + b, y)

        start = np.clip(start, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
        return least_squares(r, start, bounds=(lb, ub), x_scale="jac").x

    agg = agg_of()
    h = float(np.max(plateau)) if plateau.size else 0.0  # plateau height per bin
    h1 = h2 = h
    region = (centers >= mu - 4 * s1) & (centers <= m2 + 4 * s2)
    rest = np.where(region, counts - deb - agg, 0.0)
    g1 = width * n1 * _gauss(centers, mu, s1)
    g2 = width * n2 * _gauss(centers, m2, s2)
    for _ in range(iterations):
        left = (centers >= max(fit_lo, mu - 4 * s1)) & (centers <= mu + 1.5 * s1)
        right = (centers >= m2 - 1.5 * s2) & (centers <= min(fit_hi, m2 + 4 * s2))
        try:
            if np.count_nonzero(left) >= 4:
                bg1 = h1 * ndtr((centers - mu) / s1) + deb + agg
                n1, mu, s1 = (float(q) for q in refit(left, [n1, mu, s1], [0, 0.9 * mu, 0.5 * s1], [np.inf, 1.1 * mu, 2.0 * s1], bg1))
            if np.count_nonzero(right) >= 4:
                bg2 = h2 * ndtr((m2 - centers) / s2) + deb + agg
                lo2, hi2 = max(1.8 * mu, 0.95 * m2), min(2.2 * mu, 1.05 * m2)
                if hi2 <= lo2:
                    lo2, hi2 = 1.8 * mu, 2.2 * mu
                n2, m2, s2 = (float(q) for q in refit(right, [n2, m2, s2], [0, lo2, 0.5 * s2], [np.inf, hi2, 2.0 * s2], bg2))
        except Exception as exc:  # pragma: no cover - defensive
            ok, msg = False, str(exc)
            break
        g1 = width * n1 * _gauss(centers, mu, s1)
        g2 = width * n2 * _gauss(centers, m2, s2)
        region = (centers >= mu - 4 * s1) & (centers <= m2 + 4 * s2)
        rest = np.where(region, counts - g1 - g2 - deb - agg, 0.0)
        ns = max(float(rest.sum()), 0.0)
        # S level just inside each peak
        w1 = (centers >= mu + 2.5 * s1) & (centers <= mu + 4.0 * s1)
        w2 = (centers >= m2 - 4.0 * s2) & (centers <= m2 - 2.5 * s2)
        if w1.any():
            h1 = max(float(rest[w1].mean()), 0.0)
        if w2.any():
            h2 = max(float(rest[w2].mean()), 0.0)
        agg = agg_of()
    s_curve = np.where(region, np.clip(gaussian_filter1d(rest, 1.0), 0.0, None), 0.0)
    return n1, ns, n2, mu, s1, s2, m2 / mu, g1, s_curve, g2, (agg if na > 0 else None), ok, msg


# =============================================================================
# proliferation
# =============================================================================
@dataclass
class ProliferationResult:
    percents: np.ndarray  # percent of cells in generation 0, 1, ...
    counts_per_generation: np.ndarray
    positions: np.ndarray  # generation peak positions in fit coordinates
    sd: float
    division_index: float
    proliferation_index: float
    percent_divided: float
    expansion_index: float
    replication_index: float
    centers: np.ndarray
    counts: np.ndarray
    curves: dict[str, np.ndarray]  # "Gen 0", "Gen 1", ..., "Model"
    rms: float
    n_events: int
    log: bool = True

    @property
    def n_generations(self) -> int:
        return int(self.percents.size)

    @property
    def generation0(self) -> float:
        """Generation-0 peak position in data units (10**pos when fitted in log10)."""
        p = float(self.positions[0])
        return 10 ** p if self.log else p

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, pct in enumerate(self.percents):
            out[f"%Gen {k}"] = float(pct)
        out.update(
            {
                "Division Index": self.division_index,
                "Proliferation Index": self.proliferation_index,
                "%Divided": self.percent_divided,
                "Expansion Index": self.expansion_index,
                "Replication Index": self.replication_index,
                "Gen 0 Peak": self.generation0,
                "RMS": self.rms,
                "Events": self.n_events,
            }
        )
        return out


def proliferation_indices(cells_per_generation) -> dict[str, float]:
    """FlowJo/Roederer proliferation indices from cell counts per generation."""
    n = np.asarray(cells_per_generation, dtype=float)
    gen = np.arange(n.size)
    precursors = n / 2.0 ** gen
    total_p = precursors.sum()
    divided_p = precursors[1:].sum()
    nan = float("nan")
    return {
        "division_index": float((gen * precursors).sum() / total_p) if total_p > 0 else nan,
        "proliferation_index": float((gen * precursors).sum() / divided_p) if divided_p > 0 else nan,
        "percent_divided": float(100 * divided_p / total_p) if total_p > 0 else nan,
        "expansion_index": float(n.sum() / total_p) if total_p > 0 else nan,
        "replication_index": float(n[1:].sum() / divided_p) if divided_p > 0 else nan,
    }


def fit_proliferation(
    values,
    bins: int = 256,
    max_generations: int = 10,
    log: bool = True,
    g0_guess: float | None = None,
    forward: Callable | None = None,
    inverse: Callable | None = None,
    spacing: float = LOG2,
    fix_g0: bool = False,
) -> ProliferationResult:
    """Fit a dye-dilution histogram with one Gaussian per generation.

    log=True: `values` are linear intensities and are fitted as log10(values)
    (non-positive values are dropped); generation k sits at pos0 - k*log10(2).
    log=False: `values` are already in a log-like coordinate (e.g. the 0..1
    display space). With `forward`/`inverse` (the display transform and its
    inverse) generation k sits at forward(inverse(pos0) / 2**k), which is exact
    for any transform; without them peaks are spaced by `spacing`.
    g0_guess is in the fitting coordinate (log10 units when log=True); it is
    where the search for the generation-0 (undivided) peak starts -- without
    it the brightest clear peak is taken. fix_g0=True holds generation 0
    exactly at g0_guess (e.g. the median of an undivided control) and only
    fits the peak width and the amplitudes.
    """
    v = _finite(values)
    if log:
        v = np.log10(v[v > 0])
        forward = inverse = None
        spacing = LOG2
    n = int(v.size)
    if n < 50:
        raise ValueError("at least 50 events are needed for a proliferation fit")
    max_generations = max(1, int(max_generations))
    lo, hi = float(np.percentile(v, 0.05)), float(np.percentile(v, 99.95))
    if g0_guess is not None and math.isfinite(float(g0_guess)):
        hi = max(hi, float(g0_guess))  # keep a given generation 0 on the histogram
    if hi <= lo:
        hi = lo + 1.0

    def positions(p0: float) -> np.ndarray:
        k = np.arange(max_generations)
        if forward is not None and inverse is not None:
            base = float(np.asarray(inverse(np.array([p0], dtype=float)))[0])
            return np.asarray(forward(base / 2.0 ** k), dtype=float)
        return p0 - k * spacing

    # local spacing near the top, used to size the search grid
    def local_spacing(p0: float) -> float:
        if forward is not None and inverse is not None:
            pos = positions(p0)
            return float(abs(pos[0] - pos[1])) if pos.size > 1 else spacing
        return spacing

    pad = 0.5 * local_spacing(hi)
    counts, edges = np.histogram(v, bins=int(bins), range=(lo - pad, hi + pad))
    counts = counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])

    if g0_guess is None:
        fix_g0 = False
        smooth = gaussian_filter1d(counts, 1.5)
        peaks, _ = find_peaks(smooth, height=0.05 * smooth.max(), prominence=0.02 * smooth.max())
        g0 = float(centers[peaks[-1]]) if peaks.size else float(np.percentile(v, 99))
    else:
        g0 = float(g0_guess)
    dsp = local_spacing(g0)
    wts = 1.0 / np.sqrt(counts + 1.0)

    def solve(p0: float, sd: float):
        pos = positions(p0)
        keep = pos > edges[0] - 3 * sd
        keep[0] = True
        basis = np.zeros((centers.size, max_generations))
        basis[:, keep] = width * _gauss(centers[:, None], pos[None, keep], sd)
        amp, rnorm = nnls(basis * wts[:, None], counts * wts, maxiter=50 * max_generations)
        return amp, rnorm, basis, pos

    if fix_g0:
        # generation 0 is given: only the shared width is searched
        sds = np.linspace(0.06, 0.5, 23) * dsp
        res = [solve(g0, sd)[1] for sd in sds]
        k = int(np.argmin(res))
        a, b = sds[max(k - 1, 0)], sds[min(k + 1, sds.size - 1)]
        a = max(a, 0.26 * width)
        p0, sd = g0, float(sds[k])
        if b > a:
            opt = minimize_scalar(lambda z: solve(g0, z)[1], bounds=(a, b), method="bounded", options={"xatol": 1e-4 * dsp})
            if opt.fun <= res[k]:
                sd = float(opt.x)
    else:
        # coarse grid around the brightest peak, then a local polish
        best = (np.inf, g0, 0.2 * dsp)
        for p0 in g0 + np.linspace(-0.4, 0.4, 17) * dsp:
            for sd in np.linspace(0.06, 0.5, 12) * dsp:
                r = solve(p0, sd)[1]
                if r < best[0]:
                    best = (r, p0, sd)

        def objective(z):
            p0, sd = z
            if sd <= 0.25 * width or abs(p0 - g0) > 0.6 * dsp:
                return 1e300
            return solve(p0, sd)[1]

        opt = minimize(objective, [best[1], best[2]], method="Nelder-Mead", options={"xatol": 1e-4 * dsp, "fatol": 1e-6, "maxiter": 400})
        p0, sd = (opt.x if opt.fun <= best[0] else (best[1], best[2]))
    amp, _, basis, pos = solve(float(p0), float(sd))
    curves = {f"Gen {k}": basis[:, k] * amp[k] for k in range(max_generations)}
    model = basis @ amp
    curves["Model"] = model
    # trim generations that never got any cells at the dim end
    last = int(np.max(np.nonzero(amp > 0)[0])) + 1 if np.any(amp > 0) else 1
    amp = amp[:last]
    pos = pos[:last]
    curves = {k: c for k, c in curves.items() if k == "Model" or int(k.split()[1]) < last}
    total = amp.sum()
    pct = 100 * amp / total if total > 0 else np.full(last, np.nan)
    idx = proliferation_indices(amp)
    return ProliferationResult(
        percents=pct,
        counts_per_generation=amp,
        positions=pos,
        sd=float(sd),
        centers=centers,
        counts=counts,
        curves=curves,
        rms=float(np.sqrt(np.mean((model - counts) ** 2))),
        n_events=n,
        log=bool(log),
        **idx,
    )


# =============================================================================
# kinetics
# =============================================================================
KINETIC_STATISTICS = ("median", "mean", "percent_positive")


@dataclass
class KineticsResult:
    centers: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    lo: np.ndarray  # lower quartile of the parameter per bin
    hi: np.ndarray  # upper quartile
    edges: np.ndarray
    statistic: str = "median"
    threshold: float | None = None
    overall: float = float("nan")
    slope: float = float("nan")  # linear trend of the statistic per time unit
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        finite = self.values[np.isfinite(self.values)]
        vmin = float(finite.min()) if finite.size else float("nan")
        vmax = float(finite.max()) if finite.size else float("nan")
        spread = 100 * (vmax - vmin) / abs(self.overall) if finite.size and self.overall else float("nan")
        return {
            "Events": int(self.counts.sum()),
            "Bins": int(self.centers.size),
            "Overall": self.overall,
            "Min Bin": vmin,
            "Max Bin": vmax,
            "Range % of Overall": spread,
            "Slope / time": self.slope,
        }


def _bin_stat(values: np.ndarray, statistic: str, threshold: float | None) -> float:
    if values.size == 0:
        return float("nan")
    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(values.mean())
    return float(100.0 * np.count_nonzero(values > threshold) / values.size)


def kinetics(time_values, param_values, n_bins: int = 20, statistic: str = "median", mask=None, threshold: float | None = None, binning: str = "equal_count") -> KineticsResult:
    """A statistic of `param_values` per time bin.

    binning: "equal_count" (each bin holds ~the same number of events) or
    "equal_width". statistic: "median", "mean" or "percent_positive" (percent of
    events above `threshold`). Returns bin centres, statistic, counts and the
    lower/upper quartiles of the parameter per bin.
    """
    statistic = statistic.lower().replace(" ", "_").replace("%", "percent_").strip("_")
    if statistic in ("percent", "percent_positive", "positive", "percentpositive"):
        statistic = "percent_positive"
    if statistic not in KINETIC_STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}")
    if statistic == "percent_positive" and threshold is None:
        raise ValueError("percent_positive needs a threshold")
    t = np.asarray(time_values, dtype=np.float64).ravel()
    y = np.asarray(param_values, dtype=np.float64).ravel()
    if t.shape != y.shape:
        raise ValueError("time and parameter values differ in length")
    keep = np.isfinite(t) & np.isfinite(y)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool).ravel()
    t, y = t[keep], y[keep]
    n_bins = max(1, int(n_bins))
    if t.size == 0:
        empty = np.full(n_bins, np.nan)
        return KineticsResult(empty, empty.copy(), np.zeros(n_bins, int), empty.copy(), empty.copy(), np.full(n_bins + 1, np.nan), statistic, threshold)
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    if binning == "equal_width":
        edges = np.linspace(t[0], t[-1], n_bins + 1)
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
        splits = np.searchsorted(idx, np.arange(1, n_bins))
    elif binning == "equal_count":
        splits = np.linspace(0, t.size, n_bins + 1).round().astype(int)[1:-1]
        bounds = np.concatenate([[0], splits, [t.size]])
        edges = np.array([t[0]] + [t[min(b, t.size - 1)] for b in bounds[1:-1]] + [t[-1]], dtype=float)
    else:
        raise ValueError("binning must be 'equal_count' or 'equal_width'")
    t_chunks = np.split(t, splits)
    y_chunks = np.split(y, splits)
    centers = np.empty(n_bins)
    vals = np.empty(n_bins)
    counts = np.empty(n_bins, dtype=int)
    q1 = np.empty(n_bins)
    q3 = np.empty(n_bins)
    for i, (tc, yc) in enumerate(zip(t_chunks, y_chunks)):
        counts[i] = yc.size
        if binning == "equal_width":
            centers[i] = 0.5 * (edges[i] + edges[i + 1])
        else:
            centers[i] = float(tc.mean()) if tc.size else float("nan")
        vals[i] = _bin_stat(yc, statistic, threshold)
        if yc.size:
            q1[i], q3[i] = np.percentile(yc, [25, 75])
        else:
            q1[i] = q3[i] = float("nan")
    ok = np.isfinite(vals) & np.isfinite(centers)
    slope = float(np.polyfit(centers[ok], vals[ok], 1)[0]) if ok.sum() >= 2 and np.ptp(centers[ok]) > 0 else float("nan")
    return KineticsResult(centers, vals, counts, q1, q3, edges, statistic, threshold, _bin_stat(y, statistic, threshold), slope)


KINETICS_HEADER = ["Bin", "Time", "Statistic", "Count", "Q1", "Q3"]


def kinetics_table(result: KineticsResult, include_header: bool = False) -> list[list]:
    """One row per time bin (Bin, Time, Statistic, Count, Q1, Q3), for CSV."""
    rows: list[list] = [list(KINETICS_HEADER)] if include_header else []
    for i in range(result.centers.size):
        rows.append([i + 1, float(result.centers[i]), float(result.values[i]), int(result.counts[i]), float(result.lo[i]), float(result.hi[i])])
    return rows


# =============================================================================
# parameter choice and population helpers (duck-typed Sample / Population)
# =============================================================================
def _is_fluorescence(name: str) -> bool:
    upper = name.upper()
    return not any(upper.startswith(p) for p in SCATTER_PREFIXES)


def _is_linear(sample, name: str) -> bool:
    from .transforms import LinearScale

    try:
        return isinstance(sample.scale(name), LinearScale)
    except Exception:
        return False


def _matches(sample, name: str, keywords) -> bool:
    label = f"{name} {sample.param_stain(name) if hasattr(sample, 'param_stain') else ''}".upper()
    tokens = label.replace("-", " ").replace("_", " ").replace("/", " ").split()
    for kw in keywords:
        if len(kw) <= 2:  # short keys (PI) must be whole tokens
            if kw in tokens:
                return True
        elif kw in label or kw.replace("-", "") in label.replace("-", ""):
            return True
    return False


def _candidate_params(sample) -> list[str]:
    try:
        names = sample.parameter_names()
    except Exception:
        names = list(getattr(sample, "channel_names", []))
    return [n for n in names if n not in getattr(sample, "computed", {})]


def default_dna_parameter(sample) -> str | None:
    """A linear parameter named like a DNA dye, else the first linear
    fluorescence parameter, else any DNA-named / fluorescence parameter."""
    names = _candidate_params(sample)
    fluo = [n for n in names if _is_fluorescence(n)]
    named = [n for n in names if _matches(sample, n, DNA_KEYWORDS)]
    for group in ([n for n in named if _is_linear(sample, n)], [n for n in fluo if _is_linear(sample, n)], named, fluo, names):
        if group:
            return group[0]
    return None


def default_dye_parameter(sample) -> str | None:
    names = _candidate_params(sample)
    fluo = [n for n in names if _is_fluorescence(n)]
    named = [n for n in names if _matches(sample, n, DYE_KEYWORDS)]
    for group in (named, fluo, names):
        if group:
            return group[0]
    return None


def default_time_parameter(sample) -> str | None:
    names = _candidate_params(sample)
    for n in names:
        if n.upper() == "TIME" or n.upper().startswith("TIME"):
            return n
    return names[0] if names else None


def population_values(population, param: str, display: bool = False) -> np.ndarray:
    sample = population.sample
    if sample is None or not sample.ensure_loaded():
        raise ValueError("sample data is not available")
    data = sample.display(param) if display else sample.data(param)
    return np.asarray(data)[population.mask()]


def cell_cycle_for_population(population, param: str, bins: int = 256, g1_guess: float | None = None, model: str = "dean_jett_fox", debris: bool = False, aggregates: bool = False) -> CellCycleResult:
    return fit_cell_cycle(population_values(population, param), bins=bins, g1_guess=g1_guess, model=model, debris=debris, aggregates=aggregates)


def control_g0(population, param: str) -> float:
    """Generation-0 position from an undivided control: the median of the
    control population's `param` values, in data units (the median commutes
    with the monotonic display transforms, so it is the peak position on
    every scale)."""
    values = _finite(population_values(population, param))
    if values.size == 0:
        raise ValueError(f"the control population {getattr(population, 'display_name', '')!r} has no events")
    return float(np.median(values))


def proliferation_for_population(
    population,
    param: str,
    bins: int = 256,
    max_generations: int = 10,
    display: bool = True,
    g0_guess: float | None = None,
    g0: float | None = None,
    fix_g0: bool = False,
) -> ProliferationResult:
    """display=True fits the 0..1 display values (the sample's logicle/log
    transform) with generations placed exactly at halved intensities;
    display=False fits log10 of the raw values.

    g0_guess is in the fitting coordinate; g0 is the generation-0 (undivided)
    position in data units (e.g. `control_g0(control, param)`) and is
    converted to the fitting coordinate. fix_g0 holds generation 0 there."""
    if g0 is not None:
        g0 = float(g0)
        if display:
            g0_guess = float(population.sample.scale(param).forward_scalar(g0))
        else:
            if g0 <= 0:
                raise ValueError("the generation-0 position must be positive for a log10 fit")
            g0_guess = math.log10(g0)
    if display:
        scale = population.sample.scale(param)
        return fit_proliferation(population_values(population, param, display=True), bins=bins, max_generations=max_generations, log=False, g0_guess=g0_guess, forward=scale.forward, inverse=scale.inverse, fix_g0=fix_g0)
    return fit_proliferation(population_values(population, param), bins=bins, max_generations=max_generations, log=True, g0_guess=g0_guess, fix_g0=fix_g0)


def kinetics_for_population(population, time_param: str, param: str, n_bins: int = 20, statistic: str = "median", threshold: float | None = None, binning: str = "equal_count") -> KineticsResult:
    sample = population.sample
    if sample is None or not sample.ensure_loaded():
        raise ValueError("sample data is not available")
    return kinetics(sample.data(time_param), sample.data(param), n_bins=n_bins, statistic=statistic, mask=population.mask(), threshold=threshold, binning=binning)


# =============================================================================
# fits as population statistics (Table Editor, plate editor, Add Statistic)
# =============================================================================
FIT_CACHE_SIZE = 128
_FIT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_FIT_CACHE_INFO = {"hits": 0, "misses": 0}
FIT_KINDS = ("cell_cycle", "proliferation")


def clear_fit_cache():
    _FIT_CACHE.clear()
    _FIT_CACHE_INFO.update(hits=0, misses=0)


def fit_cache_info() -> dict:
    return {**_FIT_CACHE_INFO, "size": len(_FIT_CACHE)}


def _scale_key(sample, param: str) -> str:
    try:
        return json.dumps(sample.scale(param).to_dict(), sort_keys=True, default=str)
    except Exception:
        return ""


def cached_population_fit(kind: str, population, param: str):
    """The default cell-cycle ("cell_cycle": Dean-Jett-Fox, 256 bins) or
    proliferation ("proliferation": display scale, 10 generations, automatic
    generation 0) fit of a population, from a small LRU cache keyed by the
    population identity, parameter, event count and a fingerprint of the
    values (and the display scale for proliferation), so that several table
    columns over the same population run the fit once. Raises ValueError when
    the fit is not possible (no data, too few events)."""
    if kind not in FIT_KINDS:
        raise ValueError(f"unknown fit {kind!r}")
    sample = population.sample
    if sample is None or not sample.ensure_loaded():
        raise ValueError("sample data is not available")
    raw = np.ascontiguousarray(population_values(population, param))
    key = (
        kind,
        getattr(population, "id", id(population)),
        getattr(sample, "id", None),
        param,
        int(raw.size),
        hash(raw.tobytes()),
        _scale_key(sample, param) if kind == "proliferation" else "",
    )
    if key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        _FIT_CACHE_INFO["hits"] += 1
        result = _FIT_CACHE[key]
    else:
        _FIT_CACHE_INFO["misses"] += 1
        try:
            if kind == "cell_cycle":
                result = fit_cell_cycle(raw)
            else:
                result = proliferation_for_population(population, param)
        except (ValueError, KeyError, RuntimeError) as exc:
            result = ValueError(str(exc))
        _FIT_CACHE[key] = result
        while len(_FIT_CACHE) > FIT_CACHE_SIZE:
            _FIT_CACHE.popitem(last=False)
    if isinstance(result, Exception):
        raise ValueError(str(result))
    return result
