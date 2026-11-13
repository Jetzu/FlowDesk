"""Spillover matrices and their estimation from single-stain controls."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .transforms import default_biex

ACQUISITION_DEFINED = "Acquisition-defined"


class SpilloverMatrix:
    """Square spillover matrix: row = fluorochrome (source), column = detector.

    Compensated values are obtained as observed @ inverse(matrix).
    """

    def __init__(self, name: str, params: list[str], matrix: np.ndarray, prefix: str = "Comp-", suffix: str = ""):
        self.name = name
        self.params = list(params)
        self.matrix = np.array(matrix, dtype=float).reshape(len(self.params), len(self.params))
        self.prefix = prefix
        self.suffix = suffix
        # filled by estimate_spillover / refine_spillover (never saved)
        self.diagnostics: dict = {}
        self.report = None

    @classmethod
    def identity(cls, params: list[str], name: str = "Identity") -> "SpilloverMatrix":
        return cls(name, params, np.eye(len(params)))

    @classmethod
    def from_fcs(cls, fcs, name: str = ACQUISITION_DEFINED) -> "SpilloverMatrix | None":
        spill = fcs.spillover()
        if spill is None:
            return None
        names, matrix = spill
        return cls(name, names, matrix)

    def compensated_name(self, param: str) -> str:
        return f"{self.prefix}{param}{self.suffix}"

    def base_name(self, comp_name: str) -> str | None:
        if self.prefix and not comp_name.startswith(self.prefix):
            return None
        if self.suffix and not comp_name.endswith(self.suffix):
            return None
        inner = comp_name[len(self.prefix) : len(comp_name) - len(self.suffix) if self.suffix else None]
        return inner if inner in self.params else None

    @property
    def compensated_names(self) -> list[str]:
        return [self.compensated_name(p) for p in self.params]

    @property
    def is_identity(self) -> bool:
        return np.allclose(self.matrix, np.eye(len(self.params)))

    def inverse(self) -> np.ndarray:
        try:
            return np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.matrix)

    def apply(self, events: np.ndarray, channel_names: list[str]) -> np.ndarray | None:
        """Return the compensated columns (N x n) for self.params, or None if a
        parameter is missing from the sample."""
        try:
            idx = [channel_names.index(p) for p in self.params]
        except ValueError:
            return None
        observed = events[:, idx].astype(np.float64)
        return (observed @ self.inverse()).astype(np.float32)

    # ------------------------------------------------------ channel matching
    def resolve_params(self, channels) -> dict[str, str]:
        """Matrix parameter -> sample channel ($PnN) name, for the parameters
        found by $PnN name or, failing that, by $PnS stain name (exact, then
        case-insensitive). `channels`: Channel objects or plain names."""
        names = [getattr(c, "name", c) for c in channels]
        stains = [getattr(c, "stain", "") or "" for c in channels]
        lower = {n.lower(): n for n in names}
        out: dict[str, str] = {}
        for p in self.params:
            if p in names:
                out[p] = p
                continue
            hits = [n for n, st in zip(names, stains) if st and st == p]
            if not hits:
                hits = [n for n, st in zip(names, stains) if st and st.strip().lower() == p.strip().lower()]
            if len(hits) == 1:
                out[p] = hits[0]
            elif p.lower() in lower:
                out[p] = lower[p.lower()]
        return out

    def missing_params(self, channels) -> list[str]:
        """Matrix parameters the sample does not have (the matrix cannot be
        applied while this is not empty)."""
        found = self.resolve_params(channels)
        return [p for p in self.params if p not in found]

    def for_channels(self, channels) -> "SpilloverMatrix | None":
        """This matrix with its parameters renamed to the sample's $PnN names
        (a matrix written with $PnS stain names), itself when nothing needs
        renaming, None when parameters are missing."""
        found = self.resolve_params(channels)
        if len(found) != len(self.params) or len(set(found.values())) != len(self.params):
            return None
        if all(found[p] == p for p in self.params):
            return self
        return SpilloverMatrix(self.name, [found[p] for p in self.params], self.matrix.copy(), self.prefix, self.suffix)

    def copy(self, name: str | None = None) -> "SpilloverMatrix":
        return SpilloverMatrix(name or self.name, self.params, self.matrix.copy(), self.prefix, self.suffix)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": list(self.params),
            "matrix": self.matrix.tolist(),
            "prefix": self.prefix,
            "suffix": self.suffix,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpilloverMatrix":
        return cls(d["name"], d["params"], np.array(d["matrix"], dtype=float), d.get("prefix", "Comp-"), d.get("suffix", ""))

    def __eq__(self, other):
        return isinstance(other, SpilloverMatrix) and self.params == other.params and np.allclose(self.matrix, other.matrix)

    def __hash__(self):
        return id(self)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Two-class threshold on a 1D array (Otsu's method)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    hist, edges = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return float(edges[0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_total = (hist * centers).sum()
    sum_bg = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~np.isfinite(between)] = -1
    k = int(np.argmax(between))
    return float(edges[k + 1])


def split_positive(primary: np.ndarray, top: float = 262144.0) -> np.ndarray:
    """Boolean mask of the positive population of a single-stain control."""
    scale = default_biex(max(top, float(np.nanmax(primary)) if primary.size else top))
    disp = scale.forward(primary)
    thr = otsu_threshold(disp)
    pos = disp >= thr
    if pos.sum() < 10 or (~pos).sum() < 10:
        # fall back to the brightest 10%
        cutoff = np.percentile(primary, 90)
        pos = primary >= cutoff
    return pos


# ------------------------------------------------------------- control gates
# The wizard separates every single-stain control into a positive and a
# negative population on its primary (own) detector. By default the split is
# automatic (Otsu on the biex display scale); the user can replace it per
# control by explicit raw-value ranges.

MIN_EVENTS = 20  # fewer events in a control: its row is left uncompensated
MIN_POPULATION = 10  # fewer positive (or negative) events: population missing
NO_POSITIVE_STAIN_INDEX = 3.0  # (pos - neg) / (2 rSD neg) below this: no distinct positive population
WEAK_STAIN_INDEX = 10.0  # below this: weak positive population (row computed, flagged)


@dataclass
class ControlGates:
    """User-set positive / negative ranges (raw primary-detector values) of
    one single-stain control. None keeps the automatic choice."""

    pos: tuple[float, float] | None = None
    neg: tuple[float, float] | None = None

    @classmethod
    def from_threshold(cls, threshold: float) -> "ControlGates":
        return cls((float(threshold), math.inf), (-math.inf, float(threshold)))

    @property
    def is_auto(self) -> bool:
        return self.pos is None and self.neg is None

    def to_dict(self) -> dict:
        def enc(r):
            return None if r is None else [_enc_float(r[0]), _enc_float(r[1])]

        return {"pos": enc(self.pos), "neg": enc(self.neg)}

    @classmethod
    def from_dict(cls, d) -> "ControlGates":
        def dec(r):
            if r is None:
                return None
            return (_dec_float(r[0]), _dec_float(r[1]))

        d = d or {}
        return cls(dec(d.get("pos")), dec(d.get("neg")))


def _enc_float(v: float):
    v = float(v)
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _dec_float(v) -> float:
    return float(v)


@dataclass
class ControlDiagnostics:
    """What the estimator saw in one single-stain control."""

    param: str
    n_events: int = 0
    n_pos: int = 0
    n_neg: int = 0
    pos_median: float = math.nan  # primary detector
    neg_median: float = math.nan
    method: str = ""  # otsu | brightest 10% | gates | masks
    negative: str = ""  # internal | unstained | zero
    threshold: float = math.nan  # raw primary value of the automatic split
    separability: float = math.nan  # Otsu's eta on the display scale (1 = perfect)
    stain_index: float = math.nan
    status: str = "ok"  # ok | weak | no positives | too few events | missing
    message: str = ""
    pos_range: tuple[float, float] | None = None  # raw primary range actually used
    neg_range: tuple[float, float] | None = None
    pos_medians: np.ndarray | None = None  # every detector, raw
    neg_medians: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def used(self) -> bool:
        """The control produced a matrix row."""
        return self.status in ("ok", "weak")

    def summary(self) -> str:
        if self.status == "missing":
            return f"{self.param}: no control"
        return f"{self.param}: {self.message}" if self.message else f"{self.param}: {self.status}"


def _otsu(values: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """(threshold, separability eta = between-class / total variance)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    thr = otsu_threshold(values, bins)
    lo, hi = values[values < thr], values[values >= thr]
    var = float(values.var())
    if lo.size == 0 or hi.size == 0 or var <= 0:
        return thr, 0.0
    w0, w1 = lo.size / values.size, hi.size / values.size
    between = w0 * w1 * (float(lo.mean()) - float(hi.mean())) ** 2
    return thr, float(min(between / var, 1.0))


def auto_split(primary: np.ndarray, top: float = 262144.0) -> tuple[np.ndarray, float, str, float]:
    """Automatic positive / negative split of a control's primary detector:
    (positive mask, raw threshold, method, separability)."""
    primary = np.asarray(primary, dtype=float)
    if primary.size == 0:
        return np.zeros(0, dtype=bool), math.nan, "otsu", 0.0
    scale = default_biex(max(top, float(np.nanmax(primary))))
    disp = scale.forward(primary)
    thr_disp, eta = _otsu(disp)
    pos = disp >= thr_disp
    method = "otsu"
    thr = float(scale.inverse_scalar(thr_disp))
    if pos.sum() < MIN_POPULATION or (~pos).sum() < MIN_POPULATION:
        thr = float(np.percentile(primary, 90))
        pos = primary >= thr
        method = "brightest 10%"
    return pos, thr, method, eta


def _in_range(values: np.ndarray, rng: tuple[float, float]) -> np.ndarray:
    lo, hi = rng
    if lo > hi:
        lo, hi = hi, lo
    return (values >= lo) & (values <= hi)


def robust_sd(values: np.ndarray) -> float:
    """Robust standard deviation (half the 15.87-84.13 percentile range)."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    lo, hi = np.percentile(values, [15.865, 84.135])
    return float(hi - lo) / 2.0


def control_masks(primary: np.ndarray, gates: ControlGates | None = None, masks=None, top: float = 262144.0, diag: ControlDiagnostics | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Positive and negative masks of one control. The negative mask is None
    when it is left to the caller (universal negative, if any)."""
    primary = np.asarray(primary, dtype=float)
    if masks is not None:
        pos, neg = (np.asarray(m, dtype=bool) for m in masks)
        if diag is not None:
            diag.method = "masks"
        return pos, neg
    auto_pos, thr, method, eta = auto_split(primary, top)
    if diag is not None:
        diag.threshold, diag.separability = thr, eta
    if gates is None or gates.is_auto:
        if diag is not None:
            diag.method = method
        return auto_pos, None
    if diag is not None:
        diag.method = "gates"
    pos = _in_range(primary, gates.pos) if gates.pos is not None else auto_pos
    if gates.neg is not None:
        neg = _in_range(primary, gates.neg)
    elif gates.pos is not None:
        neg = primary < min(gates.pos)
    else:
        neg = None
    return pos, neg


def estimate_spillover(
    controls: dict[str, np.ndarray],
    params: list[str],
    unstained: np.ndarray | None = None,
    name: str = "Computed",
    *,
    gates: dict[str, ControlGates] | None = None,
    masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    top: float = 262144.0,
    clip_negative: bool = True,
) -> SpilloverMatrix:
    """Estimate a spillover matrix (median-difference method).

    controls: parameter name -> event matrix (N x len(params)) of its single-stain
    control, columns ordered as `params`. unstained: optional N x len(params)
    matrix used for the negative medians (universal negative). gates: optional
    per-control positive / negative ranges replacing the automatic split; a
    control whose negative range is set uses its own negatives even with an
    unstained control. masks: explicit (positive, negative) event masks.

    The result carries `diagnostics`: parameter -> ControlDiagnostics."""
    n = len(params)
    gates = gates or {}
    masks = masks or {}
    matrix = np.eye(n)
    diagnostics: dict[str, ControlDiagnostics] = {}
    has_unstained = unstained is not None and np.size(unstained) > 0
    neg_global = np.median(unstained, axis=0) if has_unstained else None
    for i, p in enumerate(params):
        d = ControlDiagnostics(p)
        diagnostics[p] = d
        events = controls.get(p)
        if events is None:
            d.status, d.message = "missing", "no control assigned (not compensated)"
            continue
        events = np.asarray(events, dtype=np.float64)
        d.n_events = int(events.shape[0])
        if d.n_events < MIN_EVENTS:
            d.status, d.message = "too few events", f"only {d.n_events} events: row left uncompensated"
            continue
        primary = events[:, i]
        g = gates.get(p)
        pos_mask, neg_mask = control_masks(primary, g, masks.get(p), top, d)
        d.n_pos = int(pos_mask.sum())
        if d.n_pos < MIN_POPULATION:
            d.status, d.message = "no positives", f"only {d.n_pos} events in the positive gate: row left uncompensated"
            continue
        pos_med = np.median(events[pos_mask], axis=0)
        neg_events = None
        if neg_mask is not None and int(neg_mask.sum()) >= MIN_POPULATION:
            neg_events = events[neg_mask]
            d.negative = "internal"
        elif neg_global is not None:
            neg_events = np.asarray(unstained, dtype=np.float64)
            d.negative = "unstained"
        elif neg_mask is None and int((~pos_mask).sum()) >= MIN_POPULATION:
            neg_events = events[~pos_mask]
            d.negative = "internal"
        if neg_events is not None:
            neg_med = neg_global if d.negative == "unstained" else np.median(neg_events, axis=0)
            d.n_neg = int(neg_events.shape[0])
            neg_sd = robust_sd(neg_events[:, i])
        else:
            neg_med = np.zeros(n)
            d.negative = "zero"
            neg_sd = math.nan
        d.pos_medians, d.neg_medians = pos_med, np.asarray(neg_med, dtype=float)
        d.pos_median, d.neg_median = float(pos_med[i]), float(neg_med[i])
        pos_vals = primary[pos_mask]
        d.pos_range = (float(pos_vals.min()), float(pos_vals.max()))
        if neg_events is not None and d.negative == "internal":
            d.neg_range = (float(neg_events[:, i].min()), float(neg_events[:, i].max()))
        denom = pos_med[i] - neg_med[i]
        if denom <= 0:
            d.status, d.message = "no positives", "positive median not above the negative: row left uncompensated"
            continue
        if neg_sd and np.isfinite(neg_sd) and neg_sd > 0:
            d.stain_index = float(denom / (2.0 * neg_sd))
        auto = d.method in ("otsu", "brightest 10%")
        if auto and (d.method == "brightest 10%" or (np.isfinite(d.stain_index) and d.stain_index < NO_POSITIVE_STAIN_INDEX)):
            how = "brightest 10% used" if d.method == "brightest 10%" else f"stain index {d.stain_index:.1f}"
            d.status = "no positives"
            d.message = f"no distinct positive population ({how}): row left uncompensated, set the positive gate"
            continue
        row = (pos_med - neg_med) / denom
        if clip_negative:
            row = np.clip(row, 0.0, None)
        row[i] = 1.0
        matrix[i] = row
        if np.isfinite(d.stain_index) and d.stain_index < WEAK_STAIN_INDEX:
            d.status, d.message = "weak", f"weak positive population (stain index {d.stain_index:.1f})"
        else:
            d.message = f"{d.n_pos:,} positive / {d.n_neg:,} negative events"
    out = SpilloverMatrix(name, params, matrix)
    out.diagnostics = diagnostics
    return out


def diagnostics_warnings(diagnostics: dict[str, ControlDiagnostics]) -> list[str]:
    """One line per control that needs attention."""
    return [d.summary() for d in diagnostics.values() if d.status != "ok"]


# ----------------------------------------------------- AutoSpill refinement
@dataclass
class RefinementReport:
    """Convergence of refine_spillover and the unmixing residuals left."""

    iterations: int = 0
    converged: bool = False
    tolerance: float = 0.0
    history: list[float] = field(default_factory=list)  # max |change| per iteration
    residuals: np.ndarray | None = None  # residual spill slope (control i -> detector j) after compensation
    params: list[str] = field(default_factory=list)

    @property
    def max_residual(self) -> float:
        if self.residuals is None or not self.residuals.size:
            return 0.0
        off = self.residuals[~np.eye(len(self.residuals), dtype=bool)]
        off = off[np.isfinite(off)]
        return float(np.abs(off).max()) if off.size else 0.0

    @property
    def rms_residual(self) -> float:
        if self.residuals is None or not self.residuals.size:
            return 0.0
        off = self.residuals[~np.eye(len(self.residuals), dtype=bool)]
        off = off[np.isfinite(off)]
        return float(np.sqrt(np.mean(off ** 2))) if off.size else 0.0

    def summary(self) -> str:
        state = f"converged after {self.iterations} iteration(s)" if self.converged else f"stopped after {self.iterations} iteration(s) without converging"
        last = self.history[-1] if self.history else 0.0
        return f"AutoSpill {state} (last change {last * 100:.4f} %); unmixing residual max {self.max_residual * 100:.3f} %, RMS {self.rms_residual * 100:.3f} %"

    def worst(self, k: int = 3) -> list[tuple[str, str, float]]:
        if self.residuals is None:
            return []
        n = len(self.params)
        cells = [(self.params[i], self.params[j], float(self.residuals[i, j])) for i in range(n) for j in range(n) if i != j and np.isfinite(self.residuals[i, j])]
        cells.sort(key=lambda c: -abs(c[2]))
        return cells[:k]


def _quantile_bins(x: np.ndarray, bins: int = 8) -> list[np.ndarray] | None:
    """Index arrays of the events in `bins` quantile bins of x (None when x
    has too few distinct values)."""
    edges = np.unique(np.percentile(x, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return None
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    order = np.argsort(idx, kind="stable")
    counts = np.bincount(idx, minlength=len(edges) - 1)
    groups = np.split(order, np.cumsum(counts)[:-1])
    return [g for g in groups if g.size >= 5]


def _noise_model(x: np.ndarray, r: np.ndarray, groups: list[np.ndarray] | None) -> np.ndarray:
    """Per-event residual scale sigma(x) = a + b |x| fitted to the robust
    residual spread in quantile bins of x (positive populations spread more
    the brighter they are)."""
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    flat = np.full(x.shape, max(mad, 1e-9))
    if not groups or len(groups) < 2:
        return flat
    cx = np.array([float(np.median(np.abs(x[g]))) for g in groups])
    cs = np.array([1.4826 * float(np.median(np.abs(r[g] - np.median(r[g])))) for g in groups])
    if cs.max() <= 0:
        return flat
    a, b = np.linalg.lstsq(np.column_stack([np.ones_like(cx), cx]), cs, rcond=None)[0]
    floor = max(float(cs[cs > 0].min()) if (cs > 0).any() else mad, 1e-9)
    return np.maximum(a + max(b, 0.0) * np.abs(x), floor)


def robust_slope(x: np.ndarray, y: np.ndarray, k: float = 1.345, iterations: int = 50, heteroscedastic: bool = True) -> float:
    """Slope of a robust straight-line fit y = a + b x: Huber IRLS whose
    residuals are scaled by a fitted noise model sigma(x) = a + b|x| (weights
    psi(r / sigma) / sigma^2), or by one global MAD scale when
    `heteroscedastic` is False."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return 0.0
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    groups = _quantile_bins(x) if heteroscedastic else None
    for _ in range(iterations):
        r = y - X @ beta
        if heteroscedastic:
            sig = _noise_model(x, r, groups)
        else:
            s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            if not s > 0:
                break
            sig = np.full(x.shape, s)
        u = np.abs(r) / (k * sig)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12)) / sig ** 2
        Xw = X * w[:, None]
        try:
            new = np.linalg.solve(X.T @ Xw, Xw.T @ y)
        except np.linalg.LinAlgError:
            break
        done = abs(new[1] - beta[1]) <= 1e-7 * max(abs(new[1]), 1e-3)
        beta = new
        if done:
            break
    return float(beta[1])


def _control_subsets(controls, params, gates, masks, top, max_events, seed=0) -> dict[int, np.ndarray]:
    """Positive-gated events of every usable control (subsampled)."""
    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    for i, p in enumerate(params):
        events = controls.get(p)
        if events is None or np.shape(events)[0] < MIN_EVENTS:
            continue
        events = np.asarray(events, dtype=np.float64)
        pos, _neg = control_masks(events[:, i], (gates or {}).get(p), (masks or {}).get(p), top)
        if int(pos.sum()) < MIN_POPULATION:
            continue
        data = events[pos]
        if max_events and data.shape[0] > max_events:
            data = data[np.sort(rng.choice(data.shape[0], max_events, replace=False))]
        out[i] = data
    return out


def _residual_slopes(subsets: dict[int, np.ndarray], matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    try:
        inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(matrix)
    res = np.full((n, n), np.nan)
    for i, data in subsets.items():
        c = data @ inv
        res[i] = 0.0
        for j in range(n):
            if j != i:
                res[i, j] = robust_slope(c[:, i], c[:, j])
    return res


def refine_spillover(
    controls: dict[str, np.ndarray],
    params: list[str],
    initial: SpilloverMatrix | None = None,
    *,
    unstained: np.ndarray | None = None,
    gates: dict[str, ControlGates] | None = None,
    masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    max_iter: int = 25,
    tol: float = 1e-4,
    name: str | None = None,
    top: float = 262144.0,
    max_events: int = 10000,
) -> SpilloverMatrix:
    """AutoSpill-style refinement (after Roca et al. 2021).

    Starting from `initial` (default: robust regression of every detector on
    the primary detector over the positive-gated events of each control,
    whose intercept absorbs the negative / autofluorescence level), the controls
    are compensated with the current matrix, every compensated detector is
    regressed (Huber IRLS) on the compensated primary detector, and the
    residual slopes are folded back into the matrix, until the largest change
    is below `tol` or `max_iter` is reached. Negative coefficients are allowed.
    The result carries `report` (RefinementReport) and `diagnostics`."""
    n = len(params)
    base = estimate_spillover(controls, params, unstained, name or "AutoSpill", gates=gates, masks=masks, top=top, clip_negative=False)
    subsets = _control_subsets(controls, params, gates, masks, top, max_events)
    if initial is not None:
        idx = [initial.params.index(p) if p in initial.params else None for p in params]
        S = np.eye(n)
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                if ia is not None and ib is not None:
                    S[a, b] = initial.matrix[ia, ib]
    else:
        S = np.eye(n)
        for i, data in subsets.items():
            for j in range(n):
                if j != i:
                    S[i, j] = robust_slope(data[:, i], data[:, j])
    report = RefinementReport(tolerance=tol, params=list(params))
    for _ in range(max(int(max_iter), 0)):
        R = _residual_slopes(subsets, S)
        R = np.nan_to_num(R, nan=0.0)
        new = (np.eye(n) + R) @ S
        d = np.diag(new).copy()
        d[d == 0] = 1.0
        new = new / d[:, None]
        # rows without a usable control stay as they were
        for i in range(n):
            if i not in subsets:
                new[i] = S[i]
        change = float(np.abs(new - S).max()) if n else 0.0
        S = new
        report.iterations += 1
        report.history.append(change)
        if change < tol:
            report.converged = True
            break
    report.residuals = _residual_slopes(subsets, S)
    np.fill_diagonal(S, 1.0)
    out = SpilloverMatrix(name or "AutoSpill", params, S)
    out.diagnostics = base.diagnostics
    out.report = report
    return out


# ------------------------------------------------- spillover spreading matrix
def spillover_spreading(
    controls: dict[str, np.ndarray],
    params: list[str],
    matrix: SpilloverMatrix,
    unstained: np.ndarray | None = None,
    *,
    gates: dict[str, ControlGates] | None = None,
    masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    top: float = 262144.0,
) -> np.ndarray:
    """Spillover Spreading Matrix (after Nguyen et al. 2013).

    SSM[i, j] = (rSD_pos,j - rSD_neg,j) / sqrt(median_pos,i - median_neg,i),
    measured on the compensated single-stain control of fluorochrome i
    (row) in detector j (column), robust SDs from the 15.87-84.13 percentile
    range; units sqrt(intensity). The negatives are the control's own, or the
    unstained control when it has none. Rows without a usable control are NaN;
    the diagonal is 0; negative spreads are reported as 0."""
    n = len(params)
    if list(matrix.params) == list(params):
        S = matrix.matrix
    else:
        S = np.eye(n)
        idx = [matrix.params.index(p) if p in matrix.params else None for p in params]
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                if ia is not None and ib is not None:
                    S[a, b] = matrix.matrix[ia, ib]
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(S)
    unst_c = np.asarray(unstained, dtype=np.float64) @ inv if unstained is not None and np.size(unstained) else None
    ssm = np.full((n, n), np.nan)
    for i, p in enumerate(params):
        events = controls.get(p)
        if events is None or np.shape(events)[0] < MIN_EVENTS:
            continue
        events = np.asarray(events, dtype=np.float64)
        pos, neg = control_masks(events[:, i], (gates or {}).get(p), (masks or {}).get(p), top)
        if neg is None:
            neg = ~pos
        if int(pos.sum()) < MIN_POPULATION:
            continue
        c = events @ inv
        cpos = c[pos]
        cneg = c[neg] if int(neg.sum()) >= MIN_POPULATION else unst_c
        if cneg is None or len(cneg) < MIN_POPULATION:
            continue
        df = float(np.median(cpos[:, i]) - np.median(cneg[:, i]))
        if df <= 0:
            continue
        for j in range(n):
            if j == i:
                ssm[i, j] = 0.0
                continue
            spread = robust_sd(cpos[:, j]) - robust_sd(cneg[:, j])
            ssm[i, j] = max(spread, 0.0) / math.sqrt(df)
    return ssm


def residual_spill(controls: dict[str, np.ndarray], params: list[str], matrix: SpilloverMatrix, unstained: np.ndarray | None = None, *, gates=None, masks=None, top: float = 262144.0) -> np.ndarray:
    """Residual spill left after compensating each control with `matrix`:
    (median_pos,j - median_neg,j) / (median_pos,i - median_neg,i) on the
    compensated data (0 = perfectly compensated, > 0 under-, < 0 over-
    compensated). Rows without a usable control are NaN."""
    n = len(params)
    S = np.eye(n)
    idx = [matrix.params.index(p) if p in matrix.params else None for p in params]
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            if ia is not None and ib is not None:
                S[a, b] = matrix.matrix[ia, ib]
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(S)
    res = np.full((n, n), np.nan)
    unst_med = np.median(np.asarray(unstained, dtype=np.float64) @ inv, axis=0) if unstained is not None and np.size(unstained) else None
    for i, p in enumerate(params):
        events = controls.get(p)
        if events is None or np.shape(events)[0] < MIN_EVENTS:
            continue
        events = np.asarray(events, dtype=np.float64)
        pos, neg = control_masks(events[:, i], (gates or {}).get(p), (masks or {}).get(p), top)
        if int(pos.sum()) < MIN_POPULATION:
            continue
        c = events @ inv
        if neg is not None and int(neg.sum()) >= MIN_POPULATION:
            neg_med = np.median(c[neg], axis=0)
        elif unst_med is not None:
            neg_med = unst_med
        elif int((~pos).sum()) >= MIN_POPULATION:
            neg_med = np.median(c[~pos], axis=0)
        else:
            continue
        pos_med = np.median(c[pos], axis=0)
        denom = pos_med[i] - neg_med[i]
        if denom <= 0:
            continue
        res[i] = (pos_med - neg_med) / denom
        res[i, i] = 0.0
    return res


# ---------------------------------------------------------- group matrices
# A matrix attached to a group is applied to every sample that joins the group
# later (FlowJo group-level compensation). The association lives in
# workspace.settings["group_matrices"][group name] =
#   {"matrix": matrix name, "group_id": group id, "applied": [sample ids]}
# ("applied": members that already received it, so a sample whose matrix was
# changed by hand afterwards keeps that choice). A plain matrix name is
# accepted too (every member lacking the matrix receives it).

GROUP_MATRICES = "group_matrices"


def _group_records(ws) -> dict:
    recs = ws.settings.get(GROUP_MATRICES)
    return recs if isinstance(recs, dict) else {}


def _record_group(ws, key: str, rec):
    group = ws.group_by_name(key)
    if group is None and isinstance(rec, dict) and rec.get("group_id") is not None:
        group = next((g for g in ws.groups if g.id == rec.get("group_id")), None)
    return group


def group_matrix_name(ws, group) -> str | None:
    """Name of the matrix attached to `group`, or None."""
    for key, rec in _group_records(ws).items():
        if _record_group(ws, key, rec) is group:
            return rec.get("matrix") if isinstance(rec, dict) else (rec or None)
    return None


def groups_with_matrix(ws, matrix_name: str) -> list:
    out = []
    for key, rec in _group_records(ws).items():
        name = rec.get("matrix") if isinstance(rec, dict) else rec
        g = _record_group(ws, key, rec)
        if g is not None and name == matrix_name:
            out.append(g)
    return out


def set_group_matrix(ws, group, matrix_name: str | None, applied=()) -> None:
    """Attach `matrix_name` to `group` (None detaches). `applied`: ids of the
    members that already use it."""
    recs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _group_records(ws).items() if _record_group(ws, k, v) is not group}
    if matrix_name:
        recs[group.name] = {"matrix": matrix_name, "group_id": group.id, "applied": sorted(int(i) for i in applied)}
    if recs:
        ws.settings[GROUP_MATRICES] = recs
    else:
        ws.settings.pop(GROUP_MATRICES, None)


def pending_group_compensation(ws) -> list[tuple]:
    """(group, sample, matrix) for every member of a group with an attached
    matrix that has not received it yet (matrices that no longer exist are
    skipped)."""
    out = []
    for key, rec in _group_records(ws).items():
        group = _record_group(ws, key, rec)
        if group is None:
            continue
        name = rec.get("matrix") if isinstance(rec, dict) else rec
        matrix = ws.matrices.get(name) if name else None
        if matrix is None:
            continue
        applied = set(rec.get("applied", [])) if isinstance(rec, dict) else None
        for s in ws.members(group):
            if applied is not None:
                if s.id not in applied:
                    out.append((group, s, matrix))
            elif not (s.comp is not None and s.comp.name == matrix.name and s.comp == matrix):
                out.append((group, s, matrix))
    return out


def mark_group_compensation(ws, group, sample_ids) -> None:
    """Record that members `sample_ids` of `group` received its matrix (and
    forget members that left the group)."""
    name = group_matrix_name(ws, group)
    if not name:
        return
    members = {s.id for s in ws.members(group)}
    old = None
    for key, rec in _group_records(ws).items():
        if _record_group(ws, key, rec) is group and isinstance(rec, dict):
            old = set(rec.get("applied", []))
    applied = ((old or set()) | set(sample_ids)) & members
    set_group_matrix(ws, group, name, applied)


# ------------------------------------------------------------ matrix files
def _is_number(text: str) -> bool:
    try:
        float(text.strip().rstrip("%"))
        return True
    except ValueError:
        return False


def matrix_from_text(text: str, name: str = "Matrix") -> SpilloverMatrix:
    """Parse a matrix table (CSV, semicolon or tab separated, as written by
    spreadsheets, FlowJo text matrices and write_matrix_csv): a header row of
    detector names (optionally preceded by an empty corner cell) followed by
    rows of values (fractions or percent), optionally labelled."""
    import csv

    text = text.lstrip("﻿")
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("The text does not contain a matrix")
    first = lines[0]
    if "\t" in first:
        delim = "\t"
    else:
        try:
            delim = csv.Sniffer().sniff(first, delimiters=",;\t").delimiter
        except csv.Error:
            delim = ","
    rows = [r for r in csv.reader(lines, delimiter=delim) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("The text does not contain a matrix")
    header = [c.strip() for c in rows[0]]
    has_corner = header[0] == "" or len(header) == len(rows[1]) and not _is_number(rows[1][0])
    params = header[1:] if has_corner else header
    params = [p for p in params if p]
    n = len(params)
    values = []
    for r in rows[1 : n + 1]:
        cells = r[1:] if has_corner or not _is_number(r[0]) else r
        values.append([float(c.strip().rstrip("%").replace(",", ".") if delim != "," else c.strip().rstrip("%")) for c in cells[:n]])
    m = np.array(values, dtype=float) if values and all(len(v) == n for v in values) else np.zeros((len(values), 0))
    if m.shape != (n, n):
        raise ValueError(f"Expected a {n}x{n} matrix, got {m.shape}")
    if np.allclose(np.diag(m), 100.0):
        m = m / 100.0
    return SpilloverMatrix(name, params, m)


def matrix_to_text(matrix: SpilloverMatrix, delimiter: str = "\t", percent: bool = False, labels: bool = True) -> str:
    """The matrix as a table: header row of detectors (after an empty corner
    cell when `labels`), then one (labelled) row per fluorochrome."""
    scale = 100.0 if percent else 1.0
    lines = [delimiter.join(([""] if labels else []) + list(matrix.params))]
    for p, row in zip(matrix.params, matrix.matrix):
        cells = [f"{v * scale:.6g}" for v in row]
        lines.append(delimiter.join(([p] if labels else []) + cells))
    return "\n".join(lines) + "\n"


def read_matrix_mtx(path) -> SpilloverMatrix:
    """Read a FlowJo matrix file: the XML (Gating-ML) form, whose root is a
    gatingML element holding a spilloverMatrix or the spilloverMatrix itself,
    or the text form (tab separated, first row the parameters)."""
    import xml.etree.ElementTree as ET
    from pathlib import Path

    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    if text.lstrip().startswith("<"):
        from .wsp_import import parse_spillover_matrix

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise ValueError(f"Not a valid matrix file: {exc}") from exc
        el = next((e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "spilloverMatrix"), None)
        if el is None:
            raise ValueError("The file contains no spilloverMatrix element")
        m = parse_spillover_matrix(el)
        if not el.get("name"):
            m.name = path.stem
        if not m.params:
            raise ValueError("The matrix has no parameters")
        return m
    lines = [l for l in text.splitlines() if l.strip()]
    # some text matrices start with a title line (a single cell)
    if len(lines) > 2 and "\t" not in lines[0] and "," not in lines[0] and ";" not in lines[0]:
        title = lines[0].strip()
        m = matrix_from_text("\n".join(lines[1:]), title or path.stem)
        return m
    return matrix_from_text(text, path.stem)


def write_matrix_mtx(matrix: SpilloverMatrix, path, fmt: str = "xml") -> None:
    """Write a FlowJo matrix file: "xml" (a gatingML document holding one
    spilloverMatrix, as FlowJo exports) or "text" (tab separated, first row
    the parameters, then one row of coefficients per fluorochrome)."""
    from pathlib import Path

    path = Path(path)
    if fmt == "text":
        path.write_text(matrix_to_text(matrix, "\t", labels=False), encoding="utf-8")
        return
    import xml.etree.ElementTree as ET

    from .wsp_export import spillover_element
    from .wsp_import import GATING_NS

    root = ET.Element("{%s}gatingML" % GATING_NS)
    root.append(spillover_element(matrix))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    # ElementTree declares every namespace used (gating, transforms,
    # data-type) on the root element, as FlowJo writes them
    path.write_bytes(ET.tostring(root, encoding="UTF-8", xml_declaration=True))


def matrix_from_fcs_file(path) -> SpilloverMatrix:
    """The $SPILLOVER (SPILL, $COMP) matrix stored in another FCS file."""
    from pathlib import Path

    from .fcs import FCSFile, parse_spillover

    _, keywords = FCSFile.read_keywords(path)
    spill = parse_spillover(keywords)
    if spill is None:
        raise ValueError(f"{Path(path).name} has no $SPILLOVER matrix")
    return SpilloverMatrix(Path(path).stem, spill[0], spill[1])
