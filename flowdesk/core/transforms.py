"""Axis scales (transformations) mapping data values to a 0..1 display fraction.

Implemented from the public definitions: linear, logarithmic, the logicle
scale of Parks, Roederer & Moore (2006, 2012) which is what cytometrists know
as the "biexponential" display, and the inverse hyperbolic sine (fasinh) from
Gating-ML 2.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

LN10 = math.log(10.0)
_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def power_label(exponent: int, negative: bool = False) -> str:
    return ("-" if negative else "") + "10" + str(exponent).translate(_SUPERSCRIPT)


def format_linear(value: float, step: float, big: bool) -> str:
    if abs(value) < 1e-12:
        return "0"
    if big:
        if abs(value) >= 1e6:
            v = value / 1e6
            return f"{v:g}M"
        v = value / 1e3
        return f"{v:g}K"
    if step >= 1:
        return f"{value:.0f}"
    decimals = max(0, -int(math.floor(math.log10(step))))
    return f"{value:.{decimals}f}"


@dataclass
class Tick:
    pos: float
    label: str
    major: bool


class Scale:
    """Base class. Subclasses map data -> display fraction and back."""

    kind = "base"

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward_scalar(self, x: float) -> float:
        return float(self.forward(np.array([x], dtype=float))[0])

    def inverse_scalar(self, y: float) -> float:
        return float(self.inverse(np.array([y], dtype=float))[0])

    @property
    def data_min(self) -> float:
        return self.inverse_scalar(0.0)

    @property
    def data_max(self) -> float:
        return self.inverse_scalar(1.0)

    def ticks(self) -> list[Tick]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    def describe(self) -> str:
        return self.kind

    def copy(self) -> "Scale":
        return scale_from_dict(self.to_dict())

    def __eq__(self, other) -> bool:
        return isinstance(other, Scale) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(repr(sorted(self.to_dict().items())))


# ----------------------------------------------------------------------------- linear
class LinearScale(Scale):
    kind = "linear"

    def __init__(self, minimum: float = 0.0, maximum: float = 262144.0):
        if maximum <= minimum:
            maximum = minimum + 1.0
        self.min = float(minimum)
        self.max = float(maximum)

    def forward(self, x):
        return (np.asarray(x, dtype=float) - self.min) / (self.max - self.min)

    def inverse(self, y):
        return np.asarray(y, dtype=float) * (self.max - self.min) + self.min

    def ticks(self) -> list[Tick]:
        span = self.max - self.min
        raw = span / 6.0
        mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
        step = mag
        for mult in (1, 2, 2.5, 5, 10):
            if mag * mult >= raw:
                step = mag * mult
                break
        big = max(abs(self.min), abs(self.max)) >= 10000
        ticks = []
        first = math.ceil(self.min / step)
        last = math.floor(self.max / step)
        minor = step / 5.0
        for k in range(first, last + 1):
            v = k * step
            ticks.append(Tick(self.forward_scalar(v), format_linear(v, step, big), True))
        start = math.ceil(self.min / minor)
        stop = math.floor(self.max / minor)
        for k in range(start, stop + 1):
            v = k * minor
            if abs(v / step - round(v / step)) > 1e-6:
                ticks.append(Tick(self.forward_scalar(v), "", False))
        return ticks

    def to_dict(self):
        return {"kind": self.kind, "min": self.min, "max": self.max}

    def describe(self):
        return f"Linear ({format_linear(self.min, 1, abs(self.min) >= 10000)} – {format_linear(self.max, 1, self.max >= 10000)})"


# -------------------------------------------------------------------------------- log
class LogScale(Scale):
    kind = "log"

    def __init__(self, minimum: float = 1.0, maximum: float = 262144.0):
        self.min = float(minimum) if minimum > 0 else 1.0
        self.max = float(maximum) if maximum > self.min else self.min * 10.0

    @property
    def decades(self) -> float:
        return math.log10(self.max) - math.log10(self.min)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = (np.log10(np.maximum(x, self.min)) - math.log10(self.min)) / self.decades
        return np.where(np.isfinite(y), y, 0.0)

    def inverse(self, y):
        return np.power(10.0, np.asarray(y, dtype=float) * self.decades + math.log10(self.min))

    def ticks(self) -> list[Tick]:
        return _loglike_ticks(self, zero=False)

    def to_dict(self):
        return {"kind": self.kind, "min": self.min, "max": self.max}

    def describe(self):
        return f"Log ({self.decades:.1f} decades)"


# ------------------------------------------------------------------------------ logicle
class LogicleScale(Scale):
    """Logicle ("biexponential") scale.

    T: top of scale, M: positive decades, W: linear-region width in decades,
    A: additional negative decades. The width basis used by many cytometry
    programs is the most negative data value shown; W is derived from it as
    W = (M - log10(T / |wb|)) / 2 (Parks et al. 2006).
    """

    kind = "logicle"

    def __init__(self, T: float = 262144.0, W: float = 0.5, M: float = 4.5, A: float = 0.0, width_basis: float | None = None):
        self.T = float(T) if T > 0 else 262144.0
        self.M = float(M) if M > 0 else 4.5
        self.W = float(min(max(W, 0.0), self.M / 2.0))
        self.A = float(max(A, -self.W))
        if self.A > self.M - 2 * self.W:
            self.A = max(0.0, self.M - 2 * self.W)
        # the width basis as given by the user / an imported workspace; kept so
        # that values outside the representable range survive a round trip
        self._width_basis = float(width_basis) if width_basis is not None else None
        self._init_params()

    @classmethod
    def from_biex(cls, T: float, pos: float, neg: float, width_basis: float) -> "LogicleScale":
        wb = abs(width_basis) if width_basis else 10.0
        W = (pos - math.log10(max(T, 1.0) / wb)) / 2.0
        return cls(T=T, W=max(0.0, W), M=pos, A=max(0.0, neg), width_basis=-wb)

    @property
    def width_basis(self) -> float:
        if self._width_basis is not None:
            return self._width_basis
        return -self.T * 10 ** (-(self.M - 2 * self.W))

    def _init_params(self):
        T, W, M, A = self.T, self.W, self.M, self.A
        self.w = W / (M + A)
        self.x2 = A / (M + A)
        self.x1 = self.x2 + self.w
        self.x0 = self.x2 + 2 * self.w
        self.b = (M + A) * LN10
        self.d = self._solve_d(self.b, self.w)
        c_a = math.exp(self.x0 * (self.b + self.d))
        mf_a = math.exp(self.b * self.x1) - c_a / math.exp(self.d * self.x1)
        self.a = T / ((math.exp(self.b) - mf_a) - c_a / math.exp(self.d))
        self.c = c_a * self.a
        self.f = mf_a * self.a
        ys = np.linspace(0.0, 1.0, 8193)
        xs = self._inverse_raw(ys)
        # guarantee strict monotonicity for interpolation
        xs = np.maximum.accumulate(xs)
        self._lut_y = ys
        self._lut_x = xs

    @staticmethod
    def _solve_d(b: float, w: float) -> float:
        if w <= 0:
            return b

        def func(d):
            return 2 * (math.log(d) - math.log(b)) + w * (b + d)

        lo, hi = 1e-12, b
        try:
            return brentq(func, lo, hi, xtol=1e-12, maxiter=200)
        except ValueError:
            return b

    def _inverse_raw(self, y):
        y = np.asarray(y, dtype=float)
        neg = y < self.x1
        s = np.where(neg, 2 * self.x1 - y, y)
        val = self.a * np.exp(self.b * s) - self.c * np.exp(-self.d * s) - self.f
        return np.where(neg, -val, val)

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, self._lut_x, self._lut_y)

    def inverse(self, y):
        return self._inverse_raw(np.clip(np.asarray(y, dtype=float), 0.0, 1.0))

    def ticks(self) -> list[Tick]:
        return _loglike_ticks(self, zero=True)

    def to_dict(self):
        d = {"kind": self.kind, "T": self.T, "W": self.W, "M": self.M, "A": self.A}
        if self._width_basis is not None:
            d["width_basis"] = self._width_basis
        return d

    def describe(self):
        return f"Biexponential (pos {self.M:.2f}, width {self.width_basis:.0f}, neg {self.A:.2f})"


# ------------------------------------------------------------------------------ arcsinh
class ArcsinhScale(Scale):
    kind = "arcsinh"

    def __init__(self, T: float = 262144.0, M: float = 4.5, A: float = 0.0):
        self.T = float(T) if T > 0 else 262144.0
        self.M = float(M) if M > 0 else 4.5
        self.A = float(max(A, 0.0))

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        y = (np.arcsinh(x * math.sinh(self.M * LN10) / self.T) + self.A * LN10) / ((self.M + self.A) * LN10)
        return np.clip(y, 0.0, 1.0)

    def inverse(self, y):
        y = np.asarray(y, dtype=float)
        return self.T * np.sinh(y * (self.M + self.A) * LN10 - self.A * LN10) / math.sinh(self.M * LN10)

    def ticks(self) -> list[Tick]:
        return _loglike_ticks(self, zero=True)

    def to_dict(self):
        return {"kind": self.kind, "T": self.T, "M": self.M, "A": self.A}

    def describe(self):
        return f"Arcsinh (pos {self.M:.2f}, neg {self.A:.2f})"


# ------------------------------------------------------------------------- tick helper
def _loglike_ticks(scale: Scale, zero: bool, min_label_gap: float = 0.055) -> list[Tick]:
    xmin, xmax = scale.data_min, scale.data_max
    ticks: list[Tick] = []
    labeled_positions: list[float] = []

    def far_enough(pos: float) -> bool:
        return all(abs(pos - p) >= min_label_gap for p in labeled_positions)

    if zero and xmin <= 0.0 <= xmax:
        pos = scale.forward_scalar(0.0)
        ticks.append(Tick(pos, "0", True))
        labeled_positions.append(pos)

    selected_pos: list[int] = []
    selected_neg: list[int] = []
    if xmax > 0:
        top = int(math.floor(math.log10(xmax)))
        for n in range(top, -3, -1):
            v = 10.0 ** n
            if v < max(xmin, 0.0) or v > xmax:
                continue
            pos = scale.forward_scalar(v)
            if far_enough(pos):
                ticks.append(Tick(pos, power_label(n), True))
                labeled_positions.append(pos)
                selected_pos.append(n)
            else:
                break
    if xmin < 0:
        top = int(math.floor(math.log10(-xmin)))
        for n in range(top, -3, -1):
            v = -(10.0 ** n)
            if v < xmin or v > min(xmax, 0.0):
                continue
            pos = scale.forward_scalar(v)
            if far_enough(pos):
                ticks.append(Tick(pos, power_label(n, negative=True), True))
                labeled_positions.append(pos)
                selected_neg.append(n)
            else:
                break

    def add_minors(decades: list[int], sign: float):
        if not decades:
            return
        lowest = min(decades)
        for n in range(lowest - 1, max(decades) + 1):
            for k in range(2, 10):
                v = sign * k * (10.0 ** n)
                if v < xmin or v > xmax:
                    continue
                ticks.append(Tick(scale.forward_scalar(v), "", False))

    add_minors(selected_pos, 1.0)
    add_minors(selected_neg, -1.0)
    ticks.sort(key=lambda t: t.pos)
    return ticks


# ---------------------------------------------------------------------------- factory
def scale_from_dict(d: dict) -> Scale:
    kind = d.get("kind", "linear")
    if kind == "linear":
        return LinearScale(d.get("min", 0.0), d.get("max", 262144.0))
    if kind == "log":
        return LogScale(d.get("min", 1.0), d.get("max", 262144.0))
    if kind in ("logicle", "biex"):
        if "pos" in d:
            return LogicleScale.from_biex(d.get("T", 262144.0), d.get("pos", 4.5), d.get("neg", 0.0), d.get("width_basis", -100.0))
        return LogicleScale(d.get("T", 262144.0), d.get("W", 0.5), d.get("M", 4.5), d.get("A", 0.0), d.get("width_basis"))
    if kind in ("arcsinh", "fasinh"):
        return ArcsinhScale(d.get("T", 262144.0), d.get("M", 4.5), d.get("A", 0.0))
    raise ValueError(f"unknown scale kind {kind!r}")


SCALE_KINDS = [("linear", "Linear"), ("logicle", "Biexponential"), ("log", "Logarithmic"), ("arcsinh", "Arcsinh")]


def default_biex(top: float) -> LogicleScale:
    top = float(top) if top and top > 0 else 262144.0
    pos = max(3.0, math.log10(top) - 1.0)
    return LogicleScale.from_biex(top, pos, 0.0, -top / 10 ** (pos - 1.0))


# Defaults for parameters nobody chose a scale for yet. The workspace window
# pushes the user preferences "default_transform" and "default_width_basis"
# here (set_scale_defaults), the way the controller pushes the polygon mode
# into gates.POLYGON_SETTINGS. A width basis of None (or the built-in -100)
# keeps the range-derived default of default_biex().
BUILTIN_WIDTH_BASIS = -100.0
SCALE_DEFAULTS: dict = {"transform": "logicle", "width_basis": None}
_KIND_ALIASES = {"biex": "logicle", "biexponential": "logicle", "logicle": "logicle", "log": "log", "logarithmic": "log",
                 "arcsinh": "arcsinh", "fasinh": "arcsinh", "asinh": "arcsinh", "linear": "linear", "lin": "linear"}


def normalize_kind(kind: str | None, fallback: str = "logicle") -> str:
    return _KIND_ALIASES.get(str(kind or "").strip().lower(), fallback)


def set_scale_defaults(transform: str | None = None, width_basis: float | None = None) -> bool:
    """Set the transform kind and biexponential width basis that fluorescence
    parameters get when their scale is defaulted. Returns True when that changed."""
    kind = normalize_kind(transform)
    wb: float | None
    try:
        wb = float(width_basis) if width_basis is not None else None
    except (TypeError, ValueError):
        wb = None
    if wb is not None and (wb == 0 or not math.isfinite(wb) or abs(-abs(wb) - BUILTIN_WIDTH_BASIS) < 1e-9):
        wb = None
    elif wb is not None:
        wb = -abs(wb)
    new = {"transform": kind, "width_basis": wb}
    if new == SCALE_DEFAULTS:
        return False
    SCALE_DEFAULTS.update(new)
    return True


def default_fluorescence_scale(top: float = 262144.0, transform: str | None = None, width_basis: float | None = None) -> Scale:
    """The default scale of a fluorescence parameter whose range is `top`,
    following SCALE_DEFAULTS unless `transform` / `width_basis` are given."""
    top = float(top) if top and top > 0 else 262144.0
    kind = normalize_kind(transform or SCALE_DEFAULTS.get("transform"))
    wb = width_basis if width_basis is not None else SCALE_DEFAULTS.get("width_basis")
    pos = max(3.0, math.log10(top) - 1.0)
    if kind == "linear":
        return LinearScale(0.0, top)
    if kind == "log":
        return LogScale(1.0 if top > 10.0 else top / 1e4, top)
    if kind == "arcsinh":
        return ArcsinhScale(top, pos, 0.0)
    if wb is None or not wb:
        return default_biex(top)
    return LogicleScale.from_biex(top, pos, 0.0, -abs(float(wb)))


def estimate_biex(values, top: float = 262144.0, pos: float | None = None, percentile: float = 5.0, neg: float = 0.0) -> LogicleScale:
    """A biexponential scale whose linear region fits the negative spread of
    `values` (typically compensated data): the width basis is the low
    `percentile` of the events (r, the most negative value worth showing, in
    Parks, Roederer & Moore 2006), kept between the narrowest width the scale
    can represent (and -10) and a linear region of 1.5 decades, rounded to two
    significant digits."""
    top = float(top) if top and top > 0 else 262144.0
    pos = float(pos) if pos and pos > 0 else max(3.0, math.log10(top) - 1.0)
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    floor = max(10.0, top / 10 ** pos)
    ceiling = max(floor, top / 10 ** max(pos - 3.0, 0.0))
    r = float(np.percentile(v, percentile)) if v.size else 0.0
    mag = min(max(-r, floor), ceiling)
    mag = float(f"{mag:.2g}")
    return LogicleScale.from_biex(top, pos, max(0.0, float(neg)), -mag)


_LINEAR_NAME_PATTERNS = ("FSC", "SSC", "TIME", "HDR-T", "EVENT", "INDEXSORT", "DELTATIME", "REGIONCLASSIFICATION", "SORT")


def default_scale_for_channel(
    name: str,
    display_hint: str = "",
    top: float = 262144.0,
    log_amplified: bool = False,
    *,
    log_decades: float = 0.0,
    log_offset: float = 0.0,
    transform: str | None = None,
    width_basis: float | None = None,
) -> Scale:
    """Default scale of a measured parameter. Log-amplified channels ($PnE)
    get a log scale over their decoded range offset .. offset*10^decades when
    the decades are given; scatter, time, width and similar parameters are
    linear; fluorescence follows SCALE_DEFAULTS (biexponential by default)."""
    upper = name.upper()
    hint = display_hint.upper()
    if log_amplified:
        if log_decades and log_decades > 0:
            lo = float(log_offset) if log_offset and log_offset > 0 else 1.0
            return LogScale(lo, lo * 10 ** float(log_decades))
        return LogScale(max(1.0, top / 10 ** 4), top)
    if hint.startswith("LIN"):
        return LinearScale(0.0, top)
    if hint.startswith("LOG"):
        return default_fluorescence_scale(top, transform, width_basis)
    if any(upper.startswith(p) or upper == p for p in _LINEAR_NAME_PATTERNS):
        return LinearScale(0.0, top)
    if upper.endswith("-W") or upper.endswith("WIDTH"):
        return LinearScale(0.0, top)
    return default_fluorescence_scale(top, transform, width_basis)


# ------------------------------------------------------------ dragging an axis
def _bisect(fn, lo: float, hi: float, iters: int = 60) -> float:
    """Root of a monotone fn on [lo, hi]; the nearer end when there is none."""
    flo, fhi = fn(lo), fn(hi)
    if flo == 0:
        return lo
    if fhi == 0 or (flo > 0) == (fhi > 0):
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def drag_edge(scale: Scale, frac: float) -> str:
    """Which part of an axis a drag that starts at display fraction `frac` changes: the low end ("min"), the high end
    ("max") or, in the middle of a linear or log axis, both together ("pan")."""
    if frac < 1.0 / 3.0:
        return "min"
    if frac > 2.0 / 3.0 or scale.kind not in ("linear", "log"):
        return "max"
    return "pan"


def rescale(scale: Scale, edge: str, value: float, frac: float) -> Scale:
    """A copy of `scale` changed so that the data value `value` (the one grabbed on the axis) lands at display fraction
    `frac`. edge "max" keeps the low end and moves the top, "min" keeps the top and moves the low end, "pan" shifts the
    range (linear and log). Logicle and arcsinh scales change their top (T) or their extra negative decades (A), so the
    shape of the transform stays what the user chose."""
    f = min(max(float(frac), 0.02), 0.98)
    v = float(value)
    if isinstance(scale, LinearScale):
        lo, hi = scale.min, scale.max
        if edge == "pan":
            span = hi - lo
            lo = v - f * span
            return LinearScale(lo, lo + span)
        if edge == "max":
            if v <= lo:
                return scale.copy()
            return LinearScale(lo, lo + (v - lo) / f)
        if v >= hi:
            return scale.copy()
        return LinearScale((v - f * hi) / (1.0 - f), hi)
    if isinstance(scale, LogScale):
        if v <= 0:
            return scale.copy()
        lv, llo, lhi = math.log10(v), math.log10(scale.min), math.log10(scale.max)
        if edge == "pan":
            span = lhi - llo
            llo = lv - f * span
            return LogScale(10 ** llo, 10 ** (llo + span))
        if edge == "max":
            if lv <= llo:
                return scale.copy()
            return LogScale(scale.min, 10 ** (llo + (lv - llo) / f))
        if lv >= lhi:
            return scale.copy()
        return LogScale(10 ** ((lv - f * lhi) / (1.0 - f)), scale.max)
    if isinstance(scale, LogicleScale):
        T, W, M, A = scale.T, scale.W, scale.M, scale.A
        if edge == "min":
            amax = max(0.0, M - 2 * W)
            a = _bisect(lambda a: LogicleScale(T, W, M, a).forward_scalar(v) - f, 0.0, amax, iters=40)
            return LogicleScale(T, W, M, a)
        if v <= 0:
            return scale.copy()
        lt = _bisect(lambda lt: LogicleScale(10 ** lt, W, M, A).forward_scalar(v) - f, math.log10(max(v, 1.0)) - 0.01, math.log10(T) + 4, iters=50)
        return LogicleScale(10 ** lt, W, M, A)
    if isinstance(scale, ArcsinhScale):
        T, M, A = scale.T, scale.M, scale.A
        if edge == "min":
            a = _bisect(lambda a: ArcsinhScale(T, M, a).forward_scalar(v) - f, 0.0, M, iters=40)
            return ArcsinhScale(T, M, a)
        if v <= 0:
            return scale.copy()
        lt = _bisect(lambda lt: ArcsinhScale(10 ** lt, M, A).forward_scalar(v) - f, math.log10(max(v, 1.0)) - 4, math.log10(T) + 4, iters=50)
        return ArcsinhScale(10 ** lt, M, A)
    return scale.copy()
