"""Gate geometry and membership evaluation.

Gate coordinates are stored in data units. Rectangles are evaluated on data
values; polygons and ellipses are evaluated in display space (the space the
user drew them in), so their edges stay straight on screen whatever the axis
scale is.
"""
from __future__ import annotations

import uuid
from typing import Protocol

import numpy as np


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class ParamAccess(Protocol):
    """What a gate needs from a sample to evaluate itself."""

    def data(self, name: str) -> np.ndarray: ...

    def display(self, name: str) -> np.ndarray: ...

    def to_display(self, name: str, values) -> np.ndarray: ...

    def resolve(self, path: str) -> np.ndarray | None: ...

    @property
    def n_events(self) -> int: ...


def points_in_polygon(px: np.ndarray, py: np.ndarray, vertices: list[tuple[float, float]]) -> np.ndarray:
    """Even-odd rule point-in-polygon test, vectorised over points."""
    inside = np.zeros(px.shape[0], dtype=bool)
    n = len(vertices)
    if n < 3:
        return inside
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        inside ^= crosses & (px < x_int)
    return inside


class Gate:
    kind = "gate"

    def __init__(self):
        self.id = new_id()
        self.dims: list[str] = []
        self.color: str | None = None
        self.line_width: float = 1.0
        self.negated: bool = False  # events *outside* the shape belong to the gate
        self.label_pos: tuple[float, float] | None = None  # manual label position (display fractions)

    def evaluate(self, acc: ParamAccess) -> np.ndarray:
        raise NotImplementedError

    def membership(self, acc: ParamAccess) -> np.ndarray:
        """evaluate() with the negation flag applied."""
        inside = self.evaluate(acc)
        return ~inside if self.negated else inside

    def to_dict(self) -> dict:
        raise NotImplementedError

    def _base_dict(self) -> dict:
        d = {"kind": self.kind, "id": self.id, "dims": list(self.dims)}
        if self.color:
            d["color"] = self.color
        if self.line_width != 1.0:
            d["line_width"] = self.line_width
        if self.negated:
            d["negated"] = True
        if self.label_pos is not None:
            d["label_pos"] = [float(self.label_pos[0]), float(self.label_pos[1])]
        return d

    def _load_base(self, d: dict):
        self.id = d.get("id", self.id)
        self.color = d.get("color")
        self.line_width = float(d.get("line_width", 1.0))
        self.negated = bool(d.get("negated", False))
        lp = d.get("label_pos")
        self.label_pos = (float(lp[0]), float(lp[1])) if lp else None

    def clone(self) -> "Gate":
        g = gate_from_dict(self.to_dict())
        g.id = new_id()
        return g

    def uses_param(self, name: str) -> bool:
        return name in self.dims

    def rename_param(self, old: str, new: str):
        self.dims = [new if d == old else d for d in self.dims]

    @property
    def is_1d(self) -> bool:
        return len(self.dims) == 1


class RectangleGate(Gate):
    """Axis-aligned range (1D) or rectangle (2D) gate. None bounds are open."""

    kind = "rectangle"

    def __init__(self, dims: list[str], lo: list[float | None], hi: list[float | None]):
        super().__init__()
        self.dims = list(dims)
        self.lo = list(lo)
        self.hi = list(hi)
        self.quadrant: tuple[str, int] | None = None

    def evaluate(self, acc):
        mask = np.ones(acc.n_events, dtype=bool)
        for dim, lo, hi in zip(self.dims, self.lo, self.hi):
            v = acc.data(dim)
            if lo is not None:
                mask &= v >= lo
            if hi is not None:
                mask &= v < hi
        return mask

    def to_dict(self):
        d = self._base_dict()
        d["lo"] = self.lo
        d["hi"] = self.hi
        if self.quadrant:
            d["quadrant"] = list(self.quadrant)
        return d

    @classmethod
    def from_dict(cls, d):
        g = cls(d["dims"], d.get("lo", [None] * len(d["dims"])), d.get("hi", [None] * len(d["dims"])))
        g._load_base(d)
        if d.get("quadrant"):
            g.quadrant = (d["quadrant"][0], int(d["quadrant"][1]))
        return g

    @property
    def is_range(self) -> bool:
        return len(self.dims) == 1


# How polygon gates are evaluated. "exact" tests every event against the
# polygon in display space. "bitmap" mimics the way FlowJo evaluates polygons
# on a binned display grid (its "gate resolution"): a bin belongs to the gate
# when at least `threshold` of its area lies inside the polygon, and events are
# assigned by bin. Measured end-to-end on a real FlowJo workspace (40 gates,
# 4 samples), the 512-bin / 25 % setting brings the difference to FlowJo's
# stored counts from rms 0.51 % (bias -0.39 %) down to rms 0.30 % (bias -0.02 %).
POLYGON_SETTINGS = {"mode": "exact", "resolution": 512, "threshold": 0.25, "supersample": 4}


def polygon_bitmap(vertices: list[tuple[float, float]], resolution: int = 256, threshold: float = 0.25, supersample: int = 4) -> np.ndarray:
    """Boolean [y, x] grid of bins covered by the polygon (display fractions 0..1)."""
    res = int(resolution)
    sub = max(1, int(supersample))
    base = np.arange(res, dtype=float)
    coverage = np.zeros((res, res), dtype=float)
    offsets = (np.arange(sub) + 0.5) / sub
    for ox in offsets:
        for oy in offsets:
            gx, gy = np.meshgrid((base + ox) / res, (base + oy) / res)
            coverage += points_in_polygon(gx.ravel(), gy.ravel(), vertices).reshape(res, res)
    coverage /= sub * sub
    return coverage >= threshold if threshold > 0 else coverage > 0


class PolygonGate(Gate):
    kind = "polygon"

    def __init__(self, dims: list[str], vertices: list[tuple[float, float]]):
        super().__init__()
        self.dims = list(dims)
        self.vertices = [(float(x), float(y)) for x, y in vertices]
        self._bitmap_cache: tuple | None = None

    def display_vertices(self, acc) -> list[tuple[float, float]]:
        xs = acc.to_display(self.dims[0], np.array([v[0] for v in self.vertices]))
        ys = acc.to_display(self.dims[1], np.array([v[1] for v in self.vertices]))
        return list(zip(xs.tolist(), ys.tolist()))

    def evaluate(self, acc):
        px = acc.display(self.dims[0])
        py = acc.display(self.dims[1])
        verts = self.display_vertices(acc)
        if POLYGON_SETTINGS.get("mode") == "bitmap":
            res = int(POLYGON_SETTINGS.get("resolution", 256))
            thr = float(POLYGON_SETTINGS.get("threshold", 0.25))
            sub = int(POLYGON_SETTINGS.get("supersample", 4))
            key = (tuple(round(v, 7) for pair in verts for v in pair), res, thr, sub)
            cache = getattr(self, "_bitmap_cache", None)
            if cache is None or cache[0] != key:
                cache = (key, polygon_bitmap(verts, res, thr, sub))
                self._bitmap_cache = cache
            mask = cache[1]
            bx = np.clip(np.floor(px * res), 0, res - 1).astype(np.intp)
            by = np.clip(np.floor(py * res), 0, res - 1).astype(np.intp)
            return mask[by, bx]
        return points_in_polygon(px, py, verts)

    def to_dict(self):
        d = self._base_dict()
        d["vertices"] = [list(v) for v in self.vertices]
        return d

    @classmethod
    def from_dict(cls, d):
        g = cls(d["dims"], [tuple(v) for v in d.get("vertices", [])])
        g._load_base(d)
        return g


class EllipseGate(Gate):
    """Ellipse defined by its centre and the end points of its two semi-axes."""

    kind = "ellipse"

    def __init__(self, dims: list[str], center: tuple[float, float], a_end: tuple[float, float], b_end: tuple[float, float]):
        super().__init__()
        self.dims = list(dims)
        self.center = (float(center[0]), float(center[1]))
        self.a_end = (float(a_end[0]), float(a_end[1]))
        self.b_end = (float(b_end[0]), float(b_end[1]))

    def display_points(self, acc) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = acc.to_display(self.dims[0], np.array([self.center[0], self.a_end[0], self.b_end[0]]))
        ys = acc.to_display(self.dims[1], np.array([self.center[1], self.a_end[1], self.b_end[1]]))
        pts = np.column_stack([xs, ys])
        return pts[0], pts[1], pts[2]

    @staticmethod
    def points_inside(px, py, c, pa, pb) -> np.ndarray:
        u = pa - c
        v = pb - c
        basis = np.array([[u[0], v[0]], [u[1], v[1]]], dtype=float)
        det = np.linalg.det(basis)
        if abs(det) < 1e-18:
            return np.zeros(px.shape[0], dtype=bool)
        inv = np.linalg.inv(basis)
        dx = px - c[0]
        dy = py - c[1]
        alpha = inv[0, 0] * dx + inv[0, 1] * dy
        beta = inv[1, 0] * dx + inv[1, 1] * dy
        return alpha * alpha + beta * beta <= 1.0

    def evaluate(self, acc):
        c, pa, pb = self.display_points(acc)
        return self.points_inside(acc.display(self.dims[0]), acc.display(self.dims[1]), c, pa, pb)

    def to_dict(self):
        d = self._base_dict()
        d.update({"center": list(self.center), "a_end": list(self.a_end), "b_end": list(self.b_end)})
        return d

    @classmethod
    def from_dict(cls, d):
        g = cls(d["dims"], tuple(d["center"]), tuple(d["a_end"]), tuple(d["b_end"]))
        g._load_base(d)
        return g


class BooleanGate(Gate):
    """AND / OR / NOT combination of other populations (absolute paths)."""

    kind = "boolean"

    def __init__(self, op: str, refs: list[str]):
        super().__init__()
        self.op = op.lower()
        self.refs = list(refs)
        self.dims = []

    def evaluate(self, acc):
        masks = []
        for ref in self.refs:
            m = acc.resolve(ref)
            masks.append(np.zeros(acc.n_events, dtype=bool) if m is None else m)
        if not masks:
            return np.zeros(acc.n_events, dtype=bool)
        if self.op == "not":
            return ~masks[0]
        if self.op == "or":
            out = masks[0].copy()
            for m in masks[1:]:
                out |= m
            return out
        out = masks[0].copy()
        for m in masks[1:]:
            out &= m
        return out

    def to_dict(self):
        d = self._base_dict()
        d.update({"op": self.op, "refs": list(self.refs)})
        return d

    @classmethod
    def from_dict(cls, d):
        g = cls(d.get("op", "and"), d.get("refs", []))
        g._load_base(d)
        return g

    def describe(self) -> str:
        names = [r.rsplit("/", 1)[-1] for r in self.refs]
        if self.op == "not":
            return f"NOT {names[0]}" if names else "NOT"
        return f" {self.op.upper()} ".join(names)


def gate_from_dict(d: dict) -> Gate:
    kind = d.get("kind")
    if kind == "rectangle":
        return RectangleGate.from_dict(d)
    if kind == "polygon":
        return PolygonGate.from_dict(d)
    if kind == "ellipse":
        return EllipseGate.from_dict(d)
    if kind == "boolean":
        return BooleanGate.from_dict(d)
    raise ValueError(f"unknown gate kind {kind!r}")


def make_quadrant_gates(xdim: str, ydim: str, x_split: float, y_split: float, xlabel: str | None = None, ylabel: str | None = None) -> list[tuple[str, RectangleGate]]:
    """Four rectangle gates sharing a centre, named the way cytometrists expect:
    Q1 upper-left, Q2 upper-right, Q3 lower-right, Q4 lower-left."""
    xl = xlabel or xdim
    yl = ylabel or ydim
    group = new_id()
    spec = [
        (f"Q1: {xl}- , {yl}+", None, x_split, y_split, None),
        (f"Q2: {xl}+ , {yl}+", x_split, None, y_split, None),
        (f"Q3: {xl}+ , {yl}-", x_split, None, None, y_split),
        (f"Q4: {xl}- , {yl}-", None, x_split, None, y_split),
    ]
    gates = []
    for index, (name, xlo, xhi, ylo, yhi) in enumerate(spec, start=1):
        g = RectangleGate([xdim, ydim], [xlo, ylo], [xhi, yhi])
        g.quadrant = (group, index)
        gates.append((name, g))
    return gates
