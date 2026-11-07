"""Spillover matrices and their estimation from single-stain controls."""
from __future__ import annotations

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


def estimate_spillover(controls: dict[str, np.ndarray], params: list[str], unstained: np.ndarray | None = None, name: str = "Computed") -> SpilloverMatrix:
    """Estimate a spillover matrix.

    controls: parameter name -> event matrix (N x len(params)) of its single-stain
    control, columns ordered as `params`. unstained: optional N x len(params)
    matrix used for the negative medians.
    """
    n = len(params)
    matrix = np.eye(n)
    neg_global = np.median(unstained, axis=0) if unstained is not None and unstained.size else None
    for i, p in enumerate(params):
        events = controls.get(p)
        if events is None or events.shape[0] < 20:
            continue
        pos_mask = split_positive(events[:, i])
        neg_mask = ~pos_mask
        pos_med = np.median(events[pos_mask], axis=0)
        if neg_global is not None:
            neg_med = neg_global
        elif neg_mask.sum() >= 10:
            neg_med = np.median(events[neg_mask], axis=0)
        else:
            neg_med = np.zeros(n)
        denom = pos_med[i] - neg_med[i]
        if denom <= 0:
            continue
        row = (pos_med - neg_med) / denom
        row = np.clip(row, 0.0, None)
        row[i] = 1.0
        matrix[i] = row
    return SpilloverMatrix(name, params, matrix)
