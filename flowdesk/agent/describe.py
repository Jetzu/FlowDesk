"""Numerical descriptions of cytometry data an agent can read without looking at an image."""
from __future__ import annotations

import numpy as np

from .ops_base import clean

SHADES = " .:-=+*#%@"


def _raw(scale, display_values) -> list:
    return [clean(float(scale.inverse(np.array([v]))[0]), 4) for v in display_values]


def density_map(x_disp: np.ndarray, y_disp: np.ndarray, x_scale, y_scale, cols: int = 48, rows: int = 18) -> dict:
    """An ASCII density map of two display-space coordinate arrays (0..1), with raw-unit tick labels.

    Rows run from the top (high y) to the bottom; each cell shows its event count on a square-root scale
    (' ' empty ... '@' densest)."""
    n = int(len(x_disp))
    if n == 0:
        return {"events": 0, "map": [], "note": "no events"}
    h, _, _ = np.histogram2d(x_disp, y_disp, bins=[cols, rows], range=[[0, 1], [0, 1]])
    top = h.max() or 1.0
    lines = []
    for j in range(rows - 1, -1, -1):
        row = ""
        for i in range(cols):
            v = h[i, j]
            row += " " if v <= 0 else SHADES[1 + min(len(SHADES) - 2, int((v / top) ** 0.5 * (len(SHADES) - 2)))]
        lines.append(row)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    return {"events": n, "map": lines, "columns": cols, "rows": rows,
            "x_ticks_raw": dict(zip([str(t) for t in ticks], _raw(x_scale, ticks))),
            "y_ticks_raw": dict(zip([str(t) for t in ticks], _raw(y_scale, ticks))),
            "legend": "x runs left to right, y bottom to top (ticks give the raw value at display positions 0, .25, .5, .75, 1)"}


def quantiles(raw: np.ndarray, qs=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    if len(raw) == 0:
        return {}
    return {"p%g" % q: clean(float(np.percentile(raw, q)), 4) for q in qs}


def histogram_summary(raw: np.ndarray, disp: np.ndarray, scale, bins: int = 48) -> dict:
    """Counts per display-space bin (edges given in raw units), peaks, and the valley between the two main peaks."""
    n = int(len(disp))
    if n == 0:
        return {"events": 0}
    counts, edges = np.histogram(disp, bins=bins, range=(0.0, 1.0))
    raw_edges = _raw(scale, edges)
    out = {"events": n, "bins": bins, "edges_raw": raw_edges, "counts": counts.astype(int).tolist(), "percentiles_raw": quantiles(raw)}
    k = np.ones(3) / 3.0
    smooth = np.convolve(counts.astype(float), k, mode="same")
    peaks: list[int] = []
    try:
        from scipy.signal import find_peaks

        idx, _ = find_peaks(smooth, prominence=max(1.0, 0.04 * smooth.max()), distance=max(2, bins // 16))
        peaks = list(idx)
    except Exception:
        pass
    centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    out["peaks"] = [{"raw": _raw(scale, [centers[i]])[0], "events_in_bin": int(counts[i])}
                    for i in sorted(peaks, key=lambda i: -smooth[i])[:4]]
    out["peaks"].sort(key=lambda p: p["raw"] if p["raw"] is not None else 0)
    if len(peaks) >= 2:
        two = sorted(sorted(peaks, key=lambda i: -smooth[i])[:2])
        seg = smooth[two[0]: two[1] + 1]
        v = two[0] + int(np.argmin(seg))
        depth = 1.0 - smooth[v] / max(1e-9, min(smooth[two[0]], smooth[two[1]]))
        thr_disp = float(centers[v])
        thr_raw = _raw(scale, [thr_disp])[0]
        above = float((disp >= thr_disp).mean() * 100)
        out["bimodal"] = bool(depth > 0.25)
        out["valley"] = {"raw": thr_raw, "depth": clean(depth, 3)}
        out["suggested_threshold_raw"] = thr_raw
        out["percent_above_threshold"] = clean(above, 3)
    else:
        out["bimodal"] = False
    return out
