"""Population statistics."""
from __future__ import annotations

import math

import numpy as np

STATISTICS = [
    "Count",
    "Freq. of Parent",
    "Freq. of Total",
    "Freq. of Grandparent",
    "Mean",
    "Median",
    "Geometric Mean",
    "Mode",
    "SD",
    "CV",
    "Robust SD",
    "Robust CV",
    "Min",
    "Max",
    "Percentile",
    "Sum",
]

NEEDS_PARAM = {"Mean", "Median", "Geometric Mean", "Mode", "SD", "CV", "Robust SD", "Robust CV", "Min", "Max", "Percentile", "Sum"}
PERCENT_STATS = {"Freq. of Parent", "Freq. of Total", "Freq. of Grandparent", "CV", "Robust CV"}


def compute_statistic(name: str, population, param: str | None = None, percentile: float | None = None) -> float:
    """Compute a statistic for a population. Returns NaN when undefined."""
    if name == "Count":
        return float(population.count)
    if name == "Freq. of Parent":
        return population.freq_of_parent
    if name == "Freq. of Total":
        return population.freq_of_total
    if name == "Freq. of Grandparent":
        gp = population.parent.parent if population.parent is not None else None
        if gp is None:
            return population.freq_of_total
        n = gp.count
        return 100.0 * population.count / n if n else float("nan")
    if name not in NEEDS_PARAM:
        raise ValueError(f"unknown statistic {name!r}")
    if not param:
        return float("nan")
    sample = population.sample
    if sample is None or not sample.ensure_loaded():
        return float("nan")
    try:
        values = sample.data(param)[population.mask()]
    except KeyError:
        return float("nan")
    if values.size == 0:
        return float("nan")
    values = values.astype(np.float64)
    if name == "Mean":
        return float(values.mean())
    if name == "Median":
        return float(np.median(values))
    if name == "Geometric Mean":
        positive = values[values > 0]
        if positive.size == 0:
            return float("nan")
        return float(np.exp(np.log(positive).mean()))
    if name == "Mode":
        disp = sample.display(param)[population.mask()]
        hist, edges = np.histogram(np.clip(disp, 0, 1), bins=256, range=(0.0, 1.0))
        k = int(np.argmax(hist))
        return float(sample.scale(param).inverse_scalar((edges[k] + edges[k + 1]) / 2.0))
    if name == "SD":
        return float(values.std(ddof=1)) if values.size > 1 else 0.0
    if name == "CV":
        m = values.mean()
        return float(100.0 * values.std(ddof=1) / m) if values.size > 1 and m != 0 else float("nan")
    if name == "Robust SD":
        med = np.median(values)
        return float(1.4826 * np.median(np.abs(values - med)))
    if name == "Robust CV":
        med = np.median(values)
        if med == 0:
            return float("nan")
        return float(100.0 * 1.4826 * np.median(np.abs(values - med)) / med)
    if name == "Min":
        return float(values.min())
    if name == "Max":
        return float(values.max())
    if name == "Percentile":
        p = 50.0 if percentile is None else float(percentile)
        return float(np.percentile(values, min(max(p, 0.0), 100.0)))
    if name == "Sum":
        return float(values.sum())
    return float("nan")


def format_value(name: str, value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return ""
    if name == "Count":
        return f"{int(round(value))}"
    if name in PERCENT_STATS:
        return f"{value:.{decimals}f}"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def statistic_label(name: str, param: str | None = None, percentile: float | None = None) -> str:
    if name == "Percentile" and percentile is not None:
        base = f"Percentile {percentile:g}"
    else:
        base = name
    if name in NEEDS_PARAM and param:
        return f"{base} : {param}"
    return base
