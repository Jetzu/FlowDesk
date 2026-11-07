import math

import numpy as np
import pytest

from flowdesk.core.gates import RectangleGate
from flowdesk.core.model import Population, Sample
from flowdesk.core.stats import NEEDS_PARAM, PERCENT_STATS, STATISTICS, compute_statistic, format_value, statistic_label
from flowdesk.core.transforms import LinearScale


@pytest.fixture
def tree(synthetic_fcs):
    """root -> Cells (FSC > 50000) -> FITC+ (FITC > 5000) -> PE+ (PE > 5000)."""
    s = Sample(synthetic_fcs["path"], 1)
    assert s.ensure_loaded()
    cells = s.root.add_child(Population("Cells", RectangleGate(["FSC-A"], [50000], [None])))
    fitc = cells.add_child(Population("FITC+", RectangleGate(["FITC-A"], [5000], [None])))
    pe = fitc.add_child(Population("PE+", RectangleGate(["PE-A"], [5000], [None])))
    ev = synthetic_fcs["events"].astype(np.float64)
    m_cells = ev[:, 0] >= 50000
    m_fitc = m_cells & (ev[:, 2] >= 5000)
    m_pe = m_fitc & (ev[:, 3] >= 5000)
    return {"sample": s, "cells": cells, "fitc": fitc, "pe": pe, "ev": ev, "m_cells": m_cells, "m_fitc": m_fitc, "m_pe": m_pe}


def test_statistics_list_consistent():
    assert len(STATISTICS) == len(set(STATISTICS)) == 22  # 16 + 6 cell-cycle / proliferation fits
    assert NEEDS_PARAM <= set(STATISTICS)
    assert PERCENT_STATS <= set(STATISTICS)


def test_counts_and_frequencies(tree):
    n = tree["ev"].shape[0]
    nc, nf, np_ = tree["m_cells"].sum(), tree["m_fitc"].sum(), tree["m_pe"].sum()
    # ground truth: ~80% cells, ~30% of those FITC+, ~50% of those PE+
    assert 0.77 < nc / n < 0.83
    assert 0.27 < nf / nc < 0.33
    assert 0.45 < np_ / nf < 0.55
    pe, fitc, cells, root = tree["pe"], tree["fitc"], tree["cells"], tree["sample"].root
    assert compute_statistic("Count", pe) == np_
    assert compute_statistic("Count", root) == n
    assert compute_statistic("Freq. of Parent", pe) == pytest.approx(100 * np_ / nf)
    assert compute_statistic("Freq. of Total", pe) == pytest.approx(100 * np_ / n)
    assert compute_statistic("Freq. of Grandparent", pe) == pytest.approx(100 * np_ / nc)
    # grandparent of FITC+ is the root
    assert compute_statistic("Freq. of Grandparent", fitc) == pytest.approx(100 * nf / n)
    # no grandparent: falls back to Freq. of Total
    assert compute_statistic("Freq. of Grandparent", cells) == pytest.approx(100 * nc / n)
    assert compute_statistic("Freq. of Grandparent", root) == pytest.approx(100.0)
    assert compute_statistic("Freq. of Parent", root) == 100.0


@pytest.mark.parametrize("param,col", [("FITC-A", 2), ("PE-A", 3), ("SSC-A", 1)])
def test_param_statistics_match_numpy(tree, param, col):
    pop = tree["fitc"]
    v = tree["ev"][tree["m_fitc"], col].astype(np.float32).astype(np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    expected = {
        "Mean": v.mean(),
        "Median": med,
        "Geometric Mean": np.exp(np.log(v[v > 0]).mean()),
        "SD": v.std(ddof=1),
        "CV": 100 * v.std(ddof=1) / v.mean(),
        "Robust SD": 1.4826 * mad,
        "Robust CV": 100 * 1.4826 * mad / med,
        "Min": v.min(),
        "Max": v.max(),
        "Sum": v.sum(),
    }
    for name, ref in expected.items():
        assert compute_statistic(name, pop, param) == pytest.approx(ref, rel=1e-9), name
    assert compute_statistic("Percentile", pop, param, 25) == pytest.approx(np.percentile(v, 25))
    assert compute_statistic("Percentile", pop, param, 97.5) == pytest.approx(np.percentile(v, 97.5))
    assert compute_statistic("Percentile", pop, param) == pytest.approx(med)  # default 50
    assert compute_statistic("Percentile", pop, param, 150) == pytest.approx(v.max())  # clamped
    assert compute_statistic("Percentile", pop, param, -5) == pytest.approx(v.min())


def test_statistic_values_by_stain(tree):
    # CD3 is the stain of FITC-A; data() resolves stains through the FCS file
    assert compute_statistic("Mean", tree["fitc"], "CD3") == pytest.approx(compute_statistic("Mean", tree["fitc"], "FITC-A"))


def test_mode(tree):
    s = tree["sample"]
    pop = tree["fitc"]
    v = tree["ev"][tree["m_fitc"], 2]
    mode = compute_statistic("Mode", pop, "FITC-A")
    # reference: 256-bin histogram in display space, centre of the fullest bin
    scale = s.scale("FITC-A")
    disp = np.clip(scale.forward(v), 0, 1)
    hist, edges = np.histogram(disp, bins=256, range=(0, 1))
    k = int(np.argmax(hist))
    assert mode == pytest.approx(scale.inverse_scalar((edges[k] + edges[k + 1]) / 2))
    # the positive population is lognormal around 20000
    assert 12000 < mode < 30000
    # on a linear scale the mode is reported in data units too
    s.set_scale("SSC-A", LinearScale(0, 262144))
    ssc_mode = compute_statistic("Mode", tree["cells"], "SSC-A")
    assert abs(ssc_mode - 50000) < 3000


def test_undefined_statistics(tree, tmp_path):
    pop = tree["fitc"]
    assert math.isnan(compute_statistic("Mean", pop, None))
    assert math.isnan(compute_statistic("Mean", pop, "APC-A"))  # unknown parameter
    with pytest.raises(ValueError):
        compute_statistic("Bogus", pop, "FITC-A")
    empty = tree["sample"].root.add_child(Population("Empty", RectangleGate(["FSC-A"], [1e9], [None])))
    assert compute_statistic("Count", empty) == 0
    assert math.isnan(compute_statistic("Mean", empty, "FSC-A"))
    assert compute_statistic("Freq. of Parent", empty) == 0.0
    assert math.isnan(compute_statistic("Freq. of Parent", empty.add_child(Population("Sub"))))
    # all-negative values: geometric mean undefined
    neg = tree["sample"].root.add_child(Population("Neg", RectangleGate(["FITC-A"], [None], [0])))
    assert neg.count > 0
    assert math.isnan(compute_statistic("Geometric Mean", neg, "FITC-A"))
    # single event: SD is 0
    one_val = float(tree["ev"][0, 4])
    one = tree["sample"].root.add_child(Population("One", RectangleGate(["Time"], [one_val], [one_val + 1e-3])))
    assert one.count == 1
    assert compute_statistic("SD", one, "Time") == 0.0
    assert math.isnan(compute_statistic("CV", one, "Time"))
    # sample that cannot be loaded
    missing = Sample(tmp_path / "missing.fcs", 9)
    assert math.isnan(compute_statistic("Mean", missing.root, "FSC-A"))
    assert math.isnan(compute_statistic("Freq. of Total", missing.root))


def test_statistics_skip_events_a_tool_did_not_compute(tree):
    """Computed parameters (t-SNE, UMAP, cluster ids, MESF...) are NaN for the
    events a tool did not run on (tools subsample to max_events): statistics
    are taken over the computed events only; Count still counts the gate."""
    s, pop, m = tree["sample"], tree["fitc"], tree["m_fitc"]
    rng = np.random.default_rng(11)
    n = tree["ev"].shape[0]
    values = rng.lognormal(1.0, 0.8, n) * np.where(rng.random(n) < 0.1, -1.0, 1.0)
    done = rng.random(n) < 0.5
    emb = np.where(done, values, np.nan)
    emb[tree["m_pe"]] = np.nan  # PE+ (inside FITC+): the tool ran on none of its events
    s.add_computed("emb", emb, {"tool": "test"})
    # reference: the stored float32 values of FITC+ events, NaN-aware numpy
    full = emb.astype(np.float32).astype(np.float64)[m]
    v = full[np.isfinite(full)]
    assert 100 < v.size < m.sum() and np.isnan(full).sum() > 100
    med = np.nanmedian(full)
    mad = np.nanmedian(np.abs(full - med))
    expected = {
        "Mean": np.nanmean(full),
        "Median": med,
        "Geometric Mean": np.exp(np.mean(np.log(v[v > 0]))),
        "SD": np.nanstd(full, ddof=1),
        "CV": 100 * np.nanstd(full, ddof=1) / np.nanmean(full),
        "Robust SD": 1.4826 * mad,
        "Robust CV": 100 * 1.4826 * mad / med,
        "Min": np.nanmin(full),
        "Max": np.nanmax(full),
        "Sum": np.nansum(full),
    }
    for name, ref in expected.items():
        got = compute_statistic(name, pop, "emb")
        assert math.isfinite(got), name
        assert got == pytest.approx(ref, rel=1e-9), name
    for q in (5, 50, 97.5):
        assert compute_statistic("Percentile", pop, "emb", q) == pytest.approx(np.nanpercentile(full, q))
    assert compute_statistic("Count", pop, "emb") == m.sum()
    # Mode: 256-bin display-space histogram of the computed events
    scale = s.scale("emb")
    hist, edges = np.histogram(np.clip(scale.forward(v), 0, 1), bins=256, range=(0, 1))
    k = int(np.argmax(hist))
    assert compute_statistic("Mode", pop, "emb") == pytest.approx(scale.inverse_scalar((edges[k] + edges[k + 1]) / 2))
    # no computed event at all: NaN, not an exception
    pe = tree["pe"]
    assert pe.count > 0
    for name in ("Mean", "Median", "Geometric Mean", "Mode", "SD", "CV", "Robust SD", "Robust CV", "Min", "Max", "Percentile", "Sum"):
        assert math.isnan(compute_statistic(name, pe, "emb", 50)), name
    assert compute_statistic("Count", pe, "emb") == tree["m_pe"].sum()


def test_format_value():
    assert format_value("Count", 12.6) == "13"
    assert format_value("Freq. of Parent", 12.3456) == "12.3"
    assert format_value("Freq. of Parent", 12.3456, decimals=2) == "12.35"
    assert format_value("CV", 5.0) == "5.0"
    assert format_value("Mean", 1234.7) == "1235"
    assert format_value("Mean", -1234.7) == "-1235"
    assert format_value("Median", 56.78) == "56.8"
    assert format_value("Median", 1.23456) == "1.23"
    assert format_value("Median", 0.000123456) == "0.000123"
    assert format_value("Mean", float("nan")) == ""
    assert format_value("Mean", float("inf")) == ""
    assert format_value("Mean", None) == ""
    assert format_value("Mean", np.float64(20.0)) == "20.0"


def test_statistic_label():
    assert statistic_label("Median", "FITC-A") == "Median : FITC-A"
    assert statistic_label("Percentile", "PE-A", 95) == "Percentile 95 : PE-A"
    assert statistic_label("Percentile", "PE-A", 2.5) == "Percentile 2.5 : PE-A"
    assert statistic_label("Percentile", "PE-A") == "Percentile : PE-A"
    assert statistic_label("Count", "FITC-A") == "Count"
    assert statistic_label("Freq. of Parent") == "Freq. of Parent"
    assert statistic_label("Mean") == "Mean"
