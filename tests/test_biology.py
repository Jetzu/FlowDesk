"""Cell cycle, proliferation and kinetics models on synthetic data."""
import csv

import numpy as np
import pytest

from flowdesk.core.biology import (
    fit_cell_cycle,
    fit_proliferation,
    kinetics,
    kinetics_table,
    proliferation_indices,
    write_csv,
)
from flowdesk.core.transforms import LogicleScale


def synth_cell_cycle(n=20000, fractions=(0.60, 0.15, 0.25), g1=100.0, cv=0.04, g2_cv=0.04, seed=0):
    rng = np.random.default_rng(seed)
    n1, ns = int(fractions[0] * n), int(fractions[1] * n)
    n2 = n - n1 - ns
    return np.concatenate([
        rng.normal(g1, cv * g1, n1),
        rng.uniform(g1, 2 * g1, ns),
        rng.normal(2 * g1, g2_cv * 2 * g1, n2),
    ])


GEN_FRACTIONS = np.array([0.10, 0.20, 0.30, 0.25, 0.15])


def synth_proliferation(n=20000, fractions=GEN_FRACTIONS, gen0=50000.0, sd=0.06, seed=0):
    rng = np.random.default_rng(seed)
    gens = rng.choice(len(fractions), size=n, p=np.asarray(fractions) / np.sum(fractions))
    return 10 ** (np.log10(gen0) - gens * np.log10(2) + rng.normal(0, sd, n))


def synth_kinetics(n=12000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 600, n))
    y = rng.normal(1000 + 2.0 * t, 50, n)  # median rises by 2 per time unit
    return t, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cell_cycle_recovers_fractions(seed):
    r = fit_cell_cycle(synth_cell_cycle(seed=seed))
    assert abs(r.percent_g1 - 60) < 3
    assert abs(r.percent_s - 15) < 3
    assert abs(r.percent_g2m - 25) < 3
    assert abs(r.percent_g1 + r.percent_s + r.percent_g2m - 100) < 1e-6
    assert abs(r.g1_mean - 100) < 2
    assert 1.9 <= r.g2_g1_ratio <= 2.1 and abs(r.g2_mean - 200) < 4
    assert abs(r.g1_cv - 4) < 1
    assert r.centers.shape == r.counts.shape == r.curves["Model"].shape == (256,)
    assert set(r.curves) == {"G1", "S", "G2/M", "Model"}
    assert np.isfinite(r.rms) and r.rms >= 0


def test_cell_cycle_other_mixture_and_guess():
    v = synth_cell_cycle(n=30000, fractions=(0.45, 0.30, 0.25), g1=50000, cv=0.05, g2_cv=0.045, seed=4)
    r = fit_cell_cycle(v, bins=200)
    assert abs(r.percent_g1 - 45) < 3 and abs(r.percent_s - 30) < 3 and abs(r.percent_g2m - 25) < 3
    r2 = fit_cell_cycle(v, bins=200, g1_guess=48000)
    assert abs(r2.percent_g1 - r.percent_g1) < 1


def test_cell_cycle_g2_dominant():
    # more G2/M than G1: G1 must still be the lower peak
    r = fit_cell_cycle(synth_cell_cycle(fractions=(0.35, 0.20, 0.45), seed=5))
    assert abs(r.g1_mean - 100) < 3
    assert abs(r.percent_g1 - 35) < 3 and abs(r.percent_g2m - 45) < 3


def test_cell_cycle_too_few_events():
    with pytest.raises(ValueError):
        fit_cell_cycle(np.arange(10.0))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_proliferation_recovers_generations(seed):
    r = fit_proliferation(synth_proliferation(seed=seed), max_generations=10)
    pct = np.zeros(10)
    pct[: r.n_generations] = r.percents
    assert np.all(np.abs(pct[:5] - 100 * GEN_FRACTIONS) < 3)
    assert np.all(pct[5:] < 1.5)
    assert abs(r.generation0 / 50000 - 1) < 0.05
    truth = proliferation_indices(GEN_FRACTIONS)
    assert abs(r.division_index - truth["division_index"]) < 0.1
    assert abs(r.percent_divided - truth["percent_divided"]) < 3
    assert r.curves["Model"].shape == r.counts.shape


def test_proliferation_in_logicle_display_space():
    scale = LogicleScale(T=262144.0)
    raw = synth_proliferation(seed=7)
    disp = scale.forward(raw)
    r = fit_proliferation(disp, log=False, forward=scale.forward, inverse=scale.inverse, max_generations=8)
    pct = np.zeros(8)
    pct[: r.n_generations] = r.percents
    assert np.all(np.abs(pct[:5] - 100 * GEN_FRACTIONS) < 3)
    assert abs(scale.inverse_scalar(r.positions[0]) / 50000 - 1) < 0.05


def test_proliferation_indices_known_values():
    # 100 undivided cells and 200 cells in generation 1 (from 100 precursors)
    idx = proliferation_indices([100, 200])
    assert idx["division_index"] == pytest.approx(0.5)
    assert idx["proliferation_index"] == pytest.approx(1.0)
    assert idx["percent_divided"] == pytest.approx(50.0)
    assert idx["expansion_index"] == pytest.approx(1.5)
    assert idx["replication_index"] == pytest.approx(2.0)


def test_kinetics_median_trend_equal_count():
    t, y = synth_kinetics()
    r = kinetics(t, y, n_bins=12, statistic="median")
    assert r.centers.size == 12 and r.counts.sum() == t.size
    assert r.counts.max() - r.counts.min() <= 1
    assert np.all(np.diff(r.centers) > 0)
    assert np.allclose(r.values, 1000 + 2 * r.centers, atol=15)
    assert r.slope == pytest.approx(2.0, rel=0.05)
    assert np.all(r.lo < r.values) and np.all(r.values < r.hi)


def test_kinetics_equal_width_mean_and_mask():
    t, y = synth_kinetics()
    mask = t < 300
    r = kinetics(t, y, n_bins=10, statistic="mean", mask=mask, binning="equal_width")
    assert r.counts.sum() == mask.sum()
    width = np.diff(r.edges)
    assert np.allclose(width, width[0])
    assert np.allclose(r.centers, 0.5 * (r.edges[:-1] + r.edges[1:]))
    assert np.allclose(r.values, 1000 + 2 * r.centers, atol=15)


def test_kinetics_percent_positive_and_table(tmp_path):
    t = np.linspace(0, 100, 10000)
    y = np.where(t < 50, 0.0, 10.0)
    r = kinetics(t, y, n_bins=4, statistic="percent_positive", threshold=5)
    assert list(np.round(r.values)) == [0, 0, 100, 100]
    with pytest.raises(ValueError):
        kinetics(t, y, statistic="percent_positive")
    with pytest.raises(ValueError):
        kinetics(t, y, statistic="mode")
    rows = kinetics_table(r, include_header=True)
    assert rows[0] == ["Bin", "Time", "Statistic", "Count", "Q1", "Q3"]
    assert len(rows) == 5 and rows[1][0] == 1 and rows[1][3] == 2500
    path = tmp_path / "k.csv"
    write_csv(path, rows[0], rows[1:])
    with open(path) as fh:
        back = list(csv.reader(fh))
    assert len(back) == 5 and float(back[3][2]) == 100.0
