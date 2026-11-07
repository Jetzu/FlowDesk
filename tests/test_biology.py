"""Cell cycle, proliferation and kinetics models on synthetic data."""
import csv

import numpy as np
import pytest

from flowdesk.core.biology import (
    CELL_CYCLE_MODELS,
    aggregate_component,
    cell_cycle_model_key,
    debris_component,
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


def synth_cell_cycle_mixed(n=20000, fractions=(0.60, 0.15, 0.25), g1=100.0, cv=0.04, debris=0.10, doublets=0.05,
                           debris_decay=0.35, s_shape="flat", seed=0):
    """A DNA histogram with debris and doublets. `debris` and `doublets` are
    fractions of all events; `fractions` (G1, S, G2/M) are of the singlets.
    S-phase cells are spread between G1 and G2/M (flat, or accumulating early /
    late) and measured with the same CV as the peaks; debris is exponential
    from 0 with 1/e length debris_decay*g1; a doublet is the sum of two
    randomly paired singlets (G1+G1 at 2x, G1+G2 at 3x, G2+G2 at 4x G1)."""
    rng = np.random.default_rng(seed)
    nd, na = int(debris * n), int(doublets * n)
    nsing = n - nd - na
    n1, ns = int(fractions[0] * nsing), int(fractions[1] * nsing)
    n2 = nsing - n1 - ns
    u = rng.uniform(0, 1, ns)
    frac = {"flat": u, "early": 1 - np.sqrt(u), "late": np.sqrt(u)}[s_shape]
    s = g1 * (1 + frac) * (1 + cv * rng.standard_normal(ns))
    singles = np.concatenate([rng.normal(g1, cv * g1, n1), s, rng.normal(2 * g1, cv * 2 * g1, n2)])
    doublet = rng.choice(singles, size=(na, 2)).sum(axis=1)
    junk = rng.exponential(debris_decay * g1, nd)
    out = np.concatenate([singles, doublet, junk])
    rng.shuffle(out)
    return out


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


# ----------------------------------------------------------- cell-cycle models
MIXES = [(0.60, 0.15, 0.25), (0.45, 0.30, 0.25), (0.35, 0.20, 0.45), (0.70, 0.10, 0.20)]


@pytest.mark.parametrize("model", ["dean_jett_fox", "watson"])
@pytest.mark.parametrize("mix", MIXES)
@pytest.mark.parametrize("seed", [0, 1])
def test_cell_cycle_with_debris_and_doublets(model, mix, seed):
    """10 % debris and 5 % doublets: both models recover %G1/%S/%G2M to
    within 3 points and the debris / aggregate fractions."""
    v = synth_cell_cycle_mixed(fractions=mix, seed=seed)
    r = fit_cell_cycle(v, model=model, debris=True, aggregates=True)
    assert r.model == model and r.model_name == CELL_CYCLE_MODELS[model]
    assert abs(r.percent_g1 - 100 * mix[0]) < 3
    assert abs(r.percent_s - 100 * mix[1]) < 3
    assert abs(r.percent_g2m - 100 * mix[2]) < 3
    assert r.percent_g1 + r.percent_s + r.percent_g2m == pytest.approx(100.0)
    assert abs(r.percent_debris - 10) < 1.5
    assert abs(r.percent_aggregates - 5) < 1.5
    assert abs(r.g1_mean - 100) < 2 and abs(r.g2_g1_ratio - 2) < 0.05
    assert {"G1", "S", "G2/M", "Debris", "Aggregates", "Model"} == set(r.curves)
    assert r.fit_range == (pytest.approx(r.centers[0] - (r.centers[1] - r.centers[0]) / 2), pytest.approx(r.centers[-1] + (r.centers[1] - r.centers[0]) / 2))
    s = r.summary()
    assert s["%Debris"] == r.percent_debris and s["%Aggregates"] == r.percent_aggregates
    # the debris curve sits below G1, the aggregates around 2x-4x G1
    deb, agg, x = r.curves["Debris"], r.curves["Aggregates"], r.centers
    assert np.average(x, weights=deb) < 60
    assert 200 < np.average(x, weights=agg) < 320
    assert np.isfinite(r.rms) and r.success


@pytest.mark.parametrize("seed", [3, 4])
def test_cell_cycle_ignoring_debris_and_doublets_is_worse(seed):
    # G1+G1 doublets sit exactly on G2/M: without the aggregate model they
    # inflate %G2M
    v = synth_cell_cycle_mixed(doublets=0.10, seed=seed)
    plain = fit_cell_cycle(v)
    full = fit_cell_cycle(v, debris=True, aggregates=True)
    truth = np.array([60, 15, 25])

    def err(r):
        return np.abs(np.array([r.percent_g1, r.percent_s, r.percent_g2m]) - truth).max()

    assert err(full) < 1.5 < err(plain)
    assert plain.percent_g2m > 26.5
    assert abs(full.percent_aggregates - 10) < 1.5
    assert set(plain.curves) == {"G1", "S", "G2/M", "Model"}
    assert np.isnan(plain.percent_debris) and np.isnan(plain.percent_aggregates)
    assert "%Debris" not in plain.summary()


@pytest.mark.parametrize("shape", ["early", "late"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_watson_follows_a_skewed_s_phase(shape, seed):
    """The pragmatic model assumes no S-phase shape: with S cells piling up
    early or late it stays within 3 points where the flat-plateau model drifts."""
    mix = (0.45, 0.30, 0.25)
    v = synth_cell_cycle_mixed(fractions=mix, s_shape=shape, seed=seed)
    w = fit_cell_cycle(v, model="Watson Pragmatic", debris=True, aggregates=True)
    errs = np.abs(np.array([w.percent_g1, w.percent_s, w.percent_g2m]) - 100 * np.array(mix))
    assert errs.max() < 3
    assert w.curves["S"].min() >= 0


def test_watson_without_debris_on_clean_data():
    for seed in range(3):
        r = fit_cell_cycle(synth_cell_cycle(seed=seed), model="watson")
        assert abs(r.percent_g1 - 60) < 3 and abs(r.percent_s - 15) < 3 and abs(r.percent_g2m - 25) < 3
        assert set(r.curves) == {"G1", "S", "G2/M", "Model"}
    # enabling the extra components on data without debris/doublets is harmless
    r = fit_cell_cycle(synth_cell_cycle(seed=1), model="watson", debris=True, aggregates=True)
    assert r.percent_debris < 1 and r.percent_aggregates < 1
    assert abs(r.percent_g1 - 60) < 3


def test_cell_cycle_model_names():
    assert cell_cycle_model_key(None) == "dean_jett_fox"
    assert cell_cycle_model_key("Dean-Jett-Fox") == "dean_jett_fox"
    assert cell_cycle_model_key("DJF") == "dean_jett_fox"
    assert cell_cycle_model_key("Watson Pragmatic") == "watson"
    with pytest.raises(ValueError):
        cell_cycle_model_key("ModFit")
    with pytest.raises(ValueError):
        fit_cell_cycle(synth_cell_cycle(), model="bogus")


def test_debris_and_aggregate_components():
    width = 1.0
    x = np.arange(0.5, 600, width)
    deb = debris_component(x, 1000.0, 30.0, 0.0, width)
    assert deb.sum() == pytest.approx(1000.0, rel=1e-6)
    assert np.all(np.diff(deb) < 0)
    # 50/0/50 G1/G2 singlets -> doublets 25 % at 2x, 50 % at 3x, 25 % at 4x
    agg = aggregate_component(x, 1000.0, 500.0, 0.0, 500.0, 100.0, 4.0, 8.0, 2.0, width)
    assert agg.sum() == pytest.approx(1000.0, rel=1e-3)
    for centre, share in ((200, 0.25), (300, 0.50), (400, 0.25)):
        near = np.abs(x - centre) < 40
        assert agg[near].sum() / 1000.0 == pytest.approx(share, abs=0.01)
        assert abs(x[near][np.argmax(agg[near])] - centre) <= 1
    # G1+G1 width adds in quadrature: sqrt(2) * 4
    near = np.abs(x - 200) < 40
    sd = np.sqrt(np.average((x[near] - 200) ** 2, weights=agg[near]))
    assert sd == pytest.approx(np.sqrt(2) * 4, rel=0.05)
    assert not aggregate_component(x, 0.0, 1, 1, 1, 100, 4, 8, 2).any()


# ------------------------------------------------------- proliferation g0
NO_UNDIVIDED = np.array([0.0, 0.06, 0.30, 0.40, 0.24])


@pytest.mark.parametrize("seed", [0, 1])
def test_proliferation_generation0_from_control(seed):
    """Strong stimulation leaves no undivided cells: the automatic choice
    takes the brightest peak (generation 1) as generation 0, while the median
    of an undivided control places the generations correctly."""
    v = synth_proliferation(fractions=NO_UNDIVIDED, seed=seed)
    control = synth_proliferation(n=5000, fractions=[1.0], seed=100 + seed)
    g0 = np.log10(np.median(control))
    truth = proliferation_indices(NO_UNDIVIDED)
    auto = fit_proliferation(v)
    assert abs(auto.division_index - truth["division_index"]) > 0.5  # off by a generation
    for fix in (True, False):
        r = fit_proliferation(v, g0_guess=g0, fix_g0=fix)
        pct = np.zeros(10)
        pct[: r.n_generations] = r.percents
        assert np.all(np.abs(pct[:5] - 100 * NO_UNDIVIDED) < 3)
        assert abs(r.division_index - truth["division_index"]) < 0.1
        assert abs(r.percent_divided - 100) < 1.5
        if fix:
            assert r.positions[0] == pytest.approx(g0)
    # held exactly where the control says, in display space too
    scale = LogicleScale(T=262144.0)
    pos = scale.forward_scalar(float(np.median(control)))
    r = fit_proliferation(scale.forward(v), log=False, forward=scale.forward, inverse=scale.inverse, g0_guess=pos, fix_g0=True)
    assert r.positions[0] == pytest.approx(pos)
    assert abs(r.division_index - truth["division_index"]) < 0.1


def test_proliferation_fixed_g0_beyond_the_data():
    # a generation 0 above every event stays on the histogram and gets ~0 cells
    v = synth_proliferation(fractions=NO_UNDIVIDED, seed=4)
    r = fit_proliferation(v, g0_guess=np.log10(50000.0), fix_g0=True)
    assert r.centers.max() >= np.log10(50000.0)
    assert r.percents[0] < 1.0
    # fix_g0 without a position is the automatic search
    a = fit_proliferation(v, fix_g0=True)
    b = fit_proliferation(v)
    assert np.allclose(a.percents, b.percents)
