"""Dimensionality reduction, clustering and population comparison (core)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_score

from flowdesk.core import analysis as A
from flowdesk.core.fcs import Channel, write_fcs
from flowdesk.core.model import Workspace
from flowdesk.core.serialization import load_workspace, save_workspace

NAMES = ["FL1-A", "FL2-A", "FL3-A", "FL4-A", "Time"]
CENTRES = np.array(
    [
        [500.0, 20000.0, 3000.0, 800.0],
        [20000.0, 500.0, 800.0, 3000.0],
        [3000.0, 3000.0, 30000.0, 30000.0],
    ]
)


def make_events(n: int, seed: int, shift: dict[int, float] | None = None):
    """Three log-normal clusters in four fluorescence parameters plus Time."""
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 3, n)
    centres = CENTRES.copy()
    for col, factor in (shift or {}).items():
        centres[:, col] *= factor
    values = centres[truth] * rng.lognormal(0.0, 0.15, (n, 4))
    time = np.linspace(0, 100, n)[:, None]
    return np.hstack([values, time]).astype(np.float32), truth


def write_sample(path: Path, n: int, seed: int, shift=None):
    events, truth = make_events(n, seed, shift)
    write_fcs(path, events, [Channel(i + 1, name, "", 262144.0) for i, name in enumerate(NAMES)])
    return path, truth


@pytest.fixture
def ws3(tmp_path):
    ws = Workspace()
    truths = {}
    for k in range(2):
        path, truth = write_sample(tmp_path / f"s{k}.fcs", 1500, 10 + k)
        s = ws.add_sample(path)
        s.ensure_loaded()
        truths[s.id] = truth
    return ws, truths


FL = NAMES[:4]


def test_prepare_matrix_subsamples_and_indexes(ws3):
    ws, _ = ws3
    X, index = A.prepare_matrix(ws.samples, "", FL, 400, seed=1)
    assert X.shape == (800, 4)
    assert [sid for sid, _ in index] == [s.id for s in ws.samples]
    for _, rows in index:
        assert len(rows) == 400 and np.all(np.diff(rows) > 0)
    assert 0.0 <= X.min() and X.max() <= 1.0  # display space
    s = ws.samples[0]
    np.testing.assert_allclose(X[:400, 2], s.display("FL3-A")[index[0][1]])
    # no subsampling
    X2, index2 = A.prepare_matrix(ws.samples, ["", ""], FL, None, seed=1)
    assert X2.shape[0] == 3000
    with pytest.raises(ValueError):
        A.prepare_matrix(ws.samples, "", ["nope"], 10)


def test_pca_separates_clusters(ws3):
    ws, truths = ws3
    X, index = A.prepare_matrix(ws.samples, "", FL, None)
    truth = np.concatenate([truths[sid][rows] for sid, rows in index])
    emb = A.run_pca(X, 2)
    assert emb.shape == (X.shape[0], 2)
    assert silhouette_score(emb, truth) > 0.6
    centroids = np.array([emb[truth == k].mean(axis=0) for k in range(3)])
    spread = max(emb[truth == k].std(axis=0).max() for k in range(3))
    dists = [np.linalg.norm(centroids[i] - centroids[j]) for i in range(3) for j in range(i + 1, 3)]
    assert min(dists) > 4 * spread
    ev = A.pca_explained_variance(X, 2)
    assert ev[0] >= ev[1] > 0 and ev.sum() <= 1.0 + 1e-9


@pytest.mark.parametrize("engine", ["default", "sklearn"])
def test_tsne_writes_nan_outside_subsample(ws3, monkeypatch, engine):
    ws, truths = ws3
    if engine == "sklearn":
        monkeypatch.setattr(A, "has_opentsne", lambda: False)
    s = ws.samples[0]
    X, index = A.prepare_matrix([s], "", FL, 600, seed=3)
    emb = A.run_tsne(X, perplexity=30, iterations=250, seed=0)
    assert emb.shape == (600, 2) and np.all(np.isfinite(emb))
    A.write_back(ws, index, {"tSNE_1": emb[:, 0], "tSNE_2": emb[:, 1]}, {"tool": "t-SNE"})
    v = s.data("tSNE_1")
    assert v.shape == (s.n_events,)
    assert int(np.isfinite(v).sum()) == 600
    assert np.all(np.isnan(np.delete(v, index[0][1])))
    assert "tSNE_1" in s.parameter_names() and "tSNE_2" in s.parameter_names()
    assert s.scale("tSNE_1").kind == "linear"
    assert silhouette_score(emb, truths[s.id][index[0][1]]) > 0.5
    # the other sample is untouched
    assert "tSNE_1" not in ws.samples[1].computed


def test_umap_optional(ws3, monkeypatch):
    ws, truths = ws3
    X, index = A.prepare_matrix(ws.samples[:1], "", FL, 500)
    if A.has_umap():
        emb = A.run_umap(X, n_neighbors=15, min_dist=0.1, seed=0)
        assert emb.shape == (500, 2)
    monkeypatch.setattr(A, "has_umap", lambda: False)
    with pytest.raises(ImportError, match="umap-learn"):
        A.run_umap(X)


def test_kmeans_and_flowsom_recover_clusters(ws3):
    ws, truths = ws3
    X, index = A.prepare_matrix(ws.samples, "", FL, None)
    truth = np.concatenate([truths[sid][rows] for sid, rows in index])
    km = A.run_kmeans(X, 3, seed=0)
    assert set(np.unique(km)) == {1, 2, 3}
    assert adjusted_rand_score(truth, km) >= 0.9
    labels, codes, meta = A.run_flowsom(X, grid=(10, 10), n_metaclusters=3, iterations=10, seed=0)
    assert codes.shape == (100, 4) and meta.shape == (100,)
    assert set(np.unique(labels)) == {1, 2, 3}
    assert adjusted_rand_score(truth, labels) >= 0.9
    # numpy fallback for k-means
    fallback = A._kmeans_numpy(X, 3, seed=0)
    assert adjusted_rand_score(truth, fallback) >= 0.9


def test_cluster_populations_counts_and_persistence(ws3, tmp_path):
    ws, truths = ws3
    X, index = A.prepare_matrix(ws.samples, "", FL, None)
    labels = A.run_kmeans(X, 3, seed=0)
    A.write_back(ws, index, {"kmeans": labels.astype(np.float32)}, {"tool": "k-means"})
    names = A.clusters_to_populations(ws, ws.groups[0], "", "kmeans", np.unique(labels), "kmeans")
    assert names == ["kmeans 1", "kmeans 2", "kmeans 3"]
    offset = 0
    expected = {}
    for sid, rows in index:
        s = ws.sample_by_id(sid)
        own = labels[offset:offset + len(rows)]
        offset += len(rows)
        for k in (1, 2, 3):
            pop = s.root.find(f"/kmeans {k}")
            assert pop is not None and pop.count == int((own == k).sum())
            expected[(sid, k)] = pop.count
    assert ws.groups[0].root.find("/kmeans 2") is not None
    assert sum(expected.values()) == 3000

    # without a group: per sample, under a sub-population; existing names replaced
    names2 = A.clusters_to_populations(ws, None, "/kmeans 1", "kmeans", [1, 2], "sub")
    assert names2 == ["sub 1", "sub 2"]
    s0 = ws.samples[0]
    assert s0.root.find("/kmeans 1/sub 1").count == expected[(s0.id, 1)]
    assert s0.root.find("/kmeans 1/sub 2").count == 0
    A.clusters_to_populations(ws, None, "/kmeans 1", "kmeans", [1], "sub")
    assert len([c for c in s0.root.find("/kmeans 1").children if c.name == "sub 1"]) == 1

    # save / load keeps computed parameters and the cluster gates
    path = tmp_path / "analysis.fdw"
    save_workspace(ws, path)
    ws2 = load_workspace(path)
    for s in ws2.samples:
        s.ensure_loaded()
        assert "kmeans" in s.computed
        old = ws.sample_by_id(s.id)
        np.testing.assert_array_equal(s.computed["kmeans"], old.computed["kmeans"])
        assert s.computed_info["kmeans"]["tool"] == "k-means"
        for k in (1, 2, 3):
            assert s.root.find(f"/kmeans {k}").count == expected[(s.id, k)]


def test_write_back_with_nan_and_repeated_samples(ws3):
    ws, _ = ws3
    s = ws.samples[0]
    index = [(s.id, np.array([0, 1, 2])), (s.id, np.array([5]))]
    A.write_back(ws, index, {"X": np.array([1.0, 2.0, 3.0, 4.0])})
    v = s.data("X")
    assert list(v[[0, 1, 2, 5]]) == [1, 2, 3, 4]
    assert int(np.isfinite(v).sum()) == 4
    with pytest.raises(ValueError):
        A.write_back(ws, index, {"Y": np.zeros(3)})


def test_compare_ranks_differing_parameter_first(tmp_path):
    ws = Workspace()
    a = ws.add_sample(write_sample(tmp_path / "ctrl.fcs", 3000, 1)[0])
    b = ws.add_sample(write_sample(tmp_path / "stim.fcs", 3000, 2, shift={2: 6.0})[0])
    params = FL
    rows = A.compare_populations(A.population_values(a.root, params), A.population_values(b.root, params))
    assert [r.param for r in rows][0] == "FL3-A"
    top, rest = rows[0], rows[1:]
    assert top.ks_statistic > 0.3 and top.chi_squared_T > 10
    assert all(r.difference_score < top.difference_score for r in rest)
    assert all(r.ks_statistic < 0.1 for r in rest)
    # identical distributions: T(x) ~ 0
    same = A.compare_populations({"p": np.random.default_rng(0).random(5000)}, {"p": np.random.default_rng(1).random(5000)})
    assert same[0].chi_squared_T < 4 and same[0].ks_statistic < 0.05


def test_unique_names_and_medians(ws3):
    ws, _ = ws3
    assert A.unique_names(ws, ["tSNE_1", "tSNE_2"]) == ["tSNE_1", "tSNE_2"]
    s = ws.samples[0]
    s.add_computed("tSNE_1", np.zeros(s.n_events))
    assert A.unique_names(ws, ["tSNE_1", "tSNE_2"]) == ["tSNE2_1", "tSNE2_2"]
    s.add_computed("FlowSOM", np.zeros(s.n_events))
    assert A.unique_names(ws, ["FlowSOM"]) == ["FlowSOM2"]
    values = np.array([[1.0, 10.0], [3.0, 30.0], [100.0, 0.0]])
    ids, med, counts = A.cluster_medians(values, np.array([1, 1, 2]))
    assert list(ids) == [1, 2] and list(counts) == [2, 1]
    np.testing.assert_allclose(med, [[2.0, 20.0], [100.0, 0.0]])
