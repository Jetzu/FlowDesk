"""Dimensionality reduction, clustering and population comparison (core)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")  # scikit-learn is a dependency; skip cleanly if an environment lacks it
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

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


# ------------------------------------------------------------------ Phenograph
def _planted_graph(sizes, p_in, p_out, seed):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(len(sizes)), sizes)
    p = np.where(z[:, None] == z[None, :], p_in, p_out)
    upper = np.triu(rng.random((z.size, z.size)) < p, 1)
    return (upper | upper.T).astype(np.float64), z


def test_louvain_recovers_planted_partition():
    import scipy.sparse as sp

    adj, truth = _planted_graph([60, 50, 40, 30], 0.3, 0.02, seed=1)
    labels = A.louvain(sp.csr_matrix(adj), resolution=1.0, seed=0)
    assert adjusted_rand_score(truth, labels) == 1.0
    # ids 1..m, largest community first
    assert list(np.bincount(labels)[1:]) == [60, 50, 40, 30]
    assert A.modularity(adj, labels) == pytest.approx(A.modularity(adj, truth))
    # deterministic for a seed, dense input accepted, other seeds agree too
    np.testing.assert_array_equal(A.louvain(adj, 1.0, seed=0), labels)
    assert adjusted_rand_score(truth, A.louvain(adj, 1.0, seed=7)) == 1.0
    # weighted graph: heavy intra-block weights recover the blocks even with
    # many inter-block edges
    rng = np.random.default_rng(3)
    z = np.repeat(np.arange(3), 40)
    w = np.where(z[:, None] == z[None, :], 1.0, 0.05) * (rng.random((120, 120)) < 0.5)
    w = np.triu(w, 1)
    w = w + w.T
    assert adjusted_rand_score(z, A.louvain(w, 1.0, 0)) == 1.0
    # lower resolution merges communities; no edges -> singletons
    assert len(np.unique(A.louvain(adj, 0.01, 0))) < 4
    assert list(A.louvain(np.zeros((3, 3)))) == [1, 2, 3]


def test_knn_jaccard_graph_weights():
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    k = 8
    W = A.knn_jaccard_graph(X, k)
    assert W.shape == (200, 200)
    assert abs(W - W.T).max() < 1e-12
    assert W.diagonal().max() == 0
    assert 0 < W.data.min() and W.data.max() <= 1.0
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    nbrs = [set(row[1:]) for row in idx]
    for i in range(0, 200, 17):
        for j in nbrs[i]:
            jac = lambda a, b: len(nbrs[a] & nbrs[b]) / len(nbrs[a] | nbrs[b])  # noqa: E731
            expected = 0.5 * (jac(i, j) + (jac(j, i) if i in nbrs[j] else 0.0))
            assert W[i, j] == pytest.approx(expected)


def test_phenograph_recovers_clusters(ws3):
    ws, truths = ws3
    X, index = A.prepare_matrix(ws.samples, "", FL, None)
    truth = np.concatenate([truths[sid][rows] for sid, rows in index])
    labels = A.run_phenograph(X, k=30, seed=0, resolution=0.5)
    assert set(np.unique(labels)) == {1, 2, 3}
    counts = np.bincount(labels)[1:]
    assert list(counts) == sorted(counts, reverse=True)
    assert adjusted_rand_score(truth, labels) >= 0.9
    np.testing.assert_array_equal(A.run_phenograph(X, k=30, seed=0, resolution=0.5), labels)
    # at resolution 1 modularity prefers splitting these compact clusters; the
    # result must still be at least as modular as the planted partition and
    # never mix events of different planted clusters
    W = A.knn_jaccard_graph(X, 30)
    fine = A.run_phenograph(X, k=30, seed=0, resolution=1.0)
    assert A.modularity(W, fine) >= A.modularity(W, truth) - 1e-9
    for c in np.unique(fine):
        assert len(np.unique(truth[fine == c])) == 1


def test_phenograph_scales_to_20k_events():
    import time

    rng = np.random.default_rng(0)
    centres = rng.normal(0, 6, (4, 6))
    X = np.vstack([rng.normal(c, 1.0, (5000, 6)) for c in centres])
    truth = np.repeat(np.arange(4), 5000)
    t0 = time.perf_counter()
    labels = A.run_phenograph(X, k=30, seed=0)
    elapsed = time.perf_counter() - t0
    assert adjusted_rand_score(truth, labels) >= 0.9
    assert elapsed < 30  # typically ~2 s; generous for slow CI machines


# ------------------------------------------------------------------ projection
@pytest.fixture
def ws_ab(tmp_path):
    """Samples A, B (and C) with the same planted clusters, different events."""
    ws = Workspace()
    truths = {}
    for k in range(3):
        path, truth = write_sample(tmp_path / f"p{k}.fcs", 1500, 30 + k)
        s = ws.add_sample(path)
        s.ensure_loaded()
        truths[s.id] = truth
    return ws, truths


def _display_matrix(sample, params=FL):
    return np.column_stack([np.asarray(sample.display(p), dtype=np.float64) for p in params])


def test_project_pca_matches_source_model(ws_ab, tmp_path):
    ws, _ = ws_ab
    a, b, c = ws.samples
    X, index = A.prepare_matrix([a], "", FL, None)
    scores, model = A.fit_pca(X, 2)
    np.testing.assert_allclose(scores, A.run_pca(X, 2), atol=1e-12)
    A.write_result(ws, index, {"PC_1": scores[:, 0], "PC_2": scores[:, 1]}, {"tool": "PCA", "parameters": FL, "populations": [""]}, model=model)
    info = a.computed_info["PC_1"]
    assert info["outputs"] == ["PC_1", "PC_2"] and info["model"]["parameters"] == FL
    assert info["run"] == a.computed_info["PC_2"]["run"]

    (res,) = A.computed_results(ws)
    assert res.outputs == ["PC_1", "PC_2"] and res.kind == "pca" and res.projectable
    assert res.samples == [a] and res.missing == [b, c]

    proj = A.project_onto(ws, "PC_2", [b])
    assert proj.outputs == ["PC_1", "PC_2"] and proj.targets == [b]
    # independent reference: A's principal axes applied to B's display values
    mean = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mean, full_matrices=False)
    Xb = _display_matrix(b)
    for j, name in enumerate(["PC_1", "PC_2"]):
        ref_a = (X - mean) @ vt[j]
        sign = np.sign(ref_a @ np.asarray(a.data(name), dtype=np.float64))
        expected = (Xb - mean) @ vt[j] * sign
        np.testing.assert_allclose(b.data(name), expected, atol=1e-5)
    assert b.computed_info["PC_1"]["projected"] is True
    assert b.scale("PC_1").to_dict() == a.scale("PC_1").to_dict()  # same axes
    (res,) = A.computed_results(ws)
    assert res.samples == [a, b] and res.missing == [c] and res.source is a

    # the analysed sample cannot be projected onto its own result
    with pytest.raises(ValueError, match="original analysis"):
        A.project_onto(ws, "PC_1", [a])

    # the model is saved with the workspace: projection works after reloading
    path = tmp_path / "pca.fdw"
    save_workspace(ws, path)
    ws2 = load_workspace(path)
    c2 = ws2.sample_by_id(c.id)
    c2.ensure_loaded()
    (res2,) = A.computed_results(ws2)
    assert res2.projectable and res2.missing == [c2]
    A.project_onto(ws2, "PC_1", [c2])
    Xc = _display_matrix(c2)
    expected = Xc @ np.asarray(model["components"]).T - np.asarray(model["mean"]) @ np.asarray(model["components"]).T
    np.testing.assert_allclose(c2.data("PC_1"), expected[:, 0], atol=1e-5)


def test_project_flowsom_and_kmeans(ws_ab):
    ws, truths = ws_ab
    a, b, c = ws.samples
    X, index = A.prepare_matrix([a], "", FL, None)
    labels, codes, meta = A.run_flowsom(X, grid=(10, 10), n_metaclusters=3, iterations=10, seed=0)
    model = A.flowsom_model(codes, meta)
    np.testing.assert_array_equal(A.apply_model(model, X)[:, 0], labels)  # reproduces the fit
    A.write_result(ws, index, {"FlowSOM": labels.astype(np.float32)}, {"tool": "FlowSOM", "parameters": FL, "min": 0.0, "max": 4.0, "populations": [""]}, model=model)

    proj = A.project_onto(ws, "FlowSOM", [b])
    assert proj.created == ["FlowSOM 1", "FlowSOM 2", "FlowSOM 3"]
    projected = b.data("FlowSOM")
    assert np.all(np.isfinite(projected))
    assert adjusted_rand_score(truths[b.id], projected) >= 0.9
    # agrees with a fresh FlowSOM fit of B
    fresh, _, _ = A.run_flowsom(_display_matrix(b), grid=(10, 10), n_metaclusters=3, iterations=10, seed=1)
    assert adjusted_rand_score(fresh, projected) >= 0.9
    counts = [b.root.find(f"/FlowSOM {k}").count for k in (1, 2, 3)]
    assert sum(counts) == b.n_events and min(counts) > 300
    assert a.root.find("/FlowSOM 1") is None  # the analysed sample is untouched
    # an existing population of the same name keeps its children
    from flowdesk.core.gates import RectangleGate
    from flowdesk.core.model import Population

    pop = c.root.add_child(Population("FlowSOM 2", RectangleGate(["FL1-A"], [0.0], [1.0])))
    pop.add_child(Population("child"))
    A.project_onto(ws, "FlowSOM", [c], max_events=500, seed=2)
    assert int(np.isfinite(c.data("FlowSOM")).sum()) == 500
    node = c.root.find("/FlowSOM 2")
    assert node is pop and node.gate.dims == ["FlowSOM"] and node.child_by_name("child") is not None
    assert sum(c.root.find(f"/FlowSOM {k}").count for k in (1, 2, 3)) == 500

    # k-means: nearest stored centre
    km = A.run_kmeans(X, 3, seed=0)
    kmodel = A.kmeans_model(X, km)
    assert adjusted_rand_score(km, A.apply_model(kmodel, X)[:, 0]) >= 0.99
    A.write_result(ws, index, {"kmeans": km.astype(np.float32)}, {"tool": "k-means", "parameters": FL}, model=kmodel)
    A.project_onto(ws, "kmeans", [b])
    assert adjusted_rand_score(truths[b.id], b.data("kmeans")) >= 0.9
    assert [b.root.find(f"/kmeans {k}") is not None for k in (1, 2, 3)] == [True] * 3


def test_results_not_projectable(ws_ab, monkeypatch):
    ws, _ = ws_ab
    a, b, _c = ws.samples
    X, index = A.prepare_matrix([a], "", FL, 300)
    ph = A.run_phenograph(X, 15, 0, 0.5)
    A.write_result(ws, index, {"Phenograph": ph.astype(np.float32)}, {"tool": "Phenograph", "parameters": FL}, model={"kind": "phenograph"})
    # parameters written without run metadata (older workspaces): grouped by stem
    a.add_computed("old_1", np.zeros(a.n_events), {"tool": "t-SNE", "parameters": FL})
    a.add_computed("old_2", np.zeros(a.n_events), {"tool": "t-SNE", "parameters": FL})
    a.add_computed("PCX_1", np.zeros(a.n_events), {"tool": "PCA", "parameters": FL})
    results = {r.outputs[0]: r for r in A.computed_results(ws)}
    assert set(results) == {"Phenograph", "old_1", "PCX_1"}
    assert results["old_1"].outputs == ["old_1", "old_2"]
    assert not results["Phenograph"].projectable and "k-nearest-neighbour" in results["Phenograph"].reason
    assert not results["old_1"].projectable and "session" in results["old_1"].reason
    assert not results["PCX_1"].projectable and "no stored model" in results["PCX_1"].reason
    with pytest.raises(ValueError, match="Phenograph"):
        A.project_onto(ws, "Phenograph", [b])
    with pytest.raises(ValueError, match="no sample"):
        A.project_onto(ws, "nope", [b])


@pytest.mark.skipif(not A.has_umap(), reason="umap-learn not installed")
def test_project_umap_within_session(ws_ab):
    ws, truths = ws_ab
    a, b, _c = ws.samples
    X, index = A.prepare_matrix([a], "", FL, 600, seed=0)
    emb, fitted = A.fit_umap(X, n_neighbors=15, min_dist=0.1, seed=0)
    A.write_result(ws, index, {"UMAP_1": emb[:, 0], "UMAP_2": emb[:, 1]}, {"tool": "UMAP", "parameters": FL}, model={"kind": "umap"}, session=fitted)
    assert A.session_model(ws, "UMAP_2") is fitted
    proj = A.project_onto(ws, "UMAP_1", [b], max_events=400, seed=1)
    rows = proj.index[0][1]
    yb = np.column_stack([b.data("UMAP_1")[rows], b.data("UMAP_2")[rows]])
    assert yb.shape == (400, 2) and np.all(np.isfinite(yb))
    # projected events land next to A's events of the same planted cluster
    ta = truths[a.id][index[0][1]]
    centroids = np.array([emb[ta == k].mean(axis=0) for k in range(3)])
    nearest = np.argmin(((yb[:, None, :] - centroids[None]) ** 2).sum(axis=2), axis=1)
    assert np.mean(nearest == truths[b.id][rows]) >= 0.9
    # a new session (or reloaded workspace) has no fitted model
    A._SESSION_MODELS.clear()
    (res,) = A.computed_results(ws)
    assert not res.projectable and "session" in res.reason


@pytest.mark.skipif(not A.has_opentsne(), reason="openTSNE not installed")
def test_project_tsne_with_opentsne(ws_ab, monkeypatch):
    ws, truths = ws_ab
    a, b, _c = ws.samples
    X, index = A.prepare_matrix([a], "", FL, 500, seed=0)
    emb, fitted = A.fit_tsne(X, perplexity=30, iterations=250, seed=0)
    assert fitted is not None and hasattr(fitted, "transform")
    A.write_result(ws, index, {"tSNE_1": emb[:, 0], "tSNE_2": emb[:, 1]}, {"tool": "t-SNE", "parameters": FL}, model={"kind": "tsne", "engine": "openTSNE"}, session=fitted)
    proj = A.project_onto(ws, "tSNE_1", [b], max_events=300, seed=0)
    rows = proj.index[0][1]
    yb = np.column_stack([b.data("tSNE_1")[rows], b.data("tSNE_2")[rows]])
    assert np.all(np.isfinite(yb))
    ta = truths[a.id][index[0][1]]
    centroids = np.array([emb[ta == k].mean(axis=0) for k in range(3)])
    nearest = np.argmin(((yb[:, None, :] - centroids[None]) ** 2).sum(axis=2), axis=1)
    assert np.mean(nearest == truths[b.id][rows]) >= 0.9
    # scikit-learn's engine returns no model: the result cannot be projected
    monkeypatch.setattr(A, "has_opentsne", lambda: False)
    emb2, none = A.fit_tsne(X[:100], perplexity=10, iterations=250, seed=0)
    assert none is None
    A.write_result(ws, [(a.id, index[0][1][:100])], {"sk_1": emb2[:, 0], "sk_2": emb2[:, 1]}, {"tool": "t-SNE", "parameters": FL}, model={"kind": "tsne", "engine": "scikit-learn"})
    res = A.find_result(ws, "sk_2")
    assert res.outputs == ["sk_1", "sk_2"] and not res.projectable and "openTSNE" in res.reason
