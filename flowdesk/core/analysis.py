"""Dimensionality reduction, clustering and population comparison.

Pure numpy / scipy / scikit-learn computations (no Qt). The usual flow is

    X, index = prepare_matrix(samples, paths, params, 5000, seed)
    emb = run_tsne(X)                       # or run_pca / run_umap
    write_back(workspace, index, {"tSNE_1": emb[:, 0], "tSNE_2": emb[:, 1]}, info)

`index` records which event rows of which sample produced each row of X, so
results can be stored as computed parameters (NaN for events that were not
part of the subsample).

Clustering: k-means, FlowSOM (SOM + consensus metaclustering) and
Phenograph (k-nearest-neighbour graph with Jaccard weights, partitioned by
the Louvain modularity algorithm implemented here in numpy/scipy.sparse).

Projection: results written with `write_result` keep what is needed to place
the events of further samples on the same axes / clusters without
recomputing (`computed_results`, `project_onto`). PCA, FlowSOM and k-means
store their model in the parameter's `computed_info["model"]` (saved with the
workspace, so projection always works). t-SNE (openTSNE) and UMAP keep the
fitted embedding object in a per-workspace in-memory registry, so they can be
projected only within the session that computed them.

Optional dependencies: openTSNE (faster t-SNE that can also embed new events;
falls back to sklearn.manifold.TSNE) and umap-learn (required for UMAP).

Cancellation: the long computations accept `should_cancel`, a callable polled
inside the fits (t-SNE iterations, UMAP epochs, SOM training iterations,
k-means initialisations, kNN-graph chunks and Louvain moves); when it returns
True they raise `Cancelled`.
"""
from __future__ import annotations

import importlib.util
import threading
import uuid
import warnings
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Sequence

import numpy as np

IndexEntry = tuple[int, np.ndarray]  # (sample id, event row indices)
ProgressFn = Callable[[str], None]


class Cancelled(Exception):
    """Raised by long computations when their `should_cancel()` returns True."""


def _check_cancel(should_cancel) -> None:
    if should_cancel is not None and should_cancel():
        raise Cancelled()


def _floats(a) -> list:
    """Nested lists of floats rounded to 10 significant digits (compact JSON)."""
    arr = np.asarray(a, dtype=np.float64)
    flat = np.array([float(f"{v:.10g}") for v in arr.ravel()], dtype=np.float64)
    return flat.reshape(arr.shape).tolist()

# --------------------------------------------------------------- availability
def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_opentsne() -> bool:
    return has_module("openTSNE")


def has_umap() -> bool:
    return has_module("umap")


def has_sklearn() -> bool:
    return has_module("sklearn")


# --------------------------------------------------------------------- matrix
def prepare_matrix(samples: Sequence, population_paths, params: Sequence[str], max_events_per_sample: int | None = None, seed: int = 0, should_cancel=None) -> tuple[np.ndarray, list[IndexEntry]]:
    """Stack the display-space (0..1) values of `params` for a random subsample
    of each population.

    `population_paths` is one path per sample (a single string applies to all;
    Population objects are accepted too). Events with a non-finite value in
    any chosen parameter are skipped. Returns (X, index) where index is a list
    of (sample_id, sorted event rows), in the row order of X.
    """
    params = list(params)
    if not params:
        raise ValueError("choose at least one parameter")
    samples = list(samples)
    if isinstance(population_paths, str) or population_paths is None:
        population_paths = [population_paths or ""] * len(samples)
    population_paths = list(population_paths)
    if len(population_paths) != len(samples):
        raise ValueError("one population path per sample is required")
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    index: list[IndexEntry] = []
    for sample, path in zip(samples, population_paths):
        _check_cancel(should_cancel)
        if not sample.ensure_loaded():
            raise ValueError(f"{sample.name}: {sample.load_error or 'file not found'}")
        if hasattr(path, "mask"):
            pop = path
        else:
            pop = sample.root.find(path or "")
        if pop is None:
            raise ValueError(f"{sample.name} has no population {path}")
        available = set(sample.parameter_names())
        missing = [p for p in params if p not in available]
        if missing:
            raise ValueError(f"{sample.name} lacks parameter(s): {', '.join(missing)}")
        cols = [np.asarray(sample.display(p), dtype=np.float64) for p in params]
        mask = pop.mask().copy()
        for c in cols:
            mask &= np.isfinite(c)
        rows = np.flatnonzero(mask)
        if max_events_per_sample and rows.size > max_events_per_sample:
            rows = np.sort(rng.choice(rows, size=int(max_events_per_sample), replace=False))
        if rows.size == 0:
            continue
        blocks.append(np.column_stack([c[rows] for c in cols]))
        index.append((sample.id, rows))
    if not blocks:
        raise ValueError("the chosen populations contain no events")
    return np.vstack(blocks), index


def index_size(index: list[IndexEntry]) -> int:
    return int(sum(len(r) for _, r in index))


# ---------------------------------------------------------------- embeddings
def fit_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, dict]:
    """Principal component scores (as `run_pca`) and a JSON-friendly model
    {"kind": "pca", "mean", "components"} with scores = (X - mean) @ components.T,
    which places new rows on the same components (`apply_model`)."""
    X = np.asarray(X, dtype=np.float64)
    n_components = max(1, min(n_components, X.shape[1], X.shape[0]))
    mean = X.mean(axis=0)
    centred = X - mean
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    # deterministic sign: largest loading of each component positive
    signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
    signs[signs == 0] = 1
    scores = (u[:, :n_components] * s[:n_components]) * signs[:n_components]
    components = vt[:n_components] * signs[:n_components, None]
    return scores, {"kind": "pca", "mean": _floats(mean), "components": _floats(components)}


def run_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Principal component scores (via SVD of the centred matrix)."""
    return fit_pca(X, n_components)[0]


def pca_explained_variance(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    var = s**2
    total = var.sum() or 1.0
    return var[:n_components] / total


def run_tsne(X: np.ndarray, perplexity: float = 30, iterations: int = 500, seed: int = 0, should_cancel=None) -> np.ndarray:
    """2-D t-SNE embedding (openTSNE when installed, else scikit-learn)."""
    return fit_tsne(X, perplexity, iterations, seed, should_cancel)[0]


# scikit-learn's TSNE has no callback. Its optimiser calls the module-level
# objective (_kl_divergence / _kl_divergence_bh) once per iteration, looked up
# at call time, so those are wrapped (once, process-wide) to poll the
# should_cancel of the calling thread; other threads and callers are
# unaffected. Wrapping the objective keeps the optimisation exactly as
# scikit-learn runs it (re-fitting in chunks of iterations would restart the
# momentum / gains and recompute the affinities at every chunk).
_SKLEARN_CANCEL = threading.local()
_SKLEARN_HOOK_LOCK = threading.Lock()


def _install_sklearn_tsne_hook() -> bool:
    try:
        from sklearn.manifold import _t_sne
    except ImportError:  # pragma: no cover
        return False
    with _SKLEARN_HOOK_LOCK:
        if getattr(_t_sne, "_flowdesk_cancel_hook", False):
            return True
        originals = {name: getattr(_t_sne, name, None) for name in ("_kl_divergence", "_kl_divergence_bh")}
        if any(not callable(f) for f in originals.values()):
            return False  # a scikit-learn without these internals: no mid-fit cancellation

        def wrap(objective):
            def polled(*args, **kwargs):
                _check_cancel(getattr(_SKLEARN_CANCEL, "should_cancel", None))
                return objective(*args, **kwargs)

            polled.__wrapped__ = objective
            return polled

        for name, objective in originals.items():
            setattr(_t_sne, name, wrap(objective))
        _t_sne._flowdesk_cancel_hook = True
        return True


def _opentsne_cancel_callback(should_cancel):
    # openTSNE stops the optimisation when a callback returns True and returns
    # the embedding so far; fit_tsne then raises Cancelled
    return lambda _iteration, _error, _embedding: bool(should_cancel())


def fit_tsne(X: np.ndarray, perplexity: float = 30, iterations: int = 500, seed: int = 0, should_cancel=None) -> tuple[np.ndarray, object | None]:
    """(embedding, fitted model). The model is openTSNE's TSNEEmbedding, whose
    `transform` places new events in the existing map; None with the
    scikit-learn engine, which cannot embed new events. `should_cancel` is
    polled during the optimisation (every 10 iterations with openTSNE, every
    iteration with scikit-learn)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 5:
        raise ValueError("t-SNE needs at least 5 events")
    perplexity = float(max(2.0, min(perplexity, (n - 1) / 3.0)))
    iterations = int(max(250, iterations))
    _check_cancel(should_cancel)
    if has_opentsne():
        from openTSNE import TSNE

        extra = {}
        if should_cancel is not None:
            extra = {"callbacks": _opentsne_cancel_callback(should_cancel), "callbacks_every_iters": 10}
        emb = TSNE(n_components=2, perplexity=perplexity, n_iter=iterations, random_state=seed, n_jobs=-1, verbose=False, **extra).fit(X)
        _check_cancel(should_cancel)
        return np.asarray(emb, dtype=np.float64), emb
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("t-SNE requires openTSNE or scikit-learn (pip install 'flowdesk[analysis]')") from exc
    kwargs = dict(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    try:
        model = TSNE(max_iter=iterations, **kwargs)
    except TypeError:  # scikit-learn < 1.5
        model = TSNE(n_iter=iterations, **kwargs)
    hooked = should_cancel is not None and _install_sklearn_tsne_hook()
    previous = getattr(_SKLEARN_CANCEL, "should_cancel", None)
    if hooked:
        _SKLEARN_CANCEL.should_cancel = should_cancel
    try:
        emb = model.fit_transform(X)
    finally:
        if hooked:
            _SKLEARN_CANCEL.should_cancel = previous
    _check_cancel(should_cancel)
    return np.asarray(emb, dtype=np.float64), None


def run_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0, should_cancel=None) -> np.ndarray:
    """2-D UMAP embedding (requires umap-learn)."""
    return fit_umap(X, n_neighbors, min_dist, seed, should_cancel)[0]


class _CancelStream:
    """A write-only stream for umap-learn's per-epoch progress bar
    (`tqdm_kwds`, the only per-epoch hook its API offers): each refresh polls
    `should_cancel` and raises Cancelled out of the epoch loop (once; the
    bar's closing writes are swallowed)."""

    def __init__(self, should_cancel):
        self.should_cancel = should_cancel
        self.raised = False

    def write(self, _text):
        if not self.raised and self.should_cancel():
            self.raised = True
            raise Cancelled()
        return 0

    def flush(self):
        pass


def fit_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0, should_cancel=None) -> tuple[np.ndarray, object]:
    """(embedding, fitted umap.UMAP); the model's `transform` places new events.
    `should_cancel` is polled before and after the neighbour graph and during
    the layout epochs."""
    if not has_umap():
        raise ImportError("UMAP requires the optional package umap-learn (pip install umap-learn, or pip install 'flowdesk[analysis]')")
    import umap

    X = np.asarray(X, dtype=np.float64)
    n_neighbors = int(max(2, min(n_neighbors, X.shape[0] - 1)))
    _check_cancel(should_cancel)
    extra = {}
    if should_cancel is not None:
        extra["tqdm_kwds"] = {"disable": False, "file": _CancelStream(should_cancel), "mininterval": 0.05, "miniters": 1}
    try:
        model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=float(min_dist), random_state=seed, **extra)
    except TypeError:  # an umap-learn without tqdm_kwds
        model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=float(min_dist), random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
        emb = model.fit_transform(X)
    _check_cancel(should_cancel)
    return np.asarray(emb, dtype=np.float64), model


# ---------------------------------------------------------------- clustering
def _sq_dist_argmin(X: np.ndarray, C: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Index of the nearest row of C for every row of X."""
    out = np.empty(X.shape[0], dtype=np.int64)
    cc = (C**2).sum(axis=1)
    for start in range(0, X.shape[0], chunk):
        xb = X[start:start + chunk]
        d = cc[None, :] - 2.0 * xb @ C.T
        out[start:start + chunk] = np.argmin(d, axis=1)
    return out


def _kmeans_numpy(X: np.ndarray, k: int, seed: int, n_init: int = 5, max_iter: int = 100, should_cancel=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        _check_cancel(should_cancel)
        # k-means++ seeding
        centres = [X[rng.integers(X.shape[0])]]
        d2 = ((X - centres[0]) ** 2).sum(axis=1)
        for _ in range(1, k):
            p = d2 / d2.sum() if d2.sum() > 0 else None
            centres.append(X[rng.choice(X.shape[0], p=p)])
            d2 = np.minimum(d2, ((X - centres[-1]) ** 2).sum(axis=1))
        C = np.array(centres)
        labels = np.zeros(X.shape[0], dtype=np.int64)
        for it in range(max_iter):
            new = _sq_dist_argmin(X, C)
            if it and np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                members = X[labels == j]
                if len(members):
                    C[j] = members.mean(axis=0)
        inertia = ((X - C[labels]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = labels.copy(), inertia
    return best


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True when two label vectors describe the same partition (ids may differ)."""
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    return len(pairs) == len(np.unique(a)) == len(np.unique(b))


def _kmeans_sklearn(X: np.ndarray, k: int, seed: int, n_init: int = 10, should_cancel=None) -> np.ndarray:
    """scikit-learn's KMeans(n_init=10, random_state=seed), one initialisation
    at a time so `should_cancel` is polled between them. The initialisations
    draw from one shared RandomState in the same order as a single
    n_init=10 fit, and the best one is kept by the same rule (lower inertia,
    unless it is the same partition), so the labels are identical."""
    from sklearn.cluster import KMeans
    from sklearn.utils import check_random_state

    rs = check_random_state(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        _check_cancel(should_cancel)
        km = KMeans(n_clusters=k, n_init=1, random_state=rs).fit(X)
        if best_labels is None or (km.inertia_ < best_inertia and not _same_partition(km.labels_, best_labels)):
            best_labels, best_inertia = km.labels_, km.inertia_
    return best_labels


def run_kmeans(X: np.ndarray, k: int, seed: int = 0, should_cancel=None) -> np.ndarray:
    """k-means cluster ids 1..k (one per row of X); `should_cancel` is polled
    between the initialisations."""
    X = np.asarray(X, dtype=np.float64)
    k = int(max(1, min(k, X.shape[0])))
    if has_sklearn():
        labels = _kmeans_sklearn(X, k, seed, should_cancel=should_cancel)
    else:
        labels = _kmeans_numpy(X, k, seed, should_cancel=should_cancel)
    return _renumber(np.asarray(labels))


def _renumber(labels: np.ndarray) -> np.ndarray:
    """Map labels to 1..m, ordered by decreasing cluster size."""
    uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(1, len(uniq) + 1)
    return rank[inverse.ravel()]


def train_som(X: np.ndarray, grid: tuple[int, int] = (10, 10), iterations: int = 10, seed: int = 0, sigma_start: float | None = None, sigma_end: float = 0.5, should_cancel=None) -> np.ndarray:
    """Batch self-organising map. Returns node codes (gx*gy x n_params).

    Each epoch assigns every event to its best-matching node and replaces each
    code by the neighbourhood-weighted mean of the events, with a Gaussian
    neighbourhood on the grid whose width shrinks geometrically from
    `sigma_start` (default: half the larger grid side) to `sigma_end`.
    """
    X = np.asarray(X, dtype=np.float64)
    gx, gy = int(grid[0]), int(grid[1])
    n_nodes = gx * gy
    rng = np.random.default_rng(seed)
    replace = X.shape[0] < n_nodes
    codes = X[rng.choice(X.shape[0], size=n_nodes, replace=replace)].copy()
    if replace:
        codes += rng.normal(0, 1e-3, codes.shape)
    coords = np.array([(i, j) for i in range(gx) for j in range(gy)], dtype=np.float64)
    grid_d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    s0 = float(sigma_start if sigma_start is not None else max(gx, gy) / 2.0)
    s0 = max(s0, sigma_end)
    iterations = max(1, int(iterations))
    for t in range(iterations):
        _check_cancel(should_cancel)
        frac = t / (iterations - 1) if iterations > 1 else 1.0
        sigma = s0 * (sigma_end / s0) ** frac
        bmu = _sq_dist_argmin(X, codes)
        counts = np.bincount(bmu, minlength=n_nodes).astype(np.float64)
        sums = np.zeros_like(codes)
        np.add.at(sums, bmu, X)
        H = np.exp(-grid_d2 / (2.0 * sigma * sigma))
        num = H @ sums
        den = H @ counts
        ok = den > 1e-12
        codes[ok] = num[ok] / den[ok, None]
    return codes


def consensus_metaclusters(codes: np.ndarray, k: int, reps: int = 20, fraction: float = 0.9, seed: int = 0, should_cancel=None) -> np.ndarray:
    """ConsensusClusterPlus-style metaclustering of SOM codes.

    Repeatedly clusters random subsets of nodes (average-linkage hierarchical
    clustering cut at k clusters), builds the co-clustering consensus matrix
    and cuts an average-linkage tree of 1 - consensus into k metaclusters.
    Returns one metacluster id (1..k) per node.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    codes = np.asarray(codes, dtype=np.float64)
    n = codes.shape[0]
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.ones(n, dtype=np.int64)
    if n <= 2:
        return np.arange(1, n + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    together = np.zeros((n, n))
    sampled = np.zeros((n, n))
    size = max(k + 1, int(round(fraction * n)))
    size = min(size, n)
    for _ in range(max(1, reps)):
        _check_cancel(should_cancel)
        idx = np.sort(rng.choice(n, size=size, replace=False))
        labels = fcluster(linkage(codes[idx], method="average"), k, criterion="maxclust")
        same = (labels[:, None] == labels[None, :]).astype(np.float64)
        together[np.ix_(idx, idx)] += same
        sampled[np.ix_(idx, idx)] += 1.0
    consensus = np.divide(together, sampled, out=np.zeros_like(together), where=sampled > 0)
    dist = 1.0 - consensus
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), k, criterion="maxclust")
    return labels.astype(np.int64)


def run_flowsom(X: np.ndarray, grid: tuple[int, int] = (10, 10), n_metaclusters: int = 10, iterations: int = 10, seed: int = 0, should_cancel=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FlowSOM: SOM + consensus hierarchical metaclustering.

    Returns (labels, som_codes, metacluster_of_node) where labels holds the
    metacluster id (1..k, largest first) of every row of X and
    metacluster_of_node the id of each SOM node. `should_cancel` is polled
    every training iteration and metaclustering repetition.
    """
    X = np.asarray(X, dtype=np.float64)
    codes = train_som(X, grid, iterations, seed, should_cancel=should_cancel)
    meta = consensus_metaclusters(codes, n_metaclusters, seed=seed, should_cancel=should_cancel)
    bmu = _sq_dist_argmin(X, codes)
    event_meta = meta[bmu]
    # renumber metaclusters 1..m by decreasing event count; empty ones last
    counts = np.bincount(event_meta, minlength=meta.max() + 1)
    ids = np.unique(meta)
    order = sorted(ids, key=lambda m: (-counts[m], m))
    mapping = np.zeros(meta.max() + 1, dtype=np.int64)
    for new, old in enumerate(order, start=1):
        mapping[old] = new
    return mapping[event_meta], codes, mapping[meta]


def flowsom_model(codes: np.ndarray, metacluster_of_node: np.ndarray) -> dict:
    """JSON-friendly FlowSOM model: an event belongs to the metacluster of its
    nearest SOM node (`apply_model`)."""
    return {"kind": "flowsom", "som_codes": _floats(codes), "metacluster_of_node": [int(m) for m in np.asarray(metacluster_of_node).ravel()]}


def kmeans_model(X: np.ndarray, labels: np.ndarray) -> dict:
    """JSON-friendly k-means model: the centre (mean) of each cluster id 1..k;
    new events go to the nearest centre."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    k = int(labels.max()) if labels.size else 0
    centres = np.zeros((k, X.shape[1]))
    for j in range(1, k + 1):
        sel = labels == j
        # an id without events (not produced by run_kmeans) is put far away
        centres[j - 1] = X[sel].mean(axis=0) if sel.any() else 1e6
    return {"kind": "kmeans", "centres": _floats(centres)}


def apply_model(model: dict, X: np.ndarray) -> np.ndarray:
    """Values (n x outputs) of a stored PCA / FlowSOM / k-means model for the
    display-space rows of X (columns in the order of the model's parameters)."""
    X = np.asarray(X, dtype=np.float64)
    kind = model.get("kind")
    if kind == "pca":
        mean = np.asarray(model["mean"], dtype=np.float64)
        comps = np.atleast_2d(np.asarray(model["components"], dtype=np.float64))
        if comps.shape[1] != X.shape[1]:
            raise ValueError(f"the PCA model has {comps.shape[1]} parameters, the data {X.shape[1]}")
        return (X - mean) @ comps.T
    if kind == "flowsom":
        codes = np.asarray(model["som_codes"], dtype=np.float64)
        meta = np.asarray(model["metacluster_of_node"], dtype=np.int64)
        if codes.shape[1] != X.shape[1]:
            raise ValueError(f"the FlowSOM model has {codes.shape[1]} parameters, the data {X.shape[1]}")
        return meta[_sq_dist_argmin(X, codes)].astype(np.float64)[:, None]
    if kind == "kmeans":
        centres = np.asarray(model["centres"], dtype=np.float64)
        if centres.shape[1] != X.shape[1]:
            raise ValueError(f"the k-means model has {centres.shape[1]} parameters, the data {X.shape[1]}")
        return (_sq_dist_argmin(X, centres) + 1).astype(np.float64)[:, None]
    raise ValueError(f"cannot apply a model of kind {kind!r}")


# ---------------------------------------------------------------- phenograph
def knn_jaccard_graph(X: np.ndarray, k: int = 30, should_cancel=None):
    """Phenograph graph: each event is linked to its k nearest neighbours
    (Euclidean, itself excluded) with weight |N(i) ∩ N(j)| / |N(i) ∪ N(j)|
    (Jaccard index of the two neighbour sets); the directed graph is
    symmetrised as (W + Wᵀ) / 2. Returns a scipy.sparse CSR matrix."""
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return sp.csr_matrix((n, n))
    k = int(max(1, min(k, n - 1)))
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    # drop each event itself (with duplicate events it need not come first)
    own = idx == np.arange(n)[:, None]
    keep = ~own
    keep[~own.any(axis=1), -1] = False
    idx = idx[keep].reshape(n, k)
    # shared-neighbour counts of every edge i -> j: mark N(i) in a boolean
    # row, then gather the marks at N(j); chunks keep the mask at ~20 MB
    shared = np.empty((n, k), dtype=np.float64)
    chunk = max(1, min(n, int(2e7 // n)))
    mark = np.zeros((chunk, n), dtype=bool)
    for start in range(0, n, chunk):
        _check_cancel(should_cancel)
        stop = min(n, start + chunk)
        rows = np.arange(stop - start)[:, None]
        nb = idx[start:stop]
        mark[rows, nb] = True
        shared[start:stop] = mark[rows[:, :, None], idx[nb]].sum(axis=2)
        mark[rows, nb] = False
    weights = shared / (2.0 * k - shared)
    W = sp.csr_matrix((weights.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    W = ((W + W.T) * 0.5).tocsr()
    W.eliminate_zeros()
    return W


def _louvain_local_moving(W, resolution: float, rng, should_cancel=None) -> tuple[np.ndarray, int]:
    """One Louvain level: starting from singletons, move nodes to the
    neighbouring community with the largest modularity gain until no move
    improves modularity. Nodes are visited in random order from a queue; a
    node that moves re-queues its neighbours outside its new community (the
    "fast local moving" bookkeeping). That alone can miss nodes whose stay
    gain dropped because their community grew, so whenever the queue runs
    empty after moves, every node is queued again for a full sweep; the phase
    ends after a full sweep without moves, i.e. at a node-level local optimum
    (no single move to a neighbouring community raises modularity), or when
    the visit budget runs out. Returns (community per node, number of moves)."""
    n = W.shape[0]
    indptr = W.indptr.tolist()
    indices = W.indices.tolist()
    data = W.data.tolist()
    degree_arr = np.asarray(W.sum(axis=1), dtype=np.float64).ravel()
    m2 = float(degree_arr.sum())  # 2m
    comm = list(range(n))
    if m2 <= 0:
        return np.asarray(comm, dtype=np.int64), 0
    degree = degree_arr.tolist()
    tot = list(degree)  # total degree of each community
    queue = deque(rng.permutation(n).tolist())
    queued = [True] * n
    moves = 0
    sweep_start = 0  # moves made before the current full sweep was queued
    budget = 200 * n + 1000  # safety net; convergence needs far fewer visits
    while budget > 0:
        if not queue:
            if moves == sweep_start:
                break  # a full sweep without moves: local optimum
            sweep_start = moves
            queue.extend(rng.permutation(n).tolist())
            queued = [True] * n
        budget -= 1
        if should_cancel is not None and budget % 4096 == 0:
            _check_cancel(should_cancel)
        i = queue.popleft()
        queued[i] = False
        ki = degree[i]
        if ki <= 0:
            continue
        ci = comm[i]
        links: dict[int, float] = {}
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if j != i:
                c = comm[j]
                links[c] = links.get(c, 0.0) + data[p]
        tot[ci] -= ki
        # gain of joining c (times m): k_i,c - resolution * tot_c * k_i / 2m
        f = resolution * ki / m2
        best, best_gain = ci, links.get(ci, 0.0) - f * tot[ci]
        tol = 1e-12 * ki
        for c, w in links.items():
            gain = w - f * tot[c]
            if gain > best_gain + tol:
                best, best_gain = c, gain
        tot[best] += ki
        if best != ci:
            comm[i] = best
            moves += 1
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if not queued[j] and comm[j] != best:
                    queued[j] = True
                    queue.append(j)
    return np.asarray(comm, dtype=np.int64), moves


def louvain(W, resolution: float = 1.0, seed: int = 0, should_cancel=None) -> np.ndarray:
    """Louvain community detection on a weighted undirected graph.

    `W` is a (scipy.sparse or dense) non-negative adjacency matrix; it is
    symmetrised as (W + Wᵀ) / 2. Alternates the local moving phase with the
    aggregation of each community into one node (W' = Pᵀ W P, internal weight
    kept as a self-loop) until a level moves no node. The modularity
    Q = 1/2m Σ (A_ij − γ k_i k_j / 2m) δ(c_i, c_j) uses the resolution γ;
    lower values give fewer, larger communities. Deterministic for a given
    seed (which only sets the node visiting order). Every level ends at a
    node-level local optimum (see `_louvain_local_moving`). `should_cancel`
    is polled between levels and during the moves. Returns community ids
    1..m numbered by decreasing size.
    """
    import scipy.sparse as sp

    W = sp.csr_matrix(W, dtype=np.float64)
    n = W.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    W = ((W + W.T) * 0.5).tocsr()
    W.sum_duplicates()
    rng = np.random.default_rng(seed)
    labels = np.arange(n)
    for _level in range(100):
        _check_cancel(should_cancel)
        comm, moves = _louvain_local_moving(W, float(resolution), rng, should_cancel)
        uniq, comm = np.unique(comm, return_inverse=True)
        if moves == 0 or len(uniq) == W.shape[0]:
            break
        labels = comm[labels]
        P = sp.csr_matrix((np.ones(W.shape[0]), (np.arange(W.shape[0]), comm)), shape=(W.shape[0], len(uniq)))
        W = (P.T @ W @ P).tocsr()
    return _renumber(labels)


def modularity(W, labels, resolution: float = 1.0) -> float:
    """Newman modularity of a partition (see `louvain`)."""
    import scipy.sparse as sp

    W = sp.csr_matrix(W, dtype=np.float64)
    W = ((W + W.T) * 0.5).tocoo()
    labels = np.asarray(labels)
    _, labels = np.unique(labels, return_inverse=True)
    deg = np.bincount(W.row, weights=W.data, minlength=W.shape[0])
    m2 = float(deg.sum())
    if m2 <= 0:
        return 0.0
    inside = float(W.data[labels[W.row] == labels[W.col]].sum())
    tot = np.bincount(labels, weights=deg)
    return inside / m2 - float(resolution) * float((tot**2).sum()) / (m2 * m2)


def run_phenograph(X: np.ndarray, k: int = 30, seed: int = 0, resolution: float = 1.0, should_cancel=None) -> np.ndarray:
    """Phenograph clustering (Levine et al. 2015): Louvain communities of the
    Jaccard-weighted k-nearest-neighbour graph. Cluster ids 1..m, largest
    first. Modularity tends to split compact Gaussian-like populations into
    several communities at resolution 1; lower the resolution for coarser
    clusters. `should_cancel` is polled while the graph is built, between the
    Louvain levels and during the moves."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3:
        return np.ones(X.shape[0], dtype=np.int64)
    W = knn_jaccard_graph(X, k, should_cancel=should_cancel)
    return louvain(W, resolution, seed, should_cancel=should_cancel)


def cluster_medians(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cluster ids, medians [clusters x params], counts) of a value matrix."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    ids = np.unique(labels[np.isfinite(labels)] if labels.dtype.kind == "f" else labels)
    med = np.full((len(ids), values.shape[1]), np.nan)
    counts = np.zeros(len(ids), dtype=np.int64)
    for i, k in enumerate(ids):
        sel = labels == k
        counts[i] = int(sel.sum())
        if counts[i]:
            med[i] = np.nanmedian(values[sel], axis=0)
    return ids.astype(np.int64), med, counts


# ---------------------------------------------------------------- comparison
class ComparisonRow(NamedTuple):
    param: str
    chi_squared_T: float
    ks_statistic: float
    difference_score: float


def chi_squared_distance(a: np.ndarray, b: np.ndarray, bins: int = 256, value_range: tuple[float, float] | None = None) -> tuple[float, float]:
    """(chi-squared distance, T(x)) between two univariate distributions.

    Both are binned into `bins` equal-width bins and normalised to unit area;
    the distance is chi' = sum((p - q)^2 / (p + q)). Following Roederer et al.
    (probability binning), T(x) = max(0, (chi' - chi_min) / sigma) with
    chi_min = B / min(nA, nB) and sigma = sqrt(B) / min(nA, nB), B being the
    number of occupied bins; T(x) above ~4 indicates a significant difference.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    if value_range is None:
        lo = min(a.min(), b.min(), 0.0)
        hi = max(a.max(), b.max(), 1.0)
        value_range = (lo, hi if hi > lo else lo + 1.0)
    p, _ = np.histogram(a, bins=bins, range=value_range)
    q, _ = np.histogram(b, bins=bins, range=value_range)
    p = p / p.sum()
    q = q / q.sum()
    s = p + q
    occ = s > 0
    chi = float(((p[occ] - q[occ]) ** 2 / s[occ]).sum())
    n = float(min(a.size, b.size))
    B = float(occ.sum())
    chi_min = B / n
    sigma = np.sqrt(B) / n
    T = max(0.0, (chi - chi_min) / sigma) if sigma > 0 else 0.0
    return chi, float(T)


def compare_populations(a_values: dict, b_values: dict, bins: int = 256) -> list[ComparisonRow]:
    """Per-parameter comparison of two populations (display-space arrays).

    Returns ComparisonRow(param, chi_squared_T, ks_statistic, difference_score)
    sorted by decreasing difference_score, which is the mean of the KS
    statistic and the half chi-squared distance (both 0..1).
    """
    from scipy.stats import ks_2samp

    rows: list[ComparisonRow] = []
    for param, a in a_values.items():
        if param not in b_values:
            continue
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b_values[param], dtype=np.float64)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            rows.append(ComparisonRow(param, float("nan"), float("nan"), float("nan")))
            continue
        chi, T = chi_squared_distance(a, b, bins)
        ks = float(ks_2samp(a, b).statistic)
        rows.append(ComparisonRow(param, T, ks, 0.5 * (ks + chi / 2.0)))
    rows.sort(key=lambda r: (-r.difference_score if np.isfinite(r.difference_score) else np.inf))
    return rows


def population_values(population, params: Sequence[str]) -> dict:
    """Display-space values of a population for the given parameters."""
    sample = population.sample
    mask = population.mask()
    available = set(sample.parameter_names())
    return {p: np.asarray(sample.display(p))[mask] for p in params if p in available}


# ---------------------------------------------------------------- write back
def _sample_map(workspace) -> dict:
    return {s.id: s for s in workspace.samples}


def write_back(workspace, index: list[IndexEntry], columns: dict, info: dict | None = None) -> list:
    """Store result columns (rows ordered as in `index`) as computed parameters
    on every involved sample; events not processed get NaN. Returns the
    samples that received the parameters."""
    by_id = _sample_map(workspace)
    total = index_size(index)
    columns = {name: np.asarray(v) for name, v in columns.items()}
    for name, v in columns.items():
        if v.shape[0] != total:
            raise ValueError(f"column {name} has {v.shape[0]} values, the index {total}")
    # one array per (sample, parameter); a sample may occur several times
    arrays: dict[int, dict[str, np.ndarray]] = {}
    offset = 0
    order: list[int] = []
    for sid, rows in index:
        sample = by_id.get(sid)
        n = len(rows)
        if sample is not None and sample.ensure_loaded():
            if sid not in arrays:
                arrays[sid] = {name: np.full(sample.n_events, np.nan, dtype=np.float32) for name in columns}
                order.append(sid)
            for name, v in columns.items():
                arrays[sid][name][rows] = v[offset:offset + n]
        offset += n
    touched = []
    for sid in order:
        sample = by_id[sid]
        for name, arr in arrays[sid].items():
            sample.add_computed(name, arr, dict(info or {}))
        touched.append(sample)
    return touched


def clusters_to_populations(workspace, group, parent_path: str, param_name: str, labels_present, prefix: str) -> list[str]:
    """Create one population per cluster id k, gated as
    RectangleGate([param_name], [k - 0.5], [k + 0.5]) and named f"{prefix} {k}",
    under `parent_path`. With a group, the populations are applied to the
    group (template and members); otherwise to every sample carrying the
    computed parameter. Existing populations of the same name are replaced.
    Returns the created population names."""
    from .gates import RectangleGate
    from .model import Population

    ids = sorted({int(k) for k in np.asarray(list(labels_present)).ravel() if np.isfinite(k)})
    names = []
    targets = [] if group is not None else [s for s in workspace.samples if param_name in s.computed]
    for k in ids:
        name = f"{prefix} {k}"
        names.append(name)
        pop = Population(name, RectangleGate([param_name], [k - 0.5], [k + 0.5]))
        if group is not None:
            workspace.apply_population_to_group(group, parent_path, pop)
            continue
        for sample in targets:
            parent = sample.root.find(parent_path or "")
            if parent is None:
                continue
            old = parent.child_by_name(name)
            index = None
            clone = pop.clone(deep=True)
            if old is not None:
                index = parent.children.index(old)
                parent.remove_child(old)
            parent.add_child(clone, index)
            clone.invalidate()
    workspace.modified = True
    return names


def unique_names(workspace, bases: Sequence[str]) -> list[str]:
    """`bases` unchanged if no sample has any of them, else with a number
    inserted before the trailing _<n> (tSNE_1 -> tSNE2_1) or appended."""
    existing: set[str] = set()
    for s in workspace.samples:
        existing.update(s.computed.keys())
        existing.update(s.derived.keys())
        existing.update(s.channel_names)

    def variant(base: str, n: int) -> str:
        if n == 1:
            return base
        stem, sep, tail = base.rpartition("_")
        if sep and tail.isdigit():
            return f"{stem}{n}_{tail}"
        return f"{base}{n}"

    n = 1
    while any(variant(b, n) in existing for b in bases):
        n += 1
    return [variant(b, n) for b in bases]


# ----------------------------------------------------- stored results / projection
_KIND_BY_TOOL = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap", "FlowSOM": "flowsom", "k-means": "kmeans", "Phenograph": "phenograph"}
MODEL_KINDS = ("pca", "flowsom", "kmeans")  # projectable from the saved model
SESSION_KINDS = ("tsne", "umap")  # projectable only with the in-memory model
CLUSTER_KINDS = ("flowsom", "kmeans", "phenograph")

# Fitted t-SNE (openTSNE) / UMAP objects cannot be serialised cheaply, so they
# live here for the lifetime of the workspace object only:
# workspace -> {parameter name: (run id, fitted model)}.
_SESSION_MODELS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _session_store(workspace, create: bool = False):
    try:
        if create:
            return _SESSION_MODELS.setdefault(workspace, {})
        return _SESSION_MODELS.get(workspace)
    except TypeError:  # not weak-referenceable
        return None


def register_session_model(workspace, names: Sequence[str], run_id: str, model) -> None:
    """Keep a fitted embedding (an object with `transform`) for the computed
    parameters `names` of `workspace`, for this session only."""
    store = _session_store(workspace, create=True)
    if store is None:
        return
    alive = set(names)
    for s in workspace.samples:
        alive.update(s.computed.keys())
    for stale in [n for n in store if n not in alive]:
        del store[stale]
    for n in names:
        store[n] = (run_id, model)


def session_model(workspace, name: str):
    """The in-memory fitted model of computed parameter `name`, or None (other
    session, workspace reloaded, or the parameter was recomputed)."""
    store = _session_store(workspace)
    entry = store.get(name) if store else None
    if entry is None:
        return None
    run_id, model = entry
    for s in workspace.samples:
        if name in s.computed:
            info = s.computed_info.get(name) or {}
            return model if info.get("run") == run_id else None
    return None


def write_result(workspace, index: list[IndexEntry], columns: dict, info: dict | None = None, model: dict | None = None, session=None) -> list:
    """`write_back` plus what projection needs.

    All `columns` share a run id (they are one result, e.g. tSNE_1 / tSNE_2)
    stored with the output names in every parameter's computed_info. `model`
    is a JSON-friendly dict (see fit_pca / flowsom_model / kmeans_model) saved
    as computed_info["model"] together with the input parameters; `session`
    is an unserialisable fitted object (openTSNE embedding, umap.UMAP) kept in
    memory for this workspace. Returns the samples written."""
    info = dict(info or {})
    run_id = uuid.uuid4().hex[:12]
    info["run"] = run_id
    info["outputs"] = list(columns)
    if model is not None:
        model = dict(model)
        if info.get("parameters"):
            model.setdefault("parameters", list(info["parameters"]))
        info["model"] = model
    touched = write_back(workspace, index, columns, info)
    if session is not None:
        register_session_model(workspace, list(columns), run_id, session)
    return touched


@dataclass
class ComputedResult:
    """One run of a tool: its output parameters and where they exist."""

    key: str
    tool: str
    kind: str
    outputs: list[str]
    parameters: list[str]  # input parameters (display space)
    samples: list  # samples carrying the result (analysed or projected)
    missing: list  # samples without it that have the input parameters
    populations: list[str]  # population paths the tool analysed
    info: dict
    source: object  # a sample the tool was run on
    projectable: bool = False
    reason: str = ""  # why it cannot be projected

    @property
    def label(self) -> str:
        return f"{self.tool}: {' / '.join(self.outputs)}"

    @property
    def is_clustering(self) -> bool:
        return self.kind in CLUSTER_KINDS


def _has_parameters(sample, params: Sequence[str]) -> bool:
    if not params:
        return False
    available = set(sample.parameter_names())
    return all(p in available for p in params)


def _projectable(workspace, r: ComputedResult) -> tuple[bool, str]:
    if not r.parameters:
        return False, "the input parameters of this result were not recorded"
    model = r.info.get("model") or {}
    if r.kind in MODEL_KINDS:
        if model.get("kind") == r.kind:
            return True, ""
        return False, f"this {r.tool} result has no stored model (older or imported workspace); re-run {r.tool} to enable projection"
    if r.kind in SESSION_KINDS:
        obj = session_model(workspace, r.outputs[0])
        if obj is not None and hasattr(obj, "transform"):
            return True, ""
        if r.kind == "tsne" and model.get("engine") == "scikit-learn":
            return False, "scikit-learn's t-SNE cannot place new events; install openTSNE and re-run t-SNE"
        return False, f"{r.tool} results can be projected only in the session that computed them (the fitted embedding is not saved with the workspace); re-run {r.tool} including the new samples"
    if r.kind == "phenograph":
        return False, "Phenograph clusters are communities of the k-nearest-neighbour graph of the analysed events and cannot be extended to new events; re-run Phenograph including the new samples"
    return False, "this result cannot be projected"


def computed_results(workspace) -> list[ComputedResult]:
    """The computed results of a workspace, one entry per run (parameters
    written together, e.g. tSNE_1 and tSNE_2), with the samples that carry
    them, the samples that lack them and whether they can be projected.
    Parameters written without a run id (older workspaces, imports) are
    grouped by tool and name stem."""
    groups: dict[str, dict] = {}
    for s in workspace.samples:
        for name in s.computed:
            info = s.computed_info.get(name) or {}
            run = info.get("run")
            if run:
                key = f"run:{run}"
            else:
                stem, sep, tail = name.rpartition("_")
                key = f"legacy:{info.get('tool', '')}:{stem if sep and tail.isdigit() else name}"
            g = groups.get(key)
            if g is None:
                outputs = [str(o) for o in info.get("outputs") or []] if run else []
                g = groups[key] = {"info": info, "source": s, "outputs": outputs, "samples": []}
            if name not in g["outputs"] and not (run and info.get("outputs")):
                g["outputs"].append(name)
            if not any(x is s for x in g["samples"]):
                g["samples"].append(s)
            if g["info"].get("projected") and not info.get("projected"):
                g["info"], g["source"] = info, s  # prefer an analysed sample
    results = []
    for key, g in groups.items():
        info = g["info"]
        model = info.get("model") or {}
        tool = str(info.get("tool") or g["outputs"][0])
        kind = str(model.get("kind") or _KIND_BY_TOOL.get(tool, ""))
        params = [str(p) for p in info.get("parameters") or model.get("parameters") or []]
        have = {id(s) for s in g["samples"]}
        missing = [s for s in workspace.samples if id(s) not in have and _has_parameters(s, params)]
        pops = [str(p) for p in info.get("populations") or []]
        r = ComputedResult(key, tool, kind, list(g["outputs"]), params, list(g["samples"]), missing, pops, info, g["source"])
        r.projectable, r.reason = _projectable(workspace, r)
        results.append(r)
    return results


def find_result(workspace, param_name: str) -> ComputedResult | None:
    """The result one of whose outputs is `param_name`."""
    for r in computed_results(workspace):
        if param_name in r.outputs:
            return r
    return None


@dataclass
class Projection:
    """Projected values of new samples, ready for `apply_projection`."""

    result: ComputedResult
    targets: list
    population_path: str
    index: list
    columns: dict
    info: dict
    cluster_ids: list[int] = field(default_factory=list)
    created: list[str] = field(default_factory=list)  # cluster populations

    @property
    def outputs(self) -> list[str]:
        return list(self.columns)


def compute_projection(workspace, source_param_name: str, target_samples, max_events: int | None = None, seed: int = 0, population_path: str | None = None, progress: ProgressFn | None = None, should_cancel=None) -> Projection:
    """Place the events of `target_samples` on an existing result (no changes
    to the workspace; see apply_projection).

    PCA: the stored mean and components; FlowSOM: nearest stored SOM node and
    its metacluster; k-means: nearest stored centre; t-SNE / UMAP: the fitted
    embedding's `transform` (this session only). The input is the targets'
    display-space values of the result's parameters in `population_path`
    (default: the population the result was computed on when unique, else
    Ungated), subsampled to `max_events` per sample. Samples the tool was run
    on are skipped."""
    result = find_result(workspace, source_param_name)
    if result is None:
        raise ValueError(f"no sample has the computed parameter {source_param_name}")
    if not result.projectable:
        raise ValueError(result.reason)
    analysed = {id(s) for s in result.samples if not (s.computed_info.get(result.outputs[0]) or {}).get("projected")}
    targets = [s for s in target_samples if id(s) not in analysed]
    if not targets:
        raise ValueError("choose samples that were not part of the original analysis")
    if population_path is None:
        paths = list(dict.fromkeys(result.populations))
        population_path = paths[0] if len(paths) == 1 else ""
    population_path = population_path or ""
    lacking = [s.name for s in targets if s.root.find(population_path) is None]
    if lacking:
        raise ValueError(f"{', '.join(lacking)}: no population {population_path}")
    if progress:
        progress("Collecting events…")
    X, index = prepare_matrix(targets, population_path, result.parameters, max_events, seed)
    _check_cancel(should_cancel)
    if progress:
        progress(f"Projecting {X.shape[0]:,} events onto {result.label}…")
    model = result.info.get("model") or {}
    if result.kind in MODEL_KINDS:
        Y = apply_model(model, X)
    else:
        obj = session_model(workspace, result.outputs[0])
        if obj is None:
            raise ValueError(f"the fitted {result.tool} model is no longer available")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = np.asarray(obj.transform(X), dtype=np.float64)
    _check_cancel(should_cancel)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != X.shape[0] or Y.shape[1] < len(result.outputs):
        raise ValueError(f"the {result.tool} model returned {Y.shape} values for {X.shape[0]} events")
    columns = {name: Y[:, j] for j, name in enumerate(result.outputs)}
    info = dict(result.info)
    info.update(projected=True, projected_from=getattr(result.source, "name", ""), events=index_size(index), population=population_path)
    ids: list[int] = []
    if result.kind == "flowsom":
        ids = sorted({int(m) for m in model["metacluster_of_node"]})
    elif result.kind == "kmeans":
        ids = list(range(1, len(model["centres"]) + 1))
    return Projection(result, targets, population_path, index, columns, info, ids)


def apply_projection(workspace, projection: Projection) -> list:
    """Store a projection: the parameters on the target samples (same names,
    same run, the source sample's axis scales) and, for clusterings, the
    cluster populations "<param> <k>" under the projected population (an
    existing population of that name gets the gate and keeps its children).
    Returns the samples written."""
    from .gates import RectangleGate
    from .model import Population

    result = projection.result
    touched = write_back(workspace, projection.index, projection.columns, projection.info)
    source = result.source
    for name in projection.columns:
        if source is not None and name in getattr(source, "computed", {}):
            scale = source.scale(name)
            for t in touched:
                t.set_scale(name, scale.copy())
    created: list[str] = []
    if projection.cluster_ids:
        param = result.outputs[0]
        created = [f"{param} {k}" for k in projection.cluster_ids]
        for t in touched:
            parent = t.root.find(projection.population_path)
            if parent is None:
                continue
            for k, name in zip(projection.cluster_ids, created):
                gate = RectangleGate([param], [k - 0.5], [k + 0.5])
                pop = parent.child_by_name(name)
                if pop is None:
                    parent.add_child(Population(name, gate))
                else:
                    pop.gate = gate
                    pop.invalidate()
    projection.created = created
    workspace.modified = True
    return touched


def project_onto(workspace, source_param_name: str, target_samples, max_events: int | None = None, seed: int = 0, population_path: str | None = None) -> Projection:
    """Project `target_samples` onto the existing result that produced
    `source_param_name` and store the projected parameters (and cluster
    populations) on them. PCA, FlowSOM and k-means results can always be
    projected (their model is saved with the workspace); t-SNE (openTSNE) and
    UMAP only within the session that computed them."""
    projection = compute_projection(workspace, source_param_name, target_samples, max_events, seed, population_path)
    apply_projection(workspace, projection)
    return projection
