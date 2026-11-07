"""Dimensionality reduction, clustering and population comparison.

Pure numpy / scipy / scikit-learn computations (no Qt). The usual flow is

    X, index = prepare_matrix(samples, paths, params, 5000, seed)
    emb = run_tsne(X)                       # or run_pca / run_umap
    write_back(workspace, index, {"tSNE_1": emb[:, 0], "tSNE_2": emb[:, 1]}, info)

`index` records which event rows of which sample produced each row of X, so
results can be stored as computed parameters (NaN for events that were not
part of the subsample).

Optional dependencies: openTSNE (faster t-SNE; falls back to
sklearn.manifold.TSNE) and umap-learn (required for UMAP).
"""
from __future__ import annotations

import importlib.util
from typing import Callable, NamedTuple, Sequence

import numpy as np

IndexEntry = tuple[int, np.ndarray]  # (sample id, event row indices)


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
def prepare_matrix(samples: Sequence, population_paths, params: Sequence[str], max_events_per_sample: int | None = None, seed: int = 0) -> tuple[np.ndarray, list[IndexEntry]]:
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
def run_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Principal component scores (via SVD of the centred matrix)."""
    X = np.asarray(X, dtype=np.float64)
    n_components = max(1, min(n_components, X.shape[1], X.shape[0]))
    centred = X - X.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    # deterministic sign: largest loading of each component positive
    signs = np.sign(vt[np.arange(vt.shape[0]), np.argmax(np.abs(vt), axis=1)])
    signs[signs == 0] = 1
    return (u[:, :n_components] * s[:n_components]) * signs[:n_components]


def pca_explained_variance(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    var = s**2
    total = var.sum() or 1.0
    return var[:n_components] / total


def run_tsne(X: np.ndarray, perplexity: float = 30, iterations: int = 500, seed: int = 0) -> np.ndarray:
    """2-D t-SNE embedding (openTSNE when installed, else scikit-learn)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 5:
        raise ValueError("t-SNE needs at least 5 events")
    perplexity = float(max(2.0, min(perplexity, (n - 1) / 3.0)))
    iterations = int(max(250, iterations))
    if has_opentsne():
        from openTSNE import TSNE

        emb = TSNE(n_components=2, perplexity=perplexity, n_iter=iterations, random_state=seed, n_jobs=-1, verbose=False).fit(X)
        return np.asarray(emb, dtype=np.float64)
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("t-SNE requires openTSNE or scikit-learn (pip install 'flowdesk[analysis]')") from exc
    kwargs = dict(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    try:
        model = TSNE(max_iter=iterations, **kwargs)
    except TypeError:  # scikit-learn < 1.5
        model = TSNE(n_iter=iterations, **kwargs)
    return np.asarray(model.fit_transform(X), dtype=np.float64)


def run_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0) -> np.ndarray:
    """2-D UMAP embedding (requires umap-learn)."""
    if not has_umap():
        raise ImportError("UMAP requires the optional package umap-learn (pip install umap-learn, or pip install 'flowdesk[analysis]')")
    import umap

    X = np.asarray(X, dtype=np.float64)
    n_neighbors = int(max(2, min(n_neighbors, X.shape[0] - 1)))
    import warnings

    model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=float(min_dist), random_state=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_jobs.*", category=UserWarning)
        return np.asarray(model.fit_transform(X), dtype=np.float64)


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


def _kmeans_numpy(X: np.ndarray, k: int, seed: int, n_init: int = 5, max_iter: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
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


def run_kmeans(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """k-means cluster ids 1..k (one per row of X)."""
    X = np.asarray(X, dtype=np.float64)
    k = int(max(1, min(k, X.shape[0])))
    if has_sklearn():
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    else:
        labels = _kmeans_numpy(X, k, seed)
    return _renumber(np.asarray(labels))


def _renumber(labels: np.ndarray) -> np.ndarray:
    """Map labels to 1..m, ordered by decreasing cluster size."""
    uniq, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    rank = np.empty(len(uniq), dtype=np.int64)
    rank[np.argsort(-counts, kind="stable")] = np.arange(1, len(uniq) + 1)
    return rank[inverse.ravel()]


def train_som(X: np.ndarray, grid: tuple[int, int] = (10, 10), iterations: int = 10, seed: int = 0, sigma_start: float | None = None, sigma_end: float = 0.5) -> np.ndarray:
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


def consensus_metaclusters(codes: np.ndarray, k: int, reps: int = 20, fraction: float = 0.9, seed: int = 0) -> np.ndarray:
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


def run_flowsom(X: np.ndarray, grid: tuple[int, int] = (10, 10), n_metaclusters: int = 10, iterations: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FlowSOM: SOM + consensus hierarchical metaclustering.

    Returns (labels, som_codes, metacluster_of_node) where labels holds the
    metacluster id (1..k, largest first) of every row of X and
    metacluster_of_node the id of each SOM node.
    """
    X = np.asarray(X, dtype=np.float64)
    codes = train_som(X, grid, iterations, seed)
    meta = consensus_metaclusters(codes, n_metaclusters, seed=seed)
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


ProgressFn = Callable[[str], None]
