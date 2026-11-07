"""Tools ▸ Dimensionality Reduction / Clustering / Compare.

Dialogs for t-SNE, UMAP, PCA, k-means, FlowSOM and population comparison,
plus `install(window)` that adds them to the Tools ribbon page and menu.

Every computation is also available without Qt widgets through
`run_dimreduction`, `run_clustering` and `compare_population_objects`
(they accept a WorkspaceController or a bare Workspace).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import analysis as A
from ..core.model import Population, Workspace
from .icons import icon

DIM_METHODS = {
    "pca": ("PCA", ("PC_1", "PC_2")),
    "tsne": ("t-SNE", ("tSNE_1", "tSNE_2")),
    "umap": ("UMAP", ("UMAP_1", "UMAP_2")),
}
CLUSTER_METHODS = {"kmeans": ("k-means", "kmeans"), "flowsom": ("FlowSOM", "FlowSOM")}


class Cancelled(Exception):
    pass


# ============================================================== helpers (no Qt)
def _split(target) -> tuple[object | None, Workspace]:
    """(controller or None, workspace) from a controller or a workspace."""
    if isinstance(target, Workspace):
        return None, target
    return target, target.workspace


def default_populations(target) -> list[Population]:
    """Selected populations, else Ungated of every sample in the current group
    (all samples without a controller)."""
    controller, ws = _split(target)
    if controller is not None:
        pops = [p for p in controller.selected_populations() if p.sample is not None]
        if pops:
            return pops
        return [s.root for s in ws.members(controller.current_group)]
    return [s.root for s in ws.samples]


def common_parameters(populations: list[Population]) -> list[str]:
    names: list[str] | None = None
    for pop in populations:
        sample = pop.sample
        sample.ensure_loaded()
        own = sample.parameter_names()
        names = list(own) if names is None else [n for n in names if n in set(own)]
    return names or []


def default_parameters(populations: list[Population]) -> list[str]:
    """Compensated parameters if any, else everything except scatter, Time and
    computed parameters."""
    names = common_parameters(populations)
    computed: set[str] = set()
    for pop in populations:
        computed.update(pop.sample.computed.keys())
    names = [n for n in names if n not in computed]
    comp = [n for n in names if n.startswith("Comp-")]
    if comp:
        return comp
    out = []
    for n in names:
        u = n.upper()
        if u.startswith("FSC") or u.startswith("SSC") or u.startswith("TIME") or u == "SAMPLEID":
            continue
        out.append(n)
    return out


def _check(should_cancel):
    if should_cancel is not None and should_cancel():
        raise Cancelled()


@dataclass
class DimResult:
    method: str
    params: list[str]
    index: list
    embedding: np.ndarray
    populations: list[Population]
    options: dict = field(default_factory=dict)


@dataclass
class ClusterResult:
    method: str
    params: list[str]
    index: list
    labels: np.ndarray
    X: np.ndarray
    populations: list[Population]
    options: dict = field(default_factory=dict)


def compute_dimreduction(populations, params, method="tsne", max_events=5000, options=None, seed=0, progress=None, should_cancel=None) -> DimResult:
    options = dict(options or {})
    if progress:
        progress("Collecting events…")
    X, index = A.prepare_matrix([p.sample for p in populations], [p.path for p in populations], params, max_events, seed)
    _check(should_cancel)
    if progress:
        progress(f"Running {DIM_METHODS[method][0]} on {X.shape[0]} events…")
    if method == "pca":
        emb = A.run_pca(X, 2)
        if emb.shape[1] < 2:
            emb = np.column_stack([emb, np.zeros(emb.shape[0])])
        options["explained_variance"] = [float(v) for v in A.pca_explained_variance(X, 2)]
    elif method == "tsne":
        emb = A.run_tsne(X, options.get("perplexity", 30), options.get("iterations", 500), seed)
    elif method == "umap":
        emb = A.run_umap(X, options.get("n_neighbors", 15), options.get("min_dist", 0.1), seed)
    else:
        raise ValueError(f"unknown method {method}")
    _check(should_cancel)
    return DimResult(method, list(params), index, emb, list(populations), options)


def apply_dimreduction(target, result: DimResult) -> list[str]:
    """Write the embedding back as computed parameters; returns their names."""
    controller, ws = _split(target)
    names = A.unique_names(ws, DIM_METHODS[result.method][1])
    info = {"tool": DIM_METHODS[result.method][0], "parameters": list(result.params), "events": A.index_size(result.index), **{k: v for k, v in result.options.items()}}
    A.write_back(ws, result.index, {names[0]: result.embedding[:, 0], names[1]: result.embedding[:, 1]}, info)
    if controller is not None:
        controller.mark_modified()
        controller.gatingChanged.emit(None)
        controller.samplesChanged.emit()
        controller.statusMessage.emit(f"{DIM_METHODS[result.method][0]}: created {names[0]} / {names[1]} on {len({sid for sid, _ in result.index})} sample(s)")
    else:
        ws.modified = True
    return names


def open_result_graph(controller, population: Population, x: str, y: str | None):
    """Open a graph window of `population` showing x / y."""
    if controller is None or getattr(controller, "open_graph", None) is None:
        return None
    population.graph.x = x
    population.graph.y = y
    if population.graph.plot_type == "histogram" and y is not None:
        population.graph.plot_type = "pseudocolor"
    return controller.open_graph(population.sample, population)


def run_dimreduction(target, method="tsne", populations=None, params=None, max_events=5000, options=None, seed=0, open_graph=True) -> list[str]:
    """Synchronous t-SNE / UMAP / PCA: compute, write back and (with a
    controller) open a graph of the first population on the new axes."""
    populations = populations or default_populations(target)
    params = params or default_parameters(populations)
    result = compute_dimreduction(populations, params, method, max_events, options, seed)
    names = apply_dimreduction(target, result)
    controller, _ = _split(target)
    if open_graph and controller is not None:
        open_result_graph(controller, result.populations[0], names[0], names[1])
    return names


def compute_clustering(populations, params, method="flowsom", max_events=None, options=None, seed=0, progress=None, should_cancel=None) -> ClusterResult:
    options = dict(options or {})
    if progress:
        progress("Collecting events…")
    X, index = A.prepare_matrix([p.sample for p in populations], [p.path for p in populations], params, max_events, seed)
    _check(should_cancel)
    if progress:
        progress(f"Running {CLUSTER_METHODS[method][0]} on {X.shape[0]} events…")
    if method == "kmeans":
        labels = A.run_kmeans(X, int(options.get("k", 10)), seed)
    elif method == "flowsom":
        grid = tuple(options.get("grid", (10, 10)))
        labels, _codes, _meta = A.run_flowsom(X, grid, int(options.get("n_metaclusters", 10)), int(options.get("iterations", 10)), seed)
    else:
        raise ValueError(f"unknown method {method}")
    _check(should_cancel)
    return ClusterResult(method, list(params), index, np.asarray(labels), X, list(populations), options)


def _cluster_group(controller, ws, paths: list[str], populations):
    """The current group when every population lies in it and the parent
    path exists in its template (or is Ungated)."""
    if controller is None:
        return None
    group = controller.current_group
    if len(set(paths)) != 1:
        return None
    path = paths[0]
    if not all(group.contains(p.sample) for p in populations):
        return None
    if path and group.root.find(path) is None:
        return None
    return group


def apply_clustering(target, result: ClusterResult) -> tuple[str, list[str]]:
    """Write cluster ids as a computed parameter and create one population per
    cluster below the analysed population(s). Returns (param, population names)."""
    controller, ws = _split(target)
    name = A.unique_names(ws, [CLUSTER_METHODS[result.method][1]])[0]
    info = {"tool": CLUSTER_METHODS[result.method][0], "parameters": list(result.params), "events": A.index_size(result.index), "min": 0.0, "max": float(result.labels.max()) + 1.0}
    info.update({k: (list(v) if isinstance(v, tuple) else v) for k, v in result.options.items()})
    if controller is not None:
        controller.begin_change(f"{CLUSTER_METHODS[result.method][0]} clusters")
    A.write_back(ws, result.index, {name: result.labels.astype(np.float32)}, info)
    ids = np.unique(result.labels)
    paths = [p.path for p in result.populations]
    group = _cluster_group(controller, ws, paths, result.populations)
    created: list[str] = []
    if group is not None:
        created = A.clusters_to_populations(ws, group, paths[0], name, ids, name)
    else:
        for path in dict.fromkeys(paths):
            created = A.clusters_to_populations(ws, None, path, name, ids, name)
    if controller is not None:
        controller.mark_modified()
        controller.gatingChanged.emit(None)
        controller.groupsChanged.emit()
        controller.samplesChanged.emit()
        controller.statusMessage.emit(f"{CLUSTER_METHODS[result.method][0]}: {len(ids)} clusters in {name}")
    return name, created


def cluster_heatmap_data(target, result: ClusterResult) -> dict:
    """Medians per cluster × parameter: data units (for the table text) and
    display units 0..1 (for the colour)."""
    _, ws = _split(target)
    by_id = {s.id: s for s in ws.samples}
    raw = []
    for sid, rows in result.index:
        s = by_id[sid]
        raw.append(np.column_stack([np.asarray(s.data(p), dtype=np.float64)[rows] for p in result.params]))
    raw = np.vstack(raw)
    ids, med_raw, counts = A.cluster_medians(raw, result.labels)
    _, med_disp, _ = A.cluster_medians(result.X, result.labels)
    return {"ids": ids, "params": list(result.params), "medians": med_raw, "display": med_disp, "counts": counts}


def run_clustering(target, method="flowsom", populations=None, params=None, max_events=None, options=None, seed=0) -> tuple[str, list[str], ClusterResult]:
    populations = populations or default_populations(target)
    params = params or default_parameters(populations)
    result = compute_clustering(populations, params, method, max_events, options, seed)
    name, created = apply_clustering(target, result)
    return name, created, result


def compare_population_objects(pop_a: Population, pop_b: Population, params=None, bins=256) -> list:
    params = params or [n for n in common_parameters([pop_a, pop_b]) if not n.upper().startswith("TIME")]
    return A.compare_populations(A.population_values(pop_a, params), A.population_values(pop_b, params), bins)


def rows_to_text(headers, rows, delimiter="\t") -> str:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    w.writerow(headers)
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def _pop_label(pop: Population) -> str:
    return f"{pop.sample.name} : {pop.path or 'Ungated'}" if pop.sample else pop.display_name


# =================================================================== threading
class _Worker(QThread):
    """Runs the computation in its own thread.

    A QThread subclass owned by the dialog (not a parentless QObject moved to a
    thread): Qt deletes it with the dialog, so nothing is torn down from the
    wrong thread at interpreter shutdown.
    """

    done_result = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, fn, kwargs, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.kwargs = kwargs
        self.cancelled = False

    def run(self):
        try:
            result = self.fn(**self.kwargs, progress=self.progress.emit, should_cancel=lambda: self.cancelled)
        except Cancelled:
            self.failed.emit("")
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.done_result.emit(result)


# ===================================================================== dialogs
class _AnalysisDialog(QDialog):
    """Shared population / parameter selection and threaded execution."""

    title = "Analysis"

    def __init__(self, controller, method: str, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.method = method
        self.last_result = None
        self._thread: QThread | None = None
        self._worker: _Worker | None = None
        self._progress: QProgressDialog | None = None
        self.populations = default_populations(controller)
        layout = QVBoxLayout(self)

        box = QGroupBox("Populations")
        bl = QVBoxLayout(box)
        self.pop_list = QListWidget()
        for pop in self.populations:
            item = QListWidgetItem(f"{_pop_label(pop)}  ({pop.count:,} events)")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, pop)
            self.pop_list.addItem(item)
        self.pop_list.setMaximumHeight(110)
        bl.addWidget(self.pop_list)
        layout.addWidget(box)

        box = QGroupBox("Parameters")
        pl = QVBoxLayout(box)
        self.param_list = QListWidget()
        defaults = set(default_parameters(self.populations)) if self.populations else set()
        for name in common_parameters(self.populations) if self.populations else []:
            item = QListWidgetItem(self.populations[0].sample.param_label(name))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in defaults else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.param_list.addItem(item)
        pl.addWidget(self.param_list)
        row = QHBoxLayout()
        for text, state in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=state: self._check_all(s))
            row.addWidget(b)
        row.addStretch(1)
        pl.addLayout(row)
        layout.addWidget(box)

        self.form = QFormLayout()
        self.events_spin = QSpinBox()
        self.events_spin.setRange(0, 10_000_000)
        self.events_spin.setSingleStep(1000)
        self.events_spin.setSpecialValueText("All")
        self.form.addRow("Events per sample:", self.events_spin)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.form.addRow("Random seed:", self.seed_spin)
        layout.addLayout(self.form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.run_button = self.buttons.addButton("Run", QDialogButtonBox.ButtonRole.AcceptRole)
        self.run_button.setDefault(True)
        self.buttons.accepted.connect(self.start)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.resize(440, 560)
        if not self.populations:
            self.run_button.setEnabled(False)

    def _check_all(self, state: bool):
        for i in range(self.param_list.count()):
            self.param_list.item(i).setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def chosen_populations(self) -> list[Population]:
        out = []
        for i in range(self.pop_list.count()):
            it = self.pop_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def chosen_parameters(self) -> list[str]:
        out = []
        for i in range(self.param_list.count()):
            it = self.param_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    # subclasses
    def options(self) -> dict:
        return {}

    def compute_fn(self):
        raise NotImplementedError

    def compute_kwargs(self) -> dict:
        return {
            "populations": self.chosen_populations(),
            "params": self.chosen_parameters(),
            "method": self.method,
            "max_events": self.events_spin.value() or None,
            "options": self.options(),
            "seed": self.seed_spin.value(),
        }

    def apply(self, result):
        raise NotImplementedError

    def _validate(self, kwargs) -> bool:
        if not kwargs["populations"]:
            QMessageBox.warning(self, self.title, "Choose at least one population.")
            return False
        if len(kwargs["params"]) < 2:
            QMessageBox.warning(self, self.title, "Choose at least two parameters.")
            return False
        return True

    def run_now(self):
        """Run synchronously (no thread, no progress dialog)."""
        kwargs = self.compute_kwargs()
        result = self.compute_fn()(**kwargs)
        self.last_result = result
        self.apply(result)
        return result

    def start(self):
        kwargs = self.compute_kwargs()
        if not self._validate(kwargs):
            return
        self.run_button.setEnabled(False)
        self._worker = _Worker(self.compute_fn(), kwargs, self)
        self._thread = self._worker
        self._worker.done_result.connect(self._finished)
        self._worker.failed.connect(self._failed)
        self._worker.progress.connect(self._on_progress)
        self._progress = QProgressDialog("Starting…", "Cancel", 0, 0, self)
        self._progress.setWindowTitle(self.title)
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.canceled.connect(self._cancel)
        self._progress.show()
        self._thread.start()

    def _on_progress(self, text: str):
        if self._progress is not None:
            self._progress.setLabelText(text)

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancelled = True
            if self._progress is not None:
                self._progress.setLabelText("Cancelling after the current step…")

    def _close_progress(self):
        if self._progress is not None:
            self._progress.canceled.disconnect(self._cancel)
            self._progress.close()
            self._progress = None

    def _settle_thread(self):
        # the worker emits its result just before run() returns; make sure the
        # thread has fully stopped before anything else happens
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(10000)

    def _finished(self, result):
        self._settle_thread()
        cancelled = self._worker is not None and self._worker.cancelled
        self._close_progress()
        self.run_button.setEnabled(True)
        if cancelled:
            return
        self.last_result = result
        try:
            self.apply(result)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, self.title, f"Could not store the results:\n{exc}")
            return
        self.accept()

    def _failed(self, message: str):
        self._settle_thread()
        self._close_progress()
        self.run_button.setEnabled(True)
        if message:
            QMessageBox.warning(self, self.title, message)

    def done(self, r):
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.cancelled = True
            self._thread.quit()
            self._thread.wait(30000)
        super().done(r)


class DimReductionDialog(_AnalysisDialog):
    def __init__(self, controller, method: str = "tsne", parent=None):
        if method not in DIM_METHODS:
            raise ValueError(method)
        self.title = DIM_METHODS[method][0]
        super().__init__(controller, method, parent)
        self.setWindowTitle(self.title)
        self.events_spin.setValue(5000)
        self.perplexity = self.iterations = self.neighbors = self.min_dist = None
        if method == "tsne":
            self.perplexity = QDoubleSpinBox()
            self.perplexity.setRange(2, 500)
            self.perplexity.setValue(30)
            self.iterations = QSpinBox()
            self.iterations.setRange(250, 10000)
            self.iterations.setSingleStep(250)
            self.iterations.setValue(500)
            self.form.addRow("Perplexity:", self.perplexity)
            self.form.addRow("Iterations:", self.iterations)
            engine = "openTSNE" if A.has_opentsne() else "scikit-learn"
            self.form.addRow("Engine:", QLabel(engine))
        elif method == "umap":
            self.neighbors = QSpinBox()
            self.neighbors.setRange(2, 500)
            self.neighbors.setValue(15)
            self.min_dist = QDoubleSpinBox()
            self.min_dist.setRange(0.0, 1.0)
            self.min_dist.setSingleStep(0.05)
            self.min_dist.setDecimals(3)
            self.min_dist.setValue(0.1)
            self.form.addRow("Nearest neighbours:", self.neighbors)
            self.form.addRow("Minimum distance:", self.min_dist)
            if not A.has_umap():
                self.run_button.setEnabled(False)
                self.form.addRow(QLabel("umap-learn is not installed (pip install umap-learn)."))

    def options(self):
        if self.method == "tsne":
            return {"perplexity": self.perplexity.value(), "iterations": self.iterations.value()}
        if self.method == "umap":
            return {"n_neighbors": self.neighbors.value(), "min_dist": self.min_dist.value()}
        return {}

    def compute_fn(self):
        return compute_dimreduction

    def apply(self, result: DimResult):
        names = apply_dimreduction(self.controller, result)
        self.created = names
        open_result_graph(self.controller, result.populations[0], names[0], names[1])


class ClusteringDialog(_AnalysisDialog):
    def __init__(self, controller, method: str = "flowsom", parent=None):
        if method not in CLUSTER_METHODS:
            raise ValueError(method)
        self.title = CLUSTER_METHODS[method][0]
        super().__init__(controller, method, parent)
        self.setWindowTitle(self.title)
        self.events_spin.setValue(0)
        self.heatmap: ClusterHeatmapDialog | None = None
        self.k_spin = self.grid_x = self.grid_y = self.meta_spin = self.iter_spin = None
        if method == "kmeans":
            self.k_spin = QSpinBox()
            self.k_spin.setRange(2, 200)
            self.k_spin.setValue(10)
            self.form.addRow("Clusters (k):", self.k_spin)
        else:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            self.grid_x = QSpinBox()
            self.grid_y = QSpinBox()
            for s in (self.grid_x, self.grid_y):
                s.setRange(2, 50)
                s.setValue(10)
            rl.addWidget(self.grid_x)
            rl.addWidget(QLabel("×"))
            rl.addWidget(self.grid_y)
            self.form.addRow("SOM grid:", row)
            self.meta_spin = QSpinBox()
            self.meta_spin.setRange(2, 200)
            self.meta_spin.setValue(10)
            self.form.addRow("Metaclusters:", self.meta_spin)
            self.iter_spin = QSpinBox()
            self.iter_spin.setRange(1, 1000)
            self.iter_spin.setValue(10)
            self.form.addRow("Training iterations:", self.iter_spin)

    def options(self):
        if self.method == "kmeans":
            return {"k": self.k_spin.value()}
        return {"grid": (self.grid_x.value(), self.grid_y.value()), "n_metaclusters": self.meta_spin.value(), "iterations": self.iter_spin.value()}

    def compute_fn(self):
        return compute_clustering

    def apply(self, result: ClusterResult):
        name, created = apply_clustering(self.controller, result)
        self.created = (name, created)
        data = cluster_heatmap_data(self.controller, result)
        self.heatmap = ClusterHeatmapDialog(data, title=f"{name} – cluster medians", parent=self.parentWidget())
        self.heatmap.show()


def _tint(value: float) -> QColor:
    """White → blue → red ramp for 0..1."""
    if not np.isfinite(value):
        return QColor("#eeeeee")
    v = float(min(1.0, max(0.0, value)))
    if v < 0.5:
        t = v / 0.5
        a, b = (255, 255, 255), (120, 160, 230)
    else:
        t = (v - 0.5) / 0.5
        a, b = (120, 160, 230), (220, 60, 50)
    return QColor(*[int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)])


class _TableDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.layout_ = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.layout_.addWidget(self.table)
        self.button_row = QHBoxLayout()
        self.copy_button = QPushButton(icon("copy"), "Copy")
        self.csv_button = QPushButton(icon("export"), "Export CSV…")
        close = QPushButton("Close")
        self.copy_button.clicked.connect(self.copy)
        self.csv_button.clicked.connect(self.export_csv)
        close.clicked.connect(self.close)
        self.button_row.addWidget(self.copy_button)
        self.button_row.addWidget(self.csv_button)
        self.button_row.addStretch(1)
        self.button_row.addWidget(close)
        self.layout_.addLayout(self.button_row)
        self.headers: list[str] = []
        self.rows: list[list] = []

    def set_rows(self, headers, rows, colors=None):
        self.headers = list(headers)
        self.rows = [list(r) for r in rows]
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(headers)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                text = f"{value:.4g}" if isinstance(value, float) else str(value)
                item = QTableWidgetItem(text)
                if isinstance(value, (int, float)):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if colors is not None and colors[r][c] is not None:
                    item.setBackground(colors[r][c])
                self.table.setItem(r, c, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def text(self, delimiter="\t") -> str:
        return rows_to_text(self.headers, self.rows, delimiter)

    def copy(self):
        QGuiApplication.clipboard().setText(self.text())

    def export_csv(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{self.windowTitle()}.csv", "CSV files (*.csv)")
        if not path:
            return None
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(self.text(","))
        return path


class ClusterHeatmapDialog(_TableDialog):
    """Cluster × parameter medians; cells tinted by the per-parameter
    normalised display-space median."""

    def __init__(self, data: dict, title: str = "Cluster medians", parent=None):
        super().__init__(title, parent)
        self.data = data
        params = data["params"]
        headers = ["Cluster", "Count", "%"] + params
        total = float(data["counts"].sum()) or 1.0
        disp = np.asarray(data["display"], dtype=np.float64)
        lo = np.nanmin(disp, axis=0) if disp.size else disp
        hi = np.nanmax(disp, axis=0) if disp.size else disp
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        norm = (disp - lo) / span
        rows, colors = [], []
        for i, k in enumerate(data["ids"]):
            n = int(data["counts"][i])
            rows.append([int(k), n, round(100.0 * n / total, 2)] + [float(v) for v in data["medians"][i]])
            colors.append([None, None, None] + [_tint(v) for v in norm[i]])
        self.set_rows(headers, rows, colors)
        self.resize(min(1200, 160 + 90 * len(headers)), min(800, 140 + 26 * len(rows)))


class CompareDialog(_TableDialog):
    """Compare two populations parameter by parameter (chi-squared T(x), KS)."""

    def __init__(self, controller, parent=None):
        super().__init__("Compare Populations", parent)
        self.controller = controller
        ws = controller.workspace
        self.choices: list[Population] = []
        for s in ws.samples:
            for p in s.root.walk():
                self.choices.append(p)
        form = QFormLayout()
        self.a_combo = QComboBox()
        self.b_combo = QComboBox()
        for p in self.choices:
            label = _pop_label(p)
            self.a_combo.addItem(label)
            self.b_combo.addItem(label)
        sel = controller.selected_populations()
        if sel:
            self.a_combo.setCurrentIndex(self._index_of(sel[0]))
        if len(sel) > 1:
            self.b_combo.setCurrentIndex(self._index_of(sel[1]))
        elif len(self.choices) > 1:
            self.b_combo.setCurrentIndex(min(len(self.choices) - 1, self.a_combo.currentIndex() + 1))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(16, 1024)
        self.bins_spin.setValue(256)
        form.addRow("Population A (control):", self.a_combo)
        form.addRow("Population B:", self.b_combo)
        form.addRow("Histogram bins:", self.bins_spin)
        self.layout_.insertLayout(0, form)
        self.compare_button = QPushButton(icon("statistic"), "Compare")
        self.compare_button.clicked.connect(self.compare)
        self.button_row.insertWidget(0, self.compare_button)
        self.results: list = []
        self.resize(620, 520)

    def _index_of(self, pop) -> int:
        for i, p in enumerate(self.choices):
            if p is pop:
                return i
        return 0

    def set_pair(self, a: Population, b: Population):
        self.a_combo.setCurrentIndex(self._index_of(a))
        self.b_combo.setCurrentIndex(self._index_of(b))

    def compare(self):
        if not self.choices:
            return []
        a = self.choices[self.a_combo.currentIndex()]
        b = self.choices[self.b_combo.currentIndex()]
        self.results = compare_population_objects(a, b, bins=self.bins_spin.value())
        rows = [[r.param, r.chi_squared_T, r.ks_statistic, r.difference_score] for r in self.results]
        colors = [[None, None, _tint(r.ks_statistic), None] for r in self.results]
        self.set_rows(["Parameter", "Chi² T(x)", "KS D", "Difference"], rows, colors)
        return self.results


# ===================================================================== install
class AnalysisTools:
    """References to the actions and open dialogs of this extension."""

    def __init__(self, window):
        self.window = window
        self.controller = window.controller
        self.actions: dict[str, QAction] = {}
        self.dialogs: list[QDialog] = []

    def _parent(self):
        return self.window if isinstance(self.window, QWidget) else None

    def _keep(self, dlg):
        self.dialogs.append(dlg)
        dlg.finished.connect(lambda _r, d=dlg: self.dialogs.remove(d) if d in self.dialogs and not isinstance(d, _TableDialog) else None)
        return dlg

    def open_dimreduction(self, method: str):
        if not self.controller.workspace.samples:
            QMessageBox.information(self._parent(), DIM_METHODS[method][0], "Add samples to the workspace first.")
            return None
        dlg = self._keep(DimReductionDialog(self.controller, method, self._parent()))
        dlg.show()
        return dlg

    def open_clustering(self, method: str):
        if not self.controller.workspace.samples:
            QMessageBox.information(self._parent(), CLUSTER_METHODS[method][0], "Add samples to the workspace first.")
            return None
        dlg = self._keep(ClusteringDialog(self.controller, method, self._parent()))
        dlg.show()
        return dlg

    def open_compare(self):
        dlg = CompareDialog(self.controller, self._parent())
        self.dialogs.append(dlg)
        dlg.show()
        return dlg


def install(window):
    tools = AnalysisTools(window)
    parent = window

    def act(key, text, icon_name, tip, slot):
        a = QAction(icon(icon_name), text, parent)
        a.setToolTip(tip)
        a.setStatusTip(tip)
        a.triggered.connect(lambda _=False: slot())
        tools.actions[key] = a
        return a

    act("tsne", "t-SNE", "graph", "t-SNE embedding of the selected populations", lambda: tools.open_dimreduction("tsne"))
    act("umap", "UMAP", "graph", "UMAP embedding of the selected populations", lambda: tools.open_dimreduction("umap"))
    act("pca", "PCA", "derive", "Principal components of the selected populations", lambda: tools.open_dimreduction("pca"))
    act("flowsom", "FlowSOM", "group", "FlowSOM clustering (self-organising map + metaclustering)", lambda: tools.open_clustering("flowsom"))
    act("kmeans", "k-means", "table", "k-means clustering of the selected populations", lambda: tools.open_clustering("kmeans"))
    act("compare", "Compare", "statistic", "Compare two populations (chi-squared T(x), KS)", tools.open_compare)
    if not A.has_umap():
        a = tools.actions["umap"]
        a.setEnabled(False)
        a.setToolTip("UMAP needs the optional package umap-learn (pip install umap-learn)")
        a.setStatusTip(a.toolTip())

    pages = getattr(window, "ribbon_pages", {})
    page = pages.get("tools")
    if page is not None:
        band = page.add_band("Dimensionality Reduction")
        band.add_large(tools.actions["tsne"])
        band.add_large(tools.actions["umap"])
        band.add_small(tools.actions["pca"])
        band = page.add_band("Clustering / Compare")
        band.add_large(tools.actions["flowsom"])
        band.add_small(tools.actions["kmeans"])
        band.add_small(tools.actions["compare"])
    menus = getattr(window, "menus", {})
    menu = menus.get("tools")
    if menu is not None:
        menu.addSeparator()
        sub = menu.addMenu("Dimensionality Reduction")
        for k in ("tsne", "umap", "pca"):
            sub.addAction(tools.actions[k])
        sub = menu.addMenu("Clustering")
        for k in ("flowsom", "kmeans"):
            sub.addAction(tools.actions[k])
        menu.addAction(tools.actions["compare"])
        tools.menus = menu
    return tools
