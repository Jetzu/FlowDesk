"""Tools ▸ Dimensionality Reduction / Clustering / Compare (UI)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from flowdesk.core import analysis as A  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.model import Population, Workspace  # noqa: E402
from flowdesk.ui import tools_analysis as T  # noqa: E402
from flowdesk.ui.controller import WorkspaceController  # noqa: E402

from .test_analysis import FL, write_sample  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture
def ws(tmp_path):
    ws = Workspace()
    for k in range(2):
        ws.add_sample(write_sample(tmp_path / f"u{k}.fcs", 1200, 20 + k)[0])
    for s in ws.samples:
        s.ensure_loaded()
    return ws


@pytest.fixture
def window(app, ws):
    from flowdesk.ui.main_window import MainWindow

    controller = WorkspaceController(ws)
    win = MainWindow(controller)
    yield win
    for w in list(win.graph_windows.values()):
        w.close()
    ext = win.extensions.get("flowdesk.ui.tools_analysis")
    for d in list(getattr(ext, "dialogs", [])):
        d.close()
    controller.mark_saved()
    win.close()
    app.processEvents()


def _wait(app, cond, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_install_adds_bands_and_menu(window):
    ext = window.extensions["flowdesk.ui.tools_analysis"]
    assert isinstance(ext, T.AnalysisTools)
    assert set(ext.actions) == {"tsne", "umap", "pca", "project", "flowsom", "phenograph", "kmeans", "compare"}
    titles = [a.text() for a in window.menus["tools"].actions()]
    assert "Dimensionality Reduction" in titles and "Clustering" in titles and "Compare" in titles
    assert "Project onto Existing…" in titles
    clustering = next(a.menu() for a in window.menus["tools"].actions() if a.text() == "Clustering")
    assert [a.text() for a in clustering.actions()] == ["FlowSOM", "Phenograph", "k-means"]
    assert ext.actions["umap"].isEnabled() == A.has_umap()


def test_install_without_umap(app, ws, monkeypatch):
    from flowdesk.ui.main_window import MainWindow

    monkeypatch.setattr(A, "has_umap", lambda: False)
    win = MainWindow(WorkspaceController(ws))
    ext = win.extensions["flowdesk.ui.tools_analysis"]
    assert not ext.actions["umap"].isEnabled()
    assert "umap-learn" in ext.actions["umap"].toolTip()
    dlg = T.DimReductionDialog(win.controller, "umap")
    assert not dlg.run_button.isEnabled()
    win.controller.mark_saved()
    win.close()


def test_dialogs_construct_with_defaults(app, ws):
    c = WorkspaceController(ws)
    for method in ("pca", "tsne", "umap"):
        dlg = T.DimReductionDialog(c, method)
        assert dlg.pop_list.count() == 2  # Ungated of every sample of the group
        assert dlg.chosen_parameters() == FL  # Time excluded
        assert dlg.events_spin.value() == 5000
        dlg.close()
    for method in ("kmeans", "flowsom", "phenograph"):
        dlg = T.ClusteringDialog(c, method)
        assert dlg.chosen_parameters() == FL
        dlg.close()
    dlg = T.ClusteringDialog(c, "phenograph")
    assert dlg.options() == {"k": 30, "resolution": 1.0}
    assert dlg.events_spin.value() == 10000
    dlg.close()
    # selected populations win over the default
    s = ws.samples[0]
    c.set_selection([(s, s.root)])
    dlg = T.DimReductionDialog(c, "tsne")
    assert dlg.chosen_populations() == [s.root]
    assert dlg.options() == {"perplexity": 30.0, "iterations": 500}
    dlg.close()
    cmp = T.CompareDialog(c)
    assert cmp.a_combo.count() == 2
    cmp.close()


def test_run_dimreduction_writes_parameters_and_opens_graph(window, ws):
    c = window.controller
    names = T.run_dimreduction(c, "pca", max_events=500)
    assert names == ["PC_1", "PC_2"]
    for s in ws.samples:
        v = s.data("PC_1")
        assert int(np.isfinite(v).sum()) == 500
    assert ws.modified
    first = ws.samples[0]
    assert first.root.graph.x == "PC_1" and first.root.graph.y == "PC_2"
    assert any(k[0] == first.id for k in window.graph_windows)
    # a second run gets new names
    names2 = T.run_dimreduction(c, "pca", max_events=300, open_graph=False)
    assert names2 == ["PC2_1", "PC2_2"]


def test_tsne_dialog_threaded(window, app, ws):
    c = window.controller
    dlg = T.DimReductionDialog(c, "tsne")
    dlg.events_spin.setValue(300)
    dlg.iterations.setValue(250)
    dlg.start()
    assert _wait(app, lambda: getattr(dlg, "created", None) is not None and not dlg.running)
    assert dlg.created == ["tSNE_1", "tSNE_2"]
    for s in ws.samples:
        assert int(np.isfinite(s.data("tSNE_1")).sum()) == 300
    assert len(window.graph_windows) >= 1


def test_clustering_dialog_creates_populations_and_heatmap(window, app, ws):
    c = window.controller
    dlg = T.ClusteringDialog(c, "flowsom")
    dlg.meta_spin.setValue(3)
    dlg.run_now()
    name, created = dlg.created
    assert name == "FlowSOM" and created == ["FlowSOM 1", "FlowSOM 2", "FlowSOM 3"]
    for s in ws.samples:
        counts = [s.root.find(f"/{n}").count for n in created]
        assert sum(counts) == s.n_events and min(counts) > 200
    assert c.current_group.root.find("/FlowSOM 1") is not None  # applied to the group
    heat = dlg.heatmap
    assert heat.table.rowCount() == 3 and heat.table.columnCount() == 3 + len(FL)
    text = heat.text()
    assert text.splitlines()[0].split("\t")[:3] == ["Cluster", "Count", "%"]
    heat.copy()
    assert QApplication.clipboard().text() == text
    dlg.close()
    heat.close()


def test_kmeans_on_subpopulation_per_sample(app, ws, tmp_path):
    c = WorkspaceController(ws)
    s = ws.samples[0]
    pos = s.root.add_child(Population("High FL3", RectangleGate(["FL3-A"], [10000.0], [None])))
    c.set_selection([(s, pos)])
    name, created, result = T.run_clustering(c, "kmeans", options={"k": 2})
    assert name == "kmeans" and created == ["kmeans 1", "kmeans 2"]
    assert sum(pos.find(f"/{n}").count for n in created) == pos.count
    assert ws.samples[1].root.find("/High FL3") is None
    assert c.current_group.root.find("/High FL3/kmeans 1") is None  # not in the group template
    data = T.cluster_heatmap_data(c, result)
    assert data["medians"].shape == (2, 4)


def test_compare_dialog(app, tmp_path):
    ws = Workspace()
    a = ws.add_sample(write_sample(tmp_path / "a.fcs", 2000, 1)[0])
    b = ws.add_sample(write_sample(tmp_path / "b.fcs", 2000, 2, shift={1: 8.0})[0])
    c = WorkspaceController(ws)
    c.set_selection([(a, a.root), (b, b.root)])
    dlg = T.CompareDialog(c)
    rows = dlg.compare()
    assert rows[0].param == "FL2-A"
    assert dlg.table.rowCount() == len(rows) and dlg.table.item(0, 0).text() == "FL2-A"
    out = dlg.export_csv(str(tmp_path / "cmp.csv"))
    content = open(out, encoding="utf-8").read().splitlines()
    assert content[0].startswith("Parameter,") and content[1].startswith("FL2-A,")
    dlg.close()


def test_phenograph_dialog_creates_populations_and_heatmap(window, app, ws):
    c = window.controller
    dlg = T.ClusteringDialog(c, "phenograph")
    dlg.resolution_spin.setValue(0.5)
    dlg.run_now()
    name, created = dlg.created
    assert name == "Phenograph" and created == ["Phenograph 1", "Phenograph 2", "Phenograph 3"]
    for s in ws.samples:
        counts = [s.root.find(f"/{n}").count for n in created]
        assert sum(counts) == s.n_events and min(counts) > 200
    assert ws.samples[0].computed_info["Phenograph"]["resolution"] == 0.5
    assert dlg.heatmap.table.rowCount() == 3
    dlg.close()
    dlg.heatmap.close()


def _row_of(dlg, param):
    return next(i for i, r in enumerate(dlg.results) if param in r.outputs)


def test_project_dialog_lists_results_and_projects(window, app, ws):
    c = window.controller
    a, b = ws.samples
    T.run_dimreduction(c, "pca", populations=[a.root], max_events=600, open_graph=False)
    T.run_clustering(c, "flowsom", populations=[a.root], options={"n_metaclusters": 3})
    # a t-SNE result whose fitted embedding is not in this session
    X, index = A.prepare_matrix([a], "", FL, 200)
    A.write_result(ws, index, {"tSNE_1": X[:, 0], "tSNE_2": X[:, 1]}, {"tool": "t-SNE", "parameters": FL}, model={"kind": "tsne", "engine": "openTSNE"})

    dlg = T.ProjectDialog(c)
    assert dlg.result_list.count() == 3
    texts = [dlg.result_list.item(i).text() for i in range(3)]
    assert texts[0].startswith("PCA: PC_1 / PC_2") and texts[1].startswith("FlowSOM: FlowSOM") and texts[2].startswith("t-SNE: tSNE_1 / tSNE_2")
    tsne_item = dlg.result_list.item(_row_of(dlg, "tSNE_1"))
    assert not tsne_item.flags() & Qt.ItemFlag.ItemIsEnabled
    assert "session" in tsne_item.toolTip()
    assert dlg.chosen_result().outputs == ["PC_1", "PC_2"]  # first projectable one
    assert [dlg.sample_list.item(i).text() for i in range(dlg.sample_list.count())] == [b.name]
    assert dlg.population_path() == "" and dlg.events_spin.value() == 0 and dlg.run_button.isEnabled()
    assert not dlg.select_result("tSNE_2")

    dlg.run_now()
    assert dlg.created == ["PC_1", "PC_2"]
    assert int(np.isfinite(b.data("PC_1")).sum()) == b.n_events
    assert b.computed_info["PC_1"]["projected"] and ws.modified
    assert any(k[0] == b.id for k in window.graph_windows)  # opened on the new axes

    dlg.refresh("FlowSOM")
    assert dlg.chosen_result().outputs == ["FlowSOM"]
    assert dlg.sample_list.count() == 1
    dlg.run_now()
    assert dlg.created == ["FlowSOM"] and dlg.populations_created == ["FlowSOM 1", "FlowSOM 2", "FlowSOM 3"]
    counts = [b.root.find(f"/FlowSOM {k}").count for k in (1, 2, 3)]
    assert sum(counts) == b.n_events and min(counts) > 200
    # same cluster ids as the analysed sample's populations for the same planted cluster
    from sklearn.metrics import adjusted_rand_score

    from .test_analysis import make_events

    _, truth_b = make_events(1200, 21)
    assert adjusted_rand_score(truth_b, b.data("FlowSOM")) >= 0.9

    dlg.refresh()
    pca_row = _row_of(dlg, "PC_1")
    assert "0 without" in dlg.result_list.item(pca_row).text()
    dlg.close()


def test_project_dialog_threaded_and_empty(window, app, ws):
    c = window.controller
    empty = T.ProjectDialog(c)
    assert empty.result_list.count() == 0 and not empty.run_button.isEnabled()
    assert "no t-SNE" in empty.note.text()
    empty.close()

    a, b = ws.samples
    T.run_dimreduction(c, "pca", populations=[a.root], max_events=500, open_graph=False)
    dlg = T.ProjectDialog(c, param="PC_2")
    dlg.events_spin.setValue(400)
    dlg.start()
    assert _wait(app, lambda: getattr(dlg, "created", None) is not None and not dlg.running)
    assert dlg.created == ["PC_1", "PC_2"]
    assert int(np.isfinite(b.data("PC_2")).sum()) == 400
    ext = window.extensions["flowdesk.ui.tools_analysis"]
    opened = ext.open_project()
    assert isinstance(opened, T.ProjectDialog) and opened in ext.dialogs
    opened.close()


# ------------------------------------------------- workspace binding / threading
def _gated(real, release, uninterruptible_s=0.0):
    """A compute function that waits for `release` (polling should_cancel,
    except during the first `uninterruptible_s` seconds), then runs `real`."""
    calls = {"started": False}

    def fn(progress=None, should_cancel=None, **kwargs):
        calls["started"] = True
        deaf_until = time.time() + uninterruptible_s
        while not release.is_set():
            if time.time() >= deaf_until and should_cancel():
                raise T.Cancelled()
            time.sleep(0.005)
        return real(progress=progress, should_cancel=should_cancel, **kwargs)

    fn.calls = calls
    return fn


def _flush_deletes(app):
    from PySide6.QtCore import QEvent

    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, int(QEvent.Type.DeferredDelete))


def _alive(obj):
    import shiboken6

    return shiboken6.isValid(obj)


def _boxes(monkeypatch):
    shown = []
    monkeypatch.setattr(T.QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")))
    return shown


def test_dialogs_close_when_workspace_replaced(window, app, tmp_path):
    """Finding 3: a dialog opened on one workspace must not compute stale
    populations into the next one (sample ids restart at 1 per workspace)."""
    c = window.controller
    ext = window.extensions["flowdesk.ui.tools_analysis"]
    dims = ext.open_dimreduction("pca")
    clus = ext.open_clustering("kmeans")
    proj = ext.open_project()
    new = Workspace()
    for k in range(2):
        new.add_sample(write_sample(tmp_path / f"n{k}.fcs", 1200, 40 + k)[0])
    c.set_workspace(new)
    _flush_deletes(app)
    assert not any(_alive(d) for d in (dims, clus, proj))
    assert ext.dialogs == []
    assert all(not s.computed for s in new.samples) and not c.can_undo


def test_run_cancelled_when_workspace_replaced_mid_run(window, app, ws, tmp_path):
    """Finding 3 (macOS file-open event during a modal run): the finished
    k-means of the old workspace must not land on the new one's samples."""
    import threading

    c = window.controller
    messages = []
    c.statusMessage.connect(messages.append)
    ext = window.extensions["flowdesk.ui.tools_analysis"]
    dlg = ext.open_clustering("kmeans")
    dlg.k_spin.setValue(3)
    release = threading.Event()
    fn = _gated(T.compute_clustering, release)
    dlg.compute_fn = lambda: fn
    dlg.start()
    assert _wait(app, lambda: fn.calls["started"])
    new = Workspace()
    for k in range(2):  # same ids and event counts as the analysed samples
        new.add_sample(write_sample(tmp_path / f"m{k}.fcs", 1200, 60 + k)[0])
    c.set_workspace(new)
    release.set()  # the old code would now finish and write onto `new`
    assert _wait(app, lambda: not _alive(dlg) or not dlg.running, 20)
    _wait(app, lambda: False, 0.3)
    _flush_deletes(app)
    assert all("kmeans" not in s.computed for s in new.samples)
    assert all(s.root.children == [] for s in new.samples)
    assert not c.can_undo
    assert all("kmeans" not in s.computed for s in ws.samples)
    assert any("workspace was replaced" in m for m in messages)
    assert not _alive(dlg) and dlg not in ext.dialogs


def test_result_discarded_when_sample_removed_mid_run(window, app, ws, monkeypatch):
    """Finding 3: apply refuses (with a message) when a sample the run read
    is no longer in the workspace; nothing is written on the others."""
    import threading

    c = window.controller
    shown = _boxes(monkeypatch)
    a, b = ws.samples
    dlg = T.ClusteringDialog(c, "kmeans")
    dlg.k_spin.setValue(3)
    release = threading.Event()
    fn = _gated(T.compute_clustering, release)
    dlg.compute_fn = lambda: fn
    dlg.start()
    assert _wait(app, lambda: fn.calls["started"])
    ws.remove_sample(b)
    release.set()
    assert _wait(app, lambda: not dlg.running, 30)
    assert len(shown) == 1 and "u1.fcs" in shown[0] and "removed" in shown[0]
    assert "kmeans" not in a.computed and a.root.children == []
    assert not c.can_undo
    assert dlg.result() == 0 and dlg.run_button.isEnabled()  # left open
    # a new start refuses the stale population
    dlg.start()
    assert len(shown) == 2 and "no longer in the workspace" in shown[1]
    assert not dlg.running
    dlg.close()


def test_apply_refuses_result_of_another_workspace(app, ws, tmp_path):
    """Findings 3 / 29: results are matched to samples by identity, and a
    refused apply leaves no undo step and keeps the redo stack."""
    other = Workspace()
    for k in range(2):  # sample ids 1 and 2 again, same event counts
        other.add_sample(write_sample(tmp_path / f"o{k}.fcs", 1200, 80 + k)[0])
    c = WorkspaceController(other)
    c.begin_change("something")
    c.undo()
    assert c.can_redo
    pops = [s.root for s in ws.samples]
    dim = T.compute_dimreduction(pops, FL, "pca", max_events=300)
    clu = T.compute_clustering(pops, FL, "kmeans", options={"k": 3})
    for apply, result in ((T.apply_dimreduction, dim), (T.apply_clustering, clu)):
        with pytest.raises(ValueError, match="no longer in the workspace"):
            apply(c, result)
    # projection computed on `ws` applied to `other`
    T.run_dimreduction(ws, "pca", populations=[ws.samples[0].root], max_events=300)
    proj = A.compute_projection(ws, "PC_1", [ws.samples[1]], max_events=200)
    with pytest.raises(ValueError, match="no longer in the workspace"):
        T.apply_projection(c, proj)
    assert all(not s.computed for s in other.samples)
    assert not c.can_undo and c.can_redo


def test_failed_apply_pushes_no_undo_step(app, ws):
    """Finding 29: an apply that cannot succeed fails before begin_change."""
    c = WorkspaceController(ws)
    c.begin_change("earlier")
    c.undo()
    pops = [s.root for s in ws.samples]
    res = T.compute_clustering(pops, FL, "kmeans", options={"k": 3})
    short = T.ClusterResult(res.method, res.params, res.index, res.labels[:-5], res.X, res.populations, res.options, res.model)
    with pytest.raises(ValueError, match="values"):
        T.apply_clustering(c, short)
    dim = T.compute_dimreduction(pops, FL, "pca", max_events=200)
    bad_rows = T.DimResult(dim.method, dim.params, [(dim.index[0][0], dim.index[0][1] + 5000)] + dim.index[1:], dim.embedding, dim.populations, dim.options, dim.model)
    with pytest.raises(ValueError, match="rows do not match"):
        T.apply_dimreduction(c, bad_rows)
    assert not c.can_undo and c.can_redo and c.undo_label() == ""
    assert all(not s.computed for s in ws.samples)
    # the valid result still applies with exactly one step
    T.apply_clustering(c, res)
    assert c.undo_label() == "k-means clusters" and len(c._undo) == 1


def test_cancel_keeps_progress_until_worker_stops(app, ws):
    """Finding 12: Cancel must not hide the (modal) progress dialog while the
    worker still runs; it closes when the worker has stopped."""
    import threading

    from PySide6.QtWidgets import QPushButton

    c = WorkspaceController(ws)
    dlg = T.DimReductionDialog(c, "pca")
    dlg.show()
    release = threading.Event()
    fn = _gated(T.compute_dimreduction, release, uninterruptible_s=2.0)
    dlg.compute_fn = lambda: fn
    dlg.start()
    assert _wait(app, lambda: fn.calls["started"])
    progress = dlg._progress
    progress.findChild(QPushButton).click()
    app.processEvents()
    assert dlg.running and progress.isVisible() and progress.windowModality() == Qt.WindowModality.WindowModal
    assert "Cancelling" in progress.labelText()
    assert not progress.findChild(QPushButton).isEnabled()
    progress.reject()  # Escape: still cancelling, still visible
    app.processEvents()
    assert progress.isVisible()
    assert _wait(app, lambda: not dlg.running, 20)
    assert not progress.isVisible() and dlg._progress is None
    assert dlg.run_button.isEnabled() and dlg.isVisible()
    assert all("PC_1" not in s.computed for s in ws.samples) and not c.can_undo
    dlg.close()


def test_closing_dialog_with_running_worker_does_not_block(app, ws):
    """Finding 12: done() cancels the worker and returns at once; the dialog
    closes (and is deleted) when the thread has stopped."""
    import threading

    c = WorkspaceController(ws)
    dlg = T.ClusteringDialog(c, "kmeans")
    dlg.show()
    finished = []
    dlg.finished.connect(finished.append)
    release = threading.Event()
    fn = _gated(T.compute_clustering, release, uninterruptible_s=3.0)
    dlg.compute_fn = lambda: fn
    dlg.start()
    assert _wait(app, lambda: fn.calls["started"])
    t0 = time.perf_counter()
    dlg.reject()
    assert time.perf_counter() - t0 < 1.0  # ~0.1 s; the old done() waited for the worker (up to 30 s)
    assert dlg.running and finished == [] and dlg._worker.cancelled
    assert _wait(app, lambda: finished != [], 20)
    _flush_deletes(app)
    assert not _alive(dlg)
    assert all("kmeans" not in s.computed for s in ws.samples)


def test_tsne_fit_interrupted_by_cancel(app, ws):
    """Finding 12: cancelling a real t-SNE run stops it inside the fit; the
    10000-iteration run on 2400 events would otherwise go on for ~30 s."""
    c = WorkspaceController(ws)
    dlg = T.DimReductionDialog(c, "tsne")
    dlg.events_spin.setValue(0)  # all events
    dlg.iterations.setValue(10000)
    dlg.start()
    assert _wait(app, lambda: dlg._progress is not None and "Running" in dlg._progress.labelText(), 30)
    _wait(app, lambda: False, 1.5)
    assert dlg.running
    t0 = time.time()
    dlg._progress.canceled.emit()
    assert _wait(app, lambda: not dlg.running, 60)
    assert time.time() - t0 < 5  # typically well under a second
    assert all("tSNE_1" not in s.computed for s in ws.samples)
    dlg.close()


def test_stop_all_cancels_workers_without_blocking(window, app, ws):
    """The main window calls stop_all() before quitting: it cancels every run
    and joins for at most ~500 ms in total."""
    import threading

    c = window.controller
    ext = window.extensions["flowdesk.ui.tools_analysis"]
    assert ext.stop_workers == ext.stop_all and ext.stop_all() and not ext.running()
    slow = ext.open_clustering("kmeans")
    quick = ext.open_dimreduction("pca")
    release = threading.Event()
    slow_fn = _gated(T.compute_clustering, release, uninterruptible_s=5.0)
    quick_fn = _gated(T.compute_dimreduction, release)
    slow.compute_fn = lambda: slow_fn
    quick.compute_fn = lambda: quick_fn
    slow.start()
    quick.start()
    assert _wait(app, lambda: slow_fn.calls["started"] and quick_fn.calls["started"])
    assert ext.running()
    t0 = time.perf_counter()
    assert ext.stop_all() is False  # `slow` ignores cancellation for 5 s
    assert time.perf_counter() - t0 < 2.0  # joins for ~0.5 s in total
    assert not quick.running and slow.running and slow._worker.cancelled
    assert _wait(app, lambda: not slow.running, 20)
    assert ext.stop_all() and not ext.running()
    release.set()
    assert all(not s.computed for s in ws.samples) and not c.can_undo


def test_closed_dialogs_are_deleted_and_results_released(window, app, ws):
    """Finding 32: dialogs, workers and progress dialogs do not pile up under
    the main window, and a dialog does not keep its last result."""
    from PySide6.QtWidgets import QDialog, QProgressDialog

    ext = window.extensions["flowdesk.ui.tools_analysis"]

    def count():
        return (len(window.findChildren(QDialog)), len(window.findChildren(QProgressDialog)), len(window.findChildren(T._Worker)))

    _flush_deletes(app)
    before = count()
    for _ in range(3):
        d = ext.open_dimreduction("pca")
        d.events_spin.setValue(200)
        accepted = []
        d.accepted.connect(lambda: accepted.append(True))
        d.start()
        assert _wait(app, lambda: accepted and not d.running)
        assert d.last_result is None and d._worker is None
    ext.open_compare().close()
    p = ext.open_project()
    p.close()
    k = ext.open_clustering("kmeans")
    k.k_spin.setValue(2)
    k.run_now()
    assert k.last_result is None
    k.heatmap.close()
    k.close()
    _flush_deletes(app)
    assert count() == before
    assert ext.dialogs == []
