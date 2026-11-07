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
    assert _wait(app, lambda: dlg.result() == 1 or getattr(dlg, "created", None) is not None)
    _wait(app, lambda: not dlg._thread.isRunning())
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
    assert _wait(app, lambda: getattr(dlg, "created", None) is not None)
    _wait(app, lambda: not dlg._thread.isRunning())
    assert dlg.created == ["PC_1", "PC_2"]
    assert int(np.isfinite(b.data("PC_2")).sum()) == 400
    ext = window.extensions["flowdesk.ui.tools_analysis"]
    opened = ext.open_project()
    assert isinstance(opened, T.ProjectDialog) and opened in ext.dialogs
    opened.close()
