"""Background loading (thread pool, progress, cancel), header prefetch, the
memory budget wiring and the shared progress helper."""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flowdesk.core.gates import PolygonGate, RectangleGate  # noqa: E402
from flowdesk.core.model import HEADER_CACHE, Population  # noqa: E402

from .conftest import SYNTH_NAMES, SYNTH_STAINS, make_synthetic_events, write_fcs  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def _files(tmp_path, n, events=3000, prefix="s"):
    return [write_fcs(tmp_path / f"{prefix}{i:03d}.fcs", make_synthetic_events(events, seed=i)["events"], SYNTH_NAMES, SYNTH_STAINS)
            for i in range(n)]


@pytest.fixture
def controller(app):
    from flowdesk.ui.controller import WorkspaceController

    c = WorkspaceController()
    yield c
    c.workspace.modified = False


def _gate(sample):
    cells = sample.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [50000, 20000], [None, None])))
    cells.add_child(Population("Pos", PolygonGate(["FITC-A", "PE-A"], [(1000, 1000), (1e5, 1000), (1e5, 1e5), (1000, 1e5)])))
    return cells


def test_loader_fills_counts_in_a_thread_pool(controller, tmp_path):
    from PySide6.QtCore import QThreadPool

    from flowdesk.ui.loader import LoaderProgress, SampleLoader

    samples = controller.add_samples(_files(tmp_path, 8))
    for s in samples:
        _gate(s)
    loader = SampleLoader(controller, threads=3)
    widget = LoaderProgress(loader)
    assert isinstance(loader.pool, QThreadPool) and loader.pool.maxThreadCount() == 3
    progress, filled, finished = [], [], []
    loader.progress.connect(lambda d, t, _txt: progress.append((d, t)))
    loader.countsReady.connect(lambda lst: filled.extend(lst))
    loader.finished.connect(lambda d, c: finished.append((d, c)))
    assert loader.load(samples) == 8
    assert loader.busy and not widget.isHidden()
    loader.pool.waitForDone(20000)
    # workers only touched copies: nothing is installed before the GUI thread runs
    assert all(s.fcs is None and s.root.children[0]._mask is None for s in samples)
    assert loader.wait(20000)
    assert finished == [(8, False)] and progress[-1] == (8, 8) and len(filled) == 8
    assert widget.isHidden() and not loader.busy
    for i, s in enumerate(samples):
        cells = s.root.children[0]
        assert s.loaded and cells._mask is not None and cells.children[0]._mask is not None
        ev = make_synthetic_events(3000, seed=i)["events"]
        assert cells.count == int(((ev[:, 0] >= 50000) & (ev[:, 1] >= 20000)).sum())
    # everything complete: nothing to queue again
    assert loader.load(samples) == 0


def test_loader_cancel_discards_results(controller, tmp_path):
    from flowdesk.ui.loader import LoaderProgress, SampleLoader

    samples = controller.add_samples(_files(tmp_path, 12, events=20000, prefix="c"))
    loader = SampleLoader(controller, threads=1)
    widget = LoaderProgress(loader)
    finished = []
    loader.finished.connect(lambda d, c: finished.append((d, c)))
    loader.load(samples, counts=False)
    loader.cancel()
    assert finished and finished[-1][1] is True and not loader.busy and widget.isHidden()
    loader.pool.waitForDone(20000)
    loader.wait(2000)
    assert all(s.fcs is None for s in samples)  # late results of the cancelled job are ignored
    # the loader is usable again
    assert loader.load(samples[:2], counts=False) == 2 and loader.wait(20000)
    assert samples[0].loaded and samples[1].loaded and not samples[2].loaded
    assert loader.stop_all(2000)


def test_memory_limit_releases_least_recently_used(controller, tmp_path):
    from flowdesk.ui.loader import SampleLoader, apply_memory_budget

    samples = controller.add_samples(_files(tmp_path, 3, events=40000, prefix="m"))
    controller.preferences._values["memory_limit_mb"] = 1  # session only
    try:
        apply_memory_budget(controller)
        assert controller.workspace.memory_budget.limit_mb == 1
        loader = SampleLoader(controller, threads=1)
        loader.load(samples, counts=False)
        assert loader.wait(20000)
        released = [s for s in samples if s.fcs is None]
        assert released and samples[-1].loaded  # the most recent stays
        lru = released[0]
        assert lru.data("FSC-A").shape == (40000,) and lru.loaded  # transparently reloaded
        assert controller.workspace.memory_budget.resident_bytes() <= 1024 * 1024
    finally:
        controller.preferences._values.pop("memory_limit_mb", None)
    apply_memory_budget(controller)
    assert controller.workspace.memory_budget.limit_mb == float(controller.preferences.get("memory_limit_mb", 2048))


def test_add_paths_reads_headers_in_workers(controller, tmp_path, monkeypatch):
    from flowdesk.core.fcs import FCSFile
    from flowdesk.ui.loader import SampleLoader

    paths = _files(tmp_path, 10, events=200, prefix="h")
    main = threading.current_thread()
    calls = []
    original = FCSFile.read_keywords.__func__

    def counting(cls, path):
        calls.append(threading.current_thread() is main)
        return original(cls, path)

    monkeypatch.setattr(FCSFile, "read_keywords", classmethod(counting))
    loader = SampleLoader(controller, threads=2)
    read = []
    loader.headersRead.connect(lambda lst: read.extend(lst))
    assert loader.add_paths(paths) == 10
    assert loader.wait(20000)
    assert len(controller.workspace.samples) == 10 and len(read) == 10
    assert calls and not any(calls)  # every header was read off the GUI thread
    assert all(s.keywords.get("$TOT") == "200" for s in controller.workspace.samples)
    assert not any(os.fspath(p) in HEADER_CACHE for p in paths)
    # a cancelled prefetch adds nothing and leaves no cache entries
    more = _files(tmp_path, 4, events=200, prefix="x")
    loader.add_paths(more)
    loader.cancel()
    loader.pool.waitForDone(5000)
    loader.wait(2000)
    assert len(controller.workspace.samples) == 10 and not any(os.fspath(p) in HEADER_CACHE for p in more)


def test_run_with_progress(app):
    from flowdesk.ui.loader import run_with_progress

    seen = []

    def work(step):
        for i in range(5):
            if not step(i + 1, f"item {i + 1}"):
                return "stopped"
            seen.append(i)
        return "done"

    assert run_with_progress(None, "Working", 5, work) == "done" and seen == [0, 1, 2, 3, 4]

    def cancelled(step):
        for i in range(5):
            if i == 2:
                step.dialog.cancel()  # the user presses Cancel
            if not step(i):
                return i
        return None

    assert run_with_progress(None, "Working", 5, cancelled) == 2
    assert run_with_progress(None, "Busy", 0, lambda step: step(0) and "ok", cancel=False) == "ok"


def test_install_status_bar_and_pins(app, tmp_path):
    from types import SimpleNamespace

    from flowdesk.ui import loader as loader_mod
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    try:
        ld = loader_mod.install(win)
        assert ld.progress_widget.parent() is not None and ld.memory_indicator.parent() is not None
        assert ld.progress_widget.isHidden()
        assert "Data" in ld.memory_indicator.text()
        budget = win.controller.workspace.memory_budget
        default = float(win.controller.preferences.get("memory_limit_mb", 2048))
        assert budget.limit_mb == default and budget.pin_provider is not None
        s = win.controller.add_samples(_files(tmp_path, 1, events=100, prefix="p"))[0]
        win.graph_windows[("fake",)] = SimpleNamespace(sample=s)
        assert s.id in budget.pinned_ids()
        win.graph_windows.pop(("fake",))
        # a new workspace gets the cap as well
        from flowdesk.core.model import Workspace

        win.controller.set_workspace(Workspace())
        assert win.controller.workspace.memory_budget.limit_mb == default
        assert ld.stop_all(2000)
    finally:
        win.controller.workspace.modified = False
        win.close()


def test_auto_width_basis_after_compensation(controller, tmp_path):
    from flowdesk.core.compensation import SpilloverMatrix
    from flowdesk.ui.loader import auto_adjust_after_compensation

    rng = np.random.default_rng(5)
    n = 5000
    ev = np.column_stack([rng.normal(9e4, 8e3, n), rng.normal(5e4, 6e3, n), rng.normal(0, 800, n), rng.normal(0, 40, n), np.linspace(0, 100, n)]).astype(np.float32)
    s = controller.add_samples([write_fcs(tmp_path / "wide.fcs", ev, SYNTH_NAMES, SYNTH_STAINS)])[0]
    s.ensure_loaded()
    controller.set_compensation([s], SpilloverMatrix("m", ["FITC-A", "PE-A"], np.array([[1, 0.02], [0.02, 1]])))
    assert auto_adjust_after_compensation(controller) == []  # preference off: nothing
    controller.preferences._values["auto_width_basis"] = True
    try:
        assert auto_adjust_after_compensation(controller) == [s]
    finally:
        controller.preferences._values.pop("auto_width_basis", None)
    wide, narrow = s.scale("Comp-FITC-A"), s.scale("Comp-PE-A")
    assert wide is s.scale("FITC-A") and wide.width_basis < -500
    assert -100 < narrow.width_basis <= -10
    # a scale somebody chose is left alone
    from flowdesk.core.transforms import LinearScale

    s.set_scale("FITC-A", LinearScale(0, 1000))
    assert "FITC-A" not in s.auto_adjust_scales()


def test_tree_refreshes_when_a_loader_job_finishes_without_filling_anything(monkeypatch):
    """countsReady only fires for samples the job filled; a job that ends empty (its samples were
    loaded in the meantime) must still refresh the tree, or "\u2026" placeholders stay forever."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from flowdesk.app import create_app

    create_app([])
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    try:
        calls = []
        monkeypatch.setattr(win, "_refresh_tree", lambda: calls.append("refresh"))
        win.loader.finished.emit(0, False)
        assert calls == ["refresh"]
    finally:
        win.controller.workspace.modified = False
        win.close()
