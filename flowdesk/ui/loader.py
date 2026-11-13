"""Background sample loading, progress indication and the memory budget.

``SampleLoader`` reads FCS event data - and, optionally, every population's
membership ("counts") - for many samples in a QThreadPool so the workspace
stays responsive while 100+ samples load (FlowJo loads in the background with
a progress bar). Workers only touch detached copies of the samples
(``Sample.detached_copy``); results are taken back on the GUI thread
(``Sample.adopt``), so every model mutation stays on the GUI thread. It can
also read the headers of many files ahead of ``WorkspaceController.add_samples``
(``add_paths``) so that adding hundreds of files from a slow volume shows
progress and can be cancelled.

``LoaderProgress`` is the non-modal status-bar activity indicator (label,
progress bar, cancel button); ``MemoryIndicator`` shows the event data held in
memory against the workspace's ``memory_budget``. ``run_with_progress`` is the
shared cancelable progress dialog for long operations that must run on the
GUI thread (layout batch export, table creation, file export).

``install(window)`` (an EXTENSION_MODULES entry) puts the two indicators into
the window's status bar, applies the "memory_limit_mb" preference to every
workspace, pins the samples shown in open windows and returns the loader
(``window.extensions["flowdesk.ui.loader"]``); ``stop_all()`` cancels and
waits for the workers when the window closes.

Preference keys read with ``prefs.get(key, fallback)``: ``memory_limit_mb``
(2048; 0 = no cap), ``loader_threads`` (0 = automatic), ``auto_width_basis``
(False: when True, a compensation change gives the loaded samples'
biexponential parameters a data-driven width basis).
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QProgressDialog,
    QToolButton,
    QWidget,
)

from ..core.fcs import FCSError
from ..core.model import HEADER_CACHE, Sample, read_header_entry

DEFAULT_MEMORY_LIMIT_MB = 2048
PREF_MEMORY_LIMIT = "memory_limit_mb"
PREF_THREADS = "loader_threads"
PREF_AUTO_WIDTH = "auto_width_basis"


def _pref(controller, key: str, default):
    prefs = getattr(controller, "preferences", None)
    if prefs is None:
        return default
    try:
        value = prefs.get(key, default)
    except Exception:  # noqa: BLE001 - preferences must never stop loading
        return default
    return default if value is None else value


def format_bytes(n: float) -> str:
    n = float(n)
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{int(n)} B"


# ------------------------------------------------------------------ workers
class _Signals(QObject):
    done = Signal(int, object, object, object)  # generation, key, status ("ok" / "error"), value


class _Task(QRunnable):
    def __init__(self, signals: _Signals, generation: int, key, fn: Callable[[], object], cancelled: Callable[[], bool]):
        super().__init__()
        self.signals = signals
        self.generation = generation
        self.key = key
        self.fn = fn
        self.cancelled = cancelled
        self.setAutoDelete(True)

    def run(self):
        if self.cancelled():
            return
        try:
            status, value = "ok", self.fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the GUI thread
            status, value = "error", f"{type(exc).__name__}: {exc}"
        if not self.cancelled():
            try:
                self.signals.done.emit(self.generation, self.key, status, value)
            except RuntimeError:  # the loader was deleted while this task ran (shutdown)
                pass


class _Job:
    def __init__(self, label: str, items: list, on_finish: Callable[[bool], None] | None = None):
        self.label = label
        # (key, fn run in a worker, on_result(status, value) run on the GUI thread)
        self.items = items
        self.on_finish = on_finish
        self.pending: dict = {}
        self.done = 0
        self.errors: list[str] = []


class SampleLoader(QObject):
    """Loads sample data in a thread pool, one job at a time (later jobs wait)."""

    started = Signal(int, str)  # total, label
    progress = Signal(int, int, str)  # done, total, text
    sampleLoaded = Signal(object)  # Sample whose data (and masks) arrived
    countsReady = Signal(list)  # samples whose population masks were filled by a finished job
    headersRead = Signal(list)  # paths whose headers a finished add_paths job read
    finished = Signal(int, bool)  # done, cancelled

    def __init__(self, controller=None, parent: QObject | None = None, threads: int | None = None):
        super().__init__(parent)
        self.controller = controller
        self.pool = QThreadPool(self)
        if threads is None:
            threads = int(_pref(controller, PREF_THREADS, 0) or 0)
        self.set_threads(threads)
        self._signals = _Signals(self)
        self._signals.done.connect(self._on_done, Qt.ConnectionType.QueuedConnection)
        self._generation = 0
        self._cancel_flag = False
        self._job: _Job | None = None
        self._queue: list[_Job] = []
        self._filled: list[Sample] = []
        self.last_errors: list[str] = []

    # ------------------------------------------------------------- config
    def set_threads(self, threads: int = 0):
        if not threads or threads <= 0:
            threads = max(1, min(4, (os.cpu_count() or 2) - 1))
        self.pool.setMaxThreadCount(int(threads))

    @property
    def busy(self) -> bool:
        return self._job is not None or bool(self._queue)

    @property
    def total(self) -> int:
        return len(self._job.items) if self._job is not None else 0

    @property
    def done(self) -> int:
        return self._job.done if self._job is not None else 0

    # ------------------------------------------------------------- jobs
    def load(self, samples: Iterable[Sample], counts: bool = True, label: str | None = None) -> int:
        """Read the event data of `samples` (and, with counts=True, compute
        every population's membership) in the background. Samples already
        complete are skipped. Returns the number of samples queued."""
        items = []
        queued = {id(s) for job in ([self._job] if self._job else []) + self._queue for s in getattr(job, "samples", [])}
        samples_out = []
        for sample in samples:
            if id(sample) in queued or sample.load_error and not sample.path.exists():
                continue
            if sample.fcs is not None and (not counts or all(p._mask is not None for p in sample.root.walk())):
                continue
            copy = sample.detached_copy()
            fingerprint = sample.state_fingerprint()
            items.append((id(sample), self._worker_fn(copy, counts), self._result_fn(sample, copy, fingerprint)))
            samples_out.append(sample)
            queued.add(id(sample))
        if not items:
            return 0
        n = len(items)
        job = _Job(label or f"Loading {n} sample{'s' if n != 1 else ''}", items, self._finish_load)
        job.samples = samples_out
        self._submit(job)
        return n

    @staticmethod
    def _worker_fn(copy: Sample, counts: bool) -> Callable[[], Sample]:
        def run():
            if not copy.ensure_loaded():
                raise FCSError(copy.load_error or f"cannot read {copy.path}")
            if counts:
                for pop in copy.root.walk():
                    pop.mask()
            return copy

        return run

    def _result_fn(self, sample: Sample, copy: Sample, fingerprint: str):
        def result(status, value):
            if status != "ok":
                sample.load_error = str(value)
                return
            if sample.adopt(value, fingerprint):
                self._filled.append(sample)
            self.sampleLoaded.emit(sample)

        return result

    def _finish_load(self, cancelled: bool):
        filled, self._filled = self._filled, []
        if filled and not cancelled:
            self.countsReady.emit(filled)

    def add_paths(self, paths: Iterable, on_ready: Callable[[list], None] | None = None) -> int:
        """Read the headers of `paths` in the background (progress, cancel);
        when all are read, ``on_ready(paths)`` runs on the GUI thread - by
        default ``controller.add_samples(paths)``, which then finds the
        headers in model.HEADER_CACHE. Returns the number of files queued."""
        paths = [Path(p) for p in paths]
        if not paths:
            return 0
        read: list = []

        def item(path: Path):
            def result(status, value):
                if status == "ok":
                    HEADER_CACHE[os.fspath(path)] = value
                    read.append(path)

            return (os.fspath(path), lambda: read_header_entry(path), result)

        def finish(cancelled: bool):
            if cancelled:
                for p in read:
                    HEADER_CACHE.pop(os.fspath(p), None)
                return
            self.headersRead.emit(list(read))
            target = on_ready
            if target is None and self.controller is not None:
                target = self.controller.add_samples
            try:
                if target is not None:
                    target(list(paths))
            finally:
                for p in read:  # entries add_samples did not use (duplicates, ...)
                    HEADER_CACHE.pop(os.fspath(p), None)

        n = len(paths)
        self._submit(_Job(f"Reading {n} file header{'s' if n != 1 else ''}", [item(p) for p in paths], finish))
        return n

    def _submit(self, job: _Job):
        self._queue.append(job)
        if self._job is None:
            self._start_next()

    def _start_next(self):
        if not self._queue:
            self._job = None
            return
        job = self._queue.pop(0)
        self._job = job
        self._generation += 1
        self._cancel_flag = False
        gen = self._generation
        self.started.emit(len(job.items), job.label)
        self.progress.emit(0, len(job.items), job.label)
        if not job.items:
            self._finish(job, False)
            return
        for key, fn, on_result in job.items:
            job.pending[key] = on_result
            self.pool.start(_Task(self._signals, gen, key, fn, self._is_cancelled))

    def _is_cancelled(self) -> bool:
        return self._cancel_flag

    def _on_done(self, generation: int, key, status, value):
        job = self._job
        if job is None or generation != self._generation:
            return  # a cancelled job's late result
        on_result = job.pending.pop(key, None)
        if on_result is None:
            return
        try:
            on_result(status, value)
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the job
            status, value = "error", f"{type(exc).__name__}: {exc}"
        if status != "ok":
            job.errors.append(str(value))
        job.done += 1
        self.progress.emit(job.done, len(job.items), f"{job.label}: {job.done} of {len(job.items)}")
        if not job.pending:
            self._finish(job, False)

    def _finish(self, job: _Job, cancelled: bool):
        self.last_errors = list(job.errors)
        self._job = None
        if job.on_finish is not None:
            try:
                job.on_finish(cancelled)
            except Exception:  # noqa: BLE001
                import traceback

                traceback.print_exc()
        self.finished.emit(job.done, cancelled)
        if not cancelled and self._job is None:  # on_finish may already have started the next job
            self._start_next()

    def cancel(self):
        """Stop the running job (finishing tasks are discarded) and drop the
        queued ones."""
        self._cancel_flag = True
        self.pool.clear()
        self._generation += 1
        queued, self._queue = self._queue, []
        job = self._job
        if job is not None:
            job.pending.clear()
            self._finish(job, True)
        for j in queued:
            if j.on_finish is not None:
                j.on_finish(True)
        self._filled = []

    def stop_all(self, timeout_ms: int = 3000) -> bool:
        """Cancel everything and wait for running workers (window shutdown)."""
        self.cancel()
        return bool(self.pool.waitForDone(int(timeout_ms)))

    def wait(self, timeout_ms: int = 30000) -> bool:
        """Process events until every job finished (tests, scripts). Returns
        False on timeout."""
        timer = QElapsedTimer()
        timer.start()
        app = QApplication.instance()
        while self.busy:
            if timer.elapsed() > timeout_ms:
                return False
            self.pool.waitForDone(20)
            if app is not None:
                app.processEvents()
        return True


# ------------------------------------------------------------- indicators
class LoaderProgress(QWidget):
    """Status-bar activity indicator: what runs, a progress bar and Cancel.
    Hidden while the loader is idle."""

    def __init__(self, loader: SampleLoader, parent: QWidget | None = None):
        super().__init__(parent)
        self.loader = loader
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.label = QLabel()
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(140)
        self.bar.setMaximumHeight(14)
        self.bar.setTextVisible(False)
        self.cancel_button = QToolButton()
        self.cancel_button.setText("✕")
        self.cancel_button.setAutoRaise(True)
        self.cancel_button.setToolTip("Cancel")
        self.cancel_button.clicked.connect(loader.cancel)
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        lay.addWidget(self.cancel_button)
        loader.started.connect(self._started)
        loader.progress.connect(self._progress)
        loader.finished.connect(self._finished)
        self.setVisible(False)

    def _started(self, total: int, label: str):
        self.bar.setRange(0, max(total, 0))
        self.bar.setValue(0)
        self.label.setText(label)
        self.setVisible(True)

    def _progress(self, done: int, total: int, text: str):
        self.bar.setRange(0, max(total, 0))
        self.bar.setValue(done)
        self.label.setText(text)

    def _finished(self, _done: int, _cancelled: bool):
        if not self.loader.busy:
            self.setVisible(False)


class MemoryIndicator(QLabel):
    """Event data held in memory, against the workspace's memory budget."""

    def __init__(self, controller, parent: QWidget | None = None, interval_ms: int = 2000):
        super().__init__(parent)
        self.controller = controller
        self.timer = QTimer(self)
        self.timer.setInterval(int(interval_ms))
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def refresh(self):
        ws = getattr(self.controller, "workspace", None)
        if ws is None:
            return
        budget = ws.memory_budget
        used = budget.resident_bytes()
        limit = budget.limit_bytes
        loaded = len(budget.loaded_samples())
        self.setText(f"Data {format_bytes(used)}" + (f" / {format_bytes(limit)}" if limit else ""))
        tip = f"Event data in memory: {format_bytes(used)} for {loaded} of {len(ws.samples)} samples"
        if limit:
            tip += f"\nLimit {format_bytes(limit)} (Preferences: memory_limit_mb); least-recently used samples not shown in a window are released"
        if budget.released:
            tip += f"\nReleased so far: {budget.released}"
        self.setToolTip(tip)


# ----------------------------------------------------------------- helpers
def run_with_progress(parent, title: str, total: int, fn: Callable[[Callable[..., bool]], object], cancel: bool = True,
                      minimum_duration_ms: int = 400):
    """Run ``fn(step)`` on the GUI thread behind a progress dialog (busy
    cursor, window kept responsive). ``step(done, text="")`` reports progress
    and returns False once the user pressed Cancel - fn should then stop.
    total <= 0 shows a busy indicator. Returns fn's result."""
    dlg = QProgressDialog(title, "Cancel", 0, max(int(total), 0), parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(int(minimum_duration_ms))
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    if not cancel:
        dlg.setCancelButton(None)
    dlg.setValue(0)

    def step(done: int, text: str = "") -> bool:
        if total > 0:
            dlg.setValue(min(int(done), int(total)))
        if text:
            dlg.setLabelText(text)
        QApplication.processEvents()
        return not (cancel and dlg.wasCanceled())

    step.dialog = dlg
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        return fn(step)
    finally:
        QApplication.restoreOverrideCursor()
        dlg.close()
        dlg.deleteLater()


def shown_samples(window) -> list[Sample]:
    """Samples shown in open windows (graph windows and any visible top-level
    window with a ``sample``): the memory budget never releases them."""
    out: list[Sample] = []
    for w in list(getattr(window, "graph_windows", {}).values()):
        s = getattr(w, "sample", None)
        if isinstance(s, Sample):
            out.append(s)
    app = QApplication.instance()
    if app is not None:
        for w in app.topLevelWidgets():
            s = getattr(w, "sample", None)
            if isinstance(s, Sample) and w.isVisible():
                out.append(s)
    return out


def apply_memory_budget(controller, pin_provider: Callable[[], Iterable] | None = None):
    """Set the current workspace's memory cap from the "memory_limit_mb"
    preference (and who is pinned)."""
    ws = getattr(controller, "workspace", None)
    if ws is None:
        return
    try:
        limit = float(_pref(controller, PREF_MEMORY_LIMIT, DEFAULT_MEMORY_LIMIT_MB))
    except (TypeError, ValueError):
        limit = DEFAULT_MEMORY_LIMIT_MB
    ws.memory_budget.limit_mb = limit if limit > 0 else None
    if pin_provider is not None:
        ws.memory_budget.pin_provider = pin_provider


def auto_adjust_after_compensation(controller, sample_id=None) -> list[Sample]:
    """With the "auto_width_basis" preference on, give the loaded samples
    whose compensation changed a data-driven biexponential width basis on
    their compensated parameters (parameters with a chosen scale are kept)."""
    if not _pref(controller, PREF_AUTO_WIDTH, False):
        return []
    ws = controller.workspace
    samples = [s for s in ws.samples if (sample_id is None or s.id == sample_id) and s.loaded and s._comp_active() is not None]
    changed = [s for s in samples if s.auto_adjust_scales()]
    if changed:
        controller.scalesChanged.emit(changed[0].id if len(changed) == 1 else None)
        controller.gatingChanged.emit(changed[0].id if len(changed) == 1 else None)
    return changed


# ----------------------------------------------------------------- install
def install(window) -> SampleLoader:
    """Status-bar indicators, the memory budget from the preferences and a
    loader the window starts with ``loader.load(samples)``."""
    controller = window.controller
    loader = SampleLoader(controller, parent=window)
    loader.progress_widget = LoaderProgress(loader)
    loader.memory_indicator = MemoryIndicator(controller)
    bar = window.statusBar() if hasattr(window, "statusBar") else None
    if bar is not None:
        bar.addPermanentWidget(loader.progress_widget)
        bar.addPermanentWidget(loader.memory_indicator)

    def pinned():
        return shown_samples(window)

    def on_workspace():
        loader.cancel()
        apply_memory_budget(controller, pinned)
        loader.memory_indicator.refresh()

    def on_preferences():
        apply_memory_budget(controller, pinned)
        loader.set_threads(int(_pref(controller, PREF_THREADS, 0) or 0))

    controller.workspaceReplaced.connect(on_workspace)
    if hasattr(controller, "preferencesChanged"):
        controller.preferencesChanged.connect(on_preferences)
    if hasattr(controller, "compensationChanged"):
        controller.compensationChanged.connect(lambda sid: auto_adjust_after_compensation(controller, sid))
    apply_memory_budget(controller, pinned)
    return loader
