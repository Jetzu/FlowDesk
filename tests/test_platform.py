"""Cross-platform guards: launchers, entry point, no macOS-only texts, portable paths."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "flowdesk"

# "⌘" written literally, as an HTML entity or as a Python escape
COMMAND_KEY = re.compile("⌘|&#8984;|&#x2318;|\\\\u2318", re.I)
# files allowed to mention the key: platform helpers, plus the Quick Start text for now
# (main_window.QUICK_START_HTML says "⌘/Ctrl" - TODO: render it with the native key name)
ALLOWED = {
    "flowdesk/ui/style.py",  # modifier_text() / shortcut_text() / modifier_html()
    "flowdesk/ui/platform.py",
    "flowdesk/platform.py",
    "flowdesk/ui/main_window.py",
}


def test_no_command_key_symbol_outside_platform_helpers():
    offenders = []
    for path in PKG_DIR.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if COMMAND_KEY.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, "macOS-only ⌘ text outside a platform helper:\n" + "\n".join(offenders)


def test_main_window_command_key_only_in_quick_start():
    src = (PKG_DIR / "ui" / "main_window.py").read_text(encoding="utf-8")
    lines = [n for n, line in enumerate(src.splitlines(), 1) if COMMAND_KEY.search(line)]
    if not lines:
        return
    start = src.find("QUICK_START_HTML")
    assert start >= 0
    end_line = src[: src.index('"""', src.index('"""', start) + 3)].count("\n") + 1
    start_line = src[:start].count("\n") + 1
    assert all(start_line <= n <= end_line for n in lines)


@pytest.mark.parametrize("name", ["flowdesk.sh", "flowdesk.bat", "FlowDesk.command"])
def test_launch_scripts_exist(name):
    assert (ROOT / name).is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize("name", ["flowdesk.sh", "FlowDesk.command", "packaging/install_linux_desktop.sh"])
def test_posix_launchers_are_executable(name):
    path = ROOT / name
    assert os.access(path, os.X_OK), f"{name} is not executable (chmod +x)"
    assert path.read_bytes().startswith(b"#!"), f"{name} has no shebang"
    assert b"\r\n" not in path.read_bytes(), f"{name} has CRLF line endings"


def test_batch_launcher_uses_crlf_and_no_errorlevel_in_block():
    data = (ROOT / "flowdesk.bat").read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n"), "flowdesk.bat must use CRLF line endings"
    # %errorlevel% inside a parenthesised block is expanded when the block is parsed
    assert b"%errorlevel%" not in data.lower()


def test_packaging_files_present():
    for name in ("flowdesk.spec", "build.py", "make_icons.py", "flowdesk_main.py", "flowdesk.desktop", "icon.png", "icon.ico"):
        assert (ROOT / "packaging" / name).is_file(), name
    desktop = (ROOT / "packaging" / "flowdesk.desktop").read_text(encoding="utf-8")
    assert desktop.startswith("[Desktop Entry]")
    assert "Icon=flowdesk" in desktop and "Exec=" in desktop


def test_module_help_runs_from_any_cwd(tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    out = subprocess.run(
        [sys.executable, "-m", "flowdesk", "--help"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "usage: flowdesk" in out.stdout
    out = subprocess.run(
        [sys.executable, str(PKG_DIR / "app.py"), "--version"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("FlowDesk ")


def test_create_app_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from flowdesk import APP_NAME
    from flowdesk.app import create_app, main

    app = create_app([])
    assert isinstance(app, QApplication)
    assert create_app([]) is app  # idempotent, reuses the instance
    assert QApplication.applicationName() == APP_NAME
    assert not app.windowIcon().isNull()
    assert main(["flowdesk", "--version"]) == 0


def test_sample_to_dict_survives_paths_on_different_drives(monkeypatch, tmp_path):
    from flowdesk.core import serialization
    from flowdesk.core.model import Sample

    def relpath(*_args, **_kw):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(serialization.os.path, "relpath", relpath)
    sample = Sample(tmp_path / "a.fcs", 1)
    d = serialization.sample_to_dict(sample, tmp_path / "ws")
    assert d["rel_path"] is None
    assert d["path"] == str(tmp_path / "a.fcs")


def test_file_open_event_opens_workspace(tmp_path):
    """macOS Finder double-click / Dock drop arrives as a QFileOpenEvent."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from flowdesk.app import _FileOpenFilter, create_app, open_path_in_window

    app = create_app([])
    from PySide6.QtGui import QFileOpenEvent
    from PySide6.QtWidgets import QApplication

    from flowdesk.core.fcs import Channel, write_fcs
    from flowdesk.ui.main_window import MainWindow

    import numpy as np

    fcs = tmp_path / "evt.fcs"
    write_fcs(fcs, np.random.default_rng(0).normal(1000, 100, (300, 2)).astype(np.float32), [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144)])
    win = MainWindow()
    flt = _FileOpenFilter(win)
    app.installEventFilter(flt)
    try:
        QApplication.sendEvent(app, QFileOpenEvent(str(fcs)))
        app.processEvents()
        assert [s.name for s in win.controller.workspace.samples] == ["evt.fcs"]
        assert open_path_in_window(win, tmp_path / "missing.fdw") is False
        assert open_path_in_window(win, tmp_path / "notes.txt") is False
    finally:
        app.removeEventFilter(flt)
        win.controller.workspace.modified = False
        win.close()


# ----------------------------------------------------- main window lifecycle
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from flowdesk.app import create_app

    return create_app([])


def _synthetic_fcs(path, n=300, seed=0):
    import numpy as np

    from flowdesk.core.fcs import Channel, write_fcs

    rng = np.random.default_rng(seed)
    write_fcs(path, rng.normal(1000, 100, (n, 2)).astype(np.float32), [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144)])
    return path


def _process_until(app, cond, timeout=3.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return cond()


def _worker_class():
    import threading
    import time

    from PySide6.QtCore import QThread

    class Worker(QThread):
        """Like tools_analysis._Worker: a QThread child of its dialog with a `cancelled` flag."""

        def __init__(self, parent, obeys_cancel):
            super().__init__(parent)
            self.cancelled = False
            self.obeys_cancel = obeys_cancel
            self.release = threading.Event()

        def run(self):
            while not self.release.is_set():
                if self.obeys_cancel and self.cancelled:
                    return
                time.sleep(0.005)

    return Worker


def test_close_stops_running_analysis_workers(qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox

    from flowdesk.ui.main_window import MainWindow

    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(a) or QMessageBox.StandardButton.No))
    win = MainWindow()
    stops = []

    class Ext:
        def stop_all(self):
            stops.append(True)

    win.extensions["test.ext"] = Ext()
    win.extensions["test.plain"] = {"action": None}  # extensions without stop_all are skipped
    worker = _worker_class()(QDialog(win), obeys_cancel=True)
    worker.start()
    try:
        assert worker.isRunning()
        assert win.close() is True
        assert stops == [True]
        assert not worker.isRunning() and worker.cancelled  # stopped before the window can be destroyed
        assert asked == []
    finally:
        worker.release.set()
        worker.wait(5000)


def test_close_asks_when_an_analysis_does_not_stop(qapp, monkeypatch):
    import time

    from PySide6.QtWidgets import QDialog, QMessageBox

    from flowdesk.ui import main_window as mw

    answers = [QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes]
    asked = []

    def question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return answers.pop(0)

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    win = mw.MainWindow()
    win.WORKER_STOP_TIMEOUT_MS = 150
    dlg = QDialog(win)
    worker = _worker_class()(dlg, obeys_cancel=False)
    worker.start()
    try:
        t0 = time.monotonic()
        assert win.close() is False  # "No": the window stays
        assert 0.1 <= time.monotonic() - t0 < 2.0
        assert asked == ["An analysis is still running. Quit anyway?"]
        assert worker.isRunning() and worker.parent() is dlg
        assert win.close() is True  # "Quit anyway"
        assert len(asked) == 2
        # detached from the window so destroying it cannot destroy a running thread
        assert worker.parent() is None and worker in mw.abandoned_threads()
    finally:
        worker.release.set()
        worker.wait(5000)
    assert worker not in mw.abandoned_threads()


def test_quit_with_unstoppable_analysis_exits_cleanly(tmp_path):
    """Before: 'QThread: Destroyed while thread is still running' and SIGABRT (exit 134)."""
    script = tmp_path / "quit_running.py"
    script.write_text(
        "import os, sys, time\n"
        "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from PySide6.QtCore import QThread, QTimer\n"
        "from PySide6.QtWidgets import QDialog, QMessageBox\n"
        "import flowdesk.ui.main_window as mw\n"
        "QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)\n"
        "mw.MainWindow.WORKER_STOP_TIMEOUT_MS = 100\n"
        "class Stubborn(QThread):\n"
        "    def run(self):\n"
        "        time.sleep(300)  # never looks at a cancellation flag\n"
        "orig = mw.MainWindow.__init__\n"
        "def init(self, *a, **k):\n"
        "    orig(self, *a, **k)\n"
        "    def go():\n"
        "        self._worker = Stubborn(QDialog(self))\n"
        "        self._worker.start()\n"
        "        QTimer.singleShot(50, self.close)\n"
        "    QTimer.singleShot(0, go)\n"
        "mw.MainWindow.__init__ = init\n"
        "from flowdesk.app import main\n"
        "sys.exit(main(['flowdesk']))\n",
        encoding="utf-8",
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
    assert "Destroyed while thread" not in out.stderr


def test_file_open_event_waits_while_a_modal_dialog_is_up(qapp, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFileOpenEvent
    from PySide6.QtWidgets import QApplication, QDialog

    from flowdesk.app import _FileOpenFilter
    from flowdesk.ui.main_window import MainWindow

    fcs = _synthetic_fcs(tmp_path / "late.fcs")
    win = MainWindow()
    flt = _FileOpenFilter(win)
    modal = QDialog(win)
    modal.setWindowModality(Qt.WindowModality.WindowModal)  # like an analysis' progress dialog
    modal.show()
    try:
        assert QApplication.activeModalWidget() is modal
        # handled (True), but deferred: nothing is opened from inside the event filter
        assert flt.eventFilter(qapp, QFileOpenEvent(str(fcs))) is True
        assert win.controller.workspace.samples == []
        # still waiting while the modal dialog is up (past several retry intervals)
        assert not _process_until(qapp, lambda: bool(win.controller.workspace.samples), timeout=3 * flt.RETRY_MS / 1000)
        assert flt.pending() == [fcs]
        modal.close()
        assert _process_until(qapp, lambda: bool(win.controller.workspace.samples))
        assert [s.name for s in win.controller.workspace.samples] == ["late.fcs"]
        assert flt.pending() == []
        # files it does not open are not swallowed
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        assert flt.eventFilter(qapp, QFileOpenEvent(str(tmp_path / "notes.txt"))) is False
        assert flt.eventFilter(qapp, QFileOpenEvent(str(tmp_path / "missing.fdw"))) is False
        assert flt.pending() == []
    finally:
        modal.close()
        win.controller.workspace.modified = False
        win.close()


def test_broken_extension_module_does_not_prevent_the_main_window(qapp, tmp_path, monkeypatch, capsys):
    from flowdesk.ui.main_window import MainWindow

    (tmp_path / "fdtest_ext_attr.py").write_text("import flowdesk.core.analysis as A\nX = A.NO_SUCH_NAME\ndef install(w):\n    return 1\n", encoding="utf-8")
    (tmp_path / "fdtest_ext_syntax.py").write_text("def install(w)\n    return 1\n", encoding="utf-8")
    (tmp_path / "fdtest_ext_dep.py").write_text("import fdtest_no_such_dependency\n", encoding="utf-8")
    (tmp_path / "fdtest_ext_install.py").write_text("def install(w):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    (tmp_path / "fdtest_ext_ok.py").write_text("def install(w):\n    return 'installed'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    added = ("fdtest_ext_attr", "fdtest_ext_syntax", "fdtest_ext_dep", "fdtest_ext_install", "fdtest_ext_missing", "fdtest_ext_ok")
    monkeypatch.setattr(MainWindow, "EXTENSION_MODULES", MainWindow.EXTENSION_MODULES + added)
    win = MainWindow()
    try:
        assert win.extensions["fdtest_ext_ok"] == "installed"
        assert "flowdesk.ui.cytometer" in win.extensions  # the real tools still load
        errors = win.extension_errors
        assert set(errors) == {"fdtest_ext_attr", "fdtest_ext_syntax", "fdtest_ext_dep", "fdtest_ext_install"}
        assert errors["fdtest_ext_attr"].startswith("AttributeError")
        assert errors["fdtest_ext_syntax"].startswith("SyntaxError")
        assert errors["fdtest_ext_dep"].startswith("ModuleNotFoundError")
        assert errors["fdtest_ext_install"] == "RuntimeError: boom"
        err = capsys.readouterr().err
        assert "Traceback" in err and "fdtest_ext_attr" in err and "NO_SUCH_NAME" in err
    finally:
        win.close()


def test_missing_computed_data_folder_is_reported_and_can_be_located(qapp, tmp_path, monkeypatch):
    import shutil

    import numpy as np
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from flowdesk.core.gates import RectangleGate
    from flowdesk.core.model import Population, Workspace
    from flowdesk.core.serialization import save_workspace
    from flowdesk.ui import main_window as mw

    ws = Workspace()
    s = ws.add_sample(_synthetic_fcs(tmp_path / "a.fcs", n=300))
    s.ensure_loaded()
    values = np.linspace(0.0, 1.0, s.n_events).astype(np.float32)
    assert not np.any(values == 0.5)
    expected = int(np.count_nonzero(values < 0.5))
    s.add_computed("tSNE 1", values, {"tool": "tSNE"})
    s.root.add_child(Population("left", RectangleGate(["tSNE 1"], [None], [0.5])))
    save_workspace(ws, tmp_path / "exp.fdw")
    # renamed in the file manager: the exp.fdw.data folder did not come along
    shutil.copy(tmp_path / "exp.fdw", tmp_path / "exp final.fdw")

    real_load = mw.load_workspace

    def load(path, *args, **kwargs):  # independent of whether the loader reports it itself
        out = real_load(path, *args, **kwargs)
        if not mw._workspace_load_warnings(out) and not out.samples[0].computed:
            out.load_warnings = ["a.fcs: computed parameter 'tSNE 1' was not loaded: the data folder exp final.fdw.data is missing"]
        return out

    monkeypatch.setattr(mw, "load_workspace", load)
    boxes = []
    choice = ["Continue"]

    def exec_(box):
        boxes.append(box.text() + "\n" + box.informativeText())
        next(b for b in box.buttons() if b.text().startswith(choice[0])).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    picked = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: picked.append(a) or str(tmp_path / "exp.fdw.data")))
    win = mw.MainWindow()
    try:
        win._open_path(str(tmp_path / "exp final.fdw"))
        assert len(boxes) == 1 and picked == []
        assert "exp final.fdw.data" in boxes[0] and "tSNE 1" in boxes[0]
        assert win.controller.workspace.samples[0].computed == {}

        choice[0] = "Locate"
        win._open_path(str(tmp_path / "exp final.fdw"))
        assert len(boxes) == 2 and len(picked) == 1
        s2 = win.controller.workspace.samples[0]
        np.testing.assert_array_equal(s2.computed["tSNE 1"], values)
        assert s2.root.find("/left").count == expected
        assert win.controller.workspace.modified  # saving stores the arrays next to "exp final.fdw"
        save_workspace(win.controller.workspace, tmp_path / "exp final.fdw")
        assert (tmp_path / "exp final.fdw.data").is_dir()
    finally:
        win.controller.workspace.modified = False
        win.close()
