"""Application entry point (cross-platform: macOS, Windows, Linux)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # started as a plain script: ``python flowdesk/app.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "flowdesk"  # noqa: A001 - lets the relative imports below resolve

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from . import APP_NAME, __version__

APP_USER_MODEL_ID = "FlowDesk.FlowDesk"

USAGE = f"""usage: flowdesk [-h] [--version] [--agent-headless] [--mcp] [FILE_OR_FOLDER ...]

{APP_NAME} {__version__} - flow cytometry analysis workspace.

positional arguments:
  FILE_OR_FOLDER  .fdw / .wsp workspaces to open, .fcs files or folders to add

options:
  -h, --help      show this help message and exit
  --version       print the version and exit
  --agent-headless  run without a window, serving only the agent bridge (Claude Code / Codex)
  --mcp           run the FlowDesk MCP server on stdin/stdout (what flowdesk-mcp does; used by the packaged app)
"""


def _ensure_std_streams() -> None:
    """pythonw.exe / windowed PyInstaller builds have no console: sys.stdout/stderr are None."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity so Windows shows the FlowDesk icon (not python's)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def _configure_before_qapplication() -> None:
    """Settings that only take effect if applied before the QApplication exists."""
    if QCoreApplication.instance() is not None:
        return
    # Fractional scaling (125 %, 150 % on Windows / Linux) without rounding to whole factors.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    _set_windows_app_id()


def create_app(argv: list[str] | None = None) -> QApplication:
    QCoreApplication.setOrganizationName("FlowDesk")
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion(__version__)
    _configure_before_qapplication()
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    if hasattr(app, "setDesktopFileName"):
        app.setDesktopFileName("flowdesk")  # matches packaging/flowdesk.desktop (Wayland/X11 icon)
    from .ui.style import apply_theme

    apply_theme(app)
    from .ui.icons import icon

    app.setWindowIcon(icon("app"))
    return app


def main(argv: list[str] | None = None) -> int:
    _ensure_std_streams()
    argv = list(sys.argv if argv is None else argv)
    args = argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(USAGE, end="")
        return 0
    if "--version" in args:
        print(f"{APP_NAME} {__version__}")
        return 0
    if "--mcp" in args:
        from .agent.server import main as mcp_main

        mcp_main()
        return 0
    if "--agent-headless" in args:
        from .agent.headless import run_headless

        paths = [Path(a).expanduser().absolute() for a in args if not a.startswith("-")]
        return run_headless(paths)
    # Resolve file arguments against the launch directory (relative paths, any cwd / drive).
    paths = [Path(a).expanduser().absolute() for a in args if not a.startswith("-")]
    app = create_app(argv)
    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    for p in paths:
        open_path_in_window(window, p)
    # macOS delivers files opened by double-click / drag onto the Dock icon as
    # QFileOpenEvent instead of command-line arguments (Finder, Launch Services)
    file_filter = _FileOpenFilter(window)
    app.installEventFilter(file_filter)
    rc = app.exec()
    from .ui.main_window import abandoned_threads

    if abandoned_threads():
        # the user quit while an analysis could not be stopped: Qt aborts the
        # process when a running QThread is destroyed, so leave without teardown
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        os._exit(rc)
    return rc


WORKSPACE_SUFFIXES = (".fdw", ".wsp", ".fdwt", ".wspt")


def can_open_path(p: Path) -> bool:
    """True for what open_path_in_window handles: an existing workspace /
    template, FCS file or folder."""
    p = Path(p)
    if not p.exists():
        return False
    return p.suffix.lower() in WORKSPACE_SUFFIXES or p.suffix.lower() == ".fcs" or p.is_dir()


def open_path_in_window(window, p: Path) -> bool:
    """Open a workspace / template, add an FCS file or a folder. Returns True if handled."""
    p = Path(p)
    if not can_open_path(p):
        return False
    suffix = p.suffix.lower()
    if suffix in WORKSPACE_SUFFIXES:
        window._open_path(str(p))
    elif suffix == ".fcs":
        window.controller.add_samples([p])
    elif p.is_dir():
        window.controller.add_folder(p)
    else:
        return False
    window.raise_()
    window.activateWindow()
    return True


class _FileOpenFilter(QObject):
    """Opens files the OS hands to the running app (macOS Finder double-click,
    drop onto the Dock icon). The open is deferred out of the event filter and
    waits while a modal dialog (an analysis' progress, a wizard, Save As ...)
    is up: replacing the workspace under it would pull the model from beneath
    the running analysis or the open dialog."""

    RETRY_MS = 200  # how often a queued file checks whether the modal dialog has closed

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._pending: list[Path] = []
        self._busy = False
        self._retry = QTimer(self)
        self._retry.setInterval(self.RETRY_MS)
        self._retry.timeout.connect(self._flush)

    def pending(self) -> list[Path]:
        return list(self._pending)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        if event.type() == QEvent.Type.FileOpen:
            path = event.file()
            if path and can_open_path(Path(path)):
                self._pending.append(Path(path))
                QTimer.singleShot(0, self._flush)  # never run dialogs inside an event filter
                return True
            return False  # not ours: let Qt handle (ignore) it
        return super().eventFilter(obj, event)

    def _flush(self):
        if self._busy:
            return  # an open is in progress (e.g. its maybe_save question): the timer retries
        if QApplication.activeModalWidget() is not None:
            if self._pending and not self._retry.isActive():
                self._retry.start()
                try:
                    names = ", ".join(p.name for p in self._pending)
                    self._window.statusBar().showMessage(f"{names} will open when the dialog closes", 5000)
                except (AttributeError, RuntimeError):
                    pass
            return
        self._retry.stop()
        self._busy = True
        try:
            while self._pending and QApplication.activeModalWidget() is None:
                open_path_in_window(self._window, self._pending.pop(0))
        finally:
            self._busy = False
        if self._pending:  # a modal dialog came up while opening: wait again
            self._retry.start()


if __name__ == "__main__":
    sys.exit(main())
