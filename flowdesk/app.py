"""Application entry point (cross-platform: macOS, Windows, Linux)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # started as a plain script: ``python flowdesk/app.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "flowdesk"  # noqa: A001 - lets the relative imports below resolve

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from . import APP_NAME, __version__

APP_USER_MODEL_ID = "FlowDesk.FlowDesk"

USAGE = f"""usage: flowdesk [-h] [--version] [FILE_OR_FOLDER ...]

{APP_NAME} {__version__} - flow cytometry analysis workspace.

positional arguments:
  FILE_OR_FOLDER  .fdw / .wsp workspaces to open, .fcs files or folders to add

options:
  -h, --help      show this help message and exit
  --version       print the version and exit
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
    # Resolve file arguments against the launch directory (relative paths, any cwd / drive).
    paths = [Path(a).expanduser().absolute() for a in args if not a.startswith("-")]
    app = create_app(argv)
    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    for p in paths:
        if not p.exists():
            continue
        if p.suffix.lower() in (".fdw", ".wsp"):
            window._open_path(str(p))
        elif p.suffix.lower() == ".fcs":
            window.controller.add_samples([p])
        elif p.is_dir():
            window.controller.add_folder(p)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
