"""Application entry point."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

from . import APP_NAME, __version__


def create_app(argv: list[str] | None = None) -> QApplication:
    QCoreApplication.setOrganizationName("FlowDesk")
    QCoreApplication.setApplicationName(APP_NAME)
    QCoreApplication.setApplicationVersion(__version__)
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    from .ui.style import apply_theme

    apply_theme(app)
    from .ui.icons import icon

    app.setWindowIcon(icon("app"))
    return app


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = create_app(argv)
    from .ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    for arg in argv[1:]:
        p = Path(arg)
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
