"""FlowDesk without a window: a controller and the agent bridge, for servers and for ``launch(mode="headless")``.

``python -m flowdesk --agent-headless [workspace.fdw | files...]``.  Everything the tools do works except what needs a
window (graph windows, dialogs); plots still render, because rendering is offscreen.
"""
from __future__ import annotations

import os
import signal
import sys
from pathlib import Path


def run_headless(paths=()) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QTimer
    from PySide6.QtWidgets import QApplication

    from .. import APP_NAME, __version__
    from ..core.model import Workspace
    from ..ui.controller import WorkspaceController
    from .hostapp import AgentHost

    QCoreApplication.setOrganizationName("FlowDesk")
    QCoreApplication.setApplicationName(APP_NAME)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    controller = WorkspaceController(Workspace())
    host = AgentHost(controller, window=None)
    for p in paths:
        p = Path(p)
        try:
            if p.suffix.lower() in (".fdw", ".wsp"):
                host.ops.call("open_workspace", {"path": str(p), "discard_changes": True})
            elif p.suffix.lower() == ".fcs":
                host.ops.call("add_samples", {"paths": [str(p)]})
            elif p.is_dir():
                host.ops.call("add_samples", {"folder": str(p)})
        except Exception as exc:  # noqa: BLE001
            print(f"FlowDesk headless: could not open {p}: {exc}", file=sys.stderr)
    host.start()
    app.aboutToQuit.connect(host.stop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda *_: app.quit())
        except (ValueError, OSError):
            pass
    tick = QTimer()  # lets Python signal handlers run inside the Qt loop
    tick.start(250)
    tick.timeout.connect(lambda: None)
    print(f"FlowDesk {__version__} headless: bridge on 127.0.0.1:{host.bridge.port} (pid {os.getpid()})", flush=True)
    return app.exec()
