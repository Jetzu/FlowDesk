"""Hosts the bridge inside a running FlowDesk (window or headless).

Bridge requests arrive on HTTP threads; every operation is executed on the Qt GUI thread (Qt objects must only be touched
there) and the HTTP thread waits for the result.  Controller signals are turned into events that tell the agent what the
*user* changed between its calls.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication

from . import APP
from .bridgecore import Bridge, EventLog
from .ops import Ops

OP_TIMEOUT = 1800.0  # seconds a single operation may take on the GUI thread (t-SNE on many events, ...)


class GuiThreadInvoker(QObject):
    """Runs callables on the thread this object was created in (the GUI thread) and hands back the result."""

    _call = Signal(object)

    def __init__(self):
        super().__init__()
        self._call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def _run(self, item):
        fn, fut = item
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(fn())
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiting HTTP thread
            fut.set_exception(exc)

    def invoke(self, fn, timeout: float | None = OP_TIMEOUT):
        if QThread.currentThread() is self.thread():
            return fn()
        fut: Future = Future()
        self._call.emit((fn, fut))
        return fut.result(timeout)


class DirectInvoker:
    """Executes in the calling thread (tests and headless scripts without an event loop)."""

    def invoke(self, fn, timeout=None):
        return fn()


class AgentHost:
    def __init__(self, controller, window=None, name: str | None = None, invoker=None):
        from .. import __version__

        self.c = controller
        self.window = window
        self.invoker = invoker or GuiThreadInvoker()
        self.events = EventLog()
        self.ops = Ops(controller, self.events, window)
        self._in_agent_op = False
        self._last_event: tuple[str, float] | None = None
        self.bridge = Bridge(APP, __version__, self._dispatch, info=self._info)
        if name:
            self.bridge.name = name
        self._hook_signals()

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        return self.bridge.start()

    def stop(self):
        self.bridge.stop()

    @property
    def running(self) -> bool:
        return self.bridge.running

    # ---------------------------------------------------------------- dispatching
    def _info(self) -> dict:
        return {"gui": self.window is not None, "app_version": self.bridge.version}

    def _dispatch(self, op: str, args: dict):
        def run():
            self._in_agent_op = True
            try:
                if op == "quit":
                    return self._quit(**args)
                if op == "name":
                    return {"name": self.bridge.set_name(args.get("name", ""))}
                result = self.ops.call(op, args)
                if op == "ping":
                    result.update({"name": self.bridge.name, "window_active": self.bridge.window_active,
                                   "last_activity": self.bridge.last_activity, "pid": os.getpid(),
                                   "version": self.bridge.version})
                return result
            finally:
                self._in_agent_op = False

        return self.invoker.invoke(run)

    def _quit(self, discard_changes: bool = False) -> dict:
        if self.c.workspace.modified and not discard_changes:
            from .ops_base import OpError

            raise OpError("the workspace has unsaved changes; save_workspace first or repeat with discard_changes=true if the user agrees")
        self.c.workspace.modified = False
        app = QApplication.instance()
        QTimer.singleShot(200, (self.window.close if self.window is not None else app.quit))
        return {"ok": True, "quitting": True}

    # --------------------------------------------------------------------- events
    def _user_event(self, what: str, **data):
        """Record a change the user (not the agent) made; bursts of the same kind collapse into one event."""
        if self._in_agent_op:
            return
        now = time.time()
        if self._last_event and self._last_event[0] == what and now - self._last_event[1] < 0.5:
            self._last_event = (what, now)
            return
        self._last_event = (what, now)
        self.bridge.touch()
        self.events.add("user_" + what, **data)

    def _hook_signals(self):
        c = self.c
        for name, label in (("gatingChanged", "gating"), ("samplesChanged", "samples"), ("groupsChanged", "groups"),
                            ("scalesChanged", "scales"), ("compensationChanged", "compensation"), ("layoutsChanged", "layouts"),
                            ("tablesChanged", "tables"), ("workspaceReplaced", "workspace_replaced"), ("settingsChanged", "settings")):
            sig = getattr(c, name, None)
            if sig is not None:
                sig.connect(lambda *_a, label=label: self._user_event(label))
        sel = getattr(c, "selectionChanged", None)
        if sel is not None:
            sel.connect(self._on_selection)
        app = QApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(lambda st: setattr(self.bridge, "window_active", st == Qt.ApplicationState.ApplicationActive))
            self.bridge.window_active = app.applicationState() == Qt.ApplicationState.ApplicationActive

    def _on_selection(self):
        sel = [(s.name, p.path or "/") for s, p in self.c.selected][:5]
        self._user_event("selection", selected=sel)
