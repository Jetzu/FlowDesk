"""Agent integration: the bridge that lets Claude Code / Codex / Claude Desktop drive this window, and the Agent Chat dock.

Installed like the other extension modules.  The bridge (localhost, token-protected, discovery file under ~/.flowdesk) is
started with the window unless switched off in the Agent menu (stored in QSettings ``agent/bridge``) or by the environment
variable ``FLOWDESK_BRIDGE=0``; ``FLOWDESK_BRIDGE=1`` forces it on.
"""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QDockWidget, QMessageBox

from .icons import icon


def bridge_wanted() -> bool:
    env = os.environ.get("FLOWDESK_BRIDGE")
    if env is not None and env != "":
        return env.strip().lower() not in ("0", "false", "no", "off")
    return QSettings("FlowDesk", "FlowDesk").value("agent/bridge", True, type=bool)


class AgentExtension:
    def __init__(self, window):
        from ..agent import agentcli
        from ..agent.hostapp import AgentHost

        self.window = window
        self.cli = agentcli
        self.sessions = {"claude": None, "codex": None}
        self.dock: QDockWidget | None = None
        self.host = AgentHost(window.controller, window)
        self.actions: dict[str, QAction] = {}
        self._build_actions()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop_all)
        if bridge_wanted():
            self.start_bridge(quiet=True)

    # ------------------------------------------------------------------ bridge
    def start_bridge(self, quiet: bool = False) -> bool:
        try:
            self.host.start()
        except Exception as exc:  # noqa: BLE001 - port busy, read-only home ...
            if not quiet:
                QMessageBox.warning(self.window, "Agent bridge", f"The bridge could not be started:\n{exc}")
            self.window.statusBar().showMessage(f"Agent bridge not started: {exc}", 8000)
            self._sync()
            return False
        self._sync()
        return True

    def stop_bridge(self):
        self.host.stop()
        self._sync()

    def stop_all(self):
        try:
            self.host.stop()
        except Exception:  # noqa: BLE001
            pass
        if self.dock is not None:
            chat = getattr(self.dock, "chat", None)
            run = getattr(chat, "run", None)
            if run is not None:
                run.cancel()

    def toggle_bridge(self, on: bool):
        QSettings("FlowDesk", "FlowDesk").setValue("agent/bridge", bool(on))
        if on and not self.host.running:
            self.start_bridge()
        elif not on and self.host.running:
            self.stop_bridge()

    def status_text(self) -> str:
        if self.host.running:
            b = self.host.bridge
            return f"Agent bridge listening on 127.0.0.1:{b.port} (pid {os.getpid()}); Claude Code, Codex and Claude Desktop can connect"
        return "Agent bridge is off"

    # --------------------------------------------------------------------- UI
    def _act(self, key, text, icon_name, tip, slot, checkable=False, shortcut=None):
        a = QAction(icon(icon_name), text, self.window)
        a.setToolTip(tip)
        a.setStatusTip(tip)
        a.setCheckable(checkable)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if checkable:
            a.toggled.connect(lambda on: slot(on))
        else:
            a.triggered.connect(lambda _=False: slot())
        self.actions[key] = a
        return a

    def _build_actions(self):
        self._act("chat", "Agent Chat", "agent", "Chat with Claude Code or Codex about this workspace: they gate, measure and plot in this window (every change is one undo step)", self.toggle_chat, shortcut="Ctrl+Shift+A")
        self._act("bridge", "Agent Bridge", "agent-bridge", "Let Claude Code, Codex and Claude Desktop connect to this FlowDesk (flowdesk-mcp)", self.toggle_bridge, checkable=True)
        self._act("setup", "Connect Claude / Codex…", "agent-connect", "Register the FlowDesk MCP server with Claude Code, Codex and Claude Desktop", self.setup_dialog)
        a = self.actions["bridge"]
        a.blockSignals(True)
        a.setChecked(self.host.running)
        a.blockSignals(False)
        w = self.window
        page = getattr(w, "ribbon_pages", {}).get("tools")
        if page is not None:
            band = page.add_band("Agent")
            band.add_large(self.actions["chat"])
            band.add_small(self.actions["bridge"])
            band.add_small(self.actions["setup"])
        menu = getattr(w, "menus", {}).get("tools")
        if menu is not None:
            menu.addSeparator()
            sub = menu.addMenu("Agent")
            for k in ("chat", "bridge", "setup"):
                sub.addAction(self.actions[k])

    def _sync(self):
        a = self.actions.get("bridge")
        if a is not None:
            a.blockSignals(True)
            a.setChecked(self.host.running)
            a.blockSignals(False)
            a.setToolTip(self.status_text())

    def start_run(self, backend, prompt, callback):
        def cb(kind, payload):
            if kind == "session":
                self.sessions[backend] = payload
            callback(kind, payload)

        if not self.host.running:
            self.start_bridge()
        cwd = None
        ws = getattr(self.window.controller, "workspace", None)
        path = getattr(ws, "path", None)
        if path and os.path.isdir(os.path.dirname(str(path))):
            cwd = os.path.dirname(str(path))
        run = self.cli.AgentRun(backend, prompt, "flowdesk", os.getpid(), cb, session_id=self.sessions.get(backend), cwd=cwd or os.path.expanduser("~"))
        return run.start()

    def ensure_dock(self) -> QDockWidget:
        if self.dock is None:
            from PySide6 import QtCore, QtGui, QtWidgets

            from ..agent import chat_ui

            widget = chat_ui.make_chat_widget(QtWidgets, QtCore, QtGui, app="flowdesk", pid=os.getpid(), start_run=self.start_run,
                                              sessions=self.sessions, cwd=os.path.expanduser("~"), find_backend=self.cli.find_backend,
                                              title="Agent Chat for FlowDesk", agentcli=self.cli)
            dock = QDockWidget("Agent Chat (Claude Code / Codex)", self.window)
            dock.setObjectName("flowdesk_agent_chat")
            dock.setWidget(widget)
            dock.chat = widget
            self.window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            dock.resize(520, 680)
            self.dock = dock
        return self.dock

    def toggle_chat(self):
        if self.dock is not None and self.dock.isVisible():
            self.dock.hide()
            return
        self.open_chat()

    def open_chat(self) -> QDockWidget:
        dock = self.ensure_dock()
        if not self.host.running:
            self.start_bridge()
        dock.show()
        dock.raise_()
        dock.chat.prompt.setFocus()
        return dock

    def ask(self, backend: str, prompt: str):
        dock = self.open_chat()
        dock.chat.set_backend(backend)
        dock.chat.submit(prompt)
        return dock.chat.run

    def setup_dialog(self):
        from ..agent import install

        try:
            report = install.check()
        except Exception as exc:  # noqa: BLE001
            report = {"error": str(exc)}
        text = install.describe(report)
        box = QMessageBox(self.window)
        box.setWindowTitle("Connect Claude / Codex")
        box.setText("Register the FlowDesk MCP server with the agents installed on this computer?")
        box.setInformativeText(text)
        install_btn = box.addButton("Register", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is install_btn:
            try:
                res = install.install()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self.window, "Connect Claude / Codex", str(exc))
                return
            QMessageBox.information(self.window, "Connect Claude / Codex", install.describe(res) + "\n\nRestart Claude Desktop (if used); Claude Code and Codex pick it up in new sessions.")


def install(window):
    return AgentExtension(window)
