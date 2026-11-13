"""The agent integration around the operations: bridge transport, MCP server, installer, CLI runner, chat widget, in-app extension."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs"))
import make_testset  # noqa: E402

from flowdesk.agent import agentcli, discovery, install  # noqa: E402
from flowdesk.agent.client import BridgeClient, BridgeError, NotConnected  # noqa: E402
from flowdesk.agent.hostapp import AgentHost, DirectInvoker  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture(scope="module")
def testset(tmp_path_factory):
    out = tmp_path_factory.mktemp("bridge_testset")
    make_testset.write_testset(out, seed=7, n=2000)
    return out


@pytest.fixture
def host(app, testset):
    from flowdesk.core.serialization import load_workspace
    from flowdesk.ui.controller import WorkspaceController

    c = WorkspaceController(load_workspace(testset / "FlowDesk testset.fdw"))
    c.workspace.modified = False
    h = AgentHost(c, invoker=DirectInvoker())
    h.start()
    yield h
    h.stop()


def _wait_threads(app, thread, limit=60.0):
    """Spin a real Qt event loop until the worker thread is done (QTest.qWait would hold the GIL)."""
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: loop.quit() if not thread.is_alive() else None)
    timer.start(20)
    QTimer.singleShot(int(limit * 1000), loop.quit)
    loop.exec()
    timer.stop()
    thread.join(1)


# ---------------------------------------------------------------- bridge transport
def test_bridge_roundtrip_discovery_and_auth(host):
    insts = discovery.list_instances("flowdesk")
    mine = [i for i in insts if i["pid"] == os.getpid()]
    assert mine and mine[0]["port"] == host.bridge.port
    client = BridgeClient("flowdesk")
    info = client.connect(pid=os.getpid())
    assert info["port"] == host.bridge.port
    pong = client.call("ping")
    assert pong["samples"] == 6 and pong["pid"] == os.getpid() and pong["workspace"] == "FlowDesk testset"
    with pytest.raises(BridgeError, match="unknown"):
        client.call("no_such_op")
    bad = BridgeClient("flowdesk")
    bad.connect(pid=os.getpid())
    bad.info = dict(bad.info, token="wrong")
    with pytest.raises(Exception):
        bad.call("ping")
    host.stop()
    assert not [i for i in discovery.list_instances("flowdesk") if i["pid"] == os.getpid()]


def test_discovery_file_is_private(host):
    files = [Path(discovery.instances_dir()) / f for f in os.listdir(discovery.instances_dir())]
    mine = [f for f in files if f.name.endswith("%d.json" % os.getpid())]
    assert mine
    if os.name == "posix":
        assert oct(mine[0].stat().st_mode & 0o777) == "0o600"


def test_quit_guard_and_named_window(host):
    client = BridgeClient("flowdesk")
    client.connect(pid=os.getpid())
    assert client.call("name", name="analysis A")["name"] == "analysis A"
    host.c.workspace.modified = True
    with pytest.raises(BridgeError, match="unsaved"):
        client.call("quit")
    host.c.workspace.modified = False


# ---------------------------------------------------------------- MCP server
def _text(res) -> str:
    content = getattr(res, "content", None)
    if content is None and isinstance(res, tuple):
        content = res[0]
    if content is None:
        content = res
    return "\n".join(getattr(c, "text", "") for c in content if getattr(c, "type", "") == "text")


def _call(server, tool, **args):
    return asyncio.run(server.mcp.call_tool(tool, args))


def test_mcp_tools_registered_and_documented():
    from flowdesk.agent import server

    tools = asyncio.run(server.mcp.list_tools())
    names = {t.name for t in tools}
    expected = {"status", "launch", "connect", "workspace", "sample_info", "populations", "create_gate", "edit_gate", "statistics", "histogram",
                "plot", "compensation", "dimension_reduction", "cluster", "cell_cycle", "proliferation", "data_cleaning", "export_events",
                "export_statistics", "export_flowjo_workspace", "undo", "redo", "python", "events", "log", "skills", "skill", "save_skill", "quit_flowdesk"}
    assert expected <= names
    assert all((t.description or "").strip() for t in tools)


def test_mcp_tool_calls_reach_the_bridge(host, monkeypatch):
    from flowdesk.agent import server

    monkeypatch.setenv("FLOWDESK_PID", str(os.getpid()))
    monkeypatch.setenv("FLOWDESK_AUTOLAUNCH", "0")
    server.SESSION.client = BridgeClient("flowdesk")
    st = json.loads(_text(_call(server, "status")))
    assert st["connected"] and st["connected_pid"] == os.getpid()
    ws = json.loads(_text(_call(server, "workspace")))
    assert len(ws["samples"]) == 6
    g = json.loads(_text(_call(server, "create_gate", name="Scatter", shape="rectangle", x="FSC-A", y="SSC-A", group="All Samples",
                               x_range=[20000, 200000], y_range=[0, 150000])))
    assert g.get("ok", True) and "error" not in g
    stats = json.loads(_text(_call(server, "statistics", populations="/Scatter", statistics=["Count"])))
    assert len(stats["rows"]) == 6
    bad = json.loads(_text(_call(server, "create_gate", name="Scatter", shape="rectangle", x="FSC-A", y="SSC-A", group="All Samples", x_range=[0, 1])))
    assert bad["ok"] is False and "already has a child" in bad["error"] and not bad["error"].startswith("OpError")
    undone = json.loads(_text(_call(server, "undo")))
    assert undone["ok"]


def test_mcp_plot_returns_png_and_numbers(host, monkeypatch):
    from flowdesk.agent import server

    monkeypatch.setenv("FLOWDESK_PID", str(os.getpid()))
    server.SESSION.client = BridgeClient("flowdesk")
    res = _call(server, "plot", sample="Donor1_Control", x="FSC-A", y="SSC-A")
    content = res[0] if isinstance(res, tuple) else getattr(res, "content", res)
    kinds = [getattr(c, "type", "") for c in content]
    assert "text" in kinds and "image" in kinds
    desc = json.loads(next(c.text for c in content if c.type == "text"))
    assert desc["events"] == 2000 and os.path.exists(desc["image"]["path"])


def test_pinned_pid_that_is_gone_is_a_clear_error(monkeypatch):
    from flowdesk.agent import server

    monkeypatch.setenv("FLOWDESK_PID", "999999")
    monkeypatch.setenv("FLOWDESK_AUTOLAUNCH", "0")
    server.SESSION.client = BridgeClient("flowdesk")
    out = json.loads(_text(_call(server, "workspace")))
    assert out["ok"] is False and "no longer running" in out["error"]


def test_skills_listed_and_readable():
    from flowdesk.agent import server

    listing = json.loads(_text(_call(server, "skills")))["skills"]
    names = {s["name"] for s in listing}
    assert {"gating-strategy", "compensation", "batch-analysis", "python-api"} <= names
    body = _text(_call(server, "skill", name="gating-strategy"))
    assert "histogram" in body and "create_gate" in body
    assert json.loads(_text(_call(server, "skill", name="nope")))["ok"] is False


def test_every_skill_mentions_only_real_tools():
    from flowdesk.agent import server

    tools = {t.name for t in asyncio.run(server.mcp.list_tools())}
    import re

    for name, path in server._skills().items():
        text = open(path, encoding="utf-8").read()
        for m in re.finditer(r"`(\w+)\(", text):
            fn = m.group(1)
            if fn in ("ops", "np", "float", "int", "sum", "median", "mask", "data", "param", "population", "sample", "call", "range", "len", "print", "ws", "controller"):
                continue
            assert fn in tools, "%s mentions unknown tool %s()" % (name, fn)


# ---------------------------------------------------------------- installer
@pytest.fixture
def fakehome(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    return tmp_path


def test_installer_registers_all_clients_idempotently_and_removes(fakehome):
    (fakehome / ".codex").mkdir()
    (fakehome / ".codex" / "config.toml").write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "a"\n')
    (fakehome / ".claude.json").write_text(json.dumps({"mcpServers": {"pymol": {"command": "p"}}, "numStartups": 3}))
    rep = install.install(["claude", "codex", "claude_desktop"])
    assert all(rep["clients"][c]["registered"] and rep["clients"][c]["current"] for c in install.CLIENTS)
    install.install(["claude", "codex", "claude_desktop"])
    claude = json.loads((fakehome / ".claude.json").read_text())
    assert claude["numStartups"] == 3 and "pymol" in claude["mcpServers"]
    assert claude["mcpServers"]["flowdesk"]["command"] == install.server_entry()[0]
    toml = (fakehome / ".codex" / "config.toml").read_text()
    assert toml.count("[mcp_servers.flowdesk]") == 1 and "[mcp_servers.other]" in toml
    try:
        import tomllib

        data = tomllib.loads(toml)
        assert data["mcp_servers"]["flowdesk"]["command"] == install.server_entry()[0]
    except ImportError:
        pass
    assert (fakehome / ".claude.json.flowdesk-backup").exists()
    desktop = json.loads(Path(install.claude_desktop_config_path()).read_text())
    assert "flowdesk" in desktop["mcpServers"]
    install.install(["claude", "codex", "claude_desktop"], remove=True)
    assert "flowdesk" not in json.loads((fakehome / ".claude.json").read_text())["mcpServers"]
    assert "flowdesk" not in (fakehome / ".codex" / "config.toml").read_text()
    assert "[mcp_servers.other]" in (fakehome / ".codex" / "config.toml").read_text()


def test_installer_dry_run_changes_nothing_and_keeps_foreign_entries(fakehome):
    (fakehome / ".claude.json").write_text(json.dumps({"mcpServers": {"flowdesk": {"command": "someone-elses-server"}}}))
    before = (fakehome / ".claude.json").read_text()
    out = install.install(["claude"], dry=True)
    assert "would register as 'flowdesk-mcp'" in out["clients"]["claude"]["action"]
    assert (fakehome / ".claude.json").read_text() == before
    install.install(["claude"])
    servers = json.loads((fakehome / ".claude.json").read_text())["mcpServers"]
    assert servers["flowdesk"]["command"] == "someone-elses-server" and "flowdesk-mcp" in servers


def test_installer_refuses_to_corrupt_unparseable_config(fakehome):
    (fakehome / ".claude.json").write_text("{ not json")
    out = install.install(["claude"])
    assert "cannot parse" in out["clients"]["claude"]["action"]
    assert (fakehome / ".claude.json").read_text() == "{ not json"


def test_frozen_server_entry_uses_the_app_executable(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    exe = os.path.abspath("/Applications/FlowDesk.app/Contents/MacOS/FlowDesk")
    monkeypatch.setattr(sys, "executable", exe)
    assert install.server_entry() == (exe, ["--mcp"])
    assert install.is_ours({"command": exe, "args": ["--mcp"]})
    assert discovery.find_flowdesk() == [exe]


def test_check_cli(fakehome, capsys):
    assert install.main(["--check"]) == 0
    assert "Server command" in capsys.readouterr().out


# ---------------------------------------------------------------- CLI runner
def test_build_argv_for_both_backends():
    a = agentcli.build_argv("claude", "/x/claude", "hello", "flowdesk", 42, session_id="S1", model="opus", effort="high")
    assert a[:3] == ["/x/claude", "-p", "hello"] and "mcp__flowdesk" in a and a[a.index("--resume") + 1] == "S1" and "opus" in a
    assert "process id 42" in a[a.index("--append-system-prompt") + 1]
    c = agentcli.build_argv("codex", "/x/codex", "hello", "flowdesk", 42)
    assert c[:2] == ["/x/codex", "exec"] and "--json" in c and "flowdesk" in c[-1].lower()
    r = agentcli.build_argv("codex", "/x/codex", "again", "flowdesk", 42, session_id="T9")
    assert r[:3] == ["/x/codex", "exec", "resume"] and r[-1] == "again"


def test_stream_parsers_claude_and_codex():
    got = []
    cb = lambda k, p: got.append((k, p))  # noqa: E731
    agentcli._parse_claude({"type": "system", "subtype": "init", "session_id": "abc"}, cb)
    agentcli._parse_claude({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__flowdesk__statistics", "input": {"group": "All Samples"}}]}}, cb)
    agentcli._parse_claude({"type": "user", "message": {"content": [{"type": "tool_result", "content": [{"type": "text", "text": '{"rows": []}'}]}]}}, cb)
    agentcli._parse_claude({"type": "result", "result": "done", "total_cost_usd": 0.01}, cb)
    kinds = [k for k, _ in got]
    assert kinds == ["session", "tool", "tool_result", "stats", "done"]
    assert got[1][1].startswith("flowdesk/statistics ")
    got.clear()
    agentcli._parse_codex({"type": "thread.started", "thread_id": "T"}, cb)
    agentcli._parse_codex({"type": "item.started", "item": {"type": "mcp_tool_call", "server": "flowdesk", "tool": "workspace", "arguments": {}}}, cb)
    agentcli._parse_codex({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}, cb)
    agentcli._parse_codex({"type": "turn.completed", "usage": {}}, cb)
    assert [k for k, _ in got] == ["session", "tool", "text", "stats", "done"]


def test_clean_env_drops_interpreter_overrides(monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "/nope")
    monkeypatch.setenv("QT_PLUGIN_PATH", "/nope")
    env = agentcli.clean_env()
    assert "PYTHONHOME" not in env and "QT_PLUGIN_PATH" not in env


def test_agent_run_pins_the_pid_and_streams(tmp_path, monkeypatch):
    fake = tmp_path / "fakeclaude"
    fake.write_text('#!/bin/sh\necho "{\\"type\\":\\"system\\",\\"subtype\\":\\"init\\",\\"session_id\\":\\"S\\"}"\n'
                    'echo "{\\"type\\":\\"result\\",\\"result\\":\\"pid=$FLOWDESK_PID\\"}"\n')
    fake.chmod(0o755)
    if os.name != "posix":
        pytest.skip("shell script stand-in")
    events = []
    run = agentcli.AgentRun("claude", "hi", "flowdesk", 4242, lambda k, p: events.append((k, p)), exe=str(fake))
    run.start().wait(20)
    assert run.session_id == "S" and run.final_text == "pid=4242"


# ---------------------------------------------------------------- in-app extension + chat
def test_extension_actions_dock_and_marshalling(app, testset, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from flowdesk.ui.main_window import MainWindow

    monkeypatch.setenv("FLOWDESK_BRIDGE", "0")
    w = MainWindow()
    ext = w.extensions["flowdesk.ui.agent"]
    assert not ext.host.running
    assert {"chat", "bridge", "setup"} <= set(ext.actions)
    w.controller.set_workspace(__import__("flowdesk.core.serialization", fromlist=["x"]).load_workspace(testset / "FlowDesk testset.fdw"))
    assert ext.start_bridge()
    result = {}

    def worker():
        c = BridgeClient("flowdesk")
        c.connect(pid=os.getpid())
        result["ping"] = c.call("ping")
        result["gate"] = c.call("create_gate", name="Scatter", shape="rectangle", x="FSC-A", y="SSC-A", group="All Samples", x_range=[20000, 200000])
        result["thread_is_main"] = None

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    _wait_threads(app, t)
    assert result["ping"]["samples"] == 6
    assert result["gate"]["ok"] and result["gate"]["population"] == "/Scatter"
    assert w.controller.can_undo
    dock = ext.open_chat()
    assert dock.chat.backend in ("claude", "codex")
    ext.stop_all()
    assert not ext.host.running
    w.controller.workspace.modified = False
    w.close()


def test_chat_widget_slash_commands_and_events(app):
    from PySide6 import QtCore, QtGui, QtWidgets

    from flowdesk.agent import chat_ui

    sessions = {"claude": None, "codex": None}
    w = chat_ui.make_chat_widget(QtWidgets, QtCore, QtGui, app="flowdesk", pid=1, start_run=lambda *a, **k: None, sessions=sessions,
                                 find_backend=agentcli.find_backend, title="Agent Chat for FlowDesk", agentcli=agentcli)
    w.submit("/help")
    w.on_event("tool", 'flowdesk/python {"code": "result = 1"}')
    w.on_event("tool_result", '{"ok": true}')
    w.on_event("text", "I gated the **lymphocytes**.")
    w.on_event("done", "")
    w._render()
    html = w.view.toPlainText()
    assert "FlowDesk" in html and "lymphocytes" in html and "flowdesk/python" in html
    w.submit("/clear")
    assert sessions["claude"] is None
    w.submit("/bogus")
    w._render()
    assert "unknown command" in w.view.toPlainText()


def test_chat_theme_follows_flowdesk_and_switches_in_place(app, tmp_path, monkeypatch):
    from PySide6 import QtCore, QtGui, QtWidgets

    from flowdesk.agent import chat_ui

    monkeypatch.setenv("FLOWDESK_AGENT_SETTINGS", str(tmp_path / "settings.json"))
    w = chat_ui.make_chat_widget(QtWidgets, QtCore, QtGui, app="flowdesk", pid=1, start_run=lambda *a, **k: None,
                                 sessions={"claude": None, "codex": None}, agentcli=agentcli)
    assert w.theme_mode == "auto" and chat_ui.P == chat_ui.THEMES["light"]  # FlowDesk's window is light
    w.on_event("text", "hello")
    assert chat_ui.THEMES["light"]["ORANGE"] in "".join(w.blocks)
    w.submit("/theme dark")
    joined = "".join(w.blocks)
    assert chat_ui.P == chat_ui.THEMES["dark"] and chat_ui.THEMES["dark"]["ORANGE"] in joined
    assert chat_ui.THEMES["light"]["ORANGE"] not in joined
    assert json.loads((tmp_path / "settings.json").read_text())["theme"] == "dark"
    assert chat_ui.THEMES["dark"]["BG"].lower() in w.styleSheet().lower()
    w2 = chat_ui.make_chat_widget(QtWidgets, QtCore, QtGui, app="flowdesk", pid=1, start_run=lambda *a, **k: None,
                                  sessions={"claude": None, "codex": None}, agentcli=agentcli)
    assert w2.theme_mode == "dark"  # remembered
    w2.set_theme("auto")
    assert chat_ui.P == chat_ui.THEMES["light"]


def test_theme_palettes_have_unique_colours():
    from flowdesk.agent import chat_ui

    for theme in chat_ui.THEMES.values():
        vals = [v.lower() for v in theme.values()]
        assert len(vals) == len(set(vals))
    assert set(chat_ui.THEMES["dark"]) == set(chat_ui.THEMES["light"])


def test_agent_icons_are_drawn(app):
    from flowdesk.ui.icons import pixmap

    for name in ("agent", "agent-bridge", "agent-connect"):
        img = pixmap(name, 32).toImage()
        painted = sum(1 for x in range(32) for y in range(32) if img.pixelColor(x, y).alpha() > 0)
        assert painted > 120, name
