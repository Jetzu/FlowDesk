"""Register the FlowDesk MCP server with Claude Code, Codex and Claude Desktop.

    flowdesk-mcp-install              detect the agents on this machine and register FlowDesk with each
    flowdesk-mcp-install --check      only report what is detected and registered
    flowdesk-mcp-install --dry-run    show what would change
    flowdesk-mcp-install --uninstall  remove our registrations
    flowdesk-mcp-install --claude --codex --claude-desktop   limit to named clients (also when not detected yet)

From a source checkout or pip install the entry is the ``flowdesk-mcp`` script next to this interpreter; from the
packaged app it is the app's own executable with ``--mcp``.  Existing configs are backed up (``*.flowdesk-backup``) and
only our ``flowdesk`` entry is touched.  Idempotent; works on macOS, Linux and Windows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import List, Optional, Tuple

from . import SERVER_NAME

CLIENTS = ("claude", "codex", "claude_desktop")
CLIENT_LABELS = {"claude": "Claude Code", "codex": "Codex", "claude_desktop": "Claude Desktop"}
BACKUP_SUFFIX = ".flowdesk-backup"


def _win() -> bool:
    return sys.platform.startswith("win")


def home() -> str:
    return os.path.expanduser("~")


# --------------------------------------------------------------------------- server entry
def server_entry() -> Tuple[str, List[str]]:
    """(command, args) an MCP client runs to start the FlowDesk server."""
    if getattr(sys, "frozen", False):
        return os.path.abspath(sys.executable), ["--mcp"]
    script = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "flowdesk-mcp" + (".exe" if _win() else ""))
    if os.path.exists(script):
        return script, []
    return os.path.abspath(sys.executable), ["-m", "flowdesk.agent.server"]


def is_ours(entry: Optional[dict]) -> bool:
    entry = entry or {}
    cmd = str(entry.get("command", "") or "").replace("\\", "/")
    base = cmd.rsplit("/", 1)[-1].lower()
    args = " ".join(str(a) for a in (entry.get("args") or []))
    return base in ("flowdesk-mcp", "flowdesk-mcp.exe") or "flowdesk.agent.server" in args or (
        "--mcp" in args and "flowdesk" in cmd.lower())


# --------------------------------------------------------------------------- paths
def claude_config_path() -> str:
    return os.path.join(home(), ".claude.json")


def codex_config_path() -> str:
    return os.path.join(os.environ.get("CODEX_HOME") or os.path.join(home(), ".codex"), "config.toml")


def claude_desktop_config_path() -> str:
    if sys.platform == "darwin":
        return os.path.join(home(), "Library", "Application Support", "Claude", "claude_desktop_config.json")
    if _win():
        return os.path.join(os.environ.get("APPDATA", home()), "Claude", "claude_desktop_config.json")
    return os.path.join(home(), ".config", "Claude", "claude_desktop_config.json")


def _config_path(client: str) -> str:
    return {"claude": claude_config_path, "codex": codex_config_path, "claude_desktop": claude_desktop_config_path}[client]()


def _backup(path: str):
    if os.path.exists(path):
        shutil.copy2(path, path + BACKUP_SUFFIX)


# --------------------------------------------------------------------------- detection
def detect_clients() -> dict:
    """client -> evidence that it is installed (None = not found)."""
    from . import agentcli

    out: dict = {}
    try:
        claude_exe, codex_exe = agentcli.find_claude(), agentcli.find_codex()
    except Exception:  # noqa: BLE001
        claude_exe = codex_exe = None
    out["claude"] = claude_exe or next((p for p in (claude_config_path(), os.path.join(home(), ".claude")) if os.path.exists(p)), None)
    codex_home = os.path.dirname(codex_config_path())
    out["codex"] = codex_exe or (codex_home if os.path.isdir(codex_home) else None)
    desktop = None
    if os.path.exists(claude_desktop_config_path()):
        desktop = claude_desktop_config_path()
    elif sys.platform == "darwin" and any(os.path.isdir(p) for p in ("/Applications/Claude.app", os.path.join(home(), "Applications", "Claude.app"))):
        desktop = "Claude Desktop app"
    elif _win() and any(os.path.isdir(os.path.join(os.environ.get("LOCALAPPDATA", ""), d)) for d in ("AnthropicClaude", os.path.join("Programs", "Claude"))):
        desktop = "Claude Desktop app"
    out["claude_desktop"] = desktop
    return out


def _read_json(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return json.loads(text) if text.strip() else {}


def registered(client: str) -> Optional[dict]:
    """The current FlowDesk entry of a client, or None."""
    path = _config_path(client)
    try:
        if client == "codex":
            if not os.path.exists(path):
                return None
            m = re.search(r"(?ms)^\[mcp_servers\.%s\]\n(.*?)(?=^\[|\Z)" % re.escape(SERVER_NAME), open(path, encoding="utf-8").read())
            if not m:
                return None
            cmd = re.search(r'^command\s*=\s*(".*")\s*$', m.group(1), re.M)
            args = re.search(r"^args\s*=\s*(\[.*\])\s*$", m.group(1), re.M)
            return {"command": json.loads(cmd.group(1)) if cmd else "", "args": json.loads(args.group(1)) if args else []}
        servers = _read_json(path).get("mcpServers", {})
    except Exception:  # noqa: BLE001
        return None
    for name in (SERVER_NAME, SERVER_NAME + "-mcp"):
        if name in servers:
            return servers[name]
    return None


def check() -> dict:
    """What is detected and what is registered: {"entry": [...], "clients": {client: {"found", "config", "registered", "current"}}}."""
    cmd, args = server_entry()
    found = detect_clients()
    out = {"entry": [cmd] + args, "clients": {}}
    for c in CLIENTS:
        reg = registered(c)
        out["clients"][c] = {"found": found.get(c), "config": _config_path(c), "registered": reg is not None,
                             "current": bool(reg) and reg.get("command") == cmd and list(reg.get("args") or []) == args}
    return out


def describe(report: dict) -> str:
    lines = []
    if "error" in report:
        return "error: " + report["error"]
    if "entry" in report:
        lines.append("Server command: " + " ".join(report["entry"]))
    for c, st in report.get("clients", {}).items():
        label = CLIENT_LABELS[c]
        if "action" in st:
            lines.append("%s: %s" % (label, st["action"]))
        elif st.get("registered"):
            lines.append("%s: registered%s" % (label, "" if st.get("current") else " (points elsewhere; Register updates it)"))
        elif st.get("found"):
            lines.append("%s: found, not registered yet" % label)
        else:
            lines.append("%s: not found" % label)
    return "\n".join(lines)


# --------------------------------------------------------------------------- registration
def _register_json(path: str, client: str, dry: bool, remove: bool) -> str:
    try:
        data = _read_json(path)
    except Exception as exc:  # noqa: BLE001
        return "cannot parse %s (%s); not touching it" % (path, exc)
    servers = data.setdefault("mcpServers", {})
    name = SERVER_NAME
    if remove:
        removed = [n for n in (SERVER_NAME, SERVER_NAME + "-mcp") if n in servers and is_ours(servers[n])]
        for n in removed:
            servers.pop(n)
        msg = "removed" if removed else "not registered"
        if not removed or dry:
            return msg if not dry else "would remove" if removed else msg
    else:
        if name in servers and not is_ours(servers[name]):
            name = SERVER_NAME + "-mcp"  # never overwrite somebody else's server called flowdesk
        cmd, args = server_entry()
        servers[name] = {"type": "stdio", "command": cmd, "args": args, "env": {}} if client == "claude" else {"command": cmd, "args": args}
        msg = "registered as %r -> %s" % (name, " ".join([cmd] + args))
        if dry:
            return "would register as %r -> %s" % (name, " ".join([cmd] + args))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _backup(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return msg


def _toml_str(s: str) -> str:
    return json.dumps(s)


def _register_codex(path: str, dry: bool, remove: bool) -> str:
    text = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    pat = re.compile(r"(?ms)^\[mcp_servers\.%s\]\n.*?(?=^\[|\Z)" % re.escape(SERVER_NAME))
    if remove:
        if not pat.search(text):
            return "not registered"
        if dry:
            return "would remove"
        text = pat.sub("", text)
        msg = "removed"
    else:
        cmd, args = server_entry()
        block = "[mcp_servers.%s]\ncommand = %s\nargs = [%s]\n\n" % (SERVER_NAME, _toml_str(cmd), ", ".join(_toml_str(a) for a in args))
        if dry:
            return "would register -> %s" % " ".join([cmd] + args)
        if pat.search(text):
            text = pat.sub(lambda _m: block, text)
        else:
            if text and not text.endswith("\n"):
                text += "\n"
            if text and not text.endswith("\n\n"):
                text += "\n"
            text += block
        msg = "registered -> %s" % " ".join([cmd] + args)
    try:
        import tomllib

        tomllib.loads(text)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return "NOT written: the result would not be valid TOML (%s)" % exc
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _backup(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return msg


def install(clients: Optional[List[str]] = None, dry: bool = False, remove: bool = False) -> dict:
    """Register (or remove) FlowDesk with the named clients, default: every detected client. Returns a report like check()."""
    found = detect_clients()
    chosen = list(clients) if clients else [c for c in CLIENTS if found.get(c)]
    report = check()
    for c in CLIENTS:
        if c not in chosen:
            continue
        path = _config_path(c)
        action = _register_codex(path, dry, remove) if c == "codex" else _register_json(path, c, dry, remove)
        report["clients"][c]["action"] = action
    if not dry:
        after = check()
        for c in CLIENTS:
            after["clients"][c]["action"] = report["clients"][c].get("action")
        report = after
    report["chosen"] = chosen
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flowdesk-mcp-install", description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="only report what is detected and registered")
    p.add_argument("--dry-run", action="store_true", help="show what would change")
    p.add_argument("--uninstall", action="store_true", help="remove our registrations")
    p.add_argument("--claude", action="store_true", help="Claude Code")
    p.add_argument("--codex", action="store_true", help="Codex")
    p.add_argument("--claude-desktop", action="store_true", help="Claude Desktop")
    a = p.parse_args(argv)
    named = [c for c, on in (("claude", a.claude), ("codex", a.codex), ("claude_desktop", a.claude_desktop)) if on]
    if a.check:
        print(describe(check()))
        return 0
    report = install(named or None, dry=a.dry_run, remove=a.uninstall)
    print(describe(report))
    if not report["chosen"]:
        print("\nNo Claude Code, Codex or Claude Desktop found. Install one and run this again, or name a client (e.g. --claude).")
    elif not a.dry_run and not a.uninstall:
        print("\nDone. Claude Code / Codex pick FlowDesk up in new sessions (Claude Code: /mcp); restart Claude Desktop.")
        print("Try: \"open the FlowDesk test set and tell me the CD4/CD8 ratio per donor\".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
