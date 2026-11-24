"""Drive Claude Code or Codex CLI non-interactively from inside FlowDesk.

Standard library only.  The CLI is the user's own installed Claude Code / Codex; it loads FlowDesk's MCP server
from the user's MCP config (set up by ``flowdesk-mcp-install``) and is pinned to *this* FlowDesk process through
``FLOWDESK_PID``.

Events delivered to the callback as (kind, payload):
    ("session", id)          conversation id usable for follow-up turns
    ("delta", str)           streamed partial assistant text (Claude only)
    ("text", str)            complete assistant message
    ("tool", str)            tool call: "<server>/<tool> <json args>"
    ("tool_result", str)     tool result text (truncated)
    ("note", str)            remark for the user (e.g. a saved model replaced by its current equivalent)
    ("stats", dict)          cost/usage at the end of a turn
    ("done", str)            final answer (also delivered as text)
    ("error", str)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Optional, Tuple

BACKENDS = ("claude", "codex")

SYSTEM_CONTEXT = (
    "You are embedded in {app_title} (process id {pid}), a flow cytometry analysis app. The user is looking at its "
    "window right now. Use the '{app}' MCP server tools to inspect and change this workspace (workspace, populations, "
    "histogram, plot, create_gate, statistics, compensation, ...). The server is already pinned to this process, so do "
    "not launch another instance. Look at the data with histogram/plot numbers before placing gates, report what you "
    "changed (every change is an undo step for the user) and never save over or discard the user's work unasked. "
    "Keep answers concise."
)


def _win() -> bool:
    return sys.platform.startswith("win")


def _candidates_from_env(*names: str) -> List[str]:
    out = []
    for n in names:
        v = os.environ.get(n)
        if v:
            out.append(v)
    return out


_validated: dict = {}


def _works(exe: str) -> bool:
    """A candidate counts only if `exe --version` succeeds (stale npm installs, wrong-arch builds...)."""
    if exe in _validated:
        return _validated[exe]
    ok = False
    try:
        kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=20,
                  env=clean_env())
        if _win():
            kw["creationflags"] = 0x08000000
        r = subprocess.run([exe, "--version"], **kw)
        ok = r.returncode == 0
    except Exception:
        ok = False
    _validated[exe] = ok
    return ok


def _first(paths: List[str]) -> Optional[str]:
    for p in paths:
        for hit in sorted(glob.glob(os.path.expanduser(os.path.expandvars(p))), reverse=True):
            if os.path.isfile(hit) and os.access(hit, os.X_OK) and _works(hit):
                return hit
    return None


def find_claude() -> Optional[str]:
    for v in _candidates_from_env("FLOWDESK_CLAUDE_EXE", "CLAUDE_CLI"):
        if os.path.exists(v):
            return v
    for name in ("claude", "claude.exe", "claude.cmd"):
        p = shutil.which(name)
        if p and _works(p):
            return p
    home = os.path.expanduser("~")
    pats = [
        os.path.join(home, ".claude", "local", "claude"),
        os.path.join(home, "Library", "Application Support", "Claude", "claude-code", "*", "*", "claude.app", "Contents", "MacOS", "claude"),
        os.path.join(home, ".local", "bin", "claude"),
        os.path.join(home, ".claude", "local", "node_modules", ".bin", "claude"),
        os.path.join(home, ".vscode*", "extensions", "anthropic.claude-code-*", "resources", "native-binary", "claude*"),
        os.path.join(home, ".cursor", "extensions", "anthropic.claude-code-*", "resources", "native-binary", "claude*"),
        os.path.join(home, ".npm-global", "bin", "claude"),
        "/usr/local/bin/claude", "/opt/homebrew/bin/claude",
    ]
    if _win():
        pats += [os.path.join(os.environ.get("APPDATA", ""), "npm", "claude.cmd"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "claude*", "claude.exe")]
    return _first(pats)


def find_codex() -> Optional[str]:
    for v in _candidates_from_env("FLOWDESK_CODEX_EXE", "CODEX_CLI"):
        if os.path.exists(v):
            return v
    for name in ("codex", "codex.exe", "codex.cmd"):
        p = shutil.which(name)
        if p and _works(p):
            return p
    home = os.path.expanduser("~")
    pats = []
    if sys.platform == "darwin":
        pats += ["/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex",
                 os.path.join(home, "Applications", "ChatGPT.app", "Contents", "Resources", "codex-cli", "bin", "codex")]
    pats += [os.path.join(home, ".npm-global", "bin", "codex"), "/usr/local/bin/codex", "/opt/homebrew/bin/codex"]
    if _win():
        pats += [os.path.join(os.environ.get("APPDATA", ""), "npm", "codex.cmd"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "ChatGPT", "resources", "codex-cli", "bin", "codex*")]
    return _first(pats)


def find_backend(backend: str) -> Optional[str]:
    return find_claude() if backend == "claude" else find_codex()


def available_backends() -> List[str]:
    """Backends whose CLI is installed and runs (so the UI can hide the missing one)."""
    return [b for b in BACKENDS if find_backend(b)]


BACKEND_LABELS = {"claude": "Claude Code", "codex": "Codex"}
# ---------------------------------------------------------------------------
# model catalog (follows new model releases without code changes)
# ---------------------------------------------------------------------------
# The model / effort menus are read from the caches the CLIs keep for their own /model pickers and refresh by
# themselves:
#   Claude Code: <CLAUDE_CONFIG_DIR or ~/.claude>/cache/model-catalog/*-cc.json
#   Codex:       <CODEX_HOME or ~/.codex>/models_cache.json   (+ the default model from config.toml)
# Claude models are offered by family alias (opus, sonnet, ...), which Claude Code resolves to the newest model of
# that family, so a saved choice keeps following new releases.  Without a cache the built-in lists below are used.
CLAUDE_ALIASES = ("fable", "opus", "sonnet", "haiku")  # fallback; the cache lists the aliases it knows
MODEL_CHOICES = {  # fallback when the CLI has not written its model cache yet
    "claude": [("", "default")] + [(a, a) for a in CLAUDE_ALIASES],
    "codex": [("", "default (config.toml)")],
}
EFFORT_CHOICES = {  # fallback, see effort_choices()
    "claude": ["", "low", "medium", "high", "xhigh", "max"],
    "codex": ["", "minimal", "low", "medium", "high", "xhigh"],
}
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_catalog_memo: dict = {}


def _claude_config_dir() -> str:
    return os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join("~", ".claude"))


def _codex_home() -> str:
    return os.path.expanduser(os.environ.get("CODEX_HOME") or os.path.join("~", ".codex"))


def _json_file(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sorted_efforts(levels) -> List[str]:
    seen = list(dict.fromkeys(str(e) for e in levels if e))
    return [e for e in _EFFORT_ORDER if e in seen] + [e for e in seen if e not in _EFFORT_ORDER]


def _read_claude_catalog(files: List[str]) -> Tuple[List[dict], Dict[str, str]]:
    best, best_t = None, -1.0
    for p in files:
        d = _json_file(p)
        if not isinstance(d, dict) or not p.endswith("-cc.json"):
            continue
        try:
            t = float(d.get("fetchedAt") or 0)
        except (TypeError, ValueError):
            t = 0.0
        if t > best_t:
            best, best_t = d, t
    if best is None:
        return [], {}
    masked = _json_file(os.path.join(_claude_config_dir(), "cache", "model-catalog", "masked-ids.json"))
    masked = set(masked.get("ids") or []) if isinstance(masked, dict) else set()
    try:
        config = best["catalog"]["config"]
        models = config["models"]
    except (KeyError, TypeError):
        return [], {}
    aliases = {}
    targets = config.get("provider_alias_targets") if isinstance(config, dict) else None
    for alias, t in (targets.items() if isinstance(targets, dict) else []):
        if isinstance(t, dict) and t.get("default"):
            aliases[str(alias)] = str(t["default"])
    out = []
    for m in models if isinstance(models, list) else []:
        if not isinstance(m, dict) or not m.get("id") or m["id"] in masked:
            continue
        rt = m.get("runtime") if isinstance(m.get("runtime"), dict) else {}
        out.append({"id": str(m["id"]), "name": str(m.get("name") or m["id"]),
                    "listed": str(m.get("section") or "main") == "main",
                    "family": str(rt.get("family") or "").lower(),
                    "efforts": _sorted_efforts(rt.get("effort_levels") or []),
                    "default_effort": str(rt.get("default_effort") or "")})
    return out, aliases


def _read_codex_catalog(files: List[str]) -> List[dict]:
    d = _json_file(files[0]) if files else None
    models = d.get("models") if isinstance(d, dict) else None
    out = []
    for m in models if isinstance(models, list) else []:
        if not isinstance(m, dict) or not m.get("slug"):
            continue
        levels = [r.get("effort") for r in m.get("supported_reasoning_levels") or [] if isinstance(r, dict)]
        prio = m.get("priority")
        out.append({"id": str(m["slug"]), "name": str(m.get("display_name") or m["slug"]),
                    "listed": m.get("visibility", "list") == "list", "family": "",
                    "priority": prio if isinstance(prio, (int, float)) else 1e9,
                    "efforts": _sorted_efforts(levels), "default_effort": str(m.get("default_reasoning_level") or "")})
    out.sort(key=lambda m: m["priority"])
    return out, {}


def model_catalog(backend: str) -> List[dict]:
    """Models the installed CLI offers, newest/most relevant first ([] when its cache is missing).  Each entry:
    {"id", "name", "family", "listed", "efforts", "default_effort"}.  Re-read whenever the cache file changes."""
    if backend == "claude":
        files = sorted(glob.glob(os.path.join(_claude_config_dir(), "cache", "model-catalog", "*.json")))
    else:
        files = [os.path.join(_codex_home(), "models_cache.json")]
    stamp = []
    for p in files:
        try:
            stamp.append((p, os.path.getmtime(p)))
        except OSError:
            pass
    stamp = tuple(stamp)
    hit = _catalog_memo.get(backend)
    if not (hit and hit[0] == stamp):
        try:
            cat, aliases = (_read_claude_catalog(files) if backend == "claude"
                            else _read_codex_catalog([p for p, _ in stamp]))
        except Exception:
            cat, aliases = [], {}
        hit = _catalog_memo[backend] = (stamp, cat, aliases)
    return hit[1]


def model_aliases(backend: str) -> Dict[str, str]:
    """Claude Code's aliases and the model each currently points to ({} for Codex or without a cache)."""
    model_catalog(backend)
    hit = _catalog_memo.get(backend)
    return dict(hit[2]) if hit else {}


def _codex_config_value(key: str) -> str:
    """A top-level string value of Codex's config.toml ("" when unset)."""
    try:
        with open(os.path.join(_codex_home(), "config.toml"), encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("["):
                    break
                m = re.match(r'%s\s*=\s*["\']([^"\']*)["\']' % re.escape(key), s)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def model_choices(backend: str) -> List[Tuple[str, str]]:
    """(value, label) pairs for the model menu; value "" = the CLI's own default.  Any other name still works."""
    cat = model_catalog(backend)
    if backend == "claude":
        if not cat:
            return list(MODEL_CHOICES["claude"])
        by_target = {}
        for alias, target in model_aliases(backend).items():
            by_target.setdefault(target, alias)
        out = [("", "default")]
        for m in cat:
            if not m["listed"]:
                continue  # older models: still accepted by name (/model <id>)
            alias = by_target.get(m["id"])
            out.append((alias, "%s · %s" % (alias, m["name"])) if alias else (m["id"], m["name"]))
        return out
    cfg = _codex_config_value("model")
    out = [("", "default (%s)" % cfg if cfg else "default (config.toml)")]
    listed = [(m["id"], m["id"]) for m in cat if m["listed"]]
    if not listed and cfg:
        listed = [(cfg, cfg)]
    return out + listed


def _catalog_entry(backend: str, model: str) -> Optional[dict]:
    cat = model_catalog(backend)
    model = (model or "").strip()
    if not cat:
        return None
    if not model:
        if backend != "codex":
            return None
        model = _codex_config_value("model")
    base = model.split("[")[0]
    if backend == "claude":
        base = model_aliases(backend).get(base, base)
    return next((m for m in cat if m["id"] == base), None)


def effort_choices(backend: str, model: Optional[str] = None) -> List[str]:
    """Effort levels ("" = default first): those of ``model`` when the catalog knows it, else the union over every
    model of the backend (or the built-in list when there is no catalog)."""
    if model is not None:
        m = _catalog_entry(backend, model)
        if m and m["efforts"]:
            return [""] + m["efforts"]
    levels = _sorted_efforts(e for m in model_catalog(backend) for e in m["efforts"])
    return [""] + levels if levels else list(EFFORT_CHOICES[backend])


def resolve_model(backend: str, model: str) -> Tuple[str, str]:
    """(model to run, note).  A saved model the CLI no longer offers in its menu is replaced by one that works: a
    superseded Claude model by its family alias (= the newest model of that family), an unknown Codex model by
    Codex's default."""
    model = (model or "").strip()
    cat = model_catalog(backend)
    if not model or not cat:
        return model, ""
    base = model.split("[")[0]
    ids = {m["id"] for m in cat}
    if backend == "claude":
        aliases = model_aliases(backend) or {a: "" for a in CLAUDE_ALIASES}
        if base in aliases or base in {m["id"] for m in cat if m["listed"]}:
            return model, ""
        entry = next((m for m in cat if m["id"] == base), None)
        fam = entry["family"] if entry and entry["family"] else next(
            (a for a in aliases if base.startswith("claude-%s-" % a)), "")
        if fam in aliases:
            new = fam + model[len(base):]  # keeps a context suffix such as [1m]
            return new, "%s has been superseded; using %s (the newest %s model)" % (model, new, fam)
        return model, ""
    if base in ids or _codex_config_value("model_provider") not in ("", "openai"):
        return model, ""
    return "", "%s is not in Codex's model list; using Codex's default model" % model


def resolve_effort(backend: str, model: str, effort: str) -> Tuple[str, str]:
    """(effort to run, note): an effort the chosen model does not support falls back to the model's default."""
    effort = (effort or "").strip()
    m = _catalog_entry(backend, model)
    if not effort or not m or not m["efforts"] or effort in m["efforts"]:
        return effort, ""
    return "", "%s does not support effort %s; using its default" % (m["name"], effort)


def catalog_snapshot() -> dict:
    """Menus for a panel: model options (value, label), effort levels per backend and per offered model."""
    out: dict = {"model_options": {}, "model_choices": {}, "effort_choices": {}, "model_efforts": {}}
    for b in BACKENDS:
        opts = model_choices(b)
        out["model_options"][b] = [list(o) for o in opts]
        out["model_choices"][b] = [o[0] for o in opts]
        out["effort_choices"][b] = effort_choices(b)
        out["model_efforts"][b] = {o[0]: effort_choices(b, o[0]) for o in opts}
    return out


# Codex's --approve-for-me as config overrides: the flag exists on `codex exec` but not on `codex exec resume`, so
# follow-up turns failed with "unexpected argument"; -c works on both.
CODEX_APPROVAL_ARGS = ["-c", 'approvals_reviewer="auto_review"', "-c", 'approval_policy="on-request"',
                       "-c", 'sandbox_mode="workspace-write"']


_SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".flowdesk", "agent_settings.json")


def load_settings() -> dict:
    path = os.environ.get("FLOWDESK_AGENT_SETTINGS", _SETTINGS_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}


def save_settings(d: dict) -> None:
    path = os.environ.get("FLOWDESK_AGENT_SETTINGS", _SETTINGS_FILE)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
    except Exception:
        pass


def backend_settings(backend: str) -> dict:
    """{"model": str, "effort": str} for a backend, persisted per user."""
    d = load_settings().get(backend, {})
    return {"model": d.get("model", ""), "effort": d.get("effort", "")}


def set_backend_settings(backend: str, model: Optional[str] = None, effort: Optional[str] = None) -> dict:
    d = load_settings()
    cur = d.get(backend, {})
    if model is not None:
        cur["model"] = model
    if effort is not None:
        cur["effort"] = effort
    d[backend] = cur
    d.setdefault("default_backend", backend)
    save_settings(d)
    return cur


def default_backend() -> str:
    d = load_settings().get("default_backend")
    avail = available_backends()
    if d in avail:
        return d
    return avail[0] if avail else "claude"


def set_default_backend(backend: str) -> None:
    d = load_settings()
    d["default_backend"] = backend
    save_settings(d)


# Variables FlowDesk's launcher sets for its *own* bundled Python/Qt; inherited by the CLI (and by the MCP
# server it spawns from a different Python) they break interpreter start-up ("connection closed").
_STRIP_ENV_EXACT = {"PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "PYTHONNOUSERSITE", "PYTHONSTARTUP", "PYTHONSAFEPATH",
                    "__PYVENV_LAUNCHER__", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
                    "LD_LIBRARY_PATH", "LD_PRELOAD", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH",
                    "QT_QPA_PLATFORM", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "FONTCONFIG_PATH",
                    "FONTCONFIG_FILE", "GSETTINGS_SCHEMA_DIR", "GIO_MODULE_DIR", "XDG_DATA_DIRS"}
_STANDARD_PATH = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]


def clean_env() -> dict:
    """A copy of the environment without FlowDesk's private interpreter settings; PATH gets the usual
    user directories back (GUI apps started from a dock/Finder/Start menu see a minimal PATH)."""
    env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV_EXACT}
    if not _win():
        home = os.path.expanduser("~")
        parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
        for extra in [os.path.join(home, ".local", "bin"), os.path.join(home, ".npm-global", "bin"),
                      os.path.join(home, ".claude", "local")] + _STANDARD_PATH:
            if extra not in parts:
                parts.append(extra)
        env["PATH"] = os.pathsep.join(parts)
    return env


def _app_title(app: str) -> str:
    return {"flowdesk": "FlowDesk"}.get(app, app)


def build_argv(backend: str, exe: str, prompt: str, app: str, pid: int, session_id: Optional[str] = None,
               allow_shell: bool = False, model: str = "", effort: str = "") -> List[str]:
    context = SYSTEM_CONTEXT.format(app=app, app_title=_app_title(app), pid=pid)
    if backend == "claude":
        argv = [exe, "-p", prompt, "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                "--allowedTools", "mcp__%s" % app + (",Bash,Edit,Write" if allow_shell else ""),
                "--append-system-prompt", context]
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        if session_id:
            argv += ["--resume", session_id]
        return argv
    if backend == "codex":
        full_prompt = prompt if session_id else "%s\n\nUser request: %s" % (context, prompt)
        opts = ["--json"] + CODEX_APPROVAL_ARGS + ["--skip-git-repo-check"]
        if model:
            opts += ["-m", model]
        if effort:
            opts += ["-c", 'model_reasoning_effort="%s"' % effort]
        if session_id:
            return [exe, "exec", "resume"] + opts + [session_id, full_prompt]
        return [exe, "exec"] + opts + [full_prompt]
    raise ValueError("unknown backend %r" % backend)


def _unwrap_result(obj) -> str:
    """MCP result envelopes ({"content":[{"type":"text","text":...}]}, {"result": "<json>"}) -> readable text."""
    for _ in range(3):
        if isinstance(obj, str):
            try:
                parsed = json.loads(obj)
            except Exception:
                return obj
            if isinstance(parsed, (dict, list)):
                obj = parsed
                continue
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("content"), list):
                parts = []
                for c in obj["content"]:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(str(c.get("text", "")))
                    elif isinstance(c, dict) and c.get("type") == "image":
                        parts.append("[image]")
                obj = "\n".join(parts)
                continue
            if set(obj) == {"result"} and isinstance(obj["result"], str):
                obj = obj["result"]
                continue
        break
    return obj if isinstance(obj, str) else json.dumps(obj)


def _tool_result_text(block) -> str:
    content = block.get("content")
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(str(c.get("text", "")))
            elif isinstance(c, dict) and c.get("type") == "image":
                parts.append("[image]")
        text = "\n".join(parts)
    else:
        text = str(content or "")
    return _unwrap_result(text)


def _parse_claude(ev: dict, cb: Callable):
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init" and ev.get("session_id"):
        cb("session", ev["session_id"])
    elif t == "stream_event":
        e = ev.get("event", {}) or {}
        if e.get("type") == "content_block_delta":
            d = e.get("delta", {}) or {}
            if d.get("type") == "text_delta" and d.get("text"):
                cb("delta", d["text"])
    elif t == "assistant":
        for c in ev.get("message", {}).get("content", []) or []:
            if c.get("type") == "text" and c.get("text"):
                cb("text", c["text"])
            elif c.get("type") == "tool_use":
                name = str(c.get("name", "")).replace("mcp__", "").replace("__", "/")
                args = json.dumps(c.get("input", {}))
                cb("tool", "%s %s" % (name, args[:300]))
    elif t == "user":
        for c in ev.get("message", {}).get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                cb("tool_result", _tool_result_text(c)[:4000])
    elif t == "result":
        cb("stats", {k: ev.get(k) for k in ("duration_ms", "total_cost_usd", "num_turns", "usage") if k in ev})
        if ev.get("is_error") or str(ev.get("subtype", "")).startswith("error"):
            cb("error", str(ev.get("result") or ev.get("subtype")))
        cb("done", str(ev.get("result") or ""))


def _parse_codex(ev: dict, cb: Callable):
    t = ev.get("type", "")
    if t == "thread.started" and ev.get("thread_id"):
        cb("session", ev["thread_id"])
    elif t.startswith("item."):
        it = ev.get("item", {}) or {}
        kind = it.get("type")
        if kind == "agent_message" and t == "item.completed" and it.get("text"):
            cb("text", it["text"])
        elif kind == "mcp_tool_call":
            if t == "item.started":
                cb("tool", "%s/%s %s" % (it.get("server"), it.get("tool"), json.dumps(it.get("arguments", it.get("input", {})))[:300]))
            elif t == "item.completed":
                res = it.get("result")
                if res is None:
                    res = it.get("output")
                if res is not None:
                    cb("tool_result", _unwrap_result(res)[:4000])
                elif it.get("status") and it.get("status") != "completed":
                    cb("error", "%s/%s %s" % (it.get("server"), it.get("tool"), it.get("status")))
        elif kind == "command_execution" and t == "item.completed":
            cb("tool", "shell %s" % str(it.get("command", ""))[:200])
            if it.get("aggregated_output"):
                cb("tool_result", str(it["aggregated_output"])[:4000])
    elif t == "turn.completed":
        cb("stats", {"usage": ev.get("usage") or {}})
        cb("done", "")
    elif t in ("turn.failed", "error"):
        cb("error", json.dumps({k: v for k, v in ev.items() if k != "type"})[:500])


class AgentRun:
    """One prompt → one CLI process, streamed on a background thread."""

    def __init__(self, backend: str, prompt: str, app: str, pid: int, callback: Callable,
                 session_id: Optional[str] = None, cwd: Optional[str] = None, allow_shell: bool = False,
                 exe: Optional[str] = None, extra_env: Optional[dict] = None, model: Optional[str] = None,
                 effort: Optional[str] = None):
        if backend not in BACKENDS:
            raise ValueError("backend must be one of %s" % (BACKENDS,))
        self.backend = backend
        self.exe = exe or find_backend(backend)
        if not self.exe:
            raise RuntimeError("%s CLI not found. Install it or set %s." % (
                "Claude Code" if backend == "claude" else "Codex", "FLOWDESK_CLAUDE_EXE" if backend == "claude" else "FLOWDESK_CODEX_EXE"))
        st = backend_settings(backend)
        self.model = st["model"] if model is None else model
        self.effort = st["effort"] if effort is None else effort
        # a saved model/effort the CLI no longer offers -> the current equivalent (and remember it)
        self.notes: List[str] = []
        healed_model, note = resolve_model(backend, self.model)
        if note:
            self.notes.append(note)
            if model is None:
                set_backend_settings(backend, model=healed_model)
        self.model = healed_model
        healed_effort, note = resolve_effort(backend, self.model, self.effort)
        if note:
            self.notes.append(note)
            if effort is None:
                set_backend_settings(backend, effort=healed_effort)
        self.effort = healed_effort
        self.argv = build_argv(backend, self.exe, prompt, app, pid, session_id, allow_shell, self.model, self.effort)
        self.app, self.pid, self.callback, self.cwd = app, pid, callback, cwd
        self.env = clean_env()
        self.env["FLOWDESK_PID"] = str(pid)
        self.env.setdefault("FLOWDESK_HOME", os.path.join(os.path.expanduser("~"), ".flowdesk"))
        if extra_env:
            self.env.update(extra_env)
        self.proc: Optional[subprocess.Popen] = None
        self.session_id: Optional[str] = session_id
        self.final_text = ""
        self.thread: Optional[threading.Thread] = None
        self.returncode: Optional[int] = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="flowdesk-agent-%s" % self.backend, daemon=True)
        self.thread.start()
        return self

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def wait(self, timeout: Optional[float] = None):
        if self.thread:
            self.thread.join(timeout)
        return self.final_text

    def _emit(self, kind, payload):
        if kind == "session":
            self.session_id = payload
        if kind == "done":
            self.final_text = payload or self.final_text
        if kind == "text":
            self.final_text = payload
        elif kind == "delta":
            self._draft = getattr(self, "_draft", "") + payload
        try:
            self.callback(kind, payload)
        except Exception:
            pass

    def _run(self):
        kw = dict(cwd=self.cwd, env=self.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE, text=True, bufsize=1)
        if _win():
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        try:
            self.proc = subprocess.Popen(self.argv, **kw)
        except Exception as e:
            self._emit("error", "could not start %s: %s" % (self.exe, e))
            self._emit("done", "")
            return
        for note in self.notes:
            self._emit("note", note)
        parser = _parse_claude if self.backend == "claude" else _parse_codex
        err_lines = []

        def drain_err():
            try:
                for line in self.proc.stderr:
                    err_lines.append(line.rstrip())
            except Exception:
                pass
        threading.Thread(target=drain_err, daemon=True).start()
        done_seen = False
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    if line.startswith("Reading additional input"):
                        continue
                    self._emit("text", line)
                    continue
                before = self.final_text
                parser(ev, self._emit)
                if ev.get("type") in ("result", "turn.completed"):
                    done_seen = True
        finally:
            self.returncode = self.proc.wait()
            if self.returncode not in (0, None) and not done_seen:
                tail = "\n".join(err_lines[-8:])
                self._emit("error", "%s exited with code %s%s" % (self.backend, self.returncode, (": " + tail) if tail else ""))
            if not done_seen:
                self._emit("done", self.final_text)
