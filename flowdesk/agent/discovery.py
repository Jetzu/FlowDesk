"""Finding running FlowDesk bridges and starting FlowDesk, on every platform."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

from . import APP


def flowdesk_home() -> str:
    return os.environ.get("FLOWDESK_HOME") or os.path.join(os.path.expanduser("~"), ".flowdesk")


def instances_dir() -> str:
    d = os.path.join(flowdesk_home(), "instances")
    os.makedirs(d, exist_ok=True)
    return d


def renders_dir() -> str:
    d = os.environ.get("FLOWDESK_OUTPUT_DIR") or os.path.join(flowdesk_home(), "renders")
    os.makedirs(d, exist_ok=True)
    return d


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_start_time(pid: int) -> Optional[float]:
    """Process creation time as a Unix timestamp (best effort, None when unknown)."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes

            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not h:
                return None
            try:
                ct, et, kt, ut = (wintypes.FILETIME() for _ in range(4))
                if not ctypes.windll.kernel32.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut)):
                    return None
                ticks = (ct.dwHighDateTime << 32) | ct.dwLowDateTime
                return ticks / 1e7 - 11644473600.0
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        if sys.platform.startswith("linux"):
            return float(os.stat("/proc/%d" % pid).st_ctime)
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return time.mktime(time.strptime(out, "%a %b %d %H:%M:%S %Y")) if out else None
    except Exception:
        return None


def pid_reused(pid: int, file_started: Optional[float]) -> bool:
    """True when ``pid`` now belongs to a process that started after the discovery file was written."""
    if not file_started:
        return False
    t = pid_start_time(pid)
    return t is not None and t > float(file_started) + 5.0


def list_instances(app: str = APP) -> List[dict]:
    """Discovery files of live bridges, newest first (stale files are removed)."""
    out = []
    for path in glob.glob(os.path.join(instances_dir(), "%s-*.json" % app)):
        try:
            with open(path) as f:
                info = json.load(f)
        except Exception:
            continue
        info["_file"] = path
        pid = int(info.get("pid", 0))
        if not pid_alive(pid) or pid_reused(pid, info.get("started")):
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        out.append(info)
    out.sort(key=lambda i: i.get("started", 0), reverse=True)
    return out


# ---------------------------------------------------------------------------
# starting FlowDesk
# ---------------------------------------------------------------------------
def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def find_flowdesk() -> Optional[List[str]]:
    """argv that starts FlowDesk: $FLOWDESK_EXE, this executable when frozen, this interpreter running
    ``-m flowdesk`` from a source install, else the installed app bundle / executable on PATH."""
    exe = os.environ.get("FLOWDESK_EXE")
    if exe and os.path.exists(exe):
        return [exe]
    if is_frozen():
        return [sys.executable]
    try:
        import flowdesk.app  # noqa: F401

        return [sys.executable, "-m", "flowdesk"]
    except Exception:
        pass
    candidates = [
        "/Applications/FlowDesk.app/Contents/MacOS/FlowDesk",
        os.path.expanduser("~/Applications/FlowDesk.app/Contents/MacOS/FlowDesk"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "FlowDesk", "FlowDesk.exe"),
        os.path.expanduser("~/Applications/FlowDesk/FlowDesk"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return [c]
    import shutil

    which = shutil.which("flowdesk")
    return [which] if which else None


def popen_detached(argv: List[str], cwd: Optional[str] = None, extra_env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """Start an application that outlives the MCP server process."""
    env = dict(os.environ)
    env.setdefault("FLOWDESK_HOME", flowdesk_home())
    if extra_env:
        env.update(extra_env)
    kw = dict(cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(argv, **kw)


def wait_for_instance(app: str = APP, timeout: float = 60.0, not_pids=()) -> Optional[dict]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for inst in list_instances(app):
            if inst.get("pid") not in not_pids:
                return inst
        time.sleep(0.25)
    return None
