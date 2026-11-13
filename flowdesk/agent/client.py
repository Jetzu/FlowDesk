"""HTTP client for the in-app bridge (persistent keep-alive connection)."""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from typing import Any, Optional

from . import APP, discovery


class BridgeError(RuntimeError):
    """The bridge reported an error for an operation."""

    def __init__(self, message: str, traceback_text: Optional[str] = None):
        super().__init__(message)
        self.traceback_text = traceback_text


class NotConnected(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, app: str = APP, default_timeout: float = 120.0):
        self.app = app
        self.info: Optional[dict] = None
        self._conn: Optional[http.client.HTTPConnection] = None
        self._lock = threading.Lock()
        self.default_timeout = default_timeout
        self.stats = {"calls": 0, "errors": 0, "total_ms": 0.0}

    # -- connection management ---------------------------------------------
    def ping_info(self, info: dict, timeout: float = 2.0) -> Optional[dict]:
        """The /ping payload of a bridge, or None when it does not answer."""
        try:
            conn = http.client.HTTPConnection(info["host"], int(info["port"]), timeout=timeout)
            conn.request("GET", "/ping")
            data = json.loads(conn.getresponse().read().decode("utf-8"))
            conn.close()
            if data.get("ok") and data.get("app") == self.app:
                return data
        except Exception:
            pass
        return None

    def identify(self, info: dict) -> Optional[dict]:
        """Ask the bridge for its identity through the authenticated channel (workspace, focus, ...)."""
        try:
            conn = http.client.HTTPConnection(info["host"], int(info["port"]), timeout=4)
            body = json.dumps({"op": "ping", "args": {}}).encode("utf-8")
            conn.request("POST", "/call", body=body, headers={"Content-Type": "application/json",
                                                              "Authorization": "Bearer " + info["token"],
                                                              "Content-Length": str(len(body))})
            data = json.loads(conn.getresponse().read().decode("utf-8"))
            conn.close()
            return data.get("result") if data.get("ok") else None
        except Exception:
            return None

    def describe_instances(self) -> list:
        """Live instances with identity, best first (focused window > most recently used > newest)."""
        out = []
        for c in discovery.list_instances(self.app):
            d = self.ping_info(c)
            if not d:
                continue
            ident = self.identify(c)
            if ident:
                d.update({k: v for k, v in ident.items() if v is not None})
            d["port"] = c.get("port")
            d["started"] = c.get("started")
            d["name"] = d.get("name") or c.get("name") or ""
            out.append(d)
        out.sort(key=lambda d: (not bool(d.get("window_active")), -(d.get("last_activity") or 0), -(d.get("started") or 0)))
        return out

    def connect(self, info: Optional[dict] = None, pid: Optional[int] = None, port: Optional[int] = None,
                name: Optional[str] = None) -> dict:
        """Attach to a live bridge; without arguments the window that has focus, else the most recently used."""
        if info is None:
            candidates = self.describe_instances()
            if pid is not None:
                candidates = [c for c in candidates if int(c.get("pid", 0)) == int(pid)]
            if port is not None:
                candidates = [c for c in candidates if int(c.get("port", 0)) == int(port)]
            if name:
                candidates = [c for c in candidates if (c.get("name") or "").lower() == name.lower()]
            files = {int(i.get("pid", 0)): i for i in discovery.list_instances(self.app)}
            for c in candidates:
                f = files.get(int(c.get("pid", 0)))
                if f and self._ping(f):
                    info = f
                    break
            if info is None:
                raise NotConnected("no running FlowDesk bridge found%s" % (" matching the request" if (pid or port or name) else ""))
        with self._lock:
            self._close()
            self.info = info
        return info

    def ensure(self) -> dict:
        if self.info and self._ping(self.info):
            return self.info
        return self.connect()

    @property
    def connected(self) -> bool:
        return self.info is not None

    def _close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _ping(self, info: dict, timeout: float = 2.0) -> bool:
        return self.ping_info(info, timeout) is not None

    # -- calls ---------------------------------------------------------------
    def call(self, op: str, timeout: Optional[float] = None, **args) -> Any:
        if self.info is None:
            self.connect()
        assert self.info is not None
        body = json.dumps({"op": op, "args": args}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.info["token"],
                   "Content-Length": str(len(body))}
        timeout = timeout or self.default_timeout
        t0 = time.time()
        with self._lock:
            last_exc = None
            for _attempt in range(2):
                try:
                    if self._conn is None:
                        self._conn = http.client.HTTPConnection(self.info["host"], int(self.info["port"]), timeout=timeout)
                    elif self._conn.sock:
                        self._conn.sock.settimeout(timeout)
                    self._conn.request("POST", "/call", body=body, headers=headers)
                    resp = self._conn.getresponse()
                    raw = resp.read()
                    break
                except (http.client.HTTPException, ConnectionError, socket.timeout, OSError) as e:
                    last_exc = e
                    self._close()
                    if isinstance(e, socket.timeout):
                        raise BridgeError("FlowDesk did not answer within %.0fs on %r (is it busy or showing a dialog?)" % (timeout, op))
            else:
                self.info = None
                raise NotConnected("lost the connection to FlowDesk: %s" % last_exc)
        self.stats["calls"] += 1
        self.stats["total_ms"] += (time.time() - t0) * 1000
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            raise BridgeError("bad response from the bridge: %r" % raw[:200])
        if resp.status == 401:
            self.info = None
            raise NotConnected("the bridge rejected the token; reconnecting on the next call")
        if not data.get("ok"):
            self.stats["errors"] += 1
            raise BridgeError(data.get("error", "unknown error"), data.get("traceback"))
        return data.get("result")
