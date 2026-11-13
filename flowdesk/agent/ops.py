"""Everything an agent can do in FlowDesk, as plain functions on a workspace controller.

``Ops`` is the single place where agent requests meet the application.  It is used by the in-app bridge (window or
headless) and in tests; the MCP server only forwards to it.  Every operation goes through the same controller methods
as the mouse does, so undo, signals and group propagation behave identically.  Operations raise ``OpError`` for
mistakes the caller can fix; the bridge turns those into normal error results.
"""
from __future__ import annotations

import contextlib
import io
import time
import traceback

from .ops_analysis import AnalysisOps
from .ops_base import OpError, OpsBase, clean, op
from .ops_export import ExportOps
from .ops_gating import GatingOps
from .ops_plot import PlotOps
from .ops_stats import StatsOps
from .ops_workspace import WorkspaceOps


class ControlOps:
    @op
    def set_scale(self, parameter: str, kind: str, samples=None, group: str | None = None, minimum: float | None = None,
                  maximum: float | None = None, top: float | None = None, positive_decades: float | None = None,
                  negative_decades: float | None = None, width_basis: float | None = None, decades: float | None = None) -> dict:
        """Change an axis transform: linear (minimum, maximum), logicle/biex (top, positive_decades, negative_decades,
        width_basis), log (minimum, maximum) or arcsinh (top, decades). Applies to the raw and compensated parameter."""
        from ..core.transforms import scale_from_dict

        kind = {"biexponential": "logicle", "biex": "logicle", "logarithmic": "log", "lin": "linear"}.get(str(kind).lower(), str(kind).lower())
        if kind not in ("linear", "logicle", "log", "arcsinh"):
            raise OpError("kind must be linear, logicle (biexponential), log or arcsinh")
        spec: dict = {"kind": kind}
        for key, val in (("min", minimum), ("max", maximum), ("T", top), ("pos", positive_decades), ("neg", negative_decades),
                         ("width_basis", width_basis), ("M", decades)):
            if val is not None:
                spec[key] = float(val)
        if kind == "logicle" and "pos" in spec:
            spec.setdefault("neg", 0.0)
        try:
            scale = scale_from_dict(spec)
        except Exception as exc:
            raise OpError("could not build that scale: %s" % exc)
        targets = self.samples_of(samples, group)
        name = self.param(targets[0], parameter)
        for s in targets:
            if name not in s.parameter_names():
                raise OpError("%s has no parameter %s" % (s.name, name))
        self.c.set_scale(targets, name, scale)
        self.note("set_scale", parameter=name, scale_kind=kind)
        return {"ok": True, "parameter": name, "kind": kind, "samples": [s.name for s in targets]}

    @op
    def python(self, code: str, undoable: bool = False, undo_label: str = "agent script") -> dict:
        """Run Python inside this FlowDesk process. Available: `controller`, `workspace`, `window` (None when headless), `np`,
        `ops` and the `flowdesk` package; assign `result` to return data. With undoable=True the script is one undo step and the
        views refresh afterwards. Prefer the structured tools; use this for anything they do not cover."""
        import numpy as np

        import flowdesk

        if undoable:
            self.c.begin_change(undo_label)
        ns = {"controller": self.c, "workspace": self.ws, "ws": self.ws, "window": self.window, "np": np, "ops": self,
              "flowdesk": flowdesk, "result": None}
        buf = io.StringIO()
        err = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, "<agent>", "exec"), ns)  # noqa: S102 - the whole point of this tool
        except Exception:
            err = traceback.format_exc()
        if undoable:
            self.c.mark_modified()
            for sig in ("gatingChanged", "samplesChanged", "groupsChanged"):
                try:
                    s = getattr(self.c, sig)
                    s.emit(None) if sig == "gatingChanged" else s.emit()
                except Exception:
                    pass
        self.note("python", ok=err is None, chars=len(code))
        out = {"ok": err is None, "stdout": buf.getvalue(), "result": ns.get("result")}
        if err:
            out["traceback"] = err
        return out

    @op
    def events(self, since: int = 0, limit: int = 100) -> dict:
        """What happened since `since` (an id from a previous call): operations run, workspace changes, undo steps."""
        if self.event_log is None:
            return {"events": [], "seq": 0}
        evs, truncated = self.event_log.since(int(since or 0), int(limit))
        for e in evs:
            e["age_s"] = clean(time.time() - e["t"], 1)
        return {"events": evs, "seq": self.event_log.seq, "truncated": truncated}

    @op
    def log(self, n: int = 50) -> dict:
        """The last n events as plain text lines."""
        if self.event_log is None:
            return {"lines": []}
        evs, _ = self.event_log.since(0, 100000)
        lines = []
        for e in evs[-int(n):]:
            rest = ", ".join("%s=%s" % (k, v) for k, v in e.items() if k not in ("seq", "t", "kind"))
            lines.append("%s  %s  %s" % (time.strftime("%H:%M:%S", time.localtime(e["t"])), e["kind"], rest))
        return {"lines": lines}


class Ops(WorkspaceOps, GatingOps, StatsOps, PlotOps, AnalysisOps, ExportOps, ControlOps, OpsBase):
    """All agent operations on one workspace controller."""

    def names(self) -> list[str]:
        return sorted(n for n in dir(self) if getattr(getattr(type(self), n, None), "_is_op", False))

    def call(self, name: str, args: dict | None = None):
        fn = getattr(self, name, None)
        if name.startswith("_") or not getattr(fn, "_is_op", False):
            raise OpError("unknown operation %r; available: %s" % (name, ", ".join(self.names())))
        return fn(**(args or {}))
