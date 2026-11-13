"""Shared plumbing of the agent operations: errors, the ``@op`` marker and tolerant name resolution.

An agent names things loosely ("donor1 control", "CD4", "/Lymphocytes/CD3+"), so every reference is resolved here
with clear errors that list the candidates instead of failing silently.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

import numpy as np


class OpError(ValueError):
    """A mistake the caller can fix (unknown sample, bad argument): reported as a normal error result."""


def op(fn):
    """Mark a method as an operation callable through the bridge."""
    fn._is_op = True
    return fn


def clean(value: Any, digits: int = 6):
    """JSON-friendly number: NaN and inf become None, floats are rounded."""
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(f) or math.isinf(f):
        return None
    if float(f).is_integer() and abs(f) < 1e15:
        return int(f)
    return round(f, digits)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _loose(text: Any) -> str:
    """Lower case with _ - . and spaces treated alike: "donor1 control" matches "Donor1_Control.fcs"."""
    return re.sub(r"[\s_\-.]+", " ", str(text)).strip().lower()


def _stem(name: str) -> str:
    return name[:-4] if name.lower().endswith(".fcs") else name


def listing(items: Iterable[str], limit: int = 12) -> str:
    items = list(items)
    shown = ", ".join(repr(i) for i in items[:limit])
    return shown + (" ... (%d more)" % (len(items) - limit) if len(items) > limit else "")


class OpsBase:
    """Holds the controller and resolves references; all operations run on the GUI thread (see hostapp)."""

    def __init__(self, controller, events=None, window=None):
        self.c = controller
        self.window = window
        self.event_log = events

    @property
    def ws(self):
        return self.c.workspace

    def note(self, kind: str, **data):
        if self.event_log is not None:
            self.event_log.add(kind, **data)

    # ------------------------------------------------------------------ samples
    def sample(self, ref) -> Any:
        """A sample from an id (3 or "#3"), a file name (with or without .fcs) or a unique fragment of one."""
        samples = self.ws.samples
        if not samples:
            raise OpError("the workspace has no samples; add some with add_samples or open_workspace")
        if isinstance(ref, bool) or ref is None or ref == "":
            raise OpError("no sample given")
        if isinstance(ref, int) or (isinstance(ref, str) and re.fullmatch(r"#?\d+", ref.strip())):
            sid = int(str(ref).lstrip("#"))
            s = self.ws.sample_by_id(sid)
            if s is not None:
                return s
        key = _norm(ref)
        exact = [s for s in samples if _norm(s.name) == key or _norm(_stem(s.name)) == key]
        if len(exact) == 1:
            return exact[0]
        if not exact:
            lk = _loose(ref)
            exact = [s for s in samples if _loose(s.name) == lk or _loose(_stem(s.name)) == lk]
            if len(exact) == 1:
                return exact[0]
        part = [s for s in samples if key in _norm(s.name) or _loose(ref) in _loose(s.name)]
        if len(exact) > 1 or len(part) > 1:
            raise OpError("%r matches several samples: %s" % (ref, listing(s.name for s in (exact or part))))
        if len(part) == 1:
            return part[0]
        raise OpError("no sample %r; the samples are: %s" % (ref, listing(s.name for s in samples)))

    def samples_of(self, samples=None, group=None) -> list:
        """The samples an operation applies to: explicit list, a group, or (default) every sample."""
        if samples not in (None, "", [], "all", "*"):
            refs = [samples] if isinstance(samples, (str, int)) else list(samples)
            out = []
            for r in refs:
                if isinstance(r, str) and r.lower().startswith("group:"):
                    out.extend(self.ws.members(self.group(r[6:])))
                else:
                    out.append(self.sample(r))
            return list(dict.fromkeys(out))
        if group:
            return list(self.ws.members(self.group(group)))
        if not self.ws.samples:
            raise OpError("the workspace has no samples")
        return list(self.ws.samples)

    def group(self, name) -> Any:
        groups = self.ws.groups
        if isinstance(name, str) and name.lower().startswith("group:"):
            name = name[6:]
        key = _norm(name)
        hit = [g for g in groups if _norm(g.name) == key]
        if not hit:
            hit = [g for g in groups if key in _norm(g.name)]
        if len(hit) == 1:
            return hit[0]
        if len(hit) > 1:
            raise OpError("%r matches several groups: %s" % (name, listing(g.name for g in hit)))
        raise OpError("no group %r; the groups are: %s" % (name, listing(g.name for g in groups)))

    # -------------------------------------------------------------- populations
    def population(self, sample, ref) -> Any:
        """A population of `sample` from its path ("/Lymphocytes/CD3+"), a unique name ("CD3+") or "" for ungated."""
        root = sample.root
        if ref in (None, "", "/", "Ungated", "ungated", "root"):
            return root
        ref = str(ref).strip()
        if ref.startswith("/"):
            hit = root.find(ref)
            if hit is not None:
                return hit
        else:
            named = [p for p in root.walk(include_self=False) if p.name == ref]
            if len(named) == 1:
                return named[0]
            if len(named) > 1:
                raise OpError("%r is ambiguous in %s: %s; use the full path" % (ref, sample.name, listing(p.path for p in named)))
            hit = root.find("/" + ref)
            if hit is not None:
                return hit
        paths = [p.path for p in root.walk(include_self=False)]
        raise OpError("no population %r in %s; its populations are: %s" % (ref, sample.name, listing(paths) if paths else "(none: it is ungated)"))

    def parent_in(self, container_root, parent_path: str, what: str):
        parent = container_root.find(parent_path) if parent_path not in ("", "/") else container_root
        if parent is None:
            raise OpError("the parent population %r does not exist in %s" % (parent_path, what))
        return parent

    # --------------------------------------------------------------- parameters
    def param(self, sample, text) -> str:
        """A parameter name of `sample` from a name, a stain ("CD4") or a loose spelling; prefers the compensated one."""
        names = sample.parameter_names()
        if text in names:
            return text
        key = _norm(text)
        ci = [n for n in names if _norm(n) == key]
        if len(ci) == 1:
            return ci[0]
        loose = [n for n in names if _loose(n) == _loose(text)]
        if len(loose) == 1:
            return loose[0]
        stain_hits = []
        for n in names:
            try:
                stain = sample.param_stain(n)
            except Exception:
                stain = ""
            if stain and _norm(stain) == key:
                stain_hits.append(n)
        if not stain_hits:
            for n in names:
                try:
                    label = sample.param_label(n)
                except Exception:
                    label = ""
                if label and _norm(label) == key:
                    stain_hits.append(n)
        if stain_hits:
            comp = [n for n in stain_hits if sample.is_compensated_param(n)]
            return (comp or stain_hits)[0]
        partial = [n for n in names if key in _norm(n)]
        comp_partial = [n for n in partial if sample.is_compensated_param(n)]
        if len(comp_partial) == 1:
            return comp_partial[0]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise OpError("%r matches several parameters of %s: %s" % (text, sample.name, listing(partial)))
        raise OpError("no parameter %r in %s; it has: %s" % (text, sample.name, listing(names, 40)))

    # ------------------------------------------------------------------ results
    def pop_row(self, pop) -> dict:
        s = pop.sample
        parent = pop.parent
        n = pop.count
        row = {"sample": s.name if s is not None else None, "population": pop.path or "/", "count": int(n)}
        if parent is not None:
            pn = parent.count
            row["percent_of_parent"] = clean(100.0 * n / pn, 4) if pn else None
        total = s.n_events if s is not None else 0
        row["percent_of_total"] = clean(100.0 * n / total, 4) if total else None
        return row

    def describe_gate(self, gate) -> Optional[dict]:
        if gate is None:
            return None
        d = gate.to_dict()
        d = {k: v for k, v in d.items() if k not in ("id",)}
        if getattr(gate, "negated", False):
            d["negated"] = True
        return d

    def pop_tree(self, pop, depth: int = 6, with_counts: bool = True, budget: Optional[list] = None) -> dict:
        budget = budget if budget is not None else [400]
        budget[0] -= 1
        node = {"name": pop.name, "path": pop.path or "/"}
        if pop.gate is not None:
            node["gate"] = self.describe_gate(pop.gate)
        if with_counts and pop.sample is not None:
            row = self.pop_row(pop)
            node["count"] = row["count"]
            if "percent_of_parent" in row:
                node["percent_of_parent"] = row["percent_of_parent"]
        if pop.children:
            if depth <= 0 or budget[0] <= 0:
                node["children_omitted"] = len(pop.children)
            else:
                node["children"] = [self.pop_tree(c, depth - 1, with_counts, budget) for c in pop.children if budget[0] > 0]
        return node
