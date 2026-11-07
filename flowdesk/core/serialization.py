"""Save and load workspaces as JSON (.fdw files)."""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path, PurePath, PureWindowsPath

from .compensation import SpilloverMatrix
from .gates import gate_from_dict
from .layouts import Layout, TableDef
from .model import Criterion, GraphSettings, Group, Population, Sample, StatNode, Workspace
from .transforms import scale_from_dict

FORMAT_VERSION = 1
EXTENSION = ".fdw"


# ------------------------------------------------------------------ populations
def _plain_copy(value):
    """Copy nested lists / dicts (JSON-like data) so the result shares no
    mutable container with the object it was read from."""
    if isinstance(value, dict):
        return {k: _plain_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_plain_copy(v) for v in value)
    return value


def population_to_dict(pop: Population) -> dict:
    return {
        "name": pop.name,
        "id": pop.id,
        "gate": _plain_copy(pop.gate.to_dict()) if pop.gate is not None else None,
        "graph": pop.graph.to_dict(),
        "stats": [s.to_dict() for s in pop.stats],
        "expanded": pop.expanded,
        "children": [population_to_dict(c) for c in pop.children],
    }


def population_from_dict(d: dict) -> Population:
    gate = gate_from_dict(d["gate"]) if d.get("gate") else None
    pop = Population(d.get("name", "Population"), gate)
    pop.id = d.get("id", pop.id)
    pop.graph = GraphSettings.from_dict(d.get("graph"))
    pop.stats = [StatNode.from_dict(s) for s in d.get("stats", [])]
    pop.expanded = bool(d.get("expanded", True))
    for c in d.get("children", []):
        pop.add_child(population_from_dict(c))
    return pop


def copy_population(pop: Population) -> Population:
    return population_from_dict(population_to_dict(pop))


# ----------------------------------------------------------------------- samples
def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:80]


def sample_to_dict(sample: Sample, base_dir: Path | None, data_dir: Path | None = None) -> dict:
    rel = None
    if base_dir is not None:
        try:
            # stored in POSIX form so that a workspace written on Windows opens on
            # macOS/Linux and vice versa
            rel = PurePath(os.path.relpath(sample.path, base_dir)).as_posix()
        except ValueError:  # different drives on Windows
            rel = None
    computed: dict[str, dict] = {}
    if sample.computed and data_dir is not None:
        import numpy as np

        data_dir.mkdir(parents=True, exist_ok=True)
        for i, (name, values) in enumerate(sample.computed.items()):
            # the index keeps "tSNE 1" and "tSNE_1" apart after name sanitising
            fname = f"sample{sample.id}_{i}_{_safe_name(name)}.npy"
            np.save(data_dir / fname, values)
            computed[name] = {"file": fname, "info": dict(sample.computed_info.get(name, {}))}
    return {
        "computed": computed,
        "id": sample.id,
        "path": str(sample.path),
        "rel_path": rel,
        "name": sample.name,
        "scales": {k: v.to_dict() for k, v in sample.scales.items()},
        "comp": sample.comp.to_dict() if sample.comp is not None else None,
        "comp_enabled": sample.comp_enabled,
        "comp_cleared": sample.comp_cleared,
        "derived": dict(sample.derived),
        "keywords": dict(sample.keywords),
        "root": population_to_dict(sample.root),
    }


def sample_from_dict(d: dict, base_dir: Path | None, data_dir: Path | None = None) -> Sample:
    path = Path(d["path"])
    if not path.exists() and base_dir is not None and d.get("rel_path"):
        # normpath, not resolve(): keep the spelling the user sees (resolve()
        # expands Windows 8.3 short names and symlinks, which surprises nobody
        # except path comparisons)
        candidate = Path(os.path.normpath(base_dir / d["rel_path"]))
        if candidate.exists():
            path = candidate
    if not path.exists() and base_dir is not None:
        # a path written on Windows has backslashes that POSIX does not split
        name = PureWindowsPath(d["path"]).name if "\\" in d["path"] else path.name
        candidate = base_dir / name
        if candidate.exists():
            path = candidate
    sample = Sample(path, int(d["id"]))
    sample.name = d.get("name", sample.name)
    sample.keywords = dict(d.get("keywords", {}))
    if sample.keywords:
        from .fcs import channels_from_keywords

        sample.channels = channels_from_keywords(sample.keywords)
    sample.scales = {k: scale_from_dict(v) for k, v in d.get("scales", {}).items()}
    sample.comp = SpilloverMatrix.from_dict(d["comp"]) if d.get("comp") else None
    sample.comp_enabled = bool(d.get("comp_enabled", True))
    sample.comp_cleared = bool(d.get("comp_cleared", False))
    sample.derived = dict(d.get("derived", {}))
    root = population_from_dict(d["root"]) if d.get("root") else Population(sample.name)
    root.name = sample.name
    sample.root = root
    root._attach(sample)
    if not sample.keywords:
        sample.load_header()
    if d.get("computed") and data_dir is not None:
        import numpy as np

        for name, entry in d["computed"].items():
            f = data_dir / entry.get("file", "")
            if f.exists():
                try:
                    arr = np.load(f).astype(np.float32)
                except (OSError, ValueError):
                    continue
                total = int(float(sample.keywords.get("$TOT", 0) or 0))
                if arr.ndim != 1 or (total and arr.shape[0] != total):
                    continue  # the FCS file was replaced; a stale array would break every mask
                sample.computed[name] = arr
                sample.computed_info[name] = dict(entry.get("info", {}))
    return sample


# ------------------------------------------------------------------------ groups
def group_to_dict(group: Group) -> dict:
    return {
        "id": group.id,
        "name": group.name,
        "role": group.role,
        "criteria": [c.to_dict() for c in group.criteria],
        "sample_ids": sorted(group.sample_ids),
        "root": population_to_dict(group.root),
        "color": group.color,
        "bold": group.bold,
        "live": group.live,
    }


def group_from_dict(d: dict) -> Group:
    g = Group(d.get("name", "Group"), d.get("role", "custom"), [Criterion.from_dict(c) for c in d.get("criteria", [])], d.get("color", "#000000"), bool(d.get("bold", False)))
    g.id = d.get("id", g.id)
    g.sample_ids = set(int(i) for i in d.get("sample_ids", []))
    g.live = bool(d.get("live", True))
    if d.get("root"):
        g.root = population_from_dict(d["root"])
        g.root.name = g.name
    return g


# --------------------------------------------------------------------- workspace
def data_dir_for(path: Path | str | None) -> Path | None:
    """Folder holding computed-parameter arrays next to a workspace file."""
    if path is None:
        return None
    path = Path(path)
    return path.with_name(path.name + ".data")


def workspace_to_dict(ws: Workspace, base_dir: Path | None = None, data_dir: Path | None = None) -> dict:
    base_dir = base_dir if base_dir is not None else (ws.path.parent if ws.path else None)
    return {
        "format": "flowdesk-workspace",
        "version": FORMAT_VERSION,
        "settings": dict(ws.settings),
        "samples": [sample_to_dict(s, base_dir, data_dir) for s in ws.samples],
        "groups": [group_to_dict(g) for g in ws.groups],
        "matrices": {k: v.to_dict() for k, v in ws.matrices.items()},
        "layouts": [l.to_dict() for l in ws.layouts],
        "tables": [t.to_dict() for t in ws.tables],
    }


def workspace_from_dict(d: dict, base_dir: Path | None = None, data_dir: Path | None = None) -> Workspace:
    ws = Workspace()
    ws.settings = dict(d.get("settings", {}))
    ws.samples = [sample_from_dict(s, base_dir, data_dir) for s in d.get("samples", [])]
    groups = [group_from_dict(g) for g in d.get("groups", [])]
    if groups:
        ws.groups = groups
    if not any(g.role == "all" for g in ws.groups):
        ws.groups.insert(0, Group("All Samples", role="all", bold=True))
    ws.matrices = {k: SpilloverMatrix.from_dict(v) for k, v in d.get("matrices", {}).items()}
    ws.layouts = [Layout.from_dict(l) for l in d.get("layouts", [])]
    ws.tables = [TableDef.from_dict(t) for t in d.get("tables", [])]
    ws.modified = False
    return ws


def save_workspace(ws: Workspace, path: Path | str):
    path = Path(path)
    data_dir = data_dir_for(path) if any(s.computed for s in ws.samples) else None
    data = workspace_to_dict(ws, path.parent, data_dir)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    ws.path = path
    ws.modified = False


def load_workspace(path: Path | str) -> Workspace:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") != "flowdesk-workspace":
        raise ValueError("not a FlowDesk workspace file")
    ws = workspace_from_dict(data, path.parent, data_dir_for(path))
    ws.path = path
    return ws




# ------------------------------------------------------------------- snapshots
# An undo snapshot is an in-memory record of the state undo brings back. It
# always holds the analysis state of every sample (gating tree with statistics
# and graph settings, scales, compensation, derived and computed parameters)
# and the groups; optional sections add the rest:
#   samples   the sample list and its order (Sample objects are kept by
#             reference so undoing a removal brings back the very same
#             object), sample names and keywords
#   matrices  the workspace compensation matrices
#   settings  workspace.settings (deep copy; view-only keys are left alone)
#   layouts   the layouts
#   tables    the tables
# Snapshots stay cheap: trees, scales, groups and matrices are serialised
# (a few dicts per population); the per-event float32 arrays of computed
# parameters (and their computed_info) are kept by reference. That is safe
# because they are only ever replaced (Sample.add_computed / remove_computed),
# never written into, so a snapshot costs O(populations + keywords), not
# O(events).

SNAPSHOT_SECTIONS = ("samples", "matrices", "settings", "layouts", "tables")
VIEW_SETTINGS = frozenset({"columns"})  # workspace-tree keyword columns: view state, never undone
_OPTIONAL_SAMPLE_ATTRS = ("comp_override",)  # compensation state some builds add to Sample
_MISSING = object()


def snapshot_sections(kind=None) -> frozenset:
    """Optional sections covered by a snapshot `kind`: "gating" (or None) for
    none, "all" for every section, or one section name / an iterable of them."""
    if kind is None or kind == "gating":
        return frozenset()
    if isinstance(kind, str):
        kind = (kind,)
    out: set[str] = set()
    for k in kind:
        if k == "all":
            return frozenset(SNAPSHOT_SECTIONS)
        if k == "gating":
            continue
        if k not in SNAPSHOT_SECTIONS:
            raise ValueError(f"unknown undo snapshot section {k!r}")
        out.add(k)
    return frozenset(out)


def snapshot_kind(snap: dict) -> frozenset:
    """The optional sections a snapshot holds (pass it back to gating_snapshot
    to take a snapshot of the same kind)."""
    return frozenset(snap.get("sections", ()))


def _equal(a, b) -> bool:
    try:
        return bool(a == b)
    except (ValueError, TypeError):  # numpy arrays somewhere inside: assume changed
        return False


def _same_arrays(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k] is b[k] for k in a)


def _sample_entry(s: Sample, sections: frozenset) -> dict:
    entry = {
        "root": population_to_dict(s.root),
        "scales": {k: v.to_dict() for k, v in s.scales.items()},
        "comp": s.comp.to_dict() if s.comp is not None else None,
        "comp_enabled": s.comp_enabled,
        "comp_cleared": s.comp_cleared,
        "derived": {k: _plain_copy(v) for k, v in s.derived.items()},
        # references, see above; computed_info is replaced together with its
        # array (add_computed), so a shallow copy suffices and its possibly
        # large "model" entry is shared rather than copied
        "computed": dict(s.computed),
        "computed_info": {k: dict(v) for k, v in s.computed_info.items()},
    }
    for attr in _OPTIONAL_SAMPLE_ATTRS:
        value = getattr(s, attr, _MISSING)
        if value is not _MISSING:
            entry[attr] = copy.deepcopy(value)
    if "samples" in sections:
        entry["name"] = s.name
        entry["keywords"] = dict(s.keywords)
    return entry


def _add_sections(ws: Workspace, snap: dict, sections: frozenset):
    if "samples" in sections:
        snap["order"] = [s.id for s in ws.samples]
        snap["objects"] = {s.id: s for s in ws.samples}
    if "matrices" in sections:
        snap["matrices"] = {k: v.to_dict() for k, v in ws.matrices.items()}
    if "settings" in sections:
        snap["settings"] = {k: copy.deepcopy(v) for k, v in ws.settings.items() if k not in VIEW_SETTINGS}
    if "layouts" in sections:
        snap["layouts"] = [l.to_dict() for l in ws.layouts]
    if "tables" in sections:
        snap["tables"] = [t.to_dict() for t in ws.tables]


def gating_snapshot(ws: Workspace, kind=None) -> dict:
    """An undo snapshot of `ws` (no event data is copied). `kind` selects the
    optional sections, see snapshot_sections()."""
    sections = snapshot_sections(kind)
    snap = {
        "sections": sections,
        "samples": {s.id: _sample_entry(s, sections) for s in ws.samples},
        "groups": [group_to_dict(g) for g in ws.groups],
        # references: undoing a group's deletion brings back the same object
        "group_objects": {g.id: g for g in ws.groups},
    }
    _add_sections(ws, snap, sections)
    return snap


def extend_snapshot(ws: Workspace, snap: dict, kind) -> dict:
    """Add the sections of `kind` that `snap` lacks, read from the present
    state (a nested change inside one undo step needing more than the outer
    change recorded)."""
    missing = snapshot_sections(kind) - snapshot_kind(snap)
    if not missing:
        return snap
    if "samples" in missing:
        for s in ws.samples:
            entry = snap["samples"].get(s.id)
            if entry is not None:
                entry["name"] = s.name
                entry["keywords"] = dict(s.keywords)
    _add_sections(ws, snap, missing)
    snap["sections"] = snapshot_kind(snap) | missing
    return snap


def _merge_population(pop: Population, d: dict):
    """Make `pop` and its subtree match `d` in place. Children whose name (and
    so whose path) exists in both are kept as the same objects."""
    pop.id = d.get("id", pop.id)
    if pop.parent is not None:
        pop.name = d.get("name", pop.name)
    gd = d.get("gate")
    if gd is None:
        pop.gate = None
    elif pop.gate is None or not _equal(pop.gate.to_dict(), gd):
        pop.gate = gate_from_dict(gd)
    graph = GraphSettings.from_dict(d.get("graph"))
    for f in dataclasses.fields(graph):  # in place: graph windows hold pop.graph
        setattr(pop.graph, f.name, getattr(graph, f.name))
    old_stats = {st.id: st for st in pop.stats}
    stats = []
    for sd in d.get("stats", []):
        st = old_stats.get(sd.get("id"))
        stats.append(st if st is not None and _equal(st.to_dict(), sd) else StatNode.from_dict(sd))
    pop.stats = stats
    pop.expanded = bool(d.get("expanded", True))
    pool: dict[str, list[Population]] = {}
    for c in pop.children:
        pool.setdefault(c.name, []).append(c)
    children = []
    for cd in d.get("children", []):
        same = pool.get(cd.get("name"))
        if same:
            child = same.pop(0)
            _merge_population(child, cd)
        else:
            child = population_from_dict(cd)
        children.append(child)
    for rest in pool.values():
        for c in rest:
            c.parent = None
    pop.children = children
    for c in children:
        c.parent = pop
    pop._mask = None


def _restore_sample(s: Sample, entry: dict, cur: dict) -> set[str]:
    changes: set[str] = set()
    dirty = False
    if "name" in entry and entry["name"] != s.name:
        s.name = entry["name"]
        changes.add("samples")
    if "keywords" in entry and not _equal(entry["keywords"], s.keywords):
        s.keywords = dict(entry["keywords"])
        s.refresh_channels_from_keywords()
        changes.add("samples")
        dirty = True
    if not _equal(entry["root"], cur["root"]):
        _merge_population(s.root, entry["root"])
        s.root._attach(s)
        changes.add("gating")
        dirty = True
    s.root.name = s.name
    if not _equal(entry["scales"], cur["scales"]):
        old = s.scales
        s.scales = {k: (old[k] if k in old and _equal(cur["scales"].get(k), d) else scale_from_dict(d)) for k, d in entry["scales"].items()}
        changes.add("scales")
        dirty = True
    comp_dirty = False
    if not _equal(entry["comp"], cur["comp"]):
        s.comp = SpilloverMatrix.from_dict(entry["comp"]) if entry["comp"] else None
        comp_dirty = True
    if bool(entry["comp_enabled"]) != s.comp_enabled or bool(entry["comp_cleared"]) != s.comp_cleared:
        s.comp_enabled = bool(entry["comp_enabled"])
        s.comp_cleared = bool(entry["comp_cleared"])
        comp_dirty = True
    for attr in _OPTIONAL_SAMPLE_ATTRS:
        if attr in entry and not _equal(entry[attr], cur.get(attr, _MISSING)):
            setattr(s, attr, copy.deepcopy(entry[attr]))
            comp_dirty = True
    if comp_dirty:
        s._comp_cache = None
        changes.add("compensation")
        dirty = True
    if not _equal(entry["derived"], cur["derived"]):
        s.derived = {k: _plain_copy(v) for k, v in entry["derived"].items()}
        changes.add("samples")
        dirty = True
    if not _same_arrays(entry["computed"], s.computed) or not _equal(entry["computed_info"], cur["computed_info"]):
        s.computed = dict(entry["computed"])
        s.computed_info = {k: dict(v) for k, v in entry["computed_info"].items()}
        changes.add("samples")
        dirty = True
    if dirty:
        s._derived_cache.clear()
        s.invalidate_display()
    return changes


def _restore_groups(ws: Workspace, dicts: list[dict], objects: dict | None = None):
    pool: dict[str, list[Group]] = {}
    for g in ws.groups:
        pool.setdefault(g.id, []).append(g)
    for gid, g in (objects or {}).items():
        if gid not in pool:
            pool[gid] = [g]
    out = []
    for d in dicts:
        same = pool.get(d.get("id"))
        if not same:
            out.append(group_from_dict(d))
            continue
        g = same.pop(0)
        g.name = d.get("name", g.name)
        g.role = d.get("role", g.role)
        g.criteria = [Criterion.from_dict(c) for c in d.get("criteria", [])]
        g.sample_ids = set(int(i) for i in d.get("sample_ids", []))
        g.color = d.get("color", g.color)
        g.bold = bool(d.get("bold", False))
        g.live = bool(d.get("live", True))
        if d.get("root"):
            _merge_population(g.root, d["root"])
        g.root.name = g.name
        g.root._attach(None)
        out.append(g)
    ws.groups[:] = out


def _update_in_place(old, new):
    for f in dataclasses.fields(new):
        value = getattr(new, f.name)
        current = getattr(old, f.name, None)
        if isinstance(value, list) and isinstance(current, list) and value and all(dataclasses.is_dataclass(v) and hasattr(v, "id") and hasattr(v, "to_dict") for v in value):
            value = _restore_by_id(current, [v.to_dict() for v in value], type(value[0]).from_dict)
        setattr(old, f.name, value)


def _restore_by_id(objs: list, dicts: list[dict], from_dict) -> list:
    """Objects for `dicts`, reusing (and updating in place) those of `objs`
    with the same id."""
    pool: dict = {}
    for o in objs:
        pool.setdefault(getattr(o, "id", None), []).append(o)
    out = []
    for d in dicts:
        same = pool.get(d.get("id"))
        if not same:
            out.append(from_dict(d))
            continue
        o = same.pop(0)
        if not _equal(o.to_dict(), d):
            new = from_dict(d)
            if dataclasses.is_dataclass(o) and type(o) is type(new):
                _update_in_place(o, new)
            else:
                o = new
        out.append(o)
    return out


def restore_snapshot(ws: Workspace, snap: dict, current: dict | None = None) -> set[str]:
    """Bring `ws` back to the state recorded in `snap`, in place: Sample,
    Group, Layout and TableDef objects keep their identity, and so does every
    Population whose path exists before and after. `current` is a snapshot of
    the present state with the same sections (taken here when omitted; undo
    passes its redo snapshot). Returns what changed, a subset of {"samples",
    "gating", "scales", "compensation", "groups", "matrices", "settings",
    "layouts", "tables"} ("samples": list/order, names, keywords, derived or
    computed parameters)."""
    sections = snapshot_kind(snap)
    if current is None:
        current = gating_snapshot(ws, sections)
    changes: set[str] = set()
    if "samples" in sections and "order" in snap:
        objects = snap.get("objects", {})
        present = {s.id: s for s in ws.samples}
        restored = [s for s in (objects.get(i) or present.get(i) for i in snap["order"]) if s is not None]
        if len(restored) != len(ws.samples) or any(a is not b for a, b in zip(restored, ws.samples)):
            ws.samples[:] = restored
            changes.add("samples")
    current_objects = current.get("objects", {})
    for s in ws.samples:
        entry = snap["samples"].get(s.id)
        if entry is None:
            continue  # added after the snapshot by a change it does not cover
        cur = current["samples"].get(s.id)
        if cur is None or current_objects.get(s.id, s) is not s:
            cur = _sample_entry(s, sections)
        changes |= _restore_sample(s, entry, cur)
    if not _equal(snap["groups"], current["groups"]):
        _restore_groups(ws, snap["groups"], snap.get("group_objects"))
        changes.add("groups")
    if "matrices" in sections and not _equal(snap["matrices"], current.get("matrices")):
        old = dict(ws.matrices)
        ws.matrices.clear()
        for k, d in snap["matrices"].items():
            m = old.get(k)
            ws.matrices[k] = m if m is not None and _equal(m.to_dict(), d) else SpilloverMatrix.from_dict(d)
        changes.add("matrices")
    if "settings" in sections and not _equal(snap["settings"], current.get("settings")):
        for k in [k for k in ws.settings if k not in VIEW_SETTINGS and k not in snap["settings"]]:
            del ws.settings[k]
        for k, v in snap["settings"].items():
            ws.settings[k] = copy.deepcopy(v)
        changes.add("settings")
    if "layouts" in sections and not _equal(snap["layouts"], current.get("layouts")):
        ws.layouts[:] = _restore_by_id(ws.layouts, snap["layouts"], Layout.from_dict)
        changes.add("layouts")
    if "tables" in sections and not _equal(snap["tables"], current.get("tables")):
        ws.tables[:] = _restore_by_id(ws.tables, snap["tables"], TableDef.from_dict)
        changes.add("tables")
    ws.modified = True
    return changes
