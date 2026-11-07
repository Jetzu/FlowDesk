"""Save and load workspaces as JSON (.fdw files)."""
from __future__ import annotations

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
def population_to_dict(pop: Population) -> dict:
    return {
        "name": pop.name,
        "id": pop.id,
        "gate": pop.gate.to_dict() if pop.gate is not None else None,
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
        for name, values in sample.computed.items():
            fname = f"sample{sample.id}_{_safe_name(name)}.npy"
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
                    sample.computed[name] = np.load(f).astype(np.float32)
                    sample.computed_info[name] = dict(entry.get("info", {}))
                except (OSError, ValueError):
                    continue
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
def gating_snapshot(ws: Workspace) -> dict:
    """A compact, data-free snapshot used for undo."""
    return {
        "samples": {s.id: {"root": population_to_dict(s.root), "scales": {k: v.to_dict() for k, v in s.scales.items()}, "comp": s.comp.to_dict() if s.comp else None, "comp_enabled": s.comp_enabled, "comp_cleared": s.comp_cleared, "derived": dict(s.derived)} for s in ws.samples},
        "groups": [group_to_dict(g) for g in ws.groups],
    }


def restore_snapshot(ws: Workspace, snap: dict):
    for s in ws.samples:
        entry = snap["samples"].get(s.id)
        if entry is None:
            continue
        root = population_from_dict(entry["root"])
        root.name = s.name
        s.root = root
        root._attach(s)
        s.scales = {k: scale_from_dict(v) for k, v in entry.get("scales", {}).items()}
        s.comp = SpilloverMatrix.from_dict(entry["comp"]) if entry.get("comp") else None
        s.comp_enabled = bool(entry.get("comp_enabled", True))
        s.comp_cleared = bool(entry.get("comp_cleared", False))
        if "derived" in entry:
            s.derived = dict(entry["derived"])
        s._comp_cache = None
        s._derived_cache.clear()
        s.invalidate_display()
    ws.groups = [group_from_dict(g) for g in snap["groups"]]
    ws.modified = True
