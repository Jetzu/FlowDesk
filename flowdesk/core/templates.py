"""Workspace templates: the analysis without the samples.

A template carries groups with their gating trees, axis scales per parameter,
compensation matrices, layouts, tables and workspace settings, so that a new
set of samples can be analysed the way a previous experiment was. The same
idea cytometrists know as a workspace template (.wspt).
"""
from __future__ import annotations

import json
from pathlib import Path

from .compensation import SpilloverMatrix
from .layouts import Layout, TableDef
from .model import Group, Workspace
from .transforms import scale_from_dict

TEMPLATE_EXTENSION = ".fdwt"


def build_template(ws: Workspace) -> dict:
    from .serialization import group_to_dict

    groups = []
    for g in ws.groups:
        d = group_to_dict(g)
        d["sample_ids"] = []
        groups.append(d)
    scales: dict[str, dict] = {}
    for s in ws.samples:
        for name, scale in s.scales.items():
            scales.setdefault(name, scale.to_dict())
    return {
        "format": "flowdesk-template",
        "version": 1,
        "groups": groups,
        "scales": scales,
        "matrices": {k: v.to_dict() for k, v in ws.matrices.items()},
        "layouts": [l.to_dict() for l in ws.layouts],
        "tables": [t.to_dict() for t in ws.tables],
        "settings": {k: v for k, v in ws.settings.items() if k not in ("imported_from",)},
    }


def save_template(ws: Workspace, path: Path | str):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build_template(ws), fh, indent=1)


def load_template(path: Path | str) -> dict:
    path = Path(path)
    if path.suffix.lower() in (".wsp", ".wspt"):
        from .wsp_import import import_wsp

        ws, _ = import_wsp(path, search_dirs=[path.parent])
        return build_template(ws)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format") == "flowdesk-workspace":
        from .serialization import workspace_from_dict

        return build_template(workspace_from_dict(data, path.parent))
    if data.get("format") != "flowdesk-template":
        raise ValueError("not a FlowDesk template file")
    return data


def apply_template(ws: Workspace, template: dict, replace_groups: bool = True, apply_scales: bool = True, add_layouts: bool = True) -> dict:
    """Apply a template to a workspace. Returns a small report dict."""
    from .serialization import group_from_dict

    report = {"groups": 0, "samples": 0, "scales": 0, "matrices": 0, "layouts": 0, "tables": 0}
    new_groups = [group_from_dict(d) for d in template.get("groups", [])]
    if replace_groups and new_groups:
        existing_ids = {g.name: g.sample_ids for g in ws.groups}
        for g in new_groups:
            g.sample_ids = set(existing_ids.get(g.name, set()))
        if not any(g.role == "all" for g in new_groups):
            new_groups.insert(0, Group("All Samples", role="all", bold=True))
        ws.groups = new_groups
    else:
        for g in new_groups:
            existing = ws.group_by_name(g.name)
            if existing is None:
                ws.groups.append(g)
            else:
                ws._merge_into(g.root, existing.root)
    report["groups"] = len(new_groups)
    for g in ws.groups:
        for s in ws.members(g):
            ws.apply_template(g, s)
            report["samples"] += 1
    if apply_scales:
        for s in ws.samples:
            for name, d in template.get("scales", {}).items():
                if name in s.channel_names or s.base_channel(name) is not None:
                    s.scales[name] = scale_from_dict(d)
                    report["scales"] += 1
            s.invalidate_display()
    for name, d in template.get("matrices", {}).items():
        ws.matrices[name] = SpilloverMatrix.from_dict(d)
        report["matrices"] += 1
    if add_layouts:
        existing_layout_names = {l.name for l in ws.layouts}
        for d in template.get("layouts", []):
            lay = Layout.from_dict(d)
            if lay.name in existing_layout_names:
                lay.name = f"{lay.name} (template)"
            ws.layouts.append(lay)
            report["layouts"] += 1
        existing_table_names = {t.name for t in ws.tables}
        for d in template.get("tables", []):
            tab = TableDef.from_dict(d)
            if tab.name in existing_table_names:
                tab.name = f"{tab.name} (template)"
            ws.tables.append(tab)
            report["tables"] += 1
    for key, value in template.get("settings", {}).items():
        ws.settings.setdefault(key, value)
    ws.modified = True
    return report
