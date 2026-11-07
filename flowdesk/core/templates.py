"""Workspace templates: the analysis without the samples.

A template carries groups with their gating trees, axis scales per parameter,
compensation matrices, layouts, tables and workspace settings, so that a new
set of samples can be analysed the way a previous experiment was. The same
idea cytometrists know as a workspace template (.wspt).

A template never carries per-sample data: computed parameters (tSNE/UMAP/PCA
axes, cluster ids) are values of particular events, so neither their arrays
nor their (data-dependent) scales are stored. Gates on computed parameters
stay in the group trees: on samples without the parameter they select no
events, and they start working as soon as the parameter is computed. Derived
parameters are formulas, so their definitions travel with the template and
are recreated on every sample that has the input parameters. Likewise a bead
calibration stored in the settings ("calibrations") is re-applied to the new
samples when the template gates on its calibrated parameter ("FITC-A MESF").
"""
from __future__ import annotations

import json
from pathlib import Path

from .compensation import SpilloverMatrix
from .layouts import Layout, TableDef
from .model import Group, Workspace
from .transforms import scale_from_dict

TEMPLATE_EXTENSION = ".fdwt"


# settings entries that refer to particular samples of the source workspace
_PER_SAMPLE_SETTINGS = {"plate_editor": ("manual",)}
_SKIPPED_SETTINGS = ("imported_from",)


def _template_settings(settings: dict) -> dict:
    out = {}
    for key, value in settings.items():
        if key in _SKIPPED_SETTINGS:
            continue
        value = json.loads(json.dumps(value, default=str))  # detached plain copy
        if isinstance(value, dict):
            for sub in _PER_SAMPLE_SETTINGS.get(key, ()):
                value.pop(sub, None)
        out[key] = value
    return out


def _carries_scale(sample, name: str, computed: set[str]) -> bool:
    """Whether the scale of `name` on `sample` belongs in a template: FCS
    channels, their compensated versions and derived parameters, but not
    computed parameters (their auto-scale depends on the data) nor scales a
    sample cached for a parameter it does not have."""
    if name in computed or name in sample.computed:
        return False
    if name in sample.derived:
        return True
    return name in sample.channel_names or sample.base_channel(name) is not None


def build_template(ws: Workspace) -> dict:
    from .serialization import group_to_dict

    groups = []
    for g in ws.groups:
        d = group_to_dict(g)
        d["sample_ids"] = []
        groups.append(d)
    computed: set[str] = set()
    for s in ws.samples:
        computed.update(s.computed)
    scales: dict[str, dict] = {}
    derived: dict[str, dict] = {}
    for s in ws.samples:
        for name, spec in s.derived.items():
            if name not in computed:
                derived.setdefault(name, dict(spec))
        for name, scale in s.scales.items():
            if _carries_scale(s, name, computed):
                scales.setdefault(name, scale.to_dict())
    return {
        "format": "flowdesk-template",
        "version": 1,
        "groups": groups,
        "scales": scales,
        "derived": derived,
        "matrices": {k: v.to_dict() for k, v in ws.matrices.items()},
        "layouts": [l.to_dict() for l in ws.layouts],
        "tables": [t.to_dict() for t in ws.tables],
        "settings": _template_settings(ws.settings),
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

    report = {"groups": 0, "samples": 0, "scales": 0, "derived": 0, "calibrated": 0, "matrices": 0, "layouts": 0, "tables": 0}
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
    for s in ws.samples:
        report["derived"] += _add_derived(s, template.get("derived") or {})
    report["calibrated"] = _recreate_calibrations(ws, template)
    if apply_scales:
        for s in ws.samples:
            for name, d in template.get("scales", {}).items():
                if name in s.computed:
                    continue  # computed parameters keep their own auto-scale
                if name in s.channel_names or s.base_channel(name) is not None or name in s.derived:
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


def _add_derived(sample, definitions: dict) -> int:
    """Recreate the template's derived parameters on `sample` where its
    inputs exist and the name is free. Returns the number added."""
    n = 0
    for name, spec in definitions.items():
        if not isinstance(spec, dict) or name in sample.derived or name in sample.computed or name in sample.channel_names:
            continue
        available = set(sample.parameter_names())
        inputs = [spec.get("a"), spec.get("b")] if spec.get("b") else [spec.get("a")]
        if not all(i and (i in available or sample.base_channel(i) is not None) for i in inputs):
            continue
        sample.add_derived(name, spec)
        n += 1
    return n


def _gate_dims(template: dict) -> set[str]:
    """Every parameter a gate of the template's group trees uses."""
    dims: set[str] = set()

    def walk(node: dict):
        gate = node.get("gate") or {}
        dims.update(d for d in gate.get("dims") or [] if d)
        for child in node.get("children") or []:
            walk(child)

    for g in template.get("groups") or []:
        if g.get("root"):
            walk(g["root"])
    return dims


def _recreate_calibrations(ws: Workspace, template: dict) -> int:
    """Re-apply the template's bead calibrations whose calibrated parameter is
    gated on to the samples that have the source parameter (a calibrated
    parameter is a computed one, so its values are never in the template).
    Returns the number of (sample, parameter) pairs created."""
    stored = (template.get("settings") or {}).get("calibrations") or {}
    if not isinstance(stored, dict) or not stored:
        return 0
    try:
        from . import calibration as cal
    except ImportError:  # pragma: no cover - module not available
        return 0
    used = _gate_dims(template)
    n = 0
    for param, d in stored.items():
        try:
            fit = cal.calibration_from_dict(d)
            name = cal.calibrated_name(param, fit.units)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if name not in used:
            continue
        for s in ws.samples:
            if name in s.computed or param not in s.parameter_names():
                continue
            try:
                cal.apply_calibration(s, fit, param, name)
            except (KeyError, OSError, ValueError):
                continue
            n += 1
    return n
