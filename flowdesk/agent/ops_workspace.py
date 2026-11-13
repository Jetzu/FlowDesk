"""Workspace, sample, group and undo operations."""
from __future__ import annotations

from pathlib import Path

from .ops_base import OpError, clean, listing, op


class WorkspaceOps:
    # ------------------------------------------------------------- identity
    @op
    def ping(self) -> dict:
        """Identity of this window for `connect` / `status`."""
        names = [s.name for s in self.ws.samples]
        return {"app": "flowdesk", "workspace": self.ws.name, "samples": len(names), "objects": names[:8],
                "gui": self.window is not None, "modified": bool(self.ws.modified)}

    # ---------------------------------------------------------------- state
    @op
    def workspace(self, detail: str = "summary", group: str | None = None) -> dict:
        """Structured snapshot: samples, groups, populations, parameters, matrices, undo state."""
        if detail not in ("summary", "full"):
            raise OpError("detail must be 'summary' or 'full'")
        ws, c = self.ws, self.c
        full = detail == "full"
        grp = self.group(group) if group else None
        members = ws.members(grp) if grp is not None else ws.samples
        member_groups: dict[int, list[str]] = {}
        for g in ws.groups:
            if g.role == "all":
                continue
            for s in ws.members(g):
                member_groups.setdefault(s.id, []).append(g.name)
        samples = []
        for s in members:
            entry = {"id": s.id, "name": s.name, "events": int(s.n_events), "loaded": bool(s.loaded)}
            kws = {k: v for k, v in s.keywords.items() if full or (not k.startswith("$") and len(v) < 80)}
            if not full:
                kws = dict(list(kws.items())[:12])
                if "$CYT" in s.keywords:
                    kws["$CYT"] = s.keywords["$CYT"]
            entry["keywords"] = kws
            if member_groups.get(s.id):
                entry["groups"] = member_groups[s.id]
            if s.load_error:
                entry["load_error"] = str(s.load_error)
            if s.comp is not None:
                entry["compensation"] = {"matrix": s.comp.name, "enabled": bool(s.comp_enabled)}
            samples.append(entry)
        groups = []
        for g in ws.groups:
            d = {"name": g.name, "role": g.role, "members": len(ws.members(g))}
            if g.criteria:
                d["criteria"] = ["%s %s %r" % (k.keyword, k.function, k.value) for k in g.criteria]
            groups.append(d)
        paths = ws.population_paths(grp)
        out = {
            "name": ws.name, "path": str(ws.path) if ws.path else None, "modified": bool(ws.modified),
            "samples": samples, "groups": groups,
            "populations": paths if full else paths[:80],
            "parameters": ws.all_parameter_names(grp),
            "matrices": list(ws.matrices.keys()),
            "computed_parameters": sorted({n for s in ws.samples for n in s.computed}),
            "undo": {"can_undo": c.can_undo, "next_undo": c.undo_label(), "can_redo": c.can_redo},
        }
        if len(paths) > 80 and not full:
            out["populations_omitted"] = len(paths) - 80
        exp = ws.settings.get("experiment")
        if exp:
            out["experiment"] = exp
        if self.window is not None:
            out["window"] = {"current_group": c.current_group.name if c.current_group else None,
                             "selected": [(s.name, p.path or "/") for s, p in c.selected][:10]}
        self.note("snapshot")
        return out

    @op
    def sample_info(self, sample) -> dict:
        """Header and parameters of one sample: events, cytometer, parameters with stain and scale, keywords, compensation."""
        s = self.sample(sample)
        s.ensure_loaded()
        params = []
        for name in s.parameter_names():
            entry = {"name": name}
            try:
                stain = s.param_stain(name)
                if stain:
                    entry["stain"] = stain
                sc = s.scale(name)
                entry["scale"] = sc.kind
                entry["display_range"] = [clean(sc.inverse(0.0), 4), clean(sc.inverse(1.0), 4)]
            except Exception:
                pass
            if name in s.computed:
                entry["computed"] = True
            if name in s.derived:
                entry["derived"] = True
            params.append(entry)
        info = {"id": s.id, "name": s.name, "path": str(s.path), "events": int(s.n_events), "keywords": dict(s.keywords),
                "parameters": params, "load_error": str(s.load_error) if s.load_error else None,
                "warnings": list(getattr(s, "load_warnings", []) or [])}
        if s.comp is not None:
            info["compensation"] = {"matrix": s.comp.name, "enabled": bool(s.comp_enabled), "parameters": list(s.comp.params)}
        info["populations"] = [self.pop_row(p) for p in s.root.walk(include_self=False)][:60]
        return info

    @op
    def populations(self, sample=None, group: str | None = None, population: str = "", depth: int = 6) -> dict:
        """The gating tree with counts of one sample, or the gating template of a group."""
        if group and not sample:
            g = self.group(group)
            root = self.parent_in(g.root, population, "the group template") if population else g.root
            return {"group": g.name, "members": [m.name for m in self.ws.members(g)], "tree": self.pop_tree(root, depth, with_counts=False)}
        if sample in (None, ""):
            if len(self.ws.samples) != 1:
                raise OpError("give sample= (one of: %s) or group=" % listing(s.name for s in self.ws.samples))
            sample = self.ws.samples[0].name
        s = self.sample(sample)
        pop = self.population(s, population)
        return {"sample": s.name, "events": int(s.n_events), "tree": self.pop_tree(pop, depth)}

    # ----------------------------------------------------------- file level
    def _guard_unsaved(self, discard_changes: bool):
        if self.ws.modified and not discard_changes:
            raise OpError("the current workspace has unsaved changes; save_workspace first, or repeat with discard_changes=true "
                          "if the user agrees to lose them")

    @op
    def open_workspace(self, path: str, discard_changes: bool = False) -> dict:
        """Open a FlowDesk .fdw or a FlowJo workspace (replaces the current one)."""
        p = Path(path).expanduser()
        if not p.exists():
            raise OpError("no such file: %s" % p)
        if p.suffix.lower() in (".fdwt", ".wspt"):
            raise OpError("%s is a template; open a workspace first, then use apply_template" % p.name)
        self._guard_unsaved(discard_changes)
        warnings: list[str] = []
        if p.suffix.lower() == ".wsp":
            from ..core.wsp_import import import_wsp

            ws, report = import_wsp(p, search_dirs=[p.parent])
            warnings = [str(w) for w in getattr(report, "warnings", [])]
        else:
            from ..core.serialization import load_workspace

            ws = load_workspace(p)
            warnings = [str(w) for w in getattr(ws, "load_warnings", [])]
        self.c.set_workspace(ws)
        missing = [s.name for s in ws.samples if s.load_error]
        self.note("open_workspace", path=str(p))
        return {"ok": True, "name": ws.name, "samples": len(ws.samples), "groups": [g.name for g in ws.groups],
                "missing_files": missing, "warnings": warnings[:20]}

    @op
    def new_workspace(self, discard_changes: bool = False) -> dict:
        """Start an empty workspace (replaces the current one)."""
        from ..core.model import Workspace

        self._guard_unsaved(discard_changes)
        self.c.set_workspace(Workspace())
        return {"ok": True}

    @op
    def save_workspace(self, path: str | None = None) -> dict:
        """Save as a FlowDesk workspace (.fdw); without a path, to where it was opened from."""
        from ..core.serialization import save_workspace

        target = Path(path).expanduser() if path else self.ws.path
        if target is None:
            raise OpError("this workspace has no file yet; give path=")
        target = Path(target)
        if target.suffix.lower() != ".fdw":
            target = target.with_name(target.name + ".fdw")
        target.parent.mkdir(parents=True, exist_ok=True)
        save_workspace(self.ws, target)
        self.c.mark_saved()
        self.note("save_workspace", path=str(target))
        return {"ok": True, "path": str(target)}

    @op
    def add_samples(self, paths: list[str] | str | None = None, folder: str | None = None, recursive: bool = True, group: str | None = None) -> dict:
        """Add FCS files (a list of paths) or a whole folder; optionally into a group."""
        if not paths and not folder:
            raise OpError("give paths= (FCS files) or folder=")
        added = []
        if folder:
            f = Path(folder).expanduser()
            if not f.is_dir():
                raise OpError("no such folder: %s" % f)
            added += self.c.add_folder(f, recursive=recursive)
        if paths:
            plist = [paths] if isinstance(paths, str) else list(paths)
            files = [Path(p).expanduser() for p in plist]
            bad = [str(p) for p in files if not p.exists()]
            if bad:
                raise OpError("no such file: %s" % listing(bad))
            added += self.c.add_samples(files)
        if group and added:
            self.c.add_samples_to_group(self.group(group), added)
        self.note("add_samples", n=len(added))
        return {"ok": True, "added": [{"id": s.id, "name": s.name, "events": int(s.n_events)} for s in added], "total_samples": len(self.ws.samples)}

    @op
    def remove_samples(self, samples) -> dict:
        """Remove samples from the workspace (undoable; the files stay on disk)."""
        targets = self.samples_of(samples)
        names = [s.name for s in targets]
        self.c.remove_samples(targets)
        return {"ok": True, "removed": names}

    # ----------------------------------------------------- keywords & groups
    @op
    def set_keyword(self, samples, name: str, value: str) -> dict:
        """Set a keyword (e.g. Donor, Condition) on samples; keyword columns and groups update."""
        from ..ui.experiment import set_keyword

        targets = self.samples_of(samples)
        n = set_keyword(self.c, targets, name, str(value))
        return {"ok": True, "samples": [s.name for s in targets], "changed": int(n)}

    @op
    def create_group(self, name: str, samples=None, criteria: list[dict] | None = None, color: str = "#2e7d32") -> dict:
        """A group from explicit samples, or a live keyword group: criteria = [{keyword, function, value}]
        with function one of Contains, Does not contain, Equals, Not equals, Starts with, Ends with."""
        from ..core.model import CRITERION_FUNCTIONS, Criterion

        if any(g.name == name for g in self.ws.groups):
            raise OpError("a group named %r already exists" % name)
        crit = None
        members = None
        if criteria:
            crit = []
            for k in criteria:
                fn = k.get("function", "Equals")
                if fn not in CRITERION_FUNCTIONS:
                    raise OpError("criterion function %r is not one of %s" % (fn, CRITERION_FUNCTIONS))
                crit.append(Criterion(k.get("keyword", "$FIL"), fn, str(k.get("value", "")), k.get("connector", "And")))
        elif samples is not None:
            members = self.samples_of(samples)
        else:
            raise OpError("give samples= or criteria=")
        g = self.c.create_group(name, members, crit, color)
        return {"ok": True, "group": g.name, "members": [s.name for s in self.ws.members(g)]}

    @op
    def delete_group(self, name: str) -> dict:
        g = self.group(name)
        if g.role in ("all", "compensation"):
            raise OpError("the %s group cannot be deleted" % g.name)
        self.c.delete_group(g)
        return {"ok": True, "deleted": g.name}

    # ------------------------------------------------------------------ undo
    @op
    def undo(self) -> dict:
        """Undo the last change (gates, keywords, sample order, scales, matrices, layouts, ...)."""
        if not self.c.can_undo:
            raise OpError("nothing to undo")
        label = self.c.undo_label()
        self.c.undo()
        self.note("undo", label=label)
        return {"ok": True, "undone": label, "next_undo": self.c.undo_label() or None, "can_undo": self.c.can_undo, "can_redo": self.c.can_redo}

    @op
    def redo(self) -> dict:
        if not self.c.can_redo:
            raise OpError("nothing to redo")
        self.c.redo()
        self.note("redo")
        return {"ok": True, "can_undo": self.c.can_undo, "can_redo": self.c.can_redo}
