"""Gating operations: create, edit, rename and delete populations on one sample or on a whole group."""
from __future__ import annotations

from .ops_base import OpError, listing, op

SHAPES = ("rectangle", "range", "polygon", "ellipse", "boolean")


class GatingOps:
    # ---------------------------------------------------------------- targets
    def _target(self, sample, group):
        """('group', Group, first member) or ('sample', None, Sample)."""
        if group and sample:
            raise OpError("give either sample= or group=, not both")
        if group:
            g = self.group(group)
            members = self.ws.members(g)
            if not members:
                raise OpError("the group %r has no samples" % g.name)
            return "group", g, members[0]
        if sample in (None, ""):
            if len(self.ws.samples) == 1:
                return "sample", None, self.ws.samples[0]
            raise OpError("say where the gate goes: sample= (one of: %s) or group=" % listing(s.name for s in self.ws.samples))
        return "sample", None, self.sample(sample)

    def _locate(self, kind, g, members, ref):
        """A gated population from a path or a name. On a group: from the template if it is there, else from the first
        member that has it (a group's own template stays empty until gates are applied through it)."""
        if kind == "sample":
            pop = self.population(members[0], ref)
        else:
            path = str(ref) if str(ref).startswith("/") else None
            if path is None:
                for m in members:
                    try:
                        path = self.population(m, ref).path
                        break
                    except OpError:
                        continue
            pop = None
            if path:
                for root in [g.root] + [m.root for m in members]:
                    node = root.find(path)
                    if node is not None:
                        pop = node
                        break
            if pop is None:
                raise OpError("no population %r in the group %r (its members have: %s)" % (
                    ref, g.name, listing(self.ws.population_paths(g))))
        if pop.parent is None:
            raise OpError("%r is the ungated root, not a gated population" % ref)
        return pop

    def _dims(self, ref_sample, members, x, y, need_y: bool):
        if not x:
            raise OpError("x (a parameter name or stain, e.g. 'CD3' or 'Comp-FITC-A') is required")
        if need_y and not y:
            raise OpError("y is required for this shape")
        dims = [self.param(ref_sample, x)] + ([self.param(ref_sample, y)] if y else [])
        for m in members:
            missing = [d for d in dims if d not in m.parameter_names()]
            if missing:
                raise OpError("sample %s has no parameter %s; use sample= or a group whose members share the panel" % (m.name, listing(missing)))
        return dims

    @staticmethod
    def _range(value, label):
        if value is None:
            return None, None
        try:
            lo, hi = value
        except (TypeError, ValueError):
            raise OpError("%s must be [low, high] (use null for an open end)" % label)
        return (None if lo is None else float(lo)), (None if hi is None else float(hi))

    def _make_gate(self, shape, dims, x_range, y_range, vertices, center, a_end, b_end, operation, refs):
        from ..core.gates import BooleanGate, EllipseGate, PolygonGate, RectangleGate

        if shape == "rectangle":
            xl, xh = self._range(x_range, "x_range")
            yl, yh = self._range(y_range, "y_range")
            if len(dims) != 2:
                raise OpError("a rectangle needs x and y")
            if xl is None and xh is None and yl is None and yh is None:
                raise OpError("give x_range=[lo, hi] and/or y_range=[lo, hi] in the raw units of the axes")
            return RectangleGate(dims, [xl, yl], [xh, yh])
        if shape == "range":
            if len(dims) != 1:
                raise OpError("a range gate is one-dimensional: give x only")
            xl, xh = self._range(x_range, "x_range")
            if xl is None and xh is None:
                raise OpError("give x_range=[lo, hi] (null for an open end), e.g. [4000, null] for 'positive'")
            return RectangleGate(dims, [xl], [xh])
        if shape == "polygon":
            if len(dims) != 2:
                raise OpError("a polygon needs x and y")
            if not vertices or len(vertices) < 3:
                raise OpError("a polygon needs vertices=[[x, y], ...] with at least 3 points (raw axis units)")
            return PolygonGate(dims, [(float(a), float(b)) for a, b in vertices])
        if shape == "ellipse":
            if len(dims) != 2:
                raise OpError("an ellipse needs x and y")
            if not (center and a_end and b_end):
                raise OpError("an ellipse needs center=[x, y], a_end=[x, y] and b_end=[x, y] (the ends of its two semi-axes)")
            return EllipseGate(dims, tuple(center), tuple(a_end), tuple(b_end))
        if shape == "boolean":
            if operation not in ("and", "or", "not"):
                raise OpError("operation must be 'and', 'or' or 'not'")
            if not refs:
                raise OpError("refs must list the population paths to combine, e.g. ['/Lymphocytes/CD3+', '/Lymphocytes/CD19+']")
            return BooleanGate(operation, list(refs))
        raise OpError("shape must be one of %s" % ", ".join(SHAPES))

    def _rows_after(self, kind, g, container, parent_path, name):
        path = (parent_path.rstrip("/") + "/" + name) if parent_path not in ("", "/") else "/" + name
        if kind == "group":
            rows = []
            for m in self.ws.members(g):
                pop = m.root.find(path)
                if pop is not None:
                    rows.append(self.pop_row(pop))
            return rows
        pop = container.root.find(path)
        return [self.pop_row(pop)] if pop is not None else []

    # -------------------------------------------------------------- operations
    @op
    def create_gate(self, name: str, shape: str, x: str | None = None, y: str | None = None, sample=None, group: str | None = None,
                    parent: str = "", x_range=None, y_range=None, vertices=None, center=None, a_end=None, b_end=None,
                    operation: str | None = None, refs=None, negated: bool = False) -> dict:
        """Create a population below `parent` ("" = ungated). Values are raw axis units (compensated intensities for
        Comp- parameters). On a group, the gate goes onto the group template and every member sample.

        shape: rectangle (x_range and/or y_range), range (one parameter, x_range), polygon (vertices), ellipse
        (center, a_end, b_end) or boolean (operation and refs)."""
        from ..core.model import Population

        if shape not in SHAPES:
            raise OpError("shape must be one of %s" % ", ".join(SHAPES))
        name = str(name).strip()
        if not name:
            raise OpError("a population needs a name")
        kind, g, first = self._target(sample, group)
        members = self.ws.members(g) if kind == "group" else [first]
        if shape == "boolean":
            dims = []
        else:
            dims = self._dims(first, members, x, y, need_y=shape in ("rectangle", "polygon", "ellipse"))
            if shape == "rectangle" and not y:
                raise OpError("a rectangle needs y; for one parameter use shape='range'")
        gate = self._make_gate(shape, dims, x_range, y_range, vertices, center, a_end, b_end, operation, refs)
        gate.negated = bool(negated)
        if kind == "group":
            holders = [m for m in members if parent in ("", "/") or m.root.find(parent) is not None]
            if not holders:
                raise OpError("the parent population %r does not exist in any member of the group %r (its members have: %s)" % (
                    parent, g.name, listing(self.ws.population_paths(g))))
            parent_pop = holders[0].root if parent in ("", "/") else holders[0].root.find(parent)
            clash = [m.name for m in holders if (m.root if parent in ("", "/") else m.root.find(parent)).child_by_name(name) is not None]
            if clash:
                raise OpError("%r already has a child named %r in %s; use edit_gate to change it or pick another name" % (
                    parent or "/", name, listing(clash)))
            skipped = [m.name for m in members if m not in holders]
        else:
            parent_pop = self.parent_in(first.root, parent, first.name)
            if parent_pop.child_by_name(name) is not None:
                raise OpError("%r already has a child named %r; use edit_gate to change it or pick another name" % (parent_pop.path or "/", name))
            skipped = []
        self.c.add_population(parent_pop, Population(name, gate), g if kind == "group" else None)
        self.note("create_gate", name=name, shape=shape, target=g.name if kind == "group" else first.name)
        rows = self._rows_after(kind, g, first, parent, name)
        out = {"ok": True, "population": (parent.rstrip("/") + "/" + name) if parent not in ("", "/") else "/" + name,
               "applied_to": [r["sample"] for r in rows], "counts": rows}
        if skipped:
            out["skipped_without_parent"] = skipped
        return out

    @op
    def edit_gate(self, population: str, sample=None, group: str | None = None, x_range=None, y_range=None, vertices=None,
                  center=None, a_end=None, b_end=None, refs=None, operation: str | None = None, negated: bool | None = None) -> dict:
        """Change the geometry of an existing gate in place (same shape and parameters). Children are kept and
        their counts update; on a group the change reaches every member."""
        kind, g, first = self._target(sample, group)
        members = self.ws.members(g) if kind == "group" else [first]
        pop = self._locate(kind, g, members, population)
        if pop.gate is None:
            raise OpError("%r has no gate to edit" % population)
        old = pop.gate
        shape = {"rectangle": "range" if len(old.dims) == 1 else "rectangle"}.get(old.kind, old.kind)
        new = self._make_gate(
            shape, list(old.dims),
            x_range if x_range is not None else ([old.lo[0], old.hi[0]] if old.kind == "rectangle" else None),
            y_range if y_range is not None else ([old.lo[1], old.hi[1]] if old.kind == "rectangle" and len(old.dims) == 2 else None),
            vertices if vertices is not None else getattr(old, "vertices", None),
            center if center is not None else getattr(old, "center", None),
            a_end if a_end is not None else getattr(old, "a_end", None),
            b_end if b_end is not None else getattr(old, "b_end", None),
            operation or getattr(old, "op", None), refs if refs is not None else getattr(old, "refs", None))
        new.negated = old.negated if negated is None else bool(negated)
        for attr in ("color", "line_width", "tint", "locked", "label_pos"):
            setattr(new, attr, getattr(old, attr, None))
        self.c.begin_change("edit gate")
        pop.gate = new
        self.c.gate_changed(pop, g if kind == "group" else None)
        self.note("edit_gate", population=pop.path)
        rows = self._rows_after(kind, g, first, pop.parent.path, pop.name)
        return {"ok": True, "population": pop.path, "counts": rows}

    @op
    def rename_population(self, population: str, new_name: str, sample=None, group: str | None = None) -> dict:
        kind, g, first = self._target(sample, group)
        pop = self._locate(kind, g, self.ws.members(g) if kind == "group" else [first], population)
        if pop.parent.child_by_name(new_name.strip()) is not None:
            raise OpError("%r already has a child named %r" % (pop.parent.path or "/", new_name))
        self.c.rename_population(pop, new_name, g if kind == "group" else None)
        return {"ok": True, "renamed": new_name.strip(), "path": pop.path}

    @op
    def delete_population(self, population: str, sample=None, group: str | None = None) -> dict:
        """Delete a gated population and everything below it (undoable)."""
        kind, g, first = self._target(sample, group)
        pop = self._locate(kind, g, self.ws.members(g) if kind == "group" else [first], population)
        removed = [p.path for p in pop.walk()]
        self.c.delete_populations([pop], g if kind == "group" else None)
        self.note("delete_population", path=removed[0])
        return {"ok": True, "deleted": removed}
