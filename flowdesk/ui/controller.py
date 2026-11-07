"""The controller owns the workspace, the undo history, the user preferences
and the Qt signals that keep every window in sync.

All windows (workspace, graph, layout editor, table editor, compensation)
receive the same WorkspaceController instance and must change the model only
through it (or call the notify_* methods after changing the model directly).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QSettings, Signal

from ..core.compensation import SpilloverMatrix
from ..core.gates import Gate
from ..core.model import Group, Population, Sample, StatNode, Workspace
from ..core.serialization import extend_snapshot, gating_snapshot, restore_snapshot, snapshot_kind
from ..core.transforms import Scale

POPULATION_MIME = "application/x-flowdesk-population"
MAX_UNDO = 60
# begin_change() callers written before `kind` existed that change more than
# the gating state (labels as they pass them)
LEGACY_UNDO_KINDS = {"apply template": "all", "add samples": "samples", "remove samples": "samples"}

DEFAULT_PREFERENCES = {
    "polygon_gating": "exact",  # exact | bitmap (FlowJo-compatible 256-bin evaluation)
    "plot_type": "pseudocolor",  # default plot type for new graph windows
    "dots_to_draw": 50000,  # max events rendered per plot
    "dot_size": 1,  # 1..4 pixels
    "show_outliers": True,  # draw single events below the lowest contour level
    "smoothing": 1,  # 0 (none) .. 3
    "contour_level": 5.0,  # percent per contour step
    "gate_color": "#000000",
    "gate_line_width": 1.0,
    "selected_gate_color": "#d81b1b",
    "label_font_size": 11,
    "number_font_size": 9,
    "show_gate_names": True,
    "show_gate_freqs": True,
    "axis_label_style": "name_stain",  # name_stain | name | stain
    "default_transform": "logicle",  # logicle | log | linear | arcsinh
    "default_width_basis": -100.0,
    "histogram_bins": 256,
    "histogram_ymode": "count",  # count | mode (normalize to mode)
    "apply_to_group_default": False,  # new gates are copied to the current group
    "freq_decimals": 1,
    "colormap": "pseudocolor",
    "plot_background": "#ffffff",
    "recent_files": [],
}


class Preferences:
    """Dictionary-like preferences persisted through QSettings."""

    def __init__(self):
        self._settings = QSettings("FlowDesk", "FlowDesk")
        self._values = dict(DEFAULT_PREFERENCES)
        for key, default in DEFAULT_PREFERENCES.items():
            stored = self._settings.value(f"prefs/{key}")
            if stored is None:
                continue
            self._values[key] = self._coerce(stored, default)

    @staticmethod
    def _coerce(value, default):
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes")
            return bool(value)
        if isinstance(default, int):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default
        if isinstance(default, float):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        if isinstance(default, list):
            if isinstance(value, str):
                return [value] if value else []
            return list(value) if value is not None else []
        return str(value)

    def get(self, key: str, default=None):
        return self._values.get(key, DEFAULT_PREFERENCES.get(key, default))

    def __getitem__(self, key: str):
        return self.get(key)

    def set(self, key: str, value):
        self._values[key] = value
        self._settings.setValue(f"prefs/{key}", value)

    def update(self, values: dict):
        for k, v in values.items():
            self.set(k, v)

    def as_dict(self) -> dict:
        return dict(self._values)

    def add_recent(self, path: str):
        recent = [p for p in self.get("recent_files", []) if p != path]
        recent.insert(0, path)
        self.set("recent_files", recent[:10])


class WorkspaceController(QObject):
    # -- model changes
    workspaceReplaced = Signal()  # a different Workspace object is now current
    samplesChanged = Signal()  # samples added / removed / renamed
    gatingChanged = Signal(object)  # sample id (int) or None when several samples changed
    groupsChanged = Signal()  # groups added / removed / membership or template changed
    scalesChanged = Signal(object)  # sample id or None
    compensationChanged = Signal(object)  # sample id or None
    layoutsChanged = Signal()
    tablesChanged = Signal()
    settingsChanged = Signal()  # workspace.settings replaced by undo / redo
    preferencesChanged = Signal()
    # -- ui state
    currentGroupChanged = Signal(object)  # Group
    selectionChanged = Signal()  # selected rows in the workspace tree
    statusMessage = Signal(str)
    modifiedChanged = Signal(bool)

    def __init__(self, workspace: Workspace | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self.workspace: Workspace = workspace or Workspace()
        self.current_group: Group = self.workspace.groups[0]
        self.selected: list[tuple[Sample, Population]] = []  # (sample, population) rows selected in the tree
        self.selected_groups: list[Group] = []
        self.clipboard: list[Population] = []  # copied populations (clones)
        self.preferences = Preferences()
        self._undo: list[tuple[str, dict]] = []
        self._redo: list[tuple[str, dict]] = []
        self._undo_group_depth = 0
        # Callbacks installed by the main window so that other windows can open things.
        self.open_graph: Callable[[Sample, Population], object] | None = None
        self.open_group_graph: Callable[[Group, Population | None], object] | None = None
        self.open_layout_editor: Callable[[], object] | None = None
        self.open_table_editor: Callable[[], object] | None = None
        self.open_compensation: Callable[[], object] | None = None
        self.add_to_layout: Callable[[Sample, Population], object] | None = None
        self.add_to_table: Callable[[list[Population]], object] | None = None
        self.preferencesChanged.connect(self.apply_polygon_mode)
        self.apply_polygon_mode(emit=False)

    def apply_polygon_mode(self, emit: bool = True):
        """Push the polygon-evaluation preference into the gating engine."""
        from ..core.gates import POLYGON_SETTINGS

        mode = "bitmap" if self.preferences.get("polygon_gating") == "bitmap" else "exact"
        if POLYGON_SETTINGS.get("mode") != mode:
            POLYGON_SETTINGS["mode"] = mode
            for s in self.workspace.samples:
                s.root.invalidate()
            if emit:
                self.gatingChanged.emit(None)

    # ------------------------------------------------------------- workspace
    def set_workspace(self, ws: Workspace):
        self.workspace = ws
        self.current_group = ws.groups[0]
        self.selected = []
        self.selected_groups = []
        self._undo.clear()
        self._redo.clear()
        self.workspaceReplaced.emit()
        self.currentGroupChanged.emit(self.current_group)
        self.modifiedChanged.emit(ws.modified)

    def mark_modified(self):
        if not self.workspace.modified:
            self.workspace.modified = True
            self.modifiedChanged.emit(True)

    def mark_saved(self):
        self.workspace.modified = False
        self.modifiedChanged.emit(False)

    # ------------------------------------------------------------------- undo
    def begin_change(self, label: str = "change", kind=None):
        """Call before mutating the workspace; undo brings back what `kind` covers.

        kind "gating" (the default) covers every sample's gates, statistics,
        graph settings, scales, compensation, derived and computed parameters
        plus the groups. Changes to more than that pass "all" or the extra
        sections they touch: "samples" (sample list and order, names,
        keywords), "matrices", "settings", "layouts", "tables" (one name or a
        tuple). Inside `undo_group` a call adds no step of its own; it only
        widens the open step when it needs more sections. The widened
        sections are read from the present state, so call it before changing
        them; code that has already changed a section (an editor announcing
        an in-place edit) writes its own earlier state into
        undo_top_snapshot() instead, see SectionUndoRecorder."""
        if kind is None:
            kind = LEGACY_UNDO_KINDS.get(label, "gating")
        if self._undo_group_depth and self._undo:
            extend_snapshot(self.workspace, self._undo[-1][1], kind)
            return
        self._push_undo(label, gating_snapshot(self.workspace, kind))

    def _push_undo(self, label: str, snap: dict):
        self._undo.append((label, snap))
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def push_undo(self, label: str, snapshot: dict):
        """Record an undo step from a snapshot taken earlier (editors that
        mutate the model during a drag and announce the change afterwards)."""
        self._push_undo(label, snapshot)

    def undo_top_snapshot(self) -> dict | None:
        """The snapshot of the step an undo would restore next, or None."""
        return self._undo[-1][1] if self._undo else None

    @contextmanager
    def undo_group(self, label: str, kind=None):
        """One undo step for everything done in the block (begin_change calls
        made inside, e.g. by set_scale, add none)."""
        self.begin_change(label, kind)
        self._undo_group_depth += 1
        try:
            yield
        finally:
            self._undo_group_depth -= 1

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_label(self) -> str:
        return self._undo[-1][0] if self._undo else ""

    def undo(self):
        if not self._undo:
            return
        label, snap = self._undo.pop()
        # the redo snapshot has the same sections as the one restored
        current = gating_snapshot(self.workspace, snapshot_kind(snap))
        changes = restore_snapshot(self.workspace, snap, current)
        self._redo.append((label, current))
        self._after_restore(changes)
        self.statusMessage.emit(f"Undid {label}")

    def redo(self):
        if not self._redo:
            return
        label, snap = self._redo.pop()
        current = gating_snapshot(self.workspace, snapshot_kind(snap))
        changes = restore_snapshot(self.workspace, snap, current)
        self._undo.append((label, current))
        self._after_restore(changes)
        self.statusMessage.emit(f"Redid {label}")

    def _after_restore(self, changes: set[str]):
        """Re-point the ui state at the restored model and emit the signals
        for what `changes` (from restore_snapshot) says changed. The tree
        selection and the current group are reset only by a restore that can
        invalidate them (samples, gating or groups changed); undoing a layout,
        table, settings, scale or compensation edit keeps them."""
        groups = self.workspace.groups
        cur = self.current_group
        tree_changed = bool(changes & {"samples", "gating", "groups"})
        group_moved = not any(g is cur for g in groups)
        if group_moved:
            self.current_group = next((g for g in groups if g.id == cur.id), None) or next((g for g in groups if g.name == cur.name), groups[0])
        if tree_changed:
            self.selected = []
        self.selected_groups = [g for g in self.selected_groups if any(g is x for x in groups)]
        self.mark_modified()
        if "samples" in changes:  # first: windows of samples that left close
            self.samplesChanged.emit()
        if changes & {"samples", "groups"}:  # keywords / sample list change memberships
            self.groupsChanged.emit()
        if changes & {"samples", "gating", "scales", "compensation", "groups"}:
            self.gatingChanged.emit(None)
        if "scales" in changes:
            self.scalesChanged.emit(None)
        if changes & {"compensation", "matrices"}:
            self.compensationChanged.emit(None)
        if "layouts" in changes:
            self.layoutsChanged.emit()
        if "tables" in changes:
            self.tablesChanged.emit()
        if "settings" in changes:
            self.settingsChanged.emit()
        if tree_changed or group_moved:
            self.currentGroupChanged.emit(self.current_group)
        if tree_changed:
            self.selectionChanged.emit()

    # --------------------------------------------------------------- samples
    def add_samples(self, paths: list[str | Path]) -> list[Sample]:
        if self._undo_group_depth:
            self.begin_change("add samples", "samples")
            snap = None
        else:
            snap = gating_snapshot(self.workspace, "samples")
        added = []
        for p in paths:
            p = Path(p)
            if p.suffix.lower() != ".fcs":
                continue
            before = len(self.workspace.samples)
            s = self.workspace.add_sample(p)
            if len(self.workspace.samples) > before:
                added.append(s)
        if added:
            if snap is not None:
                self._push_undo("add samples", snap)
            self.mark_modified()
            self.samplesChanged.emit()
            self.groupsChanged.emit()
            self.statusMessage.emit(f"Added {len(added)} sample(s)")
        return added

    def add_folder(self, folder: str | Path, recursive: bool = True) -> list[Sample]:
        folder = Path(folder)
        pattern = "**/*.fcs" if recursive else "*.fcs"
        paths = sorted(p for p in folder.glob(pattern) if p.is_file())
        return self.add_samples(paths)

    def remove_samples(self, samples: list[Sample]):
        if not samples:
            return
        self.begin_change("remove samples", "samples")
        for s in samples:
            self.workspace.remove_sample(s)
        self.selected = [(s, p) for s, p in self.selected if s in self.workspace.samples]
        self.mark_modified()
        self.samplesChanged.emit()
        self.groupsChanged.emit()
        self.selectionChanged.emit()

    def rename_sample(self, sample: Sample, name: str):
        if name and name != sample.name:
            self.begin_change("rename sample", "samples")
            sample.name = name
            sample.root.name = name
            self.mark_modified()
            self.samplesChanged.emit()

    # ------------------------------------------------------------------ groups
    def set_current_group(self, group: Group):
        if group is not self.current_group:
            self.current_group = group
            self.currentGroupChanged.emit(group)

    def set_selection(self, rows: list[tuple[Sample, Population]], groups: list[Group] | None = None):
        self.selected = list(rows)
        if groups is not None:
            self.selected_groups = list(groups)
        self.selectionChanged.emit()

    def selected_samples(self) -> list[Sample]:
        seen: list[Sample] = []
        for s, _ in self.selected:
            if s not in seen:
                seen.append(s)
        return seen

    def selected_populations(self) -> list[Population]:
        return [p for _, p in self.selected]

    def create_group(self, name: str, samples: list[Sample] | None = None, criteria=None, color: str = "#000000") -> Group:
        self.begin_change("create group")
        g = self.workspace.add_group(name, samples, criteria, color)
        for s in self.workspace.members(g):
            self.workspace.apply_template(g, s)
        self.mark_modified()
        self.groupsChanged.emit()
        return g

    def delete_group(self, group: Group):
        if group.role == "all":
            return
        self.begin_change("delete group")
        self.workspace.remove_group(group)
        if self.current_group is group:
            self.current_group = self.workspace.groups[0]
            self.currentGroupChanged.emit(self.current_group)
        self.mark_modified()
        self.groupsChanged.emit()

    def rename_group(self, group: Group, name: str):
        if name and name != group.name and self.workspace.group_by_name(name) is None:
            group.name = name
            group.root.name = name
            self.mark_modified()
            self.groupsChanged.emit()

    def add_samples_to_group(self, group: Group, samples: list[Sample]):
        self.begin_change("add to group")
        for s in samples:
            group.sample_ids.add(s.id)
            self.workspace.apply_template(group, s)
        self.mark_modified()
        self.groupsChanged.emit()
        self.gatingChanged.emit(None)

    def remove_samples_from_group(self, group: Group, samples: list[Sample]):
        if group.role == "all":
            return
        self.begin_change("remove from group")
        for s in samples:
            group.sample_ids.discard(s.id)
        self.mark_modified()
        self.groupsChanged.emit()

    # ------------------------------------------------------------------ gating
    def add_population(self, parent: Population, population: Population, group: Group | None = None) -> Population:
        """Add a population below `parent`. With a group, the population is
        applied to every member sample (and the group template) instead."""
        self.begin_change("create gate")
        sample = parent.sample
        if group is not None:
            self.workspace.apply_population_to_group(group, parent.path, population)
            result = parent.find("/" + population.name) if sample is not None else None
            self.mark_modified()
            self.gatingChanged.emit(None)
            self.groupsChanged.emit()
            return result or population
        parent.add_child(population)
        population.invalidate()
        self.mark_modified()
        self.gatingChanged.emit(sample.id if sample is not None else None)
        return population

    def gate_changed(self, population: Population, group: Group | None = None):
        """Call after editing a gate's geometry in place."""
        population.invalidate()
        sample = population.sample
        if group is not None:
            self.workspace.apply_population_to_group(group, population.parent.path if population.parent else "", population)
            self.mark_modified()
            self.gatingChanged.emit(None)
            return
        self.mark_modified()
        self.gatingChanged.emit(sample.id if sample is not None else None)

    def delete_populations(self, populations: list[Population], group: Group | None = None):
        if not populations:
            return
        self.begin_change("delete gate")
        affected: set[int | None] = set()
        for pop in populations:
            if pop.parent is None:
                continue
            if group is not None:
                self.workspace.remove_population_from_group(group, pop.path)
                affected.add(None)
            else:
                pop.parent.remove_child(pop)
                affected.add(pop.sample.id if pop.sample else None)
        self.selected = [(s, p) for s, p in self.selected if p not in populations]
        self.mark_modified()
        if None in affected or len(affected) > 1:
            self.gatingChanged.emit(None)
            self.groupsChanged.emit()
        else:
            self.gatingChanged.emit(next(iter(affected)))
        self.selectionChanged.emit()

    def rename_population(self, population: Population, name: str, group: Group | None = None):
        name = name.strip()
        if not name or name == population.name or population.parent is None:
            return
        if population.parent.child_by_name(name) is not None:
            return
        self.begin_change("rename gate")
        if group is not None:
            self.workspace.rename_population_in_group(group, population.path, name)
            self.mark_modified()
            self.gatingChanged.emit(None)
            return
        population.name = name
        self.mark_modified()
        self.gatingChanged.emit(population.sample.id if population.sample else None)

    def copy_populations_to_group(self, populations: list[Population], group: Group) -> int:
        self.begin_change("copy to group")
        n = 0
        for pop in populations:
            if pop.parent is None:
                continue
            n += self.workspace.apply_population_to_group(group, pop.parent.path, pop)
        self.mark_modified()
        self.gatingChanged.emit(None)
        self.groupsChanged.emit()
        return n

    def paste_populations(self, populations: list[Population], target_parent: Population, group: Group | None = None):
        self.begin_change("paste gates")
        for pop in populations:
            clone = pop.clone(deep=True)
            base = clone.name
            k = 2
            while target_parent.child_by_name(clone.name) is not None:
                clone.name = f"{base} {k}"
                k += 1
            if group is not None:
                self.workspace.apply_population_to_group(group, target_parent.path, clone)
            else:
                target_parent.add_child(clone)
        self.mark_modified()
        self.gatingChanged.emit(None if group is not None else (target_parent.sample.id if target_parent.sample else None))

    def create_boolean(self, parent: Population, op: str, refs: list[str], name: str, group: Group | None = None) -> Population:
        from ..core.gates import BooleanGate

        pop = Population(name, BooleanGate(op, refs))
        return self.add_population(parent, pop, group)

    def add_statistic(self, populations: list[Population], statistic: str, param: str | None, percentile: float | None, group: Group | None = None):
        self.begin_change("add statistic")
        for pop in populations:
            node = StatNode(statistic, param, percentile)
            if group is not None and pop.parent is not None:
                template = group.root.find(pop.path)
                if template is None:
                    template = self.workspace._ensure_path(group.root, pop.path)
                template.stats.append(StatNode.from_dict(node.to_dict()))
                for s in self.workspace.members(group):
                    target = s.root.find(pop.path)
                    if target is not None and not any(x.statistic == statistic and x.param == param and x.percentile == percentile for x in target.stats):
                        target.stats.append(StatNode.from_dict(node.to_dict()))
            else:
                if not any(x.statistic == statistic and x.param == param and x.percentile == percentile for x in pop.stats):
                    pop.stats.append(node)
        self.mark_modified()
        self.gatingChanged.emit(None)

    def remove_statistic(self, population: Population, stat: StatNode):
        if stat in population.stats:
            self.begin_change("remove statistic")
            population.stats.remove(stat)
            self.mark_modified()
            self.gatingChanged.emit(population.sample.id if population.sample else None)

    # ------------------------------------------------------------------ scales
    def set_scale(self, samples: list[Sample], param: str, scale: Scale):
        self.begin_change("change axis scale")
        for s in samples:
            s.set_scale(param, scale.copy())
        self.mark_modified()
        self.scalesChanged.emit(samples[0].id if len(samples) == 1 else None)
        self.gatingChanged.emit(samples[0].id if len(samples) == 1 else None)

    # ------------------------------------------------------------ compensation
    def set_compensation(self, samples: list[Sample], matrix: SpilloverMatrix | None, enabled: bool = True):
        self.begin_change("apply compensation")
        for s in samples:
            s.set_compensation(matrix.copy() if matrix is not None else None, enabled)
        self.mark_modified()
        self.compensationChanged.emit(samples[0].id if len(samples) == 1 else None)
        self.gatingChanged.emit(samples[0].id if len(samples) == 1 else None)

    def store_matrix(self, matrix: SpilloverMatrix):
        self.begin_change("edit matrix", "matrices")
        self.workspace.matrices[matrix.name] = matrix
        self.mark_modified()
        self.compensationChanged.emit(None)

    def remove_matrix(self, name: str):
        if name in self.workspace.matrices:
            self.begin_change("delete matrix", "matrices")
            del self.workspace.matrices[name]
            self.mark_modified()
            self.compensationChanged.emit(None)

    # ----------------------------------------------------------- layouts etc.
    def notify_layouts(self):
        self.mark_modified()
        self.layoutsChanged.emit()

    def notify_tables(self):
        self.mark_modified()
        self.tablesChanged.emit()

    def notify_gating(self, sample: Sample | None = None):
        self.mark_modified()
        self.gatingChanged.emit(sample.id if sample is not None else None)

    def set_preference(self, key: str, value):
        self.preferences.set(key, value)
        self.preferencesChanged.emit()
