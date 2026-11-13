"""Undo / redo covers every kind of workspace state, stays cheap and emits the
signals for what a restore changed."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flowdesk.core.compensation import SpilloverMatrix  # noqa: E402
from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.layouts import Layout, LayoutItem, TableColumn, TableDef  # noqa: E402
from flowdesk.core.model import Criterion, Population  # noqa: E402
from flowdesk.core.serialization import gating_snapshot, restore_snapshot  # noqa: E402
from flowdesk.core.transforms import LinearScale  # noqa: E402

NAMES = ["Well_B2.fcs", "Well_A10.fcs", "Well_A2.fcs", "Well_A1.fcs"]
SIGNALS = ("samplesChanged", "gatingChanged", "groupsChanged", "scalesChanged", "compensationChanged", "layoutsChanged", "tablesChanged", "settingsChanged")


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def make_files(folder, n_events=(300, 400, 500, 600), names=NAMES, cytometer="Aurora"):
    rng = np.random.default_rng(7)
    paths = []
    for k, name in enumerate(names):
        n = n_events[k] if isinstance(n_events, (list, tuple)) else n_events
        events = np.column_stack([
            rng.normal(50000, 8000, n),
            rng.normal(30000, 6000, n),
            rng.lognormal(7, 1, n),
            rng.lognormal(6, 1, n),
        ]).astype(np.float32)
        channels = [Channel(1, "FSC-A"), Channel(2, "SSC-A"), Channel(3, "FITC-A", "CD3"), Channel(4, "PE-A", "CD4")]
        write_fcs(folder / name, events, channels, extra_keywords={"$CYT": cytometer, "Well": name.split("_")[1][:-4]})
        paths.append(folder / name)
    return paths


@pytest.fixture
def controller(app, tmp_path):
    from flowdesk.ui.controller import WorkspaceController

    c = WorkspaceController()
    c.add_samples(make_files(tmp_path))
    c._undo.clear()
    yield c
    c.workspace.modified = False


class Recorder:
    def __init__(self, controller):
        self.seen: list[str] = []
        for name in SIGNALS:
            getattr(controller, name).connect(lambda *_a, _n=name: self.seen.append(_n))

    def take(self) -> set[str]:
        out = set(self.seen)
        self.seen.clear()
        return out


def gate(c, sample, name="Cells", parent=None, lo=20000.0):
    parent = parent or sample.root
    return c.add_population(parent, Population(name, RectangleGate(["FSC-A"], [lo], [None])))


# ------------------------------------------------------------------ computed
def test_computed_arrays_are_replaced_never_mutated(controller):
    """The snapshot keeps computed arrays by reference, which is only safe
    because every writer replaces the array."""
    from flowdesk.core import analysis as A

    s = controller.workspace.samples[0]
    index = [(s.id, np.arange(s.n_events))]
    A.write_back(controller.workspace, index, {"tSNE_1": np.arange(s.n_events, dtype=float)})
    first = s.computed["tSNE_1"]
    saved = first.copy()
    snap = gating_snapshot(controller.workspace)
    assert snap["samples"][s.id]["computed"]["tSNE_1"] is first
    A.write_back(controller.workspace, index, {"tSNE_1": -np.arange(s.n_events, dtype=float)})
    assert s.computed["tSNE_1"] is not first
    np.testing.assert_array_equal(first, saved)  # the old array was not written into
    s.add_computed("tSNE_1", np.zeros(s.n_events))
    s.remove_computed("tSNE_1")
    np.testing.assert_array_equal(first, saved)
    restore_snapshot(controller.workspace, snap)
    assert s.computed["tSNE_1"] is first


def test_undo_clustering_removes_populations_and_parameter(controller):
    from flowdesk.ui.tools_analysis import apply_clustering, compute_clustering

    c = controller
    samples = c.workspace.samples[:2]
    rec = Recorder(c)
    result = compute_clustering([s.root for s in samples], ["FSC-A", "SSC-A"], "kmeans", options={"k": 3})
    name, created = apply_clustering(c, result)
    assert created and all(name in s.computed for s in samples)
    arrays = [s.computed[name] for s in samples]
    assert all(s.root.child_by_name(created[0]) is not None for s in samples)
    rec.take()
    c.undo()
    assert {"samplesChanged", "gatingChanged"} <= rec.take()
    for s in samples:
        assert name not in s.computed and name not in s.computed_info
        assert s.root.child_by_name(created[0]) is None
        assert name not in s.parameter_names()
    c.redo()
    for s, arr in zip(samples, arrays):
        assert s.computed[name] is arr  # the very same array comes back
        pop = s.root.child_by_name(created[0])
        assert pop is not None and 0 < pop.count < s.n_events


def test_undo_derived_parameter_with_legacy_begin_change(controller):
    c = controller
    s = c.workspace.samples[0]
    c.begin_change("derive parameter")  # as the main window calls it
    s.add_derived("ratio", {"op": "ratio", "a": "FITC-A", "b": "PE-A"})
    rec = Recorder(c)
    c.undo()
    assert "ratio" not in s.derived and "ratio" not in s.parameter_names()
    assert "samplesChanged" in rec.take()
    c.redo()
    assert s.derived["ratio"]["op"] == "ratio"


# ------------------------------------------------------- keywords / samples
def test_sort_samples_undo_restores_order(controller):
    from flowdesk.ui.experiment import sort_samples_by_keyword

    c = controller
    before = list(c.workspace.samples)
    sort_samples_by_keyword(c, "Well")
    assert [s.keywords["Well"] for s in c.workspace.samples] == ["A1", "A2", "A10", "B2"]
    sorted_order = list(c.workspace.samples)
    rec = Recorder(c)
    c.undo()
    assert c.workspace.samples == before
    assert all(a is b for a, b in zip(c.workspace.samples, before))  # Sample identity
    assert "samplesChanged" in rec.take()
    c.redo()
    assert all(a is b for a, b in zip(c.workspace.samples, sorted_order))
    # sorting into the order the samples already have adds no undo step
    n = len(c._undo)
    sort_samples_by_keyword(c, "Well")
    assert len(c._undo) == n


def test_add_keyword_undo_restores_keywords_and_group_membership(controller):
    from flowdesk.ui.experiment import set_keyword

    c = controller
    ws = c.workspace
    g = c.create_group("Stimulated", criteria=[Criterion("Condition", "Equals", "stim")])
    c.add_population(g.root, Population("Lymph", RectangleGate(["SSC-A"], [None], [40000.0])), g)
    assert ws.members(g) == []
    s0, s1 = ws.samples[:2]
    set_keyword(c, [s0, s1], "Condition", "stim")
    assert ws.members(g) == [s0, s1]
    assert s0.root.child_by_name("Lymph") is not None  # template applied to new members
    rec = Recorder(c)
    c.undo()
    assert "Condition" not in s0.keywords and "Condition" not in s1.keywords
    assert ws.members(g) == []
    assert s0.root.child_by_name("Lymph") is None
    assert {"samplesChanged", "groupsChanged", "gatingChanged"} <= rec.take()
    c.redo()
    assert s0.keywords["Condition"] == "stim" and ws.members(g) == [s0, s1]


def test_rename_sample_undo(controller):
    c = controller
    s = c.workspace.samples[0]
    old = s.name
    c.rename_sample(s, "Renamed")
    assert s.name == s.root.name == "Renamed"
    rec = Recorder(c)
    c.undo()
    assert s.name == s.root.name == old
    assert "samplesChanged" in rec.take()


def test_annotate_experiment_undo_restores_settings_and_stains(controller):
    from flowdesk.ui.experiment import apply_annotation

    c = controller
    ws = c.workspace
    ws.settings["columns"] = ["Well"]
    info = {"name": "Screen", "panel": [{"parameter": "FITC-A", "stain": "CD8", "fluorochrome": "FITC"}]}
    n = len(c._undo)
    assert apply_annotation(c, info, {"FITC-A": "CD3"}) == 4
    assert len(c._undo) == n + 1  # settings and stains are one step
    assert all(s.base_channel("FITC-A").stain == "CD8" for s in ws.samples)
    ws.settings["columns"] = ["Well", "$CYT"]  # view state changed afterwards
    rec = Recorder(c)
    c.undo()
    assert "experiment" not in ws.settings
    assert ws.settings["columns"] == ["Well", "$CYT"]  # view state is not undone
    for s in ws.samples:
        assert s.keywords["$P3S"] == "CD3" and s.base_channel("FITC-A").stain == "CD3"
    assert {"samplesChanged", "settingsChanged"} <= rec.take()
    c.redo()
    assert ws.settings["experiment"]["name"] == "Screen"
    assert ws.samples[0].base_channel("FITC-A").stain == "CD8"


def test_add_and_remove_samples_undo(controller, tmp_path):
    c = controller
    ws = c.workspace
    victim = ws.samples[1]
    pop = gate(c, victim)
    c.remove_samples([victim])
    assert victim not in ws.samples
    rec = Recorder(c)
    c.undo()
    assert ws.samples[1] is victim and victim.root.child_by_name("Cells") is pop
    assert "samplesChanged" in rec.take()
    (tmp_path / "more").mkdir()
    new = c.add_samples(make_files(tmp_path / "more", names=["Extra_C3.fcs"], n_events=[250]))
    assert new and new[0] in ws.samples and c.undo_label() == "add samples"
    c.undo()
    assert new[0] not in ws.samples and len(ws.samples) == 4
    c.redo()
    assert ws.samples[-1] is new[0]


# ------------------------------------------------------------------- scales
def test_cytometer_settings_undo_restores_scales_and_settings(controller):
    from flowdesk.ui.cytometer import apply_cytometer_settings

    c = controller
    ws = c.workspace
    before = [s.scale("FITC-A").to_dict() for s in ws.samples]
    n = len(c._undo)
    apply_cytometer_settings(c, "Aurora", {"FITC-A": {"kind": "linear", "min": 0.0}})
    assert len(c._undo) == n + 1
    assert all(s.scale("FITC-A").kind == "linear" for s in ws.samples)
    rec = Recorder(c)
    c.undo()
    assert [s.scale("FITC-A").to_dict() for s in ws.samples] == before
    assert "Aurora" not in ws.settings.get("cytometers", {})
    assert {"scalesChanged", "gatingChanged", "settingsChanged"} <= rec.take()
    c.redo()
    assert all(s.scale("FITC-A").kind == "linear" for s in ws.samples)
    assert ws.settings["cytometers"]["Aurora"]["FITC-A"]["kind"] == "linear"


def test_set_scale_undo(controller):
    c = controller
    s = c.workspace.samples[0]
    old = s.scale("PE-A").to_dict()
    c.set_scale([s], "PE-A", LinearScale(0, 1000))
    rec = Recorder(c)
    c.undo()
    assert s.scale("PE-A").to_dict() == old
    assert "scalesChanged" in rec.take()


# ------------------------------------------------------------- compensation
def test_compensation_and_matrices_undo(controller):
    c = controller
    ws = c.workspace
    s = ws.samples[0]
    m = SpilloverMatrix("panel", ["FITC-A", "PE-A"], [[1, 0.2], [0, 1]])
    c.store_matrix(m)
    c.set_compensation([s], m)
    assert s.comp == m and "Comp-PE-A" in s.parameter_names()
    rec = Recorder(c)
    c.undo()  # the sample's compensation
    assert s.comp is None and not s.comp_cleared
    assert "compensationChanged" in rec.take()
    assert "panel" in ws.matrices
    c.undo()  # the stored matrix
    assert "panel" not in ws.matrices
    assert "compensationChanged" in rec.take()
    c.redo()
    c.redo()
    assert ws.matrices["panel"] == m and s.comp == m
    c.set_compensation([s], None)
    assert s.comp_cleared
    c.undo()
    assert s.comp == m and not s.comp_cleared and s.comp_enabled
    c.remove_matrix("panel")
    c.undo()
    assert ws.matrices["panel"] == m


# ------------------------------------------------------------------- groups
def test_groups_undo_keeps_identity_and_restores_membership(controller):
    c = controller
    ws = c.workspace
    s0, s1, s2, _ = ws.samples
    g = c.create_group("Donors", [s0], criteria=[Criterion("$FIL", "Contains", "A10")], color="#ff0000")
    c.add_samples_to_group(g, [s2])
    assert g.sample_ids == {s0.id, s2.id}
    c.undo()
    assert g in ws.groups and g.sample_ids == {s0.id}  # same Group object
    assert [x.name for x in ws.members(g)] == [s0.name, s1.name]  # criteria still live
    c.delete_group(g)
    assert g not in ws.groups
    c.set_current_group(ws.groups[0])
    rec = Recorder(c)
    c.undo()
    assert any(x is g for x in ws.groups) and g.criteria[0].value == "A10" and g.color == "#ff0000"
    assert "groupsChanged" in rec.take()
    c.undo()  # the creation
    assert ws.group_by_name("Donors") is None


# -------------------------------------------------- template / layouts / tables
def _template_source(tmp_path, app):
    from flowdesk.ui.controller import WorkspaceController

    src = WorkspaceController()
    folder = tmp_path / "src"
    folder.mkdir(exist_ok=True)
    src.add_samples(make_files(folder, names=["T1_A1.fcs", "T1_A2.fcs"], n_events=[300, 300]))
    src.create_group("Panel", src.workspace.samples)
    everyone = src.workspace.groups[0]
    src.add_population(everyone.root, Population("Big", RectangleGate(["FSC-A"], [45000.0], [None])), everyone)
    src.set_scale(src.workspace.samples, "FITC-A", LinearScale(0, 5000))
    src.workspace.layouts.append(Layout("Figure 1", [LayoutItem(pop_path="/Big")]))
    src.workspace.tables.append(TableDef("Freqs", [TableColumn("/Big")]))
    src.workspace.settings["experiment"] = {"name": "From template"}
    return src


def test_apply_template_undo_restores_groups_gates_scales_layouts_tables(controller, tmp_path, app):
    from flowdesk.ui.experiment import apply_workspace_template, save_workspace_template

    src = _template_source(tmp_path, app)
    path = save_workspace_template(src, tmp_path / "panel.fdwt")
    c = controller
    ws = c.workspace
    own_layout = Layout("Mine")
    ws.layouts.append(own_layout)
    ws.tables.append(TableDef("Mine"))
    groups_before = list(ws.groups)
    scales_before = [s.scale("FITC-A").to_dict() for s in ws.samples]
    apply_workspace_template(c, path)
    assert ws.group_by_name("Panel") is not None and ws.samples[0].root.child_by_name("Big") is not None
    assert [l.name for l in ws.layouts] == ["Mine", "Figure 1"] and len(ws.tables) == 2
    assert ws.samples[0].scale("FITC-A").kind == "linear" and ws.settings["experiment"]["name"] == "From template"
    rec = Recorder(c)
    c.undo()
    assert ws.groups == groups_before and ws.group_by_name("Panel") is None
    assert all(s.root.child_by_name("Big") is None for s in ws.samples)
    assert [s.scale("FITC-A").to_dict() for s in ws.samples] == scales_before
    assert ws.layouts == [own_layout] and ws.layouts[0] is own_layout  # Layout identity
    assert [t.name for t in ws.tables] == ["Mine"]
    assert "experiment" not in ws.settings
    assert {"groupsChanged", "gatingChanged", "scalesChanged", "layoutsChanged", "tablesChanged", "settingsChanged"} <= rec.take()
    c.redo()
    assert ws.group_by_name("Panel") is not None and [l.name for l in ws.layouts] == ["Mine", "Figure 1"]


def test_main_window_template_path_gets_a_full_snapshot(controller, tmp_path, app, monkeypatch):
    """Opening a .fdwt in the main window (MainWindow._apply_template_path, the
    File > Open / Finder / command-line path) is one undo step that restores
    groups, gates, scales, layouts, tables and settings."""
    from flowdesk.ui import main_window as mw
    from flowdesk.ui.experiment import save_workspace_template

    path = save_workspace_template(_template_source(tmp_path, app), tmp_path / "panel.fdwt")
    c = controller
    ws = c.workspace
    own_layout = Layout("Mine")
    ws.layouts.append(own_layout)
    ws.tables.append(TableDef("Mine"))
    groups_before = list(ws.groups)
    scales_before = [s.scale("FITC-A").to_dict() for s in ws.samples]
    asked = []
    monkeypatch.setattr(mw, "confirm", lambda _parent, title, *_a, **_k: asked.append(title) or True)
    win = mw.MainWindow(c)
    try:
        c._undo.clear()
        win._apply_template_path(path)
        assert asked == ["Apply template"]
        assert ws.group_by_name("Panel") is not None and ws.samples[0].root.child_by_name("Big") is not None
        assert [l.name for l in ws.layouts] == ["Mine", "Figure 1"] and [t.name for t in ws.tables] == ["Mine", "Freqs"]
        assert ws.samples[0].scale("FITC-A").kind == "linear" and ws.settings["experiment"]["name"] == "From template"
        assert len(c._undo) == 1  # one step
        c.undo()
        assert ws.groups == groups_before and ws.group_by_name("Panel") is None
        assert all(s.root.child_by_name("Big") is None for s in ws.samples)
        assert [s.scale("FITC-A").to_dict() for s in ws.samples] == scales_before
        assert ws.layouts == [own_layout] and ws.layouts[0] is own_layout
        assert [t.name for t in ws.tables] == ["Mine"]
        assert "experiment" not in ws.settings
        assert not c.can_undo and c.can_redo
    finally:
        c.mark_saved()
        win.hide()
        win.deleteLater()
        app.processEvents()


def test_legacy_apply_template_label_gets_a_full_snapshot(controller, tmp_path, app):
    """begin_change("apply template") without a kind (callers written before
    kinds existed) still snapshots layouts and tables (LEGACY_UNDO_KINDS)."""
    from flowdesk.core.templates import apply_template, build_template

    template = build_template(_template_source(tmp_path, app).workspace)
    c = controller
    c.begin_change("apply template")
    apply_template(c.workspace, template)
    assert len(c.workspace.layouts) == 1 and len(c.workspace.tables) == 1
    c.undo()
    assert c.workspace.layouts == [] and c.workspace.tables == []
    assert c.workspace.group_by_name("Panel") is None


def test_layout_and_table_edits_undo_in_place(controller):
    c = controller
    ws = c.workspace
    lay = Layout("L", [LayoutItem(x=10.0)])
    tab = TableDef("T", [TableColumn("/Cells")])
    ws.layouts.append(lay)
    ws.tables.append(tab)
    item = lay.items[0]
    c.begin_change("move item", ("layouts", "tables"))
    item.x = 99.0
    lay.items.append(LayoutItem(kind="text", text="hello"))
    tab.columns.clear()
    rec = Recorder(c)
    c.undo()
    assert ws.layouts[0] is lay and lay.items == [item] and lay.items[0] is item and item.x == 10.0
    assert ws.tables[0] is tab and [col.pop_path for col in tab.columns] == ["/Cells"]
    assert {"layoutsChanged", "tablesChanged"} <= rec.take()


# ---------------------------------------------------------------- gating
def test_population_identity_kept_where_path_survives(controller):
    c = controller
    s = c.workspace.samples[0]
    cells = gate(c, s, "Cells")
    child = gate(c, s, "Small", parent=cells, lo=0.0)
    count = cells.count
    graph = cells.graph
    c.begin_change("edit gate")
    cells.gate.lo[0] = 55000.0  # in place, like a quadrant drag
    cells.invalidate()
    cells.graph.x = "PE-A"
    cells.remove_child(child)
    assert cells.count != count
    rec = Recorder(c)
    c.undo()
    assert s.root.child_by_name("Cells") is cells and cells.graph is graph and cells.graph.x is None
    assert cells.gate.lo[0] == 20000.0 and cells.count == count
    restored = cells.child_by_name("Small")
    assert restored is not None and restored.sample is s and restored.parent is cells
    seen = rec.take()
    assert "gatingChanged" in seen
    # a gating undo touches nothing it does not cover
    assert not seen & {"layoutsChanged", "tablesChanged", "settingsChanged", "samplesChanged"}


def test_gating_undo_leaves_sections_it_does_not_cover(controller):
    c = controller
    ws = c.workspace
    s = ws.samples[0]
    gate(c, s)
    ws.layouts.append(Layout("made after the gate"))
    ws.settings["plate_editor"] = {"format": 384}
    s.keywords["Note"] = "x"
    c.undo()
    assert s.root.child_by_name("Cells") is None
    assert [l.name for l in ws.layouts] == ["made after the gate"]
    assert ws.settings["plate_editor"] == {"format": 384} and s.keywords["Note"] == "x"


def test_restore_without_changes_emits_no_model_signals(controller):
    c = controller
    c.begin_change("nothing")
    rec = Recorder(c)
    c.undo()
    assert rec.take() == set()
    c.begin_change("nothing at all", "all")
    c.undo()
    assert rec.take() == set()


def test_undo_group_is_one_step_and_widens(controller):
    c = controller
    ws = c.workspace
    s = ws.samples[0]
    n = len(c._undo)
    with c.undo_group("batch"):  # gating only at first
        gate(c, s)
        c.set_scale([s], "PE-A", LinearScale(0, 10))
        c.rename_sample(s, "Batch renamed")  # widens the step to the sample section
        ws.settings["experiment"] = {"name": "not covered"}
    assert len(c._undo) == n + 1 and c.undo_label() == "batch"
    c.undo()
    assert s.root.child_by_name("Cells") is None and s.name == NAMES[0]
    assert ws.settings["experiment"] == {"name": "not covered"}  # settings were never part of the step


def test_undo_rename_keeps_population_objects(controller):
    """Finding 10: a rename undone / redone keeps the Population objects that
    graph windows and the tree hold (matched by id, not by the changed name)."""
    c = controller
    ws = c.workspace
    s = ws.samples[0]
    cells = gate(c, s, "Cells")
    small = gate(c, s, "Small", parent=cells, lo=0.0)
    count = cells.count
    c.rename_population(cells, "Lymph")
    c.undo()
    assert s.root.children == [cells] and cells.name == "Cells" and cells.parent is s.root
    assert cells.children == [small] and small.parent is cells and cells.count == count
    c.redo()
    assert s.root.children == [cells] and cells.name == "Lymph" and cells.count == count
    # group gates: the template node and every member's node survive too
    g = ws.groups[0]
    c.add_population(s.root, Population("G1", RectangleGate(["FSC-A"], [40000.0], [None])), group=g)
    template = g.root.child_by_name("G1")
    members = {x.id: x.root.child_by_name("G1") for x in ws.samples}
    assert template is not None and all(m is not None for m in members.values())
    c.rename_population(s.root.child_by_name("G1"), "G2", group=g)
    c.undo()
    assert g.root.child_by_name("G1") is template and g.root.child_by_name("G2") is None
    assert all(x.root.child_by_name("G1") is members[x.id] for x in ws.samples)


def test_section_recorder_inside_undo_group_restores_its_baseline(controller, app):
    """Finding 27: an editor that changed its section in place and records the
    change inside an undo group must not widen the step from the edited state."""
    from flowdesk.ui.layout_editor import SectionUndoRecorder

    c = controller
    ws = c.workspace
    s = ws.samples[0]
    item = LayoutItem(x=150.0)
    ws.layouts.append(Layout("L", [item]))
    tab = TableDef("T", [TableColumn("/Cells")])
    ws.tables.append(tab)
    layouts = SectionUndoRecorder(c, "layouts", "edit layout")
    tables = SectionUndoRecorder(c, "tables", "edit table")
    n = len(c._undo)
    with c.undo_group("outer"):  # gating only
        gate(c, s)
        item.x = 999.0  # in place, like a drag
        assert layouts.record() is False  # no step of its own
        item.x = 500.0
        layouts.record()  # a second edit keeps the first baseline
        tab.columns.append(TableColumn("/Other"))
        tables.record()
    assert len(c._undo) == n + 1 and c.undo_label() == "outer"
    c.undo()
    assert ws.layouts[0].items[0] is item and item.x == 150.0
    assert [col.pop_path for col in tab.columns] == ["/Cells"]
    assert s.root.child_by_name("Cells") is None
    c.redo()
    assert item.x == 500.0 and [col.pop_path for col in tab.columns] == ["/Cells", "/Other"]
    # a group that covered the section from the start keeps its own snapshot
    layouts.reset()
    with c.undo_group("layout batch", "layouts"):
        item.x = 1.0
        ws.layouts[0].name = "changed before the recorder saw it"
        layouts.reset()
        item.x = 2.0
        layouts.record()
    c.undo()
    assert item.x == 500.0 and ws.layouts[0].name == "L"


def test_undo_of_non_tree_changes_keeps_selection(controller):
    """Finding 30: undoing a layout / table / settings / scale change leaves the
    workspace-tree selection and current group alone; a gating undo resets."""
    c = controller
    ws = c.workspace
    s, t = ws.samples[0], ws.samples[1]
    cells = gate(c, s)
    group = c.create_group("Panel", [s])
    c.set_current_group(group)
    rows = [(s, cells), (t, t.root)]
    c.set_selection(rows, [group])
    seen = []
    c.selectionChanged.connect(lambda: seen.append("selection"))
    c.currentGroupChanged.connect(lambda g: seen.append("group"))
    ws.layouts.append(Layout("L", [LayoutItem(x=1.0)]))
    ws.tables.append(TableDef("T"))
    for kind, edit in (
        ("layouts", lambda: setattr(ws.layouts[0].items[0], "x", 5.0)),
        ("tables", lambda: ws.tables[0].columns.append(TableColumn("/Cells"))),
        ("settings", lambda: ws.settings.__setitem__("experiment", {"name": "x"})),
    ):
        c.begin_change("edit", kind)
        edit()
        c.undo()
        c.redo()
    c.set_scale([s], "PE-A", LinearScale(0, 10))
    c.undo()
    assert c.selected == rows and c.selected_groups == [group] and c.current_group is group
    assert seen == []
    # a gating change can invalidate the rows: those are reset as before
    c.begin_change("edit gate")
    cells.gate.lo[0] = 1.0
    c.undo()
    assert c.selected == [] and c.current_group is group
    assert seen == ["group", "selection"]


# ------------------------------------------------------------------ cost
def test_snapshots_share_computed_arrays_and_are_fast(app, tmp_path):
    from flowdesk.ui.controller import WorkspaceController

    c = WorkspaceController()
    c.add_samples(make_files(tmp_path, n_events=200_000))
    c._undo.clear()
    ws = c.workspace
    rng = np.random.default_rng(0)
    for s in ws.samples:
        gate(c, s)
        gate(c, s, "Small", parent=s.root.child_by_name("Cells"), lo=0.0)
        for name in ("tSNE_1", "tSNE_2", "FlowSOM"):
            s.add_computed(name, rng.random(s.n_events, dtype=np.float32))
    assert all(s.computed["tSNE_1"].shape == (200_000,) for s in ws.samples)
    c._undo.clear()
    t0 = time.perf_counter()
    for k in range(30):
        c.begin_change(f"change {k}", "all" if k % 2 else "gating")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"30 snapshots took {elapsed:.3f}s"
    assert len(c._undo) == 30
    for _, snap in c._undo:
        for s in ws.samples:
            for name, arr in snap["samples"][s.id]["computed"].items():
                assert arr is s.computed[name] and np.shares_memory(arr, s.computed[name])
    # undo / redo of a change that did not touch the arrays keeps them too
    arrays = {s.id: dict(s.computed) for s in ws.samples}
    c.undo()
    c.redo()
    assert all(s.computed[n] is a for s in ws.samples for n, a in arrays[s.id].items())


# ------------------------------------------------------------ workspace hub
def test_move_gate_undo_restores_tree_booleans_and_layouts(controller, app):
    """Re-parenting a gate (drag inside a sample) is one undo step that brings
    back the tree, the Boolean references and the layout plots that followed."""
    from flowdesk.core.gates import BooleanGate
    from flowdesk.ui import main_window as mw

    c = controller
    ws = c.workspace
    s = ws.samples[0]
    cells = gate(c, s)
    cd3 = gate(c, s, "CD3+", parent=cells, lo=25000.0)
    live = gate(c, s, "Live", parent=cells, lo=22000.0)
    both = c.add_population(s.root, Population("Both", BooleanGate("and", ["/Cells/CD3+", "/Cells/Live"])))
    ws.layouts.append(Layout("L", [LayoutItem(sample_id=s.id, pop_path="/Cells/CD3+")]))
    c._undo.clear()
    win = mw.MainWindow(c)
    try:
        rec = Recorder(c)
        assert win.move_populations(s, [cd3], live) == 1
        assert s.root.find("/Cells/Live/CD3+") is cd3 and s.root.find("/Cells/CD3+") is None
        assert both.gate.refs == ["/Cells/Live/CD3+", "/Cells/Live"]
        assert ws.layouts[0].items[0].pop_path == "/Cells/Live/CD3+"
        assert len(c._undo) == 1 and c.undo_label() == "move gate"
        assert {"gatingChanged", "layoutsChanged"} <= rec.take()
        # moving a gate into its own sub-population is refused
        assert win.move_populations(s, [cells], live) == 0 and len(c._undo) == 1
        c.undo()
        assert [p.path for p in s.root.walk(False)] == ["/Cells", "/Cells/CD3+", "/Cells/Live", "/Both"]
        assert s.root.find("/Both").gate.refs == ["/Cells/CD3+", "/Cells/Live"]
        assert ws.layouts[0].items[0].pop_path == "/Cells/CD3+"
        c.redo()
        assert s.root.find("/Cells/Live/CD3+") is not None
    finally:
        c.mark_saved()
        win.hide()
        win.deleteLater()
        app.processEvents()


def test_group_tree_move_rename_and_file_drop_undo(controller, app, tmp_path):
    """Moving and renaming in the group tree changes every member and the
    template in one step; files dropped on a group join it in one step."""
    from flowdesk.ui import main_window as mw

    c = controller
    ws = c.workspace
    s0, s1 = ws.samples[:2]
    g = c.create_group("Panel", [s0, s1])
    c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A"], [20000.0], [None])), g)
    c.add_population(s0.root.find("/Cells"), Population("A", RectangleGate(["FSC-A"], [30000.0], [None])), g)
    c.add_population(s0.root.find("/Cells"), Population("B", RectangleGate(["SSC-A"], [10000.0], [None])), g)
    c._undo.clear()
    win = mw.MainWindow(c)
    try:
        assert win.move_group_populations(g, [g.root.find("/Cells/A")], g.root.find("/Cells/B")) == 1
        for root in (g.root, s0.root, s1.root):
            assert root.find("/Cells/B/A") is not None and root.find("/Cells/A") is None
        assert len(c._undo) == 1
        c.undo()
        for root in (g.root, s0.root, s1.root):
            assert root.find("/Cells/A") is not None and root.find("/Cells/B/A") is None
        # F2 on the group row renames the group; on a template row, the gate everywhere
        assert win.rename_group(g, "Donors") and g.name == "Donors"
        assert not win.rename_group(g, "Compensation")  # taken
        win._group_item_renamed(g, g.root.find("/Cells"), "Lymphs")
        assert all(r.find("/Lymphs/A") is not None for r in (g.root, s0.root, s1.root))
        c.undo()
        c.undo()
        assert g.name == "Panel" and all(r.find("/Cells/A") is not None for r in (g.root, s0.root, s1.root))
        # files dropped on the group: added and members in one step
        c._undo.clear()
        (tmp_path / "x").mkdir()
        extra = make_files(tmp_path / "x", names=["Well_X1.fcs"], n_events=[100])[0]
        added = win._files_dropped([str(extra)], g)
        assert [s.path for s in added] == [extra] and g.contains(added[0]) and added[0].root.find("/Cells/A") is not None
        assert len(c._undo) == 1
        c.undo()
        assert ws.sample_by_path(extra) is None and g.sample_ids == {s0.id, s1.id}
    finally:
        c.mark_saved()
        win.hide()
        win.deleteLater()
        app.processEvents()
