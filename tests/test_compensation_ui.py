import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from flowdesk.core.compensation import SpilloverMatrix
from flowdesk.core.model import Workspace
from flowdesk.ui.compensation import (
    CompensationWindow,
    CompensationWizard,
    compute_spillover,
    fluorescence_params,
    guess_control_assignment,
    read_matrix_csv,
    write_matrix_csv,
)
from flowdesk.ui.controller import WorkspaceController

DATA = Path("/Users/lukasgerstung/Coding/Plotting/FACS-Data/Experiment_20260430")
CONTROLS = [
    "Compensation Controls_Unstained Control.fcs",
    "Compensation Controls_FITC Stained Control.fcs",
    "Compensation Controls_PE Stained Control.fcs",
    "Compensation Controls_APC Stained Control.fcs",
]
needs_data = pytest.mark.skipif(not all((DATA / c).exists() for c in CONTROLS), reason="real FCS files not available")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _specimens(n=2):
    return sorted(DATA.glob("Specimen_001_*.fcs"))[:n]


@pytest.fixture
def controller(qapp):
    ws = Workspace()
    for c in CONTROLS:
        ws.add_sample(DATA / c)
    for p in _specimens(2):
        ws.add_sample(p)
    return WorkspaceController(ws)


def test_guess_assignment_real_names():
    params = ["FITC-A", "APC-A", "PE-A"]
    g = guess_control_assignment(CONTROLS, params, {})
    assert g == {
        CONTROLS[0]: "Unstained",
        CONTROLS[1]: "FITC-A",
        CONTROLS[2]: "PE-A",
        CONTROLS[3]: "APC-A",
    }


def test_guess_assignment_tandems_and_tokens():
    params = ["FITC-A", "PE-A", "PE-Cy7-A", "PerCP-Cy5-5-A", "APC-A", "APC-Cy7-A", "Blue_B-A", "Red_C-A"]
    names = ["PE Stained Control", "PE-Cy7 control", "PerCP-Cy5.5 beads", "APC-Cy7_ctrl", "APC only", "Blue_B comp", "unstained", "Comp_FITC.fcs", "CD3 tube", "nothing"]
    g = guess_control_assignment(names, params, {"Red_C-A": "CD3"})
    assert g["PE Stained Control"] == "PE-A"
    assert g["PE-Cy7 control"] == "PE-Cy7-A"
    assert g["APC-Cy7_ctrl"] == "APC-Cy7-A"
    assert g["APC only"] == "APC-A"
    assert g["Blue_B comp"] == "Blue_B-A"
    assert g["unstained"] == "Unstained"
    assert g["Comp_FITC.fcs"] == "FITC-A"
    assert g["CD3 tube"] == "Red_C-A"
    assert "nothing" not in g
    assert g.get("PerCP-Cy5.5 beads") != "PE-A"


def test_csv_roundtrip(tmp_path):
    m = SpilloverMatrix("M", ["A-A", "B-A"], np.array([[1, 0.12], [0.03, 1]]))
    path = tmp_path / "m.csv"
    write_matrix_csv(m, path)
    back = read_matrix_csv(path)
    assert back == m and back.name == "m"


@needs_data
def test_fluorescence_params_and_comp_group(controller):
    ws = controller.workspace
    s = ws.samples[0]
    assert fluorescence_params(s) == ["FITC-A", "APC-A", "PE-A"]
    comp = next(g for g in ws.groups if g.role == "compensation")
    assert sorted(x.name for x in ws.members(comp)) == sorted(CONTROLS)


@needs_data
def test_wizard_computation(controller):
    wiz = CompensationWizard(controller)
    assignment = wiz.assignment()
    assert sorted(assignment.values()) == ["APC-A", "FITC-A", "PE-A", "Unstained"]
    assert wiz.controls_page.universal_negative.isChecked()
    m = compute_spillover(assignment, True, "Computed")
    assert m.matrix.shape == (3, 3)
    assert np.allclose(np.diag(m.matrix), 1.0)
    off = m.matrix[~np.eye(3, dtype=bool)]
    assert np.all(off >= 0) and np.all(off < 0.5)
    # finishing the wizard stores the matrix and applies it to All Samples
    wiz.show()
    wiz.next()
    assert wiz.matrix is not None and wiz.matrix == m
    wiz.matrix_page.name_edit.setText("Wizard matrix")
    wiz.accept()
    assert "Wizard matrix" in controller.workspace.matrices
    assert all(s.comp is not None and s.comp.name == "Wizard matrix" for s in controller.workspace.samples)


@needs_data
def test_window_lists_and_applies(controller):
    win = CompensationWindow(controller)
    win.show()
    labels = win.matrix_labels()
    assert any(l.startswith("Acquisition-defined") for l in labels)
    # pick a non-identity acquisition-defined matrix and apply it to all samples
    target = next(e for e in win._entries if not e.matrix.is_identity)
    win.select_matrix(target.label)
    win.apply_to_all()
    for s in controller.workspace.samples:
        assert s.comp is not None and s.comp.name == target.label
        assert s.parameter_names()[0].startswith("Comp-")
    assert win.status_table.rowCount() == len(controller.workspace.samples)
    assert win.status_table.item(0, 1).text() == target.label
    # removing compensation
    controller.set_selection([(controller.workspace.samples[0], controller.workspace.samples[0].root)])
    win.remove_from_selected()
    assert controller.workspace.samples[0].comp is None
    assert win.status_table.item(0, 1).text() == "none"
    win.close()
    assert not win.isVisible()


@needs_data
def test_editing_acquisition_matrix_creates_copy(controller):
    win = CompensationWindow(controller)
    acq = next(e for e in win._entries if not e.stored)
    win.select_matrix(acq.label)
    assert not (win.grid.item(0, 0).flags() & win.grid.item(0, 0).flags().ItemIsEditable)
    win.grid.item(0, 1).setText("12.5")
    name = f"{acq.label} (edited)"
    assert name in controller.workspace.matrices
    assert controller.workspace.matrices[name].matrix[0, 1] == pytest.approx(0.125)
    assert win.current_entry().label == name
    assert win.grid.item(0, 1).text() == "12.50"
    # a second edit changes the stored copy in place
    win.grid.item(1, 0).setText("3")
    assert controller.workspace.matrices[name].matrix[1, 0] == pytest.approx(0.03)
    assert len([k for k in controller.workspace.matrices if k.startswith(acq.label)]) == 1


@needs_data
def test_new_duplicate_rename_delete(controller):
    win = CompensationWindow(controller)
    m = win.new_matrix()
    assert m is not None and m.is_identity and m.params == ["FITC-A", "APC-A", "PE-A"]
    d = win.duplicate_matrix()
    assert d.name in controller.workspace.matrices
    win.rename_matrix("Renamed")
    assert "Renamed" in controller.workspace.matrices and d.name not in controller.workspace.matrices
    win.delete_matrix(confirm=False)
    assert "Renamed" not in controller.workspace.matrices


def test_wizard_finish_is_one_undo_step(qapp, tmp_path):
    """Storing the matrix and applying it to the group undo together."""
    from .conftest import SYNTH_NAMES, SYNTH_STAINS, make_synthetic_events, write_fcs

    ws = Workspace()
    for i in range(2):
        ws.add_sample(write_fcs(tmp_path / f"w{i}.fcs", make_synthetic_events(500, seed=40 + i)["events"], SYNTH_NAMES, SYNTH_STAINS))
    c = WorkspaceController(ws)
    before = [(s.comp.name if s.comp is not None else None, s.comp_enabled) for s in ws.samples]
    wiz = CompensationWizard(c)
    wiz.matrix = SpilloverMatrix("W", ["FITC-A", "PE-A"], np.array([[1.0, 0.1], [0.05, 1.0]]))
    page = wiz.matrix_page
    page.name_edit.setText("Wizard W")
    page.group_combo.clear()
    page.group_combo.addItem(next(g for g in ws.groups if g.role == "all").name)
    assert page.apply_check.isChecked()
    steps = len(c._undo)
    wiz.accept()
    assert len(c._undo) == steps + 1
    assert "Wizard W" in ws.matrices
    assert all(s.comp is not None and s.comp.name == "Wizard W" and s.comp_enabled for s in ws.samples)
    c.undo()  # one Ctrl+Z reverts the whole Finish
    assert "Wizard W" not in ws.matrices
    assert [(s.comp.name if s.comp is not None else None, s.comp_enabled) for s in ws.samples] == before
    assert not c.can_undo
    c.redo()
    assert "Wizard W" in ws.matrices
    assert all(s.comp is not None and s.comp.name == "Wizard W" for s in ws.samples)
    wiz.deleteLater()


# ====================================================== synthetic controls
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest

from flowdesk.core.compensation import ControlGates, read_matrix_mtx
from flowdesk.core.gates import RectangleGate
from flowdesk.core.model import Population
from flowdesk.ui.compensation import CONTROLS_KEY, ControlSet, default_control_population, ensure_group_hook

from .conftest import write_fcs

SYN_NAMES = ["FSC-A", "SSC-A", "FITC-A", "PE-A", "APC-A", "Time"]
SYN_STAINS = ["", "", "CD3", "CD4", "CD8", ""]
SYN_PARAMS = ["FITC-A", "PE-A", "APC-A"]
S3 = np.array([[1.0, 0.15, 0.02], [0.05, 1.0, 0.20], [0.0, 0.08, 1.0]])


def _write_controls(folder, n=3000, debris=0.0, seed=0, prefix="Comp_", roles=("Unstained", "FITC", "PE", "APC")):
    rng = np.random.default_rng(seed)
    paths = []
    for role in roles:
        true = rng.normal(0, 60, (n, 3))
        if role in ("FITC", "PE", "APC"):
            i = ["FITC", "PE", "APC"].index(role)
            pos = rng.random(n) < 0.5
            true[pos, i] += rng.lognormal(np.log(40000), 0.3, pos.sum())
        fl = true @ S3 + rng.normal(0, 20, (n, 3))
        fsc = rng.normal(90000, 8000, n)
        ssc = rng.normal(50000, 6000, n)
        if debris:
            d = rng.random(n) < debris
            fsc[d] = rng.normal(12000, 3000, d.sum())
            ssc[d] = rng.normal(6000, 1500, d.sum())
            # bright, flat-spectrum autofluorescent debris
            fl[d] = rng.lognormal(np.log(20000), 0.3, (d.sum(), 1)) * np.ones((1, 3))
        events = np.column_stack([fsc, ssc, fl, np.linspace(0, 100, n)]).astype(np.float32)
        paths.append(write_fcs(Path(folder) / f"{prefix}{role}.fcs", events, SYN_NAMES, SYN_STAINS))
    return paths


def _add_cells_gate(sample):
    pop = Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [50000.0, 25000.0], [None, None]))
    sample.root.add_child(pop)
    pop.invalidate()
    return pop


@pytest.fixture
def syn(qapp, tmp_path):
    ws = Workspace()
    for p in _write_controls(tmp_path):
        ws.add_sample(p)
    return WorkspaceController(ws)


def _err(m):
    idx = [m.params.index(p) for p in SYN_PARAMS]
    return np.abs(m.matrix[np.ix_(idx, idx)] - S3).max()


def test_wizard_pages_and_synthetic_accuracy(syn):
    wiz = CompensationWizard(syn)
    assert [wiz.page(i).title() for i in wiz.pageIds()] == ["Controls", "Gates", "Matrix", "Diagnostics"]
    assert sorted(wiz.assignment().values()) == ["APC-A", "FITC-A", "PE-A", "Unstained"]
    wiz.show()
    wiz.next()  # Gates: the matrix is computed with the automatic split
    assert wiz.currentPage() is wiz.gates_page
    assert wiz.matrix is not None and _err(wiz.matrix) < 0.02
    assert wiz.message.endswith("matrix computed.")
    assert set(wiz.gates_page.tiles) == set(SYN_PARAMS)
    tile = wiz.gates_page.tile("FITC-A")
    assert tile.view.pixmap() is not None and not tile.view.pixmap().isNull()
    assert "OK" in tile.info.text() and "stain index" in tile.info.text()
    wiz.deleteLater()


def test_moving_the_positive_gate_changes_the_matrix(syn):
    wiz = CompensationWizard(syn)
    wiz.show()
    wiz.next()
    before = wiz.matrix.copy()
    tile = wiz.gates_page.tile("PE-A")
    view = tile.view
    # drag a positive range from the negatives to the top of the scale
    x0, x1 = int(view.pixel_at(-100.0)), int(view.pixel_at(200000.0))
    y = int(view.result.plot_rect.center().y())
    QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x0, y))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x1, y))
    g = wiz.gates["PE-A"]
    assert g.pos is not None and g.pos[0] < 0 < 100000 < g.pos[1]
    assert not np.allclose(wiz.matrix.matrix, before.matrix)
    assert wiz.diagnostics["PE-A"].method == "gates"
    assert tile.pos_lo.value() == pytest.approx(g.pos[0], abs=1.0)
    # typing a range works the same way; "Auto" goes back to the automatic split
    tile.pos_lo.setValue(10000)
    assert wiz.gates["PE-A"].pos[0] == pytest.approx(10000)
    tile.auto_button.click()
    assert "PE-A" not in wiz.gates
    assert np.allclose(wiz.matrix.matrix, before.matrix)
    # a right-button drag sets the negative range
    x2 = int(view.pixel_at(2000.0))
    QTest.mousePress(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, QPoint(int(view.pixel_at(-500.0)), y))
    QTest.mouseRelease(view, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, QPoint(x2, y))
    assert wiz.gates["PE-A"].neg is not None and wiz.gates["PE-A"].pos is None
    assert wiz.diagnostics["PE-A"].negative == "internal"
    wiz.deleteLater()


def test_control_without_positives_shows_a_warning(qapp, tmp_path):
    ws = Workspace()
    for p in _write_controls(tmp_path, roles=("Unstained", "FITC", "APC")):
        ws.add_sample(p)
    # a second unstained tube, named like a PE control
    for p in _write_controls(tmp_path, roles=("tube",), prefix="Comp_PE_", seed=9):
        ws.add_sample(p)
    c = WorkspaceController(ws)
    wiz = CompensationWizard(c)
    assert "PE-A" in wiz.assignment().values()
    wiz.show()
    wiz.next()
    assert "matrix computed" not in wiz.message
    assert "PE-A" in wiz.message and "no distinct positive population" in wiz.message
    assert wiz.gates_page.message.text() == wiz.message
    tile = wiz.gates_page.tile("PE-A")
    assert "no distinct positive population" in tile.info.text()
    pe = wiz.matrix.params.index("PE-A")
    np.testing.assert_allclose(np.delete(wiz.matrix.matrix[pe], pe), 0.0)
    wiz.next()
    wiz.next()  # Diagnostics
    page = wiz.diagnostics_page
    rows = {page.table.item(r, 0).text(): r for r in range(page.table.rowCount())}
    assert set(rows) == set(SYN_PARAMS)
    assert "no distinct positive population" in page.table.item(rows["PE-A"], 10).text()
    assert page.table.item(rows["FITC-A"], 10).text() == "OK"
    assert int(page.table.item(rows["FITC-A"], 4).text().replace(",", "")) > 1000
    wiz.deleteLater()


def test_population_column_gates_out_debris(qapp, tmp_path):
    ws = Workspace()
    for p in _write_controls(tmp_path, debris=0.25, seed=3):
        s = ws.add_sample(p)
        _add_cells_gate(s)
    c = WorkspaceController(ws)
    s0 = ws.samples[0]
    assert default_control_population(s0).name == "Cells"
    wiz = CompensationWizard(c)
    page = wiz.controls_page
    # the scatter gate is picked by default and the Events column shows its count
    for r, s in enumerate(wiz.control_samples):
        assert page.pop_combos[r].currentData() == "/Cells"
        cells = s.find_population("/Cells")
        assert page.table.item(r, 2).text() == f"{cells.count:,}"
        assert cells.count < s.n_events
    gated = wiz.compute().copy()
    for r in range(len(wiz.control_samples)):
        page.pop_combos[r].setCurrentIndex(0)  # Ungated
    assert page.table.item(0, 2).text() == f"{wiz.control_samples[0].n_events:,}"
    ungated = wiz.compute().copy()
    assert not np.allclose(gated.matrix, ungated.matrix)
    assert _err(gated) < 0.02 < _err(ungated)
    wiz.deleteLater()


def test_wizard_group_selector(qapp, tmp_path):
    ws = Workspace()
    for p in _write_controls(tmp_path, prefix="Day1_", roles=("Blank", "FITC", "PE", "APC")):
        ws.add_sample(p)
    extra = write_fcs(tmp_path / "Specimen.fcs", np.ones((50, 6), np.float32), SYN_NAMES, SYN_STAINS)
    ws.add_sample(extra)
    c = WorkspaceController(ws)
    day1 = c.create_group("Day 1 controls", [s for s in ws.samples if s.name.startswith("Day1_")])
    wiz = CompensationWizard(c)
    # the Compensation group is empty: every sample is listed
    assert not wiz.used_comp_group and len(wiz.control_samples) == 5
    assert "empty" in wiz.controls_page.info.text()
    page = wiz.controls_page
    page.group_combo.setCurrentIndex(ws.groups.index(day1))
    assert wiz.control_group is day1 and len(wiz.control_samples) == 4
    assert sorted(wiz.assignment().values()) == ["APC-A", "FITC-A", "PE-A", "Unstained"]
    assert page.table.rowCount() == 4
    # the wizard can also be opened on a group
    wiz2 = CompensationWizard(c, group=day1)
    assert len(wiz2.control_samples) == 4
    wiz.deleteLater()
    wiz2.deleteLater()


def test_autospill_method_and_diagnostics_page(syn):
    wiz = CompensationWizard(syn)
    wiz.show()
    wiz.next()
    wiz.next()  # Matrix
    page = wiz.matrix_page
    median = wiz.matrix.copy()
    page.method_combo.setCurrentIndex(page.method_combo.findData("autospill"))
    assert wiz.method == "autospill"
    assert wiz.matrix.report is not None and wiz.matrix.report.converged
    assert "AutoSpill converged" in wiz.message and "AutoSpill" in page.message.text()
    assert _err(wiz.matrix) < 0.02 and not np.allclose(wiz.matrix.matrix, median.matrix)
    wiz.next()  # Diagnostics
    d = wiz.diagnostics_page
    assert d.ssm_grid.rowCount() == 3 and d.ssm_grid.columnCount() == 3
    assert wiz.ssm is not None and np.all(np.diag(wiz.ssm) == 0)
    assert not (d.ssm_grid.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
    assert "Largest unmixing residuals" in d.report.text()
    wiz.deleteLater()


def test_finish_stores_controls_gates_and_group_matrix(syn):
    ws = syn.workspace
    wiz = CompensationWizard(syn)
    wiz.show()
    wiz.next()
    wiz.gates_page.set_gate("FITC-A", ControlGates((5000.0, float("inf"))))
    wiz.next()
    wiz.matrix_page.name_edit.setText("Synthetic")
    steps = len(syn._undo)
    wiz.accept()
    assert len(syn._undo) == steps + 1
    stored = ws.settings[CONTROLS_KEY]
    assert stored["matrix"] == "Synthetic" and stored["gates"]["FITC-A"]["pos"] == [5000.0, "inf"]
    assert {c["role"] for c in stored["controls"]} == {"Unstained", "FITC-A", "PE-A", "APC-A"}
    allg = next(g for g in ws.groups if g.role == "all")
    assert ws.settings["group_matrices"][allg.name]["matrix"] == "Synthetic"
    # reopening restores the controls and the gates
    wiz2 = CompensationWizard(syn)
    assert wiz2.gates["FITC-A"].pos == (5000.0, float("inf"))
    cs = ControlSet.from_settings(ws, stored)
    assert set(cs.singles()) == set(SYN_PARAMS)
    syn.undo()
    assert CONTROLS_KEY not in ws.settings and "group_matrices" not in ws.settings and "Synthetic" not in ws.matrices
    wiz.deleteLater()
    wiz2.deleteLater()


# ====================================================== group-level compensation
def test_group_matrix_applies_to_samples_joining_later(qapp, tmp_path):
    ws = Workspace()
    paths = _write_controls(tmp_path, n=200, prefix="S_")
    first = ws.add_sample(paths[0])
    c = WorkspaceController(ws)
    g = c.create_group("Panel", [first])
    m = SpilloverMatrix("Panel matrix", SYN_PARAMS, S3)
    c.store_matrix(m)
    win = CompensationWindow(c)
    win.select_matrix("Panel matrix")
    win.apply_to_group(g, attach=True)
    assert first.comp.name == "Panel matrix"
    assert ws.settings["group_matrices"]["Panel"]["matrix"] == "Panel matrix"
    assert "attached to Panel" in win.title_label.text()
    # a sample added later (manual membership) inherits the matrix
    later = c.add_samples([paths[1]])[0]
    assert later.comp is None or later.comp.name != "Panel matrix"
    steps = len(c._undo)
    c.add_samples_to_group(g, [later])
    assert later.comp is not None and later.comp.name == "Panel matrix" and later.comp_enabled
    assert len(c._undo) == steps + 1  # joins the "add to group" step
    c.undo()
    assert later.comp is None or later.comp.name != "Panel matrix"
    assert not g.contains(later)
    c.redo()
    assert later.comp.name == "Panel matrix" and g.contains(later)
    # a sample whose matrix was changed by hand keeps that choice
    c.set_compensation([later], None)
    c.groupsChanged.emit()
    assert later.comp is None
    win.deleteLater()


def test_group_matrix_from_settings_and_live_criteria(qapp, tmp_path):
    from flowdesk.core.model import Criterion

    ws = Workspace()
    paths = _write_controls(tmp_path, n=200, prefix="Tube_")
    c = WorkspaceController(ws)
    g = c.create_group("PE tubes", criteria=[Criterion("$FIL", "Contains", "PE", "And")])
    c.store_matrix(SpilloverMatrix("M", SYN_PARAMS, S3))
    ws.settings["group_matrices"] = {g.name: "M"}  # a plain matrix name
    ensure_group_hook(c)
    added = c.add_samples(paths)  # Tube_PE.fcs matches the live criterion
    pe = next(s for s in added if "PE" in s.name)
    other = next(s for s in added if "FITC" in s.name)
    assert pe.comp is not None and pe.comp.name == "M"
    assert other.comp is None or other.comp.name != "M"
    assert ws.settings["group_matrices"][g.name]["applied"] == [pe.id]


def test_apply_warns_about_missing_parameters(syn, tmp_path):
    ws = syn.workspace
    small = ws.add_sample(write_fcs(tmp_path / "two.fcs", np.ones((20, 3), np.float32), ["FSC-A", "FITC-A", "Time"]))
    m = SpilloverMatrix("Big", SYN_PARAMS, S3)
    syn.store_matrix(m)
    win = CompensationWindow(syn)
    win.select_matrix("Big")
    applied = win.apply_to_samples([ws.samples[0], small])
    assert ws.samples[0] in applied and small not in applied
    assert small.comp is None
    assert "two.fcs" in win.last_warning and "PE-A" in win.last_warning and "APC-A" in win.last_warning
    assert win._box is not None and win._box.isVisible()
    # a matrix that cannot be applied shows as such in the status table
    small.set_compensation(m.copy(), True)
    win.refresh()
    row = ws.samples.index(small)
    text = win.status_table.item(row, 1).text()
    assert "not applicable" in text and "PE-A" in text and "APC-A" in text
    win._box.close()
    win.deleteLater()


def test_stain_named_matrix_is_applied_by_stain(syn):
    ws = syn.workspace
    s = ws.samples[1]
    stain_matrix = SpilloverMatrix("Stains", ["CD3", "CD4", "CD8"], S3)
    win = CompensationWindow(syn)
    applied = win.apply_to_samples([s], stain_matrix)
    assert applied == [s] and s.comp.params == SYN_PARAMS and s.comp.name == "Stains"
    assert np.all(np.isfinite(s.data("Comp-PE-A")))
    # a stain-named matrix set directly (e.g. from $SPILLOVER) is renamed by the hook
    s2 = ws.samples[2]
    s2.set_compensation(stain_matrix.copy(), True)
    syn.compensationChanged.emit(None)
    assert s2.comp.params == SYN_PARAMS
    assert "Comp-FITC-A" in s2.parameter_names()
    win.deleteLater()


# ====================================================== window: files, undo, views
def test_window_mtx_and_fcs_import_export(syn, tmp_path):
    ws = syn.workspace
    syn.store_matrix(SpilloverMatrix("Exported", SYN_PARAMS, S3))
    win = CompensationWindow(syn)
    win.select_matrix("Exported")
    win.export_matrix(str(tmp_path / "out.mtx"))
    assert read_matrix_mtx(tmp_path / "out.mtx") == ws.matrices["Exported"]
    win.export_matrix(str(tmp_path / "out.txt"))
    assert (tmp_path / "out.txt").read_text().splitlines()[0].split("\t") == SYN_PARAMS
    win.export_csv(str(tmp_path / "out.csv"))
    assert read_matrix_csv(tmp_path / "out.csv") == ws.matrices["Exported"]
    imported = win.import_matrix(str(tmp_path / "out.mtx"))
    assert imported.name != "Exported" and imported == ws.matrices["Exported"]
    assert win.current_entry().label == imported.name
    t = win.import_matrix(str(tmp_path / "out.txt"))
    assert t == ws.matrices["Exported"]
    # another FCS file's $SPILLOVER
    acq = write_fcs(tmp_path / "acq.fcs", np.ones((5, 2), np.float32), ["FITC-A", "PE-A"], extra_keywords={"$SPILLOVER": "2,FITC-A,PE-A,1,0.2,0.05,1"})
    f = win.import_matrix(str(acq))
    assert f.name == "acq" and f.matrix[0, 1] == pytest.approx(0.2)
    win.deleteLater()


def test_ctrl_z_in_compensation_window_reverts_a_cell_edit(syn):
    syn.store_matrix(SpilloverMatrix("Edit me", SYN_PARAMS, S3))
    win = CompensationWindow(syn)
    win.show()
    win.raise_()
    win.activateWindow()
    qapp_ = QApplication.instance()
    qapp_.processEvents()
    win.select_matrix("Edit me")
    win.grid.item(0, 1).setText("33")
    assert syn.workspace.matrices["Edit me"].matrix[0, 1] == pytest.approx(0.33)
    assert win.act_undo.isEnabled()
    win.grid.setFocus()
    QTest.keyClick(win.grid, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    qapp_.processEvents()
    assert syn.workspace.matrices["Edit me"].matrix[0, 1] == pytest.approx(0.15)
    assert win.grid.item(0, 1).text() == "15.00"
    QTest.keySequence(win.grid, QKeySequence(QKeySequence.StandardKey.Redo))
    qapp_.processEvents()
    assert syn.workspace.matrices["Edit me"].matrix[0, 1] == pytest.approx(0.33)
    win.close()
    win.deleteLater()


def test_inverse_view_toggle(syn):
    syn.store_matrix(SpilloverMatrix("Inv", SYN_PARAMS, S3))
    win = CompensationWindow(syn)
    win.select_matrix("Inv")
    win.act_inverse.setChecked(True)
    assert win.view_mode == "inverse"
    inv = np.linalg.inv(S3)
    assert win.grid.item(0, 1).text() == f"{inv[0, 1] * 100:.2f}"
    assert win.grid.item(0, 0).text() == f"{inv[0, 0] * 100:.2f}"
    assert not (win.grid.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
    assert "inverse" in win.hint.text()
    win.set_view_mode("spill")
    assert not win.act_inverse.isChecked() and win.grid.item(0, 1).text() == "15.00"
    assert win.grid.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    win.deleteLater()


def test_copy_and_paste_matrix(syn):
    ws = syn.workspace
    syn.store_matrix(SpilloverMatrix("Copy", SYN_PARAMS, S3))
    win = CompensationWindow(syn)
    win.select_matrix("Copy")
    text = win.copy_matrix()
    assert QApplication.clipboard().text() == text
    assert text.splitlines()[0].split("\t") == [""] + SYN_PARAMS and "15" in text
    # pasting the same parameters replaces the values (one undo step)
    edited = text.replace("\t15\t", "\t25\t")
    steps = len(syn._undo)
    m = win.paste_matrix(edited)
    assert m.name == "Copy" and ws.matrices["Copy"].matrix[0, 1] == pytest.approx(0.25)
    assert len(syn._undo) == steps + 1
    syn.undo()
    assert ws.matrices["Copy"].matrix[0, 1] == pytest.approx(0.15)
    # other parameters become a new matrix; the clipboard is read when no text is given
    QApplication.clipboard().setText("\tA\tB\nA\t100\t7\nB\t2\t100\n")
    new = win.paste_matrix()
    assert new.params == ["A", "B"] and new.name.startswith("Pasted matrix") and new.name in ws.matrices
    assert win.paste_matrix("not a matrix") is None and "no matrix" in win.last_warning
    win.deleteLater()


def test_control_plot_panel_and_hot_spots(syn):
    ws = syn.workspace
    good = SpilloverMatrix("True", SYN_PARAMS, S3)
    syn.store_matrix(good)
    win = CompensationWindow(syn)
    win.select_matrix("True")
    win.tabs.setCurrentIndex(1)
    win.refresh_panel()
    panel = win.panel
    assert len(panel.tiles) == 9
    assert all(panel.images[(i, j)] is not None for i in range(3) for j in range(3))
    assert all(panel.hot(i, j) is None for i in range(3) for j in range(3))
    # an under-compensated cell is a hot spot in the panel and in the grid
    win.set_cell(1, 2, 0.0)  # PE -> APC
    win.refresh_panel()
    assert panel.hot(1, 2) is not None and panel.hot(1, 2) > 0.1
    assert win.grid.item(1, 2).background().color().name() == "#f4b6b6"
    assert "under-compensated" in win.grid.item(1, 2).toolTip()
    assert "hot spot" in panel.message.text()
    # clicking a plot selects its matrix cell; the wheel nudges the value
    panel.tiles[(1, 2)].clicked.emit(1, 2)
    assert (win.grid.currentRow(), win.grid.currentColumn()) == (1, 2)
    assert win.cell_spin.value() == pytest.approx(0.0)
    panel.tiles[(1, 2)].wheeled.emit(1, 2, 1.0)
    assert ws.matrices["True"].matrix[1, 2] == pytest.approx(0.01)
    win.cell_spin.setValue(20.0)
    assert ws.matrices["True"].matrix[1, 2] == pytest.approx(0.20)
    win.refresh_panel()
    assert panel.hot(1, 2) is None
    win.deleteLater()


def test_spreading_matrix_tab(syn):
    syn.store_matrix(SpilloverMatrix("True", SYN_PARAMS, S3))
    win = CompensationWindow(syn)
    win.select_matrix("True")
    ssm = win.compute_ssm()
    assert ssm is not None and ssm.shape == (3, 3) and win.ssm_params == SYN_PARAMS
    assert win.ssm_grid.rowCount() == 3 and win.ssm_grid.item(0, 0).text() == "0.00"
    assert not (win.ssm_grid.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable)
    assert "Largest spread" in win.ssm_message.text()
    # a matrix written with $PnS stain names gives the same spreading matrix
    syn.store_matrix(SpilloverMatrix("Stains", ["CD3", "CD4", "CD8"], S3))
    win.select_matrix("Stains")
    np.testing.assert_allclose(win.compute_ssm(), ssm)
    win.deleteLater()


def test_install_hook_once(syn):
    from types import SimpleNamespace

    from flowdesk.ui.compensation import install

    hook = install(SimpleNamespace(controller=syn))
    assert hook is ensure_group_hook(syn) and syn._group_matrix_hook is hook


def test_autospill_action_in_window(syn):
    ws = syn.workspace
    start = S3.copy()
    start[0, 1] = 0.10  # a poor starting matrix
    syn.store_matrix(SpilloverMatrix("Start", SYN_PARAMS, start))
    win = CompensationWindow(syn)
    win.select_matrix("Start")
    steps = len(syn._undo)
    m = win.refine_current()
    assert m is not None and m.name == "Start (AutoSpill)" and m.name in ws.matrices
    assert abs(m.matrix[0, 1] - 0.15) < 0.01
    assert len(syn._undo) == steps + 1 and "converged" in win.last_message
    syn.undo()
    assert "Start (AutoSpill)" not in ws.matrices
    win._box.close()
    win.deleteLater()


CYTOFLEX = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/260318/Comp_20260318_1")


@pytest.mark.skipif(not CYTOFLEX.is_dir(), reason="real CytoFLEX controls not available")
def test_wizard_on_real_cytoflex_controls(qapp):
    """Eight single-stain tubes named by fluorochrome (FITC.fcs, PC5.5.fcs, …)
    on FL1-A … FL10-A with $PnS "FITC-A" …: the wizard assigns them by stain,
    flags the dim PC5.5 control and matches CytExpert's matrix elsewhere."""
    ws = Workspace()
    for p in sorted(CYTOFLEX.glob("*.fcs")):
        ws.add_sample(p)
    c = WorkspaceController(ws)
    wiz = CompensationWizard(c)
    roles = {s.name: r for s, r in wiz.assignment().items()}
    assert roles["FITC.fcs"] == "FL1-A" and roles["PC5.5.fcs"] == "FL4-A" and roles["APC-A750.fcs"] == "FL8-A"
    m = wiz.compute()
    assert m.diagnostics["FL4-A"].status == "weak" and "FL4-A" in wiz.message
    acq = ws.samples[0].comp  # CytExpert's matrix ($SPILLOVER)
    idx = [acq.params.index(p) for p in m.params]
    ref = acq.matrix[np.ix_(idx, idx)]
    good = [i for i, p in enumerate(m.params) if m.diagnostics[p].status == "ok"]
    assert len(good) == 7
    assert np.abs(m.matrix - ref)[good].max() < 0.005
    wiz.deleteLater()
