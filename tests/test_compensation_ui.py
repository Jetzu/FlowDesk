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
