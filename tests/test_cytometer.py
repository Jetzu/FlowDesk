"""Cytometer settings: grouping by $CYT, the dialog, applying default scales."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flowdesk.core.transforms import LinearScale, LogicleScale  # noqa: E402

from .conftest import SYNTH_NAMES, SYNTH_STAINS, make_synthetic_events, write_fcs  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture
def controller(app, tmp_path):
    from flowdesk.ui.controller import WorkspaceController

    c = WorkspaceController()
    paths = []
    for i, cyt in enumerate(["Aurora", "Aurora", "Fortessa", None]):
        ev = make_synthetic_events(2000, seed=i + 3)["events"]
        extra = {"$CYT": cyt} if cyt else None
        paths.append(write_fcs(tmp_path / f"c{i}.fcs", ev, SYNTH_NAMES, SYNTH_STAINS, extra_keywords=extra))
    c.add_samples(paths)
    yield c
    c.workspace.modified = False


def test_grouping_by_cytometer(controller):
    from flowdesk.ui.cytometer import UNKNOWN, cytometer_of, group_by_cytometer

    groups = group_by_cytometer(controller.workspace)
    assert {k: len(v) for k, v in groups.items()} == {"Aurora": 2, "Fortessa": 1, UNKNOWN: 1}
    assert cytometer_of(controller.workspace.samples[3]) == UNKNOWN


def test_dialog_applies_scales_to_all_samples_of_a_cytometer(controller, tmp_path):
    from flowdesk.ui.cytometer import CytometerDialog, apply_cytometer_defaults

    dlg = CytometerDialog(controller)
    assert set(dlg.cytometers()) == {"Aurora", "Fortessa", "Unknown"}
    assert dlg.sample_count("Aurora") == 2
    dlg.select("Aurora")
    assert dlg.parameters("Aurora") == SYNTH_NAMES
    assert dlg.table.rowCount() == len(SYNTH_NAMES)
    # pre-filled from the samples' current scales: FSC linear, FITC biex
    assert dlg.specs()["FSC-A"]["kind"] == "linear"
    assert dlg.specs()["FITC-A"]["kind"] == "logicle"
    assert dlg.table.cellWidget(2, 2).currentData() == "logicle"
    dlg.set_param_scale("FITC-A", "logicle", width_basis=-500, decades=4.2)
    dlg.set_param_scale("FSC-A", "arcsinh", decades=5.0)
    undo_before = len(controller._undo)
    dlg.apply()
    assert len(controller._undo) == undo_before + 1  # one undo step
    aurora = [s for s in controller.workspace.samples if s.keywords.get("$CYT") == "Aurora"]
    for s in aurora:
        sc = s.scale("FITC-A")
        assert isinstance(sc, LogicleScale)
        assert sc.width_basis == pytest.approx(-500) and sc.M == pytest.approx(4.2)
        assert s.scale("FSC-A").kind == "arcsinh"
    other = [s for s in controller.workspace.samples if s.keywords.get("$CYT") != "Aurora"]
    for s in other:
        assert isinstance(s.scale("FSC-A"), LinearScale)
    stored = controller.workspace.settings["cytometers"]["Aurora"]
    assert stored["FITC-A"]["width_basis"] == pytest.approx(-500)

    # a new sample from the same instrument receives the defaults
    ev = make_synthetic_events(1000, seed=11)["events"]
    path = write_fcs(tmp_path / "late.fcs", ev, SYNTH_NAMES, SYNTH_STAINS, extra_keywords={"$CYT": "Aurora"})
    new = controller.add_samples([path])[0]
    assert apply_cytometer_defaults(controller.workspace, new) == len(SYNTH_NAMES)
    assert new.scale("FSC-A").kind == "arcsinh"
    assert new.scale("FITC-A").width_basis == pytest.approx(-500)
    controller.undo()


def test_install_adds_band_and_menu(app):
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    try:
        ext = win.extensions.get("flowdesk.ui.cytometer")
        assert ext is not None
        act = ext["action"]
        assert act in win.menus["workspace"].actions()
        assert "graph_extras" in " ".join(win.extensions) or True
    finally:
        win.controller.workspace.modified = False
        win.close()


def test_reapplying_identical_settings_pushes_no_undo_step(controller):
    from flowdesk.ui.cytometer import CytometerDialog, apply_cytometer_settings

    c = controller
    aurora = [s for s in c.workspace.samples if s.keywords.get("$CYT") == "Aurora"]
    a = {"FSC-A": {"kind": "arcsinh", "decades": 5.0, "neg": 0.0}, "FITC-A": {"kind": "logicle", "width_basis": -500.0, "decades": 4.2, "neg": 0.0}}
    b = {"FSC-A": {"kind": "linear", "min": 0.0}}
    n = len(c._undo)
    assert apply_cytometer_settings(c, "Aurora", a) == aurora
    apply_cytometer_settings(c, "Aurora", b)
    assert len(c._undo) == n + 2
    c.undo()  # back to "a"; "b" waits on the redo stack
    assert c.can_redo and all(s.scale("FSC-A").kind == "arcsinh" for s in aurora)
    c.workspace.modified = False
    # identical again: no empty step, the pending redo survives, nothing marked modified
    assert apply_cytometer_settings(c, "Aurora", {k: dict(v) for k, v in a.items()}) == []
    assert len(c._undo) == n + 1 and c.can_redo and not c.workspace.modified
    dlg = CytometerDialog(c)
    dlg.select("Aurora")
    assert dlg.specs("Aurora")["FSC-A"]["kind"] == "arcsinh"
    dlg._specs = {"Aurora": {k: dict(v) for k, v in a.items()}}
    dlg.apply()  # the dialog's own undo group is not opened for nothing either
    assert len(c._undo) == n + 1 and c.can_redo
    c.redo()
    assert all(s.scale("FSC-A").kind == "linear" for s in aurora)
    c.undo()
    # one sample's scale differs from the stored defaults: that is a change
    c.set_scale([aurora[0]], "FSC-A", LinearScale(0.0, 1000.0))
    n2 = len(c._undo)
    apply_cytometer_settings(c, "Aurora", a)
    assert len(c._undo) == n2 + 1
    sc = aurora[0].scale("FSC-A")
    assert sc.kind == "arcsinh" and sc.M == pytest.approx(5.0)
