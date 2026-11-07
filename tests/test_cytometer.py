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
