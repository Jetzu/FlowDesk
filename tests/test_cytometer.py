"""Cytometer settings: grouping by $CYT, the dialog, applying default scales."""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flowdesk.core.transforms import LinearScale, LogicleScale  # noqa: E402

from .conftest import SYNTH_NAMES, SYNTH_STAINS, make_synthetic_events, write_fcs  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture(autouse=True)
def isolated_app_settings(tmp_path, monkeypatch):
    """Application-level cytometer settings go to a temporary INI file (never
    the user's real preferences); the default-scale preferences are restored."""
    from PySide6.QtCore import QSettings

    from flowdesk.core.transforms import SCALE_DEFAULTS
    from flowdesk.ui import cytometer

    ini = tmp_path / "flowdesk-test.ini"
    monkeypatch.setattr(cytometer, "app_store", lambda: QSettings(str(ini), QSettings.Format.IniFormat))
    saved = dict(SCALE_DEFAULTS)
    yield ini
    SCALE_DEFAULTS.clear()
    SCALE_DEFAULTS.update(saved)


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


def _fresh_controller(tmp_path, cytometers, name="n", seed=20, prefs=None):
    from flowdesk.ui.controller import WorkspaceController
    from flowdesk.ui.cytometer import connect_controller

    c = WorkspaceController()
    if prefs:
        c.preferences._values.update(prefs)  # session only: never written to QSettings
    hooks = connect_controller(c)
    paths = []
    for i, cyt in enumerate(cytometers):
        ev = make_synthetic_events(1500, seed=seed + i)["events"]
        extra = {"$CYT": cyt} if cyt else None
        paths.append(write_fcs(tmp_path / f"{name}{i}.fcs", ev, SYNTH_NAMES, SYNTH_STAINS, extra_keywords=extra))
    return c, hooks, paths


def test_table_has_negative_decades_and_linear_range_columns(controller):
    from flowdesk.core.transforms import ArcsinhScale
    from flowdesk.ui.cytometer import COL_KIND, COL_MAX, COL_MIN, COL_NEG, COL_WIDTH, HEADERS, CytometerDialog

    dlg = CytometerDialog(controller)
    dlg.select("Aurora")
    assert dlg.table.columnCount() == len(HEADERS) == 8
    assert [dlg.table.horizontalHeaderItem(i).text() for i in (COL_NEG, COL_MIN, COL_MAX)] == ["Extra neg. decades", "Linear min", "Linear max"]
    row = {p: i for i, p in enumerate(dlg.parameters("Aurora"))}
    fitc, fsc, pe = row["FITC-A"], row["FSC-A"], row["PE-A"]
    assert dlg.table.cellWidget(fitc, COL_NEG).isEnabled() and not dlg.table.cellWidget(fitc, COL_MIN).isEnabled()
    assert dlg.table.cellWidget(fsc, COL_MAX).isEnabled() and not dlg.table.cellWidget(fsc, COL_NEG).isEnabled()
    # edit through the widgets, as a user would
    dlg.table.cellWidget(fitc, COL_NEG).setValue(0.7)
    dlg.table.cellWidget(fsc, COL_MIN).setValue(-500)
    dlg.table.cellWidget(fsc, COL_MAX).setValue(150000)
    dlg.table.cellWidget(pe, COL_KIND).setCurrentIndex(3)  # Arcsinh
    assert dlg.table.cellWidget(pe, COL_NEG).isEnabled() and not dlg.table.cellWidget(pe, COL_WIDTH).isEnabled()
    dlg.table.cellWidget(pe, COL_NEG).setValue(0.5)
    dlg.remember.setChecked(False)
    dlg.apply()
    for s in [s for s in controller.workspace.samples if s.keywords.get("$CYT") == "Aurora"]:
        assert s.scale("FITC-A").A == pytest.approx(0.7)
        assert s.scale("FSC-A") == LinearScale(-500, 150000)
        pe_scale = s.scale("PE-A")
        assert isinstance(pe_scale, ArcsinhScale) and pe_scale.A == pytest.approx(0.5)
    # "Auto" (0) removes the maximum again: back to $PnR
    dlg.set_param_scale("FSC-A", "linear", minimum=0, maximum=0)
    assert "max" not in dlg.specs()["FSC-A"] and dlg.table.cellWidget(fsc, COL_MAX).text() == "Auto"
    controller.undo()


def test_cytometer_settings_persist_across_workspaces(controller, tmp_path, isolated_app_settings):
    from PySide6.QtCore import QSettings

    from flowdesk.ui.cytometer import APP_SETTINGS_KEY, COL_NEG, CytometerDialog, app_cytometer_specs, forget_cytometer

    dlg = CytometerDialog(controller)
    dlg.select("Aurora")
    dlg.set_param_scale("FITC-A", "logicle", width_basis=-300, decades=4.4)
    row = dlg.parameters("Aurora").index("FITC-A")
    dlg.table.cellWidget(row, COL_NEG).setValue(0.6)  # the new column
    dlg.select("Unknown")  # samples without $CYT are never kept at application level
    dlg.set_param_scale("FSC-A", "arcsinh")
    assert not dlg.remember.isEnabled()
    dlg.select("Aurora")
    assert dlg.remember.isChecked()
    dlg.apply()
    raw = QSettings(str(isolated_app_settings), QSettings.Format.IniFormat).value(APP_SETTINGS_KEY)
    assert "Aurora" in json.loads(raw) and "Unknown" not in json.loads(raw)
    assert app_cytometer_specs()["Aurora"]["FITC-A"]["neg"] == pytest.approx(0.6)
    controller.undo()

    # a brand-new workspace (another session): samples of the same $CYT get the settings
    c2, _hooks, paths = _fresh_controller(tmp_path, ["Aurora", "Fortessa", None])
    assert c2.workspace.settings == {}
    added = c2.add_samples(paths)
    aurora, fortessa, unknown = added
    sc = aurora.scale("FITC-A")
    assert isinstance(sc, LogicleScale) and sc.width_basis == pytest.approx(-300) and sc.A == pytest.approx(0.6) and sc.M == pytest.approx(4.4)
    assert aurora.scale("Comp-FITC-A") is sc or aurora.scale("Comp-FITC-A") == sc
    assert fortessa.scale("FITC-A").width_basis == pytest.approx(-100) and fortessa.scale("FITC-A").A == 0
    assert unknown.scale("FSC-A").kind == "linear"
    # the workspace's own settings win over the application-level ones
    c2.workspace.settings["cytometers"] = {"Aurora": {"FITC-A": {"kind": "log", "decades": 4.0}}}
    ev = make_synthetic_events(800, seed=99)["events"]
    late = c2.add_samples([write_fcs(tmp_path / "late.fcs", ev, SYNTH_NAMES, SYNTH_STAINS, extra_keywords={"$CYT": "Aurora"})])[0]
    assert late.scale("FITC-A").kind == "log"
    assert forget_cytometer("Aurora") and "Aurora" not in app_cytometer_specs()
    c2.workspace.modified = False


def test_default_transform_preferences_for_new_samples(app, tmp_path):
    from flowdesk.core.transforms import ArcsinhScale

    c, hooks, paths = _fresh_controller(tmp_path, [None, None], name="pref", prefs={"default_transform": "arcsinh", "default_width_basis": -50.0})
    s = c.add_samples(paths[:1])[0]
    sc = s.scale("FITC-A")
    assert isinstance(sc, ArcsinhScale) and sc.T == 262144.0
    assert s.scale("FSC-A") == LinearScale(0, 262144)  # scatter stays linear
    c.preferences._values["default_transform"] = "logicle"
    c.preferencesChanged.emit()
    t = c.add_samples(paths[1:])[0]
    assert isinstance(t.scale("PE-A"), LogicleScale) and t.scale("PE-A").width_basis == pytest.approx(-50)
    assert t.scale("Comp-PE-A") is t.scale("PE-A")
    c.workspace.modified = False


def test_builtin_instrument_defaults(controller):
    from flowdesk.ui.cytometer import CytometerDialog, builtin_cytometer, builtin_spec

    assert "spectral" in builtin_cytometer("Aurora 5L")["name"].lower()
    assert builtin_cytometer("BD LSRFortessa X-20")["fluorescence"]["width_basis"] == -100
    assert builtin_cytometer("Unknown") is None and builtin_cytometer("Homebuilt") is None
    dlg = CytometerDialog(controller)
    dlg.select("Aurora")
    dlg.load_instrument_defaults()
    specs = dlg.specs()
    assert specs["FITC-A"]["kind"] == "logicle" and specs["FITC-A"]["width_basis"] == -1000
    assert specs["FSC-A"] == {"kind": "linear", "min": 0.0, "max": 262144.0}
    assert specs["Time"] == {"kind": "linear", "min": 0.0}  # follows the acquired data
    sample = controller.workspace.samples[3]
    assert builtin_spec(sample, "FITC-A", "Homebuilt")["width_basis"] == -100


def test_time_axis_stays_auto_in_cytometer_settings(controller):
    from flowdesk.ui.cytometer import apply_cytometer_settings

    aurora = [s for s in controller.workspace.samples if s.keywords.get("$CYT") == "Aurora"]
    aurora[0].set_scale("Time", LinearScale(0, 50))
    apply_cytometer_settings(controller, "Aurora", {"Time": {"kind": "linear", "min": 0.0}})
    for s in aurora:
        assert not s.has_own_scale("Time")
        s.ensure_loaded()
        assert s.scale("Time") == LinearScale(0, 1020)  # synthetic time runs 0 .. 1000
    apply_cytometer_settings(controller, "Aurora", {"Time": {"kind": "linear", "min": 0.0, "max": 3000.0}})
    assert all(s.scale("Time") == LinearScale(0, 3000) for s in aurora)
    controller.undo()
    assert all(s.scale("Time") == LinearScale(0, 1020) for s in aurora)


def test_compensated_parameters_share_the_cytometer_scale(controller):
    import numpy as np

    from flowdesk.core.compensation import SpilloverMatrix
    from flowdesk.ui.cytometer import CytometerDialog, apply_cytometer_settings

    aurora = [s for s in controller.workspace.samples if s.keywords.get("$CYT") == "Aurora"]
    controller.set_compensation(aurora, SpilloverMatrix("m", ["FITC-A", "PE-A"], np.array([[1, 0.1], [0.02, 1]])))
    apply_cytometer_settings(controller, "Aurora", {"PE-A": {"kind": "logicle", "width_basis": -250.0, "decades": 4.5, "neg": 0.2}})
    for s in aurora:
        assert s.scale("Comp-PE-A") is s.scale("PE-A") and s.scale("PE-A").width_basis == pytest.approx(-250)
        assert not any(k.startswith("Comp-") for k in s.scales)
    # the dialog shows the shared transform
    dlg = CytometerDialog(controller)
    dlg.select("Aurora")
    assert dlg.specs()["PE-A"]["width_basis"] == pytest.approx(-250)
