"""Data Cleaning and MESF Calibration dialogs on synthetic FCS files (offscreen)."""
import csv
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from flowdesk.core import calibration as cal
from flowdesk.core import qc
from flowdesk.core.gates import BooleanGate, RectangleGate
from flowdesk.core.model import Population, Workspace
from flowdesk.ui.controller import WorkspaceController
from flowdesk.ui.ribbon import RibbonPage
from flowdesk.ui.tools_qc import CalibrationDialog, DataCleaningDialog, QCTools, install

from .test_calibration import KIT_MESF, SLOPE, INTERCEPT, write_bead_fcs
from .test_qc import write_qc_fcs


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def controller(qapp, tmp_path):
    ws = Workspace()
    truth = {}
    for i in range(3):
        path, bad, sat = write_qc_fcs(tmp_path / f"run_{i}.fcs", seed=20 + i, saturated=30)
        s = ws.add_sample(path)
        assert s.ensure_loaded()
        s.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [0, 0], [None, None])))
        truth[s.id] = (bad, sat)
    beads = ws.add_sample(write_bead_fcs(tmp_path / "beads.fcs"))
    assert beads.ensure_loaded()
    c = WorkspaceController(ws)
    s0 = ws.samples[0]
    c.set_selection([(s0, s0.root.find("Cells"))])
    c.truth = truth
    return c


def _table(dlg):
    t = dlg.table
    header = [t.horizontalHeaderItem(i).text() for i in range(t.columnCount())]
    rows = [[t.item(r, c).text() if t.item(r, c) else "" for c in range(t.columnCount())] for r in range(t.rowCount())]
    return header, rows


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def _qc_group(c):
    """A group holding the three runs (not the beads)."""
    ws = c.workspace
    g = c.create_group("Runs", ws.samples[:3])
    c.set_current_group(g)
    return g


# ------------------------------------------------------------------ cleaning
def test_cleaning_dialog_defaults_and_run(controller, tmp_path):
    dlg = DataCleaningDialog(controller)
    assert dlg.sample_combo.currentData() == controller.workspace.samples[0].id
    assert dlg.pop_combo.currentData() == "/Cells"
    assert dlg.time_param() == "Time"
    assert dlg.selected_params() == ["FSC-A", "SSC-A", "FITC-A", "PE-A"]
    assert dlg.checks() == ("rate", "signal", "range")
    r = dlg.run()
    assert r is not None
    bad, sat = controller.truth[controller.workspace.samples[0].id]
    truly_bad = bad | sat
    assert r.good[~truly_bad].mean() >= 0.95 and (~r.good[truly_bad]).mean() >= 0.90
    header, rows = _table(dlg)
    assert header == qc.TABLE_HEADER
    assert [row[0] for row in rows[:4]] == ["Flow rate", "Signal stability", "Dynamic range", "Total"]
    assert len(rows) == 4 + 100
    assert any("rate" in row[5] for row in rows[4:]) and any("signal:FITC-A" in row[5] for row in rows[4:])
    assert "events kept" in dlg.status.text()
    dlg.plot.resize(600, 360)
    dlg.plot.grab()  # paints without error
    out = dlg.export_csv(str(tmp_path / "qc.csv"))
    data = _read_csv(out)
    assert data[0] == qc.TABLE_HEADER and len(data) == 1 + 4 + 100
    assert float(data[4][6]) == pytest.approx(r.percent_removed["total"], abs=0.01)
    dlg.copy_results()
    assert QGuiApplication.clipboard().text().startswith("Bin\tt start")

    # settings reach the core
    dlg.bins_spin.setValue(50)
    dlg.range_check.setChecked(False)
    dlg.set_params(["FITC-A"])
    r2 = dlg.run()
    assert r2.n_bins == 50 and r2.checks == ("rate", "signal") and r2.params == ["FITC-A"]
    assert "range" not in r2.percent_removed
    dlg.close()


def test_create_flag_population(controller):
    c = controller
    s0 = c.workspace.samples[0]
    cells = s0.root.find("Cells")
    dlg = DataCleaningDialog(c)
    r = dlg.run()
    events = []
    c.gatingChanged.connect(lambda sid: events.append(sid))
    created = dlg.apply()
    assert created is not None
    pop = created[0]
    assert pop.name == "QC pass" and pop.parent is cells
    assert isinstance(pop.gate, RectangleGate) and pop.gate.dims == ["QC_flag"]
    assert pop.count == r.n_good
    assert "QC_flag" in s0.parameter_names()
    assert events == [s0.id] and c.workspace.modified
    assert c.can_undo
    # the dialog kept its sample / population after the samplesChanged refresh
    assert dlg.pop_combo.currentData() == "/Cells" and dlg.current_sample() is s0
    # undo removes the population again
    c.undo()
    assert c.workspace.samples[0].root.find("Cells/QC pass") is None
    dlg.close()


def test_create_time_gate_population(controller):
    c = controller
    s0 = c.workspace.samples[0]
    dlg = DataCleaningDialog(c)
    dlg.set_mode("time")
    dlg.name_edit.setText("Stable")
    r = dlg.run()
    created = dlg.apply()
    assert created is not None and created[0].name == "Stable"
    union, parts = created[0], created[1:]
    assert isinstance(union.gate, BooleanGate) and len(parts) == len(r.intervals) == 3
    assert [p.name for p in parts] == ["Stable interval 1", "Stable interval 2", "Stable interval 3"]
    assert union.count == int(r.time_good.sum())
    assert s0.root.find("Cells/Stable") is union
    # without a time parameter the time mode is not offered
    dlg.time_combo.setCurrentIndex(dlg.time_combo.findData(None))
    assert dlg.mode() == "flag" and not dlg.rate_check.isEnabled()
    r2 = dlg.run()
    assert r2.checks == ("signal", "range")
    dlg.close()


def test_apply_to_all_samples_in_group(controller, tmp_path):
    c = controller
    _qc_group(c)
    dlg = DataCleaningDialog(c)
    undo_before = len(c._undo)
    rows = dlg.apply_to_group()
    assert len(rows) == 3
    assert len(c._undo) == undo_before + 1  # one undo step for the whole group
    header, table_rows = _table(dlg)
    assert header == qc.GROUP_HEADER
    for s, row in zip(c.workspace.samples[:3], rows):
        pop = s.root.find("Cells/QC pass")
        assert pop is not None and pop.count == row[3]
        bad, sat = c.truth[s.id]
        flag = s.data("QC_flag")
        good = flag == 1
        truly_bad = bad | sat
        assert good[~truly_bad].mean() >= 0.95 and (~good[truly_bad]).mean() >= 0.90
        assert 25 < row[4] < 35  # ~ 30 % of the run is bad
    out = dlg.export_csv(str(tmp_path / "group.csv"))
    data = _read_csv(out)
    assert data[0] == qc.GROUP_HEADER and [r[0] for r in data[1:]] == ["run_0.fcs", "run_1.fcs", "run_2.fcs"]
    # a sample without the population gets an error row
    s2 = c.workspace.samples[2]
    s2.root.remove_child(s2.root.find("Cells"))
    rows = dlg.apply_to_group()
    assert rows[2][-1].startswith("no population") and "failed" in dlg.status.text()
    dlg.close()


# --------------------------------------------------------------- calibration
def _beads_dialog(c):
    beads = c.workspace.samples[3]
    c.set_selection([(beads, beads.root)])
    dlg = CalibrationDialog(c)
    assert dlg.current_sample() is beads
    dlg.set_param(dlg.param_combo, "FITC-A")
    return dlg, beads


def test_calibration_dialog_fit_and_apply(controller, tmp_path):
    c = controller
    dlg, beads = _beads_dialog(c)
    assert dlg.param() == "FITC-A"  # default: first non-scatter channel
    dlg.peaks_spin.setValue(7)
    peaks = dlg.run()
    assert peaks is not None and len(peaks) == 7
    header, rows = _table(dlg)
    assert header == ["Peak", "Position", "Median", "rCV %", "Events", "MESF", "Used"]
    assert len(rows) == 7 and all(r[5] == "" for r in rows)  # the user enters the kit values
    assert dlg.intensities() == pytest.approx([p.median for p in peaks], rel=1e-3)
    assert dlg.fit() is None and "Could not fit" in dlg.status.text()
    dlg.set_mesf_values([0.0] + KIT_MESF)
    fit = dlg.fit()
    assert fit is not None
    assert abs(fit.slope / SLOPE - 1) < 0.02 and abs(fit.intercept / INTERCEPT - 1) < 0.02
    assert fit.used[0] is False  # blank (MESF 0)
    assert dlg.r2_label.text().startswith("r² ") and float(dlg.r2_label.text().split()[1]) > 0.999
    dlg.plot.resize(700, 320)
    dlg.plot.grab()
    out = dlg.export_csv(str(tmp_path / "peaks.csv"))
    data = _read_csv(out)
    assert data[0] == header and len(data) == 8 and float(data[7][5]) == KIT_MESF[-1]

    done = dlg.apply("all")
    assert len(done) == 4  # every sample has FITC-A
    for s in c.workspace.samples:
        assert "FITC-A MESF" in s.parameter_names()
        raw = s.data("FITC-A").astype(float)
        mesf = s.data("FITC-A MESF").astype(float)
        order = np.argsort(raw)
        assert np.all(np.diff(mesf[order]) >= 0)
    stored = c.workspace.settings["calibrations"]["FITC-A"]
    assert stored["slope"] == pytest.approx(fit.slope)
    # the MESF values of the bead peaks reproduce the kit
    mesf_beads = beads.data("FITC-A MESF").astype(float)
    raw_beads = beads.data("FITC-A").astype(float)
    for p, kit in zip(peaks[1:], KIT_MESF):
        sel = (raw_beads >= beads.scale("FITC-A").inverse_scalar(p.lo)) & (raw_beads < beads.scale("FITC-A").inverse_scalar(p.hi))
        assert abs(np.median(mesf_beads[sel]) / kit - 1) < 0.02
    # undo removes the calibrated parameters and the stored calibration
    c.undo()
    assert "FITC-A MESF" not in c.workspace.samples[0].parameter_names()
    assert "FITC-A" not in c.workspace.settings.get("calibrations", {})
    dlg.close()


def test_calibration_targets_save_load(controller, tmp_path):
    c = controller
    g = _qc_group(c)
    dlg, beads = _beads_dialog(c)
    dlg.peaks_spin.setValue(7)
    dlg.blank_check.setChecked(True)
    dlg.run()
    assert dlg.used()[0] is False  # blank unticked
    dlg.set_mesf_values([None] + KIT_MESF)
    fit = dlg.fit()
    assert fit is not None and fit.used == [False] + [True] * 6
    # the current group: the three runs, not the beads
    done = dlg.apply()
    assert {s.name for s, _ in done} == {"run_0.fcs", "run_1.fcs", "run_2.fcs"}
    assert "FITC-A MESF" not in beads.parameter_names()
    done = dlg.apply(g)
    assert len(done) == 3
    path = dlg.save_calibration(str(tmp_path / "fitc.json"))
    d = json.loads(open(path).read())
    assert d["param"] == "FITC-A" and d["slope"] == pytest.approx(fit.slope)

    dlg2 = CalibrationDialog(c)
    loaded = dlg2.load_calibration(path)
    assert loaded is not None and dlg2.param() == "FITC-A"
    assert dlg2.mesf_values()[1:] == KIT_MESF and dlg2.used()[0] is False
    assert dlg2.fit_result.slope == pytest.approx(fit.slope)
    dlg2.plot.resize(600, 300)
    dlg2.plot.grab()
    done = dlg2.apply([beads])
    assert done and done[0][1] == "FITC-A MESF"
    # a re-run on the beads pre-fills the kit values stored in the workspace
    dlg3, _ = _beads_dialog(c)
    dlg3.peaks_spin.setValue(7)
    dlg3.run()
    assert dlg3.mesf_values()[1:] == KIT_MESF
    assert dlg2.load_calibration(str(tmp_path / "missing.json")) is None
    for d in (dlg, dlg2, dlg3):
        d.close()


# ------------------------------------------------------------------- install
class _FakeWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.menus = {"tools": QMenu("Tools", self)}
        self.ribbon_pages = {"tools": RibbonPage(self)}


def test_install_adds_band_and_menu(controller):
    win = _FakeWindow(controller)
    ext = install(win)
    assert isinstance(ext, QCTools)
    texts = [a.text() for a in win.menus["tools"].actions() if not a.isSeparator()]
    assert texts == ["Data Cleaning…", "Calibrate (MESF)…"]
    assert ext.band is not None and ext.band.title == "Quality Control"
    dlg = ext.open("cleaning")
    assert isinstance(dlg, DataCleaningDialog) and ext.open("cleaning") is dlg
    ext.actions["calibration"].trigger()
    assert isinstance(ext.dialogs["calibration"], CalibrationDialog)
    for d in ext.dialogs.values():
        d.close()


def test_install_without_ribbon(controller):
    win = QWidget()
    win.controller = controller
    ext = install(win)
    assert ext.band is None and ext.menu is None and set(ext.actions) == {"cleaning", "calibration"}
