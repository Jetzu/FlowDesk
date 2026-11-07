"""Data Cleaning and MESF Calibration dialogs on synthetic FCS files (offscreen)."""
import csv
import json
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QPushButton, QWidget

from flowdesk.core import calibration as cal
from flowdesk.core import qc
from flowdesk.core.gates import BooleanGate, RectangleGate
from flowdesk.core.model import Population, Workspace
from flowdesk.ui.controller import WorkspaceController
from flowdesk.ui.ribbon import RibbonPage
from flowdesk.ui.tools_qc import CalibrationDialog, DataCleaningDialog, QCTools, install

from .test_calibration import KIT_MESF, SLOPE, INTERCEPT, peak_intensities, write_bead_fcs
from .test_qc import DROP, SHIFT, write_qc_fcs

QC_PARAMS = ["FSC-A", "SSC-A", "FITC-A", "PE-A"]


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


def _wait(app, cond, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def _inside(edges, a, b):
    """Bins lying wholly inside [a, b]."""
    return (edges[:-1] >= a) & (edges[1:] <= b)


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


def test_cleaning_settings_reach_the_core(controller):
    c = controller
    s0 = c.workspace.samples[0]
    cells = s0.root.find("Cells")
    # a computed parameter (no $PnR limit) with known pile-ups: 5 events on
    # its maximum, 8 on its minimum
    rng = np.random.default_rng(11)
    n = s0.n_events
    piled = rng.normal(1000.0, 50.0, n)
    idx = rng.choice(n, 13, replace=False)
    piled[idx[:5]] = piled.max() + 10.0
    piled[idx[5:]] = piled.min() - 10.0
    s0.add_computed("Piled", piled.astype(np.float32), {"tool": "test"})
    in_cells = cells.mask()
    n_top, n_bottom = int(in_cells[idx[:5]].sum()), int(in_cells[idx[5:]].sum())
    assert (n_top, n_bottom) == (5, 8)
    dlg = DataCleaningDialog(c)
    time_ = s0.data("Time")[in_cells]

    # flow rate: the 20 % drop counts with the default 10 % minimum change, not with 60 %
    dlg.set_params(QC_PARAMS)
    r = dlg.run()
    drop = _inside(r.edges, *DROP)
    assert r.settings["min_rate_change"] == pytest.approx(0.10) and r.bin_flags["rate"][drop].all()
    dlg.rate_min.setValue(60)
    dlg.rate_z.setValue(5)
    r = dlg.run()
    assert r.settings["min_rate_change"] == pytest.approx(0.60) and r.settings["rate_z"] == 5.0
    assert not r.bin_flags["rate"].any()

    # signal: the FITC shift (measured here from the display values) is above
    # the default 2 % minimum shift and below 10 %
    disp = s0.display("FITC-A")[in_cells]
    in_shift = (time_ >= SHIFT[0]) & (time_ < SHIFT[1])
    calm = ~in_shift & ~((time_ >= DROP[0]) & (time_ < DROP[1]))
    shift = float(np.median(disp[in_shift]) - np.median(disp[calm]))
    assert 0.02 < shift < 0.10
    assert r.bin_flags["signal:FITC-A"][_inside(r.edges, *SHIFT)].all()  # default min shift 2 %
    dlg.signal_min.setValue(10)
    dlg.signal_z.setValue(2)
    dlg.cp_check.setChecked(False)
    r = dlg.run()
    assert r.settings["min_shift"] == pytest.approx(0.10) and r.settings["signal_z"] == 2.0 and r.settings["z"] == 2.0
    assert r.settings["changepoints"] is False
    assert not r.bin_flags["signal:FITC-A"].any()

    # dynamic range: min pile-up and the minimum side on the computed parameter
    dlg.rate_check.setChecked(False)
    dlg.signal_check.setChecked(False)
    dlg.set_params(["Piled"])
    expected = {(3, False): n_top, (10, False): 0, (3, True): n_top + n_bottom, (6, True): n_bottom}
    for (pile, low), count in expected.items():
        dlg.pile_spin.setValue(pile)
        dlg.low_check.setChecked(low)
        r = dlg.run()
        assert r.checks == ("range",) and r.settings["min_pile"] == pile and r.settings["check_low"] is low
        assert r.range_counts["Piled"] == count
    dlg.close()


def test_create_uses_current_settings_not_last_run(controller):
    c = controller
    cells = c.workspace.samples[0].root.find("Cells")
    dlg = DataCleaningDialog(c)
    r1 = dlg.run()
    created = dlg.apply()
    assert dlg.result is r1 and created[0].count == r1.n_good  # unchanged settings: the run is reused
    dlg.bins_spin.setValue(40)
    dlg.rate_min.setValue(60)
    dlg.range_check.setChecked(False)
    created = dlg.apply()  # without pressing Run again
    used = dlg.result
    assert used is not r1 and dlg.plot.result is used
    assert used.n_bins == 40 and used.settings["min_rate_change"] == pytest.approx(0.6) and used.checks == ("rate", "signal")
    fresh = qc.qc_for_population(cells, "Time", QC_PARAMS, checks=("rate", "signal"), n_bins=40, min_rate_change=0.6)
    assert created[0].count == fresh.n_good and fresh.n_good != r1.n_good
    dlg.close()


def test_create_recomputes_after_gate_change(controller):
    c = controller
    s0 = c.workspace.samples[0]
    cells = s0.root.find("Cells")
    dlg = DataCleaningDialog(c)
    r1 = dlg.run()
    cells.gate = RectangleGate(["FSC-A", "SSC-A"], [80000, 0], [None, None])  # about half of the events
    s0.root.invalidate()
    assert cells.count < r1.n_events
    created = dlg.apply()
    fresh = qc.qc_for_population(cells, "Time", QC_PARAMS)
    assert dlg.result is not r1 and dlg.result.n_events == cells.count
    assert created[0].count == fresh.n_good
    dlg.close()


def test_create_recomputes_after_scale_or_compensation_change(controller):
    c = controller
    s0, s1 = c.workspace.samples[:2]
    dlg = DataCleaningDialog(c)
    r1 = dlg.run()
    c.scalesChanged.emit(s1.id)  # another sample: the run still holds
    dlg.apply()
    assert dlg.result is r1
    c.scalesChanged.emit(s0.id)  # display values of this sample may differ now
    dlg.apply()
    r2 = dlg.result
    assert r2 is not r1 and dlg.plot.result is r2
    dlg.apply()
    assert dlg.result is r2
    c.compensationChanged.emit(None)  # every sample
    dlg.apply()
    assert dlg.result is not r2
    dlg.close()


def test_cleaning_result_dropped_when_workspace_replaced(controller, tmp_path):
    c = controller
    dlg = DataCleaningDialog(c)
    old = dlg.run()
    old_id = c.workspace.samples[0].id
    # a new workspace whose first sample has the same id but other events
    ws2 = Workspace()
    path, _, _ = write_qc_fcs(tmp_path / "donor0.fcs", seed=99, rate=20.0)
    s_new = ws2.add_sample(path)
    assert s_new.ensure_loaded() and s_new.id == old_id and s_new.n_events != old.n_events
    cells = s_new.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [0, 0], [None, None])))
    c.set_workspace(ws2)
    assert dlg.current_sample() is s_new
    assert dlg.result is None and dlg.plot.result is None and dlg.table.rowCount() == 0
    dlg.set_population(s_new, cells)
    created = dlg.apply()
    fresh = qc.qc_for_population(cells, "Time", QC_PARAMS)
    assert dlg.result.n_events == cells.count == s_new.n_events
    assert created[0].count == fresh.n_good
    flag = s_new.data("QC_flag")
    assert flag.shape == (s_new.n_events,) and int(np.nansum(flag)) == fresh.n_good
    assert f"of {s_new.n_events} events" in dlg.status.text()
    dlg.close()


def test_group_apply_runs_in_worker_with_progress(controller, qapp):
    c = controller
    _qc_group(c)
    dlg = DataCleaningDialog(c)
    labels = []
    show = dlg._job_progress
    # progress reaches the dialog in the GUI thread (queued from the worker)
    dlg._job_progress = lambda value, text: (labels.append((text, threading.current_thread() is threading.main_thread())), show(value, text))
    undo_before = len(c._undo)
    dlg.group_button.click()  # the button starts the threaded version
    assert dlg.busy() and not dlg.group_button.isEnabled() and not dlg.create_button.isEnabled()
    assert dlg._progress is not None and dlg._progress.maximum() == 3
    assert dlg.start_group() is False  # one job at a time
    assert _wait(qapp, lambda: not dlg.busy())
    assert dlg.group_button.isEnabled() and dlg._progress is None
    assert [t for t, _ in labels if t.startswith("Sample")] == [f"Sample {i + 1} of 3: run_{i}.fcs" for i in range(3)]
    assert labels and all(main for _, main in labels)
    assert len(c._undo) == undo_before + 1  # one undo step for the whole group
    header, rows = _table(dlg)
    assert header == qc.GROUP_HEADER and [r[0] for r in rows] == ["run_0.fcs", "run_1.fcs", "run_2.fcs"]
    for s in c.workspace.samples[:3]:
        cells = s.root.find("Cells")
        assert s.root.find("Cells/QC pass").count == qc.qc_for_population(cells, "Time", QC_PARAMS).n_good
    c.undo()
    assert all(s.root.find("Cells/QC pass") is None for s in c.workspace.samples[:3])
    dlg.close()


def test_group_apply_cancel_changes_nothing(controller, qapp):
    c = controller
    _qc_group(c)
    dlg = DataCleaningDialog(c)
    undo_before = len(c._undo)
    assert dlg.start_group()
    dlg._progress.findChild(QPushButton).click()  # the progress dialog's Cancel
    assert _wait(qapp, lambda: not dlg.busy())
    assert len(c._undo) == undo_before
    for s in c.workspace.samples[:3]:
        assert s.root.find("Cells/QC pass") is None and "QC_flag" not in s.parameter_names()
    assert "Cancelled" in dlg.status.text() and dlg.group_button.isEnabled()
    dlg.close()


# --------------------------------------------------------------- calibration
def _combo_data(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def _beads_dialog(c):
    beads = c.workspace.samples[3]
    c.set_selection([(beads, beads.root)])
    dlg = CalibrationDialog(c)
    assert dlg.current_sample() is beads
    # nothing forced the parameter: the default skips scatter (and any
    # calibrated "FITC-A MESF" created by an earlier apply)
    assert dlg.param() == "FITC-A"
    assert "FITC-A MESF" not in _combo_data(dlg.param_combo)
    return dlg, beads


def test_calibration_default_parameter_skips_scatter_and_calibrated(controller):
    c = controller
    beads = c.workspace.samples[3]
    assert beads.parameter_names()[:3] == ["FSC-A", "SSC-A", "FITC-A"]
    c.set_selection([(beads, beads.root)])
    dlg = CalibrationDialog(c)
    assert _combo_data(dlg.param_combo) == ["FSC-A", "SSC-A", "FITC-A"]
    assert dlg.param() == "FITC-A"  # not the first parameter (FSC-A)
    fit = cal.fit_calibration(peak_intensities(), KIT_MESF, param="FITC-A")
    assert cal.apply_calibration(beads, fit) == "FITC-A MESF"
    c.samplesChanged.emit()  # the open dialog refreshes its combos
    assert "FITC-A MESF" not in _combo_data(dlg.param_combo) and dlg.param() == "FITC-A"
    dlg2 = CalibrationDialog(c)  # a fresh dialog has no current choice to keep
    assert _combo_data(dlg2.param_combo) == ["FSC-A", "SSC-A", "FITC-A"] and dlg2.param() == "FITC-A"
    for d in (dlg, dlg2):
        d.close()


def test_calibration_dialog_fit_and_apply(controller, tmp_path):
    c = controller
    dlg, beads = _beads_dialog(c)
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


def test_calibration_apply_runs_in_worker(controller, qapp):
    c = controller
    dlg, beads = _beads_dialog(c)
    dlg.peaks_spin.setValue(7)
    dlg.run()
    dlg.set_mesf_values([0.0] + KIT_MESF)
    fit = dlg.fit()
    assert fit is not None
    undo_before = len(c._undo)
    assert dlg.start_apply("all")
    assert dlg.busy() and not dlg.apply_button.isEnabled() and dlg._progress.maximum() == 4
    assert _wait(qapp, lambda: not dlg.busy())
    assert dlg.apply_button.isEnabled() and "on 4 of 4 sample(s)" in dlg.status.text()
    assert len(c._undo) == undo_before + 1
    for s in c.workspace.samples:
        raw = s.data("FITC-A").astype(float)
        pos = raw > 1.0
        expected = 10.0 ** (fit.slope * np.log10(raw[pos]) + fit.intercept)
        assert np.allclose(s.data("FITC-A MESF").astype(float)[pos], expected, rtol=1e-4)
    assert c.workspace.settings["calibrations"]["FITC-A"]["slope"] == pytest.approx(fit.slope)
    c.undo()
    assert all("FITC-A MESF" not in s.parameter_names() for s in c.workspace.samples)
    # Cancel (started from the button): no parameter, no undo step
    dlg.apply_button.click()
    assert dlg.busy()
    dlg._progress.findChild(QPushButton).click()
    assert _wait(qapp, lambda: not dlg.busy())
    assert all("FITC-A MESF" not in s.parameter_names() for s in c.workspace.samples)
    assert len(c._undo) == undo_before and "Cancelled" in dlg.status.text()
    dlg.close()


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
