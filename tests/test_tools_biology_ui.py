"""Biology tool dialogs on a workspace of synthetic FCS files (offscreen)."""
import csv
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMenu, QWidget

from flowdesk.core.fcs import Channel, write_fcs
from flowdesk.core.gates import RectangleGate
from flowdesk.core.model import Population, Workspace
from flowdesk.ui.controller import WorkspaceController
from flowdesk.ui.ribbon import RibbonPage
from flowdesk.ui.tools_biology import CellCycleDialog, KineticsDialog, ProliferationDialog, install

from .test_biology import GEN_FRACTIONS, synth_cell_cycle, synth_proliferation

N = 12000


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_sample(path, seed):
    rng = np.random.default_rng(seed)
    dna = synth_cell_cycle(n=N, g1=50000, seed=seed)
    cfse = synth_proliferation(n=N, seed=seed)
    rng.shuffle(dna)
    fsc = rng.normal(80000, 5000, N)
    ssc = rng.normal(40000, 4000, N)
    time = np.sort(rng.uniform(0, 500, N))
    events = np.column_stack([fsc, ssc, dna, cfse, time])
    channels = [
        Channel(1, "FSC-A"),
        Channel(2, "SSC-A"),
        Channel(3, "PE-A", "PI"),
        Channel(4, "FITC-A", "CFSE"),
        Channel(5, "Time", range=500),
    ]
    write_fcs(path, events, channels, extra_keywords={"P3DISPLAY": "LIN"})
    return path


@pytest.fixture
def controller(qapp, tmp_path):
    ws = Workspace()
    for i in range(3):
        s = ws.add_sample(_write_sample(tmp_path / f"sample_{i}.fcs", seed=i))
        s.ensure_loaded()
        s.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [0, 0], [1e9, 1e9])))
    c = WorkspaceController(ws)
    s0 = ws.samples[0]
    c.set_selection([(s0, s0.root.find("Cells"))])
    return c


def _table(dlg):
    t = dlg.table
    header = [t.horizontalHeaderItem(i).text() for i in range(t.columnCount())]
    rows = [[t.item(r, c).text() if t.item(r, c) else "" for c in range(t.columnCount())] for r in range(t.rowCount())]
    return header, rows


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


def test_cell_cycle_dialog(controller, tmp_path):
    dlg = CellCycleDialog(controller)
    assert dlg.param_combo.currentData() == "PE-A"  # stain "PI", linear display
    assert dlg.pop_combo.currentData() == "/Cells"
    r = dlg.run()
    assert r is not None
    assert abs(r.percent_g1 - 60) < 3 and abs(r.percent_s - 15) < 3 and abs(r.percent_g2m - 25) < 3
    header, rows = _table(dlg)
    assert header[:6] == ["Sample", "Population", "Parameter", "%G1", "%S", "%G2/M"]
    assert len(rows) == 1 and rows[0][0] == "sample_0.fcs" and rows[0][1] == "Cells"
    assert dlg.plot.mode == "hist" and len(dlg.plot.curves) == 4
    dlg.plot.resize(500, 300)
    dlg.plot.grab()  # paints without error

    group_rows = dlg.apply_to_group()
    assert len(group_rows) == 3
    header, rows = _table(dlg)
    assert [r[0] for r in rows] == ["sample_0.fcs", "sample_1.fcs", "sample_2.fcs"]
    for row in rows:
        assert abs(float(row[3]) - 60) < 3
    out = dlg.export_csv(str(tmp_path / "cc.csv"))
    data = _read_csv(out)
    assert data[0] == dlg.header and len(data) == 4
    assert abs(float(data[2][header.index("%G2/M")]) - 25) < 3
    dlg.copy_results()
    assert QGuiApplication.clipboard().text().startswith("Sample\tPopulation")


def test_proliferation_dialog(controller, tmp_path):
    dlg = ProliferationDialog(controller)
    assert dlg.param_combo.currentData() == "FITC-A"  # stain CFSE
    dlg.gen_spin.setValue(8)
    assert dlg.log_check.isChecked()
    r = dlg.run()
    assert r is not None
    pct = np.zeros(8)
    pct[: r.n_generations] = r.percents
    assert np.all(np.abs(pct[:5] - 100 * GEN_FRACTIONS) < 3)
    header, rows = _table(dlg)
    assert "%Gen 7" in header and "Division Index" in header
    peak = float(rows[0][header.index("Gen 0 Peak")])
    assert abs(peak / 50000 - 1) < 0.05
    dlg.plot.resize(500, 300)
    dlg.plot.grab()

    # raw log10 fitting gives the same answer
    dlg.log_check.setChecked(False)
    r2 = dlg.run()
    assert abs(r2.generation0 / 50000 - 1) < 0.05

    dlg.log_check.setChecked(True)
    rows = dlg.apply_to_group()
    assert len(rows) == 3
    out = dlg.export_csv(str(tmp_path / "prolif.csv"))
    data = _read_csv(out)
    assert len(data) == 4
    col = data[0].index("%Gen 2")
    assert all(abs(float(r[col]) - 30) < 3 for r in data[1:])


def test_kinetics_dialog(controller, tmp_path):
    dlg = KineticsDialog(controller)
    assert dlg.time_combo.currentData() == "Time"
    dlg.set_param(dlg.param_combo, "FSC-A")
    dlg.bins_spin.setValue(10)
    r = dlg.run()
    assert r is not None and r.centers.size == 10
    header, rows = _table(dlg)
    assert header[0] == "Bin" and len(rows) == 10
    assert sum(int(r[3]) for r in rows) == N
    medians = [float(r[2]) for r in rows]
    assert all(abs(m - 80000) < 1000 for m in medians)
    dlg.plot.resize(500, 300)
    dlg.plot.grab()

    dlg.stat_combo.setCurrentIndex(dlg.stat_combo.findData("percent_positive"))
    dlg.threshold_spin.setValue(80000)
    r = dlg.run()
    assert np.all(np.abs(r.values - 50) < 8)

    rows = dlg.apply_to_group()
    assert len(rows) == 3 and rows[0][3] == "% Positive"
    out = dlg.export_csv(str(tmp_path / "kin.csv"))
    data = _read_csv(out)
    assert data[0][:4] == ["Sample", "Population", "Parameter", "Statistic"] and len(data) == 4


def test_group_missing_population(controller):
    ws = controller.workspace
    ws.samples[2].root.remove_child(ws.samples[2].root.find("Cells"))
    dlg = CellCycleDialog(controller)
    rows = dlg.apply_to_group()
    assert len(rows) == 3
    assert rows[2][-1].startswith("no population")
    assert "failed" in dlg.status.text()


class _FakeWindow(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.menus = {"tools": QMenu("Tools", self)}
        self.ribbon_pages = {"tools": RibbonPage(self)}


def test_install_adds_band_and_menu(controller):
    win = _FakeWindow(controller)
    ext = install(win)
    texts = [a.text() for a in win.menus["tools"].actions() if not a.isSeparator()]
    assert texts == ["Cell Cycle…", "Proliferation…", "Kinetics…"]
    assert ext.band is not None
    dlg = ext.open("cell_cycle")
    assert isinstance(dlg, CellCycleDialog) and ext.open("cell_cycle") is dlg
    ext.actions["kinetics"].trigger()
    assert isinstance(ext.dialogs["kinetics"], KineticsDialog)
    for d in ext.dialogs.values():
        d.close()
