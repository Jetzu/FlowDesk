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

from .test_biology import GEN_FRACTIONS, NO_UNDIVIDED, synth_cell_cycle, synth_cell_cycle_mixed, synth_proliferation

N = 12000


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_sample(path, seed, dna=None, cfse=None):
    rng = np.random.default_rng(seed)
    dna = synth_cell_cycle(n=N, g1=50000, seed=seed) if dna is None else dna
    cfse = synth_proliferation(n=N, seed=seed) if cfse is None else cfse
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


def _workspace(qapp, tmp_path, specs):
    """specs: [(name, dna values, cfse values)] -> controller with a /Cells gate."""
    ws = Workspace()
    for i, (name, dna, cfse) in enumerate(specs):
        s = ws.add_sample(_write_sample(tmp_path / name, seed=10 + i, dna=dna, cfse=cfse))
        s.ensure_loaded()
        s.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [0, 0], [1e9, 1e9])))
    c = WorkspaceController(ws)
    s0 = ws.samples[0]
    c.set_selection([(s0, s0.root.find("Cells"))])
    return c


def test_cell_cycle_dialog_models_with_debris_and_doublets(qapp, tmp_path):
    mix = (0.45, 0.30, 0.25)
    specs = [(f"dna_{i}.fcs", synth_cell_cycle_mixed(n=N, fractions=mix, g1=50000, seed=i), synth_proliferation(n=N, seed=i)) for i in range(2)]
    c = _workspace(qapp, tmp_path, specs)
    dlg = CellCycleDialog(c)
    assert [dlg.model_combo.itemText(i) for i in range(dlg.model_combo.count())] == ["Dean-Jett-Fox", "Watson Pragmatic"]
    assert not dlg.debris_check.isChecked() and not dlg.aggregates_check.isChecked()
    for model in ("dean_jett_fox", "watson"):
        dlg.set_model(model, debris=True, aggregates=True)
        assert dlg.model_combo.currentData() == model
        r = dlg.run()
        assert r is not None and r.model == model
        assert abs(r.percent_g1 - 45) < 3 and abs(r.percent_s - 30) < 3 and abs(r.percent_g2m - 25) < 3
        assert abs(r.percent_debris - 10) < 2 and abs(r.percent_aggregates - 5) < 2
        assert [name for name, _, _ in dlg.plot.curves] == ["G1", "S", "G2/M", "Debris", "Aggregates", "Model"]
        dlg.plot.resize(520, 320)
        dlg.plot.grab()
        header, rows = _table(dlg)
        assert header[-4:] == ["%Debris", "%Aggregates", "Model", "Note"]
        assert abs(float(rows[0][header.index("%Debris")]) - 10) < 2
        assert rows[0][header.index("Model")] == r.model_name
        assert r.model_name in dlg.status.text() and "debris" in dlg.status.text()
    rows = dlg.apply_to_group()
    assert len(rows) == 2
    out = dlg.export_csv(str(tmp_path / "cc_watson.csv"))
    data = _read_csv(out)
    assert data[1][data[0].index("Model")] == "Watson Pragmatic"
    # without the extra components the columns stay empty
    dlg.set_model("Dean-Jett-Fox", debris=False, aggregates=False)
    dlg.run()
    header, rows = _table(dlg)
    assert rows[0][header.index("%Debris")] == "" and rows[0][header.index("%Aggregates")] == ""
    assert len(dlg.plot.curves) == 4
    dlg.close()


def test_proliferation_dialog_generation0_from_control(qapp, tmp_path):
    from flowdesk.core.biology import proliferation_indices

    specs = [(f"stim_{i}.fcs", synth_cell_cycle(n=N, g1=50000, seed=i), synth_proliferation(n=N, fractions=NO_UNDIVIDED, seed=i)) for i in range(2)]
    specs.append(("unstim.fcs", synth_cell_cycle(n=N, g1=50000, seed=7), synth_proliferation(n=N, fractions=[1.0], seed=7)))
    c = _workspace(qapp, tmp_path, specs)
    ws = c.workspace
    control = ws.samples[2]
    truth = proliferation_indices(NO_UNDIVIDED)
    dlg = ProliferationDialog(c)
    assert dlg.g0_mode.currentData() == "auto"
    assert not dlg.form.isRowVisible(dlg.control_sample_combo) and not dlg.form.isRowVisible(dlg.g0_spin)
    assert [dlg.control_sample_combo.itemText(i) for i in range(3)] == ["stim_0.fcs", "stim_1.fcs", "unstim.fcs"]
    auto = dlg.run()
    assert abs(auto.division_index - truth["division_index"]) > 0.5  # generation 1 taken as generation 0

    dlg.set_g0_control(control, "/Cells")
    assert dlg.form.isRowVisible(dlg.control_sample_combo) and dlg.form.isRowVisible(dlg.control_pop_combo)
    assert not dlg.form.isRowVisible(dlg.g0_spin) and dlg.fix_g0_check.isChecked()
    assert dlg.control_pop_combo.currentData() == "/Cells"
    r = dlg.run()
    assert r is not None
    assert abs(r.division_index - truth["division_index"]) < 0.1
    assert abs(r.percent_divided - 100) < 1.5
    median = float(np.median(control.data("FITC-A")))
    assert "Control median" in dlg.g0_info.text()
    header, rows = _table(dlg)
    peak = float(rows[0][header.index("Gen 0 Peak")])
    assert peak == pytest.approx(median, rel=1e-3)  # held at the control's median
    rows = dlg.apply_to_group()
    col = header.index("Division Index")
    assert all(abs(float(row[col]) - truth["division_index"]) < 0.1 for row in rows[:2])

    # a typed intensity works the same way; refining from it also finds generation 0
    dlg.set_g0_value(median, fix=False)
    assert dlg.form.isRowVisible(dlg.g0_spin) and not dlg.form.isRowVisible(dlg.control_sample_combo)
    r2 = dlg.run()
    assert abs(r2.division_index - truth["division_index"]) < 0.1
    # log10 fitting of the raw values with the control position
    dlg.log_check.setChecked(False)
    dlg.set_g0_control(control, "/Cells")
    r3 = dlg.run()
    assert abs(r3.generation0 / median - 1) < 1e-3
    # a control population that does not exist is reported, not raised
    dlg._refresh_control_populations("/Cells")
    control.root.remove_child(control.root.find("Cells"))
    dlg.control_pop_combo.addItem("gone", "/Cells")
    dlg.control_pop_combo.setCurrentIndex(dlg.control_pop_combo.count() - 1)
    assert dlg.run() is None and "control population" in dlg.status.text()
    dlg.set_g0_auto()
    assert not dlg.form.isRowVisible(dlg.fix_g0_check)
    dlg.close()


# ------------------------------------------------------ fits as statistics
FIT_STATS = ["Cell Cycle %G1", "Cell Cycle %S", "Cell Cycle %G2M", "Prolif. Division Index", "Prolif. Proliferation Index", "Prolif. % Divided"]


def test_fit_statistics_compute_and_cache(controller):
    from flowdesk.core import biology
    from flowdesk.core.biology import proliferation_indices
    from flowdesk.core.stats import NEEDS_PARAM, PERCENT_STATS, STATISTICS, compute_statistic, format_value, statistic_label

    assert all(name in STATISTICS and name in NEEDS_PARAM for name in FIT_STATS)
    assert {"Cell Cycle %G1", "Cell Cycle %S", "Cell Cycle %G2M", "Prolif. % Divided"} <= PERCENT_STATS
    biology.clear_fit_cache()
    pop = controller.workspace.samples[0].root.find("Cells")
    g1 = compute_statistic("Cell Cycle %G1", pop, "PE-A")
    s = compute_statistic("Cell Cycle %S", pop, "PE-A")
    g2 = compute_statistic("Cell Cycle %G2M", pop, "PE-A")
    assert abs(g1 - 60) < 3 and abs(s - 15) < 3 and abs(g2 - 25) < 3
    assert g1 + s + g2 == pytest.approx(100)
    assert biology.fit_cache_info()["misses"] == 1 and biology.fit_cache_info()["hits"] == 2
    truth = proliferation_indices(GEN_FRACTIONS)
    di = compute_statistic("Prolif. Division Index", pop, "FITC-A")
    pi = compute_statistic("Prolif. Proliferation Index", pop, "FITC-A")
    pdiv = compute_statistic("Prolif. % Divided", pop, "FITC-A")
    assert abs(di - truth["division_index"]) < 0.1
    assert abs(pi - truth["proliferation_index"]) < 0.1
    assert abs(pdiv - truth["percent_divided"]) < 3
    assert biology.fit_cache_info()["misses"] == 2
    # the same numbers as the dialog's default fit
    assert g1 == pytest.approx(biology.cell_cycle_for_population(pop, "PE-A").percent_g1)
    assert di == pytest.approx(biology.proliferation_for_population(pop, "FITC-A").division_index)
    # by stain name too ("PI" is the stain of PE-A)
    assert compute_statistic("Cell Cycle %G1", pop, "PI") == pytest.approx(g1)
    assert format_value("Cell Cycle %G1", g1, 2) == f"{g1:.2f}"
    assert statistic_label("Cell Cycle %G2M", "PE-A") == "Cell Cycle %G2M : PE-A"
    # a gate change changes the events: the fit runs again
    before = biology.fit_cache_info()["misses"]
    pop.gate = RectangleGate(["FSC-A"], [75000], [None])
    pop.invalidate()
    assert compute_statistic("Cell Cycle %G1", pop, "PE-A") != g1
    assert biology.fit_cache_info()["misses"] == before + 1
    # undefined cases are NaN, not errors
    assert np.isnan(compute_statistic("Cell Cycle %G1", pop, None))
    assert np.isnan(compute_statistic("Cell Cycle %G1", pop, "Nope-A"))
    tiny = pop.add_child(Population("Tiny", RectangleGate(["FSC-A"], [1e9], [None])))
    assert np.isnan(compute_statistic("Prolif. Division Index", tiny, "FITC-A"))
    assert np.isnan(compute_statistic("Prolif. Division Index", tiny, "FITC-A"))  # cached failure
    # the LRU stays bounded
    old = biology.FIT_CACHE_SIZE
    try:
        biology.FIT_CACHE_SIZE = 2
        for smp in controller.workspace.samples:
            compute_statistic("Cell Cycle %S", smp.root, "PE-A")
        assert biology.fit_cache_info()["size"] <= 2
    finally:
        biology.FIT_CACHE_SIZE = old
        biology.clear_fit_cache()


def test_fit_statistics_in_table_over_group(controller):
    from flowdesk.core import biology
    from flowdesk.core.biology import proliferation_indices
    from flowdesk.core.layouts import TableColumn, TableDef
    from flowdesk.ui.dialogs import AddStatisticDialog
    from flowdesk.ui.table_editor import ColumnDialog, build_table

    ws = controller.workspace
    biology.clear_fit_cache()
    cols = [TableColumn(pop_path="/Cells", statistic=name, param="PE-A" if name.startswith("Cell") else "FITC-A") for name in FIT_STATS]
    tdef = TableDef(name="Biology", columns=[TableColumn(pop_path="/Cells", statistic="Count")] + cols)
    headers, rows = build_table(ws, tdef, ws.groups[0])
    assert headers[2] == "Cells | Cell Cycle %G1 : PE-A" and headers[-1] == "Cells | Prolif. % Divided : FITC-A"
    assert [r[0] for r in rows] == ["sample_0.fcs", "sample_1.fcs", "sample_2.fcs"]
    truth = proliferation_indices(GEN_FRACTIONS)
    for row in rows:
        g1, s, g2, di, pi, pdiv = (float(v) for v in row[2:])
        assert abs(g1 - 60) < 3 and abs(s - 15) < 3 and abs(g2 - 25) < 3
        assert abs(di - truth["division_index"]) < 0.1 and abs(pi - truth["proliferation_index"]) < 0.1
        assert abs(pdiv - truth["percent_divided"]) < 3
    # one cell-cycle and one proliferation fit per sample, six columns
    assert biology.fit_cache_info()["misses"] == 2 * len(ws.samples)
    assert build_table(ws, tdef, ws.groups[0])[1] == rows
    assert biology.fit_cache_info()["misses"] == 2 * len(ws.samples)
    # a definition survives a round trip through its dict form
    assert build_table(ws, TableDef.from_dict(tdef.to_dict()), ws.groups[0])[1] == rows

    # offered by the Table Editor's column dialog and by Add Statistic…
    dlg = ColumnDialog(controller, ws.groups[0])
    items = [dlg.stat_combo.itemText(i) for i in range(dlg.stat_combo.count())]
    assert all(name in items for name in FIT_STATS)
    dlg.stat_combo.setCurrentText("Cell Cycle %S")
    assert dlg.param_combo.isEnabled() and not dlg.pct_spin.isEnabled()
    dlg.param_combo.setCurrentIndex(dlg.param_combo.findData("PE-A"))
    col = dlg.column()
    assert col.statistic == "Cell Cycle %S" and col.param == "PE-A"
    add = AddStatisticDialog(["PE-A", "FITC-A"])
    items = [add.stat.itemText(i) for i in range(add.stat.count())]
    assert all(name in items for name in FIT_STATS)
    add.stat.setCurrentText("Prolif. Division Index")
    assert add.values()[:2] == ("Prolif. Division Index", "PE-A")
    biology.clear_fit_cache()


def test_tick_labels_are_distinct():
    from flowdesk.ui.tools_biology import _fmt_tick, _nice_ticks

    labels = [_fmt_tick(t) for t in _nice_ticks(0, 250000)]
    assert len(labels) == len(set(labels)) and "1.5e+05" in labels
