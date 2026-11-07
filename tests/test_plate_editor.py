"""Plate Editor: well assignment, colour coding, export and workspace links."""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt  # noqa: E402

from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.model import Sample  # noqa: E402

NAMES = ["Plate1_A1.fcs", "Plate1_B2.fcs", "Specimen_001_C3.fcs", "Plate1_H12.fcs", "Control.fcs"]


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def make_files(folder, names=NAMES, keywords=None):
    rng = np.random.default_rng(7)
    paths = []
    for k, name in enumerate(names):
        n = 100 * (k + 1)
        events = np.column_stack([rng.normal(50000, 5000, n), rng.normal(30000, 4000, n)]).astype(np.float32)
        kw = keywords(k, name) if keywords else None
        p = folder / name
        write_fcs(p, events, [Channel(1, "FSC-A"), Channel(2, "SSC-A")], extra_keywords=kw)
        paths.append(p)
    return paths


@pytest.fixture
def window(app, tmp_path):
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    win.controller.add_samples(make_files(tmp_path))
    app.processEvents()
    yield win
    ext = win.extensions.get("flowdesk.ui.plate_editor")
    if ext is not None and ext.editor is not None:
        ext.editor.hide()
    for w in list(win.graph_windows.values()):
        w.close()
    win.controller.workspace.modified = False
    win.hide()
    win.deleteLater()


def editor(win):
    return win.extensions["flowdesk.ui.plate_editor"].show_editor()


def by_name(win, name) -> Sample:
    return next(s for s in win.controller.workspace.samples if s.name == name)


def test_parse_well():
    from flowdesk.ui.plate_editor import parse_well

    assert parse_well("Plate1_A1.fcs") == (0, 0)
    assert parse_well("Specimen_001_A1.fcs") == (0, 0)
    assert parse_well("H12") == (7, 11)
    assert parse_well("B02") == (1, 1)
    assert parse_well("A13") is None
    assert parse_well("A13", 16, 24) == (0, 12)
    assert parse_well("P24", 16, 24) == (15, 23)
    assert parse_well("Control.fcs") is None
    assert parse_well("x_D7", pattern=r"([A-P])(\d{1,2})") == (3, 6)


def test_install_and_band(window):
    exp = window.extensions["flowdesk.ui.experiment"]
    ext = window.extensions["flowdesk.ui.plate_editor"]
    assert ext.band is exp.band and not ext.own_band
    assert "Plate Editor" in [a.text() for a in window.menus["workspace"].actions()]


def test_assignment_from_names_and_manual(window):
    ed = editor(window)
    assert ed.isVisible()
    assert ed.wells[(0, 0)].name == "Plate1_A1.fcs"
    assert ed.wells[(1, 1)].name == "Plate1_B2.fcs"
    assert ed.wells[(2, 2)].name == "Specimen_001_C3.fcs"
    assert ed.wells[(7, 11)].name == "Plate1_H12.fcs"
    assert [s.name for s in ed.unassigned] == ["Control.fcs"]
    assert ed.unassigned_list.count() == 1
    # manual assignment: pick in the side list, click a free well
    ed.unassigned_list.setCurrentRow(0)
    assert ed.assign_combo.currentData() == by_name(window, "Control.fcs").id
    ed.click_well(0, 5)
    assert ed.wells[(0, 5)].name == "Control.fcs" and not ed.unassigned
    assert window.controller.workspace.settings["plate_editor"]["manual"]
    # 384-well format keeps the same wells
    ed.set_format(384)
    assert ed.plate.rows == 16 and ed.plate.cols == 24
    assert ed.wells[(7, 11)].name == "Plate1_H12.fcs"
    # closing hides
    ed.close()
    assert not ed.isVisible()
    assert window.extensions["flowdesk.ui.plate_editor"].show_editor() is ed


def test_assignment_from_keyword(app, tmp_path):
    from flowdesk.ui.main_window import MainWindow

    wells = ["D04", "E5", "G11"]
    win = MainWindow()
    try:
        win.controller.add_samples(make_files(tmp_path, ["s1.fcs", "s2.fcs", "s3.fcs"], lambda k, n: {"$WELLID": wells[k]}))
        ed = win.extensions["flowdesk.ui.plate_editor"].show_editor()
        assert ed.source_combo.currentText() == "Auto"
        assert ed.wells[(3, 3)].name == "s1.fcs" and ed.wells[(4, 4)].name == "s2.fcs" and ed.wells[(6, 10)].name == "s3.fcs"
        for s, w in zip(win.controller.workspace.samples, ["A1", "A2", "A3"]):
            s.keywords["Well"] = w
        ed.set_source("Well")
        assert ed.wells[(0, 1)].name == "s2.fcs"
        ed.set_source("Sample name (pattern)")
        assert len(ed.unassigned) == 3
        ed.hide()
    finally:
        win.controller.workspace.modified = False
        win.hide()
        win.deleteLater()


def test_colour_values(window):
    from flowdesk.ui.plate_editor import color_for

    ed = editor(window)
    ed.set_statistic("", "Count")
    assert ed.values[(0, 0)] == 100 and ed.values[(1, 1)] == 200 and ed.values[(7, 11)] == 400
    assert ed.plate.vrange == (100, 400)
    assert ed.well_color(0, 0) == color_for(0.0)
    assert ed.well_color(7, 11) == color_for(1.0)
    assert ed.well_color(5, 5).name() == "#f3f4f6"  # empty well
    assert not ed.param_combo.isEnabled()
    ed.set_statistic("", "Mean", "FSC-A")
    assert ed.param_combo.isEnabled()
    s = by_name(window, "Plate1_A1.fcs")
    assert ed.values[(0, 0)] == pytest.approx(float(s.data("FSC-A").astype(np.float64).mean()))
    assert ed.legend.vrange is not None
    img = ed.grab_image()
    assert not img.isNull()


def test_population_statistic(window):
    from flowdesk.core.gates import RectangleGate
    from flowdesk.core.model import Population

    c = window.controller
    c.add_population(c.workspace.samples[0].root, Population("High", RectangleGate(["FSC-A"], [50000], [None])), c.current_group)
    ed = editor(window)
    assert ed.pop_combo.findData("/High") >= 0
    ed.set_statistic("/High", "Freq. of Parent")
    assert 30 < ed.values[(0, 0)] < 70


def test_export_csv(window, tmp_path):
    ed = editor(window)
    ed.set_statistic("", "Count")
    out = ed.export_csv(str(tmp_path / "plate.csv"))
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0] == ["Sample"] + [str(i) for i in range(1, 13)]
    assert rows[1][0] == "A" and rows[1][1] == "Plate1_A1.fcs"
    assert rows[8][12] == "Plate1_H12.fcs"
    assert rows[10][0].startswith("Count")
    assert rows[11][1] == "100" and rows[12][2] == "200"


def test_click_selects_in_tree_and_group(window):
    from flowdesk.ui.main_window import ROLE_SAMPLE

    ed = editor(window)
    ed.click_well(1, 1)
    sel = window.tree.selectedItems()
    assert len(sel) == 1 and sel[0].data(0, ROLE_SAMPLE).name == "Plate1_B2.fcs"
    assert window.controller.selected_samples()[0].name == "Plate1_B2.fcs"
    # real mouse click on a well goes through the same path
    ed.resize(1040, 640)
    rect = ed.plate.well_rect(0, 0)
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent

    ev = QMouseEvent(QEvent.Type.MouseButtonPress, rect.center(), rect.center(), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier)
    ed.plate.mousePressEvent(ev)
    assert ed.plate.selected == {(0, 0), (1, 1)}
    names = {i.data(0, ROLE_SAMPLE).name for i in window.tree.selectedItems()}
    assert names == {"Plate1_A1.fcs", "Plate1_B2.fcs"}
    g = ed.create_group_from_selection("Wells A1+B2")
    assert g is not None and {s.name for s in window.controller.workspace.members(g)} == {"Plate1_A1.fcs", "Plate1_B2.fcs"}


def test_click_switches_group_when_needed(window):
    from flowdesk.ui.main_window import ROLE_SAMPLE

    c = window.controller
    g = c.create_group("Only A1", [by_name(window, "Plate1_A1.fcs")])
    c.set_current_group(g)
    ed = editor(window)
    ed.click_well(7, 11)
    assert c.current_group.role == "all"
    assert window.tree.selectedItems()[0].data(0, ROLE_SAMPLE).name == "Plate1_H12.fcs"


def test_double_click_opens_graph(window):
    ed = editor(window)
    gw = ed.open_well(1, 1)
    assert gw is not None
    assert any(k for k in window.graph_windows)
    assert ed.open_well(5, 5) is None


def test_refresh_on_samples_changed(window, tmp_path):
    ed = editor(window)
    extra = tmp_path / "more"
    extra.mkdir()
    window.controller.add_samples(make_files(extra, ["Plate1_F6.fcs"]))
    assert ed.wells[(5, 5)].name == "Plate1_F6.fcs"
