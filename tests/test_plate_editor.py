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


def test_percentile_spin(window):
    ed = editor(window)
    ed.set_statistic("", "Count")
    assert ed.pct_spin.isHidden() and ed.pct_label.isHidden()
    assert ed.percentile is None
    ed.set_statistic("", "Percentile", "FSC-A", 90)
    assert not ed.pct_spin.isHidden() and ed.pct_spin.value() == 90
    s = by_name(window, "Plate1_B2.fcs")
    v = s.data("FSC-A").astype(np.float64)
    assert ed.values[(1, 1)] == pytest.approx(np.percentile(v, 90))
    assert ed.legend.label == "Percentile 90 : FSC-A"
    # changing the spin recomputes (a view setting: no undo step, not saved)
    steps = window.controller.can_undo, window.controller.undo_label()
    ed.pct_spin.setValue(10)
    assert ed.values[(1, 1)] == pytest.approx(np.percentile(v, 10))
    assert (window.controller.can_undo, window.controller.undo_label()) == steps
    assert "percentile" not in window.controller.workspace.settings.get("plate_editor", {})
    rows = ed.plate_table()
    assert rows[10][0] == "Percentile 10 : FSC-A"
    # the fit statistics are offered too
    items = [ed.stat_combo.itemText(i) for i in range(ed.stat_combo.count())]
    assert "Cell Cycle %G1" in items and "Prolif. Division Index" in items
    ed.set_statistic("", "Mean", "FSC-A")
    assert ed.pct_spin.isHidden() and ed.percentile is None


def test_read_plate_map(tmp_path):
    from flowdesk.ui.plate_editor import parse_plate_map, read_plate_map, strict_well

    assert strict_well("A1") == (0, 0) and strict_well(" h-12 ") == (7, 11) and strict_well("B07") == (1, 6)
    assert strict_well("Plate1_A1") is None and strict_well("Q1") is None and strict_well("A25") is None
    two = tmp_path / "map.csv"
    two.write_text("Well,Sample\nA1,first.fcs\nB02,second\n\nC3,\n", encoding="utf-8")
    assert read_plate_map(two) == [((0, 0), "first.fcs"), ((1, 1), "second")]
    # name first, semicolons, a BOM
    rev = tmp_path / "rev.csv"
    rev.write_bytes("﻿Name;Well\nfirst.fcs;D4\n".encode("utf-8"))
    assert read_plate_map(rev) == [((3, 3), "first.fcs")]
    # plate-shaped grid with tabs
    grid = tmp_path / "grid.tsv"
    grid.write_text("\t1\t2\t3\nA\tx1\t\tx3\nB\t\ty2\t\n", encoding="utf-8")
    assert read_plate_map(grid) == [((0, 0), "x1"), ((0, 2), "x3"), ((1, 1), "y2")]
    # wells off a 96-well plate are dropped
    assert parse_plate_map("Well,Sample\nM20,big\nA1,a\n", 8, 12) == [((0, 0), "a")]
    assert parse_plate_map(",1,2,13\nA,a,b,c\nZ,z,z,z\n", 8, 12) == [((0, 0), "a"), ((0, 1), "b")]
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert read_plate_map(empty) == []
    with pytest.raises(OSError):
        read_plate_map(tmp_path / "missing.csv")


def test_fill_wells_from_csv(window, tmp_path):
    ed = editor(window)
    assert ed.wells[(0, 0)].name == "Plate1_A1.fcs"
    path = tmp_path / "platemap.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Well", "Sample"])
        w.writerow(["A1", "Control.fcs"])        # replaces the name-pattern assignment
        w.writerow(["E7", "plate1_a1"])          # case-insensitive, no extension
        w.writerow(["F8", "Missing.fcs"])
        w.writerow(["A1", "Plate1_B2.fcs"])      # A1 already listed: ignored
    report = ed.fill_from_csv(str(path))
    assert report == {"assigned": 2, "missing": ["Missing.fcs"], "conflicts": ["A1"]}
    assert ed.wells[(0, 0)].name == "Control.fcs"
    assert ed.wells[(4, 6)].name == "Plate1_A1.fcs"
    assert ed.wells[(1, 1)].name == "Plate1_B2.fcs"  # untouched
    assert not ed.unassigned
    manual = window.controller.workspace.settings["plate_editor"]["manual"]
    assert manual[str(by_name(window, "Control.fcs").id)] == "A1"
    assert "Filled 2 wells" in ed.status.text() and "1 names not found" in ed.status.text()
    assert window.controller.workspace.modified
    # a second map moves a sample that was put on a listed well by hand
    grid = tmp_path / "grid.csv"
    grid.write_text(",1,2\nA,Plate1_H12.fcs,\n", encoding="utf-8")
    report = ed.fill_from_csv(str(grid))
    assert report["assigned"] == 1
    assert ed.wells[(0, 0)].name == "Plate1_H12.fcs"
    assert str(by_name(window, "Control.fcs").id) not in window.controller.workspace.settings["plate_editor"]["manual"]
    assert [s.name for s in ed.unassigned] == ["Control.fcs"]
    # an unreadable file is reported
    bad = ed.fill_from_csv(str(tmp_path / "nope.csv"))
    assert bad["assigned"] == 0 and "Could not read" in ed.status.text()


def test_undo_well_assignment(window, tmp_path):
    c = window.controller
    ed = editor(window)
    control = by_name(window, "Control.fcs")
    assert [s.name for s in ed.unassigned] == ["Control.fcs"]
    ed.unassigned_list.setCurrentRow(0)
    ed.click_well(0, 5)
    assert ed.wells[(0, 5)] is control and not ed.unassigned
    assert c.undo_label() == "assign Control.fcs to A6"
    c.undo()
    assert (0, 5) not in ed.wells  # the well is free again
    assert [s.name for s in ed.unassigned] == ["Control.fcs"]
    assert str(control.id) not in c.workspace.settings.get("plate_editor", {}).get("manual", {})
    c.redo()
    assert ed.wells[(0, 5)] is control
    # unassigning, a CSV fill and a format change are single undo steps too
    ed.plate.selected = {(0, 5)}
    ed.unassign_selected()
    assert (0, 5) not in ed.wells
    c.undo()
    assert ed.wells[(0, 5)] is control
    path = tmp_path / "map.csv"
    path.write_text("Well,Sample\nB7,Plate1_A1.fcs\n", encoding="utf-8")
    ed.fill_from_csv(str(path))
    assert ed.wells[(1, 6)].name == "Plate1_A1.fcs" and (0, 0) not in ed.wells
    c.undo()
    assert ed.wells[(0, 0)].name == "Plate1_A1.fcs" and (1, 6) not in ed.wells
    ed.set_format(384)
    assert ed.plate.rows == 16
    c.undo()
    assert ed.plate.rows == 8 and ed.format_combo.currentData() == 96
    # nothing to unassign / an unchanged layout adds no step
    label = c.undo_label()
    ed.plate.selected = {(0, 0)}  # placed by its name, not by hand
    ed.unassign_selected()
    ed.pattern_edit.editingFinished.emit()
    assert c.undo_label() == label
