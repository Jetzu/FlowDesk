import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QMimeData
from PySide6.QtWidgets import QApplication

from flowdesk.core.gates import RectangleGate
from flowdesk.core.layouts import TableColumn, TableDef
from flowdesk.core.model import Population, Workspace
from flowdesk.ui.controller import POPULATION_MIME, WorkspaceController
from flowdesk.ui.table_editor import (
    TableEditorWindow,
    TableResultWindow,
    build_table,
    keyword_names,
    parse_population_mime,
)

DIR = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel")
FILES = [DIR / "_F1donor 6T cell pehnotype2025-08-18.0003.fcs", DIR / "_F3donor 7T cell pehnotype2025-08-18.0001.fcs"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ws():
    if not all(f.exists() for f in FILES):
        pytest.skip("real FCS files not available")
    w = Workspace()
    for f in FILES:
        w.add_sample(f)
    gate = RectangleGate(["FSC-A", "SSC-A"], [50000, 10000], [150000, 100000])
    w.apply_population_to_group(w.groups[0], "", Population("Lymphocytes", gate))
    return w


def test_build_table(ws):
    t = TableDef(
        name="T",
        columns=[
            TableColumn(pop_path="/Lymphocytes", statistic="Freq. of Parent"),
            TableColumn(pop_path="/Lymphocytes", statistic="Count"),
            TableColumn(pop_path="", statistic="Count"),
            TableColumn(pop_path="/Lymphocytes", statistic="Median", param="FSC-A"),
            TableColumn(pop_path="/Missing", statistic="Count"),
            TableColumn(keyword="$TOT"),
        ],
    )
    headers, rows = build_table(ws, t, ws.groups[0])
    assert headers[0] == "Sample"
    assert len(headers) == 1 + len(t.columns)
    assert headers[1] == "Lymphocytes | Freq. of Parent"
    assert len(rows) == len(ws.samples)
    for s, row in zip(ws.samples, rows):
        assert row[0] == s.name
        pop = s.root.find("/Lymphocytes")
        assert 0.0 <= float(row[1]) <= 100.0
        assert int(row[2]) == pop.count
        assert int(row[3]) == s.n_events
        assert float(row[4]) > 0
        assert row[5] == ""
        assert row[6] == s.keywords["$TOT"].strip()
    # group resolution from the definition
    t.group_name = "All Samples"
    assert build_table(ws, t, None)[1] == rows


def test_keyword_names(ws):
    names = keyword_names(ws, ws.groups[0])
    assert "$TOT" in names


def test_parse_mime():
    payload = json.dumps({"sample_id": 1, "paths": ["/A", "/A/B", ""]}).encode()
    assert parse_population_mime(payload) == ["/A", "/A/B", ""]
    assert parse_population_mime(b"garbage") == []


def test_editor_columns_and_export(app, ws, tmp_path):
    c = WorkspaceController(ws)
    emitted = []
    c.tablesChanged.connect(lambda: emitted.append(1))
    win = TableEditorWindow(c)
    assert win.windowTitle() == "Table Editor"
    assert len(ws.tables) == 1 and ws.tables[0].name == "Table"
    win.show()
    app.processEvents()
    assert win.isVisible()

    added = win.add_columns_for_paths(["/Lymphocytes", ""])
    t = ws.tables[0]
    assert len(t.columns) == 2 and added == t.columns
    assert t.columns[0].statistic == "Freq. of Parent"
    assert t.columns[1].statistic == "Count" and t.columns[1].pop_path == ""
    assert win.columns.rowCount() == 2
    assert emitted

    win.add_columns_for_paths(["/Lymphocytes"], statistic="Mean", param="FSC-A")
    assert t.columns[2].param == "FSC-A"
    win.add_keyword_columns(["$TOT"])
    assert t.columns[3].keyword == "$TOT"

    # move / remove
    win.columns.selectRow(3)
    win.move_columns(-1)
    assert t.columns[2].keyword == "$TOT"
    win.columns.clearSelection()
    win.columns.selectRow(2)
    win.remove_columns()
    assert len(t.columns) == 3

    # drag & drop payload
    mime = QMimeData()
    mime.setData(POPULATION_MIME, json.dumps({"sample_id": ws.samples[0].id, "paths": ["/Lymphocytes"]}).encode())
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QDropEvent

    ev = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    win.columns.dropEvent(ev)
    assert len(t.columns) == 4 and t.columns[3].pop_path == "/Lymphocytes"
    win.columns.clearSelection()
    win.columns.selectRow(3)
    win.remove_columns()

    out = tmp_path / "table.csv"
    win.export_csv(str(out))
    with open(out, newline="") as fh:
        data = list(csv.reader(fh))
    assert data[0][0] == "Sample" and len(data[0]) == 4
    assert len(data) == 1 + len(ws.samples)

    res = win.create_table()
    assert isinstance(res, TableResultWindow)
    assert res.table.rowCount() == len(ws.samples)
    assert res.table.columnCount() == 4
    res.copy_to_clipboard()
    assert QApplication.clipboard().text().startswith("Sample\t")
    out2 = tmp_path / "result.csv"
    res.export_csv(str(out2))
    assert out2.read_text().startswith("Sample,")
    res.close()

    # tables management
    t2 = win.new_table()
    assert len(ws.tables) == 2 and t2.name == "Table 2"
    d = win.duplicate_table()
    assert len(ws.tables) == 3 and d.columns == [] and d.id != t2.id
    win.delete_table()
    assert len(ws.tables) == 2

    # closing hides
    win.close()
    assert not win.isVisible()
    win.show()
    assert win.isVisible()
    win.hide()


def test_group_combo_sets_group_name(app, ws):
    c = WorkspaceController(ws)
    g = c.create_group("Test", ws.samples[:1])
    win = TableEditorWindow(c)
    i = win.group_combo.findData("Test")
    assert i >= 0
    win.group_combo.setCurrentIndex(i)
    assert ws.tables[0].group_name == "Test"
    win.add_columns_for_paths(["/Lymphocytes"])
    headers, rows = win.build()
    assert len(rows) == 1
    assert g is ws.group_by_name("Test")
    win.hide()
