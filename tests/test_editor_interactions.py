"""Table Editor and Layout Editor driven through their actions, keys and mouse."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QTableWidget  # noqa: E402

from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.model import Population  # noqa: E402

FOLDER = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel")
FCS = sorted(FOLDER.glob("*.fcs")) if FOLDER.exists() else []

pytestmark = pytest.mark.skipif(len(FCS) < 2, reason="real test data not available")


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture
def window(app):
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    c = win.controller
    c.add_samples(FCS[:2])
    s0 = c.workspace.samples[0]
    c.add_population(s0.root, Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 5000], [200000, 150000])), c.current_group)
    c.add_population(s0.root.find("/Cells"), Population("Bright", RectangleGate(["B1-A"], [1000], [None])), c.current_group)
    app.processEvents()
    yield win
    c.workspace.modified = False
    win.hide()


def test_table_editor_actions(app, window, tmp_path):
    from flowdesk.ui.table_editor import ColumnDialog, build_table

    c = window.controller
    ws = c.workspace
    te = window.show_table_editor()
    te.resize(900, 500)
    app.processEvents()
    n0 = len(ws.tables)
    t = te.new_table()
    assert len(ws.tables) == n0 + 1 and te.current_table() is t
    # rename through the list (what double-click + typing does)
    row = ws.tables.index(t)
    item = te.table_list.item(row)
    item.setText("Donor stats")
    app.processEvents()
    assert t.name == "Donor stats"
    # columns: via drop-style helper, via the dialog, keyword columns
    te.add_columns_for_paths(["/Cells", "/Cells/Bright"])
    dlg = ColumnDialog(c, c.current_group)
    dlg.stat_combo.setCurrentText("Count")
    idx = dlg.pop_combo.findText("Cells")
    if idx >= 0:
        dlg.pop_combo.setCurrentIndex(idx)
    t.columns.append(dlg.column())
    te.add_keyword_columns(["$CYT"])
    c.notify_tables()
    app.processEvents()
    assert len(t.columns) == 4
    assert te.columns.rowCount() == 4
    # reorder and remove through the grid selection
    te.columns.selectRow(3)
    te.move_columns(-1)
    assert t.columns[2].keyword == "$CYT"
    te.columns.selectRow(2)
    te.remove_columns()
    assert len(t.columns) == 3 and all(col.keyword is None for col in t.columns)
    # group selector drives the rows
    te.group_combo.setCurrentIndex(0)
    assert t.group_name == ws.groups[0].name
    # Create Table shows the computed values
    headers, rows = te.build()
    assert headers[0] == "Sample" and len(rows) == len(ws.members(ws.groups[0]))
    win_result = te.create_table()
    assert win_result is not None
    table = win_result.findChild(QTableWidget)
    assert table is not None and table.rowCount() == len(rows) and table.columnCount() == len(headers)
    assert table.item(0, 1).text() == rows[0][1]
    assert float(rows[0][1]) > 0  # Freq. of Parent of Cells
    assert rows[0][3] == str(ws.samples[0].root.find("/Cells").count)
    # export and clipboard
    out = tmp_path / "table.csv"
    te.export_csv(str(out))
    text = out.read_text()
    assert text.splitlines()[0].startswith("Sample,")
    te.copy_to_clipboard()
    assert QApplication.clipboard().text().startswith("Sample\t")
    # duplicate / delete
    dup = te.duplicate_table()
    assert dup is not None and len(ws.tables) == n0 + 2 and len(dup.columns) == 3
    te.delete_table()
    assert len(ws.tables) == n0 + 1
    win_result.close()
    te.close()


def test_layout_editor_actions_keys_and_rubber_band(app, window, tmp_path):
    c = window.controller
    ws = c.workspace
    s0 = ws.samples[0]
    le = window.show_layout_editor()
    le.resize(1000, 800)
    app.processEvents()
    layout = le.current_layout()
    a = le.add_population(s0, s0.root.find("/Cells"))
    b = le.add_population(s0, s0.root.find("/Cells/Bright"))
    txt = le.add_text("Donor 6", edit=False)
    assert txt.kind == "text" and len(layout.items) == 3
    # select both plots and align / size them
    le.scene.clearSelection()
    le.scene_item(a).setSelected(True)
    le.scene_item(b).setSelected(True)
    b.w, b.h = 140, 140
    le.scene_item(b).sync_from_model()
    le.same_size()
    assert (b.w, b.h) == (a.w, a.h)
    le.align_left()
    assert b.x == a.x
    le.align_top()
    assert b.y == a.y
    le.bring_to_front(le.selected_layout_items())
    dups = le.duplicate_items(le.selected_layout_items())
    assert len(dups) == 2 and len(layout.items) == 5
    # page setup and zoom
    le.set_page("Letter", True)
    assert layout.page == "Letter" and layout.landscape and layout.page_size[0] > layout.page_size[1]
    before = le.view.transform().m11()
    le.zoom_by(1.25)
    assert le.view.transform().m11() == pytest.approx(before * 1.25, rel=1e-3)
    le.fit_page()
    app.processEvents()
    # rubber-band selection with the mouse over the text item, then Delete key
    le.scene.clearSelection()
    txt.x, txt.y = 560.0, 480.0  # away from the plots stacked at the top-left
    gi = le.scene_item(txt)
    gi.sync_from_model()
    app.processEvents()
    r = gi.sceneBoundingRect()
    assert not any(le.scene_item(it).sceneBoundingRect().intersects(r.adjusted(-20, -20, 20, 20)) for it in layout.items if it is not txt)
    p0 = le.view.mapFromScene(QPointF(r.left() - 15, r.top() - 15))
    p1 = le.view.mapFromScene(QPointF(r.right() + 15, r.bottom() + 15))
    QTest.mousePress(le.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
    QTest.mouseMove(le.view.viewport(), p1)
    QTest.mouseRelease(le.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1)
    app.processEvents()
    assert gi.isSelected()
    n_before = len(layout.items)
    QTest.keyClick(le.view, Qt.Key.Key_Delete)
    app.processEvents()
    assert len(layout.items) == n_before - 1 and txt not in layout.items
    # arrow keys nudge the selected plot
    le.scene.clearSelection()
    le.scene_item(a).setSelected(True)
    x_before = a.x
    QTest.keyClick(le.view, Qt.Key.Key_Right)
    app.processEvents()
    le.flush_notify()
    assert a.x > x_before
    # layouts list: new / duplicate / delete
    second = le.new_layout("Second")
    assert le.current_layout() is second and len(ws.layouts) == 2
    le.set_current_layout(layout)
    dup = le.duplicate_layout()
    assert len(dup.items) == len(layout.items) and len(ws.layouts) == 3
    le.delete_layout(confirm=False)
    assert len(ws.layouts) == 2
    # batch PDF: one page per member of the group
    le.set_current_layout(layout)
    pages = le.export_pdf(str(tmp_path / "batch.pdf"), batch_group=ws.groups[0])
    assert pages == len(ws.members(ws.groups[0])) and (tmp_path / "batch.pdf").stat().st_size > 1000
    le.close()
