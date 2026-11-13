"""Workspace tools: find, header sort, move, select equivalent, groups by
keyword, bulk rename, keyword cells, display names and row commands."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QTreeWidgetItemIterator  # noqa: E402

from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.model import Population  # noqa: E402

# name, events, Donor, TUBE NAME
SPECS = [
    ("S_A1.fcs", 400, "D2", "CD3 stim"),
    ("S_A2.fcs", 150, "D1", "CD3 unstim"),
    ("S_A10.fcs", 900, "D1", "Iso"),
    ("S_B1.fcs", 250, "", "FMO"),
]
MOD = Qt.KeyboardModifier.NoModifier


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def make_files(folder, specs=SPECS):
    rng = np.random.default_rng(11)
    paths = []
    for name, n, donor, tube in specs:
        events = np.column_stack([rng.normal(50000, 5000, n), rng.normal(30000, 4000, n), rng.lognormal(7, 1, n)]).astype(np.float32)
        kw = {"$CYT": "TestCyt", "TUBE NAME": tube, "$BTIM": "10:00:00", "$ETIM": "10:00:10"}
        if donor:
            kw["Donor"] = donor
        p = folder / name
        write_fcs(p, events, [Channel(1, "FSC-A"), Channel(2, "SSC-A"), Channel(3, "FITC-A", "CD3")], extra_keywords=kw)
        paths.append(p)
    return paths


def add_gates(sample):
    cells = sample.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [0, 0], [1e6, 1e6])))
    cells.add_child(Population("CD3+", RectangleGate(["FITC-A"], [1000], [1e7])))
    cells.add_child(Population("CD4", RectangleGate(["FITC-A"], [0], [1e7])))


@pytest.fixture
def window(app, tmp_path):
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    c = win.controller
    c.add_samples(make_files(tmp_path))
    for s in c.workspace.samples[:3]:  # S_B1 has no gates
        add_gates(s)
    c.gatingChanged.emit(None)
    app.processEvents()
    yield win
    for w in list(win.graph_windows.values()):
        w.close()
    c.workspace.modified = False
    win.hide()
    win.deleteLater()


def ext(win):
    return win.extensions["flowdesk.ui.workspace_tools"]


def by_name(win, name):
    return next(s for s in win.controller.workspace.samples if s.name == name)


def names(win):
    return [s.name for s in win.controller.workspace.samples]


def items(tree):
    out = []
    it = QTreeWidgetItemIterator(tree)
    while it.value():
        out.append(it.value())
        it += 1
    return out


def pop_item(tree, sample, path):
    from flowdesk.ui.main_window import ROLE_POP, ROLE_SAMPLE, ROLE_STAT

    for item in items(tree):
        p = item.data(0, ROLE_POP)
        if item.data(0, ROLE_SAMPLE) is sample and p is not None and item.data(0, ROLE_STAT) is None and p.path == path:
            return item
    return None


def sample_item(tree, sample):
    return pop_item(tree, sample, "")


def test_install_menus_band_and_find_bar(window):
    e = ext(window)
    ws_titles = [a.text() for a in window.menus["workspace"].actions()]
    for t in ("Find…", "Select Equivalent", "Create Groups by Keyword…", "Rename Samples…", "Name Samples by Keyword…", "Move Up", "Move Down"):
        assert t in ws_titles
    edit_titles = [a.text() for a in window.menus["edit"].actions()]
    assert "Duplicate" in edit_titles and "Copy as Text" in edit_titles
    assert edit_titles.index("Duplicate") < edit_titles.index("Preferences…")
    assert e.band is not None
    assert e.act_find.shortcut().matches(e.act_find.shortcut()) and not e.act_find.shortcut().isEmpty()
    # the find box sits above the tree inside the splitter
    assert window.splitter.indexOf(window.tree) < 0
    assert e.tree_container.isAncestorOf(window.tree) and e.tree_container.isAncestorOf(e.find_edit)
    assert window.tree.header().sectionsClickable()


def test_group_by_keyword(window):
    from flowdesk.core.model import Criterion
    from flowdesk.ui.experiment import set_keyword
    from flowdesk.ui.workspace_tools import GroupByKeywordDialog, keyword_value_counts

    c = window.controller
    assert keyword_value_counts(c.workspace.samples, "Donor") == [("D1", 2), ("D2", 1)]
    n_groups = len(c.workspace.groups)
    dlg = GroupByKeywordDialog(c, window)
    dlg.keyword_combo.setCurrentText("Donor")
    assert dlg.preview.count() == 2 and "2 groups will be created" in dlg.status.text()
    assert dlg.preview.item(0).text().startswith("Donor D1 — 2 samples")
    created = dlg.apply()
    assert [g.name for g in created] == ["Donor D1", "Donor D2"]
    assert len(c.workspace.groups) == n_groups + 2
    g1 = c.workspace.group_by_name("Donor D1")
    assert g1.criteria == [Criterion("Donor", "Equals", "D1")] and g1.live
    assert {s.name for s in c.workspace.members(g1)} == {"S_A2.fcs", "S_A10.fcs"}
    assert c.undo_label() == "create groups by Donor"
    # one undo step removes every group
    c.undo()
    assert len(c.workspace.groups) == n_groups
    c.redo()
    g1 = c.workspace.group_by_name("Donor D1")
    # existing names are skipped
    assert ext(window).group_by_keyword("Donor") == []
    # a sample given the value later joins the group with its gates
    g1.root.add_child(Population("Lymph", RectangleGate(["FSC-A", "SSC-A"], [0, 0], [1e6, 1e6])))
    late = by_name(window, "S_B1.fcs")
    assert late.root.find("Lymph") is None
    set_keyword(c, [late], "Donor", "D1")
    assert late in c.workspace.members(g1) and late.root.find("Lymph") is not None


def test_header_click_sorts_samples(window):
    from flowdesk.ui.main_window import ROLE_SAMPLE

    c = window.controller
    tree = window.tree
    header = tree.header()
    header.sectionClicked.emit(2)  # #Cells
    assert [s.n_events for s in c.workspace.samples] == [150, 250, 400, 900]
    assert header.isSortIndicatorShown() and header.sortIndicatorSection() == 2
    assert header.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
    assert tree.topLevelItem(0).data(0, ROLE_SAMPLE).name == "S_A2.fcs"
    header.sectionClicked.emit(2)  # again: descending
    assert [s.n_events for s in c.workspace.samples] == [900, 400, 250, 150]
    assert header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    assert c.undo_label() == "sort samples by #Cells (descending)"
    tree.rebuild()  # the indicator survives a rebuild
    assert header.isSortIndicatorShown() and header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    c.undo()
    assert [s.n_events for s in c.workspace.samples] == [150, 250, 400, 900]
    assert not header.isSortIndicatorShown() or header.sortIndicatorOrder() != Qt.SortOrder.DescendingOrder
    # name column: natural order; statistic column does nothing
    header.sectionClicked.emit(0)
    assert names(window) == ["S_A1.fcs", "S_A2.fcs", "S_A10.fcs", "S_B1.fcs"]
    order = names(window)
    header.sectionClicked.emit(1)
    assert names(window) == order
    # keyword column: samples without the keyword go last
    tree.keyword_columns = ["Donor"]
    tree.rebuild()
    header.sectionClicked.emit(3)
    assert names(window)[-1] == "S_B1.fcs" and [s.keywords.get("Donor") for s in c.workspace.samples[:3]] == ["D1", "D1", "D2"]


def test_find_filters_tree_and_survives_rebuild(window, app):
    from flowdesk.ui.main_window import ROLE_POP

    e = ext(window)
    tree = window.tree
    a10 = by_name(window, "S_A10.fcs")
    a10.root.find("Cells").expanded = False
    tree.rebuild()
    assert not pop_item(tree, a10, "/Cells").isExpanded()
    e.act_find.trigger()
    e.find_edit.setText("CD3")
    visible = [it for it in items(tree) if not it.isHidden()]
    pops = {it.data(0, ROLE_POP).path for it in visible if it.data(0, ROLE_POP) is not None}
    assert "/Cells/CD3+" in pops and "/Cells" in pops  # match plus its ancestors
    assert "/Cells/CD4" not in pops  # non-matching sibling hidden
    assert sample_item(tree, by_name(window, "S_B1.fcs")).isHidden()  # no CD3 anywhere
    assert pop_item(tree, a10, "/Cells").isExpanded()  # opened to show the match ...
    assert not a10.root.find("Cells").expanded  # ... without changing the saved state
    assert e.match_label.text() == "3 matches"
    tree.rebuild()
    assert sample_item(tree, by_name(window, "S_B1.fcs")).isHidden()
    assert pop_item(tree, by_name(window, "S_A1.fcs"), "/Cells/CD4").isHidden()
    assert not pop_item(tree, by_name(window, "S_A1.fcs"), "/Cells/CD3+").isHidden()
    # a sample name matches with all of its rows
    e.set_filter("S_B1")
    assert not sample_item(tree, by_name(window, "S_B1.fcs")).isHidden()
    assert sample_item(tree, by_name(window, "S_A1.fcs")).isHidden()
    # keyword columns are searched too
    tree.keyword_columns = ["TUBE NAME"]
    tree.rebuild()
    e.set_filter("unstim")
    assert not sample_item(tree, by_name(window, "S_A2.fcs")).isHidden()
    assert sample_item(tree, by_name(window, "S_A1.fcs")).isHidden()
    # clearing shows every row with the saved expansion
    e.set_filter("CD3")
    e.set_filter("")
    assert not any(it.isHidden() for it in items(tree))
    assert not pop_item(tree, a10, "/Cells").isExpanded()
    assert e.match_label.text() == ""


def test_select_equivalent(window):
    c = window.controller
    tree = window.tree
    s0 = c.workspace.samples[0]
    tree.clearSelection()
    pop_item(tree, s0, "/Cells/CD3+").setSelected(True)
    assert [p.path for _s, p in c.selected] == ["/Cells/CD3+"]
    assert ext(window).act_select_equivalent.isEnabled()
    n = ext(window).select_equivalent()
    assert n == 3
    assert {(s.name, p.path) for s, p in c.selected} == {(s.name, "/Cells/CD3+") for s in c.workspace.samples[:3]}


def test_rename_samples_dialog(window):
    from flowdesk.ui.workspace_tools import RenameSamplesDialog

    c = window.controller
    samples = c.workspace.samples[:3]
    c.set_selection([(s, s.root) for s in samples])
    dlg = RenameSamplesDialog(c, c.selected_samples(), window)
    dlg.keyword_combo.setCurrentText("TUBE NAME")
    assert dlg.preview.rowCount() == 3
    assert [dlg.preview.item(r, 1).text() for r in range(3)] == ["CD3 stim", "CD3 unstim", "Iso"]
    assert "3 of 3 samples will be renamed" in dlg.status.text()
    assert dlg.apply() == 3
    assert names(window)[:3] == ["CD3 stim", "CD3 unstim", "Iso"]
    assert samples[0].root.name == "CD3 stim"
    assert c.undo_label() == "rename samples"
    window.tree.rebuild()
    assert window.tree.topLevelItem(0).text(0) == "CD3 stim"
    c.undo()  # one step for all three
    assert names(window)[:3] == ["S_A1.fcs", "S_A2.fcs", "S_A10.fcs"]
    # find / replace (regular expression) and prefix / suffix
    dlg = RenameSamplesDialog(c, c.workspace.samples, window)
    dlg.find_edit.setText(r"S_([A-H])(\d+)")
    assert dlg.mode() == RenameSamplesDialog.MODE_REPLACE  # typing selects the mode
    dlg.regex_check.setChecked(True)
    dlg.replace_edit.setText(r"Well \1\2")
    assert dlg.new_names()[c.workspace.samples[2].id] == "Well A10.fcs"
    dlg.find_edit.setText("(")
    assert dlg.new_names() == {} and "Invalid expression" in dlg.status.text()
    dlg.prefix_edit.setText("Exp1_")
    dlg.suffix_edit.setText("_r")
    assert dlg.mode() == RenameSamplesDialog.MODE_AFFIX
    dlg.accept()
    assert names(window)[0] == "Exp1_S_A1_r.fcs"


def test_rename_helpers():
    from pathlib import Path

    from flowdesk.core.model import Sample
    from flowdesk.ui.workspace_tools import names_affix, names_from_keyword, names_replace

    a, b = Sample(Path("x/a_1.fcs"), 1), Sample(Path("x/b_2.fcs"), 2)
    a.keywords["$SRC"] = "Blood"
    assert names_from_keyword([a, b], "$SRC") == {1: "Blood"}
    assert names_replace([a, b], "A_", "Z-") == {1: "Z-1.fcs"}  # case-insensitive, literal
    assert names_replace([a, b], "a_", "\\1", regex=False) == {1: "\\11.fcs"}
    assert names_replace([a, b], "A_", "Z-", case_sensitive=True) == {}
    assert names_affix([a], "p_", "") == {1: "p_a_1.fcs"}


def test_keyword_cell_edit_writes_keyword(window, app):
    from flowdesk.ui.main_window import ROLE_SAMPLE

    c = window.controller
    tree = window.tree
    tree.keyword_columns = ["Donor", "$CYT"]
    tree.rebuild()
    s = c.workspace.samples[0]
    item = sample_item(tree, s)
    e = ext(window)
    assert e._keyword_cell(item, 3) == (s, "Donor")
    assert e._keyword_cell(item, 4) is None  # $ keywords stay read-only
    assert e._keyword_cell(pop_item(tree, s, "/Cells"), 3) is None  # population rows
    item.setText(3, "D7")  # what committing the cell editor does
    app.processEvents()
    assert s.keywords["Donor"] == "D7"
    assert c.undo_label() == "set keyword Donor"
    tree.rebuild()
    item = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).data(0, ROLE_SAMPLE) is s)
    assert item.text(3) == "D7"
    c.undo()
    assert s.keywords["Donor"] == "D2"


def test_double_click_keyword_cell_edits_instead_of_graph(window, app):
    from PySide6.QtWidgets import QAbstractItemView

    window.resize(1100, 700)
    window.show()
    tree = window.tree
    tree.keyword_columns = ["Donor"]
    tree.rebuild()
    app.processEvents()
    s = window.controller.workspace.samples[0]
    item = sample_item(tree, s)
    tree.scrollToItem(item)
    rect = tree.visualItemRect(item)
    x = tree.header().sectionViewportPosition(3) + 5
    pos = QPointF(x, rect.center().y())
    btn = Qt.MouseButton.LeftButton
    opened = len(window.graph_windows)
    app.sendEvent(tree.viewport(), QMouseEvent(QEvent.Type.MouseButtonDblClick, pos, tree.viewport().mapToGlobal(pos.toPoint()), btn, btn, MOD))
    assert tree.state() == QAbstractItemView.State.EditingState
    assert len(window.graph_windows) == opened
    tree.closePersistentEditor(item, 3)
    window.hide()


def test_name_samples_by_keyword_display(window, monkeypatch):
    from flowdesk.ui.main_window import ROLE_SAMPLE
    from flowdesk.ui.workspace_tools import NameByKeywordDialog

    c = window.controller
    tree = window.tree
    e = ext(window)
    e.set_name_keyword("TUBE NAME")
    shown = {tree.topLevelItem(i).data(0, ROLE_SAMPLE).name: tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())}
    assert shown["S_A1.fcs"] == "CD3 stim" and shown["S_B1.fcs"] == "FMO"
    assert names(window)[0] == "S_A1.fcs"  # display only
    assert "Sample: S_A1.fcs" in tree.topLevelItem(0).toolTip(0)
    assert c.workspace.settings["name_keyword"] == "TUBE NAME"
    tree.rebuild()
    assert tree.topLevelItem(0).text(0) == "CD3 stim"
    c.undo()
    assert tree.topLevelItem(0).text(0) == "S_A1.fcs"
    # the preference is the default for workspaces without their own choice
    assert "name_keyword" not in c.workspace.settings
    monkeypatch.setitem(c.preferences._values, "sample_name_keyword", "Donor")
    tree.rebuild()
    assert tree.topLevelItem(0).text(0) == "D2" and tree.topLevelItem(3).text(0) == "S_B1.fcs"
    dlg = NameByKeywordDialog(c, e.name_keyword(), window)
    assert dlg.keyword() == "Donor"
    dlg.keyword_combo.setCurrentText(NameByKeywordDialog.FILE_NAME)
    assert dlg.keyword() == ""


def test_move_up_down(window):
    c = window.controller
    s2 = c.workspace.samples[2]
    c.set_selection([(s2, s2.root)])
    assert ext(window).move_selected(-1)
    assert names(window) == ["S_A1.fcs", "S_A10.fcs", "S_A2.fcs", "S_B1.fcs"]
    assert c.undo_label() == "move samples"
    assert [s.name for s in c.selected_samples()] == ["S_A10.fcs"]  # selection kept
    assert ext(window).move_selected(-1)
    assert not ext(window).move_selected(-1)  # already first
    assert names(window)[0] == "S_A10.fcs"
    ext(window).act_move_down.trigger()
    assert names(window)[1] == "S_A10.fcs"
    c.undo()
    assert names(window)[0] == "S_A10.fcs"
    # within a group the other samples keep their places
    from flowdesk.ui.workspace_tools import move_samples

    ws = c.workspace
    g = c.create_group("Pair", [by_name(window, "S_A1.fcs"), by_name(window, "S_B1.fcs")])
    assert move_samples(c, [by_name(window, "S_B1.fcs")], -1, ws.members(g))
    assert names(window) == ["S_A10.fcs", "S_B1.fcs", "S_A2.fcs", "S_A1.fcs"]


def test_copy_as_text_path_and_reveal(window, app):
    from flowdesk.ui.workspace_tools import reveal_command, reveal_label

    c = window.controller
    tree = window.tree
    s0 = c.workspace.samples[0]
    tree.clearSelection()
    sample_item(tree, s0).setSelected(True)
    pop_item(tree, s0, "/Cells").setSelected(True)
    text = ext(window).copy_rows_as_text()
    assert QApplication.clipboard().text() == text
    lines = text.splitlines()
    assert lines[0].split("\t")[:5] == ["Sample", "Population", "Name", "Statistic", "#Cells"]
    row = lines[2].split("\t")
    assert row[0] == "S_A1.fcs" and row[1] == "Cells" and row[2] == "Cells"
    assert row[4] in ("400", "…")  # thousands separators removed
    tree.clearSelection()
    sample_item(tree, s0).setSelected(True)
    assert ext(window).copy_selected_paths() == str(s0.path)
    assert QApplication.clipboard().text() == str(s0.path)
    assert reveal_command("/x/a.fcs", "darwin") == ["open", "-R", "/x/a.fcs"]
    assert reveal_command("C:/x/a.fcs", "win32")[0] == "explorer"
    assert reveal_command("/x/a.fcs", "linux") is None
    assert reveal_label("darwin") == "Reveal in Finder" and reveal_label("win32") == "Show in Explorer"


def test_duplicate_gate(window):
    c = window.controller
    tree = window.tree
    s0 = c.workspace.samples[0]
    tree.clearSelection()
    pop_item(tree, s0, "/Cells/CD3+").setSelected(True)
    assert ext(window).duplicate_selected() == 1
    assert s0.root.find("/Cells/CD3+ 2") is not None
    assert c.workspace.samples[1].root.find("/Cells/CD3+ 2") is None
    assert c.undo_label() == "duplicate gates"
    c.undo()
    assert s0.root.find("/Cells/CD3+ 2") is None


def test_sample_tooltips_and_context_menu(window):
    from PySide6.QtWidgets import QMenu

    tree = window.tree
    s0 = window.controller.workspace.samples[0]
    tip = sample_item(tree, s0).toolTip(0)
    assert str(s0.path) in tip and "TUBE NAME: CD3 stim" in tip and "Events: 400" in tip
    assert "events/s" in tip  # 400 events in 10 s
    menu = QMenu()
    ext(window)._context_hook(menu, sample_item(tree, s0))
    titles = [a.text() for a in menu.actions()]
    assert "Move Up" in titles and "Copy File Path" in titles and "Rename Samples…" in titles
    menu = QMenu()
    ext(window)._context_hook(menu, pop_item(tree, s0, "/Cells"))
    titles = [a.text() for a in menu.actions()]
    assert "Select Equivalent" in titles and "Duplicate" in titles and "Copy as Text" in titles


def test_dialog_entry_points_and_keys(window, app, monkeypatch):
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QAbstractItemView, QDialog

    from flowdesk.ui import workspace_tools as wt

    monkeypatch.setattr(QDialog, "exec", lambda self: 0)  # every dialog is cancelled
    e = ext(window)
    c = window.controller
    assert isinstance(e.rename_samples(), wt.RenameSamplesDialog)
    assert e.group_by_keyword() == []
    assert isinstance(e.name_by_keyword(), wt.NameByKeywordDialog)
    assert names(window)[0] == "S_A1.fcs" and "name_keyword" not in c.workspace.settings
    # F2 on a keyword cell edits the cell, not the name
    window.resize(1100, 700)
    window.show()
    tree = window.tree
    tree.keyword_columns = ["Donor"]
    tree.rebuild()
    app.processEvents()
    item = sample_item(tree, c.workspace.samples[0])
    tree.setCurrentItem(item, 3)
    tree.setFocus()
    override = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_F2, MOD)
    app.sendEvent(tree, override)
    assert override.isAccepted()
    app.sendEvent(tree, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_F2, MOD))
    assert tree.state() == QAbstractItemView.State.EditingState
    tree.closePersistentEditor(item, 3)
    # Escape in the find box clears the filter
    e.find_edit.setText("CD3")
    app.sendEvent(e.find_edit, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, MOD))
    assert e.find_edit.text() == "" and not any(it.isHidden() for it in items(tree))
    window.hide()
