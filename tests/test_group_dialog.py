"""The Create / Edit Group dialog's role, live and font-style controls, and the
Boolean gate dialog that can build the gate for a whole group."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.model import Criterion, Population, Workspace  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def _files(folder, names=("Stim_1.fcs", "Stim_2.fcs", "Ctrl_1.fcs")):
    rng = np.random.default_rng(3)
    out = []
    for name in names:
        events = np.column_stack([rng.normal(50000, 8000, 200), rng.normal(30000, 5000, 200)]).astype(np.float32)
        write_fcs(folder / name, events, [Channel(1, "FSC-A"), Channel(2, "SSC-A")], extra_keywords={"Cond": name.split("_")[0]})
        out.append(folder / name)
    return out


def test_dialog_role_style_live_controls(app, tmp_path):
    from flowdesk.ui.group_dialog import GroupDialog, group_font_style

    ws = Workspace()
    for p in _files(tmp_path):
        ws.add_sample(p)
    g = ws.add_group("Stim", [], [Criterion("Cond", "Equals", "Stim")])
    dlg = GroupDialog(ws.samples, ["Cond"], g)
    # defaults mirror the group: custom shows as Test and stays custom when untouched
    assert dlg.role_combo.currentText() == "Test" and dlg.role() == "custom"
    assert dlg.style_combo.currentText() == "Plain" and dlg.live_check.isChecked()
    dlg.role_combo.setCurrentIndex(dlg.role_combo.findData("control"))
    dlg.style_combo.setCurrentIndex(dlg.style_combo.findData("italic"))
    dlg.live_check.setChecked(False)
    assert dlg.extra_values() == {"role": "control", "style": "italic", "live": False}
    dlg.apply_extras(g, ws)
    assert g.role == "control" and g.italic and not g.bold and group_font_style(g) == "italic"
    # freezing keeps the samples that matched as members
    assert not g.live and {s.name for s in ws.members(g)} == {"Stim_1.fcs", "Stim_2.fcs"}
    # an edit dialog on the frozen group shows its state; a new dialog defaults to Test / live
    dlg2 = GroupDialog(ws.samples, ["Cond"], g)
    assert dlg2.role_combo.currentText() == "Control" and dlg2.style_combo.currentText() == "Italic" and not dlg2.live_check.isChecked()
    new = GroupDialog(ws.samples, ["Cond"], None)
    assert new.role() == "custom" and new.live() and new.font_style() == "plain"
    new.role_combo.setCurrentIndex(new.role_combo.findData("test"))
    new.role_combo.setCurrentIndex(new.role_combo.findData("compensation"))
    assert new.role() == "compensation"


def test_set_group_live_freezes_and_thaws_membership(app, tmp_path):
    from flowdesk.ui.group_dialog import set_group_live

    ws = Workspace()
    for p in _files(tmp_path):
        ws.add_sample(p)
    g = ws.add_group("Stim", [], [Criterion("Cond", "Equals", "Stim")])
    assert len(ws.members(g)) == 2
    assert set_group_live(ws, g, False)
    # a sample that matches later does not join a frozen group
    ws.samples[2].keywords["Cond"] = "Stim"
    assert len(ws.members(g)) == 2
    assert not set_group_live(ws, g, False)  # already frozen
    assert set_group_live(ws, g, True)
    assert len(ws.members(g)) == 3 and g.sample_ids == set()


def test_window_create_and_edit_group_with_role_style_and_live(app, tmp_path, monkeypatch):
    from flowdesk.ui import main_window as mw

    win = mw.MainWindow()
    try:
        c = win.controller
        c.add_samples(_files(tmp_path))
        c._undo.clear()
        answers = {}

        def fake_exec(dlg):
            dlg.name_edit.setText(answers["name"])
            dlg._add_criterion_row(Criterion("Cond", "Equals", "Ctrl"))
            dlg.role_combo.setCurrentIndex(dlg.role_combo.findData(answers["role"]))
            dlg.style_combo.setCurrentIndex(dlg.style_combo.findData(answers["style"]))
            dlg.live_check.setChecked(answers["live"])
            return True

        monkeypatch.setattr(mw.GroupDialog, "exec", fake_exec)
        answers.update(name="Controls", role="control", style="bold", live=True)
        win.create_group()
        g = c.workspace.group_by_name("Controls")
        assert g is not None and c.current_group is g
        assert g.role == "control" and g.bold and g.live and len(c.workspace.members(g)) == 1
        assert len(c._undo) == 1  # one step
        # the group pane shows the role and the style
        win.group_pane.rebuild()
        row = next(r for r in range(win.group_pane.rowCount()) if win.group_pane.group_at(r) is g)
        assert win.group_pane.item(row, 2).text().startswith("Control · ")
        assert win.group_pane.item(row, 0).font().bold()
        # Edit Group…: italic, frozen
        monkeypatch.setattr(mw.GroupDialog, "exec", lambda dlg: (dlg.style_combo.setCurrentIndex(dlg.style_combo.findData("italic")), dlg.live_check.setChecked(False))[0] is None)
        win.edit_group()
        assert g.italic and not g.bold and not g.live
        win.group_pane.rebuild()
        row = next(r for r in range(win.group_pane.rowCount()) if win.group_pane.group_at(r) is g)
        assert win.group_pane.item(row, 0).font().italic()
        assert "(frozen)" in win.group_pane.item(row, 2).text()
        # the live toggle in the Workspace menu / context menu thaws it again (undoable)
        assert win.act_live_group.isEnabled() and not win.act_live_group.isChecked()
        win.act_live_group.trigger()
        assert g.live and win.act_live_group.isChecked()
        c.undo()
        assert not g.live
    finally:
        c.workspace.modified = False
        win.hide()
        win.deleteLater()
        app.processEvents()


def test_boolean_dialog_group_option(app):
    from flowdesk.core.model import Group
    from flowdesk.ui.group_dialog import GroupBooleanGateDialog

    a, b = Population("A", RectangleGate(["FSC-A"], [1], [None])), Population("B", RectangleGate(["SSC-A"], [1], [None]))
    root = Population("root")
    root.add_child(a)
    root.add_child(b)
    g = Group("Panel")
    dlg = GroupBooleanGateDialog([a, b], "or", g, checked=True)
    dlg.check([a, b])
    assert dlg.apply_to_group() and "Panel" in dlg.group_check.text()
    assert dlg.values() == ("or", ["/A", "/B"], "A OR B")
    dlg.group_check.setChecked(False)
    assert not dlg.apply_to_group()
    locked = GroupBooleanGateDialog([a, b], "and", g, checked=True, locked=True)
    assert locked.apply_to_group() and not locked.group_check.isEnabled()
    none = GroupBooleanGateDialog([a, b], "and", None, checked=True)
    assert not none.apply_to_group() and not none.group_check.isEnabled()
