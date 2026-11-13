"""Workspace tools: the sample-list commands of FlowJo's workspace window.

* Find (Ctrl+F): a search box above the sample tree hides every row that does
  not match the typed text (sample and population names, keyword columns);
  ancestors of a match stay visible and open, and the filter survives every
  rebuild of the tree.
* Header-click sorting: clicking Name, #Cells or a keyword column header
  reorders the samples (natural / numeric order, a second click reverses it)
  with a sort indicator; the order is the workspace's sample order (saved,
  exported, undoable).
* Move Up / Move Down for the selected samples (manual order).
* Select Equivalent: selects the same population path in every sample shown.
* Create Groups by Keyword…: one live keyword group per distinct value.
* Rename Samples…: from a keyword (TUBE NAME, $SRC …), find / replace (plain or
  regular expression) or prefix / suffix, with a preview, as one undo step.
* Name Samples by Keyword…: the sample list shows each sample under a keyword
  value instead of its name (display only; workspace setting "name_keyword",
  default from the preference "sample_name_keyword").
* Keyword cells: double-click (or F2 on) a keyword column cell of a sample row
  to edit that keyword in place.
* Duplicate (Ctrl+D), Copy as Text (Ctrl+Shift+C), Reveal in Finder / Show in
  Explorer, Copy File Path, and hover tooltips on sample rows.

`install(window)` adds the actions to the Workspace and Edit menus, a "Sample
List" band on the Workspace ribbon page and entries in the sample-tree context
menu; the logic is in plain functions usable without the window.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from ..core.model import Criterion, Group, Sample
from .experiment import acquisition_duration, format_duration, format_rate, keyword_union, natural_key, set_keyword
from .icons import icon

NAME_PREFERENCE = "sample_name_keyword"  # application default for the displayed sample name
NAME_SETTING = "name_keyword"  # workspace.settings key (overrides the preference)
GROUP_PATTERN = "{keyword} {value}"
GROUP_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
TOOLTIP_KEYWORDS = ("TUBE NAME", "$SRC", "WELL ID", "$WELLID", "$PLATEID", "EXPERIMENT NAME")
FIRST_KEYWORD_COLUMN = 3  # Name, Statistic, #Cells, then the keyword columns


def _roles():
    from .main_window import ROLE_GROUP, ROLE_POP, ROLE_SAMPLE, ROLE_STAT

    return ROLE_SAMPLE, ROLE_POP, ROLE_STAT, ROLE_GROUP


# =============================================================== pure helpers
def keyword_value_counts(samples: list[Sample], keyword: str) -> list[tuple[str, int]]:
    """Distinct non-empty values of a keyword (case-insensitive, first
    spelling kept) with their sample counts, in natural order."""
    counts: dict[str, list] = {}
    for s in samples:
        v = str(s.keywords.get(keyword, "") or "").strip()
        if not v and keyword in ("$FIL", "FILENAME"):
            v = s.name
        if not v:
            continue
        entry = counts.setdefault(v.lower(), [v, 0])
        entry[1] += 1
    return sorted(((v, n) for v, n in counts.values()), key=lambda t: natural_key(t[0]))


def group_plan(samples: list[Sample], keyword: str, pattern: str = GROUP_PATTERN) -> list[tuple[str, str, int]]:
    """(value, group name, sample count) for every distinct keyword value."""
    pattern = pattern or "{value}"
    out = []
    for value, n in keyword_value_counts(samples, keyword):
        try:
            name = pattern.format(keyword=keyword, value=value).strip()
        except (KeyError, IndexError, ValueError):
            name = f"{keyword} {value}"
        out.append((value, name or value, n))
    return out


def create_groups_by_keyword(controller, keyword: str, pattern: str = GROUP_PATTERN, colors: bool = False, samples: list[Sample] | None = None) -> list[Group]:
    """One live keyword group per distinct value: criterion (keyword Equals
    value). Groups whose name exists already are skipped. One undo step;
    returns the groups created."""
    ws = controller.workspace
    keyword = (keyword or "").strip()
    if not keyword:
        return []
    plan = [(v, name) for v, name, _n in group_plan(ws.samples if samples is None else samples, keyword, pattern) if ws.group_by_name(name) is None]
    if not plan:
        return []
    created = []
    with controller.undo_group(f"create groups by {keyword}"):
        for k, (value, name) in enumerate(plan):
            color = GROUP_COLORS[k % len(GROUP_COLORS)] if colors else "#000000"
            g = ws.add_group(name, None, [Criterion(keyword, "Equals", value)], color)
            for s in ws.members(g):
                ws.apply_template(g, s)
            created.append(g)
    controller.mark_modified()
    controller.groupsChanged.emit()
    controller.gatingChanged.emit(None)
    controller.statusMessage.emit(f"Created {len(created)} group{'s' if len(created) != 1 else ''} by {keyword}")
    return created


def names_from_keyword(samples: list[Sample], keyword: str) -> dict[int, str]:
    """sample id -> keyword value, for the samples that have the keyword and
    whose name would change."""
    out = {}
    for s in samples:
        v = str(s.keywords.get(keyword, "") or "").strip()
        if v and v != s.name:
            out[s.id] = v
    return out


def names_replace(samples: list[Sample], find: str, replace: str, regex: bool = False, case_sensitive: bool = False) -> dict[int, str]:
    """sample id -> name with `find` replaced (plain text or a regular
    expression with \\1 back-references). Raises re.error for a bad pattern."""
    if not find:
        return {}
    flags = 0 if case_sensitive else re.IGNORECASE
    rx = re.compile(find if regex else re.escape(find), flags)
    repl = replace if regex else (lambda _m: replace)
    out = {}
    for s in samples:
        new = rx.sub(repl, s.name).strip()
        if new and new != s.name:
            out[s.id] = new
    return out


def names_affix(samples: list[Sample], prefix: str = "", suffix: str = "", before_extension: bool = True) -> dict[int, str]:
    """sample id -> prefix + name + suffix (the suffix goes before ".fcs")."""
    out = {}
    if not prefix and not suffix:
        return out
    for s in samples:
        name = s.name
        stem, ext = (name[:-4], name[-4:]) if before_extension and name.lower().endswith(".fcs") else (name, "")
        new = f"{prefix}{stem}{suffix}{ext}"
        if new != name:
            out[s.id] = new
    return out


def set_sample_names(controller, samples: list[Sample], mapping: dict[int, str], label: str = "rename samples") -> int:
    """Rename samples ({sample id: new name}) as one undo step; empty names
    are skipped. Returns the number renamed."""
    targets = []
    for s in samples:
        new = str(mapping.get(s.id, "") or "").strip()
        if new and new != s.name:
            targets.append((s, new))
    if not targets:
        return 0
    controller.begin_change(label, "samples")
    for s, new in targets:
        s.name = new
        s.root.name = new
    controller.mark_modified()
    controller.samplesChanged.emit()
    controller.statusMessage.emit(f"Renamed {len(targets)} sample{'s' if len(targets) != 1 else ''}")
    return len(targets)


def numeric_or_natural(text: str):
    """Sort key: numbers ("1,200", "2.5e3") by value before any text."""
    t = str(text).strip().replace(",", "")
    try:
        return (0, float(t), ())
    except ValueError:
        return (1, 0.0, natural_key(text))


def reorder_samples(controller, order: list[Sample], label: str) -> bool:
    """Make `order` the workspace's sample order (one undo step)."""
    ws = controller.workspace
    if len(order) != len(ws.samples) or all(a is b for a, b in zip(order, ws.samples)):
        return False
    controller.begin_change(label, "samples")
    ws.samples[:] = order
    controller.mark_modified()
    controller.samplesChanged.emit()
    return True


def sorted_samples(samples: list[Sample], key, descending: bool = False) -> list[Sample]:
    """Samples ordered by key(sample) (None = no value: always last, in their
    present order)."""
    have = [s for s in samples if key(s) is not None]
    missing = [s for s in samples if key(s) is None]
    have.sort(key=key, reverse=descending)
    return have + missing


def move_samples(controller, samples: list[Sample], offset: int, shown: list[Sample] | None = None) -> bool:
    """Move the samples one place up (offset < 0) or down among the samples
    shown (the current group's members), keeping every other sample's place
    in the workspace list. One undo step; False when nothing moved."""
    ws = controller.workspace
    shown = list(ws.samples if shown is None else shown)
    chosen = {id(s) for s in samples}
    order = list(shown)
    if offset < 0:
        for i in range(1, len(order)):
            if id(order[i]) in chosen and id(order[i - 1]) not in chosen:
                order[i - 1], order[i] = order[i], order[i - 1]
    else:
        for i in range(len(order) - 2, -1, -1):
            if id(order[i]) in chosen and id(order[i + 1]) not in chosen:
                order[i], order[i + 1] = order[i + 1], order[i]
    if all(a is b for a, b in zip(order, shown)):
        return False
    shown_ids = {id(s) for s in shown}
    slots = [k for k, s in enumerate(ws.samples) if id(s) in shown_ids]
    new = list(ws.samples)
    for k, s in zip(slots, order):
        new[k] = s
    return reorder_samples(controller, new, "move samples")


def display_name(sample: Sample, keyword: str) -> str:
    """The name the sample list shows: the keyword value, else the name."""
    if keyword:
        v = str(sample.keywords.get(keyword, "") or "").strip()
        if v:
            return v
    return sample.name


def sample_tooltip(sample: Sample, shown_name: str | None = None) -> str:
    kw = sample.keywords
    lines = []
    if sample.load_error:
        lines.append(f"File not found: {sample.path}")
    if shown_name is not None and shown_name != sample.name:
        lines.append(f"Sample: {sample.name}")
    lines.append(f"File: {sample.path}")
    if kw.get("$CYT"):
        lines.append(f"Cytometer: {kw['$CYT']}")
    date = " ".join(t for t in (kw.get("$DATE", ""), kw.get("$BTIM", "")) if t)
    if date:
        lines.append(f"Acquired: {date}")
    lines.append(f"Events: {sample.n_events:,}")
    duration = acquisition_duration(sample, use_data=False)  # keywords only: runs on every tree rebuild
    if duration:
        lines.append(f"Acquisition time: {format_duration(duration)} · {format_rate(sample.n_events / duration)}")
    for key in TOOLTIP_KEYWORDS:
        if kw.get(key):
            lines.append(f"{key}: {kw[key]}")
    return "\n".join(lines)


def reveal_command(path, platform: str | None = None) -> list[str] | None:
    """The command that shows the file selected in the system file manager,
    None where only the folder can be opened (Linux)."""
    platform = platform or sys.platform
    if platform == "darwin":
        return ["open", "-R", str(path)]
    if platform.startswith("win"):
        return ["explorer", f"/select,{Path(path)}"]
    return None


def reveal_label(platform: str | None = None) -> str:
    platform = platform or sys.platform
    if platform == "darwin":
        return "Reveal in Finder"
    if platform.startswith("win"):
        return "Show in Explorer"
    return "Show in File Manager"


def reveal_in_file_manager(path) -> bool:
    path = Path(path)
    cmd = reveal_command(path)
    if cmd is not None and path.exists():
        res = QProcess.startDetached(cmd[0], cmd[1:])
        if (res[0] if isinstance(res, tuple) else bool(res)):
            return True
    folder = path.parent if path.parent.exists() else path
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


# ===================================================================== dialogs
def _buttons(dialog: QDialog, ok_text: str) -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    box.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    return box


def _preview_table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    return t


class GroupByKeywordDialog(QDialog):
    """Create Groups by Keyword: one live group per distinct value."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.created: list[Group] = []
        self.setWindowTitle("Create Groups by Keyword")
        self.resize(480, 440)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.keyword_combo = QComboBox()
        self.keyword_combo.setEditable(True)
        self.keyword_combo.addItems(keyword_union(controller.workspace.samples))
        self.keyword_combo.setCurrentText("")
        self.keyword_combo.lineEdit().setPlaceholderText("e.g. Donor")
        self.pattern_edit = QLineEdit(GROUP_PATTERN)
        self.pattern_edit.setToolTip("Group name: {keyword} and {value} are replaced")
        self.colors = QCheckBox("Give each group its own colour")
        form.addRow("Keyword:", self.keyword_combo)
        form.addRow("Group name:", self.pattern_edit)
        form.addRow("", self.colors)
        layout.addLayout(form)
        layout.addWidget(QLabel("Groups (live: samples given the value later join them):"))
        self.preview = QListWidget()
        layout.addWidget(self.preview, 1)
        self.status = QLabel()
        layout.addWidget(self.status)
        layout.addWidget(_buttons(self, "Create"))
        self.keyword_combo.currentTextChanged.connect(lambda *_: self.update_preview())
        self.pattern_edit.textChanged.connect(lambda *_: self.update_preview())
        self.update_preview()

    def plan(self) -> list[tuple[str, str, int, bool]]:
        ws = self.controller.workspace
        return [(v, name, n, ws.group_by_name(name) is not None) for v, name, n in group_plan(ws.samples, self.keyword_combo.currentText().strip(), self.pattern_edit.text())]

    def update_preview(self):
        self.preview.clear()
        plan = self.plan() if self.keyword_combo.currentText().strip() else []
        new = 0
        for _v, name, n, exists in plan:
            self.preview.addItem(f"{name} — {n} sample{'s' if n != 1 else ''}" + (" (exists, skipped)" if exists else ""))
            new += not exists
        self.status.setText(f"{new} group{'s' if new != 1 else ''} will be created")

    def apply(self) -> list[Group]:
        self.created = create_groups_by_keyword(self.controller, self.keyword_combo.currentText(), self.pattern_edit.text(), self.colors.isChecked())
        return self.created

    def accept(self):
        if not self.keyword_combo.currentText().strip():
            return
        self.apply()
        super().accept()


class RenameSamplesDialog(QDialog):
    """Rename Samples: from a keyword, find / replace or prefix / suffix."""

    MODE_KEYWORD, MODE_REPLACE, MODE_AFFIX = 0, 1, 2

    def __init__(self, controller, samples: list[Sample], parent=None):
        super().__init__(parent)
        self.controller = controller
        self.samples = list(samples)
        self.setWindowTitle("Rename Samples")
        self.resize(620, 560)
        layout = QVBoxLayout(self)
        self.mode_group = QButtonGroup(self)
        self.rb_keyword = QRadioButton("Name from a keyword")
        self.rb_replace = QRadioButton("Find and replace")
        self.rb_affix = QRadioButton("Add a prefix / suffix")
        for k, rb in enumerate((self.rb_keyword, self.rb_replace, self.rb_affix)):
            self.mode_group.addButton(rb, k)
        self.rb_keyword.setChecked(True)
        self.keyword_combo = QComboBox()
        self.keyword_combo.setEditable(True)
        keys = keyword_union(self.samples)
        preferred = [k for k in ("TUBE NAME", "$SRC", "WELL ID", "$FIL") if k in keys]
        self.keyword_combo.addItems(preferred + [k for k in keys if k not in preferred])
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find")
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Replace with")
        self.regex_check = QCheckBox("Regular expression (\\1 … in the replacement)")
        self.case_check = QCheckBox("Match case")
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Prefix")
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setPlaceholderText("Suffix (before .fcs)")
        layout.addWidget(self.rb_keyword)
        row = QHBoxLayout()
        row.addWidget(QLabel("Keyword:"))
        row.addWidget(self.keyword_combo, 1)
        layout.addLayout(row)
        layout.addWidget(self.rb_replace)
        row = QHBoxLayout()
        row.addWidget(self.find_edit, 1)
        row.addWidget(self.replace_edit, 1)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.regex_check)
        row.addWidget(self.case_check)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self.rb_affix)
        row = QHBoxLayout()
        row.addWidget(self.prefix_edit, 1)
        row.addWidget(self.suffix_edit, 1)
        layout.addLayout(row)
        layout.addWidget(QLabel(f"Preview ({len(self.samples)} samples):"))
        self.preview = _preview_table(["Current name", "New name"])
        layout.addWidget(self.preview, 1)
        self.status = QLabel()
        layout.addWidget(self.status)
        layout.addWidget(_buttons(self, "Rename"))
        self.keyword_combo.currentTextChanged.connect(lambda *_: self._edited(self.MODE_KEYWORD))
        for w in (self.find_edit, self.replace_edit):
            w.textChanged.connect(lambda *_: self._edited(self.MODE_REPLACE))
        for w in (self.prefix_edit, self.suffix_edit):
            w.textChanged.connect(lambda *_: self._edited(self.MODE_AFFIX))
        for cb in (self.regex_check, self.case_check):
            cb.toggled.connect(lambda *_: self._edited(self.MODE_REPLACE))
        self.mode_group.idToggled.connect(lambda *_: self.update_preview())
        self.update_preview()

    def _edited(self, mode: int):
        if self.mode() != mode:
            self.set_mode(mode)  # toggling the radio button refreshes the preview
        else:
            self.update_preview()

    def set_mode(self, mode: int):
        self.mode_group.button(mode).setChecked(True)

    def mode(self) -> int:
        return self.mode_group.checkedId()

    def new_names(self) -> dict[int, str]:
        mode = self.mode()
        if mode == self.MODE_KEYWORD:
            return names_from_keyword(self.samples, self.keyword_combo.currentText().strip())
        if mode == self.MODE_REPLACE:
            try:
                return names_replace(self.samples, self.find_edit.text(), self.replace_edit.text(), self.regex_check.isChecked(), self.case_check.isChecked())
            except (re.error, IndexError) as exc:
                self.status.setText(f"Invalid expression: {exc}")
                return {}
        return names_affix(self.samples, self.prefix_edit.text(), self.suffix_edit.text())

    def update_preview(self):
        self.status.setText("")
        names = self.new_names()
        self.preview.setRowCount(len(self.samples))
        for row, s in enumerate(self.samples):
            self.preview.setItem(row, 0, QTableWidgetItem(s.name))
            self.preview.setItem(row, 1, QTableWidgetItem(names.get(s.id, "(unchanged)")))
        if not self.status.text():
            final = [names.get(s.id, s.name) for s in self.samples]
            dupes = len(final) - len(set(final))
            msg = f"{len(names)} of {len(self.samples)} samples will be renamed"
            if dupes:
                msg += f" · {dupes} duplicate name{'s' if dupes != 1 else ''}"
            self.status.setText(msg)

    def apply(self) -> int:
        return set_sample_names(self.controller, self.samples, self.new_names())

    def accept(self):
        self.apply()
        super().accept()


class NameByKeywordDialog(QDialog):
    """Name Samples by Keyword: the keyword whose value the sample list shows."""

    FILE_NAME = "(sample name)"

    def __init__(self, controller, current: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Name Samples by Keyword")
        form = QFormLayout(self)
        self.keyword_combo = QComboBox()
        self.keyword_combo.setEditable(True)
        keys = keyword_union(controller.workspace.samples)
        preferred = [k for k in ("TUBE NAME", "$SRC", "$FIL", "WELL ID") if k in keys]
        self.keyword_combo.addItems([self.FILE_NAME] + preferred + [k for k in keys if k not in preferred])
        self.keyword_combo.setCurrentText(current or self.FILE_NAME)
        form.addRow(QLabel("The sample list shows each sample under this keyword's value\n(samples without it keep their name; the names themselves do not change)."))
        form.addRow("Show samples as:", self.keyword_combo)
        form.addRow(_buttons(self, "OK"))

    def keyword(self) -> str:
        text = self.keyword_combo.currentText().strip()
        return "" if text == self.FILE_NAME else text


# ================================================================== extension
class WorkspaceToolsExtension(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.controller = window.controller
        self.tree = window.tree
        self.filter_text = ""
        self.sort_column = -1  # header section the samples are sorted by
        self.sort_order = Qt.SortOrder.AscendingOrder
        self._applied_name_keyword = None
        self._applying = False
        self._viewport = None

        def act(text, icon_name, slot, tip, shortcut=None):
            a = QAction(icon(icon_name), text, window) if icon_name else QAction(text, window)
            a.setStatusTip(tip)
            a.setToolTip(tip)
            if shortcut is not None:
                a.setShortcut(QKeySequence(shortcut))
                window.addAction(a)
            a.triggered.connect(lambda _checked=False: slot())
            return a

        self.act_find = act("Find…", "zoom", self.focus_find, "Filter the sample list by sample, population or keyword text", QKeySequence.StandardKey.Find)
        self.act_select_equivalent = act("Select Equivalent", "check", self.select_equivalent, "Select the same population in every sample shown", "Ctrl+E")
        self.act_group_by_keyword = act("Create Groups by Keyword…", "group", self.group_by_keyword, "Create one live group per distinct value of a keyword")
        self.act_rename_samples = act("Rename Samples…", "rename", self.rename_samples, "Rename the selected samples from a keyword, by find / replace or with a prefix / suffix")
        self.act_name_by_keyword = act("Name Samples by Keyword…", "keyword", self.name_by_keyword, "Show each sample under a keyword value (e.g. TUBE NAME) in the sample list")
        self.act_move_up = act("Move Up", "up", lambda: self.move_selected(-1), "Move the selected samples up", "Ctrl+Shift+Up")
        self.act_move_down = act("Move Down", "down", lambda: self.move_selected(1), "Move the selected samples down", "Ctrl+Shift+Down")
        self.act_duplicate = act("Duplicate", "copy", self.duplicate_selected, "Copy the selected gates next to themselves (same parent)", "Ctrl+D")
        self.act_copy_text = act("Copy as Text", "copy", self.copy_rows_as_text, "Copy the selected rows (name, statistic, count, keywords) as tab-separated text", "Ctrl+Shift+C")
        self.act_reveal = act(reveal_label(), "folder", self.reveal_selected, "Show the selected sample's FCS file in the file manager")
        self.act_copy_path = act("Copy File Path", "copy", self.copy_selected_paths, "Copy the selected samples' file paths to the clipboard")
        self.actions = [
            self.act_find, self.act_select_equivalent, self.act_group_by_keyword, self.act_rename_samples, self.act_name_by_keyword,
            self.act_move_up, self.act_move_down, self.act_duplicate, self.act_copy_text, self.act_reveal, self.act_copy_path,
        ]
        for a, short in ((self.act_group_by_keyword, "Groups by Keyword"), (self.act_rename_samples, "Rename Samples"), (self.act_name_by_keyword, "Name by Keyword")):
            a.setIconText(short)

        menus = getattr(window, "menus", {})
        if "workspace" in menus:
            m = menus["workspace"]
            m.addSeparator()
            for a in (self.act_find, self.act_select_equivalent, self.act_group_by_keyword, self.act_rename_samples, self.act_name_by_keyword, self.act_move_up, self.act_move_down, self.act_reveal, self.act_copy_path):
                m.addAction(a)
        if "edit" in menus:
            m = menus["edit"]
            prefs = getattr(window, "act_prefs", None)
            items = (self.act_duplicate, self.act_copy_text, self.act_select_equivalent, self.act_find)
            if prefs is not None and prefs in m.actions():
                for a in items:
                    m.insertAction(prefs, a)
                m.insertSeparator(prefs)
            else:
                for a in items:
                    m.addAction(a)
        pages = getattr(window, "ribbon_pages", {})
        self.band = None
        if "workspace" in pages:
            self.band = pages["workspace"].add_band("Sample List")
            self.band.add_large(self.act_find)
            for a in (self.act_rename_samples, self.act_name_by_keyword, self.act_group_by_keyword, self.act_move_up, self.act_move_down, self.act_select_equivalent):
                self.band.add_small(a)

        self._build_find_bar()
        header = self.tree.header()
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self.header_clicked)
        self._viewport = self.tree.viewport()
        self._viewport.installEventFilter(self)
        self.tree.installEventFilter(self)
        self.tree.itemChanged.connect(self._item_changed)
        if hasattr(self.tree, "rebuilt"):
            self.tree.rebuilt.connect(self.after_rebuild)
        hooks = getattr(window, "tree_context_hooks", None)
        if hooks is not None:
            hooks.append(self._context_hook)
        c = self.controller
        c.selectionChanged.connect(self.update_states)
        c.samplesChanged.connect(self.update_states)
        c.workspaceReplaced.connect(self._workspace_replaced)
        c.settingsChanged.connect(self._settings_changed)
        self.update_states()

    # ------------------------------------------------------------- find bar
    def _build_find_bar(self):
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 2)
        row.setSpacing(4)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find samples and populations (Ctrl+F)")
        self.find_edit.setClearButtonEnabled(True)
        self.find_edit.addAction(icon("zoom"), QLineEdit.ActionPosition.LeadingPosition)
        self.find_edit.textChanged.connect(self.set_filter)
        self.find_edit.installEventFilter(self)
        self.match_label = QLabel()
        self.match_label.setStyleSheet("color: #6b7280;")
        row.addWidget(self.find_edit, 1)
        row.addWidget(self.match_label)
        self.find_bar = bar
        splitter = getattr(self.window, "splitter", None)
        idx = splitter.indexOf(self.tree) if splitter is not None else -1
        if idx < 0:
            return
        sizes = splitter.sizes()
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(bar)
        splitter.replaceWidget(idx, container)
        lay.addWidget(self.tree, 1)
        self.tree.show()
        splitter.setStretchFactor(idx, 1)
        splitter.setSizes(sizes)
        self.tree_container = container

    def focus_find(self):
        self.find_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.find_edit.selectAll()

    def set_filter(self, text: str):
        if self.find_edit.text() != text:
            self.find_edit.setText(text)  # comes back here through textChanged
            return
        self.filter_text = text.strip()
        self.apply_filter()

    def _expanded_in_model(self, item) -> bool:
        _s, ROLE_POP, ROLE_STAT, ROLE_GROUP = _roles()
        pop = item.data(0, ROLE_POP)
        group = item.data(0, ROLE_GROUP)
        if group is not None and pop is group.root:
            return bool(getattr(group, "expanded", True))
        return bool(getattr(pop, "expanded", False)) if pop is not None else item.isExpanded()

    def apply_filter(self) -> int:
        """Hide the rows that neither match the find text nor lie above or
        below a match; returns the number of matching rows."""
        tree = self.tree
        text = self.filter_text.lower()
        cols = [0] + list(range(FIRST_KEYWORD_COLUMN, tree.columnCount()))
        matches = 0

        def visit(item, under_match: bool) -> bool:
            nonlocal matches
            hit = bool(text) and any(text in item.text(c).lower() for c in cols)
            matches += hit
            below = False
            for k in range(item.childCount()):
                below |= visit(item.child(k), under_match or hit)
            if not text:
                item.setHidden(False)
                if item.childCount():
                    item.setExpanded(self._expanded_in_model(item))
                return False
            item.setHidden(not (hit or below or under_match))
            if below:
                item.setExpanded(True)  # open the way to the match (not saved)
            return hit or below

        tree.blockSignals(True)  # expanding here must not change the saved state
        try:
            for i in range(tree.topLevelItemCount()):
                visit(tree.topLevelItem(i), False)
        finally:
            tree.blockSignals(False)
        self.match_label.setText(f"{matches} match{'es' if matches != 1 else ''}" if text else "")
        return matches

    def visible_items(self) -> list:
        out = []
        it = QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            hidden = False
            p = item
            while p is not None:
                hidden |= p.isHidden()
                p = p.parent()
            if not hidden:
                out.append(item)
            it += 1
        return out

    # ------------------------------------------------------------- rebuild
    def after_rebuild(self):
        """Re-apply what a tree rebuild resets: display names, tooltips, the
        sort indicator and the find filter."""
        self._apply_names_and_tips()
        self._apply_indicator()
        if self.filter_text:
            self.apply_filter()

    def name_keyword(self) -> str:
        ws = self.controller.workspace
        if NAME_SETTING in ws.settings:
            return str(ws.settings.get(NAME_SETTING) or "")
        prefs = getattr(self.controller, "preferences", None)
        return str(prefs.get(NAME_PREFERENCE, "") or "") if prefs is not None else ""

    def _sample_rows(self):
        ROLE_SAMPLE, ROLE_POP, ROLE_STAT, ROLE_GROUP = _roles()
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            s = item.data(0, ROLE_SAMPLE)
            if s is not None and item.data(0, ROLE_GROUP) is None and item.data(0, ROLE_STAT) is None and item.data(0, ROLE_POP) is s.root:
                yield item, s

    def _apply_names_and_tips(self):
        keyword = self.name_keyword()
        self._applied_name_keyword = keyword
        self._applying = True
        self.tree.blockSignals(True)  # a display name is not a rename
        try:
            for item, s in self._sample_rows():
                shown = display_name(s, keyword)
                if shown != item.text(0):
                    item.setText(0, shown)
                item.setToolTip(0, sample_tooltip(s, shown))
        finally:
            self.tree.blockSignals(False)
            self._applying = False

    def _settings_changed(self):
        # undo / redo of Name Samples by Keyword
        if self.name_keyword() != self._applied_name_keyword:
            self.tree.rebuild()

    def _workspace_replaced(self):
        self.sort_column = -1
        self.update_states()

    # ------------------------------------------------------------- sorting
    def _sort_key(self, column: int):
        keyword = self.name_keyword()
        if column == 0:
            return lambda s: natural_key(display_name(s, keyword))
        if column == 2:
            return lambda s: s.n_events
        cols = self.tree.keyword_columns
        k = column - FIRST_KEYWORD_COLUMN
        if 0 <= k < len(cols):
            key = cols[k]

            def value(s):
                v = str(s.keywords.get(key, "") or "").strip()
                return numeric_or_natural(v) if v else None

            return value
        return None

    def _column_label(self, column: int) -> str:
        item = self.tree.headerItem()
        return item.text(column) if item is not None else str(column)

    def sort_by_column(self, column: int, order: Qt.SortOrder | None = None) -> bool:
        """Sort the workspace's samples by a sample-list column (Name,
        #Cells or a keyword column). Without `order`, the same column again
        reverses the order. Returns True when the order changed."""
        key = self._sort_key(column)
        if key is None:
            self._apply_indicator()  # undo the header's own flip
            return False
        if order is None:
            same = column == self.sort_column
            order = (Qt.SortOrder.DescendingOrder if self.sort_order == Qt.SortOrder.AscendingOrder else Qt.SortOrder.AscendingOrder) if same else Qt.SortOrder.AscendingOrder
        self.sort_column, self.sort_order = column, order
        ws = self.controller.workspace
        new = sorted_samples(ws.samples, key, order == Qt.SortOrder.DescendingOrder)
        arrow = "descending" if order == Qt.SortOrder.DescendingOrder else "ascending"
        changed = reorder_samples(self.controller, new, f"sort samples by {self._column_label(column)} ({arrow})")
        self._apply_indicator()
        return changed

    def header_clicked(self, column: int):
        self.sort_by_column(column)

    def _is_sorted(self) -> bool:
        key = self._sort_key(self.sort_column) if self.sort_column >= 0 else None
        if key is None:
            return False
        ws = self.controller.workspace
        expect = sorted_samples(ws.samples, key, self.sort_order == Qt.SortOrder.DescendingOrder)
        values = [key(s) for s in ws.samples]
        return [key(s) for s in expect] == values

    def _apply_indicator(self):
        header = self.tree.header()
        if self.sort_column >= 0 and self.sort_column < self.tree.columnCount() and self._is_sorted():
            header.setSortIndicatorShown(True)
            header.setSortIndicator(self.sort_column, self.sort_order)
        else:
            self.sort_column = -1
            header.setSortIndicatorShown(False)

    def clear_sort_indicator(self):
        self.sort_column = -1
        self._apply_indicator()

    # ------------------------------------------------------------ commands
    def _shown_samples(self) -> list[Sample]:
        c = self.controller
        return c.workspace.members(c.current_group)

    def _target_samples(self) -> list[Sample]:
        sel = self.controller.selected_samples()
        return sel if sel else self._shown_samples()

    def move_selected(self, offset: int) -> bool:
        samples = self.controller.selected_samples()
        if not samples:
            return False
        moved = move_samples(self.controller, samples, offset, self._shown_samples())
        if moved:
            self.clear_sort_indicator()
        return moved

    def select_equivalent(self) -> int:
        """Select the rows of the same population path (of the selected
        populations) in every sample shown; returns the rows selected."""
        ROLE_SAMPLE, ROLE_POP, ROLE_STAT, ROLE_GROUP = _roles()
        tree = self.tree
        paths = {p.path for _s, p in tree.selected_rows() if p.parent is not None}
        paths |= {p.path for _g, p in tree.selected_group_rows() if p.parent is not None}
        if not paths:
            return 0
        items = []
        it = QTreeWidgetItemIterator(tree)
        while it.value():
            item = it.value()
            s, p = item.data(0, ROLE_SAMPLE), item.data(0, ROLE_POP)
            if s is not None and p is not None and item.data(0, ROLE_STAT) is None and p.path in paths:
                items.append(item)
            it += 1
        tree.blockSignals(True)
        try:
            for item in items:
                item.setSelected(True)
                parent = item.parent()
                while parent is not None:
                    pop = parent.data(0, ROLE_POP)
                    if pop is not None and parent.data(0, ROLE_GROUP) is None:
                        pop.expanded = True
                    parent.setExpanded(True)
                    parent = parent.parent()
        finally:
            tree.blockSignals(False)
        tree.itemSelectionChanged.emit()
        if items:
            tree.scrollToItem(items[0])
        return len(items)

    def group_by_keyword(self, keyword: str | None = None, pattern: str = GROUP_PATTERN, colors: bool = False):
        if keyword is not None:
            return create_groups_by_keyword(self.controller, keyword, pattern, colors)
        dlg = GroupByKeywordDialog(self.controller, self.window)
        dlg.exec()
        return dlg.created

    def rename_samples(self):
        samples = self._target_samples()
        if not samples:
            return None
        dlg = RenameSamplesDialog(self.controller, samples, self.window)
        dlg.exec()
        return dlg

    def set_name_keyword(self, keyword: str):
        """Show samples under `keyword` ("" = their names) in this workspace (undoable)."""
        ws = self.controller.workspace
        keyword = (keyword or "").strip()
        if NAME_SETTING in ws.settings and ws.settings[NAME_SETTING] == keyword:
            return
        self.controller.begin_change("name samples by keyword", "settings")
        ws.settings[NAME_SETTING] = keyword
        self.controller.mark_modified()
        self.tree.rebuild()

    def name_by_keyword(self):
        dlg = NameByKeywordDialog(self.controller, self.name_keyword(), self.window)
        if dlg.exec():
            self.set_name_keyword(dlg.keyword())
        return dlg

    def duplicate_selected(self) -> int:
        """Copy each selected gate next to itself ("CD3+ 2"); group-template
        gates are duplicated for the whole group. One undo step."""
        c = self.controller
        pops = [p for _s, p in self.tree.selected_rows() if p.parent is not None]
        gpops = [(g, p) for g, p in self.tree.selected_group_rows() if p.parent is not None]
        if not pops and not gpops:
            return 0
        with c.undo_group("duplicate gates"):
            for p in pops:
                c.paste_populations([p], p.parent, None)
            for g, p in gpops:
                c.paste_populations([p], p.parent, g)
        return len(pops) + len(gpops)

    def rows_as_text(self, items=None) -> str:
        """Tab-separated text of tree rows (the selection, in tree order):
        sample, population path, row name, statistic, count, keyword columns."""
        ROLE_SAMPLE, ROLE_POP, ROLE_STAT, ROLE_GROUP = _roles()
        tree = self.tree
        if items is None:
            items = [it for it in self._all_items() if it.isSelected()]
        kw = list(tree.keyword_columns)
        lines = ["\t".join(["Sample", "Population", "Name", "Statistic", "#Cells"] + kw)]
        for item in items:
            s, g, p = item.data(0, ROLE_SAMPLE), item.data(0, ROLE_GROUP), item.data(0, ROLE_POP)
            owner = s.name if s is not None else (g.name if g is not None else "")
            path = p.path.strip("/") if p is not None else ""
            count = item.text(2).replace(",", "")
            cells = [owner, path, item.text(0), item.text(1), count]
            cells += [s.keywords.get(k, "") if s is not None else "" for k in kw]
            lines.append("\t".join(str(c).replace("\t", " ").replace("\n", " ") for c in cells))
        return "\n".join(lines) + "\n"

    def _all_items(self) -> list:
        out = []
        it = QTreeWidgetItemIterator(self.tree)
        while it.value():
            out.append(it.value())
            it += 1
        return out

    def copy_rows_as_text(self) -> str:
        if not self.tree.selectedItems():
            return ""
        text = self.rows_as_text()
        QGuiApplication.clipboard().setText(text)
        self.controller.statusMessage.emit(f"Copied {text.count(chr(10)) - 1} rows as text")
        return text

    def copy_selected_paths(self) -> str:
        samples = self.controller.selected_samples()
        if not samples:
            return ""
        text = "\n".join(str(s.path) for s in samples)
        QGuiApplication.clipboard().setText(text)
        return text

    def reveal_selected(self) -> bool:
        samples = self.controller.selected_samples()
        if not samples:
            return False
        return reveal_in_file_manager(samples[0].path)

    # ------------------------------------------------------- keyword cells
    def _keyword_cell(self, item, column: int) -> tuple[Sample, str] | None:
        """(sample, keyword) when (item, column) is an editable keyword cell."""
        if item is None or column < FIRST_KEYWORD_COLUMN:
            return None
        ROLE_SAMPLE, ROLE_POP, ROLE_STAT, ROLE_GROUP = _roles()
        s = item.data(0, ROLE_SAMPLE)
        if s is None or item.data(0, ROLE_GROUP) is not None or item.data(0, ROLE_STAT) is not None or item.data(0, ROLE_POP) is not s.root:
            return None
        cols = self.tree.keyword_columns
        k = column - FIRST_KEYWORD_COLUMN
        if not 0 <= k < len(cols) or cols[k].startswith("$"):
            return None  # FCS ($) keywords stay read-only, as in Edit Keywords
        return s, cols[k]

    def edit_keyword_cell(self, item, column: int) -> bool:
        if self._keyword_cell(item, column) is None:
            return False
        self.tree.setCurrentItem(item, column)
        self.tree.editItem(item, column)
        return True

    def _item_changed(self, item, column: int):
        if self._applying or getattr(self.tree, "_building", False):
            return
        cell = self._keyword_cell(item, column)
        if cell is None:
            return
        sample, key = cell
        value = item.text(column).strip()
        if value == str(sample.keywords.get(key, "")):
            return
        # after the editor has closed: writing the keyword rebuilds the tree
        QTimer.singleShot(0, lambda: self.set_cell_keyword(sample, key, value))

    def set_cell_keyword(self, sample: Sample, key: str, value: str) -> bool:
        if sample not in self.controller.workspace.samples or value == str(sample.keywords.get(key, "")):
            return False
        return set_keyword(self.controller, [sample], key, value) > 0

    _FILTERED = (QEvent.Type.KeyPress, QEvent.Type.ShortcutOverride, QEvent.Type.MouseButtonDblClick)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        try:
            et = event.type()
            if et not in self._FILTERED:
                return False
            return self._filter_event(obj, event, et)
        except RuntimeError:  # the tree is being destroyed
            return False

    def _filter_event(self, obj, event, et) -> bool:
        if obj is self.find_edit and et == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            self.find_edit.clear()
            self.tree.setFocus()
            return True
        if obj is self._viewport and et == QEvent.Type.MouseButtonDblClick:
            pos = event.position().toPoint()
            item = self.tree.itemAt(pos)
            column = self.tree.columnAt(pos.x())
            if self.edit_keyword_cell(item, column):
                return True  # edit the keyword instead of opening a graph
        if obj is self.tree and et in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress) and event.key() == Qt.Key.Key_F2:
            item = self.tree.currentItem()
            column = self.tree.currentColumn()
            if self.tree.state() != QAbstractItemView.State.EditingState and self._keyword_cell(item, column) is not None:
                event.accept()
                if et == QEvent.Type.KeyPress:
                    self.edit_keyword_cell(item, column)
                return True
        return False

    # ------------------------------------------------------- context menu
    def _context_hook(self, menu, item):
        if item is None:
            return
        ROLE_SAMPLE, ROLE_POP, ROLE_STAT, ROLE_GROUP = _roles()
        s, p, st, g = item.data(0, ROLE_SAMPLE), item.data(0, ROLE_POP), item.data(0, ROLE_STAT), item.data(0, ROLE_GROUP)
        menu.addSeparator()
        if st is None and p is not None and p.parent is not None:
            menu.addAction(self.act_select_equivalent)
            menu.addAction(self.act_duplicate)
        elif s is not None and g is None and st is None:
            menu.addAction(self.act_move_up)
            menu.addAction(self.act_move_down)
            menu.addAction(self.act_rename_samples)
            menu.addAction(self.act_name_by_keyword)
            menu.addAction(self.act_reveal)
            menu.addAction(self.act_copy_path)
        menu.addAction(self.act_copy_text)

    # -------------------------------------------------------------- states
    def update_states(self):
        c = self.controller
        samples = c.selected_samples()
        has_samples = bool(c.workspace.samples)
        pops = [p for _s, p in c.selected if p.parent is not None]
        gpops = [p for _g, p in self.tree.selected_group_rows() if p.parent is not None]
        self.act_select_equivalent.setEnabled(bool(pops or gpops))
        self.act_duplicate.setEnabled(bool(pops or gpops))
        self.act_copy_text.setEnabled(bool(self.tree.selectedItems()))
        self.act_move_up.setEnabled(bool(samples))
        self.act_move_down.setEnabled(bool(samples))
        self.act_reveal.setEnabled(len(samples) == 1)
        self.act_copy_path.setEnabled(bool(samples))
        self.act_rename_samples.setEnabled(has_samples)
        self.act_group_by_keyword.setEnabled(has_samples)
        self.act_name_by_keyword.setEnabled(has_samples)


def install(window) -> WorkspaceToolsExtension:
    return WorkspaceToolsExtension(window)
