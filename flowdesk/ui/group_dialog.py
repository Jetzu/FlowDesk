"""Create / Edit Group dialog with FlowJo's group role, live membership and
font style controls, plus the Boolean gate dialog variant that can build the
gate for every sample of a group.

Group role strings: "all" (All Samples), "compensation", "test", "control"
and "custom" (a group made before roles could be picked; shown and exported
as Test). The font style lives in Group.bold and Group.italic (italic is set
as an attribute when the model has no such field yet).
"""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QLayout, QVBoxLayout

from ..core.model import Group, Population, Sample, Workspace
from .dialogs import BooleanGateDialog
from .dialogs import GroupDialog as _BaseGroupDialog

GROUP_ROLES = (("Test", "test"), ("Control", "control"), ("Compensation", "compensation"))
FONT_STYLES = (("Plain", "plain"), ("Bold", "bold"), ("Italic", "italic"))
ROLE_LABELS = {"all": "All Samples", "test": "Test", "control": "Control", "compensation": "Compensation", "custom": "Test"}


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role.title() if role else "Test")


def group_font_style(group: Group) -> str:
    if group.bold:
        return "bold"
    return "italic" if getattr(group, "italic", False) else "plain"


def set_group_font_style(group: Group, style: str):
    group.bold = style == "bold"
    group.italic = style == "italic"


def group_font(group: Group, base: QFont | None = None, force_bold: bool = False) -> QFont:
    """The font a group row is drawn with (All Samples is always bold)."""
    font = QFont(base) if base is not None else QFont()
    font.setBold(force_bold or group.bold or group.role == "all")
    font.setItalic(bool(getattr(group, "italic", False)))
    return font


def set_group_live(ws: Workspace, group: Group, live: bool) -> bool:
    """Make a keyword group live or freeze its membership. Freezing keeps the
    samples that match the criteria now (as manual members), so the group
    does not lose them; going live again drops the manual entries the
    criteria cover. Returns True when something changed."""
    live = bool(live)
    if group.role == "all" or group.live == live:
        return False
    if not live:
        group.sample_ids |= {s.id for s in ws.samples if group.matches_criteria(s)}
    else:
        group.sample_ids -= {s.id for s in ws.samples if group.matches_criteria(s)}
    group.live = live
    return True


def _find_layout(layout: QLayout, kind):
    if isinstance(layout, kind):
        return layout
    for i in range(layout.count()):
        child = layout.itemAt(i).layout()
        if child is not None:
            found = _find_layout(child, kind)
            if found is not None:
                return found
    return None


class GroupDialog(_BaseGroupDialog):
    """The shared Create / Edit Group dialog plus Role, Style and Live."""

    def __init__(self, samples: list[Sample], keywords: list[str], group: Group | None = None, parent=None):
        super().__init__(samples, keywords, group, parent)
        self._group = group
        self._initial_role = group.role if group is not None else "custom"
        form = _find_layout(self.layout(), QFormLayout)
        if form is None:  # the base dialog changed: keep the controls reachable
            form = QFormLayout()
            self.layout().insertLayout(0, form)
        self.role_combo = QComboBox()
        for label, value in GROUP_ROLES:
            self.role_combo.addItem(label, value)
        initial = "test" if self._initial_role in ("custom", "all") else self._initial_role
        idx = self.role_combo.findData(initial)
        self.role_combo.setCurrentIndex(max(0, idx))
        self.role_combo.setToolTip("Test and Control groups hold samples; the Compensation group's samples are the controls the Compensation Wizard uses")
        form.addRow("Role:", self.role_combo)
        self.style_combo = QComboBox()
        for label, value in FONT_STYLES:
            self.style_combo.addItem(label, value)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(group_font_style(group) if group is not None else "plain")))
        form.addRow("Style:", self.style_combo)
        self.live_check = QCheckBox("Live (membership follows the keyword criteria)")
        self.live_check.setChecked(group.live if group is not None else True)
        self.live_check.setToolTip("Untick to freeze the group: samples that match the criteria now stay members, samples added later do not join")
        box = next((b for b in self.findChildren(QGroupBox) if "keyword" in b.title().lower()), None)
        box_layout = box.layout() if box is not None else None
        if isinstance(box_layout, QVBoxLayout):
            box_layout.insertWidget(0, self.live_check)
        else:
            form.addRow("", self.live_check)

    def role(self) -> str:
        value = self.role_combo.currentData() or "test"
        # an untouched Test choice keeps a "custom" group as it was
        if value == "test" and self._initial_role == "custom":
            return "custom"
        return value

    def font_style(self) -> str:
        return self.style_combo.currentData() or "plain"

    def live(self) -> bool:
        return self.live_check.isChecked()

    def extra_values(self) -> dict:
        return {"role": self.role(), "style": self.font_style(), "live": self.live()}

    def apply_extras(self, group: Group, ws: Workspace | None = None):
        """Write role and font style into `group`, and the live flag (through
        set_group_live when `ws` is given, so freezing keeps the members)."""
        if group.role != "all":
            group.role = self.role()
        set_group_font_style(group, self.font_style())
        if ws is not None:
            set_group_live(ws, group, self.live())
        elif group.role != "all":
            group.live = self.live()


class GroupBooleanGateDialog(BooleanGateDialog):
    """BooleanGateDialog with an "Add to all samples in <group>" option."""

    def __init__(self, candidates: list[Population], op: str = "and", group: Group | None = None, checked: bool = False, parent=None, locked: bool = False):
        super().__init__(candidates, op, parent)
        self.group = group
        self.group_check = QCheckBox(f"Add to all samples in “{group.name}”" if group is not None else "Add to all samples in the group")
        self.group_check.setChecked(bool(checked) and group is not None)
        self.group_check.setEnabled(group is not None and not locked)
        layout = self.layout()
        layout.insertWidget(max(0, layout.count() - 1), self.group_check)

    def apply_to_group(self) -> bool:
        return self.group is not None and self.group_check.isChecked()
