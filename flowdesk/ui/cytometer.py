"""Cytometer settings: default axis scales per instrument.

Samples are grouped by the cytometer that acquired them ($CYT keyword). For
each cytometer the dialog lists the union of its samples' parameters with a
default scale (Linear / Biexponential / Logarithmic / Arcsinh, width basis and
positive decades). "Apply" writes the scales to every sample of that cytometer
and remembers them in ``workspace.settings["cytometers"][name]`` so samples of
the same instrument added later receive them too (``apply_cytometer_defaults``).
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.transforms import ArcsinhScale, LinearScale, LogicleScale, LogScale, Scale
from .icons import icon

UNKNOWN = "Unknown"
SETTINGS_KEY = "cytometers"
SCALE_CHOICES = [("linear", "Linear"), ("logicle", "Biexponential"), ("log", "Logarithmic"), ("arcsinh", "Arcsinh")]
DEFAULT_TOP = 262144.0

COL_PARAM, COL_STAIN, COL_KIND, COL_WIDTH, COL_DECADES = range(5)


# ------------------------------------------------------------------- model side
def cytometer_of(sample) -> str:
    """The instrument a sample was acquired on ($CYT), "Unknown" when absent."""
    kw = getattr(sample, "keywords", None) or {}
    value = kw.get("$CYT")
    if value is None:
        value = next((v for k, v in kw.items() if str(k).upper() == "$CYT"), None)
    value = str(value).strip() if value is not None else ""
    return value or UNKNOWN


def group_by_cytometer(workspace) -> dict[str, list]:
    out: dict[str, list] = {}
    for s in workspace.samples:
        out.setdefault(cytometer_of(s), []).append(s)
    return out


def _top(sample, param: str) -> float:
    ch = sample.base_channel(param)
    return float(ch.range) if ch is not None and ch.range and ch.range > 1 else DEFAULT_TOP


def _comp_names_for(sample, param: str) -> list[str]:
    """Compensated parameters of `sample` whose base channel is `param`."""
    comp = sample._comp_active() if hasattr(sample, "_comp_active") else None
    if comp is None:
        return []
    return [n for n in comp.compensated_names if comp.base_name(n) == param and n != param]


def scale_to_spec(scale: Scale) -> dict:
    """The editable description of a scale used by the dialog and the settings."""
    if isinstance(scale, LogicleScale):
        return {"kind": "logicle", "width_basis": float(scale.width_basis), "decades": float(scale.M), "neg": float(scale.A)}
    if isinstance(scale, LogScale):
        return {"kind": "log", "decades": float(round(scale.decades, 3)) if scale.decades > 0 else 4.0}
    if isinstance(scale, ArcsinhScale):
        return {"kind": "arcsinh", "decades": float(scale.M), "neg": float(scale.A)}
    if isinstance(scale, LinearScale):
        return {"kind": "linear", "min": float(scale.min)}
    return {"kind": "linear", "min": 0.0}


def scale_from_spec(spec: dict, top: float = DEFAULT_TOP) -> Scale:
    """Build a scale for a parameter whose range is `top` from a spec."""
    top = float(top) if top and top > 0 else DEFAULT_TOP
    kind = spec.get("kind", "linear")
    decades = float(spec.get("decades") or max(3.0, math.log10(top) - 1.0))
    if kind == "logicle":
        wb = float(spec.get("width_basis", -100.0) or -100.0)
        return LogicleScale.from_biex(top, decades, float(spec.get("neg", 0.0) or 0.0), -abs(wb))
    if kind == "log":
        return LogScale(max(1e-3, top / 10 ** decades), top)
    if kind == "arcsinh":
        return ArcsinhScale(top, decades, float(spec.get("neg", 0.0) or 0.0))
    return LinearScale(float(spec.get("min", 0.0) or 0.0), top)


def cytometer_settings(workspace) -> dict:
    return workspace.settings.setdefault(SETTINGS_KEY, {})


def apply_cytometer_defaults(workspace, sample) -> int:
    """Give `sample` the stored default scales of its cytometer. Returns the number
    of parameters set (0 when no defaults are stored for that instrument)."""
    specs = (workspace.settings.get(SETTINGS_KEY) or {}).get(cytometer_of(sample))
    if not specs:
        return 0
    names = set(sample.channel_names)
    n = 0
    for param, spec in specs.items():
        if param not in names:
            continue
        scale = scale_from_spec(spec, _top(sample, param))
        sample.set_scale(param, scale)
        for comp_name in _comp_names_for(sample, param):
            sample.set_scale(comp_name, scale.copy())
        n += 1
    return n


def apply_cytometer_settings(controller, cytometer: str, specs: dict[str, dict]) -> list:
    """Write `specs` ({param: spec}) to every sample of `cytometer` as one undo step
    and store them as that cytometer's defaults. Returns the affected samples."""
    ws = controller.workspace
    samples = group_by_cytometer(ws).get(cytometer, [])
    # one undo step for the stored defaults and every scale written (the
    # set_scale calls inside the group push no step of their own)
    with controller.undo_group(f"cytometer settings ({cytometer})", "settings"):
        cytometer_settings(ws)[cytometer] = {p: dict(sp) for p, sp in specs.items()}
        if not samples:
            controller.mark_modified()
            return []
        for param, spec in specs.items():
            holders = [s for s in samples if param in s.channel_names]
            if not holders:
                continue
            top = max(_top(s, param) for s in holders)
            scale = scale_from_spec(spec, top)
            controller.set_scale(holders, param, scale)
            comp_names: dict[str, list] = {}
            for s in holders:
                for cn in _comp_names_for(s, param):
                    comp_names.setdefault(cn, []).append(s)
            for cn, owners in comp_names.items():
                controller.set_scale(owners, cn, scale)
    return samples


# ----------------------------------------------------------------------- dialog
class CytometerDialog(QDialog):
    """Cytometers of the workspace and their default parameter scales."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Cytometer Settings")
        self.resize(820, 480)
        self._groups = group_by_cytometer(controller.workspace)
        self._specs: dict[str, dict[str, dict]] = {}
        self._current: str | None = None
        self._loading = False

        layout = QVBoxLayout(self)
        intro = QLabel("Default axis scales per instrument. <b>Apply</b> sets them for every sample acquired on the "
                       "selected cytometer; samples added later from that cytometer receive them automatically.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        split = QSplitter()
        self.list = QListWidget()
        self.list.setMinimumWidth(180)
        for name, samples in self._groups.items():
            item = QListWidgetItem(icon("cytometer"), f"{name}  ({len(samples)} sample{'s' if len(samples) != 1 else ''})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.list.addItem(item)
        split.addWidget(self.list)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight:600;")
        rl.addWidget(self.title)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Parameter", "Stain", "Scale", "Width basis", "Positive decades"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.horizontalHeader().setSectionResizeMode(COL_PARAM, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(COL_STAIN, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rl.addWidget(self.table, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        self.buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.list.currentItemChanged.connect(lambda cur, _prev: cur is not None and self.select(cur.data(Qt.ItemDataRole.UserRole)))
        if self.list.count():
            self.list.setCurrentRow(0)
        else:
            self.title.setText("No samples in the workspace")

    # ------------------------------------------------------------- state
    def cytometers(self) -> list[str]:
        return list(self._groups)

    def sample_count(self, cytometer: str) -> int:
        return len(self._groups.get(cytometer, []))

    def parameters(self, cytometer: str) -> list[str]:
        """Union of the (raw) parameters of the cytometer's samples, in file order."""
        out: list[str] = []
        seen = set()
        for s in self._groups.get(cytometer, []):
            for n in s.channel_names:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
        return out

    def _initial_specs(self, cytometer: str) -> dict[str, dict]:
        stored = (self.controller.workspace.settings.get(SETTINGS_KEY) or {}).get(cytometer, {})
        specs = {}
        samples = self._groups.get(cytometer, [])
        for param in self.parameters(cytometer):
            if param in stored:
                specs[param] = dict(stored[param])
                continue
            owner = next(s for s in samples if param in s.channel_names)
            # what the user sees: the compensated parameter's scale when there is one
            shown = next(iter(_comp_names_for(owner, param)), param)
            specs[param] = scale_to_spec(owner.scale(shown))
        return specs

    def specs(self, cytometer: str | None = None) -> dict[str, dict]:
        cyt = cytometer or self._current
        if cyt not in self._specs:
            self._specs[cyt] = self._initial_specs(cyt)
        return self._specs[cyt]

    def select(self, cytometer: str):
        if cytometer not in self._groups:
            return
        self._current = cytometer
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == cytometer and self.list.currentRow() != i:
                self.list.setCurrentRow(i)
        n = self.sample_count(cytometer)
        self.title.setText(f"{cytometer} – {n} sample{'s' if n != 1 else ''}")
        self._fill_table()

    def set_param_scale(self, param: str, kind: str, width_basis: float | None = None, decades: float | None = None):
        """Programmatic edit of one row (also used by the tests)."""
        spec = self.specs().setdefault(param, {"kind": kind})
        spec["kind"] = kind
        if width_basis is not None:
            spec["width_basis"] = float(width_basis)
        if decades is not None:
            spec["decades"] = float(decades)
        self._fill_table()

    # ------------------------------------------------------------- table
    def _fill_table(self):
        self._loading = True
        cyt = self._current
        specs = self.specs(cyt)
        params = self.parameters(cyt)
        sample = self._groups[cyt][0] if self._groups.get(cyt) else None
        self.table.setRowCount(len(params))
        for row, param in enumerate(params):
            spec = specs[param]
            self.table.setItem(row, COL_PARAM, QTableWidgetItem(param))
            self.table.setItem(row, COL_STAIN, QTableWidgetItem(sample.param_stain(param) if sample is not None and sample.param_stain(param) != param else ""))
            kind = QComboBox()
            for key, label in SCALE_CHOICES:
                kind.addItem(label, key)
            kind.setCurrentIndex(max(0, [k for k, _ in SCALE_CHOICES].index(spec.get("kind", "linear")) if spec.get("kind") in dict(SCALE_CHOICES) else 0))
            kind.currentIndexChanged.connect(lambda _i, p=param, w=kind: self._edited(p, "kind", w.currentData()))
            self.table.setCellWidget(row, COL_KIND, kind)
            wb = QDoubleSpinBox()
            wb.setRange(-1e6, -0.01)
            wb.setDecimals(1)
            wb.setSingleStep(10)
            wb.setValue(float(spec.get("width_basis", -100.0)))
            wb.setEnabled(spec.get("kind") == "logicle")
            wb.valueChanged.connect(lambda v, p=param: self._edited(p, "width_basis", v))
            self.table.setCellWidget(row, COL_WIDTH, wb)
            dec = QDoubleSpinBox()
            dec.setRange(1.0, 8.0)
            dec.setDecimals(2)
            dec.setSingleStep(0.1)
            top = _top(sample, param) if sample is not None else DEFAULT_TOP
            dec.setValue(float(spec.get("decades") or max(3.0, math.log10(top) - 1.0)))
            dec.setEnabled(spec.get("kind") != "linear")
            dec.valueChanged.connect(lambda v, p=param: self._edited(p, "decades", v))
            self.table.setCellWidget(row, COL_DECADES, dec)
        self._loading = False

    def _edited(self, param: str, key: str, value):
        if self._loading:
            return
        spec = self.specs()[param]
        spec[key] = value
        if key == "kind":
            row = self.parameters(self._current).index(param)
            self.table.cellWidget(row, COL_WIDTH).setEnabled(value == "logicle")
            self.table.cellWidget(row, COL_DECADES).setEnabled(value != "linear")
            if value == "logicle" and "width_basis" not in spec:
                spec["width_basis"] = float(self.table.cellWidget(row, COL_WIDTH).value())
            if value != "linear" and "decades" not in spec:
                spec["decades"] = float(self.table.cellWidget(row, COL_DECADES).value())

    def apply(self):
        """Apply every cytometer viewed in this dialog (the current one first)."""
        order = [c for c in [self._current] + [c for c in self._specs if c != self._current] if c is not None]
        if not order:
            return
        label = f"cytometer settings ({order[0]})" if len(order) == 1 else "cytometer settings"
        with self.controller.undo_group(label, "settings"):  # one undo step for all
            for cyt in order:
                apply_cytometer_settings(self.controller, cyt, self.specs(cyt))


# ---------------------------------------------------------------------- install
def install(window):
    """Add the "Cytometry" ribbon band and menu entry; apply stored defaults to
    samples that join the workspace later."""
    controller = window.controller
    act = QAction(icon("cytometer"), "Cytometer Settings…", window)
    act.setToolTip("Default axis scales per cytometer ($CYT)")
    act.setStatusTip("Default axis scales for the samples of each cytometer")

    def open_dialog():
        CytometerDialog(controller, window).exec()

    act.triggered.connect(open_dialog)
    page = getattr(window, "ribbon_pages", {}).get("workspace")
    if page is not None:
        band = page.add_band("Cytometry")
        band.add_large(act)
    menu = getattr(window, "menus", {}).get("workspace")
    if menu is not None:
        menu.addSeparator()
        menu.addAction(act)

    known = {s.id for s in controller.workspace.samples}

    def on_replaced():
        known.clear()
        known.update(s.id for s in controller.workspace.samples)

    def on_samples():
        ws = controller.workspace
        new = [s for s in ws.samples if s.id not in known]
        known.update(s.id for s in new)
        if not ws.settings.get(SETTINGS_KEY):
            return
        changed = [s for s in new if apply_cytometer_defaults(ws, s)]
        if changed:
            controller.scalesChanged.emit(changed[0].id if len(changed) == 1 else None)
            controller.gatingChanged.emit(changed[0].id if len(changed) == 1 else None)

    controller.workspaceReplaced.connect(on_replaced)
    controller.samplesChanged.connect(on_samples)
    return {"action": act, "dialog": CytometerDialog}
