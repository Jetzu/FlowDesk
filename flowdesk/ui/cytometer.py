"""Cytometer settings: default axis scales per instrument.

Samples are grouped by the cytometer that acquired them ($CYT keyword). For
each cytometer the dialog lists the union of its samples' parameters with a
default scale (Linear / Biexponential / Logarithmic / Arcsinh, width basis,
positive decades, extra negative decades, linear minimum / maximum). "Apply"
writes the scales to every sample of that cytometer and remembers them in
``workspace.settings["cytometers"][name]`` so samples of the same instrument
added later receive them too (``apply_cytometer_defaults``).

As in FlowJo's Preferences > Cytometers the settings can also be kept at
application level (QSettings key ``prefs/cytometers``, a JSON object
{cytometer: {parameter: spec}}): they then apply to samples of that
instrument in every workspace that has no settings of its own. A small
built-in table (``BUILTIN_CYTOMETERS``) offers starting points per
instrument family ("Instrument Defaults" in the dialog).

Transforms belong to the raw parameter: a compensated parameter (Comp-X)
always shows the transform of X (see ``flowdesk.core.model.ScaleMap``).
"""
from __future__ import annotations

import json
import math

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.transforms import (
    ArcsinhScale,
    LinearScale,
    LogicleScale,
    LogScale,
    Scale,
    default_scale_for_channel,
    set_scale_defaults,
)
from .icons import icon

UNKNOWN = "Unknown"
SETTINGS_KEY = "cytometers"
APP_SETTINGS_KEY = "prefs/cytometers"
SCALE_CHOICES = [("linear", "Linear"), ("logicle", "Biexponential"), ("log", "Logarithmic"), ("arcsinh", "Arcsinh")]
DEFAULT_TOP = 262144.0

COL_PARAM, COL_STAIN, COL_KIND, COL_WIDTH, COL_DECADES, COL_NEG, COL_MIN, COL_MAX = range(8)
HEADERS = ["Parameter", "Stain", "Scale", "Width basis", "Positive decades", "Extra neg. decades", "Linear min", "Linear max"]

# Starting points per instrument family, matched case-insensitively against
# $CYT. Generic recommendations, not vendor settings: spectral analysers get a
# wider linear region (unmixed data spread further below zero), log-amplified
# analog instruments a log scale over four decades; scatter, width and time
# parameters stay linear (time ranged to the acquired data).
BUILTIN_CYTOMETERS = [
    {"name": "Cytek Aurora / Northern Lights (spectral)", "match": ("aurora", "northern lights", "northernlights"),
     "fluorescence": {"kind": "logicle", "width_basis": -1000.0, "neg": 0.0}},
    {"name": "Sony ID7000 / SP6800 (spectral)", "match": ("id7000", "sp6800"),
     "fluorescence": {"kind": "logicle", "width_basis": -1000.0, "neg": 0.0}},
    {"name": "BD FACSCalibur (log amplifiers)", "match": ("facscalibur", "calibur"),
     "fluorescence": {"kind": "log", "decades": 4.0}},
    {"name": "BD FACSDiva instruments", "match": ("fortessa", "lsr", "canto", "aria", "symphony", "celesta", "lyric", "verse"),
     "fluorescence": {"kind": "logicle", "width_basis": -100.0, "neg": 0.0}},
    {"name": "Beckman Coulter CytoFLEX", "match": ("cytoflex",),
     "fluorescence": {"kind": "logicle", "width_basis": -100.0, "neg": 0.0}},
    {"name": "Thermo Fisher Attune", "match": ("attune",),
     "fluorescence": {"kind": "logicle", "width_basis": -100.0, "neg": 0.0}},
    {"name": "Miltenyi MACSQuant", "match": ("macsquant",),
     "fluorescence": {"kind": "logicle", "width_basis": -100.0, "neg": 0.0}},
    {"name": "Agilent NovoCyte", "match": ("novocyte",),
     "fluorescence": {"kind": "logicle", "width_basis": -100.0, "neg": 0.0}},
]


# ------------------------------------------------------------- app-level store
def app_store():
    """The QSettings holding application-level preferences (tests replace it)."""
    return QSettings("FlowDesk", "FlowDesk")


def app_cytometer_specs() -> dict[str, dict[str, dict]]:
    """Cytometer settings kept at application level: {cytometer: {param: spec}}."""
    try:
        raw = app_store().value(APP_SETTINGS_KEY)
    except Exception:  # noqa: BLE001 - a broken settings file must not stop the app
        return {}
    if raw is None or raw == "":
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for cyt, specs in raw.items():
        if isinstance(specs, dict):
            out[str(cyt)] = {str(p): dict(sp) for p, sp in specs.items() if isinstance(sp, dict)}
    return out


def _write_app_specs(data: dict):
    store = app_store()
    store.setValue(APP_SETTINGS_KEY, json.dumps(data, sort_keys=True))
    store.sync()


def remember_cytometer(cytometer: str, specs: dict[str, dict]) -> bool:
    """Keep `specs` as the application-level settings of `cytometer` (every
    workspace). Not for samples without $CYT. Returns True when stored."""
    if not cytometer or cytometer == UNKNOWN:
        return False
    data = app_cytometer_specs()
    data[cytometer] = {p: dict(sp) for p, sp in specs.items()}
    _write_app_specs(data)
    return True


def forget_cytometer(cytometer: str) -> bool:
    data = app_cytometer_specs()
    if data.pop(cytometer, None) is None:
        return False
    _write_app_specs(data)
    return True


def apply_scale_preferences(prefs) -> bool:
    """Push the preferences "default_transform" / "default_width_basis" into
    the default scale of fluorescence parameters nobody chose a scale for."""
    get = getattr(prefs, "get", None)
    if get is None:
        return False
    return set_scale_defaults(get("default_transform", "logicle"), get("default_width_basis", -100.0))


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


def _is_time(sample, param: str) -> bool:
    fn = getattr(sample, "_is_time_channel", None)
    return bool(fn(sample.base_channel(param))) if fn is not None else False


def _comp_names_for(sample, param: str) -> list[str]:
    """Compensated parameters of `sample` whose base channel is `param` (they
    share its transform)."""
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
        return {"kind": "linear", "min": float(scale.min), "max": float(scale.max)}
    return {"kind": "linear", "min": 0.0}


def scale_from_spec(spec: dict, top: float = DEFAULT_TOP) -> Scale:
    """Build a scale for a parameter whose range is `top` from a spec. A
    linear spec without "max" ends at `top`."""
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
    maximum = spec.get("max")
    try:
        maximum = float(maximum) if maximum not in (None, "") else None
    except (TypeError, ValueError):
        maximum = None
    return LinearScale(float(spec.get("min", 0.0) or 0.0), maximum if maximum and maximum > 0 else top)


def _auto_linear(sample, param: str, spec: dict) -> bool:
    """A linear spec without a maximum on a time parameter: the axis keeps
    following the acquired data (the sample's own default)."""
    return spec.get("kind", "linear") == "linear" and spec.get("max") in (None, "", 0, 0.0) and _is_time(sample, param) \
        and float(spec.get("min", 0.0) or 0.0) == 0.0


def cytometer_settings(workspace) -> dict:
    return workspace.settings.setdefault(SETTINGS_KEY, {})


def builtin_cytometer(cytometer: str) -> dict | None:
    """The BUILTIN_CYTOMETERS entry matching a $CYT value, None when unknown."""
    text = (cytometer or "").lower()
    if not text or cytometer == UNKNOWN:
        return None
    for entry in BUILTIN_CYTOMETERS:
        if any(m in text for m in entry["match"]):
            return entry
    return None


def builtin_spec(sample, param: str, cytometer: str | None = None) -> dict:
    """Starting-point spec of one parameter: the built-in instrument entry for
    fluorescence, linear (time: following the data) for scatter, width and time."""
    linear = isinstance(default_scale_for_channel(param, transform="logicle"), LinearScale)
    ch = sample.base_channel(param) if sample is not None else None
    if ch is not None and ch.is_log_amplified:
        return scale_to_spec(sample.default_scale(param))
    if linear:
        spec = {"kind": "linear", "min": 0.0}
        if sample is not None and not _is_time(sample, param):
            spec["max"] = _top(sample, param)
        return spec
    entry = builtin_cytometer(cytometer or "")
    top = _top(sample, param) if sample is not None else DEFAULT_TOP
    spec = dict(entry["fluorescence"]) if entry else {"kind": "logicle", "width_basis": -100.0, "neg": 0.0}
    spec.setdefault("decades", round(max(3.0, math.log10(top) - 1.0), 3))
    return spec


def builtin_cytometer_specs(cytometer: str, samples: list) -> dict[str, dict]:
    """Built-in starting points for every raw parameter of `samples`."""
    out: dict[str, dict] = {}
    for s in samples:
        for n in s.channel_names:
            if n not in out:
                out[n] = builtin_spec(s, n, cytometer)
    return out


def specs_for(workspace, cytometer: str, app_level: dict | None = None) -> dict[str, dict] | None:
    """The settings that apply to samples of `cytometer`: the workspace's own,
    else the application-level ones (``app_level``, {cytometer: specs})."""
    specs = (workspace.settings.get(SETTINGS_KEY) or {}).get(cytometer)
    if specs:
        return specs
    if app_level and cytometer != UNKNOWN:
        return app_level.get(cytometer) or None
    return None


def _sample_plan(sample, specs: dict[str, dict]) -> list[tuple[str, Scale | None]]:
    """(parameter, scale) pairs `specs` gives one sample; None = its own default."""
    names = set(sample.channel_names)
    plan = []
    for param, spec in specs.items():
        if param not in names:
            continue
        plan.append((param, None if _auto_linear(sample, param, spec) else scale_from_spec(spec, _top(sample, param))))
    return plan


def apply_cytometer_defaults(workspace, sample, app_level: dict | None = None) -> int:
    """Give `sample` the stored default scales of its cytometer (the
    workspace's, else ``app_level``). Returns the number of parameters set (0
    when no defaults are stored for that instrument)."""
    specs = specs_for(workspace, cytometer_of(sample), app_level)
    if not specs:
        return 0
    n = 0
    for param, scale in _sample_plan(sample, specs):
        if scale is None:
            sample.reset_scale(param)
        else:
            sample.set_scale(param, scale)
        n += 1
    return n


def _planned_scales(samples: list, specs: dict[str, dict]) -> list[tuple[list, str, Scale | None]]:
    """(samples, parameter, scale) for every scale that applying `specs`
    writes; scale None resets the parameter to the sample's own default (time
    axes following the data). Compensated parameters share these transforms."""
    plan = []
    for param, spec in specs.items():
        holders = [s for s in samples if param in s.channel_names]
        if not holders:
            continue
        auto = [s for s in holders if _auto_linear(s, param, spec)]
        if auto:
            plan.append((auto, param, None))
        rest = [s for s in holders if s not in auto]
        if rest:
            top = max(_top(s, param) for s in rest)
            plan.append((rest, param, scale_from_spec(spec, top)))
    return plan


def _differs(sample, name: str, scale: Scale | None) -> bool:
    if scale is None:
        return sample.has_own_scale(name)
    return sample.scales.get(name) != scale


def cytometer_settings_would_change(workspace, cytometer: str, specs: dict[str, dict]) -> bool:
    """False when applying `specs` to `cytometer` changes nothing: the stored
    defaults are the same and every sample already has exactly these scales."""
    new = {p: dict(sp) for p, sp in specs.items()}
    if (workspace.settings.get(SETTINGS_KEY) or {}).get(cytometer) != new:
        return True
    samples = group_by_cytometer(workspace).get(cytometer, [])
    return any(_differs(s, name, scale) for owners, name, scale in _planned_scales(samples, specs) for s in owners)


def apply_cytometer_settings(controller, cytometer: str, specs: dict[str, dict], remember: bool = False) -> list:
    """Write `specs` ({param: spec}) to every sample of `cytometer` as one undo step
    and store them as that cytometer's defaults (with remember=True also at
    application level, for every workspace). Returns the affected samples
    (none, and no undo step, when nothing would change)."""
    ws = controller.workspace
    if remember:
        remember_cytometer(cytometer, specs)
    if not cytometer_settings_would_change(ws, cytometer, specs):
        return []
    samples = group_by_cytometer(ws).get(cytometer, [])
    plan = _planned_scales(samples, specs)
    # one undo step for the stored defaults and every scale written (the
    # set_scale calls inside the group push no step of their own)
    with controller.undo_group(f"cytometer settings ({cytometer})", "settings"):
        cytometer_settings(ws)[cytometer] = {p: dict(sp) for p, sp in specs.items()}
        if not samples:
            controller.mark_modified()
            return []
        reset = []
        for owners, name, scale in plan:
            if scale is None:
                for s in owners:
                    if s.has_own_scale(name):
                        s.reset_scale(name)
                        reset.append(s)
            else:
                controller.set_scale(owners, name, scale)
        if reset:
            controller.mark_modified()
            controller.scalesChanged.emit(reset[0].id if len(reset) == 1 else None)
            controller.gatingChanged.emit(reset[0].id if len(reset) == 1 else None)
    return samples


# ----------------------------------------------------------------------- dialog
class CytometerDialog(QDialog):
    """Cytometers of the workspace and their default parameter scales."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Cytometer Settings")
        self.resize(1040, 500)
        self._groups = group_by_cytometer(controller.workspace)
        self._specs: dict[str, dict[str, dict]] = {}
        self._current: str | None = None
        self._loading = False

        layout = QVBoxLayout(self)
        intro = QLabel("Default axis scales per instrument. <b>Apply</b> sets them for every sample acquired on the "
                       "selected cytometer; samples added later from that cytometer receive them automatically. "
                       "Compensated parameters share the scale of their raw parameter.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        split = QSplitter()
        self.list = QListWidget()
        self.list.setMinimumWidth(180)
        for name, samples in self._groups.items():
            item = QListWidgetItem(icon("cytometer"), f"{name}  ({len(samples)} sample{'s' if len(samples) != 1 else ''})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            entry = builtin_cytometer(name)
            if entry is not None:
                item.setToolTip(f"Built-in defaults: {entry['name']}")
            self.list.addItem(item)
        split.addWidget(self.list)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel()
        self.title.setStyleSheet("font-weight:600;")
        rl.addWidget(self.title)
        self.table = QTableWidget(0, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.horizontalHeader().setSectionResizeMode(COL_PARAM, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(COL_STAIN, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        rl.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.defaults_button = QPushButton("Instrument Defaults")
        self.defaults_button.setToolTip("Fill the table with the built-in starting points for this instrument")
        self.defaults_button.clicked.connect(lambda: self.load_instrument_defaults())
        row.addWidget(self.defaults_button)
        row.addStretch(1)
        self.remember = QCheckBox("Use in all workspaces")
        self.remember.setToolTip("Also keep these settings as application preferences (Preferences ▸ Cytometers), "
                                 "for samples of this cytometer in every workspace")
        self.remember.setChecked(True)
        row.addWidget(self.remember)
        rl.addLayout(row)
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
        stored = specs_for(self.controller.workspace, cytometer, app_cytometer_specs()) or {}
        specs = {}
        samples = self._groups.get(cytometer, [])
        for param in self.parameters(cytometer):
            if param in stored:
                specs[param] = dict(stored[param])
                continue
            owner = next(s for s in samples if param in s.channel_names)
            specs[param] = scale_to_spec(owner.scale(param))  # shared with the compensated parameter
            if specs[param]["kind"] == "linear" and _is_time(owner, param) and not owner.has_own_scale(param):
                specs[param].pop("max", None)  # "Auto": follows the acquired data
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
        entry = builtin_cytometer(cytometer)
        extra = f"  (built-in defaults: {entry['name']})" if entry else ""
        self.title.setText(f"{cytometer} – {n} sample{'s' if n != 1 else ''}{extra}")
        self.remember.setEnabled(cytometer != UNKNOWN)
        self._fill_table()

    def set_param_scale(self, param: str, kind: str, width_basis: float | None = None, decades: float | None = None,
                        neg: float | None = None, minimum: float | None = None, maximum: float | None = None):
        """Programmatic edit of one row (also used by the tests). maximum <= 0
        means "Auto" (the parameter's range; time: the acquired data)."""
        spec = self.specs().setdefault(param, {"kind": kind})
        spec["kind"] = kind
        if width_basis is not None:
            spec["width_basis"] = float(width_basis)
        if decades is not None:
            spec["decades"] = float(decades)
        if neg is not None:
            spec["neg"] = float(neg)
        if minimum is not None:
            spec["min"] = float(minimum)
        if maximum is not None:
            if maximum > 0:
                spec["max"] = float(maximum)
            else:
                spec.pop("max", None)
        self._fill_table()

    def load_instrument_defaults(self, cytometer: str | None = None):
        """Replace the rows of a cytometer by the built-in starting points."""
        cyt = cytometer or self._current
        if cyt is None:
            return
        self._specs[cyt] = builtin_cytometer_specs(cyt, self._groups.get(cyt, []))
        if cyt == self._current:
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
            kind_key = spec.get("kind", "linear")
            self.table.setItem(row, COL_PARAM, QTableWidgetItem(param))
            self.table.setItem(row, COL_STAIN, QTableWidgetItem(sample.param_stain(param) if sample is not None and sample.param_stain(param) != param else ""))
            kind = QComboBox()
            for key, label in SCALE_CHOICES:
                kind.addItem(label, key)
            kind.setCurrentIndex(max(0, [k for k, _ in SCALE_CHOICES].index(kind_key) if kind_key in dict(SCALE_CHOICES) else 0))
            kind.currentIndexChanged.connect(lambda _i, p=param, w=kind: self._edited(p, "kind", w.currentData()))
            self.table.setCellWidget(row, COL_KIND, kind)
            wb = QDoubleSpinBox()
            wb.setRange(-1e6, -0.01)
            wb.setDecimals(1)
            wb.setSingleStep(10)
            wb.setValue(float(spec.get("width_basis", -100.0)))
            wb.valueChanged.connect(lambda v, p=param: self._edited(p, "width_basis", v))
            self.table.setCellWidget(row, COL_WIDTH, wb)
            dec = QDoubleSpinBox()
            dec.setRange(1.0, 8.0)
            dec.setDecimals(2)
            dec.setSingleStep(0.1)
            top = _top(sample, param) if sample is not None else DEFAULT_TOP
            dec.setValue(float(spec.get("decades") or max(3.0, math.log10(top) - 1.0)))
            dec.valueChanged.connect(lambda v, p=param: self._edited(p, "decades", v))
            self.table.setCellWidget(row, COL_DECADES, dec)
            neg = QDoubleSpinBox()
            neg.setRange(0.0, 4.0)
            neg.setDecimals(2)
            neg.setSingleStep(0.1)
            neg.setValue(float(spec.get("neg", 0.0) or 0.0))
            neg.valueChanged.connect(lambda v, p=param: self._edited(p, "neg", v))
            self.table.setCellWidget(row, COL_NEG, neg)
            lo = QDoubleSpinBox()
            lo.setRange(-1e9, 1e9)
            lo.setDecimals(1)
            lo.setSingleStep(100)
            lo.setValue(float(spec.get("min", 0.0) or 0.0))
            lo.valueChanged.connect(lambda v, p=param: self._edited(p, "min", v))
            self.table.setCellWidget(row, COL_MIN, lo)
            hi = QDoubleSpinBox()
            hi.setRange(0.0, 1e9)
            hi.setDecimals(1)
            hi.setSingleStep(1000)
            hi.setSpecialValueText("Auto")
            hi.setToolTip("Auto: the parameter's range ($PnR); time parameters follow the acquired data")
            hi.setValue(float(spec.get("max") or 0.0))
            hi.valueChanged.connect(lambda v, p=param: self._edited(p, "max", v))
            self.table.setCellWidget(row, COL_MAX, hi)
            self._enable_row(row, kind_key)
        self._loading = False

    def _enable_row(self, row: int, kind: str):
        self.table.cellWidget(row, COL_WIDTH).setEnabled(kind == "logicle")
        self.table.cellWidget(row, COL_DECADES).setEnabled(kind != "linear")
        self.table.cellWidget(row, COL_NEG).setEnabled(kind in ("logicle", "arcsinh"))
        self.table.cellWidget(row, COL_MIN).setEnabled(kind == "linear")
        self.table.cellWidget(row, COL_MAX).setEnabled(kind == "linear")

    def _edited(self, param: str, key: str, value):
        if self._loading:
            return
        spec = self.specs()[param]
        if key == "max" and (value is None or float(value) <= 0):
            spec.pop("max", None)
        else:
            spec[key] = value
        if key == "kind":
            row = self.parameters(self._current).index(param)
            self._enable_row(row, value)
            if value == "logicle" and "width_basis" not in spec:
                spec["width_basis"] = float(self.table.cellWidget(row, COL_WIDTH).value())
            if value != "linear" and "decades" not in spec:
                spec["decades"] = float(self.table.cellWidget(row, COL_DECADES).value())
            if value in ("logicle", "arcsinh") and "neg" not in spec:
                spec["neg"] = float(self.table.cellWidget(row, COL_NEG).value())

    def apply(self):
        """Apply every cytometer viewed in this dialog (the current one first)."""
        order = [c for c in [self._current] + [c for c in self._specs if c != self._current] if c is not None]
        if self.remember.isChecked():
            for cyt in order:
                if cyt != UNKNOWN:
                    remember_cytometer(cyt, self.specs(cyt))
        order = [c for c in order if cytometer_settings_would_change(self.controller.workspace, c, self.specs(c))]
        if not order:
            return  # nothing changed: no empty undo step
        label = f"cytometer settings ({order[0]})" if len(order) == 1 else "cytometer settings"
        with self.controller.undo_group(label, "settings"):  # one undo step for all
            for cyt in order:
                apply_cytometer_settings(self.controller, cyt, self.specs(cyt))


# ---------------------------------------------------------------------- install
def connect_controller(controller) -> dict:
    """Keep a controller's workspaces in step with the cytometer settings and
    the default-scale preferences: samples that join later receive the stored
    defaults of their cytometer (the workspace's, else the application-level
    ones), and "default_transform" / "default_width_basis" decide the scale of
    parameters nobody chose one for."""
    known = {s.id for s in controller.workspace.samples}

    def on_replaced():
        known.clear()
        known.update(s.id for s in controller.workspace.samples)

    def on_samples():
        ws = controller.workspace
        new = [s for s in ws.samples if s.id not in known]
        known.update(s.id for s in new)
        if not new:
            return
        app_level = app_cytometer_specs()
        if not ws.settings.get(SETTINGS_KEY) and not app_level:
            return
        changed = [s for s in new if apply_cytometer_defaults(ws, s, app_level)]
        if changed:
            controller.scalesChanged.emit(changed[0].id if len(changed) == 1 else None)
            controller.gatingChanged.emit(changed[0].id if len(changed) == 1 else None)

    def on_preferences():
        prefs = getattr(controller, "preferences", None)
        if prefs is not None:
            apply_scale_preferences(prefs)

    controller.workspaceReplaced.connect(on_replaced)
    controller.samplesChanged.connect(on_samples)
    if hasattr(controller, "preferencesChanged"):
        controller.preferencesChanged.connect(on_preferences)
    on_preferences()
    return {"on_samples": on_samples, "on_replaced": on_replaced, "on_preferences": on_preferences}


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
    hooks = connect_controller(controller)
    return {"action": act, "dialog": CytometerDialog, "hooks": hooks}
