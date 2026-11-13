"""Compensation window (matrix editor + sample status) and the compensation wizard.

The window lists the spillover matrices of the workspace together with the
acquisition-defined matrices found in the samples' $SPILLOVER keywords, shows
the selected matrix as an editable grid of spill percentages (or its inverse,
the compensation matrix), an N x N panel of the compensated single-stain
controls with hot-spot highlighting, the spillover spreading matrix, and lists
which matrix every sample uses. The wizard computes a new matrix from gated
single-stain controls (FlowJo-style "Compensation Wizard": controls -> gates
-> matrix -> diagnostics), by the median-difference method or by AutoSpill-
style iterative robust regression.
"""
from __future__ import annotations

import csv
import datetime as _dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from ..core.compensation import (
    ACQUISITION_DEFINED,
    ControlDiagnostics,
    ControlGates,
    SpilloverMatrix,
    diagnostics_warnings,
    estimate_spillover,
    groups_with_matrix,
    mark_group_compensation,
    matrix_from_fcs_file,
    matrix_from_text,
    matrix_to_text,
    pending_group_compensation,
    read_matrix_mtx,
    refine_spillover,
    residual_spill,
    set_group_matrix,
    spillover_spreading,
    write_matrix_mtx,
)
from ..core.fcs import parse_spillover
from .icons import icon

UNSTAINED = "Unstained"
NOT_USED = "— not used —"
UNGATED = "Ungated"
CONTROLS_KEY = "compensation_controls"  # workspace.settings: the wizard's last control set
METHODS = [("median", "Median difference (classic)"), ("autospill", "AutoSpill (iterative robust regression)")]
HOTSPOT = 0.02  # residual spill beyond +-2 % marks a cell / control plot as a hot spot

_SCATTER_RE = re.compile(r"^(FSC|SSC|TIME|EVENT|WIDTH|HEIGHT)", re.IGNORECASE)


# ---------------------------------------------------------------- helpers
def fluorescence_params(sample) -> list[str]:
    """Fluorescence parameters of a sample: every channel except scatter, time
    and pulse width/height channels; area ("-A") channels are preferred."""
    names = list(sample.channel_names)
    fluo = []
    for n in names:
        up = n.upper()
        if _SCATTER_RE.match(up) or "TIME" in up:
            continue
        if up.endswith("-W") or up.endswith("-H"):
            continue
        fluo.append(n)
    area = [n for n in fluo if n.upper().endswith("-A")]
    return area or fluo


def _is_scatter(name: str) -> bool:
    up = str(name).upper()
    return bool(_SCATTER_RE.match(up)) or "TIME" in up


def default_control_population(sample):
    """The population a control's spillover is computed from by default: the
    deepest gate along the first-child chain that only uses scatter / time
    parameters (cells, singlets, beads), or None (ungated)."""
    pop, best = sample.root, None
    while pop.children:
        child = pop.children[0]
        dims = list(getattr(child.gate, "dims", []) or [])
        if child.gate is None or not dims or not all(_is_scatter(d) for d in dims):
            break
        best, pop = child, child
    return best


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text.lower()).strip()


def _token_pattern(token: str) -> re.Pattern:
    parts = [re.escape(p) for p in _norm(token).split(" ") if p]
    body = r"[\s_\-]?".join(parts)
    # word boundaries; a token followed by "-<alnum>" (e.g. "PE" in "PE-Cy7")
    # does not count as a match.
    return re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])(?!-[a-z0-9])", re.IGNORECASE)


def _param_tokens(param: str, stain: str | None) -> list[str]:
    tokens = {param}
    base = re.sub(r"-[AHW]$", "", param, flags=re.IGNORECASE)
    if base:
        tokens.add(base)
    if stain and stain.strip() and stain.strip() != param:
        tokens.add(stain.strip())
        # a stain written like a detector ("FITC-A" on FL1-A) also matches "FITC"
        stain_base = re.sub(r"-[AHW]$", "", stain.strip(), flags=re.IGNORECASE)
        if stain_base:
            tokens.add(stain_base)
    return [t for t in tokens if len(t) >= 2]


def guess_control_assignment(sample_names: list[str], params: list[str], stains: dict[str, str]) -> dict[str, str]:
    """Guess which control each sample is: sample name -> "Unstained" | param.

    Samples that cannot be assigned are left out."""
    candidates: list[tuple[re.Pattern, int, str]] = []
    for p in params:
        for tok in _param_tokens(p, (stains or {}).get(p)):
            candidates.append((_token_pattern(tok), len(_norm(tok)), p))
    out: dict[str, str] = {}
    for name in sample_names:
        stem = re.sub(r"\.fcs$", "", name, flags=re.IGNORECASE)
        if "unstain" in stem.lower() or re.search(r"(?<![a-z])(blank|no[\s_\-]?stain)(?![a-z])", stem, re.IGNORECASE):
            out[name] = UNSTAINED
            continue
        best: tuple[int, str] | None = None
        for pat, length, p in candidates:
            if pat.search(stem) and (best is None or length > best[0]):
                best = (length, p)
        if best is not None:
            out[name] = best[1]
    return out


def control_columns(sample, population, params: list[str]) -> np.ndarray:
    """Raw (uncompensated) events of `params` in `population` of a control
    (all events when population is None or the root)."""
    if not sample.ensure_loaded() or sample.fcs is None or sample.fcs.events is None:
        raise ValueError(f"Cannot load {sample.name}")
    idx = []
    for p in params:
        try:
            idx.append(sample.fcs.channel_index(p))
        except KeyError:
            raise ValueError(f"{sample.name} has no parameter {p}") from None
    events = sample.fcs.events
    if population is not None and population.parent is not None:
        mask = population.mask()
        if mask.shape[0] == events.shape[0]:
            events = events[mask]
    return np.asarray(events[:, idx], dtype=np.float64)


def _split_assignment(assignment: dict) -> tuple[dict, object]:
    singles: dict[str, object] = {}
    unstained = None
    for sample, role in assignment.items():
        if role == UNSTAINED:
            if unstained is None:
                unstained = sample
        elif role and role != NOT_USED and role not in singles:
            singles[role] = sample
    return singles, unstained


def control_data(assignment: dict, use_unstained: bool = True, populations: dict | None = None, params: list[str] | None = None, cache: dict | None = None):
    """(params, controls, unstained events, singles) of an assignment
    (Sample -> "Unstained" | parameter). The parameters are the assigned
    ones in channel order of the first control unless `params` is given."""
    singles, unstained = _split_assignment(assignment)
    if not singles:
        raise ValueError("No single-stain controls assigned")
    if params is None:
        first = next(iter(singles.values()))
        order = {n: i for i, n in enumerate(first.channel_names)}
        params = sorted(singles, key=lambda p: order.get(p, len(order)))
    populations = populations or {}

    def columns(sample):
        pop = populations.get(sample)
        if cache is None:
            return control_columns(sample, pop, params)
        mask = pop.mask() if pop is not None and pop.parent is not None else None
        key = (id(sample), id(mask) if mask is not None else 0, tuple(params))
        hit = cache.get(key)
        if hit is None or hit[0] is not mask:
            hit = (mask, control_columns(sample, pop, params))
            cache[key] = hit
        return hit[1]

    controls = {p: columns(s) for p, s in singles.items() if p in params}
    unst = columns(unstained) if (use_unstained and unstained is not None) else None
    return params, controls, unst, singles


def compute_spillover(assignment: dict, use_unstained: bool = True, name: str = "Computed", populations: dict | None = None, gates: dict | None = None, method: str = "median", cache: dict | None = None) -> SpilloverMatrix:
    """Compute a spillover matrix from control samples.

    assignment: Sample -> "Unstained" | parameter name (other values ignored).
    populations: Sample -> Population whose events are used (default: all).
    gates: parameter -> ControlGates overriding the automatic split. method:
    "median" (median difference) or "autospill" (iterative robust regression
    started from the median estimate). The matrix parameters are the assigned
    parameters, in channel order of the first control; the result carries
    `diagnostics` (and `report` for AutoSpill)."""
    params, controls, unst, _singles = control_data(assignment, use_unstained, populations, cache=cache)
    m = estimate_spillover(controls, params, unst, name, gates=gates)
    if method == "autospill":
        refined = refine_spillover(controls, params, m, unstained=unst, gates=gates, name=name)
        refined.diagnostics = m.diagnostics
        return refined
    return m


def _unique_name(existing, base: str) -> str:
    name, k = base, 2
    while name in existing:
        name = f"{base} {k}"
        k += 1
    return name


def _tint(value: float) -> QColor:
    v = min(abs(float(value)), 0.5) / 0.5
    if abs(value) < 1e-9:
        return QColor("#ffffff")
    light, strong = (234, 242, 251), (40, 100, 190)
    c = [round(a + (b - a) * v) for a, b in zip(light, strong)]
    return QColor(*c)


def _hot_color(residual: float) -> QColor | None:
    if residual is None or not np.isfinite(residual) or abs(residual) <= HOTSPOT:
        return None
    return QColor("#f4b6b6") if residual > 0 else QColor("#b9cdf3")


def _hot_text(residual: float) -> str:
    kind = "under-compensated" if residual > 0 else "over-compensated"
    return f"residual spill {residual * 100:+.2f} %: {kind}"


def fill_matrix_grid(table: QTableWidget, matrix: SpilloverMatrix | None, editable: bool = True, mode: str = "spill", hotspots: np.ndarray | None = None):
    """Show a matrix as percentages (row = source fluorochrome, column =
    detector). mode "inverse" shows the compensation (inverse) matrix x 100,
    read-only. `hotspots`: residual spill per cell; cells beyond HOTSPOT are
    tinted red (under-compensated) or blue (over-compensated)."""
    blocked = table.blockSignals(True)
    try:
        if matrix is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        n = len(matrix.params)
        if table.rowCount() != n or table.columnCount() != n:
            table.setRowCount(n)
            table.setColumnCount(n)
        table.setHorizontalHeaderLabels(list(matrix.params))
        table.setVerticalHeaderLabels(list(matrix.params))
        values = matrix.inverse() if mode == "inverse" else matrix.matrix
        for i in range(n):
            for j in range(n):
                v = float(values[i, j])
                item = table.item(i, j)
                if item is None:
                    item = QTableWidgetItem()
                    table.setItem(i, j, item)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip("")
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if i == j:
                    item.setText("100.00" if mode != "inverse" else f"{v * 100:.2f}")
                    item.setBackground(QBrush(QColor("#e4e4e4")))
                    item.setForeground(QBrush(QColor("#555555")))
                else:
                    item.setText(f"{v * 100:.2f}")
                    bg = _tint(v)
                    hot = None
                    if mode != "inverse" and hotspots is not None and i < hotspots.shape[0] and j < hotspots.shape[1]:
                        hot = _hot_color(float(hotspots[i, j]))
                        if hot is not None:
                            bg = hot
                            item.setToolTip(_hot_text(float(hotspots[i, j])))
                    item.setBackground(QBrush(bg))
                    item.setForeground(QBrush(QColor("#ffffff" if bg.lightness() < 130 else "#000000")))
                    if editable and mode != "inverse":
                        flags |= Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
    finally:
        table.blockSignals(blocked)


def fill_values_grid(table: QTableWidget, params: list[str], values: np.ndarray | None, decimals: int = 2, rows: list[str] | None = None):
    """A read-only grid of numbers (NaN shown as "—"), tinted by magnitude."""
    blocked = table.blockSignals(True)
    try:
        if values is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        rows = rows or params
        table.setRowCount(len(rows))
        table.setColumnCount(len(params))
        table.setHorizontalHeaderLabels(list(params))
        table.setVerticalHeaderLabels(list(rows))
        finite = values[np.isfinite(values)]
        top = float(np.abs(finite).max()) if finite.size else 0.0
        for i in range(len(rows)):
            for j in range(len(params)):
                v = float(values[i, j])
                item = QTableWidgetItem("—" if not np.isfinite(v) else f"{v:.{decimals}f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if i == j and len(rows) == len(params):
                    item.setBackground(QBrush(QColor("#e4e4e4")))
                elif np.isfinite(v) and top > 0:
                    bg = _tint(0.5 * abs(v) / top)
                    item.setBackground(QBrush(bg))
                    item.setForeground(QBrush(QColor("#ffffff" if bg.lightness() < 130 else "#000000")))
                table.setItem(i, j, item)
    finally:
        table.blockSignals(blocked)


def read_matrix_csv(path: str | Path) -> SpilloverMatrix:
    """Read a matrix CSV: header row of detector names (optionally preceded by
    an empty corner cell) followed by rows of values (fractions or percent)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        text = fh.read()
    return matrix_from_text(text, Path(path).stem)


def write_matrix_csv(matrix: SpilloverMatrix, path: str | Path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + list(matrix.params))
        for p, row in zip(matrix.params, matrix.matrix):
            w.writerow([p] + [f"{v:.6g}" for v in row])


def read_matrix_file(path: str | Path) -> SpilloverMatrix:
    """A matrix from a FlowJo .mtx (XML or text), a CSV / TXT table or the
    $SPILLOVER keyword of an FCS file, chosen by the file suffix."""
    suffix = Path(path).suffix.lower()
    if suffix == ".fcs":
        return matrix_from_fcs_file(path)
    if suffix in (".mtx", ".xml", ".txt"):
        return read_matrix_mtx(path)
    return read_matrix_csv(path)


def write_matrix_file(matrix: SpilloverMatrix, path: str | Path, fmt: str | None = None):
    """Write `matrix` as fmt "xml" (FlowJo .mtx), "text" (FlowJo text
    matrix) or "csv"; by default chosen by the suffix (.mtx -> xml, .txt ->
    text, anything else -> csv)."""
    suffix = Path(path).suffix.lower()
    fmt = fmt or {".mtx": "xml", ".xml": "xml", ".txt": "text"}.get(suffix, "csv")
    if fmt == "csv":
        write_matrix_csv(matrix, path)
    else:
        write_matrix_mtx(matrix, path, fmt)


# ------------------------------------------------------------ control sets
@dataclass
class ControlSet:
    """Which samples are the compensation controls, the population of each
    and the user's positive / negative gates (what the wizard last used)."""

    assignment: dict = field(default_factory=dict)  # Sample -> "Unstained" | parameter
    populations: dict = field(default_factory=dict)  # Sample -> Population (None: ungated)
    gates: dict = field(default_factory=dict)  # parameter -> ControlGates
    universal: bool = True
    method: str = "median"
    group: str | None = None

    def singles(self) -> dict:
        return _split_assignment(self.assignment)[0]

    def unstained(self):
        return _split_assignment(self.assignment)[1]

    def to_settings(self) -> dict:
        return {
            "group": self.group,
            "controls": [
                {"sample": s.id, "role": r, "population": (self.populations.get(s).path if self.populations.get(s) is not None else "")}
                for s, r in self.assignment.items()
                if r and r != NOT_USED
            ],
            "gates": {p: g.to_dict() for p, g in self.gates.items() if not g.is_auto},
            "universal_negative": bool(self.universal),
            "method": self.method,
        }

    @classmethod
    def from_settings(cls, ws, d: dict | None) -> "ControlSet | None":
        if not isinstance(d, dict) or not d.get("controls"):
            return None
        cs = cls(universal=bool(d.get("universal_negative", True)), method=d.get("method", "median"), group=d.get("group"))
        for c in d.get("controls", []):
            s = ws.sample_by_id(int(c.get("sample", -1)))
            if s is None:
                continue
            cs.assignment[s] = c.get("role")
            path = c.get("population") or ""
            pop = s.find_population(path) if path else None
            cs.populations[s] = pop
        cs.gates = {p: ControlGates.from_dict(g) for p, g in (d.get("gates") or {}).items()}
        return cs if cs.assignment else None


def guessed_control_set(ws) -> ControlSet | None:
    """Controls guessed from the Compensation group's sample names."""
    comp = next((g for g in ws.groups if g.role == "compensation"), None)
    samples = ws.members(comp) if comp is not None else []
    if not samples:
        return None
    params: list[str] = []
    stains: dict[str, str] = {}
    for s in samples:
        for p in fluorescence_params(s):
            if p not in params:
                params.append(p)
        for ch in s.channels:
            if ch.stain and ch.name not in stains:
                stains[ch.name] = ch.stain
    guess = guess_control_assignment([s.name for s in samples], params, stains)
    cs = ControlSet(group=comp.name)
    for s in samples:
        role = guess.get(s.name)
        if role:
            cs.assignment[s] = role
            cs.populations[s] = default_control_population(s)
    cs.universal = any(r == UNSTAINED for r in cs.assignment.values())
    return cs if cs.singles() else None


def workspace_control_set(ws) -> ControlSet | None:
    """The wizard's stored control set, or one guessed from the Compensation group."""
    return ControlSet.from_settings(ws, ws.settings.get(CONTROLS_KEY)) or guessed_control_set(ws)


def matrix_control_data(cs: ControlSet, matrix: SpilloverMatrix, cache: dict | None = None):
    """(matrix, params, controls, unstained) of a control set restricted to
    the parameters of `matrix`; the returned matrix has the controls' $PnN
    names (a stain-named matrix is matched by $PnS)."""
    singles = cs.singles()
    if not singles:
        raise ValueError("No single-stain controls")
    first = next(iter(singles.values()))
    mm = matrix.for_channels(first.channels)
    if mm is None:
        raise ValueError(f"The controls lack {', '.join(matrix.missing_params(first.channels))}")
    params = list(mm.params)
    assignment = {s: r for s, r in cs.assignment.items() if r == UNSTAINED or r in params}
    if not _split_assignment(assignment)[0]:
        raise ValueError("No control for the parameters of this matrix")
    params, controls, unst, _singles = control_data(assignment, cs.universal, cs.populations, params, cache)
    return mm, params, controls, unst


# ---------------------------------------------------------- group matrices
class GroupMatrixHook(QObject):
    """Keeps group-level compensation: a sample that joins a group with an
    attached matrix (workspace.settings["group_matrices"]) receives it, by
    manual membership or a live keyword criterion. Also renames matrices
    written with $PnS stain names to the samples' $PnN channel names so they
    can be applied. The changes join the undo step that made the sample join."""

    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self._busy = False
        self.last_warnings: list[str] = []
        controller.samplesChanged.connect(self.sync)
        controller.groupsChanged.connect(self.sync)
        controller.workspaceReplaced.connect(self.sync)
        controller.compensationChanged.connect(lambda *_a: self.sync())

    def sync(self) -> int:
        if self._busy:
            return 0
        c = self.controller
        ws = c.workspace
        pending = pending_group_compensation(ws)
        renames = []
        for s in ws.samples:
            comp = s.comp
            if comp is None or not s.channels or all(p in s.channel_names for p in comp.params):
                continue
            mm = comp.for_channels(s.channels)
            if mm is not None and mm is not comp:
                renames.append((s, mm))
        if not pending and not renames:
            return 0
        self._busy = True
        try:
            snap = c.undo_top_snapshot()
            if snap is not None and pending:
                from ..core.serialization import extend_snapshot

                extend_snapshot(ws, snap, "settings")
            for s, mm in renames:
                s.set_compensation(mm, s.comp_enabled)
            done: dict = {}
            warnings: list[str] = []
            for group, s, m in pending:
                mm = m.for_channels(s.channels)
                if mm is None:
                    warnings.append(f"“{m.name}” cannot be applied to {s.name}: {', '.join(m.missing_params(s.channels))} missing")
                else:
                    s.set_compensation(mm.copy(m.name), True)
                done.setdefault(id(group), (group, []))[1].append(s.id)
            for group, ids in done.values():
                mark_group_compensation(ws, group, ids)
            self.last_warnings = warnings
            c.mark_modified()
            c.compensationChanged.emit(None)
            c.gatingChanged.emit(None)
            if warnings:
                c.statusMessage.emit("; ".join(warnings))
        finally:
            self._busy = False
        return len(pending) + len(renames)


def ensure_group_hook(controller) -> GroupMatrixHook:
    """The controller's GroupMatrixHook (created and synced on first use)."""
    hook = getattr(controller, "_group_matrix_hook", None)
    if hook is None:
        hook = GroupMatrixHook(controller)
        controller._group_matrix_hook = hook
        hook.sync()
    return hook


def install(window):
    """Extension entry point: keep group-level compensation working from the
    moment the workspace window opens."""
    return ensure_group_hook(window.controller)


def apply_matrix(controller, samples, matrix: SpilloverMatrix | None, enabled: bool = True) -> tuple[list, list[tuple]]:
    """Apply `matrix` to the samples it fits (stain-named parameters are
    renamed per sample); returns (applied samples, [(sample, missing params)])."""
    if matrix is None:
        if samples:
            controller.set_compensation(list(samples), None)
        return list(samples), []
    same, renamed, problems = [], [], []
    for s in samples:
        mm = matrix.for_channels(s.channels) if s.channels else matrix
        if mm is None:
            problems.append((s, matrix.missing_params(s.channels)))
        elif mm is matrix:
            same.append(s)
        else:
            renamed.append((s, mm))
    if same or renamed:
        with controller.undo_group("apply compensation"):
            if same:
                controller.set_compensation(same, matrix, enabled)
            for s, mm in renamed:
                controller.set_compensation([s], mm, enabled)
    return same + [s for s, _ in renamed], problems


def missing_text(problems: list[tuple], matrix_name: str) -> str:
    lines = [f"“{matrix_name}” cannot be applied to:"]
    for s, missing in problems[:20]:
        lines.append(f"  {s.name}: {', '.join(missing)} missing")
    if len(problems) > 20:
        lines.append(f"  … and {len(problems) - 20} more")
    return "\n".join(lines)


# ------------------------------------------------------- control plot panel
class _PlotTile(QLabel):
    clicked = Signal(int, int)
    wheeled = Signal(int, int, float)

    def __init__(self, i: int, j: int, size: int):
        super().__init__()
        self.i, self.j = i, j
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #ffffff; color: #888;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.i, self.j)
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if not steps or self.i == self.j:
            return super().wheelEvent(event)
        fine = not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self.wheeled.emit(self.i, self.j, steps * (0.1 if fine else 1.0))
        event.accept()


class ControlPlotPanel(QWidget):
    """N x N panel of the single-stain controls compensated with a matrix:
    row = control of fluorochrome i, column = detector j (Comp-i vs Comp-j;
    the diagonal shows the control's histogram). Cells whose residual spill
    exceeds HOTSPOT get a red (under-compensated) or blue (over-compensated)
    frame; clicking selects the matching matrix cell, the mouse wheel nudges
    it by 0.1 % (Shift: 1 %)."""

    cellSelected = Signal(int, int)
    nudged = Signal(int, int, float)

    def __init__(self, parent=None, tile: int = 128):
        super().__init__(parent)
        self.tile = tile
        self.matrix: SpilloverMatrix | None = None
        self.rows: dict[int, tuple] = {}  # matrix row -> (sample, population)
        self.residuals: np.ndarray | None = None
        self.selected: tuple[int, int] | None = None
        self.images: dict[tuple[int, int], object] = {}
        self.tiles: dict[tuple[int, int], _PlotTile] = {}
        self._shape: tuple = ()
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #666;")
        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.grid.setSpacing(3)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.inner)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.message)
        lay.addWidget(self.scroll, 1)

    def _rebuild(self, params: list[str]):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.tiles = {}
        n = len(params)
        for j, p in enumerate(params):
            lab = QLabel(p)
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setStyleSheet("font-weight: bold;")
            self.grid.addWidget(lab, 0, j + 1)
        for i, p in enumerate(params):
            lab = QLabel(f"{p}\ncontrol")
            lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            lab.setStyleSheet("font-weight: bold;")
            self.grid.addWidget(lab, i + 1, 0)
            for j in range(n):
                t = _PlotTile(i, j, self.tile)
                t.clicked.connect(self._on_clicked)
                t.wheeled.connect(self.nudged)
                self.grid.addWidget(t, i + 1, j + 1)
                self.tiles[(i, j)] = t
        self._shape = tuple(params)

    def set_content(self, matrix: SpilloverMatrix | None, rows: dict[int, tuple], residuals: np.ndarray | None, message: str = ""):
        self.matrix = matrix
        self.rows = dict(rows)
        self.residuals = residuals
        params = list(matrix.params) if matrix is not None else []
        if tuple(params) != self._shape:
            self._rebuild(params)
        self.message.setText(message)
        self.render()

    def _render_tile(self, i: int, j: int, copies: dict):
        from ..core.model import GraphSettings
        from .renderer import RenderOptions, render_plot

        row = self.rows.get(i)
        if row is None or self.matrix is None:
            return None
        sample, population = row
        entry = copies.get(i)
        if entry is None:
            mm = self.matrix.for_channels(sample.channels)
            if mm is None or not sample.ensure_loaded():
                copies[i] = (None, None, None)
                return None
            copy = sample.detached_copy()
            copy.set_compensation(mm.copy(self.matrix.name), True)
            pop = copy.find_population(population.path) if population is not None and population.parent is not None else None
            entry = (copy, pop or copy.root, mm)
            copies[i] = entry
        copy, pop, mm = entry
        if copy is None:
            return None
        pi, pj = mm.params[i], mm.params[j]
        if i == j:
            graph = GraphSettings(x=mm.compensated_name(pi), y=None, plot_type="histogram")
        else:
            graph = GraphSettings(x=mm.compensated_name(pi), y=mm.compensated_name(pj), plot_type="pseudocolor")
        opts = RenderOptions(plot_type=graph.plot_type, show_axes=False, show_gates=False, dots_to_draw=6000, font_dpi=72.0, show_legend=False)
        return render_plot(copy, pop, graph, self.tile, self.tile, opts).image

    def render(self):
        self.images = {}
        copies: dict = {}
        for (i, j), tile in self.tiles.items():
            try:
                img = self._render_tile(i, j, copies)
            except Exception as exc:  # noqa: BLE001 - one bad control must not break the panel
                img = None
                tile.setToolTip(f"Could not draw: {exc}")
            self.images[(i, j)] = img
        self._decorate()

    def hot(self, i: int, j: int) -> float | None:
        r = self.residuals
        if r is None or i == j or i >= r.shape[0] or j >= r.shape[1] or not np.isfinite(r[i, j]):
            return None
        return float(r[i, j]) if abs(r[i, j]) > HOTSPOT else None

    def _decorate(self):
        for (i, j), tile in self.tiles.items():
            img = self.images.get((i, j))
            if img is None:
                tile.setPixmap(QPixmap())
                tile.setText("no control" if i not in self.rows else "")
                continue
            pm = QPixmap.fromImage(img)
            p = QPainter(pm)
            hot = self.hot(i, j)
            if hot is not None:
                p.setPen(QPen(QColor("#d01c1c" if hot > 0 else "#1f5fd0"), 4))
                p.drawRect(QRectF(2, 2, pm.width() - 4, pm.height() - 4))
                tile.setToolTip(_hot_text(hot))
            else:
                r = self.residuals
                res = "" if r is None or i == j or not np.isfinite(r[i, j]) else f"\nresidual spill {r[i, j] * 100:+.2f} %"
                tile.setToolTip(f"{self.matrix.params[i]} control: {self.matrix.params[j]}{res}" if self.matrix else "")
            if self.selected == (i, j):
                p.setPen(QPen(QColor("#ff9800"), 3, Qt.PenStyle.DashLine))
                p.drawRect(QRectF(6, 6, pm.width() - 12, pm.height() - 12))
            p.end()
            tile.setPixmap(pm)

    def select(self, i: int, j: int):
        if self.selected != (i, j):
            self.selected = (i, j)
            self._decorate()

    def _on_clicked(self, i: int, j: int):
        self.select(i, j)
        self.cellSelected.emit(i, j)


# ---------------------------------------------------------------- window
class _Entry:
    __slots__ = ("label", "matrix", "stored")

    def __init__(self, label: str, matrix: SpilloverMatrix, stored: bool):
        self.label = label
        self.matrix = matrix
        self.stored = stored


def _matrix_key(m: SpilloverMatrix | None):
    return None if m is None else (m.name, tuple(m.params), m.matrix.tobytes())


class _GroupApplyDialog(QDialog):
    def __init__(self, parent, groups, current: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Apply to Group")
        self.combo = QComboBox()
        for g in groups:
            self.combo.addItem(icon("group"), g.name)
        self.combo.setCurrentIndex(current)
        self.attach = QCheckBox("Attach to the group: samples added later inherit the matrix")
        self.attach.setChecked(True)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form = QFormLayout(self)
        form.addRow("Group:", self.combo)
        form.addRow(self.attach)
        form.addRow(bb)


class CompensationWindow(QMainWindow):
    """Matrix list, matrix editor, control plots, spreading matrix and
    per-sample compensation status."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Compensation")
        self.setWindowIcon(icon("compensation"))
        self.resize(1080, 720)
        self._entries: list[_Entry] = []
        self._current_label: str | None = None
        self._wizard: CompensationWizard | None = None
        self.view_mode = "spill"  # or "inverse"
        self.last_warning = ""
        self.last_message = ""
        self._box: QMessageBox | None = None
        self._event_cache: dict = {}
        self._residuals: tuple | None = None  # (matrix key, residual array)
        self.ssm: np.ndarray | None = None
        self.ssm_params: list[str] = []
        ensure_group_hook(controller)

        self._build_toolbar()

        self.matrix_list = QListWidget()
        self.matrix_list.setMinimumWidth(200)
        self.matrix_list.currentRowChanged.connect(self._on_matrix_selected)
        self.matrix_list.itemDoubleClicked.connect(lambda _i: self.rename_matrix())

        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-weight: bold;")
        self.title_label.setWordWrap(True)
        self.grid = QTableWidget()
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.grid.cellChanged.connect(self._on_cell_changed)
        self.grid.currentCellChanged.connect(lambda r, c, _pr, _pc: self._on_grid_current(r, c))
        self.hint = QLabel()
        self.hint.setStyleSheet("color: #666;")
        matrix_tab = QWidget()
        ml = QVBoxLayout(matrix_tab)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.grid, 1)
        ml.addWidget(self.hint)

        # control plots
        self.panel = ControlPlotPanel()
        self.panel.cellSelected.connect(self._on_panel_selected)
        self.panel.nudged.connect(self._on_panel_nudged)
        self.cell_label = QLabel("Click a plot to select its matrix cell")
        self.cell_spin = QDoubleSpinBox()
        self.cell_spin.setRange(-1000.0, 1000.0)
        self.cell_spin.setDecimals(2)
        self.cell_spin.setSingleStep(0.1)
        self.cell_spin.setSuffix(" %")
        self.cell_spin.setKeyboardTracking(False)
        self.cell_spin.setEnabled(False)
        self.cell_spin.valueChanged.connect(self._on_cell_spin)
        legend = QLabel("<span style='color:#d01c1c'>■</span> under-compensated &nbsp; <span style='color:#1f5fd0'>■</span> over-compensated "
                        f"(|residual| &gt; {HOTSPOT * 100:g} %) • wheel over a plot: ±0.1 % (Shift ±1 %)")
        legend.setStyleSheet("color: #666;")
        bar = QHBoxLayout()
        bar.addWidget(self.cell_label)
        bar.addWidget(self.cell_spin)
        bar.addStretch(1)
        bar.addWidget(legend)
        plots_tab = QWidget()
        pl = QVBoxLayout(plots_tab)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.panel, 1)
        pl.addLayout(bar)

        # spreading matrix
        self.ssm_grid = QTableWidget()
        self.ssm_grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ssm_grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.ssm_message = QLabel("Spillover spreading of the controls compensated with the selected matrix (Nguyen et al.): "
                                  "row = fluorochrome, column = detector, units √intensity.")
        self.ssm_message.setWordWrap(True)
        self.ssm_message.setStyleSheet("color: #666;")
        self.ssm_button = QPushButton("Compute")
        self.ssm_button.clicked.connect(lambda: self.compute_ssm())
        self.ssm_export = QPushButton("Export CSV…")
        self.ssm_export.clicked.connect(lambda: self.export_ssm())
        sb = QHBoxLayout()
        sb.addWidget(self.ssm_button)
        sb.addWidget(self.ssm_export)
        sb.addStretch(1)
        ssm_tab = QWidget()
        sl = QVBoxLayout(ssm_tab)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addLayout(sb)
        sl.addWidget(self.ssm_grid, 1)
        sl.addWidget(self.ssm_message)

        self.tabs = QTabWidget()
        self.tabs.addTab(matrix_tab, "Spillover Matrix")
        self.tabs.addTab(plots_tab, "Control Plots")
        self.tabs.addTab(ssm_tab, "Spreading Matrix")
        self.tabs.currentChanged.connect(lambda _i: self._schedule_panel())

        centre = QWidget()
        cl = QVBoxLayout(centre)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.addWidget(self.title_label)
        cl.addWidget(self.tabs, 1)

        self.status_table = QTableWidget(0, 3)
        self.status_table.setHorizontalHeaderLabels(["Sample", "Matrix in use", "Enabled"])
        self.status_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.status_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.status_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.status_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.status_table.cellDoubleClicked.connect(self._on_status_double_clicked)
        self.status_table.itemChanged.connect(self._on_status_item_changed)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.matrix_list)
        top.addWidget(centre)
        top.setStretchFactor(1, 1)
        top.setSizes([240, 840])
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.status_table)
        main.setSizes([480, 240])
        self.setCentralWidget(main)

        self._panel_timer = QTimer(self)
        self._panel_timer.setSingleShot(True)
        self._panel_timer.setInterval(0)
        self._panel_timer.timeout.connect(self.refresh_panel)

        controller.compensationChanged.connect(lambda *_: self.refresh())
        controller.samplesChanged.connect(self._data_changed)
        controller.workspaceReplaced.connect(self._data_changed)
        controller.gatingChanged.connect(lambda *_: self._gating_changed())
        controller.groupsChanged.connect(self._update_actions)
        controller.selectionChanged.connect(self._update_actions)
        for sig in (controller.settingsChanged, controller.modifiedChanged, controller.groupsChanged, controller.gatingChanged, controller.layoutsChanged, controller.tablesChanged):
            sig.connect(lambda *_a: self._sync_undo())
        self.refresh()

    # ------------------------------------------------------------ toolbar
    def _build_toolbar(self):
        tb = QToolBar("Compensation")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.addToolBar(tb)

        def act(text, ic, slot, tip="", shortcut=None):
            a = QAction(icon(ic), text, self)
            a.triggered.connect(lambda _checked=False: slot())
            if tip:
                a.setToolTip(tip)
                a.setStatusTip(tip)
            if shortcut is not None:
                a.setShortcuts(shortcut)  # every platform binding of a standard key
            return a

        # Undo / Redo work while this window is active (the workspace window's
        # shortcuts only reach that window)
        self.act_undo = act("Undo", "undo", self.controller.undo, "Undo the last change", QKeySequence.StandardKey.Undo)
        self.act_redo = act("Redo", "redo", self.controller.redo, "Redo", QKeySequence.StandardKey.Redo)
        self.addAction(self.act_undo)
        self.addAction(self.act_redo)
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()

        self.act_new = act("New", "plus", self.new_matrix, "New identity matrix for the selected sample's parameters")
        self.act_duplicate = act("Duplicate", "copy", self.duplicate_matrix)
        self.act_rename = act("Rename", "rename", self.rename_matrix)
        self.act_delete = act("Delete", "delete", self.delete_matrix)
        self.act_import = act("Import…", "import", self.import_csv, "Import a matrix: FlowJo .mtx (XML or text), CSV / TXT table, or the $SPILLOVER of an FCS file")
        self.act_export = act("Export…", "export", self.export_csv, "Export the matrix as FlowJo .mtx (XML or text) or CSV")
        self.act_copy = act("Copy", "copy", self.copy_matrix, "Copy the matrix (tab separated, %) to the clipboard", QKeySequence.StandardKey.Copy)
        self.act_paste = act("Paste", "paste", self.paste_matrix, "Paste a matrix from the clipboard (spreadsheet cells)", QKeySequence.StandardKey.Paste)
        for a in (self.act_copy, self.act_paste):
            a.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.addAction(a)
        self.act_inverse = QAction(icon("compensation"), "Inverse", self)
        self.act_inverse.setCheckable(True)
        self.act_inverse.setToolTip("Show the compensation (inverse) matrix instead of the spillover %")
        self.act_inverse.toggled.connect(lambda on: self.set_view_mode("inverse" if on else "spill"))
        self.act_wizard = act("Compensation Wizard…", "wizard", self.open_wizard)
        self.act_refine = act("AutoSpill", "wizard", self.refine_current, "Refine the matrix by AutoSpill-style iterative robust regression on the compensation controls")
        for a in (self.act_new, self.act_duplicate, self.act_rename, self.act_delete):
            tb.addAction(a)
        tb.addSeparator()
        for a in (self.act_import, self.act_export, self.act_copy, self.act_paste):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_inverse)
        tb.addAction(self.act_wizard)
        tb.addAction(self.act_refine)
        tb.addSeparator()

        self.act_apply_selected = act("Apply to selected samples", "check", self.apply_to_selected)
        self.act_apply_group = act("Apply to group…", "group", self.apply_to_group_dialog, "Apply to a group; attached, samples added to the group later inherit the matrix")
        self.act_apply_all = act("Apply to all samples", "play", self.apply_to_all)
        self.act_detach = act("Detach from groups", "minus", self.detach_from_groups, "Samples added later no longer inherit this matrix")
        self.act_remove = act("Remove compensation from selected samples", "minus", self.remove_from_selected)
        menu = QMenu(self)
        for a in (self.act_apply_selected, self.act_apply_group, self.act_apply_all):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(self.act_detach)
        menu.addAction(self.act_remove)
        self.apply_button = QToolButton()
        self.apply_button.setText("Apply to…")
        self.apply_button.setIcon(icon("play"))
        self.apply_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.apply_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.apply_button.setMenu(menu)
        self.apply_button.clicked.connect(self.apply_to_selected)
        tb.addWidget(self.apply_button)

    def _update_actions(self):
        entry = self.current_entry()
        has = entry is not None
        self.act_duplicate.setEnabled(has)
        self.act_rename.setEnabled(has and entry.stored)
        self.act_delete.setEnabled(has and entry.stored)
        self.act_export.setEnabled(has)
        self.act_copy.setEnabled(has)
        self.act_refine.setEnabled(has)
        self.act_apply_selected.setEnabled(has)
        self.act_apply_group.setEnabled(has)
        self.act_apply_all.setEnabled(has)
        self.act_detach.setEnabled(has and bool(groups_with_matrix(self.workspace, entry.label)))
        self.apply_button.setEnabled(True)
        self._sync_undo()

    def _sync_undo(self):
        c = self.controller
        self.act_undo.setEnabled(c.can_undo)
        self.act_redo.setEnabled(c.can_redo)
        label = c.undo_label()
        self.act_undo.setToolTip(f"Undo {label}" if label else "Undo")

    def _warn(self, title: str, text: str):
        """A non-blocking warning (also kept in last_warning)."""
        self.last_warning = text
        self._show_box(QMessageBox.Icon.Warning, title, text)

    def _inform(self, title: str, text: str):
        self.last_message = text
        self._show_box(QMessageBox.Icon.Information, title, text)

    def _show_box(self, kind, title: str, text: str):
        if self._box is not None:
            self._box.close()
        box = QMessageBox(kind, title, text, QMessageBox.StandardButton.Ok, self)
        box.setModal(False)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.destroyed.connect(lambda *_a: setattr(self, "_box", None))
        self._box = box
        box.show()

    # ------------------------------------------------------------ model
    @property
    def workspace(self):
        return self.controller.workspace

    def available_matrices(self) -> list[tuple[str, SpilloverMatrix, bool]]:
        """(label, matrix, stored_in_workspace) for every matrix in the list."""
        out: list[tuple[str, SpilloverMatrix, bool]] = []
        for name, m in self.workspace.matrices.items():
            out.append((name, m, True))
        acq: list[tuple[SpilloverMatrix, str]] = []
        for s in self.workspace.samples:
            found = []
            spill = parse_spillover(s.keywords) if s.keywords else None
            if spill is not None:
                found.append(SpilloverMatrix(ACQUISITION_DEFINED, spill[0], spill[1]))
            if s.comp is not None and s.comp.name == ACQUISITION_DEFINED:
                found.append(s.comp)
            for m in found:
                if not any(m == other for other, _ in acq):
                    acq.append((m, s.name))
        stored_names = set(self.workspace.matrices)
        for m, sname in acq:
            label = ACQUISITION_DEFINED if len(acq) == 1 else f"{ACQUISITION_DEFINED} ({sname})"
            label = _unique_name(stored_names, label) if label in stored_names else label
            out.append((label, m.copy(label), False))
        return out

    def matrix_labels(self) -> list[str]:
        return [e.label for e in self._entries]

    def current_entry(self) -> _Entry | None:
        row = self.matrix_list.currentRow()
        return self._entries[row] if 0 <= row < len(self._entries) else None

    def current_matrix(self) -> SpilloverMatrix | None:
        e = self.current_entry()
        return e.matrix if e is not None else None

    def select_matrix(self, label: str) -> bool:
        for i, e in enumerate(self._entries):
            if e.label == label:
                self.matrix_list.setCurrentRow(i)
                return True
        return False

    # ------------------------------------------------------------ refresh
    def _data_changed(self):
        self._event_cache.clear()
        self._residuals = None
        self.refresh()

    def _gating_changed(self):
        # control populations may have moved: re-read their events
        self._event_cache.clear()
        self._residuals = None

    def refresh(self):
        self._entries = [_Entry(l, m, st) for l, m, st in self.available_matrices()]
        keep = self._current_label
        blocked = self.matrix_list.blockSignals(True)
        self.matrix_list.clear()
        row = -1
        for i, e in enumerate(self._entries):
            item = QListWidgetItem(icon("compensation" if e.stored else "sample"), e.label)
            n = len(e.matrix.params)
            groups = groups_with_matrix(self.workspace, e.label) if e.stored else []
            tip = f"{n}×{n}: {', '.join(e.matrix.params)}" + ("" if e.stored else "\n(read from the FCS $SPILLOVER keyword)")
            if groups:
                tip += "\nAttached to: " + ", ".join(g.name for g in groups)
            item.setToolTip(tip)
            if not e.stored:
                f = item.font()
                f.setItalic(True)
                item.setFont(f)
            self.matrix_list.addItem(item)
            if e.label == keep:
                row = i
        if row < 0 and self._entries:
            row = 0
        self.matrix_list.setCurrentRow(row)
        self.matrix_list.blockSignals(blocked)
        self._show_current()
        self._refresh_status()
        self._update_actions()

    def _residual_key(self, m: SpilloverMatrix):
        # residuals depend on the matrix and on the controls / gates in use
        return (_matrix_key(m), repr(self.workspace.settings.get(CONTROLS_KEY)))

    def _hotspots(self, m: SpilloverMatrix | None):
        r = self._residuals
        if m is None or r is None or r[0] != self._residual_key(m):
            return None
        return r[1]

    def _show_current(self):
        e = self.current_entry()
        self._current_label = e.label if e else None
        if e is None:
            self.title_label.setText("No matrix — create one or run the Compensation Wizard")
        else:
            kind = "workspace matrix" if e.stored else "acquisition-defined, edits create a copy"
            groups = groups_with_matrix(self.workspace, e.label) if e.stored else []
            extra = f" • attached to {', '.join(g.name for g in groups)}" if groups else ""
            self.title_label.setText(f"{e.label}  ({kind}){extra}")
        m = e.matrix if e else None
        fill_matrix_grid(self.grid, m, editable=True, mode=self.view_mode, hotspots=self._hotspots(m))
        if self.view_mode == "inverse":
            self.hint.setText("Compensation (inverse) matrix × 100, read-only: compensated = observed × inverse")
        else:
            self.hint.setText("Rows: source fluorochrome • Columns: detector • values are % spillover")
        self._schedule_panel()

    def _on_matrix_selected(self, _row: int):
        self._show_current()
        self._update_actions()

    def set_view_mode(self, mode: str):
        mode = "inverse" if mode == "inverse" else "spill"
        if mode != self.view_mode:
            self.view_mode = mode
            blocked = self.act_inverse.blockSignals(True)
            self.act_inverse.setChecked(mode == "inverse")
            self.act_inverse.blockSignals(blocked)
            self._show_current()

    def _refresh_status(self):
        t = self.status_table
        blocked = t.blockSignals(True)
        samples = self.workspace.samples
        t.setRowCount(len(samples))
        for r, s in enumerate(samples):
            it = QTableWidgetItem(icon("sample"), s.name)
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            t.setItem(r, 0, it)
            if s.comp is None:
                mi = QTableWidgetItem("none")
            else:
                missing = s.comp.missing_params(s.channels) if s.channels else []
                if missing:
                    mi = QTableWidgetItem(f"{s.comp.name} — not applicable ({', '.join(missing)} missing)")
                    mi.setForeground(QBrush(QColor("#c00000")))
                    mi.setToolTip(f"The sample has no {', '.join(missing)}: the matrix cannot be applied")
                else:
                    mi = QTableWidgetItem(s.comp.name)
            t.setItem(r, 1, mi)
            en = QTableWidgetItem()
            en.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            en.setCheckState(Qt.CheckState.Checked if s.comp_enabled and s.comp is not None else Qt.CheckState.Unchecked)
            t.setItem(r, 2, en)
        t.blockSignals(blocked)

    def _on_status_double_clicked(self, row: int, _col: int):
        if not (0 <= row < len(self.workspace.samples)):
            return
        comp = self.workspace.samples[row].comp
        if comp is None:
            return
        for i, e in enumerate(self._entries):
            if e.matrix.name == comp.name and e.matrix == comp:
                self.matrix_list.setCurrentRow(i)
                return
        for i, e in enumerate(self._entries):
            if e.matrix == comp:
                self.matrix_list.setCurrentRow(i)
                return

    def _on_status_item_changed(self, item: QTableWidgetItem):
        if item.column() != 2:
            return
        row = item.row()
        if not (0 <= row < len(self.workspace.samples)):
            return
        s = self.workspace.samples[row]
        enabled = item.checkState() == Qt.CheckState.Checked
        if s.comp is None:
            self._refresh_status()
            return
        if enabled != s.comp_enabled:
            self.controller.set_compensation([s], s.comp, enabled)

    # ------------------------------------------------------------ editing
    def _store_following(self, e: _Entry, m: SpilloverMatrix):
        """Store `m` (the edited version of entry `e`); samples using the
        stored matrix follow the edit. Call inside an undo group."""
        users = [s for s in self.workspace.samples if e.stored and s.comp is not None and s.comp.name == m.name]
        self._current_label = m.name
        self.controller.store_matrix(m)
        for enabled in (True, False):
            group = [s for s in users if s.comp_enabled == enabled]
            if group:
                apply_matrix(self.controller, group, m, enabled)
        if self._current_label != (self.current_entry().label if self.current_entry() else None):
            self.select_matrix(m.name)

    def set_cell(self, row: int, col: int, percent: float):
        """Set one spill value (in percent) of the current matrix."""
        e = self.current_entry()
        if e is None or row == col:
            return
        n = len(e.matrix.params)
        if not (0 <= row < n and 0 <= col < n):
            return
        with self.controller.undo_group('edit matrix', kind="matrices"):
            if e.stored:
                m = e.matrix.copy()
            else:
                m = e.matrix.copy(_unique_name(self.workspace.matrices, f"{e.label} (edited)"))
            m.matrix[row, col] = percent / 100.0
            self._store_following(e, m)

    def _on_cell_changed(self, row: int, col: int):
        item = self.grid.item(row, col)
        e = self.current_entry()
        if item is None or e is None:
            return
        if self.view_mode == "inverse":
            self._show_current()
            return
        text = item.text().strip().rstrip("%").replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            fill_matrix_grid(self.grid, e.matrix, hotspots=self._hotspots(e.matrix))
            return
        if abs(value / 100.0 - e.matrix.matrix[row, col]) < 1e-12:
            fill_matrix_grid(self.grid, e.matrix, hotspots=self._hotspots(e.matrix))
            return
        self.set_cell(row, col, value)

    # ------------------------------------------------------------ control plots
    def _schedule_panel(self):
        if self.tabs.currentIndex() == 1:
            self._panel_timer.start()

    def control_rows(self, matrix: SpilloverMatrix) -> tuple[dict, str]:
        """Matrix row -> (control sample, population) and a message."""
        cs = workspace_control_set(self.workspace)
        if cs is None:
            return {}, "No compensation controls: run the Compensation Wizard or put single-stain controls in the Compensation group."
        rows = {}
        for role, s in cs.singles().items():
            mm = matrix.for_channels(s.channels)
            if mm is None:
                continue
            for i, p in enumerate(mm.params):
                if p == role and i not in rows:
                    rows[i] = (s, cs.populations.get(s))
        missing = [matrix.params[i] for i in range(len(matrix.params)) if i not in rows]
        msg = f"Single-stain controls compensated with “{matrix.name}” (x: own detector, y: column detector)."
        if missing:
            msg += f" No control for {', '.join(missing)}."
        return rows, msg

    def residuals_for(self, matrix: SpilloverMatrix) -> np.ndarray | None:
        """Residual spill of the controls compensated with `matrix` (cached)."""
        key = self._residual_key(matrix)
        if self._residuals is not None and self._residuals[0] == key:
            return self._residuals[1]
        cs = workspace_control_set(self.workspace)
        if cs is None:
            return None
        try:
            mm, params, controls, unst = matrix_control_data(cs, matrix, self._event_cache)
        except (ValueError, KeyError):
            return None
        res = residual_spill(controls, params, mm, unst, gates=cs.gates)
        self._residuals = (key, res)
        return res

    def refresh_panel(self):
        m = self.current_matrix()
        if m is None:
            self.panel.set_content(None, {}, None, "No matrix selected.")
            return
        rows, msg = self.control_rows(m)
        res = self.residuals_for(m) if rows else None
        hot = int(np.sum(np.abs(np.nan_to_num(res)) > HOTSPOT)) if res is not None else 0
        if hot:
            msg += f" {hot} hot spot(s)."
        self.panel.set_content(m, rows, res, msg)
        if res is not None and self.view_mode == "spill":
            fill_matrix_grid(self.grid, m, editable=True, mode=self.view_mode, hotspots=res)

    def _on_panel_selected(self, i: int, j: int):
        blocked = self.grid.blockSignals(True)
        self.grid.setCurrentCell(i, j)
        self.grid.blockSignals(blocked)
        self._show_cell(i, j)

    def _on_grid_current(self, i: int, j: int):
        if i < 0 or j < 0:
            return
        self.panel.select(i, j)
        self._show_cell(i, j)

    def _show_cell(self, i: int, j: int):
        m = self.current_matrix()
        if m is None or not (0 <= i < len(m.params) and 0 <= j < len(m.params)):
            return
        self._cell = (i, j)
        text = f"{m.params[i]} → {m.params[j]}:"
        hot = self.panel.hot(i, j)
        if hot is not None:
            text = f"{text} ({_hot_text(hot)})"
        self.cell_label.setText(text)
        blocked = self.cell_spin.blockSignals(True)
        self.cell_spin.setValue(float(m.matrix[i, j]) * 100.0)
        self.cell_spin.blockSignals(blocked)
        self.cell_spin.setEnabled(i != j)

    def _on_cell_spin(self, value: float):
        cell = getattr(self, "_cell", None)
        m = self.current_matrix()
        if cell is None or m is None:
            return
        i, j = cell
        if i != j and abs(value / 100.0 - m.matrix[i, j]) > 1e-12:
            self.set_cell(i, j, value)
            self._show_cell(i, j)

    def _on_panel_nudged(self, i: int, j: int, delta: float):
        m = self.current_matrix()
        if m is None or i == j:
            return
        self.panel.select(i, j)
        self.set_cell(i, j, float(m.matrix[i, j]) * 100.0 + delta)
        self._show_cell(i, j)

    # ------------------------------------------------------------ spreading matrix
    def compute_ssm(self) -> np.ndarray | None:
        m = self.current_matrix()
        if m is None:
            return None
        cs = workspace_control_set(self.workspace)
        if cs is None:
            self.ssm_message.setText("No compensation controls: run the Compensation Wizard first.")
            return None
        try:
            mm, params, controls, unst = matrix_control_data(cs, m, self._event_cache)
        except (ValueError, KeyError) as exc:
            self.ssm_message.setText(f"Cannot compute the spreading matrix: {exc}")
            return None
        self.ssm = spillover_spreading(controls, params, mm, unst, gates=cs.gates)
        self.ssm_params = list(params)
        fill_values_grid(self.ssm_grid, params, self.ssm)
        finite = self.ssm[np.isfinite(self.ssm)]
        worst = ""
        if finite.size and float(finite.max()) > 0:
            i, j = np.unravel_index(np.nanargmax(self.ssm), self.ssm.shape)
            worst = f" Largest spread: {params[i]} → {params[j]} ({self.ssm[i, j]:.2f})."
        self.ssm_message.setText(f"Spillover spreading of the controls compensated with “{m.name}” (Nguyen et al.): row = fluorochrome, "
                                 f"column = detector, units √intensity.{worst}")
        return self.ssm

    def export_ssm(self, path: str | None = None):
        if self.ssm is None and self.compute_ssm() is None:
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Spreading Matrix", "spreading_matrix.csv", "CSV files (*.csv)")
            if not path:
                return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([""] + self.ssm_params)
            for p, row in zip(self.ssm_params, self.ssm):
                w.writerow([p] + ["" if not np.isfinite(v) else f"{v:.6g}" for v in row])

    # ------------------------------------------------------------ AutoSpill
    def refine_current(self) -> SpilloverMatrix | None:
        """Refine the selected matrix by AutoSpill on the compensation
        controls; stores the result as a new matrix."""
        e = self.current_entry()
        if e is None:
            return None
        cs = workspace_control_set(self.workspace)
        if cs is None:
            self._warn("AutoSpill", "No compensation controls: run the Compensation Wizard or put the single-stain controls in the Compensation group.")
            return None
        try:
            mm, params, controls, unst = matrix_control_data(cs, e.matrix, self._event_cache)
        except (ValueError, KeyError) as exc:
            self._warn("AutoSpill", f"Cannot refine “{e.label}”: {exc}")
            return None
        name = _unique_name(self.workspace.matrices, f"{e.label} (AutoSpill)")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            refined = refine_spillover(controls, params, mm, unstained=unst, gates=cs.gates, name=name)
        finally:
            QApplication.restoreOverrideCursor()
        m = refined.copy(name)
        m.prefix, m.suffix = e.matrix.prefix, e.matrix.suffix
        with self.controller.undo_group("AutoSpill refinement", kind="matrices"):
            self._current_label = m.name
            self.controller.store_matrix(m)
            self.select_matrix(m.name)
        self._inform("AutoSpill", f"“{m.name}”: {refined.report.summary()}")
        return m

    # ------------------------------------------------------------ matrix actions
    def _reference_sample(self):
        sel = self.controller.selected_samples()
        if sel:
            return sel[0]
        return self.workspace.samples[0] if self.workspace.samples else None

    def new_matrix(self) -> SpilloverMatrix | None:
        s = self._reference_sample()
        if s is None:
            QMessageBox.information(self, "Compensation", "Add samples to the workspace first.")
            return None
        params = fluorescence_params(s)
        if not params:
            QMessageBox.information(self, "Compensation", f"{s.name} has no fluorescence parameters.")
            return None
        m = SpilloverMatrix.identity(params, _unique_name(self.workspace.matrices, "Matrix"))
        with self.controller.undo_group("new matrix", kind="matrices"):
            self._current_label = m.name
            self.controller.store_matrix(m)
            self.select_matrix(m.name)
        return m

    def duplicate_matrix(self) -> SpilloverMatrix | None:
        e = self.current_entry()
        if e is None:
            return None
        m = e.matrix.copy(_unique_name(self.workspace.matrices, f"{e.label} copy"))
        with self.controller.undo_group("duplicate matrix", kind="matrices"):
            self._current_label = m.name
            self.controller.store_matrix(m)
            self.select_matrix(m.name)
        return m

    def rename_matrix(self, new_name: str | None = None):
        e = self.current_entry()
        if e is None or not e.stored:
            return
        if new_name is None:
            new_name, ok = QInputDialog.getText(self, "Rename Matrix", "Name:", QLineEdit.EchoMode.Normal, e.label)
            if not ok:
                return
        new_name = new_name.strip()
        if not new_name or new_name == e.label:
            return
        if new_name in self.workspace.matrices:
            QMessageBox.warning(self, "Rename Matrix", f"A matrix named “{new_name}” already exists.")
            return
        with self.controller.undo_group('rename matrix', kind=("matrices", "settings")):
            old = e.label
            m = e.matrix.copy(new_name)
            users = [s for s in self.workspace.samples if s.comp is not None and s.comp.name == old]
            # group attachments follow the new name
            for g in groups_with_matrix(self.workspace, old):
                applied = [s.id for s in self.workspace.members(g)]
                set_group_matrix(self.workspace, g, new_name, applied)
            self._current_label = new_name
            self.controller.remove_matrix(old)
            self.controller.store_matrix(m)
            for enabled in (True, False):
                group = [s for s in users if s.comp_enabled == enabled]
                if group:
                    apply_matrix(self.controller, group, m, enabled)
            self.select_matrix(new_name)

    def delete_matrix(self, confirm: bool = True):
        e = self.current_entry()
        if e is None or not e.stored:
            return
        if confirm:
            res = QMessageBox.question(self, "Delete Matrix", f"Delete the matrix “{e.label}”?\nSamples using it keep their copy.")
            if res != QMessageBox.StandardButton.Yes:
                return
        with self.controller.undo_group("delete matrix", kind=("matrices", "settings")):
            for g in groups_with_matrix(self.workspace, e.label):
                set_group_matrix(self.workspace, g, None)
            self._current_label = None
            self.controller.remove_matrix(e.label)

    def _store_imported(self, m: SpilloverMatrix) -> SpilloverMatrix:
        m.name = _unique_name(self.workspace.matrices, m.name)
        with self.controller.undo_group("import matrix", kind="matrices"):
            self._current_label = m.name
            self.controller.store_matrix(m)
            self.select_matrix(m.name)
        return m

    def import_matrix(self, path: str | None = None) -> SpilloverMatrix | None:
        """Import a matrix from a FlowJo .mtx (XML or text), a CSV / TXT
        table or another FCS file's $SPILLOVER."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import Matrix", "",
                "Matrix files (*.mtx *.csv *.txt *.xml *.fcs);;FlowJo matrix (*.mtx *.xml *.txt);;CSV files (*.csv *.txt);;FCS file $SPILLOVER (*.fcs);;All files (*)",
            )
            if not path:
                return None
        try:
            m = read_matrix_file(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import Matrix", f"Could not read the matrix:\n{exc}")
            return None
        return self._store_imported(m)

    import_csv = import_matrix  # older name (CSV was the only format)

    def export_matrix(self, path: str | None = None, fmt: str | None = None):
        m = self.current_matrix()
        if m is None:
            return
        if path is None:
            filters = "FlowJo matrix (*.mtx);;FlowJo text matrix (*.txt);;CSV files (*.csv)"
            path, chosen = QFileDialog.getSaveFileName(self, "Export Matrix", f"{m.name}.mtx", filters)
            if not path:
                return
            fmt = "text" if chosen.startswith("FlowJo text") else "csv" if chosen.startswith("CSV") else "xml"
            suffix = {"xml": ".mtx", "text": ".txt", "csv": ".csv"}[fmt]
            if not Path(path).suffix:
                path += suffix
        try:
            write_matrix_file(m, path, fmt)
        except OSError as exc:
            QMessageBox.warning(self, "Export Matrix", f"Could not write the file:\n{exc}")

    def export_csv(self, path: str | None = None):
        self.export_matrix(path, "csv" if path is not None and Path(path).suffix.lower() == ".csv" else None)

    def copy_matrix(self) -> str:
        """Put the spillover matrix (%, tab separated, labelled) on the clipboard."""
        m = self.current_matrix()
        if m is None:
            return ""
        text = matrix_to_text(m, "\t", percent=True)
        QApplication.clipboard().setText(text)
        return text

    def paste_matrix(self, text: str | None = None) -> SpilloverMatrix | None:
        """Matrix cells from the clipboard: they replace the values of the
        current workspace matrix when the parameters match, otherwise they
        become a new matrix."""
        if text is None:
            text = QApplication.clipboard().text()
        try:
            m = matrix_from_text(text or "", "Pasted matrix")
        except ValueError as exc:
            self._warn("Paste Matrix", f"The clipboard holds no matrix:\n{exc}")
            return None
        e = self.current_entry()
        if e is not None and e.stored and list(e.matrix.params) == list(m.params):
            new = e.matrix.copy()
            new.matrix = m.matrix.copy()
            with self.controller.undo_group("paste matrix", kind="matrices"):
                self._store_following(e, new)
            return new
        m.name = _unique_name(self.workspace.matrices, "Pasted matrix")
        return self._store_imported(m)

    def open_wizard(self):
        self._wizard = CompensationWizard(self.controller, self)
        self._wizard.setModal(False)
        self._wizard.show()
        return self._wizard

    # ------------------------------------------------------------ applying
    def apply_to_samples(self, samples, matrix: SpilloverMatrix | None = None, enabled: bool = True) -> list:
        """Apply the matrix (default: the current one); samples lacking matrix
        parameters are skipped with a warning naming them."""
        if matrix is None:
            matrix = self.current_matrix()
        if matrix is None or not samples:
            return []
        applied, problems = apply_matrix(self.controller, list(samples), matrix, enabled)
        if problems:
            self._warn("Compensation", missing_text(problems, matrix.name))
        return applied

    def apply_to_selected(self):
        samples = self.controller.selected_samples()
        if not samples:
            rows = sorted({i.row() for i in self.status_table.selectedItems()})
            samples = [self.workspace.samples[r] for r in rows if r < len(self.workspace.samples)]
        if not samples:
            QMessageBox.information(self, "Compensation", "Select samples in the workspace (or in the table below) first.")
            return
        self.apply_to_samples(samples)

    def apply_to_group(self, group, attach: bool = True):
        """Apply the current matrix to the group's members; attached, samples
        that join the group later inherit it."""
        e = self.current_entry()
        if e is None:
            return
        members = self.workspace.members(group)
        with self.controller.undo_group("apply matrix to group", kind=("matrices", "settings")):
            matrix = e.matrix
            if attach:
                ensure_group_hook(self.controller)
                if not e.stored:
                    matrix = e.matrix.copy(e.label)
                    self.controller.store_matrix(matrix)
                set_group_matrix(self.workspace, group, matrix.name, [s.id for s in members])
            self.apply_to_samples(members, matrix)
            self.controller.mark_modified()
        if attach:
            self.select_matrix(matrix.name)
            self._show_current()
            self._update_actions()

    def detach_from_groups(self):
        e = self.current_entry()
        if e is None:
            return
        groups = groups_with_matrix(self.workspace, e.label)
        if not groups:
            return
        with self.controller.undo_group("detach matrix", kind="settings"):
            for g in groups:
                set_group_matrix(self.workspace, g, None)
            self.controller.mark_modified()
        self._show_current()
        self._update_actions()

    def apply_to_group_dialog(self):
        groups = self.workspace.groups
        names = [g.name for g in groups]
        cur = self.controller.current_group
        idx = names.index(cur.name) if cur is not None and cur.name in names else 0
        dlg = _GroupApplyDialog(self, groups, idx)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            g = self.workspace.group_by_name(dlg.combo.currentText())
            if g is not None:
                self.apply_to_group(g, dlg.attach.isChecked())

    def apply_to_all(self):
        self.apply_to_samples(self.workspace.samples)

    def remove_from_selected(self):
        samples = self.controller.selected_samples()
        if not samples:
            rows = sorted({i.row() for i in self.status_table.selectedItems()})
            samples = [self.workspace.samples[r] for r in rows if r < len(self.workspace.samples)]
        if samples:
            self.controller.set_compensation(samples, None)

    def closeEvent(self, event):
        event.ignore()
        self.hide()


# ---------------------------------------------------------------- wizard
_STATUS_COLORS = {"ok": None, "weak": "#ffe7c2", "no positives": "#f8d0d0", "too few events": "#f8d0d0", "missing": "#f8d0d0"}


class _ControlsPage(QWizardPage):
    def __init__(self, wizard: "CompensationWizard"):
        super().__init__()
        self.wiz = wizard
        self.setTitle("Controls")
        self.setSubTitle("Choose the group holding the compensation controls, the population of each control and the fluorochrome it is stained with.")
        self.group_combo = QComboBox()
        self.group_combo.currentIndexChanged.connect(self._group_changed)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Sample", "Population", "Events", "Control for"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.universal_negative = QCheckBox("Use unstained control as the universal negative")
        self.universal_negative.toggled.connect(lambda _on: self.wiz.invalidate())
        self.info = QLabel()
        self.info.setStyleSheet("color: #666;")
        self.info.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(QLabel("Controls from group:"))
        top.addWidget(self.group_combo, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.universal_negative)
        lay.addWidget(self.info)
        self.combos: list[QComboBox] = []
        self.pop_combos: list[QComboBox] = []

    def populate(self):
        ws = self.wiz.controller.workspace
        blocked = self.group_combo.blockSignals(True)
        self.group_combo.clear()
        for g in ws.groups:
            self.group_combo.addItem(icon("group"), g.name)
        if self.wiz.control_group is not None and self.wiz.control_group in ws.groups:
            self.group_combo.setCurrentIndex(ws.groups.index(self.wiz.control_group))
        self.group_combo.blockSignals(blocked)

        samples = self.wiz.control_samples
        params = self.wiz.params
        stains: dict[str, str] = {}
        for s in samples:
            for ch in s.channels:
                if ch.name in params and ch.stain and ch.name not in stains:
                    stains[ch.name] = ch.stain
        guess = guess_control_assignment([s.name for s in samples], params, stains)
        stored = self.wiz.stored
        if stored is not None:
            for s, r in stored.assignment.items():
                if s in samples:
                    guess[s.name] = r
        self.table.setRowCount(len(samples))
        self.combos = []
        self.pop_combos = []
        for r, s in enumerate(samples):
            self.table.setItem(r, 0, QTableWidgetItem(icon("sample"), s.name))
            pc = QComboBox()
            pc.addItem(UNGATED, "")
            for pop in s.populations():
                pc.addItem("  " * max(pop.depth - 1, 0) + pop.name, pop.path)
            if stored is not None and s in stored.populations:
                pop = stored.populations.get(s)
            else:
                pop = default_control_population(s)
            if pop is not None:
                k = pc.findData(pop.path)
                pc.setCurrentIndex(max(k, 0))
            pc.currentIndexChanged.connect(lambda _i, row=r: self._pop_changed(row))
            self.table.setCellWidget(r, 1, pc)
            self.pop_combos.append(pc)
            ev = QTableWidgetItem()
            ev.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 2, ev)
            combo = QComboBox()
            combo.addItem(NOT_USED)
            combo.addItem(UNSTAINED)
            for p in params:
                label = f"{p} :: {stains[p]}" if p in stains and stains[p] != p else p
                combo.addItem(label, p)
            role = guess.get(s.name)
            if role == UNSTAINED:
                combo.setCurrentIndex(1)
            elif role in params:
                combo.setCurrentIndex(2 + params.index(role))
            combo.currentIndexChanged.connect(lambda _i, row=r: self._role_changed(row))
            self.table.setCellWidget(r, 3, combo)
            self.combos.append(combo)
            self._update_count(r)
        if stored is not None:
            self.universal_negative.setChecked(stored.universal and any(v == UNSTAINED for v in self.roles()))
        else:
            self.universal_negative.setChecked(any(v == UNSTAINED for v in guess.values()))
        self._changed()

    def population(self, row: int):
        if not (0 <= row < len(self.pop_combos)):
            return None
        path = self.pop_combos[row].currentData() or ""
        return self.wiz.control_samples[row].find_population(path) if path else None

    def populations(self) -> dict:
        return {s: self.population(r) for r, s in enumerate(self.wiz.control_samples)}

    def set_population(self, sample, path: str):
        r = self.wiz.control_samples.index(sample)
        k = self.pop_combos[r].findData(path or "")
        if k >= 0:
            self.pop_combos[r].setCurrentIndex(k)

    def _update_count(self, row: int):
        s = self.wiz.control_samples[row]
        pop = self.population(row)
        item = self.table.item(row, 2)
        if item is None:
            return
        used = row < len(self.combos) and self.combos[row].currentIndex() > 0
        if pop is None or not used:  # gated counts only for controls in use (they read the events)
            item.setText(f"{s.n_events:,}")
            item.setToolTip("all events" if pop is None else "all events (the gated count is shown once the sample is assigned)")
        else:
            try:
                n = pop.count
            except Exception:  # noqa: BLE001 - unreadable file: show the total
                n = s.n_events
            item.setText(f"{n:,}")
            item.setToolTip(f"events in {pop.path} (of {s.n_events:,})")

    def _pop_changed(self, row: int):
        self._update_count(row)
        self.wiz.invalidate()

    def _role_changed(self, row: int):
        self._update_count(row)
        self._changed()

    def _group_changed(self, index: int):
        ws = self.wiz.controller.workspace
        if 0 <= index < len(ws.groups):
            self.wiz.set_control_group(ws.groups[index])

    def roles(self) -> list[str]:
        out = []
        for c in self.combos:
            i = c.currentIndex()
            out.append(NOT_USED if i <= 0 else UNSTAINED if i == 1 else c.itemData(i))
        return out

    def _changed(self):
        roles = self.roles()
        singles = [r for r in roles if r not in (NOT_USED, UNSTAINED)]
        dup = sorted({r for r in singles if singles.count(r) > 1})
        text = f"{len(set(singles))} single-stain control(s)"
        if UNSTAINED in roles:
            text += ", unstained control assigned"
        if dup:
            text += f". Several controls for {', '.join(dup)}: the first one is used."
        if not self.wiz.used_comp_group:
            prefix = "The Compensation group is empty: showing all samples. " if self.wiz.control_group is None or self.wiz.control_group.role in ("compensation", "all") else "The group is empty: showing all samples. "
            text = prefix + text
        self.info.setText(text)
        self.wiz.invalidate()
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return any(r not in (NOT_USED, UNSTAINED) for r in self.roles())


class _HistogramView(QLabel):
    """A control's primary-detector histogram; drag with the left button to
    set the positive range, with the right button (or Shift) the negative."""

    rangeDragged = Signal(str, float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.result = None
        self._start: tuple[float, str] | None = None

    def raw_at(self, x: float) -> float:
        r = self.result
        fx, _ = r.to_fraction(float(x), r.plot_rect.center().y())
        return float(r.x_scale.inverse_scalar(fx))

    def pixel_at(self, raw: float) -> float:
        r = self.result
        fx = r.x_scale.forward_scalar(float(raw))
        return float(r.to_pixel(fx, 0.5).x())

    def mousePressEvent(self, event):
        if self.result is None or self.result.x_scale is None:
            return super().mousePressEvent(event)
        neg = event.button() == Qt.MouseButton.RightButton or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self._start = (event.position().x(), "neg" if neg else "pos")
        event.accept()

    def mouseReleaseEvent(self, event):
        start, self._start = self._start, None
        if start is None or self.result is None:
            return super().mouseReleaseEvent(event)
        x0, which = start
        x1 = event.position().x()
        if abs(x1 - x0) < 3:
            return
        a, b = self.raw_at(min(x0, x1)), self.raw_at(max(x0, x1))
        self.rangeDragged.emit(which, a, b)
        event.accept()


def _spin(tip: str) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(-1e9, 1e9)
    s.setDecimals(0)
    s.setSingleStep(100.0)
    s.setSpecialValueText("auto")
    s.setValue(-1e9)
    s.setKeyboardTracking(False)
    s.setToolTip(tip)
    s.setMinimumWidth(90)
    return s


def _spin_value(s: QDoubleSpinBox) -> float | None:
    v = s.value()
    return None if v <= s.minimum() else float(v)


class _GateTile(QFrame):
    """One single-stain control: histogram of its primary detector with the
    positive (green) and negative (grey) ranges, range spin boxes and the
    control's diagnostics."""

    W, H = 340, 150

    def __init__(self, page: "_GatesPage", param: str, sample, population, label: str):
        super().__init__()
        self.page, self.param, self.sample, self.population = page, param, sample, population
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.title = QLabel(f"<b>{label}</b> — {sample.name}" + (f" / {population.name}" if population is not None else ""))
        self.title.setWordWrap(True)
        self.view = _HistogramView()
        self.view.rangeDragged.connect(self._dragged)
        self._base = None
        self.pos_lo, self.pos_hi = _spin("Lower end of the positive range"), _spin("Upper end of the positive range")
        self.neg_lo, self.neg_hi = _spin("Lower end of the negative range"), _spin("Upper end of the negative range")
        for s in (self.pos_lo, self.pos_hi, self.neg_lo, self.neg_hi):
            s.valueChanged.connect(lambda _v: self._spun())
        self.auto_button = QPushButton("Auto")
        self.auto_button.setToolTip("Back to the automatic positive / negative split")
        self.auto_button.clicked.connect(lambda: self.page.set_gate(self.param, None))
        self.info = QLabel()
        self.info.setWordWrap(True)
        grid = QGridLayout()
        grid.addWidget(QLabel("Positive:"), 0, 0)
        grid.addWidget(self.pos_lo, 0, 1)
        grid.addWidget(QLabel("–"), 0, 2)
        grid.addWidget(self.pos_hi, 0, 3)
        grid.addWidget(QLabel("Negative:"), 1, 0)
        grid.addWidget(self.neg_lo, 1, 1)
        grid.addWidget(QLabel("–"), 1, 2)
        grid.addWidget(self.neg_hi, 1, 3)
        grid.addWidget(self.auto_button, 0, 4, 2, 1)
        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.view)
        lay.addLayout(grid)
        lay.addWidget(self.info)
        self.render()

    def render(self):
        from ..core.model import GraphSettings
        from .renderer import RenderOptions, render_plot

        pop = self.population if self.population is not None else self.sample.root
        opts = RenderOptions(plot_type="histogram", show_gates=False, font_dpi=72.0, label_font_size=9, number_font_size=8, histogram_bins=128)
        try:
            res = render_plot(self.sample, pop, GraphSettings(x=self.param, y=None, plot_type="histogram"), self.W, self.H, opts)
        except Exception as exc:  # noqa: BLE001 - unreadable control
            self.view.setText(f"Cannot draw {self.sample.name}: {exc}")
            return
        self.view.result = res if res.x_scale is not None else None
        self._base = res.image
        self.view.setFixedSize(self.W, self.H)

    def _draw(self, pos, neg, threshold):
        if self._base is None or self.view.result is None:
            return
        pm = QPixmap.fromImage(self._base)
        r = self.view.result.plot_rect
        p = QPainter(pm)

        def band(rng, color, label):
            lo, hi = rng
            lo = max(lo, -1e12)
            hi = min(hi, 1e12)
            x0 = max(self.view.pixel_at(lo), r.left())
            x1 = min(self.view.pixel_at(hi), r.right())
            if x1 <= x0:
                return
            c = QColor(color)
            c.setAlpha(55)
            p.fillRect(QRectF(x0, r.top(), x1 - x0, r.height()), c)
            c.setAlpha(255)
            p.setPen(QPen(c, 1.5))
            p.drawLine(QPointF(x0, r.top()), QPointF(x0, r.bottom()))
            p.drawLine(QPointF(x1, r.top()), QPointF(x1, r.bottom()))
            p.drawText(QRectF(x0 + 2, r.top() + 1, max(x1 - x0 - 4, 30), 14), Qt.AlignmentFlag.AlignLeft, label)

        if neg is not None:
            band(neg, "#6b6b6b", "neg")
        if pos is not None:
            band(pos, "#1a9a3a", "pos")
        if threshold is not None and np.isfinite(threshold) and pos is None:
            x = self.view.pixel_at(threshold)
            p.setPen(QPen(QColor("#1a9a3a"), 1.5, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.end()
        self.view.setPixmap(pm)

    def update_from(self, diag: ControlDiagnostics | None, gates: ControlGates | None):
        for s, v in ((self.pos_lo, gates.pos[0] if gates and gates.pos else None), (self.pos_hi, gates.pos[1] if gates and gates.pos else None),
                     (self.neg_lo, gates.neg[0] if gates and gates.neg else None), (self.neg_hi, gates.neg[1] if gates and gates.neg else None)):
            blocked = s.blockSignals(True)
            s.setValue(s.minimum() if v is None or not np.isfinite(v) else v)
            s.blockSignals(blocked)
        if diag is None:
            self.info.setText("")
            self._draw(None, None, None)
            return
        pos = gates.pos if gates and gates.pos else diag.pos_range
        neg = gates.neg if gates and gates.neg else diag.neg_range
        self._draw(pos, neg, diag.threshold if (gates is None or gates.is_auto) else None)
        neg_text = "unstained control" if diag.negative == "unstained" else ("none" if diag.negative == "zero" else "own negatives")
        parts = []
        if diag.n_events:
            parts.append(f"Positive: {diag.n_pos:,} events, median {diag.pos_median:,.0f}" if np.isfinite(diag.pos_median) else f"Positive: {diag.n_pos:,} events")
            if np.isfinite(diag.neg_median):
                parts.append(f"Negative ({neg_text}): {diag.n_neg:,} events, median {diag.neg_median:,.0f}")
            if np.isfinite(diag.stain_index):
                parts.append(f"stain index {diag.stain_index:.1f}")
            if diag.method in ("otsu", "brightest 10%") and np.isfinite(diag.threshold):
                parts.append(f"auto split at {diag.threshold:,.0f}")
        color = {"ok": "#1a7a2e", "weak": "#b26a00"}.get(diag.status, "#c00000")
        status = "OK" if diag.status == "ok" else diag.message or diag.status
        self.info.setText(" • ".join(parts) + f"<br><span style='color:{color}'><b>{status}</b></span>")
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.setStyleSheet("" if diag.status == "ok" else f"_GateTile {{ border: 2px solid {color}; }}")

    def _spun(self):
        plo, phi, nlo, nhi = (_spin_value(s) for s in (self.pos_lo, self.pos_hi, self.neg_lo, self.neg_hi))
        pos = None if plo is None and phi is None else (plo if plo is not None else -math.inf, phi if phi is not None else math.inf)
        neg = None if nlo is None and nhi is None else (nlo if nlo is not None else -math.inf, nhi if nhi is not None else math.inf)
        self.page.set_gate(self.param, ControlGates(pos, neg))

    def _dragged(self, which: str, lo: float, hi: float):
        cur = self.page.wiz.gates.get(self.param) or ControlGates()
        if which == "neg":
            g = ControlGates(cur.pos, (lo, hi))
        else:
            g = ControlGates((lo, hi), cur.neg)
        self.page.set_gate(self.param, g)


class _GatesPage(QWizardPage):
    def __init__(self, wizard: "CompensationWizard"):
        super().__init__()
        self.wiz = wizard
        self.setTitle("Gates")
        self.setSubTitle("Positive and negative populations of every control, found automatically. Drag on a histogram "
                         "(left button: positive, right button or Shift: negative) or type a range to change them.")
        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.grid.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.inner)
        self.message = QLabel()
        self.message.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll, 1)
        lay.addWidget(self.message)
        self.tiles: dict[str, _GateTile] = {}

    def initializePage(self):
        self.wiz.compute()
        self.build()

    def build(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.tiles = {}
        singles, _unst = _split_assignment(self.wiz.assignment())
        pops = self.wiz.populations()
        stains = {}
        for s in self.wiz.control_samples:
            for ch in s.channels:
                if ch.stain and ch.name not in stains:
                    stains[ch.name] = ch.stain
        params = [p for p in self.wiz.params if p in singles]
        for k, p in enumerate(params):
            s = singles[p]
            label = f"{p} :: {stains[p]}" if p in stains and stains[p] != p else p
            tile = _GateTile(self, p, s, pops.get(s), label)
            self.grid.addWidget(tile, k // 2, k % 2)
            self.tiles[p] = tile
        self.refresh()

    def tile(self, param: str) -> _GateTile | None:
        return self.tiles.get(param)

    def set_gate(self, param: str, gates: ControlGates | None):
        self.wiz.set_gates(param, gates)
        self.wiz.compute()
        self.refresh()

    def refresh(self):
        diags = self.wiz.diagnostics or {}
        for p, tile in self.tiles.items():
            tile.update_from(diags.get(p), self.wiz.gates.get(p))
        self.message.setText(self.wiz.message)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self.wiz.matrix is not None


class _MatrixPage(QWizardPage):
    def __init__(self, wizard: "CompensationWizard"):
        super().__init__()
        self.wiz = wizard
        self.setTitle("Matrix")
        self.setSubTitle("The spillover matrix computed from the controls (values in %).")
        self.method_combo = QComboBox()
        for key, label in METHODS:
            self.method_combo.addItem(label, key)
        self.method_combo.currentIndexChanged.connect(self._method_changed)
        self.grid = QTableWidget()
        self.grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.name_edit = QLineEdit()
        self.apply_check = QCheckBox("Apply to the group")
        self.apply_check.setChecked(True)
        self.group_combo = QComboBox()
        self.attach_check = QCheckBox("Samples added to the group later inherit the matrix")
        self.attach_check.setChecked(True)
        self.apply_check.toggled.connect(self.group_combo.setEnabled)
        self.apply_check.toggled.connect(self.attach_check.setEnabled)
        self.name_edit.textChanged.connect(lambda _t: self.completeChanged.emit())
        form = QFormLayout()
        form.addRow("Method:", self.method_combo)
        form2 = QFormLayout()
        form2.addRow("Name:", self.name_edit)
        row = QHBoxLayout()
        row.addWidget(self.apply_check)
        row.addWidget(self.group_combo, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.grid, 1)
        lay.addWidget(self.message)
        lay.addLayout(form2)
        lay.addLayout(row)
        lay.addWidget(self.attach_check)
        self._fill_groups()

    def _fill_groups(self):
        ws = self.wiz.controller.workspace
        keep = self.group_combo.currentText()
        self.group_combo.clear()
        for g in ws.groups:
            self.group_combo.addItem(icon("group"), g.name)
        idx = self.group_combo.findText(keep) if keep else -1
        if idx < 0:
            idx = next((i for i, g in enumerate(ws.groups) if g.role == "all"), 0)
        self.group_combo.setCurrentIndex(idx)

    def initializePage(self):
        ws = self.wiz.controller.workspace
        if not self.name_edit.text():
            self.name_edit.setText(_unique_name(ws.matrices, f"Computed {_dt.date.today().isoformat()}"))
        blocked = self.method_combo.blockSignals(True)
        self.method_combo.setCurrentIndex(max(self.method_combo.findData(self.wiz.method), 0))
        self.method_combo.blockSignals(blocked)
        self._fill_groups()
        self._recompute()

    def _recompute(self):
        self.wiz.compute()
        fill_matrix_grid(self.grid, self.wiz.matrix, editable=False)
        self.message.setText(self.wiz.message)
        self.completeChanged.emit()

    def _method_changed(self, _i: int):
        self.wiz.method = self.method_combo.currentData() or "median"
        self.wiz.invalidate()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._recompute()
        finally:
            QApplication.restoreOverrideCursor()

    def isComplete(self) -> bool:
        return self.wiz.matrix is not None and bool(self.name_edit.text().strip())


class _DiagnosticsPage(QWizardPage):
    COLUMNS = ["Control", "Sample", "Population", "Events", "Positive", "Negative", "Pos. median", "Neg. median", "Stain index", "Split", "Status"]

    def __init__(self, wizard: "CompensationWizard"):
        super().__init__()
        self.wiz = wizard
        self.setTitle("Diagnostics")
        self.setSubTitle("Quality of every control, the spillover spreading matrix of the compensated controls and the unmixing residuals.")
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.ssm_grid = QTableWidget()
        self.ssm_grid.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ssm_grid.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.report = QLabel()
        self.report.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.table, 2)
        lay.addWidget(QLabel("<b>Spillover spreading matrix</b> (row = fluorochrome, column = detector, √intensity)"))
        lay.addWidget(self.ssm_grid, 1)
        lay.addWidget(self.report)

    def initializePage(self):
        if self.wiz.matrix is None:
            self.wiz.compute()
        self.wiz.compute_ssm()
        self.fill()

    def fill(self):
        singles, _unst = _split_assignment(self.wiz.assignment())
        pops = self.wiz.populations()
        diags = self.wiz.diagnostics or {}
        rows = [p for p in self.wiz.params if p in diags or p in singles]
        self.table.setRowCount(len(rows))

        def num(v, fmt="{:,.0f}"):
            return fmt.format(v) if v is not None and np.isfinite(v) else "—"

        for r, p in enumerate(rows):
            d = diags.get(p) or ControlDiagnostics(p, status="missing", message="no control assigned (not compensated)")
            s = singles.get(p)
            pop = pops.get(s) if s is not None else None
            cells = [p, s.name if s is not None else "—", pop.path if pop is not None else (UNGATED if s is not None else "—"),
                     f"{d.n_events:,}", f"{d.n_pos:,}", f"{d.n_neg:,}" + (" (unstained)" if d.negative == "unstained" else ""),
                     num(d.pos_median), num(d.neg_median), num(d.stain_index, "{:.1f}"), d.method or "—",
                     "OK" if d.status == "ok" else (d.message or d.status)]
            bg = _STATUS_COLORS.get(d.status)
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if 3 <= c <= 8:
                    it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if bg:
                    it.setBackground(QBrush(QColor(bg)))
                self.table.setItem(r, c, it)
        fill_values_grid(self.ssm_grid, self.wiz.ssm_params, self.wiz.ssm)
        lines = [self.wiz.message]
        m = self.wiz.matrix
        if m is not None and getattr(m, "report", None) is not None:
            worst = ", ".join(f"{a} → {b}: {v * 100:+.3f} %" for a, b, v in m.report.worst())
            if worst:
                lines.append(f"Largest unmixing residuals: {worst}.")
        if self.wiz.residuals is not None:
            res = self.wiz.residuals
            off = res[~np.eye(len(res), dtype=bool)] if res.size else res
            off = off[np.isfinite(off)]
            if off.size:
                lines.append(f"Residual spill of the compensated controls (median based): max {np.abs(off).max() * 100:.2f} %.")
        self.report.setText("\n".join(l for l in lines if l))


class CompensationWizard(QWizard):
    """Compute a spillover matrix from single-stain controls:
    controls -> gates -> matrix -> diagnostics."""

    def __init__(self, controller, parent=None, group=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Compensation Wizard")
        self.setWindowIcon(icon("wizard"))
        self.setWizardStyle(QWizard.WizardStyle.ClassicStyle)
        self.resize(820, 600)
        ensure_group_hook(controller)
        ws = controller.workspace
        self.stored = ControlSet.from_settings(ws, ws.settings.get(CONTROLS_KEY))
        self.gates: dict[str, ControlGates] = dict(self.stored.gates) if self.stored is not None else {}
        self.method = self.stored.method if self.stored is not None else "median"
        self.matrix: SpilloverMatrix | None = None
        self.diagnostics: dict = {}
        self.residuals: np.ndarray | None = None
        self.ssm: np.ndarray | None = None
        self.ssm_params: list[str] = []
        self.message = ""
        self._cache: dict = {}
        if group is None and self.stored is not None and self.stored.group:
            group = ws.group_by_name(self.stored.group)
        self.control_group = None
        self.control_samples: list = []
        self.params: list[str] = []
        self.used_comp_group = False
        self.controls_page = _ControlsPage(self)
        self.gates_page = _GatesPage(self)
        self.matrix_page = _MatrixPage(self)
        self.diagnostics_page = _DiagnosticsPage(self)
        for page in (self.controls_page, self.gates_page, self.matrix_page, self.diagnostics_page):
            self.addPage(page)
        self.set_control_group(group)

    # ------------------------------------------------------------ controls
    def set_control_group(self, group=None):
        """Use the members of `group` as the controls (default: the
        Compensation group, or every sample when it is empty)."""
        ws = self.controller.workspace
        comp_group = next((g for g in ws.groups if g.role == "compensation"), None)
        if group is None:
            members = ws.members(comp_group) if comp_group is not None else []
            group = comp_group if members else next((g for g in ws.groups if g.role == "all"), comp_group)
            self.used_comp_group = bool(members)
        else:
            members = ws.members(group)
            self.used_comp_group = bool(members)
        self.control_group = group
        self.control_samples = members or list(ws.samples)
        params: list[str] = []
        for s in self.control_samples:
            for p in fluorescence_params(s):
                if p not in params:
                    params.append(p)
        self.params = params
        self.invalidate()
        self.controls_page.populate()

    def assignment(self) -> dict:
        """Sample -> "Unstained" | parameter for the assigned rows."""
        return {s: r for s, r in zip(self.control_samples, self.controls_page.roles()) if r != NOT_USED}

    def populations(self) -> dict:
        """Sample -> Population (None: ungated) the controls are read from."""
        return self.controls_page.populations()

    def control_set(self) -> ControlSet:
        assignment = self.assignment()
        pops = self.populations()
        return ControlSet(assignment, {s: pops.get(s) for s in assignment}, {p: g for p, g in self.gates.items() if not g.is_auto},
                          self.controls_page.universal_negative.isChecked(), self.method, self.control_group.name if self.control_group else None)

    def set_gates(self, param: str, gates: ControlGates | None):
        if gates is None or gates.is_auto:
            self.gates.pop(param, None)
        else:
            self.gates[param] = gates
        self.invalidate()

    def active_gates(self) -> dict:
        return {p: g for p, g in self.gates.items() if not g.is_auto and p in self.params}

    def invalidate(self):
        """The controls, populations, gates or method changed: the spreading
        matrix and residuals of the last matrix no longer apply."""
        self.ssm, self.ssm_params, self.residuals = None, [], None

    # ------------------------------------------------------------ computing
    def compute(self) -> SpilloverMatrix | None:
        name = self.matrix_page.name_edit.text().strip() or "Computed"
        try:
            m = compute_spillover(self.assignment(), self.controls_page.universal_negative.isChecked(), name,
                                  populations=self.populations(), gates=self.active_gates(), method=self.method, cache=self._cache)
        except (ValueError, KeyError) as exc:
            self.matrix = None
            self.diagnostics = {}
            self.message = f"Could not compute the matrix: {exc}"
            return None
        self.matrix = m
        self.diagnostics = dict(m.diagnostics)
        n = len(m.params)
        missing = [p for p in self.params if p not in m.params]
        warnings = diagnostics_warnings(m.diagnostics)
        if warnings:
            self.message = f"{n}×{n} matrix — {len(warnings)} control(s) need attention: " + "; ".join(warnings) + "."
        else:
            self.message = f"{n}×{n} matrix computed."
        if missing:
            self.message += f" No control for: {', '.join(missing)} (not compensated)."
        if m.report is not None:
            self.message += f" {m.report.summary()}."
        return m

    def compute_ssm(self) -> np.ndarray | None:
        """Spreading matrix and median residuals of the controls compensated
        with the computed matrix."""
        self.ssm, self.ssm_params, self.residuals = None, [], None
        if self.matrix is None:
            return None
        try:
            params, controls, unst, _singles = control_data(self.assignment(), self.controls_page.universal_negative.isChecked(),
                                                            self.populations(), list(self.matrix.params), self._cache)
        except (ValueError, KeyError):
            return None
        gates = self.active_gates()
        self.ssm = spillover_spreading(controls, params, self.matrix, unst, gates=gates)
        self.ssm_params = list(params)
        self.residuals = residual_spill(controls, params, self.matrix, unst, gates=gates)
        return self.ssm

    def accept(self):
        if self.matrix is None:
            self.compute()
        if self.matrix is None:
            return
        ws = self.controller.workspace
        name = self.matrix_page.name_edit.text().strip() or self.matrix.name
        if name in ws.matrices:
            res = QMessageBox.question(self, "Compensation Wizard", f"Replace the existing matrix “{name}”?")
            if res != QMessageBox.StandardButton.Yes:
                return
        m = self.matrix.copy(name)
        settings = self.control_set().to_settings()
        settings["matrix"] = name
        # one undo step: storing the matrix, remembering the controls and gates,
        # attaching the matrix to the group and applying it to the members
        with self.controller.undo_group("compensation wizard", kind=("matrices", "settings")):
            self.controller.store_matrix(m)
            ws.settings[CONTROLS_KEY] = settings
            if self.matrix_page.apply_check.isChecked():
                g = ws.group_by_name(self.matrix_page.group_combo.currentText())
                if g is not None:
                    members = ws.members(g)
                    if self.matrix_page.attach_check.isChecked():
                        set_group_matrix(ws, g, name, [s.id for s in members])
                    if members:
                        apply_matrix(self.controller, members, m, True)
            self.controller.mark_modified()
        super().accept()
