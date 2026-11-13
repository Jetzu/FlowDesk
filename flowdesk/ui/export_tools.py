"""Export dialog (FlowJo's Export / Concatenate Populations) and batch export.

`install(window)` returns an `ExportTools` object. The main window hands its
"Export Population as FCS…", "Export Events (CSV)…" and "Concatenate
Samples…" commands to it (`export_fcs`, `export_events`,
`concatenate_samples`); all three open one `ExportDialog` offering:

* format FCS 3.1 or CSV (header row, parameter labels);
* parameters: All / Compensated only / Uncompensated / Selected (checkable
  list), with or without derived and computed parameters;
* linear ("channel") values or transformed display-space ("scale") values;
* all events, at most N random events or a random percentage (seeded);
* file names from a pattern ({sample}, {population}, {group}, {index},
  keywords such as {$WELLID}) and keywords to add to the files;
* "for every sample of group X": the selected population path(s) are
  exported for every member of the group (batch export);
* for concatenation: parameter matching by name (common / all / identical),
  keywords encoded as parameters, a SampleID parameter;
* adding the written FCS files to the workspace.

It also adds "Export…" and "Batch Export…" to the Tools ▸ Export band,
context-menu entries to the sample tree and the group pane, "Split Data Sets
into Files…" for multi-data-set files ($NEXTDATA / .lmd), and the
"Memory-Map Large FCS Files" option (preference ``fcs_memmap``).

Preference keys read with ``prefs.get(key, fallback)``: ``fcs_memmap``
(bool, default True) and ``fcs_memmap_min_mb`` (int, default 64).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core import export as core_export
from ..core import fcs as core_fcs
from ..core.model import Group, Population, Sample
from .icons import icon

PREF_MEMMAP = "fcs_memmap"
PREF_MEMMAP_MIN_MB = "fcs_memmap_min_mb"
MODES = ("fcs", "csv", "concatenate")
_PARAM_MODES = [
    ("all", "All parameters"),
    ("compensated", "Compensated parameters only"),
    ("uncompensated", "Uncompensated parameters only"),
    ("selected", "Selected parameters:"),
]
_MATCH = [
    ("intersection", "Parameters common to all samples"),
    ("union", "All parameters (empty where a sample has none)"),
    ("strict", "Identical parameter sets only"),
]
_STRUCTURAL_PREFIXES = ("$BEGIN", "$END", "$TOT", "$PAR", "$NEXTDATA", "$BYTEORD", "$DATATYPE", "$MODE")


def _pref(prefs, key: str, default):
    """A preference, also when the controller's Preferences does not list
    the key among its defaults (then the stored QSettings value is used)."""
    if prefs is None:
        return default
    value = prefs.get(key, None)
    if value is None:
        settings = getattr(prefs, "_settings", None)
        if settings is not None:
            try:
                value = settings.value(f"prefs/{key}")
            except Exception:  # noqa: BLE001 - unreadable settings: use the default
                value = None
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes") if isinstance(value, str) else bool(value)
    if isinstance(default, int):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default
    return value


def _is_structural(key: str) -> bool:
    import re

    upper = key.upper()
    return upper.startswith(_STRUCTURAL_PREFIXES) or bool(re.match(r"^\$?P\d+", upper)) or upper.startswith("FLOWDESK_")


def keyword_candidates(samples: list[Sample]) -> list[str]:
    """Keywords offered for encoding as parameters: those whose value differs
    between the samples first (conditions such as STIM, DOSE), then the rest."""
    names: dict[str, None] = {}
    for s in samples:
        for k in s.keywords:
            if not _is_structural(k):
                names.setdefault(k, None)

    def varies(k: str) -> bool:
        return len({s.keywords.get(k) for s in samples}) > 1

    return sorted(names, key=lambda k: (not varies(k), k.upper()))


class ExportDialog(QDialog):
    """Options of an export or concatenation; `run()` writes the files."""

    def __init__(
        self,
        controller,
        rows: list[tuple[Sample, Population]],
        mode: str = "fcs",
        group: Group | None = None,
        batch: bool = False,
        paths: list[str] | None = None,
        parent=None,
        state: dict | None = None,
    ):
        super().__init__(parent)
        if mode not in MODES:
            raise ValueError(f"unknown export mode {mode!r}")
        self.controller = controller
        self.rows = list(rows)
        self.mode = mode
        self.state = state if state is not None else {}
        self.paths_from_rows = list(dict.fromkeys(paths if paths is not None else [p.path for _, p in self.rows]))
        self.written: list[Path] = []
        self.errors: list[str] = []
        self.missing: list[tuple[Sample, str]] = []  # group members without the population
        self.events_written = 0
        self._ready = False
        concat = mode == "concatenate"
        self.setWindowTitle("Concatenate Populations" if concat else "Export Populations")
        layout = QVBoxLayout(self)

        # ---- what is exported
        top = QFormLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItem("FCS 3.1", "fcs")
        self.format_combo.addItem("CSV (comma-separated values)", "csv")
        self.format_combo.setCurrentIndex(1 if mode == "csv" else 0)
        self.format_combo.currentIndexChanged.connect(self._sync)
        top.addRow("Format:", self.format_combo)
        self.summary = QLabel()
        top.addRow("Populations:", self.summary)
        batch_row = QHBoxLayout()
        self.batch_check = QCheckBox("For every sample of group")
        self.batch_check.setToolTip("Export the selected population(s) for every sample of the group (FlowJo batch export)")
        self.group_combo = QComboBox()
        ws = controller.workspace
        current = group or getattr(controller, "current_group", None)
        for g in ws.groups:
            self.group_combo.addItem(g.name, g)
            if g is current:
                self.group_combo.setCurrentIndex(self.group_combo.count() - 1)
        self.batch_check.setChecked(bool(batch))
        self.batch_check.toggled.connect(self._sync)
        self.group_combo.currentIndexChanged.connect(self._sync)
        batch_row.addWidget(self.batch_check)
        batch_row.addWidget(self.group_combo, 1)
        top.addRow("", batch_row)
        layout.addLayout(top)

        columns = QHBoxLayout()
        layout.addLayout(columns, 1)
        left = QVBoxLayout()
        right = QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        # ---- parameters
        box = QGroupBox("Parameters")
        pl = QVBoxLayout(box)
        self.param_group = QButtonGroup(self)
        self.param_radios: dict[str, QRadioButton] = {}
        for key, text in _PARAM_MODES:
            rb = QRadioButton(text)
            self.param_group.addButton(rb)
            self.param_radios[key] = rb
            pl.addWidget(rb)
            rb.toggled.connect(self._sync)
        self.param_radios[self.state.get("param_mode", "all") if self.state.get("param_mode") in self.param_radios else "all"].setChecked(True)
        self.param_list = QListWidget()
        self.param_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.param_list.setMinimumHeight(140)
        pl.addWidget(self.param_list, 1)
        row = QHBoxLayout()
        for text, state in (("All", True), ("None", False)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(lambda _=False, st=state: self._check_all(st))
            row.addWidget(b)
        row.addStretch(1)
        pl.addLayout(row)
        self.derived_check = QCheckBox("Include derived parameters (ratios, …)")
        self.derived_check.setChecked(bool(self.state.get("derived", True)))
        self.computed_check = QCheckBox("Include computed parameters (t-SNE / UMAP / PCA, clusters, QC, MESF)")
        self.computed_check.setChecked(bool(self.state.get("computed", True)))
        pl.addWidget(self.derived_check)
        pl.addWidget(self.computed_check)
        left.addWidget(box, 1)

        # ---- concatenation
        self.concat_box = QGroupBox("Concatenation")
        cl = QFormLayout(self.concat_box)
        self.match_combo = QComboBox()
        for key, text in _MATCH:
            self.match_combo.addItem(text, key)
        cl.addRow("Match by name:", self.match_combo)
        self.kw_list = QListWidget()
        self.kw_list.setMaximumHeight(110)
        self.kw_list.setToolTip("Checked keywords become parameters (text values are coded 1, 2, …; the table is kept in FLOWDESK_KW_<keyword>)")
        cl.addRow("Keywords as parameters:", self.kw_list)
        self.sample_id_check = QCheckBox("Add a SampleID parameter")
        self.sample_id_check.setChecked(True)
        cl.addRow("", self.sample_id_check)
        left.addWidget(self.concat_box)
        self.concat_box.setVisible(concat)

        # ---- values
        vbox = QGroupBox("Values")
        vl = QVBoxLayout(vbox)
        self.linear_radio = QRadioButton("Linear values (as acquired / compensated)")
        self.scaled_radio = QRadioButton("Transformed values (display scale)")
        self.scaled_radio.setToolTip("The values on the axis transforms (biexponential, log, …) as 0 … range; FlowJo calls these channel values")
        (self.scaled_radio if self.state.get("scaled") else self.linear_radio).setChecked(True)
        vl.addWidget(self.linear_radio)
        srow = QHBoxLayout()
        srow.addWidget(self.scaled_radio)
        self.range_spin = QSpinBox()
        self.range_spin.setRange(1, 1 << 22)
        self.range_spin.setValue(int(self.state.get("scale_range", core_export.SCALE_RANGE)))
        self.range_spin.setPrefix("0 – ")
        srow.addWidget(self.range_spin)
        srow.addStretch(1)
        vl.addLayout(srow)
        self.linear_radio.toggled.connect(self._sync)
        right.addWidget(vbox)

        # ---- events
        ebox = QGroupBox("Events per sample" if concat else "Events")
        el = QFormLayout(ebox)
        self.events_group = QButtonGroup(self)
        self.all_radio = QRadioButton("All events")
        self.count_radio = QRadioButton("At most")
        self.percent_radio = QRadioButton("Random")
        for rb in (self.all_radio, self.count_radio, self.percent_radio):
            self.events_group.addButton(rb)
            rb.toggled.connect(self._sync)
        self.all_radio.setChecked(True)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 1_000_000_000)
        self.count_spin.setSingleStep(1000)
        self.count_spin.setValue(int(self.state.get("max_events", 10000)))
        self.count_spin.setSuffix(" random events")
        self.percent_spin = QDoubleSpinBox()
        self.percent_spin.setRange(0.01, 100.0)
        self.percent_spin.setDecimals(2)
        self.percent_spin.setValue(float(self.state.get("percent", 10.0)))
        self.percent_spin.setSuffix(" %")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 1_000_000)
        self.seed_spin.setValue(int(self.state.get("seed", 0)))
        el.addRow(self.all_radio)
        el.addRow(self.count_radio, self.count_spin)
        el.addRow(self.percent_radio, self.percent_spin)
        el.addRow("Random seed:", self.seed_spin)
        right.addWidget(ebox)

        # ---- keywords to add
        self.kw_box = QGroupBox("Keywords to add")
        kl = QVBoxLayout(self.kw_box)
        self.add_table = QTableWidget(0, 2)
        self.add_table.setHorizontalHeaderLabels(["Keyword", "Value"])
        self.add_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.add_table.verticalHeader().setVisible(False)
        self.add_table.setMaximumHeight(110)
        kl.addWidget(self.add_table)
        krow = QHBoxLayout()
        add = QPushButton("Add")
        add.setAutoDefault(False)
        add.clicked.connect(lambda: self.add_keyword_row("", ""))
        rem = QPushButton("Remove")
        rem.setAutoDefault(False)
        rem.clicked.connect(self._remove_keyword_rows)
        krow.addWidget(add)
        krow.addWidget(rem)
        krow.addStretch(1)
        kl.addLayout(krow)
        right.addWidget(self.kw_box)

        # ---- output
        obox = QGroupBox("Output")
        ol = QFormLayout(obox)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        ol.addRow("File:" if concat else "Folder:", path_row)
        self.pattern_edit = QLineEdit(self.state.get("pattern", core_export.DEFAULT_PATTERN))
        self.pattern_edit.setToolTip("{sample} file name, {name} sample name, {population}, {path}, {group}, {index}, or any keyword such as {$WELLID} or {TUBE NAME}")
        self.pattern_label = QLabel("File names:")
        ol.addRow(self.pattern_label, self.pattern_edit)
        self.header_check = QCheckBox("Header row")
        self.header_check.setChecked(bool(self.state.get("header", True)))
        self.labels_check = QCheckBox("Parameter labels (name :: stain) in the header")
        self.labels_check.setChecked(bool(self.state.get("header_labels", False)))
        ol.addRow("CSV:", self.header_check)
        ol.addRow("", self.labels_check)
        self.add_check = QCheckBox("Add the written FCS files to the workspace")
        self.add_check.setChecked(bool(self.state.get("add_to_workspace" + ("_concat" if concat else ""), concat)))
        ol.addRow("", self.add_check)
        right.addWidget(obox)

        if concat:
            default = Path(self.state.get("folder") or Path.home()) / "concatenated.fcs"
            self.path_edit.setText(str(default))
            self.pattern_edit.setVisible(False)
            self.pattern_label.setVisible(False)
        else:
            folder = self.state.get("folder")
            if not folder and self.rows:
                folder = str(self.rows[0][0].path.parent)
            self.path_edit.setText(str(folder or Path.home()))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Concatenate" if concat else "Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.buttons = buttons
        self._chosen_before = set(self.state.get("chosen", []))
        self._ready = True
        self._refill()
        self._sync()

    # ------------------------------------------------------------- contents
    def selected_group(self) -> Group | None:
        return self.group_combo.currentData()

    def target_rows(self) -> list[tuple[Sample, Population]]:
        """The (sample, population) rows exported: the selection, or the
        selected population path(s) for every member of the chosen group."""
        if self.batch_check.isChecked() and self.selected_group() is not None:
            rows, self.missing = core_export.group_rows(self.controller.workspace, self.selected_group(), self.paths_from_rows)
            return rows
        self.missing = []
        return list(self.rows)

    def _samples(self) -> list[Sample]:
        out: list[Sample] = []
        for s, _ in self.target_rows():
            if s not in out:
                out.append(s)
        return out

    def available_parameters(self) -> list[str]:
        """Parameters of the exported samples (from their headers; no event
        data is read)."""
        names: dict[str, None] = {}
        for s in self._samples():
            for n in s.parameter_names():
                names.setdefault(n, None)
        return list(names)

    def _refill(self):
        """Fill the parameter and keyword lists for the current rows."""
        checked = set(self.chosen_parameters()) if self.param_list.count() else self._chosen_before
        samples = self._samples()
        label_of = {}
        for s in samples:
            for n in s.parameter_names():
                label_of.setdefault(n, s.param_label(n))
        self.param_list.clear()
        for name in self.available_parameters():
            item = QListWidgetItem(label_of.get(name, name))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if (not checked or name in checked) else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.param_list.addItem(item)
        kws = set(self.chosen_keywords())
        self.kw_list.clear()
        for k in keyword_candidates(samples):
            item = QListWidgetItem(k)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if k in kws else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, k)
            self.kw_list.addItem(item)
        rows = self.target_rows()
        n_samples = len({id(s) for s, _ in rows})
        pops = sorted({p.path or "Ungated" for _, p in rows})
        text = f"{len(rows)} population(s) from {n_samples} sample(s)"
        if len(pops) <= 3:
            text += ": " + ", ".join(p.rsplit("/", 1)[-1] or "Ungated" for p in pops)
        if self.missing:
            text += f" ({len(self.missing)} sample(s) lack the population)"
        self.summary.setText(text)
        self._last_key = (self.batch_check.isChecked(), id(self.selected_group()))

    def _check_all(self, state: bool):
        for i in range(self.param_list.count()):
            self.param_list.item(i).setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def set_checked_parameters(self, names: list[str]):
        wanted = set(names)
        for i in range(self.param_list.count()):
            it = self.param_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in wanted else Qt.CheckState.Unchecked)

    def set_checked_keywords(self, names: list[str]):
        wanted = set(names)
        for i in range(self.kw_list.count()):
            it = self.kw_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in wanted else Qt.CheckState.Unchecked)

    def chosen_parameters(self) -> list[str]:
        return [
            self.param_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.param_list.count())
            if self.param_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def chosen_keywords(self) -> list[str]:
        return [
            self.kw_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.kw_list.count())
            if self.kw_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def add_keyword_row(self, key: str, value: str):
        r = self.add_table.rowCount()
        self.add_table.insertRow(r)
        self.add_table.setItem(r, 0, QTableWidgetItem(key))
        self.add_table.setItem(r, 1, QTableWidgetItem(value))

    def _remove_keyword_rows(self):
        for r in sorted({i.row() for i in self.add_table.selectedIndexes()}, reverse=True):
            self.add_table.removeRow(r)

    def keywords_to_add(self) -> dict[str, str]:
        out = {}
        for r in range(self.add_table.rowCount()):
            k = self.add_table.item(r, 0)
            v = self.add_table.item(r, 1)
            key = k.text().strip() if k else ""
            if key:
                out[key] = v.text() if v else ""
        return out

    def param_mode(self) -> str:
        return next((k for k, rb in self.param_radios.items() if rb.isChecked()), "all")

    def fmt(self) -> str:
        return self.format_combo.currentData()

    def _sync(self, *_):
        if not self._ready:
            return
        if getattr(self, "_last_key", None) is not None and self._last_key != (self.batch_check.isChecked(), id(self.selected_group())):
            self._refill()
        selected = self.param_mode() == "selected"
        self.param_list.setEnabled(selected)
        self.derived_check.setEnabled(not selected)
        self.computed_check.setEnabled(not selected)
        self.group_combo.setEnabled(self.batch_check.isChecked())
        self.range_spin.setEnabled(self.scaled_radio.isChecked())
        self.count_spin.setEnabled(self.count_radio.isChecked())
        self.percent_spin.setEnabled(self.percent_radio.isChecked())
        self.seed_spin.setEnabled(not self.all_radio.isChecked())
        csv = self.fmt() == "csv"
        self.header_check.setEnabled(csv)
        self.labels_check.setEnabled(csv)
        self.kw_box.setEnabled(not csv)
        self.add_check.setEnabled(not csv)
        if self.mode == "concatenate":
            text = self.path_edit.text().strip()
            if text:
                stem = text.rsplit(".", 1)[0] if text.lower().endswith((".fcs", ".csv")) else text
                want = stem + (".csv" if csv else ".fcs")
                if want != text:
                    self.path_edit.setText(want)

    def _browse(self):
        if self.mode == "concatenate":
            filt = "CSV (*.csv)" if self.fmt() == "csv" else "FCS (*.fcs)"
            path, _ = QFileDialog.getSaveFileName(self, "Concatenated file", self.path_edit.text(), filt)
        else:
            path = QFileDialog.getExistingDirectory(self, "Folder for the exported files", self.path_edit.text())
        if path:
            self.path_edit.setText(path)
            self._sync()

    # ------------------------------------------------------------- options
    def options(self) -> dict:
        """Keyword arguments for core.export.export_population."""
        mode = self.param_mode()
        opts = {
            "mode": "all" if mode == "selected" else mode,
            "include_derived": self.derived_check.isChecked(),
            "include_computed": self.computed_check.isChecked(),
            "scaled": self.scaled_radio.isChecked(),
            "scale_range": float(self.range_spin.value()),
            "max_events": self.count_spin.value() if self.count_radio.isChecked() else None,
            "percent": self.percent_spin.value() if self.percent_radio.isChecked() else None,
            "seed": self.seed_spin.value(),
        }
        if mode == "selected":
            opts["parameters"] = self.chosen_parameters()
        if self.fmt() == "fcs":
            opts["extra_keywords"] = self.keywords_to_add()
        else:
            opts["header"] = self.header_check.isChecked()
            opts["header_labels"] = self.labels_check.isChecked()
        return opts

    def _remember(self):
        st = self.state
        st["param_mode"] = self.param_mode()
        st["chosen"] = self.chosen_parameters()
        st["derived"] = self.derived_check.isChecked()
        st["computed"] = self.computed_check.isChecked()
        st["scaled"] = self.scaled_radio.isChecked()
        st["scale_range"] = self.range_spin.value()
        st["max_events"] = self.count_spin.value()
        st["percent"] = self.percent_spin.value()
        st["seed"] = self.seed_spin.value()
        st["header"] = self.header_check.isChecked()
        st["header_labels"] = self.labels_check.isChecked()
        st["add_to_workspace" + ("_concat" if self.mode == "concatenate" else "")] = self.add_check.isChecked()
        text = self.path_edit.text().strip()
        if text:
            st["folder"] = str(Path(text).parent if self.mode == "concatenate" else Path(text))
        if self.mode != "concatenate":
            st["pattern"] = self.pattern_edit.text()

    def validate(self) -> str | None:
        rows = self.target_rows()
        if not rows:
            return "There is nothing to export: select a population (or a group whose samples have it)."
        if not self.path_edit.text().strip():
            return "Choose where to write the files."
        if self.param_mode() == "selected" and not self.chosen_parameters():
            return "Select at least one parameter."
        return None

    def run(self) -> list[Path]:
        """Write the files; returns their paths (problems in `errors`)."""
        self.errors = []
        self.written = []
        rows = self.target_rows()
        for s, reason in self.missing:
            self.errors.append(f"{s.name}: {reason}")
        opts = self.options()
        fmt = self.fmt()
        if self.mode == "concatenate":
            path = Path(self.path_edit.text().strip())
            if path.suffix.lower() != f".{fmt}":
                path = path.with_name(path.name + f".{fmt}")
            mode = self.param_mode()
            kwargs = {
                "parameters": opts.get("parameters"),
                "mode": None if mode == "selected" else mode,
                "match": self.match_combo.currentData(),
                "include_compensated": mode in ("all", "compensated"),
                "include_derived": opts["include_derived"],
                "include_computed": opts["include_computed"],
                "keywords": self.chosen_keywords(),
                "max_events_per_sample": opts["max_events"],
                "percent": opts["percent"],
                "seed": opts["seed"],
                "sample_id": self.sample_id_check.isChecked(),
                "scaled": opts["scaled"],
                "scale_range": opts["scale_range"],
                "extra_keywords": opts.get("extra_keywords"),
                "fmt": fmt,
            }
            try:
                self.events_written = core_export.concatenate([s for s, _ in rows], [p for _, p in rows], path, **kwargs)
                self.written = [path]
            except (OSError, ValueError, KeyError, core_fcs.FCSError) as exc:
                self.errors.append(str(exc).strip("'\""))
        else:
            folder = Path(self.path_edit.text().strip())
            group = self.selected_group() if self.batch_check.isChecked() else None
            if "parameters" in opts:
                opts["skip_missing"] = True
            try:
                result = core_export.export_rows(rows, folder, self.pattern_edit.text().strip() or core_export.DEFAULT_PATTERN, fmt, group, **opts)
            except OSError as exc:
                self.errors.append(str(exc))
                return []
            self.written = result.paths
            self.events_written = sum(n for _, _, n in result.written)
            self.errors.extend(f"{s.name}: {reason}" for s, reason in result.skipped)
        if self.written and fmt == "fcs" and self.add_check.isChecked():
            window = self.parent()
            adder = getattr(window, "add_sample_paths", None)
            if callable(adder):
                adder([str(p) for p in self.written])
            else:
                self.controller.add_samples([str(p) for p in self.written])
        self._remember()
        return self.written

    def accept(self):
        problem = self.validate()
        if problem:
            QMessageBox.warning(self, self.windowTitle(), problem)
            return
        self.run()
        if self.errors and not self.written:
            QMessageBox.warning(self, self.windowTitle(), "Nothing was written:\n" + "\n".join(self.errors[:12]))
            return
        if self.errors:
            QMessageBox.warning(self, self.windowTitle(), f"{len(self.written)} file(s) written; some were skipped:\n" + "\n".join(self.errors[:12]))
        super().accept()


class ExportTools:
    """What `install` returns: the export actions and the dialog launcher."""

    def __init__(self, window):
        self.window = window
        self.controller = window.controller
        self.state: dict = {}  # dialog settings kept for the session
        self.last_dialog: ExportDialog | None = None
        self.actions: dict[str, QAction] = {}
        specs = [
            ("export", "Export…", "export", "Export the selected populations: parameters, linear or transformed values, random subsample, file names, keywords to add"),
            ("batch", "Batch Export…", "group", "Export the selected population(s) for every sample of a group, one file per sample"),
            ("split", "Split Data Sets into Files…", "export", "Write every data set of a multi-data-set file ($NEXTDATA, Beckman .lmd) as its own FCS file and add them"),
        ]
        for key, text, ic, tip in specs:
            act = QAction(icon(ic), text, window)
            act.setToolTip(tip)
            act.setStatusTip(tip)
            self.actions[key] = act
        self.actions["export"].triggered.connect(lambda: self.open("fcs"))
        self.actions["batch"].triggered.connect(lambda: self.open("fcs", batch=True))
        self.actions["split"].triggered.connect(self.split_datasets)
        memmap = QAction("Memory-Map Large FCS Files", window)
        memmap.setCheckable(True)
        memmap.setToolTip("Read large FCS files through a memory map instead of loading them into memory at once (preference fcs_memmap)")
        memmap.toggled.connect(self._memmap_toggled)
        self.actions["memmap"] = memmap
        self.band = None
        page = getattr(window, "ribbon_pages", {}).get("tools")
        if page is not None:
            band = next((b for b in getattr(page, "bands", []) if getattr(b, "title", "") == "Export"), None)
            if band is None:
                band = page.add_band("Export")
            band.add_small(self.actions["export"])
            band.add_small(self.actions["batch"])
            self.band = band
        menu = getattr(window, "menus", {}).get("tools")
        self.menu = menu
        if menu is not None:
            menu.addSeparator()
            for key in ("export", "batch", "split", "memmap"):
                menu.addAction(self.actions[key])
        if hasattr(window, "tree_context_hooks"):
            window.tree_context_hooks.append(self._tree_hook)
        if hasattr(window, "group_context_hooks"):
            window.group_context_hooks.append(self._group_hook)
        sig = getattr(self.controller, "preferencesChanged", None)
        if sig is not None:
            sig.connect(self.apply_preferences)
        sel = getattr(self.controller, "selectionChanged", None)
        if sel is not None:
            sel.connect(self._sync_actions)
        self.apply_preferences()
        self._sync_actions()

    # ---------------------------------------------------------- preferences
    def apply_preferences(self):
        prefs = getattr(self.controller, "preferences", None)
        enabled = bool(_pref(prefs, PREF_MEMMAP, True))
        min_mb = int(_pref(prefs, PREF_MEMMAP_MIN_MB, 64))
        core_fcs.set_memmap(enabled, max(0, min_mb) * 1024 * 1024)
        act = self.actions["memmap"]
        if act.isChecked() != enabled:
            act.blockSignals(True)
            act.setChecked(enabled)
            act.blockSignals(False)

    def _memmap_toggled(self, on: bool):
        setter = getattr(self.controller, "set_preference", None)
        if callable(setter):
            setter(PREF_MEMMAP, bool(on))
        else:
            core_fcs.set_memmap(bool(on))

    # ------------------------------------------------------------- commands
    def _sync_actions(self, *_):
        rows = list(getattr(self.controller, "selected", []))
        self.actions["export"].setEnabled(bool(rows))
        self.actions["split"].setEnabled(any(self._has_datasets(s) for s in {id(s): s for s, _ in rows}.values()))

    @staticmethod
    def _has_datasets(sample: Sample) -> bool:
        try:
            return int(float(sample.keywords.get("$NEXTDATA", "0") or 0)) > 0 or sample.path.suffix.lower() == ".lmd"
        except ValueError:
            return False

    def export_fcs(self):
        return self.open("fcs")

    def export_events(self):
        return self.open("csv")

    def concatenate_samples(self):
        return self.open("concatenate")

    def make_dialog(self, mode: str = "fcs", rows=None, group: Group | None = None, batch: bool = False, paths: list[str] | None = None) -> ExportDialog:
        rows = list(self.controller.selected) if rows is None else list(rows)
        dlg = ExportDialog(self.controller, rows, mode, group, batch, paths, self.window, self.state)
        self.last_dialog = dlg
        return dlg

    def open(self, mode: str = "fcs", rows=None, group: Group | None = None, batch: bool = False, paths: list[str] | None = None) -> ExportDialog | None:
        rows = list(self.controller.selected) if rows is None else list(rows)
        if not rows and not batch:
            QMessageBox.information(self.window, "Export", "Select one or more samples or populations first.")
            return None
        dlg = self.make_dialog(mode, rows, group, batch, paths)
        if dlg.exec():
            what = f"{dlg.events_written:,} events to {dlg.written[0].name}" if mode == "concatenate" and dlg.written else f"{len(dlg.written)} file(s)"
            status = getattr(self.window, "statusBar", None)
            if callable(status):
                status().showMessage(f"Wrote {what}", 6000)
        return dlg

    def export_for_group(self, group: Group, paths: list[str] | None = None, mode: str = "fcs") -> ExportDialog | None:
        """The export dialog set to batch over `group` for `paths` (default:
        the selected populations, or the whole samples)."""
        if paths is None:
            paths = list(dict.fromkeys(p.path for _, p in self.controller.selected)) or [""]
        return self.open(mode, rows=list(self.controller.selected), group=group, batch=True, paths=paths)

    def split_datasets(self, samples: list[Sample] | None = None) -> list[Path]:
        samples = samples if samples is not None else self.controller.selected_samples()
        written: list[Path] = []
        problems = []
        for s in samples:
            try:
                written.extend(core_export.split_datasets(s.path))
            except (OSError, core_fcs.FCSError) as exc:
                problems.append(f"{s.name}: {exc}")
        if written:
            adder = getattr(self.window, "add_sample_paths", None)
            if callable(adder):
                adder([str(p) for p in written])
            else:
                self.controller.add_samples([str(p) for p in written])
        status = getattr(self.window, "statusBar", None)
        if callable(status):
            status().showMessage(f"Wrote {len(written)} data set file(s)" if written else "No file with several data sets selected", 6000)
        if problems:
            QMessageBox.warning(self.window, "Split Data Sets", "\n".join(problems))
        return written

    # ------------------------------------------------------- context menus
    def _tree_hook(self, menu, item):
        from .main_window import ROLE_GROUP, ROLE_POP, ROLE_SAMPLE, ROLE_STAT

        if item is None or item.data(0, ROLE_STAT) is not None:
            return
        group = item.data(0, ROLE_GROUP)
        pop = item.data(0, ROLE_POP)
        if group is not None:
            if pop is None:
                return
            label = pop.name if pop.parent is not None else "all samples"
            menu.addSeparator()
            act = menu.addAction(icon("export"), f"Export “{label}” for every sample of “{group.name}”…")
            act.triggered.connect(lambda _=False, g=group, p=pop.path: self.export_for_group(g, [p]))
            return
        sample = item.data(0, ROLE_SAMPLE)
        menu.addSeparator()
        current = getattr(self.controller, "current_group", None)
        if current is not None and pop is not None:
            label = pop.name if pop.parent is not None else "Ungated"
            act = menu.addAction(icon("group"), f"Export “{label}” for every sample of “{current.name}”…")
            act.triggered.connect(lambda _=False, g=current: self.export_for_group(g))
        if sample is not None and self._has_datasets(sample):
            menu.addAction(self.actions["split"])

    def _group_hook(self, menu, group):
        if group is None:
            return
        pops = [p for _, p in self.controller.selected]
        what = f"{len(set(p.path for p in pops))} selected population(s)" if pops else "all events"
        menu.addSeparator()
        act = menu.addAction(icon("export"), f"Export {what} for every sample of this group…")
        act.triggered.connect(lambda _=False, g=group: self.export_for_group(g))


def install(window) -> ExportTools:
    return ExportTools(window)


__all__ = ["ExportDialog", "ExportTools", "install", "keyword_candidates", "PREF_MEMMAP", "PREF_MEMMAP_MIN_MB"]
