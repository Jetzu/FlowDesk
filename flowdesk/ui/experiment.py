"""Experiment band: annotate the experiment and its staining panel, inspect a
sample, add / export keywords, sort samples by a keyword, and save or apply
workspace templates.

The module is a main-window extension: `install(window)` adds an "Experiment"
band to the Workspace ribbon page, a "Templates" band to the FlowDesk page and
the matching menu entries, and returns an `ExperimentExtension` holding them.
The logic lives in plain functions (usable without any dialog) so that the
dialogs only collect input and call them.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from PySide6.QtCore import QObject, Qt
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.model import Sample
from ..core.templates import TEMPLATE_EXTENSION, apply_template, load_template, save_template
from .icons import icon

EXPERIMENT_KEYS = ("name", "investigator", "date", "description", "notes")
SCATTER_PREFIXES = ("FSC", "SSC", "TIME", "EVENT", "WIDTH", "HEIGHT", "SAMPLEID")


# =============================================================== pure helpers
def natural_key(text: str):
    """Sort key that orders "A2" before "A10" (case-insensitive)."""
    parts = re.split(r"(\d+)", str(text or ""))
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p.lower()) for p in parts if p != "")


def guess_fluorochrome(parameter: str) -> str:
    name = parameter.strip()
    if name.upper().startswith(SCATTER_PREFIXES):
        return ""
    name = re.sub(r"^(Comp-|comp-|\[|<)", "", name)
    name = re.sub(r"[\]>]$", "", name)
    name = re.sub(r"[-_ ]?[AHW]$", "", name)
    return name


def panel_parameters(samples: list[Sample]) -> list[tuple[str, str]]:
    """Union of ($PnN, $PnS) over the samples, in first-seen order. The stain
    is the first non-empty stain found for that parameter."""
    seen: dict[str, str] = {}
    for s in samples:
        for ch in s.channels:
            if ch.name not in seen:
                seen[ch.name] = ch.stain or ""
            elif not seen[ch.name] and ch.stain:
                seen[ch.name] = ch.stain
    return list(seen.items())


def set_panel_stains(controller, stains: dict[str, str]) -> int:
    """Write $PnS for every sample channel named in `stains`. Returns the number
    of samples changed. Emits the refresh signals when anything changed."""
    changed = 0
    for s in controller.workspace.samples:
        touched = False
        for ch in s.channels:
            if ch.name not in stains:
                continue
            new = (stains[ch.name] or "").strip()
            key = f"$P{ch.index}S"
            if s.keywords.get(key, "").strip() == new:
                continue
            if new:
                s.keywords[key] = new
            else:
                s.keywords.pop(key, None)
            touched = True
        if touched:
            s.refresh_channels_from_keywords()
            changed += 1
    if changed:
        controller.mark_modified()
        controller.samplesChanged.emit()
        controller.gatingChanged.emit(None)
    return changed


def apply_annotation(controller, info: dict, original_stains: dict[str, str] | None = None) -> int:
    """Store the experiment annotation in workspace.settings["experiment"] and
    propagate edited stains. Only stains that differ from `original_stains`
    (what the dialog showed) are written, so per-sample differences survive."""
    info = dict(info)
    panel = [dict(row) for row in info.get("panel", [])]
    info["panel"] = panel
    controller.workspace.settings["experiment"] = info
    controller.mark_modified()
    stains = {}
    for row in panel:
        param = row.get("parameter")
        if not param:
            continue
        stain = row.get("stain", "")
        if original_stains is None or original_stains.get(param, "") != stain:
            stains[param] = stain
    return set_panel_stains(controller, stains) if stains else 0


def keyword_values_from_regex(samples: list[Sample], pattern: str) -> dict[int, str]:
    """sample id -> captured value (first group, or the whole match)."""
    rx = re.compile(pattern, re.IGNORECASE)
    out = {}
    for s in samples:
        m = rx.search(s.name)
        if m is None:
            continue
        out[s.id] = m.group(1) if m.groups() else m.group(0)
    return out


def read_keyword_csv(path: Path | str) -> tuple[dict[str, str], str | None]:
    """Read a two-column CSV (sample name, value). Returns (mapping, header
    keyword name or None). A header row is recognised by its first cell."""
    mapping: dict[str, str] = {}
    header_name = None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample_text = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        for k, row in enumerate(csv.reader(fh, dialect)):
            if len(row) < 2 or not row[0].strip():
                continue
            if k == 0 and row[0].strip().lower() in ("sample", "name", "file", "filename", "$fil", "sample name"):
                header_name = row[1].strip() or None
                continue
            mapping[row[0].strip()] = row[1].strip()
    return mapping, header_name


def keyword_values_from_mapping(samples: list[Sample], mapping: dict[str, str]) -> dict[int, str]:
    lower = {k.lower(): v for k, v in mapping.items()}
    out = {}
    for s in samples:
        for key in (s.name, s.path.name, s.path.stem, Path(s.name).stem):
            v = lower.get(key.lower())
            if v is not None:
                out[s.id] = v
                break
    return out


def set_keyword(controller, samples: list[Sample], name: str, values: dict[int, str] | str) -> int:
    """Set keyword `name` on samples (one value for all, or per sample id).
    Returns the number of samples changed."""
    name = name.strip()
    if not name or not samples:
        return 0
    per_sample = values if isinstance(values, dict) else {s.id: values for s in samples}
    targets = [s for s in samples if s.id in per_sample]
    if not targets:
        return 0
    controller.begin_change(f"set keyword {name}")
    ws = controller.workspace
    for s in targets:
        s.keywords[name] = str(per_sample[s.id])
    # keyword-based groups may now include more samples: give them the gates
    for g in ws.groups:
        if g.criteria:
            for s in ws.members(g):
                ws.apply_template(g, s)
    controller.mark_modified()
    controller.samplesChanged.emit()
    controller.groupsChanged.emit()
    controller.gatingChanged.emit(None)
    return len(targets)


def keyword_union(samples: list[Sample]) -> list[str]:
    seen: dict[str, None] = {}
    for s in samples:
        for k in s.keywords:
            seen.setdefault(k, None)
    return sorted(seen, key=lambda k: (k.startswith("$"), natural_key(k)))


def export_keywords_csv(samples: list[Sample], path: Path | str, keywords: list[str] | None = None) -> int:
    """Write a table: one row per sample, one column per keyword."""
    keys = keywords if keywords is not None else keyword_union(samples)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Sample"] + keys)
        for s in samples:
            w.writerow([s.name] + [s.keywords.get(k, "") for k in keys])
    return len(samples)


def sort_samples_by_keyword(controller, keyword: str, descending: bool = False):
    """Reorder workspace.samples by a keyword value (natural order). Samples
    without that keyword keep their order at the end."""
    ws = controller.workspace

    def value(s: Sample):
        if keyword in ("$FIL", "Name", "FILENAME") and keyword not in s.keywords:
            return s.name
        return s.keywords.get(keyword)

    have = [s for s in ws.samples if value(s) not in (None, "")]
    missing = [s for s in ws.samples if value(s) in (None, "")]
    have.sort(key=lambda s: natural_key(value(s)), reverse=descending)
    ws.samples[:] = have + missing
    controller.mark_modified()
    controller.samplesChanged.emit()


def sample_info(sample: Sample) -> dict:
    """Facts shown by Inspect Sample."""
    kw = sample.keywords
    try:
        size = sample.path.stat().st_size
    except OSError:
        size = None
    comp = sample.comp if (sample.comp is not None and sample.comp_enabled and not sample.comp.is_identity) else None
    params = []
    for ch in sample.channels:
        try:
            kind = sample.scale(ch.name).kind
        except Exception:  # pragma: no cover - defensive
            kind = ""
        params.append({"name": ch.name, "stain": ch.stain, "range": ch.range, "scale": kind})
    date = kw.get("$DATE", "")
    times = " – ".join(t for t in (kw.get("$BTIM", ""), kw.get("$ETIM", "")) if t)
    return {
        "name": sample.name,
        "path": str(sample.path),
        "size": size,
        "version": sample.version or "",
        "cytometer": kw.get("$CYT", ""),
        "date": f"{date} {times}".strip(),
        "events": sample.n_events,
        "parameters": params,
        "compensation": comp.name if comp is not None else "None",
        "keyword_count": len(kw),
    }


def _format_size(size: int | None) -> str:
    if size is None:
        return "file not found"
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
        size /= 1024.0
    return str(size)


def sample_info_text(info: dict) -> str:
    lines = [
        f"Sample: {info['name']}",
        f"File: {info['path']}",
        f"Size: {_format_size(info['size'])}",
        f"FCS version: {info['version']}",
        f"Cytometer: {info['cytometer']}",
        f"Acquired: {info['date']}",
        f"Total events: {info['events']:,}",
        f"Compensation: {info['compensation']}",
        f"Keywords: {info['keyword_count']}",
        "",
        "Parameter\tStain\tRange\tScale",
    ]
    for p in info["parameters"]:
        lines.append(f"{p['name']}\t{p['stain']}\t{p['range']:g}\t{p['scale']}")
    return "\n".join(lines)


def save_workspace_template(controller, path: Path | str) -> Path:
    path = Path(path)
    if path.suffix.lower() != TEMPLATE_EXTENSION:
        path = path.with_suffix(TEMPLATE_EXTENSION)
    save_template(controller.workspace, path)
    controller.statusMessage.emit(f"Template saved to {path.name}")
    return path


def apply_workspace_template(controller, path: Path | str, replace_groups: bool = True, apply_scales: bool = True, add_layouts: bool = True) -> dict:
    template = load_template(path)
    controller.begin_change("apply template")
    report = apply_template(controller.workspace, template, replace_groups, apply_scales, add_layouts)
    ws = controller.workspace
    if controller.current_group not in ws.groups:
        name = controller.current_group.name
        controller.current_group = ws.group_by_name(name) or ws.groups[0]
        controller.currentGroupChanged.emit(controller.current_group)
    controller.selected = []
    controller.mark_modified()
    controller.groupsChanged.emit()
    controller.gatingChanged.emit(None)
    controller.scalesChanged.emit(None)
    controller.compensationChanged.emit(None)
    controller.layoutsChanged.emit()
    controller.tablesChanged.emit()
    controller.selectionChanged.emit()
    controller.statusMessage.emit(f"Applied template: {report['groups']} groups, {report['samples']} sample updates")
    return report


def _buttons(dialog: QDialog, ok_text: str = "OK") -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    box.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    return box


# ===================================================================== dialogs
class AnnotateExperimentDialog(QDialog):
    """Experiment metadata plus the staining panel ($PnS per parameter)."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Annotate Experiment")
        self.resize(620, 620)
        info = dict(controller.workspace.settings.get("experiment", {}))
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(info.get("name", "") or controller.workspace.name)
        self.investigator_edit = QLineEdit(info.get("investigator", ""))
        self.date_edit = QLineEdit(info.get("date", "") or self._default_date())
        self.date_edit.setPlaceholderText("YYYY-MM-DD")
        self.description_edit = QLineEdit(info.get("description", ""))
        self.notes_edit = QPlainTextEdit(info.get("notes", ""))
        self.notes_edit.setFixedHeight(70)
        form.addRow("Experiment name:", self.name_edit)
        form.addRow("Investigator:", self.investigator_edit)
        form.addRow("Date:", self.date_edit)
        form.addRow("Description:", self.description_edit)
        form.addRow("Notes:", self.notes_edit)
        layout.addLayout(form)

        box = QGroupBox("Panel (editing a stain writes $PnS into every sample)")
        bl = QVBoxLayout(box)
        self.panel = QTableWidget(0, 3)
        self.panel.setHorizontalHeaderLabels(["Parameter", "Stain ($PnS)", "Fluorochrome"])
        self.panel.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.panel.verticalHeader().setVisible(False)
        bl.addWidget(self.panel)
        layout.addWidget(box, 1)
        stored = {row.get("parameter"): row for row in info.get("panel", []) if isinstance(row, dict)}
        self.original_stains: dict[str, str] = {}
        for param, stain in panel_parameters(controller.workspace.samples):
            self.original_stains[param] = stain
            row = self.panel.rowCount()
            self.panel.insertRow(row)
            p = QTableWidgetItem(param)
            p.setFlags(p.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.panel.setItem(row, 0, p)
            self.panel.setItem(row, 1, QTableWidgetItem(stain))
            fluor = stored.get(param, {}).get("fluorochrome") or guess_fluorochrome(param)
            self.panel.setItem(row, 2, QTableWidgetItem(fluor))
        layout.addWidget(_buttons(self))

    @staticmethod
    def _default_date() -> str:
        from datetime import date

        return date.today().isoformat()

    def set_stain(self, parameter: str, stain: str) -> bool:
        for row in range(self.panel.rowCount()):
            if self.panel.item(row, 0).text() == parameter:
                self.panel.item(row, 1).setText(stain)
                return True
        return False

    def values(self) -> dict:
        panel = []
        for row in range(self.panel.rowCount()):
            panel.append({
                "parameter": self.panel.item(row, 0).text(),
                "stain": self.panel.item(row, 1).text().strip(),
                "fluorochrome": self.panel.item(row, 2).text().strip(),
            })
        return {
            "name": self.name_edit.text().strip(),
            "investigator": self.investigator_edit.text().strip(),
            "date": self.date_edit.text().strip(),
            "description": self.description_edit.text().strip(),
            "notes": self.notes_edit.toPlainText(),
            "panel": panel,
        }

    def apply(self) -> int:
        return apply_annotation(self.controller, self.values(), self.original_stains)

    def accept(self):
        self.apply()
        super().accept()


class InspectSampleDialog(QDialog):
    def __init__(self, sample: Sample, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.info = sample_info(sample)
        self.setWindowTitle(f"Inspect Sample – {sample.name}")
        self.resize(600, 560)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        i = self.info
        for label, value in (
            ("File:", i["path"]),
            ("Size:", _format_size(i["size"])),
            ("FCS version:", i["version"]),
            ("Cytometer ($CYT):", i["cytometer"]),
            ("Acquired:", i["date"]),
            ("Total events:", f"{i['events']:,}"),
            ("Compensation:", i["compensation"]),
            ("Keywords:", str(i["keyword_count"])),
        ):
            lab = QLabel(value)
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lab.setWordWrap(True)
            form.addRow(label, lab)
        layout.addLayout(form)
        self.params = QTableWidget(len(i["parameters"]), 4)
        self.params.setHorizontalHeaderLabels(["Parameter", "Stain", "Range", "Scale"])
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.params.verticalHeader().setVisible(False)
        self.params.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, p in enumerate(i["parameters"]):
            for col, v in enumerate((p["name"], p["stain"], f"{p['range']:g}", p["scale"])):
                self.params.setItem(row, col, QTableWidgetItem(v))
        layout.addWidget(self.params, 1)
        btns = QHBoxLayout()
        copy = QPushButton(icon("copy"), "Copy")
        copy.clicked.connect(self.copy)
        btns.addWidget(copy)
        btns.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        btns.addWidget(close)
        layout.addLayout(btns)

    def text(self) -> str:
        return sample_info_text(self.info)

    def copy(self):
        QGuiApplication.clipboard().setText(self.text())


class AddKeywordDialog(QDialog):
    MODE_VALUE, MODE_REGEX, MODE_CSV = 0, 1, 2

    def __init__(self, controller, samples: list[Sample], parent=None):
        super().__init__(parent)
        self.controller = controller
        self.samples = list(samples)
        self.setWindowTitle("Add Keyword")
        self.resize(560, 480)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Donor")
        form.addRow("Keyword:", self.name_edit)
        layout.addLayout(form)
        self.mode_group = QButtonGroup(self)
        self.rb_value = QRadioButton("Same value for the selected samples")
        self.rb_regex = QRadioButton("Capture from the file name (regular expression)")
        self.rb_csv = QRadioButton("Read from a CSV file (sample name, value)")
        for k, rb in enumerate((self.rb_value, self.rb_regex, self.rb_csv)):
            self.mode_group.addButton(rb, k)
        self.rb_value.setChecked(True)
        self.value_edit = QLineEdit()
        self.regex_edit = QLineEdit(r"donor (\d+)")
        self.csv_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_edit, 1)
        csv_row.addWidget(browse)
        layout.addWidget(self.rb_value)
        layout.addWidget(self.value_edit)
        layout.addWidget(self.rb_regex)
        layout.addWidget(self.regex_edit)
        layout.addWidget(self.rb_csv)
        layout.addLayout(csv_row)
        self.preview = QTableWidget(0, 2)
        self.preview.setHorizontalHeaderLabels(["Sample", "Value"])
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.preview.verticalHeader().setVisible(False)
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(QLabel(f"Preview ({len(self.samples)} selected samples):"))
        layout.addWidget(self.preview, 1)
        self.status = QLabel()
        layout.addWidget(self.status)
        layout.addWidget(_buttons(self, "Apply"))
        for w in (self.value_edit, self.regex_edit, self.csv_edit):
            w.textChanged.connect(self.update_preview)
        self.mode_group.idToggled.connect(lambda *_: self.update_preview())
        self.update_preview()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Keyword CSV", "", "CSV (*.csv *.tsv *.txt);;All files (*)")
        if path:
            self.csv_edit.setText(path)
            self.rb_csv.setChecked(True)

    def set_mode(self, mode: int):
        self.mode_group.button(mode).setChecked(True)

    def mode(self) -> int:
        return self.mode_group.checkedId()

    def computed_values(self) -> dict[int, str]:
        mode = self.mode()
        if mode == self.MODE_REGEX:
            try:
                return keyword_values_from_regex(self.samples, self.regex_edit.text())
            except re.error as exc:
                self.status.setText(f"Invalid expression: {exc}")
                return {}
        if mode == self.MODE_CSV:
            path = self.csv_edit.text().strip()
            if not path or not Path(path).is_file():
                return {}
            try:
                mapping, header = read_keyword_csv(path)
            except (OSError, UnicodeDecodeError) as exc:
                self.status.setText(str(exc))
                return {}
            if header and not self.name_edit.text().strip():
                self.name_edit.setText(header)
            return keyword_values_from_mapping(self.samples, mapping)
        return {s.id: self.value_edit.text() for s in self.samples}

    def update_preview(self):
        self.status.setText("")
        values = self.computed_values()
        self.preview.setRowCount(len(self.samples))
        for row, s in enumerate(self.samples):
            self.preview.setItem(row, 0, QTableWidgetItem(s.name))
            self.preview.setItem(row, 1, QTableWidgetItem(values.get(s.id, "—")))
        if not self.status.text():
            self.status.setText(f"{len(values)} of {len(self.samples)} samples will get a value")

    def apply(self) -> int:
        return set_keyword(self.controller, self.samples, self.name_edit.text(), self.computed_values())

    def accept(self):
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Add Keyword", "Enter a keyword name.")
            return
        self.apply()
        super().accept()


class SortSamplesDialog(QDialog):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Sort Samples by Keyword")
        layout = QFormLayout(self)
        self.keyword_combo = QComboBox()
        self.keyword_combo.setEditable(True)
        self.keyword_combo.addItems(["$FIL"] + [k for k in keyword_union(controller.workspace.samples) if k != "$FIL"])
        self.descending = QCheckBox("Descending")
        layout.addRow("Keyword:", self.keyword_combo)
        layout.addRow("", self.descending)
        layout.addRow(_buttons(self, "Sort"))

    def apply(self):
        sort_samples_by_keyword(self.controller, self.keyword_combo.currentText().strip(), self.descending.isChecked())

    def accept(self):
        self.apply()
        super().accept()


class ApplyTemplateDialog(QDialog):
    def __init__(self, controller, path: str = "", parent=None):
        super().__init__(parent)
        self.controller = controller
        self.report: dict | None = None
        self.setWindowTitle("Apply Template")
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.path_edit = QLineEdit(path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(QLabel("Template:"))
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        self.replace_groups = QCheckBox("Replace groups (otherwise merge gates into groups of the same name)")
        self.replace_groups.setChecked(True)
        self.apply_scales = QCheckBox("Apply axis scales")
        self.apply_scales.setChecked(True)
        self.add_layouts = QCheckBox("Add layouts and tables")
        self.add_layouts.setChecked(True)
        for cb in (self.replace_groups, self.apply_scales, self.add_layouts):
            layout.addWidget(cb)
        layout.addWidget(_buttons(self, "Apply"))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Apply Template", "", f"Templates (*{TEMPLATE_EXTENSION} *.wspt *.wsp *.fdw);;All files (*)")
        if path:
            self.path_edit.setText(path)

    def apply(self) -> dict:
        self.report = apply_workspace_template(self.controller, self.path_edit.text().strip(), self.replace_groups.isChecked(), self.apply_scales.isChecked(), self.add_layouts.isChecked())
        return self.report

    def accept(self):
        if not self.path_edit.text().strip():
            return
        try:
            self.apply()
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Apply Template", f"Could not apply the template:\n{exc}")
            return
        super().accept()


# ================================================================== extension
class ExperimentExtension(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.controller = window.controller
        self.dialogs: list[QDialog] = []

        def act(text, icon_name, slot, tip):
            a = QAction(icon(icon_name), text, window)
            a.setStatusTip(tip)
            a.setToolTip(tip)
            a.triggered.connect(lambda _checked=False: slot())
            return a

        self.act_annotate = act("Annotate Experiment…", "page", self.annotate, "Describe the experiment and its staining panel")
        self.act_inspect = act("Inspect Sample…", "sample", self.inspect, "Show file, acquisition and parameter details of the selected sample")
        self.act_add_keyword = act("Add Keyword…", "keyword", self.add_keyword, "Set a keyword on the selected samples (value, file-name pattern or CSV)")
        self.act_export_keywords = act("Export Keywords (CSV)…", "export", self.export_keywords, "Write every sample's keywords to a CSV table")
        self.act_sort = act("Sort Samples by Keyword…", "table", self.sort_samples, "Reorder the samples by a keyword value")
        self.act_save_template = act("Save as Template…", "save-as", self.save_template, f"Save groups, gates, scales and layouts without samples ({TEMPLATE_EXTENSION})")
        self.act_apply_template = act("Apply Template…", "import", self.apply_template, "Apply a FlowDesk or FlowJo template to the current samples")
        self.actions = [self.act_annotate, self.act_inspect, self.act_add_keyword, self.act_export_keywords, self.act_sort, self.act_save_template, self.act_apply_template]

        pages = getattr(window, "ribbon_pages", {})
        self.band = None
        if "workspace" in pages:
            self.band = pages["workspace"].add_band("Experiment")
            self.band.add_large(self.act_annotate)
            self.band.add_large(self.act_inspect)
            self.band.add_small(self.act_add_keyword)
            self.band.add_small(self.act_export_keywords)
            self.band.add_small(self.act_sort)
        self.templates_band = None
        if "flowdesk" in pages:
            self.templates_band = pages["flowdesk"].add_band("Templates")
            self.templates_band.add_large(self.act_save_template)
            self.templates_band.add_large(self.act_apply_template)

        menus = getattr(window, "menus", {})
        if "workspace" in menus:
            m = menus["workspace"]
            m.addSeparator()
            for a in (self.act_annotate, self.act_inspect, self.act_add_keyword, self.act_export_keywords, self.act_sort):
                m.addAction(a)
        if "file" in menus:
            m = menus["file"]
            quit_act = getattr(window, "act_quit", None)
            if quit_act is not None and quit_act in m.actions():
                for a in (self.act_save_template, self.act_apply_template, self.act_export_keywords):
                    m.insertAction(quit_act, a)
                m.insertSeparator(quit_act)
            else:
                m.addSeparator()
                for a in (self.act_save_template, self.act_apply_template, self.act_export_keywords):
                    m.addAction(a)
        self.controller.selectionChanged.connect(self.update_states)
        self.controller.samplesChanged.connect(self.update_states)
        self.controller.workspaceReplaced.connect(self.update_states)
        self.update_states()

    # ------------------------------------------------------------------ state
    def update_states(self):
        has_samples = bool(self.controller.workspace.samples)
        has_sel = bool(self.controller.selected_samples())
        self.act_inspect.setEnabled(has_sel or has_samples)
        self.act_add_keyword.setEnabled(has_samples)
        self.act_export_keywords.setEnabled(has_samples)
        self.act_sort.setEnabled(has_samples)

    def _target_samples(self) -> list[Sample]:
        sel = self.controller.selected_samples()
        if sel:
            return sel
        return self.controller.workspace.members(self.controller.current_group)

    def _keep(self, dlg: QDialog) -> QDialog:
        self.dialogs = [d for d in self.dialogs if d.isVisible()]
        self.dialogs.append(dlg)
        return dlg

    # ---------------------------------------------------------------- actions
    def annotate(self):
        dlg = AnnotateExperimentDialog(self.controller, self.window)
        dlg.exec()
        return dlg

    def inspect(self, sample: Sample | None = None):
        if sample is None:
            targets = self._target_samples()
            if not targets:
                return None
            sample = targets[0]
        dlg = self._keep(InspectSampleDialog(sample, self.window))
        dlg.show()
        return dlg

    def add_keyword(self):
        samples = self._target_samples()
        if not samples:
            return None
        dlg = AddKeywordDialog(self.controller, samples, self.window)
        dlg.exec()
        return dlg

    def export_keywords(self, path: str | None = None) -> Path | None:
        samples = self.controller.workspace.samples
        if not samples:
            return None
        if path is None:
            start = str(Path.home() / f"{self.controller.workspace.name}_keywords.csv")
            path, _ = QFileDialog.getSaveFileName(self.window, "Export Keywords", start, "CSV (*.csv)")
            if not path:
                return None
        export_keywords_csv(samples, path)
        self.controller.statusMessage.emit(f"Exported keywords of {len(samples)} samples")
        return Path(path)

    def sort_samples(self, keyword: str | None = None, descending: bool = False):
        if keyword is not None:
            sort_samples_by_keyword(self.controller, keyword, descending)
            return None
        dlg = SortSamplesDialog(self.controller, self.window)
        dlg.exec()
        return dlg

    def save_template(self, path: str | None = None) -> Path | None:
        if path is None:
            ws = self.controller.workspace
            start = str((ws.path.parent if ws.path else Path.home()) / f"{ws.name}{TEMPLATE_EXTENSION}")
            path, _ = QFileDialog.getSaveFileName(self.window, "Save as Template", start, f"FlowDesk template (*{TEMPLATE_EXTENSION})")
            if not path:
                return None
        try:
            return save_workspace_template(self.controller, path)
        except OSError as exc:
            QMessageBox.warning(self.window, "Save as Template", str(exc))
            return None

    def apply_template(self, path: str | None = None, replace_groups: bool = True, apply_scales: bool = True, add_layouts: bool = True):
        if path is not None:
            return apply_workspace_template(self.controller, path, replace_groups, apply_scales, add_layouts)
        dlg = ApplyTemplateDialog(self.controller, "", self.window)
        dlg.exec()
        return dlg.report


def install(window) -> ExperimentExtension:
    return ExperimentExtension(window)
