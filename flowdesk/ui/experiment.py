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
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
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


def _stain_edits(samples: list[Sample], stains: dict[str, str]) -> list[tuple[Sample, dict[str, str]]]:
    """(sample, {$PnS keyword: new stain or ""}) for the samples whose stains differ."""
    out = []
    for s in samples:
        edits = {}
        for ch in s.channels:
            if ch.name not in stains:
                continue
            new = (stains[ch.name] or "").strip()
            key = f"$P{ch.index}S"
            if s.keywords.get(key, "").strip() != new:
                edits[key] = new
        if edits:
            out.append((s, edits))
    return out


def set_panel_stains(controller, stains: dict[str, str]) -> int:
    """Write $PnS for every sample channel named in `stains` (one undo step).
    Returns the number of samples changed. Emits the refresh signals when
    anything changed."""
    edits = _stain_edits(controller.workspace.samples, stains)
    if edits:
        controller.begin_change("edit stains", "samples")
    for s, keys in edits:
        for key, new in keys.items():
            if new:
                s.keywords[key] = new
            else:
                s.keywords.pop(key, None)
        s.refresh_channels_from_keywords()
    changed = len(edits)
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
    stains = {}
    for row in panel:
        param = row.get("parameter")
        if not param:
            continue
        stain = row.get("stain", "")
        if original_stains is None or original_stains.get(param, "") != stain:
            stains[param] = stain
    ws = controller.workspace
    if ws.settings.get("experiment") == info and not _stain_edits(ws.samples, stains):
        controller.mark_modified()
        return 0
    # one undo step for the annotation and the stains it propagates
    with controller.undo_group("annotate experiment", ("samples", "settings")):
        ws.settings["experiment"] = info
        controller.mark_modified()
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


# first cells that mark the header row of a keyword table
SAMPLE_HEADERS = ("sample", "name", "file", "filename", "file name", "$fil", "sample name", "fcs file")
_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_XLSX_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _xlsx_col_index(ref: str) -> int:
    """0-based column of a cell reference such as "C7"."""
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return max(n - 1, 0)


def read_xlsx_rows(path: Path | str) -> list[list[str]]:
    """Cell texts of the first worksheet of an .xlsx file (standard library
    only: the file is a zip of XML parts). Numbers keep their stored text
    ("10", "2.5"); empty cells are ""."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _XLSX_NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_XLSX_NS['m']}}}t")))
        sheet = "xl/worksheets/sheet1.xml"
        try:  # the first sheet listed in the workbook, wherever it is stored
            wb = ElementTree.fromstring(zf.read("xl/workbook.xml"))
            first = wb.find("m:sheets/m:sheet", _XLSX_NS)
            rels = ElementTree.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rid = first.get(_XLSX_REL) if first is not None else None
            for rel in rels:
                if rel.get("Id") == rid:
                    target = rel.get("Target", "").lstrip("/")
                    target = target if target.startswith("xl/") else f"xl/{target}"
                    if target in names:
                        sheet = target
        except (KeyError, ElementTree.ParseError):
            pass
        root = ElementTree.fromstring(zf.read(sheet))
    rows: list[list[str]] = []
    for row in root.iter(f"{{{_XLSX_NS['m']}}}row"):
        cells: dict[int, str] = {}
        for k, c in enumerate(row.findall("m:c", _XLSX_NS)):
            col = _xlsx_col_index(c.get("r", "")) if c.get("r") else k
            kind = c.get("t", "n")
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in c.iter(f"{{{_XLSX_NS['m']}}}t"))
            else:
                v = c.find("m:v", _XLSX_NS)
                text = v.text if v is not None and v.text is not None else ""
                if kind == "s" and text.isdigit() and int(text) < len(shared):
                    text = shared[int(text)]
                elif kind == "b":
                    text = "TRUE" if text == "1" else "FALSE"
                elif kind == "n" and text:
                    try:  # 10.0 -> "10", keep other numbers as stored
                        f = float(text)
                        text = str(int(f)) if f.is_integer() and abs(f) < 1e15 else text
                    except ValueError:
                        pass
            cells[col] = text
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows


def read_keyword_table(path: Path | str) -> tuple[list[str] | None, list[list[str]]]:
    """Read a keyword table: the first column names the sample, every further
    column holds one keyword. CSV / TSV / text (comma, semicolon or tab) or
    Excel .xlsx. Returns (header, rows): header is the list of column names
    when the first row is a header (its first cell reads "Sample", "Name",
    "File", ...), else None; rows are the stripped data rows whose first cell
    is not empty."""
    if str(path).lower().endswith((".xlsx", ".xlsm")):
        table = read_xlsx_rows(path)
    else:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            sample_text = fh.read(4096)
            fh.seek(0)
            try:
                table = list(csv.reader(fh, csv.Sniffer().sniff(sample_text, delimiters=",;\t")))
            except csv.Error:  # ragged rows defeat the sniffer: the commonest separator
                fh.seek(0)
                table = list(csv.reader(fh, csv.excel, delimiter=max((",", ";", "\t"), key=sample_text.count)))
    header = None
    rows: list[list[str]] = []
    for row in table:
        row = [str(c).strip() for c in row]
        if not row or not row[0]:
            continue
        if header is None and not rows and row[0].lower() in SAMPLE_HEADERS:
            header = row
            continue
        rows.append(row)
    return header, rows


def table_keywords(header: list[str] | None, rows: list[list[str]]) -> list[str]:
    """The keyword names of a table's value columns ("Column 2", ... where the
    header has no name for a column)."""
    width = max([len(r) for r in rows] + [len(header or [])] + [0])
    names = []
    for i in range(1, width):
        name = header[i].strip() if header is not None and i < len(header) else ""
        names.append(name or f"Column {i + 1}")
    return names


def read_keyword_csv(path: Path | str) -> tuple[dict[str, str], str | None]:
    """Read a two-column CSV (sample name, value). Returns (mapping, header
    keyword name or None). A header row is recognised by its first cell.
    Further columns are ignored here; see read_keyword_table."""
    header, rows = read_keyword_table(path)
    mapping = {r[0]: r[1] for r in rows if len(r) >= 2}
    header_name = header[1].strip() or None if header is not None and len(header) > 1 else None
    return mapping, header_name


def match_table_rows(samples: list[Sample], rows: list[list[str]]) -> dict[int, list[str]]:
    """sample id -> its table row, matching the first cell to the sample name,
    file name or either without extension (case-insensitive)."""
    lower = {}
    for r in rows:
        lower.setdefault(r[0].lower(), r)
    out = {}
    for s in samples:
        for key in (s.name, s.path.name, s.path.stem, Path(s.name).stem):
            r = lower.get(key.lower())
            if r is not None:
                out[s.id] = r
                break
    return out


def keyword_table_values(samples: list[Sample], header: list[str] | None, rows: list[list[str]], columns: list[str] | None = None) -> dict[str, dict[int, str]]:
    """{keyword: {sample id: value}} for the chosen columns (all when None)
    of a keyword table; empty cells set nothing."""
    names = table_keywords(header, rows)
    wanted = names if columns is None else [c for c in columns if c in names]
    matched = match_table_rows(samples, rows)
    out: dict[str, dict[int, str]] = {}
    for name in wanted:
        i = names.index(name) + 1
        values = {sid: r[i] for sid, r in matched.items() if i < len(r) and r[i] != ""}
        out[name] = values
    return out


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
    return set_keywords(controller, samples, {name: values})


def set_keywords(controller, samples: list[Sample], keywords: dict[str, dict[int, str] | str], label: str | None = None) -> int:
    """Set several keywords at once ({name: one value or {sample id: value}})
    as one undo step. Returns the number of samples that received a value."""
    if not samples:
        return 0
    plan: list[tuple[str, dict[int, str]]] = []
    for name, values in keywords.items():
        name = (name or "").strip()
        if not name:
            continue
        per_sample = values if isinstance(values, dict) else {s.id: values for s in samples}
        per_sample = {s.id: per_sample[s.id] for s in samples if s.id in per_sample}
        if per_sample:
            plan.append((name, per_sample))
    if not plan:
        return 0
    if label is None:
        label = f"set keyword {plan[0][0]}" if len(plan) == 1 else f"set {len(plan)} keywords"
    controller.begin_change(label, "samples")
    by_id = {s.id: s for s in samples}
    touched: set[int] = set()
    for name, per_sample in plan:
        for sid, value in per_sample.items():
            by_id[sid].keywords[name] = str(value)
            touched.add(sid)
    notify_keywords_changed(controller)
    return len(touched)


def notify_keywords_changed(controller):
    """After keywords were written (inside an undo step): samples that now
    match a keyword-based group receive its gates, then the views refresh."""
    ws = controller.workspace
    for g in ws.groups:
        if g.criteria:
            for s in ws.members(g):
                ws.apply_template(g, s)
    controller.mark_modified()
    controller.samplesChanged.emit()
    controller.groupsChanged.emit()
    controller.gatingChanged.emit(None)


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
    order = have + missing
    if any(a is not b for a, b in zip(order, ws.samples)):
        controller.begin_change(f"sort samples by {keyword}", "samples")
        ws.samples[:] = order
    controller.mark_modified()
    controller.samplesChanged.emit()


_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{1,2}):(\d{1,2})(?:([:.])(\d+))?\s*$")


def parse_fcs_time(text: str | None) -> float | None:
    """Seconds since midnight of a $BTIM / $ETIM value: hh:mm:ss, FCS 3.0
    hh:mm:ss:tt (tt in 1/60 s) or FCS 3.1 hh:mm:ss.cc (fractions)."""
    m = _TIME_RE.match(str(text or ""))
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if h > 23 or mi > 59 or s > 60:
        return None
    sec = h * 3600 + mi * 60 + s
    if m.group(5):
        sec += int(m.group(5)) / 60.0 if m.group(4) == ":" else float("0." + m.group(5))
    return float(sec)


def acquisition_duration(sample: Sample, use_data: bool = True) -> float | None:
    """Acquisition time in seconds: the span of the Time channel times
    $TIMESTEP when the events are loaded and both exist (and use_data),
    otherwise $ETIM - $BTIM (an end before the start wraps past midnight).
    None if unknown."""
    kw = sample.keywords
    try:
        step = float(kw.get("$TIMESTEP", "") or "nan")
    except ValueError:
        step = float("nan")
    fcs = getattr(sample, "fcs", None) if use_data else None
    if fcs is not None and step == step and step > 0:
        from ..core.qc import is_time_parameter

        for ch in sample.channels:
            if not is_time_parameter(ch.name):
                continue
            try:
                values = np.asarray(fcs.column(ch.name), dtype=float)
                values = values[np.isfinite(values)]
            except (KeyError, ValueError, IndexError, TypeError):
                break
            if values.size > 1 and values.max() > values.min():
                return float((values.max() - values.min()) * step)
            break
    start, end = parse_fcs_time(kw.get("$BTIM")), parse_fcs_time(kw.get("$ETIM"))
    if start is None or end is None:
        return None
    if end < start:
        end += 24 * 3600.0
    return end - start if end > start else None


def event_rate(sample: Sample, use_data: bool = True) -> float | None:
    """Mean acquisition rate in events per second, None if the duration is unknown."""
    duration = acquisition_duration(sample, use_data)
    if not duration:
        return None
    return sample.n_events / duration


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes, sec = divmod(seconds, 60.0)
    return f"{seconds:,.1f} s ({int(minutes)}:{sec:04.1f})" if minutes else f"{seconds:,.1f} s"


def format_rate(rate: float | None) -> str:
    return "unknown" if rate is None else f"{rate:,.0f} events/s"


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
        "duration": acquisition_duration(sample),
        "event_rate": event_rate(sample),
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
        f"Acquisition time: {format_duration(info.get('duration'))}",
        f"Event rate: {format_rate(info.get('event_rate'))}",
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
    controller.begin_change("apply template", "all")
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
            ("Acquisition time:", format_duration(i.get("duration"))),
            ("Event rate:", format_rate(i.get("event_rate"))),
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
    ALL_COLUMNS = "All columns"

    def __init__(self, controller, samples: list[Sample], parent=None):
        super().__init__(parent)
        self.controller = controller
        self.samples = list(samples)
        self._table_cache: tuple | None = None  # (path, mtime, header, rows)
        self._auto_name = ""  # keyword name filled in from the table header
        self.setWindowTitle("Add Keyword")
        self.resize(640, 500)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Donor")
        form.addRow("Keyword:", self.name_edit)
        layout.addLayout(form)
        self.mode_group = QButtonGroup(self)
        self.rb_value = QRadioButton("Same value for the selected samples")
        self.rb_regex = QRadioButton("Capture from the file name (regular expression)")
        self.rb_csv = QRadioButton("Read from a table (CSV / TSV / Excel: sample name, then one column per keyword)")
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
        self.column_combo = QComboBox()
        self.column_combo.setToolTip("Import every keyword column of the table, or only one of them")
        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("Columns:"))
        col_row.addWidget(self.column_combo, 1)
        layout.addWidget(self.rb_value)
        layout.addWidget(self.value_edit)
        layout.addWidget(self.rb_regex)
        layout.addWidget(self.regex_edit)
        layout.addWidget(self.rb_csv)
        layout.addLayout(csv_row)
        layout.addLayout(col_row)
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
        for w in (self.value_edit, self.regex_edit, self.csv_edit, self.name_edit):
            w.textChanged.connect(self.update_preview)
        self.csv_edit.textChanged.connect(lambda *_: self._fill_columns())
        self.column_combo.currentIndexChanged.connect(lambda *_: self._column_changed())
        self.mode_group.idToggled.connect(lambda *_: self.update_preview())
        self.update_preview()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Keyword Table", "", "Tables (*.csv *.tsv *.txt *.xlsx);;All files (*)")
        if path:
            self.csv_edit.setText(path)
            self.rb_csv.setChecked(True)

    def set_mode(self, mode: int):
        self.mode_group.button(mode).setChecked(True)

    def mode(self) -> int:
        return self.mode_group.checkedId()

    # ------------------------------------------------------------- table
    def table(self) -> tuple[list[str] | None, list[list[str]]] | None:
        """(header, rows) of the chosen table file, None when it cannot be read."""
        path = self.csv_edit.text().strip()
        if not path or not Path(path).is_file():
            return None
        try:
            mtime = Path(path).stat().st_mtime
            if self._table_cache is None or self._table_cache[:2] != (path, mtime):
                header, rows = read_keyword_table(path)
                self._table_cache = (path, mtime, header, rows)
        except (OSError, UnicodeDecodeError, csv.Error, zipfile.BadZipFile, ElementTree.ParseError, KeyError) as exc:
            self.status.setText(f"Could not read the table: {exc}")
            self._table_cache = None
            return None
        return self._table_cache[2], self._table_cache[3]

    def table_columns(self) -> list[str]:
        t = self.table()
        return table_keywords(*t) if t is not None else []

    def _fill_columns(self):
        cols = self.table_columns()
        keep = self.column_combo.currentText()
        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        if cols:
            if len(cols) > 1:
                self.column_combo.addItem(self.ALL_COLUMNS)
            self.column_combo.addItems(cols)
            i = self.column_combo.findText(keep)
            self.column_combo.setCurrentIndex(i if i >= 0 else 0)
        self.column_combo.blockSignals(False)
        self._column_changed()

    def selected_columns(self) -> list[str]:
        cols = self.table_columns()
        cur = self.column_combo.currentText()
        if cur == self.ALL_COLUMNS or (not cur and len(cols) > 1):
            return cols
        return [cur] if cur in cols else cols[:1]

    def multi_column(self) -> bool:
        """True when several table columns are imported, each under its header name."""
        return self.mode() == self.MODE_CSV and len(self.selected_columns()) > 1

    def _column_changed(self):
        cols = self.selected_columns()
        header = (self.table() or (None, []))[0]
        if len(cols) == 1:
            # one column: the keyword is named after its header unless typed
            name = cols[0] if header is not None or not cols[0].startswith("Column ") else ""
            if name and self.name_edit.text().strip() in ("", self._auto_name):
                self._auto_name = name
                self.name_edit.setText(name)
        self.name_edit.setEnabled(not self.multi_column())
        self.update_preview()

    # ------------------------------------------------------------ values
    def computed_table(self) -> dict[str, dict[int, str]]:
        """{keyword: {sample id: value}} that Apply would write."""
        mode = self.mode()
        if mode == self.MODE_CSV:
            t = self.table()
            if t is None:
                return {}
            cols = self.selected_columns()
            values = keyword_table_values(self.samples, t[0], t[1], cols)
            if len(cols) == 1:
                name = self.name_edit.text().strip() or cols[0]
                return {name: values.get(cols[0], {})}
            return values
        return {self.name_edit.text().strip(): self.computed_values()}

    def computed_values(self) -> dict[int, str]:
        mode = self.mode()
        if mode == self.MODE_REGEX:
            try:
                return keyword_values_from_regex(self.samples, self.regex_edit.text())
            except re.error as exc:
                self.status.setText(f"Invalid expression: {exc}")
                return {}
        if mode == self.MODE_CSV:
            table = self.computed_table()
            return next(iter(table.values())) if table else {}
        return {s.id: self.value_edit.text() for s in self.samples}

    def update_preview(self):
        self.status.setText("")
        if self.mode() == self.MODE_CSV and self.column_combo.count() == 0 and self.table() is not None:
            self._fill_columns()
            return
        table = self.computed_table()
        names = list(table) if self.mode() == self.MODE_CSV else ["Value"]
        self.preview.setColumnCount(1 + len(names))
        self.preview.setHorizontalHeaderLabels(["Sample"] + [n or "Value" for n in names])
        self.preview.setRowCount(len(self.samples))
        got: set[int] = set()
        for row, s in enumerate(self.samples):
            self.preview.setItem(row, 0, QTableWidgetItem(s.name))
            for col, values in enumerate(table.values()):
                v = values.get(s.id)
                if v is not None:
                    got.add(s.id)
                self.preview.setItem(row, col + 1, QTableWidgetItem("—" if v is None else str(v)))
        if not self.status.text():
            what = f"{len(table)} keywords" if len(table) > 1 else "a value"
            self.status.setText(f"{len(got)} of {len(self.samples)} samples will get {what}")

    def apply(self) -> int:
        if self.mode() == self.MODE_CSV:
            table = self.computed_table()
            label = f"import {len(table)} keywords" if len(table) > 1 else None
            return set_keywords(self.controller, self.samples, table, label)
        return set_keyword(self.controller, self.samples, self.name_edit.text(), self.computed_values())

    def accept(self):
        if not self.multi_column() and not self.name_edit.text().strip():
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
