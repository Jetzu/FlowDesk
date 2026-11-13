"""Experiment band: annotate, inspect, keywords, sorting and templates."""
from __future__ import annotations

import csv
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.model import Population  # noqa: E402

NAMES = ["Plate1_A1.fcs", "Plate1_B2.fcs", "Plate1_A10.fcs", "Plate1_A2.fcs"]


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def make_files(folder, names=NAMES, extra=None):
    paths = []
    rng = np.random.default_rng(3)
    for k, name in enumerate(names):
        n = 200 + 100 * k
        events = np.column_stack([rng.normal(50000, 5000, n), rng.normal(30000, 4000, n), rng.lognormal(7, 1, n)]).astype(np.float32)
        channels = [Channel(1, "FSC-A"), Channel(2, "SSC-A"), Channel(3, "FITC-A", "CD3")]
        kw = {"$CYT": "TestCyt", "$DATE": "07-NOV-2026"}
        if extra:
            kw.update(extra(k, name))
        p = folder / name
        write_fcs(p, events, channels, extra_keywords=kw)
        paths.append(p)
    return paths


@pytest.fixture
def window(app, tmp_path):
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    win.controller.add_samples(make_files(tmp_path))
    app.processEvents()
    yield win
    win.controller.workspace.modified = False
    for w in list(win.graph_windows.values()):
        w.close()
    win.hide()
    win.deleteLater()


def ext(win):
    return win.extensions["flowdesk.ui.experiment"]


def titles(menu):
    return [a.text() for a in menu.actions()]


def test_install_adds_actions(window):
    ws_titles = titles(window.menus["workspace"])
    for t in ("Annotate Experiment…", "Inspect Sample…", "Add Keyword…", "Export Keywords (CSV)…", "Sort Samples by Keyword…"):
        assert t in ws_titles
    file_titles = titles(window.menus["file"])
    assert "Save as Template…" in file_titles and "Apply Template…" in file_titles
    # templates come before Quit, after a separator
    acts = window.menus["file"].actions()
    i = file_titles.index("Save as Template…")
    assert acts[i - 1].isSeparator()
    assert file_titles.index("Quit") > i
    e = ext(window)
    assert e.band is not None and e.templates_band is not None


def test_annotate_writes_settings_and_stains(window):
    from flowdesk.ui.experiment import AnnotateExperimentDialog

    c = window.controller
    dlg = AnnotateExperimentDialog(c, window)
    params = [dlg.panel.item(r, 0).text() for r in range(dlg.panel.rowCount())]
    assert params == ["FSC-A", "SSC-A", "FITC-A"]
    assert dlg.panel.item(2, 1).text() == "CD3"
    assert dlg.panel.item(2, 2).text() == "FITC"
    dlg.name_edit.setText("T cell screen")
    dlg.investigator_edit.setText("L. G.")
    assert dlg.set_stain("FITC-A", "CD8")
    seen = []
    c.samplesChanged.connect(lambda: seen.append(1))
    dlg.accept()
    info = c.workspace.settings["experiment"]
    assert info["name"] == "T cell screen" and info["investigator"] == "L. G."
    assert {"parameter": "FITC-A", "stain": "CD8", "fluorochrome": "FITC"} in info["panel"]
    for s in c.workspace.samples:
        assert s.keywords["$P3S"] == "CD8"
        assert s.base_channel("FITC-A").stain == "CD8"
        assert s.param_label("FITC-A") == "FITC-A :: CD8"
    assert seen and c.workspace.modified


def test_inspect_sample(window):
    c = window.controller
    s = c.workspace.samples[0]
    dlg = ext(window).inspect(s)
    text = dlg.text()
    assert str(s.path) in text and "TestCyt" in text and "Total events: 200" in text
    assert "FITC-A\tCD3" in text and "Compensation: None" in text
    assert dlg.params.rowCount() == 3
    dlg.copy()
    from PySide6.QtGui import QGuiApplication

    assert QGuiApplication.clipboard().text() == text
    dlg.close()


def test_add_keyword_value_regex_csv(window, tmp_path):
    from flowdesk.ui.experiment import AddKeywordDialog

    c = window.controller
    samples = c.workspace.samples
    c.set_selection([(s, s.root) for s in samples[:2]])
    dlg = AddKeywordDialog(c, c.selected_samples(), window)
    dlg.name_edit.setText("Condition")
    dlg.value_edit.setText("stim")
    groups_seen = []
    c.groupsChanged.connect(lambda: groups_seen.append(1))
    assert dlg.apply() == 2
    assert [s.keywords.get("Condition") for s in samples] == ["stim", "stim", None, None]
    assert groups_seen and c.can_undo

    dlg = AddKeywordDialog(c, samples, window)
    dlg.name_edit.setText("Row")
    dlg.set_mode(AddKeywordDialog.MODE_REGEX)
    dlg.regex_edit.setText(r"_([A-H])\d+")
    assert dlg.apply() == 4
    assert [s.keywords["Row"] for s in samples] == ["A", "B", "A", "A"]

    path = tmp_path / "donors.csv"
    path.write_text("Sample,Donor\nPlate1_A1.fcs,D1\nPlate1_B2,D2\n", encoding="utf-8")
    dlg = AddKeywordDialog(c, samples, window)
    dlg.set_mode(AddKeywordDialog.MODE_CSV)
    dlg.csv_edit.setText(str(path))
    assert dlg.name_edit.text() == "Donor"  # header column name
    assert dlg.apply() == 2
    assert samples[0].keywords["Donor"] == "D1" and samples[1].keywords["Donor"] == "D2"
    assert "Donor" not in samples[2].keywords


def test_keyword_group_gets_gates(window):
    from flowdesk.core.model import Criterion
    from flowdesk.ui.experiment import set_keyword

    c = window.controller
    g = c.create_group("Stim", criteria=[Criterion("Condition", "Equals", "stim")])
    g.root.add_child(Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [0, 0], [1e6, 1e6])))
    s = c.workspace.samples[3]
    assert set_keyword(c, [s], "Condition", "stim") == 1
    assert s.root.find("Cells") is not None


def test_export_keywords_and_sort(window, tmp_path):
    from flowdesk.ui.experiment import natural_key

    c = window.controller
    for s, v in zip(c.workspace.samples, ["10", "2", "1", "x"]):
        s.keywords["Order"] = v
    out = ext(window).export_keywords(str(tmp_path / "kw.csv"))
    rows = list(csv.reader(open(out, encoding="utf-8")))
    assert rows[0][0] == "Sample" and "Order" in rows[0] and "$CYT" in rows[0]
    assert len(rows) == 5 and rows[1][0] == "Plate1_A1.fcs"
    assert rows[1][rows[0].index("Order")] == "10"

    ext(window).sort_samples("Order")
    assert [s.keywords["Order"] for s in c.workspace.samples] == ["1", "2", "10", "x"]
    ext(window).sort_samples("$FIL")
    assert [s.name for s in c.workspace.samples] == ["Plate1_A1.fcs", "Plate1_A2.fcs", "Plate1_A10.fcs", "Plate1_B2.fcs"]
    window.tree.rebuild()
    assert window.tree.topLevelItem(1).text(0) == "Plate1_A2.fcs"
    assert natural_key("A2") < natural_key("a10")


def test_sort_dialog(window):
    from flowdesk.ui.experiment import SortSamplesDialog

    c = window.controller
    dlg = SortSamplesDialog(c, window)
    dlg.keyword_combo.setCurrentText("$FIL")
    dlg.descending.setChecked(True)
    dlg.apply()
    assert c.workspace.samples[0].name == "Plate1_B2.fcs"


def test_template_round_trip(window, app, tmp_path):
    from flowdesk.core.transforms import LinearScale
    from flowdesk.ui.experiment import ApplyTemplateDialog
    from flowdesk.ui.main_window import MainWindow

    c = window.controller
    group = c.create_group("Panel", c.workspace.samples[:2])
    c.add_population(c.workspace.samples[0].root, Population("Lymphocytes", RectangleGate(["FSC-A", "SSC-A"], [30000, 10000], [70000, 50000])), group)
    c.set_scale(c.workspace.samples, "FSC-A", LinearScale(0, 100000))
    path = ext(window).save_template(str(tmp_path / "panel"))
    assert path.suffix == ".fdwt" and path.exists()

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    win2 = MainWindow()
    try:
        c2 = win2.controller
        c2.add_samples(make_files(other_dir, ["X_A1.fcs", "X_A2.fcs"]))
        c2.create_group("Panel", c2.workspace.samples[:1])
        signals = []
        for sig in ("groupsChanged", "layoutsChanged", "tablesChanged"):
            getattr(c2, sig).connect(lambda sig=sig: signals.append(sig))
        c2.gatingChanged.connect(lambda _x: signals.append("gating"))
        c2.scalesChanged.connect(lambda _x: signals.append("scales"))
        c2.compensationChanged.connect(lambda _x: signals.append("comp"))
        dlg = ApplyTemplateDialog(c2, str(path), win2)
        dlg.apply_scales.setChecked(True)
        dlg.accept()
        assert dlg.report["groups"] >= 2
        assert {"groupsChanged", "layoutsChanged", "tablesChanged", "gating", "scales", "comp"} <= set(signals)
        g2 = c2.workspace.group_by_name("Panel")
        assert g2 is not None and g2.root.find("Lymphocytes") is not None
        s0, s1 = c2.workspace.samples
        assert s0.root.find("Lymphocytes") is not None  # kept membership
        assert s1.root.find("Lymphocytes") is None
        assert s0.scale("FSC-A").kind == "linear"
        assert c2.current_group in c2.workspace.groups
        assert c2.workspace.modified
    finally:
        c2.workspace.modified = False
        win2.hide()
        win2.deleteLater()


def test_apply_fdw_as_template(window, tmp_path):
    from flowdesk.core.serialization import save_workspace

    c = window.controller
    g = c.create_group("FromWorkspace", c.workspace.samples[:1])
    c.add_population(c.workspace.samples[0].root, Population("Big", RectangleGate(["FSC-A"], [40000], [1e6])), g)
    fdw = tmp_path / "ws.fdw"
    save_workspace(c.workspace, fdw)
    c.delete_group(g)
    report = ext(window).apply_template(str(fdw), replace_groups=False)
    assert report["groups"] >= 1
    assert c.workspace.group_by_name("FromWorkspace") is not None


def _write_xlsx(path, rows):
    """A minimal .xlsx (shared strings for text, numbers as values)."""
    import zipfile
    from xml.sax.saxutils import escape

    shared: list[str] = []

    def cell(ref, value):
        if isinstance(value, (int, float)):
            return f'<c r="{ref}"><v>{value}</v></c>'
        if value not in shared:
            shared.append(value)
        return f'<c r="{ref}" t="s"><v>{shared.index(value)}</v></c>'

    body = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(cell(f"{chr(65 + k)}{r}", v) for k, v in enumerate(row) if v != "")
        body.append(f'<row r="{r}">{cells}</row>')
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel_ns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    sheet = f'<?xml version="1.0" encoding="UTF-8"?><worksheet {ns}><sheetData>{"".join(body)}</sheetData></worksheet>'
    sst = f'<?xml version="1.0" encoding="UTF-8"?><sst {ns}>' + "".join(f"<si><t>{escape(s)}</t></si>" for s in shared) + "</sst>"
    workbook = f'<?xml version="1.0" encoding="UTF-8"?><workbook {ns} {rel_ns}><sheets><sheet name="Keywords" sheetId="1" r:id="rId1"/></sheets></workbook>'
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/data.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        zf.writestr("xl/worksheets/data.xml", sheet)
        zf.writestr("xl/sharedStrings.xml", sst)
    return path


def test_keyword_table_import_all_columns(window, tmp_path):
    from flowdesk.ui.experiment import AddKeywordDialog

    c = window.controller
    samples = c.workspace.samples
    path = tmp_path / "design.csv"
    path.write_text(
        "Sample,Donor,Treatment,Timepoint\n"
        "Plate1_A1.fcs,D1,Stim,0h\n"
        "plate1_b2,D2,Unstim,24h\n"
        "Plate1_A10.fcs,D1,,48h\n"
        "Unknown.fcs,D9,Stim,0h\n",
        encoding="utf-8",
    )
    undo_depth = len(c._undo)
    dlg = AddKeywordDialog(c, samples, window)
    dlg.set_mode(AddKeywordDialog.MODE_CSV)
    dlg.csv_edit.setText(str(path))
    assert dlg.column_combo.currentText() == AddKeywordDialog.ALL_COLUMNS
    assert dlg.multi_column() and not dlg.name_edit.isEnabled()
    # the preview shows one column per keyword
    assert dlg.preview.columnCount() == 4
    assert [dlg.preview.horizontalHeaderItem(k).text() for k in range(4)] == ["Sample", "Donor", "Treatment", "Timepoint"]
    row = [s.name for s in samples].index("Plate1_B2.fcs")
    assert [dlg.preview.item(row, k).text() for k in range(1, 4)] == ["D2", "Unstim", "24h"]
    assert "3 of 4 samples will get 3 keywords" in dlg.status.text()
    dlg.accept()  # no keyword name needed for a multi-column import
    by = {s.name: s for s in samples}
    assert by["Plate1_A1.fcs"].keywords["Donor"] == "D1" and by["Plate1_A1.fcs"].keywords["Treatment"] == "Stim"
    assert by["Plate1_B2.fcs"].keywords["Timepoint"] == "24h"
    assert "Treatment" not in by["Plate1_A10.fcs"].keywords and by["Plate1_A10.fcs"].keywords["Timepoint"] == "48h"
    assert "Donor" not in by["Plate1_A2.fcs"].keywords
    # all three keywords in one undo step
    assert len(c._undo) == undo_depth + 1 and c.undo_label() == "import 3 keywords"
    c.undo()
    assert not any(k in s.keywords for s in samples for k in ("Donor", "Treatment", "Timepoint"))

    # one column only: the keyword is named after its header (or as typed)
    dlg = AddKeywordDialog(c, samples, window)
    dlg.set_mode(AddKeywordDialog.MODE_CSV)
    dlg.csv_edit.setText(str(path))
    dlg.column_combo.setCurrentText("Treatment")
    assert dlg.name_edit.isEnabled() and dlg.name_edit.text() == "Treatment"
    assert dlg.preview.columnCount() == 2
    dlg.name_edit.setText("Condition")
    assert dlg.apply() == 2
    assert by["Plate1_A1.fcs"].keywords["Condition"] == "Stim" and "Donor" not in by["Plate1_A1.fcs"].keywords


def test_keyword_table_xlsx(window, tmp_path):
    from flowdesk.ui.experiment import AddKeywordDialog, read_keyword_table

    path = _write_xlsx(tmp_path / "design.xlsx", [["Sample", "Donor", "Dose"], ["Plate1_A1.fcs", "D1", 10], ["Plate1_A2", "D2", 2.5]])
    header, rows = read_keyword_table(path)
    assert header == ["Sample", "Donor", "Dose"]
    assert rows == [["Plate1_A1.fcs", "D1", "10"], ["Plate1_A2", "D2", "2.5"]]
    c = window.controller
    dlg = AddKeywordDialog(c, c.workspace.samples, window)
    dlg.set_mode(AddKeywordDialog.MODE_CSV)
    dlg.csv_edit.setText(str(path))
    assert dlg.preview.columnCount() == 3
    assert dlg.apply() == 2
    by = {s.name: s for s in c.workspace.samples}
    assert by["Plate1_A2.fcs"].keywords["Dose"] == "2.5" and by["Plate1_A1.fcs"].keywords["Donor"] == "D1"


def test_read_keyword_table_formats(tmp_path):
    from flowdesk.ui.experiment import read_keyword_csv, read_keyword_table, table_keywords

    p = tmp_path / "t.tsv"
    p.write_text("a.fcs\tx\ty\nb.fcs\tz\n", encoding="utf-8")
    header, rows = read_keyword_table(p)
    assert header is None and rows == [["a.fcs", "x", "y"], ["b.fcs", "z"]]
    assert table_keywords(header, rows) == ["Column 2", "Column 3"]
    mapping, name = read_keyword_csv(p)
    assert mapping == {"a.fcs": "x", "b.fcs": "z"} and name is None


def test_acquisition_time_and_event_rate(app, tmp_path):
    from pathlib import Path

    from flowdesk.core.model import Sample
    from flowdesk.ui.experiment import acquisition_duration, event_rate, parse_fcs_time, sample_info, sample_info_text

    assert parse_fcs_time("10:00:30") == 36030.0
    assert parse_fcs_time("10:00:30:30") == 36030.5  # FCS 3.0: 1/60 s
    assert parse_fcs_time("10:00:30.25") == 36030.25  # FCS 3.1: fractions
    assert parse_fcs_time("garbage") is None
    s = Sample(Path(tmp_path / "x.fcs"), 1)
    s.keywords.update({"$BTIM": "23:59:50", "$ETIM": "00:00:10", "$TOT": "1000"})
    assert acquisition_duration(s) == 20.0  # wraps past midnight
    assert event_rate(s) == 50.0
    info = sample_info(s)
    text = sample_info_text(info)
    assert "Acquisition time: 20.0 s" in text and "Event rate: 50 events/s" in text
    s.keywords.pop("$ETIM")
    assert acquisition_duration(s) is None and event_rate(s) is None
    # loaded data with $TIMESTEP: the Time channel span decides
    events = np.column_stack([np.linspace(0, 4000, 500), np.ones(500)]).astype(np.float32)
    p = tmp_path / "timed.fcs"
    write_fcs(p, events, [Channel(1, "Time"), Channel(2, "FSC-A")], extra_keywords={"$TIMESTEP": "0.01", "$BTIM": "10:00:00", "$ETIM": "10:05:00"})
    t = Sample(p, 2)
    t.load_header()
    assert acquisition_duration(t) == 300.0  # header only: $ETIM - $BTIM
    assert t.ensure_loaded()
    assert acquisition_duration(t) == pytest.approx(40.0)
