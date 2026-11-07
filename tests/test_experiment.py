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
