"""Layout Editor: PowerPoint / per-sample image export, batch options, text
tokens and formatting, plot options, overlays hand-off, arrange tools,
clipboard and layout defaults - on small synthetic FCS files."""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QMimeData, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from flowdesk.core import layout_text  # noqa: E402
from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.layout_text import TextContext, expand_text, parse_statistic, stat_token  # noqa: E402
from flowdesk.core.layouts import Layout, LayoutItem, TableColumn, TableDef  # noqa: E402
from flowdesk.core.model import Population, Workspace  # noqa: E402
from flowdesk.core.pptx_export import PptxPicture, PptxSlide, PptxText, write_pptx  # noqa: E402
from flowdesk.core.serialization import load_workspace, save_workspace  # noqa: E402
from flowdesk.core.stats import compute_statistic, format_value  # noqa: E402
from flowdesk.ui.controller import POPULATION_MIME, WorkspaceController  # noqa: E402
from flowdesk.ui.layout_editor import (  # noqa: E402
    BatchDialog,
    LayoutDefaults,
    LayoutEditorWindow,
    PageSetupDialog,
    PlotGraphicsItem,
    PlotOptionsDialog,
    TextGraphicsItem,
    caption_height,
    parse_population_entries,
    plot_rect,
)

CHANNELS = [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144), Channel(3, "FITC-A", "CD4", 262144), Channel(4, "PE-A", "CD8", 262144)]
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _fcs(path: Path, n: int, seed: int, tube: str) -> Path:
    rng = np.random.default_rng(seed)
    events = np.column_stack(
        [rng.normal(50000 + 8000 * seed, 9000, n), rng.normal(30000, 8000, n), rng.lognormal(6, 1.2, n), rng.lognormal(5, 1.0, n)]
    ).astype(np.float32)
    write_fcs(path, events, CHANNELS, {"$CYT": "Fortessa"}, {"TUBE": tube})
    return path


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_defaults(monkeypatch):
    """Never touch the user's QSettings: layout defaults live in a dict."""
    store: dict = {}
    monkeypatch.setattr(LayoutEditorWindow, "defaults", LayoutDefaults(store))
    return store


@pytest.fixture
def ws(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    w = Workspace()
    for k, (name, tube) in enumerate((("a.fcs", "B"), ("b.fcs", "A"), ("c.fcs", "B"))):
        w.add_sample(_fcs(data / name, 600 + 100 * k, k + 1, tube))
    for s in w.samples:
        s.ensure_loaded()
    w.apply_population_to_group(w.groups[0], "", Population("Cells", RectangleGate(["FSC-A", "SSC-A"], [30000, 0], [None, None])))
    return w


@pytest.fixture
def editor(app, ws):
    c = WorkspaceController(ws)
    win = LayoutEditorWindow(c)
    win.resize(1000, 800)
    win.show()
    app.processEvents()
    yield win
    win.flush_notify()
    win.hide()
    win.deleteLater()
    app.processEvents()


def _mime(payload) -> QMimeData:
    m = QMimeData()
    m.setData(POPULATION_MIME, json.dumps(payload).encode())
    return m


def _slides(path) -> tuple[zipfile.ZipFile, list[str]]:
    z = zipfile.ZipFile(path)
    names = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.findall(r"\d+", n)[0]))
    return z, names


def _slide_texts(z, name) -> list[str]:
    root = ET.fromstring(z.read(name))
    out = []
    for sp in root.iter(f"{{{NS['p']}}}sp"):
        paras = ["".join(t.text or "" for t in p.iter(f"{{{NS['a']}}}t")) for p in sp.iter(f"{{{NS['a']}}}p")]
        out.append("\n".join(paras))
    return out


def _slide_images(z, name) -> list[str]:
    rels = ET.fromstring(z.read(name.replace("slides/", "slides/_rels/") + ".rels"))
    targets = {r.get("Id"): r.get("Target") for r in rels.iter(f"{{{NS['rel']}}}Relationship")}
    root = ET.fromstring(z.read(name))
    out = []
    for blip in root.iter(f"{{{NS['a']}}}blip"):
        target = targets[blip.get(f"{{{NS['r']}}}embed")]
        out.append("ppt/" + target.replace("../", ""))
    return out


# ------------------------------------------------------------------ model
def test_props_roundtrip_custom_page_and_margin():
    item = LayoutItem(kind="text", text="x", props={"bold": True, "annotations": [{"statistic": "Count"}]})
    lay = Layout("L", items=[item], page="Custom", props={"custom_w": 500.0, "custom_h": 300.0, "margin": 18.0})
    back = Layout.from_dict(json.loads(json.dumps(lay.to_dict())))
    assert back.props == lay.props and back.items[0].props == item.props
    assert back.page_size == (500.0, 300.0) and back.margin == 18.0
    back.landscape = True
    assert back.page_size == (300.0, 500.0)
    assert Layout("Default").margin == 36.0 and Layout.from_dict({"name": "old"}).props == {}
    col = TableColumn(props={"width": 80})
    td = TableDef(columns=[col], props={"sort": "Sample"})
    td2 = TableDef.from_dict(td.to_dict())
    assert td2.props == {"sort": "Sample"} and td2.columns[0].props == {"width": 80}
    # props are copied, not shared
    d = item.to_dict()
    d["props"]["annotations"].append({"keyword": "$FIL"})
    assert len(item.props["annotations"]) == 1


# ------------------------------------------------------------- text tokens
def test_text_tokens_resolve(ws):
    s = ws.samples[0]
    cells = s.root.find("/Cells")
    ctx = TextContext(sample=s, layout_name="Fig 1", group_name="All Samples", page=2, pages=3)
    median = format_value("Median", compute_statistic("Median", s.root, "FSC-A"))
    assert expand_text("{$FIL} | {Median FSC-A}", ctx) == f"a.fcs | {median}"
    assert expand_text("{Sample} {Group} {Layout} {Page}/{Pages}", ctx) == f"{s.name} All Samples Fig 1 2/3"
    assert expand_text("{TUBE} {tube} {$CYT}", ctx) == "B B Fortessa"
    freq = format_value("Freq. of Parent", compute_statistic("Freq. of Parent", cells))
    assert expand_text("{/Cells | Freq. of Parent}% {Cells | Count}", ctx) == f"{freq}% {cells.count}"
    assert expand_text("{{literal}} {NoSuchKeyword}", ctx) == "{literal} {NoSuchKeyword}"
    assert expand_text("{$FIL}", TextContext()) == "{$FIL}"  # unresolved without a sample
    assert parse_statistic("Percentile 95 : FSC-A") == ("Percentile", "FSC-A", 95.0)
    assert parse_statistic("Count of beads") is None
    assert stat_token("Median", "FSC-A", None, "/Cells") == "{/Cells | Median FSC-A}"
    hook = lambda token, c: "42" if token == "Answer" else None  # noqa: E731
    LayoutEditorWindow.register_text_expander(hook)
    try:
        assert expand_text("{Answer}", ctx) == "42"
    finally:
        layout_text.unregister_text_expander(hook)


# ----------------------------------------------------------------- pptx
def test_write_pptx_package_is_consistent(tmp_path):
    png = QImage(4, 4, QImage.Format.Format_ARGB32)
    png.fill(0xFF2F6FD6)
    from flowdesk.ui.layout_editor import image_bytes

    data = image_bytes(png)
    slides = [
        PptxSlide([PptxPicture(data, 10, 10, 100, 100), PptxText("A & <B>\n\nline 3", 10, 120, 200, 40, 14, bold=True, italic=True, color="#ff0000", align="center", font_family="Arial")]),
        PptxSlide([PptxPicture(data, 0, 0, 50, 50)]),
    ]
    path = tmp_path / "t.pptx"
    assert write_pptx(path, slides, (595, 842), "T") == 2
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    for n in names:
        if n.endswith((".xml", ".rels")):
            ET.fromstring(z.read(n))  # well-formed
    ct = ET.fromstring(z.read("[Content_Types].xml"))
    parts = {o.get("PartName") for o in ct}
    assert {"/ppt/slides/slide1.xml", "/ppt/slides/slide2.xml", "/ppt/presentation.xml"} <= parts
    for rel_file in [n for n in names if n.endswith(".rels")]:
        base = rel_file.replace("_rels/", "").replace(".rels", "")
        folder = base.rsplit("/", 1)[0] if "/" in base else ""
        for r in ET.fromstring(z.read(rel_file)):
            target = os.path.normpath(os.path.join(folder, r.get("Target"))).replace(os.sep, "/")
            assert target in names, (rel_file, target)
    assert len([n for n in names if n.startswith("ppt/media/")]) == 1  # identical images stored once
    pres = ET.fromstring(z.read("ppt/presentation.xml"))
    sz = pres.find("p:sldSz", NS)
    assert (int(sz.get("cx")), int(sz.get("cy"))) == (595 * 12700, 842 * 12700)
    texts = _slide_texts(z, "ppt/slides/slide1.xml")
    assert texts == ["A & <B>\n\nline 3"]
    rpr = ET.fromstring(z.read("ppt/slides/slide1.xml")).find(".//a:rPr", NS)
    assert rpr.get("b") == "1" and rpr.get("i") == "1" and rpr.get("sz") == "1400"
    assert rpr.find("a:solidFill/a:srgbClr", NS).get("val") == "FF0000"


def test_export_pptx_batch_and_single(editor, ws, tmp_path):
    s = ws.samples[0]
    plot = editor.add_population(s, s.root.find("/Cells"))
    plot.show_title = True
    editor.add_text("{$FIL}: {Median FSC-A}", edit=False)
    g = ws.groups[0]
    members = ws.members(g)
    path = tmp_path / "batch.pptx"
    assert editor.export_pptx(path, batch_group=g) == len(members) == 3
    z, slides = _slides(path)
    assert len(slides) == len(members)
    seen_texts = set()
    for name, sample in zip(slides, members):
        images = _slide_images(z, name)
        assert images, name
        for img in images:
            assert z.read(img).startswith(b"\x89PNG")
        texts = _slide_texts(z, name)
        assert sample.name in texts  # the page header
        median = format_value("Median", compute_statistic("Median", sample.root, "FSC-A"))
        assert f"{sample.path.name}: {median}" in texts
        assert "Cells" in texts  # the plot title as a text box
        seen_texts.add(tuple(texts))
    assert len(seen_texts) == 3
    # the plot picture sits where the plot area is, rendered at 2x
    pic = ET.fromstring(z.read(slides[0])).find(".//p:pic/p:spPr/a:xfrm", NS)
    r = plot_rect(plot)
    assert int(pic.find("a:off", NS).get("x")) == round((plot.x + r.x()) * 12700)
    img = QImage.fromData(z.read(_slide_images(z, slides[0])[0]))
    assert img.width() == pytest.approx(2 * r.width(), abs=2)
    # a single layout -> one slide; page-as-picture mode -> one picture per slide
    one = tmp_path / "one.pptx"
    assert editor.export_pptx(one) == 1
    pages = tmp_path / "pages.pptx"
    assert editor.export_pptx(pages, batch_group=g, as_page_image=True) == 3
    z2, slides2 = _slides(pages)
    assert all(len(_slide_images(z2, n)) == 1 for n in slides2)


def test_export_images_per_sample_and_formats(editor, ws, tmp_path):
    s = ws.samples[0]
    editor.add_population(s, s.root.find("/Cells"))
    g = ws.groups[0]
    folder = tmp_path / "pngs"
    files = editor.export_images(folder, g, "png", dpi=300)
    assert len(files) == len(ws.members(g)) == 3
    assert sorted(Path(f).name for f in files) == sorted(f"Layout - {m.name}.png" for m in ws.members(g))
    w, h = editor.current_layout().page_size
    img = QImage(files[0])
    assert (img.width(), img.height()) == (round(w * 300 / 72), round(h * 300 / 72))
    assert img.dotsPerMeterX() == pytest.approx(300 / 0.0254, rel=0.01)
    assert len(editor.export_images(tmp_path / "jpg", g, "jpeg", dpi=72)) == 3
    assert all(Path(f).suffix == ".svg" for f in editor.export_images(tmp_path / "svg", g, "svg"))
    # single page in other formats, at a chosen resolution
    for ext in ("jpg", "tif", "bmp", "png"):
        out = tmp_path / f"page.{ext}"
        assert editor.export_image(out, dpi=150)
        im = QImage(str(out))
        assert im.width() == round(w * 150 / 72), ext
    assert editor.export_png(tmp_path / "default.png")
    assert QImage(str(tmp_path / "default.png")).width() == round(w * 2)


def test_batch_progress_cancel(editor, ws, tmp_path):
    s = ws.samples[0]
    editor.add_population(s, s.root.find("/Cells"))
    g = ws.groups[0]
    calls = []

    def progress(done, text=""):
        calls.append(done)
        return True

    assert editor.export_pdf(tmp_path / "all.pdf", g, progress) == 3
    assert calls[:3] == [0, 1, 2] and calls[-1] == 3

    def cancel_after_one(done, text=""):
        return done < 1

    pdf = tmp_path / "cancel.pdf"
    assert editor.export_pdf(pdf, g, cancel_after_one) == 0 and not pdf.exists()
    pptx = tmp_path / "cancel.pptx"
    assert editor.export_pptx(pptx, g, cancel_after_one) == 0 and not pptx.exists()
    assert len(editor.export_images(tmp_path / "imgs", g, "png", 72, cancel_after_one)) == 1


def test_batch_options_keyword_header_and_new_layouts(editor, ws, tmp_path):
    ctl = editor.controller
    s = ws.samples[0]
    plot = editor.add_population(s, s.root.find("/Cells"))
    plot.overlays.append({"sample_id": s.id, "pop_path": ""})
    editor.add_text("{Sample}", edit=False)
    g = ws.groups[0]
    # iterate by keyword value: one page per distinct TUBE value, ordered by it
    pages = editor.batch_pages(g, "TUBE")
    assert [p.keywords["TUBE"] for p in pages] == ["A", "B"]
    lay = editor.current_layout()
    assert editor.batch_header(lay, pages[0], 1, 2, "{TUBE} - page {Page} of {Pages}") == "A - page 1 of 2"
    assert editor.export_pdf(tmp_path / "kw.pdf", g, keyword="TUBE") == 2
    # batch to new layouts: one undo step
    n_layouts = len(ws.layouts)
    steps = len(ctl._undo)
    created = editor.batch_to_layouts(g)
    assert len(created) == 3 and len(ws.layouts) == n_layouts + 3
    assert len(ctl._undo) == steps + 1
    for new, sample in zip(created, ws.members(g)):
        p = next(i for i in new.items if i.kind == "plot")
        assert p.sample_id == sample.id and p.overlays == [{"sample_id": sample.id, "pop_path": ""}]
        texts = [i for i in new.items if i.kind == "text"]
        assert all(t.sample_id == sample.id for t in texts)
        assert sample.name in new.name
    assert editor.display_text(next(i for i in created[1].items if i.kind == "text"), layout=created[1]) == ws.samples[1].name
    ctl.undo()
    assert len(ws.layouts) == n_layouts
    # the batch dialog collects the options; run_batch stores them in the layout
    dlg = BatchDialog(editor, lay, "images")
    dlg.by_keyword.setChecked(True)
    dlg.keyword_combo.setCurrentText("TUBE")
    dlg.header_edit.setText("{TUBE}")
    v = dlg.values()
    assert v["output"] == "images" and v["by"] == "keyword" and v["keyword"] == "TUBE"
    files = editor.run_batch(v, path=str(tmp_path / "dlg"))
    assert len(files) == 2
    assert lay.props["batch_by"] == "keyword" and lay.props["batch_keyword"] == "TUBE" and lay.props["batch_header"] == "{TUBE}"
    out = editor.run_batch({**v, "output": "pptx", "by": "group"}, path=str(tmp_path / "dlg.pptx"))
    assert out == 3


def test_text_tokens_per_page_and_formatting_survive(editor, ws, tmp_path):
    a, b = ws.samples[:2]
    editor.add_population(a, a.root.find("/Cells"))
    item = editor.add_text("{$FIL} {Median FSC-A}", edit=False)
    lay = editor.current_layout()
    t1 = editor.display_text(item, a, lay)
    t2 = editor.display_text(item, b, lay)
    assert t1 != t2 and t1.startswith("a.fcs ") and t2.startswith("b.fcs ")
    gi = editor.scene_item(item)
    assert gi.toPlainText() == t1  # the scene shows the values of the first plot's sample
    gi.start_editing()
    assert gi.toPlainText() == "{$FIL} {Median FSC-A}"  # editing shows the template
    gi.finish_editing()
    assert item.text == "{$FIL} {Median FSC-A}" and gi.toPlainText() == t1
    # pages differ per sample
    assert editor.page_pixmap(lay, a, 1.0).toImage() != editor.page_pixmap(lay, b, 1.0).toImage()
    # formatting through the editor (one undo step) ...
    editor.scene.clearSelection()
    gi.setSelected(True)
    steps = len(editor.controller._undo)
    editor.set_text_format(bold=True, italic=True, color="#d81b1b", align="center", font_family="Courier New")
    assert len(editor.controller._undo) == steps + 1
    assert item.props == {"bold": True, "italic": True, "color": "#d81b1b", "align": "center", "font_family": "Courier New"}
    assert gi.font().bold() and gi.font().italic() and gi.defaultTextColor().name() == "#d81b1b"
    # ... survives save / load
    editor.flush_notify()
    path = tmp_path / "ws.flowdesk"
    save_workspace(ws, path)
    ws2 = load_workspace(path)
    t = next(i for i in ws2.layouts[0].items if i.kind == "text")
    assert t.props == item.props and t.text == item.text
    # ... and shows in the SVG export
    svg = tmp_path / "page.svg"
    assert editor.export_svg(svg)
    text = svg.read_text()
    assert re.search(r'font-weight="(bold|[6-9]00)"', text)
    assert 'font-style="italic"' in text
    assert "#d81b1b" in text.lower()
    assert "a.fcs" in text
    # and in the PowerPoint export
    pptx = tmp_path / "fmt.pptx"
    editor.export_pptx(pptx)
    z = zipfile.ZipFile(pptx)
    root = ET.fromstring(z.read("ppt/slides/slide1.xml"))
    rpr = [r for r in root.iter(f"{{{NS['a']}}}rPr") if r.get("b") == "1" and r.get("i") == "1"]
    assert rpr and rpr[0].find("a:latin", NS).get("typeface") == "Courier New"
    assert root.find(".//a:pPr[@algn='ctr']", NS) is not None


# -------------------------------------------------------------- plot dialog
def test_plot_options_repoint_and_display(editor, ws):
    a, b, c = ws.samples
    item = editor.add_population(a, a.root.find("/Cells"))
    item.overlays.append({"sample_id": c.id, "pop_path": "/Cells"})
    editor.page_pixmap(editor.current_layout(), scale=1.0)
    geom = (item.x, item.y, item.w, item.h)
    renders = editor.cache.renders
    # unchanged dialog: only the overlay options are written (as before)
    dlg = PlotOptionsDialog(editor, item, editor)
    dlg.apply()
    assert set(item.graph.options) == {"show_legend", "histogram_offset"}
    dlg = PlotOptionsDialog(editor, item, editor)
    dlg.sample_combo.setCurrentIndex(dlg.sample_combo.findData(b.id))
    dlg.pop_combo.setCurrentIndex(dlg.pop_combo.findData(""))
    dlg.display_combos["smoothing"].setCurrentIndex(dlg.display_combos["smoothing"].findData(3))
    dlg.display_combos["dot_size"].setCurrentIndex(dlg.display_combos["dot_size"].findData(3))
    dlg.display_combos["contour_level"].setCurrentIndex(dlg.display_combos["contour_level"].findData(10.0))
    dlg.display_checks["show_gates"].setChecked(False)
    dlg.font_spins["caption_font_size"].setValue(9.0)
    assert editor.apply_plot_dialog(dlg, [])
    assert (item.sample_id, item.pop_path) == (b.id, "")
    assert (item.x, item.y, item.w) == geom[:3]
    assert item.h == pytest.approx(geom[3] + caption_height(item) - 34.0)  # the caption grew with its font
    assert item.overlays == [{"sample_id": c.id, "pop_path": "/Cells"}]
    opts = item.graph.options
    assert opts["smoothing"] == 3 and opts["dot_size"] == 3 and opts["contour_level"] == 10.0 and opts["show_gates"] is False
    assert item.props["caption_font_size"] == 9.0
    s, p = editor.resolve(item)
    assert s is b and p is b.root
    editor.page_pixmap(editor.current_layout(), scale=1.0)
    assert editor.cache.renders > renders  # re-rendered with the new population
    # display changes apply to other selected plots too; reset clears them
    other = editor.add_population(a, a.root)
    dlg = PlotOptionsDialog(editor, item, editor)
    dlg.display_combos["dot_size"].setCurrentIndex(dlg.display_combos["dot_size"].findData(4))
    editor.apply_plot_dialog(dlg, [other])
    assert other.graph.options.get("dot_size") == 4 and "smoothing" not in other.graph.options
    dlg = PlotOptionsDialog(editor, item, editor)
    dlg._reset_to_prefs()
    editor.apply_plot_dialog(dlg, [])
    assert not any(k in item.graph.options for k in ("smoothing", "dot_size", "contour_level", "show_gates"))
    # overlay chooser helpers: all samples of the group
    dlg = PlotOptionsDialog(editor, other, editor)
    dlg._add_group_overlays()
    assert {o["sample_id"] for o in dlg.overlays} == {b.id, c.id}


def test_overlays_handoff_and_multi_sample_drop(editor, ws):
    a, b, c = ws.samples
    item = editor.add_population(a, a.root.find("/Cells"), overlays=[{"sample_id": b.id, "pop_path": "/Cells"}, (c, c.root.find("/Cells")), {"sample_id": a.id, "pop_path": "/Cells"}])
    assert item.overlays == [{"sample_id": b.id, "pop_path": "/Cells"}, {"sample_id": c.id, "pop_path": "/Cells"}]
    editor.page_pixmap(editor.current_layout(), scale=1.0)
    legend = editor.render_legends[item.id]
    assert [e[0] for e in legend] == [f"{a.name} / Cells", f"{b.name} / Cells", f"{c.name} / Cells"]
    # a multi-sample list payload dropped on a plot: one overlay per sample
    plain = editor.add_population(a, a.root, QPointF(320, 60))
    inside = QPointF(plain.x + plain.w / 2, plain.y + 60)
    payload = [{"sample_id": a.id, "paths": [""]}, {"sample_id": b.id, "paths": [""]}, {"sample_id": c.id, "paths": [""]}]
    assert len(parse_population_entries(_mime(payload))) == 3
    assert editor.drop_mime(_mime(payload), inside) == [plain]
    assert plain.overlays == [{"sample_id": b.id, "pop_path": ""}, {"sample_id": c.id, "pop_path": ""}]
    # a group population dropped on a plot: every member
    third = editor.add_population(b, b.root.find("/Cells"), QPointF(60, 420))
    g = ws.groups[0]
    editor.drop_mime(_mime({"group_id": g.id, "sample_id": a.id, "paths": ["/Cells"]}), QPointF(third.x + 50, third.y + 60))
    assert {o["sample_id"] for o in third.overlays} == {a.id, c.id}
    # a list payload dropped on the page: one plot per sample
    n = len(editor.current_layout().items)
    created = editor.drop_mime(_mime(payload[:2]), QPointF(40, 700))
    assert [i.sample_id for i in created] == [a.id, b.id] and len(editor.current_layout().items) == n + 2
    # a statistic row dropped on the page -> live text; on a plot -> annotation
    stat = {"sample_id": a.id, "paths": [], "stats": [{"path": "/Cells", "statistic": "Freq. of Parent", "param": None, "percentile": None}]}
    made = editor.drop_mime(_mime(stat), QPointF(300, 450))
    assert len(made) == 1 and made[0].kind == "text" and made[0].pop_path == "/Cells"
    dec = editor.text_context(None).decimals  # the user's "freq_decimals" preference
    freq = format_value("Freq. of Parent", compute_statistic("Freq. of Parent", a.root.find("/Cells")), dec)
    assert editor.display_text(made[0]).endswith(freq)
    r_before = plot_rect(plain)
    editor.drop_mime(_mime(stat), inside)
    assert plain.props["annotations"] == [{"statistic": "Freq. of Parent", "pop_path": "/Cells"}]
    assert plot_rect(plain).height() == pytest.approx(r_before.height())  # the plot keeps its size
    s, p = editor.resolve(plain)
    assert editor.caption_lines(plain, s, p)[-1] == f"Cells Freq. of Parent: {freq}"


# ------------------------------------------------------- arrange / keyboard
def test_align_distribute_size_and_undo_key(editor, ws, app):
    a = ws.samples[0]
    ctl = editor.controller
    items = [editor.add_population(a, a.root, QPointF(x, y)) for x, y in ((40, 40), (300, 120), (120, 400))]
    items[1].w, items[2].w = 150, 100
    editor._sync_items(items)
    editor.flush_notify()
    editor.scene.clearSelection()
    QTest.keyClick(editor.view, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)  # select all
    assert len(editor.selected_layout_items()) == 3
    before = [(i.x, i.y) for i in items]
    steps = len(ctl._undo)
    assert editor.align_right()
    right = max(x + i.w for (x, _), i in zip(before, items))
    assert all(i.x + i.w == pytest.approx(right) for i in items)
    assert len(ctl._undo) == steps + 1
    editor.view.setFocus()
    QTest.keyClick(editor.view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    app.processEvents()
    assert [(i.x, i.y) for i in editor.current_layout().items] == before
    items = editor.current_layout().items  # undo rebuilt the scene from the model
    editor.select_all()
    assert editor.align_bottom()
    bottom = max(i.y + i.h for i in items)
    assert all(i.y + i.h == pytest.approx(bottom) for i in items)
    assert editor.align_hcenter()
    centres = {round(i.x + i.w / 2, 6) for i in items}
    assert len(centres) == 1
    # distribute horizontally: equal gaps, outer edges stay
    items[0].x, items[1].x, items[2].x = 40, 100, 400
    editor._sync_items(items)
    editor.flush_notify()
    left, right_edge = 40, 400 + items[2].w
    assert editor.distribute_horizontally()
    order = sorted(items, key=lambda i: i.x)
    gaps = [order[k + 1].x - (order[k].x + order[k].w) for k in range(2)]
    assert gaps[0] == pytest.approx(gaps[1])
    assert order[0].x == pytest.approx(left) and order[-1].x + order[-1].w == pytest.approx(right_edge)
    assert editor.distribute_vertically()
    order = sorted(items, key=lambda i: i.y)
    vg = [order[k + 1].y - (order[k].y + order[k].h) for k in range(2)]
    assert vg[0] == pytest.approx(vg[1])
    # size to match: the first selected (layout order) is the reference
    assert editor.same_width() and {i.w for i in items} == {items[0].w}
    items[2].h = 90
    assert editor.same_height() and {i.h for i in items} == {items[0].h}
    assert editor.align_vcenter()
    # Ctrl+Z three more times walks back through the steps
    n = len(ctl._undo)
    for _ in range(3):
        QTest.keyClick(editor.view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(ctl._undo) == n - 3


def test_copy_paste_between_layouts_and_close_key(editor, ws, app):
    a = ws.samples[0]
    first = editor.current_layout()
    p = editor.add_population(a, a.root.find("/Cells"))
    t = editor.add_text("Note {Sample}", edit=False)
    editor.set_text_format([t], bold=True)
    editor.scene.clearSelection()
    editor.select_all()
    QTest.keyClick(editor.view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    md = QApplication.clipboard().mimeData()
    assert md.hasImage()  # pastes into PowerPoint / Word as a picture
    second = editor.new_layout("Second")
    assert editor.current_layout() is second and not second.items
    steps = len(editor.controller._undo)
    QTest.keyClick(editor.view, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert [i.kind for i in second.items] == ["plot", "text"]
    assert len(editor.controller._undo) == steps + 1
    pasted = second.items
    assert {i.id for i in pasted}.isdisjoint({p.id, t.id})
    assert pasted[0].pop_path == "/Cells" and pasted[1].props == {"bold": True}
    assert (pasted[0].x, pasted[0].y) == (p.x, p.y)  # empty layout: same place
    assert len(editor.selected_layout_items()) == 2
    # pasting again into the same layout shifts the copies
    again = editor.paste_items()
    assert again[0].x == pasted[0].x + 16
    # cut removes, select all + delete
    editor.scene.clearSelection()
    editor.scene_item(again[1]).setSelected(True)
    assert editor.cut_selected() == [again[1]] and again[1] not in second.items
    editor.set_current_layout(first)
    assert len(first.items) == 2  # the source is unchanged
    # copy with nothing selected copies the page as an image
    editor.scene.clearSelection()
    img = editor.copy_selected()
    assert isinstance(img, QImage) and not img.isNull()
    # Ctrl+W closes (hides) the editor
    assert editor.isVisible()
    editor.activateWindow()
    QTest.keyClick(editor.view, Qt.Key.Key_W, Qt.KeyboardModifier.ControlModifier)
    app.processEvents()
    assert not editor.isVisible()


def test_menus_ribbon_and_shortcut_scope(editor):
    titles = [m.title().replace("&", "") for m in editor.menuBar().findChildren(type(editor.arrange_menu)) if m.parent() is editor.menuBar()]
    assert {"File", "Edit", "Arrange", "Text"} <= set(titles)
    subs = {a.text(): a.menu() for a in editor.arrange_menu.actions() if a.menu() is not None}
    assert {"Align", "Distribute", "Size"} <= set(subs)
    assert [a.text() for a in subs["Distribute"].actions()] == ["Distribute Horizontally", "Distribute Vertically"]
    assert {"Same Width", "Same Height", "Same Size"} == {a.text() for a in subs["Size"].actions()}
    tabs = [editor.ribbon.tabs.tabText(i) for i in range(editor.ribbon.tabs.count())]
    assert "Arrange" in tabs
    bands = [b.title for b in editor.ribbon.pages[tabs.index("Arrange")].bands]
    assert bands[:3] == ["Align", "Distribute", "Size"]
    for name in ("undo", "redo", "copy", "paste", "select_all", "close"):
        act = editor.acts[name]
        assert act in editor.actions() and not act.shortcut().isEmpty(), name
        assert act.shortcutContext() == Qt.ShortcutContext.WindowShortcut


# ---------------------------------------------------- page setup / defaults
def test_page_setup_custom_margin_and_defaults(editor, ws, isolated_defaults):
    ctl = editor.controller
    lay = editor.current_layout()
    dlg = PageSetupDialog(lay)
    dlg.page_combo.setCurrentText("Custom")
    dlg.unit_combo.setCurrentText("pt")
    dlg.width_spin.setValue(500)
    dlg.height_spin.setValue(300)
    dlg.margin_spin.setValue(18)
    dlg.landscape.setChecked(True)
    v = dlg.result_values()
    assert v["page"] == "Custom" and v["custom_size"] == (500.0, 300.0) and v["margin"] == 18.0 and v["landscape"]
    steps = len(ctl._undo)
    editor.set_page(v["page"], v["landscape"], margin=v["margin"], custom_size=v["custom_size"])
    assert lay.page_size == (300.0, 500.0) and lay.margin == 18.0 and len(ctl._undo) == steps + 1
    pos = editor._free_position(100, 100)
    assert (pos.x(), pos.y()) == (18.0, 18.0)
    ctl.undo()
    lay = editor.current_layout()
    assert lay.page == "A4" and lay.margin == 36.0
    # defaults for new layouts
    editor.set_page("Letter", True, margin=24)
    editor.set_grid(True)
    a = ws.samples[0]
    plot = editor.add_population(a, a.root)
    plot.w = 160
    editor.scene.clearSelection()
    editor.scene_item(plot).setSelected(True)
    editor.save_layout_defaults()
    assert isolated_defaults["page"] == "Letter" and isolated_defaults["landscape"] is True and isolated_defaults["plot_size"] == 160
    new = editor.new_layout("Fresh")
    assert new.page == "Letter" and new.landscape and new.margin == 24
    item = editor.add_population(a, a.root)
    assert item.w == 160


def test_registered_item_kind(editor, ws, tmp_path):
    calls = []

    def painter(ed, qp, item, sample, scale):
        calls.append((item.kind, sample.name if sample is not None else None))
        qp.fillRect(0, 0, int(item.w), int(item.h), Qt.GlobalColor.darkGreen)

    LayoutEditorWindow.register_item_kind("swatch", painter=painter)
    try:
        lay = editor.current_layout()
        it = LayoutItem(kind="swatch", x=50, y=50, w=40, h=40, props={"color": "green"})
        lay.items.append(it)
        editor.refresh()
        gi = editor.scene_item(it)
        assert isinstance(gi, PlotGraphicsItem) and editor.plot_item_at(QPointF(60, 60)) is None
        img = editor.page_pixmap(lay, ws.samples[1], 1.0).toImage()
        assert ("swatch", ws.samples[1].name) in calls
        assert img.pixelColor(70, 70).green() > img.pixelColor(70, 70).red()
        assert editor.export_pptx(tmp_path / "k.pptx") == 1
        with pytest.raises(ValueError):
            LayoutEditorWindow.register_item_kind("plot", painter=painter)
    finally:
        LayoutEditorWindow._item_kinds.pop("swatch", None)


def test_insert_tokens_and_annotations(editor, ws):
    a = ws.samples[0]
    plot = editor.add_population(a, a.root.find("/Cells"))
    t = editor.add_text("Donor", edit=False)
    editor.scene.clearSelection()
    editor.scene_item(t).setSelected(True)
    editor.insert_token("{$FIL}")
    assert t.text == "Donor {$FIL}" and editor.scene_item(t).toPlainText() == "Donor a.fcs"
    # inline editing inserts at the cursor
    gi = editor.scene_item(t)
    gi.start_editing()
    cur = gi.textCursor()
    cur.movePosition(cur.MoveOperation.End)
    gi.setTextCursor(cur)
    editor.insert_token(" {Sample}")
    gi.finish_editing()
    assert t.text.endswith("{$FIL} {Sample}")
    # annotations on the selected plot keep the plot area
    editor.scene.clearSelection()
    editor.scene_item(plot).setSelected(True)
    r = plot_rect(plot)
    assert editor.add_annotation([plot], {"keyword": "TUBE"}) == 1
    assert plot_rect(plot).height() == pytest.approx(r.height())
    s, p = editor.resolve(plot)
    assert editor.caption_lines(plot, s, p)[-1] == "TUBE: B"
    assert editor.caption_lines(plot, ws.samples[1], ws.samples[1].root.find("/Cells"))[-1] == "TUBE: A"
    assert isinstance(editor.scene_item(t), TextGraphicsItem)


def test_toggle_format_keys_open_graph_and_undo_actions(editor, ws, app):
    a = ws.samples[0]
    plot = editor.add_population(a, a.root.find("/Cells"))
    t = editor.add_text("Title", edit=False)
    editor.scene.clearSelection()
    editor.scene_item(t).setSelected(True)
    editor.toggle_text_format("bold")
    assert t.props.get("bold") is True and editor.acts["bold"].isChecked()
    editor.acts["bold"].trigger()  # Ctrl+B again: off
    assert "bold" not in t.props and not editor.acts["bold"].isChecked()
    editor.set_text_format([t], align="right", color="#000000")
    assert t.props == {"align": "right"}  # black is the default, not stored
    # open the plot's population in a graph window (the main window's hook)
    opened = []
    editor.controller.open_graph = lambda s, p: opened.append((s, p)) or "graph window"
    editor.scene.clearSelection()
    editor.scene_item(plot).setSelected(True)
    assert editor.open_graph_selected() == "graph window" and opened == [(a, a.root.find("/Cells"))]
    # Undo / Redo actions of the editor window walk the controller's history
    n = len(editor.controller._undo)
    editor.acts["undo"].trigger()
    assert len(editor.controller._undo) == n - 1 and editor.controller.can_redo
    editor._update_undo_actions()
    editor.acts["redo"].trigger()
    assert len(editor.controller._undo) == n
