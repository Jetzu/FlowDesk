import json
import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QMimeData, QPointF
from PySide6.QtWidgets import QApplication

from flowdesk.core.gates import RectangleGate
from flowdesk.core.layouts import Layout, LayoutItem
from flowdesk.core.model import Population, Workspace
from flowdesk.core.serialization import load_workspace, save_workspace
from flowdesk.ui.controller import POPULATION_MIME, WorkspaceController
from flowdesk.ui.layout_editor import (
    LayoutEditorWindow,
    PlotGraphicsItem,
    TextGraphicsItem,
    parse_population_mime,
)

DIR = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel")
FILES = [DIR / "_F1donor 6T cell pehnotype2025-08-18.0003.fcs", DIR / "_F3donor 7T cell pehnotype2025-08-18.0001.fcs"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ws():
    if not all(f.exists() for f in FILES):
        pytest.skip("real FCS files not available")
    w = Workspace()
    for f in FILES:
        w.add_sample(f)
    gate = RectangleGate(["FSC-A", "SSC-A"], [50000, 10000], [150000, 100000])
    w.apply_population_to_group(w.groups[0], "", Population("Lymphocytes", gate))
    return w


@pytest.fixture
def editor(app, ws):
    c = WorkspaceController(ws)
    win = LayoutEditorWindow(c)
    win.show()
    app.processEvents()
    yield win
    win.flush_notify()
    win.hide()
    win.deleteLater()


def mime(sample_id, paths):
    m = QMimeData()
    m.setData(POPULATION_MIME, json.dumps({"sample_id": sample_id, "paths": paths}).encode())
    return m


def plot_items(win):
    return [i for i in win.scene.items() if isinstance(i, PlotGraphicsItem)]


def test_opens_with_default_layout(editor, ws):
    assert editor.windowTitle() == "Layout Editor"
    assert len(ws.layouts) == 1 and isinstance(ws.layouts[0], Layout)
    assert editor.current_layout() is ws.layouts[0]
    assert editor.layout_list.count() == 1
    editor.close()
    assert not editor.isVisible()  # closing hides


def test_add_population_creates_item(editor, ws):
    s = ws.samples[0]
    pop = s.root.find("/Lymphocytes")
    item = editor.add_population(s, pop)
    assert isinstance(item, LayoutItem)
    assert item.kind == "plot" and item.sample_id == s.id and item.pop_path == "/Lymphocytes"
    assert item.graph.x and item.graph.y
    assert editor.current_layout().items == [item]
    gi = editor.scene_item(item)
    assert isinstance(gi, PlotGraphicsItem)
    # rendering is cached: a second render of the unchanged item is free
    editor.page_pixmap(editor.current_layout())
    n = editor.cache.renders
    assert n >= 1
    editor.page_pixmap(editor.current_layout())
    assert editor.cache.renders == n
    # a second add flows to a free spot instead of overlapping
    item2 = editor.add_population(s, s.root)
    assert (item2.x, item2.y) != (item.x, item.y)


def test_arrange_grid(editor, ws):
    s = ws.samples[0]
    a = editor.add_population(s, s.root.find("/Lymphocytes"), QPointF(300, 400))
    b = editor.add_population(s, s.root, QPointF(40, 40))
    editor.arrange_grid(2)
    # reading order: b (top) first, then a
    assert b.y == a.y
    assert b.x < a.x
    assert a.w == b.w and a.h == b.h
    assert a.x >= b.x + b.w  # side by side, no overlap
    assert editor.scene_item(a).pos().x() == a.x


def test_exports(editor, ws, tmp_path):
    s = ws.samples[0]
    editor.add_population(s, s.root.find("/Lymphocytes"))
    editor.add_text("Hello layout", edit=False)
    png, pdf, svg = tmp_path / "l.png", tmp_path / "l.pdf", tmp_path / "l.svg"
    assert editor.export_png(png)
    assert editor.export_pdf(pdf) == 1
    assert editor.export_svg(svg)
    for f in (png, pdf, svg):
        assert f.exists() and f.stat().st_size > 1000
    assert pdf.read_bytes().startswith(b"%PDF")
    pm = editor.page_pixmap(editor.current_layout(), scale=1.0)
    w, h = editor.current_layout().page_size
    assert (pm.width(), pm.height()) == (round(w), round(h))


def test_batch_pdf(editor, ws, tmp_path):
    s = ws.samples[0]
    editor.add_population(s, s.root.find("/Lymphocytes"))
    missing = editor.add_population(s, s.root)
    missing.pop_path = "/Nope"
    single, batch = tmp_path / "one.pdf", tmp_path / "batch.pdf"
    assert editor.export_pdf(single) == 1
    assert editor.export_pdf(batch, batch_group=ws.groups[0]) == 2
    assert batch.stat().st_size > single.stat().st_size
    data = batch.read_bytes()
    assert len(re.findall(rb"/Type\s*/Page[^s]", data)) == 2
    assert len(re.findall(rb"/Type\s*/Page[^s]", single.read_bytes())) == 1
    # per-sample pages differ (different data, different header)
    lay = editor.current_layout()
    img1 = editor.page_pixmap(lay, ws.samples[0], 1.0).toImage()
    img2 = editor.page_pixmap(lay, ws.samples[1], 1.0).toImage()
    assert img1 != img2
    # the re-pointed population of the second sample is used
    s2, p2 = editor.resolve(lay.items[0], ws.samples[1])
    assert s2 is ws.samples[1] and p2 is ws.samples[1].root.find("/Lymphocytes")
    assert editor.resolve(missing, ws.samples[1])[1] is None


def test_drop_creates_items(editor, ws):
    s = ws.samples[0]
    created = editor.drop_mime(mime(s.id, ["/Lymphocytes", ""]), QPointF(50, 50))
    assert len(created) == 2
    assert created[0].pop_path == "/Lymphocytes" and created[1].pop_path == ""
    assert created[1].x > created[0].x  # flowing to the right
    assert len(plot_items(editor)) == 2
    assert parse_population_mime(mime(s.id, ["/A"])) == (s.id, ["/A"])
    assert editor.drop_mime(QMimeData(), QPointF(0, 0)) == []


def test_drop_on_item_adds_overlay(editor, ws):
    s1, s2 = ws.samples
    item = editor.add_population(s1, s1.root.find("/Lymphocytes"), QPointF(60, 60))
    inside = QPointF(item.x + item.w / 2, item.y + 60)
    result = editor.drop_mime(mime(s2.id, ["/Lymphocytes"]), inside)
    assert result == [item]
    assert item.overlays == [{"sample_id": s2.id, "pop_path": "/Lymphocytes"}]
    assert len(editor.current_layout().items) == 1
    overlays = editor.resolve_overlays(item)
    assert overlays[0][0] is s2
    assert not editor.page_pixmap(editor.current_layout(), scale=1.0).isNull()


def test_text_item_and_editing(editor, ws):
    item = editor.add_text("Title", edit=False)
    gi = editor.scene_item(item)
    assert isinstance(gi, TextGraphicsItem)
    gi.start_editing()
    gi.setPlainText("Changed")
    gi.finish_editing()
    assert item.text == "Changed"
    editor.set_font_size([item], 24)
    assert item.font_size == 24


def test_geometry_writes_back(editor, ws, app):
    s = ws.samples[0]
    item = editor.add_population(s, s.root)
    gi = editor.scene_item(item)
    gi.setPos(120, 130)
    assert (item.x, item.y) == (120, 130)
    gi.resize_to("br", gi.box().translated(item.x, item.y), QPointF(50, 20), keep_aspect=False)
    assert item.w == 250
    editor.flush_notify()
    assert ws.modified


def test_layout_survives_save_load(editor, ws, tmp_path):
    s1, s2 = ws.samples
    item = editor.add_population(s1, s1.root.find("/Lymphocytes"))
    item.overlays.append({"sample_id": s2.id, "pop_path": "/Lymphocytes"})
    editor.add_text("Note", edit=False)
    editor.set_page("Letter", True)
    editor.flush_notify()
    path = tmp_path / "ws.flowdesk"
    save_workspace(ws, path)
    ws2 = load_workspace(path)
    assert len(ws2.layouts) == 1
    lay = ws2.layouts[0]
    assert lay.page == "Letter" and lay.landscape
    kinds = [i.kind for i in lay.items]
    assert kinds == ["plot", "text"]
    p = lay.items[0]
    assert p.pop_path == "/Lymphocytes" and p.sample_id == s1.id and p.overlays == item.overlays
    assert p.graph.to_dict() == item.graph.to_dict()
    assert lay.items[1].text == "Note"
    # the editor follows a workspace replacement
    editor.controller.set_workspace(ws2)
    assert editor.current_layout() is lay
    assert len(plot_items(editor)) == 1


def test_batch_pptx_and_images_real_samples(editor, ws, tmp_path):
    """One slide / one image file per sample of the group (real sample names
    with spaces and dots make safe file names)."""
    import zipfile

    s = ws.samples[0]
    editor.add_population(s, s.root.find("/Lymphocytes"))
    editor.add_text("{$FIL}", edit=False)
    g = ws.groups[0]
    out = tmp_path / "batch.pptx"
    assert editor.export_pptx(out, batch_group=g) == 2
    names = zipfile.ZipFile(out).namelist()
    assert sum(1 for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")) == 2
    files = editor.export_images(tmp_path / "imgs", g, "tif", dpi=150)
    assert len(files) == 2 and all(Path(f).exists() and Path(f).suffix == ".tif" for f in files)
    assert {Path(f).stem for f in files} == {f"Layout - {m.name}" for m in ws.members(g)}
