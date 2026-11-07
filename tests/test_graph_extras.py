"""Graph window extras: export, gate tint / lock, toggles, legend, stats, keys."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402

from flowdesk.core.gates import PolygonGate, RectangleGate, gate_from_dict  # noqa: E402
from flowdesk.core.model import Population  # noqa: E402
from flowdesk.ui import plot_widget as plot_widget_module  # noqa: E402

from .conftest import SYNTH_NAMES, SYNTH_STAINS, make_synthetic_events, write_fcs  # noqa: E402

FOLDER = Path("/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel")
REAL = sorted(FOLDER.glob("*.fcs")) if FOLDER.exists() else []


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


@pytest.fixture
def files(tmp_path):
    if len(REAL) >= 2:
        return REAL[:2]
    out = []
    for i in range(2):
        ev = make_synthetic_events(4000, seed=i + 1)["events"]
        out.append(write_fcs(tmp_path / f"s{i}.fcs", ev, SYNTH_NAMES, SYNTH_STAINS, extra_keywords={"$CYT": "TestCyt"}))
    return out


@pytest.fixture
def window(app, monkeypatch, files):
    from flowdesk.ui.main_window import MainWindow

    class FakeNameDialog:
        def __init__(self, default_name, group=None, apply_default=False, parent=None):
            self.default_name = default_name

        def exec(self):
            return True

        def values(self):
            return (self.default_name, False)

    monkeypatch.setattr(plot_widget_module, "GateNameDialog", FakeNameDialog)
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    win.controller.add_samples(files)
    app.processEvents()
    yield win
    for gw in list(win.graph_windows.values()):
        gw.close()
    win.controller.workspace.modified = False
    win.hide()


def _graph(app, window, with_gate=True):
    c = window.controller
    s0 = c.workspace.samples[0]
    gw = window.open_graph(s0, s0.root)
    gw.resize(620, 660)
    app.processEvents()
    x, y = s0.default_axes()
    gw.plot.set_axes(x=x, y=y)
    pop = None
    if with_gate:
        d = s0.data(x)
        e = s0.data(y)
        lo = [float(np.percentile(d, 30)), float(np.percentile(e, 30))]
        hi = [float(np.percentile(d, 80)), float(np.percentile(e, 80))]
        pop = c.add_population(s0.root, Population("Cells", RectangleGate([x, y], lo, hi)))
    gw.plot.repaint()
    app.processEvents()
    return gw, s0, pop


def test_save_plot_as_png_svg_pdf(app, window, tmp_path):
    gw, _, _ = _graph(app, window)
    for suffix, magic in ((".png", b"\x89PNG"), (".svg", b"<?xml"), (".pdf", b"%PDF")):
        out = gw.save_plot(tmp_path / f"plot{suffix}")
        assert out.exists() and out.stat().st_size > 500
        assert out.read_bytes()[:5].startswith(magic)
    from PySide6.QtGui import QImage

    img = QImage(str(tmp_path / "plot.png"))
    assert img.width() >= 2 * gw.plot.width() - 2  # 2x resolution
    svg = (tmp_path / "plot.svg").read_text(errors="ignore")
    assert "Cells" in svg  # gate labels are vector text


def test_print_to_pdf_printer(app, window, tmp_path):
    from PySide6.QtPrintSupport import QPrinter

    gw, _, _ = _graph(app, window)
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(str(tmp_path / "printed.pdf"))
    gw.print_plot(printer)
    assert (tmp_path / "printed.pdf").stat().st_size > 500


def test_tint_and_lock_persist_through_dict():
    g = PolygonGate(["a", "b"], [(0, 0), (1, 0), (1, 1)])
    g.tint = 0.3
    g.locked = True
    d = g.to_dict()
    g2 = gate_from_dict(d)
    assert g2.tint == pytest.approx(0.3) and g2.locked is True
    plain = gate_from_dict(RectangleGate(["a"], [0], [1]).to_dict())
    assert plain.tint == 0.0 and plain.locked is False
    assert "tint" not in RectangleGate(["a"], [0], [1]).to_dict()


def test_tint_changes_rendering(app, window):
    gw, s0, pop = _graph(app, window)
    from flowdesk.ui.renderer import render_plot

    opts = gw.plot.options()
    before = render_plot(s0, s0.root, gw.plot.graph, 300, 300, opts).image
    gw._set_gate_tint(pop, 0.3)
    pop = s0.root.child_by_name("Cells")
    assert pop.gate.tint == pytest.approx(0.3)
    after = render_plot(s0, s0.root, gw.plot.graph, 300, 300, opts).image
    assert before != after


def test_show_gates_and_labels_toggles(app, window):
    gw, _, _ = _graph(app, window)
    assert any(s.population.name == "Cells" for s in gw.plot.result().gates)
    gw.option_actions["show_gate_labels"].setChecked(False)
    gw.plot.repaint()
    app.processEvents()
    assert all(not s.label for s in gw.plot.result().gates)
    gw.option_actions["show_gates"].setChecked(False)
    gw.plot.repaint()
    app.processEvents()
    assert gw.plot.result().gates == []
    gw.option_actions["show_gates"].setChecked(True)
    gw.plot.repaint()
    app.processEvents()
    assert gw.plot.result().gates


def test_locked_gate_cannot_be_dragged(app, window):
    gw, s0, pop = _graph(app, window)
    gw._set_gate_locked(pop, True)
    pop = s0.root.child_by_name("Cells")
    gw.plot.repaint()
    app.processEvents()
    before = pop.gate.to_dict()
    shape = next(s for s in gw.plot.result().gates if s.population is pop)
    a = shape.bbox.center()
    b = QPointF(a.x() + 30, a.y() + 20)
    btn = Qt.MouseButton.LeftButton
    for typ, pos in ((QEvent.Type.MouseButtonPress, a), (QEvent.Type.MouseMove, b), (QEvent.Type.MouseButtonRelease, b)):
        app.sendEvent(gw.plot, QMouseEvent(typ, pos, gw.plot.mapToGlobal(pos.toPoint()), btn, btn, Qt.KeyboardModifier.NoModifier))
    app.processEvents()
    assert s0.root.child_by_name("Cells").gate.to_dict() == before


def test_overlay_legend_and_histogram_offset(app, window):
    from flowdesk.ui.renderer import render_plot

    gw, s0, _ = _graph(app, window, with_gate=False)
    s1 = window.controller.workspace.samples[1]
    opts = gw.plot.options()
    plain = render_plot(s0, s0.root, gw.plot.graph, 320, 320, opts)
    assert plain.legend_rect is None
    res = render_plot(s0, s0.root, gw.plot.graph, 320, 320, opts, overlays=[(s1, s1.root)])
    assert res.legend_rect is not None and res.plot_rect.contains(res.legend_rect)
    assert len(res.legend_entries) == 2 and s1.name in res.legend_entries[1][0]
    assert plain.image != res.image
    gw.plot.set_axes(plot_type="histogram")
    opts = gw.plot.options()
    flat = render_plot(s0, s0.root, gw.plot.graph, 320, 320, opts, overlays=[(s1, s1.root)])
    opts.histogram_offset = True
    staggered = render_plot(s0, s0.root, gw.plot.graph, 320, 320, opts, overlays=[(s1, s1.root)])
    assert staggered.legend_rect is not None
    assert getattr(staggered, "extra_offset", 0) > 0
    assert flat.image != staggered.image


def test_stats_panel_median_mean_and_copy(app, window):
    gw, s0, pop = _graph(app, window)
    gw.stats_action.setChecked(True)
    app.processEvents()
    t = gw.stats_table
    headers = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
    x, y = gw.plot.graph.x, gw.plot.graph.y
    assert headers[:4] == ["Population", "Events", "% of Parent", "% of Total"]
    assert any(h.startswith(f"Median {x}") for h in headers) and any(h.startswith(f"Mean {y}") for h in headers)
    assert t.rowCount() == 2
    col = next(j for j, h in enumerate(headers) if h.startswith(f"Median {x}"))
    expected = float(np.median(s0.data(x)[pop.mask()]))
    from flowdesk.core.stats import format_value

    assert t.item(1, col).text() == format_value("Median", expected)
    gw.copy_stats()
    from PySide6.QtWidgets import QApplication

    text = QApplication.clipboard().text()
    lines = text.strip().split("\n")
    assert lines[0].split("\t") == headers and len(lines) == 3


def test_keyboard_zoom_and_label_rename(app, window, monkeypatch):
    gw, s0, pop = _graph(app, window)
    plot = gw.plot

    def key(k, text=""):
        app.sendEvent(plot, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier, text))
        app.processEvents()

    key(Qt.Key.Key_Plus, "+")
    x0, x1, y0, y1 = plot.view
    assert x1 - x0 < 1.0 and abs((x0 + x1) / 2 - 0.5) < 1e-6
    key(Qt.Key.Key_Minus, "-")
    key(Qt.Key.Key_Minus, "-")
    assert plot.view == (0.0, 1.0, 0.0, 1.0)
    key(Qt.Key.Key_Plus, "+")
    key(Qt.Key.Key_0, "0")
    assert plot.view == (0.0, 1.0, 0.0, 1.0)
    plot.repaint()
    app.processEvents()

    renamed = []
    monkeypatch.setattr(gw, "_rename", lambda p: renamed.append(p))
    plot.renameRequested.disconnect()
    plot.renameRequested.connect(gw._rename)
    shape = next(s for s in plot.result().gates if s.population is pop)
    pos = shape.label_rect.center()
    btn = Qt.MouseButton.LeftButton
    app.sendEvent(plot, QMouseEvent(QEvent.Type.MouseButtonDblClick, pos, plot.mapToGlobal(pos.toPoint()), btn, btn, Qt.KeyboardModifier.NoModifier))
    assert renamed and renamed[0] is pop


def test_platform_modifier_helpers():
    from flowdesk.ui import style

    assert style.modifier_text("darwin") == "⌘"
    assert style.modifier_text("win32") == "Ctrl"
    assert style.modifier_text("linux") == "Ctrl"
    assert style.shortcut_text("Z", "darwin") == "⌘Z"
    assert style.shortcut_text("S", "win32", shift=True) == "Ctrl+Shift+S"
    assert style.modifier_html("darwin") == "&#8984;" and style.modifier_html("linux") == "Ctrl"
    assert style.base_font_px("darwin") == 12 and style.base_font_px("win32") == 11 and style.base_font_px("linux") == 11


def test_plot_fonts_are_pixel_sized(app):
    from flowdesk.ui.renderer import RenderOptions, plot_font

    f = plot_font(11, RenderOptions(font_dpi=72.0))
    assert f.pixelSize() == 11
    assert plot_font(9, RenderOptions()).pixelSize() == 12  # 96 dpi basis (layout editor)
