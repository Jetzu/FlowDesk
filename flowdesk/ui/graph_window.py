"""The graph window: one population, gate tools, axis selectors, ancestry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QMarginsF, QPointF, QRect, QRectF, QSize, QSizeF, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFontMetrics, QIcon, QKeySequence, QPageSize, QPainter, QPdfWriter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyleOptionToolButton,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.model import PLOT_TYPES, GraphSettings, Group, Population, Sample
from ..core.stats import format_value
from .dialogs import OverlayDialog, ScaleDialog, confirm
from .icons import icon
from .plot_widget import PlotWidget
from .renderer import RenderOptions, render_plot
from .style import PLOT_FONT_DPI, shortcut_text

PLOT_FILE_FILTERS = "PNG image (*.png);;SVG vector graphic (*.svg);;PDF document (*.pdf)"
TINT_CHOICES = [("None", 0.0), ("Light", 0.15), ("Medium", 0.3)]

TOOL_ACTIONS = [
    ("pointer", "Select", "pointer", "Select and edit gates (double-click a gate to open it)"),
    ("rect", "Rectangle", "gate-rectangle", "Rectangle gate: drag on the plot"),
    ("polygon", "Polygon", "gate-polygon", "Polygon gate: click vertices, double-click or click the first vertex to close"),
    ("ellipse", "Ellipse", "gate-ellipse", "Ellipse gate: drag on the plot"),
    ("quadrant", "Quadrant", "gate-quadrant", "Quadrant gates: click the centre"),
    ("range", "Range", "gate-range", "Range gate on a histogram: drag horizontally"),
    ("bisector", "Bisector", "gate-bisector", "Bisector: click to split into two populations"),
    ("freehand", "Freehand", "gate-freehand", "Freehand gate: draw a closed shape"),
    ("zoom", "Zoom", "zoom", "Zoom: drag a region, double-click to reset"),
]


class AxisButton(QToolButton):
    """A parameter selector drawn along an axis (vertical for the y-axis)."""

    def __init__(self, vertical: bool = False, parent=None):
        super().__init__(parent)
        self.vertical = vertical
        self.setObjectName("axisButton")
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if vertical:
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self.font())
        w = fm.horizontalAdvance(self.text()) + 26
        h = fm.height() + 10
        return QSize(h, w) if self.vertical else QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event):
        if not self.vertical:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = self.rect().adjusted(0, 0, -1, -1)
        hover = self.underMouse()
        painter.setPen(QColor("#c9cdd3"))
        painter.setBrush(QColor("#e8f0fd" if hover else "#fafbfc"))
        painter.drawRoundedRect(r, 3, 3)
        painter.setPen(QColor("#1f2328"))
        painter.translate(r.center())
        painter.rotate(-90)
        text = self.text() + "  ▾"
        painter.drawText(QRectF(-r.height() / 2, -r.width() / 2, r.height(), r.width()), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()


class AncestryBar(QWidget):
    """Thumbnails of the parent chain, like the ancestry strip cytometrists expect."""

    navigate = Signal(object)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("ancestryBar")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 3, 6, 3)
        self._layout.setSpacing(4)
        self.setFixedHeight(96)

    def rebuild(self, sample: Sample, population: Population):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        chain = population.ancestors() + [population]
        for i, node in enumerate(chain):
            child = chain[i + 1] if i + 1 < len(chain) else None
            btn = QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setIconSize(QSize(56, 56))
            btn.setText(_elide(node.display_name, 14))
            btn.setToolTip(node.display_name)
            btn.setIcon(QIcon(self._thumbnail(sample, node, child)))
            btn.setCheckable(True)
            btn.setChecked(node is population)
            btn.setAutoRaise(True)
            btn.clicked.connect(lambda _=False, n=node: self.navigate.emit(n))
            self._layout.addWidget(btn)
            if child is not None:
                arrow = QLabel("▶")
                arrow.setStyleSheet("color:#8a8f98; font-size:10px;")
                self._layout.addWidget(arrow)
        self._layout.addStretch(1)

    def _thumbnail(self, sample: Sample, node: Population, child: Population | None) -> QPixmap:
        graph = GraphSettings.from_dict(node.graph.to_dict())
        if child is not None and child.gate is not None and len(child.gate.dims) == 2:
            graph.x, graph.y = child.gate.dims[0], child.gate.dims[1]
            if graph.plot_type == "histogram":
                graph.plot_type = "pseudocolor"
        elif child is not None and child.gate is not None and len(child.gate.dims) == 1:
            graph.x, graph.y, graph.plot_type = child.gate.dims[0], None, "histogram"
        if not graph.x:
            graph.x, graph.y = sample.default_axes()
        opts = RenderOptions.from_prefs(self.controller.preferences, graph, show_axes=False, show_gate_names=False, show_gate_freqs=False, dots_to_draw=6000, label_font_size=7, number_font_size=6, font_dpi=PLOT_FONT_DPI)
        opts.plot_type = graph.plot_type if graph.plot_type != "histogram" or graph.y is None else "pseudocolor"
        res = render_plot(sample, node, graph, 56, 56, opts, selected=child, dpr=2.0)
        return QPixmap.fromImage(res.image)


def _elide(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


class GraphWindow(QMainWindow):
    closed = Signal(object)

    def __init__(self, controller, sample: Sample, population: Population, group: Group | None = None, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.sample = sample
        self.population = population
        self.group = group
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(640, 700)

        self.plot = PlotWidget(controller, sample, population)
        self.plot.group = group
        self.plot.selectionChanged.connect(self._on_selection)
        self.plot.populationActivated.connect(self.open_population)
        self.plot.graphChanged.connect(self._sync_axes_ui)
        self.plot.statusText.connect(lambda t: self.statusBar().showMessage(t) if t else self.statusBar().clearMessage())
        self.plot.toolDone.connect(lambda: self._select_tool("pointer"))
        self.plot.contextMenuRequested.connect(self._context_menu)
        self.plot.zoomChanged.connect(self._sync_axes_ui)
        self.plot.renameRequested.connect(self._rename)

        self._build_toolbar()
        self._build_central()
        self._build_stats_dock()
        self.statusBar()
        self._update_title()
        self._sync_axes_ui()
        self.ancestry.rebuild(sample, population)
        self._refresh_stats()
        controller.gatingChanged.connect(self._on_gating_changed)
        controller.samplesChanged.connect(self._on_samples_changed)
        controller.workspaceReplaced.connect(self.close)

    # ------------------------------------------------------------- building
    def _build_toolbar(self):
        tb = QToolBar("Tools")
        tb.setIconSize(QSize(22, 22))
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(tb)
        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_actions: dict[str, QAction] = {}
        for key, label, icon_name, tip in TOOL_ACTIONS:
            act = QAction(icon(icon_name), label, self)
            act.setCheckable(True)
            act.setToolTip(f"{label} – {tip}")
            act.setStatusTip(tip)
            act.triggered.connect(lambda checked=False, k=key: self.plot.set_tool(k))
            self.tool_group.addAction(act)
            tb.addAction(act)
            self.tool_actions[key] = act
            if key in ("pointer", "quadrant", "freehand"):
                pass
        self.tool_actions["pointer"].setChecked(True)
        tb.addSeparator()
        self.plot_type = QComboBox()
        for key, label in PLOT_TYPES:
            self.plot_type.addItem(icon(key if key != "pseudocolor" else "pseudocolor"), label, key)
        self.plot_type.currentIndexChanged.connect(self._plot_type_changed)
        tb.addWidget(self.plot_type)
        tb.addSeparator()
        options_btn = QToolButton()
        options_btn.setText("Options")
        options_btn.setIcon(icon("options"))
        options_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        options_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        options_btn.setMenu(self._options_menu())
        tb.addWidget(options_btn)
        overlay = QAction(icon("overlay"), "Overlay…", self)
        overlay.setToolTip("Overlay other populations on this plot")
        overlay.triggered.connect(self._choose_overlays)
        tb.addAction(overlay)
        add_layout = QAction(icon("layout-editor"), "Add to Layout", self)
        add_layout.setToolTip("Add this plot to the Layout Editor")
        add_layout.triggered.connect(self._add_to_layout)
        tb.addAction(add_layout)
        copy_img = QAction(icon("copy"), "Copy Image", self)
        copy_img.setToolTip("Copy the plot to the clipboard")
        copy_img.triggered.connect(self._copy_image)
        tb.addAction(copy_img)
        self.save_plot_action = QAction(icon("export"), "Save Plot As…", self)
        self.save_plot_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.save_plot_action.setToolTip(f"Save the plot as PNG, SVG or PDF ({shortcut_text('S', shift=True)})")
        self.save_plot_action.triggered.connect(self._save_plot_dialog)
        tb.addAction(self.save_plot_action)
        self.print_action = QAction(icon("print"), "Print…", self)
        self.print_action.setShortcut(QKeySequence.StandardKey.Print)
        self.print_action.setToolTip(f"Print the plot ({shortcut_text('P')})")
        self.print_action.triggered.connect(self._print_dialog)
        tb.addAction(self.print_action)
        tb.addSeparator()
        self.stats_action = QAction(icon("statistic"), "Statistics", self)
        self.stats_action.setCheckable(True)
        self.stats_action.setChecked(False)
        self.stats_action.setToolTip("Show statistics of the displayed populations")
        tb.addAction(self.stats_action)
        self.ancestry_action = QAction(icon("ancestry"), "Ancestry", self)
        self.ancestry_action.setCheckable(True)
        self.ancestry_action.setChecked(True)
        self.ancestry_action.setToolTip("Show the ancestry strip")
        self.ancestry_action.toggled.connect(lambda on: self.ancestry.setVisible(on))
        tb.addAction(self.ancestry_action)
        # keyboard shortcuts for the tools
        for key, shortcut in (("pointer", "V"), ("rect", "R"), ("polygon", "P"), ("ellipse", "E"), ("quadrant", "Q"), ("range", "H"), ("bisector", "B"), ("freehand", "F"), ("zoom", "Z")):
            self.tool_actions[key].setShortcut(QKeySequence(shortcut))
        close_act = QAction("Close", self)
        close_act.setShortcut(QKeySequence.StandardKey.Close)
        close_act.triggered.connect(self.close)
        self.addAction(close_act)
        undo = QAction("Undo", self)
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        undo.triggered.connect(self.controller.undo)
        self.addAction(undo)
        redo = QAction("Redo", self)
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        redo.triggered.connect(self.controller.redo)
        self.addAction(redo)

    def _options_menu(self) -> QMenu:
        menu = QMenu(self)
        prefs = self.controller.preferences
        opts = self.population.graph.options

        self.option_actions: dict[str, QAction] = {}

        def toggle(key, label, default=False, target=None):
            target = target or menu
            act = QAction(label, target)
            act.setCheckable(True)
            act.setChecked(bool(opts.get(key, prefs.get(key, default))))
            act.toggled.connect(lambda on, k=key: self._set_option(k, on))
            target.addAction(act)
            self.option_actions[key] = act

        toggle("show_gates", "Show gates", True)
        toggle("show_gate_labels", "Show gate labels", True)
        toggle("show_gate_names", "Show gate names")
        toggle("show_gate_freqs", "Show gate frequencies")
        toggle("show_outliers", "Show outliers (contour plots)")
        menu.addSeparator()
        ov = menu.addMenu("Overlays")
        toggle("show_legend", "Show legend", True, ov)
        toggle("histogram_offset", "Offset (staggered) histograms", False, ov)
        lw = ov.addMenu("Histogram line width")
        grp = QActionGroup(lw)
        current_lw = float(opts.get("histogram_line_width", 1.2))
        for label, value in (("Thin (0.8)", 0.8), ("Normal (1.2)", 1.2), ("Medium (2)", 2.0), ("Thick (3)", 3.0)):
            act = QAction(label, lw)
            act.setCheckable(True)
            act.setChecked(abs(current_lw - value) < 1e-6)
            act.triggered.connect(lambda checked=False, v=value: self._set_option("histogram_line_width", v))
            grp.addAction(act)
            lw.addAction(act)
        menu.addSeparator()
        for title, key, choices in (
            ("Smoothing", "smoothing", [("None", 0), ("Low", 1), ("Medium", 2), ("High", 3)]),
            ("Dot size", "dot_size", [("1 px", 1), ("2 px", 2), ("3 px", 3), ("4 px", 4)]),
            ("Events to draw", "dots_to_draw", [("10,000", 10000), ("25,000", 25000), ("50,000", 50000), ("100,000", 100000), ("All", 10_000_000)]),
            ("Contour level", "contour_level", [("2%", 2.0), ("5%", 5.0), ("10%", 10.0), ("20%", 20.0)]),
            ("Histogram y-axis", "histogram_ymode", [("Count", "count"), ("% of Max (normalized to mode)", "mode")]),
        ):
            sub = menu.addMenu(title)
            grp = QActionGroup(sub)
            grp.setExclusive(True)
            current = opts.get(key, prefs.get(key))
            for label, value in choices:
                act = QAction(label, sub)
                act.setCheckable(True)
                act.setChecked(current == value)
                act.triggered.connect(lambda checked=False, k=key, v=value: self._set_option(k, v))
                grp.addAction(act)
                sub.addAction(act)
        menu.addSeparator()
        for label, slot in (("Zoom In\t+", lambda: self.plot.zoom_by(0.8)), ("Zoom Out\t-", lambda: self.plot.zoom_by(1.25)), ("Reset Zoom\t0", self.plot.reset_zoom)):
            act = QAction(label, menu)
            act.triggered.connect(slot)
            menu.addAction(act)
        menu.addSeparator()
        default = QAction("Use these settings for new plots", menu)
        default.triggered.connect(self._make_default)
        menu.addAction(default)
        reset = QAction("Reset to defaults", menu)
        reset.triggered.connect(self._reset_options)
        menu.addAction(reset)
        return menu

    def _set_option(self, key: str, value):
        self.population.graph.options[key] = value
        self.controller.mark_modified()
        self.plot.invalidate()

    def _make_default(self):
        self.controller.preferences.update(dict(self.population.graph.options))
        self.controller.preferencesChanged.emit()

    def _reset_options(self):
        self.population.graph.options.clear()
        self.plot.invalidate()
        prefs = self.controller.preferences
        defaults = {"show_gates": True, "show_gate_labels": True, "show_legend": True, "histogram_offset": False}
        for key, act in getattr(self, "option_actions", {}).items():
            act.blockSignals(True)
            act.setChecked(bool(prefs.get(key, defaults.get(key, False))))
            act.blockSignals(False)

    def _build_central(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.ancestry = AncestryBar(self.controller)
        self.ancestry.navigate.connect(self.open_population)
        outer.addWidget(self.ancestry)

        body = QWidget()
        grid = QHBoxLayout(body)
        grid.setContentsMargins(6, 6, 6, 2)
        grid.setSpacing(4)
        # y axis column
        ycol = QVBoxLayout()
        ycol.setSpacing(2)
        self.y_transform = QToolButton()
        self.y_transform.setObjectName("transformButton")
        self.y_transform.setText("T")
        self.y_transform.setToolTip("Customize the y-axis scale")
        self.y_transform.clicked.connect(lambda: self._customize_axis("y"))
        ycol.addWidget(self.y_transform, 0, Qt.AlignmentFlag.AlignHCenter)
        ycol.addStretch(1)
        self.y_button = AxisButton(vertical=True)
        self.y_button.setMenu(QMenu(self.y_button))
        self.y_button.menu().aboutToShow.connect(lambda: self._fill_axis_menu("y"))
        ycol.addWidget(self.y_button, 0, Qt.AlignmentFlag.AlignHCenter)
        ycol.addStretch(1)
        grid.addLayout(ycol)
        # plot + x axis row
        pcol = QVBoxLayout()
        pcol.setSpacing(2)
        frame = QFrame()
        frame.setObjectName("plotFrame")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.plot)
        pcol.addWidget(frame, 1)
        xrow = QHBoxLayout()
        xrow.addStretch(1)
        self.x_button = AxisButton(vertical=False)
        self.x_button.setMenu(QMenu(self.x_button))
        self.x_button.menu().aboutToShow.connect(lambda: self._fill_axis_menu("x"))
        xrow.addWidget(self.x_button)
        xrow.addStretch(1)
        self.x_transform = QToolButton()
        self.x_transform.setObjectName("transformButton")
        self.x_transform.setText("T")
        self.x_transform.setToolTip("Customize the x-axis scale")
        self.x_transform.clicked.connect(lambda: self._customize_axis("x"))
        xrow.addWidget(self.x_transform)
        pcol.addLayout(xrow)
        grid.addLayout(pcol, 1)
        outer.addWidget(body, 1)
        self.setCentralWidget(central)

    def _build_stats_dock(self):
        self.stats_dock = QDockWidget("Statistics", self)
        self.stats_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.addStretch(1)
        self.stats_copy = QToolButton()
        self.stats_copy.setText("Copy")
        self.stats_copy.setIcon(icon("copy"))
        self.stats_copy.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.stats_copy.setAutoRaise(True)
        self.stats_copy.setToolTip("Copy the table (tab-separated) for a spreadsheet")
        self.stats_copy.clicked.connect(self.copy_stats)
        bar.addWidget(self.stats_copy)
        lay.addLayout(bar)
        self.stats_table = QTableWidget(0, 4)
        self.stats_table.setObjectName("statsTable")
        self.stats_table.setHorizontalHeaderLabels(["Population", "Events", "% of Parent", "% of Total"])
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.stats_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.verticalHeader().setDefaultSectionSize(20)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self.stats_table, 1)
        self.stats_dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.stats_dock)
        self.stats_dock.setVisible(False)
        self.stats_action.toggled.connect(self.stats_dock.setVisible)
        self.stats_action.toggled.connect(lambda on: on and self._refresh_stats())
        self.stats_dock.visibilityChanged.connect(lambda v: self.stats_action.setChecked(v))

    # ------------------------------------------------------------- behaviour
    def _update_title(self):
        prefix = f"Group {self.group.name}: " if self.group is not None else ""
        self.setWindowTitle(f"{prefix}{self.sample.name} – {self.population.display_name}")

    def _select_tool(self, key: str):
        self.tool_actions[key].setChecked(True)
        self.plot.set_tool(key)

    def _sync_axes_ui(self):
        g = self.plot.graph
        self.x_button.setText(self.sample.param_label(g.x) if g.x else "x")
        self.y_button.setText(self.sample.param_label(g.y) if g.y else "Histogram")
        self.y_button.updateGeometry()
        idx = self.plot_type.findData(g.plot_type)
        self.plot_type.blockSignals(True)
        self.plot_type.setCurrentIndex(max(0, idx))
        self.plot_type.blockSignals(False)
        self.y_transform.setEnabled(g.y is not None)
        self.statusBar().showMessage(self._event_summary(), 4000)
        if getattr(self, "_stats_axes", None) != (g.x, g.y):
            self._refresh_stats()

    def _event_summary(self) -> str:
        pop = self.population
        total = self.sample.n_events
        return f"{pop.count:,} events ({pop.freq_of_total:.1f}% of {total:,})"

    def _fill_axis_menu(self, axis: str):
        button = self.x_button if axis == "x" else self.y_button
        menu = button.menu()
        menu.clear()
        g = self.plot.graph
        current = g.x if axis == "x" else g.y
        if axis == "y":
            act = menu.addAction("Histogram")
            act.setCheckable(True)
            act.setChecked(g.y is None)
            act.triggered.connect(lambda: self.plot.set_axes(y=None))
            menu.addSeparator()
        names = self.sample.parameter_names()
        comp_names = [n for n in names if self.sample.is_compensated_param(n)]
        raw_names = [n for n in names if n not in comp_names]
        for block in (comp_names, raw_names):
            for name in block:
                act = menu.addAction(self.sample.param_label(name))
                act.setCheckable(True)
                act.setChecked(name == current)
                if axis == "x":
                    act.triggered.connect(lambda checked=False, n=name: self.plot.set_axes(x=n))
                else:
                    act.triggered.connect(lambda checked=False, n=name: self.plot.set_axes(y=n))
            if block and block is comp_names and raw_names:
                menu.addSeparator()
        menu.addSeparator()
        act = menu.addAction(icon("transform"), "Customize axis…")
        act.triggered.connect(lambda: self._customize_axis(axis))
        if self.plot.view != (0.0, 1.0, 0.0, 1.0):
            act = menu.addAction("Reset zoom")
            act.triggered.connect(self.plot.reset_zoom)

    def _plot_type_changed(self, index: int):
        key = self.plot_type.itemData(index)
        if key:
            self.plot.set_axes(plot_type=key)

    def _customize_axis(self, axis: str):
        g = self.plot.graph
        param = g.x if axis == "x" else g.y
        if not param:
            return
        group = self.group or self.controller.current_group
        dlg = ScaleDialog(self.sample, param, self.sample.scale(param), group, self)
        if not dlg.exec():
            return
        scale = dlg.scale()
        samples = self.controller.workspace.members(group) if dlg.apply_to_group() else [self.sample]
        if self.sample not in samples:
            samples.append(self.sample)
        params = [param]
        if dlg.apply_to_all_params():
            params = [n for n in self.sample.parameter_names() if self.sample.scale(n).kind != "linear" or n == param]
        for p in params:
            self.controller.set_scale(samples, p, scale)
        self.plot.invalidate()

    def _on_selection(self, pop):
        if pop is not None:
            self.statusBar().showMessage(f"{pop.name}: {pop.count:,} events, {pop.freq_of_parent:.1f}% of parent", 5000)

    def open_population(self, pop: Population):
        """Navigate this window to another population of the same sample."""
        if pop is self.population:
            return
        self.population = pop
        self.plot.set_population(pop)
        self._update_title()
        self._sync_axes_ui()
        self.ancestry.rebuild(self.sample, pop)
        self._refresh_stats()

    def _on_gating_changed(self, sample_id):
        if sample_id is not None and sample_id != self.sample.id:
            return
        # the population object may have been replaced (group copy / undo)
        pop = self.sample.root.find(self.population.path) if self.population.parent is not None or self.population is not self.sample.root else self.sample.root
        if pop is None:
            self.close()
            return
        if pop is not self.population:
            self.population = pop
            self.plot.population = pop
            self.plot.graph = pop.graph
            self.plot._ensure_axes()
        self.plot.invalidate()
        self.ancestry.rebuild(self.sample, self.population)
        self._refresh_stats()
        self._sync_axes_ui()

    def _on_samples_changed(self):
        if self.sample not in self.controller.workspace.samples:
            self.close()
        else:
            self._update_title()

    def stats_rows(self) -> list[tuple[str, Sample, Population]]:
        """The populations shown on this plot: the population, its gates and the overlays."""
        rows = [(self.population.display_name, self.sample, self.population)]
        rows += [(p.display_name, self.sample, p) for p in self.population.children]
        for s, p in self.plot.overlays:
            rows.append((f"{s.name} / {p.display_name}", s, p))
        return rows

    def stats_headers(self) -> list[str]:
        g = self.plot.graph
        headers = ["Population", "Events", "% of Parent", "% of Total"]
        for param in (g.x, g.y):
            if param:
                stain = self.sample.param_stain(param)
                label = f"{param} ({stain})" if stain and stain != param else param
                headers += [f"Median {label}", f"Mean {label}"]
        return headers

    def _refresh_stats(self):
        g = self.plot.graph
        self._stats_axes = (g.x, g.y)
        if not self.stats_action.isChecked():
            return  # computed when the panel is shown
        headers = self.stats_headers()
        params = [p for p in (g.x, g.y) if p]
        rows = self.stats_rows()
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        self.stats_table.setRowCount(len(rows))
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for i, (label, s, p) in enumerate(rows):
            values = [f"{p.count:,}", format_value("Freq. of Parent", p.freq_of_parent), format_value("Freq. of Total", p.freq_of_total)]
            for param in params:
                med = mean = float("nan")
                try:
                    v = np.asarray(s.data(param))[p.mask()]
                    v = v[np.isfinite(v)]
                    if v.size:
                        med, mean = float(np.median(v)), float(np.mean(v))
                except (KeyError, IndexError, ValueError):
                    pass
                values += [format_value("Median", med), format_value("Mean", mean)]
            item = QTableWidgetItem(label)
            item.setToolTip(label)
            self.stats_table.setItem(i, 0, item)
            for j, text in enumerate(values, start=1):
                cell = QTableWidgetItem(text)
                cell.setTextAlignment(right)
                self.stats_table.setItem(i, j, cell)
        hh = self.stats_table.horizontalHeader()
        for j in range(1, len(headers)):
            hh.setSectionResizeMode(j, QHeaderView.ResizeMode.ResizeToContents)

    def stats_tsv(self) -> str:
        t = self.stats_table
        lines = ["\t".join(t.horizontalHeaderItem(j).text() for j in range(t.columnCount()))]
        for i in range(t.rowCount()):
            lines.append("\t".join((t.item(i, j).text() if t.item(i, j) else "") for j in range(t.columnCount())))
        return "\n".join(lines) + "\n"

    def copy_stats(self):
        if not self.stats_action.isChecked():
            self.stats_action.setChecked(True)
        self._refresh_stats()
        QApplication.clipboard().setText(self.stats_tsv())
        self.statusBar().showMessage("Statistics copied to the clipboard", 3000)

    def _choose_overlays(self):
        group = self.group or self.controller.current_group
        samples = self.controller.workspace.members(group)
        if self.sample not in samples:
            samples = [self.sample] + samples
        dlg = OverlayDialog(samples, self.plot.overlays, self)
        if dlg.exec():
            self.plot.set_overlays([(s, p) for s, p in dlg.values() if not (s is self.sample and p is self.population)])
            self._refresh_stats()

    def _add_to_layout(self):
        if self.controller.add_to_layout is not None:
            self.controller.add_to_layout(self.sample, self.population)

    def _copy_image(self):
        QApplication.clipboard().setImage(self.plot_image(2.0))
        self.statusBar().showMessage("Plot copied to the clipboard", 3000)

    # ---------------------------------------------------------------- export
    def _export_size(self) -> tuple[int, int]:
        return max(240, self.plot.width()), max(240, self.plot.height())

    def _render_onto(self, painter: QPainter | None, dpr: float):
        w, h = self._export_size()
        return render_plot(self.sample, self.population, self.plot.graph, w, h, self.plot.options(), overlays=self.plot.overlays or None, dpr=dpr, painter=painter)

    def plot_image(self, dpr: float = 2.0):
        return self._render_onto(None, dpr).image

    def save_plot(self, path) -> Path:
        """Write the plot to `path`: PNG (2x resolution), SVG or PDF by suffix."""
        path = Path(path)
        suffix = path.suffix.lower()
        w, h = self._export_size()
        if suffix == ".svg":
            from PySide6.QtSvg import QSvgGenerator

            gen = QSvgGenerator()
            gen.setFileName(str(path))
            gen.setSize(QSize(w, h))
            gen.setViewBox(QRect(0, 0, w, h))
            gen.setTitle(self.windowTitle())
            gen.setDescription("FlowDesk plot")
            painter = QPainter(gen)
            self._render_onto(painter, 3.0)
            painter.end()
        elif suffix == ".pdf":
            writer = QPdfWriter(str(path))
            writer.setResolution(300)
            writer.setPageSize(QPageSize(QSizeF(w, h), QPageSize.Unit.Point, "plot"))
            writer.setPageMargins(QMarginsF(0, 0, 0, 0))
            writer.setTitle(self.windowTitle())
            painter = QPainter(writer)
            k = writer.resolution() / 72.0  # one logical pixel = one point
            painter.scale(k, k)
            self._render_onto(painter, 3.0)
            painter.end()
        else:
            if suffix not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                path = path.with_suffix(".png")
            if not self.plot_image(2.0).save(str(path)):
                raise OSError(f"could not write {path}")
        return path

    def _save_plot_dialog(self):
        base = f"{self.sample.name} {self.population.display_name}".replace("/", "_").replace(":", "_")
        path, chosen = QFileDialog.getSaveFileName(self, "Save Plot As", base + ".png", PLOT_FILE_FILTERS)
        if not path:
            return
        if not Path(path).suffix:
            path += ".svg" if "svg" in chosen.lower() else ".pdf" if "pdf" in chosen.lower() else ".png"
        try:
            out = self.save_plot(path)
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save the plot: {exc}", 6000)
            return
        self.statusBar().showMessage(f"Saved {out.name}", 4000)

    def print_plot(self, printer):
        """Paint the plot, scaled to fit, on a QPrinter (or any paged device)."""
        from PySide6.QtPrintSupport import QPrinter

        w, h = self._export_size()
        painter = QPainter(printer)
        page = printer.pageRect(QPrinter.Unit.DevicePixel)
        k = min(page.width() / w, page.height() / h)
        painter.translate((page.width() - w * k) / 2, (page.height() - h * k) / 2)
        painter.scale(k, k)
        self._render_onto(painter, 3.0)
        painter.end()

    def _print_dialog(self):
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        dlg = QPrintDialog(printer, self)
        if dlg.exec():
            self.print_plot(printer)

    # ------------------------------------------------------------ context menu
    def _context_menu(self, pop: Population | None, global_pos):
        menu = QMenu(self)
        group = self.group or self.controller.current_group
        if pop is not None:
            act = menu.addAction(icon("graph"), f"Open “{pop.name}”")
            act.triggered.connect(lambda: self.open_population(pop))
            menu.addSeparator()
            act = menu.addAction(icon("rename"), "Rename…")
            act.triggered.connect(lambda: self._rename(pop))
            act = menu.addAction(icon("delete"), "Delete")
            act.triggered.connect(self.plot.delete_selected)
            act = menu.addAction(icon("copy"), "Copy")
            act.triggered.connect(lambda: self._copy_gates([pop]))
            if self.group is None:
                act = menu.addAction(icon("group"), f"Apply to group “{group.name}”")
                act.triggered.connect(lambda: self.controller.copy_populations_to_group([pop], group))
            menu.addSeparator()
            act = menu.addAction("Invert gate (NOT)")
            act.setCheckable(True)
            act.setChecked(bool(pop.gate.negated))
            act.triggered.connect(lambda on: self._toggle_negated(pop, on))
            color = menu.addMenu("Gate color")
            for name, value in (("Black", "#000000"), ("Red", "#d81b1b"), ("Blue", "#2f6fd6"), ("Green", "#2e9e5b"), ("Orange", "#e8912d"), ("Purple", "#7c4dd6")):
                a = color.addAction(name)
                a.triggered.connect(lambda checked=False, v=value: self._set_gate_color(pop, v))
            tint = menu.addMenu("Tint")
            grp = QActionGroup(tint)
            current = float(getattr(pop.gate, "tint", 0.0) or 0.0)
            for name, value in TINT_CHOICES:
                a = tint.addAction(name)
                a.setCheckable(True)
                a.setChecked(abs(current - value) < 1e-6 or (value == 0.0 and current == 0.0))
                grp.addAction(a)
                a.triggered.connect(lambda checked=False, v=value: self._set_gate_tint(pop, v))
            act = menu.addAction(icon("lock"), "Lock gate")
            act.setCheckable(True)
            act.setChecked(bool(getattr(pop.gate, "locked", False)))
            act.triggered.connect(lambda on: self._set_gate_locked(pop, on))
            act = menu.addAction("Reset label position")
            act.triggered.connect(lambda: self._reset_label(pop))
            menu.addSeparator()
        if self.controller.clipboard:
            act = menu.addAction(icon("paste"), f"Paste {len(self.controller.clipboard)} gate(s) here")
            act.triggered.connect(self._paste_gates)
        act = menu.addAction(icon("layout-editor"), "Add plot to Layout")
        act.triggered.connect(self._add_to_layout)
        act = menu.addAction(icon("copy"), "Copy plot image")
        act.triggered.connect(self._copy_image)
        menu.addAction(self.save_plot_action)
        menu.addAction(self.print_action)
        if self.plot.view != (0.0, 1.0, 0.0, 1.0):
            act = menu.addAction("Reset zoom")
            act.triggered.connect(self.plot.reset_zoom)
        menu.exec(global_pos)

    def _rename(self, pop: Population):
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Rename population", "Name:", text=pop.name)
        if ok and name.strip():
            self.controller.rename_population(pop, name.strip(), self.group)

    def _copy_gates(self, pops: list[Population]):
        self.controller.clipboard = [p.clone(deep=True) for p in pops]
        self.statusBar().showMessage(f"Copied {len(pops)} gate(s)", 3000)

    def _paste_gates(self):
        self.controller.paste_populations(self.controller.clipboard, self.population, self.group)

    def _toggle_negated(self, pop: Population, on: bool):
        self.controller.begin_change("invert gate")
        pop.gate.negated = on
        self.controller.gate_changed(pop, self.group)

    def _set_gate_color(self, pop: Population, value: str):
        pop.gate.color = None if value == "#000000" else value
        self.controller.gate_changed(pop, self.group)

    def _set_gate_tint(self, pop: Population, value: float):
        self.controller.begin_change("gate tint")
        pop.gate.tint = float(value)
        self.controller.gate_changed(pop, self.group)

    def _set_gate_locked(self, pop: Population, on: bool):
        self.controller.begin_change("unlock gate" if not on else "lock gate")
        targets = [pop] + self.plot._quadrant_siblings(pop)
        for p in targets:
            p.gate.locked = bool(on)
        for p in targets[1:]:
            self.controller.gate_changed(p, self.group)
        self.controller.gate_changed(pop, self.group)

    def _reset_label(self, pop: Population):
        pop.gate.label_pos = None
        self.controller.gate_changed(pop, self.group)

    def closeEvent(self, event):
        self.closed.emit(self)
        super().closeEvent(event)
