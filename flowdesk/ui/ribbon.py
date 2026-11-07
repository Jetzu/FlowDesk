"""A ribbon: tabbed pages of titled bands holding large and small tool buttons.

When a page is wider than the window, its bands shrink from the right the way
ribbon groups do: first to small stacked buttons, then to a single drop-down
button carrying the band's name. Every representation is built once and
shown/hidden, so nothing is destroyed while the user works.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QFontMetrics
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QScrollArea, QSizePolicy, QStackedWidget, QTabBar, QToolButton, QVBoxLayout, QWidget


def wrap_label(text: str, max_chars: int = 11) -> str:
    """Break a long button label into two lines, the way ribbon buttons do."""
    text = text.replace("&", "").strip()
    if len(text) <= max_chars or " " not in text:
        return text
    words = text.split()
    best, best_score = None, None
    for i in range(1, len(words)):
        first, second = " ".join(words[:i]), " ".join(words[i:])
        score = abs(len(first) - len(second)) + max(0, max(len(first), len(second)) - max_chars) * 3
        if best_score is None or score < best_score:
            best, best_score = (first, second), score
    return "\n".join(best) if best else text


class LargeRibbonButton(QToolButton):
    """A large ribbon button whose label wraps onto two lines instead of being elided."""

    def setDefaultAction(self, action: QAction):  # noqa: N802 - Qt naming
        super().setDefaultAction(action)
        action.changed.connect(self.refresh_text)

    def refresh_text(self):
        action = self.defaultAction()
        if action is None:
            return
        wrapped = wrap_label(action.iconText())
        if self.text() != wrapped:
            self.setText(wrapped)
        fm = QFontMetrics(self.font())
        widest = max(fm.horizontalAdvance(line) for line in wrapped.split("\n"))
        self.setMinimumWidth(max(58, widest + 14))


def make_large_button(action: QAction, menu=None) -> QToolButton:
    btn = LargeRibbonButton()
    btn.setObjectName("ribbonButton")
    btn.setDefaultAction(action)
    btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    btn.setIconSize(QSize(32, 32))
    btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
    if menu is not None:
        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
    btn.refresh_text()
    return btn


def make_small_button(action: QAction, menu=None) -> QToolButton:
    btn = QToolButton()
    btn.setObjectName("ribbonSmall")
    btn.setDefaultAction(action)
    btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    btn.setIconSize(QSize(16, 16))
    btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    if menu is not None:
        btn.setMenu(menu)
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    return btn


class _Representation:
    """One visual form of a band's content."""

    def __init__(self):
        self.widget = QWidget()
        self.row = QHBoxLayout(self.widget)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(2)
        self._small_column: QVBoxLayout | None = None
        self._small_count = 0

    def add_large(self, action: QAction, menu=None) -> QToolButton:
        btn = make_large_button(action, menu)
        self.row.addWidget(btn)
        self._small_column = None
        return btn

    def add_small(self, action: QAction, menu=None) -> QToolButton:
        if self._small_column is None or self._small_count >= 3:
            self._small_column = QVBoxLayout()
            self._small_column.setContentsMargins(0, 0, 0, 0)
            self._small_column.setSpacing(1)
            self._small_column.addStretch(1)
            self.row.addLayout(self._small_column)
            self._small_count = 0
        btn = make_small_button(action, menu)
        self._small_column.insertWidget(self._small_column.count() - 1, btn)
        self._small_count += 1
        return btn

    def add_widget(self, widget: QWidget):
        self.row.addWidget(widget)
        self._small_column = None


class RibbonBand(QWidget):
    MODES = ("large", "compact", "collapsed")

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbonBand")
        self.title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 1)
        outer.setSpacing(1)
        self._reps_box = QHBoxLayout()
        self._reps_box.setContentsMargins(0, 0, 0, 0)
        self._reps_box.setSpacing(0)
        outer.addLayout(self._reps_box, 1)
        label = QLabel(title)
        label.setObjectName("ribbonBandTitle")
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        outer.addWidget(label)
        self._label = label
        self._entries: list[tuple[str, object, object]] = []  # (kind, action-or-widget, menu)
        self._reps: dict[str, _Representation] = {}
        self.mode = "large"
        self._primary = self._add_rep("large")

    # ---- building (clients only ever add to the large representation)
    def _add_rep(self, mode: str) -> _Representation:
        rep = _Representation()
        self._reps[mode] = rep
        self._reps_box.addWidget(rep.widget)
        rep.widget.setVisible(mode == self.mode)
        return rep

    @property
    def row(self) -> QHBoxLayout:
        return self._primary.row

    def add_large(self, action: QAction, menu=None) -> QToolButton:
        self._entries.append(("large", action, menu))
        return self._primary.add_large(action, menu)

    def add_small(self, action: QAction, menu=None) -> QToolButton:
        self._entries.append(("small", action, menu))
        return self._primary.add_small(action, menu)

    def add_widget(self, widget: QWidget):
        self._entries.append(("widget", widget, None))
        self._primary.add_widget(widget)

    def _build_compact(self) -> _Representation:
        rep = self._add_rep("compact")
        for kind, obj, menu in self._entries:
            if kind != "widget":
                rep.add_small(obj, menu)
        return rep

    def _build_collapsed(self) -> _Representation:
        rep = self._add_rep("collapsed")
        btn = QToolButton()
        btn.setObjectName("ribbonButton")
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setIconSize(QSize(32, 32))
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        btn.setText(wrap_label(self.title) + " ▾")
        btn.setToolTip(self.title)
        menu = QMenu(btn)
        for kind, obj, sub in self._entries:
            if kind == "widget":
                continue
            if btn.icon().isNull() and isinstance(obj, QAction) and not obj.icon().isNull():
                btn.setIcon(obj.icon())
            if sub is not None:
                sub_menu = menu.addMenu(obj.icon(), obj.text().replace("&", ""))
                sub_menu.addAction(obj)
                for a in sub.actions():
                    if a is not obj:
                        sub_menu.addAction(a)
            else:
                menu.addAction(obj)
        btn.setMenu(menu)
        fm = QFontMetrics(btn.font())
        widest = max(fm.horizontalAdvance(line) for line in btn.text().split("\n"))
        btn.setMinimumWidth(max(58, widest + 14))
        btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        rep.row.addWidget(btn)
        return rep

    # ---- modes
    @property
    def compact(self) -> bool:
        return self.mode != "large"

    def natural_width(self) -> int:
        """Width this band needs in its current mode (independent of layout caches)."""
        rep = self._reps[self.mode]
        rep.row.activate()
        margins = self.layout().contentsMargins()
        return max(rep.widget.sizeHint().width(), self._label.sizeHint().width()) + margins.left() + margins.right()

    def set_compact(self, compact: bool):
        self.set_mode("compact" if compact else "large")

    def set_mode(self, mode: str):
        if mode not in self.MODES or mode == self.mode:
            return
        if mode not in self._reps:
            if mode == "compact":
                self._build_compact()
            else:
                self._build_collapsed()
        self.mode = mode
        for name, rep in self._reps.items():
            rep.widget.setVisible(name == mode)
        self._reps_box.invalidate()
        self.layout().invalidate()
        self.updateGeometry()


class RibbonPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbonPage")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(0)
        self._layout.addStretch(1)
        self.bands: list[RibbonBand] = []

    def add_band(self, title: str) -> RibbonBand:
        band = RibbonBand(title)
        self._layout.insertWidget(self._layout.count() - 1, band)
        self.bands.append(band)
        return band

    def natural_width(self) -> int:
        """Width the page needs for its bands in their current modes.

        Computed from the bands' own (freshly activated) size hints so that
        it is correct immediately after a mode change, before the event loop
        has delivered the pending layout requests."""
        margins = self._layout.contentsMargins()
        total = margins.left() + margins.right()
        for band in self.bands:
            total += band.natural_width() + max(0, self._layout.spacing())
        return total

    def fit(self, width: int):
        """Shrink bands from the right (small buttons, then one drop-down) until the page fits."""
        for band in self.bands:
            band.set_mode("large")
        if self.natural_width() <= width:
            return
        for band in reversed(self.bands):
            band.set_mode("compact")
            if self.natural_width() <= width:
                return
            band.set_mode("collapsed")
            if self.natural_width() <= width:
                return


class Ribbon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbon")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 0)
        layout.setSpacing(0)
        self.tabs = QTabBar()
        self.tabs.setObjectName("ribbonTabs")
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.stack = QStackedWidget()
        self.stack.setFixedHeight(104)
        layout.addWidget(self.tabs)
        layout.addWidget(self.stack)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)
        self.tabs.currentChanged.connect(lambda _i: self.fit_current())
        self.pages: list[RibbonPage] = []

    def fit_current(self):
        idx = self.stack.currentIndex()
        if 0 <= idx < len(self.pages):
            self.pages[idx].fit(max(200, self.stack.width() - 4))

    def fit_all(self):
        for page in self.pages:
            page.fit(max(200, self.stack.width() - 4))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_current()

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_all()

    def add_page(self, title: str) -> RibbonPage:
        page = RibbonPage()
        self.pages.append(page)
        # pages scroll horizontally when even the collapsed bands do not fit
        scroll = QScrollArea()
        scroll.setObjectName("ribbonScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(page)
        self.tabs.addTab(title)
        self.stack.addWidget(scroll)
        return page
