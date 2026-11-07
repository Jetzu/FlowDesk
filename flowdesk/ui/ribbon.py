"""A ribbon: tabbed pages of titled bands holding large and small tool buttons."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QStackedWidget, QTabBar, QToolButton, QVBoxLayout, QWidget


class RibbonBand(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbonBand")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 1)
        outer.setSpacing(1)
        self.row = QHBoxLayout()
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(2)
        outer.addLayout(self.row, 1)
        label = QLabel(title)
        label.setObjectName("ribbonBandTitle")
        label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        outer.addWidget(label)
        self._small_column: QVBoxLayout | None = None
        self._small_count = 0

    def add_large(self, action: QAction, menu=None) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("ribbonButton")
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setIconSize(QSize(32, 32))
        btn.setMinimumWidth(58)
        btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        if menu is not None:
            btn.setMenu(menu)
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
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
        btn = QToolButton()
        btn.setObjectName("ribbonSmall")
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setIconSize(QSize(16, 16))
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if menu is not None:
            btn.setMenu(menu)
            btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._small_column.insertWidget(self._small_column.count() - 1, btn)
        self._small_count += 1
        return btn

    def add_widget(self, widget: QWidget):
        self.row.addWidget(widget)
        self._small_column = None


class RibbonPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ribbonPage")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(0)
        self._layout.addStretch(1)

    def add_band(self, title: str) -> RibbonBand:
        band = RibbonBand(title)
        self._layout.insertWidget(self._layout.count() - 1, band)
        return band


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
        self.stack.setFixedHeight(84)
        layout.addWidget(self.tabs)
        layout.addWidget(self.stack)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)

    def add_page(self, title: str) -> RibbonPage:
        page = RibbonPage()
        self.tabs.addTab(title)
        self.stack.addWidget(page)
        return page
