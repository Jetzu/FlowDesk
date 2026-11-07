"""Application theme: a light, ribbon-style look."""
from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#2f6fd6"
SELECTION = "#cfe0f7"
WINDOW_BG = "#f2f3f5"
PANE_BG = "#ffffff"
BORDER = "#c9cdd3"
TEXT = "#1f2328"
MUTED = "#6b7280"
COMP_GROUP = "#bc1900"


# ------------------------------------------------------------- platform text
def is_mac(platform: str | None = None) -> bool:
    return (platform or sys.platform) == "darwin"


def modifier_text(platform: str | None = None) -> str:
    """The primary shortcut modifier as users read it: "⌘" on macOS, "Ctrl" elsewhere."""
    return "\u2318" if is_mac(platform) else "Ctrl"


def shortcut_text(key: str, platform: str | None = None, shift: bool = False, alt: bool = False) -> str:
    """Human-readable shortcut, e.g. shortcut_text("Z") -> "⌘Z" / "Ctrl+Z"."""
    if is_mac(platform):
        return ("\u2325" if alt else "") + ("\u21e7" if shift else "") + "\u2318" + key
    parts = ["Ctrl"] + (["Shift"] if shift else []) + (["Alt"] if alt else []) + [key]
    return "+".join(parts)


def modifier_html(platform: str | None = None) -> str:
    """modifier_text() for rich text (an HTML entity on macOS)."""
    return "&#8984;" if is_mac(platform) else "Ctrl"


def base_font_px(platform: str | None = None) -> int:
    """Base UI font size in pixels: macOS 12 px; Windows (9 pt) and Linux 11 px."""
    return 12 if is_mac(platform) else 11


# Point sizes of plot text are converted with this dpi in graph windows, so
# 1 pt = 1 px on every platform (see renderer.RenderOptions.font_dpi).
PLOT_FONT_DPI = 72.0
BASE_FONT_PX = base_font_px()
SMALL_FONT_PX = BASE_FONT_PX - 1
TINY_FONT_PX = BASE_FONT_PX - 2

STYLESHEET = f"""
QMainWindow, QDialog {{
    background: {WINDOW_BG};
}}
QWidget {{
    color: {TEXT};
    font-size: {BASE_FONT_PX}px;
}}
QToolTip {{
    background: #ffffe1;
    color: #000;
    border: 1px solid #a0a0a0;
    padding: 3px;
}}
/* ---------------------------------------------------------------- ribbon */
QWidget#ribbon {{
    background: {WINDOW_BG};
    border-bottom: 1px solid {BORDER};
}}
QTabBar#ribbonTabs::tab {{
    background: transparent;
    border: none;
    padding: 5px 14px 5px 14px;
    margin-right: 2px;
    color: {TEXT};
}}
QTabBar#ribbonTabs::tab:selected {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-bottom: 1px solid #ffffff;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QTabBar#ribbonTabs::tab:hover:!selected {{
    background: #e6e9ee;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}}
QWidget#ribbonPage {{
    background: #ffffff;
    border: 1px solid {BORDER};
    border-top: none;
}}
QWidget#ribbonBand {{
    background: transparent;
    border-right: 1px solid #e3e6ea;
}}
QLabel#ribbonBandTitle {{
    color: {MUTED};
    font-size: {TINY_FONT_PX}px;
    padding-top: 1px;
    padding-bottom: 1px;
}}
QToolButton#ribbonButton {{
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 2px 4px;
    background: transparent;
    font-size: {SMALL_FONT_PX}px;
}}
QToolButton#ribbonButton:hover {{
    background: #e8f0fd;
    border: 1px solid #b7cff2;
}}
QToolButton#ribbonButton:pressed, QToolButton#ribbonButton:checked {{
    background: #d6e4fa;
    border: 1px solid {ACCENT};
}}
QToolButton#ribbonButton:disabled {{
    color: #9aa0a6;
}}
QToolButton#ribbonSmall {{
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 1px 4px;
    background: transparent;
    font-size: {SMALL_FONT_PX}px;
    text-align: left;
}}
QToolButton#ribbonSmall:hover {{
    background: #e8f0fd;
    border: 1px solid #b7cff2;
}}
QToolButton#ribbonSmall:checked {{
    background: #d6e4fa;
    border: 1px solid {ACCENT};
}}
/* ----------------------------------------------------------------- panes */
QTreeView, QTableView, QListView, QTreeWidget, QTableWidget, QListWidget {{
    background: {PANE_BG};
    alternate-background-color: #f6f8fb;
    border: 1px solid {BORDER};
    selection-background-color: {SELECTION};
    selection-color: {TEXT};
    outline: 0;
}}
QTreeView::item, QTableView::item {{
    padding: 1px 2px;
    min-height: 20px;
}}
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected {{
    background: {SELECTION};
    color: {TEXT};
}}
QTreeView::item:hover, QTableView::item:hover {{
    background: #eaf1fb;
}}
QHeaderView::section {{
    background: #f7f8fa;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 3px 6px;
    font-weight: 500;
}}
QSplitter::handle {{
    background: {WINDOW_BG};
}}
QSplitter::handle:vertical {{
    height: 5px;
}}
QSplitter::handle:horizontal {{
    width: 5px;
}}
QStatusBar {{
    background: {WINDOW_BG};
    border-top: 1px solid {BORDER};
    color: {MUTED};
}}
/* --------------------------------------------------------------- toolbar */
QToolBar {{
    background: {WINDOW_BG};
    border-bottom: 1px solid {BORDER};
    spacing: 2px;
    padding: 2px;
}}
QToolBar QToolButton {{
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 2px;
}}
QToolBar QToolButton:hover {{
    background: #e8f0fd;
    border: 1px solid #b7cff2;
}}
QToolBar QToolButton:checked {{
    background: #d6e4fa;
    border: 1px solid {ACCENT};
}}
QToolButton#axisButton {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: #fafbfc;
    padding: 2px 8px;
}}
QToolButton#axisButton:hover {{
    background: #e8f0fd;
    border: 1px solid #b7cff2;
}}
QToolButton#axisButton:pressed {{
    background: #d6e4fa;
    border: 1px solid {ACCENT};
}}
QToolButton#axisButton::menu-indicator {{
    image: none;
    width: 0px;
}}
QToolButton#transformButton {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: #fafbfc;
    font-weight: 600;
    min-width: 18px;
    min-height: 18px;
    padding: 0px;
}}
QToolButton#transformButton:hover {{
    background: #e8f0fd;
    border: 1px solid #b7cff2;
}}
QToolButton#transformButton:disabled {{
    color: #b0b5bc;
}}
/* ---------------------------------------------------- statistics dock */
QTableWidget#statsTable {{
    border: none;
    gridline-color: #e3e6ea;
}}
QTableWidget#statsTable::item {{
    padding: 0px 4px;
    min-height: 18px;
}}
QDockWidget::title {{
    background: {WINDOW_BG};
    padding: 3px 6px;
    border-bottom: 1px solid {BORDER};
}}
QFrame#plotFrame {{
    background: #ffffff;
}}
QWidget#ancestryBar {{
    background: #fafbfc;
    border-bottom: 1px solid {BORDER};
}}
QPushButton {{
    padding: 4px 12px;
}}
QComboBox {{
    padding: 2px 6px;
}}
QGroupBox {{
    font-weight: 600;
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
}}
"""


def apply_theme(app: QApplication):
    if app.property("flowdesk_theme_applied"):
        return
    app.setProperty("flowdesk_theme_applied", True)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(WINDOW_BG))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANE_BG))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#f6f8fb"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(SELECTION))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor("#f7f8fa"))
    pal.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    app.setPalette(pal)
    font = QFont()
    # pixel sizes look alike on every platform (12 pt would be 16 px on Windows)
    font.setPixelSize(BASE_FONT_PX)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
