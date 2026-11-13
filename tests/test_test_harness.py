"""Guards for the test harness itself."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def test_a_simulated_ctrl_key_is_left_held(app):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLineEdit

    w = QLineEdit()
    w.show()
    QTest.keyClick(w, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier  # what QTest does
    w.close()


def test_b_the_next_test_starts_without_a_held_modifier(app):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QTableWidget

    assert QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier
    # the visible symptom: selectRow() used to toggle instead of select
    table = QTableWidget(4, 2)
    table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.selectRow(3)
    assert sorted({i.row() for i in table.selectedIndexes()}) == [3]
