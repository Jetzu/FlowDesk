"""Graph-window extras.

Save Plot As / Print, gate tint and lock, show-gates / label toggles, overlay
legend and staggered histograms, median / mean statistics and keyboard zoom
are built into GraphWindow, PlotWidget and the renderer directly, so this
extension hook has nothing to add to the main window.
"""
from __future__ import annotations


def install(window):
    return None
