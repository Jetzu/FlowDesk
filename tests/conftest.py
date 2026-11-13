"""Shared fixtures for the FlowDesk core test suite.

`write_fcs` produces small but valid FCS 3.0 files so the reader and the model
can be exercised without shipping binary fixtures.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

REAL_COMP_CONTROL = Path(
    "/Users/lukasgerstung/Coding/Plotting/FACS-Data/Experiment_20260430/Compensation Controls_PE Stained Control.fcs"
)
REAL_TCELL_PANEL = Path(
    "/Users/lukasgerstung/Desktop/Desktop Staubsauger/recompensated T cell panel/_F1donor 6T cell pehnotype2025-08-18.0003.fcs"
)

DELIM = "/"

# the test suite never opens the agent bridge or touches the user's ~/.flowdesk
os.environ.setdefault("FLOWDESK_BRIDGE", "0")
os.environ.setdefault("FLOWDESK_HOME", tempfile.mkdtemp(prefix="flowdesk-test-home-"))


def _escape(value: str) -> str:
    """Escape the delimiter inside a keyword value by doubling it."""
    return str(value).replace(DELIM, DELIM + DELIM)


def write_fcs(
    path,
    data: np.ndarray,
    names: list[str],
    stains: list[str] | None = None,
    datatype: str = "F",
    byteord: str = "1,2,3,4",
    extra_keywords: dict[str, str] | None = None,
    log_amplified: dict[int, tuple[float, float]] | None = None,
    ranges: list[float] | None = None,
) -> Path:
    """Write a minimal FCS 3.0 file.

    data: N x P array. For datatype 'I' the values are the raw 16-bit channel
    numbers. log_amplified maps a 0-based column index to ($PnE decades, offset);
    it only makes sense for integer data. ranges overrides $PnR per column
    (default 262144 for float data, 1024 for integer data).
    extra_keywords are written last and override generated keywords (e.g. $TOT).
    """
    path = Path(path)
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError("data must be N x len(names)")
    n_events, n_par = data.shape
    datatype = datatype.upper()
    big = byteord.startswith("4") or byteord.startswith("2,1")
    order = ">" if big else "<"
    if datatype == "F":
        dtype, bits = np.dtype(order + "f4"), 32
    elif datatype == "D":
        dtype, bits = np.dtype(order + "f8"), 64
    elif datatype == "I":
        dtype, bits = np.dtype(order + "u2"), 16
    else:
        raise ValueError(f"unsupported datatype {datatype}")
    payload = np.ascontiguousarray(data.astype(dtype)).tobytes()

    keywords: dict[str, str] = {
        "$BYTEORD": byteord,
        "$DATATYPE": datatype,
        "$MODE": "L",
        "$NEXTDATA": "0",
        "$PAR": str(n_par),
        "$TOT": str(n_events),
    }
    for i, name in enumerate(names, start=1):
        keywords[f"$P{i}N"] = name
        if stains and stains[i - 1]:
            keywords[f"$P{i}S"] = stains[i - 1]
        keywords[f"$P{i}B"] = str(bits)
        default_range = 1024 if datatype == "I" else 262144
        keywords[f"$P{i}R"] = str(int(ranges[i - 1]) if ranges else default_range)
        if log_amplified and (i - 1) in log_amplified:
            decades, offset = log_amplified[i - 1]
            keywords[f"$P{i}E"] = f"{decades:g},{offset:g}"
        else:
            keywords[f"$P{i}E"] = "0,0"
    if extra_keywords:
        keywords.update({k: str(v) for k, v in extra_keywords.items()})

    text_start = 58
    # $BEGINDATA / $ENDDATA depend on the TEXT length; use fixed-width values so
    # the length is known before the offsets are.
    def build_text(begin: int, end: int) -> bytes:
        kw = dict(keywords)
        kw["$BEGINDATA"] = f"{begin:020d}"
        kw["$ENDDATA"] = f"{end:020d}"
        kw["$BEGINANALYSIS"] = "0"
        kw["$ENDANALYSIS"] = "0"
        kw["$BEGINSTEXT"] = "0"
        kw["$ENDSTEXT"] = "0"
        body = DELIM + "".join(f"{_escape(k)}{DELIM}{_escape(v)}{DELIM}" for k, v in kw.items())
        return body.encode("utf-8")

    probe = build_text(0, 0)
    text_end = text_start + len(probe) - 1
    data_start = text_end + 1
    data_end = data_start + len(payload) - 1 if payload else 0
    text = build_text(data_start, data_end)
    assert len(text) == len(probe)

    def off(v: int) -> str:
        s = str(v)
        return s.rjust(8) if len(s) <= 8 else "0".rjust(8)

    header = ("FCS3.0" + " " * 4 + off(text_start) + off(text_end) + off(data_start) + off(data_end) + off(0) + off(0)).encode("ascii")
    assert len(header) == 58
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(text)
        fh.write(payload)
    return path


@pytest.fixture
def fcs_writer():
    """Expose write_fcs as a fixture for tests that prefer injection."""
    return write_fcs


# --------------------------------------------------------------------------- synthetic
SYNTH_NAMES = ["FSC-A", "SSC-A", "FITC-A", "PE-A", "Time"]
SYNTH_STAINS = ["", "", "CD3", "CD4", ""]


def make_synthetic_events(n: int = 10000, seed: int = 1) -> dict:
    """Two scatter clusters (cells / debris) and independent FITC/PE positive
    and negative populations. Returns the event matrix and ground-truth masks."""
    rng = np.random.default_rng(seed)
    cells = rng.random(n) < 0.8
    fsc = np.where(cells, rng.normal(90000, 8000, n), rng.normal(15000, 3000, n))
    ssc = np.where(cells, rng.normal(50000, 6000, n), rng.normal(8000, 2000, n))
    fitc_pos = rng.random(n) < 0.3
    pe_pos = rng.random(n) < 0.5
    fitc = np.where(fitc_pos, rng.lognormal(np.log(20000), 0.25, n), rng.normal(200, 150, n))
    pe = np.where(pe_pos, rng.lognormal(np.log(50000), 0.25, n), rng.normal(300, 150, n))
    time = np.linspace(0, 1000, n)
    events = np.column_stack([fsc, ssc, fitc, pe, time]).astype(np.float32)
    return {"events": events, "cells": cells, "fitc_pos": fitc_pos, "pe_pos": pe_pos}


@pytest.fixture
def synthetic():
    return make_synthetic_events()


@pytest.fixture
def synthetic_fcs(tmp_path, synthetic):
    """Path to a synthetic FCS 3.0 file, plus its ground truth."""
    path = write_fcs(tmp_path / "synthetic_sample.fcs", synthetic["events"], SYNTH_NAMES, SYNTH_STAINS)
    return {"path": path, **synthetic, "names": list(SYNTH_NAMES), "stains": list(SYNTH_STAINS)}


# ------------------------------------------------------------------------- real files
@pytest.fixture
def real_comp_control() -> Path:
    if not REAL_COMP_CONTROL.exists():
        pytest.skip("real compensation-control FCS file not available")
    return REAL_COMP_CONTROL


@pytest.fixture
def real_tcell_panel() -> Path:
    if not REAL_TCELL_PANEL.exists():
        pytest.skip("real T-cell panel FCS file not available")
    return REAL_TCELL_PANEL


# ------------------------------------------------------- simulated key state
@pytest.fixture(autouse=True)
def _clear_simulated_modifiers():
    """Undo what QTest.keyClick(..., ControlModifier) leaves behind.

    On the offscreen platform a simulated Ctrl+key keeps QApplication.keyboardModifiers()
    at ControlModifier until a modifier-less key event arrives.  Every later test then
    sees a held Ctrl: QTableWidget.selectRow() toggles instead of selecting, clicks
    extend selections, and so on.  A plain key release clears it.
    """
    yield
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication, QLineEdit
    except ImportError:  # core-only environment
        return
    if QApplication.instance() is None:
        return
    if QApplication.keyboardModifiers() != Qt.KeyboardModifier.NoModifier:
        probe = QLineEdit()
        QTest.keyRelease(probe, Qt.Key.Key_Control)
        probe.deleteLater()


# ---------------------------------------------------------------- Qt shutdown
# Widgets that PySide believes Python owns (e.g. buttons added to a layout
# before that layout had a parent widget, then re-parented by Qt) are freed
# twice when the interpreter finalises after a UI test run, which aborts the
# process with "shared QObject was deleted directly".  The running
# application never gets there (its event loop closes everything first), so
# the test process mirrors that: close, flush, report, then leave before
# Python's finalisation touches Qt.
_EXIT_STATUS = {"code": None}


def pytest_sessionfinish(session, exitstatus):
    try:
        _EXIT_STATUS["code"] = int(exitstatus)
    except (TypeError, ValueError):
        _EXIT_STATUS["code"] = 1 if exitstatus else 0


def pytest_unconfigure(config):
    try:
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication
    except ImportError:  # core-only environment
        return
    app = QApplication.instance()
    if app is None or _EXIT_STATUS["code"] is None:
        return
    import gc
    import os
    import sys

    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
        except RuntimeError:
            pass
    gc.collect()
    for _ in range(3):
        app.processEvents()
        app.sendPostedEvents(None, int(QEvent.Type.DeferredDelete))
    sys.stdout.flush()
    sys.stderr.flush()
    code = _EXIT_STATUS["code"]
    if sys.platform == "win32":
        # os._exit() -> ExitProcess still unloads DLLs, and Qt's DLL teardown
        # faults while the QApplication is alive; TerminateProcess skips that.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        # the pseudo-handle is (HANDLE)-1; declaring the types keeps all 64 bits
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    os._exit(code)
