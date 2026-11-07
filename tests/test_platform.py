"""Cross-platform guards: launchers, entry point, no macOS-only texts, portable paths."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "flowdesk"

# "⌘" written literally, as an HTML entity or as a Python escape
COMMAND_KEY = re.compile("⌘|&#8984;|&#x2318;|\\\\u2318", re.I)
# files allowed to mention the key: platform helpers, plus the Quick Start text for now
# (main_window.QUICK_START_HTML says "⌘/Ctrl" - TODO: render it with the native key name)
ALLOWED = {
    "flowdesk/ui/style.py",  # modifier_text() / shortcut_text() / modifier_html()
    "flowdesk/ui/platform.py",
    "flowdesk/platform.py",
    "flowdesk/ui/main_window.py",
}


def test_no_command_key_symbol_outside_platform_helpers():
    offenders = []
    for path in PKG_DIR.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if COMMAND_KEY.search(line):
                offenders.append(f"{rel}:{n}: {line.strip()}")
    assert not offenders, "macOS-only ⌘ text outside a platform helper:\n" + "\n".join(offenders)


def test_main_window_command_key_only_in_quick_start():
    src = (PKG_DIR / "ui" / "main_window.py").read_text(encoding="utf-8")
    lines = [n for n, line in enumerate(src.splitlines(), 1) if COMMAND_KEY.search(line)]
    if not lines:
        return
    start = src.find("QUICK_START_HTML")
    assert start >= 0
    end_line = src[: src.index('"""', src.index('"""', start) + 3)].count("\n") + 1
    start_line = src[:start].count("\n") + 1
    assert all(start_line <= n <= end_line for n in lines)


@pytest.mark.parametrize("name", ["flowdesk.sh", "flowdesk.bat", "FlowDesk.command"])
def test_launch_scripts_exist(name):
    assert (ROOT / name).is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize("name", ["flowdesk.sh", "FlowDesk.command", "packaging/install_linux_desktop.sh"])
def test_posix_launchers_are_executable(name):
    path = ROOT / name
    assert os.access(path, os.X_OK), f"{name} is not executable (chmod +x)"
    assert path.read_bytes().startswith(b"#!"), f"{name} has no shebang"
    assert b"\r\n" not in path.read_bytes(), f"{name} has CRLF line endings"


def test_batch_launcher_uses_crlf_and_no_errorlevel_in_block():
    data = (ROOT / "flowdesk.bat").read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n"), "flowdesk.bat must use CRLF line endings"
    # %errorlevel% inside a parenthesised block is expanded when the block is parsed
    assert b"%errorlevel%" not in data.lower()


def test_packaging_files_present():
    for name in ("flowdesk.spec", "build.py", "make_icons.py", "flowdesk_main.py", "flowdesk.desktop", "icon.png", "icon.ico"):
        assert (ROOT / "packaging" / name).is_file(), name
    desktop = (ROOT / "packaging" / "flowdesk.desktop").read_text(encoding="utf-8")
    assert desktop.startswith("[Desktop Entry]")
    assert "Icon=flowdesk" in desktop and "Exec=" in desktop


def test_module_help_runs_from_any_cwd(tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    out = subprocess.run(
        [sys.executable, "-m", "flowdesk", "--help"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "usage: flowdesk" in out.stdout
    out = subprocess.run(
        [sys.executable, str(PKG_DIR / "app.py"), "--version"], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("FlowDesk ")


def test_create_app_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from flowdesk import APP_NAME
    from flowdesk.app import create_app, main

    app = create_app([])
    assert isinstance(app, QApplication)
    assert create_app([]) is app  # idempotent, reuses the instance
    assert QApplication.applicationName() == APP_NAME
    assert not app.windowIcon().isNull()
    assert main(["flowdesk", "--version"]) == 0


def test_sample_to_dict_survives_paths_on_different_drives(monkeypatch, tmp_path):
    from flowdesk.core import serialization
    from flowdesk.core.model import Sample

    def relpath(*_args, **_kw):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(serialization.os.path, "relpath", relpath)
    sample = Sample(tmp_path / "a.fcs", 1)
    d = serialization.sample_to_dict(sample, tmp_path / "ws")
    assert d["rel_path"] is None
    assert d["path"] == str(tmp_path / "a.fcs")
