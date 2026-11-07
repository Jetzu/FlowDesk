"""Render the FlowDesk app icon into packaging/icon.png, icon.ico and (on macOS) icon.icns.

The artwork is the procedurally drawn ``flowdesk.ui.icons.pixmap("app", size)``; this script
only rasterises it, so it runs headless (QT_QPA_PLATFORM=offscreen is set automatically).

    python packaging/make_icons.py
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
ICNS_SIZES = (16, 32, 128, 256, 512)  # each also rendered @2x


def _png_bytes(size: int) -> bytes:
    from flowdesk.ui.icons import pixmap

    buf = QByteArray()
    dev = QBuffer(buf)
    dev.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pixmap("app", size).save(dev, "PNG"):
        raise RuntimeError(f"could not encode {size}px icon")
    dev.close()
    return bytes(buf.data())


def write_png(path: Path, size: int = 256) -> None:
    path.write_bytes(_png_bytes(size))


def write_ico(path: Path, sizes=ICO_SIZES) -> None:
    """Multi-resolution .ico with PNG-compressed entries (supported since Windows Vista)."""
    images = [(s, _png_bytes(s)) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = 6 + 16 * len(images)
    entries, blobs = b"", b""
    for s, data in images:
        dim = 0 if s >= 256 else s  # 0 means 256 in the ICONDIRENTRY
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    path.write_bytes(header + entries + blobs)


def write_icns(path: Path) -> bool:
    """Build an .icns with Apple's iconutil; returns False when iconutil is unavailable."""
    iconutil = shutil.which("iconutil")
    if iconutil is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "FlowDesk.iconset"
        iconset.mkdir()
        for s in ICNS_SIZES:
            (iconset / f"icon_{s}x{s}.png").write_bytes(_png_bytes(s))
            (iconset / f"icon_{s}x{s}@2x.png").write_bytes(_png_bytes(2 * s))
        subprocess.run([iconutil, "-c", "icns", str(iconset), "-o", str(path)], check=True)
    return True


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 - needed for QPixmap
    write_png(HERE / "icon.png", 256)
    write_ico(HERE / "icon.ico")
    print(f"wrote {HERE / 'icon.png'}\nwrote {HERE / 'icon.ico'}")
    if write_icns(HERE / "icon.icns"):
        print(f"wrote {HERE / 'icon.icns'}")
    else:
        print("iconutil not found (not macOS): icon.icns not regenerated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
