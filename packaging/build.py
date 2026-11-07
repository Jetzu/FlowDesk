"""Build a standalone FlowDesk with PyInstaller (run with the project's Python).

    python packaging/build.py            # -> dist/FlowDesk[/.app]
    python packaging/build.py --zip      # also writes dist/FlowDesk-<os>.zip
    python packaging/build.py --keep-build

Installs PyInstaller into the running interpreter if it is missing (uv if available, else pip).
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"
DIST = ROOT / "dist"
BUILD = ROOT / "build"


def os_tag() -> str:
    return {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")


def ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is not None:
        return
    print("PyInstaller not found - installing it into", sys.executable)
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, "pyinstaller>=6.10"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "pyinstaller>=6.10"]
    subprocess.run(cmd, check=True)


def ensure_icons() -> None:
    wanted = ["icon.png", "icon.ico"] + (["icon.icns"] if sys.platform == "darwin" else [])
    if not all((PKG / f).exists() for f in wanted):
        subprocess.run([sys.executable, str(PKG / "make_icons.py")], check=True, cwd=ROOT)


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def artifact() -> Path:
    app = DIST / "FlowDesk.app"
    return app if sys.platform == "darwin" and app.exists() else DIST / "FlowDesk"


def make_zip() -> Path:
    target = artifact()
    out = DIST / f"FlowDesk-{os_tag()}.zip"
    out.unlink(missing_ok=True)
    if sys.platform == "darwin":  # ditto keeps symlinks / bundle metadata intact
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(target), str(out)], check=True)
    else:
        shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=DIST, base_dir=target.name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", action="store_true", help="also create dist/FlowDesk-<os>.zip")
    ap.add_argument("--keep-build", action="store_true", help="keep the intermediate build/ directory")
    args = ap.parse_args(argv)

    ensure_pyinstaller()
    ensure_icons()
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(BUILD), str(PKG / "flowdesk.spec")],
        check=True, cwd=ROOT,
    )
    if not args.keep_build:
        shutil.rmtree(BUILD, ignore_errors=True)
    if sys.platform == "darwin" and (DIST / "FlowDesk.app").exists():
        shutil.rmtree(DIST / "FlowDesk", ignore_errors=True)  # the .app holds the same files
    target = artifact()
    print(f"built {target}  ({dir_size(target) / 1e6:.0f} MB)")
    if args.zip:
        z = make_zip()
        print(f"zipped {z}  ({z.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
