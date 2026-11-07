"""Build a standalone FlowDesk with PyInstaller (run with the project's Python).

    python packaging/build.py            # -> dist/FlowDesk[/.app]
    python packaging/build.py --zip      # also writes dist/FlowDesk-<os>.zip
    python packaging/build.py --keep-build
    python packaging/build.py --check-only   # only run the frozen-app check on dist/

Installs PyInstaller into the running interpreter if it is missing (uv if available, else pip).
After the build, the frozen app runs k-means, FlowSOM and the kNN graph (scikit-learn), and
t-SNE / UMAP when openTSNE / umap-learn are installed in the build environment, so a library
that is missing from the bundle fails the build instead of the user's analysis.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
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


def executable(target: Path) -> Path:
    if target.suffix == ".app":
        return target / "Contents" / "MacOS" / "FlowDesk"
    return target / ("FlowDesk.exe" if sys.platform == "win32" else "FlowDesk")


# Runs inside the frozen app (see check_frozen_app); writes a report and exits with os._exit so
# that a windowed build never shows a traceback dialog.
FROZEN_CHECK = r"""
import os, traceback
with open(os.environ["FLOWDESK_CHECK_OUT"], "w", encoding="utf-8") as fh:
    fh.write("started")
report, code = [], 1
try:
    import numpy as np
    from flowdesk.core import analysis
    wanted = set(filter(None, os.environ.get("FLOWDESK_CHECK_LIBS", "").split(",")))
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (150, 3)), rng.normal(8.0, 1.0, (150, 3))])
    two = [150, 150]
    from sklearn.cluster import KMeans  # noqa: F401
    from sklearn.manifold import TSNE  # noqa: F401
    import joblib, threadpoolctl  # noqa: F401,E401
    assert analysis.has_sklearn()
    assert sorted(np.bincount(analysis.run_kmeans(X, 2))[1:].tolist()) == two
    report.append("k-means (scikit-learn) ok")
    labels, _codes, _meta = analysis.run_flowsom(X, grid=(4, 4), n_metaclusters=2, iterations=5)
    assert sorted(np.bincount(labels)[1:].tolist()) == two
    report.append("FlowSOM ok")
    assert analysis.knn_jaccard_graph(X, 10).shape == (300, 300)
    report.append("kNN graph (sklearn.neighbors) ok")
    if "openTSNE" in wanted:
        assert analysis.has_opentsne()
        emb, model = analysis.fit_tsne(X, perplexity=10, iterations=250)
        assert emb.shape == (300, 2) and model is not None and np.isfinite(emb).all()
        report.append("t-SNE (openTSNE) ok")
    if "umap" in wanted:
        assert analysis.has_umap()
        emb, _model = analysis.fit_umap(X, n_neighbors=10)
        assert emb.shape == (300, 2) and np.isfinite(emb).all()
        report.append("UMAP (umap-learn) ok")
    code = 0
except BaseException:
    report.append(traceback.format_exc())
with open(os.environ["FLOWDESK_CHECK_OUT"], "w", encoding="utf-8") as fh:
    fh.write("\n".join(report))
os._exit(code)
"""


def analysis_libs() -> list[str]:
    """Optional analysis libraries installed here, which the spec therefore bundles."""
    return [m for m in ("openTSNE", "umap") if importlib.util.find_spec(m) is not None]


# The interpreter flags a frozen app reports (PyInstaller 6 bootloader: -B -S -I), which the
# runtime hook expects before "-c"; the others are fallbacks for other PyInstaller versions.
FLAG_CANDIDATES = (["-B", "-S", "-I"], ["-B", "-S", "-E", "-s"], ["-B", "-I"], ["-I"], [])


def check_frozen_app(target: Path, timeout: float = 900.0, start_timeout: float = 120.0) -> None:
    """Run FROZEN_CHECK inside the built app.

    The frozen executable has no ``-c`` of its own, but PyInstaller's multiprocessing runtime
    hook (active because flowdesk_main calls multiprocessing.freeze_support()) executes a
    ``-c`` command that starts like multiprocessing's resource-tracker launch line, provided
    the arguments before it are the app's interpreter flags. That runs our code with the
    bundle's own modules only. When the flags do not match, the app starts normally instead;
    it is then stopped and the next candidate is tried."""
    import time

    exe = executable(target)
    if not exe.exists():
        raise SystemExit(f"frozen-app check: {exe} not found")
    libs = analysis_libs()
    command = "from multiprocessing.resource_tracker import main; exec(%r, {'__name__': '__flowdesk_check__'})" % FROZEN_CHECK
    with tempfile.TemporaryDirectory() as tmp:
        out, log_path = Path(tmp) / "report.txt", Path(tmp) / "output.txt"
        env = dict(os.environ, FLOWDESK_CHECK_OUT=str(out), FLOWDESK_CHECK_LIBS=",".join(libs))
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.pop("PYTHONPATH", None)
        for flags in FLAG_CANDIDATES:
            out.unlink(missing_ok=True)
            with open(log_path, "wb") as log:
                proc = subprocess.Popen([str(exe), *flags, "-c", command], env=env, cwd=tmp, stdout=log, stderr=subprocess.STDOUT)
                t0 = time.monotonic()
                while not out.exists() and proc.poll() is None and time.monotonic() - t0 < start_timeout:
                    time.sleep(0.2)
                if not out.exists():  # not intercepted: the GUI started (or the app died)
                    proc.kill()
                    proc.wait()
                    continue
                try:
                    returncode = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    raise SystemExit(f"frozen-app check timed out after {timeout:.0f} s") from None
            report = out.read_text(encoding="utf-8")
            if returncode != 0:
                tail = log_path.read_bytes().decode(errors="replace")[-2000:]
                raise SystemExit(f"frozen-app check FAILED (exit {returncode}):\n{report}\n{tail}")
            print(f"frozen-app check passed ({' '.join(flags)}): " + "; ".join(report.splitlines()))
            return
        tail = log_path.read_bytes().decode(errors="replace")[-2000:]
    raise SystemExit(f"frozen-app check: the app did not run the check with any of {FLAG_CANDIDATES}\n{tail}")


def opened_suffixes() -> set[str]:
    """File types the app opens (flowdesk/app.py WORKSPACE_SUFFIXES, plus .fcs), read
    from the source so that this does not import Qt."""
    import ast

    tree = ast.parse((ROOT / "flowdesk" / "app.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "WORKSPACE_SUFFIXES" for t in node.targets):
            return {s.lstrip(".").lower() for s in ast.literal_eval(node.value)} | {"fcs"}
    raise SystemExit("WORKSPACE_SUFFIXES not found in flowdesk/app.py")


def check_document_types(target: Path) -> None:
    """macOS: every file type the app opens is declared in Info.plist, so Finder and the
    Dock hand it to FlowDesk."""
    if target.suffix != ".app":
        return
    import plistlib

    with open(target / "Contents" / "Info.plist", "rb") as fh:
        info = plistlib.load(fh)
    declared = {e.lower() for t in info.get("CFBundleDocumentTypes", []) for e in t.get("CFBundleTypeExtensions", [])}
    missing = sorted(opened_suffixes() - declared)
    if missing:
        raise SystemExit(f"Info.plist declares no document type for: {', '.join(missing)}")
    print("document types declared: " + ", ".join(sorted(declared)))


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
    ap.add_argument("--no-check", action="store_true", help="skip running the analysis check in the frozen app")
    ap.add_argument("--check-only", action="store_true", help="only run the frozen-app check on an existing build")
    args = ap.parse_args(argv)

    if args.check_only:
        check_document_types(artifact())
        check_frozen_app(artifact())
        return 0
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
    check_document_types(target)
    if not args.no_check:
        check_frozen_app(target)
    if args.zip:
        z = make_zip()
        print(f"zipped {z}  ({z.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
