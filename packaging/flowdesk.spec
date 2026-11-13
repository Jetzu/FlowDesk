# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for FlowDesk: onedir, windowed, on macOS / Windows / Linux.

Build with ``python packaging/build.py`` (or ``pyinstaller packaging/flowdesk.spec`` from the
repository root). Output: dist/FlowDesk/ (FlowDesk.exe on Windows, FlowDesk on Linux) and,
on macOS, dist/FlowDesk.app.
"""
import importlib.util
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected by PyInstaller
PKG = ROOT / "packaging"
sys.path.insert(0, str(ROOT))
from flowdesk import __version__  # noqa: E402

# Analysis libraries that FlowDesk imports lazily (scikit-learn: k-means, kNN graphs, the t-SNE
# fallback; openTSNE; umap-learn), each with the runtime dependencies it needs. FlowDesk never
# imports those dependencies itself, so they are not detected from the sources: they come with
# their entry library. An entry library is bundled when the FlowDesk sources import it and it is
# installed (so the "analysis" extra is picked up automatically); the dependencies of the entry
# libraries that are not bundled are excluded, so that unrelated packages in the build
# environment do not end up in the app.
ANALYSIS_LIBS = {
    "sklearn": ["joblib", "threadpoolctl"],
    "openTSNE": ["sklearn", "joblib", "threadpoolctl"],
    "umap": ["sklearn", "joblib", "threadpoolctl", "pynndescent", "numba", "llvmlite", "tqdm"],
}
_src = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "flowdesk").rglob("*.py"))


def _imported_by_flowdesk(module):
    return re.search(rf"^\s*(import|from)\s+{re.escape(module)}\b", _src, re.M) is not None


def _installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


BUNDLED_LIBS = [m for m in ANALYSIS_LIBS if _imported_by_flowdesk(m) and _installed(m)]
_keep = set(BUNDLED_LIBS).union(*(ANALYSIS_LIBS[m] for m in BUNDLED_LIBS))
_unused = sorted(set(ANALYSIS_LIBS).union(*ANALYSIS_LIBS.values()) - _keep)
print(f"FlowDesk spec: bundling {', '.join(BUNDLED_LIBS) or 'no analysis libraries'}"
      f"; excluding {', '.join(_unused) or 'nothing'}")

QT_EXCLUDES = [
    f"PySide6.{m}"
    for m in (
        "QtWebEngine", "QtWebEngineCore", "QtWebEngineWidgets", "QtWebEngineQuick", "QtWebChannel",
        "QtWebSockets", "QtWebView", "QtQuick", "QtQuick3D", "QtQuickControls2", "QtQuickWidgets",
        "QtQuickTest", "QtQml", "Qt3DCore", "Qt3DRender", "Qt3DInput", "Qt3DLogic", "Qt3DAnimation",
        "Qt3DExtras", "QtMultimedia", "QtMultimediaWidgets", "QtSpatialAudio", "QtPdf", "QtPdfWidgets",
        "QtCharts", "QtDataVisualization", "QtGraphs", "QtGraphsWidgets", "QtBluetooth", "QtNfc",
        "QtPositioning", "QtLocation", "QtSensors", "QtSerialPort", "QtSerialBus", "QtRemoteObjects",
        "QtScxml", "QtStateMachine", "QtTextToSpeech", "QtDesigner", "QtHelp", "QtUiTools", "QtSql",
        "QtTest", "QtOpenGLWidgets", "QtNetworkAuth", "QtHttpServer", "QtAxContainer",
    )
]
EXCLUDES = QT_EXCLUDES + _unused + [
    "tkinter", "matplotlib", "IPython", "pytest", "_pytest", "tests", "flowdesk.tests",
    "PyInstaller", "setuptools", "pip", "numpy.tests", "scipy.tests",
]

hiddenimports = (
    ["contourpy", "contourpy._contourpy", "scipy.ndimage", "scipy.optimize",
     "scipy.special._special_ufuncs",
     "PySide6.QtSvg", "PySide6.QtPrintSupport"]
    + collect_submodules("flowdesk")
)
try:  # the MCP server (flowdesk --mcp) lets the packaged app serve Claude Code / Codex on its own
    import mcp  # noqa: F401

    # mcp.cli (the `mcp` dev command) exits at import without typer and is not needed by the server
    hiddenimports += collect_submodules("mcp", filter=lambda name: not name.startswith("mcp.cli"))
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
if "openTSNE" in BUNDLED_LIBS:
    # its Cython modules import openTSNE._matrix_mul (and the bundled annoy) at C level,
    # which the module graph does not see
    hiddenimports += collect_submodules("openTSNE")

if sys.platform == "win32":
    ICON = str(PKG / "icon.ico")
elif sys.platform == "darwin":
    ICON = str(PKG / "icon.icns")
else:
    ICON = str(PKG / "icon.png")

a = Analysis(  # noqa: F821
    [str(PKG / "flowdesk_main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[(str(PKG / "icon.png"), "packaging"), (str(ROOT / "flowdesk" / "agent" / "skills"), "flowdesk/agent/skills")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)


# Keep only the Qt plugin families a widgets app needs (platforms, imageformats, iconengines,
# styles, platformthemes, xcbglintegrations / wayland on Linux, printsupport).
_KEEP_PLUGINS = ("platforms", "imageformats", "iconengines", "styles", "platformthemes",
                 "platforminputcontexts", "xcbglintegrations", "wayland-shell-integration",
                 "wayland-decoration-client", "wayland-graphics-integration-client", "printsupport",
                 "egldeviceintegrations", "generic", "tls", "networkinformation")
_DROP_QT_LIBS = re.compile(r"Qt6?(WebEngine|Quick|Qml|3D|Multimedia|Pdf|Charts|Graphs|DataVis|Designer|"
                           r"VirtualKeyboard|Location|Positioning|ShaderTools|Sensors|Bluetooth|Sql)", re.I)


def _keep(entry):
    dest = entry[0].replace("\\", "/")
    if "/plugins/" in dest:
        family = dest.split("/plugins/", 1)[1].split("/", 1)[0]
        if family not in _KEEP_PLUGINS:
            return False
    if "/translations/" in dest and "qtbase_" not in dest:
        return False
    if "PySide6/" in dest and _DROP_QT_LIBS.search(dest.rsplit("/", 1)[-1]):
        return False
    return True


a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FlowDesk",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=ICON,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FlowDesk",
)

if sys.platform == "darwin":
    app = BUNDLE(  # noqa: F821
        coll,
        name="FlowDesk.app",
        icon=ICON,
        bundle_identifier="org.flowdesk.FlowDesk",
        version=__version__,
        info_plist={
            "CFBundleName": "FlowDesk",
            "CFBundleDisplayName": "FlowDesk",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleDocumentTypes": [
                {"CFBundleTypeName": "FlowDesk Workspace", "CFBundleTypeExtensions": ["fdw"],
                 "CFBundleTypeRole": "Editor", "LSHandlerRank": "Owner"},
                {"CFBundleTypeName": "FCS Data File", "CFBundleTypeExtensions": ["fcs"],
                 "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate"},
                {"CFBundleTypeName": "FlowJo Workspace", "CFBundleTypeExtensions": ["wsp"],
                 "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate"},
                {"CFBundleTypeName": "FlowDesk Workspace Template", "CFBundleTypeExtensions": ["fdwt"],
                 "CFBundleTypeRole": "Editor", "LSHandlerRank": "Owner"},
                {"CFBundleTypeName": "FlowJo Workspace Template", "CFBundleTypeExtensions": ["wspt"],
                 "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate"},
            ],
        },
    )
