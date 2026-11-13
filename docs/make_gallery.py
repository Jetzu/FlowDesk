#!/usr/bin/env python
"""Regenerate the screenshots used in README.md (docs/images/*.png).

FlowDesk renders itself offscreen, so no display is needed:

    python docs/make_gallery.py --data "<folder with 19-Aug-2025.wsp>" --controls "<folder of single-stain controls>"

Shots that need real data (workspace, layout, tsne, flowsom_heatmap, compensation) are skipped when the matching
folder is not given. The cell-cycle, proliferation and plate shots use generated test data and always run.
Use --only NAME [NAME ...] to redo a few; --list shows the names.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from flowdesk.app import create_app  # noqa: E402
from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.model import Population, Workspace  # noqa: E402

OUT = ROOT / "docs" / "images"
APP: QApplication | None = None


def pump(n: int = 3) -> None:
    for _ in range(n):
        APP.processEvents()


def wait_until(condition, timeout: float = 90.0) -> bool:
    """Run a real Qt event loop until `condition()` is true or `timeout` seconds pass.

    A real loop (like the application's own) lets the background loader's worker threads run;
    polling with QTest.qWait would hold Python's lock and starve them."""
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    result = {"ok": False}
    poll = QTimer()
    poll.setInterval(100)

    def check():
        if condition():
            result["ok"] = True
            loop.quit()

    poll.timeout.connect(check)
    poll.start()
    QTimer.singleShot(int(timeout * 1000), loop.quit)
    loop.exec()
    poll.stop()
    return result["ok"]


def tree_is_loading(tree) -> bool:
    from PySide6.QtWidgets import QTreeWidgetItemIterator

    it = QTreeWidgetItemIterator(tree)
    while it.value() is not None:
        if it.value().text(2) in ("\u2026", "..."):
            return True
        it += 1
    return False


def save(widget, name: str, size: tuple[int, int] | None = None) -> None:
    if size:
        widget.resize(*size)
    pump()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    widget.grab().save(str(path))
    print(f"  wrote {path.relative_to(ROOT)}")


def new_window():
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    win.resize(1280, 800)
    win.show()
    pump()
    return win


def close_window(win) -> None:
    for w in list(win.graph_windows.values()):
        w.close()
    win.controller.workspace.modified = False
    win.close()
    pump()


def find_population(sample, name: str):
    stack = list(sample.root.children)
    while stack:
        p = stack.pop(0)
        if p.name == name:
            return p
        stack.extend(p.children)
    return None


# --------------------------------------------------------------------------- real data
def shot_workspace(data: Path) -> None:
    win = new_window()
    try:
        win._open_path(str(data / "19-Aug-2025.wsp"))
        pump()
        c = win.controller
        ws = c.workspace
        try:  # a custom group whose gating template is shown above its members
            group = c.create_group("T-cell donors", ws.samples, color="#1f5fa8")
            s0 = ws.samples[0]
            parent = ""
            for name in ("Lymphocytes", "Single Cells"):
                pop = find_population(s0, name)
                if pop is None or pop.gate is None:
                    break
                ws.apply_population_to_group(group, parent, Population(name, pop.gate))
                parent = f"{parent}/{name}"
            c.set_current_group(group)
        except Exception:
            traceback.print_exc()
        win.ribbon.tabs.setCurrentIndex(1)
        win.tree.expandAll()
        wait_until(lambda: not tree_is_loading(win.tree))  # the loader fills in the counts in the background
        win.tree.expandAll()
        save(win, "workspace", (1280, 800))
    finally:
        close_window(win)


def shot_layout(data: Path) -> None:
    win = new_window()
    try:
        win._open_path(str(data / "19-Aug-2025.wsp"))
        pump()
        s0 = win.controller.workspace.samples[0]
        picks = []
        for name in ("Lymphocytes", "Single Cells", "CD4+", "CD8+", "CD62L+", "central memory"):
            pop = find_population(s0, name)
            if pop is not None:
                picks.append(pop)
        editor = win.show_layout_editor()
        for pop in picks[:6]:
            editor.add_population(s0, pop)
        pump(5)
        save(editor, "layout", (1100, 760))
        editor.close()
    finally:
        close_window(win)


def shot_tsne_and_flowsom(data: Path) -> None:
    from flowdesk.ui import tools_analysis as T

    win = new_window()
    try:
        win._open_path(str(data / "19-Aug-2025.wsp"))
        pump()
        c = win.controller
        s0 = c.workspace.samples[0]
        pop = find_population(s0, "CD4+")
        if pop is None:
            raise RuntimeError("no CD4+ population in the workspace")
        params = [p for p in T.default_parameters([pop]) if p.endswith("-A") and not p.upper().startswith(("FSC", "SSC", "HDR"))]
        T.run_dimreduction(c, "tsne", populations=[pop], params=params, max_events=6000, options={"perplexity": 30.0, "iterations": 500}, open_graph=False)
        name, created, result = T.run_clustering(c, "flowsom", populations=[pop], params=params, options={"grid": (6, 6), "n_metaclusters": 8, "iterations": 10})
        graph = T.open_result_graph(c, pop, "tSNE_1", "tSNE_2")
        clusters = [find_population(s0, n) for n in created]
        graph.plot.set_overlays([(s0, p) for p in clusters if p is not None])
        pump(5)
        save(graph, "tsne", (760, 640))
        graph.close()
        data_for_heatmap = T.cluster_heatmap_data(c, result)
        heat = T.ClusterHeatmapDialog(data_for_heatmap, title=f"{name} - cluster medians")
        heat.show()
        save(heat, "flowsom_heatmap", (760, 520))
        heat.close()
    finally:
        close_window(win)


def shot_compensation(controls: Path) -> None:
    from flowdesk.ui.compensation import CompensationWindow, CompensationWizard
    from flowdesk.ui.controller import WorkspaceController

    ws = Workspace()
    for p in sorted(controls.glob("*.fcs")):
        ws.add_sample(p)
    c = WorkspaceController(ws)
    wiz = CompensationWizard(c)
    wiz.compute()
    name = "Single-stain controls"
    wiz.matrix_page.name_edit.setText(name)
    wiz.accept()  # Finish: stores the matrix, applies it and remembers the control set the plots need
    wiz.deleteLater()
    win = CompensationWindow(c)
    win.show()
    win.select_matrix(name)
    win.tabs.setCurrentIndex(1)  # control plots
    wait_until(lambda: False, timeout=4.0)  # the 8 x 8 panel renders in the background
    save(win, "compensation", (1100, 720))
    win.close()
    ws.modified = False


# ------------------------------------------------------------------- generated data
def _write(path: Path, columns: dict[str, np.ndarray], stains: dict[str, str] | None = None) -> Path:
    names = list(columns)
    events = np.column_stack([columns[n] for n in names]).astype(np.float32)
    channels = [Channel(i + 1, n, (stains or {}).get(n, ""), 262144) for i, n in enumerate(names)]
    write_fcs(path, events, channels, {"$CYT": "Synthetic"})
    return path


def _window_with(files: list[Path]):
    win = new_window()
    ws = Workspace()
    for f in files:
        ws.add_sample(f)
    win.controller.set_workspace(ws)
    win.controller.samplesChanged.emit()
    pump()
    return win


def shot_cell_cycle(tmp: Path) -> None:
    rng = np.random.default_rng(1)
    n = 24000
    cycling = int(n * 0.85)
    g1 = rng.normal(100, 4, int(cycling * 0.60))
    g2 = rng.normal(200, 8, int(cycling * 0.25))
    s = rng.uniform(100, 200, int(cycling * 0.15)) + rng.normal(0, 5, int(cycling * 0.15))
    debris = rng.exponential(22, int(n * 0.10))
    doublets = rng.normal(300, 10, int(n * 0.05))
    dna = np.concatenate([g1, s, g2, debris[debris < 95], doublets]) * 600
    dna = dna[dna > 0]
    rng.shuffle(dna)
    path = _write(tmp / "DNA content.fcs", {"FSC-A": rng.normal(60000, 8000, dna.size), "SSC-A": rng.normal(30000, 6000, dna.size), "PI-A": dna}, {"PI-A": "DNA (PI)"})
    win = _window_with([path])
    try:
        dlg = win.extensions["flowdesk.ui.tools_biology"].open("cell_cycle")
        dlg.model_combo.setCurrentIndex(1)  # Watson pragmatic
        dlg.debris_check.setChecked(True)
        dlg.aggregates_check.setChecked(True)
        dlg.run()
        save(dlg, "cell_cycle", (860, 620))
        dlg.close()
    finally:
        close_window(win)


def shot_proliferation(tmp: Path) -> None:
    rng = np.random.default_rng(2)
    fractions = [0.10, 0.20, 0.30, 0.25, 0.15]
    sizes = (np.array(fractions) * 30000).astype(int)
    dye = np.concatenate([50000 / 2 ** k * rng.lognormal(0, 0.12, m) for k, m in enumerate(sizes)])
    rng.shuffle(dye)
    path = _write(tmp / "CellTrace dilution.fcs", {"FSC-A": rng.normal(60000, 8000, dye.size), "SSC-A": rng.normal(30000, 6000, dye.size), "CTV-A": dye}, {"CTV-A": "CellTrace Violet"})
    win = _window_with([path])
    try:
        dlg = win.extensions["flowdesk.ui.tools_biology"].open("proliferation")
        dlg.run()
        save(dlg, "proliferation", (860, 620))
        dlg.close()
    finally:
        close_window(win)


def shot_plate(tmp: Path) -> None:
    rng = np.random.default_rng(3)
    files = []
    for r, row in enumerate("ABCDEFGH"):
        for col in range(1, 13):
            frac = np.clip(0.85 - 0.065 * (col - 1) + rng.normal(0, 0.03) + 0.01 * r, 0.05, 0.95)
            n = 400
            positive = rng.random(n) < frac
            cd3 = np.where(positive, rng.normal(30000, 6000, n), rng.normal(3000, 1500, n))
            files.append(_write(tmp / f"Plate1_{row}{col}.fcs", {"FSC-A": rng.normal(60000, 8000, n), "SSC-A": rng.normal(30000, 6000, n), "CD3-A": cd3}, {"CD3-A": "CD3"}))
    win = _window_with(files)
    try:
        ws = win.controller.workspace
        ws.apply_population_to_group(ws.groups[0], "", Population("CD3+", RectangleGate(["CD3-A"], [15000], [None])))
        win.controller.gatingChanged.emit(None)
        pump()
        editor = win.extensions["flowdesk.ui.plate_editor"].show_editor()
        editor.set_format(96)
        editor.set_statistic("/CD3+", "Freq. of Parent")
        editor.refresh()
        save(editor, "plate", (1000, 620))
        editor.close()
    finally:
        close_window(win)


SHOTS = {
    "workspace": ("data", shot_workspace),
    "layout": ("data", shot_layout),
    "tsne": ("data", shot_tsne_and_flowsom),
    "compensation": ("controls", shot_compensation),
    "cell_cycle": ("tmp", shot_cell_cycle),
    "proliferation": ("tmp", shot_proliferation),
    "plate": ("tmp", shot_plate),
}


def main() -> int:
    global APP
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("FLOWDESK_GALLERY_DATA"), help="folder with 19-Aug-2025.wsp and its FCS files")
    ap.add_argument("--controls", default=os.environ.get("FLOWDESK_GALLERY_CONTROLS"), help="folder of single-stain control FCS files")
    ap.add_argument("--only", nargs="*", help="shot names to (re)generate")
    ap.add_argument("--list", action="store_true", help="list the shot names")
    args = ap.parse_args()
    if args.list:
        print("\n".join(f"{k:14s} needs {v[0]}" for k, v in SHOTS.items()))
        return 0
    APP = create_app([])
    wanted = args.only or list(SHOTS)
    failed = []
    with tempfile.TemporaryDirectory(prefix="flowdesk-gallery-") as tmp:
        for name in wanted:
            kind, fn = SHOTS[name]
            if kind == "tmp":
                arg = Path(tmp)
            else:
                folder = args.data if kind == "data" else args.controls
                if not folder or not Path(folder).is_dir():
                    print(f"{name}: skipped (no --{kind} folder)")
                    continue
                arg = Path(folder).expanduser()
            print(f"{name}:")
            try:
                fn(arg)
            except Exception:
                traceback.print_exc()
                failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # Qt objects are torn down by the OS; avoids shutdown-order aborts
