#!/usr/bin/env python
"""Write a small synthetic test set for trying FlowDesk out.

    python docs/make_testset.py [--out FOLDER] [--seed N]

The folder it writes ("FlowDesk testset" in the current directory by default) holds

  FlowDesk testset.fdw       a ready workspace: groups, a gating tree and an experiment annotation
  samples/                   6 four-colour T-cell samples (3 donors, control and stimulated) with an
                             acquisition spillover matrix, keywords (Donor, Condition, WELL ID) and a
                             flow-rate dip at the start of one run for the data-cleaning tool
  compensation controls/     an unstained tube and four single-stain controls for the compensation wizard
  extras/                    a DNA-content sample (cell cycle), a CellTrace dilution sample (proliferation)
                             and a six-peak bead sample (MESF calibration)
  READ ME.txt                what to try

All data are generated, so the set can be shared. Seeded: the same seed gives the same files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flowdesk.core.fcs import Channel, write_fcs  # noqa: E402
from flowdesk.core.gates import PolygonGate, RectangleGate, make_quadrant_gates  # noqa: E402
from flowdesk.core.model import Population, Workspace  # noqa: E402
from flowdesk.core.serialization import save_workspace  # noqa: E402

CHANNELS = ["FSC-A", "FSC-H", "SSC-A", "FITC-A", "PE-A", "APC-A", "PE-Cy7-A", "Time"]
STAINS = {"FITC-A": "CD3", "PE-A": "CD4", "APC-A": "CD8", "PE-Cy7-A": "CD69"}
FLUORS = ["FITC-A", "PE-A", "APC-A", "PE-Cy7-A"]
# row = the fluorochrome, column = the detector it spills into (observed = true @ SPILL)
SPILL = np.array(
    [
        [1.000, 0.140, 0.003, 0.000],
        [0.010, 1.000, 0.025, 0.220],
        [0.000, 0.002, 1.000, 0.070],
        [0.000, 0.000, 0.010, 1.000],
    ]
)
DONORS = {  # CD4 fraction, CD8 fraction of T cells; CD69+ fraction of T cells (control, stimulated)
    "Donor1": (0.62, 0.31, 0.07, 0.55),
    "Donor2": (0.50, 0.42, 0.05, 0.62),
    "Donor3": (0.70, 0.24, 0.09, 0.48),
}


def _channels() -> list[Channel]:
    return [Channel(i + 1, n, STAINS.get(n, ""), 262144) for i, n in enumerate(CHANNELS)]


def _spillover_keyword() -> str:
    values = ",".join(f"{v:g}" for v in SPILL.flatten())
    return f"{len(FLUORS)},{','.join(FLUORS)},{values}"


def _fluor(rng, n: int, positive: np.ndarray, median: float, sigma: float) -> np.ndarray:
    """Compensated signal of one marker: a dim noisy negative and a lognormal positive."""
    neg = rng.lognormal(np.log(250), 0.7, n) + rng.normal(0, 150, n)
    pos = rng.lognormal(np.log(median), sigma, n)
    return np.where(positive, pos, neg)


def make_sample(rng, n: int, cd4: float, cd8: float, cd69: float, dip: bool = False) -> np.ndarray:
    kind = rng.choice(4, size=n, p=[0.10, 0.13, 0.10, 0.67])  # debris, myeloid, B cells, T cells
    debris, myeloid, bcell, tcell = (kind == k for k in range(4))
    fsc = np.where(debris, rng.lognormal(np.log(12000), 0.4, n), np.where(myeloid, rng.normal(120000, 18000, n), rng.normal(80000, 9000, n)))
    ssc = np.where(debris, rng.lognormal(np.log(5000), 0.4, n), np.where(myeloid, rng.normal(90000, 20000, n), rng.normal(25000, 6000, n)))
    doublet = rng.random(n) < 0.05
    fsc_h = np.where(doublet, fsc * rng.uniform(0.5, 0.7, n), fsc * (1 + rng.normal(0, 0.04, n)))
    subset = rng.random(n)
    is_cd4 = tcell & (subset < cd4)
    is_cd8 = tcell & (subset >= cd4) & (subset < cd4 + cd8)
    activated = (tcell | bcell) & (rng.random(n) < cd69)
    true = np.column_stack(
        [
            _fluor(rng, n, tcell, 30000, 0.35),  # CD3
            np.where(myeloid, _fluor(rng, n, np.ones(n, bool), 2500, 0.4), _fluor(rng, n, is_cd4, 25000, 0.4)),  # CD4
            _fluor(rng, n, is_cd8, 22000, 0.4),  # CD8
            _fluor(rng, n, activated, 9000, 0.5),  # CD69
        ]
    )
    observed = true @ SPILL
    rate = np.full(n, 1.0)
    if dip:
        rate[: n // 10] = 0.2  # the first tenth of the run is acquired at a fifth of the rate
    time = np.cumsum(rng.exponential(1.0 / rate)) * 0.4
    return np.column_stack([fsc, fsc_h, ssc, observed, time])


def make_controls(rng, n: int = 6000) -> dict[str, np.ndarray]:
    out = {}
    for i, name in enumerate(["Unstained"] + FLUORS):
        true = np.zeros((n, 4))
        if i:
            positive = rng.random(n) < 0.5
            true[:, i - 1] = np.where(positive, rng.lognormal(np.log(40000), 0.3, n), rng.lognormal(np.log(250), 0.7, n))
        else:
            true += rng.lognormal(np.log(250), 0.7, (n, 4))
        observed = true @ SPILL + rng.normal(0, 120, (n, 4))
        fsc = rng.normal(80000, 9000, n)
        out[name] = np.column_stack([fsc, fsc * (1 + rng.normal(0, 0.04, n)), rng.normal(25000, 6000, n), observed, np.cumsum(rng.exponential(1.0, n)) * 0.4])
    return out


def _write(path: Path, events: np.ndarray, keywords: dict[str, str]) -> None:
    write_fcs(path, events.astype(np.float32), _channels(), {"$CYT": "Synthetic cytometer", "$TIMESTEP": "0.01", **keywords})


def _write_simple(path: Path, columns: dict[str, np.ndarray], stains: dict[str, str]) -> None:
    names = list(columns)
    events = np.column_stack([columns[k] for k in names]).astype(np.float32)
    write_fcs(path, events, [Channel(i + 1, k, stains.get(k, ""), 262144) for i, k in enumerate(names)], {"$CYT": "Synthetic cytometer"})


def write_extras(rng, folder: Path) -> None:
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
    _write_simple(folder / "DNA content.fcs", {"FSC-A": rng.normal(60000, 8000, dna.size), "SSC-A": rng.normal(30000, 6000, dna.size), "PI-A": dna}, {"PI-A": "DNA (PI)"})
    sizes = (np.array([0.10, 0.20, 0.30, 0.25, 0.15]) * 30000).astype(int)
    dye = np.concatenate([50000 / 2 ** k * rng.lognormal(0, 0.12, m) for k, m in enumerate(sizes)])
    rng.shuffle(dye)
    _write_simple(folder / "CellTrace dilution.fcs", {"FSC-A": rng.normal(60000, 8000, dye.size), "SSC-A": rng.normal(30000, 6000, dye.size), "CTV-A": dye}, {"CTV-A": "CellTrace Violet"})
    peaks = [120.0 * 4 ** i for i in range(6)]
    beads = np.concatenate([p * rng.lognormal(0, 0.05, 4000) for p in peaks])
    rng.shuffle(beads)
    _write_simple(folder / "Rainbow beads.fcs", {"FSC-A": rng.normal(60000, 3000, beads.size), "SSC-A": rng.normal(30000, 2000, beads.size), "FITC-A": beads}, {"FITC-A": "FITC"})


def build_workspace(out: Path, samples: list[Path]) -> Workspace:
    ws = Workspace()
    loaded = [ws.add_sample(p) for p in samples]
    ws.add_group("Control", [s for s in loaded if "Control" in s.name], color="#2e7d32")
    ws.add_group("Stimulated", [s for s in loaded if "Stimulated" in s.name], color="#c62828")
    everyone = ws.groups[0]
    cd3 = "/Lymphocytes/Single cells/CD3+"
    tree = [
        ("", "Lymphocytes", RectangleGate(["FSC-A", "SSC-A"], [50000, 5000], [110000, 55000])),
        ("/Lymphocytes", "Single cells", PolygonGate(["FSC-A", "FSC-H"], [(50000, 45000), (110000, 99000), (110000, 121000), (50000, 55000)])),
        ("/Lymphocytes/Single cells", "CD3+", RectangleGate(["Comp-FITC-A"], [4000], [None])),
        (cd3, "CD69+", RectangleGate(["Comp-PE-Cy7-A"], [3000], [None])),
    ]
    for parent, name, gate in tree:
        ws.apply_population_to_group(everyone, parent, Population(name, gate))
    for name, gate in make_quadrant_gates("Comp-PE-A", "Comp-APC-A", 4000, 4000, "CD4", "CD8"):
        ws.apply_population_to_group(everyone, cd3, Population(name, gate))
    ws.settings["experiment"] = {
        "name": "FlowDesk test set",
        "investigator": "FlowDesk",
        "description": "Synthetic T-cell activation experiment: 3 donors, control and stimulated.",
        "notes": "All data are generated; see READ ME.txt.",
        "panel": [{"parameter": p, "stain": s, "fluorochrome": p.replace("-A", "")} for p, s in STAINS.items()],
    }
    save_workspace(ws, out / "FlowDesk testset.fdw")
    ws.modified = False
    return ws


README = """FlowDesk test set  (synthetic data)
===================================

Open "FlowDesk testset.fdw" in FlowDesk (File > Open, or drag it onto the window).
It holds six samples, three donors in two conditions, with the groups Control and Stimulated, a gating
tree (Lymphocytes > Single cells > CD3+ > CD4 / CD8 quadrants and CD69+) and an experiment annotation.

Things to try
-------------
Gating        Double-click a sample, then a population. The gates are drawn in the plots; move one and
              watch the statistics change in the workspace window. Ctrl+Z undoes it.
Groups        Double-click the group "Stimulated" and draw a gate: it applies to all its samples.
Statistics    Add Statistic: Median of CD69 in CD3+, or the frequency of CD69+ in CD4+ cells.
Table         Table Editor: columns "CD69+ - Freq. of Parent" per sample, grouped by Condition.
Layout        Layout Editor: drop the CD4 / CD8 plot on a page, batch it over the group, export a PDF.
Compensation  The samples carry an acquisition matrix (Comp- parameters). To compute one yourself:
              File > New Workspace, Add Folder "compensation controls", then Tools > Compensation
              Wizard. The true spillover is FITC>PE 14 %, PE>PE-Cy7 22 %, APC>PE-Cy7 7 %.
Plate         Workspace > Experiment > Plate Editor: the WELL ID keywords place the samples in A1-A3
              and B1-B3.
Clustering    Select CD3+ in Donor1_Stimulated, Tools > FlowSOM or t-SNE on FITC, PE, APC, PE-Cy7.
Cleaning      Open samples/Donor3_Stimulated.fcs, Tools > Data Cleaning: the first tenth of the run
              was acquired at a fifth of the flow rate and is flagged.

Extras (open each with Add Samples)
-----------------------------------
extras/DNA content.fcs          Tools > Cell Cycle (Watson model, tick debris and aggregates):
                                about 60 % G1, 15 % S, 25 % G2/M among the cycling cells.
extras/CellTrace dilution.fcs   Tools > Proliferation: five generations of 10, 20, 30, 25 and 15 %.
extras/Rainbow beads.fcs        Tools > Calibrate (MESF): six peaks; enter 900, 3600, 14400, 57600,
                                230400 and 921600 as the MESF values.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="FlowDesk testset", help="folder to write (created if needed)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "compensation controls").mkdir(exist_ok=True)
    (out / "extras").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    paths = []
    wells = iter(["A1", "A2", "A3", "B1", "B2", "B3"])
    for condition, col in (("Control", 2), ("Stimulated", 3)):
        for donor, params in DONORS.items():
            events = make_sample(rng, 25000, params[0], params[1], params[col], dip=(donor == "Donor3" and condition == "Stimulated"))
            path = out / "samples" / f"{donor}_{condition}.fcs"
            _write(path, events, {"Donor": donor, "Condition": condition, "WELL ID": next(wells), "$SPILLOVER": _spillover_keyword()})
            paths.append(path)
    # wells in the order the files were written: control row A, stimulated row B
    for name, events in make_controls(rng).items():
        label = "Unstained" if name == "Unstained" else f"{STAINS[name]} {name.replace('-A', '')}"
        _write(out / "compensation controls" / f"{label}.fcs", events, {})
    write_extras(rng, out / "extras")
    ws = build_workspace(out, sorted(paths, key=lambda p: ("Stimulated" in p.name, p.name)))
    (out / "READ ME.txt").write_text(README, encoding="utf-8")
    for s in ws.samples:
        print(f"{s.name:26s} {s.n_events:6d} events  Lymphocytes {s.root.find('/Lymphocytes').count:6d}"
              f"  CD3+ {s.root.find('/Lymphocytes/Single cells/CD3+').count:6d}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
