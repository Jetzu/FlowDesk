"""Rendering a plot for the agent: a PNG plus a numerical description of what it shows."""
from __future__ import annotations

import os
import time

import numpy as np

from . import describe
from .discovery import renders_dir
from .ops_base import OpError, clean, op

PLOT_TYPES = ("pseudocolor", "dot", "contour", "zebra", "density", "histogram")


class PlotOps:
    @op
    def plot(self, sample, population: str = "", x: str | None = None, y: str | None = None, plot_type: str | None = None,
             width: int = 640, height: int = 480, overlays: list[dict] | None = None, return_image: bool = True,
             path: str | None = None, grid: str = "48x18") -> dict:
        """Render a plot of a population (like a graph window) and describe it numerically.

        Without y (or with plot_type='histogram') a histogram of x is drawn. The result lists the axes with their
        raw-unit ranges, an ASCII density map, and the gates drawn on this plot (child populations) with counts.
        overlays: [{"sample": ..., "population": ...}] drawn on top."""
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice

        from ..core.model import GraphSettings
        from ..ui.renderer import RenderOptions, render_plot

        s = self.sample(sample)
        pop = self.population(s, population)
        ax, ay = s.default_axes()
        xname = self.param(s, x) if x else (pop.graph.x or ax)
        yname = None if (plot_type == "histogram") else (self.param(s, y) if y else (None if x and not y else (pop.graph.y or ay)))
        ptype = plot_type or ("histogram" if yname is None else (pop.graph.plot_type if pop.graph.plot_type != "histogram" else "pseudocolor"))
        if ptype not in PLOT_TYPES:
            raise OpError("plot_type must be one of %s" % ", ".join(PLOT_TYPES))
        if ptype == "histogram":
            yname = None
        elif yname is None:
            raise OpError("a %s plot needs y; use plot_type='histogram' for one parameter" % ptype)
        width, height = int(max(160, min(2000, width))), int(max(120, min(2000, height)))
        try:
            cols, rows = (int(v) for v in str(grid).lower().split("x"))
        except ValueError:
            raise OpError("grid must look like '48x18'")
        ov = []
        for o in overlays or []:
            os_ = self.sample(o.get("sample", s.name))
            ov.append((os_, self.population(os_, o.get("population", ""))))
        graph = GraphSettings(xname, yname, ptype, {})
        opts = RenderOptions(plot_type=ptype)
        res = render_plot(s, pop, graph, width, height, opts, overlays=ov or None)
        if res.error:
            raise OpError("could not draw the plot: %s" % res.error)
        out_path = path or os.path.join(renders_dir(), "plot_%s_%03d.png" % (time.strftime("%Y%m%d_%H%M%S"), int(time.time() * 1000) % 1000))
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        res.image.save(out_path, "PNG")
        mask = pop.mask()
        info = {"sample": s.name, "population": pop.path or "/", "plot_type": ptype, "events": int(mask.sum()), "image": {"path": out_path, "width": width, "height": height}}
        row = self.pop_row(pop)
        for k in ("percent_of_parent", "percent_of_total"):
            if k in row:
                info[k] = row[k]
        axes = []
        for name in [n for n in (xname, yname) if n]:
            sc = s.scale(name)
            axes.append({"parameter": name, "label": s.param_label(name), "scale": sc.kind,
                         "range_raw": [clean(sc.inverse(0.0), 4), clean(sc.inverse(1.0), 4)]})
        info["axes"] = axes
        if ptype == "histogram":
            raw = np.asarray(s.data(xname))[mask]
            disp = np.asarray(s.display(xname))[mask]
            ok = np.isfinite(raw) & np.isfinite(disp)
            h = describe.histogram_summary(raw[ok], disp[ok], s.scale(xname), 40)
            info["histogram"] = {k: h[k] for k in h if k not in ("edges_raw", "counts")}
        else:
            xd, yd = np.asarray(s.display(xname))[mask], np.asarray(s.display(yname))[mask]
            ok = np.isfinite(xd) & np.isfinite(yd)
            info["density"] = describe.density_map(xd[ok], yd[ok], s.scale(xname), s.scale(yname), max(8, min(160, cols)), max(4, min(80, rows)))
        gates = []
        for child in pop.children:
            g = child.gate
            if g is None or not g.dims or not set(g.dims) <= {xname, yname}:
                continue
            r = self.pop_row(child)
            gates.append({"name": child.name, "path": child.path, "shape": g.kind, "count": r["count"],
                          "percent_of_parent": r.get("percent_of_parent"), "gate": self.describe_gate(g)})
        info["gates_on_plot"] = gates
        if ov:
            info["overlays"] = [{"sample": a.name, "population": b.path or "/"} for a, b in ov]
        if return_image:
            ba = QByteArray()
            buf = QBuffer(ba)
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            res.image.save(buf, "PNG")
            buf.close()
            info["png"] = bytes(ba.data())
        self.note("plot", sample=s.name, population=pop.path or "/", x=xname, y=yname)
        return info
