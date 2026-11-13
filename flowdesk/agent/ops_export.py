"""Export events, statistics and workspaces."""
from __future__ import annotations

import csv
from pathlib import Path

from .ops_base import OpError, op


class ExportOps:
    @op
    def export_events(self, path: str, population: str = "", samples=None, group: str | None = None, parameters=None,
                      fmt: str | None = None, include_computed: bool = True, include_derived: bool = True,
                      max_events: int | None = None, percent: float | None = None, seed: int = 0) -> dict:
        """Write the events of a population as FCS 3.1 or CSV. One sample: `path` is the file. Several samples: `path` is a folder
        and files are named <sample>_<population>. parameters=[...] limits the columns; computed (t-SNE, cluster ids, QC flag) and
        derived parameters are included unless switched off."""
        from ..core import export as E

        targets = self.samples_of(samples, group)
        dest = Path(path).expanduser()
        written = []
        cols = None
        for s in targets:
            pop = self.population(s, population)
            if parameters:
                cols = [self.param(s, p) for p in ([parameters] if isinstance(parameters, str) else parameters)]
            if len(targets) > 1:
                dest.mkdir(parents=True, exist_ok=True)
                ext = ".csv" if (fmt or "").lower() == "csv" else ".fcs"
                tag = E.safe_filename((pop.path or "ungated").strip("/").replace("/", "_") or "ungated")
                file = dest / (E.safe_filename(s.name.rsplit(".", 1)[0]) + "_" + tag + ext)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                file = dest
            try:
                n = E.export_population(s, pop, file, cols, include_computed=include_computed, include_derived=include_derived,
                                        max_events=max_events, percent=percent, seed=int(seed), fmt=fmt)
            except (KeyError, ValueError) as exc:
                raise OpError("could not export %s: %s" % (s.name, str(exc).strip("'\"")))
            written.append({"sample": s.name, "file": str(file), "events": int(n)})
        self.note("export_events", files=len(written))
        return {"ok": True, "written": written}

    @op
    def export_statistics(self, path: str, populations=None, statistics=None, parameters=None, samples=None, group: str | None = None,
                          percentile: float | None = None) -> dict:
        """The same table as `statistics`, written as CSV."""
        table = self.statistics(populations=populations, statistics=statistics, parameters=parameters, samples=samples, group=group,
                                percentile=percentile)
        dest = Path(path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(table["columns"])
            w.writerows(table["rows"])
        return {"ok": True, "path": str(dest), "rows": len(table["rows"]), "columns": len(table["columns"])}

    @op
    def export_flowjo_workspace(self, path: str) -> dict:
        """Write the workspace as a FlowJo 10 .wsp. Gates on computed or derived parameters cannot be expressed in FlowJo and are
        left out; the warnings say which."""
        from ..core.wsp_export import export_wsp

        dest = Path(path).expanduser()
        if dest.suffix.lower() != ".wsp":
            dest = dest.with_name(dest.name + ".wsp")
        dest.parent.mkdir(parents=True, exist_ok=True)
        warnings = export_wsp(self.ws, dest) or []
        return {"ok": True, "path": str(dest), "warnings": [str(w) for w in warnings][:30]}
