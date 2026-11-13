"""Statistics tables and 1-D histograms."""
from __future__ import annotations

import numpy as np

from . import describe
from .ops_base import OpError, clean, listing, op

_ALIASES = {
    "frequency of parent": "Freq. of Parent", "freq of parent": "Freq. of Parent", "% of parent": "Freq. of Parent",
    "percent of parent": "Freq. of Parent", "% parent": "Freq. of Parent", "frequency of total": "Freq. of Total",
    "freq of total": "Freq. of Total", "% of total": "Freq. of Total", "percent of total": "Freq. of Total",
    "geomean": "Geometric Mean", "geo mean": "Geometric Mean", "gmean": "Geometric Mean", "events": "Count",
    "number of events": "Count", "n": "Count", "median fluorescence": "Median", "mfi": "Median", "stdev": "SD",
    "std": "SD", "standard deviation": "SD",
}


def stat_name(text: str) -> str:
    from ..core.stats import STATISTICS

    key = " ".join(str(text).split()).lower()
    for s in STATISTICS:
        if s.lower() == key:
            return s
    if key in _ALIASES:
        return _ALIASES[key]
    plain = [s for s in STATISTICS if key and key in s.lower()]
    if len(plain) == 1:
        return plain[0]
    raise OpError("unknown statistic %r; available: %s" % (text, listing(STATISTICS, 40)))


class StatsOps:
    @op
    def statistics(self, populations=None, statistics=None, parameters=None, samples=None, group: str | None = None,
                   percentile: float | None = None) -> dict:
        """A table of statistics: one row per sample, one column per population x statistic (x parameter).

        statistics: Count, Freq. of Parent, Freq. of Total, Median, Mean, Geometric Mean, CV, SD, Mode, Min, Max,
        Percentile (with percentile=), ... Frequencies are in percent. parameters (name or stain) are needed for
        Median, Mean and the other intensity statistics. Values are NaN-safe (null when undefined)."""
        from ..core.stats import NEEDS_PARAM, compute_statistic

        targets = self.samples_of(samples, group)
        stats = [stat_name(s) for s in (statistics or ["Count", "Freq. of Parent"])]
        if isinstance(populations, str):
            populations = [populations]
        if not populations:
            grp = self.group(group) if group else None
            populations = [p for p in self.ws.population_paths(grp)][:40]
            if not populations:
                populations = [""]
        params = [parameters] if isinstance(parameters, str) else list(parameters or [])
        if any(s in NEEDS_PARAM for s in stats) and not params:
            raise OpError("%s need parameters= (a name or a stain such as 'CD69')" % listing([s for s in stats if s in NEEDS_PARAM]))
        columns: list[tuple] = []
        for p in populations:
            for st in stats:
                if st in NEEDS_PARAM:
                    for prm in params:
                        columns.append((p, st, prm))
                else:
                    columns.append((p, st, None))
        if len(columns) * len(targets) > 4000:
            raise OpError("that is %d values; narrow samples, populations or statistics" % (len(columns) * len(targets)))
        rows = []
        for s in targets:
            row = [s.name]
            for pop_ref, st, prm in columns:
                try:
                    pop = self.population(s, pop_ref)
                except OpError:
                    row.append(None)
                    continue
                try:
                    name = self.param(s, prm) if prm else None
                except OpError:
                    row.append(None)
                    continue
                row.append(clean(compute_statistic(st, pop, name, percentile if st == "Percentile" else None), 5))
            rows.append(row)
        def used(prm):
            try:
                return self.param(targets[0], prm) if prm else None
            except OpError:
                return prm

        labels = ["sample"] + [" | ".join(x for x in (p or "ungated", st, used(prm)) if x) for p, st, prm in columns]
        return {"columns": labels, "rows": rows, "note": "frequencies are percent; null = undefined or missing in that sample"}

    @op
    def histogram(self, sample, parameter: str, population: str = "", bins: int = 48) -> dict:
        """Distribution of one parameter inside a population: counts per bin (edges in raw units), percentiles, peaks and,
        for a bimodal marker, the valley between the two peaks as a suggested positive/negative threshold."""
        s = self.sample(sample)
        pop = self.population(s, population)
        name = self.param(s, parameter)
        mask = pop.mask()
        bins = int(max(8, min(200, bins)))
        raw = np.asarray(s.data(name))[mask]
        disp = np.asarray(s.display(name))[mask]
        ok = np.isfinite(raw) & np.isfinite(disp)
        out = describe.histogram_summary(raw[ok], disp[ok], s.scale(name), bins)
        out.update({"sample": s.name, "population": pop.path or "/", "parameter": name, "scale": s.scale(name).kind})
        return out
