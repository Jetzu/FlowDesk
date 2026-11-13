"""Compensation, dimensionality reduction, clustering, cell cycle, proliferation, data cleaning and comparison."""
from __future__ import annotations

import dataclasses

import numpy as np

from .ops_base import OpError, clean, listing, op


def _diag(d) -> dict:
    out = {"parameter": d.param, "status": d.status, "events": int(d.n_events), "positive_events": int(d.n_pos),
           "negative_events": int(d.n_neg), "stain_index": clean(d.stain_index, 3), "method": d.method}
    if d.message:
        out["message"] = d.message
    return out


def snake(key: str) -> str:
    """'%G2/M' -> 'percent_g2m', 'Division Index' -> 'division_index'."""
    special = {"G2/G1": "g2_g1_ratio", "Gen 0 Peak": "generation0_peak"}
    if key in special:
        return special[key]
    return key.replace("%", "percent_").replace("/", "").strip().lower().replace(" ", "_")


def _matrix_json(m) -> dict:
    return {"name": m.name, "parameters": list(m.params), "spillover_percent": [[clean(100.0 * v, 3) for v in row] for row in m.matrix],
            "note": "row = the fluorochrome, column = the detector it spills into; the diagonal is 100"}


class AnalysisOps:
    # ------------------------------------------------------------ compensation
    @op
    def compensation(self, action: str = "list", controls_group: str | None = None, method: str = "median", name: str = "Computed",
                     apply_to_group: str | None = None, replace: bool = False, matrix: str | None = None, samples=None) -> dict:
        """Compensation. action 'list': the matrices in the workspace and in the files. 'compute': run the wizard logic on the
        single-stain controls (controls_group, default the Compensation group) with method 'median' or 'autospill' and report the
        matrix and per-control diagnostics without changing anything. 'apply': compute, store the matrix as `name` and apply it to
        apply_to_group (one undo step). 'assign': apply the stored matrix `matrix` (or 'acquisition' for the file's own) to samples."""
        ws = self.ws
        if action == "list":
            out = {"workspace_matrices": [_matrix_json(m) for m in ws.matrices.values()], "samples": []}
            for s in ws.samples:
                out["samples"].append({"sample": s.name, "matrix": s.comp.name if s.comp is not None else None, "enabled": bool(s.comp_enabled)})
            return out
        if action == "assign":
            if not matrix:
                raise OpError("assign needs matrix= (a workspace matrix name, or 'acquisition')")
            targets = self.samples_of(samples, apply_to_group)
            if matrix.lower() == "acquisition":
                from ..core.compensation import SpilloverMatrix

                n = 0
                for s in targets:
                    s.ensure_loaded()
                    m = SpilloverMatrix.from_fcs(s.fcs)
                    if m is not None:
                        self.c.set_compensation([s], m, True)
                        n += 1
                return {"ok": True, "assigned": n, "samples": [s.name for s in targets]}
            if matrix not in ws.matrices:
                raise OpError("no matrix %r; the workspace has: %s" % (matrix, listing(ws.matrices)))
            self.c.set_compensation(targets, ws.matrices[matrix], True)
            return {"ok": True, "assigned": len(targets), "matrix": matrix, "samples": [s.name for s in targets]}
        if action not in ("compute", "apply"):
            raise OpError("action must be list, compute, apply or assign")
        if method not in ("median", "autospill"):
            raise OpError("method must be 'median' or 'autospill'")
        from ..ui.compensation import CompensationWizard

        grp = self.group(controls_group) if controls_group else None
        if grp is None:
            comp = next((g for g in ws.groups if g.role == "compensation"), None)
            if comp is None or len(ws.members(comp)) < 2:
                # the Compensation group is defined by names ("comp", "unstained"); with one member it is not the control set
                grp = next((g for g in ws.groups if g.role == "all"), None)
        if action == "apply" and name in ws.matrices and not replace:
            raise OpError("a matrix named %r exists; choose another name or pass replace=true" % name)
        wiz = CompensationWizard(self.c, group=grp)
        try:
            wiz.method = method
            wiz.matrix_page.name_edit.setText(name)
            m = wiz.compute()
            if m is None:
                raise OpError(wiz.message or "could not compute a matrix from those controls")
            out = {"ok": True, "message": wiz.message, "matrix": _matrix_json(m),
                   "controls": [_diag(d) for d in wiz.diagnostics.values()],
                   "control_group": wiz.control_group.name if wiz.control_group is not None else None}
            if action == "apply":
                target = apply_to_group or (wiz.control_group.name if wiz.control_group is not None else "All Samples")
                target_group = self.group(target)
                if replace and name in ws.matrices:
                    self.c.remove_matrix(name)
                wiz.matrix_page.apply_check.setChecked(True)
                wiz.matrix_page.attach_check.setChecked(True)
                idx = wiz.matrix_page.group_combo.findText(target_group.name)
                if idx >= 0:
                    wiz.matrix_page.group_combo.setCurrentIndex(idx)
                wiz.accept()
                out["stored_as"] = name
                out["applied_to"] = [s.name for s in ws.members(target_group)]
                self.note("compensation", action="apply", name=name)
            return out
        finally:
            wiz.deleteLater()

    # --------------------------------------------------- reduction & clustering
    def _pops_and_params(self, population, samples, group, parameters):
        from ..ui import tools_analysis as T

        targets = self.samples_of(samples, group)
        pops = [self.population(s, population) for s in targets]
        if parameters:
            names = [parameters] if isinstance(parameters, str) else list(parameters)
            params = [self.param(targets[0], p) for p in names]
        else:
            params = T.default_parameters(pops)
        if len(params) < 2:
            raise OpError("need at least two parameters; pass parameters=[...] (e.g. the compensated fluorescence ones)")
        return T, targets, pops, params

    @op
    def dimension_reduction(self, method: str = "tsne", population: str = "", samples=None, group: str | None = None,
                            parameters=None, max_events: int = 5000, perplexity: float = 30.0, iterations: int = 500,
                            n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0) -> dict:
        """t-SNE, UMAP or PCA of a population: writes two new parameters (e.g. tSNE_1, tSNE_2) you can plot and gate on.
        max_events is per sample (random subsample). Runs on the GUI thread: keep max_events modest."""
        method = method.lower()
        if method not in ("tsne", "umap", "pca"):
            raise OpError("method must be tsne, umap or pca")
        T, targets, pops, params = self._pops_and_params(population, samples, group, parameters)
        options = {"tsne": {"perplexity": float(perplexity), "iterations": int(iterations)},
                   "umap": {"n_neighbors": int(n_neighbors), "min_dist": float(min_dist)}, "pca": {}}[method]
        try:
            names = T.run_dimreduction(self.c, method, populations=pops, params=params, max_events=int(max_events) or None,
                                       options=options, seed=int(seed), open_graph=False)
        except ImportError as exc:
            raise OpError(str(exc))
        self.note("dimension_reduction", method=method, created=names)
        return {"ok": True, "method": method, "created_parameters": names, "samples": [s.name for s in targets],
                "parameters_used": params, "next": "plot x=%s y=%s to see the map" % (names[0], names[1])}

    @op
    def cluster(self, method: str = "flowsom", population: str = "", samples=None, group: str | None = None, parameters=None,
                clusters: int = 10, grid: int = 6, k: int = 30, resolution: float = 1.0, iterations: int = 10, max_events: int | None = None,
                seed: int = 0) -> dict:
        """Cluster a population with flowsom (clusters = metaclusters), kmeans (clusters) or phenograph (k, resolution). Creates one
        population per cluster below it and returns each cluster's size and median per parameter."""
        method = method.lower()
        if method not in ("flowsom", "kmeans", "phenograph"):
            raise OpError("method must be flowsom, kmeans or phenograph")
        T, targets, pops, params = self._pops_and_params(population, samples, group, parameters)
        options = {"flowsom": {"grid": (int(grid), int(grid)), "n_metaclusters": int(clusters), "iterations": int(iterations)},
                   "kmeans": {"k": int(clusters)}, "phenograph": {"k": int(k), "resolution": float(resolution)}}[method]
        name, created, result = T.run_clustering(self.c, method, populations=pops, params=params, max_events=max_events,
                                                 options=options, seed=int(seed))
        data = T.cluster_heatmap_data(self.c, result)
        self.note("cluster", method=method, parameter=name, clusters=len(created))
        return {"ok": True, "method": method, "cluster_parameter": name, "populations": created, "parameters_used": params,
                "medians": {k: data[k] for k in data if k in ("rows", "columns", "parameters", "clusters", "counts", "medians")} or None}

    @op
    def compare_populations(self, sample_a, population_a: str, sample_b=None, population_b: str | None = None, parameters=None) -> dict:
        """How different two populations are per parameter: chi-squared T(x) (probability-binning style) and the KS statistic."""
        from ..ui import tools_analysis as T

        sa = self.sample(sample_a)
        sb = self.sample(sample_b) if sample_b not in (None, "") else sa
        pa, pb = self.population(sa, population_a), self.population(sb, population_b if population_b is not None else "")
        params = None
        if parameters:
            names = [parameters] if isinstance(parameters, str) else list(parameters)
            params = [self.param(sa, n) for n in names]
        rows = T.compare_population_objects(pa, pb, params)
        return {"a": "%s %s" % (sa.name, pa.path or "/"), "b": "%s %s" % (sb.name, pb.path or "/"),
                "rows": [{"parameter": r.param, "chi_squared_T": clean(r.chi_squared_T, 4), "ks": clean(r.ks_statistic, 4),
                          "score": clean(r.difference_score, 4)} for r in rows]}

    # ------------------------------------------------------------- biology / QC
    @op
    def cell_cycle(self, parameter: str, population: str = "", samples=None, group: str | None = None, model: str = "dean_jett_fox",
                   debris: bool = False, aggregates: bool = False, bins: int = 256) -> dict:
        """Fit the cell cycle of a DNA-content histogram (Dean-Jett-Fox or watson): %G1, %S, %G2/M, peak CVs. One row per sample."""
        from ..core import biology as B

        rows = []
        for s in self.samples_of(samples, group):
            pop = self.population(s, population)
            try:
                res = B.cell_cycle_for_population(pop, self.param(s, parameter), bins=int(bins), model=model, debris=bool(debris),
                                                  aggregates=bool(aggregates))
                rows.append({"sample": s.name, **{snake(k): clean(v, 4) for k, v in res.summary().items()}})
            except (ValueError, OpError) as exc:
                rows.append({"sample": s.name, "error": str(exc)})
        return {"model": model, "rows": rows}

    @op
    def proliferation(self, parameter: str, population: str = "", samples=None, group: str | None = None, bins: int = 256,
                      max_generations: int = 10) -> dict:
        """Dye-dilution model: per-generation percentages, division / proliferation / expansion index, % divided."""
        from ..core import biology as B

        rows = []
        for s in self.samples_of(samples, group):
            pop = self.population(s, population)
            try:
                res = B.proliferation_for_population(pop, self.param(s, parameter), bins=int(bins), max_generations=int(max_generations))
                summ = {snake(k): (clean(v, 4) if not isinstance(v, (list, tuple, np.ndarray)) else [clean(x, 3) for x in v]) for k, v in res.summary().items()}
                rows.append({"sample": s.name, **summ})
            except (ValueError, OpError) as exc:
                rows.append({"sample": s.name, "error": str(exc)})
        return {"rows": rows}

    @op
    def data_cleaning(self, population: str = "", samples=None, group: str | None = None, time_parameter: str | None = None,
                      create_population: bool = False, name: str = "QC pass") -> dict:
        """Flag unstable flow rate, signal drift and saturated events. Reports the percent removed per check; with
        create_population=True also adds a clean-events population below `population` (one undo step)."""
        from ..core import qc

        targets = self.samples_of(samples, group)
        rows = []
        for s in targets:
            pop = self.population(s, population)
            s.ensure_loaded()
            tparam = None
            if time_parameter:
                tparam = self.param(s, time_parameter)
            else:
                tparam = next((n for n in s.channel_names if n.upper() in ("TIME", "HDR-T") or n.upper().startswith("TIME")), None)
            try:
                res = qc.qc_for_population(pop, tparam)
            except (ValueError, KeyError) as exc:
                rows.append({"sample": s.name, "error": str(exc)})
                continue
            summ = res.summary()
            row = {"sample": s.name, "time_parameter": tparam, "events": summ["events"], "kept": summ["kept"], "removed": summ["removed"],
                   "percent_removed": {k: clean(v, 3) for k, v in summ["percent_removed"].items()},
                   "flagged_bins": summ["flagged_bins"], "bins": summ["bins"]}
            if create_population:
                self.c.begin_change("data cleaning")
                qc.apply_qc(pop, res, name=name)
                self.c.mark_modified()
                self.c.gatingChanged.emit(s.id)
                row["created"] = (pop.path.rstrip("/") + "/" + name) if pop.path else "/" + name
            rows.append(row)
        return {"rows": rows}
