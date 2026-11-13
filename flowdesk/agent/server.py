"""flowdesk-mcp: MCP server that drives a live FlowDesk through the in-app bridge.

Run by the agent's MCP client (Claude Code, Codex, Claude Desktop) over stdio; ``flowdesk-mcp-install`` registers it.
The server owns no data: every tool is forwarded to the bridge inside FlowDesk, so the agent works on the same window
the user is looking at (undo included).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from typing import Any, List, Optional, Union

from . import APP, PID_ENV, discovery
from .client import BridgeClient, BridgeError, NotConnected
from .mcpcompat import FastMCP, Image

INSTRUCTIONS = """FlowDesk MCP server. You control a live FlowDesk (flow cytometry analysis: gating, compensation, clustering) through a
bridge running inside it, on the same window the user sees; every change is an undo step.

Workflow:
1. `status` - is a FlowDesk bridge running?  `launch` starts one if needed (`mode="headless"` for no window).
2. `workspace` - structured snapshot: samples (events, keywords, groups), groups, populations, parameters, matrices, undo state.
   Call it before assuming what is loaded.  `sample_info` gives one sample's parameters (stain, scale, range); `populations`
   the gating tree with counts.
3. Look at the data: `histogram` (counts per bin, peaks and the valley between them = a threshold suggestion) and `plot` (a PNG plus a
   numerical description: density map, axis ranges, gates drawn with counts) - read the numbers instead of guessing from pictures.
4. Gate: `create_gate` (rectangle, range, polygon, ellipse, boolean).  Values are the raw units of the axes (compensated intensities for
   Comp- parameters).  Use `group=` to put a gate on every sample of a group, or `sample=` for one.  `edit_gate`, `rename_population`,
   `delete_population` change what exists.  Name parameters by stain: x="CD4" resolves to the compensated detector.
5. Measure: `statistics` (one row per sample; Count, Freq. of Parent, Median, ...).  Batch work = the same call on a group.
6. Compensation: `compensation` (list / compute / apply / assign).  Analysis: `dimension_reduction` (t-SNE, UMAP, PCA), `cluster`
   (FlowSOM, k-means, Phenograph), `compare_populations`, `cell_cycle`, `proliferation`, `data_cleaning`.
7. `export_events`, `export_statistics`, `export_flowjo_workspace`, `save_workspace`.
8. `undo` / `redo` revert anything done here or by the user.  `events` / `log` show what the *user* did since your last call.
9. `python` runs code inside FlowDesk (`controller`, `workspace`, `np`) for anything the tools do not cover; use undoable=true for changes.
10. `skills` / `skill` - how-to notes (gating strategy, compensation, batch analysis, FlowJo migration, the Python API).  Read the
    matching skill before a non-trivial job.

Conventions: samples are named like their file ("Donor1_Control", "donor1 control" both work, or an id); populations are paths
("/Lymphocytes/Single cells/CD3+") or a unique name; percentages are 0-100; nothing is saved until `save_workspace`.
Several FlowDesk windows may be open; `status` lists them and `connect(pid=...)` switches.  Never close the user's window
(`quit_flowdesk` needs confirm=true) and never discard unsaved work unless the user agrees.
"""

mcp = FastMCP("flowdesk", instructions=INSTRUCTIONS)


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=str)


def _clean_error(e: Exception) -> str:
    msg = str(e)
    for prefix in ("OpError: ", "ValueError: "):
        if msg.startswith(prefix):
            msg = msg[len(prefix):]
    return msg


class Session:
    def __init__(self):
        self.client = BridgeClient(APP)
        self.last_launch: Optional[dict] = None

    def connect(self, auto_launch: bool = True, mode: str = "gui") -> dict:
        pinned = os.environ.get(PID_ENV)
        if pinned:
            if self.client.info is not None and str(self.client.info.get("pid")) == pinned:
                return self.client.info
            try:
                return self.client.connect(pid=int(pinned))
            except (NotConnected, ValueError):
                raise NotConnected("the FlowDesk window this chat is bound to (pid %s) is no longer running; restart the chat from a live window" % pinned)
        try:
            return self.client.ensure()
        except NotConnected:
            files = discovery.list_instances()
            if files:
                raise NotConnected("%d FlowDesk window(s) are running but none answered (busy or showing a dialog); retry in a moment or call `status`" % len(files))
            if auto_launch and os.environ.get("FLOWDESK_AUTOLAUNCH", "1") != "0":
                return self.launch(mode)
            raise

    def launch(self, mode: str = "gui", wait_s: float = 120.0) -> dict:
        known = {int(i.get("pid", 0)) for i in discovery.list_instances()}
        argv = discovery.find_flowdesk()
        if not argv:
            raise RuntimeError("FlowDesk was not found. Install it, or set FLOWDESK_EXE to its executable.")
        env = {"FLOWDESK_BRIDGE": "1"}
        if mode in ("headless", "nogui"):
            argv = argv + ["--agent-headless"]
            env["QT_QPA_PLATFORM"] = "offscreen"
        t0 = time.time()
        proc = discovery.popen_detached(argv, cwd=os.environ.get("FLOWDESK_APP_CWD") or os.path.expanduser("~"), extra_env=env)
        self.last_launch = {"argv": argv, "pid": proc.pid, "mode": mode, "t": t0}
        inst = discovery.wait_for_instance(APP, timeout=wait_s, not_pids=known)
        if inst is None:
            rc = proc.poll()
            raise RuntimeError("started %s but no bridge appeared within %.0fs%s" % (" ".join(argv), wait_s, "" if rc is None else " (exit code %s)" % rc))
        self.client.connect(inst)
        return inst

    def call(self, op: str, timeout: Optional[float] = None, **args) -> Any:
        self.connect()
        return self.client.call(op, timeout=timeout, **args)


SESSION = Session()


def _instance_note() -> Optional[str]:
    try:
        files = discovery.list_instances()
    except Exception:
        return None
    if len(files) < 2 or not SESSION.client.info:
        return None
    try:
        ident = SESSION.client.call("ping", timeout=5)
    except Exception:
        ident = {}
    return "FlowDesk window pid %s%s · %s (%d windows open; use `connect` to switch)" % (
        SESSION.client.info.get("pid"), (" '%s'" % ident["name"]) if ident.get("name") else "", ident.get("workspace") or "empty", len(files))


def call(op: str, timeout: Optional[float] = None, **args) -> str:
    """Forward one operation to the bridge and return its result (or a clear error) as JSON text."""
    try:
        r = SESSION.call(op, timeout=timeout, **{k: v for k, v in args.items() if v is not None})
    except (BridgeError, NotConnected, RuntimeError) as e:
        return dumps({"ok": False, "error": _clean_error(e)})
    if isinstance(r, dict):
        note = _instance_note()
        if note:
            r = {"window": note, **r}
    return dumps(r)


LONG = 1800.0  # seconds: clustering / t-SNE on many events

# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------
@mcp.tool()
def status() -> str:
    """Connection status: running FlowDesk windows (pid, name, workspace, focus), which one is connected, the FlowDesk executable
    and the folder plots are saved to."""
    instances = SESSION.client.describe_instances()
    responsive = {int(i.get("pid", 0)) for i in instances}
    for f in discovery.list_instances():
        if int(f.get("pid", 0)) not in responsive:
            instances.append({"pid": f.get("pid"), "name": f.get("name", ""), "responsive": False})
    out: dict = {"app": APP, "executable": discovery.find_flowdesk(), "renders_dir": discovery.renders_dir(), "connected": False,
                 "windows": [{"pid": i.get("pid"), "name": i.get("name") or "", "workspace": i.get("workspace"), "samples": i.get("samples"),
                              "gui": i.get("gui"), "window_active": i.get("window_active"), "responsive": i.get("responsive", True)} for i in instances]}
    try:
        info = SESSION.client.ensure() if instances else None
    except NotConnected:
        info = None
    if info:
        out["connected"] = True
        out["connected_pid"] = info.get("pid")
        try:
            out["ping"] = SESSION.client.call("ping", timeout=10)
        except Exception as e:
            out["ping_error"] = str(e)
        if len(instances) > 1:
            out["hint"] = "%d FlowDesk windows are open; connected to pid %s. `connect(pid=...)` switches." % (len(instances), info.get("pid"))
    elif instances:
        out["hint"] = "A FlowDesk bridge exists but did not answer (busy or showing a dialog). Wait and call `status` again."
    else:
        out["hint"] = "No FlowDesk is running. Call `launch` (or any tool: the server starts FlowDesk on demand)."
    if SESSION.last_launch:
        out["last_launch"] = SESSION.last_launch
    return dumps(out)


@mcp.tool()
def launch(mode: str = "gui", wait_seconds: float = 120) -> str:
    """Start a new FlowDesk with the bridge and connect to it. mode: "gui" (normal window) or "headless" (no window; everything except
    opening windows works, plots still render)."""
    try:
        inst = SESSION.launch(mode, wait_seconds)
    except Exception as e:
        return dumps({"ok": False, "error": str(e), "executable": discovery.find_flowdesk(), "mode": mode})
    return dumps({"ok": True, "pid": inst.get("pid"), "port": inst.get("port"), "mode": mode, "version": inst.get("app_version"), "gui": inst.get("gui")})


@mcp.tool()
def connect(pid: Optional[int] = None, port: Optional[int] = None, name: Optional[str] = None) -> str:
    """Attach to a specific running FlowDesk window (see `status` for pids and names). Without arguments: the focused window, else the
    most recently used one."""
    try:
        info = SESSION.client.connect(pid=pid, port=port, name=name)
    except NotConnected as e:
        return dumps({"ok": False, "error": str(e)})
    try:
        ident = SESSION.client.call("ping", timeout=5)
    except Exception:
        ident = {}
    return dumps({"ok": True, "pid": info.get("pid"), "port": info.get("port"), "workspace": ident.get("workspace"), "samples": ident.get("samples")})


@mcp.tool()
def quit_flowdesk(confirm: bool = False, discard_changes: bool = False) -> str:
    """Close the connected FlowDesk. Refused unless confirm=true, and refused while the workspace has unsaved changes unless
    discard_changes=true. Ask the user before closing their window."""
    if not confirm:
        return dumps({"ok": False, "error": "refused without confirm=true; ask the user first"})
    return call("quit", discard_changes=discard_changes)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
@mcp.tool()
def workspace(detail: str = "summary", group: Optional[str] = None) -> str:
    """Structured snapshot of the workspace: samples (id, events, keywords, groups, compensation), groups (role, members, criteria),
    population paths, parameters, matrices, computed parameters, experiment annotation, undo state and the window's selection.
    detail="full" lists every keyword and every population; group= restricts samples and populations to one group."""
    return call("workspace", detail=detail, group=group)


@mcp.tool()
def sample_info(sample: Union[str, int]) -> str:
    """One sample in detail: file, events, all keywords, every parameter with its stain, scale and display range, compensation, the
    populations with counts. `sample` is a name (with or without .fcs, loosely spelled) or an id."""
    return call("sample_info", sample=sample)


@mcp.tool()
def populations(sample: Union[str, int, None] = None, group: Optional[str] = None, population: str = "", depth: int = 6) -> str:
    """The gating tree: with sample= the counts and % of parent of every population; with group= the group's gating template.
    population= starts at a subtree."""
    return call("populations", sample=sample, group=group, population=population, depth=depth)


@mcp.tool()
def open_workspace(path: str, discard_changes: bool = False) -> str:
    """Open a FlowDesk .fdw or a FlowJo workspace, replacing the current one. Refused while the current workspace has unsaved
    changes unless discard_changes=true (only with the user's agreement)."""
    return call("open_workspace", path=path, discard_changes=discard_changes)


@mcp.tool()
def new_workspace(discard_changes: bool = False) -> str:
    """Start an empty workspace (refused with unsaved changes unless discard_changes=true)."""
    return call("new_workspace", discard_changes=discard_changes)


@mcp.tool()
def save_workspace(path: Optional[str] = None) -> str:
    """Save as a FlowDesk workspace (.fdw); without a path, where it was opened from. Computed parameters (t-SNE, clusters) are stored
    next to it in a .fdw.data folder."""
    return call("save_workspace", path=path)


@mcp.tool()
def add_samples(paths: Union[str, List[str], None] = None, folder: Optional[str] = None, recursive: bool = True, group: Optional[str] = None) -> str:
    """Add FCS files (paths) or a whole folder to the workspace, optionally into a group."""
    return call("add_samples", paths=paths, folder=folder, recursive=recursive, group=group)


@mcp.tool()
def remove_samples(samples: Union[str, List[str]]) -> str:
    """Remove samples from the workspace (undoable; the files stay on disk)."""
    return call("remove_samples", samples=samples)


@mcp.tool()
def set_keyword(samples: Union[str, List[str]], name: str, value: str) -> str:
    """Set a keyword (Donor, Condition, ...) on samples. Keyword columns and live keyword groups update."""
    return call("set_keyword", samples=samples, name=name, value=value)


@mcp.tool()
def create_group(name: str, samples: Union[str, List[str], None] = None, criteria: Optional[List[dict]] = None, color: str = "#2e7d32") -> str:
    """Create a group from explicit samples, or a live keyword group: criteria=[{"keyword": "Condition", "function": "Equals",
    "value": "Stimulated"}] (functions: Contains, Does not contain, Equals, Not equals, Starts with, Ends with)."""
    return call("create_group", name=name, samples=samples, criteria=criteria, color=color)


@mcp.tool()
def delete_group(name: str) -> str:
    """Delete a custom group (samples stay in the workspace)."""
    return call("delete_group", name=name)


# ---------------------------------------------------------------------------
# gating
# ---------------------------------------------------------------------------
@mcp.tool()
def create_gate(name: str, shape: str, x: Optional[str] = None, y: Optional[str] = None, sample: Union[str, int, None] = None,
                group: Optional[str] = None, parent: str = "", x_range: Optional[List[Optional[float]]] = None,
                y_range: Optional[List[Optional[float]]] = None, vertices: Optional[List[List[float]]] = None,
                center: Optional[List[float]] = None, a_end: Optional[List[float]] = None, b_end: Optional[List[float]] = None,
                operation: Optional[str] = None, refs: Optional[List[str]] = None, negated: bool = False) -> str:
    """Create a population below `parent` ("" = ungated). Values are in the raw units of the axes (compensated intensities for
    Comp- parameters); x and y are parameter names or stains ("CD4"). With group= the gate goes onto the group template and every
    member sample; with sample= onto one sample.

    shape="rectangle": x, y and x_range=[lo, hi] and/or y_range=[lo, hi] (null = open end).
    shape="range": one parameter x and x_range, e.g. [4000, null] for "positive".
    shape="polygon": x, y and vertices=[[x, y], ...] (at least 3).
    shape="ellipse": x, y, center=[x, y], a_end=[x, y], b_end=[x, y] (ends of the two semi-axes).
    shape="boolean": operation "and"/"or"/"not" and refs=[population paths].
    negated=true selects the events outside the shape. Returns the counts and % of parent per sample."""
    return call("create_gate", name=name, shape=shape, x=x, y=y, sample=sample, group=group, parent=parent, x_range=x_range, y_range=y_range,
                vertices=vertices, center=center, a_end=a_end, b_end=b_end, operation=operation, refs=refs, negated=negated or None)


@mcp.tool()
def edit_gate(population: str, sample: Union[str, int, None] = None, group: Optional[str] = None, x_range: Optional[List[Optional[float]]] = None,
              y_range: Optional[List[Optional[float]]] = None, vertices: Optional[List[List[float]]] = None, center: Optional[List[float]] = None,
              a_end: Optional[List[float]] = None, b_end: Optional[List[float]] = None, refs: Optional[List[str]] = None,
              operation: Optional[str] = None, negated: Optional[bool] = None) -> str:
    """Change the geometry of an existing gate in place (same shape and parameters); sub-populations are kept and recounted. With
    group= the change reaches every member."""
    return call("edit_gate", population=population, sample=sample, group=group, x_range=x_range, y_range=y_range, vertices=vertices, center=center,
                a_end=a_end, b_end=b_end, refs=refs, operation=operation, negated=negated)


@mcp.tool()
def rename_population(population: str, new_name: str, sample: Union[str, int, None] = None, group: Optional[str] = None) -> str:
    """Rename a gated population (on one sample, or on every member of a group)."""
    return call("rename_population", population=population, new_name=new_name, sample=sample, group=group)


@mcp.tool()
def delete_population(population: str, sample: Union[str, int, None] = None, group: Optional[str] = None) -> str:
    """Delete a gated population and everything below it (undoable)."""
    return call("delete_population", population=population, sample=sample, group=group)


@mcp.tool()
def set_scale(parameter: str, kind: str, samples: Union[str, List[str], None] = None, group: Optional[str] = None, minimum: Optional[float] = None,
              maximum: Optional[float] = None, top: Optional[float] = None, positive_decades: Optional[float] = None,
              negative_decades: Optional[float] = None, width_basis: Optional[float] = None, decades: Optional[float] = None) -> str:
    """Change an axis transform: kind linear (minimum, maximum), logicle/biexponential (top, positive_decades, negative_decades,
    width_basis), log (minimum, maximum) or arcsinh (top, decades). Applies to the raw and the compensated parameter."""
    return call("set_scale", parameter=parameter, kind=kind, samples=samples, group=group, minimum=minimum, maximum=maximum, top=top,
                positive_decades=positive_decades, negative_decades=negative_decades, width_basis=width_basis, decades=decades)


# ---------------------------------------------------------------------------
# looking and measuring
# ---------------------------------------------------------------------------
@mcp.tool()
def statistics(populations: Union[str, List[str], None] = None, statistics: Union[str, List[str], None] = None,
               parameters: Union[str, List[str], None] = None, samples: Union[str, List[str], None] = None, group: Optional[str] = None,
               percentile: Optional[float] = None) -> str:
    """A table: one row per sample, one column per population x statistic (x parameter). statistics: Count, Freq. of Parent,
    Freq. of Total, Median, Mean, Geometric Mean, CV, SD, Mode, Min, Max, Percentile (percentile=) ...; frequencies are percent.
    Median and the other intensity statistics need parameters (names or stains). Default: Count and Freq. of Parent of every
    population of all samples."""
    return call("statistics", populations=populations, statistics=statistics, parameters=parameters, samples=samples, group=group, percentile=percentile)


@mcp.tool()
def histogram(sample: Union[str, int], parameter: str, population: str = "", bins: int = 48) -> str:
    """Distribution of one parameter in a population: counts per bin (edges in raw units), percentiles, peaks and - for a bimodal
    marker - the valley between the two main peaks as `suggested_threshold_raw` with `percent_above_threshold`. Use this to place
    positive/negative gates from the data."""
    return call("histogram", sample=sample, parameter=parameter, population=population, bins=bins)


@mcp.tool()
def plot(sample: Union[str, int], population: str = "", x: Optional[str] = None, y: Optional[str] = None, plot_type: Optional[str] = None,
         width: int = 640, height: int = 480, overlays: Optional[List[dict]] = None, return_image: bool = True, path: Optional[str] = None,
         grid: str = "48x18"):
    """Render a plot like a graph window and describe it numerically: events, axes with raw-unit ranges, an ASCII density map
    (grid = columns x rows), the gates drawn on it (child populations) with counts, histogram peaks. Without y a histogram of x is
    drawn. plot_type: pseudocolor, dot, contour, zebra, density, histogram. overlays=[{"sample":..., "population":...}] are drawn on top.
    The PNG is saved (path= to choose) and returned as an image when return_image=true."""
    try:
        r = SESSION.call("plot", timeout=300, **{k: v for k, v in dict(sample=sample, population=population, x=x, y=y, plot_type=plot_type,
                                                                        width=width, height=height, overlays=overlays, return_image=return_image,
                                                                        path=path, grid=grid).items() if v is not None})
    except (BridgeError, NotConnected, RuntimeError) as e:
        return dumps({"ok": False, "error": _clean_error(e)})
    png = r.pop("png", None)
    text = dumps(r)
    if png and return_image:
        return [text, Image(data=base64.b64decode(png), format="png")]
    return text


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
@mcp.tool()
def compensation(action: str = "list", controls_group: Optional[str] = None, method: str = "median", name: str = "Computed",
                 apply_to_group: Optional[str] = None, replace: bool = False, matrix: Optional[str] = None,
                 samples: Union[str, List[str], None] = None) -> str:
    """Compensation. action="list": the matrices in the workspace and per sample. "compute": estimate the spillover matrix from the single-stain
    controls (controls_group; default the Compensation group, else all samples; controls are recognised by file name vs channel/stain) with
    method "median" or "autospill"; reports the matrix and per-control diagnostics, changes nothing. "apply": compute, store as `name` and apply
    to apply_to_group (one undo step; replace=true overwrites an existing name). "assign": apply the stored matrix `matrix` (or "acquisition" for
    the file's own) to samples/apply_to_group."""
    return call("compensation", timeout=LONG, action=action, controls_group=controls_group, method=method, name=name, apply_to_group=apply_to_group,
                replace=replace or None, matrix=matrix, samples=samples)


@mcp.tool()
def dimension_reduction(method: str = "tsne", population: str = "", samples: Union[str, List[str], None] = None, group: Optional[str] = None,
                        parameters: Union[str, List[str], None] = None, max_events: int = 5000, perplexity: float = 30.0, iterations: int = 500,
                        n_neighbors: int = 15, min_dist: float = 0.1, seed: int = 0) -> str:
    """t-SNE, UMAP or PCA of a population (default parameters: compensated fluorescence). Writes two new parameters (tSNE_1/tSNE_2, UMAP_1/2,
    PC_1/2) you can then plot and gate on. max_events is per sample (random subsample); the run blocks FlowDesk meanwhile, so keep it modest."""
    return call("dimension_reduction", timeout=LONG, method=method, population=population, samples=samples, group=group, parameters=parameters,
                max_events=max_events, perplexity=perplexity, iterations=iterations, n_neighbors=n_neighbors, min_dist=min_dist, seed=seed)


@mcp.tool()
def cluster(method: str = "flowsom", population: str = "", samples: Union[str, List[str], None] = None, group: Optional[str] = None,
            parameters: Union[str, List[str], None] = None, clusters: int = 10, grid: int = 6, k: int = 30, resolution: float = 1.0,
            iterations: int = 10, max_events: Optional[int] = None, seed: int = 0) -> str:
    """Cluster a population: flowsom (clusters = metaclusters, grid), kmeans (clusters) or phenograph (k, resolution). Creates one population
    per cluster below it and returns their names and the median of every parameter per cluster."""
    return call("cluster", timeout=LONG, method=method, population=population, samples=samples, group=group, parameters=parameters, clusters=clusters,
                grid=grid, k=k, resolution=resolution, iterations=iterations, max_events=max_events, seed=seed)


@mcp.tool()
def compare_populations(sample_a: Union[str, int], population_a: str, sample_b: Union[str, int, None] = None, population_b: Optional[str] = None,
                        parameters: Union[str, List[str], None] = None) -> str:
    """How different two populations are, per parameter: chi-squared T(x) (probability-binning style) and the KS statistic, ranked."""
    return call("compare_populations", sample_a=sample_a, population_a=population_a, sample_b=sample_b, population_b=population_b, parameters=parameters)


@mcp.tool()
def cell_cycle(parameter: str, population: str = "", samples: Union[str, List[str], None] = None, group: Optional[str] = None,
               model: str = "dean_jett_fox", debris: bool = False, aggregates: bool = False, bins: int = 256) -> str:
    """Fit the cell cycle of a DNA-content histogram (model dean_jett_fox or watson, optional debris/aggregate components): %G1, %S, %G2/M,
    peak means and CVs, one row per sample."""
    return call("cell_cycle", parameter=parameter, population=population, samples=samples, group=group, model=model, debris=debris or None,
                aggregates=aggregates or None, bins=bins)


@mcp.tool()
def proliferation(parameter: str, population: str = "", samples: Union[str, List[str], None] = None, group: Optional[str] = None, bins: int = 256,
                  max_generations: int = 10) -> str:
    """Dye-dilution model (CFSE / CellTrace): percent per generation, division / proliferation / expansion index and % divided per sample."""
    return call("proliferation", parameter=parameter, population=population, samples=samples, group=group, bins=bins, max_generations=max_generations)


@mcp.tool()
def data_cleaning(population: str = "", samples: Union[str, List[str], None] = None, group: Optional[str] = None, time_parameter: Optional[str] = None,
                  create_population: bool = False, name: str = "QC pass") -> str:
    """Flag unstable flow rate, signal drift and saturated events; reports the percent removed per check. create_population=true adds a
    clean-events population below `population` (one undo step)."""
    return call("data_cleaning", population=population, samples=samples, group=group, time_parameter=time_parameter,
                create_population=create_population or None, name=name)


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
@mcp.tool()
def export_events(path: str, population: str = "", samples: Union[str, List[str], None] = None, group: Optional[str] = None,
                  parameters: Union[str, List[str], None] = None, fmt: Optional[str] = None, include_computed: bool = True,
                  include_derived: bool = True, max_events: Optional[int] = None, percent: Optional[float] = None, seed: int = 0) -> str:
    """Write the events of a population as FCS 3.1 or CSV (fmt, or by suffix). One sample: path is the file. Several: path is a folder, files are
    named <sample>_<population>. Computed parameters (t-SNE axes, cluster ids, QC flag) are included unless switched off."""
    return call("export_events", path=path, population=population, samples=samples, group=group, parameters=parameters, fmt=fmt,
                include_computed=include_computed, include_derived=include_derived, max_events=max_events, percent=percent, seed=seed)


@mcp.tool()
def export_statistics(path: str, populations: Union[str, List[str], None] = None, statistics: Union[str, List[str], None] = None,
                      parameters: Union[str, List[str], None] = None, samples: Union[str, List[str], None] = None, group: Optional[str] = None,
                      percentile: Optional[float] = None) -> str:
    """The `statistics` table written as CSV."""
    return call("export_statistics", path=path, populations=populations, statistics=statistics, parameters=parameters, samples=samples, group=group,
                percentile=percentile)


@mcp.tool()
def export_flowjo_workspace(path: str) -> str:
    """Write the workspace as a FlowJo 10 .wsp. Gates on computed or derived parameters cannot exist in FlowJo and are left out; the
    warnings say which."""
    return call("export_flowjo_workspace", path=path)


# ---------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------
@mcp.tool()
def undo() -> str:
    """Undo the last change (gates, keywords, scales, matrices, layouts, ... - also the user's)."""
    return call("undo")


@mcp.tool()
def redo() -> str:
    """Redo what was undone."""
    return call("redo")


@mcp.tool()
def python(code: str, undoable: bool = False) -> str:
    """Run Python inside FlowDesk. Available: `controller`, `workspace`, `window` (None when headless), `np`, `ops` and the `flowdesk` package;
    assign `result` to return data. Prefer the structured tools; use this for anything they do not cover. undoable=true makes the script one undo
    step and refreshes the views afterwards (use it whenever the code changes something)."""
    return call("python", timeout=LONG, code=code, undoable=undoable or None)


@mcp.tool()
def events(since: Optional[int] = None, limit: int = 100) -> str:
    """What happened since `since` (an id from a previous call): the operations run and the changes the *user* made in the window (gating,
    selection, samples, ...). Call it after the user has been working to see what changed."""
    return call("events", since=since, limit=limit)


@mcp.tool()
def log(n: int = 50) -> str:
    """The last n events as plain text lines."""
    return call("log", n=n)


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------
def _skill_dirs() -> List[str]:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
    return [here, os.path.join(discovery.flowdesk_home(), "skills")]


def _skills() -> dict:
    out = {}
    for d in _skill_dirs():
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".md"):
                    out[f[:-3]] = os.path.join(d, f)
    return out


@mcp.tool()
def skills() -> str:
    """List the skill notes (how-to guides for common jobs): name and first line of each. Read one with `skill(name)`."""
    items = []
    for name, path in _skills().items():
        with open(path, encoding="utf-8") as fh:
            first = next((ln.strip("# \n") for ln in fh if ln.strip()), "")
        items.append({"name": name, "summary": first, "user_defined": not path.startswith(_skill_dirs()[0])})
    return dumps({"skills": items})


@mcp.tool()
def skill(name: str) -> str:
    """The text of one skill note. Read the matching one before a non-trivial task."""
    found = _skills()
    if name not in found:
        return dumps({"ok": False, "error": "no skill %r; available: %s" % (name, ", ".join(found))})
    with open(found[name], encoding="utf-8") as fh:
        return fh.read()


@mcp.tool()
def save_skill(name: str, content: str) -> str:
    """Store your own recipe as a skill note (markdown) for next time. Stored in ~/.flowdesk/skills."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-").lower()
    if not safe:
        return dumps({"ok": False, "error": "give a name made of letters, digits, - or _"})
    d = _skill_dirs()[1]
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, safe + ".md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return dumps({"ok": True, "path": path})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
