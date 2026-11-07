"""Import FlowJo 10 workspace files (.wsp).

A .wsp file is XML: a FlowJo envelope (groups, sample list, population tree,
graphs, layouts) that embeds ISAC Gating-ML 2.0 gates, transformations and
spillover matrices. This reader is written from the XML structure of the files
alone; layouts, tables and other FlowJo-only sections are ignored.

Gates on parameters FlowDesk cannot compute - FlowJo derived parameters,
Gating-ML "new dimensions" (ratio transforms) or parameters missing from the
FCS file - are kept with their dimension names, so the population tree stays
complete; such populations evaluate as empty (0 events) and are listed in the
import report's warnings. Dimensions without a parameter name (Gating-ML new
dimensions, unnamed dimensions) are named with the `UNRESOLVED_DIMENSION`
prefix, which the FlowJo export never writes back as a parameter.
"""
from __future__ import annotations

import math
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compensation import SpilloverMatrix
from .fcs import FCSError, FCSFile, channels_from_keywords
from .gates import BooleanGate, EllipseGate, Gate, PolygonGate, RectangleGate
from .model import Criterion, GraphSettings, Group, Population, Sample, StatNode, Workspace
from .stats import STATISTICS
from .transforms import ArcsinhScale, LinearScale, LogicleScale, LogScale, Scale

GATING_NS = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
TRANSFORMS_NS = "http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
DATATYPE_NS = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

G = "{%s}" % GATING_NS
T = "{%s}" % TRANSFORMS_NS
D = "{%s}" % DATATYPE_NS

BOOLEAN_NODES = {"NotNode": "not", "AndNode": "and", "OrNode": "or"}
POPULATION_TAGS = {"Population", *BOOLEAN_NODES}
COMPENSATION_ROLE = "ws.group.dlog.compensation"
ALL_SAMPLES = "All Samples"
# prefix of the placeholder names given to gate dimensions that have no
# parameter name (ratio / new dimensions); never the name of an FCS parameter
UNRESOLVED_DIMENSION = "[unresolved] "

GRAPH_TYPES = {
    "pseudocolor": "pseudocolor",
    "pseudocolour": "pseudocolor",
    "dot": "dot",
    "dotplot": "dot",
    "contour": "contour",
    "contourplot": "contour",
    "zebra": "zebra",
    "zebraplot": "zebra",
    "pseudocolorcontour": "zebra",
    "density": "density",
    "densityplot": "density",
    "histogram": "histogram",
}

_STAT_ALIASES = {
    "geom. mean": "Geometric Mean",
    "geometric mean": "Geometric Mean",
    "std. dev.": "SD",
    "standard deviation": "SD",
    "sd": "SD",
    "cv": "CV",
    "robust sd": "Robust SD",
    "robust cv": "Robust CV",
    "minimum": "Min",
    "maximum": "Max",
    "frequency of parent": "Freq. of Parent",
    "frequency of total": "Freq. of Total",
    "frequency of grandparent": "Freq. of Grandparent",
}
_STAT_BY_LOWER = {s.lower(): s for s in STATISTICS}


@dataclass
class ImportReport:
    warnings: list[str] = field(default_factory=list)
    samples: int = 0
    samples_found: int = 0
    samples_missing: int = 0
    populations: int = 0
    gates: int = 0
    boolean_gates: int = 0
    groups: int = 0
    matrices: int = 0
    statistics: int = 0

    def warn(self, msg: str):
        self.warnings.append(msg)


# ---------------------------------------------------------------- XML helpers
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _get(el: ET.Element, name: str, default: str | None = None) -> str | None:
    """Attribute lookup that accepts the namespaced and the plain spelling."""
    for key in (name, T + name, G + name, D + name):
        if key in el.attrib:
            return el.attrib[key]
    return default


def _float(el: ET.Element, name: str, default: float | None = None) -> float | None:
    v = _get(el, name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _param_name(el: ET.Element) -> str | None:
    """Name of the data-type:parameter / data-type:fcs-dimension child."""
    for child in el.iter():
        if _local(child.tag) in ("parameter", "fcs-dimension") and child is not el:
            name = _get(child, "name")
            if name is not None:
                return name
    return None


# ----------------------------------------------------------------- transforms
def parse_transform(el: ET.Element, report: ImportReport | None = None) -> tuple[str, Scale] | None:
    kind = _local(el.tag)
    name = _param_name(el)
    if name is None:
        return None
    if kind == "linear":
        return name, LinearScale(_float(el, "minRange", 0.0), _float(el, "maxRange", 262144.0))
    if kind == "log":
        offset = _float(el, "offset", 1.0) or 1.0
        decades = _float(el, "decades", 4.5) or 4.5
        return name, LogScale(offset, offset * 10.0 ** decades)
    if kind == "biex":
        return name, LogicleScale.from_biex(
            T=_float(el, "maxRange", 262144.0),
            pos=_float(el, "pos", 4.5),
            neg=_float(el, "neg", 0.0),
            width_basis=_float(el, "width", -10.0),
        )
    if kind == "fasinh":
        return name, ArcsinhScale(_float(el, "T", 262144.0), _float(el, "M", 4.5), _float(el, "A", 0.0))
    if kind in ("logicle", "hyperlog"):
        if kind == "hyperlog" and report is not None:
            report.warn(f"hyperlog transform on {name} approximated by logicle")
        return name, LogicleScale(_float(el, "T", 262144.0), _float(el, "W", 0.5), _float(el, "M", 4.5), _float(el, "A", 0.0))
    if report is not None:
        report.warn(f"unsupported transform {kind!r} on {name}")
    return None


# ---------------------------------------------------------------- compensation
def parse_spillover_matrix(el: ET.Element) -> SpilloverMatrix:
    name = el.get("name") or _get(el, "id") or "Matrix"
    prefix = el.get("prefix", "Comp-")
    suffix = el.get("suffix", "")
    params: list[str] = []
    for container in el:
        if _local(container.tag) == "parameters":
            for p in container:
                pname = _get(p, "name")
                if pname is not None:
                    params.append(pname)
    rows: dict[str, dict[str, float]] = {}
    for sp in el:
        if _local(sp.tag) != "spillover":
            continue
        src = _get(sp, "parameter")
        if src is None:
            continue
        coeffs = rows.setdefault(src, {})
        for c in sp:
            if _local(c.tag) == "coefficient":
                det = _get(c, "parameter")
                val = _float(c, "value")
                if det is not None and val is not None:
                    coeffs[det] = val
    if not params:
        params = list(rows.keys())
    n = len(params)
    matrix = np.eye(n)
    for i, src in enumerate(params):
        for det, val in rows.get(src, {}).items():
            if det in params:
                matrix[i, params.index(det)] = val
    return SpilloverMatrix(name, params, matrix, prefix, suffix)


# ----------------------------------------------------------------------- gates
def _dimensions(gel: ET.Element) -> list[ET.Element]:
    return [c for c in gel if c.tag == G + "dimension"]


def _vertex_points(container: ET.Element) -> list[tuple[float, ...]]:
    pts = []
    for v in container:
        if v.tag != G + "vertex":
            continue
        coords = [_float(c, "value", 0.0) for c in v if c.tag == G + "coordinate"]
        pts.append(tuple(coords))
    return pts


def _ellipse_from_xml(gel: ET.Element, dims: list[str]) -> EllipseGate:
    edge = next((c for c in gel if c.tag == G + "edge"), None)
    if edge is not None:
        pts = _vertex_points(edge)
        if len(pts) >= 4:
            arr = np.array([p[:2] for p in pts[:4]], dtype=float)
            center = arr.mean(axis=0)
            return EllipseGate(dims, tuple(center), tuple(arr[0]), tuple(arr[2]))
    # Gating-ML form: mean, covariance matrix and squared Mahalanobis distance
    mean_el = next((c for c in gel if c.tag == G + "mean"), None)
    cov_el = next((c for c in gel if c.tag == G + "covarianceMatrix"), None)
    dist_el = next((c for c in gel if c.tag == G + "distanceSquare"), None)
    if mean_el is None or cov_el is None:
        raise ValueError("ellipsoid gate without edge points or covariance")
    mean = np.array([_float(c, "value", 0.0) for c in mean_el if c.tag == G + "coordinate"][:2])
    rows = [[_float(e, "value", 0.0) for e in row if _local(e.tag) == "entry"] for row in cov_el if _local(row.tag) == "row"]
    cov = np.array(rows, dtype=float)[:2, :2]
    dsq = _float(dist_el, "value", 1.0) if dist_el is not None else 1.0
    vals, vecs = np.linalg.eigh(cov * dsq)
    vals = np.maximum(vals, 0.0)
    a = mean + math.sqrt(vals[1]) * vecs[:, 1]
    b = mean + math.sqrt(vals[0]) * vecs[:, 0]
    return EllipseGate(dims, tuple(mean), tuple(a), tuple(b))


def parse_gate(gate_el: ET.Element, report: ImportReport, where: str) -> tuple[Gate | None, list[tuple[str, list[str]]] | None]:
    """Parse the content of a <Gate> element.

    Returns (gate, boolean_spec). For Gating-ML boolean gates the gate is a
    placeholder BooleanGate whose refs are gate ids to resolve later; the
    second item is then [(op, [gate ids])].
    """
    gel = next(iter(gate_el), None)
    if gel is None:
        return None, None
    kind = _local(gel.tag)
    dim_els = _dimensions(gel)
    dims = [_param_name(d) for d in dim_els]
    if any(d is None for d in dims):
        # a Gating-ML new-dimension (ratio transform) or an unnamed dimension:
        # keep the gate on a placeholder name, the population evaluates empty
        for k, (d_el, d) in enumerate(zip(dim_els, dims)):
            if d is None:
                ref = next((_get(c, "transformation-ref") for c in d_el.iter() if _local(c.tag) == "new-dimension"), None)
                label = ref or "unnamed dimension"
                dims[k] = UNRESOLVED_DIMENSION + label
                report.warn(f"{where}: gate dimension {label!r} is a transformed (ratio / new) dimension FlowDesk cannot compute – the population stays empty")
    gate: Gate | None = None
    boolean = None
    if kind == "RectangleGate":
        lo = [_float(d, "min") for d in dim_els]
        hi = [_float(d, "max") for d in dim_els]
        gate = RectangleGate(dims, lo, hi)
    elif kind == "PolygonGate":
        pts = _vertex_points(gel)
        gate = PolygonGate(dims[:2], [(p[0], p[1]) for p in pts if len(p) >= 2])
    elif kind in ("EllipsoidGate", "EllipseGate"):
        try:
            gate = _ellipse_from_xml(gel, dims[:2])
        except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
            report.warn(f"{where}: cannot read ellipsoid gate ({exc})")
            return None, None
    elif kind == "BooleanGate":
        for op_el in gel:
            op = _local(op_el.tag)
            if op in ("and", "or", "not"):
                ids = [_get(r, "ref") for r in op_el if _local(r.tag) == "gateReference"]
                ids = [i for i in ids if i]
                gate = BooleanGate(op, [])
                boolean = [(op, ids)]
                break
        if gate is None:
            report.warn(f"{where}: empty boolean gate")
            return None, None
    else:
        report.warn(f"{where}: unsupported gate type {kind}")
        return None, None
    if gel.get("eventsInside", "1") == "0":
        gate.negated = True
    tint = gel.get("tint")
    if tint and gel.get("isTinted", "0") == "1":
        gate.color = tint
    return gate, boolean


# ---------------------------------------------------------------- populations
def parse_graph(el: ET.Element | None) -> GraphSettings:
    g = GraphSettings()
    if el is None:
        return g
    ptype = (el.get("type") or "Pseudocolor").replace(" ", "").replace("_", "").lower()
    g.plot_type = GRAPH_TYPES.get(ptype, "pseudocolor")
    for ax in el:
        if ax.tag != "Axis":
            continue
        name = ax.get("name") or None
        if ax.get("dimension") == "x":
            g.x = name
        elif ax.get("dimension") == "y":
            g.y = name
    if g.plot_type == "histogram":
        g.y = None
    return g


def parse_statistic(el: ET.Element, report: ImportReport) -> StatNode | None:
    raw = (el.get("name") or "").strip()
    stat = _STAT_BY_LOWER.get(raw.lower()) or _STAT_ALIASES.get(raw.lower())
    if stat is None:
        report.warn(f"unsupported statistic {raw!r}")
        return None
    param = el.get("parameter") or el.get("id") or None
    pct = el.get("percent") or el.get("percentile")
    try:
        percentile = float(pct) if pct not in (None, "") else None
    except ValueError:
        percentile = None
    if stat == "Percentile" and percentile is None:
        percentile = 50.0
    report.statistics += 1
    return StatNode(stat, param, percentile)


class _TreeBuilder:
    """Builds population trees and resolves boolean references afterwards."""

    def __init__(self, report: ImportReport, where: str):
        self.report = report
        self.where = where
        self.gate_ids: dict[str, Population] = {}
        self.pending_names: list[tuple[Population, str, list[str]]] = []
        self.pending_ids: list[tuple[Population, str, list[str]]] = []

    def children_of(self, el: ET.Element, parent: Population):
        """Add the populations / statistics listed under `el` to `parent`."""
        containers = [el] + [c for c in el if c.tag == "Subpopulations"]
        seen: set[str] = set(c.name for c in parent.children)
        for container in containers:
            for child in container:
                if child.tag == "Statistic":
                    st = parse_statistic(child, self.report)
                    if st is not None:
                        parent.stats.append(st)
                elif child.tag in POPULATION_TAGS and container is not el:
                    pop = self.population(child)
                    if pop.name in seen:
                        self.report.warn(f"{self.where}: duplicate population name {pop.name!r} under {parent.path or '/'}")
                    seen.add(pop.name)
                    parent.add_child(pop)

    def population(self, el: ET.Element) -> Population:
        name = el.get("name", "Population")
        pop = Population(name)
        pop.expanded = el.get("expanded", "1") != "0"
        pop.graph = parse_graph(next((c for c in el if c.tag == "Graph"), None))
        self.report.populations += 1
        if el.tag in BOOLEAN_NODES:
            op = BOOLEAN_NODES[el.tag]
            deps = []
            for dep_container in el:
                if dep_container.tag == "Dependents":
                    deps.extend(d.get("name", "") for d in dep_container if d.tag == "Dependent")
            pop.gate = BooleanGate(op, [])
            self.pending_names.append((pop, op, [d for d in deps if d]))
            self.report.boolean_gates += 1
        gate_el = next((c for c in el if c.tag == "Gate"), None)
        if gate_el is not None:
            gid = _get(gate_el, "id")
            gate, boolean = parse_gate(gate_el, self.report, f"{self.where}: {name}")
            if gate is not None and pop.gate is None:
                pop.gate = gate
                self.report.gates += 1
                if boolean:
                    self.pending_ids.append((pop, boolean[0][0], boolean[0][1]))
                    self.report.boolean_gates += 1
            if gid:
                self.gate_ids[gid] = pop
        elif el.tag == "Population":
            self.report.warn(f"{self.where}: population {name!r} has no gate")
        self.children_of(el, pop)
        return pop

    # ------------------------------------------------------------ booleans
    @staticmethod
    def _find_by_name(root: Population, name: str, exclude: Population) -> Population | None:
        for p in root.walk(include_self=False):
            if p is not exclude and p.name == name:
                return p
        return None

    def _resolve_name(self, node: Population, ref: str) -> Population | None:
        parent = node.parent
        root = node.root
        if parent is not None:
            sib = parent.child_by_name(ref)
            if sib is not None and sib is not node:
                return sib
            if "/" in ref:
                hit = parent.find(ref)
                if hit is not None:
                    return hit
        if "/" in ref:
            hit = root.find(ref)
            if hit is not None:
                return hit
        hit = self._find_by_name(root, ref, node)
        if hit is None and "/" in ref:
            hit = self._find_by_name(root, ref.rstrip("/").rsplit("/", 1)[-1], node)
        return hit

    def resolve(self):
        for pop, op, names in self.pending_names:
            refs = []
            for n in names:
                target = self._resolve_name(pop, n)
                if target is None:
                    self.report.warn(f"{self.where}: boolean gate {pop.name!r} references unknown population {n!r}")
                else:
                    refs.append(target.path)
            pop.gate = BooleanGate(op, refs)
        for pop, op, ids in self.pending_ids:
            refs = []
            for gid in ids:
                target = self.gate_ids.get(gid)
                if target is None:
                    self.report.warn(f"{self.where}: boolean gate {pop.name!r} references unknown gate id {gid!r}")
                else:
                    refs.append(target.path)
            negated = pop.gate.negated if pop.gate is not None else False
            pop.gate = BooleanGate(op, refs)
            pop.gate.negated = negated


# -------------------------------------------------------------- file lookup
_MATCH_KEYS = ("$DATE", "$BTIM", "$ETIM", "$TOT")


class _FileFinder:
    def __init__(self, wsp_dir: Path, search_dirs: list[Path]):
        self.wsp_dir = wsp_dir
        self.dirs = [d for d in search_dirs if d != wsp_dir]
        self._listing: list[Path] | None = None
        self._headers: dict[Path, dict[str, str]] = {}
        self.used: set[Path] = set()

    def files(self) -> list[Path]:
        """FCS files in the workspace folder (top level only) and in the
        search folders (recursively, depth <= 3)."""
        if self._listing is None:
            out: list[Path] = []
            seen: set[Path] = set()
            try:
                top = sorted(self.wsp_dir.iterdir()) if self.wsp_dir.is_dir() else []
            except OSError:
                top = []
            for p in top:
                if p.suffix.lower() in (".fcs", ".lmd") and p.is_file():
                    seen.add(p)
                    out.append(p)
            for base in self.dirs:
                if not base.is_dir():
                    continue
                base_depth = len(base.parts)
                for dirpath, dirnames, filenames in os.walk(base):
                    depth = len(Path(dirpath).parts) - base_depth
                    if depth >= 3:
                        dirnames[:] = []
                    for fn in filenames:
                        if fn.lower().endswith((".fcs", ".lmd")):
                            p = Path(dirpath) / fn
                            if p not in seen:
                                seen.add(p)
                                out.append(p)
            self._listing = out
        return self._listing

    def by_basename(self, name: str) -> Path | None:
        direct = self.wsp_dir / name
        if direct.is_file():
            return direct
        for p in self.files():
            if p.name == name:
                return p
        return None

    def by_keywords(self, keywords: dict[str, str]) -> Path | None:
        want = {k: keywords.get(k, "").strip() for k in _MATCH_KEYS}
        if not want["$BTIM"] or not want["$DATE"]:
            return None
        for p in self.files():
            if p in self.used:
                continue
            kw = self._headers.get(p)
            if kw is None:
                try:
                    kw = FCSFile.read_keywords(p)[1]
                except (OSError, FCSError):
                    kw = {}
                self._headers[p] = kw
            if all(kw.get(k, "").strip() == v for k, v in want.items()):
                return p
        return None


def uri_to_path(uri: str, base_dir: Path | None = None) -> Path:
    uri = uri.strip()
    if uri.lower().startswith("file:"):
        rest = uri[5:]
        if rest.startswith("//"):
            rest = rest[2:]
            if not rest.startswith("/"):  # file://host/path
                rest = "/" + rest.split("/", 1)[-1] if "/" in rest else "/" + rest
        path = urllib.parse.unquote(rest)
        if re.match(r"^/[A-Za-z]:[/\\]", path):
            path = path[1:]
    else:
        path = urllib.parse.unquote(uri)
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p


# ----------------------------------------------------------------- workspace
def _parse_keywords(sample_el: ET.Element) -> dict[str, str]:
    kws: dict[str, str] = {}
    kel = next((c for c in sample_el if c.tag == "Keywords"), None)
    if kel is not None:
        for k in kel:
            if k.tag == "Keyword" and k.get("name") is not None:
                kws[k.get("name")] = k.get("value", "")
    return kws


def _parse_group(node: ET.Element, report: ImportReport) -> Group:
    gname = node.get("name", "Group")
    gel = next((c for c in node if c.tag == "Group"), None)
    role_attr = gel.get("role", "") if gel is not None else ""
    if gname == ALL_SAMPLES:
        role = "all"
    elif role_attr == COMPENSATION_ROLE:
        role = "compensation"
    else:
        role = "custom"
    criteria: list[Criterion] = []
    color = "#000000"
    bold = False
    live = True
    ids: set[int] = set()
    if gel is not None:
        color = gel.get("foreground") or color
        bold = "bold" in (gel.get("fontStyle") or "").lower()
        live = gel.get("live", "1") != "0"
        for c in gel:
            if c.tag == "Criteria":
                for cr in c:
                    if cr.tag == "Criterion":
                        criteria.append(Criterion(cr.get("keyword", "$FIL"), cr.get("function", "Contains"), cr.get("value", ""), cr.get("connector", "And")))
            elif c.tag == "SampleRefs":
                for ref in c:
                    try:
                        ids.add(int(ref.get("sampleID", "")))
                    except ValueError:
                        pass
    group = Group(gname, role, criteria, color, bold)
    group.live = live
    group.sample_ids = ids
    group.expanded = node.get("expanded", "1") != "0"
    group.root.graph = parse_graph(next((c for c in node if c.tag == "Graph"), None))
    builder = _TreeBuilder(report, f"group {gname}")
    builder.children_of(node, group.root)
    builder.resolve()
    report.groups += 1
    return group


def _derived_parameter_names(sample_el: ET.Element) -> list[str]:
    """Names of the FlowJo derived parameters defined for a sample."""
    out = []
    for el in sample_el.iter():
        if _local(el.tag) == "DerivedParameter" and el.get("name"):
            out.append(el.get("name"))
    return out


def missing_parameters(sample: Sample) -> dict[str, list[str]]:
    """Gate dimensions of `sample`'s populations that the sample cannot
    provide: {parameter: [population paths]}. Empty when the sample has no
    channel information (file and keywords missing). A compensated name of
    an identity or disabled matrix is provided: it reads the raw channel.
    Placeholder dimensions (`UNRESOLVED_DIMENSION`) are not listed, parsing
    the gate already reported them."""
    if not sample.channels:
        return {}
    channels = set(sample.channel_names)
    known = set(sample.parameter_names())
    known.update(c.stain for c in sample.channels if c.stain)
    comp = sample.comp

    def missing(d: str) -> bool:
        if d in known or d.startswith(UNRESOLVED_DIMENSION):
            return False
        # identity / disabled matrix: "Comp-X" reads the raw channel X
        return comp is None or comp.base_name(d) not in channels

    out: dict[str, list[str]] = {}
    for pop in sample.root.walk(include_self=False):
        gate = pop.gate
        if gate is None or isinstance(gate, BooleanGate):
            continue
        for d in gate.dims:
            if missing(d):
                paths = out.setdefault(d, [])
                if pop.path not in paths:
                    paths.append(pop.path)
    return out


def _warn_missing(report: ImportReport, missing: dict[str, dict], flowjo_derived: dict[str, None]):
    for param, entry in missing.items():
        samples = entry["samples"]
        paths = list(entry["paths"])
        if param in flowjo_derived:
            what = "a FlowJo derived parameter (not imported)"
        else:
            what = "not a parameter of the FCS file"
        shown = ", ".join(paths[:3]) + (", …" if len(paths) > 3 else "")
        where = ", ".join(samples[:3]) + (f" and {len(samples) - 3} more" if len(samples) > 3 else "")
        report.warn(f"{param} is {what}: {len(paths)} population(s) gated on it stay empty ({shown}) in {where}")
    unused = [n for n in flowjo_derived if n not in missing]
    if unused:
        report.warn(f"FlowJo derived parameters are not imported: {', '.join(unused)}")


def import_wsp(path, search_dirs=None) -> tuple[Workspace, ImportReport]:
    path = Path(path)
    report = ImportReport()
    tree = ET.parse(path)
    root = tree.getroot()
    if _local(root.tag) != "Workspace":
        raise ValueError(f"{path} is not a FlowJo workspace")
    ws = Workspace()
    ws.path = None
    ws.settings["imported_from"] = str(path)
    if root.get("flowJoVersion"):
        ws.settings["flowjo_version"] = root.get("flowJoVersion")

    # workspace-level matrices
    mats = root.find("Matrices")
    if mats is not None:
        for m in mats:
            if m.tag == T + "spilloverMatrix":
                matrix = parse_spillover_matrix(m)
                ws.matrices[matrix.name] = matrix
                report.matrices += 1

    # groups
    groups: list[Group] = []
    groups_el = root.find("Groups")
    if groups_el is not None:
        for node in groups_el:
            if node.tag == "GroupNode":
                groups.append(_parse_group(node, report))
    all_groups = [g for g in groups if g.role == "all"]
    if all_groups:
        groups.remove(all_groups[0])
        groups.insert(0, all_groups[0])
    else:
        groups.insert(0, Group(ALL_SAMPLES, role="all", bold=True))
    ws.groups = groups

    # samples
    dirs = [Path(d) for d in (search_dirs or [])]
    finder = _FileFinder(path.parent, dirs)
    flowjo_derived: dict[str, None] = {}
    missing: dict[str, dict] = {}
    sample_list = root.find("SampleList")
    for sel in (sample_list if sample_list is not None else []):
        if sel.tag != "Sample":
            continue
        ds = sel.find("DataSet")
        node = sel.find("SampleNode")
        sid_text = (ds.get("sampleID") if ds is not None else None) or (node.get("sampleID") if node is not None else None)
        try:
            sid = int(sid_text)
        except (TypeError, ValueError):
            sid = ws.next_sample_id()
            report.warn(f"sample without a valid sampleID, assigned {sid}")
        keywords = _parse_keywords(sel)
        uri = ds.get("uri", "") if ds is not None else ""
        orig = uri_to_path(uri, path.parent) if uri else Path(node.get("name", f"sample{sid}.fcs") if node is not None else f"sample{sid}.fcs")
        fpath: Path | None = orig if orig.is_file() else None
        if fpath is None:
            fpath = finder.by_basename(orig.name)
            if fpath is None and node is not None and node.get("name"):
                fpath = finder.by_basename(node.get("name"))
        if fpath is None:
            fpath = finder.by_keywords(keywords)
            if fpath is not None:
                report.warn(f"sample {sid}: {orig.name} not found, matched {fpath} by acquisition keywords")
        if fpath is None:
            fpath = orig
            report.warn(f"sample {sid}: FCS file not found: {orig}")
            report.samples_missing += 1
        else:
            finder.used.add(fpath)
            report.samples_found += 1
        sample = Sample(fpath, sid)
        if node is not None and node.get("name"):
            sample.name = node.get("name")
        sample.keywords = dict(keywords)
        if fpath.is_file():
            if not sample.load_header():
                report.warn(f"sample {sid}: cannot read {fpath}: {sample.load_error}")
        elif keywords:
            sample.channels = channels_from_keywords(keywords)
        # transforms
        tel = sel.find("Transformations")
        if tel is not None:
            for t in tel:
                parsed = parse_transform(t, report)
                if parsed is not None:
                    sample.scales[parsed[0]] = parsed[1]
        # sample-level compensation (the workspace wins over $SPILLOVER)
        mel = sel.find(T + "spilloverMatrix")
        if mel is not None:
            sample.comp = parse_spillover_matrix(mel)
            sample.comp_enabled = True
            report.matrices += 1
        # population tree
        sample.root.name = sample.name
        if node is not None:
            sample.root.graph = parse_graph(node.find("Graph"))
            sample.root.expanded = node.get("expanded", "1") != "0"
            builder = _TreeBuilder(report, f"sample {sid}")
            builder.children_of(node, sample.root)
            builder.resolve()
        for name in _derived_parameter_names(sel):
            flowjo_derived.setdefault(name, None)
        for param, paths in missing_parameters(sample).items():
            entry = missing.setdefault(param, {"samples": [], "paths": {}})
            entry["samples"].append(sample.name)
            for p in paths:
                entry["paths"].setdefault(p, None)
        ws.samples.append(sample)
        report.samples += 1
    _warn_missing(report, missing, flowjo_derived)
    ws._next_id = max([s.id for s in ws.samples], default=0) + 1
    ws.modified = False
    return ws, report


def flowjo_counts(path) -> dict[tuple[int, str], int]:
    """FlowJo's stored event counts: (sample id, population path) -> count.
    The sample itself is stored under the path ""."""
    root = ET.parse(Path(path)).getroot()
    out: dict[tuple[int, str], int] = {}

    def walk(el: ET.Element, sid: int, prefix: str):
        for sub in el:
            if sub.tag != "Subpopulations":
                continue
            for child in sub:
                if child.tag not in POPULATION_TAGS:
                    continue
                p = f"{prefix}/{child.get('name', '')}"
                try:
                    c = int(float(child.get("count", "-1")))
                except ValueError:
                    c = -1
                out.setdefault((sid, p), c)
                walk(child, sid, p)

    sl = root.find("SampleList")
    for sel in (sl if sl is not None else []):
        node = sel.find("SampleNode")
        if node is None:
            continue
        ds = sel.find("DataSet")
        try:
            sid = int((ds.get("sampleID") if ds is not None else None) or node.get("sampleID"))
        except (TypeError, ValueError):
            continue
        try:
            out[(sid, "")] = int(float(node.get("count", "-1")))
        except ValueError:
            out[(sid, "")] = -1
        walk(node, sid, "")
    return out
