"""Export a workspace as a FlowJo-10-style workspace file (.wsp).

Writes the same element structure that wsp_import reads: Groups with their
templates, SampleList with Gating-ML 2.0 transforms, keywords, spillover
matrices and the population tree (Gating-ML gates, FlowJo boolean nodes,
statistics and graph settings). Layouts and tables are written as empty
editors.
"""
from __future__ import annotations

import math
import re
import time
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .compensation import SpilloverMatrix
from .gates import BooleanGate, EllipseGate, Gate, PolygonGate, RectangleGate
from .model import GraphSettings, Group, Population, Sample, Workspace
from .transforms import ArcsinhScale, LinearScale, LogicleScale, LogScale, Scale
from .wsp_import import ALL_SAMPLES, COMPENSATION_ROLE, DATATYPE_NS, GATING_NS, TRANSFORMS_NS, XSI_NS

G = "{%s}" % GATING_NS
T = "{%s}" % TRANSFORMS_NS
D = "{%s}" % DATATYPE_NS
XSI = "{%s}" % XSI_NS

ET.register_namespace("gating", GATING_NS)
ET.register_namespace("transforms", TRANSFORMS_NS)
ET.register_namespace("data-type", DATATYPE_NS)
ET.register_namespace("xsi", XSI_NS)

SCHEMA_LOCATION = (
    f"{GATING_NS} {GATING_NS}/Gating-ML.v2.0.xsd "
    f"{TRANSFORMS_NS} {GATING_NS}/Transformations.v2.0.xsd "
    f"{DATATYPE_NS} {GATING_NS}/DataTypes.v2.0.xsd "
)
FLOWJO_VERSION = "10.10.0"

GRAPH_TYPE_OUT = {
    "pseudocolor": "Pseudocolor",
    "dot": "DotPlot",
    "contour": "Contour",
    "zebra": "Zebra",
    "density": "Density",
    "histogram": "Histogram",
}

_INVALID_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _clean(text) -> str:
    return _INVALID_XML.sub("", "" if text is None else str(text))


def _num(v: float) -> str:
    v = float(v)
    if math.isfinite(v) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def path_to_uri(path: Path) -> str:
    p = Path(path)
    try:
        p = p.absolute()
    except OSError:
        pass
    posix = p.as_posix()
    if not posix.startswith("/"):
        posix = "/" + posix
    return "file:" + urllib.parse.quote(posix, safe="/+()!,;=:@&$'*~-._")


class _Ids:
    def __init__(self):
        self._used: set[str] = set()
        self._rng = np.random.default_rng()

    def new(self) -> str:
        while True:
            gid = f"ID{int(self._rng.integers(10_000_000, 2_000_000_000))}"
            if gid not in self._used:
                self._used.add(gid)
                return gid


# --------------------------------------------------------------- transforms
def transform_element(name: str, scale: Scale) -> ET.Element | None:
    if isinstance(scale, LinearScale):
        el = ET.Element(T + "linear", {T + "minRange": _num(scale.min), T + "maxRange": _num(scale.max), "gain": "1"})
    elif isinstance(scale, LogScale):
        el = ET.Element(T + "log", {T + "offset": _num(scale.min), T + "decades": _num(scale.decades)})
    elif isinstance(scale, LogicleScale):
        if scale.A >= 0:
            el = ET.Element(
                T + "biex",
                {
                    T + "length": "256",
                    T + "maxRange": _num(scale.T),
                    T + "neg": _num(scale.A),
                    T + "width": _num(scale.width_basis),
                    T + "pos": _num(scale.M),
                },
            )
        else:
            el = ET.Element(T + "logicle", {T + "T": _num(scale.T), T + "W": _num(scale.W), T + "M": _num(scale.M), T + "A": _num(scale.A)})
    elif isinstance(scale, ArcsinhScale):
        el = ET.Element(T + "fasinh", {T + "length": "256", T + "T": _num(scale.T), T + "A": _num(scale.A), T + "M": _num(scale.M)})
    else:
        return None
    ET.SubElement(el, D + "parameter", {D + "name": name})
    return el


def spillover_element(m: SpilloverMatrix, editable: bool = True) -> ET.Element:
    el = ET.Element(
        T + "spilloverMatrix",
        {
            "spectral": "0",
            "prefix": m.prefix,
            "name": m.name,
            "editable": "1" if editable else "0",
            "color": "#c0c0c0",
            "version": f"FlowJo-{FLOWJO_VERSION}",
            "status": "FINALIZED",
            T + "id": str(uuid.uuid4()),
            "suffix": m.suffix,
        },
    )
    params = ET.SubElement(el, D + "parameters")
    for p in m.params:
        ET.SubElement(params, D + "parameter", {D + "name": p, "userProvidedCompInfix": m.compensated_name(p)})
    for i, src in enumerate(m.params):
        row = ET.SubElement(el, T + "spillover", {D + "parameter": src, "userProvidedCompInfix": m.compensated_name(src)})
        for j, det in enumerate(m.params):
            ET.SubElement(row, T + "coefficient", {D + "parameter": det, T + "value": _num(m.matrix[i, j])})
    return el


# --------------------------------------------------------------------- gates
def _dimension(parent: ET.Element, name: str, lo=None, hi=None) -> ET.Element:
    attrs = {}
    if lo is not None:
        attrs[G + "min"] = _num(lo)
    if hi is not None:
        attrs[G + "max"] = _num(hi)
    d = ET.SubElement(parent, G + "dimension", attrs)
    ET.SubElement(d, D + "fcs-dimension", {D + "name": name})
    return d


def _vertex(parent: ET.Element, pt) -> ET.Element:
    v = ET.SubElement(parent, G + "vertex")
    for c in pt:
        ET.SubElement(v, G + "coordinate", {D + "value": _num(c)})
    return v


def _gate_common(gate: Gate) -> dict:
    attrs = {
        "eventsInside": "0" if gate.negated else "1",
        "annoOffsetX": "0",
        "annoOffsetY": "0",
        "tint": gate.color or "#000000",
        "isTinted": "1" if gate.color else "0",
        "lineWeight": "Normal",
        "userDefined": "1",
    }
    return attrs


def gate_element(gate: Gate, gid: str, parent_gid: str | None) -> ET.Element:
    attrs = {G + "id": gid}
    if parent_gid:
        attrs[G + "parent_id"] = parent_gid
    wrapper = ET.Element("Gate", attrs)
    common = _gate_common(gate)
    if isinstance(gate, RectangleGate):
        gel = ET.SubElement(wrapper, G + "RectangleGate", common)
        for dim, lo, hi in zip(gate.dims, gate.lo, gate.hi):
            _dimension(gel, dim, lo, hi)
    elif isinstance(gate, PolygonGate):
        common.update({"quadId": "-1", "gateResolution": "256"})
        gel = ET.SubElement(wrapper, G + "PolygonGate", common)
        for dim in gate.dims[:2]:
            _dimension(gel, dim)
        for v in gate.vertices:
            _vertex(gel, v)
    elif isinstance(gate, EllipseGate):
        gel = ET.SubElement(wrapper, G + "EllipsoidGate", common)
        for dim in gate.dims[:2]:
            _dimension(gel, dim)
        c = np.array(gate.center)
        u = np.array(gate.a_end) - c
        v = np.array(gate.b_end) - c
        # foci along the longer semi-axis (exact when the axes are orthogonal)
        major, minor = (u, v) if np.linalg.norm(u) >= np.linalg.norm(v) else (v, u)
        la, lb = np.linalg.norm(major), np.linalg.norm(minor)
        fd = math.sqrt(max(la * la - lb * lb, 0.0))
        dirv = major / la if la > 0 else np.array([1.0, 0.0])
        foci = ET.SubElement(gel, G + "foci")
        _vertex(foci, c + fd * dirv)
        _vertex(foci, c - fd * dirv)
        edge = ET.SubElement(gel, G + "edge")
        for pt in (c + u, c - u, c + v, c - v):
            _vertex(edge, pt)
        # Gating-ML form: {x : (x-mean)' inv(S) (x-mean) <= 1} with S = M M'
        M = np.column_stack([u, v])
        S = M @ M.T
        mean = ET.SubElement(gel, G + "mean")
        for val in c:
            ET.SubElement(mean, G + "coordinate", {D + "value": _num(val)})
        cov = ET.SubElement(gel, G + "covarianceMatrix")
        for r in range(2):
            row = ET.SubElement(cov, G + "row")
            for k in range(2):
                ET.SubElement(row, G + "entry", {D + "value": _num(S[r, k])})
        ET.SubElement(gel, G + "distanceSquare", {D + "value": "1"})
    else:
        raise ValueError(f"cannot export gate {gate!r}")
    return wrapper


# --------------------------------------------------------------- populations
def graph_element(g: GraphSettings) -> ET.Element:
    el = ET.Element("Graph", {"smoothing": "0", "backColor": "#ffffff", "foreColor": "#000000", "type": GRAPH_TYPE_OUT.get(g.plot_type, "Pseudocolor"), "fast": "1"})
    ET.SubElement(el, "Axis", {"dimension": "x", "name": g.x or "", "label": "", "auto": "auto"})
    y = "" if g.plot_type == "histogram" else (g.y or "")
    ET.SubElement(el, "Axis", {"dimension": "y", "name": y, "label": "", "auto": "auto"})
    return el


def _node_attrs(name: str, owning: str, expanded: bool, count: int) -> dict:
    return {"name": _clean(name), "annotation": "", "owningGroup": owning, "expanded": "1" if expanded else "0", "sortPriority": "10", "count": str(count)}


class _Writer:
    def __init__(self, with_counts: bool):
        self.ids = _Ids()
        self.with_counts = with_counts

    def count(self, pop: Population) -> int:
        if self.with_counts and pop.sample is not None and pop.sample.loaded:
            return pop.count
        return -1

    def dependent_name(self, node: Population, ref: str, root: Population) -> str:
        target = root.find(ref)
        if target is not None and node.parent is not None and target.parent is node.parent:
            return target.name
        return ref.strip("/")

    def subpopulations(self, parent_el: ET.Element, pop: Population, parent_gid: str | None, owning: str, root: Population):
        if not pop.children and not pop.stats:
            return
        sub = ET.SubElement(parent_el, "Subpopulations")
        for child in pop.children:
            self.population(sub, child, parent_gid, owning, root)
        for st in pop.stats:
            attrs = {"name": st.statistic, "id": st.param or ""}
            if st.percentile is not None:
                attrs["percent"] = _num(st.percentile)
            ET.SubElement(sub, "Statistic", attrs)

    def population(self, parent_el: ET.Element, pop: Population, parent_gid: str | None, owning: str, root: Population):
        gate = pop.gate
        attrs = _node_attrs(pop.name, owning, pop.expanded, self.count(pop))
        if isinstance(gate, BooleanGate):
            tag = {"not": "NotNode", "and": "AndNode", "or": "OrNode"}.get(gate.op, "AndNode")
            el = ET.SubElement(parent_el, tag, attrs)
            el.append(graph_element(pop.graph))
            deps = ET.SubElement(el, "Dependents")
            for ref in gate.refs:
                ET.SubElement(deps, "Dependent", {"name": _clean(self.dependent_name(pop, ref, root))})
            gid = parent_gid
        else:
            el = ET.SubElement(parent_el, "Population", attrs)
            el.append(graph_element(pop.graph))
            gid = parent_gid
            if gate is not None:
                gid = self.ids.new()
                el.append(gate_element(gate, gid, parent_gid))
        self.subpopulations(el, pop, gid, owning, root)


# ----------------------------------------------------------------- workspace
def _group_role(group: Group) -> str:
    return COMPENSATION_ROLE if group.role == "compensation" else "ws.group.dlog.test"


def build_tree(ws: Workspace, with_counts: bool = True) -> ET.ElementTree:
    writer = _Writer(with_counts)
    root = ET.Element(
        "Workspace",
        {
            "version": "20.0",
            "modDate": time.strftime("%a %b %d %H:%M:%S %Z %Y"),
            "clientTimestamp": str(int(time.time() * 1000)),
            "flowJoVersion": FLOWJO_VERSION,
            "drawRowBorders": "1",
            "drawColumnBorders": "1",
            "hideCompNodes": "1",
            "curGroup": ALL_SAMPLES,
            "groupPaneHeight": "80",
            XSI + "schemaLocation": SCHEMA_LOCATION,
        },
    )
    mats = ET.SubElement(root, "Matrices")
    for m in ws.matrices.values():
        mats.append(spillover_element(m))
    ET.SubElement(root, "Cytometers")

    groups = ET.SubElement(root, "Groups")
    for group in ws.groups:
        name = ALL_SAMPLES if group.role == "all" else group.name
        node = ET.SubElement(groups, "GroupNode", _node_attrs(name, name, group.expanded, -1))
        node.append(graph_element(group.root.graph))
        writer.subpopulations(node, group.root, None, name, group.root)
        gel = ET.SubElement(
            node,
            "Group",
            {
                "name": _clean(name),
                "live": "1" if group.live else "0",
                "role": _group_role(group),
                "key": "",
                "synchronized": "0",
                "foreground": group.color or "#000000",
                "fontStyle": "bold" if group.bold else "plain",
            },
        )
        crit = ET.SubElement(gel, "Criteria")
        for c in group.criteria:
            ET.SubElement(crit, "Criterion", {"connector": c.connector, "keyword": _clean(c.keyword), "function": c.function, "value": _clean(c.value)})
        ids = sorted(s.id for s in ws.members(group))
        if ids:
            refs = ET.SubElement(gel, "SampleRefs")
            for sid in ids:
                ET.SubElement(refs, "SampleRef", {"sampleID": str(sid)})
        ET.SubElement(gel, "Keywords")

    slist = ET.SubElement(root, "SampleList")
    for sample in ws.samples:
        _sample_element(slist, sample, writer)

    ET.SubElement(root, "TableEditor", {"title": "Table Editor", "current": ""})
    ET.SubElement(root, "LayoutEditor", {"title": "Layout Editor", "current": ""})
    return ET.ElementTree(root)


def _sample_element(parent: ET.Element, sample: Sample, writer: _Writer):
    sel = ET.SubElement(parent, "Sample")
    ET.SubElement(sel, "DataSet", {"uri": path_to_uri(sample.path), "sampleID": str(sample.id)})
    tel = ET.SubElement(sel, "Transformations")
    for name, scale in sample.scales.items():
        el = transform_element(name, scale)
        if el is not None:
            tel.append(el)
    kel = ET.SubElement(sel, "Keywords")
    for k, v in sample.keywords.items():
        ET.SubElement(kel, "Keyword", {"name": _clean(k), "value": _clean(v)})
    if sample.comp is not None and sample.comp_enabled:
        sel.append(spillover_element(sample.comp, editable=False))
    count = sample.n_events if sample.loaded or sample.keywords else -1
    node = ET.SubElement(sel, "SampleNode", {**_node_attrs(sample.name, "", sample.root.expanded, count), "sampleID": str(sample.id)})
    node.append(graph_element(sample.root.graph))
    writer.subpopulations(node, sample.root, None, "", sample.root)


def export_wsp(ws: Workspace, path, with_counts: bool = True):
    """Write `ws` as a FlowJo 10 workspace. Population counts are written for
    samples whose event data is already loaded (otherwise -1)."""
    tree = build_tree(ws, with_counts)
    ET.indent(tree, space="  ")
    path = Path(path)
    with open(path, "wb") as fh:
        tree.write(fh, encoding="UTF-8", xml_declaration=True)
