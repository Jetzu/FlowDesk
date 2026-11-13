"""Workspace model: samples, population hierarchies and groups."""
from __future__ import annotations

import itertools
import json
import math
import operator
import os
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compensation import ACQUISITION_DEFINED, SpilloverMatrix
from .fcs import Channel, FCSError, FCSFile
from .gates import BooleanGate, Gate, RectangleGate, new_id
from .transforms import Scale, default_scale_for_channel

COMP_PREFIX = "Comp-"

# Operations of derived parameters contributed by other modules: op name ->
# function(sample, spec) returning one value per event. Sample._derived_values
# consults it for ops it does not implement itself (ratio, product, sum,
# difference, log10, identity). A spec may list the parameters it reads under
# "inputs" so that changing other parameters keeps its cached values; without
# it the parameter is recomputed after any data change.
DERIVED_OPS: dict[str, Callable] = {}
BUILTIN_DERIVED_OPS = ("ratio", "product", "sum", "difference", "log10", "identity")

# FCS headers read ahead of Workspace.add_sample (e.g. by a background loader):
# path string -> (mtime_ns, size, version, keywords). load_header() takes an
# entry when the file has not changed since.
HEADER_CACHE: dict[str, tuple] = {}

_USE_CLOCK = itertools.count(1)  # Sample.last_used stamps (least-recently-used order)
_BUDGETS: "weakref.WeakSet[MemoryBudget]" = weakref.WeakSet()


def read_header_entry(path: Path | str) -> tuple:
    """(mtime_ns, size, version, keywords) of an FCS file; safe in a worker thread."""
    st = os.stat(path)
    version, keywords = FCSFile.read_keywords(path)
    return (st.st_mtime_ns, st.st_size, version, keywords)


def _take_cached_header(path: Path) -> tuple[str, dict] | None:
    entry = HEADER_CACHE.pop(os.fspath(path), None)
    if entry is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if (st.st_mtime_ns, st.st_size) != (entry[0], entry[1]):
        return None
    return entry[2], dict(entry[3])


def _derived_inputs(spec: dict) -> list[str] | None:
    """The parameters a derived parameter reads, None when unknown."""
    op = spec.get("op", "ratio")
    if op in BUILTIN_DERIVED_OPS or op not in DERIVED_OPS:
        return [n for n in (spec.get("a"), spec.get("b")) if n]
    inputs = spec.get("inputs")
    return list(inputs) if isinstance(inputs, (list, tuple)) else None


class ScaleMap(dict):
    """``Sample.scales``: one transform per raw parameter.

    As in FlowJo, the transform belongs to the raw parameter: a compensated
    name ("Comp-PE-A") is stored under, and read from, its raw parameter
    ("PE-A"), so the raw and the compensated axis always agree. Entries
    written under a compensated spelling (old .fdw files, templates, direct
    writes before the sample knew its channels) collapse onto the raw key as
    soon as the sample can resolve them; when both spellings were stored, the
    compensated one wins (it was only ever written by an explicit edit, while
    a raw entry may be a cached default).
    """

    def __init__(self, owner: "Sample | None" = None, data=None):
        super().__init__()
        self._owner = weakref.ref(owner) if owner is not None else None
        self._sig: tuple | None = None
        self._dirty = False
        # keys holding a default that Sample.scale() cached (not chosen by anyone)
        self.auto: set[str] = set()
        if data:
            items = list(data.items()) if hasattr(data, "items") else list(data)
            own = self._sample()
            if own is not None:
                items.sort(key=lambda kv: own._scale_key(kv[0]) != kv[0])
            for k, v in items:
                self[k] = v

    def _sample(self) -> "Sample | None":
        return self._owner() if self._owner is not None else None

    def _key(self, name):
        own = self._sample()
        return own._scale_key(name) if own is not None and isinstance(name, str) else name

    def settle(self):
        """Move entries stored under a compensated spelling to their raw key."""
        own = self._sample()
        if own is None:
            return
        sig = (id(own.comp), id(own.channels), len(own.channels))
        if not self._dirty and sig == self._sig:
            return
        self._sig = sig
        self._dirty = False
        moves = [(k, own._scale_key(k)) for k in dict.keys(self) if isinstance(k, str)]
        for k, key in moves:
            if key != k:
                dict.__setitem__(self, key, dict.pop(self, k))
                self.auto.discard(key)
                if k in self.auto:
                    self.auto.discard(k)

    def __setitem__(self, name, scale):
        key = self._key(name)
        dict.__setitem__(self, key, scale)
        if key != name:
            dict.pop(self, name, None)
        self.auto.discard(key)
        self._dirty = True

    def __getitem__(self, name):
        self.settle()
        return dict.__getitem__(self, self._key(name))

    def __contains__(self, name) -> bool:
        self.settle()
        return dict.__contains__(self, self._key(name))

    def __delitem__(self, name):
        self.settle()
        key = self._key(name)
        dict.__delitem__(self, key)
        self.auto.discard(key)

    def get(self, name, default=None):
        self.settle()
        return dict.get(self, self._key(name), default)

    def pop(self, name, *default):
        self.settle()
        key = self._key(name)
        self.auto.discard(key)
        return dict.pop(self, key, *default)

    def setdefault(self, name, default=None):
        self.settle()
        key = self._key(name)
        if not dict.__contains__(self, key):
            self[key] = default
        return dict.__getitem__(self, key)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __iter__(self):
        self.settle()
        return dict.__iter__(self)

    def keys(self):
        self.settle()
        return dict.keys(self)

    def values(self):
        self.settle()
        return dict.values(self)

    def items(self):
        self.settle()
        return dict.items(self)

    def __len__(self) -> int:
        self.settle()
        return dict.__len__(self)

    def copy(self) -> dict:
        return dict(self.items())

    def __reduce__(self):
        return (dict, (dict(self.items()),))

PLOT_TYPES = [
    ("pseudocolor", "Pseudocolor"),
    ("dot", "Dot Plot"),
    ("contour", "Contour Plot"),
    ("zebra", "Zebra Plot"),
    ("density", "Density Plot"),
    ("histogram", "Histogram"),
]


@dataclass
class GraphSettings:
    """What a node's graph window last showed."""

    x: str | None = None
    y: str | None = None
    plot_type: str = "pseudocolor"
    options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "plot_type": self.plot_type, "options": dict(self.options)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "GraphSettings":
        if not d:
            return cls()
        return cls(d.get("x"), d.get("y"), d.get("plot_type", "pseudocolor"), dict(d.get("options", {})))

    def copy(self) -> "GraphSettings":
        return GraphSettings.from_dict(self.to_dict())


@dataclass
class StatNode:
    """A statistic shown as a row below a population in the workspace."""

    statistic: str
    param: str | None = None
    percentile: float | None = None
    id: str = field(default_factory=new_id)

    @property
    def label(self) -> str:
        from .stats import statistic_label

        return statistic_label(self.statistic, self.param, self.percentile)

    def to_dict(self) -> dict:
        return {"statistic": self.statistic, "param": self.param, "percentile": self.percentile, "id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> "StatNode":
        return cls(d["statistic"], d.get("param"), d.get("percentile"), d.get("id", new_id()))


class Population:
    """A node of the gating tree. The root has no gate and holds every event."""

    def __init__(self, name: str, gate: Gate | None = None):
        self.id = new_id()
        self.name = name
        self.gate = gate
        self.parent: Population | None = None
        self.children: list[Population] = []
        self.stats: list[StatNode] = []
        self.graph = GraphSettings()
        self.sample: Sample | None = None
        self.expanded = True
        self._mask: np.ndarray | None = None

    # ----------------------------------------------------------------- tree ops
    def add_child(self, pop: "Population", index: int | None = None) -> "Population":
        pop.parent = self
        if index is None:
            self.children.append(pop)
        else:
            self.children.insert(index, pop)
        pop._attach(self.sample)
        return pop

    def remove_child(self, pop: "Population"):
        if pop in self.children:
            self.children.remove(pop)
            pop.parent = None

    def _attach(self, sample: "Sample | None"):
        self.sample = sample
        self._mask = None
        for c in self.children:
            c._attach(sample)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def path(self) -> str:
        if self.parent is None:
            return ""
        return f"{self.parent.path}/{self.name}"

    @property
    def display_name(self) -> str:
        return self.name if self.parent is not None else "Ungated"

    @property
    def depth(self) -> int:
        d = 0
        p = self.parent
        while p is not None:
            d += 1
            p = p.parent
        return d

    @property
    def root(self) -> "Population":
        p = self
        while p.parent is not None:
            p = p.parent
        return p

    def ancestors(self) -> list["Population"]:
        out = []
        p = self.parent
        while p is not None:
            out.append(p)
            p = p.parent
        out.reverse()
        return out

    def child_by_name(self, name: str) -> "Population | None":
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find(self, path: str) -> "Population | None":
        parts = [p for p in path.split("/") if p]
        node = self
        for part in parts:
            node = node.child_by_name(part)
            if node is None:
                return None
        return node

    def walk(self, include_self: bool = True):
        if include_self:
            yield self
        for c in self.children:
            yield from c.walk(True)

    def clone(self, deep: bool = True) -> "Population":
        p = Population(self.name, self.gate.clone() if self.gate is not None else None)
        p.graph = self.graph.copy()
        p.stats = [StatNode.from_dict(s.to_dict()) for s in self.stats]
        p.expanded = self.expanded
        if deep:
            for c in self.children:
                p.add_child(c.clone(True))
        return p

    # --------------------------------------------------------------- membership
    def invalidate(self):
        self._mask = None
        for c in self.children:
            c.invalidate()

    def invalidate_where(self, gate_depends: Callable[[Gate], bool]) -> int:
        """Selective invalidation over the whole tree this node belongs to:
        drop the cached membership of every population whose own gate depends
        on what changed (``gate_depends(gate)``), of all their descendants and
        of boolean gates combining any of them. Everything else keeps its
        mask. Returns the number of masks dropped."""
        root = self.root
        nodes = list(root.walk())
        dirty: set[int] = set()
        for node in nodes:
            g = node.gate
            if g is None or isinstance(g, BooleanGate):
                continue
            try:
                hit = bool(gate_depends(g))
            except Exception:  # noqa: BLE001 - when in doubt, recompute
                hit = True
            if hit:
                dirty.add(id(node))
        changed = True
        while changed:
            changed = False
            for node in nodes:
                if id(node) in dirty:
                    continue
                hit = node.parent is not None and id(node.parent) in dirty
                if not hit and isinstance(node.gate, BooleanGate):
                    for ref in node.gate.refs:
                        target = root.find(ref)
                        if target is not None and id(target) in dirty:
                            hit = True
                            break
                if hit:
                    dirty.add(id(node))
                    changed = True
        dropped = 0
        for node in nodes:
            if id(node) in dirty and node._mask is not None:
                node._mask = None
                dropped += 1
        return dropped

    def mask(self) -> np.ndarray:
        if self._mask is not None:
            return self._mask
        sample = self.sample
        if sample is None or not sample.ensure_loaded():
            return np.zeros(0, dtype=bool)
        if self.parent is None:
            self._mask = np.ones(sample.n_events, dtype=bool)
        else:
            parent_mask = self.parent.mask()
            if self.gate is None:
                self._mask = parent_mask.copy()
            else:
                try:
                    own = self.gate.membership(sample)
                except KeyError:  # a parameter the sample does not have
                    own = None
                except ValueError:
                    # a computed parameter whose values do not fit the events
                    # (stale data): empty, like a missing parameter
                    if not any(d in sample.computed for d in getattr(self.gate, "dims", ())):
                        raise
                    own = None
                if own is None or np.shape(own) != parent_mask.shape:
                    own = np.zeros(parent_mask.shape, dtype=bool)
                self._mask = parent_mask & own
        return self._mask

    @property
    def count(self) -> int:
        m = self.mask()
        return int(m.sum()) if m.size else 0

    @property
    def freq_of_parent(self) -> float:
        if self.parent is None:
            return 100.0
        n = self.parent.count
        return 100.0 * self.count / n if n else float("nan")

    @property
    def freq_of_total(self) -> float:
        if self.sample is None:
            return float("nan")
        n = self.sample.n_events
        return 100.0 * self.count / n if n else float("nan")

    def __repr__(self):
        return f"<Population {self.path or '/'}>"


class Sample:
    """An FCS file inside the workspace, with its gating tree and axis scales."""

    def __init__(self, path: Path | str, sample_id: int):
        self.id = sample_id
        self.path = Path(path)
        self.name = self.path.name
        self.version = ""
        self.keywords: dict[str, str] = {}
        self.channels: list[Channel] = []
        self.fcs: FCSFile | None = None
        self.load_error: str | None = None
        # problems met while loading the workspace or the data that did not
        # stop the sample from opening (computed parameters that could not be
        # restored, ...), as messages for the user; never saved
        self.load_warnings: list[str] = []
        self.root = Population(self.name)
        self.root.sample = self
        self.comp: SpilloverMatrix | None = None
        self.comp_enabled = True
        self.comp_cleared = False  # the user removed the acquisition matrix on purpose
        self.derived: dict[str, dict] = {}
        # parameters computed by tools (tSNE/UMAP axes, cluster ids, ...): one
        # float32 value per event, NaN where the tool did not run
        self.computed: dict[str, np.ndarray] = {}
        self.computed_info: dict[str, dict] = {}
        self.scales = {}
        self._comp_cache: tuple[SpilloverMatrix, np.ndarray] | None = None
        self._display_cache: dict[str, np.ndarray] = {}
        self._derived_cache: dict[str, np.ndarray] = {}
        self._chan_set: tuple | None = None
        self._was_unloaded = False  # unload() released the data: keep edited channels on reload
        self._detached = False  # a worker-thread copy (detached_copy), not part of a workspace
        self.last_used = 0  # _USE_CLOCK stamp of the last data access

    # ------------------------------------------------------------------- scales map
    @property
    def scales(self) -> ScaleMap:
        return self._scales

    @scales.setter
    def scales(self, value):
        if isinstance(value, ScaleMap) and value._sample() is self:
            self._scales = value
        else:
            self._scales = ScaleMap(self, value or {})

    # ------------------------------------------------------------------ loading
    def _file_is_compensated(self, spill_params: list[str] | None = None) -> bool:
        """The file already holds compensated data (an exported compensated
        population): its matrix must not be applied a second time."""
        flag = next((v for k, v in self.keywords.items() if str(k).upper() == "FLOWDESK_COMPENSATED"), "")
        if str(flag).strip().upper() in ("TRUE", "1", "YES"):
            return True
        if spill_params:
            names = set(self.channel_names)
            return any(f"{COMP_PREFIX}{p}" in names for p in spill_params)
        return False

    def load_header(self) -> bool:
        try:
            cached = _take_cached_header(self.path)
            if cached is not None:
                self.version, self.keywords = cached
            else:
                self.version, self.keywords = FCSFile.read_keywords(self.path)
        except (OSError, FCSError) as exc:
            self.load_error = str(exc)
            return False
        from .fcs import channels_from_keywords, parse_spillover

        self.channels = channels_from_keywords(self.keywords)
        if self.comp is None and not self.comp_cleared:
            spill = parse_spillover(self.keywords)
            if spill is not None and not self._file_is_compensated(spill[0]):
                self.comp = SpilloverMatrix(ACQUISITION_DEFINED, spill[0], spill[1])
        self.load_error = None
        return True

    def ensure_loaded(self) -> bool:
        if self.fcs is not None:
            self.last_used = next(_USE_CLOCK)
            return True
        if self.load_error and not self.path.exists():
            return False
        try:
            fcs = FCSFile.read(self.path)
        except (OSError, FCSError) as exc:
            self.load_error = str(exc)
            return False
        self.attach_fcs(fcs)
        return True

    def attach_fcs(self, fcs: FCSFile):
        """Install event data read from this sample's file (by ensure_loaded,
        or by a loader that read it in a worker thread). GUI thread only."""
        kept = self.channels if self._was_unloaded else None
        self.fcs = fcs
        self.version = fcs.version
        if not self.keywords:
            self.keywords = dict(fcs.keywords)
        if kept and len(kept) == len(fcs.channels):
            fcs.channels = kept  # $PnN / $PnS edited before the data was released
        self.channels = fcs.channels
        if self.comp is None and not self.comp_cleared:
            comp = SpilloverMatrix.from_fcs(fcs)
            if comp is not None and not self._file_is_compensated(comp.params):
                self.comp = comp
        self.load_error = None
        self._was_unloaded = False
        self._drop_stale_computed()
        self.last_used = next(_USE_CLOCK)
        if not self._detached:
            _notify_loaded(self)

    def unload(self) -> bool:
        """Release the event data and everything computed from it (compensated,
        display and derived values, population masks). Header, keywords,
        scales, gates and computed parameters stay; the next data() / mask()
        reads the file again. Returns False (keeping the data) when nothing was
        loaded or the file cannot be read back."""
        if self.fcs is None or self._detached or not self.path.is_file():
            return False
        self._was_unloaded = True
        self.fcs = None
        self._comp_cache = None
        self._display_cache.clear()
        self._derived_cache.clear()
        self.root.invalidate()
        return True

    def release_events(self) -> bool:
        """Alias of unload()."""
        return self.unload()

    def resident_bytes(self) -> int:
        """Bytes held for this sample's events and what is derived from them
        (computed parameters, which cannot be re-read, are not counted)."""
        if self.fcs is None:
            return 0
        total = int(getattr(self.fcs.events, "nbytes", 0) or 0)
        if self._comp_cache is not None:
            total += int(self._comp_cache[1].nbytes)
        total += sum(int(a.nbytes) for a in self._display_cache.values())
        total += sum(int(a.nbytes) for a in self._derived_cache.values())
        total += sum(int(p._mask.nbytes) for p in self.root.walk() if p._mask is not None)
        return total

    def _drop_stale_computed(self):
        """Computed parameters restored from a workspace whose length does not
        match the events of the file now on disk (the FCS file was replaced or
        relocated to a different one) cannot belong to it: drop them, with a
        load warning, instead of letting every mask on them fail."""
        n = self.n_events
        stale = [k for k, v in self.computed.items() if np.shape(v) != (n,)]
        for name in stale:
            size = int(np.size(self.computed.pop(name)))
            self.computed_info.pop(name, None)
            self._display_cache.pop(name, None)
            self.load_warnings.append(
                f"computed parameter {name!r} was dropped: {self.path.name} has {n} events, the saved values are for {size}"
            )
        if stale:
            self._derived_cache.clear()
            self.root.invalidate()

    @property
    def loaded(self) -> bool:
        return self.fcs is not None

    @property
    def n_events(self) -> int:
        if self.fcs is not None:
            return self.fcs.n_events
        try:
            return int(float(self.keywords.get("$TOT", "0") or 0))
        except ValueError:
            return 0

    @property
    def display_count(self) -> int:
        return self.n_events

    # --------------------------------------------------------------- parameters
    @property
    def channel_names(self) -> list[str]:
        return [c.name for c in self.channels]

    def _comp_active(self) -> SpilloverMatrix | None:
        if self.comp is not None and self.comp_enabled and not self.comp.is_identity:
            return self.comp
        return None

    def _channel_set(self) -> frozenset:
        cached = self._chan_set
        if cached is None or cached[0] is not self.channels or cached[1] != len(self.channels):
            cached = (self.channels, len(self.channels), frozenset(c.name for c in self.channels))
            self._chan_set = cached
        return cached[2]

    def _scale_key(self, name: str) -> str:
        """The key a parameter's transform is stored under: the raw parameter
        for a compensated name (FlowJo keeps one transform per raw parameter),
        the name itself otherwise."""
        chans = self._channel_set()
        if name in chans or name in self.derived or name in self.computed:
            return name
        comp = self.comp
        if comp is not None:
            base = comp.base_name(name)
            if base is not None and base != name:
                return base
        if name.startswith(COMP_PREFIX) and name[len(COMP_PREFIX):] in chans:
            return name[len(COMP_PREFIX):]
        return name

    def parameter_names(self) -> list[str]:
        names: list[str] = []
        comp = self._comp_active()
        if comp is not None:
            names.extend(c for c in comp.compensated_names if comp.base_name(c) in self.channel_names)
        names.extend(self.channel_names)
        names.extend(self.derived.keys())
        names.extend(self.computed.keys())
        return names

    def is_compensated_param(self, name: str) -> bool:
        comp = self._comp_active()
        return comp is not None and comp.base_name(name) is not None and name not in self.channel_names

    def base_channel(self, name: str) -> Channel | None:
        for c in self.channels:
            if c.name == name:
                return c
        comp = self.comp
        if comp is not None:
            base = comp.base_name(name)
            if base is not None:
                for c in self.channels:
                    if c.name == base:
                        return c
        for c in self.channels:
            if c.stain == name:
                return c
        return None

    def param_label(self, name: str) -> str:
        ch = self.base_channel(name)
        if ch is None:
            return name
        if ch.stain and ch.stain != ch.name:
            return f"{name} :: {ch.stain}"
        return name

    def param_stain(self, name: str) -> str:
        ch = self.base_channel(name)
        if ch is None:
            return name
        return ch.stain or name

    def default_axes(self) -> tuple[str | None, str | None]:
        names = self.channel_names
        if not names:
            return None, None

        def pick(prefix):
            for n in names:
                if n.upper().startswith(prefix):
                    return n
            return None

        x = pick("FSC-A") or pick("FSC") or names[0]
        y = pick("SSC-A") or pick("SSC") or (names[1] if len(names) > 1 else None)
        return x, y

    # --------------------------------------------------------------------- data
    def data(self, name: str) -> np.ndarray:
        if not self.ensure_loaded():
            raise KeyError(name)
        assert self.fcs is not None
        if name in self.computed:
            values = self.computed[name]
            if values.shape != (self.fcs.n_events,):
                raise KeyError(name)  # stale values (see _drop_stale_computed)
            return values
        if name in self.derived:
            return self._derived_values(name)
        comp = self._comp_active()
        if comp is not None:
            base = comp.base_name(name)
            if base is not None and name not in self.channel_names:
                cache = self._comp_cache
                if cache is None or cache[0] is not comp:
                    values = comp.apply(self.fcs.events, self.channel_names)
                    if values is None:
                        raise KeyError(name)
                    cache = (comp, values)
                    self._comp_cache = cache
                return cache[1][:, comp.params.index(base)]
        if name not in self.channel_names and self.comp is not None:
            # identity or disabled matrix: "Comp-X" is the raw channel X, so
            # gates drawn on compensated parameters keep working
            base = self.comp.base_name(name)
            if base is not None and base in self.channel_names:
                return self.fcs.column(base)
        return self.fcs.column(name)

    def _derived_values(self, name: str) -> np.ndarray:
        if name in self._derived_cache:
            return self._derived_cache[name]
        spec = self.derived[name]
        op = spec.get("op", "ratio")
        if op not in BUILTIN_DERIVED_OPS and op in DERIVED_OPS:
            values = np.asarray(DERIVED_OPS[op](self, spec), dtype=np.float32)
            if values.shape != (self.n_events,):
                raise KeyError(name)
            self._derived_cache[name] = values
            return values
        a = self.data(spec["a"]).astype(np.float64)
        b = self.data(spec["b"]).astype(np.float64) if spec.get("b") else None
        with np.errstate(divide="ignore", invalid="ignore"):
            if op == "ratio" and b is not None:
                out = np.where(b != 0, a / b, 0.0)
            elif op == "product" and b is not None:
                out = a * b
            elif op == "sum" and b is not None:
                out = a + b
            elif op == "difference" and b is not None:
                out = a - b
            elif op == "log10":
                out = np.where(a > 0, np.log10(np.maximum(a, 1e-30)), 0.0)
            else:
                out = a
        values = out.astype(np.float32)
        self._derived_cache[name] = values
        return values

    def to_display(self, name: str, values) -> np.ndarray:
        return self.scale(name).forward(np.asarray(values, dtype=float))

    def display(self, name: str) -> np.ndarray:
        cached = self._display_cache.get(name)
        if cached is None:
            cached = self.scale(name).forward(self.data(name)).astype(np.float32)
            self._display_cache[name] = cached
        return cached

    def resolve(self, path: str) -> np.ndarray | None:
        pop = self.root.find(path)
        return None if pop is None else pop.mask()

    # ------------------------------------------------------------------- scales
    def _is_time_channel(self, ch: Channel | None) -> bool:
        if ch is None or ch.is_log_amplified or (ch.display or "").upper().startswith("LOG"):
            return False
        from .qc import is_time_parameter

        return is_time_parameter(ch.name)

    def time_span(self, name: str) -> tuple[float, float] | None:
        """Axis range covering the acquired time values of a time channel
        (0 .. 2 % above the last event), None before the data is loaded."""
        ch = self.base_channel(name)
        if self.fcs is None or not self._is_time_channel(ch):
            return None
        try:
            values = np.asarray(self.fcs.column(ch.name), dtype=float)
        except (KeyError, ValueError, IndexError):
            return None
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        lo, hi = float(values.min()), float(values.max())
        top = float(math.ceil(hi * 1.02)) if hi > 0 else 0.0
        bottom = 0.0 if lo >= 0 else float(math.floor(lo * 1.02))
        if top <= bottom:
            return None
        return bottom, top

    def default_scale(self, name: str) -> Scale:
        from .transforms import ArcsinhScale, LinearScale, LogicleScale, LogScale, default_biex, normalize_kind

        if name in self.computed:
            info = self.computed_info.get(name, {})
            values = self.computed[name]
            finite = values[np.isfinite(values)]
            if "min" in info and "max" in info:
                return LinearScale(float(info["min"]), float(info["max"]))
            if finite.size == 0:
                return LinearScale(0.0, 1.0)
            lo, hi = float(finite.min()), float(finite.max())
            pad = (hi - lo) * 0.05 or 1.0
            return LinearScale(lo - pad, hi + pad)
        if name in self.derived:
            spec = self.derived[name]
            kind = spec.get("kind")
            if kind:
                kind = normalize_kind(kind, "linear")
                if kind == "linear":
                    return LinearScale(float(spec.get("min", 0.0)), float(spec.get("max", 10.0)))
                if kind == "log":
                    return LogScale(float(spec.get("min", 1.0)), float(spec.get("max", 262144.0)))
                if kind == "arcsinh":
                    return ArcsinhScale(T=float(spec.get("max", 262144.0)))
                return default_biex(float(spec.get("max", 262144.0)))
            if spec.get("op") in ("ratio", "log10"):
                return LinearScale(spec.get("min", 0.0), spec.get("max", 10.0))
            return LogicleScale(T=spec.get("max", 262144.0))
        ch = self.base_channel(name)
        if ch is None:
            return default_scale_for_channel(name)
        top = ch.range if ch.range and ch.range > 1 else 262144.0
        scale = default_scale_for_channel(
            ch.name, ch.display, top, ch.is_log_amplified, log_decades=ch.log_decades, log_offset=ch.log_offset
        )
        if isinstance(scale, LinearScale) and self._is_time_channel(ch):
            span = self.time_span(ch.name)
            if span is not None:
                return LinearScale(*span)
        return scale

    def _default_needs_data(self, key: str) -> bool:
        """The default scale of `key` is computed from the events (time axes)."""
        if key in self.computed or key in self.derived:
            return False
        return self._is_time_channel(self.base_channel(key))

    def scale(self, name: str) -> Scale:
        """The transform of a parameter (shared by a raw parameter and its
        compensated version), its default when nobody chose one."""
        sc = self._scales
        sc.settle()
        key = self._scale_key(name)
        s = dict.get(sc, key)
        if s is None and key != name:
            s = dict.pop(sc, name, None)  # stored under the compensated spelling
            if s is not None:
                dict.__setitem__(sc, key, s)
        if s is not None:
            return s
        s = self.default_scale(key)
        if self.fcs is None and self._default_needs_data(key):
            return s  # header only: computed from the data once it is loaded
        dict.__setitem__(sc, key, s)
        sc.auto.add(key)
        return s

    def has_own_scale(self, name: str) -> bool:
        """A transform was chosen for this parameter (not a cached default)."""
        key = self._scale_key(name)
        return key in self._scales and key not in self._scales.auto

    def set_scale(self, name: str, scale: Scale):
        """Set the transform of a parameter. A compensated name sets its raw
        parameter's transform, which both share. Only the display values and
        the gates that depend on this transform are invalidated."""
        key = self._scale_key(name)
        old = dict.get(self._scales, key)
        self._scales[key] = scale
        if old is not None and old == scale:
            return
        self.invalidate_param(key)

    def reset_scale(self, name: str):
        """Forget the chosen transform: the parameter shows its default again."""
        key = self._scale_key(name)
        if self._scales.pop(key, None) is not None:
            self.invalidate_param(key)

    def invalidate_display(self):
        self._display_cache.clear()
        self.root.invalidate()

    def invalidate_param(self, name: str) -> int:
        """The transform of `name` changed: drop the display values of every
        parameter using that transform (the raw and the compensated name) and
        the masks of gates evaluated in display space on them (polygons,
        ellipses; rectangles and ranges work on data values), their
        descendants and boolean gates combining them. Returns the number of
        masks dropped."""
        return self.invalidate_params([name], data=False)

    def invalidate_params(self, names: Iterable[str], data: bool = False) -> int:
        """Selective invalidation. data=False: the transforms of `names`
        changed. data=True: their values changed (derived parameters computed
        from them follow)."""
        names = list(names)
        if data:
            wanted = set(names)
            return self._invalidate_data(self._data_dependency(lambda n: n in wanted))
        keys = {self._scale_key(n) for n in names}

        def shown(n: str) -> bool:
            return self._scale_key(n) in keys

        for n in [n for n in self._display_cache if shown(n)]:
            del self._display_cache[n]
        return self.root.invalidate_where(lambda g: not isinstance(g, RectangleGate) and any(shown(d) for d in g.dims))

    def _data_dependency(self, changed: Callable[[str], bool]) -> Callable[[str], bool]:
        """Extend `changed` (name -> its values changed) to the derived
        parameters computed from such names."""
        memo: dict[str, bool] = {}

        def dep(n: str, stack: tuple = ()) -> bool:
            if n in memo:
                return memo[n]
            hit = bool(changed(n))
            if not hit and n in self.derived and n not in stack:
                inputs = _derived_inputs(self.derived[n])
                hit = True if inputs is None else any(dep(i, stack + (n,)) for i in inputs)
            memo[n] = hit
            return hit

        return dep

    def _invalidate_data(self, dep: Callable[[str], bool]) -> int:
        for n in [n for n in self._display_cache if dep(n)]:
            del self._display_cache[n]
        for n in [n for n in self._derived_cache if dep(n)]:
            del self._derived_cache[n]
        return self.root.invalidate_where(lambda g: any(dep(d) for d in g.dims))

    def set_compensation(self, matrix: SpilloverMatrix | None, enabled: bool = True):
        """Change the matrix. Raw parameters, computed parameters and
        everything derived only from them keep their caches and masks."""
        self.comp = matrix
        self.comp_enabled = enabled
        self.comp_cleared = matrix is None
        self._comp_cache = None
        raw = self._channel_set()

        def changed(n: str) -> bool:
            # compensated names, and names nothing provides (a matrix may add them)
            return n not in raw and n not in self.computed and n not in self.derived

        self._invalidate_data(self._data_dependency(changed))

    def add_derived(self, name: str, spec: dict):
        self.derived[name] = dict(spec)
        if name in self._scales.auto:
            self._scales.pop(name, None)  # default of the previous definition
        self._derived_cache.pop(name, None)
        self.invalidate_params([name], data=True)

    def remove_derived(self, name: str):
        """Remove a derived parameter, its transform and what was computed from it."""
        if self.derived.pop(name, None) is None:
            return
        self._scales.pop(name, None)
        self._derived_cache.pop(name, None)
        self.invalidate_params([name], data=True)

    def add_computed(self, name: str, values: np.ndarray, info: dict | None = None):
        """Add a computed parameter (one value per event; NaN = not computed)."""
        if not self.ensure_loaded():
            raise FCSError(self.load_error or "file not found")
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape != (self.n_events,):
            raise ValueError(f"computed parameter must have {self.n_events} values, got {arr.shape}")
        self.computed[name] = arr
        self.computed_info[name] = dict(info or {})
        self.scales.pop(name, None)
        self.invalidate_params([name], data=True)

    def remove_computed(self, name: str):
        self.computed.pop(name, None)
        self.computed_info.pop(name, None)
        self.scales.pop(name, None)
        self.invalidate_params([name], data=True)

    def estimate_scale(self, name: str, percentile: float = 5.0) -> Scale:
        """A biexponential transform whose width basis fits the negative
        spread of this parameter's (compensated) data (transforms.estimate_biex)."""
        from .transforms import estimate_biex

        ch = self.base_channel(name)
        top = ch.range if ch is not None and ch.range and ch.range > 1 else 262144.0
        return estimate_biex(self.data(name), top, percentile=percentile)

    def auto_adjust_scales(self, names: Iterable[str] | None = None, only_defaults: bool = True) -> list[str]:
        """Give compensated fluorescence parameters a data-driven width basis
        (FlowJo's auto-adjusted biexponential after compensation). Parameters
        whose transform somebody chose are left alone unless only_defaults is
        False; only biexponential transforms are adjusted. Returns the raw keys
        changed."""
        from .transforms import LogicleScale

        comp = self._comp_active()
        if names is None:
            names = [c for c in comp.compensated_names if comp.base_name(c) in self._channel_set()] if comp else []
        changed = []
        for name in names:
            key = self._scale_key(name)
            if only_defaults and self.has_own_scale(key):
                continue
            if not isinstance(self.scale(name), LogicleScale):
                continue
            try:
                new = self.estimate_scale(name)
            except KeyError:
                continue
            if new != self.scale(name):
                self.set_scale(key, new)
                changed.append(key)
        return changed

    # ------------------------------------------------------------ worker copies
    def detached_copy(self) -> "Sample":
        """A copy sharing no mutable state with this sample (gates cloned,
        scales copied, event arrays shared read-only) that a worker thread may
        load and gate; adopt() takes the results back on the GUI thread."""
        c = Sample(self.path, self.id)
        c._detached = True
        c.name = self.name
        c.version = self.version
        c.keywords = dict(self.keywords)
        c.channels = list(self.channels)
        c.comp = self.comp.copy(self.comp.name) if self.comp is not None else None
        c.comp_enabled = self.comp_enabled
        c.comp_cleared = self.comp_cleared
        c.derived = {k: dict(v) for k, v in self.derived.items()}
        c.computed = dict(self.computed)
        c.computed_info = {k: dict(v) for k, v in self.computed_info.items()}
        c.scales = {k: v.copy() for k, v in self.scales.items()}
        c.scales.auto = set(self.scales.auto)
        c._was_unloaded = self._was_unloaded
        root = self.root.clone(deep=True)
        root.name = self.name
        c.root = root
        root._attach(c)
        c.fcs = self.fcs
        return c

    def state_fingerprint(self) -> str:
        """Everything population membership depends on, as a string (used to
        tell whether results computed on a detached copy still apply)."""
        from .gates import POLYGON_SETTINGS

        self._scales.settle()
        explicit = {k: v.to_dict() for k, v in dict.items(self._scales) if k not in self._scales.auto}
        payload = {
            "path": str(self.path),
            "channels": [c.name for c in self.channels],
            "scales": explicit,
            "comp": self.comp.to_dict() if self.comp is not None else None,
            "comp_enabled": bool(self.comp_enabled),
            "derived": self.derived,
            "computed": sorted((k, id(v)) for k, v in self.computed.items()),
            "tree": [(p.path, p.gate.to_dict() if p.gate is not None else None) for p in self.root.walk()],
            "polygon": dict(POLYGON_SETTINGS),
        }
        return json.dumps(payload, sort_keys=True, default=str)

    def adopt(self, copy: "Sample", fingerprint: str | None = None) -> bool:
        """Take over what a detached copy computed in a worker thread: its
        event data when this sample has none, and - when nothing membership
        depends on changed since the copy was made (`fingerprint`, from
        state_fingerprint() at that time) - its cached values and population
        masks. GUI thread only. Returns True when the masks were taken."""
        if copy is None or copy.fcs is None or copy.path != self.path:
            return False
        if self.fcs is None:
            self.attach_fcs(copy.fcs)
        if self.fcs is None or self.fcs.n_events != copy.fcs.n_events:
            return False
        if fingerprint is None or fingerprint != self.state_fingerprint():
            return False
        mine, theirs = self._scales, copy._scales
        for k in theirs.auto:
            a, b = dict.get(mine, k), dict.get(theirs, k)
            if a is not None and b is not None and a != b:
                return False
        for k in theirs.auto:
            if dict.get(mine, k) is None and dict.get(theirs, k) is not None:
                dict.__setitem__(mine, k, dict.get(theirs, k))
                mine.auto.add(k)
        comp = self._comp_active()
        if copy._comp_cache is not None and self._comp_cache is None and comp is not None:
            self._comp_cache = (comp, copy._comp_cache[1])
        for k, v in copy._display_cache.items():
            self._display_cache.setdefault(k, v)
        for k, v in copy._derived_cache.items():
            self._derived_cache.setdefault(k, v)
        n = self.n_events
        for own, other in zip(self.root.walk(), copy.root.walk()):
            if own._mask is None and other._mask is not None and other._mask.shape == (n,):
                own._mask = other._mask
        return True

    # -------------------------------------------------------------- populations
    def export_population_fcs(self, population: Population, path: Path | str, compensated: bool = False):
        """Write the events of a population to a new FCS 3.1 file."""
        from .fcs import write_fcs

        if not self.ensure_loaded():
            raise FCSError(self.load_error or "file not found")
        assert self.fcs is not None
        mask = population.mask()
        if compensated and self._comp_active() is not None:
            names = self.parameter_names()
            names = [n for n in names if n not in self.derived]
            columns = [self.data(n)[mask] for n in names]
            channels = []
            for i, n in enumerate(names, start=1):
                base = self.base_channel(n)
                channels.append(Channel(i, n, base.stain if base else "", base.range if base else 262144.0))
            events = np.column_stack(columns) if columns else np.zeros((0, 0), dtype=np.float32)
        else:
            events = self.fcs.events[mask]
            channels = list(self.channels)
        extra = {"FLOWDESK_POPULATION": population.path or "Ungated", "FLOWDESK_SOURCE": self.path.name}
        write_fcs(path, events, channels, self.keywords, extra)

    def find_population(self, path: str) -> Population | None:
        return self.root.find(path)

    def populations(self) -> list[Population]:
        return [p for p in self.root.walk() if p.parent is not None]

    def __repr__(self):
        return f"<Sample {self.id} {self.name}>"

    def refresh_channels_from_keywords(self):
        """Re-read $PnN/$PnS after the keywords were edited."""
        from .fcs import channels_from_keywords

        channels = channels_from_keywords(self.keywords)
        if self.fcs is not None and len(channels) == len(self.fcs.channels):
            self.fcs.channels = channels
        if channels:
            self.channels = channels
        self.invalidate_display()


def concatenate_samples(samples: list["Sample"], populations: list[Population] | None, path: Path | str, max_events_per_sample: int | None = None) -> int:
    """Write one FCS file holding the events of several samples (or one
    population per sample) plus a SampleID parameter. Returns the event count."""
    from .fcs import Channel, write_fcs

    if not samples:
        raise ValueError("no samples")
    blocks = []
    names = None
    for k, sample in enumerate(samples):
        if not sample.ensure_loaded():
            continue
        assert sample.fcs is not None
        if names is None:
            names = sample.channel_names
            channels = list(sample.channels)
        elif sample.channel_names != names:
            raise ValueError(f"{sample.name} has different parameters than {samples[0].name}")
        pop = populations[k] if populations else sample.root
        events = sample.fcs.events[pop.mask()]
        if max_events_per_sample and events.shape[0] > max_events_per_sample:
            rng = np.random.default_rng(k)
            events = events[np.sort(rng.choice(events.shape[0], size=max_events_per_sample, replace=False))]
        ids = np.full((events.shape[0], 1), float(k + 1), dtype=np.float32)
        blocks.append(np.hstack([events, ids]))
    if not blocks or names is None:
        raise ValueError("no events to write")
    all_events = np.vstack(blocks)
    channels = channels + [Channel(len(channels) + 1, "SampleID", "SampleID", float(len(samples) + 1))]
    extra = {"FLOWDESK_CONCATENATED": ";".join(s.name for s in samples)}
    write_fcs(path, all_events, channels, samples[0].keywords, extra)
    return int(all_events.shape[0])


@dataclass
class Criterion:
    keyword: str
    function: str = "Contains"
    value: str = ""
    connector: str = "And"

    def matches(self, keywords: dict[str, str], sample_name: str) -> bool:
        actual = keywords.get(self.keyword)
        if actual is None and self.keyword in ("$FIL", "FILENAME"):
            actual = sample_name
        actual = (actual or "").strip().lower()
        expect = self.value.strip().lower()
        f = self.function.lower().replace(" ", "")
        if f == "contains":
            return expect in actual
        if f in ("doesnotcontain", "notcontains"):
            return expect not in actual
        if f in ("equals", "is", "equal"):
            return actual == expect
        if f in ("startswith", "beginswith"):
            return actual.startswith(expect)
        if f == "endswith":
            return actual.endswith(expect)
        if f in ("notequals", "isnot"):
            return actual != expect
        return expect in actual

    def to_dict(self):
        return {"keyword": self.keyword, "function": self.function, "value": self.value, "connector": self.connector}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("keyword", "$FIL"), d.get("function", "Contains"), d.get("value", ""), d.get("connector", "And"))


CRITERION_FUNCTIONS = ["Contains", "Does not contain", "Equals", "Not equals", "Starts with", "Ends with"]


class Group:
    """A set of samples sharing a gating template."""

    def __init__(self, name: str, role: str = "custom", criteria: list[Criterion] | None = None, color: str = "#000000", bold: bool = False):
        self.id = new_id()
        self.name = name
        self.role = role
        self.criteria = criteria or []
        self.sample_ids: set[int] = set()
        self.root = Population(name)
        self.color = color
        self.bold = bold
        self.italic = False
        self.live = True
        self.expanded = True

    def matches_criteria(self, sample: Sample) -> bool:
        if not self.criteria:
            return False
        result = self.criteria[0].matches(sample.keywords, sample.name)
        for crit in self.criteria[1:]:
            m = crit.matches(sample.keywords, sample.name)
            result = (result or m) if crit.connector.lower() == "or" else (result and m)
        return result

    def contains(self, sample: Sample) -> bool:
        if self.role == "all":
            return True
        if sample.id in self.sample_ids:
            return True
        return self.live and self.matches_criteria(sample)

    def __repr__(self):
        return f"<Group {self.name}>"


def default_compensation_group() -> Group:
    g = Group(
        "Compensation",
        role="compensation",
        criteria=[Criterion("$FIL", "Contains", "comp", "And"), Criterion("$FIL", "Contains", "unstained", "Or")],
        color="#bc1900",
        bold=True,
    )
    return g


def _notify_loaded(sample: Sample):
    """A sample read its events: let the memory budget of its workspace
    release least-recently-used samples if the cap is exceeded."""
    for budget in list(_BUDGETS):
        if budget.limit_bytes is not None and budget.owns(sample):
            budget.enforce(keep=sample)


class MemoryBudget:
    """Cap on the event data a workspace keeps in memory (FlowJo's
    size-limited data cache).

    When a sample reads its events and the samples' resident data
    (Sample.resident_bytes) exceeds ``limit_mb``, the least-recently used
    samples are released (Sample.unload) until the total fits again; they
    reload transparently on their next data() / mask(). Samples shown in open
    windows must stay: the window layer lists them in ``pinned`` (sample ids)
    or provides ``pin_provider`` (a callable returning samples or ids), e.g.::

        ws.memory_budget.limit_mb = prefs.get("memory_limit_mb", 2048)
        ws.memory_budget.pin_provider = lambda: [w.sample for w in graph_windows]

    ``limit_mb`` None or 0 means no cap (the default of a bare Workspace).
    """

    def __init__(self, workspace: "Workspace", limit_mb: float | None = None):
        self._workspace = weakref.ref(workspace)
        self.limit_mb = limit_mb
        self.pinned: set[int] = set()
        self.pin_provider: Callable[[], Iterable] | None = None
        self.released = 0  # samples released so far
        _BUDGETS.add(self)

    @property
    def limit_bytes(self) -> int | None:
        try:
            mb = float(self.limit_mb or 0)
        except (TypeError, ValueError):
            return None
        return int(mb * 1024 * 1024) if mb > 0 else None

    def samples(self) -> list[Sample]:
        ws = self._workspace()
        return list(ws.samples) if ws is not None else []

    def owns(self, sample: Sample) -> bool:
        return any(s is sample for s in self.samples())

    def loaded_samples(self) -> list[Sample]:
        return [s for s in self.samples() if s.fcs is not None]

    def resident_bytes(self) -> int:
        return sum(s.resident_bytes() for s in self.loaded_samples())

    def pinned_ids(self) -> set[int]:
        ids = set(self.pinned)
        if self.pin_provider is not None:
            try:
                for item in self.pin_provider() or ():
                    if item is None:
                        continue
                    ids.add(item.id if isinstance(item, Sample) else int(item))
            except Exception:  # noqa: BLE001 - a broken provider must not stop loading
                pass
        return ids

    def enforce(self, keep: Sample | None = None) -> list[Sample]:
        """Release least-recently-used, unpinned samples (never `keep`) until
        the resident data fits the cap. Returns the samples released."""
        limit = self.limit_bytes
        if limit is None:
            return []
        loaded = self.loaded_samples()
        sizes = {id(s): s.resident_bytes() for s in loaded}
        total = sum(sizes.values())
        if total <= limit:
            return []
        pinned = self.pinned_ids()
        released = []
        for s in sorted((s for s in loaded if s is not keep and s.id not in pinned), key=lambda s: s.last_used):
            if total <= limit:
                break
            if s.unload():
                total -= sizes[id(s)]
                released.append(s)
        self.released += len(released)
        return released


_SAMPLE_PATH = operator.attrgetter("path")


def _path_key(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(path))


class Workspace:
    def __init__(self):
        self.memory_budget = MemoryBudget(self)
        self._path_seen: dict[int, tuple] = {}  # id(sample) -> (sample, path, resolved path)
        self._path_index: dict[Path, Sample] = {}
        self._index_ids: tuple = ()
        self._index_paths: tuple = ()
        self.samples: list[Sample] = []
        self.groups: list[Group] = [Group("All Samples", role="all", bold=True), default_compensation_group()]
        self.matrices: dict[str, SpilloverMatrix] = {}
        self.layouts: list = []
        self.tables: list = []
        self.path: Path | None = None
        self.settings: dict = {}
        self.modified = False
        # messages about what load_workspace could not restore (computed
        # parameters whose data folder or file is missing, unreadable or
        # stale), one per problem, prefixed with the sample name; never saved
        self.load_warnings: list[str] = []
        self._next_id = 1

    # ------------------------------------------------------------------ samples
    @property
    def name(self) -> str:
        return self.path.stem if self.path else "Untitled"

    def next_sample_id(self) -> int:
        sid = self._next_id
        if any(s.id >= sid for s in self.samples):
            used = {s.id for s in self.samples}
            while sid in used:
                sid += 1
        self._next_id = sid + 1
        return sid

    def _sync_path_index(self):
        """Keep the resolved-path index in step with the sample list (which
        loaders append to directly) and with relocated sample paths; only new
        or moved samples are resolved again. The index holds the samples, so
        their ids stay unique while they are compared."""
        samples = self.samples
        ids = tuple(map(id, samples))
        paths = tuple(map(_SAMPLE_PATH, samples))
        if ids == self._index_ids and paths == self._index_paths:
            return
        seen = self._path_seen
        new_seen: dict[int, tuple] = {}
        index: dict[Path, Sample] = {}
        for s in samples:
            e = seen.get(id(s))
            if e is None or e[0] is not s or e[1] != s.path:
                e = (s, s.path, _path_key(s.path))
            new_seen[id(s)] = e
            index.setdefault(e[2], s)
        self._path_seen = new_seen
        self._path_index = index
        self._index_ids = ids
        self._index_paths = paths

    def sample_by_path(self, path: Path | str) -> Sample | None:
        self._sync_path_index()
        return self._path_index.get(_path_key(Path(path)))

    def add_sample(self, path: Path | str, sample_id: int | None = None, load_header: bool = True) -> Sample:
        key = _path_key(Path(path))
        self._sync_path_index()
        existing = self._path_index.get(key)
        if existing is not None:
            return existing
        sample = Sample(path, sample_id if sample_id is not None else self.next_sample_id())
        if load_header:
            sample.load_header()
        self.samples.append(sample)
        self._path_seen[id(sample)] = (sample, sample.path, key)
        self._path_index.setdefault(key, sample)
        self._index_ids += (id(sample),)
        self._index_paths += (sample.path,)
        for group in self.groups:
            if group.contains(sample):
                self.apply_template(group, sample)
        self.modified = True
        return sample

    def remove_sample(self, sample: Sample):
        if sample in self.samples:
            self.samples.remove(sample)
            for g in self.groups:
                g.sample_ids.discard(sample.id)
            self.modified = True

    def sample_by_id(self, sample_id: int) -> Sample | None:
        for s in self.samples:
            if s.id == sample_id:
                return s
        return None

    # ------------------------------------------------------------------- groups
    def group_by_name(self, name: str) -> Group | None:
        for g in self.groups:
            if g.name == name:
                return g
        return None

    def members(self, group: Group) -> list[Sample]:
        return [s for s in self.samples if group.contains(s)]

    def add_group(self, name: str, samples: list[Sample] | None = None, criteria: list[Criterion] | None = None, color: str = "#000000") -> Group:
        base = name
        k = 2
        while self.group_by_name(name) is not None:
            name = f"{base} {k}"
            k += 1
        g = Group(name, role="custom", criteria=criteria, color=color)
        for s in samples or []:
            g.sample_ids.add(s.id)
        self.groups.append(g)
        self.modified = True
        return g

    def remove_group(self, group: Group):
        if group.role != "all" and group in self.groups:
            self.groups.remove(group)
            self.modified = True

    # ------------------------------------------------------------ group gating
    def apply_template(self, group: Group, sample: Sample):
        """Copy the group's template populations into a sample (merging by name)."""
        self._merge_into(group.root, sample.root)

    def _merge_into(self, template: Population, target: Population):
        for tchild in template.children:
            existing = target.child_by_name(tchild.name)
            if existing is None:
                existing = target.add_child(tchild.clone(deep=False))
            elif tchild.gate is not None:
                existing.gate = tchild.gate.clone()
                existing.invalidate()
            for st in tchild.stats:
                if not any(s.statistic == st.statistic and s.param == st.param and s.percentile == st.percentile for s in existing.stats):
                    existing.stats.append(StatNode.from_dict(st.to_dict()))
            self._merge_into(tchild, existing)

    def apply_population_to_group(self, group: Group, parent_path: str, population: Population) -> int:
        """Add (or replace) a population under parent_path for every member sample
        and in the group template. Returns the number of samples changed."""
        tparent = group.root.find(parent_path)
        if tparent is None:
            tparent = self._ensure_path(group.root, parent_path)
        existing = tparent.child_by_name(population.name)
        if existing is not None:
            tparent.remove_child(existing)
        tparent.add_child(population.clone(deep=True))
        changed = 0
        for sample in self.members(group):
            parent = sample.root.find(parent_path)
            if parent is None:
                continue
            old = parent.child_by_name(population.name)
            index = None
            if old is not None:
                index = parent.children.index(old)
                parent.remove_child(old)
            clone = population.clone(deep=True)
            if old is not None:
                # keep sub-populations the sample already had
                for c in list(old.children):
                    if clone.child_by_name(c.name) is None:
                        old.remove_child(c)
                        clone.add_child(c)
            parent.add_child(clone, index)
            clone.invalidate()
            changed += 1
        self.modified = True
        return changed

    def remove_population_from_group(self, group: Group, path: str) -> int:
        node = group.root.find(path)
        if node is not None and node.parent is not None:
            node.parent.remove_child(node)
        changed = 0
        for sample in self.members(group):
            pop = sample.root.find(path)
            if pop is not None and pop.parent is not None:
                pop.parent.remove_child(pop)
                changed += 1
        self.modified = True
        return changed

    def rename_population_in_group(self, group: Group, path: str, new_name: str) -> int:
        node = group.root.find(path)
        if node is not None:
            node.name = new_name
        changed = 0
        for sample in self.members(group):
            pop = sample.root.find(path)
            if pop is not None:
                pop.name = new_name
                changed += 1
        self.modified = True
        return changed

    @staticmethod
    def _ensure_path(root: Population, path: str) -> Population:
        node = root
        for part in [p for p in path.split("/") if p]:
            child = node.child_by_name(part)
            if child is None:
                child = node.add_child(Population(part))
            node = child
        return node

    def population_paths(self, group: Group | None = None) -> list[str]:
        """Union of population paths across the group's members, in tree order."""
        seen: dict[str, None] = {}
        samples = self.members(group) if group is not None else self.samples
        for s in samples:
            for p in s.root.walk(include_self=False):
                seen.setdefault(p.path, None)
        return list(seen.keys())

    def all_parameter_names(self, group: Group | None = None) -> list[str]:
        seen: dict[str, None] = {}
        for s in self.members(group) if group is not None else self.samples:
            for n in s.parameter_names():
                seen.setdefault(n, None)
        return list(seen.keys())
