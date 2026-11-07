"""Workspace model: samples, population hierarchies and groups."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compensation import ACQUISITION_DEFINED, SpilloverMatrix
from .fcs import Channel, FCSError, FCSFile
from .gates import Gate, new_id
from .transforms import Scale, default_scale_for_channel

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
                except KeyError:
                    own = np.zeros(sample.n_events, dtype=bool)
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
        self.root = Population(self.name)
        self.root.sample = self
        self.scales: dict[str, Scale] = {}
        self.comp: SpilloverMatrix | None = None
        self.comp_enabled = True
        self.comp_cleared = False  # the user removed the acquisition matrix on purpose
        self.derived: dict[str, dict] = {}
        # parameters computed by tools (tSNE/UMAP axes, cluster ids, ...): one
        # float32 value per event, NaN where the tool did not run
        self.computed: dict[str, np.ndarray] = {}
        self.computed_info: dict[str, dict] = {}
        self._comp_cache: tuple[SpilloverMatrix, np.ndarray] | None = None
        self._display_cache: dict[str, np.ndarray] = {}
        self._derived_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ loading
    def load_header(self) -> bool:
        try:
            self.version, self.keywords = FCSFile.read_keywords(self.path)
        except (OSError, FCSError) as exc:
            self.load_error = str(exc)
            return False
        from .fcs import channels_from_keywords, parse_spillover

        self.channels = channels_from_keywords(self.keywords)
        if self.comp is None and not self.comp_cleared:
            spill = parse_spillover(self.keywords)
            if spill is not None:
                self.comp = SpilloverMatrix(ACQUISITION_DEFINED, spill[0], spill[1])
        self.load_error = None
        return True

    def ensure_loaded(self) -> bool:
        if self.fcs is not None:
            return True
        if self.load_error and not self.path.exists():
            return False
        try:
            self.fcs = FCSFile.read(self.path)
        except (OSError, FCSError) as exc:
            self.load_error = str(exc)
            return False
        self.version = self.fcs.version
        if not self.keywords:
            self.keywords = dict(self.fcs.keywords)
        self.channels = self.fcs.channels
        if self.comp is None and not self.comp_cleared:
            self.comp = SpilloverMatrix.from_fcs(self.fcs)
        self.load_error = None
        return True

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
            return self.computed[name]
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
        return self.fcs.column(name)

    def _derived_values(self, name: str) -> np.ndarray:
        if name in self._derived_cache:
            return self._derived_cache[name]
        spec = self.derived[name]
        a = self.data(spec["a"]).astype(np.float64)
        op = spec.get("op", "ratio")
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
    def default_scale(self, name: str) -> Scale:
        ch = self.base_channel(name)
        if name in self.computed:
            from .transforms import LinearScale

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
            from .transforms import LinearScale, LogicleScale

            if spec.get("op") in ("ratio", "log10"):
                return LinearScale(spec.get("min", 0.0), spec.get("max", 10.0))
            return LogicleScale(T=spec.get("max", 262144.0))
        if ch is None:
            return default_scale_for_channel(name)
        top = ch.range if ch.range and ch.range > 1 else 262144.0
        return default_scale_for_channel(ch.name, ch.display, top, ch.is_log_amplified)

    def scale(self, name: str) -> Scale:
        s = self.scales.get(name)
        if s is not None:
            return s
        comp = self.comp
        if comp is not None:
            base = comp.base_name(name)
            if base is not None and base != name:
                # a compensated parameter follows its base channel's scale
                # unless it was given one of its own
                return self.scale(base)
        s = self.default_scale(name)
        self.scales[name] = s
        return s

    def set_scale(self, name: str, scale: Scale):
        self.scales[name] = scale
        self.invalidate_display()

    def invalidate_display(self):
        self._display_cache.clear()
        self.root.invalidate()

    def set_compensation(self, matrix: SpilloverMatrix | None, enabled: bool = True):
        self.comp = matrix
        self.comp_enabled = enabled
        self.comp_cleared = matrix is None
        self._comp_cache = None
        self._derived_cache.clear()
        self.invalidate_display()

    def add_derived(self, name: str, spec: dict):
        self.derived[name] = dict(spec)
        self._derived_cache.pop(name, None)
        self.invalidate_display()

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
        self._display_cache.pop(name, None)
        self.root.invalidate()

    def remove_computed(self, name: str):
        self.computed.pop(name, None)
        self.computed_info.pop(name, None)
        self.scales.pop(name, None)
        self._display_cache.pop(name, None)
        self.root.invalidate()

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


class Workspace:
    def __init__(self):
        self.samples: list[Sample] = []
        self.groups: list[Group] = [Group("All Samples", role="all", bold=True), default_compensation_group()]
        self.matrices: dict[str, SpilloverMatrix] = {}
        self.layouts: list = []
        self.tables: list = []
        self.path: Path | None = None
        self.settings: dict = {}
        self.modified = False
        self._next_id = 1

    # ------------------------------------------------------------------ samples
    @property
    def name(self) -> str:
        return self.path.stem if self.path else "Untitled"

    def next_sample_id(self) -> int:
        used = {s.id for s in self.samples}
        while self._next_id in used:
            self._next_id += 1
        sid = self._next_id
        self._next_id += 1
        return sid

    def sample_by_path(self, path: Path | str) -> Sample | None:
        p = Path(path).resolve()
        for s in self.samples:
            try:
                if s.path.resolve() == p:
                    return s
            except OSError:
                continue
        return None

    def add_sample(self, path: Path | str, sample_id: int | None = None, load_header: bool = True) -> Sample:
        existing = self.sample_by_path(path)
        if existing is not None:
            return existing
        sample = Sample(path, sample_id if sample_id is not None else self.next_sample_id())
        if load_header:
            sample.load_header()
        self.samples.append(sample)
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
