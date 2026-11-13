"""Population export and concatenation (FlowJo's Export / Concatenate).

Qt-free helpers behind the Export dialog (flowdesk/ui/export_tools.py):

* `select_parameters` - All / Compensated only / Uncompensated / Selected
  parameters, with or without derived and computed parameters;
* `export_population` - one population as FCS 3.1 or CSV: chosen columns,
  linear ("channel") or transformed display-space ("scale") values, random
  subsample by count or percent, keywords to add; source keywords are kept
  and the compensation keywords are made consistent with the written data;
* `batch_export` - one population path for every sample of a group, files
  named from a pattern ({sample}, {population}, {group}, {index}, keywords);
* `concatenate` - several samples / populations in one file, parameters
  matched by $PnN (common or all parameters, any order), derived / computed /
  compensated parameters on request, keyword values as coded parameters
  (code table in FLOWDESK_KW_<keyword>) and a SampleID parameter;
* `split_datasets` - every data set of a multi-data-set file ($NEXTDATA,
  .lmd) as a separate FCS file.
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fcs import SPILLOVER_KEYS, Channel, FCSError, FCSFile, format_spillover, write_fcs
from .model import COMP_PREFIX, Group, Population, Sample, Workspace

# display-space values ("scale values") are written as fraction x SCALE_RANGE
SCALE_RANGE = 1024.0
PARAMETER_MODES = ("all", "compensated", "uncompensated", "selected")
MATCH_MODES = ("intersection", "union", "strict")
DEFAULT_PATTERN = "{sample}_{population}"
_MATRIX_KEYS = {k.upper() for k in SPILLOVER_KEYS}
_BAD_CHARS = '/\\:*?"<>|\x00'


# ------------------------------------------------------------------ parameters
def parameter_kind(sample: Sample, name: str) -> str:
    """'compensated' (computed from a matrix), 'raw' (a channel of the file),
    'derived' or 'computed'."""
    if name in sample.computed:
        return "computed"
    if name in sample.derived:
        return "derived"
    if name in sample.channel_names:
        return "raw"
    return "compensated" if sample.is_compensated_param(name) else "raw"


def _native_comp(sample: Sample, name: str) -> bool:
    """A channel of the file named "Comp-X" next to a channel X (a
    pre-compensated export)."""
    return name.startswith(COMP_PREFIX) and name in sample.channel_names and name[len(COMP_PREFIX):] in sample.channel_names


def _compensated_bases(sample: Sample, names: list[str]) -> set[str]:
    """Raw channels that have a compensated counterpart among `names`."""
    out = set()
    comp = sample._comp_active()
    for n in names:
        if comp is not None and sample.is_compensated_param(n):
            base = comp.base_name(n)
            if base:
                out.add(base)
        elif _native_comp(sample, n):
            out.add(n[len(COMP_PREFIX):])
    return out


def select_parameters(
    sample: Sample,
    mode: str = "all",
    chosen: list[str] | None = None,
    include_derived: bool = True,
    include_computed: bool = True,
) -> list[str]:
    """The parameters an export writes.

    all: every parameter (compensated, raw, derived, computed);
    compensated: the compensated version of each compensated channel instead
    of the raw one, the other raw channels (scatter, time) as they are (in a
    pre-compensated file its "Comp-X" channels replace X);
    uncompensated: the channels of the file; selected: `chosen`, in that order.
    Derived / computed parameters follow the include flags except in
    'selected' mode."""
    if mode not in PARAMETER_MODES:
        raise ValueError(f"unknown parameter mode {mode!r}")
    sample.ensure_loaded()
    available = sample.parameter_names()
    if mode == "selected":
        have = set(available)
        return [n for n in dict.fromkeys(chosen or []) if n in have]
    kinds = {n: parameter_kind(sample, n) for n in available}
    extra = [n for n in available if (kinds[n] == "derived" and include_derived) or (kinds[n] == "computed" and include_computed)]
    measured = [n for n in available if kinds[n] in ("raw", "compensated")]
    if mode == "all":
        names = measured
    elif mode == "uncompensated":
        names = [n for n in measured if kinds[n] == "raw"]
    else:
        comp = [n for n in measured if kinds[n] == "compensated" or _native_comp(sample, n)]
        replaced = _compensated_bases(sample, comp)
        names = [n for n in measured if n not in replaced]
    return list(dict.fromkeys(names + extra))


def _finite_range(values: np.ndarray) -> tuple[float, float] | None:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    return float(v.min()), float(v.max())


def _channel_for(sample: Sample, name: str, index: int, values: np.ndarray, scaled: bool, scale_range: float) -> Channel:
    """FCS channel description of one exported parameter."""
    if scaled:
        base = sample.base_channel(name) if parameter_kind(sample, name) in ("raw", "compensated") else None
        return Channel(index, name, base.stain if base else "", float(scale_range), display=f"Linear,0,{scale_range:g}")
    kind = parameter_kind(sample, name)
    if kind in ("raw", "compensated"):
        base = sample.base_channel(name)
        if base is None:
            return Channel(index, name, "", 262144.0)
        gain = base.gain if kind == "raw" else 1.0
        return Channel(index, name, base.stain, base.range, base.bits, 0.0, 0.0, gain, base.display)
    info = sample.computed_info.get(name, {}) if kind == "computed" else sample.derived.get(name, {})
    lohi = _finite_range(values)
    top = None
    try:
        if "max" in info:
            top = float(info["max"])
    except (TypeError, ValueError):
        top = None
    if top is None or not math.isfinite(top):
        top = lohi[1] if lohi else 1.0
    lo = None
    try:
        if "min" in info:
            lo = float(info["min"])
    except (TypeError, ValueError):
        lo = None
    if lo is None:
        lo = min(0.0, lohi[0]) if lohi else 0.0
    rng = max(1.0, float(math.floor(top)) + 1.0)  # values 0 .. $PnR - 1, as for integer data
    return Channel(index, name, "", rng, display=f"Linear,{lo:g},{max(top, lo + 1e-9):g}")


def _values(sample: Sample, name: str, index: np.ndarray, scaled: bool, scale_range: float) -> np.ndarray:
    if scaled:
        return (sample.display(name)[index].astype(np.float64) * scale_range).astype(np.float32)
    return np.asarray(sample.data(name)[index], dtype=np.float32)


# ------------------------------------------------------------------ subsampling
def subsample_index(mask: np.ndarray, max_events: int | None = None, percent: float | None = None, seed: int = 0) -> np.ndarray:
    """Indices of the events to write: the events of `mask`, optionally a
    random `percent` of them and/or at most `max_events` (seeded, in the
    original order)."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    rng = np.random.default_rng(seed)
    if percent is not None and 0 < float(percent) < 100 and idx.size:
        k = int(round(idx.size * float(percent) / 100.0))
        idx = np.sort(rng.choice(idx, size=k, replace=False))
    if max_events and int(max_events) > 0 and idx.size > int(max_events):
        idx = np.sort(rng.choice(idx, size=int(max_events), replace=False))
    return idx


# ------------------------------------------------------------- compensation keys
def _matrix_free(keywords: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in keywords.items() if k.upper() not in _MATRIX_KEYS}


def _compensation_keywords(sample: Sample, names: list[str], scaled: bool) -> tuple[dict[str, str], dict[str, str]]:
    """(source keywords to pass on, keywords to add) so that the file's
    compensation keywords match its columns: written Comp- columns mean the
    data are compensated (FLOWDESK_COMPENSATED=TRUE, no matrix to apply
    again); raw columns carry the matrix in use (restricted to the written
    channels); transformed values carry none."""
    keywords = dict(sample.keywords)
    extra: dict[str, str] = {}
    comp_cols = [n for n in names if sample.is_compensated_param(n)]
    comp = sample.comp
    if comp_cols or scaled:
        keywords = _matrix_free(keywords)
        if comp_cols:
            extra["FLOWDESK_COMPENSATED"] = "TRUE"
            active = sample._comp_active()
            if active is not None:
                extra["FLOWDESK_COMPENSATION"] = active.name
                extra["FLOWDESK_SPILLOVER"] = format_spillover(active.params, active.matrix)
        return keywords, extra
    if comp is not None:
        keywords = _matrix_free(keywords)
        keep = [i for i, p in enumerate(comp.params) if p in names]
        if keep:
            sub = comp.matrix[np.ix_(keep, keep)]
            extra["$SPILLOVER"] = format_spillover([comp.params[i] for i in keep], sub)
    elif sample.comp_cleared:
        keywords = _matrix_free(keywords)
    return keywords, extra


# ---------------------------------------------------------------------- writing
def write_csv(path: Path | str, names: list[str], columns: list[np.ndarray], header: bool = True, labels: list[str] | None = None):
    """Comma-separated values, one row per event (6 significant digits)."""
    n = int(columns[0].shape[0]) if columns else 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if header:
            csv.writer(fh).writerow(labels or names)
        if not columns:
            return
        step = 50000
        for a in range(0, n, step):
            block = np.column_stack([c[a:a + step] for c in columns])
            np.savetxt(fh, block, fmt="%.6g", delimiter=",")


def _fmt_of(path: Path | str, fmt: str | None) -> str:
    if fmt:
        fmt = fmt.lower()
        if fmt not in ("fcs", "csv"):
            raise ValueError(f"unknown export format {fmt!r}")
        return fmt
    return "csv" if str(path).lower().endswith((".csv", ".txt")) else "fcs"


def export_population(
    sample: Sample,
    population: Population | None,
    path: Path | str,
    parameters: list[str] | None = None,
    *,
    mode: str = "all",
    include_computed: bool = True,
    include_derived: bool = True,
    scaled: bool = False,
    scale_range: float = SCALE_RANGE,
    max_events: int | None = None,
    percent: float | None = None,
    seed: int = 0,
    extra_keywords: dict[str, str] | None = None,
    fmt: str | None = None,
    header: bool = True,
    header_labels: bool = False,
    skip_missing: bool = False,
) -> int:
    """Write the events of `population` (the whole sample when None) to an
    FCS 3.1 (default) or CSV file (`fmt`, or a .csv / .txt suffix). With
    `parameters` exactly those columns are written; otherwise `mode` and the
    include flags choose them (see select_parameters). Returns the number of
    events written. A parameter the sample lacks raises KeyError unless
    `skip_missing` (batch export over samples with different panels)."""
    if not sample.ensure_loaded():
        raise FCSError(sample.load_error or f"file not found: {sample.path}")
    pop = population if population is not None else sample.root
    if parameters is not None:
        names = select_parameters(sample, "selected", parameters)
        missing = [p for p in parameters if p not in names]
        if missing and not skip_missing:
            raise KeyError(f"{sample.name} has no parameter(s) {', '.join(missing)}")
    else:
        names = select_parameters(sample, mode, None, include_derived, include_computed)
    if not names:
        raise ValueError("no parameters to export")
    index = subsample_index(pop.mask(), max_events, percent, seed)
    columns = [_values(sample, n, index, scaled, scale_range) for n in names]
    if _fmt_of(path, fmt) == "csv":
        labels = [sample.param_label(n) for n in names] if header_labels else None
        write_csv(path, names, columns, header, labels)
        return int(index.size)
    channels = [_channel_for(sample, n, i, col, scaled, scale_range) for i, (n, col) in enumerate(zip(names, columns), start=1)]
    keywords, extra = _compensation_keywords(sample, names, scaled)
    extra.update({"FLOWDESK_POPULATION": pop.path or "Ungated", "FLOWDESK_SOURCE": sample.path.name})
    if scaled:
        extra["FLOWDESK_VALUES"] = f"scale 0-{scale_range:g}"
    if index.size != int(np.count_nonzero(pop.mask())):
        extra["FLOWDESK_SUBSAMPLE"] = f"{index.size} of {int(np.count_nonzero(pop.mask()))} (seed {seed})"
    extra.update({str(k): str(v) for k, v in (extra_keywords or {}).items() if str(k).strip()})
    events = np.column_stack(columns) if index.size else np.zeros((0, len(names)), dtype=np.float32)
    write_fcs(path, events, channels, keywords, extra)
    return int(index.size)


# ------------------------------------------------------------------ file naming
def _keyword(sample: Sample, key: str) -> str | None:
    kw = sample.keywords
    if key in kw:
        return kw[key]
    upper = key.upper()
    for k, v in kw.items():
        if k.upper() == upper or k.upper() == "$" + upper:
            return v
    return None


def safe_filename(name: str) -> str:
    out = "".join("_" if ch in _BAD_CHARS or ord(ch) < 32 else ch for ch in name).strip().strip(".")
    return out or "export"


def format_name(pattern: str, sample: Sample, population: Population | None = None, group: Group | None = None, index: int | None = None) -> str:
    """File name (without suffix) from a pattern: {sample} (file name without
    suffix), {name} (sample name), {population} (population name, "Ungated"
    for the whole sample), {path} (population path), {group}, {index}
    (1-based position in a batch) and any keyword ({$WELLID}, {TUBE NAME})."""
    pop = population

    def repl(m: re.Match) -> str:
        token = m.group(1).strip()
        low = token.lower()
        if low in ("sample", "stem", "file"):
            return sample.path.stem
        if low == "name":
            return sample.name
        if low in ("population", "pop"):
            return pop.name if pop is not None and pop.parent is not None else "Ungated"
        if low == "path":
            return (pop.path.strip("/").replace("/", "_") if pop is not None and pop.parent is not None else "Ungated")
        if low == "group":
            return group.name if group is not None else ""
        if low == "index":
            return f"{index:03d}" if index is not None else ""
        value = _keyword(sample, token)
        return value if value is not None else ""

    return safe_filename(re.sub(r"\{([^{}]+)\}", repl, pattern or DEFAULT_PATTERN))


def unique_path(path: Path, taken: set | None = None) -> Path:
    """`path`, or "name (2).ext", "name (3).ext"... when it exists or was used."""
    taken = taken if taken is not None else set()
    cand = path
    k = 2
    while cand.exists() or str(cand).lower() in taken:
        cand = path.with_name(f"{path.stem} ({k}){path.suffix}")
        k += 1
    taken.add(str(cand).lower())
    return cand


def find_population(sample: Sample, path: str | None) -> Population | None:
    if not path or path in ("/", ""):
        return sample.root
    return sample.root.find(path)


@dataclass
class BatchResult:
    written: list[tuple[Sample, Path, int]] = field(default_factory=list)
    skipped: list[tuple[Sample, str]] = field(default_factory=list)  # (sample, reason)

    @property
    def paths(self) -> list[Path]:
        return [p for _, p, _ in self.written]


def export_rows(
    rows: list[tuple[Sample, Population]],
    folder: Path | str,
    pattern: str = DEFAULT_PATTERN,
    fmt: str = "fcs",
    group: Group | None = None,
    overwrite: bool = False,
    **options,
) -> BatchResult:
    """Export each (sample, population) row to `folder`, files named from
    `pattern`; options as for export_population."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fmt = _fmt_of("", fmt)
    result = BatchResult()
    taken: set = set()
    for k, (sample, pop) in enumerate(rows, start=1):
        target = folder / f"{format_name(pattern, sample, pop, group, k)}.{fmt}"
        target = target if overwrite and str(target).lower() not in taken else unique_path(target, taken)
        taken.add(str(target).lower())
        try:
            n = export_population(sample, pop, target, fmt=fmt, **options)
        except (OSError, FCSError, KeyError, ValueError) as exc:
            result.skipped.append((sample, str(exc).strip("'\"")))
            continue
        result.written.append((sample, target, n))
    return result


def group_rows(workspace: Workspace, group: Group, population_paths: list[str]) -> tuple[list[tuple[Sample, Population]], list[tuple[Sample, str]]]:
    """(sample, population) for every member of `group` and every path, and
    the (sample, reason) pairs of members that lack a population."""
    rows, missing = [], []
    for sample in workspace.members(group):
        for path in population_paths or [""]:
            pop = find_population(sample, path)
            if pop is None:
                missing.append((sample, f"no population {path}"))
            else:
                rows.append((sample, pop))
    return rows, missing


def batch_export(
    workspace: Workspace,
    group: Group,
    population_path: str | list[str],
    folder: Path | str,
    pattern: str = DEFAULT_PATTERN,
    fmt: str = "fcs",
    **options,
) -> BatchResult:
    """FlowJo's batch export: the population(s) at `population_path` for every
    sample of `group`, one file per sample and population."""
    paths = [population_path] if isinstance(population_path, str) else list(population_path)
    rows, missing = group_rows(workspace, group, paths)
    result = export_rows(rows, folder, pattern, fmt, group, **options)
    result.skipped[:0] = missing
    return result


# ------------------------------------------------------------------ concatenate
def _is_number(text: str) -> bool:
    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError):
        return False


def _natural_key(text: str):
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", text) if t]


def keyword_codes(values: list[str | None]) -> tuple[list[float], str]:
    """Per-sample codes for keyword values and the code table: numbers are
    used as they are ("numeric"), text values get 1..k in natural sort order
    ("1=PMA;2=unstim"; a missing value is 0)."""
    present = [v for v in values if v is not None and str(v).strip() != ""]
    if present and all(_is_number(v) for v in present):
        return [float(v) if v is not None and str(v).strip() != "" else math.nan for v in values], "numeric"
    uniq = sorted(set(present), key=_natural_key)
    codes = {v: float(i) for i, v in enumerate(uniq, start=1)}
    table = ";".join(f"{int(c)}={v}" for v, c in codes.items())
    if len(present) < len(values):
        table = "0=;" + table if table else "0="
    return [codes.get(v, 0.0) if v is not None else 0.0 for v in values], table


def parse_code_table(text: str) -> dict[float, str]:
    """Code -> value of a FLOWDESK_KW_<keyword> table ({} for "numeric")."""
    out: dict[float, str] = {}
    if not text or text.strip().lower() == "numeric":
        return out
    for part in text.split(";"):
        code, sep, value = part.partition("=")
        if sep and _is_number(code):
            out[float(code)] = value
    return out


def _unique_name(name: str, taken: set[str]) -> str:
    cand = name
    k = 2
    while cand in taken:
        cand = f"{name}_{k}"
        k += 1
    taken.add(cand)
    return cand


def _concat_candidates(sample: Sample, parameters, mode, include_compensated, include_derived, include_computed) -> list[str]:
    if parameters is not None:
        have = set(sample.parameter_names())
        return [p for p in dict.fromkeys(parameters) if p in have]
    if mode is None:
        mode = "all" if include_compensated else "uncompensated"
    if mode == "all" and not include_compensated:
        mode = "uncompensated"
    return select_parameters(sample, mode, None, include_derived, include_computed)


def common_keywords(samples: list[Sample]) -> dict[str, str]:
    """The keywords that have the same value in every sample."""
    if not samples:
        return {}
    first = samples[0].keywords
    return {k: v for k, v in first.items() if all(s.keywords.get(k) == v for s in samples[1:])}


def concatenate(
    samples: list[Sample],
    populations: list[Population | None] | None,
    path: Path | str,
    *,
    parameters: list[str] | None = None,
    mode: str | None = None,
    match: str = "intersection",
    include_compensated: bool = False,
    include_derived: bool = False,
    include_computed: bool = False,
    keywords: list[str] | None = None,
    max_events_per_sample: int | None = None,
    percent: float | None = None,
    seed: int = 0,
    sample_id: bool = True,
    scaled: bool = False,
    scale_range: float = SCALE_RANGE,
    extra_keywords: dict[str, str] | None = None,
    fmt: str | None = None,
) -> int:
    """Write the events of several samples (or one population per sample) to
    one file. Parameters are matched by name: 'intersection' keeps the ones
    every sample has, 'union' all of them (NaN where a sample lacks one),
    'strict' requires the same set (any order). `keywords` become coded
    parameters (see keyword_codes) with their table in FLOWDESK_KW_<keyword>.
    Returns the number of events written."""
    if match not in MATCH_MODES:
        raise ValueError(f"unknown match mode {match!r}")
    if not samples:
        raise ValueError("no samples")
    loaded: list[tuple[int, Sample]] = []
    for k, s in enumerate(samples):
        if s.ensure_loaded():
            loaded.append((k, s))
    if not loaded:
        raise ValueError("none of the samples could be read")
    per_sample = {k: _concat_candidates(s, parameters, mode, include_compensated, include_derived, include_computed) for k, s in loaded}
    first = per_sample[loaded[0][0]]
    if match == "strict":
        for k, s in loaded[1:]:
            if set(per_sample[k]) != set(first):
                raise ValueError(f"{s.name} has different parameters than {loaded[0][1].name}")
        names = list(first)
    elif match == "intersection":
        common = set(first)
        for k, _ in loaded[1:]:
            common &= set(per_sample[k])
        names = [n for n in first if n in common]
    else:
        names = list(dict.fromkeys(n for k, _ in loaded for n in per_sample[k]))
    if parameters is not None:
        order = {p: i for i, p in enumerate(parameters)}
        names.sort(key=lambda n: order.get(n, len(order)))
    if not names:
        raise ValueError("the samples have no parameters in common")

    blocks: list[np.ndarray] = []
    owners: dict[str, Sample] = {}
    sizes: list[int] = []
    for k, s in loaded:
        pop = (populations[k] if populations and k < len(populations) else None) or s.root
        index = subsample_index(pop.mask(), max_events_per_sample, percent, seed + k)
        cols = []
        have = set(per_sample[k])
        for n in names:
            if n in have:
                col = _values(s, n, index, scaled, scale_range)
                owners.setdefault(n, s)
            else:
                col = np.full(index.size, np.nan, dtype=np.float32)
            cols.append(col)
        blocks.append(np.column_stack(cols) if cols else np.zeros((index.size, 0), dtype=np.float32))
        sizes.append(int(index.size))
    events = np.vstack(blocks)
    channels: list[Channel] = []
    sources: list[dict | None] = []
    for i, n in enumerate(names, start=1):
        s = owners.get(n, loaded[0][1])
        ch = _channel_for(s, n, i, events[:, i - 1], scaled, scale_range)
        channels.append(ch)
        sources.append(s.keywords)
    extra: dict[str, str] = {}
    taken = set(names)
    kw_cols = []
    for kw in keywords or []:
        values = [_keyword(s, kw) for _, s in loaded]
        codes, table = keyword_codes(values)
        col = np.concatenate([np.full(size, code, dtype=np.float32) for size, code in zip(sizes, codes)]) if sizes else np.zeros(0, dtype=np.float32)
        label = kw.lstrip("$").strip() or kw
        pname = _unique_name(label, taken)
        finite = [c for c in codes if math.isfinite(c)]
        top = max(finite) if finite else 1.0
        channels.append(Channel(len(channels) + 1, pname, kw, max(1.0, float(math.ceil(top)) + 1.0), display=f"Linear,0,{max(top, 0.0) + 1:g}"))
        sources.append({})
        kw_cols.append(col)
        extra[f"FLOWDESK_KW_{label}"] = table
    if sample_id:
        ids = np.concatenate([np.full(size, float(k + 1), dtype=np.float32) for (k, _), size in zip(loaded, sizes)]) if sizes else np.zeros(0, dtype=np.float32)
        kw_cols.append(ids)
        channels.append(Channel(len(channels) + 1, _unique_name("SampleID", taken), "SampleID", float(len(samples) + 1), display=f"Linear,0,{len(samples) + 1}"))
        sources.append({})
    if kw_cols:
        events = np.hstack([events, np.column_stack(kw_cols)])
    total = int(events.shape[0])
    if _fmt_of(path, fmt) == "csv":
        write_csv(path, [c.name for c in channels], [events[:, j] for j in range(events.shape[1])])
        return total
    base = common_keywords([s for _, s in loaded])
    written = set(names)
    comp_cols = any(s.is_compensated_param(n) for k, s in loaded for n in per_sample[k] if n in written)
    base = {k: v for k, v in base.items() if k.upper() not in _MATRIX_KEYS}
    if comp_cols:
        extra["FLOWDESK_COMPENSATED"] = "TRUE"
    elif not scaled:
        mats = [s.comp for _, s in loaded]
        if all(m is not None for m in mats) and all(m == mats[0] for m in mats[1:]):
            comp = mats[0]
            keep = [i for i, p in enumerate(comp.params) if p in names]
            if keep:
                extra["$SPILLOVER"] = format_spillover([comp.params[i] for i in keep], comp.matrix[np.ix_(keep, keep)])
    if scaled:
        extra["FLOWDESK_VALUES"] = f"scale 0-{scale_range:g}"
    extra["FLOWDESK_CONCATENATED"] = ";".join(s.name for s in samples)
    extra.update({str(k): str(v) for k, v in (extra_keywords or {}).items() if str(k).strip()})
    # per-parameter keywords ($PnV, $PnF, ...) come from the sample each
    # channel was taken from; the file keywords are the ones all samples share
    write_fcs(path, events, channels, base, extra, channel_sources=sources)
    return total


# ------------------------------------------------------------------- data sets
def split_datasets(path: Path | str, folder: Path | str | None = None) -> list[Path]:
    """Write every data set of a multi-data-set file ($NEXTDATA chain, .lmd)
    as its own FCS 3.1 file "<stem>_dataset<k>.fcs" (in `folder`, default
    next to the file). Returns the written paths (empty for a single data set)."""
    path = Path(path)
    offsets = FCSFile.list_datasets(path)
    if len(offsets) < 2:
        return []
    folder = Path(folder) if folder is not None else path.parent
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    taken: set = set()
    for k in range(len(offsets)):
        f = FCSFile.read(path, dataset=k)
        target = unique_path(folder / f"{path.stem}_dataset{k + 1}.fcs", taken)
        write_fcs(target, f.events, f.channels, f.keywords, {"FLOWDESK_SOURCE": path.name, "FLOWDESK_DATASET": f"{k + 1} of {len(offsets)} ({f.version})"})
        out.append(target)
    return out
