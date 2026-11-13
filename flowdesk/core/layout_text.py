"""Dynamic text for layout pages: keyword, field and statistic tokens.

A layout text box (and a plot's annotation list) may contain tokens in curly
braces that are replaced when the page is painted, so the same template shows
the right values on every page of a batch:

    {$FIL}  {$DATE}  {TUBE NAME}      FCS / user keyword of the sample
    {Sample}  {Group}  {Population}  {Path}  {Layout}  {Page}  {Pages}
    {Date}  {Time}  {File}            built-in fields
    {Count}  {Freq. of Parent}        a statistic of the bound population
    {Median FSC-A}  {CV : B1-A}       a statistic with its parameter
    {Percentile 95 FSC-A}
    {/Cells/Bright | Freq. of Parent} a statistic of another population
                                      (path, or a population name)

``{{`` and ``}}`` give literal braces. A token that cannot be resolved (no
sample bound yet, unknown keyword) is left as it is, so the template stays
visible. Other packages add their own tokens with register_text_expander().
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Callable

from .stats import NEEDS_PARAM, STATISTICS, compute_statistic, format_value, statistic_label

TOKEN_RE = re.compile(r"\{\{|\}\}|\{([^{}\n]+)\}")

BUILTIN_FIELDS = ["Sample", "Group", "Population", "Path", "Layout", "Page", "Pages", "Date", "Time", "File"]

# fn(token, context) -> str | None ; tried before the built-in resolution
_EXPANDERS: list[Callable[[str, "TextContext"], str | None]] = []


def register_text_expander(fn: Callable[[str, "TextContext"], str | None]) -> None:
    """Add a resolver for custom tokens. It receives the token text (without
    braces, stripped) and the TextContext and returns the replacement, or
    None when the token is not its own."""
    if fn not in _EXPANDERS:
        _EXPANDERS.append(fn)


def unregister_text_expander(fn) -> None:
    if fn in _EXPANDERS:
        _EXPANDERS.remove(fn)


@dataclass
class TextContext:
    sample: object | None = None  # flowdesk.core.model.Sample
    population: object | None = None  # bound population (default: the sample root)
    workspace: object | None = None
    layout_name: str = ""
    group_name: str = ""
    page: int = 1
    pages: int = 1
    decimals: int = 1
    extra: dict = field(default_factory=dict)  # name -> value, highest priority
    layout: object | None = None  # the Layout being painted (for expanders)

    @property
    def base_population(self):
        if self.population is not None:
            return self.population
        return getattr(self.sample, "root", None)


# ------------------------------------------------------------------ parsing
def parse_statistic(text: str) -> tuple[str, str | None, float | None] | None:
    """'Median FSC-A' / 'Median : FSC-A' / 'Percentile 95 FSC-A' / 'Count'
    -> (statistic, param, percentile), or None when it is not a statistic."""
    t = text.strip()
    low = t.lower()
    best = None
    for name in STATISTICS:
        n = name.lower()
        if low == n or (low.startswith(n) and len(low) > len(n) and low[len(n)] in " :\t"):
            if best is None or len(name) > len(best):
                best = name
    if best is None:
        return None
    rest = t[len(best):].strip()
    if rest.startswith(":"):
        rest = rest[1:].strip()
    percentile = None
    if best == "Percentile":
        m = re.match(r"^([0-9]+(?:\.[0-9]*)?)\s*(?::\s*)?(.*)$", rest)
        if m:
            percentile = float(m.group(1))
            rest = m.group(2).strip()
    param = rest or None
    if best not in NEEDS_PARAM and param:
        return None  # e.g. "Count of beads": not this statistic
    return best, param, percentile


def stat_token(statistic: str, param: str | None = None, percentile: float | None = None, pop_path: str | None = None) -> str:
    """The token text that shows a statistic, e.g. '{Median FSC-A}'."""
    body = statistic
    if statistic == "Percentile" and percentile is not None:
        body = f"Percentile {percentile:g}"
    if param and statistic in NEEDS_PARAM:
        body = f"{body} {param}"
    if pop_path is not None:
        body = f"{pop_path or '/'} | {body}"
    return "{" + body + "}"


def keyword_token(keyword: str) -> str:
    return "{" + keyword + "}"


def _keyword_value(sample, name: str) -> str | None:
    kws = getattr(sample, "keywords", None) or {}
    if name in kws:
        return str(kws[name])
    up = name.upper()
    for k, v in kws.items():
        if str(k).upper() == up:
            return str(v)
    return None


def _find_population(sample, spec: str):
    root = getattr(sample, "root", None)
    if root is None:
        return None
    spec = spec.strip()
    if spec in ("", "/", "Ungated", "ungated"):
        return root
    if spec.startswith("/"):
        return root.find(spec)
    hit = root.find(spec)
    if hit is not None:
        return hit
    for p in root.walk(False):  # a bare population name: the first match
        if p.name == spec:
            return p
    return None


def _builtin(name: str, ctx: TextContext) -> str | None:
    key = name.strip().lower()
    s = ctx.sample
    pop = ctx.base_population
    if key in ("sample", "sample name", "name", "$sample"):
        return getattr(s, "name", None)
    if key == "file":
        p = getattr(s, "path", None)
        return getattr(p, "name", None) if p is not None else None
    if key == "group":
        return ctx.group_name or None
    if key in ("population", "pop"):
        return getattr(pop, "display_name", None)
    if key == "path":
        return (getattr(pop, "path", "") or "/") if pop is not None else None
    if key == "layout":
        return ctx.layout_name or None
    if key == "page":
        return str(ctx.page)
    if key == "pages":
        return str(ctx.pages)
    if key == "date":
        return _dt.date.today().isoformat()
    if key == "time":
        return _dt.datetime.now().strftime("%H:%M")
    return None


def statistic_text(sample, population, statistic: str, param: str | None, percentile: float | None, decimals: int = 1) -> str:
    if population is None:
        return ""
    try:
        value = compute_statistic(statistic, population, param, percentile)
    except (ValueError, KeyError):
        return ""
    return format_value(statistic, value, decimals)


def resolve_token(token: str, ctx: TextContext) -> str | None:
    """The replacement of one token (text between the braces), or None."""
    t = token.strip()
    if not t:
        return None
    for k, v in ctx.extra.items():
        if k.lower() == t.lower():
            return str(v)
    for fn in list(_EXPANDERS):
        try:
            out = fn(t, ctx)
        except Exception:  # a broken plug-in must not break painting
            out = None
        if out is not None:
            return str(out)
    value = _builtin(t, ctx)
    if value is not None:
        return value
    s = ctx.sample
    if s is None:
        return None
    if t.startswith("$"):
        kv = _keyword_value(s, t)
        if kv is not None:
            return kv
    pop = ctx.base_population
    body = t
    scoped = "|" in t
    if scoped:
        spec, body = (x.strip() for x in t.split("|", 1))
        pop = _find_population(s, spec)
        if pop is None:
            return ""
    stat = parse_statistic(body)
    if stat is not None:
        name, param, percentile = stat
        if name in NEEDS_PARAM and not param:
            return None
        return statistic_text(s, pop, name, param, percentile, ctx.decimals)
    if not scoped:
        kv = _keyword_value(s, t)
        if kv is None and not t.startswith("$"):
            kv = _keyword_value(s, "$" + t)
        if kv is not None:
            return kv
    return None


def expand_text(text: str, ctx: TextContext | None = None, **kwargs) -> str:
    """Replace every token of `text` (see the module docstring)."""
    if not text or "{" not in text and "}" not in text:
        return text
    ctx = ctx or TextContext(**kwargs)

    def repl(m: re.Match) -> str:
        whole = m.group(0)
        if whole == "{{":
            return "{"
        if whole == "}}":
            return "}"
        value = resolve_token(m.group(1), ctx)
        return whole if value is None else value

    return TOKEN_RE.sub(repl, text)


def has_tokens(text: str) -> bool:
    return bool(text) and any(m.group(1) for m in TOKEN_RE.finditer(text))


# -------------------------------------------------------------- annotations
def annotation_label(entry: dict) -> str:
    """Display label of an annotation entry ({"statistic", "param",
    "percentile", "pop_path"} | {"keyword"} | {"text"})."""
    if entry.get("label"):
        return str(entry["label"])
    if entry.get("keyword"):
        return str(entry["keyword"])
    if entry.get("text"):
        return str(entry["text"])
    label = statistic_label(entry.get("statistic", "Count"), entry.get("param"), entry.get("percentile"))
    path = entry.get("pop_path")
    if path:
        label = f"{path.rsplit('/', 1)[-1]} {label}"
    return label


def annotation_value(entry: dict, ctx: TextContext) -> str:
    s = ctx.sample
    if entry.get("keyword"):
        return (_keyword_value(s, str(entry["keyword"])) or "") if s is not None else ""
    if entry.get("text"):
        return expand_text(str(entry["text"]), ctx)
    if s is None:
        return ""
    pop = ctx.base_population
    if entry.get("pop_path"):
        pop = _find_population(s, entry["pop_path"])
    return statistic_text(s, pop, entry.get("statistic", "Count"), entry.get("param"), entry.get("percentile"), ctx.decimals)


def annotation_lines(entries: list[dict], ctx: TextContext) -> list[str]:
    """One caption line per annotation: 'Label: value' ('text' entries are
    shown expanded, without a label)."""
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        if e.get("text") and not e.get("label"):
            out.append(expand_text(str(e["text"]), ctx))
        else:
            out.append(f"{annotation_label(e)}: {annotation_value(e, ctx)}")
    return out


def safe_filename(name: str) -> str:
    """A file-name-safe version of a sample / layout name (all platforms)."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(name)).strip(" .")
    return cleaned or "untitled"
