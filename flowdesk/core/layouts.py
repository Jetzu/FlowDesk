"""Definitions for the Layout Editor and the Table Editor."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .gates import new_id
from .model import GraphSettings

PAGE_SIZES = {
    "A4": (595.0, 842.0),
    "Letter": (612.0, 792.0),
    "Legal": (612.0, 1008.0),
    "A3": (842.0, 1191.0),
    "A5": (420.0, 595.0),
    "Tabloid": (792.0, 1224.0),
    "Slide 16:9": (720.0, 405.0),
    "Slide 4:3": (720.0, 540.0),
}
CUSTOM_PAGE = "Custom"
DEFAULT_MARGIN = 36.0


def _num(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _copy_props(props) -> dict:
    """Deep-ish copy of a JSON-like props dict (nested dicts/lists copied)."""
    if not isinstance(props, dict):
        return {}
    return copy.deepcopy(props)


@dataclass
class LayoutItem:
    kind: str = "plot"  # plot | text
    x: float = 0.0
    y: float = 0.0
    w: float = 256.0
    h: float = 256.0
    sample_id: int | None = None
    pop_path: str = ""
    graph: GraphSettings = field(default_factory=GraphSettings)
    text: str = ""
    font_size: int = 11
    show_caption: bool = True
    show_title: bool = False
    overlays: list[dict] = field(default_factory=list)
    # Generic, JSON-serialisable per-item settings shared with other packages
    # (text formatting: bold/italic/color/align/font_family, annotations,
    # custom item kinds, ...).  Round-tripped verbatim by to_dict/from_dict.
    props: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "sample_id": self.sample_id,
            "pop_path": self.pop_path,
            "graph": self.graph.to_dict(),
            "text": self.text,
            "font_size": self.font_size,
            "show_caption": self.show_caption,
            "show_title": self.show_title,
            "overlays": [dict(o) for o in self.overlays],
            "props": _copy_props(self.props),
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutItem":
        return cls(
            kind=d.get("kind", "plot"),
            x=float(d.get("x", 0)),
            y=float(d.get("y", 0)),
            w=float(d.get("w", 256)),
            h=float(d.get("h", 256)),
            sample_id=d.get("sample_id"),
            pop_path=d.get("pop_path", ""),
            graph=GraphSettings.from_dict(d.get("graph")),
            text=d.get("text", ""),
            font_size=int(d.get("font_size", 11)),
            show_caption=bool(d.get("show_caption", True)),
            show_title=bool(d.get("show_title", False)),
            overlays=[dict(o) for o in d.get("overlays", [])],
            props=_copy_props(d.get("props")),
            id=d.get("id", new_id()),
        )


@dataclass
class Layout:
    name: str = "Layout"
    items: list[LayoutItem] = field(default_factory=list)
    page: str = "A4"
    landscape: bool = False
    batch_group: str | None = None
    # Generic layout-level settings (margins, custom page size, batch header,
    # iteration mode, ...); round-tripped verbatim by to_dict/from_dict.
    props: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)

    @property
    def page_size(self) -> tuple[float, float]:
        if self.page == "Custom":
            cw = _num(self.props.get("custom_w"), 0.0)
            ch = _num(self.props.get("custom_h"), 0.0)
            if cw > 0 and ch > 0:
                w, h = cw, ch
            else:
                w, h = PAGE_SIZES["A4"]
        else:
            w, h = PAGE_SIZES.get(self.page, PAGE_SIZES["A4"])
        return (h, w) if self.landscape else (w, h)

    @property
    def margin(self) -> float:
        """Page margin in points (props["margin"], default 36 pt)."""
        return max(0.0, _num(self.props.get("margin"), DEFAULT_MARGIN))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "items": [i.to_dict() for i in self.items],
            "page": self.page,
            "landscape": self.landscape,
            "batch_group": self.batch_group,
            "props": _copy_props(self.props),
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls(
            name=d.get("name", "Layout"),
            items=[LayoutItem.from_dict(i) for i in d.get("items", [])],
            page=d.get("page", "A4"),
            landscape=bool(d.get("landscape", False)),
            batch_group=d.get("batch_group"),
            props=_copy_props(d.get("props")),
            id=d.get("id", new_id()),
        )


@dataclass
class TableColumn:
    pop_path: str = ""
    statistic: str = "Freq. of Parent"
    param: str | None = None
    percentile: float | None = None
    keyword: str | None = None  # a keyword column instead of a statistic
    label: str = ""
    props: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)

    @property
    def auto_label(self) -> str:
        if self.keyword:
            return self.keyword
        from .stats import statistic_label

        pop = self.pop_path.rsplit("/", 1)[-1] if self.pop_path else "Ungated"
        return f"{pop} | {statistic_label(self.statistic, self.param, self.percentile)}"

    @property
    def header(self) -> str:
        return self.label or self.auto_label

    def to_dict(self) -> dict:
        return {
            "pop_path": self.pop_path,
            "statistic": self.statistic,
            "param": self.param,
            "percentile": self.percentile,
            "keyword": self.keyword,
            "label": self.label,
            "props": _copy_props(self.props),
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableColumn":
        return cls(
            pop_path=d.get("pop_path", ""),
            statistic=d.get("statistic", "Freq. of Parent"),
            param=d.get("param"),
            percentile=d.get("percentile"),
            keyword=d.get("keyword"),
            label=d.get("label", ""),
            props=_copy_props(d.get("props")),
            id=d.get("id", new_id()),
        )


@dataclass
class TableDef:
    name: str = "Table"
    columns: list[TableColumn] = field(default_factory=list)
    group_name: str = "All Samples"
    props: dict = field(default_factory=dict)
    id: str = field(default_factory=new_id)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "group_name": self.group_name,
            "props": _copy_props(self.props),
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableDef":
        return cls(
            name=d.get("name", "Table"),
            columns=[TableColumn.from_dict(c) for c in d.get("columns", [])],
            group_name=d.get("group_name", "All Samples"),
            props=_copy_props(d.get("props")),
            id=d.get("id", new_id()),
        )
