"""Reader and writer for FCS 2.0 / 3.0 / 3.1 (and 3.2) list-mode files.

Only the public FCS standard is implemented here: the 58-byte header, the
delimited TEXT segment (plus supplemental TEXT), and the DATA segment in
float, double or integer form (FCS 3.2 per-parameter $PnDATATYPE included).

Files with several data sets ($NEXTDATA chain, e.g. Beckman Coulter .LMD
files) are listed by `FCSFile.list_datasets`; `read` / `read_keywords` take
a `dataset` index. Without one the first data set is read, except for .lmd
files, where the first FCS 3.x data set is used (as FlowJo does).

Large DATA segments are memory-mapped (`set_memmap`, the application's
``fcs_memmap`` preference): native float32 data is used in place, other
layouts are converted in chunks so that loading never needs twice the file
size in memory.
"""
from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FCSError(Exception):
    """Raised when a file cannot be parsed as FCS."""


@dataclass
class Channel:
    """One measured parameter ($PnN and friends)."""

    index: int
    name: str
    stain: str = ""
    range: float = 262144.0
    bits: int = 32
    log_decades: float = 0.0
    log_offset: float = 0.0
    gain: float = 1.0
    display: str = ""

    @property
    def label(self) -> str:
        if self.stain and self.stain != self.name:
            return f"{self.name} :: {self.stain}"
        return self.name

    @property
    def is_log_amplified(self) -> bool:
        return self.log_decades > 0


# file suffixes of list-mode files that hold FCS data sets
SAMPLE_SUFFIXES = (".fcs", ".lmd")

# ----------------------------------------------------------------- memory mapping
# DATA segments of at least MEMMAP_MIN_BYTES are memory-mapped when enabled
# (the application sets this from its "fcs_memmap" preference).
MEMMAP_ENABLED = True
MEMMAP_MIN_BYTES = 64 * 1024 * 1024
_CHUNK_BYTES = 32 * 1024 * 1024


def set_memmap(enabled: bool | None = None, min_bytes: int | None = None):
    """Turn memory-mapped reading of large DATA segments on or off and/or set
    the size from which it is used."""
    global MEMMAP_ENABLED, MEMMAP_MIN_BYTES
    if enabled is not None:
        MEMMAP_ENABLED = bool(enabled)
    if min_bytes is not None:
        MEMMAP_MIN_BYTES = max(0, int(min_bytes))


def memmap_settings() -> tuple[bool, int]:
    return MEMMAP_ENABLED, MEMMAP_MIN_BYTES


_KEY_RE = re.compile(r"^\$[A-Za-z0-9]+$")


def _header_int(raw: bytes) -> int:
    text = raw.decode("ascii", "replace").strip()
    try:
        return int(text) if text else 0
    except ValueError:
        return 0


def _decode(raw: bytes) -> str:
    """UTF-8 (FCS 3.1) with a fallback to Windows-1252 / Latin-1, which older
    Diva / Beckman files use for µ, °, ü in their keyword values."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_text_segment(segment: bytes) -> dict[str, str]:
    """Parse a delimited TEXT segment into a keyword dictionary.

    Standard keyword names ($...) are case-insensitive (FCS 3.1) and are
    returned in upper case, so files writing '$par' or '$p1n' read normally."""
    if not segment:
        raise FCSError("empty TEXT segment")
    delim = segment[:1]
    body = segment[1:]
    if body.endswith(delim):
        body = body[:-1]
    tokens = body.split(delim)
    # A doubled delimiter is an escaped delimiter inside a value. It shows up as
    # an empty token. Standard keywords ($...) never carry escaped delimiters, so
    # an empty token followed by a standard keyword is an (invalid but common)
    # empty value instead.
    merged: list[bytes] = []
    i = 0
    n = len(tokens)

    def looks_like_key(tok: bytes) -> bool:
        return bool(_KEY_RE.match(tok.decode("ascii", "replace").strip()))

    while i < n:
        tok = tokens[i]
        is_value = len(merged) % 2 == 1
        if is_value:
            # a value that begins with an escaped delimiter shows up as an empty token
            if tok == b"" and i + 1 < n and not looks_like_key(tokens[i + 1]):
                tok = delim + tokens[i + 1]
                i += 1
            while i + 2 < n and tokens[i + 1] == b"" and not looks_like_key(tokens[i + 2]):
                tok = tok + delim + tokens[i + 2]
                i += 2
        merged.append(tok)
        i += 1
    keywords: dict[str, str] = {}
    for k in range(0, len(merged) - 1, 2):
        key = _decode(merged[k]).strip()
        value = _decode(merged[k + 1]).strip()
        if key.startswith("$") and key.isascii():
            key = key.upper()
        if key:
            keywords[key] = value
    return keywords


def channels_from_keywords(keywords: dict[str, str]) -> list[Channel]:
    try:
        n = int(float(keywords.get("$PAR", "0") or 0))
    except ValueError:
        n = 0
    channels = []
    for i in range(1, n + 1):
        name = keywords.get(f"$P{i}N", f"P{i}").strip() or f"P{i}"
        stain = keywords.get(f"$P{i}S", "").strip()
        try:
            rng = float(keywords.get(f"$P{i}R", "262144") or 262144)
        except ValueError:
            rng = 262144.0
        try:
            bits = int(float(keywords.get(f"$P{i}B", "32") or 32))
        except ValueError:
            bits = 32
        amp = keywords.get(f"$P{i}E", "0,0")
        try:
            parts = [float(v) for v in amp.split(",")[:2]]
            f1, f2 = (parts + [0.0, 0.0])[:2]
        except ValueError:
            f1, f2 = 0.0, 0.0
        try:
            gain = float(keywords.get(f"$P{i}G", "1") or 1)
        except ValueError:
            gain = 1.0
        display = (keywords.get(f"$P{i}D") or keywords.get(f"P{i}DISPLAY") or "").strip()
        channels.append(Channel(i, name, stain, rng, bits, f1, f2, gain, display))
    return channels


SPILLOVER_KEYS = ("$SPILLOVER", "SPILL", "SPILLOVER", "$COMP", "COMP")


def parse_spillover(keywords: dict[str, str]) -> tuple[list[str], np.ndarray] | None:
    """Return (parameter names, spillover matrix) from $SPILLOVER / SPILL / $COMP.

    $COMP is also accepted in its FCS 3.0 standard form, n followed by n*n
    coefficients without parameter names (it then refers to parameters 1..n)."""
    for key in SPILLOVER_KEYS:
        text = keywords.get(key)
        if not text:
            continue
        parts = [p.strip() for p in text.split(",")]
        try:
            n = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        if n <= 0:
            continue
        if key in ("$COMP", "COMP") and len(parts) == 1 + n * n:
            names = [keywords.get(f"$P{i}N", "").strip() for i in range(1, n + 1)]
            if not all(names):
                continue
            try:
                matrix = np.array([float(v) for v in parts[1:]], dtype=float).reshape(n, n)
            except ValueError:
                continue
            return names, matrix
        names = parts[1 : 1 + n]
        values = parts[1 + n :]
        if len(names) != n or len(values) < n * n:
            continue
        try:
            matrix = np.array([float(v) for v in values[: n * n]], dtype=float).reshape(n, n)
        except ValueError:
            continue
        return names, matrix
    return None


def format_spillover(params: list[str], matrix) -> str:
    """$SPILLOVER text: n, the n parameter names, then the n*n values row by row."""
    m = np.asarray(matrix, dtype=float).reshape(len(params), len(params))
    return ",".join([str(len(params)), *params, *(f"{v:.10g}" for v in m.ravel())])


def looks_like_fcs(path: Path | str) -> bool:
    """True when the file starts with an FCS header ("FCS2.0", "FCS3.1", ...),
    whatever its suffix."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(10)
    except OSError:
        return False
    return len(head) >= 6 and head[:3] == b"FCS" and bool(re.match(rb"^\d\.\d$", head[3:6]))


def _int_kw(keywords: dict[str, str], key: str) -> int:
    try:
        return int(float(keywords.get(key, "0") or 0))
    except ValueError:
        return 0


def _column_layout(keywords: dict[str, str], channels: list[Channel]) -> list[tuple[str, int]]:
    """(type, byte width) of every parameter: 'F', 'D' or 'I' ($PnDATATYPE
    of FCS 3.2 overrides $DATATYPE per parameter)."""
    datatype = keywords.get("$DATATYPE", "F").strip().upper() or "F"
    if datatype == "A":
        raise FCSError("ASCII-encoded FCS data is not supported")
    layout = []
    bad_bits = set()
    for ch in channels:
        kind = (keywords.get(f"$P{ch.index}DATATYPE", "") or datatype).strip().upper() or datatype
        if kind == "A":
            raise FCSError("ASCII-encoded FCS data is not supported")
        if kind == "F":
            layout.append(("F", 4))
        elif kind == "D":
            layout.append(("D", 8))
        elif kind == "I":
            if ch.bits not in (8, 16, 32, 64):
                bad_bits.add(ch.bits)
            layout.append(("I", ch.bits // 8 if ch.bits in (8, 16, 32, 64) else 0))
        else:
            raise FCSError(f"unknown $DATATYPE {kind!r}")
    if bad_bits:
        bits = sorted({ch.bits for ch, (k, _) in zip(channels, layout) if k == "I"})
        raise FCSError(f"unsupported integer widths {bits}")
    return layout


def _byte_order(keywords: dict[str, str]) -> str:
    byteord = keywords.get("$BYTEORD", "1,2,3,4").strip()
    return ">" if byteord.startswith("4") or byteord.startswith("2,1") else "<"


def _np_type(kind: str, width: int, order: str) -> np.dtype:
    if kind == "F":
        return np.dtype(order + "f4")
    if kind == "D":
        return np.dtype(order + "f8")
    return np.dtype(f"{order}u{width}")


def _decode_int_column(col: np.ndarray, ch: Channel) -> np.ndarray:
    """Integer channel numbers -> values: $PnR bit mask (keep the low
    ceil(log2($PnR)) bits), $PnE log decoding or $PnG gain (linear only)."""
    rng = int(ch.range) if ch.range and ch.range >= 2 else 0
    if rng:
        nbits = (rng - 1).bit_length()
        if nbits < col.dtype.itemsize * 8:
            col = col & np.asarray((1 << nbits) - 1, dtype=col.dtype)
    values = col.astype(np.float64)
    if ch.is_log_amplified:
        offset = ch.log_offset if ch.log_offset > 0 else 1.0
        values = offset * np.power(10.0, ch.log_decades * values / max(ch.range, 1.0))
    elif ch.gain and ch.gain > 0 and ch.gain != 1.0:
        values = values / ch.gain
    return values


class FCSFile:
    """A parsed FCS file: keywords, channel descriptions and the event matrix."""

    def __init__(
        self,
        path: Path,
        version: str,
        keywords: dict[str, str],
        channels: list[Channel],
        events: np.ndarray | None,
        dataset: int = 0,
        offset: int = 0,
        warnings: list[str] | None = None,
    ):
        self.path = Path(path)
        self.version = version
        self.keywords = keywords
        self.channels = channels
        self.events = events
        self.dataset = dataset  # index of the data set in the file ($NEXTDATA chain)
        self.offset = offset  # byte offset of that data set
        # problems that did not stop the file from loading ($TOT larger than
        # the DATA segment, ...), as messages for the user
        self.warnings: list[str] = list(warnings or [])

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _read_segments(fh, base: int = 0) -> tuple[str, dict[str, str], int, int]:
        """Header and TEXT of the data set starting at byte `base`; returns
        (version, keywords, data start, data end) with absolute offsets."""
        fh.seek(base)
        header = fh.read(58)
        if len(header) < 58:
            raise FCSError("file too small to be an FCS file")
        version = header[:6].decode("ascii", "replace")
        if not version.startswith("FCS"):
            raise FCSError(f"not an FCS file (header is {version!r})")
        text_start, text_end = _header_int(header[10:18]), _header_int(header[18:26])
        h_start, h_end = _header_int(header[26:34]), _header_int(header[34:42])
        if text_end <= text_start:
            raise FCSError("invalid TEXT segment offsets")
        fh.seek(base + text_start)
        keywords = parse_text_segment(fh.read(text_end - text_start + 1))
        # Supplemental TEXT (FCS 3.x)
        s_start = _int_kw(keywords, "$BEGINSTEXT")
        s_end = _int_kw(keywords, "$ENDSTEXT")
        if s_end > s_start > 0 and (s_start, s_end) != (text_start, text_end):
            fh.seek(base + s_start)
            try:
                extra = parse_text_segment(fh.read(s_end - s_start + 1))
            except FCSError:
                extra = {}
            for key, value in extra.items():
                keywords.setdefault(key, value)
        t_start, t_end = _int_kw(keywords, "$BEGINDATA"), _int_kw(keywords, "$ENDDATA")
        data_start, data_end = FCSFile._choose_data_offsets(fh, keywords, (h_start, h_end), (t_start, t_end), base)
        return version, keywords, data_start, data_end

    @staticmethod
    def _choose_data_offsets(fh, keywords, header_pair, text_pair, base) -> tuple[int, int]:
        """The DATA offsets: the TEXT values ($BEGINDATA / $ENDDATA) when they
        are set (also when the header disagrees), unless only the header ones
        hold all $TOT events."""
        cands = [p for p in (text_pair, header_pair) if p[1] > p[0] > 0]
        if not cands:
            return 0, 0
        if len(cands) == 2 and cands[0] != cands[1]:
            try:
                channels = channels_from_keywords(keywords)
                row = sum(w for _, w in _column_layout(keywords, channels)) if channels else 0
            except FCSError:
                row = 0
            tot = _int_kw(keywords, "$TOT")
            if row and tot:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                need = row * tot
                for start, end in cands:
                    if end - start + 1 >= need and base + start + need <= size:
                        return base + start, base + end
        start, end = cands[0]
        return base + start, base + end

    @staticmethod
    def _dataset_chain(fh) -> list[tuple[int, str, dict[str, str]]]:
        """(offset, version, keywords) of every data set, following $NEXTDATA
        (relative to the current data set; absolute values are accepted too)."""
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        out = []
        seen = set()
        base = 0
        while base not in seen and len(out) < 4096:
            seen.add(base)
            version, keywords, _, _ = FCSFile._read_segments(fh, base)
            out.append((base, version, keywords))
            nxt = _int_kw(keywords, "$NEXTDATA")
            if nxt <= 0:
                break
            found = None
            for cand in (base + nxt, nxt):
                if cand in seen or cand + 58 > size:
                    continue
                fh.seek(cand)
                if fh.read(3) == b"FCS":
                    found = cand
                    break
            if found is None:
                break
            base = found
        return out

    @classmethod
    def list_datasets(cls, path: Path | str) -> list[int]:
        """Byte offsets of the data sets in the file (one for most files)."""
        with open(path, "rb") as fh:
            return [off for off, _, _ in cls._dataset_chain(fh)]

    @classmethod
    def datasets(cls, path: Path | str) -> list[tuple[int, str, dict[str, str]]]:
        """(offset, version, keywords) of every data set in the file."""
        with open(path, "rb") as fh:
            return cls._dataset_chain(fh)

    @classmethod
    def default_dataset(cls, path: Path | str) -> int:
        """Index of the data set read when none is given: the first FCS 3.x
        data set of a .lmd file (Beckman Coulter keeps an FCS 2.0 copy first),
        the first data set otherwise."""
        if Path(path).suffix.lower() != ".lmd":
            return 0
        with open(path, "rb") as fh:
            chain = cls._dataset_chain(fh)
        for i, (_, version, _) in enumerate(chain):
            if version.startswith("FCS3"):
                return i
        return 0

    @classmethod
    def _locate(cls, fh, path: Path, dataset: int | None) -> tuple[int, int]:
        """(index, byte offset) of the data set to read."""
        if dataset is None:
            if path.suffix.lower() != ".lmd":
                return 0, 0
            chain = cls._dataset_chain(fh)
            for i, (off, version, _) in enumerate(chain):
                if version.startswith("FCS3"):
                    return i, off
            return 0, 0
        dataset = int(dataset)
        if dataset == 0:
            return 0, 0
        chain = cls._dataset_chain(fh)
        if not 0 <= dataset < len(chain):
            raise FCSError(f"{path.name} has {len(chain)} data set(s), not {dataset + 1}")
        return dataset, chain[dataset][0]

    @classmethod
    def read_keywords(cls, path: Path | str, dataset: int | None = None) -> tuple[str, dict[str, str]]:
        """Fast path: read only the header and TEXT segment."""
        path = Path(path)
        with open(path, "rb") as fh:
            _, base = cls._locate(fh, path, dataset)
            version, keywords, _, _ = cls._read_segments(fh, base)
        return version, keywords

    @classmethod
    def read(cls, path: Path | str, load_data: bool = True, dataset: int | None = None) -> "FCSFile":
        path = Path(path)
        notes: list[str] = []
        with open(path, "rb") as fh:
            index, base = cls._locate(fh, path, dataset)
            version, keywords, data_start, data_end = cls._read_segments(fh, base)
            channels = channels_from_keywords(keywords)
            events = None
            if load_data:
                if data_end <= data_start or not channels:
                    raise FCSError("file has no DATA segment")
                events = cls._read_data(fh, keywords, channels, data_start, data_end, notes)
        return cls(path, version, keywords, channels, events, dataset=index, offset=base, warnings=notes)

    @staticmethod
    def _read_data(fh, keywords: dict[str, str], channels: list[Channel], data_start: int, data_end: int, notes: list[str] | None = None) -> np.ndarray:
        datatype = keywords.get("$DATATYPE", "F").strip().upper() or "F"
        if datatype == "A":
            raise FCSError("ASCII-encoded FCS data is not supported")
        if datatype not in ("F", "D", "I"):
            raise FCSError(f"unknown $DATATYPE {datatype!r}")
        layout = _column_layout(keywords, channels)
        order = _byte_order(keywords)
        n_par = len(channels)
        row = sum(w for _, w in layout)
        tot = _int_kw(keywords, "$TOT")
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        seg_len = data_end - data_start + 1
        declared = seg_len // row
        expected = declared
        if tot > 0:
            if tot > declared:
                if notes is not None:
                    notes.append(f"$TOT is {tot:,} but the DATA segment holds {declared:,} events; {declared:,} were read")
            else:
                expected = tot
        have = max(0, min(seg_len, size - data_start)) // row
        if have < expected:
            raise FCSError(f"the file is truncated: its DATA segment holds {have:,} of {expected:,} events")
        n = expected
        uniform = len(set(layout)) == 1
        nbytes = n * row
        use_map = MEMMAP_ENABLED and nbytes >= MEMMAP_MIN_BYTES and n > 0
        if uniform:
            kind, width = layout[0]
            dtype = _np_type(kind, width, order)
            if use_map and kind == "F" and dtype == np.dtype(np.float32) and data_start % 4 == 0:
                # zero-copy: the events live in the page cache (copy-on-write)
                mm = np.memmap(fh, dtype=np.float32, mode="c", offset=data_start, shape=(n, n_par))
                return np.asarray(mm)
            if use_map:
                src = np.memmap(fh, dtype=dtype, mode="r", offset=data_start, shape=(n, n_par))
            else:
                fh.seek(data_start)
                src = np.frombuffer(fh.read(nbytes), dtype=dtype, count=n * n_par).reshape(n, n_par)
        else:
            dtype = np.dtype([(f"p{j}", _np_type(k, w, order)) for j, (k, w) in enumerate(layout)])
            if use_map:
                src = np.memmap(fh, dtype=dtype, mode="r", offset=data_start, shape=(n,))
            else:
                fh.seek(data_start)
                src = np.frombuffer(fh.read(nbytes), dtype=dtype, count=n)
        events = np.empty((n, n_par), dtype=np.float32)
        step = max(1, _CHUNK_BYTES // max(row, 1)) if use_map else max(n, 1)
        int_cols = [j for j, (k, _) in enumerate(layout) if k == "I"]
        for a in range(0, n, step):
            b = min(n, a + step)
            block = src[a:b]
            if uniform:
                if layout[0][0] == "I":
                    for j in range(n_par):
                        events[a:b, j] = _decode_int_column(np.asarray(block[:, j]), channels[j])
                else:
                    events[a:b] = block
            else:
                for j in range(n_par):
                    col = np.asarray(block[f"p{j}"])
                    events[a:b, j] = _decode_int_column(col, channels[j]) if j in int_cols else col
        del src
        return events

    # ---------------------------------------------------------------- accessors
    @property
    def n_events(self) -> int:
        return 0 if self.events is None else int(self.events.shape[0])

    @property
    def channel_names(self) -> list[str]:
        return [ch.name for ch in self.channels]

    def channel_index(self, name: str) -> int:
        for i, ch in enumerate(self.channels):
            if ch.name == name:
                return i
        for i, ch in enumerate(self.channels):
            if ch.stain == name:
                return i
        raise KeyError(name)

    def column(self, name: str) -> np.ndarray:
        if self.events is None:
            raise FCSError("event data not loaded")
        return self.events[:, self.channel_index(name)]

    def spillover(self) -> tuple[list[str], np.ndarray] | None:
        return parse_spillover(self.keywords)

    def __repr__(self) -> str:
        return f"<FCSFile {self.path.name} {self.version} {self.n_events} events x {len(self.channels)} parameters>"


# ------------------------------------------------------------------------ writing
# keywords describing the layout of the DATA segment: regenerated, never copied
_STRUCTURAL_KEYS = {
    "$BEGINANALYSIS", "$ENDANALYSIS", "$BEGINSTEXT", "$ENDSTEXT", "$BEGINDATA", "$ENDDATA",
    "$BYTEORD", "$DATATYPE", "$MODE", "$NEXTDATA", "$PAR", "$TOT", "$FIL", "$UNICODE", "$CSMODE", "$CSVBITS",
}
# per-parameter keywords regenerated from the Channel objects
_REGENERATED = {"N", "S", "B", "E", "R", "G", "DATATYPE"}
_PARAM_KEY = re.compile(r"^\$P(\d+)([A-Z][A-Z0-9_]*)$")
_VENDOR_PARAM_KEY = re.compile(r"^P(\d+)([A-Za-z][A-Za-z0-9_ ]*)$")
COMP_PREFIX = "Comp-"


def _param_index(keywords: dict[str, str]) -> tuple[dict[str, int], dict[int, list[tuple[bool, str, str]]]]:
    """Of a source keyword dict: $PnN -> n and n -> [(standard?, suffix, value)]
    for every per-parameter keyword other than the regenerated ones."""
    by_name: dict[str, int] = {}
    per: dict[int, list[tuple[bool, str, str]]] = {}
    for key, value in keywords.items():
        m = _PARAM_KEY.match(key)
        if m:
            idx, suffix = int(m.group(1)), m.group(2)
            if suffix == "N":
                by_name.setdefault(str(value).strip(), idx)
            if suffix not in _REGENERATED:
                per.setdefault(idx, []).append((True, suffix, value))
            continue
        m = _VENDOR_PARAM_KEY.match(key)
        if m:
            per.setdefault(int(m.group(1)), []).append((False, m.group(2), value))
    return by_name, per


def _source_index(ch: Channel, by_name: dict[str, int]) -> int | None:
    """Index of the channel's parameter in the source keywords: by $PnN (a
    "Comp-X" column by X); by Channel.index when the source names no
    parameters at all (per-parameter keywords given for the channel list)."""
    if not by_name:
        return ch.index if ch.index and ch.index > 0 else None
    idx = by_name.get(ch.name)
    if idx is None and ch.name.startswith(COMP_PREFIX):
        idx = by_name.get(ch.name[len(COMP_PREFIX):])
    return idx


def _col_max(col: np.ndarray) -> float:
    if col.size == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(np.nanmax(col))
    return value if math.isfinite(value) else 0.0


def write_fcs(
    path: Path | str,
    events: np.ndarray,
    channels: list[Channel],
    keywords: dict[str, str] | None = None,
    extra_keywords: dict[str, str] | None = None,
    channel_sources: list[dict[str, str] | None] | None = None,
):
    """Write an FCS 3.1 file with float32 list-mode data.

    The standard $PnN/$PnS/$PnB/$PnE/$PnR keywords are generated from
    `channels`. Every other keyword of the source file (`keywords`) is kept:
    descriptive ($CYT, $DATE, $TIMESTEP, $SPILLOVER, $SYS ...) and vendor
    keywords as they are, per-parameter ones ($PnV voltage, $PnF filter, $PnL
    wavelength, $PnD display, vendor PnDISPLAY ...) re-indexed to the
    position of their channel in the new file (found by $PnN, a "Comp-X"
    column by X). `channel_sources` may give, per channel, a different
    keyword dict to take these from (concatenated files). `extra_keywords` are
    written last, verbatim. Empty values are written as a single space.

    When `extra_keywords` sets a spillover matrix, or the file gets "Comp-X"
    columns of the source matrix's parameters, the source matrix keywords
    are not copied (the data would be compensated twice), and the latter
    case is marked FLOWDESK_COMPENSATED=TRUE.
    """
    events = np.asarray(events, dtype=np.float32)
    if events.ndim != 2:
        raise FCSError("events must be a 2-D array")
    n, p = events.shape
    if p != len(channels):
        raise FCSError("channel list does not match the event matrix")
    keywords = dict(keywords or {})
    extra = dict(extra_keywords or {})
    if channel_sources is not None and len(channel_sources) != p:
        raise FCSError("channel_sources does not match the channel list")

    text: dict[str, str] = {
        "$BEGINANALYSIS": "0",
        "$ENDANALYSIS": "0",
        "$BEGINSTEXT": "0",
        "$ENDSTEXT": "0",
        "$BYTEORD": "1,2,3,4",
        "$DATATYPE": "F",
        "$MODE": "L",
        "$NEXTDATA": "0",
        "$PAR": str(p),
        "$TOT": str(n),
        "$FIL": Path(path).name,
    }
    index_cache: dict[int, tuple[dict, dict]] = {}

    def indexed(src: dict[str, str]):
        got = index_cache.get(id(src))
        if got is None:
            got = _param_index(src)
            index_cache[id(src)] = got
        return got

    per_param: dict[str, str] = {}
    for i, ch in enumerate(channels, start=1):
        src = keywords if channel_sources is None or channel_sources[i - 1] is None else channel_sources[i - 1]
        by_name, per = indexed(src)
        old = _source_index(ch, by_name)
        col_max = _col_max(events[:, i - 1]) if n else 0.0
        rng = max(ch.range if ch.range and ch.range > 0 else 0.0, float(np.ceil(col_max)), 1.0)
        text[f"$P{i}N"] = ch.name
        if ch.stain and ch.stain != ch.name:
            text[f"$P{i}S"] = ch.stain
        text[f"$P{i}B"] = "32"
        text[f"$P{i}E"] = "0,0"
        text[f"$P{i}R"] = f"{rng:.0f}"
        # integer data were divided by $PnG when read: the float values written
        # here are already scaled, so the gain is only kept for float sources
        src_type = ""
        if old is not None:
            src_type = (src.get(f"$P{old}DATATYPE") or src.get("$DATATYPE") or "F").strip().upper()
        had_gain = old is not None and f"$P{old}G" in src
        if ch.gain and (ch.gain != 1.0 or had_gain) and src_type != "I":
            text[f"$P{i}G"] = f"{ch.gain:g}"
        copied_d = False
        if old is not None:
            for standard, suffix, value in per.get(old, ()):
                if standard:
                    per_param[f"$P{i}{suffix}"] = value
                    copied_d = copied_d or suffix == "D"
                else:
                    per_param[f"P{i}{suffix}"] = value
        if not copied_d and ch.display and re.match(r"^(lin|log)[a-z]*\s*,", ch.display, re.IGNORECASE):
            per_param[f"$P{i}D"] = ch.display
    text.update(per_param)

    matrix_keys = {k.upper() for k in SPILLOVER_KEYS}
    drop_matrix = any(k.upper() in matrix_keys for k in extra)
    if not drop_matrix:
        spill = parse_spillover(keywords)
        if spill is not None:
            names = {ch.name for ch in channels}
            if any(f"{COMP_PREFIX}{param}" in names for param in spill[0]):
                drop_matrix = True
                extra.setdefault("FLOWDESK_COMPENSATED", "TRUE")
    for key, value in keywords.items():
        upper = key.upper()
        if upper in _STRUCTURAL_KEYS or key in text or upper.startswith("$CSV") or upper.startswith("$DFC"):
            continue
        if _PARAM_KEY.match(key) or _VENDOR_PARAM_KEY.match(key):
            continue  # per-parameter: re-indexed above (dropped with its channel)
        if drop_matrix and upper in matrix_keys:
            continue
        text[key] = value
    text.update(extra)
    # offsets use a fixed width so that the TEXT length does not change
    text["$BEGINDATA"] = "0" * 12
    text["$ENDDATA"] = "0" * 12
    values = {k: (str(v) if str(v) != "" else " ") for k, v in text.items()}
    # choose a delimiter that occurs in no key or value, so nothing needs escaping
    used = "".join(list(values.keys()) + list(values.values()))
    delim = next((d for d in ("/", "|", "\\", "!", "\x1e", "\x0c", "~") if d not in used), "/")

    def esc(value) -> str:
        return str(value).replace(delim, delim + delim)

    def encode(kw: dict[str, str]) -> bytes:
        body = delim + delim.join(f"{esc(k)}{delim}{esc(v)}" for k, v in kw.items()) + delim
        return body.encode("utf-8")

    header_len = 58
    text_bytes = encode(values)
    text_start = header_len
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + n * p * 4 - 1
    values["$BEGINDATA"] = f"{data_start:012d}"
    values["$ENDDATA"] = f"{data_end:012d}"
    text_bytes = encode(values)
    assert len(text_bytes) == text_end - text_start + 1

    def field(v: int) -> bytes:
        return (f"{v:8d}" if v <= 99_999_999 else f"{0:8d}").encode("ascii")

    header = b"FCS3.1    " + field(text_start) + field(text_end) + field(data_start) + field(data_end) + field(0) + field(0)
    rows = max(1, _CHUNK_BYTES // max(p * 4, 1))
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(text_bytes)
        for a in range(0, n, rows):
            fh.write(np.ascontiguousarray(events[a:a + rows], dtype="<f4").tobytes())
