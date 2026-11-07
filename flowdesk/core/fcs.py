"""Reader for FCS 2.0 / 3.0 / 3.1 list-mode files.

Only the public FCS standard is implemented here: the 58-byte header, the
delimited TEXT segment (plus supplemental TEXT), and the DATA segment in
float, double or integer form.
"""
from __future__ import annotations

import re
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


_KEY_RE = re.compile(r"^\$[A-Za-z0-9]+$")


def _header_int(raw: bytes) -> int:
    text = raw.decode("ascii", "replace").strip()
    return int(text) if text else 0


def parse_text_segment(segment: bytes) -> dict[str, str]:
    """Parse a delimited TEXT segment into a keyword dictionary."""
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
        key = merged[k].decode("utf-8", "replace").strip()
        value = merged[k + 1].decode("utf-8", "replace").strip()
        if key:
            keywords[key] = value
    return keywords


def channels_from_keywords(keywords: dict[str, str]) -> list[Channel]:
    n = int(float(keywords.get("$PAR", "0") or 0))
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


def parse_spillover(keywords: dict[str, str]) -> tuple[list[str], np.ndarray] | None:
    """Return (parameter names, spillover matrix) from $SPILLOVER / SPILL / $COMP."""
    for key in ("$SPILLOVER", "SPILL", "SPILLOVER", "$COMP", "COMP"):
        text = keywords.get(key)
        if not text:
            continue
        parts = [p.strip() for p in text.split(",")]
        try:
            n = int(parts[0])
        except (ValueError, IndexError):
            continue
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


class FCSFile:
    """A parsed FCS file: keywords, channel descriptions and the event matrix."""

    def __init__(self, path: Path, version: str, keywords: dict[str, str], channels: list[Channel], events: np.ndarray | None):
        self.path = Path(path)
        self.version = version
        self.keywords = keywords
        self.channels = channels
        self.events = events

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _read_segments(fh) -> tuple[str, dict[str, str], int, int]:
        header = fh.read(58)
        if len(header) < 58:
            raise FCSError("file too small to be an FCS file")
        version = header[:6].decode("ascii", "replace")
        if not version.startswith("FCS"):
            raise FCSError(f"not an FCS file (header is {version!r})")
        text_start, text_end = _header_int(header[10:18]), _header_int(header[18:26])
        data_start, data_end = _header_int(header[26:34]), _header_int(header[34:42])
        if text_end <= text_start:
            raise FCSError("invalid TEXT segment offsets")
        fh.seek(text_start)
        keywords = parse_text_segment(fh.read(text_end - text_start + 1))
        # Supplemental TEXT (FCS 3.x)
        try:
            s_start = int(float(keywords.get("$BEGINSTEXT", "0") or 0))
            s_end = int(float(keywords.get("$ENDSTEXT", "0") or 0))
        except ValueError:
            s_start = s_end = 0
        if s_end > s_start > 0 and (s_start, s_end) != (text_start, text_end):
            fh.seek(s_start)
            try:
                extra = parse_text_segment(fh.read(s_end - s_start + 1))
            except FCSError:
                extra = {}
            for key, value in extra.items():
                keywords.setdefault(key, value)
        if data_start == 0 or data_end == 0:
            try:
                data_start = int(float(keywords.get("$BEGINDATA", "0") or 0))
                data_end = int(float(keywords.get("$ENDDATA", "0") or 0))
            except ValueError:
                data_start = data_end = 0
        return version, keywords, data_start, data_end

    @classmethod
    def read_keywords(cls, path: Path | str) -> tuple[str, dict[str, str]]:
        """Fast path: read only the header and TEXT segment."""
        with open(path, "rb") as fh:
            version, keywords, _, _ = cls._read_segments(fh)
        return version, keywords

    @classmethod
    def read(cls, path: Path | str, load_data: bool = True) -> "FCSFile":
        path = Path(path)
        with open(path, "rb") as fh:
            version, keywords, data_start, data_end = cls._read_segments(fh)
            channels = channels_from_keywords(keywords)
            events = None
            if load_data:
                if data_end <= data_start or not channels:
                    raise FCSError("file has no DATA segment")
                events = cls._read_data(fh, keywords, channels, data_start, data_end)
        return cls(path, version, keywords, channels, events)

    @staticmethod
    def _read_data(fh, keywords: dict[str, str], channels: list[Channel], data_start: int, data_end: int) -> np.ndarray:
        datatype = keywords.get("$DATATYPE", "F").strip().upper()
        byteord = keywords.get("$BYTEORD", "1,2,3,4").strip()
        order = ">" if byteord.startswith("4") or byteord.startswith("2,1") else "<"
        n_par = len(channels)
        try:
            tot = int(float(keywords.get("$TOT", "0") or 0))
        except ValueError:
            tot = 0
        fh.seek(data_start)
        raw = fh.read(data_end - data_start + 1)
        if datatype in ("F", "D"):
            dtype = np.dtype(order + ("f4" if datatype == "F" else "f8"))
            n_values = (len(raw) // dtype.itemsize) // n_par * n_par
            events = np.frombuffer(raw, dtype=dtype, count=n_values).reshape(-1, n_par).astype(np.float32)
        elif datatype == "I":
            bits = [ch.bits for ch in channels]
            if any(b not in (8, 16, 32, 64) for b in bits):
                raise FCSError(f"unsupported integer widths {sorted(set(bits))}")
            if len(set(bits)) == 1:
                dtype = np.dtype(f"{order}u{bits[0] // 8}")
                n_values = (len(raw) // dtype.itemsize) // n_par * n_par
                ints = np.frombuffer(raw, dtype=dtype, count=n_values).reshape(-1, n_par)
            else:
                dtype = np.dtype([(f"p{i}", f"{order}u{b // 8}") for i, b in enumerate(bits)])
                n_values = len(raw) // dtype.itemsize
                rec = np.frombuffer(raw, dtype=dtype, count=n_values)
                ints = np.column_stack([rec[f"p{i}"].astype(np.uint64) for i in range(n_par)])
            events = np.empty(ints.shape, dtype=np.float32)
            for j, ch in enumerate(channels):
                col = ints[:, j].astype(np.float64)
                rng = int(ch.range) if ch.range > 0 else 0
                if rng and (rng & (rng - 1)) == 0 and rng < (1 << ch.bits):
                    col = np.mod(col, rng)
                if ch.is_log_amplified:
                    offset = ch.log_offset if ch.log_offset > 0 else 1.0
                    col = offset * np.power(10.0, ch.log_decades * col / max(ch.range, 1.0))
                events[:, j] = col
        elif datatype == "A":
            raise FCSError("ASCII-encoded FCS data is not supported")
        else:
            raise FCSError(f"unknown $DATATYPE {datatype!r}")
        if tot and events.shape[0] > tot:
            events = events[:tot]
        return np.ascontiguousarray(events)

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


_COPY_KEYWORDS = ("$CYT", "$CYTSN", "$DATE", "$BTIM", "$ETIM", "$INST", "$OP", "$PROJ", "$EXP", "$SRC", "$SMNO", "$CELLS", "$COM", "$TIMESTEP", "$SPILLOVER", "$ORIGINALITY", "$LAST_MODIFIER", "$PLATEID", "$PLATENAME", "$WELLID", "$VOL")


def write_fcs(path: Path | str, events: np.ndarray, channels: list[Channel], keywords: dict[str, str] | None = None, extra_keywords: dict[str, str] | None = None):
    """Write an FCS 3.1 file with float32 list-mode data.

    Standard $Pn* keywords are generated from `channels`; a selection of
    descriptive keywords is copied from `keywords` and `extra_keywords` is
    written verbatim (vendor keywords of the source file are copied too).
    """
    events = np.ascontiguousarray(np.asarray(events, dtype=np.float32))
    if events.ndim != 2:
        raise FCSError("events must be a 2-D array")
    n, p = events.shape
    if p != len(channels):
        raise FCSError("channel list does not match the event matrix")

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
    for i, ch in enumerate(channels, start=1):
        col_max = float(np.nanmax(events[:, i - 1])) if n else 0.0
        rng = max(ch.range if ch.range and ch.range > 0 else 0.0, float(np.ceil(col_max)) if np.isfinite(col_max) else 0.0, 1.0)
        text[f"$P{i}N"] = ch.name
        if ch.stain and ch.stain != ch.name:
            text[f"$P{i}S"] = ch.stain
        text[f"$P{i}B"] = "32"
        text[f"$P{i}E"] = "0,0"
        text[f"$P{i}R"] = f"{rng:.0f}"
        if ch.gain and ch.gain != 1.0:
            text[f"$P{i}G"] = f"{ch.gain:g}"
    if keywords:
        for key, value in keywords.items():
            if key in text or key.startswith("$P") or key.startswith("$BEGIN") or key.startswith("$END") or key in ("$TOT", "$PAR", "$NEXTDATA", "$DATATYPE", "$BYTEORD", "$MODE"):
                continue
            if key in _COPY_KEYWORDS or not key.startswith("$"):
                text[key] = value
    if extra_keywords:
        text.update(extra_keywords)
    # offsets use a fixed width so that the TEXT length does not change
    text["$BEGINDATA"] = "0" * 12
    text["$ENDDATA"] = "0" * 12
    # choose a delimiter that occurs in no key or value, so nothing needs escaping
    used = "".join(list(text.keys()) + [str(v) for v in text.values()])
    delim = next((d for d in ("/", "|", "\\", "!", "\x1e", "\x0c", "~") if d not in used), "/")

    def esc(value) -> str:
        return str(value).replace(delim, delim + delim)

    def encode(kw: dict[str, str]) -> bytes:
        body = delim + delim.join(f"{esc(k)}{delim}{esc(v)}" for k, v in kw.items()) + delim
        return body.encode("utf-8")

    header_len = 58
    text_bytes = encode(text)
    text_start = header_len
    text_end = text_start + len(text_bytes) - 1
    data_start = text_end + 1
    data_end = data_start + n * p * 4 - 1
    text["$BEGINDATA"] = f"{data_start:012d}"
    text["$ENDDATA"] = f"{data_end:012d}"
    text_bytes = encode(text)
    assert len(text_bytes) == text_end - text_start + 1

    def field(v: int) -> bytes:
        return (f"{v:8d}" if v <= 99_999_999 else f"{0:8d}").encode("ascii")

    header = b"FCS3.1    " + field(text_start) + field(text_end) + field(data_start) + field(data_end) + field(0) + field(0)
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(text_bytes)
        fh.write(events.astype("<f4").tobytes())
