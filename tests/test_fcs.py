import numpy as np
import pytest

from flowdesk.core.fcs import (
    Channel,
    FCSError,
    FCSFile,
    channels_from_keywords,
    parse_spillover,
    parse_text_segment,
)
from tests.conftest import write_fcs


def _small(n=6, p=3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, p)) * 1000).astype(np.float32)


# ----------------------------------------------------------------- header / TEXT
def test_header_offsets_and_version(tmp_path):
    data = _small()
    path = write_fcs(tmp_path / "a.fcs", data, ["A", "B", "C"])
    raw = path.read_bytes()
    assert raw[:6] == b"FCS3.0"
    text_start = int(raw[10:18])
    text_end = int(raw[18:26])
    data_start = int(raw[26:34])
    data_end = int(raw[34:42])
    assert text_start == 58
    assert raw[58:59] == b"/"
    assert data_start == text_end + 1
    assert data_end - data_start + 1 == data.size * 4
    assert data_end == len(raw) - 1

    f = FCSFile.read(path)
    assert f.version == "FCS3.0"
    assert int(f.keywords["$BEGINDATA"]) == data_start
    assert int(f.keywords["$ENDDATA"]) == data_end
    assert f.keywords["$PAR"] == "3"
    assert f.n_events == 6
    assert f.channel_names == ["A", "B", "C"]
    assert "6 events x 3 parameters" in repr(f)


def test_parse_text_segment_basic():
    kw = parse_text_segment(b"/$PAR/2/$P1N/FSC-A/$P2N/SSC-A/")
    assert kw == {"$PAR": "2", "$P1N": "FSC-A", "$P2N": "SSC-A"}


def test_parse_text_segment_other_delimiter_and_no_trailing_delim():
    kw = parse_text_segment(b"|$PAR|1|$P1N|FSC-A")
    assert kw == {"$PAR": "1", "$P1N": "FSC-A"}


def test_parse_text_segment_escaped_delimiter():
    kw = parse_text_segment(b"/$PAR/1/COMMENT/a//b//c/$P1N/FSC-A/")
    assert kw["COMMENT"] == "a/b/c"
    assert kw["$P1N"] == "FSC-A"
    assert kw["$PAR"] == "1"


def test_parse_text_segment_empty_value_tolerated():
    # "$P1S//" is an empty value (not an escaped delimiter) since the following
    # token is a standard keyword.
    kw = parse_text_segment(b"/$P1N/FSC-A/$P1S//$PAR/1/$P2N/X/")
    assert kw["$P1S"] == ""
    assert kw["$PAR"] == "1"
    assert kw["$P1N"] == "FSC-A"
    assert kw["$P2N"] == "X"


def test_parse_text_segment_empty_raises():
    with pytest.raises(FCSError):
        parse_text_segment(b"")


def test_escaped_delimiter_round_trip_through_file(tmp_path):
    path = write_fcs(tmp_path / "e.fcs", _small(), ["A", "B", "C"], extra_keywords={"$SRC": "tube 1/2", "NOTE": "x/y/z"})
    f = FCSFile.read(path)
    assert f.keywords["$SRC"] == "tube 1/2"
    assert f.keywords["NOTE"] == "x/y/z"
    assert f.n_events == 6


# ------------------------------------------------------------------------ data
def test_float_data_little_endian(tmp_path):
    data = _small(50, 4)
    f = FCSFile.read(write_fcs(tmp_path / "f.fcs", data, list("ABCD")))
    assert f.events.dtype == np.float32
    assert f.events.shape == (50, 4)
    np.testing.assert_array_equal(f.events, data)


def test_double_data(tmp_path):
    data = np.array([[1.5, -2.25], [1e6, 3.0], [0.125, 7.0]], dtype=np.float64)
    path = write_fcs(tmp_path / "d.fcs", data, ["A", "B"], datatype="D")
    assert path.stat().st_size > 0
    f = FCSFile.read(path)
    assert f.keywords["$DATATYPE"] == "D"
    assert f.keywords["$P1B"] == "64"
    np.testing.assert_allclose(f.events, data.astype(np.float32))


def test_float_data_big_endian(tmp_path):
    data = _small(20, 3)
    path = write_fcs(tmp_path / "be.fcs", data, ["A", "B", "C"], byteord="4,3,2,1")
    raw = path.read_bytes()
    start = int(raw[26:34])
    # first value is stored big-endian
    assert np.frombuffer(raw[start : start + 4], dtype=">f4")[0] == data[0, 0]
    f = FCSFile.read(path)
    np.testing.assert_array_equal(f.events, data)


def test_integer_data_linear(tmp_path):
    data = np.array([[0, 10, 1023], [512, 7, 1], [1000, 1023, 0]], dtype=np.uint16)
    f = FCSFile.read(write_fcs(tmp_path / "i.fcs", data, ["A", "B", "C"], datatype="I"))
    assert f.events.dtype == np.float32
    np.testing.assert_array_equal(f.events, data.astype(np.float32))


def test_integer_data_big_endian(tmp_path):
    data = np.array([[1, 258], [513, 1000]], dtype=np.uint16)
    f = FCSFile.read(write_fcs(tmp_path / "ibe.fcs", data, ["A", "B"], datatype="I", byteord="2,1"))
    np.testing.assert_array_equal(f.events, data.astype(np.float32))


def test_integer_range_mask_applied(tmp_path):
    # $PnR = 1024 (power of two, smaller than 2**16): upper bits are masked off.
    data = np.array([[1024 + 5, 3]], dtype=np.uint16)
    f = FCSFile.read(write_fcs(tmp_path / "m.fcs", data, ["A", "B"], datatype="I"))
    assert f.events[0, 0] == 5
    assert f.events[0, 1] == 3


def test_integer_log_amplification(tmp_path):
    # $PnE = 4,1 over $PnR = 1024: value = 1 * 10 ** (4 * x / 1024)
    raw = np.array([[0, 0, 0], [256, 256, 256], [512, 512, 512], [768, 768, 768]], dtype=np.uint16)
    path = write_fcs(
        tmp_path / "log.fcs", raw, ["FL1", "FL2", "FL3"], datatype="I", log_amplified={0: (4, 1), 1: (4, 0), 2: (3, 2)}
    )
    f = FCSFile.read(path)
    assert f.channels[0].is_log_amplified
    assert f.channels[0].log_decades == 4 and f.channels[0].log_offset == 1
    np.testing.assert_allclose(f.events[:, 0], [1, 10, 100, 1000], rtol=1e-6)
    # an offset of 0 is treated as 1 (FCS 3.0 rule)
    np.testing.assert_allclose(f.events[:, 1], [1, 10, 100, 1000], rtol=1e-6)
    # 3 decades, offset 2
    np.testing.assert_allclose(f.events[:, 2], 2 * 10 ** (3 * np.array([0, 256, 512, 768]) / 1024), rtol=1e-6)


def test_tot_truncation(tmp_path):
    data = _small(10, 2)
    f = FCSFile.read(write_fcs(tmp_path / "t.fcs", data, ["A", "B"], extra_keywords={"$TOT": "4"}))
    assert f.n_events == 4
    np.testing.assert_array_equal(f.events, data[:4])


def test_tot_larger_than_data_is_ignored(tmp_path):
    data = _small(10, 2)
    f = FCSFile.read(write_fcs(tmp_path / "t2.fcs", data, ["A", "B"], extra_keywords={"$TOT": "99"}))
    assert f.n_events == 10


def test_unsupported_datatypes(tmp_path):
    data = _small(4, 2)
    with pytest.raises(FCSError, match="ASCII"):
        FCSFile.read(write_fcs(tmp_path / "a.fcs", data, ["A", "B"], extra_keywords={"$DATATYPE": "A"}))
    with pytest.raises(FCSError, match="unknown"):
        FCSFile.read(write_fcs(tmp_path / "x.fcs", data, ["A", "B"], extra_keywords={"$DATATYPE": "Q"}))
    ints = np.zeros((4, 2), dtype=np.uint16)
    with pytest.raises(FCSError, match="integer widths"):
        FCSFile.read(write_fcs(tmp_path / "b.fcs", ints, ["A", "B"], datatype="I", extra_keywords={"$P1B": "12"}))


# ---------------------------------------------------------------- keyword helpers
def test_channels_from_keywords_defaults_and_malformed():
    kw = {
        "$PAR": "3",
        "$P1N": "FSC-A",
        "$P1R": "262144",
        "$P1B": "32",
        "$P2N": "FITC-A",
        "$P2S": "CD3",
        "$P2R": "abc",
        "$P2B": "x",
        "$P2E": "bad",
        "$P2G": "2.5",
        "$P2D": "Logarithmic,4,0.1",
        "$P3R": "1024",
        "$P3E": "4,1",
        "$P3G": "nope",
    }
    chs = channels_from_keywords(kw)
    assert [c.index for c in chs] == [1, 2, 3]
    assert chs[0] == Channel(1, "FSC-A", "", 262144.0, 32, 0.0, 0.0, 1.0, "")
    assert chs[0].label == "FSC-A"
    assert chs[1].stain == "CD3"
    assert chs[1].label == "FITC-A :: CD3"
    assert chs[1].range == 262144.0  # malformed -> default
    assert chs[1].bits == 32
    assert chs[1].log_decades == 0.0 and not chs[1].is_log_amplified
    assert chs[1].gain == 2.5
    assert chs[1].display == "Logarithmic,4,0.1"
    assert chs[2].name == "P3"  # missing $PnN
    assert chs[2].range == 1024.0
    assert chs[2].is_log_amplified and chs[2].log_offset == 1.0
    assert chs[2].gain == 1.0


def test_channels_from_keywords_par_float_and_missing():
    assert channels_from_keywords({}) == []
    assert len(channels_from_keywords({"$PAR": "2.0"})) == 2


def test_channel_label_same_stain():
    assert Channel(1, "FSC-A", "FSC-A").label == "FSC-A"


def test_parse_spillover():
    kw = {"$SPILLOVER": "3,FITC-A,PE-A,APC-A,1,0.1,0,0.05,1,0.2,0,0.03,1"}
    names, m = parse_spillover(kw)
    assert names == ["FITC-A", "PE-A", "APC-A"]
    np.testing.assert_allclose(m, [[1, 0.1, 0], [0.05, 1, 0.2], [0, 0.03, 1]])


def test_parse_spillover_alternative_keys_and_invalid():
    assert parse_spillover({"SPILL": "1,A,1"})[0] == ["A"]
    assert parse_spillover({"$COMP": "2,A,B,1,0,0,1"})[1].shape == (2, 2)
    assert parse_spillover({}) is None
    assert parse_spillover({"$SPILLOVER": "x,A,1"}) is None
    assert parse_spillover({"$SPILLOVER": "2,A,B,1,0,0"}) is None  # too few values
    assert parse_spillover({"$SPILLOVER": "2,A,B,1,0,q,1"}) is None
    # an invalid first key falls through to a valid later one
    names, _ = parse_spillover({"$SPILLOVER": "bad", "SPILL": "1,X,1"})
    assert names == ["X"]


# -------------------------------------------------------------------- accessors
def test_column_and_channel_index_by_stain(tmp_path):
    data = _small(8, 3)
    f = FCSFile.read(write_fcs(tmp_path / "s.fcs", data, ["FSC-A", "FITC-A", "PE-A"], stains=["", "CD3", "CD4"]))
    assert f.channel_index("FITC-A") == 1
    assert f.channel_index("CD4") == 2
    np.testing.assert_array_equal(f.column("CD3"), data[:, 1])
    np.testing.assert_array_equal(f.column("PE-A"), data[:, 2])
    with pytest.raises(KeyError):
        f.channel_index("APC-A")


def test_spillover_method(tmp_path):
    path = write_fcs(tmp_path / "sp.fcs", _small(4, 2), ["A", "B"], extra_keywords={"$SPILLOVER": "2,A,B,1,0.25,0,1"})
    names, m = FCSFile.read(path).spillover()
    assert names == ["A", "B"]
    assert m[0, 1] == 0.25


def test_read_keywords_fast_path(tmp_path):
    data = _small(100, 3)
    path = write_fcs(tmp_path / "k.fcs", data, ["A", "B", "C"], stains=["", "X", ""])
    version, kw = FCSFile.read_keywords(path)
    assert version == "FCS3.0"
    assert kw["$TOT"] == "100"
    assert kw["$P2S"] == "X"
    # works even if the DATA segment is gone
    raw = path.read_bytes()
    trunc = tmp_path / "trunc.fcs"
    trunc.write_bytes(raw[: int(raw[26:34])])
    version2, kw2 = FCSFile.read_keywords(trunc)
    assert kw2 == kw


def test_read_without_data(tmp_path):
    path = write_fcs(tmp_path / "nd.fcs", _small(), ["A", "B", "C"])
    f = FCSFile.read(path, load_data=False)
    assert f.events is None
    assert f.n_events == 0
    assert len(f.channels) == 3
    with pytest.raises(FCSError):
        f.column("A")


def test_data_offsets_from_keywords_when_header_zero(tmp_path):
    data = _small(5, 2)
    path = write_fcs(tmp_path / "z.fcs", data, ["A", "B"])
    raw = bytearray(path.read_bytes())
    raw[26:42] = b"       0       0"
    path.write_bytes(bytes(raw))
    np.testing.assert_array_equal(FCSFile.read(path).events, data)


# ----------------------------------------------------------------------- errors
def test_non_fcs_input(tmp_path):
    small = tmp_path / "small.fcs"
    small.write_bytes(b"FCS3.0")
    with pytest.raises(FCSError, match="too small"):
        FCSFile.read(small)
    txt = tmp_path / "notes.txt"
    txt.write_bytes(b"hello world, this is definitely not a flow cytometry file at all" * 2)
    with pytest.raises(FCSError, match="not an FCS"):
        FCSFile.read(txt)
    with pytest.raises(FCSError):
        FCSFile.read_keywords(txt)
    bad = tmp_path / "bad.fcs"
    bad.write_bytes(b"FCS3.0    " + b"     100      50       0       0       0       0" + b"x" * 100)
    with pytest.raises(FCSError, match="TEXT"):
        FCSFile.read(bad)


def test_missing_data_segment_raises(tmp_path):
    path = write_fcs(tmp_path / "nodata.fcs", np.zeros((0, 2), dtype=np.float32), ["A", "B"])
    with pytest.raises(FCSError, match="DATA"):
        FCSFile.read(path)


def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        FCSFile.read(tmp_path / "missing.fcs")


# -------------------------------------------------------------------- real files
def test_real_compensation_control(real_comp_control):
    f = FCSFile.read(real_comp_control)
    assert f.version == "FCS3.1"
    assert f.keywords["$BYTEORD"] == "4,3,2,1"
    assert f.n_events == 5000
    assert len(f.channels) == 16
    assert f.channel_names[0] == "FSC-A"
    assert f.channel_names[-1] == "Time"
    assert "PE-A" in f.channel_names
    spill = f.spillover()
    assert spill is not None
    names, m = spill
    assert len(names) == m.shape[0] == m.shape[1]
    np.testing.assert_allclose(np.diag(m), 1.0)
    assert set(names) <= set(f.channel_names)
    # big-endian floats decoded into a sensible scatter range
    fsc = f.column("FSC-A")
    assert np.all(np.isfinite(fsc))
    assert 0 < np.median(fsc) < 262144
    v, kw = FCSFile.read_keywords(real_comp_control)
    assert v == "FCS3.1" and kw == f.keywords


def test_real_tcell_panel(real_tcell_panel):
    f = FCSFile.read(real_tcell_panel)
    assert f.version == "FCS3.0"
    assert f.n_events == 73560
    assert len(f.channels) == 31
    assert f.events.shape == (73560, 31)
    assert np.all(np.isfinite(f.events))
    stains = {c.stain for c in f.channels}
    assert "CD8_FITC-A" in stains
    idx = f.channel_index("CD8_FITC-A")
    np.testing.assert_array_equal(f.column("CD8_FITC-A"), f.events[:, idx])
