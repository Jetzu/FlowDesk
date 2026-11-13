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


# ------------------------------------------------------ data sets, encodings, fixes
import shutil  # noqa: E402

from flowdesk.core import fcs as fcs_module  # noqa: E402
from flowdesk.core.fcs import format_spillover, looks_like_fcs, set_memmap, write_fcs as write_fcs31  # noqa: E402


def _two_datasets(tmp_path, name="multi.fcs", first_version=b"FCS3.0"):
    """A file holding two data sets: the first one points to the second
    through $NEXTDATA (relative to its own start)."""
    a = _small(5, 2, seed=1)
    b = _small(7, 3, seed=2)
    p1 = tmp_path / "ds1.fcs"
    write_fcs(p1, a, ["A1", "A2"], extra_keywords={"$NEXTDATA": f"{0:020d}"})
    size = p1.stat().st_size
    write_fcs(p1, a, ["A1", "A2"], extra_keywords={"$NEXTDATA": f"{size:020d}"})
    assert p1.stat().st_size == size
    p2 = write_fcs(tmp_path / "ds2.fcs", b, ["B1", "B2", "B3"])
    first = bytearray(p1.read_bytes())
    first[:6] = first_version
    out = tmp_path / name
    out.write_bytes(bytes(first) + p2.read_bytes())
    return out, a, b, size


def test_list_datasets_and_read_second(tmp_path):
    path, a, b, size = _two_datasets(tmp_path)
    assert FCSFile.list_datasets(path) == [0, size]
    first = FCSFile.read(path)
    np.testing.assert_array_equal(first.events, a)
    assert first.dataset == 0
    second = FCSFile.read(path, dataset=1)
    assert second.channel_names == ["B1", "B2", "B3"]
    np.testing.assert_array_equal(second.events, b)
    assert second.dataset == 1 and second.offset == size
    v, kw = FCSFile.read_keywords(path, dataset=1)
    assert kw["$PAR"] == "3"
    assert [d[0] for d in FCSFile.datasets(path)] == [0, size]
    with pytest.raises(FCSError, match="data set"):
        FCSFile.read(path, dataset=2)
    # a single-data-set file lists one offset
    assert FCSFile.list_datasets(write_fcs(tmp_path / "one.fcs", a, ["A1", "A2"])) == [0]


def test_lmd_reads_its_fcs3_data_set(tmp_path):
    from flowdesk.core.model import Workspace

    path, a, b, _ = _two_datasets(tmp_path, "beckman.LMD", first_version=b"FCS2.0")
    assert FCSFile.default_dataset(path) == 1
    f = FCSFile.read(path)
    assert f.version == "FCS3.0" and f.channel_names == ["B1", "B2", "B3"]
    np.testing.assert_array_equal(f.events, b)
    assert FCSFile.read_keywords(path)[1]["$PAR"] == "3"
    # the legacy FCS 2.0 copy stays reachable
    assert FCSFile.read(path, dataset=0).channel_names == ["A1", "A2"]
    s = Workspace().add_sample(path)
    assert s.load_error is None and s.n_events == 7 and s.channel_names == ["B1", "B2", "B3"]
    assert s.ensure_loaded() and s.data("B2").shape == (7,)


def test_lmd_added_through_add_samples(tmp_path):
    from flowdesk.ui import controller as controller_module

    if ".lmd" not in tuple(getattr(controller_module, "SAMPLE_SUFFIXES", ())):
        pytest.skip("controller.SAMPLE_SUFFIXES does not list .lmd (owned by the application layer)")
    path, _, b, _ = _two_datasets(tmp_path, "beckman.lmd", first_version=b"FCS2.0")
    c = controller_module.WorkspaceController()
    added = c.add_samples([path])
    assert len(added) == 1 and added[0].channel_names == ["B1", "B2", "B3"]


def test_integer_range_mask_non_power_of_two(tmp_path):
    # $PnR = 1000 -> keep the low ceil(log2(1000)) = 10 bits (flag bits above are dropped)
    data = np.array([[1024 + 5, 2048 + 999, 7]], dtype=np.uint16)
    f = FCSFile.read(write_fcs(tmp_path / "m.fcs", data, ["A", "B", "C"], datatype="I", ranges=[1000, 1000, 65536]))
    np.testing.assert_array_equal(f.events[0], [5, 999, 7])


def test_integer_gain_applied_to_linear_channels_only(tmp_path):
    data = np.array([[400, 400, 512]], dtype=np.uint16)
    path = write_fcs(
        tmp_path / "g.fcs", data, ["A", "B", "L"], datatype="I", ranges=[1024, 1024, 1024],
        log_amplified={2: (4, 1)}, extra_keywords={"$P1G": "4", "$P3G": "8"},
    )
    f = FCSFile.read(path)
    assert f.events[0, 0] == pytest.approx(100.0)
    assert f.events[0, 1] == pytest.approx(400.0)
    assert f.events[0, 2] == pytest.approx(100.0)  # log-amplified: gain ignored
    # float data are already scaled: $PnG is informational
    ff = FCSFile.read(write_fcs(tmp_path / "gf.fcs", np.array([[400.0]], dtype=np.float32), ["A"], extra_keywords={"$P1G": "4"}))
    assert ff.events[0, 0] == pytest.approx(400.0)


def test_latin1_keyword_values_have_no_replacement_character(tmp_path):
    kw = parse_text_segment("/$P1S/CD3 µm °C ü α?/LASER/Röhre/".encode("cp1252", "replace"))
    assert "�" not in "".join(kw.values())
    assert kw["$P1S"].startswith("CD3 µm °C ü")
    assert kw["LASER"] == "Röhre"
    # UTF-8 (FCS 3.1) still decodes as UTF-8
    assert parse_text_segment("/$P1S/CD3 µm/".encode("utf-8"))["$P1S"] == "CD3 µm"
    # a whole file whose TEXT is Latin-1
    path = write_fcs(tmp_path / "l1.fcs", _small(3, 2), ["FL1-H", "FL2-H"], stains=["IgG µ", "CD4"])
    raw = path.read_bytes().replace("IgG µ".encode("utf-8"), "IgG µ ".encode("latin-1"))
    path.write_bytes(raw)
    f = FCSFile.read(path)
    assert f.channels[0].stain == "IgG µ" and "�" not in f.channels[0].label


def test_lower_case_standard_keywords(tmp_path):
    data = _small(4, 2)
    path = write_fcs(tmp_path / "lc.fcs", data, ["A", "B"])
    raw = path.read_bytes()
    for key in (b"$PAR", b"$DATATYPE", b"$TOT", b"$P1N", b"$P2N", b"$BYTEORD", b"$BEGINDATA"):
        raw = raw.replace(b"/" + key + b"/", b"/" + key.lower() + b"/")
    path.write_bytes(raw)
    f = FCSFile.read(path)
    assert f.channel_names == ["A", "B"]
    np.testing.assert_array_equal(f.events, data)
    assert "$PAR" in f.keywords and "$par" not in f.keywords


def test_comp_keyword_standard_form():
    kw = {"$COMP": "2,1,0.1,0.2,1", "$P1N": "FITC-A", "$P2N": "PE-A", "$PAR": "2"}
    names, m = parse_spillover(kw)
    assert names == ["FITC-A", "PE-A"]
    np.testing.assert_allclose(m, [[1, 0.1], [0.2, 1]])
    assert parse_spillover({"$COMP": "2,1,0.1,0.2,1"}) is None  # no $PnN to refer to
    text = format_spillover(["A", "B"], np.array([[1, 0.25], [0, 1]]))
    assert text == "2,A,B,1,0.25,0,1"
    assert parse_spillover({"$SPILLOVER": text})[0] == ["A", "B"]


def test_text_data_offsets_win_over_a_wrong_header(tmp_path):
    data = _small(6, 2)
    path = write_fcs(tmp_path / "h.fcs", data, ["A", "B"])
    raw = bytearray(path.read_bytes())
    start = int(raw[26:34])
    raw[34:42] = f"{start + 2 * 2 * 4 - 1:8d}".encode()  # header claims 2 events
    path.write_bytes(bytes(raw))
    f = FCSFile.read(path)
    assert f.n_events == 6
    np.testing.assert_array_equal(f.events, data)


def test_truncated_file_raises_and_sets_load_error(tmp_path):
    from flowdesk.core.model import Workspace

    data = _small(1000, 3)
    path = write_fcs(tmp_path / "cut.fcs", data, ["A", "B", "C"])
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - 5000])
    with pytest.raises(FCSError, match="truncated"):
        FCSFile.read(path)
    s = Workspace().add_sample(path)
    assert s.load_error is None and s.n_events == 1000  # the header is intact
    assert not s.ensure_loaded()
    assert "truncated" in s.load_error


def test_real_file_truncated_at_200kb(tmp_path, real_comp_control):
    from flowdesk.core.model import Workspace

    cut = tmp_path / "cut.fcs"
    cut.write_bytes(real_comp_control.read_bytes()[:200_000])
    s = Workspace().add_sample(cut)
    assert not s.ensure_loaded()
    assert s.load_error and "truncated" in s.load_error


def test_tot_mismatch_is_reported_as_warning(tmp_path):
    f = FCSFile.read(write_fcs(tmp_path / "w.fcs", _small(10, 2), ["A", "B"], extra_keywords={"$TOT": "99"}))
    assert f.n_events == 10
    assert f.warnings and "$TOT" in f.warnings[0]


def _raw_fcs(path, keywords: dict, payload: bytes, version=b"FCS3.2"):
    kw = dict(keywords)
    kw.setdefault("$NEXTDATA", "0")

    def build(begin, end):
        k = dict(kw, **{"$BEGINDATA": f"{begin:020d}", "$ENDDATA": f"{end:020d}"})
        return ("/" + "".join(f"{a}/{b}/" for a, b in k.items())).encode("utf-8")

    probe = build(0, 0)
    begin = 58 + len(probe)
    text = build(begin, begin + len(payload) - 1)
    header = version + b"    " + b"".join(f"{v:8d}".encode() for v in (58, 58 + len(text) - 1, begin, begin + len(payload) - 1, 0, 0))
    path.write_bytes(header + text + payload)
    return path


def test_fcs32_per_parameter_datatypes(tmp_path):
    n = 4
    ints = np.array([10, 20, 30, 4096 + 40], dtype="<u2")
    floats = np.array([1.5, -2.5, 3.25, 1e6], dtype="<f4")
    doubles = np.array([0.125, 2.0, -7.0, 9.5], dtype="<f8")
    rec = np.zeros(n, dtype=[("a", "<u2"), ("b", "<f4"), ("c", "<f8")])
    rec["a"], rec["b"], rec["c"] = ints, floats, doubles
    kw = {
        "$BYTEORD": "1,2,3,4", "$DATATYPE": "F", "$MODE": "L", "$PAR": "3", "$TOT": str(n),
        "$P1N": "I16", "$P1B": "16", "$P1R": "4096", "$P1E": "0,0", "$P1DATATYPE": "I",
        "$P2N": "F32", "$P2B": "32", "$P2R": "262144", "$P2E": "0,0",
        "$P3N": "D64", "$P3B": "64", "$P3R": "262144", "$P3E": "0,0", "$P3DATATYPE": "D",
    }
    f = FCSFile.read(_raw_fcs(tmp_path / "mixed.fcs", kw, rec.tobytes()))
    np.testing.assert_array_equal(f.events[:, 0], [10, 20, 30, 40])  # $P1R mask applies
    np.testing.assert_allclose(f.events[:, 1], floats)
    np.testing.assert_allclose(f.events[:, 2], doubles)


@pytest.fixture
def memmap_always():
    before = fcs_module.memmap_settings()
    set_memmap(True, 0)
    yield
    set_memmap(*before)


def _is_mapped(arr) -> bool:
    base = arr
    while base is not None:
        if isinstance(base, np.memmap):
            return True
        base = getattr(base, "base", None)
    return False


def test_memmap_reading_matches_plain_reading(tmp_path, memmap_always):
    data = _small(500, 4)
    little = write_fcs(tmp_path / "le.fcs", data, ["A", "B", "C", "D"])
    f = FCSFile.read(little)
    np.testing.assert_array_equal(f.events, data)
    aligned = int(little.read_bytes()[26:34]) % 4 == 0
    assert _is_mapped(f.events) == aligned
    if aligned:
        f.events[0, 0] = -1.0  # copy-on-write: the file is not changed
        assert FCSFile.read(little).events[0, 0] == data[0, 0]
    # converted layouts are read through a map in chunks
    big = FCSFile.read(write_fcs(tmp_path / "be.fcs", data, ["A", "B", "C", "D"], byteord="4,3,2,1"))
    np.testing.assert_array_equal(big.events, data)
    dbl = FCSFile.read(write_fcs(tmp_path / "d.fcs", data.astype(np.float64), ["A", "B", "C", "D"], datatype="D"))
    np.testing.assert_allclose(dbl.events, data)
    ints = np.array([[1, 2], [1024 + 3, 4]], dtype=np.uint16)
    fi = FCSFile.read(write_fcs(tmp_path / "i.fcs", ints, ["A", "B"], datatype="I"))
    np.testing.assert_array_equal(fi.events, [[1, 2], [3, 4]])
    set_memmap(False)
    plain = FCSFile.read(little)
    assert not _is_mapped(plain.events)
    np.testing.assert_array_equal(plain.events, data)


def test_memmap_threshold(tmp_path):
    before = fcs_module.memmap_settings()
    try:
        set_memmap(True, 10**12)
        f = FCSFile.read(write_fcs(tmp_path / "s.fcs", _small(50, 2), ["A", "B"]))
        assert not _is_mapped(f.events)
    finally:
        set_memmap(*before)


def test_looks_like_fcs_without_suffix(tmp_path):
    p = write_fcs(tmp_path / "exported", _small(3, 2), ["A", "B"])
    assert looks_like_fcs(p)
    (tmp_path / "note").write_bytes(b"hello")
    assert not looks_like_fcs(tmp_path / "note")
    assert not looks_like_fcs(tmp_path / "missing")
    assert FCSFile.read(p).n_events == 3


# ------------------------------------------------------------ keyword-faithful writer
DIVA_KEYWORDS = {
    "$CYT": "FACSAriaIII", "$DATE": "30-APR-2026", "$BTIM": "14:41:19", "$SYS": "Windows 10 10.0", "$INST": " ",  # Diva writes a space
    "LASER1NAME": "355 UV", "LASER1WAVELENGTH": "355", "$TIMESTEP": "0.01", "CREATOR": "BD FACSDiva Software Version 9.7",
    "$SPILLOVER": "2,FITC-A,PE-A,1,0.1,0.02,1", "THRESHOLD": "SSC,1000",
    "$P1V": "220", "$P1L": "488", "P1DISPLAY": "LIN", "$P1G": "1.0",
    "$P2V": "300", "$P2F": "530/30", "$P2L": "488", "P2DISPLAY": "LOG", "P2BS": "12",
    "$P3V": "520", "$P3F": "582/15", "$P3L": "561", "$P3D": "Logarithmic,4,1", "P3DISPLAY": "LOG", "$P3CALIBRATION": "1.5,MESF",
    "$P4G": "0.01",
}


def _diva_file(tmp_path):
    data = np.abs(_small(20, 4)) + 1
    names = ["FSC-A", "FITC-A", "PE-A", "Time"]
    return write_fcs(tmp_path / "diva.fcs", data, names, stains=["", "CD3", "CD4", ""], extra_keywords=DIVA_KEYWORDS), data


def test_write_fcs_keeps_and_reindexes_source_keywords(tmp_path):
    src, data = _diva_file(tmp_path)
    f = FCSFile.read(src)
    # drop FITC-A, keep FSC-A / PE-A / Time and add a SampleID column
    keep = [0, 2, 3]
    chans = [f.channels[i] for i in keep] + [Channel(5, "SampleID", "SampleID", 3)]
    events = np.column_stack([f.events[:, keep], np.ones(20, dtype=np.float32)])
    out = tmp_path / "out.fcs"
    write_fcs31(out, events, chans, f.keywords, {"FLOWDESK_TEST": "1"})
    g = FCSFile.read(out)
    kw = g.keywords
    assert g.channel_names == ["FSC-A", "PE-A", "Time", "SampleID"]
    # $PnV / $PnF / $PnL / $PnD and vendor PnDISPLAY follow their channel
    assert kw["$P1V"] == "220" and kw["$P1L"] == "488" and kw["P1DISPLAY"] == "LIN"
    assert kw["$P2V"] == "520" and kw["$P2F"] == "582/15" and kw["$P2L"] == "561"
    assert kw["$P2D"] == "Logarithmic,4,1" and kw["P2DISPLAY"] == "LOG" and kw["$P2CALIBRATION"] == "1.5,MESF"
    assert kw["$P3G"] == "0.01" and kw["$P1G"] == "1"
    assert "$P4V" not in kw and "$P4F" not in kw and "P4DISPLAY" not in kw  # SampleID has none
    assert not any(v == "530/30" for v in kw.values())  # the dropped channel's keywords are gone
    assert "P2BS" not in kw
    # descriptive / vendor keywords are kept, empty values survive as empty
    assert kw["$SYS"] == "Windows 10 10.0" and kw["$CYT"] == "FACSAriaIII" and kw["$INST"] == ""
    assert kw["LASER1NAME"] == "355 UV" and kw["LASER1WAVELENGTH"] == "355"
    assert kw["$TIMESTEP"] == "0.01" and kw["THRESHOLD"] == "SSC,1000"
    assert parse_spillover(kw)[0] == ["FITC-A", "PE-A"]
    assert kw["FLOWDESK_TEST"] == "1"
    raw = out.read_bytes()
    assert b"$INST" + raw[58:59] + b" " + raw[58:59] in raw  # a single space, never an empty value


def test_write_fcs_drops_source_matrix_next_to_comp_columns(tmp_path):
    src, data = _diva_file(tmp_path)
    f = FCSFile.read(src)
    chans = list(f.channels) + [Channel(5, "Comp-FITC-A", "CD3", 262144), Channel(6, "Comp-PE-A", "CD4", 262144)]
    events = np.column_stack([f.events, f.events[:, 1:3]])
    out = tmp_path / "comp.fcs"
    write_fcs31(out, events, chans, f.keywords)
    kw = FCSFile.read(out).keywords
    assert parse_spillover(kw) is None
    assert kw["FLOWDESK_COMPENSATED"] == "TRUE"
    assert kw["$P5V"] == "300" and kw["$P6F"] == "582/15"  # Comp-X carries X's detector keywords


def test_real_diva_file_round_trip_keeps_keywords(tmp_path, real_comp_control):
    from flowdesk.core.model import Workspace

    ws = Workspace()
    s = ws.add_sample(real_comp_control)
    out = tmp_path / "diva_export.fcs"
    s.export_population_fcs(s.root, out)
    v, kw = FCSFile.read_keywords(out)
    src = s.keywords
    assert kw["$TIMESTEP"] == src["$TIMESTEP"]
    assert parse_spillover(kw)[0] == parse_spillover(src)[0]
    for key in ("LASER1NAME", "LASER5WAVELENGTH", "CST SETUP STATUS", "$SYS", "$INST", "$P13V", "$P13F", "$P13L", "P13DISPLAY"):
        assert kw[key] == src[key], key
    re_read = ws.add_sample(out)
    assert re_read.comp is not None and re_read.ensure_loaded()
    assert re_read.n_events == s.n_events
