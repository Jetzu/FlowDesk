"""FCS writing: population export and sample concatenation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flowdesk.core.fcs import Channel, FCSFile, write_fcs
from flowdesk.core.gates import RectangleGate
from flowdesk.core.model import Population, Workspace, concatenate_samples


def _make_file(path: Path, n: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    events = np.column_stack([rng.normal(50000, 10000, n), rng.normal(30000, 8000, n), rng.lognormal(6, 1.2, n)]).astype(np.float32)
    channels = [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144), Channel(3, "FITC-A", "CD4", 262144)]
    write_fcs(path, events, channels, {"$CYT": "TestCytometer", "VENDOR_KEY": "a/b"})
    return path


def test_write_and_read_back(tmp_path):
    path = _make_file(tmp_path / "a.fcs", 500, 1)
    f = FCSFile.read(path)
    assert f.version.startswith("FCS3.1")
    assert f.n_events == 500
    assert f.channel_names == ["FSC-A", "SSC-A", "FITC-A"]
    assert f.channels[2].stain == "CD4"
    assert f.keywords["$CYT"] == "TestCytometer"
    assert f.keywords["VENDOR_KEY"] == "a/b"  # escaped delimiter survives
    assert f.keywords["$FIL"] == "a.fcs"
    assert np.isfinite(f.events).all()


def test_population_export_and_concatenate(tmp_path):
    ws = Workspace()
    s1 = ws.add_sample(_make_file(tmp_path / "s1.fcs", 400, 2))
    s2 = ws.add_sample(_make_file(tmp_path / "s2.fcs", 300, 3))
    gate = RectangleGate(["FSC-A", "SSC-A"], [45000, 0], [None, None])
    ws.apply_population_to_group(ws.groups[0], "", Population("Big", gate))
    pop1 = s1.root.find("/Big")
    out = tmp_path / "big.fcs"
    s1.export_population_fcs(pop1, out)
    f = FCSFile.read(out)
    assert f.n_events == pop1.count
    assert (f.events[:, 0] >= 45000).all()
    assert f.keywords["FLOWDESK_POPULATION"] == "/Big"
    cat = tmp_path / "cat.fcs"
    n = concatenate_samples([s1, s2], [s1.root.find("/Big"), s2.root.find("/Big")], cat)
    f = FCSFile.read(cat)
    assert f.n_events == n == pop1.count + s2.root.find("/Big").count
    assert f.channel_names[-1] == "SampleID"
    ids = f.column("SampleID")
    assert set(np.unique(ids).tolist()) == {1.0, 2.0}
    limited = concatenate_samples([s1, s2], None, tmp_path / "lim.fcs", max_events_per_sample=100)
    assert limited == 200
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.fcs"
        write_fcs(bad, np.zeros((10, 2), dtype=np.float32), [Channel(1, "X", "", 1024), Channel(2, "Y", "", 1024)])
        s3 = ws.add_sample(bad)
        concatenate_samples([s1, s3], None, tmp_path / "mixed.fcs")
