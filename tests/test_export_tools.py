"""The Export / Concatenate dialogs (flowdesk.ui.export_tools) and their hand-off from the main window."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")

from flowdesk.core.fcs import Channel, FCSFile, write_fcs  # noqa: E402
from flowdesk.core.gates import RectangleGate  # noqa: E402
from flowdesk.core.model import Population, Workspace  # noqa: E402

BIG = RectangleGate(["FSC-A", "SSC-A"], [45000, 0], [None, None])
BRIGHT = RectangleGate(["FITC-A"], [400], [None])


@pytest.fixture(scope="module")
def app():
    from flowdesk.app import create_app

    return create_app([])


def _make_file(path: Path, n: int, seed: int) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(seed)
    events = np.column_stack([rng.normal(50000, 10000, n), rng.normal(30000, 8000, n), rng.lognormal(6, 1.2, n)]).astype(np.float32)
    channels = [Channel(1, "FSC-A", "", 262144), Channel(2, "SSC-A", "", 262144), Channel(3, "FITC-A", "CD4", 262144)]
    write_fcs(path, events, channels, {"$CYT": "TestCytometer"})
    return path, events


@pytest.fixture
def setup(app, tmp_path):
    """Three samples with a "Big" gate on all of them and "Big/Bright" on the first two only."""
    from flowdesk.ui.controller import WorkspaceController

    ws = Workspace()
    raw = {}
    samples = []
    for k, n in enumerate((400, 300, 350)):
        path, events = _make_file(tmp_path / f"s{k + 1}.fcs", n, 10 + k)
        samples.append(ws.add_sample(path))
        raw[samples[-1].name] = events
    ws.apply_population_to_group(ws.groups[0], "", Population("Big", BIG))
    for s in samples[:2]:
        s.root.find("/Big").add_child(Population("Bright", BRIGHT))
    controller = WorkspaceController(ws)
    out = tmp_path / "out"
    out.mkdir()
    return controller, samples, raw, out


def _expected_big(events: np.ndarray) -> int:
    return int(((events[:, 0] >= 45000)).sum())


def _dialog(controller, rows, mode="fcs", **kwargs):
    from flowdesk.ui.export_tools import ExportDialog

    dlg = ExportDialog(controller, rows, mode=mode, **kwargs)
    dlg.add_check.setChecked(False)  # do not add the written files to the workspace
    return dlg


def test_fcs_export_writes_the_gated_events(setup):
    controller, samples, raw, out = setup
    s1 = samples[0]
    pop = s1.root.find("/Big")
    dlg = _dialog(controller, [(s1, pop)])
    dlg.path_edit.setText(str(out))
    assert dlg.validate() is None
    written = dlg.run()
    assert len(written) == 1 and not dlg.errors
    f = FCSFile.read(written[0])
    assert f.n_events == _expected_big(raw[s1.name]) == pop.count
    assert f.channel_names == ["FSC-A", "SSC-A", "FITC-A"]
    assert np.allclose(f.events[:, 0], raw[s1.name][raw[s1.name][:, 0] >= 45000][:, 0])


def test_selected_parameters_and_added_keywords(setup):
    controller, samples, raw, out = setup
    s1 = samples[0]
    dlg = _dialog(controller, [(s1, s1.root.find("/Big"))])
    dlg.path_edit.setText(str(out))
    dlg.param_radios["selected"].setChecked(True)
    dlg.set_checked_parameters(["FSC-A", "FITC-A"])
    dlg.add_keyword_row("TREATMENT", "drug A")
    assert dlg.chosen_parameters() == ["FSC-A", "FITC-A"]
    assert dlg.keywords_to_add() == {"TREATMENT": "drug A"}
    f = FCSFile.read(dlg.run()[0])
    assert f.channel_names == ["FSC-A", "FITC-A"]
    assert f.keywords["TREATMENT"] == "drug A"


def test_validation_messages(setup):
    controller, samples, _raw, out = setup
    s1 = samples[0]
    dlg = _dialog(controller, [(s1, s1.root.find("/Big"))])
    dlg.path_edit.setText("")  # the dialog pre-fills the last folder; clear it
    assert "where to write" in dlg.validate()
    dlg.path_edit.setText(str(out))
    dlg.param_radios["selected"].setChecked(True)
    dlg.set_checked_parameters([])
    assert "at least one parameter" in dlg.validate()
    empty = _dialog(controller, [])
    empty.path_edit.setText(str(out))
    assert "nothing to export" in empty.validate().lower()


def test_csv_export_with_a_random_percentage(setup):
    controller, samples, raw, out = setup
    s1 = samples[0]
    pop = s1.root.find("/Big")
    dlg = _dialog(controller, [(s1, pop)], mode="csv")
    dlg.path_edit.setText(str(out))
    assert dlg.fmt() == "csv"
    dlg.percent_radio.setChecked(True)
    dlg.percent_spin.setValue(50)
    dlg.seed_spin.setValue(3)
    written = dlg.run()
    assert len(written) == 1 and written[0].suffix == ".csv"
    lines = written[0].read_text().splitlines()
    header, data = lines[0], lines[1:]
    assert header.split(",")[:3] == ["FSC-A", "SSC-A", "FITC-A"] or "FSC-A" in header
    assert abs(len(data) - pop.count / 2) <= 1
    # the same seed gives the same events; a different seed does not
    again = _dialog(controller, [(s1, pop)], mode="csv")
    again.path_edit.setText(str(out / "again"))
    (out / "again").mkdir()
    again.percent_radio.setChecked(True)
    again.percent_spin.setValue(50)
    again.seed_spin.setValue(3)
    assert again.run()[0].read_text().splitlines()[1:] == data


def test_concatenate_writes_one_file_with_a_sample_id_column(setup):
    controller, samples, raw, out = setup
    rows = [(s, s.root.find("/Big")) for s in samples[:2]]
    dlg = _dialog(controller, rows, mode="concatenate")
    dlg.path_edit.setText(str(out / "all.fcs"))
    dlg.sample_id_check.setChecked(True)
    written = dlg.run()
    assert written == [out / "all.fcs"] and not dlg.errors
    expected = sum(_expected_big(raw[s.name]) for s in samples[:2])
    assert dlg.events_written == expected
    f = FCSFile.read(written[0])
    assert f.n_events == expected
    assert len(f.channel_names) == 4  # the three parameters plus the sample id
    sid = f.events[:, 3]
    assert sorted(set(np.round(sid).astype(int))) == [1, 2]
    assert int((np.round(sid) == 1).sum()) == _expected_big(raw[samples[0].name])


def test_batch_export_over_a_group_reports_the_samples_without_the_population(setup):
    controller, samples, raw, out = setup
    s1 = samples[0]
    dlg = _dialog(controller, [(s1, s1.root.find("/Big/Bright"))])
    dlg.path_edit.setText(str(out))
    dlg.batch_check.setChecked(True)
    rows = dlg.target_rows()
    assert {s.name for s, _ in rows} == {samples[0].name, samples[1].name}
    written = dlg.run()
    assert len(written) == 2
    assert any(samples[2].name in e for e in dlg.errors)  # the third sample has no Bright gate
    for path, s in zip(sorted(written), samples[:2]):
        assert FCSFile.read(path).n_events == s.root.find("/Big/Bright").count


def test_main_window_hands_export_commands_to_the_extension(app, monkeypatch):
    from flowdesk.ui.main_window import MainWindow

    win = MainWindow()
    try:
        ext = win.extensions["flowdesk.ui.export_tools"]
        called = []
        for name in ("export_fcs", "export_events", "concatenate_samples"):
            monkeypatch.setattr(ext, name, lambda n=name: called.append(n))
        win.export_fcs()
        win.export_events()
        win.concatenate_samples()
        assert called == ["export_fcs", "export_events", "concatenate_samples"]
    finally:
        win.controller.workspace.modified = False
        win.close()
