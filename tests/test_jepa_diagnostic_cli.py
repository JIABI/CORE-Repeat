import json

from opal2.jepa_diagnostic import inspect, _write_json


def test_inspection_reads_only_logs_and_ignores_partial_append(tmp_path):
    _write_json(tmp_path / "status.json", {"state": "DIAGNOSING", "epoch": 5})
    arm = tmp_path / "arms" / "R1"
    arm.mkdir(parents=True)
    (arm / "training.jsonl").write_text(
        '{"epoch":1}\n{"epoch":2}\n{"epoch":3}\n{"epoch":'
    )
    snapshot = tmp_path / "source_snapshot"
    snapshot.mkdir()
    (snapshot / "not_a_run.jsonl").write_text('{"forbidden":true}\n')
    # These are deliberately unreadable as archives/checkpoints. Inspect is
    # a progress operation and must not open either kind of experimental input.
    (tmp_path / "measurements.npz").write_text("not an archive")
    (arm / "last.pt").write_text("not a checkpoint")
    report = inspect(tmp_path)
    assert report["status"]["epoch"] == 5
    assert report["logs"] == {"arms/R1/training.jsonl": [{"epoch": 2}, {"epoch": 3}]}


def test_status_write_is_complete_and_temporary_file_removed(tmp_path):
    path = tmp_path / "status.json"
    _write_json(path, {"state": "PREPARED"})
    _write_json(path, {"state": "TRAINING", "epoch": 1})
    assert json.loads(path.read_text()) == {"state": "TRAINING", "epoch": 1}
    assert not (tmp_path / "status.json.tmp").exists()
