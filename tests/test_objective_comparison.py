"""Runner ordering and freeze checks; spies are not biological performance tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opal2 import objective_comparison as study


def fixture_data():
    ids = np.array([f"fixture_{i}" for i in range(639)])
    splits = {"train": np.arange(383), "validation": np.arange(383, 479),
              "calibration": np.arange(479, 543), "evaluation": np.arange(543, 639)}
    return SimpleNamespace(ids=ids, Y=SimpleNamespace(shape=(639, 4, 3617))), splits, "fixture_only"


def test_freezes_only_intended_objective_changes_and_preserves_existing(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "splits.json").write_text('{"fixture": true}')
    monkeypatch.setattr(study, "_load_declared_data", lambda path: fixture_data())
    root = tmp_path / "run"
    manifest = study.prepare(data, root)
    configs = manifest["configurations"]
    a, b, c = (configs[name] for name in study.ARMS)
    assert {k for k in a if a[k] != b[k]} == {"objective"}
    assert {k for k in b if b[k] != c[k]} == {"utility_crps_weight"}
    assert all(v["paired_objective_rng"] for v in configs.values())
    assert a["epochs"] == 100 and a["jepa_epochs"] == 60 and a["hidden_dim"] == 256
    assert (root / "source_snapshot" / "opal2" / "model.py").is_file()
    assert (root / "PROTOCOL.md").is_file()
    assert manifest["evaluation_after_all_fitting"]
    with pytest.raises(FileExistsError):
        study.prepare(data, root)
    with pytest.raises(ValueError, match="source_snapshot"):
        study.execute(root)


def test_all_fitting_precedes_any_evaluation(tmp_path, monkeypatch):
    import opal2.training as training
    import opal2.evaluation as evaluation
    import opal2.objective_analysis as analysis
    ds, splits, scope = fixture_data()
    data = tmp_path / "data"
    data.mkdir()
    (data / "splits.json").write_text('{"fixture": true}')
    monkeypatch.setattr(study, "_load_declared_data", lambda path: (ds, splits, scope))
    root = tmp_path / "run"
    manifest = study.prepare(data, root)
    # Allow the instrumented current module to exercise the frozen-run logic.
    manifest["source_snapshot"] = str(Path(study.__file__).resolve().parents[1])
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    events = []
    configs = study.configurations()

    def fit(dataset, actual_splits, config, directory, **kwargs):
        assert dataset is ds
        assert actual_splits is splits
        directory.mkdir()
        import torch
        torch.save({"fixture_weight": torch.tensor([1.0])}, directory / "initial_state.pt")
        if directory.name == "A_ELBO":
            assert kwargs["shared_jepa_directory"] is None
        else:
            assert kwargs["shared_jepa_directory"] == root / "arms" / "A_ELBO"
        events.append(("fit", directory.name))
        return None, None

    def load(directory):
        return None, None, configs[directory.name], {"epoch": 7, "best_validation": 1.2}

    def evaluate(model, scaler, dataset, actual_splits, config, directory):
        assert len([x for x in events if x[0] == "fit"]) == 3
        events.append(("evaluate", directory.name))

    monkeypatch.setattr(training, "fit_model", fit)
    monkeypatch.setattr(training, "load_model", load)
    monkeypatch.setattr(training, "seed_everything", lambda *args: None)
    monkeypatch.setattr(evaluation, "evaluate_model", evaluate)
    monkeypatch.setattr(analysis, "generate_analysis", lambda root: events.append(("analysis", None)))
    study.execute(root)
    assert events == [("fit", n) for n in study.ARMS] + [("evaluate", n) for n in study.ARMS] + [("analysis", None)]
    assert json.loads((root / "status.json").read_text())["state"] == "COMPLETE"
