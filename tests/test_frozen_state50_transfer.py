"""Frozen old-cohort replay; these tests never read the new LKCP population."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from opal2.frozen_state50_transfer import DEFAULT_STATE_RUN, FrozenState50Transfer
from opal2.lincs_biology_experiment import load_data


@pytest.fixture(scope="module")
def source():
    if not (DEFAULT_STATE_RUN/"run_manifest.json").exists():
        pytest.skip("Local saved STATE50 artifact is not present")
    manifest = json.loads((DEFAULT_STATE_RUN/"run_manifest.json").read_text())
    data, metadata = load_data(manifest["data_directory"])
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    loader = FrozenState50Transfer.load()
    rows = np.asarray(manifest["folds"][0]["test"], int)
    yield loader, data, metadata, rows
    torch.set_num_threads(old)


def test_complete_epoch50_mean_scatter_and_target_replay(source):
    loader, data, _, rows = source
    actual = loader.calibration_records(data["Y"][rows], data["chem"][rows], data["chem_mask"][rows],
        feature_names=data["feature_names"], ids=data["ids"][rows], groups=data["groups"][rows])
    with np.load(DEFAULT_STATE_RUN/"folds/fold_0/arms/STATE50/evaluation/u_predictions.npz") as saved:
        assert np.array_equal(actual["ids"], saved["ids"])
        assert np.array_equal(actual["mean_u"], saved["mean_u"])
        assert np.array_equal(actual["base_scatter_u"], saved["covariance_u"])
        assert np.array_equal(actual["actual_u"], saved["actual_u"])
    assert all(not p.requires_grad for p in loader.model.parameters())
    assert not any(module.training for module in loader.model.modules())
    assert loader.provenance["complete_frozen_A_verified"]
    assert loader.provenance["actual_epoch"] == 50


def test_unknown_biology_matches_inactive_saved_annotations(source):
    loader, data, _, rows = source
    rows = rows[:9]
    kwargs = dict(X=data["Y"][rows, 0], chem=data["chem"][rows], chem_mask=data["chem_mask"][rows])
    bio = {key: data[key][rows] for key in ("target", "moa", "target_mask", "moa_mask")}
    assert np.array_equal(loader.predict(**kwargs)["mean_u"], loader.predict(**kwargs, biology=bio)["mean_u"])
    unknown = loader.unknown_biology(len(rows))
    assert unknown["target"].shape[1] == len(loader.target_names)
    assert not unknown["target_mask"].any() and not unknown["moa_mask"].any()


def test_input_and_annotation_schema_checks(source):
    loader, data, _, rows = source
    X, chem = data["Y"][rows[:2], 0], data["chem"][rows[:2]]
    with pytest.raises(ValueError, match="names/order"):
        loader.predict(X, chem, feature_names=data["feature_names"][::-1])
    with pytest.raises(ValueError, match="boolean"):
        loader.predict(X, chem, np.ones(2))
    with pytest.raises(ValueError, match="schema"):
        loader.predict(X, chem[:, :-1])
    with pytest.raises(ValueError, match="X must"):
        loader.predict(data["Y"][rows[:2]], chem)
    with pytest.raises(TypeError):
        loader.predict(X, chem, future_Y=data["Y"][rows[:2], 1:])
    result = loader.predict(X, np.full_like(chem, np.nan), np.zeros(2, bool))
    assert np.isfinite(result["mean_u"]).all()


def test_loading_preserves_rng_and_reports_source_overlap(source):
    loader, data, _, rows = source
    before = torch.get_rng_state().clone()
    FrozenState50Transfer.load()
    assert torch.equal(before, torch.get_rng_state())
    report = loader.overlap_report(ids=data["ids"][rows], groups=data["groups"][rows])
    assert report["roles"]["original_source_cohort"]["groups_overlap_count"] == len(rows)
    assert report["roles"]["original_model_fit"]["groups_overlap_count"] == 0
    assert not report["new_chemical_identity_claim"]


def test_old_radial_lineage_replays_scatter_amplitude_and_weights(source):
    """Reconstruct the old frozen formula, without selecting or refitting it."""
    from opal2.conditional_joint_error_experiment import reference_weights
    from opal2.empirical_radial import reference_weights as radial_weights
    from opal2.joint_contrast_scale import predict_scale

    loader, data, _, _ = source
    radial = DEFAULT_STATE_RUN.parent/"lincs_empirical_radial_20260916_v1"
    if not (radial/"summary.json").exists():
        pytest.skip("Original empirical-radial replay artifact is absent")
    summary = json.loads((radial/"summary.json").read_text())
    previous = Path(summary["previous_run"])
    earlier = json.loads((previous/"summary.json").read_text())
    lookup = {value: i for i, value in enumerate(data["ids"])}
    logamp = np.log(np.linalg.norm(data["Y"][:, 0], axis=1))
    bandwidth = max(float(logamp[loader._record["fit"]].std()), .1)
    with np.load(radial/"AMP_EMP_LOCAL.npz") as z:
        saved_scatter = z["scatter_u"].copy()
    for half in (0, 1):
        cell = next(c for c in summary["cells"] if c["fold"] == 0 and c["half"] == half)
        old = next(c for c in earlier["cells"] if c["fold"] == 0 and c["half"] == half)
        query, fit, reps = (np.array([lookup[x] for x in cell[key]]) for key in
                            ("query_ids", "fit_ids", "representative_ids"))
        weights, _ = reference_weights(data, query, fit, bandwidth)
        with np.load(previous/f"cell_0_{half}_reference.npz") as z:
            residual, local_cov = z["residual"].copy(), z["query_covariance"].copy()
        tau = np.square(np.linalg.solve(np.linalg.cholesky(loader.base_scatter), residual.T)).sum(0)/9
        beta = old["covariance_choice"]["beta"]
        reconstructed = (1-beta+beta*(weights@tau))[:, None, None]*loader.base_scatter
        assert np.array_equal(reconstructed, local_cov)
        amplitude = predict_scale(cell["amplitude_fit"], logamp[query])
        np.testing.assert_allclose(reconstructed*amplitude[:, None, None], saved_scatter[query],
                                   atol=1e-13, rtol=1e-13)
        radial_weight = radial_weights(logamp[reps], logamp[query], cell["local_bandwidth"],
                                       conditional=True)["weights"]
        with np.load(radial/f"cell_0_{half}_radial.npz") as z:
            assert np.array_equal(amplitude, z["amplitude_factor"])
            assert np.array_equal(radial_weight, z["local_weights"])
