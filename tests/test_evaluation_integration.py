"""End-to-end evaluation wiring on declared numerical unit-test fixtures.

These fixtures exercise the complete neural model, fitted 400-tree comparators,
calibration and CLI reload. They are not biological experiments or evidence.
"""
import copy
import json

import numpy as np
import pytest
import torch

from opal2.config import TrainConfig
from opal2.data import MeasurementDataset, TrainScaler, cellprofiler_groups, save_dataset
from opal2.evaluation import actual_utilities, compare_allocations, evaluate_model, predict_utilities, representations
from opal2.model import MeasurementWorldModel
from opal2.splits import save_split
from opal2.utility import UtilityResult, enumerate_actions


@pytest.fixture
def setup():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(117)
    n, w, d = 12, 4, 6
    names = [f"Nuclei_Intensity_Test{i}" for i in range(3)] + [f"Cells_AreaShape_Test{i}" for i in range(3)]
    gi, gn = cellprofiler_groups(names)
    group = np.zeros((n, w, 3), dtype=int)
    group[:, :, 1] = np.arange(w)
    group[:, :, 2] = np.arange(w)
    ds = MeasurementDataset(Y=2. + rng.normal(size=(n, w, d)), ids=[f"numerical_unit{i}" for i in range(n)],
                            feature_names=names, feature_group_index=gi, feature_group_names=gn,
                            cond=rng.normal(size=(n, w, 2)), reference=rng.normal(size=(n, w, 3, 3)),
                            reference_mask=np.ones((n, w, 3), dtype=bool), groups=group,
                            chem=rng.normal(size=(n, 4)), metadata={"fixture": True})
    splits = {"train": np.arange(6), "validation": np.array([6, 7]),
              "calibration": np.array([8, 9]), "evaluation": np.array([10, 11])}
    scaler = TrainScaler.fit(ds, splits["train"])
    config = TrainConfig(seed=301, epochs=1, jepa_epochs=1, batch_size=2, hidden_dim=8,
                         latent_rank=1, residual_rank=1, use_jepa=False, use_library=False, threads=1, samples=4)
    torch.manual_seed(307)
    model = MeasurementWorldModel(ds.feature_groups, 2, 3, 4, hidden_dim=8, latent_rank=1, residual_rank=1)
    model.set_outcome_transform(scaler.y_center,scaler.y_scale)
    yield model, scaler, ds, splits, config
    torch.set_num_threads(old_threads)


def test_future_measurements_change_scores_not_prediction_or_representation(setup):
    model, scaler, ds, splits, config = setup
    ix = splits["evaluation"]
    first, first_score = predict_utilities(model, scaler, ds, ix, config)
    first_rep = representations(model, scaler, ds, ix, config)
    changed = copy.deepcopy(ds)
    changed.Y[ix, 1:4] = 100 + 3 * ds.Y[ix, 1:4]
    second, second_score = predict_utilities(model, scaler, changed, ix, config)
    second_rep = representations(model, scaler, changed, ix, config)
    np.testing.assert_allclose(first.samples, second.samples, rtol=0, atol=0)
    np.testing.assert_allclose(first_rep, second_rep, rtol=0, atol=0)
    assert first_score["fixed_space_mse_per_coordinate"] != second_score["fixed_space_mse_per_coordinate"]
    assert first_score["monte_carlo_environment_latents_shared_across_all_evaluation_chunks"] is True


def test_measurement_scores_use_declared_observation_mask(setup):
    model, scaler, ds, splits, config = setup
    changed = copy.deepcopy(ds)
    ix = splits["evaluation"]
    # Finite storage does not make a measurement observed when its mask is false.
    changed.observed_mask[ix[0], 1] = False
    _, score = predict_utilities(model, scaler, changed, ix, config)
    expected = int(changed.observed_mask[ix][:, 1:4].sum() * ds.Y.shape[-1])
    assert score["scored_coordinates"] == expected


def test_measurement_nll_jacobian_omits_missing_wells(setup):
    from opal2.training import fixed_batch
    model, scaler, ds, splits, config = setup
    changed = copy.deepcopy(ds)
    ix = splits["evaluation"]
    changed.Y[ix[0], 1, 0] = np.nan
    # The dataset contract requires any incomplete well to be marked missing.
    changed.observed_mask[ix[0], 1] = False
    _, score = predict_utilities(model, scaler, changed, ix, config)
    normalized = scaler.transform(changed)
    inputs, target, mask = fixed_batch(normalized, ix, config)
    with torch.no_grad():
        logp = model(inputs).log_prob(target, mask).numpy()
    observed = np.isfinite(changed.Y[ix, 1:4]) & mask.numpy()[..., None]
    jacobian = np.log(scaler.y_scale)[None, None, :] * observed
    expected = (-logp.sum() + jacobian.sum()) / observed.sum()
    assert score["scored_coordinates"] == observed.sum()
    np.testing.assert_allclose(score["fixed_space_nll_per_coordinate"], expected, rtol=1e-12)


def test_four_role_utility_rejects_finite_but_unobserved_initial_x(setup):
    model, scaler, ds, splits, config = setup
    changed = copy.deepcopy(ds)
    ix = splits["evaluation"]
    changed.observed_mask[ix[0], 0] = False
    with pytest.raises(ValueError, match="observed finite initial X"):
        predict_utilities(model, scaler, changed, ix, config)


def test_allocation_report_rejects_fractional_action_indices(setup):
    from opal2.evaluation import allocation_observed
    _, _, ds, splits, _ = setup
    actual = actual_utilities(ds, splits["evaluation"])
    with pytest.raises(ValueError, match="integer dtype"):
        allocation_observed(actual, np.array([1.2, 0.]), label="invalid")


def test_unobserved_outcome_cannot_be_scored_as_actual_gamma(setup):
    _, _, ds, splits, _ = setup
    changed = copy.deepcopy(ds)
    ix = splits["evaluation"]
    changed.observed_mask[ix[0], 1] = False
    with pytest.raises(ValueError, match="observed|missing|complete"):
        actual_utilities(changed, ix)


def test_explicit_worst_completion_charges_only_required_missing_actions(setup):
    _, _, ds, splits, _ = setup
    ix = splits["evaluation"]
    original = actual_utilities(ds, ix)
    np.testing.assert_array_equal(actual_utilities(ds, ix, missing_policy="worst").samples,
                                  original.samples)
    changed = copy.deepcopy(ds)
    changed.observed_mask[ix[0], 1] = False
    completed = actual_utilities(changed, ix, missing_policy="worst")
    assert completed.samples[0, 0, 0] == 0.
    assert completed.samples[0, 0, 1] == -1.01
    assert completed.samples[0, 0, 3] == -1.02
    np.testing.assert_array_equal(completed.samples[:, 0, 2], original.samples[:, 0, 2])
    np.testing.assert_array_equal(completed.samples[:, 1], original.samples[:, 1])


def test_action_mix_random_preserves_spent_budget_and_separates_selection():
    actions = enumerate_actions([0, 1])
    costs = np.array([0., .01, .01, .02])
    prediction = UtilityResult.from_samples(actions, np.array([[
        [0., .20, -.1, .10], [0., -.1, .30, .20],
        [0., .10, .10, .15], [0., -.1, -.1, -.1],
    ]]), costs)
    # Every compound has the same potential gains: any fixed action mix must
    # obtain exactly the same total value, however compounds are permuted.
    actual = UtilityResult.from_samples(actions, np.tile([0., .10, .20, .25], (1, 4, 1)), costs)
    rows, choices = compare_allocations(prediction, actual, ["a", "b", "c", "d"], fractions=(.5,), seed=19)
    for label, action_indices in choices.items():
        selected = next(row for row in rows if row["strategy"] == label)
        random = next(row for row in rows if row["strategy"] == "action_mix_random_for_" + label)
        assert random["used_wells"] == selected["used_wells"] == 4
        assert random["activated"] == selected["activated"] == 3
        assert random["action_mix_and_spent_budget_matched"]
        assert random["randomization_interval_not_sampling_confidence"]
        np.testing.assert_allclose(random["population_mean_net_gain"], selected["population_mean_net_gain"], atol=1e-15)
        np.testing.assert_allclose(random["observed_minus_matched_random_mean"], 0., atol=1e-15)


def test_evaluation_rejects_overlapping_split_before_writing_outputs(setup, tmp_path):
    model, scaler, ds, splits, config = setup
    invalid = copy.deepcopy(splits)
    invalid["evaluation"][0] = invalid["train"][0]
    with pytest.raises(ValueError, match="leakage|overlap"):
        evaluate_model(model, scaler, ds, invalid, config, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_complete_evaluation_writes_calibration_and_fitted_baselines(setup, tmp_path):
    model, scaler, ds, splits, config = setup
    report = evaluate_model(model, scaler, ds, splits, config, tmp_path)
    json.dumps(report, allow_nan=False)
    assert report["evaluation_compounds"] == 2
    assert report["calibration_compounds"] == 2
    assert report["final_opened"] is False
    assert report["coverage_diagnostic"]["certificate"] is False
    assert report["coverage_diagnostic"]["measurement_coverage_is_not_conditional_distribution_calibration"] is True
    assert report["coverage_diagnostic"]["half_width"] is None  # infinite finite-sample radius serialized explicitly
    assert set(report["gain_metrics"]) == {"add_0", "add_1", "add_0_1"}
    assert report["same_representation_direct_heads"]["n"] == 2
    assert report["role_conditional_residual_baseline"]["n"] == 2
    with np.load(tmp_path / "utility_draws.npz") as archive:
        assert archive["samples"].shape == (4, 2, 4)
        np.testing.assert_array_equal(archive["ids"], ds.ids[splits["evaluation"]])
        np.testing.assert_array_equal(archive["calibration_ids"], ds.ids[splits["calibration"]])
    with np.load(tmp_path / "utility_intervals.npz") as archive:
        np.testing.assert_array_equal(archive["lower"], np.full((2, 3), -1.02))
        np.testing.assert_array_equal(archive["upper"], np.ones((2, 3)))


def test_cli_probe_reloads_real_model_and_keeps_predicted_observed_separate(setup, tmp_path):
    from opal2.cli import main
    from dataclasses import asdict
    model, scaler, ds, splits, config = setup
    data_dir, model_dir = tmp_path / "data", tmp_path / "model"
    data_dir.mkdir(); model_dir.mkdir()
    save_dataset(ds, data_dir / "measurements.npz")
    save_split(data_dir / "splits.json", ds.ids, splits, evidence_scope="NUMERICAL_UNIT_TEST_ONLY")
    # A full architecture checkpoint checks reload wiring; it does not claim
    # that these initialized parameters were trained on biological examples.
    torch.save({"model_config": model.config, "state_dict": model.state_dict(),
                "train_config": asdict(config), "epoch": 0,
                "train_ids":ds.ids[splits["train"]].tolist(),
                "validation_ids":ds.ids[splits["validation"]].tolist(),"feature_names":ds.feature_names.tolist()}, model_dir / "best.pt")
    scaler.save(model_dir / "scaler.json")
    common = ["probe", "--data", str(data_dir), "--model", str(model_dir), "--budget", "3",
              "--probe-role", "2", "--selection-samples", "3", "--evaluation-samples", "3", "--seed", "17"]
    predicted, observed = tmp_path / "predicted.json", tmp_path / "observed.json"
    main(common + ["--mode", "predicted", "--outer-samples", "2", "--output", str(predicted)])
    main(common + ["--mode", "observed", "--output", str(observed)])
    pred, obs = json.loads(predicted.read_text()), json.loads(observed.read_text())
    assert pred["mode"] == "MODEL_BASED_PROBE_PLANNING" and not pred["observed_result"]
    assert obs["mode"] == "OBSERVED_RETROSPECTIVE_PROBE_EVALUATION" and obs["observed_result"]
    assert pred["roles"] == obs["roles"] == {"X": 0, "P": 2, "Q": 1, "V": 3}
    assert pred["sample_counts"] == {"probe": 2, "selection_per_probe": 3, "evaluation_per_probe": 3}
    assert obs["policy"]["used_wells"] <= 3
    with pytest.raises(FileExistsError):
        main(common + ["--mode", "observed", "--output", str(observed)])
