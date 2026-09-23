"""Numerical engineering fixtures for the full-model experiment interfaces."""
from dataclasses import asdict, replace
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opal2.biology_kernel_evaluation import (actual_gains, evaluate_partition, interval_stats,
                                           predict_partition, score_measurements, utility_draws)
from opal2.biology_kernel_experiment import (ARM_FACTORS, SEEDS, _check_common_initialization,
                                           configurations, prepare)
from opal2.biology_kernel_comparison import compare_arrays, summarize_factorial, ARM_NAMES
from opal2.config import TrainConfig
from opal2.data import MeasurementDataset, TrainScaler, cellprofiler_groups
from opal2.model import JointGaussian, MeasurementWorldModel
from opal2.training import fixed_batch


@pytest.fixture
def setup():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(173)
    n, w, d = 12, 4, 6
    names = [f"Nuclei_Intensity_F{i}" for i in range(3)] + [f"Cells_AreaShape_F{i}" for i in range(3)]
    gi, gn = cellprofiler_groups(names)
    groups = np.zeros((n, w, 3), int)
    groups[:, :, 1] = np.arange(w)
    groups[:, :, 2] = np.arange(w)
    ds = MeasurementDataset(Y=2. + rng.normal(size=(n, w, d)), ids=[f"fixture{i}" for i in range(n)],
        feature_names=names, feature_group_index=gi, feature_group_names=gn,
        cond=rng.normal(size=(n, w, 2)), reference=rng.normal(size=(n, w, 3, 3)),
        reference_mask=np.ones((n, w, 3), bool), groups=groups, chem=rng.normal(size=(n, 4)),
        metadata={"fixture": True})
    splits = dict(train=np.arange(6), validation=np.array([6, 7]), calibration=np.array([8, 9]), evaluation=np.array([10, 11]))
    scaler = TrainScaler.fit(ds, splits["train"])
    cfg = TrainConfig(seed=7, epochs=1, hidden_dim=8, latent_rank=1, residual_rank=1,
        batch_size=2, use_jepa=False, use_library=False, threads=1, samples=11, mc_chunk_size=4)
    torch.manual_seed(173)
    model = MeasurementWorldModel(ds.feature_groups, 2, 3, 4, hidden_dim=8, latent_rank=1, residual_rank=1)
    model.set_outcome_transform(scaler.y_center, scaler.y_scale)
    yield model, scaler, ds, splits, cfg
    torch.set_num_threads(previous)


def test_full_main_queue_is_four_factors_three_seeds_no_hidden_method_reduction():
    configs = configurations()
    assert tuple(map(int, configs)) == SEEDS
    for seed, arms in configs.items():
        assert tuple(arms) == tuple(ARM_FACTORS)
        for name, cfg in arms.items():
            assert cfg.seed == int(seed)
            assert (cfg.biology_kernel_mode, cfg.observation_family) == ARM_FACTORS[name]
            assert cfg.hidden_dim == 256 and cfg.latent_rank == 32 and cfg.residual_rank == 8
            assert cfg.epochs == 100 and cfg.batch_size == 32
            assert cfg.learning_rate == 3e-4 and cfg.lr_schedule == "cosine"
            assert cfg.warmup_steps == 60 and cfg.min_learning_rate == 3e-6
            assert cfg.samples == 2000 and cfg.mc_chunk_size == 32
            assert not cfg.use_jepa and not cfg.use_biology_prior
            assert cfg.use_chemistry and cfg.use_references and cfg.use_library
            assert cfg.objective == "predictive_nll" and cfg.paired_objective_rng
            assert cfg.kernel_mode == "measurement"
        common = asdict(next(iter(arms.values())))
        for cfg in arms.values():
            actual = asdict(cfg)
            for key in common.keys() - {"biology_kernel_mode", "observation_family"}:
                assert actual[key] == common[key]


def test_future_answers_change_scores_not_future_draws_or_predictions(setup):
    model, scaler, ds, splits, cfg = setup
    ix = splits["evaluation"]
    first = predict_partition(model, scaler, ds, ix, cfg, seed=201, object_chunk=1)
    changed = copy.deepcopy(ds)
    changed.Y[ix, 1:] = changed.Y[ix, 1:] * 3 + 4
    second = predict_partition(model, scaler, changed, ix, cfg, seed=201, object_chunk=1)
    for field in ("prediction_mean", "utility_samples", "predicted", "p_null", "mc_se"):
        np.testing.assert_array_equal(first[field], second[field])
    assert not np.allclose(first["raw_nll"], second["raw_nll"])
    assert first["utility_samples"].shape == (11, 2, 3)
    assert first["coordinate_interval_hits"].shape == (2, 4)
    assert first["difference_interval_hits"].shape == (2, 3, 4)


def test_original_space_nll_has_correct_jacobian(setup):
    model, scaler, ds, splits, cfg = setup
    ix = splits["evaluation"]
    values = predict_partition(model, scaler, ds, ix, cfg, seed=11)
    normalized = scaler.transform(ds)
    model.eval()
    with torch.no_grad():
        x, y, mask = fixed_batch(normalized, ix, cfg)
        logp = model(x).log_prob(y, mask).numpy()
    expected = (-logp + 3 * np.log(scaler.y_scale).sum()) / (3 * ds.Y.shape[-1])
    np.testing.assert_allclose(values["raw_nll"], expected, atol=1e-7)
    np.testing.assert_allclose(values["raw_nll"] - values["standardized_nll"], np.log(scaler.y_scale).mean())


def test_copula_t4_uses_same_full_evaluation_interface(setup):
    _, scaler, ds, splits, cfg = setup
    torch.manual_seed(19)
    model = MeasurementWorldModel(ds.feature_groups, 2, 3, 4, hidden_dim=8, latent_rank=1,
                                   residual_rank=1, observation_family="copula_t4")
    model.set_outcome_transform(scaler.y_center, scaler.y_scale)
    cfg = replace(cfg, observation_family="copula_t4")
    values = predict_partition(model, scaler, ds, splits["evaluation"], cfg, seed=5)
    assert np.isfinite(values["raw_nll"]).all()
    assert values["prediction_mean"].shape == (2, 3, 6)
    assert values["difference_interval_hits"].shape == (2, 3, 4)
    assert np.all(values["p_null"] + values["p_positive"] <= 1 + 1e-12)


def test_environment_noise_cache_reused_across_object_chunks(setup, monkeypatch):
    model, scaler, ds, splits, cfg = setup
    seen = []
    original = JointGaussian.sample_joint
    def wrapped(self, n_samples, generator=None, environment_noise_cache=None):
        seen.append(id(environment_noise_cache))
        assert environment_noise_cache is not None
        return original(self, n_samples, generator, environment_noise_cache)
    monkeypatch.setattr(JointGaussian, "sample_joint", wrapped)
    predict_partition(model, scaler, ds, splits["evaluation"], cfg, seed=4, object_chunk=1)
    assert len(seen) == 6
    assert seen[:3] == seen[3:]


def test_utility_is_unchanged_and_positive_probability_preserves_ambiguous_region(setup):
    _, _, ds, _, _ = setup
    actual = actual_gains(ds.Y)
    x, z1, z2, v = np.moveaxis(ds.Y, 1, 0)
    def cos(a, b):
        return (a*b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    expected = .5 * (cos((x + z1 + z2) / 3, v) - cos(x, v)) - .02
    np.testing.assert_allclose(actual[:, 2], expected)
    np.testing.assert_array_equal(actual, utility_draws(ds.Y[:, 0], ds.Y[None, :, 1:])[0])
    with pytest.raises(ValueError):
        actual_gains(ds.Y[:, :3])
    bad = ds.Y.copy(); bad[0, 0] = 0
    with pytest.raises(ValueError):
        actual_gains(bad)


def test_joint_pair_intervals_use_real_differences_not_independent_marginals():
    rng = np.random.default_rng(4)
    shared = rng.normal(size=(2000, 2, 1, 3))
    draw = np.repeat(shared, 2, axis=2)
    observed = np.zeros((2, 3))
    hits, widths, coordinates = interval_stats(draw[:, :, 0] - draw[:, :, 1], observed)
    assert coordinates == 3
    np.testing.assert_array_equal(widths, 0)
    np.testing.assert_array_equal(hits, 3)


def test_full_evaluation_emits_trace_and_keeps_original_budget_risk(setup, tmp_path):
    model, scaler, ds, splits, cfg = setup
    report = evaluate_partition(model, scaler, ds, splits["train"], splits["evaluation"], cfg,
        tmp_path / "partition", seed=5, fractions=(.5,), n_bootstrap=5, n_random=5)
    assert report["n"] == 2 and report["all_objects_retained"]
    assert not report["formal_certificate"] and not report["original_contract_changed"]
    assert len(report["predictive_intervals"]) == 4
    assert report["samples"] == 11
    assert report["shared_environment_draws_preserved_across_object_chunks"]
    assert len(report["policy"]["row_trace"]) == 2
    assert report["policy"]["positive_margin"] == .005
    assert all(row["used_wells"] <= row["budget_wells"] for row in report["policy"]["common_budget"])
    with np.load(tmp_path / "partition" / "predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ids"], ds.ids[splits["evaluation"]])
        assert z["standardized_sse"].shape == (2,)
        assert z["prediction_mean"].shape == (2, 3, 6)
    json.dumps(report, allow_nan=False)


def test_missing_observed_roles_are_not_silently_removed(setup):
    model, scaler, ds, splits, cfg = setup
    changed = copy.deepcopy(ds)
    changed.observed_mask[splits["evaluation"][0], 3] = False
    with pytest.raises(ValueError, match="complete"):
        predict_partition(model, scaler, changed, splits["evaluation"], cfg, seed=3)


def test_response_metrics_keep_amplitude_and_zero_direction_status():
    actual = np.array([[[1., 0.]], [[0., 0.]]])
    prediction = actual * 2
    report, trace = score_measurements(actual, prediction, np.zeros((1, 2)), np.ones(2))
    assert report["profile_direction_mean"] == 1
    assert report["profile_direction_defined_count"] == 1
    assert report["profile_direction_total_count"] == 2
    assert report["rms_amplitude_mae"] > 0
    assert np.isnan(trace["profile_direction"][1, 0])
    assert report["physical"]["overall"]["sse"] == 1


def test_prepare_has_fresh_output_and_twelve_fixed_jobs(setup, tmp_path, monkeypatch):
    import opal2.biology_kernel_experiment as module
    _, _, ds, splits, _ = setup
    monkeypatch.setattr(module, "_load_study_data", lambda _: (ds, splits, {"fixture": True}))
    protocol = tmp_path / "protocol.md"; protocol.write_text("Synthetic test protocol; no experiment result.\n")
    manifest = prepare(tmp_path / "source5_primary_fullcontrols", tmp_path / "run", protocol)
    assert len(manifest["queue"]) == 12
    assert manifest["results_do_not_modify_remaining_queue"]
    assert not manifest["mechanism_annotations_active"]
    assert (tmp_path / "run" / "PROTOCOL.md").read_text() == protocol.read_text()
    assert (tmp_path / "run" / "source_snapshot" / "opal2" / "biology_kernel_experiment.py").is_file()
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "source5_primary_fullcontrols", tmp_path / "run", protocol)


def test_paired_common_initialization_detects_unintended_backbone_change(tmp_path):
    first = dict(seed=1, arm="A", directory="a")
    later = dict(seed=1, arm="C", directory="c")
    for folder in ("a", "c"):
        (tmp_path / folder).mkdir()
    torch.save({"backbone": torch.ones(2)}, tmp_path / "a" / "initial_state.pt")
    torch.save({"backbone": torch.ones(2), "new_kernel": torch.ones(3)}, tmp_path / "c" / "initial_state.pt")
    _check_common_initialization(tmp_path, {"queue": [first, later]}, later)
    torch.save({"backbone": torch.zeros(2)}, tmp_path / "c" / "initial_state.pt")
    with pytest.raises(ValueError, match="initial"):
        _check_common_initialization(tmp_path, {"queue": [first, later]}, later)


def comparison_arrays():
    n = 20
    ids = np.array([f"unit{i:02d}" for i in range(n)])
    actual = np.tile(np.linspace(-.03, .03, n)[:, None], (1, 3))
    arms = {}
    for index, name in enumerate(ARM_NAMES):
        arms[name] = dict(ids=ids, actual=actual, predicted=actual if index else -actual,
            p_null=(actual <= 0).astype(float), standardized_sse=np.ones(n) * (4 - index),
            raw_nll=np.ones(n) * (2 - index / 10), feature_dim=6)
    return arms


def test_factorial_pairing_has_correct_signs_budgets_and_no_dropped_objects():
    result, _ = compare_arrays(comparison_arrays(), fractions=(.25,), n_bootstrap=9)
    assert result["n"] == 20 and not result["pending"]
    contrast = result["contrasts"]["C_minus_A_kernel_gaussian"]["metrics"]
    assert contrast["measurement/standardized_mse"]["difference"] == pytest.approx(-2 / 18)
    key = "common_budget/Z1/expected_gain/0.25/per_eligible_net_gain"
    assert contrast[key]["difference"] > 0
    fdp = "common_budget/Z1/expected_gain/0.25/fdp"
    assert contrast[fdp]["difference"] == -1
    for arm in result["selections"].values():
        single = arm["common_budget/Z1/expected_gain/0.25"]
        double = arm["common_budget/Z1Z2/expected_gain/0.25"]
        assert single["budget_wells"] == double["budget_wells"] == 5
        assert single["used_wells"] == 5 and double["used_wells"] == 4
    assert not result["formal_certificate"]


def test_factorial_reorders_only_exact_id_sets_and_rejects_changed_actual():
    arms = comparison_arrays()
    first, _ = compare_arrays(arms, fractions=(.25,), n_bootstrap=5)
    for key, value in list(arms[ARM_NAMES[1]].items()):
        if key != "feature_dim":
            arms[ARM_NAMES[1]][key] = value[::-1]
    second, _ = compare_arrays(arms, fractions=(.25,), n_bootstrap=5)
    assert first == second
    arms[ARM_NAMES[1]]["actual"] = arms[ARM_NAMES[1]]["actual"].copy()
    arms[ARM_NAMES[1]]["actual"][0, 0] += .1
    with pytest.raises(ValueError, match="utility"):
        compare_arrays(arms, n_bootstrap=5)


def test_factorial_does_not_claim_unfinished_contrasts():
    arms = comparison_arrays()
    del arms[ARM_NAMES[2]], arms[ARM_NAMES[3]]
    result, _ = compare_arrays(arms, n_bootstrap=5)
    assert list(result["contrasts"]) == ["B_minus_A_noise_without_kernel"]
    assert len(result["pending"]) == 3


def test_average_seed_keeps_twenty_objects_not_sixty(tmp_path):
    arms = comparison_arrays()
    queue = []
    for seed in SEEDS:
        for name, values in arms.items():
            relative = f"jobs/{seed}/{name}"
            queue.append(dict(seed=seed, arm=name, directory=relative))
            folder = tmp_path / relative / "evaluation"
            folder.mkdir(parents=True)
            np.savez(folder / "predictions.npz", **{k: v for k, v in values.items() if k != "feature_dim"})
            (folder / "metrics.json").write_text(json.dumps(dict(ids=values["ids"].tolist(),
                formal_certificate=False, original_contract_changed=False)))
    manifest = dict(queue=queue, seeds=list(SEEDS), evaluation_splits=["evaluation"],
        compound_ids={"evaluation": arms[ARM_NAMES[0]]["ids"].tolist()}, data_shape=[20, 4, 6],
        fractions=[.25], n_bootstrap=5)
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    result = summarize_factorial(tmp_path)
    averaged = result["average_seed"]["evaluation"]
    assert averaged["n"] == 20
    assert all(r["independent_object_count"] == 20 for r in averaged["contrasts"].values())
    assert not averaged["pending"]
    assert (tmp_path / "factorial_comparison.json").exists()
