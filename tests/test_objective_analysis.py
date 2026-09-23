"""Analysis integrity tests use synthetic fixtures, never experimental evidence."""
import json

import numpy as np
import pandas as pd
import pytest

from opal2.objective_analysis import ACTIONS, ARMS, _metrics, fair_crps, generate_analysis


def _fixture(root):
    n, s = 8, 32
    ids = np.array([f"fixture_{i}" for i in range(n)])
    actual = np.column_stack((np.zeros(n), np.linspace(-.03, .07, n),
                              np.linspace(.04, -.06, n), np.linspace(-.06, .06, n)))
    for number, name in enumerate(ARMS):
        folder = root / "arms" / name
        folder.mkdir(parents=True)
        samples = np.random.default_rng(number).normal(0, .06, size=(s, n, 4))
        samples[..., 0] = 0
        frame = pd.DataFrame({"compound_id": ids})
        gains = {}
        for j, action in enumerate(ACTIONS):
            mean = samples[..., j].mean(0)
            null = (samples[..., j] <= 0).mean(0)
            positive = (samples[..., j] >= .005).mean(0)
            for suffix, values in (("actual", actual[:, j]), ("predicted", mean),
                                   ("p_null", null), ("p_positive", positive)):
                frame[action + "__" + suffix] = values
            if j:
                gains[action] = _metrics(actual[:, j], mean, null, positive)
        allocations = []
        for fraction in (.05, .1, .25):
            budget = 2 * int(np.ceil(fraction * n))
            choices = np.array([1, 2, 3 if budget >= 4 else 0, 0, 0, 0, 0, 0])
            for limit in (None, .35):
                planner = f"planner_budget_{fraction:g}_model_null_{limit}"
                frame[planner] = choices
                active = choices > 0
                values = actual[np.arange(n), choices]
                allocations.append(dict(strategy=planner, budget_wells=budget, used_wells=budget,
                                        activated=int(active.sum()), population_mean_net_gain=values.mean(),
                                        null_count=int(((values <= 0) & active).sum())))
        frame.to_csv(folder / "evaluation_predictions.tsv", sep="\t", index=False)
        np.savez(folder / "utility_draws.npz", samples=samples, ids=ids, calibration_ids=np.array(["cal_a", "cal_b"]))
        pd.DataFrame(allocations).to_csv(folder / "allocations.tsv", sep="\t", index=False)
        summary = dict(final_opened=False, evaluation_compounds=n, calibration_compounds=2,
                       full_feature_dimension=3617, positive_margin=.005, cost_per_optional_well=.01,
                       utility="original fixed-space half-cosine gain", gain_metrics=gains,
                       model=dict(seed=1, hidden_dim=256, samples=s,
                                  objective="elbo" if number == 0 else "predictive_nll",
                                  utility_crps_weight=1.0 if number == 2 else 0.0),
                       same_representation_direct_heads=gains["add_0_1"],
                       measurement=dict(fixed_space_nll_per_coordinate=1., fixed_space_mse_per_coordinate=1.,
                                        uncalibrated_gaussian_90pct_coordinate_coverage=.7, scored_coordinates=n * 3 * 3617),
                       coverage_diagnostic=dict(certificate=False, half_width=None))
        (folder / "evaluation.json").write_text(json.dumps(summary))
    return actual


def test_fair_crps_matches_explicit_off_diagonal_pairs():
    samples = np.array([[0., 1.], [1., 3.], [2., 6.], [4., 7.]])
    actual = np.array([1.5, 4.])
    pair_sum = sum(np.abs(samples[i] - samples[j]) for i in range(4) for j in range(4) if i != j)
    expected = np.abs(samples - actual).mean(0) - .5 * pair_sum / (4 * 3)
    np.testing.assert_allclose(fair_crps(samples, actual), expected)
    empirical_biased = np.abs(samples - actual).mean(0) - .5 * pair_sum / 16
    assert not np.allclose(expected, empirical_biased)


def test_completed_matched_analysis_scores_and_exact_action_mix_random(tmp_path):
    actual = _fixture(tmp_path)
    result = generate_analysis(tmp_path, bootstrap_replicates=5, seed=4)
    assert result["endpoints_and_order_validated"]
    assert not result["final_opened"]
    assert len(result["action_metrics"]) == 12
    assert len(result["same_budget_comparisons"]) == 36
    row = next(r for r in result["same_budget_comparisons"] if r["strategy"] == "planner_budget_0.25_model_null_None")
    assert row["action_counts"] == dict(stop=5, add_0=1, add_1=1, add_0_1=1)
    assert row["used_wells"] == 4
    assert row["matched_action_mix_random_mean"] == pytest.approx(actual[:, 1:].mean(0).sum() / len(actual))
    assert all(not r["formal_certificate"] for r in result["paired_bootstrap"])
    for name in ("comparison.json", "comparison.tsv", "allocation_comparison.tsv", "RESULT_ANALYSIS.md"):
        assert (tmp_path / name).is_file()
    text = (tmp_path / "RESULT_ANALYSIS.md").read_text()
    assert "not as proof of efficacy" in text
    assert "not sampling confidence" in text
    persisted = json.loads((tmp_path / "comparison.json").read_text())
    assert persisted["measurement_metrics"][0]["conformal_diagnostic"]["half_width"] is None


@pytest.mark.parametrize("problem", ["missing_arm", "id_order", "action_order", "actual", "nonobjective_config", "planner_value"])
def test_mismatched_evidence_fails_before_emitting_comparison(tmp_path, problem):
    _fixture(tmp_path)
    folder = tmp_path / "arms" / "B_NLL"
    if problem == "missing_arm":
        (folder / "evaluation.json").unlink()
    elif problem in {"id_order", "action_order"}:
        file = folder / "utility_draws.npz"
        with np.load(file, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        if problem == "id_order":
            arrays["ids"] = arrays["ids"][::-1]
        else:
            arrays["samples"] = arrays["samples"][:, :, [0, 2, 1, 3]]
        np.savez(file, **arrays)
    elif problem == "actual":
        file = folder / "evaluation_predictions.tsv"
        frame = pd.read_csv(file, sep="\t")
        frame.loc[0, "add_0_1__actual"] += .01
        frame.to_csv(file, sep="\t", index=False)
    elif problem == "nonobjective_config":
        file = folder / "evaluation.json"
        summary = json.loads(file.read_text())
        summary["model"]["hidden_dim"] = 128
        file.write_text(json.dumps(summary))
    else:
        file = folder / "allocations.tsv"
        frame = pd.read_csv(file, sep="\t")
        frame.loc[0, "population_mean_net_gain"] += .01
        frame.to_csv(file, sep="\t", index=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        generate_analysis(tmp_path)
    assert not (tmp_path / "comparison.json").exists()
