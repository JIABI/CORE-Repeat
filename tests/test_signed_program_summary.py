"""Engineering checks for paired aggregation, not experimental evidence."""
import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_r3_signed_program_20260921.py"
SPEC = importlib.util.spec_from_file_location("signed_program_summary", PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_group_means_keep_all_doses_and_aliases_together():
    scores = np.zeros((2, 4, 2, 3))
    scores[:, :, 0, :] = np.asarray([1., 3., 5., 10.])[None, :, None]
    scores[:, :, 1, :] = scores[:, :, 0, :] + 2
    labels, values, counts = summary.group_means(scores, np.asarray(["a", "a", "a", "b"]), np.ones(4, bool))
    np.testing.assert_array_equal(counts, [3, 1])
    np.testing.assert_allclose(values[:, 0, 0, 0], [3, 10])
    boot = summary.paired_bootstrap(values, 42, n=100)
    np.testing.assert_allclose(boot[:, :, 1, :] - boot[:, :, 0, :], 2)


def test_random_score_mean_not_counted_as_an_extra_random_draw():
    scores = np.zeros((2, 4, 4, 3))
    scores[:, :, 1, :] = 2
    scores[:, :, 2, :] = 4
    scores[:, :, 3, :] = 3
    compact, names, counts = summary.compact_random(scores, ["RIDGE_RESPONSE", "TARGET_R01_CAL", "TARGET_R02_CAL", "TARGET_RANDOM_MEAN_CAL"])
    assert names == ["RIDGE_RESPONSE", "TARGET_RANDOM_MEAN_CAL"]
    assert counts["TARGET_RANDOM_MEAN_CAL"] == 2
    np.testing.assert_array_equal(compact[:, :, 1, :], 3)


def test_simultaneous_interval_marks_exact_fallback_without_dividing_zero():
    mean = np.ones((2, 2, 3))
    boot = np.ones((100, 2, 2, 3))
    result = summary.simultaneous_mse_intervals(mean, boot, [("A_CAL", "RIDGE_RESPONSE")], ["RIDGE_RESPONSE", "A_CAL"])
    assert result["active_contrasts"] == 0
    entry = result["intervals"]["A_CAL minus RIDGE_RESPONSE"]["CROSS"]
    assert entry["deterministic_tie"]
    assert entry["simultaneous_ci95"] == [0., 0.]


def test_cross_minus_same_uses_the_paired_task_draws():
    mean = np.ones((2, 2, 3)) * 10
    mean[0, 1] = 9
    mean[1, 1] = 8
    boot = np.repeat(mean[None], 100, axis=0)
    result = summary.contrast_record(mean, boot, 1, 0)["CROSS_minus_SAME"]["profile_mse"]
    assert result["risk_difference_interaction"] == -1
    assert result["difference_ci95"] == [-1., -1.]
    np.testing.assert_allclose(result["improvement_percentage_point_interaction"], 10)
    np.testing.assert_allclose(result["improvement_pp_ci95"], [10., 10.])
