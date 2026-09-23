"""Leakage and original-query reuse checks for the decision-region adapter."""
import numpy as np
import pytest

from opal2.decision_region_cache_standard import (
    _assert_query_reuse, crossfit_core_predictions, crossfit_direct_predictions,
    grouped_calibration_folds,
)


def test_group_folds_keep_aliases_and_are_independent_of_row_order():
    groups = np.array(["a", "b", "a", "c", "d", "e", "f", "b"])
    folds = grouped_calibration_folds(groups, 20260920)
    permutation = np.array([7, 2, 4, 0, 5, 3, 1, 6])
    reordered = grouped_calibration_folds(groups[permutation], 20260920)
    np.testing.assert_array_equal(reordered, folds[permutation])
    assert len(np.unique(folds)) == 5
    assert folds[0] == folds[2]
    assert folds[1] == folds[7]
    with pytest.raises(ValueError, match="at least two"):
        grouped_calibration_folds(["a", "a"], 0)


def test_direct_held_fold_outcomes_do_not_enter_its_own_predictions():
    rng = np.random.default_rng(89)
    groups = np.array([f"g{i // 2}" for i in range(40)])
    folds = grouped_calibration_folds(groups, 11)
    point = rng.uniform(-.3, .3, len(groups))
    raw = rng.uniform(.05, .95, len(groups))
    actual = rng.uniform(-.4, .4, len(groups))
    original = crossfit_direct_predictions(point, raw, actual, groups, folds, seed=12)
    held = folds == 0
    changed = actual.copy()
    changed[held] = .98
    altered = crossfit_direct_predictions(point, raw, changed, groups, folds, seed=12)
    for arm in ("CAL", "COHERENT"):
        for key in ("cal_p", "cal_mean"):
            np.testing.assert_array_equal(original[arm][key][held], altered[arm][key][held])
    # Other folds legitimately use these changed labels to calibrate their laws.
    assert not np.array_equal(original["COHERENT"]["cal_mean"][~held],
                              altered["COHERENT"]["cal_mean"][~held])


def test_direct_single_class_inner_cal_uses_declared_original_fallback():
    groups = np.array([f"g{i}" for i in range(8)])
    folds = grouped_calibration_folds(groups, 23)
    actual = np.full(8, .1)
    result = crossfit_direct_predictions(np.zeros(8), np.full(8, .5), actual, groups, folds, seed=7)
    for record in result["audit"]:
        held = folds == record["inner_fold"]
        np.testing.assert_array_equal(result["CAL"]["cal_p"][held],
                                      np.full(held.sum(), 1 / (record["train_rows"] + 2)))
    np.testing.assert_array_equal(result["COHERENT"]["cal_p"], np.zeros(8))


def test_core_full_law_excludes_held_residuals_and_whole_alias_groups():
    rng = np.random.default_rng(913)

    def inputs(prefix, n, grouped=False):
        return {"ids": np.array([f"{prefix}{i}" for i in range(n)]),
                "groups": np.array([f"{prefix}g{i // 2 if grouped else i}" for i in range(n)]),
                "X": rng.normal(size=(n, 12)) * np.exp(rng.normal(size=(n, 1)) * .3),
                "chem": rng.integers(0, 2, size=(n, 512)).astype(float)}

    ref, cal = inputs("r", 24), inputs("c", 20, grouped=True)
    rr, cr = rng.normal(size=(24, 9)), rng.normal(size=(20, 9))
    mean = rng.normal(size=(20, 9)) * .1
    folds = grouped_calibration_folds(cal["groups"], 71)
    parameters = dict(ref_inputs=ref, ref_residual=rr, cal_inputs=cal, cal_residual=cr,
        cal_mean=mean, base_scatter=np.eye(9), training_log_amplitude_sd=.4,
        stats={"u_center": np.zeros(9), "u_scale": np.ones(9)}, inner_fold=folds,
        model_training_ids=np.array(["train"]), model_training_groups=np.array(["train_group"]),
        samples=128, seed=91)
    original = crossfit_core_predictions(**parameters)
    held = folds == 0
    changed = cr.copy()
    changed[held] *= 50
    altered = crossfit_core_predictions(**dict(parameters, cal_residual=changed))
    for key in ("cal_p", "cal_mean"):
        np.testing.assert_array_equal(original[key][held], altered[key][held])
    for record in original["audit"]:
        assert not set(record["radial_representative_ids"]) & set(record["held_ids"])
        assert record["radial_calibration_n"] == record["train_groups"]
    assert not np.array_equal(original["cal_mean"][~held], altered["cal_mean"][~held])


def test_query_reuse_rejects_any_prediction_change():
    original = {"ids": np.array(["a", "b", "c"]), "actual": np.array([.1, -.1, .2]),
                "predicted": np.array([.05, -.08, .1]), "p_null": np.array([.1, .7, .2])}
    rows = np.array([0, 2])
    query = {key: value[rows].copy() for key, value in original.items()}
    _assert_query_reuse(query, original, rows)
    query["p_null"][0] += .001
    with pytest.raises(AssertionError):
        _assert_query_reuse(query, original, rows)
