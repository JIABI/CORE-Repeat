"""Synthetic checks for fixed-budget CAL tuning and selected-set forecasts."""
from copy import deepcopy

import numpy as np
import pytest

from opal2.decision_region_calibration import select_top_k
from opal2.selected_risk_replay import (
    ARMS, LAMBDA_GRID, VARIANTS, apportion_cal_budget, choose_lambda,
    evaluate_cell, select_cal_by_fold, validate_cell,
)


def example_cell():
    rng = np.random.default_rng(103)
    n, nc = 32, 40
    return {"dataset": "synthetic", "cell": "test", "query_budget": 4,
        "cal_ids": np.array([f"c{i:03d}" for i in range(nc)]),
        "cal_groups": np.array([f"cg{i:03d}" for i in range(nc)]),
        "cal_layout": np.array([f"cp{i % 3}" for i in range(nc)]),
        "cal_inner_fold": np.repeat(np.arange(5), 8), "cal_actual": rng.uniform(-.2, .4, nc),
        "query_ids": np.array([f"q{i:03d}" for i in range(n)]),
        "query_groups": np.array([f"qg{i:03d}" for i in range(n)]),
        "query_layout": np.array([f"qp{i % 3}" for i in range(n)]),
        "query_actual": rng.uniform(-.2, .4, n),
        "arms": {arm: {"cal_p": rng.uniform(.02, .95, nc), "cal_mean": rng.uniform(-.1, .3, nc),
                       "query_p": rng.uniform(.02, .95, n), "query_mean": rng.uniform(-.1, .3, n)}
                 for arm in ARMS}}


def saved_predictions(cell):
    saved = {"ids": cell["query_ids"].copy()}
    for arm, values in cell["arms"].items():
        saved[arm + "__mean"] = values["query_mean"].copy()
        saved[arm + "__ORIGINAL__p"] = values["query_p"].copy()
        saved[arm + "__original_selected"] = select_top_k(cell["query_ids"], values["query_mean"],
                                                           values["query_p"], cell["query_budget"], .2)
        saved[arm + "__lambda0_selected"] = select_top_k(cell["query_ids"], values["query_mean"],
                                                         values["query_p"], cell["query_budget"], 0.)
    return saved


def test_half_up_budget_and_largest_remainders_are_exact_and_outcome_free():
    # 20 * 4 / 32 = 2.5, which rounds upward rather than banker's rounding.
    allocation = apportion_cal_budget(np.repeat(np.arange(5), 4), 4, 32)
    assert allocation["total_budget"] == 3
    assert allocation["fold_budgets"] == {"0": 1, "1": 1, "2": 1, "3": 0, "4": 0}
    # Equal remainders at folds 0/1; larger fold receives the final slot.
    allocation = apportion_cal_budget(np.repeat([0, 1, 2], [2, 6, 4]), 1, 4)
    assert allocation["fold_budgets"] == {"0": 0, "1": 2, "2": 1}
    permuted = np.random.default_rng(90).permutation(np.repeat([0, 1, 2], [2, 6, 4]))
    assert allocation == apportion_cal_budget(permuted, 1, 4)
    full = apportion_cal_budget(permuted, 4, 4)
    assert full["fold_budgets"] == full["fold_sizes"]


@pytest.mark.parametrize("objectives,expected", [
    ([0., 0., 0., 0.], .2),
    ([10., 9., 8., 10.], 0.),
    ([8., 10., 9., 10.], .1),
    ([0., 0., 1. - 5e-13, 1.], .2),
    ([0., 0., 1. - 2e-12, 1.], .4),
])
def test_lambda_ties_follow_fixed_preference(objectives, expected):
    chosen, index = choose_lambda(objectives)
    assert chosen == expected
    assert LAMBDA_GRID[index] == expected


def test_lambda_selection_maximizes_realized_gamma_and_risk_never_reranks():
    cell = example_cell()
    for arm in ARMS:
        values = cell["arms"][arm]
        values["cal_mean"] = np.tile([.3, .25, .235, -1., -1., -1., -1., -1.], 5)
        values["cal_p"] = np.tile([1., .1, 0., 0., 0., 0., 0., 0.], 5)
    # lambda0 chooses A, .1 chooses B, and .2/.4 choose C in every held fold.
    cell["cal_actual"] = np.tile([.1, .8, -.1, -.1, -.1, -.1, -.1, -.1], 5)
    saved = saved_predictions(cell)
    before = deepcopy(cell)
    arrays, meta = evaluate_cell(cell, saved)
    for arm in ARMS:
        info = meta["arms"][arm]
        assert info["chosen_lambda"] == .1
        assert [r["selected_mean_Gamma"] for r in info["calibration_grid"]] == pytest.approx([.1, .8, -.1, -.1])
        assert info["q_empirical"] == 1.
        assert info["q_jeffreys"] == pytest.approx(5.5 / 6)
        assert info["frozen_primary_risk"]["raw_predicted_query_NULL"] == 4.
        for key in ("cal_p", "cal_mean", "query_p", "query_mean"):
            np.testing.assert_array_equal(arrays[arm + "__" + key], before["arms"][arm][key])
            np.testing.assert_array_equal(cell["arms"][arm][key], before["arms"][arm][key])
        np.testing.assert_array_equal(arrays[arm + "__FIXED_02"], saved[arm + "__original_selected"])
        np.testing.assert_array_equal(arrays[arm + "__FIXED_0"], saved[arm + "__lambda0_selected"])
        mask = arrays[arm + "__FIXED_02"]
        np.testing.assert_array_equal(arrays[arm + "__FIXED_02_SELECTED_QUERY_P"],
                                      cell["arms"][arm]["query_p"][mask])


def test_budgets_match_for_all_arms_lambdas_and_held_folds():
    cell = example_cell()
    arrays, metadata = evaluate_cell(cell, saved_predictions(cell))
    for arm in ARMS:
        grid = arrays[arm + "__CAL_SELECTED_GRID"]
        np.testing.assert_array_equal(grid.sum(axis=1), np.full(4, metadata["cal_budget"]))
        for fold, count in metadata["budget"]["fold_budgets"].items():
            held = cell["cal_inner_fold"] == int(fold)
            np.testing.assert_array_equal(grid[:, held].sum(axis=1), np.full(4, count))
        for variant in VARIANTS:
            assert arrays[arm + "__query_selected_" + variant].sum() == cell["query_budget"]


def test_query_labels_cannot_change_lambda_risk_or_any_prediction():
    cell = example_cell()
    saved = saved_predictions(cell)
    arrays, info = evaluate_cell(cell, saved)
    cell["query_actual"] = np.random.default_rng(29).permutation(cell["query_actual"])
    changed, info_changed = evaluate_cell(cell, saved)
    assert info == info_changed
    for key in arrays:
        if key != "query_actual":
            np.testing.assert_array_equal(arrays[key], changed[key], err_msg=key)


def test_row_reordering_preserves_lambdas_risk_and_selected_id_sets():
    cell = example_cell()
    saved = saved_predictions(cell)
    original, info = evaluate_cell(cell, saved)
    rng = np.random.default_rng(101)
    ci, qi = rng.permutation(len(cell["cal_ids"])), rng.permutation(len(cell["query_ids"]))
    reordered = deepcopy(cell)
    for key in cell:
        if key.startswith("cal_"):
            reordered[key] = np.asarray(cell[key])[ci]
        elif key.startswith("query_") and key != "query_budget":
            reordered[key] = np.asarray(cell[key])[qi]
    for arm in ARMS:
        for key, value in cell["arms"][arm].items():
            reordered["arms"][arm][key] = value[ci if key.startswith("cal_") else qi]
    replay, info2 = evaluate_cell(reordered, saved)
    assert info == info2
    for arm in ARMS:
        np.testing.assert_array_equal(replay[arm + "__CAL_SELECTED_GRID"], original[arm + "__CAL_SELECTED_GRID"][:, ci])
        for variant in VARIANTS:
            np.testing.assert_array_equal(replay[arm + "__" + variant], original[arm + "__" + variant][qi])


def test_cal_selections_never_compare_scores_between_held_folds():
    cell = example_cell()
    budget = validate_cell(cell)
    values = cell["arms"]["CORE"]
    first = select_cal_by_fold(cell["cal_ids"], values["cal_mean"], values["cal_p"],
                               cell["cal_inner_fold"], budget, .2)
    changed_mean = values["cal_mean"].copy()
    held = cell["cal_inner_fold"] == 0
    changed_mean[~held] += 1000.
    second = select_cal_by_fold(cell["cal_ids"], changed_mean, values["cal_p"],
                                cell["cal_inner_fold"], budget, .2)
    np.testing.assert_array_equal(first[held], second[held])
    assert first[held].sum() == second[held].sum() == budget["fold_budgets"]["0"]


def test_group_isolation_and_saved_baseline_changes_are_rejected():
    cell = example_cell()
    cell["query_groups"][0] = cell["cal_groups"][0]
    with pytest.raises(ValueError, match="chemistry overlap"):
        validate_cell(cell)
    cell = example_cell()
    cell["cal_groups"][0] = cell["cal_groups"][9]
    with pytest.raises(ValueError, match="split across"):
        validate_cell(cell)
    cell = example_cell()
    saved = saved_predictions(cell)
    saved["CORE__original_selected"][0] ^= True
    with pytest.raises(AssertionError, match="Changed original query selection"):
        evaluate_cell(cell, saved)


def test_cached_mc_replay_uses_frozen_lambda_and_no_probability_correction():
    cell = example_cell()
    base = cell["arms"]["CORE"]
    base["query_seed_p"] = np.stack([base["query_p"], base["query_p"] * .8])
    base["query_seed_mean"] = np.stack([base["query_mean"], base["query_mean"] + .01])
    arrays, info = evaluate_cell(cell, saved_predictions(cell))
    for variant in VARIANTS:
        np.testing.assert_array_equal(arrays["CORE__MC0_" + variant], arrays["CORE__" + variant])
        np.testing.assert_array_equal(arrays["CORE__query_selected_MC0_" + variant],
                                      arrays["CORE__MC0_" + variant])
    np.testing.assert_array_equal(arrays["CORE__query_seed_p"], base["query_seed_p"])
    assert info["arms"]["CORE"]["MC_lambda_refitted"] is False
    assert info["new_monte_carlo_draws"] == 0


def test_support_diagnostics_count_alias_groups_without_calling_them_independent():
    cell = example_cell()
    cell["cal_groups"][1] = cell["cal_groups"][0]
    _, info = evaluate_cell(cell, saved_predictions(cell))
    assert info["unique_cal_ids"] == 40
    assert info["unique_cal_groups"] == 39
    support = info["arms"]["CORE"]["cal_support"]
    assert support["all"]["unique_groups"] == 39
    assert support["top_0.125"]["n"] == 5
    assert support["top_0.25"]["n"] == 10
