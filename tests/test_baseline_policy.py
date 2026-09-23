"""Synthetic accounting fixtures, not biological experimental evidence."""
import json

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from opal2.baseline_policy import ACTIONS, evaluate_predictions, stable_top_k


def fixture(n=20):
    ids = np.array([f"unit_{i:03d}" for i in range(n)])
    actual = np.column_stack((np.linspace(-.03, .04, n), np.linspace(.03, -.02, n),
                              np.linspace(-.05, .05, n)))
    return actual + .001, (actual <= 0).astype(float), actual, ids


def evaluate(*args, **kwargs):
    return evaluate_predictions(*args, n_random=19, n_bootstrap=23, **kwargs)


def row(result, section="within_action", action="Z1", fraction=.25, ranking="expected_gain"):
    return next(r for r in result[section]
                if r["action"] == action and r["fraction"] == fraction and r["ranking"] == ranking)


def test_selected_gain_is_not_whole_population_policy_value():
    mean, pn, actual, ids = fixture(100)
    actual[:25, 0] = .0184
    mean[:25, 0] = 1.
    result = evaluate(mean, pn, actual, ids, fractions=(.25,))
    selected = row(result)
    assert selected["selected_n"] == 25
    assert selected["per_selected_net_gain"] == pytest.approx(.0184)
    assert selected["per_eligible_net_gain"] == pytest.approx(.0046)
    assert selected["used_wells"] == 25
    assert selected["total_net_gain"] == pytest.approx(.46)
    assert selected["selected_null_count"] == 0
    assert not selected["formal_certificate"]


def test_common_well_cap_does_not_award_double_budget_to_two_well_action():
    result = evaluate(*fixture(100), fractions=(.25,))
    single = row(result, "common_budget")
    double = row(result, "common_budget", action="Z1Z2")
    within_double = row(result, action="Z1Z2")
    assert single["budget_wells"] == double["budget_wells"] == 25
    assert single["selected_n"] == 25
    assert double["selected_n"] == 12
    assert double["used_wells"] == 24
    assert double["unused_wells"] == 1
    assert within_double["selected_n"] == 25
    assert within_double["used_wells"] == 50
    assert all(r["used_wells"] <= r["budget_wells"] for r in result["common_budget"])
    all_double = next(r for r in result["fixed_policies"] if r["action"] == "Z1Z2")
    assert all_double["used_wells"] == 200 and not all_double["budget_matched"]


def test_tied_selection_is_id_stable_and_metrics_use_standard_ties():
    scores = np.array([2., 2., 1., 1.])
    ids = np.array(["b", "a", "d", "c"])
    assert ids[stable_top_k(scores, ids, 1)].tolist() == ["a"]
    actual = np.tile(np.array([.01, -.01, .03, -.03])[:, None], (1, 3))
    mean = np.tile(scores[:, None], (1, 3))
    pnull = np.tile(np.array([.5, .5, .8, .8])[:, None], (1, 3))
    result = evaluate(mean, pnull, actual, ids, fractions=(.25,))
    assert row(result)["selected_ids"] == ["a"]
    assert result["action_metrics"][0]["spearman"] == pytest.approx(spearmanr(actual[:, 0], mean[:, 0]).statistic)
    assert result["action_metrics"][0]["null_auc"] == pytest.approx(roc_auc_score(actual[:, 0] <= 0, pnull[:, 0]))


def test_row_reordering_changes_only_input_row_trace():
    mean, pn, actual, ids = fixture()
    mean[:] = .1
    first = evaluate(mean, pn, actual, ids, seed=13)
    perm = np.array([8, 4, 3, 5, 2, 0, 12, 6, 19, 7, 1, 10, 9, 14, 18, 15, 11, 16, 17, 13])
    second = evaluate(mean[perm], pn[perm], actual[perm], ids[perm], seed=13)
    assert first["within_action"] == second["within_action"]
    assert first["common_budget"] == second["common_budget"]
    for trace in second["row_trace"]:
        i = trace["input_row"]
        assert trace["compound_id"] == ids[perm][i]
        assert trace["actual"]["Z1"] == actual[perm][i, 0]


def test_random_expectation_and_paired_accounting_are_exact():
    mean, pn, actual, ids = fixture()
    result = evaluate(mean, pn, actual, ids, fractions=(.25,))
    selected = row(result)
    random = selected["matched_random"]["exact_expectation"]
    assert random["expected_total_net_gain"] == pytest.approx(5 * actual[:, 0].mean())
    assert random["expected_fdp"] == pytest.approx((actual[:, 0] <= 0).mean())
    assert random["expected_fpr"] == .25
    paired = selected["matched_random"]["paired_bootstrap_vs_random_expectation"]
    assert paired["per_eligible_difference"] == pytest.approx(
        selected["per_eligible_net_gain"] - .25 * actual[:, 0].mean())
    assert paired["per_selected_difference"] == pytest.approx(
        selected["per_selected_net_gain"] - actual[:, 0].mean())
    assert not paired["formal_certificate"] and paired["selection_is_frozen"]
    assert "shared fixed batches" in paired["scope"]
    assert "not population confidence" in selected["matched_random"]["finite_population_randomization"]["scope"]


def test_train_selected_fixed_action_cannot_choose_evaluation_best():
    mean, pn, actual, ids = fixture()
    actual[:, 2] = 100  # The evaluation-best action must not change the choice.
    train = np.tile([.03, .01, .04], (8, 1))
    result = evaluate(mean, pn, actual, ids, train_actual=train, fractions=(.25,))
    fixed = result["train_selected_fixed_action"]
    assert fixed["choice"] == "Z1"  # .03 per well beats .04/2.
    assert not fixed["selection_uses_evaluation_outcomes"]
    budget = fixed["budgets"][0]
    assert budget["expected_total_net_gain"] == pytest.approx(5 * actual[:, 0].mean())
    assert budget["uniform_random_subset_exact_expectation"]
    stopped = evaluate(mean, pn, actual, ids, train_actual=-np.ones((4, 3)))
    assert stopped["train_selected_fixed_action"]["choice"] == "STOP"
    assert all(r["used_wells"] == 0 for r in stopped["train_selected_fixed_action"]["budgets"])


def test_original_labels_missing_class_and_zero_budget_are_explicit():
    mean, pn, actual, ids = fixture(4)
    actual[:, 0] = [0., .0049, .005, .006]
    mean[:, 0] = actual[:, 0]
    result = evaluate(mean, pn, actual, ids, fractions=(.05, 1.))
    metric = result["action_metrics"][0]
    assert metric["null_count"] == 1
    assert metric["positive_count"] == 2
    assert metric["ambiguous_count"] == 1
    zero = row(result, fraction=.05)
    assert zero["selected_n"] == 0 and zero["fdp"] is None
    assert zero["per_selected_net_gain"] is None
    assert zero["per_eligible_net_gain"] == 0
    all_active = row(result, fraction=1.)
    assert all_active["fpr"] == all_active["sensitivity"] == 1
    assert all_active["fdp"] == .25
    degenerate = evaluate(np.zeros((4, 3)), np.ones((4, 3)) * .5, np.ones((4, 3)), ids)
    assert degenerate["action_metrics"][0]["null_auc"] is None
    assert degenerate["action_metrics"][0]["spearman"] is None


def test_contract_is_preserved_without_claim_of_certification_and_json_finite():
    contract = {"min_activations": 100, "max_fdp": .35, "max_fpr": .075}
    result = evaluate(*fixture(), original_contract=contract)
    assert result["original_contract"] == contract
    result["original_contract"]["max_fdp"] = 1
    assert contract["max_fdp"] == .35
    assert not result["original_contract_changed"] and not result["formal_certificate"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure", ["duplicate_id", "blank_id", "numeric_id", "nonfinite", "badprob", "wrongshape", "wrongrows"])
def test_bad_or_ambiguous_rows_fail_without_dropping_objects(failure):
    mean, pn, actual, ids = fixture()
    if failure == "duplicate_id":
        ids[0] = ids[1]
    elif failure == "blank_id":
        ids[0] = ""
    elif failure == "numeric_id":
        ids = np.arange(len(ids))
    elif failure == "nonfinite":
        actual[0, 0] = np.nan
    elif failure == "badprob":
        pn[0, 0] = 1.1
    elif failure == "wrongshape":
        actual = actual[:, :2]
    else:
        mean = mean[:-1]
    with pytest.raises(ValueError):
        evaluate(mean, pn, actual, ids)
