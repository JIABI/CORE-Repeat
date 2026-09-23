"""Exact weighted-distribution mechanics; no biological-data test fixture."""
import inspect

import numpy as np
import pytest

from opal2.neighbor_response_diagnostic import (fixed_neighbor_weights, weighted_crps,
    weighted_energy_score, weighted_quantiles, response_targets, uniform_policy_expectations)


def test_weighted_crps_equals_explicit_discrete_double_sum_with_ties():
    atoms = np.array([[0., 1.], [0., -2.], [3., 1.]])
    observed = np.array([[.5, 2.], [5., -3.]])
    weights = np.array([[.2, .3, .5], [0., .8, .2]])
    expected = np.empty_like(observed)
    for q in range(2):
        for j in range(2):
            expected[q,j] = (weights[q]*np.abs(atoms[:,j]-observed[q,j])).sum()
            expected[q,j] -= .5 * np.sum(weights[q,:,None]*weights[q,None,:]
                * np.abs(atoms[:,None,j]-atoms[None,:,j]))
    np.testing.assert_allclose(weighted_crps(atoms, observed, weights), expected, atol=1e-14)
    # The declared finite empirical law is not given an IID-MC correction.
    assert weighted_crps(np.array([0., 2.]), np.array([1.]), np.array([[.5,.5]]))[0,0] == .5


def test_exact_energy_score_retains_joint_atoms_not_independent_marginals():
    atoms = np.array([[-1.,-1.], [1.,1.]])
    observed = np.array([[0.,0.]])
    weights = np.array([[.5,.5]])
    assert weighted_energy_score(atoms, observed, weights)[0] == pytest.approx(np.sqrt(2)/2)
    independent_splice = np.array([[-1.,-1.], [-1.,1.], [1.,-1.], [1.,1.]])
    spliced_score = weighted_energy_score(independent_splice, observed, np.full((1,4), .25))[0]
    assert spliced_score != pytest.approx(weighted_energy_score(atoms, observed, weights)[0])


def test_fixed_weights_exclude_equal_ids_and_do_not_accept_targets():
    distance = np.array([[0., .1, .3, .7], [.1, .1, .4, .5]])
    donors, queries = ["D", "B", "C", "A"], ["D", "Q"]
    weights = fixed_neighbor_weights(distance, queries, donors, bandwidth=.3, k=2)
    assert weights[0,0] == 0
    assert np.count_nonzero(weights[0]) == 2
    np.testing.assert_allclose(weights.sum(1), 1.)
    np.testing.assert_allclose(weights[0,1]/weights[0,2], np.exp((.3-.1)/.3))
    assert set(inspect.signature(fixed_neighbor_weights).parameters) == {
        "distances", "query_ids", "donor_ids", "bandwidth", "k"}
    with pytest.raises(TypeError):
        fixed_neighbor_weights(distance, queries, donors, .3, k=2, gamma=np.ones(2))


def test_weighted_quantile_obeys_atoms_and_zero_weight_entries():
    atoms = np.array([[0.,10.], [1.,20.], [2.,30.]])
    weights = np.array([[0., .25, .75], [1.,0.,0.]])
    result = weighted_quantiles(atoms, weights, probabilities=(.1,.5,.9))
    np.testing.assert_array_equal(result[:,0,0], [1,2,2])
    np.testing.assert_array_equal(result[:,1,1], [10,10,10])


def test_future_responses_and_pair_contrasts_keep_distinct_meanings():
    y = np.array([[[0.,0.], [2.,4.], [4.,8.], [6.,12.]]])
    result = response_targets(y, np.array([1.,2.]))
    np.testing.assert_allclose(result["future_mean_profile"], [[4.,8.]])
    np.testing.assert_allclose(result["future_mean_minus_X"], [[4.,8.]])
    np.testing.assert_allclose(result["repeat_contrast_energy"], [[10.,40.,10.]])
    np.testing.assert_allclose(result["standardized_repeat_contrast_energy"], [[4.,16.,4.]])
    shifted = y+100
    other = response_targets(shifted, np.array([1.,2.]))
    np.testing.assert_allclose(other["repeat_contrast_energy"], result["repeat_contrast_energy"])
    np.testing.assert_allclose(other["future_mean_profile"], result["future_mean_profile"]+100)


def test_nonfinite_or_unnormalized_distribution_is_not_silently_repaired():
    with pytest.raises(ValueError, match="row-normalized"):
        weighted_crps(np.array([0.,1.]), np.array([.5]), np.array([[1.,1.]]))
    with pytest.raises(ValueError, match="Finite donor"):
        weighted_crps(np.array([0.,np.nan]), np.array([.5]), np.array([[.5,.5]]))
    with pytest.raises(ValueError, match="Insufficient"):
        fixed_neighbor_weights(np.zeros((1,2)), ["Q"], ["Q","A"], 1., k=2)


def test_global_uniform_policy_replaces_arbitrary_subset_by_exact_expectation():
    row = dict(action="Z1Z2", fraction=.25, budget_wells=6, budget_scope="common physical-well cap",
        eligible_n=10, selected_n=3, used_wells=6, unused_wells=0, coverage=.3,
        ranking="expected_gain", selected_ids=["A","B","C"], total_net_gain=100,
        selected_null_count=0, fdp=0)
    raw = dict(action_metrics=[dict(action="Z1Z2", n=10, actual_mean=.02, null_count=4, positive_count=5)],
        within_action=[row.copy()], common_budget=[row.copy(),{**row,"ranking":"lowest_p_null"}],
        fixed_policies=[dict(action="STOP")], statistical_scope={"score_ties":"lexical"}, row_trace=["arbitrary"])
    result = uniform_policy_expectations(raw)
    assert len(result["common_budget"]) == 1
    expected = result["common_budget"][0]
    assert expected["selected_ids"] is None
    assert expected["expected_selected_null_count"] == pytest.approx(1.2)
    assert expected["expected_selected_positive_count"] == pytest.approx(1.5)
    assert expected["expected_selected_ambiguous_count"] == pytest.approx(.3)
    assert expected["expected_total_net_gain"] == pytest.approx(.06)
    assert expected["expected_fdp"] == pytest.approx(.4)
    assert expected["expected_fpr"] == pytest.approx(.3)
    assert expected["expected_sensitivity"] == pytest.approx(.3)
    assert "total_net_gain" not in expected and "fdp" not in expected
    assert result["row_trace"] is None
    assert raw["common_budget"][0]["selected_ids"] == ["A","B","C"]
