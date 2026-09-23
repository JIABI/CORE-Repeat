"""Numerical unit tests, not biological experiments or manuscript evidence."""
from itertools import product

import numpy as np
import pytest

from opal2.utility import (Action, UtilityResult, average_precision, cosine,
                          cosine_utility_samples, enumerate_actions, fixed_strategy,
                          random_strategy, retrieval_utility_samples)
from opal2.planning import allocate_actions, finite_depth_probe_plan


def test_original_gamma_and_probability_coherence():
    x = np.array([[1., 0.], [0., 1.]])
    y = np.array([[[[0., 1.], [1., 1.], [1., 1.]],
                   [[1., 0.], [1., 1.], [1., 1.]]]])
    actions = enumerate_actions([0, 1])
    result = cosine_utility_samples(x, y, actions, 2)
    wanted = 0.5 * (cosine((x + y[0, :, 0] + y[0, :, 1]) / 3, y[0, :, 2])
                    - cosine(x, y[0, :, 2])) - 0.02
    np.testing.assert_allclose(result.samples[0, :, 3], wanted)
    np.testing.assert_allclose(result.costs, [0, .01, .01, .02])
    np.testing.assert_array_equal(result.samples[:, :, 0], 0)
    np.testing.assert_allclose(result.p_null + result.p_ambiguous + result.p_positive, 1)
    with pytest.raises(ValueError, match="validation well"):
        cosine_utility_samples(x, y, enumerate_actions([2]), 2)


def test_joint_dependence_is_preserved_not_replaced_by_marginals():
    x = np.array([[1., 0.]])
    future = np.array([[[[0., 1.], [0., 1.]]], [[[0., -1.], [0., -1.]]]])
    correlated = cosine_utility_samples(x, future, enumerate_actions([0]), 1)
    opposite = future.copy()
    opposite[:, :, 1] *= -1
    anticorrelated = cosine_utility_samples(x, opposite, enumerate_actions([0]), 1)
    # Both single-well empirical marginal distributions are unchanged.
    assert correlated.mean[0, 1] > 0
    assert anticorrelated.mean[0, 1] < 0


def test_incremental_gain_telescopes_with_exact_well_weights():
    x, p, q, v = [np.array([[1., .2]]), np.array([[.2, 1.]]),
                   np.array([[.8, .8]]), np.array([[.4, 1.]])]
    joint = np.stack([p, q, v], axis=1)[None]
    full = cosine_utility_samples(x, joint, [Action("both", (0, 1))], 2)
    first = cosine_utility_samples(x, joint, [Action("probe", (0,))], 2)
    second_joint = np.stack([q, v], axis=1)[None]
    second = cosine_utility_samples((x+p)/2, second_joint, [Action("continue", (0,))], 1, observed_count=2)
    np.testing.assert_allclose(full.samples, first.samples + second.samples, atol=1e-15)


def test_retrieval_is_fixed_gallery_and_rejects_self_reuse():
    x = np.array([[0., 1.]])
    y = np.array([[[[2., 0.]]]])
    args = dict(gallery=np.array([[1., 0.], [0., 1.]]),
                gallery_compound_ids=["A", "B"], gallery_measurement_ids=["a_repeat", "b_repeat"],
                query_compound_ids=["A"], known_measurement_ids=[["a_initial"]],
                future_measurement_ids=np.array([["a_add"]]))
    result = retrieval_utility_samples(x, y, enumerate_actions([0]), **args)
    np.testing.assert_allclose(result.mean[0], [0, .49])
    args["gallery_measurement_ids"] = ["a_add", "b_repeat"]
    with pytest.raises(ValueError, match="Self-reuse"):
        retrieval_utility_samples(x, y, enumerate_actions([0]), **args)
    with pytest.raises(ValueError, match="relevant"):
        average_precision(x[0], np.eye(2), np.zeros(2, bool), ["a", "b"])


def _allocation_problem():
    actions = enumerate_actions([0, 1])
    samples = np.array([
        [[0., .10, -.04, .14], [0., .09, .05, .10], [0., -.02, .03, -.01]],
        [[0., .08, -.02, .12], [0., -.01, .03, .08], [0., -.02, .01, -.03]],
    ])
    return UtilityResult.from_samples(actions, samples, np.array([0, .01, .01, .02]))


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 6])
def test_milp_matches_exhaustive_optimum(budget):
    result = _allocation_problem()
    answer = allocate_actions(result, budget, max_null_fraction=.25)
    feasible = []
    for choices in product(range(len(result.actions)), repeat=3):
        wells = sum(result.actions[j].wells for j in choices)
        active = sum(result.actions[j].wells > 0 for j in choices)
        null = sum(result.p_null[i, j] for i, j in enumerate(choices) if result.actions[j].wells)
        if wells <= budget and null <= .25 * active + 1e-12:
            feasible.append(sum(result.mean[i, j] for i, j in enumerate(choices)))
    np.testing.assert_allclose(answer.expected_total_gain, max(feasible), atol=1e-10)
    assert answer.total_wells <= budget and answer.optimal


def test_fixed_actions_risk_constraints_and_infeasibility():
    result = _allocation_problem()
    answer = allocate_actions(result, 4, fixed_actions={0: "add_0"}, max_expected_null=0)
    assert answer.action_names[0] == "add_0"
    assert answer.expected_null_count == 0
    with pytest.raises(RuntimeError, match="optimal feasible"):
        allocate_actions(result, 0, min_activations=1)
    with pytest.raises(ValueError, match="population"):
        allocate_actions(result, 2, max_false_activation_rate=.1)


def test_fixed_and_random_comparators_respect_budget_and_no_stop_false_count():
    result = _allocation_problem()
    stopped = fixed_strategy(result, "stop")
    assert stopped["expected_null_count"] == 0 and stopped["mean_net_gain"] == 0
    all_one = fixed_strategy(result, "add_0")
    assert all_one["total_wells"] == 3
    a = random_strategy(result, "add_0", 2, np.random.default_rng(4))
    b = random_strategy(result, "add_0", 2, np.random.default_rng(4))
    np.testing.assert_array_equal(a["action_indices"], b["action_indices"])
    assert a["total_wells"] == 2 and a["activations"] == 2


def test_probe_policy_reconditions_and_actually_changes_continuation():
    x = np.array([[1., 0.]])
    probe = np.array([[[0., 1.]], [[1., 0.]]])
    calls = []
    def resample(history, outer_index, purpose):
        calls.append((history.copy(), outer_index, purpose))
        # Only the observed history enters this conditional simulation.
        validator = history[:, 1]
        future = np.stack([np.array([[0., 1.]]), validator], axis=1)
        return np.repeat(future[None], 3, axis=0)
    plan = finite_depth_probe_plan(x, probe, resample, enumerate_actions([0]), 1,
                                  total_budget=2, cost_per_well=.01)
    assert len(calls) == 4
    assert [c[2] for c in calls] == ["selection", "evaluation", "selection", "evaluation"]
    np.testing.assert_array_equal(plan.continuation_action_indices[:, 0], [1, 0])
    np.testing.assert_array_equal(plan.total_wells_by_probe_draw, [2, 1])
    expected_first = .5 * cosine(np.array([[1., 2.]]) / 3, np.array([[0., 1.]])) - .02
    np.testing.assert_allclose(plan.samples[0, :, 0], expected_first.item())
    np.testing.assert_allclose(plan.samples[1, :, 0], -.01)
    assert plan.mean_total_wells == 1.5
    with pytest.raises(ValueError, match="pay"):
        finite_depth_probe_plan(x, probe, resample, enumerate_actions([0]), 1, total_budget=0)


def test_nonfinite_and_invalid_inputs_rejected():
    with pytest.raises(ValueError, match="distinct"):
        enumerate_actions([0, 0])
    with pytest.raises(ValueError, match="Nonfinite"):
        cosine_utility_samples(np.ones((1, 2)), np.full((2, 1, 2, 2), np.nan), enumerate_actions([0]), 1)
    with pytest.raises(ValueError, match="observed_count"):
        cosine_utility_samples(np.ones((1, 2)), np.ones((2, 1, 2, 2)), enumerate_actions([0]), 1, observed_count=0)
