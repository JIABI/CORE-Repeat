"""Algebra, isolation interfaces and numerical checks, not efficacy evidence."""
import inspect

import numpy as np
import pytest

from opal2.crossdose_response import (
    apply_response_correction, fit_convex_strength, fit_ridge_response,
    profile_scores, response_candidates,
)


def test_candidates_transport_identity_and_baseline_not_double_counted():
    xq = np.array([[1., 3.], [5., -2.], [2., 8.]])
    xr = np.array([[2., 1.], [4., -3.]])
    target = xr + np.array([10., -2.])
    weights = np.array([[1., 0.], [.25, .75], [0., 0.]])
    support = np.array([True, True, False])
    out = response_candidates(xq, xr, target, weights, support)
    np.testing.assert_allclose(out["DIRECT"][:2], weights[:2] @ target)
    np.testing.assert_allclose(out["TRANSPORT"][:2], xq[:2] + [10., -2.])
    identity = response_candidates(xq, xr, xr, weights, support)
    assert np.array_equal(identity["TRANSPORT"][:2], xq[:2])
    baseline = np.full_like(xq, 50.)
    corrected = apply_response_correction(baseline, identity["TRANSPORT"], support, .25)
    np.testing.assert_allclose(corrected[:2], .75 * baseline[:2] + .25 * xq[:2])
    assert np.array_equal(corrected[~support], baseline[~support])
    assert np.array_equal(apply_response_correction(baseline, out["DIRECT"], support, 0), baseline)


def test_weights_and_shapes_are_checked_and_empty_support_is_safe():
    x = np.ones((3, 2))
    refs = np.empty((0, 2))
    out = response_candidates(x, refs, refs, np.empty((3, 0)), np.zeros(3, bool))
    assert all(np.array_equal(value, np.zeros_like(x)) for value in out.values())
    with pytest.raises(ValueError, match="sum to one"):
        response_candidates(x, x, x, np.zeros((3, 3)), np.ones(3, bool))
    with pytest.raises(ValueError, match="exactly zero"):
        response_candidates(x, x, x, np.eye(3), np.zeros(3, bool))
    with pytest.raises(ValueError, match="nonnegative"):
        response_candidates(x, x, x, -np.eye(3), np.ones(3, bool))


def test_convex_strength_is_group_equal_not_row_equal():
    baseline = np.zeros((5, 2))
    target = np.array([[1., 1.], [1., 1.], [1., 1.], [-1., -1.], [1., 1.]])
    candidate = np.ones_like(target)
    groups = np.array(["a", "a", "a", "b", "c"])
    result = fit_convex_strength(baseline, target, candidate, groups, np.ones(5, bool), one_se=False)
    assert result["n_groups"] == 3
    assert result["alpha"] == pytest.approx(1 / 3)
    assert result["alpha"] != pytest.approx(3 / 5)
    guarded = fit_convex_strength(baseline, target, candidate, groups, np.ones(5, bool))
    assert guarded["alpha"] == 0
    assert guarded["reason"] == "paired improvement below one SE"


def test_supported_only_strength_clipping_insufficient_and_identity():
    base = np.zeros((5, 2))
    candidate, target = np.ones_like(base), np.full_like(base, .4)
    support, groups = np.array([True, True, True, False, False]), np.arange(5)
    target[~support] = 1000
    result = fit_convex_strength(base, target, candidate, groups, support)
    assert result["alpha"] == pytest.approx(.4)
    assert result["se"] == 0
    assert result["n_groups"] == 3
    assert fit_convex_strength(base, 3 * candidate, candidate, groups, support)["alpha"] == 1
    assert fit_convex_strength(base, -candidate, candidate, groups, support)["alpha"] == 0
    assert fit_convex_strength(base, target, base, groups, support)["alpha"] == 0
    empty = fit_convex_strength(base, target, candidate, groups, np.zeros(5, bool))
    assert empty["alpha"] == 0 and empty["n_groups"] == 0
    assert empty["se"] is None


def test_scores_declare_zero_norm_rule_and_no_rows_are_dropped():
    target = np.array([[1., 0.], [0., 0.], [1., 0.], [0., 0.]])
    prediction = np.array([[1., 0.], [0., 0.], [-1., 0.], [1., 0.]])
    score = profile_scores(prediction, target)
    np.testing.assert_array_equal(score["cosine_loss"], [0., 1., 2., 1.])
    np.testing.assert_array_equal(score["profile_mse"], [0., 0., 2., .5])
    assert np.isfinite(score["lognorm_squared_error"]).all()
    assert score["metadata"]["epsilon"] == 1e-12
    assert np.sum(score["prediction_zero_norm"]) == 1
    assert np.sum(score["target_zero_norm"]) == 2


@pytest.mark.parametrize("n,p", [(12, 5), (5, 12)])
def test_ridge_matches_direct_linear_solve_for_primal_and_dual(n, p):
    rng = np.random.default_rng(741)
    train_x, train_y = rng.normal(size=(n, p)), rng.normal(size=(n, 3))
    valid_x, valid_y = rng.normal(size=(4, p)), rng.normal(size=(4, 3))
    model = fit_ridge_response(train_x, train_y, valid_x, valid_y, lambdas=(.01, .1, 1.))
    x = (train_x - train_x.mean(0)) / train_x.std(0)
    y = train_y - train_y.mean(0)
    coefficient = np.linalg.solve(x.T @ x + n * model.selected_lambda * np.eye(p), x.T @ y)
    np.testing.assert_allclose(model.coefficient, coefficient, atol=1e-12)
    np.testing.assert_allclose(model.predict(valid_x),
        ((valid_x - train_x.mean(0)) / train_x.std(0)) @ coefficient + train_y.mean(0), atol=1e-12)
    assert model.report["model"] == "RIDGE_RESPONSE"
    assert model.report["eigensystem"] == ("dual" if n <= p else "primal")


def test_ridge_train_only_normalization_and_validation_only_penalty():
    rng = np.random.default_rng(38)
    train_x, train_y = rng.normal(size=(10, 3)), rng.normal(size=(10, 2))
    train_x[:, 2] = 4.
    vx, vy = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))
    first = fit_ridge_response(train_x, train_y, vx, vy, lambdas=(.1,))
    second = fit_ridge_response(train_x, train_y, vx + 100, vy + 1000, lambdas=(.1,))
    assert np.array_equal(first.coefficient, second.coefficient)
    assert np.array_equal(first.input_center, train_x.mean(0))
    assert first.input_scale[2] == 1
    assert np.array_equal(first.target_center, train_y.mean(0))
    assert np.array_equal(first.coefficient[2], np.zeros(2))


def test_ridge_group_equal_training_unchanged_by_repeating_one_identity():
    rng = np.random.default_rng(765)
    tx, ty = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))
    vx, vy = rng.normal(size=(4, 4)), rng.normal(size=(4, 3))
    group = np.array(["a", "b", "c", "d", "e", "f"])
    first = fit_ridge_response(tx, ty, vx, vy, lambdas=(.1,), training_groups=group)
    indices = np.array([0, 0, 0, 1, 2, 3, 4, 5])
    second = fit_ridge_response(tx[indices], ty[indices], vx, vy,
                                lambdas=(.1,), training_groups=group[indices])
    np.testing.assert_allclose(first.input_center, second.input_center, atol=1e-14)
    np.testing.assert_allclose(first.input_scale, second.input_scale, atol=1e-14)
    np.testing.assert_allclose(first.target_center, second.target_center, atol=1e-14)
    np.testing.assert_allclose(first.coefficient, second.coefficient, atol=1e-12)
    np.testing.assert_allclose(first.predict(vx), second.predict(vx), atol=1e-12)
    assert second.report["n_train_groups"] == 6
    assert second.report["training_weighting"] == "group equal"


def test_prediction_interfaces_accept_no_query_outcomes():
    assert "target" not in inspect.signature(response_candidates).parameters
    assert "target" not in inspect.signature(apply_response_correction).parameters
    assert set(inspect.signature(fit_convex_strength).parameters) == {
        "baseline", "target", "candidate", "groups", "support", "one_se", "min_groups"}
