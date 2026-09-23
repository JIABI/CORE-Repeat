"""Selection-information tests using synthetic arrays, not experimental rows."""
import inspect
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from opal2.gram_oof_ridge import fit_preprocessing, transform_input, transform_target
from opal2.gram_simple_models import GramSimpleGaussian, REGULARIZATION_GRID
from opal2.hierarchical_stability_ridge import fit_validation_ridge


@pytest.fixture(autouse=True)
def bounded_blas():
    with threadpool_limits(limits=1):
        yield


def fixture(seed=19, n=35, nv=13, d=7):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n, 4, d)) + np.arange(4)[None, :, None] * .15
    vy = rng.normal(size=(nv, 4, d)) + np.arange(4)[None, :, None] * .15
    beta = rng.normal(size=(d, 9))
    scales = np.linspace(.4, 4, 9)
    u = (y[:, 0] @ beta + rng.normal(size=(n, 9)) * .3) * scales + np.arange(9)
    vu = (vy[:, 0] @ beta + rng.normal(size=(nv, 9)) * .3) * scales + np.arange(9)
    return y, u, vy, vu


def test_external_validation_selects_full_grid_and_final_fit_matches_sklearn():
    y, u, vy, vu = fixture()
    model, stats = fit_validation_ridge(y, u, vy, vu, seed=9)
    x, target = transform_input(y[:, 0], stats), transform_target(u, stats)
    vx, vt = transform_input(vy[:, 0], stats), transform_target(vu, stats)
    expected_errors, references = [], []
    for penalty in REGULARIZATION_GRID:
        reference = Ridge(alpha=len(y)*penalty).fit(x, target)
        references.append(reference)
        expected_errors.append(np.square(reference.predict(vx)-vt).mean(-1))
    np.testing.assert_allclose(model.audit_arrays['final_validation_candidate_per_object_mse'],
                               expected_errors, atol=1e-11, rtol=1e-10)
    chosen = REGULARIZATION_GRID.index(model.selected_lambda)
    assert chosen == np.argmin(np.mean(expected_errors, axis=1))
    np.testing.assert_allclose(model.coefficient, references[chosen].coef_.T, atol=1e-11)
    np.testing.assert_allclose(model.intercept, references[chosen].intercept_, atol=1e-11)
    assert stats == fit_preprocessing(y, u)
    assert model.metadata['selection_information_differs_from_train_only_ridge'] is True


def test_same_external_labels_used_per_error_fold_without_error_holdout_leakage():
    y, u, vy, vu = fixture(n=30)
    model, stats = fit_validation_ridge(y, u, vy, vu, seed=5)
    audit = model.audit_arrays
    np.testing.assert_array_equal(audit['oof_count'], np.ones(len(y), dtype=int))
    for fold in model.metadata['internal_error_folds']:
        fit, error = fold['fit_indices'], fold['error_indices']
        assert set(fit).isdisjoint(error) and set(fit) | set(error) == set(range(len(y)))
        assert fold['validation_indices'] == list(range(len(vy)))
        inner_stats = fit_preprocessing(y[fit], u[fit])
        for key, value in inner_stats.items():
            np.testing.assert_array_equal(audit[fold['preprocessing_audit_prefix']+'__'+key], value)
    np.testing.assert_allclose(audit['oof_predictions'], transform_target(audit['oof_native_predictions'], stats))
    np.testing.assert_allclose(audit['oof_residuals'], transform_target(u, stats)-audit['oof_predictions'])
    np.testing.assert_allclose(model.covariance, audit['centered_residual_covariance']+
                               np.outer(audit['residual_mean'], audit['residual_mean']))
    assert np.linalg.eigvalsh(model.covariance).min() > 0
    y2, u2 = y.copy(), u.copy()
    y2[2, 1:] += 100
    u2[2] += np.arange(9)*40
    alternative, _ = fit_validation_ridge(y2, u2, vy, vu, seed=5)
    np.testing.assert_array_equal(audit['oof_native_predictions'][2],
                                  alternative.audit_arrays['oof_native_predictions'][2])


def test_validation_labels_change_selection_but_future_validation_profiles_do_not():
    y, u, vy, vu = fixture()
    stats = fit_preprocessing(y, u)
    x, target = transform_input(y[:, 0], stats), transform_target(u, stats)
    vx = transform_input(vy[:, 0], stats)
    for penalty in (REGULARIZATION_GRID[0], REGULARIZATION_GRID[-1]):
        fitted = Ridge(alpha=len(y)*penalty).fit(x, target)
        native_labels = fitted.predict(vx)*stats['u_scale'] + stats['u_center']
        model, actual_stats = fit_validation_ridge(y, u, vy, native_labels, seed=12)
        assert model.selected_lambda == penalty
        assert actual_stats == stats
    original, _ = fit_validation_ridge(y, u, vy, vu, seed=12)
    vy2 = vy.copy(); vy2[:, 1:] += 1000
    changed, _ = fit_validation_ridge(y, u, vy2, vu, seed=12)
    np.testing.assert_array_equal(original.coefficient, changed.coefficient)
    np.testing.assert_array_equal(original.covariance, changed.covariance)


def test_ties_choose_larger_lambda_and_serialization_keeps_query_label_free_api(tmp_path):
    y, u, vy, vu = fixture()
    # Constant fitting inputs give a mean-only map for every penalty; targets
    # still vary, so the uncertainty model has nondegenerate residuals.
    tied_y = y.copy(); tied_y[:, 0] = 1.
    tie_model, _ = fit_validation_ridge(tied_y, u, vy, vu, seed=12)
    assert tie_model.selected_lambda == max(REGULARIZATION_GRID)
    assert set(tie_model.audit_arrays['inner_selected_lambdas']) == {max(REGULARIZATION_GRID)}
    model, stats = fit_validation_ridge(y, u, vy, vu, seed=12)
    filename = tmp_path/'model.npz'
    model.save(filename)
    restored = GramSimpleGaussian.load(filename)
    query = transform_input(vy[:, 0], json.loads(json.dumps(stats)))
    np.testing.assert_array_equal(restored.predict_mean(query), model.predict_mean(query))
    assert restored.metadata == model.metadata
    assert tuple(inspect.signature(restored.predict_mean).parameters) == ('x',)


def test_invalid_partition_shapes_and_nonfinite_values_are_rejected():
    y, u, vy, vu = fixture(n=10)
    with pytest.raises(ValueError, match='same ordered'):
        fit_validation_ridge(y, u, vy, vu[:-1])
    with pytest.raises(ValueError, match='same feature dimension'):
        fit_validation_ridge(y, u, vy[:, :, :-1], vu)
    bad = vu.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        fit_validation_ridge(y, u, vy, bad)
