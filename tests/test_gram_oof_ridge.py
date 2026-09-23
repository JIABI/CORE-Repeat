"""Synthetic tests of nested preprocessing and complete ridge inference."""
import inspect
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from opal2.gram_oof_ridge import (
    fit_preprocessing, transform_input, transform_target, fit_ridge_oof,
)
from opal2.gram_simple_models import GramSimpleGaussian


@pytest.fixture(autouse=True)
def bounded_blas():
    with threadpool_limits(limits=1):
        yield


def fixture(n=35, d=7, seed=51):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n, 4, d))+np.arange(4)[None, :, None]*.2
    u = y[:, 0]@rng.normal(size=(d, 9))+rng.normal(size=(n, 9))*.15
    u = u*np.linspace(.5, 7., 9)+np.arange(9)*3.
    return y, u


def test_preprocessing_matches_all_four_role_population_moments_and_input_precision():
    y, u = fixture()
    y[:, :, 1] = 5.
    stats = fit_preprocessing(y, u)
    restored = json.loads(json.dumps(stats))
    np.testing.assert_allclose(stats['y_center'], y.reshape(-1, y.shape[-1]).mean(0))
    expected_scale = y.reshape(-1, y.shape[-1]).std(0)
    expected_scale[expected_scale < 1e-6] = 1.
    np.testing.assert_allclose(stats['y_scale'], expected_scale)
    expected_x = ((y[:, 0]-stats['y_center'])/stats['y_scale']).astype(np.float32).astype(np.float64)
    actual_x = transform_input(y[:, 0], restored)
    assert actual_x.shape == (len(y), y.shape[-1]+1)
    np.testing.assert_array_equal(actual_x[:, :-1], expected_x)
    np.testing.assert_allclose(transform_target(u, stats)*stats['u_scale']+stats['u_center'], u)


def test_nested_oof_complete_joint_covariance_and_final_ridge_convention():
    y, u = fixture()
    model, stats = fit_ridge_oof(y, u, seed=14)
    x, target = transform_input(y[:, 0], stats), transform_target(u, stats)
    reference = Ridge(alpha=len(y)*model.selected_lambda).fit(x, target)
    np.testing.assert_allclose(model.coefficient, reference.coef_.T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model.intercept, reference.intercept_, atol=1e-10, rtol=1e-10)
    audit = model.audit_arrays
    np.testing.assert_array_equal(audit['oof_count'], np.ones(len(y), dtype=int))
    np.testing.assert_allclose(audit['oof_predictions'], transform_target(audit['oof_native_predictions'], stats))
    np.testing.assert_allclose(audit['oof_residuals'], target-audit['oof_predictions'])
    np.testing.assert_allclose(model.covariance, audit['centered_residual_covariance']+
                               np.outer(audit['residual_mean'], audit['residual_mean']))
    assert np.linalg.eigvalsh(model.covariance).min() > 0
    assert model.coefficient.shape == (y.shape[-1]+1, 9)
    assert model.metadata['preprocessing_fits'] == 26
    assert not model.metadata['prediction_bias_correction']


def test_every_inner_preprocessing_excludes_both_held_out_sets():
    y, u = fixture(n=30)
    model, _ = fit_ridge_oof(y, u, seed=17)
    for outer in model.metadata['outer_cv']:
        check = set(outer['outer_validation_indices'])
        outer_fit = set(outer['outer_fit_indices'])
        seen = []
        for inner in outer['inner_folds']:
            ii, jj = inner['fit_indices'], inner['validation_indices']
            assert set(ii).isdisjoint(jj)
            assert (set(ii) | set(jj)) == outer_fit
            assert (set(ii) | set(jj)).isdisjoint(check)
            stats = fit_preprocessing(y[ii], u[ii])
            for key, value in stats.items():
                np.testing.assert_array_equal(model.audit_arrays[inner['preprocessing_audit_prefix']+'__'+key], value)
            seen.extend(jj)
        assert sorted(seen) == sorted(outer_fit)


def test_query_future_values_cannot_change_own_native_oof_prediction():
    y, u = fixture(n=30, d=6)
    original, _ = fit_ridge_oof(y, u, seed=16)
    y2, u2 = y.copy(), u.copy()
    y2[2, 1:] += 900.
    u2[2] += np.arange(9)*300.
    changed, _ = fit_ridge_oof(y2, u2, seed=16)
    # Full fitting coordinates/covariance may change; the held-out native mean
    # must not, since its X and all eligible fitting objects are unchanged.
    np.testing.assert_array_equal(original.audit_arrays['oof_native_predictions'][2],
                                  changed.audit_arrays['oof_native_predictions'][2])
    fold = original.audit_arrays['outer_fold_membership'][2]
    assert original.metadata['outer_cv'][fold]['selected_lambda'] == changed.metadata['outer_cv'][fold]['selected_lambda']


def test_model_and_preprocessing_roundtrip_preserves_predictions(tmp_path):
    y, u = fixture(n=30)
    model, stats = fit_ridge_oof(y, u)
    path = tmp_path/'model.bin'
    model.save(path)
    restored = GramSimpleGaussian.load(path)
    x = transform_input(y[:3, 0], json.loads(json.dumps(stats)))
    np.testing.assert_array_equal(restored.predict_mean(x), model.predict_mean(x))
    np.testing.assert_array_equal(restored.sample_coordinates(x, 11, 4), model.sample_coordinates(x, 11, 4))
    assert tuple(inspect.signature(restored.predict_mean).parameters) == ('x',)
    assert restored.metadata == model.metadata


def test_invalid_inputs_and_zero_norm_are_rejected_without_repair():
    y, u = fixture(n=10)
    with pytest.raises(ValueError, match=r'\[N,4,D\]'):
        fit_preprocessing(y[:, :3], u)
    with pytest.raises(ValueError, match='same ordered'):
        fit_ridge_oof(y, u[:-1])
    stats = fit_preprocessing(y, u)
    with pytest.raises(ValueError, match='positive norm'):
        transform_input(np.zeros_like(y[:, 0]), stats)
    assert set(stats) == {'y_center','y_scale','u_center','u_scale','lognorm_center','lognorm_scale'}
