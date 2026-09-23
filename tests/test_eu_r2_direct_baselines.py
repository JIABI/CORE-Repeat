"""Full-recipe scalar baselines on synthetic arrays, not assay experiments."""
import inspect

import joblib
import numpy as np
import pytest

from opal2.eu_r2_direct_baselines import (
    ARMS, BOUNDS, HIST_ITERATIONS, TREE_COUNT, ConstantNullClassifier,
    empirical_gamma_support, exact_empirical_crps,
    evaluate_direct_distribution, fit_direct_baselines,
)


def synthetic():
    rng = np.random.default_rng(923)
    counts = (80, 24, 20, 9)
    x = [rng.normal(size=(n, 9)) for n in counts]
    gamma = [np.clip(.12*z[:, 0]-.09*z[:, 1]+.03*z[:, 2]**2-.03
                     +rng.normal(0, .05, len(z)), -.9, .8) for z in x[:3]]
    return x[0], gamma[0], x[1], gamma[1], x[2], gamma[2], x[3]


@pytest.fixture(scope='module')
def fitted():
    data = synthetic()
    return data, fit_direct_baselines(*data, seed=124)


def test_complete_grids_full_inputs_and_separate_probability_interfaces(fitted):
    data, out = fitted
    assert tuple(out) == ARMS
    for name, arm in out.items():
        meta = arm['metadata']
        assert meta['n_train'] == 80 and meta['n_validation'] == 24
        assert meta['n_calibration'] == 20 and meta['n_query'] == 9
        assert meta['full_input_dimension'] == 9
        assert not meta['feature_truncation'] and not meta['pca_used']
        assert not meta['validation_refit'] and not meta['query_outcomes_accepted']
        assert meta['threads'] == 1
        assert meta['gamma_crps_is_fair_monte_carlo_estimator'] is False
        for key in ('predicted', 'gamma_distribution_mean', 'p_null',
                    'p_null_calibrated', 'p_null_from_gamma'):
            assert arm[key].shape == (9,) and np.isfinite(arm[key]).all()
        for key in ('p_null', 'p_null_calibrated', 'p_null_from_gamma'):
            assert np.all((arm[key] >= 0)&(arm[key] <= 1))
        support = empirical_gamma_support(arm['predicted'], arm['gamma_residuals'])
        np.testing.assert_array_equal(support.mean(1), arm['gamma_distribution_mean'])
        np.testing.assert_array_equal((support <= 0).mean(1), arm['p_null_from_gamma'])
        assert np.all((support >= BOUNDS[0])&(support <= BOUNDS[1]))
        expected = {'RIDGE': (6, 4), 'EXTRATREES': (3, 3), 'HISTGB': (3, 3)}[name]
        assert (len(meta['regression']['candidates']), len(meta['classification']['candidates'])) == expected
        if name == 'EXTRATREES':
            assert arm['model']['regression'].n_estimators == TREE_COUNT == 256
            assert arm['model']['classifier'].n_jobs == 1
        if name == 'HISTGB':
            assert arm['model']['regression'].max_iter == HIST_ITERATIONS == 200
            assert arm['model']['classifier'].early_stopping is False


def test_crps_is_exact_finite_distribution_including_ties_not_fair_mc():
    support = np.array([[-1., 0., 0., .8], [.1, .1, .1, .1], [-.8, -.2, .4, .7]])
    actual = np.array([.2, -.1, .5])
    brute = np.abs(support-actual[:, None]).mean(1) \
        -.5*np.abs(support[:, :, None]-support[:, None, :]).mean((1, 2))
    np.testing.assert_allclose(exact_empirical_crps(support, actual), brute, atol=1e-15)
    # Two support atoms at 0 and .8, observation 0: exact CRPS=.2.
    assert exact_empirical_crps([[0., .8]], [0.])[0] == pytest.approx(.2)
    # The fair m(m-1) estimator would give 0 here, a different quantity.
    assert exact_empirical_crps([[.3]], [0.])[0] == pytest.approx(.3)
    bounded = empirical_gamma_support([-.9, .9], [-.6, .6])
    np.testing.assert_allclose(bounded, [[-1.02, -.3], [.3, .98]])


def test_calibration_changes_distribution_not_train_fits_or_validation_choices(fitted):
    data, original = fitted
    changed = list(data)
    changed[5] = -changed[5]
    out = fit_direct_baselines(*changed, seed=124)
    for name in ARMS:
        a, b = original[name], out[name]
        np.testing.assert_array_equal(a['predicted'], b['predicted'])
        np.testing.assert_array_equal(a['p_null'], b['p_null'])
        assert a['metadata']['regression'] == b['metadata']['regression']
        assert a['metadata']['classification'] == b['metadata']['classification']
        assert not np.allclose(a['gamma_residuals'], b['gamma_residuals'])
        assert not np.allclose(a['p_null_calibrated'], b['p_null_calibrated'])


def test_query_order_is_not_fitted_and_actual_query_only_enters_evaluation(fitted):
    data, original = fitted
    perm = np.array([6, 2, 0, 8, 1, 5, 7, 3, 4])
    changed = list(data)
    changed[6] = changed[6][perm]
    out = fit_direct_baselines(*changed, seed=124)
    assert 'actual_query' not in inspect.signature(fit_direct_baselines).parameters
    for name in ARMS:
        for key in ('predicted', 'p_null', 'p_null_calibrated', 'gamma_distribution_mean', 'p_null_from_gamma'):
            np.testing.assert_allclose(out[name][key], original[name][key][perm], atol=1e-14, rtol=1e-14)
        arm = original[name]
        snapshot = {k: arm[k].copy() for k in ('predicted', 'p_null', 'p_null_calibrated', 'gamma_residuals')}
        scored = evaluate_direct_distribution(arm, np.linspace(-.2, .3, 9))
        assert scored['crps'].shape == (9,)
        assert scored['gamma_coverage_by_level'].shape == (9, 5)
        assert scored['predictive_distribution_sample_count'] == 20
        assert np.isfinite(scored['crps']).all()
        for key, value in snapshot.items():
            np.testing.assert_array_equal(arm[key], value)


def test_models_can_be_saved_by_caller(fitted, tmp_path):
    data, out = fitted
    path = tmp_path/'models.joblib'
    joblib.dump(out, path)
    restored = joblib.load(path)
    for name in ARMS:
        arm = restored[name]
        prediction = np.clip(arm['model']['regression'].predict(data[6]), *BOUNDS)
        np.testing.assert_array_equal(prediction, out[name]['predicted'])
        np.testing.assert_array_equal(arm['model']['platt'].predict(arm['p_null']), arm['p_null_calibrated'])


def test_missing_null_class_has_explicit_constant_and_calibration_fallback():
    data = list(synthetic())
    data[1] = np.full(len(data[1]), .1)
    data[3] = np.full(len(data[3]), .1)
    data[5] = np.full(len(data[5]), .1)
    out = fit_direct_baselines(*data, seed=19)
    for arm in out.values():
        assert isinstance(arm['model']['classifier'], ConstantNullClassifier)
        np.testing.assert_array_equal(arm['p_null'], 0.)
        np.testing.assert_allclose(arm['p_null_calibrated'], 1/22)
        assert arm['metadata']['classification']['candidates'] == []
        assert arm['metadata']['platt']['method'] == 'single-class CAL Laplace-smoothed constant'


@pytest.mark.parametrize('case', ['dimension', 'nonfinite', 'target_bound', 'seed'])
def test_invalid_inputs_rejected_before_fit(case):
    data = list(synthetic())
    seed = 3
    if case == 'dimension': data[6] = data[6][:, :-1]
    if case == 'nonfinite': data[4][0, 0] = np.nan
    if case == 'target_bound': data[5][0] = 1.1
    if case == 'seed': seed = True
    with pytest.raises(ValueError):
        fit_direct_baselines(*data, seed=seed)
