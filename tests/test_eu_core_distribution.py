"""Scope/isolation and numerical equivalence of the complete EU CORE recipe."""
import copy

import numpy as np
import pytest

from opal2.conditional_joint_error import fit_covariance_family
from opal2.conditional_joint_error_experiment import reference_weights as old_weights
from opal2.empirical_radial import fit_radial, reference_weights, variance_multiplier
from opal2.eu_core_distribution import (
    COORDINATE_SPACE, decision_reference_weights, fit_eu_distribution,
    predict_eu_distribution,
)
from opal2.joint_contrast_scale import fit_scale, predict_scale
from opal2.joint_tail_calibration import mahalanobis_scores


def fixture():
    rng = np.random.default_rng(12071)

    def inputs(role, count):
        x = rng.normal(size=(count, 15)) * np.exp(rng.normal(size=(count, 1)))
        bits = rng.integers(0, 2, size=(count, 512)).astype(float)
        return dict(ids=np.array([f'{role}_{i:02}' for i in range(count)]),
            groups=np.array([f'{role}_group_{i // 2:02}' for i in range(count)]),
            X=x, chem=np.column_stack((bits, np.ones(count))))

    ref, cal, query = inputs('ref', 32), inputs('cal', 14), inputs('query', 6)
    query['mean_u'] = rng.normal(size=(6, 9))
    a = rng.normal(size=(9, 9))
    base = a @ a.T / 9 + np.eye(9) * .5
    rr = rng.normal(size=(32, 9)) @ np.linalg.cholesky(base).T
    rr *= np.exp(.3 * np.log(np.linalg.norm(ref['X'], axis=1)))[:, None]
    cr = rng.normal(size=(14, 9)) @ np.linalg.cholesky(base).T
    return ref, rr, cal, cr, base, query


def fit_fixture():
    ref, rr, cal, cr, base, query = fixture()
    model = fit_eu_distribution(ref, rr, cal, cr, base, .8,
        model_training_ids=['train_a', 'train_b'], model_training_groups=['train_a', 'train_b'])
    return model, query


def test_complete_recipe_matches_original_local_scale_amplitude_and_radial_functions():
    ref, rr, cal, cr, base, query = fixture()
    fitted = fit_eu_distribution(ref, rr, cal, cr, base, .8)
    predicted = predict_eu_distribution(fitted, query)
    # Synthetic four-well containers are used only to replay the former X-only
    # weighting implementation; future slots are arbitrary and never read.
    all_x = np.vstack((ref['X'], cal['X'], query['X']))
    data = dict(ids=np.r_[ref['ids'], cal['ids'], query['ids']],
        groups=np.r_[ref['groups'], cal['groups'], query['groups']],
        chem=np.vstack((ref['chem'], cal['chem'], query['chem'])),
        Y=np.repeat(all_x[:, None, :], 4, axis=1))
    ri, ci, qi = np.arange(32), np.arange(32, 46), np.arange(46, 52)
    wr, _ = old_weights(data, ri, ri, .8)
    wcq, _ = old_weights(data, np.r_[ci, qi], ri, .8)
    covariance = fit_covariance_family(rr, base, wr, wcq, 'LOCAL_SCALE')
    np.testing.assert_array_equal(fitted['reference_loo_weights'], wr)
    np.testing.assert_allclose(predicted['reference_weights'], wcq[len(ci):], atol=1e-14, rtol=1e-14)
    assert fitted['covariance_choice'] == covariance['choice']
    logamp = np.log(np.linalg.norm(all_x, axis=1))
    energy = mahalanobis_scores(rr, covariance['loo_covariance'])
    amplitude = fit_scale(energy, 9, logamp[ri], conditional=True, penalty=1.)
    assert fitted['amplitude_fit'] == amplitude
    amp = predict_scale(amplitude, logamp[np.r_[ci, qi]])
    scatter = covariance['query_covariance'] * amp[:, None, None]
    reps = np.array([min(np.flatnonzero(cal['groups'] == g), key=lambda i: cal['ids'][i])
                     for g in np.unique(cal['groups'])])
    radius = np.sqrt(mahalanobis_scores(cr[reps], scatter[reps]))
    law = fit_radial(radius, 9)
    local = reference_weights(logamp[ci][reps], logamp[qi], logamp[ri].std(), conditional=True)
    np.testing.assert_allclose(predicted['base_scatter_u'], covariance['query_covariance'][len(ci):],
                               atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(predicted['scatter_u'], scatter[len(ci):], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(predicted['law']['log_centers'], law['log_centers'], atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(predicted['radial_weights'], local['weights'])
    np.testing.assert_allclose(predicted['radial_variance_multiplier'], variance_multiplier(law, local['weights']))
    np.testing.assert_array_equal(predicted['mean_u'], query['mean_u'])
    np.testing.assert_allclose(predicted['covariance_u'],
        predicted['scatter_u'] * predicted['radial_variance_multiplier'][:, None, None])
    assert predicted['report']['coordinate_space'] == COORDINATE_SPACE
    assert fitted['report']['radial_ess_shrinkage_constant'] == 20
    assert fitted['report']['radial_gaussian_guard'] == .1
    assert len(fitted['representative_ids']) == 7
    assert not np.allclose(predicted['radial_variance_multiplier'], 1.)


@pytest.mark.parametrize('future_key', ['Y', 'target', 'actual_u', 'residual', 'Gamma'])
def test_query_future_information_rejected(future_key):
    fitted, query = fit_fixture()
    query[future_key] = np.zeros((len(query['ids']), 9))
    with pytest.raises(ValueError, match='future measurements'):
        predict_eu_distribution(fitted, query)


def test_reference_loo_excludes_entire_groups_and_all_roles_are_disjoint():
    ref, rr, cal, cr, base, query = fixture()
    weights = decision_reference_weights(ref, ref, .8)['weights']
    same_group = ref['groups'][:, None] == ref['groups'][None, :]
    assert np.all(weights[same_group] == 0)
    np.testing.assert_allclose(weights.sum(1), 1.)
    fitted = fit_eu_distribution(ref, rr, cal, cr, base, .8)
    for key in ('ids', 'groups'):
        bad_cal = copy.deepcopy(cal)
        bad_cal[key][0] = ref[key][0]
        with pytest.raises(ValueError, match='overlap'):
            fit_eu_distribution(ref, rr, bad_cal, cr, base, .8)
        for other in (ref, cal):
            bad_query = copy.deepcopy(query)
            bad_query[key][0] = other[key][0]
            with pytest.raises(ValueError, match='overlap'):
                predict_eu_distribution(fitted, bad_query)
    with pytest.raises(ValueError, match='MODEL_TRAIN/REF'):
        fit_eu_distribution(ref, rr, cal, cr, base, .8,
            model_training_ids=['train'], model_training_groups=[ref['groups'][0]])
    fitted = fit_eu_distribution(ref, rr, cal, cr, base, .8,
        model_training_ids=['train'], model_training_groups=[query['groups'][0]])
    with pytest.raises(ValueError, match='QUERY/MODEL_TRAIN'):
        predict_eu_distribution(fitted, query)


def test_calibration_does_not_refit_covariance_mean_or_amplitude_and_query_is_batch_invariant():
    ref, rr, cal, cr, base, query = fixture()
    fit1 = fit_eu_distribution(ref, rr, cal, cr, base, .8)
    fit2 = fit_eu_distribution(ref, rr, cal, cr * 2, base, .8)
    p1, p2 = predict_eu_distribution(fit1, query), predict_eu_distribution(fit2, query)
    assert fit1['covariance_choice'] == fit2['covariance_choice']
    assert fit1['amplitude_fit'] == fit2['amplitude_fit']
    np.testing.assert_array_equal(p1['mean_u'], p2['mean_u'])
    np.testing.assert_array_equal(p1['scatter_u'], p2['scatter_u'])
    np.testing.assert_allclose(fit2['calibration_radii'], 2 * fit1['calibration_radii'])
    assert not np.allclose(p1['covariance_u'], p2['covariance_u'])
    for i in range(len(query['ids'])):
        one = predict_eu_distribution(fit1, {k: v[i:i + 1] for k, v in query.items()})
        for key in ('mean_u', 'scatter_u', 'radial_weights', 'covariance_u'):
            np.testing.assert_allclose(one[key], p1[key][i:i + 1], atol=1e-13, rtol=1e-13)


def test_invalid_coordinate_inputs_fail_without_fallback_or_clipping():
    ref, rr, cal, cr, base, _ = fixture()
    with pytest.raises(ValueError, match='positive definite'):
        fit_eu_distribution(ref, rr, cal, cr, np.zeros((9, 9)), .8)
    with pytest.raises(ValueError, match='standardized-u'):
        fit_eu_distribution(ref, rr[:, :8], cal, cr, base, .8)
    bad = copy.deepcopy(ref)
    bad['X'][0] = 0
    with pytest.raises(ValueError, match='positive norms'):
        fit_eu_distribution(bad, rr, cal, cr, base, .8)
