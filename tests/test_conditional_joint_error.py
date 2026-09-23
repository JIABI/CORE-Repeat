"""Donor-only error second moments, marginal controls, and joint correlation."""
import inspect

import numpy as np
import pytest

from opal2.conditional_joint_error import (
    FAMILIES, MIXING_GRID, fit_covariance_family, gaussian_score, second_moments,
)


def group_weights(groups):
    groups = np.asarray(groups)
    allowed = groups[:, None] != groups[None]
    return allowed/allowed.sum(1, keepdims=True)


def correlation(covariance):
    std = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    return covariance/std[..., :, None]/std[..., None, :]


def exact_base_case():
    dimension = 9
    cov0 = np.eye(dimension); cov0[0, 1] = cov0[1, 0] = .25
    directions = np.r_[np.eye(dimension), -np.eye(dimension)]*np.sqrt(dimension)
    block = directions@np.linalg.cholesky(cov0).T
    residual = np.tile(block, (2, 1))
    loo = group_weights(np.repeat([0, 1], len(block)))
    query = np.full((3, len(residual)), 1/len(residual))
    return residual, cov0, loo, query


def test_second_moment_retains_bias_and_complete_cross_products():
    residual = np.array([[2., -3.], [2., -3.], [4., 1.]])
    weights = np.array([[.5, .5, 0.], [.2, .3, .5]])
    moment = second_moments(weights, residual)
    np.testing.assert_array_equal(moment[0], [[4., -6.], [-6., 9.]])
    np.testing.assert_allclose(moment[1], .5*np.outer(residual[0], residual[0])
                               +.5*np.outer(residual[2], residual[2]))
    assert np.linalg.eigvalsh(moment).min() >= -1e-12


def test_full_gaussian_score_matches_logdet_quadratic_and_constant():
    residual = np.array([[1., 2.], [-3., .5]])
    cov0 = np.array([[4., 1.], [1., 2.]])
    expected = .5*(2*np.log(2*np.pi)+np.linalg.slogdet(cov0)[1]
                  +np.einsum('ni,ij,nj->n', residual, np.linalg.inv(cov0), residual))
    np.testing.assert_allclose(gaussian_score(residual, cov0), expected)
    np.testing.assert_allclose(gaussian_score(residual, np.tile(cov0, (2, 1, 1))), expected)
    with pytest.raises(ValueError, match='positive definite'):
        gaussian_score(residual, np.ones((2, 2)))
    with pytest.raises(ValueError, match='symmetric'):
        gaussian_score(residual, [[1., .5], [0., 1.]])


@pytest.mark.parametrize('family', FAMILIES)
def test_tied_moments_select_zero_and_return_exact_base(family):
    residual, cov0, loo, query = exact_base_case()
    result = fit_covariance_family(residual, cov0, loo, query, family)
    assert result['choice']['beta'] == 0.
    assert result['choice']['eta'] == 0.
    np.testing.assert_array_equal(result['loo_covariance'], np.broadcast_to(cov0, (len(loo), 9, 9)))
    np.testing.assert_array_equal(result['query_covariance'], np.broadcast_to(cov0, (len(query), 9, 9)))
    np.testing.assert_allclose(result['choice']['loo_nll'], gaussian_score(residual, cov0).mean())


@pytest.mark.parametrize('family', ['GLOBAL_SCALE', 'LOCAL_SCALE'])
def test_scalar_family_uses_full_precision_trace_and_preserves_correlation(family):
    residual = np.tile([4., 4.], (4, 1))
    cov0 = np.array([[2., 1.], [1., 2.]])
    loo = group_weights([0, 0, 1, 1]); query = np.array([[.25]*4, [0., .5, .5, 0.]])
    result = fit_covariance_family(residual, cov0, loo, query, family)
    beta = result['choice']['beta']
    assert beta > 0
    moment = second_moments(query, residual)
    tau = np.trace(np.linalg.solve(cov0, moment), axis1=-2, axis2=-1)/2
    expected = (1-beta+beta*tau)[:, None, None]*cov0
    np.testing.assert_allclose(result['query_covariance'], expected)
    np.testing.assert_allclose(correlation(result['query_covariance']),
                               np.broadcast_to(correlation(cov0), expected.shape))


def test_joint_and_global_correlation_control_keep_identical_selected_variances():
    rng = np.random.default_rng(231)
    residual = rng.normal(size=(12, 9))*np.linspace(.5, 3., 9)
    residual[:, 1] = .8*residual[:, 0]+rng.normal(size=12)*.1
    groups = np.repeat(np.arange(6), 2)
    uniform = group_weights(groups)
    local = uniform*rng.uniform(.1, 1., size=uniform.shape)
    local /= local.sum(1, keepdims=True)
    query = rng.uniform(.1, 1., size=(3, 12)); query /= query.sum(1, keepdims=True)
    cov0 = np.eye(9); cov0[2, 3] = cov0[3, 2] = .2
    diagonal = fit_covariance_family(residual, cov0, local, query, 'LOCAL_DIAG')
    joint = fit_covariance_family(residual, cov0, local, query, 'LOCAL_JOINT')
    global_corr = fit_covariance_family(residual, cov0, local, query, 'LOCAL_JOINT',
        correlation_loo_weights=uniform,
        correlation_query_weights=np.full_like(query, 1/len(residual)))
    for result in (joint, global_corr):
        assert result['choice']['beta'] == diagonal['choice']['beta']
        assert result['choice']['loo_nll'] <= diagonal['choice']['loo_nll']+1e-12
        for key in ('query_covariance', 'loo_covariance'):
            np.testing.assert_array_equal(np.diagonal(result[key], axis1=-2, axis2=-1),
                np.diagonal(diagonal[key], axis1=-2, axis2=-1))
            assert np.linalg.eigvalsh(result[key]).min() > 0
        beta, eta = result['choice']['beta'], result['choice']['eta']
        moment = result['correlation_query_second_moment']
        expected_corr = (1-eta)*correlation(cov0)+eta*correlation(.25*cov0+.75*moment)
        np.testing.assert_allclose(correlation(result['query_covariance']), expected_corr, atol=1e-14)
        assert beta in MIXING_GRID and eta in MIXING_GRID
    np.testing.assert_allclose(global_corr['correlation_loo_second_moment'],
                               second_moments(uniform, residual))


def test_zero_residuals_stay_positive_definite_without_hidden_covariance_floor():
    residual = np.zeros((4, 9)); cov0 = np.diag(np.arange(1., 10.))
    loo = group_weights([0, 0, 1, 1]); query = np.full((2, 4), .25)
    for family in FAMILIES:
        result = fit_covariance_family(residual, cov0, loo, query, family)
        assert result['choice']['beta'] == .75
        assert result['choice']['eta'] == 0
        np.testing.assert_array_equal(result['query_covariance'], np.broadcast_to(.25*cov0, (2, 9, 9)))
        assert np.linalg.eigvalsh(result['query_covariance']).min() > 0


def test_joint_nonzero_correlation_matches_hand_math_with_residual_bias():
    residual = np.tile([2., -2.], (4, 1))
    loo = group_weights([0, 0, 1, 1]); query = np.full((1, 4), .25)
    result = fit_covariance_family(residual, np.eye(2), loo, query, 'LOCAL_JOINT')
    assert result['choice']['beta'] == .75
    assert result['choice']['eta'] == .75
    # M has diagonal 4 and off-diagonal -4. Mreg has diagonal 13/4
    # and off-diagonal -3, so eta=.75 yields correlation -9/13.
    np.testing.assert_allclose(result['query_covariance'], [[[13/4, -9/4], [-9/4, 13/4]]])
    np.testing.assert_allclose(result['loo_second_moment'],
                               np.broadcast_to([[4., -4.], [-4., 4.]], (4, 2, 2)))


def test_group_exclusions_are_preserved_and_query_rows_cannot_change_selection():
    rng = np.random.default_rng(133)
    residual = rng.normal(size=(6, 9)); cov0 = np.eye(9)
    groups = np.array([0, 0, 1, 1, 2, 2]); loo = group_weights(groups)
    query = np.full((1, 6), 1/6)
    original = fit_covariance_family(residual, cov0, loo, query, 'LOCAL_JOINT')
    changed = residual.copy(); changed[:2] += 1000
    np.testing.assert_array_equal(second_moments(loo, residual)[:2], second_moments(loo, changed)[:2])
    changed_query = fit_covariance_family(residual, cov0, loo, np.eye(6), 'LOCAL_JOINT')
    assert original['choice'] == changed_query['choice']
    assert not any('target' in name or 'label' in name for name in inspect.signature(fit_covariance_family).parameters)
    with pytest.raises(TypeError):
        fit_covariance_family(residual, cov0, loo, query, 'LOCAL_DIAG', query_residual=residual)


def test_weight_shape_finiteness_normalization_and_optional_pair_validation():
    residual, cov0, loo, query = exact_base_case()
    with pytest.raises(ValueError, match='normalized'):
        second_moments(query*2, residual)
    with pytest.raises(ValueError, match='finite'):
        second_moments(query, residual*np.nan)
    with pytest.raises(ValueError, match='align'):
        second_moments(query[:, :-1], residual)
    with pytest.raises(ValueError, match='exclude'):
        fit_covariance_family(residual, cov0, np.eye(len(residual)), query, 'LOCAL_SCALE')
    with pytest.raises(ValueError, match='both'):
        fit_covariance_family(residual, cov0, loo, query, 'LOCAL_JOINT', correlation_loo_weights=loo)
    with pytest.raises(ValueError, match='only valid'):
        fit_covariance_family(residual, cov0, loo, query, 'LOCAL_DIAG',
            correlation_loo_weights=loo, correlation_query_weights=query)
    with pytest.raises(ValueError, match='align'):
        fit_covariance_family(residual, cov0, loo, query, 'LOCAL_JOINT',
            correlation_loo_weights=loo, correlation_query_weights=query[:1])
