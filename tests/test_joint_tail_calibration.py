"""Outcome-free group splitting and finite-rank radius calibration checks."""
import inspect

import numpy as np
import pytest
from scipy.stats import chi2

from opal2.joint_tail_calibration import (
    finite_sample_squared_radius, fit_tail_calibration, mahalanobis_scores,
    scale_covariance, split_donor_groups,
)


def test_group_split_is_complete_disjoint_deterministic_and_row_equivariant():
    groups = np.array(['b', 'a', 'b', 'c', 'd', 'a', 'e', 'f', 'f'])
    result = split_donor_groups(groups, seed=17)
    assert result['n_fit_groups'] == 4 and result['n_cal_groups'] == 2
    assert set(result['fit_groups']).isdisjoint(result['cal_groups'])
    np.testing.assert_array_equal(np.sort(np.r_[result['fit_indices'], result['cal_indices']]), np.arange(len(groups)))
    np.testing.assert_array_equal(result['fit_indices'], split_donor_groups(groups, seed=17)['fit_indices'])
    order = np.array([5, 3, 2, 7, 0, 8, 1, 6, 4])
    permuted = split_donor_groups(groups[order], seed=17)
    assert result['fit_groups'] == permuted['fit_groups']
    assert result['cal_groups'] == permuted['cal_groups']
    np.testing.assert_array_equal(np.sort(order[permuted['cal_indices']]), result['cal_indices'])
    assert list(inspect.signature(split_donor_groups).parameters) == ['donor_groups', 'seed']
    with pytest.raises(TypeError): split_donor_groups(groups, seed=17, outcomes=np.arange(len(groups)))
    with pytest.raises(ValueError, match='two chemistry groups'): split_donor_groups(['a', 'a'], seed=17)
    with pytest.raises(ValueError, match='seed'): split_donor_groups(groups, seed=-1)


def test_split_uses_private_generator_and_small_group_rounding():
    np.random.seed(81)
    expected = np.random.uniform(size=5)
    np.random.seed(81)
    result = split_donor_groups(['a', 'b'], seed=9)
    np.testing.assert_array_equal(np.random.uniform(size=5), expected)
    assert len(result['fit_indices']) == len(result['cal_indices']) == 1
    assert split_donor_groups(['a', 'b', 'c', 'd'], seed=9)['n_fit_groups'] == 2


def test_finite_sample_rank_uses_ceiling_order_statistic_and_infinite_edge():
    assert finite_sample_squared_radius([4., 1., 3., 2.], alpha=.4) == 3.
    assert finite_sample_squared_radius(np.arange(1., 20.)) == 19.
    assert finite_sample_squared_radius(np.arange(1., 21.)) == 20.
    assert finite_sample_squared_radius(np.arange(1., 40.)) == 38.
    assert finite_sample_squared_radius(np.arange(1., 10.), alpha=.7) == 3.
    assert np.isinf(finite_sample_squared_radius(np.arange(1., 19.)))
    assert np.isinf(finite_sample_squared_radius([]))
    assert finite_sample_squared_radius([0., 0., 0.], alpha=.5) == 0.
    assert finite_sample_squared_radius([1., 1., 8.], alpha=.5) == 1.
    for alpha in (0., 1., np.nan):
        with pytest.raises(ValueError, match='alpha'): finite_sample_squared_radius([1.], alpha=alpha)
    for scores in ([-1.], [np.inf], [np.nan], [[1.]]):
        with pytest.raises(ValueError, match='scores'): finite_sample_squared_radius(scores)


def test_mahalanobis_scores_match_full_precision_and_batched_covariance():
    residual = np.array([[1., 2.], [3., -1.]])
    cov = np.array([[2., 1.], [1., 3.]])
    expected = np.einsum('ni,ij,nj->n', residual, np.linalg.inv(cov), residual)
    np.testing.assert_allclose(mahalanobis_scores(residual, cov), expected)
    np.testing.assert_allclose(mahalanobis_scores(residual, np.tile(cov, (2, 1, 1))), expected)
    with pytest.raises(ValueError, match='positive definite'):
        mahalanobis_scores(residual, np.ones((2, 2)))
    with pytest.raises(ValueError, match='aligned'):
        mahalanobis_scores(residual, np.ones((3, 2, 2)))


def test_calibration_uses_id_min_representatives_not_outcomes_or_group_counts():
    ids = np.array(['z', 'a', 'c', 'b', 'x', 'y'])
    groups = np.array(['g1', 'g1', 'g2', 'g2', 'g3', 'g3'])
    residual = np.zeros((6, 9)); residual[:, 0] = [100., 1., 50., 2., 3., 90.]
    result = fit_tail_calibration(residual, np.eye(9), ids, groups, alpha=.5)
    assert result['representative_indices'] == [1, 3, 4]
    assert result['representative_ids'] == ['a', 'b', 'x']
    assert result['representative_scores'] == [1., 4., 9.]
    assert result['cal_group_sizes'] == [2, 2, 2]
    assert result['m'] == 3 and result['k'] == 2 and result['q'] == 4.
    np.testing.assert_allclose(result['gaussian_mle_scale'], (1.+4.+9.)/(3*9))
    assert result['raw_cal_scores'] == [10000., 1., 2500., 4., 9., 8100.]
    changed = residual.copy(); changed[[0, 2, 5]] *= 50
    different = fit_tail_calibration(changed, np.eye(9), ids, groups, alpha=.5)
    for key in ('q', 'gaussian_mle_scale', 'representative_ids', 'representative_scores'):
        assert result[key] == different[key]
    order = np.array([5, 2, 1, 0, 4, 3])
    permuted = fit_tail_calibration(residual[order], np.eye(9), ids[order], groups[order], alpha=.5)
    assert permuted['representative_ids'] == result['representative_ids']
    assert permuted['q'] == result['q']
    assert permuted['gaussian_mle_scale'] == result['gaussian_mle_scale']
    assert not result['region_only_changes_density']
    assert all(name.startswith('cal_') or name == 'alpha'
               for name in inspect.signature(fit_tail_calibration).parameters)
    with pytest.raises(TypeError):
        fit_tail_calibration(residual, np.eye(9), ids, groups, query_residual=residual)


def test_covariance_scale_matches_region_and_preserves_correlation_without_clamp():
    cov = np.array([[2., 1.], [1., 3.]])
    residual = np.array([[1., 2.], [3., -1.], [.1, .2]])
    original = cov.copy()
    for scaling in (.25, 1., 3.):
        scaled = scale_covariance(cov, scaling)
        np.testing.assert_array_equal(scaled, cov*scaling)
        np.testing.assert_allclose(mahalanobis_scores(residual, scaled),
                                   mahalanobis_scores(residual, cov)/scaling)
        sd0 = np.sqrt(np.diag(cov)); sd1 = np.sqrt(np.diag(scaled))
        np.testing.assert_allclose(scaled/sd1[:, None]/sd1[None], cov/sd0[:, None]/sd0[None])
        q = scaling*chi2.ppf(.95, 2)
        np.testing.assert_array_equal(mahalanobis_scores(residual, cov) <= q,
            mahalanobis_scores(residual, scaled) <= chi2.ppf(.95, 2))
    np.testing.assert_array_equal(cov, original)
    for invalid in (0., -1., np.inf, np.nan):
        with pytest.raises(ValueError, match='nondegenerate Gaussian'): scale_covariance(cov, invalid)


def test_calibration_records_unavailable_full_law_without_clipping_region():
    small = fit_tail_calibration(np.ones((2, 9)), np.eye(9), ['a', 'b'], ['g1', 'g2'])
    assert np.isinf(small['q']) and np.isinf(small['full_law_scale'])
    assert not small['full_law_available'] and small['mle_available']
    zero = fit_tail_calibration(np.zeros((20, 9)), np.eye(9),
                                [str(i) for i in range(20)], [str(i) for i in range(20)])
    assert zero['q'] == 0. and zero['full_law_scale'] == 0. and zero['gaussian_mle_scale'] == 0.
    assert not zero['full_law_available'] and not zero['mle_available']
    assert zero['chi2_reference'] == chi2.ppf(.95, 9)
    with pytest.raises(ValueError, match='unique'):
        fit_tail_calibration(np.ones((2, 9)), np.eye(9), ['a', 'a'], ['g1', 'g2'])
