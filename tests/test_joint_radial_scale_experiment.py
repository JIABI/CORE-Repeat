"""Finite complete-vector radial laws; calibration is separate from query scoring."""
import inspect

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import chi2, norm, multivariate_normal

import opal2.joint_radial_scale_experiment as radial
from opal2.conditional_joint_error import gaussian_score
from opal2.conditional_joint_error_experiment import score_distribution, observable_forward


def case():
    rng = np.random.default_rng(453)
    factor = rng.normal(size=(4, 9, 9))*.1
    covariance = factor@factor.swapaxes(-1, -2)+np.eye(9)*.15
    return rng.normal(size=(4, 9))*.2, covariance


@pytest.mark.parametrize('candidate', radial.CANDIDATES)
def test_exact_moments_and_non_diagonal_full_density(candidate):
    residual, covariance = case()
    small, p, large = (candidate[k] for k in ('v_small', 'p_large', 'v_large'))
    np.testing.assert_allclose((1-p)*small*covariance+p*large*covariance, covariance,
                               rtol=2e-16, atol=2e-16)
    expected = []
    for r, c in zip(residual, covariance):
        if p == 0:
            expected.append(-multivariate_normal.logpdf(r, cov=c))
        else:
            expected.append(-logsumexp([np.log1p(-p)+multivariate_normal.logpdf(r, cov=small*c),
                                       np.log(p)+multivariate_normal.logpdf(r, cov=large*c)]))
    np.testing.assert_allclose(radial.radial_nll(residual, covariance, candidate), expected,
                               rtol=1e-13, atol=1e-13)
    assert np.isfinite(radial.radial_nll(residual*10000, covariance, candidate)).all()


@pytest.mark.parametrize('candidate', radial.CANDIDATES)
def test_analytic_region_quantiles_have_declared_probabilities(candidate):
    result = radial.radial_quantiles(candidate)
    small, p, large = (candidate[k] for k in ('v_small', 'p_large', 'v_large'))
    q, h = result['joint_squared_radius'], result['coordinate_halfwidth']
    np.testing.assert_allclose((1-p)*chi2.cdf(q/small, 9)+p*chi2.cdf(q/large, 9),
                               radial.LEVELS, atol=2e-14, rtol=0)
    np.testing.assert_allclose((1-p)*norm.cdf(h/np.sqrt(small))+p*norm.cdf(h/np.sqrt(large)),
                               (1+np.asarray(radial.LEVELS))/2, atol=2e-14, rtol=0)
    assert np.all(np.diff(q) > 0) and np.all(np.diff(h) > 0)
    if p == 0:
        np.testing.assert_array_equal(q, chi2.ppf(radial.LEVELS, 9))
        np.testing.assert_array_equal(h, norm.ppf((1+np.asarray(radial.LEVELS))/2))


def test_sampler_shares_whole_vector_scale_and_is_deterministic():
    _, covariance = case()
    rng = np.random.default_rng(932)
    z = rng.normal(size=(6, 4, 9))
    uniforms = np.array([[.0, .099, .1, .8]]*6)
    baseline = np.einsum('nij,snj->sni', np.linalg.cholesky(covariance), z)
    draw = radial.sample_radial(covariance, radial.CANDIDATES[1], z, uniforms)
    expected_scale = np.broadcast_to(np.sqrt([5.5, 5.5, .5, .5]), (6, 4))
    np.testing.assert_array_equal(draw, baseline*expected_scale[..., None])
    np.testing.assert_array_equal(draw, radial.sample_radial(covariance, radial.CANDIDATES[1], z, uniforms))
    # Complete-vector signs and relative direction cannot be selected independently.
    np.testing.assert_array_equal(np.sign(draw), np.sign(baseline))
    np.testing.assert_allclose(draw/draw[..., :1], baseline/baseline[..., :1], rtol=2e-15)
    np.testing.assert_array_equal(radial.sample_radial(covariance, radial.CANDIDATES[0], z, uniforms), baseline)


def test_gaussian_score_exact_and_all_saved_scorer_fields_replayed():
    residual, covariance = case()
    np.testing.assert_array_equal(radial.radial_nll(residual, covariance, radial.CANDIDATES[0]),
                                  gaussian_score(residual, covariance))
    mean = np.zeros_like(residual)
    stats = dict(u_scale=np.full(9, .1), u_center=np.zeros(9))
    actual, obs, differences, _ = observable_forward(residual*.1)
    args = (mean, covariance, residual, stats, actual, obs, np.log1p(differences), np.ones(4))
    original = score_distribution(*args, seed=452, samples=40)
    actual_scores = radial.score_radial(mean, covariance, residual, radial.CANDIDATES[0],
        stats, actual, obs, np.log1p(differences), np.ones(4), seed=452, samples=40)
    for key, value in original.items():
        np.testing.assert_array_equal(actual_scores[key], value, err_msg=key)
    np.testing.assert_array_equal(actual_scores['joint_squared_radius'], np.full(4, chi2.ppf(.95, 9)))
    np.testing.assert_array_equal(actual_scores['coordinate_coverage'],
                                  actual_scores['coordinate_coverage_by_level'][:, 3])
    assert actual_scores['coordinate_coverage_by_level'].shape == (4, 5)


def test_calibration_only_api_and_fixed_ties(monkeypatch):
    assert list(inspect.signature(radial.select_radial_candidate).parameters) == ['cal_residual', 'cal_covariance']
    residual, covariance = case()
    scores = [radial.radial_nll(residual, covariance, c).mean() for c in radial.CANDIDATES]
    choice = radial.select_radial_candidate(residual, covariance)
    assert choice['candidate_index'] == int(np.argmin(scores))
    assert choice['calibration_n'] == 4 and len(choice['candidate_scores']) == 5
    monkeypatch.setattr(radial, 'radial_nll', lambda r, c, candidate: np.full(len(r), 3.))
    assert radial.select_radial_candidate(residual, covariance)['candidate_index'] == 0


def test_representatives_enforce_id_min_and_group_completeness():
    ids = np.array([f'g{i:02d}_{suffix}' for i in range(40) for suffix in ('b', 'a')])
    groups = np.repeat(np.arange(40), 2)
    representatives = ids[1::2].tolist()[::-1]
    positions = radial._representatives(ids, groups, representatives)
    assert ids[positions].tolist() == representatives
    with pytest.raises(ValueError, match='ID-min'):
        radial._representatives(ids, groups, ids[::2].tolist())


def test_mixture_scoring_uses_analytic_regions_with_finite_complete_draws():
    residual, covariance = case()
    mean, stats = np.zeros_like(residual), dict(u_scale=np.full(9, .1), u_center=np.zeros(9))
    actual, obs, differences, _ = observable_forward(residual*.1)
    candidate = radial.CANDIDATES[1]
    out = radial.score_radial(mean, covariance, residual, candidate, stats, actual, obs,
                             np.log1p(differences), np.ones(4), seed=452, samples=40)
    np.testing.assert_array_equal(out['nll'], radial.radial_nll(residual, covariance, candidate))
    np.testing.assert_allclose(out['joint_squared_radius'], 45.885580, atol=1e-6)
    assert all(np.isfinite(value).all() for value in out.values())


def test_invalid_law_covariance_and_streams_fail_without_repair():
    residual, covariance = case()
    with pytest.raises(ValueError, match='preserve covariance'):
        radial.radial_nll(residual, covariance, dict(v_small=.5, p_large=.1, v_large=5.))
    with pytest.raises(ValueError, match='positive definite'):
        radial.radial_nll(residual, -covariance, radial.CANDIDATES[0])
    with pytest.raises(ValueError, match='uniforms'):
        radial.sample_radial(covariance, radial.CANDIDATES[1], np.zeros((2, 4, 9)), np.ones((2, 4)))
    with pytest.raises(ValueError, match='levels'):
        radial.radial_quantiles(radial.CANDIDATES[0], levels=[1.])
