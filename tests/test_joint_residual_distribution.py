"""Complete-vector mixture laws with fixed zero mean and query covariance."""
import inspect

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from opal2.conditional_joint_error import gaussian_score
from opal2.joint_residual_distribution import (
    build_residual_mixture, mixture_nll, sample_mixture, select_mixture_alpha,
)


def random_case(seed=513):
    rng = np.random.default_rng(seed)
    donor_count, queries, dimension = 11, 4, 9
    residual = rng.normal(size=(donor_count, dimension))+np.linspace(-1., 1., dimension)
    reference_factor = rng.normal(size=(donor_count, dimension, dimension))
    reference_covariance = reference_factor@reference_factor.swapaxes(-1, -2)+np.eye(dimension)
    query_factor = rng.normal(size=(queries, dimension, dimension))
    covariance = query_factor@query_factor.swapaxes(-1, -2)+np.eye(dimension)
    weights = rng.uniform(size=(queries, donor_count)); weights[:, ::3] = 0
    weights /= weights.sum(1, keepdims=True)
    return residual, reference_covariance, weights, covariance


def test_complete_mixture_matches_zero_mean_and_every_query_covariance():
    residual, reference, weights, covariance = random_case()
    mixture = build_residual_mixture(residual, reference, weights, covariance)
    centers, component_covariance = mixture['centers'], mixture['component_covariance']
    mean = np.einsum('qn,qnd->qd', weights, centers)
    second = component_covariance+np.einsum('qn,qni,qnj->qij', weights, centers, centers)
    np.testing.assert_allclose(mean, 0., atol=2e-14)
    np.testing.assert_allclose(second, covariance, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(mixture['matched_covariance'], covariance, rtol=2e-13, atol=2e-13)
    assert np.linalg.eigvalsh(component_covariance).min() > 0
    np.testing.assert_array_equal(mixture['covariance'], covariance)
    np.testing.assert_array_equal(mixture['weights'], weights)
    # Blend with the zero-mean Gaussian at any alpha: both moments stay fixed.
    for alpha in (0., .25, .5, .75, 1.):
        np.testing.assert_allclose((1-alpha)*covariance+alpha*second, covariance, atol=2e-13)


def test_transport_orientation_matches_noncommuting_hand_construction():
    residual = np.array([[1., 2.], [-2., 1.], [.5, -3.]])
    reference = np.array([[[2., .5], [.5, 1.]], [[1., .2], [.2, 2.]], [[3., -.3], [-.3, 1.]]])
    weights = np.array([[.2, .3, .5]])
    covariance = np.array([[[2., .8], [.8, 3.]]])
    mixture = build_residual_mixture(residual, reference, weights, covariance)
    whitened = np.stack([np.linalg.solve(np.linalg.cholesky(c), r) for c, r in zip(reference, residual)])
    a = np.sqrt(.75)*(whitened-weights@whitened)
    v = sum(w*np.outer(x, x) for w, x in zip(weights[0], a))+.25*np.eye(2)
    a_transform = np.linalg.cholesky(covariance[0])@np.linalg.inv(np.linalg.cholesky(v))
    np.testing.assert_allclose(mixture['transport'][0], a_transform)
    np.testing.assert_allclose(mixture['centers'][0], a@a_transform.T)
    np.testing.assert_allclose(mixture['component_covariance'][0], .25*a_transform@a_transform.T)


def test_alpha_zero_exactly_recovers_existing_gaussian_score_and_draws():
    residual, reference, weights, covariance = random_case()
    mixture = build_residual_mixture(residual, reference, weights, covariance)
    rng = np.random.default_rng(94)
    evaluation = rng.normal(size=(len(weights), residual.shape[1]))
    np.testing.assert_array_equal(mixture_nll(evaluation, mixture, 0), gaussian_score(evaluation, covariance))
    z = rng.normal(size=(13, len(weights), residual.shape[1]))
    uc = rng.uniform(size=z.shape[:2]); um = rng.uniform(size=z.shape[:2])
    expected = np.einsum('qij,sqj->sqi', np.linalg.cholesky(covariance), z, optimize=True)
    np.testing.assert_array_equal(sample_mixture(mixture, 0, z, uc, um), expected)


@pytest.mark.parametrize('alpha', [0., .25, 1.])
def test_nll_matches_independent_multivariate_normal_mixture(alpha):
    residual = np.array([[-1., 2.], [2., -.5], [0., -1.]])
    weights = np.array([[.2, 0., .8], [.5, .25, .25]])
    covariance = np.array([[[2., .5], [.5, 1.]], [[1., -.3], [-.3, 2.]]])
    mixture = build_residual_mixture(residual, np.tile(np.eye(2), (3, 1, 1)), weights, covariance)
    evaluation = np.array([[.3, -.8], [-2., 1.]])
    expected = []
    for row, value in enumerate(evaluation):
        base = multivariate_normal.logpdf(value, mean=np.zeros(2), cov=covariance[row])
        parts = [np.log(w)+multivariate_normal.logpdf(value, mean=center, cov=mixture['component_covariance'][row])
                 for w, center in zip(weights[row], mixture['centers'][row]) if w > 0]
        empirical = logsumexp(parts)
        blended = base if alpha == 0 else empirical if alpha == 1 else np.logaddexp(np.log1p(-alpha)+base, np.log(alpha)+empirical)
        expected.append(-blended)
    np.testing.assert_allclose(mixture_nll(evaluation, mixture, alpha), expected, rtol=1e-13)
    assert np.isfinite(mixture_nll(evaluation*1000, mixture, alpha)).all()


def test_sampling_selects_complete_centers_and_whole_vector_mixture_branch():
    residual = np.array([[1., -1.], [-1., 1.], [80., 90.]])
    mixture = build_residual_mixture(residual, np.tile(np.eye(2), (3, 1, 1)),
                                     [[.5, .5, 0.]], [np.eye(2)])
    z = np.zeros((4, 1, 2)); uc = np.array([[0.], [.49], [.5], [.99]])
    um = np.array([[0.], [.24], [.25], [.99]])
    samples = sample_mixture(mixture, 1, z, uc, um)
    np.testing.assert_array_equal(samples[:, 0], mixture['centers'][0, [0, 0, 1, 1]])
    assert np.all(samples[:, 0, 0]*samples[:, 0, 1] < 0)
    blended = sample_mixture(mixture, .25, z, uc, um)
    np.testing.assert_array_equal(blended[:2], samples[:2])
    np.testing.assert_array_equal(blended[2:], 0.)
    zero_first = build_residual_mixture(residual, np.tile(np.eye(2), (3, 1, 1)),
                                        [[0., .5, .5]], [np.eye(2)])
    drawn = sample_mixture(zero_first, 1, z[:1], np.zeros((1, 1)), um[:1])
    np.testing.assert_array_equal(drawn[0, 0], zero_first['centers'][0, 1])


def test_trailing_zero_weight_is_never_selected_at_upper_uniform_boundary():
    mixture = build_residual_mixture([[-1.], [0.], [1.], [100.]],
        np.ones((4, 1, 1)), [[.2, .7, .1, 0.]], np.ones((1, 1, 1)))
    sample = sample_mixture(mixture, 1., np.zeros((1, 1, 1)),
                            [[np.nextafter(1., 0.)]], [[0.]])
    np.testing.assert_array_equal(sample[0, 0], mixture['centers'][0, 2])


def test_sampling_is_deterministic_preserves_inputs_and_does_not_access_rng():
    residual, reference, weights, covariance = random_case()
    mixture = build_residual_mixture(residual, reference, weights, covariance)
    rng = np.random.default_rng(81)
    z = rng.normal(size=(5, 4, 9)); uc = rng.uniform(size=(5, 4)); um = rng.uniform(size=(5, 4))
    originals = [v.copy() for v in (z, uc, um, weights, covariance)]
    np.random.seed(27); expected_random = np.random.uniform(size=5)
    np.random.seed(27)
    first = sample_mixture(mixture, .5, z, uc, um)
    np.testing.assert_array_equal(np.random.uniform(size=5), expected_random)
    np.testing.assert_array_equal(first, sample_mixture(mixture, .5, z, uc, um))
    for original, value in zip(originals, (z, uc, um, weights, covariance)):
        np.testing.assert_array_equal(value, original)


def test_gaussian_limit_and_single_positive_donor_are_ties_for_alpha_selection():
    residual = np.array([[1., -2.], [3., 1.]])
    reference = np.tile(np.eye(2), (2, 1, 1))
    cov = np.tile([[2., .5], [.5, 1.]], (2, 1, 1))
    for weights, bandwidth in (([[.5, .5], [.25, .75]], 1.), ([[1., 0.], [0., 1.]], .5)):
        mixture = build_residual_mixture(residual, reference, weights, cov, bandwidth)
        expected = gaussian_score(residual, cov)
        for alpha in (0., .25, 1.):
            np.testing.assert_allclose(mixture_nll(residual, mixture, alpha), expected, atol=1e-13)
        selected = select_mixture_alpha(residual, mixture, grid=(1., .75, .5, .25, 0.))
        assert selected['alpha'] == 0.


def test_cal_selection_can_prefer_shape_but_has_no_query_target_argument():
    reference = np.array([[-1.], [1.]])
    mixture = build_residual_mixture(reference, np.ones((2, 1, 1)),
                                     np.full((4, 2), .5), np.ones((4, 1, 1)))
    cal = np.array([[-1.], [1.], [-1.], [1.]])*np.sqrt(.75)
    selected = select_mixture_alpha(cal, mixture)
    assert selected['alpha'] == 1.
    np.testing.assert_allclose(selected['cal_nll'], mixture_nll(cal, mixture, 1).mean())
    assert selected['cal_nll'] < selected['gaussian_cal_nll']
    assert list(inspect.signature(select_mixture_alpha).parameters) == ['cal_residual', 'cal_mixture', 'grid']
    assert not any('query_residual' in name or 'label' in name for name in inspect.signature(build_residual_mixture).parameters)
    with pytest.raises(TypeError): select_mixture_alpha(cal, mixture, query_residual=cal)


def test_validation_rejects_invalid_weights_covariance_bandwidth_and_streams():
    residual, reference, weights, covariance = random_case()
    for bandwidth in (0., -1., 1.1, np.nan):
        with pytest.raises(ValueError, match='bandwidth'):
            build_residual_mixture(residual, reference, weights, covariance, bandwidth)
    with pytest.raises(ValueError, match='normalized'):
        build_residual_mixture(residual, reference, weights*2, covariance)
    with pytest.raises(ValueError, match='positive definite'):
        build_residual_mixture(residual, np.zeros_like(reference), weights, covariance)
    mixture = build_residual_mixture(residual, reference, weights, covariance)
    with pytest.raises(ValueError, match='alpha'):
        mixture_nll(np.zeros((4, 9)), mixture, 1.1)
    with pytest.raises(ValueError, match='normal_draws'):
        sample_mixture(mixture, .5, np.zeros((4, 3, 9)), np.zeros((4, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match='component_uniforms'):
        sample_mixture(mixture, .5, np.zeros((2, 4, 9)), np.ones((2, 4)), np.zeros((2, 4)))
    with pytest.raises(ValueError, match='grid'):
        select_mixture_alpha(np.zeros((4, 9)), mixture, grid=())
