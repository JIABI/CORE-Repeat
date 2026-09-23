"""Complete empirical radial laws: density, quantiles, moments and fixed streams."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma

from opal2.empirical_radial import (
    fit_radial, reference_weights, radial_cdf, radial_ppf, radial_nll,
    draw_radial, variance_multiplier,
)


def test_bandwidth_formula_and_degenerate_iqr_fallback():
    radii = np.exp(np.array([-.8, -.4, -.1, .2, .8, 1.]))
    law = fit_radial(radii)
    center = np.log(radii)
    expected = max(.1, .9*min(center.std(ddof=1), np.subtract(*np.percentile(center, [75, 25]))/1.349)*len(center)**(-.2))
    assert law['bandwidth'] == expected and law['epsilon'] == .1
    np.testing.assert_array_equal(law['log_centers'], center)
    fallback = fit_radial(np.exp([0., 0., 0., 0., 0., 3.]))
    assert fallback['log_radius_iqr'] == 0
    assert fallback['robust_scale'] == np.std([0., 0., 0., 0., 0., 3.], ddof=1)
    assert fit_radial([2.])['bandwidth'] == .1


def test_feature_only_local_weights_shrink_and_report_final_ess():
    cal, query = np.array([-3., -1., 0., 2., 4.]), np.array([-10., .4, 30.])
    out = reference_weights(cal, query, 1.2, conditional=True)
    logits = -.5*((query[:, None]-cal)/1.2)**2
    local = np.exp(logits-logits.max(1, keepdims=True))
    local /= local.sum(1, keepdims=True)
    ess = 1/np.square(local).sum(1)
    shrink = ess/(ess+20)
    expected = shrink[:, None]*local+(1-shrink[:, None])/len(cal)
    np.testing.assert_allclose(out['weights'], expected, atol=1e-14)
    np.testing.assert_allclose(out['ess'], 1/np.square(expected).sum(1))
    np.testing.assert_allclose(out['weights'].sum(1), 1)
    np.testing.assert_allclose(reference_weights(cal, query, 1., False)['weights'], .2)
    # A very large common log-kernel offset must not lose the normalization term.
    extreme = reference_weights([0., 0., 0.], [1e100], 1., conditional=True)
    np.testing.assert_allclose(extreme['weights'], 1/3)


@pytest.mark.parametrize('dimension', [2, 9])
def test_complete_density_integrates_to_one_including_jacobians(dimension):
    law = fit_radial([.7, 1.2, 2.], dimension=dimension)
    weights = np.array([[.2, .3, .5]])
    scatter = (np.eye(dimension)*2)[None]
    area = 2*np.pi**(dimension/2)/gamma(dimension/2)
    # Integrate in whitened spherical radius; determinant scatter factor is 2^(d/2).
    def density(r):
        residual = np.zeros((1, dimension)); residual[0, 0] = np.sqrt(2)*r
        return np.exp(-radial_nll(residual, scatter, law, weights)[0])*area*r**(dimension-1)*2**(dimension/2)
    knots = sorted(set([0., *np.exp(law['log_centers']-law['bandwidth']),
                        *np.exp(law['log_centers']), *np.exp(law['log_centers']+law['bandwidth'])]))
    integral = sum(quad(density, left, right, epsabs=1e-10)[0] for left, right in zip(knots, knots[1:]))
    integral += quad(density, knots[-1], np.inf, epsabs=1e-10)[0]
    assert integral == pytest.approx(1., abs=2e-9)
    zero = radial_nll(np.zeros((1, dimension)), scatter, law, weights)[0]
    assert zero == pytest.approx(-np.log(.1)+dimension/2*np.log(2*np.pi)+dimension/2*np.log(2))


def test_quantiles_and_exact_variance_are_not_silently_normalized():
    law = fit_radial([1., 2., 3.])
    weights = np.array([[.1, .2, .7], [.8, .2, 0.]])
    levels = np.array([0., .1, .5, .95, .99, 1.])
    result = radial_ppf(law, weights, levels)
    assert result.shape == (2, 6)
    assert np.all(result[:, 0] == 0) and np.isposinf(result[:, -1]).all()
    for j, level in enumerate(levels[1:-1], 1):
        np.testing.assert_allclose(radial_cdf(law, weights, result[:, j]), level, atol=2e-12)
    np.testing.assert_array_equal(radial_cdf(law, weights, [-1., 0.]), [0., 0.])
    np.testing.assert_allclose(radial_cdf(law, weights, [np.inf, np.inf]), 1.)
    h = law['bandwidth']
    expected = .1+.9*(weights@np.square([1., 2., 3.]))*(np.sinh(h)/h)**2/9
    np.testing.assert_allclose(variance_multiplier(law, weights), expected, rtol=2e-15)
    assert not np.allclose(expected, 1.)


def test_sampling_preserves_complete_direction_guard_and_zero_weight_support():
    law = fit_radial([1., 2., 100.], dimension=2)
    weights = np.array([[0., 1., 0.]])
    scatter = np.array([[[2., .3], [.3, 1.]]])
    normal = np.array([[[3., 4.]], [[-4., 3.]], [[3., -4.]], [[-3., -4.]]])
    uniforms = np.array([[0.], [.1], [.55], [np.nextafter(1., 0.)]])
    kernel = np.full((4, 1), .5)
    draws = draw_radial(law, weights, scatter, normal, uniforms, kernel)
    white = np.linalg.solve(np.linalg.cholesky(scatter)[None], draws[..., None])[..., 0]
    np.testing.assert_allclose(white[0], normal[0])
    np.testing.assert_allclose(np.linalg.norm(white[1:], axis=-1), 2.)
    np.testing.assert_allclose(white[1:]/normal[1:], .4)
    np.testing.assert_array_equal(draws, draw_radial(law, weights, scatter, normal, uniforms, kernel))


def test_sampling_covariance_matches_exposed_factor_and_full_scatter():
    rng = np.random.default_rng(883)
    law = fit_radial([1., 2., 3.], dimension=2)
    weights = np.array([[.2, .3, .5]])
    scatter = np.array([[[1., .45], [.45, 1.5]]])
    count = 100000
    draws = draw_radial(law, weights, scatter, rng.normal(size=(count, 1, 2)),
                        rng.random((count, 1)), rng.random((count, 1)))[:, 0]
    np.testing.assert_allclose(np.cov(draws, rowvar=False, bias=True),
                               variance_multiplier(law, weights)[0]*scatter[0], rtol=.015, atol=.015)
    np.testing.assert_allclose(draws.mean(0), 0., atol=.025)


def test_invalid_inputs_fail_without_clipping_or_covariance_repair():
    with pytest.raises(ValueError, match='positive'):
        fit_radial([0., 1.])
    law = fit_radial([1., 2.], dimension=2)
    weights = np.array([[.5, .5]])
    with pytest.raises(ValueError, match='positive'):
        reference_weights([1., 2.], [1.], 0., conditional=True)
    with pytest.raises(ValueError, match='normalized'):
        radial_cdf(law, [[1., 1.]], [1.])
    with pytest.raises(ValueError, match='positive definite'):
        radial_nll(np.zeros((1, 2)), np.zeros((1, 2, 2)), law, weights)
    with pytest.raises(ValueError, match='nonzero normal'):
        draw_radial(law, weights, np.eye(2)[None], np.zeros((1, 1, 2)), [[.5]], [[.5]])
