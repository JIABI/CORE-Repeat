import numpy as np
from opal2.joint_contrast_scale import (pair_jacobian, contrast_projector,
    projected_energy, fit_scale, predict_scale, rescale_components)
from opal2.conditional_joint_error_experiment import observable_forward
from opal2.conditional_joint_error import gaussian_score


def example():
    rng = np.random.default_rng(171)
    raw = rng.normal(size=(12, 9))*.25
    scale = rng.uniform(.3, 1.3, 9)
    matrix = rng.normal(size=(9, 9))
    covariance = np.broadcast_to(matrix@matrix.T+.5*np.eye(9), (12, 9, 9)).copy()
    return raw, scale, covariance


def test_pair_jacobian_against_centered_difference():
    raw, scale, _ = example()
    analytic = pair_jacobian(raw, scale)
    numeric = np.empty_like(analytic)
    for j in range(9):
        offset = np.zeros_like(raw)
        offset[:, j] = 1e-6*scale[j]
        numeric[:, :, j] = (observable_forward(raw+offset)[1][:, 3:6]
                            -observable_forward(raw-offset)[1][:, 3:6])/2e-6
    np.testing.assert_allclose(analytic, numeric, atol=5e-10, rtol=1e-7)


def test_projector_and_exact_fallback():
    raw, scale, covariance = example()
    d = contrast_projector(raw, scale, covariance)
    P = d['projector']
    np.testing.assert_allclose(P@P, P, atol=1e-13)
    np.testing.assert_allclose(np.trace(P, axis1=-2, axis2=-1), 3, atol=1e-13)
    np.testing.assert_allclose(d['whitened_jacobian']@(np.eye(9)-P), 0, atol=1e-13)
    np.testing.assert_array_equal(rescale_components(covariance, d, 1., 1.), covariance)
    np.testing.assert_allclose(rescale_components(covariance, d, 2., 2.), 2*covariance, atol=1e-13)


def test_projected_likelihood_matches_full_gaussian():
    raw, scale, covariance = example()
    d = contrast_projector(raw, scale, covariance)
    residual = np.random.default_rng(912).normal(size=(12, 9))
    energy = projected_energy(residual, d)
    a, b = .7, 1.8
    new = rescale_components(covariance, d, a, b)
    exact = gaussian_score(residual, new)-gaussian_score(residual, covariance)
    derived = .5*(3*np.log(a)+6*np.log(b)+energy[:, 0]*(1/a-1)+energy[:, 1]*(1/b-1))
    np.testing.assert_allclose(exact, derived, atol=1e-13)


def test_conditional_scale_recovers_known_curve_and_constant():
    x = np.linspace(-2, 2, 200)
    target = 3*np.exp(.2+.4*x)
    fitted = fit_scale(target, 3, x, conditional=True, penalty=0.)
    np.testing.assert_allclose(predict_scale(fitted, x), target/3, atol=1e-7)
    const = fit_scale(target, 3, x, conditional=False)
    np.testing.assert_allclose(predict_scale(const, x), target.mean()/3)


def test_negative_variance_rejected():
    import pytest
    raw, scale, covariance = example()
    d = contrast_projector(raw, scale, covariance)
    with pytest.raises(ValueError):
        rescale_components(covariance, d, -.5, 1.)
