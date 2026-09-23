"""Independent distribution and gradient checks, using synthetic fixtures only."""
import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm, t

from opal2.copula import StudentT4GaussianCopula, normal_from_t4, t4_from_normal, t4_logpdf
from opal2.model import JointGaussian, LazyConditionalGaussian, EnvironmentNoiseCache


@pytest.fixture(scope="module", autouse=True)
def small_tensor_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def gaussian_fixture(shared=False, b=2, wells=3, dim=2):
    generator = torch.Generator().manual_seed(155)
    mean = torch.randn(b, wells, dim, generator=generator, dtype=torch.float64) * .3
    diagonal = torch.full_like(mean, .5)
    local = torch.randn(b, wells, dim, 2, generator=generator, dtype=torch.float64) * .3
    if not shared:
        return JointGaussian(mean, diagonal, local)
    loads = tuple(torch.full((*mean.shape, 1), a, dtype=torch.float64) for a in (.4, .2, .1))
    groups = torch.zeros(b, wells, 3, dtype=torch.int64)
    return JointGaussian(mean, diagonal, torch.cat((local, *loads), -1), local, loads, groups)


def test_t4_maps_match_independent_scipy_including_extreme_tails():
    x = np.array([-1e12, -10000., -100., -3., -1., -.1, 0., .1, 1., 3., 100., 10000., 1e12])
    expected = np.sign(x) * -norm.ppf(t.sf(np.abs(x), df=4))
    actual = normal_from_t4(torch.tensor(x)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(t4_logpdf(torch.tensor(x)).numpy(), t.logpdf(x, df=4), rtol=1e-13)
    z = np.array([-12., -8., -3., -1., -.1, 0., .1, 1., 3., 8., 12.])
    expected_inverse = np.sign(z) * t.isf(norm.sf(np.abs(z)), df=4)
    np.testing.assert_allclose(t4_from_normal(torch.tensor(z)).numpy(), expected_inverse, rtol=2e-11, atol=2e-11)


def test_roundtrip_far_beyond_cdf_underflow_has_nonzero_gradient():
    x = torch.tensor([-1e300, -1e100, -10000., 0., 10000., 1e100, 1e300],
                     dtype=torch.float64, requires_grad=True)
    z = normal_from_t4(x)
    recovered = t4_from_normal(z)
    torch.testing.assert_close(recovered, x, rtol=2e-11, atol=1e-12)
    gradient, = torch.autograd.grad(z.sum(), x)
    assert torch.isfinite(z).all() and torch.isfinite(gradient).all() and torch.all(gradient > 0)
    assert z[-1] > z[-2] > z[-3]  # No probability-floor saturation.


def test_probe_conditioning_preserves_double_tail_on_float32_network():
    from opal2.probe import _gaussian_condition_on_probe, _distribution_geometry

    class IdentityScaler:
        @staticmethod
        def transform_y(value):
            return value

    base = JointGaussian(torch.zeros(1, 3, 2), torch.ones(1, 3, 2),
                         torch.full((1, 3, 2, 1), .1))
    distribution = StudentT4GaussianCopula(base)
    assert distribution.dtype == torch.float32
    assert _distribution_geometry(distribution)[2] == torch.float64
    history = np.array([[[0., 0.], [1e40, -1e40]]], dtype=np.float64)
    conditioned = _gaussian_condition_on_probe(distribution, history, IdentityScaler())
    expected_y = torch.zeros((1, 3, 2), dtype=torch.float64)
    expected_y[:, 0] = torch.tensor(history[:, 1])
    expected_mask = torch.zeros_like(expected_y, dtype=torch.bool)
    expected_mask[:, 0] = True
    expected = distribution.condition(expected_y, expected_mask, (1, 2), lazy=True)
    first = conditioned.sample_joint(3, torch.Generator().manual_seed(18))
    second = expected.sample_joint(3, torch.Generator().manual_seed(18))
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("lazy", [False, True])
def test_partial_retained_probe_float32_base_keeps_exact_observed_tails(lazy):
    base = JointGaussian(torch.zeros(2, 3, 2), torch.ones(2, 3, 2),
                         torch.full((2, 3, 2, 1), .1))
    distribution = StudentT4GaussianCopula(base)
    y = torch.zeros((2, 3, 2), dtype=torch.float64)
    y[0, 0] = torch.tensor([1e40, -1e40], dtype=torch.float64)
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[0, 0] = True
    conditional = distribution.condition(y, mask, (0, 1, 2), retain_observed=True, lazy=lazy)
    sampled = conditional.sample_joint(3, torch.Generator().manual_seed(23))
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(sampled[:, 0, 0], y[0, 0].expand(3, -1), rtol=0, atol=0)


def test_quantile_autograd_is_density_ratio_not_a_clipped_tail():
    x = torch.tensor([-20., -1., -.2, 0., .2, 1., 20.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(normal_from_t4, (x,), atol=2e-6, rtol=2e-5)
    z = torch.tensor([-6., -1., -.2, 0., .2, 1., 6.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(t4_from_normal, (z,), atol=2e-6, rtol=2e-5)


def test_independent_coordinates_are_exact_independent_t4_densities():
    mean = torch.tensor([[[.1, -.3], [.5, -.7]]], dtype=torch.float64)
    var = torch.tensor([[[.4, 2.], [1.2, .8]]], dtype=torch.float64)
    base = JointGaussian(mean, var, torch.zeros(1, 2, 2, 1, dtype=torch.float64))
    distribution = StudentT4GaussianCopula(base)
    y = torch.tensor([[[10000., -4.], [.1, 3.]]], dtype=torch.float64)
    expected = torch.distributions.StudentT(4., mean, torch.sqrt(var / 2)).log_prob(y)
    torch.testing.assert_close(distribution.joint_log_prob(y), expected.sum(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(distribution.log_prob(y), expected.sum((1, 2)), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(distribution.mean, mean)
    torch.testing.assert_close(distribution.marginal_variance, var)
    assert distribution.moment_diagnostics["method"] == "analytic_t4_marginal"


def test_one_dimensional_density_integrates_to_one():
    base = JointGaussian(torch.zeros(1, 1, 1, dtype=torch.float64),
                         torch.ones(1, 1, 1, dtype=torch.float64),
                         torch.zeros(1, 1, 1, 1, dtype=torch.float64))
    distribution = StudentT4GaussianCopula(base)

    def pdf(x):
        with torch.no_grad():
            return distribution.joint_log_prob(torch.tensor([[[x]]], dtype=torch.float64)).exp().item()

    integral, error = quad(pdf, -np.inf, np.inf, epsabs=1e-9, epsrel=1e-9)
    assert abs(integral - 1) < 1e-8 and error < 1e-8


def test_correlated_joint_density_matches_dense_change_of_variables():
    base = gaussian_fixture(b=1, wells=2, dim=2)
    distribution = StudentT4GaussianCopula(base)
    y = torch.tensor([[[1.3, -2.], [.2, .4]]], dtype=torch.float64)
    gaussian = distribution.gaussianize(y)
    factor = base.factors[0].reshape(4, -1).numpy()
    covariance = np.diag(base.diag_var[0].reshape(-1).numpy()) + factor @ factor.T
    expected = (multivariate_normal.logpdf(gaussian.numpy().ravel(), base.mean.numpy().ravel(), covariance)
                + distribution.log_abs_det_y_to_g(y).sum().item())
    assert abs(distribution.joint_log_prob(y).item() - expected) < 1e-10
    y_grad = y.clone().requires_grad_(True)
    actual = torch.autograd.functional.jacobian(distribution.gaussianize, y_grad).reshape(4, 4)
    expected_jac = distribution.log_abs_det_y_to_g(y).exp().reshape(-1)
    torch.testing.assert_close(actual, torch.diag(expected_jac), rtol=1e-9, atol=1e-9)


def test_density_gradients_include_location_variance_and_factor_jacobian():
    location = torch.tensor([[[.2, -.1], [.3, .1]]], dtype=torch.float64, requires_grad=True)
    logdiag = torch.tensor([[[-.4, -.2], [-.1, -.5]]], dtype=torch.float64, requires_grad=True)
    factors = torch.tensor([[[[.2], [.1]], [[-.3], [.25]]]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[[3., -1.], [.5, -2.]]], dtype=torch.float64)

    def objective(loc, ld, f):
        return StudentT4GaussianCopula(JointGaussian(loc, ld.exp(), f)).joint_log_prob(y)

    assert torch.autograd.gradcheck(objective, (location, logdiag, factors), atol=3e-5, rtol=3e-4)
    huge = y.clone(); huge[0, 0, 0] = 10000.
    loss = StudentT4GaussianCopula(JointGaussian(location, logdiag.exp(), factors)).joint_log_prob(huge)
    gradients = torch.autograd.grad(loss, (location, logdiag, factors))
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in gradients)


def test_missing_coordinates_are_marginalized_and_masked_values_ignored():
    distribution = StudentT4GaussianCopula(gaussian_fixture())
    y = torch.ones(distribution.shape, dtype=torch.float64)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 1, 0] = False
    clean = distribution.joint_log_prob(y, mask)
    changed = y.clone(); changed[0, 1, 0] = float("nan")
    torch.testing.assert_close(distribution.joint_log_prob(changed, mask), clean)
    zeros = torch.zeros_like(mask)
    assert distribution.joint_log_prob(torch.full_like(y, float("nan")), zeros).item() == 0
    torch.testing.assert_close(distribution.log_prob(torch.full_like(y, float("nan")), zeros), torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("lazy", [False, True])
def test_conditioning_density_and_original_maps_are_exact(lazy):
    distribution = StudentT4GaussianCopula(gaussian_fixture(shared=True))
    y = torch.tensor([[[2., -.2], [.4, .8], [-1., .3]],
                      [[-.7, 1.2], [.3, -.4], [.5, .8]]], dtype=torch.float64)
    observed = torch.zeros_like(y, dtype=torch.bool); observed[:, 0] = True
    conditional = distribution.condition(y, observed, [1, 2], lazy=lazy)
    torch.testing.assert_close(conditional.location, distribution.location[:, 1:])
    torch.testing.assert_close(conditional.variance, distribution.variance[:, 1:])
    expected = distribution.joint_log_prob(y) - distribution.joint_log_prob(y, observed)
    torch.testing.assert_close(conditional.joint_log_prob(y[:, 1:]), expected, rtol=1e-9, atol=1e-9)
    assert conditional.moment_diagnostics["converged"]
    assert conditional.mean.shape == (2, 2, 2)
    assert torch.all(conditional.marginal_variance >= 0)


def test_chained_probe_equals_one_shot_and_does_not_refit_marginal_maps():
    distribution = StudentT4GaussianCopula(gaussian_fixture(shared=True))
    y = torch.ones(distribution.shape, dtype=torch.float64) * .8
    first_mask = torch.zeros_like(y, dtype=torch.bool); first_mask[:, 0] = True
    first = distribution.condition(y, first_mask, [1, 2], lazy=True)
    second_mask = torch.zeros_like(y[:, 1:], dtype=torch.bool); second_mask[:, 0] = True
    chained = first.condition(y[:, 1:], second_mask, [1], lazy=True)
    combined = first_mask.clone(); combined[:, 1] = True
    direct = distribution.condition(y, combined, [2], lazy=True)
    torch.testing.assert_close(chained.location, distribution.location[:, 2:])
    torch.testing.assert_close(chained.mean, direct.mean, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(chained.marginal_variance, direct.marginal_variance, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(chained.joint_log_prob(y[:, 2:]), direct.joint_log_prob(y[:, 2:]), rtol=1e-9, atol=1e-9)


def test_lazy_conditional_variance_matches_materialized_and_mc_moments():
    distribution = StudentT4GaussianCopula(gaussian_fixture(shared=True))
    y = torch.ones(distribution.shape, dtype=torch.float64) * 1.4
    observed = torch.zeros_like(y, dtype=torch.bool); observed[:, 0] = True
    lazy = distribution.condition(y, observed, [1, 2], lazy=True)
    dense = distribution.condition(y, observed, [1, 2], lazy=False)
    assert isinstance(lazy.base, LazyConditionalGaussian)
    torch.testing.assert_close(lazy._gaussian_marginal_variance(), dense.base.marginal_variance, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(lazy.mean, dense.mean, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(lazy.marginal_variance, dense.marginal_variance, rtol=1e-9, atol=1e-9)
    draw = lazy.sample_joint(50000, torch.Generator().manual_seed(123))
    torch.testing.assert_close(draw.mean(0), lazy.mean, rtol=.04, atol=.035)
    torch.testing.assert_close(draw.var(0), lazy.marginal_variance, rtol=.12, atol=.04)
    naive = lazy.degaussianize(lazy.base.mean)
    assert torch.max(torch.abs(lazy.mean - naive)) > .005


def test_retained_observations_are_point_masses_without_double_jacobian():
    distribution = StudentT4GaussianCopula(gaussian_fixture(shared=True))
    y = torch.ones(distribution.shape, dtype=torch.float64) * 1.7
    mask = torch.zeros_like(y, dtype=torch.bool); mask[0, 0] = True
    conditional = distribution.condition(y, mask, [0, 1, 2], retain_observed=True, lazy=True)
    draws = conditional.sample_joint(5, torch.Generator().manual_seed(44))
    torch.testing.assert_close(draws[:, 0, 0], y[0, 0].expand(5, -1), rtol=0, atol=0)
    expected = distribution.joint_log_prob(y) - distribution.joint_log_prob(y, mask)
    torch.testing.assert_close(conditional.joint_log_prob(y), expected, rtol=1e-9, atol=1e-9)
    assert not conditional.observed_mask(y)[0, 0].any()
    changed = y.clone(); changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="conflict"):
        conditional.joint_log_prob(changed)


def test_conditional_per_object_densities_are_correct_marginals_not_joint_sum():
    distribution = StudentT4GaussianCopula(gaussian_fixture(shared=True))
    y = torch.ones(distribution.shape, dtype=torch.float64)
    observed = torch.zeros_like(y, dtype=torch.bool); observed[:, 0] = True
    conditional = distribution.condition(y, observed, [1, 2], lazy=True)
    marginal = conditional.log_prob(y[:, 1:])
    expected = []
    for i in range(2):
        only = torch.zeros_like(y[:, 1:], dtype=torch.bool); only[i] = True
        expected.append(conditional.joint_log_prob(y[:, 1:], only))
    torch.testing.assert_close(marginal, torch.stack(expected), rtol=1e-9, atol=1e-9)
    assert abs((marginal.sum() - conditional.joint_log_prob(y[:, 1:])).item()) > 1e-5


def test_environment_cache_is_preserved_across_copula_wrapped_chunks():
    def distribution(group):
        mean = torch.zeros(1, 1, 1, dtype=torch.float64)
        local = torch.zeros(1, 1, 1, 1, dtype=torch.float64)
        loads = (torch.ones_like(local), torch.zeros_like(local), torch.zeros_like(local))
        groups = torch.tensor([[[group, group, group]]], dtype=torch.int64)
        gaussian = JointGaussian(mean, torch.full_like(mean, 1e-12), torch.cat((local, *loads), -1),
                                 local, loads, groups)
        return StudentT4GaussianCopula(gaussian)

    cache = EnvironmentNoiseCache()
    a = distribution(1).sample_joint(200, torch.Generator().manual_seed(1), cache).flatten()
    b = distribution(1).sample_joint(200, torch.Generator().manual_seed(2), cache).flatten()
    c = distribution(2).sample_joint(200, torch.Generator().manual_seed(3), cache).flatten()
    assert (a - b).abs().max() < .001
    assert (a - c).abs().mean() > .2


def test_unsupported_gaussian_aggregation_fails_explicitly():
    distribution = StudentT4GaussianCopula(gaussian_fixture())
    with pytest.raises(NotImplementedError, match="original-space"):
        distribution.aggregate_wells(torch.ones(2, 3))
    with pytest.raises(NotImplementedError, match="original-space"):
        distribution.weighted_moments(torch.ones(2, 3))


def test_lazy_constructor_requires_original_maps_and_invalid_inputs_rejected():
    distribution = StudentT4GaussianCopula(gaussian_fixture(shared=True))
    y = torch.ones(distribution.shape, dtype=torch.float64)
    mask = torch.zeros_like(y, dtype=torch.bool); mask[:, 0] = True
    conditional = distribution.condition(y, mask, [1, 2], lazy=True)
    with pytest.raises(ValueError, match="original marginal"):
        StudentT4GaussianCopula(conditional.base)
    with pytest.raises(ValueError, match="positive variances"):
        StudentT4GaussianCopula(distribution.base, location=distribution.location,
                               variance=torch.zeros_like(distribution.variance))
    with pytest.raises(ValueError, match="boolean"):
        distribution.joint_log_prob(y, torch.ones_like(y))
    with pytest.raises(ValueError, match="indices"):
        distribution.condition(y, mask, [1, 1])


def test_conditional_moment_nonconvergence_is_not_silently_accepted():
    distribution = StudentT4GaussianCopula(gaussian_fixture(), moment_order=8, moment_max_order=16,
                                         moment_rtol=1e-16, moment_atol=1e-16)
    y = torch.ones(distribution.shape, dtype=torch.float64) * 10
    mask = torch.zeros_like(y, dtype=torch.bool); mask[:, 0] = True
    conditional = distribution.condition(y, mask, [1, 2])
    with pytest.raises(RuntimeError, match="did not converge"):
        _ = conditional.mean
