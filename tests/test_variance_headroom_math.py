"""Scientific invariants of the finite-grid two-scale hindsight comparator."""
import numpy as np
import pytest

from opal2.conditional_joint_error_experiment import observable_forward
from opal2.dual_branch_features import apply_increment
from opal2.empirical_radial import draw_radial, fit_radial, radial_nll
from opal2.joint_contrast_scale import contrast_projector
from opal2.radial_mixture_evaluation import fair_mixture_crps
from opal2.variance_headroom_math import (
    ETA_BOUND, adapter_grid, component_nll, evaluate_candidate_crps, fair_gamma_crps, gamma_only,
    residual_at_eta, sample_components, search_hindsight,
)


def fixture(n=3, s=128):
    rng = np.random.default_rng(188)
    mean = rng.normal(size=(n, 9))*.1
    factor = np.eye(9)[None]+rng.normal(size=(n, 9, 9))*.03
    scatter = factor@factor.swapaxes(-1, -2)*.1
    target = mean+rng.normal(size=(n, 9))*.25
    stats = dict(u_scale=np.linspace(.4, .7, 9), u_center=np.linspace(-.3, .3, 9))
    law = fit_radial(np.linspace(1., 5., 18))
    weights = rng.uniform(size=(n, 18)); weights /= weights.sum(1, keepdims=True)
    actual = observable_forward(target*stats['u_scale']+stats['u_center'])[0]
    normal = rng.normal(size=(s, n, 9)); mix = rng.random((s, n)); kernel = rng.random((s, n))
    dec = contrast_projector(mean*stats['u_scale']+stats['u_center'], stats['u_scale'], scatter)
    return mean, scatter, target, stats, law, weights, actual, normal, mix, kernel, dec


def test_gamma_fast_matches_all_original_actions_amplitude_and_joint_geometry():
    rng = np.random.default_rng(814)
    raw = rng.normal(size=(200, 7, 9))*.8
    np.testing.assert_allclose(gamma_only(raw), observable_forward(raw)[0], atol=3e-16, rtol=3e-14)


def test_fair_crps_matches_full_pair_u_statistic_and_existing_scorer():
    rng = np.random.default_rng(110)
    samples, target = rng.normal(size=(8, 3)), rng.normal(size=3)
    pair = np.abs(samples[:, None]-samples[None, :]).sum(axis=(0, 1))/(8*7)
    expected = np.abs(samples-target).mean(0)-.5*pair
    np.testing.assert_allclose(fair_gamma_crps(samples, target), expected, atol=1e-15)
    np.testing.assert_allclose(fair_gamma_crps(samples, target),
                               fair_mixture_crps((samples,), (1.,), target), atol=1e-15)


def test_zero_increment_is_exact_original_radial_sampling():
    mean, scatter, target, stats, law, weights, actual, normal, mix, kernel, dec = fixture()
    base, contrast = sample_components(scatter, dec, law, weights, normal, mix, kernel)
    original = draw_radial(law, weights, scatter, normal, mix, kernel)
    np.testing.assert_array_equal(base, original)
    np.testing.assert_array_equal(residual_at_eta(base, contrast, [0., 0.]), original)


def test_factor_parameterization_equals_updated_scatter_and_is_orthogonal_rotation():
    mean, scatter, target, stats, law, weights, actual, normal, mix, kernel, dec = fixture()
    eta = np.array([[.4, -.6], [-1., 1.2], [0., .5]])
    P, L = dec['projector'], dec['factor']
    sqrtwhite = np.exp(eta[:, :1, None]/2)*P+np.exp(eta[:, 1:, None]/2)*(np.eye(9)-P)
    factor = L@sqrtwhite
    changed = apply_increment(mean*stats['u_scale']+stats['u_center'], stats['u_scale'], scatter, eta)
    np.testing.assert_allclose(factor@factor.swapaxes(-1, -2), changed, atol=2e-15, rtol=1e-14)
    rotation = np.linalg.solve(np.linalg.cholesky(changed), factor)
    np.testing.assert_allclose(rotation@rotation.swapaxes(-1, -2),
                               np.broadcast_to(np.eye(9), rotation.shape), atol=3e-14)


def test_analytic_nll_equals_original_with_both_jacobians():
    mean, scatter, target, stats, law, weights, actual, normal, mix, kernel, dec = fixture()
    for eta in adapter_grid(3):
        increment = np.broadcast_to(eta, (len(mean), 2))
        changed = apply_increment(mean*stats['u_scale']+stats['u_center'], stats['u_scale'], scatter, increment)
        np.testing.assert_allclose(component_nll(target-mean, dec, law, weights, eta),
            radial_nll(target-mean, changed, law, weights), atol=2e-13, rtol=2e-13)


def test_search_returns_separate_nested_minima_and_exact_grid_indices():
    mean, scatter, target, stats, law, weights, actual, *_ = fixture(n=4)
    before = [x.copy() for x in (mean, scatter, target, weights)]
    out = search_hindsight(mean, scatter, target, stats, actual, law, weights, 31, samples=128, grid_size=3)
    assert out['grid_crps'].shape == out['grid_nll'].shape == (4, 9)
    for loss in ('crps', 'nll'):
        assert np.all(out[loss+'_two_'+loss] <= out[loss+'_scalar_'+loss]+1e-14)
        assert np.all(out[loss+'_scalar_'+loss] <= out['core_'+loss]+1e-14)
        for family in ('scalar', 'two'):
            key = family+'_'+loss
            np.testing.assert_array_equal(out['eta_'+key], out['eta_grid'][out['index_'+key]])
            np.testing.assert_array_equal(out['crps_'+key], out['grid_crps'][np.arange(4), out['index_'+key]])
    assert np.any(out['index_two_crps'] != out['index_two_nll'])
    for now, old in zip((mean, scatter, target, weights), before):
        np.testing.assert_array_equal(now, old)
    np.testing.assert_allclose(out['actual_total_energy'], out['actual_geometry_energy'].sum(1))


def test_grid_contains_baseline_scalar_subfamily_and_bounds():
    grid = adapter_grid(9)
    np.testing.assert_array_equal(grid[0], [0., 0.])
    assert len(grid) == 81 and len(np.unique(grid, axis=0)) == 81
    assert np.count_nonzero(grid[:, 0] == grid[:, 1]) == 9
    assert np.max(np.abs(grid)) == ETA_BOUND
    for bad in (2, 8, True, 1):
        with pytest.raises(ValueError): adapter_grid(bad)


def test_scalar_only_evaluation_matches_full_search_with_same_random_primitives():
    mean, scatter, target, stats, law, weights, actual, *_ = fixture(n=2)
    out = search_hindsight(mean, scatter, target, stats, actual, law, weights, 31, samples=128, grid_size=3)
    take = np.flatnonzero(out['grid_eta'][:, 0] == out['grid_eta'][:, 1])
    scalar = evaluate_candidate_crps(mean, scatter, stats, actual, law, weights,
                                    out['grid_eta'][take], 31, samples=128)
    np.testing.assert_array_equal(scalar, out['grid_crps'][:, take])


def test_invalid_geometry_is_not_silently_clipped():
    with pytest.raises(ValueError): gamma_only(np.full((1, 9), np.nan))
    with pytest.raises(ValueError): fair_gamma_crps(np.zeros((1, 3)), np.zeros(3))
    with pytest.raises(ValueError): residual_at_eta(np.zeros((3, 1, 9)), np.zeros((3, 1, 9)), [3., 0.])
