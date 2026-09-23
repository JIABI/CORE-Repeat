import numpy as np

from opal2.conditional_joint_error_experiment import observable_forward, score_distribution
from opal2.empirical_radial import fit_radial, radial_cdf, variance_multiplier
from opal2.empirical_radial_experiment import LEVELS, score, interval_levels


def fixture():
    rng = np.random.default_rng(13)
    mean = rng.normal(size=(3, 9))*.1
    residual = rng.normal(size=(3, 9))*.2
    factor = rng.normal(size=(3, 9, 9))*.02
    scatter = factor@factor.swapaxes(-1, -2)+np.eye(9)*.05
    stats = dict(u_center=np.zeros(9), u_scale=np.full(9, .2))
    actual, obs, diff, _ = observable_forward((mean+residual)*.2)
    return mean, scatter, mean+residual, stats, actual, obs, np.log1p(diff), np.ones(3)


def test_gaussian_replay_same_metrics_and_regions():
    args = fixture()
    expected = score_distribution(*args, seed=493, samples=200)
    actual = score(*args, seed=493, samples=200)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-13, rtol=1e-13, err_msg=key)
    assert actual['observable_coverage_by_level'].shape == (3, 10, 5)
    assert np.all(np.diff(actual['joint_squared_radius_by_level']) > 0)


def test_empirical_full_joint_scores_moments_and_multi_level_intervals():
    args = fixture()
    law = fit_radial(np.linspace(.8, 5., 40))
    weights = np.full((3, 40), 1/40)
    result = score(*args, seed=493, law=law, weights=weights, samples=500)
    assert all(np.isfinite(v).all() for v in result.values())
    np.testing.assert_allclose(result['covariance_u'], args[1]*variance_multiplier(law, weights)[:, None, None])
    for column, level in enumerate(LEVELS):
        q = np.sqrt(result['joint_squared_radius_by_level'][:, column])
        np.testing.assert_allclose(radial_cdf(law, weights, q), level, atol=2e-11)
    np.testing.assert_array_equal(result['coverage'], result['gamma_coverage_by_level'][:, 3])
    np.testing.assert_array_equal(result['observable_coverage'], result['observable_coverage_by_level'][..., 3])


def test_all_interval_levels_are_nested():
    rng = np.random.default_rng(21)
    draws = rng.normal(size=(300, 3, 4))
    cov, widths, lower, upper = interval_levels(draws, np.ones((3, 4)))
    assert cov.shape == (3, 4, 5)
    assert np.all(np.diff(cov.astype(int), axis=-1) >= 0)
    assert np.all(np.diff(widths, axis=-1) >= 0)
    assert np.all(np.diff(lower, axis=0) <= 0)
    assert np.all(np.diff(upper, axis=0) >= 0)
