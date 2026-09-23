"""Numerical engineering oracles, not experimental or publication evidence."""
import numpy as np
import pytest
from scipy.integrate import quad

from opal2.quantile_distribution import GRID, fit_quantile_offsets, make_quantile_law


def test_point_mass_mean_crps_and_inclusive_null_probability():
    law = make_quantile_law(np.tile([-.2, 0., .2], (len(GRID), 1)).T)
    np.testing.assert_allclose(law.mean(), [-.2, 0., .2], atol=1e-16)
    np.testing.assert_allclose(law.crps([0., .2, -.3]), [.2, .2, .5], atol=1e-15)
    np.testing.assert_array_equal(law.cdf(0.), [1., 1., 0.])
    np.testing.assert_array_equal(law.cdf(-np.inf), [0., 0., 0.])
    np.testing.assert_array_equal(law.cdf(np.inf), [1., 1., 1.])
    intervals = law.interval_metrics([-.2, .01, .2])
    np.testing.assert_array_equal(intervals['width'], np.zeros((3, 5)))
    np.testing.assert_array_equal(intervals['covered'],
                                  np.tile([True, False, True], (5, 1)).T)


def test_uniform_law_exact_oracles_and_intervals():
    law = make_quantile_law(np.tile(GRID, (4, 1)), bounds=(0., 1.))
    y = np.array([0., .25, .5, 2.])
    np.testing.assert_allclose(law.mean(), .5, atol=1e-15)
    np.testing.assert_allclose(law.cdf(y), [0., .25, .5, 1.], atol=1e-15)
    np.testing.assert_allclose(law.crps(y), [1/3, 7/48, 1/12, 4/3], atol=1e-15)
    result = law.interval_metrics(y)
    np.testing.assert_allclose(result['width'],
                               np.tile(result['nominal'], (4, 1)), atol=1e-15)
    np.testing.assert_allclose(law.quantile([0., .25, 1.]),
                               np.tile([0., .25, 1.], (4, 1)), atol=1e-15)


def test_plateau_is_atom_at_zero_and_cdf_is_right_continuous():
    # Half the mass is at zero and half is Uniform(0,1).
    law = make_quantile_law(np.maximum(0., 2*GRID - 1), bounds=(0., 1.))
    assert law.cdf(-1e-10)[0] == 0.
    assert law.cdf(0.)[0] == .5
    assert law.cdf(.25)[0] == .625
    assert law.cdf(1.)[0] == 1.
    np.testing.assert_allclose(law.mean(), [.25], atol=1e-15)
    np.testing.assert_allclose(law.crps(0.), [1/12], atol=1e-15)


def test_bounded_tails_and_crossing_rearrangement():
    levels = np.array([.01, .025, .5, .975, .99])
    raw = np.array([[.2, -1., -.5, .8, .96], [10., 4., 5., -3., -4.]])
    law = make_quantile_law(raw, levels=levels)
    np.testing.assert_array_equal(law.quantiles[0, 1:-1], [-1., -.5, .2, .8, .96])
    np.testing.assert_array_equal(law.quantiles[:, 0], [-1.02, -1.02])
    np.testing.assert_array_equal(law.quantiles[:, -1], [.98, .98])
    assert np.all(np.diff(law.quantiles, axis=1) >= 0)
    assert law.metadata()['monte_carlo_samples'] == 0


def test_exact_crps_matches_independent_adaptive_quadrature():
    rng = np.random.default_rng(710)
    raw = rng.normal(0., .4, (7, len(GRID)))
    raw[0] = 0.
    raw[1, :10] = -.25
    law = make_quantile_law(raw)
    outcomes = rng.uniform(-1.2, 1.2, len(raw))
    expected = []
    for q, y in zip(law.quantiles, outcomes):
        def integrand(p):
            residual = y - np.interp(p, law.probabilities, q)
            return 2 * residual * (p - (residual < 0))
        points = list(law.probabilities)
        for j in range(len(q) - 1):
            if q[j] < y < q[j + 1]:
                points.append(law.probabilities[j]
                              + (y-q[j])/(q[j+1]-q[j])
                              * (law.probabilities[j+1]-law.probabilities[j]))
        expected.append(quad(integrand, 0., 1., points=np.unique(points),
                             epsabs=1e-12, epsrel=1e-12)[0])
    np.testing.assert_allclose(law.crps(outcomes), expected, atol=2e-15, rtol=1e-14)


def test_gamma_cost_shift_preserves_scores_width_and_corresponding_event():
    rng = np.random.default_rng(2001)
    raw = rng.normal(0., .15, (6, len(GRID)))
    y = rng.normal(0., .1, 6)
    shift = -.06  # Difference between two declared two-well action costs.
    original = make_quantile_law(raw)
    shifted = make_quantile_law(raw + shift, bounds=(-1.02 + shift, .98 + shift))
    np.testing.assert_allclose(shifted.mean(), original.mean() + shift, atol=1e-15)
    np.testing.assert_allclose(shifted.crps(y + shift), original.crps(y), atol=1e-15)
    np.testing.assert_allclose(shifted.cdf(shift), original.cdf(0.), atol=1e-15)
    np.testing.assert_allclose(shifted.interval_metrics(y + shift)['width'],
                               original.interval_metrics(y)['width'], atol=1e-15)


def test_calibration_offsets_are_levelwise_empirical_residual_quantiles():
    y = np.arange(21, dtype=float) / 20 - .5
    cal = np.tile(GRID - .3, (len(y), 1))
    offsets = fit_quantile_offsets(cal, y)
    expected = np.array([np.quantile(y - cal[:, j], p, method='linear')
                         for j, p in enumerate(GRID)])
    np.testing.assert_array_equal(offsets, expected)
    query = np.tile(GRID - .3, (2, 1))
    law = make_quantile_law(query + offsets)
    np.testing.assert_allclose(law.quantile(GRID)[0],
                               np.quantile(y, GRID, method='linear'), atol=1e-15)
    # Query rows never enter offset fitting or change another row's law.
    law_extra = make_quantile_law(np.vstack([query, query + .4]) + offsets)
    np.testing.assert_array_equal(law.quantiles, law_extra.quantiles[:2])


def test_malformed_inputs_are_rejected():
    with pytest.raises(ValueError, match='increase strictly'):
        make_quantile_law([[1., 2.]], levels=[.5, .5])
    with pytest.raises(ValueError, match='finite nonempty'):
        make_quantile_law(np.full((1, len(GRID)), np.nan))
    with pytest.raises(ValueError, match='Bounds'):
        make_quantile_law(GRID, bounds=(1., 0.))
    with pytest.raises(ValueError, match='aligned'):
        fit_quantile_offsets(np.tile(GRID, (3, 1)), [1., 2.])
