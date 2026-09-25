"""Engineering checks of the new observable support, not scientific evidence."""
import json

import numpy as np
import pytest
from scipy.integrate import quad

from opal2.observable_quantile_distribution import (
    GRID, NonnegativeQuantileLaw, fit_quantile_offsets,
    make_nonnegative_quantile_law,
)
from opal2.quantile_distribution import (
    QuantileLaw, fit_quantile_offsets as gamma_fit_quantile_offsets,
)


def test_uniform_law_reuses_exact_scores_without_gamma_upper_clipping():
    law = make_nonnegative_quantile_law(np.tile(4 * GRID, (4, 1)))
    assert isinstance(law, (NonnegativeQuantileLaw, QuantileLaw))
    np.testing.assert_allclose(law.quantiles[:, 1:-1], np.tile(4 * GRID, (4, 1)))
    np.testing.assert_allclose(law.quantile([0., .5, 1.]),
                               np.tile([0., 2., 4.], (4, 1)), atol=1e-15)
    np.testing.assert_allclose(law.mean(), 2., atol=1e-15)
    np.testing.assert_allclose(law.cdf([0., 1., 2., 8.]), [0., .25, .5, 1.])
    np.testing.assert_allclose(law.crps([0., 1., 2., 8.]),
                               [4/3, 7/12, 1/3, 16/3], atol=2e-15)
    interval = law.interval_metrics([0., 1., 2., 8.])
    np.testing.assert_allclose(interval['width'],
                               np.tile(4 * interval['nominal'], (4, 1)), atol=2e-15)


def test_crossings_zero_floor_and_adjacent_extreme_tail_extrapolation():
    levels = np.array([.1, .25, .5, .75, .9])
    law = make_nonnegative_quantile_law([5., -2., 1., 3., .5], levels)
    np.testing.assert_array_equal(law.quantiles[0, 1:-1], [0., .5, 1., 3., 5.])
    assert law.quantiles[0, 0] == 0.
    np.testing.assert_allclose(law.quantiles[0, -1], 5. + .1 * 2. / .15)
    assert np.all(np.diff(law.quantiles, axis=1) >= 0.)
    assert law.cdf(-1.)[0] == 0.


def test_zero_atoms_and_unchanged_calibration_offsets():
    # A half-zero atom plus half Uniform(0,2), formed by the support floor.
    law = make_nonnegative_quantile_law(4 * GRID - 2.)
    assert law.cdf(0.)[0] == .5
    np.testing.assert_allclose(law.mean(), [.5], atol=1e-15)
    np.testing.assert_allclose(law.crps(0.), [1/6], atol=1e-15)
    assert fit_quantile_offsets is gamma_fit_quantile_offsets
    cal = np.ones((8, len(GRID)))
    offsets = fit_quantile_offsets(cal, np.zeros(8))
    np.testing.assert_array_equal(offsets, -np.ones(len(GRID)))
    calibrated = make_nonnegative_quantile_law(cal[:2] + offsets)
    np.testing.assert_array_equal(calibrated.quantiles, np.zeros((2, len(GRID) + 2)))
    np.testing.assert_array_equal(calibrated.cdf(0.), [1., 1.])
    np.testing.assert_array_equal(calibrated.crps([0., 2.]), [0., 2.])


def test_exact_crps_matches_quadrature_for_irregular_nonnegative_tails():
    levels = np.array([.1, .25, .5, .75, .9])
    law = make_nonnegative_quantile_law([[4., -1., 1., 2., .5],
                                       [1., 2., 3., 4., 5.]], levels)
    outcomes = np.array([1.3, 8.])
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


def test_metadata_is_strict_json_and_distinguishes_numerical_tail_endpoint():
    metadata = make_nonnegative_quantile_law(3. + GRID).metadata()
    encoded = json.dumps(metadata, allow_nan=False)
    assert json.loads(encoded)['bounds'] == [0., None]
    assert metadata['upper_physical_bound'] is None
    assert 'not a physical upper bound' in metadata['finite_upper_endpoint']
    assert metadata['monte_carlo_samples'] == 0


def test_nonfinite_predictions_and_invalid_levels_are_rejected():
    with pytest.raises(ValueError, match='finite nonempty'):
        make_nonnegative_quantile_law(np.full(len(GRID), np.nan))
    with pytest.raises(ValueError, match='increase strictly'):
        make_nonnegative_quantile_law([1., 2.], levels=[.5, .5])
