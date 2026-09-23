import numpy as np
from scipy.stats import chi2

from opal2.joint_tail_calibration_experiment import region_diagnostics


def test_same_region_does_not_require_same_probability_law():
    covariance = np.broadcast_to(np.diag(np.arange(1, 10)), (4, 9, 9)).copy()
    mahal = np.array([3., 12., 23., 55.])
    factor = 2.1
    threshold = chi2.ppf(.95, 9)
    region_only = region_diagnostics(covariance, mahal, factor*threshold)
    scaled_law = region_diagnostics(factor*covariance, mahal/factor, threshold)
    np.testing.assert_array_equal(region_only['joint_coverage'], scaled_law['joint_coverage'])
    np.testing.assert_allclose(region_only['region_log_volume_without_unit_ball'],
                               scaled_law['region_log_volume_without_unit_ball'])
    np.testing.assert_allclose(region_only['region_radius_factor'], np.sqrt(factor))
    np.testing.assert_allclose(scaled_law['region_radius_factor'], 1.)
