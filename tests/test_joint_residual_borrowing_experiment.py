import numpy as np
import torch

from opal2.conditional_joint_error_experiment import observable_forward, score_distribution
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates
from opal2.joint_residual_distribution import build_residual_mixture
from opal2.joint_residual_borrowing_experiment import score_mixture


def test_distribution_scoring_preserves_fallback_and_original_functionals():
    rng = np.random.default_rng(122)
    y = rng.normal(size=(5, 4, 25))
    raw = gram_to_coordinates(profiles_to_gram(torch.tensor(y))).numpy()
    actual, obs, diff, _ = observable_forward(raw)
    scale = np.square(y[:, 0]).mean(1)
    covariance = np.broadcast_to(.04*np.eye(9), (5, 9, 9)).copy()
    reference = rng.normal(size=(35, 9))
    reference_cov = np.broadcast_to(np.eye(9), (35, 9, 9)).copy()
    weights = rng.uniform(size=(5, 35))
    weights /= weights.sum(1, keepdims=True)
    mixture = build_residual_mixture(reference, reference_cov, weights, covariance)
    stats = dict(u_center=np.zeros(9), u_scale=np.ones(9))
    absolute = np.log1p(diff*scale[:, None])
    base = score_distribution(raw, covariance, raw, stats, actual, obs, absolute, scale, 18, 200)
    fallback = score_mixture(raw, raw, mixture, 0., stats, actual, obs, absolute, scale, 18, 200)
    for key in base:
        np.testing.assert_array_equal(base[key], fallback[key])
    non_gaussian = score_mixture(raw, raw, mixture, .75, stats, actual, obs, absolute, scale, 18, 200)
    assert all(np.isfinite(v).all() for v in non_gaussian.values())
    assert non_gaussian['observable_crps'].shape == (5, 10)
    assert non_gaussian['coordinate_lower'].shape == (5, 9)
    assert np.all(non_gaussian['joint_squared_radius'] > 0)
    np.testing.assert_array_equal(non_gaussian['mahalanobis2'], fallback['mahalanobis2'])
