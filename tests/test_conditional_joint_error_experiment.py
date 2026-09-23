import numpy as np
import torch

from opal2.conditional_joint_error_experiment import observable_forward, score_distribution
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains, gram_observables


def test_forward_preserves_full_original_well_geometry():
    y = np.random.default_rng(14).normal(size=(12, 4, 40))
    g = profiles_to_gram(torch.tensor(y))
    u = gram_to_coordinates(g).numpy()
    gamma, logobs, diff, cos = observable_forward(u)
    obs = gram_observables(g).numpy()
    np.testing.assert_allclose(gamma, gram_gains(g).numpy()[:, 2], atol=1e-12)
    expected = np.concatenate((np.square(obs[:, 7:10]), obs[:, 10:16], obs[:, 19:20]), axis=1)
    np.testing.assert_allclose(logobs, np.log1p(expected), atol=1e-12)
    np.testing.assert_allclose(diff, obs[:, 10:13], atol=1e-12)
    np.testing.assert_allclose(cos, obs[:, (4, 5)], atol=1e-12)


def test_scoring_uses_joint_ellipsoid_and_keeps_mean_fixed():
    y = np.random.default_rng(4).normal(size=(4, 4, 25))
    raw = gram_to_coordinates(profiles_to_gram(torch.tensor(y))).numpy()
    actual, obs, diff, _ = observable_forward(raw)
    scale = np.square(y[:, 0]).mean(1)
    covariance = np.broadcast_to(.02*np.eye(9), (4, 9, 9)).copy()
    mean = raw.copy()
    scores = score_distribution(mean, covariance, raw,
        {'u_center': np.zeros(9), 'u_scale': np.ones(9)}, actual, obs,
        np.log1p(diff*scale[:, None]), scale, seed=3, samples=100)
    np.testing.assert_array_equal(mean, raw)
    assert np.all(scores['joint_coverage'] == 1)
    assert scores['observable_crps'].shape == (4, 10)
    assert scores['absolute_crps_by_pair'].shape == (4, 3)
    assert all(np.isfinite(v).all() for v in scores.values())
