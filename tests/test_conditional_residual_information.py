import numpy as np
import torch

from opal2.conditional_residual_information import (
    ScalePredictor, bounded_scale_ratio, extend_scatter, error_targets,
    ResidualStateNetwork, projection_nll,
)


def test_geometry_scales_off_exact_and_positive():
    rng = np.random.default_rng(13)
    mean = rng.normal(0, .1, (15, 9))
    cov = np.broadcast_to(np.eye(9), (15, 9, 9)).copy()
    ratio = bounded_scale_ratio(np.exp(rng.normal(size=(15, 2))), np.ones((15, 2)))
    assert np.all((ratio > .25) & (ratio < 4))
    assert np.array_equal(extend_scatter(mean, np.ones(9), cov, ratio, enabled=False), cov)
    assert np.array_equal(extend_scatter(mean, np.ones(9), cov, np.ones((15, 2))), cov)
    changed = extend_scatter(mean, np.ones(9), cov, ratio)
    assert np.linalg.eigvalsh(changed).min() > 0
    error = rng.normal(size=(15, 9))
    energies = error_targets(mean, cov, error)
    np.testing.assert_allclose(energies.sum(1), np.square(error).sum(1), atol=1e-12)


def test_boosting_positive_nonconstant_and_no_query_targets():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(180, 4))
    energy = np.exp(x[:, :2])*np.array([3., 6.])
    fit = ScalePredictor.fit(x, energy, [0, 1, 2, 3], np.arange(180).astype(str))
    p = fit.predict(x[:8])
    assert p.shape == (8, 2) and np.all(p > 0) and p.std() > .1
    np.testing.assert_array_equal(p, fit.predict(x[:8].copy()))


def test_latent_is_trained_on_distribution_not_random_error_sign():
    torch.manual_seed(3)
    model = ResidualStateNetwork(5, 3).double()
    x, d = torch.randn(16, 5, dtype=torch.float64), torch.randn(16, 3, dtype=torch.float64)
    offset = torch.zeros(16, 2, dtype=torch.float64)
    eta, _, correction = model(x, d, offset)
    assert torch.equal(correction, torch.zeros_like(correction))
    energy = torch.exp(torch.randn(16, 2, dtype=torch.float64))
    loss = projection_nll(eta, energy).mean()
    loss.backward()
    assert model.scale_head.weight.grad.abs().sum() > 0
