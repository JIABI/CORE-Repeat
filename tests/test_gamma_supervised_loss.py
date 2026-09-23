"""Synthetic formula/gradient checks, not new experimental observations."""
from io import BytesIO

import pytest
import torch

from opal2.gamma_supervised_loss import JointGammaCRPS, gamma_from_raw_coordinates
from opal2.gram_factor_verified import decode_draws
from opal2.gram_geometry import gram_gains


DTYPE = torch.float64


def objective(covariance=None, center=None, scale=None, gamma_scale=.07):
    return JointGammaCRPS(
        torch.eye(9, dtype=DTYPE)*.025 if covariance is None else covariance,
        torch.zeros(9, dtype=DTYPE) if center is None else center,
        torch.ones(9, dtype=DTYPE) if scale is None else scale,
        gamma_scale,
    )


def test_native_gamma_matches_original_gram_functional_and_shapes():
    rng = torch.Generator().manual_seed(194)
    native = torch.randn(3, 4, 9, dtype=DTYPE, generator=rng)*.35
    gram, audit = decode_draws(native)
    expected = gram_gains(torch.from_numpy(gram))[..., 2]
    actual = gamma_from_raw_coordinates(native)
    assert actual.shape == (3, 4)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=2e-16)
    assert audit['draw_object_count'] == 12
    torch.testing.assert_close(gamma_from_raw_coordinates(native[0, 0]), actual[0, 0])


def test_high_condition_number_legal_factor_matches_verified_forward():
    native = torch.tensor([[1e10, 1.4e10, 1.1e10, 0., .25, 0., -.2, .35, 0.]], dtype=DTYPE)
    gram, audit = decode_draws(native)
    assert audit['recovered_schur_failure_count'] == 1
    expected = gram_gains(torch.from_numpy(gram))[..., 2]
    torch.testing.assert_close(gamma_from_raw_coordinates(native), expected, rtol=0, atol=1e-15)


def test_full_covariance_reparameterization_and_native_scale_restore():
    lower = torch.eye(9, dtype=DTYPE)*.3
    lower[1, 0] = .2
    lower[6, 2] = -.17
    lower[8, 3] = .11
    covariance = lower@lower.T
    center = torch.linspace(-.2, .2, 9, dtype=DTYPE)
    scale = torch.linspace(.6, 1.4, 9, dtype=DTYPE)
    model = objective(covariance, center, scale)
    mean = torch.linspace(-.1, .1, 9, dtype=DTYPE)[None]
    epsilon = torch.eye(9, dtype=DTYPE)[:, None, :]
    sampled = model.sample_standardized_coordinates(mean, epsilon)
    expected = mean[None]+epsilon@lower.T
    torch.testing.assert_close(sampled, expected, rtol=1e-14, atol=1e-15)
    # Off-diagonal entries really move joint coordinates, not independent heads.
    assert sampled[0, 0, 1]-mean[0, 1] == pytest.approx(.2)
    target = torch.tensor([.1], dtype=DTYPE)
    result = model(mean, target, epsilon, -epsilon.flip(0))
    torch.testing.assert_close(result['gamma_samples_a'], gamma_from_raw_coordinates(expected*scale+center))
    torch.testing.assert_close(result['normalized_gamma_crps'], result['gamma_crps']/.07)
    torch.testing.assert_close(result['gamma_prediction_mse'], (result['gamma_prediction_mean']-target).square().mean())


def test_gamma_and_crps_mean_autograd_gradcheck():
    generator = torch.Generator().manual_seed(552)
    native = (torch.randn(2, 9, dtype=DTYPE, generator=generator)*.2).requires_grad_()
    assert torch.autograd.gradcheck(gamma_from_raw_coordinates, (native,), eps=1e-6, atol=2e-6)
    model = objective()
    mean = (torch.randn(2, 9, dtype=DTYPE, generator=generator)*.2).requires_grad_()
    a = torch.randn(4, 2, 9, dtype=DTYPE, generator=generator)
    b = torch.randn(4, 2, 9, dtype=DTYPE, generator=generator)
    target = torch.tensor([.137, -.117], dtype=DTYPE)
    fn = lambda m: model(m, target, a, b)['normalized_gamma_crps']
    assert torch.autograd.gradcheck(fn, (mean,), eps=1e-6, atol=2e-6, rtol=1e-4)
    fn(mean).backward()
    assert torch.isfinite(mean.grad).all() and mean.grad.abs().sum() > 0
    assert not list(model.parameters())
    assert all(not value.requires_grad for value in model.buffers())


def test_independent_cartesian_pair_formula_and_coupled_pair_bias():
    # Enumerate a two-point *synthetic* distribution exactly; this tests the
    # CRPS identity, not whether these discrete epsilons are Gaussian draws.
    model = objective(torch.eye(9, dtype=DTYPE))
    mean = torch.zeros(1, 9, dtype=DTYPE)
    support = torch.zeros(2, 1, 9, dtype=DTYPE)
    support[0, 0, 2], support[1, 0, 2] = -1., 1.
    a, b = support[[0, 0, 1, 1]], support[[0, 1, 0, 1]]
    target = torch.tensor([.1], dtype=DTYPE)
    result = model(mean, target, a, b)
    g = gamma_from_raw_coordinates(support)
    exact = (g-target).abs().mean()-.5*(g[:, None]-g[None, :]).abs().mean()
    torch.testing.assert_close(result['gamma_crps'], exact)
    coupled = model(mean, target, a, a)
    assert coupled['gamma_crps'] > result['gamma_crps']
    torch.testing.assert_close(coupled['half_independent_pair_distance'], torch.tensor(0., dtype=DTYPE))
    assert result['gamma_crps'] >= 0


def test_deterministic_limit_is_absolute_error():
    model = objective(torch.eye(9, dtype=DTYPE)*1e-20)
    mean = torch.linspace(-.2, .2, 9, dtype=DTYPE)[None]
    generator = torch.Generator().manual_seed(741)
    a = torch.randn(8, 1, 9, dtype=DTYPE, generator=generator)
    b = torch.randn(8, 1, 9, dtype=DTYPE, generator=generator)
    target = torch.tensor([.17], dtype=DTYPE)
    expected = (gamma_from_raw_coordinates(mean)-target).abs().mean()
    torch.testing.assert_close(model(mean, target, a, b)['gamma_crps'], expected, rtol=0, atol=1e-9)


def test_frozen_buffers_checkpoint_roundtrip_and_input_copy():
    center = torch.linspace(-.3, .3, 9, dtype=DTYPE)
    model = objective(center=center)
    center.add_(10)
    assert model.target_center.max() < 1
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = objective(gamma_scale=.9)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    mean = torch.zeros(2, 9, dtype=DTYPE)
    generator = torch.Generator().manual_seed(612)
    a = torch.randn(5, 2, 9, dtype=DTYPE, generator=generator)
    b = torch.randn(5, 2, 9, dtype=DTYPE, generator=generator)
    target = torch.tensor([.02, -.03], dtype=DTYPE)
    one, two = model(mean, target, a, b), restored(mean, target, a, b)
    for key, value in one.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, two[key], rtol=0, atol=0)


@pytest.mark.parametrize('which', ['shape', 'asymmetric', 'not_spd', 'nonfinite', 'scale_zero', 'scale_shape', 'center_nan', 'gamma_zero'])
def test_invalid_frozen_laws_rejected(which):
    covariance = torch.eye(9, dtype=DTYPE)
    center, scale, gamma_scale = torch.zeros(9, dtype=DTYPE), torch.ones(9, dtype=DTYPE), .1
    if which == 'shape': covariance = covariance[:8, :8]
    elif which == 'asymmetric': covariance[1, 0] = .2
    elif which == 'not_spd': covariance[0, 0] = 0
    elif which == 'nonfinite': covariance[0, 0] = float('nan')
    elif which == 'scale_zero': scale[0] = 0
    elif which == 'scale_shape': scale = scale[:8]
    elif which == 'center_nan': center[0] = float('nan')
    elif which == 'gamma_zero': gamma_scale = 0
    with pytest.raises(ValueError): JointGammaCRPS(covariance, center, scale, gamma_scale)


@pytest.mark.parametrize('logdiag', [-1000., -400., 1000.])
def test_factor_underflow_or_overflow_rejected_without_replacement(logdiag):
    native = torch.zeros(1, 9, dtype=DTYPE)
    native[0, 3] = logdiag
    with pytest.raises(FloatingPointError): gamma_from_raw_coordinates(native)


def test_bad_runtime_shapes_and_nonfinite_rejected():
    model = objective()
    mean = torch.zeros(2, 9, dtype=DTYPE)
    target = torch.zeros(2, dtype=DTYPE)
    epsilon = torch.zeros(3, 2, 9, dtype=DTYPE)
    with pytest.raises(ValueError): model(mean[:, :8], target, epsilon, epsilon)
    with pytest.raises(ValueError): model(mean, target[:, None], epsilon, epsilon)
    with pytest.raises(ValueError): model(mean, target, epsilon[:0], epsilon[:0])
    with pytest.raises(ValueError): model(mean, target, epsilon, epsilon[:2])
    bad = mean.clone(); bad[0, 0] = float('nan')
    with pytest.raises(ValueError): model(bad, target, epsilon, epsilon)
    with pytest.raises(ValueError): gamma_from_raw_coordinates(torch.zeros(8, dtype=DTYPE))
