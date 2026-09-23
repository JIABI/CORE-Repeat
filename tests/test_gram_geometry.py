"""Synthetic engineering checks; no experiment profiles or protected data."""
import pytest
import torch

from opal2.gram_geometry import (
    ACTION_NAMES, COORDINATE_NAMES, OBSERVABLE_NAMES, COSINE_PAIRS,
    profiles_to_gram, gram_to_coordinates, coordinates_to_gram,
    gram_observables, gram_gains,
)


def profiles(shape=(7, 4, 13), seed=42):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def original_gains(y, cost_one=.01, cost_two=.02):
    x, z1, z2, v = y.unbind(-2)
    def cosine(a, b):
        return (a*b).sum(-1)/(a.square().sum(-1)*b.square().sum(-1)).sqrt()
    before = cosine(x, v)
    return torch.stack((
        .5*(cosine((x+z1)/2, v)-before)-cost_one,
        .5*(cosine((x+z2)/2, v)-before)-cost_one,
        .5*(cosine((x+z1+z2)/3, v)-before)-cost_two), -1)


def test_original_action_formula_all_samples_and_batch_dimensions():
    y = profiles((5, 7, 4, 13))
    expected = original_gains(y)
    for normalize in (True, False):
        gram = profiles_to_gram(y, normalize_x=normalize)
        assert gram.shape == (5, 7, 4, 4)
        actual = gram_gains(gram)
        assert actual.shape == (5, 7, 3)
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    assert len(ACTION_NAMES) == 3


def test_positive_definite_nine_coordinate_round_trip():
    gram = profiles_to_gram(profiles((3, 5, 4, 13)))
    u = gram_to_coordinates(gram)
    assert u.shape == (3, 5, 9)
    assert len(COORDINATE_NAMES) == 9
    recovered = coordinates_to_gram(u)
    torch.testing.assert_close(recovered, gram, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(gram_to_coordinates(recovered), u, rtol=2e-13, atol=2e-13)
    assert torch.all(torch.linalg.eigvalsh(recovered) > 0)
    assert torch.equal(recovered[..., 0, 0], torch.ones((3, 5), dtype=torch.float64))


def test_coordinate_order_and_schur_psd_are_explicit():
    u = torch.tensor([.3, -.7, 1.2, .2, .8, -.4, -.5, .9, .1], dtype=torch.float64)
    gram = coordinates_to_gram(u)
    p = u[:3]
    factor = torch.tensor([[torch.exp(u[3]), 0., 0.],
                           [u[4], torch.exp(u[5]), 0.],
                           [u[6], u[7], torch.exp(u[8])]], dtype=torch.float64)
    torch.testing.assert_close(gram[1:, 0], p)
    torch.testing.assert_close(gram[1:, 1:]-torch.outer(p, p), factor@factor.T)
    assert torch.all(torch.linalg.eigvalsh(gram) > 0)


def test_block_scale_invariance_and_single_well_amplitude_matters():
    y = profiles()
    scaled = y*torch.linspace(.2, 7., 7, dtype=torch.float64)[:, None, None]
    gram = profiles_to_gram(y)
    torch.testing.assert_close(profiles_to_gram(scaled), gram)
    torch.testing.assert_close(gram_gains(profiles_to_gram(scaled)), gram_gains(gram))
    changed = y.clone()
    changed[:, 1] *= 4
    changed_gram = profiles_to_gram(changed)
    assert not torch.allclose(changed_gram, gram)
    # Scaling Z1 alone preserves its pairwise cosine but changes its weight in
    # (X+Z1)/2. A correlation-only output would miss this action difference.
    torch.testing.assert_close(gram_observables(changed_gram)[..., :6], gram_observables(gram)[..., :6])
    assert not torch.allclose(gram_gains(changed_gram)[..., 0], gram_gains(gram)[..., 0])


def test_twenty_observables_match_explicit_profile_geometry():
    y = profiles((3, 5, 4, 13))
    x, z1, z2, v = y.unbind(-2)
    norm2 = y.square().sum(-1)
    s = norm2[..., 0]
    cosine = torch.stack([(y[..., i, :]*y[..., j, :]).sum(-1)/(norm2[..., i]*norm2[..., j]).sqrt()
                          for i, j in COSINE_PAIRS], -1)
    vectors = (z1-z2, z1-v, z2-v, (z1+z2)/2, (z1+v)/2, (z2+v)/2,
               (x+z1)/2, (x+z2)/2, (x+z1+z2)/3, (z1+z2+v)/3)
    energies = torch.stack([a.square().sum(-1)/s for a in vectors], -1)
    expected = torch.cat((cosine, (norm2/s[..., None]).sqrt(), energies), -1)
    actual = gram_observables(profiles_to_gram(y))
    assert actual.shape == (3, 5, 20)
    assert len(OBSERVABLE_NAMES) == len(set(OBSERVABLE_NAMES)) == 20
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(gram_observables(profiles_to_gram(y, False)), expected)


def test_joint_sampling_shape_and_nonlinear_pushforward_not_mean_plug_in():
    u = profiles((19, 4, 9))*.35
    gram = coordinates_to_gram(u)
    draws = gram_gains(gram)
    assert draws.shape == (19, 4, 3)
    assert gram_observables(gram).shape == (19, 4, 20)
    assert not torch.allclose(draws.mean(0), gram_gains(gram.mean(0)), atol=1e-5, rtol=1e-5)


def test_differentiable_decoding_and_action_pushforward():
    u = (profiles((2, 9))*.2).requires_grad_()
    assert torch.autograd.gradcheck(lambda value: gram_gains(coordinates_to_gram(value)),
                                    (u,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_raise_without_implicit_completion(bad):
    y = profiles()
    y[0, 1, 2] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        profiles_to_gram(y)
    u = torch.zeros(9, dtype=torch.float64)
    u[0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        coordinates_to_gram(u)


def test_zero_x_and_zero_validation_are_not_replaced_by_epsilon():
    y = profiles()
    y[0, 0] = 0
    with pytest.raises(ValueError, match="zero"):
        profiles_to_gram(y)
    y = profiles()
    y[0, 3] = 0
    gram = profiles_to_gram(y)
    with pytest.raises(ValueError, match="V.*zero"):
        gram_gains(gram)
    with pytest.raises(ValueError, match="zero"):
        gram_observables(gram)


def test_zero_acquired_mean_preserves_undefined_original_action():
    y = profiles()
    y[:, 1] = -y[:, 0]
    with pytest.raises(ValueError, match="acquired average"):
        gram_gains(profiles_to_gram(y))


def test_singular_schur_boundary_is_rejected_not_regularized():
    gram = torch.ones((4, 4), dtype=torch.float64)
    # The original action still has well-defined nonzero norms, but finite
    # log-Cholesky coordinates cannot express this rank-one geometry.
    torch.testing.assert_close(gram_gains(gram), torch.tensor([-.01, -.01, -.02], dtype=torch.float64))
    with pytest.raises(ValueError, match="positive definite"):
        gram_to_coordinates(gram)


def test_invalid_geometry_costs_shapes_and_coordinate_normalization():
    y = profiles()
    with pytest.raises(ValueError, match="G00"):
        gram_to_coordinates(profiles_to_gram(y, False))
    indefinite = torch.eye(4, dtype=torch.float64)
    indefinite[1, 2] = indefinite[2, 1] = 2
    with pytest.raises(ValueError, match="positive semidefinite"):
        gram_gains(indefinite)
    nonsymmetric = torch.eye(4, dtype=torch.float64)
    nonsymmetric[1, 2] = .5
    with pytest.raises(ValueError, match="symmetric"):
        gram_observables(nonsymmetric)
    with pytest.raises(ValueError, match="four declared roles"):
        profiles_to_gram(torch.ones((3, 12), dtype=torch.float64))
    with pytest.raises(ValueError, match="Nine"):
        coordinates_to_gram(torch.zeros(8, dtype=torch.float64))
    with pytest.raises(ValueError, match="cost"):
        gram_gains(profiles_to_gram(y), cost_one=-1)


@pytest.mark.parametrize("log_value", [-1000., 1000.])
def test_log_diagonal_extremes_fail_explicitly(log_value):
    u = torch.zeros(9, dtype=torch.float64)
    u[3] = log_value
    with pytest.raises(ValueError, match="overflowed or underflowed"):
        coordinates_to_gram(u)


def test_float32_is_preserved_and_small_error_is_numerical_only():
    y = profiles().float()
    gram = profiles_to_gram(y)
    assert gram.dtype == torch.float32
    u = gram_to_coordinates(gram)
    assert u.dtype == torch.float32
    torch.testing.assert_close(coordinates_to_gram(u), gram, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(gram_gains(gram), original_gains(y), atol=2e-6, rtol=2e-6)


def test_finite_diagonal_does_not_hide_squared_underflow_or_lost_schur():
    underflow = torch.zeros(9, dtype=torch.float64)
    underflow[3] = -400.  # exp(-400) is positive, but its square is zero.
    with pytest.raises(ValueError, match="floating-point singular boundary"):
        coordinates_to_gram(underflow)
    lost_residual = torch.zeros(9, dtype=torch.float64)
    lost_residual[:3] = 1e10  # Adding a unit residual to p pᵀ loses it.
    with pytest.raises(ValueError, match="floating-point singular boundary"):
        coordinates_to_gram(lost_residual)
