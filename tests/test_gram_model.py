"""Joint Gram model mechanics on synthetic tensors, not biological results."""
from copy import deepcopy

import pytest
import torch

from opal2.gram_model import GRAM_DIM, GramConditionalModel, log_profile_norm


def example():
    torch.manual_seed(450)
    model = GramConditionalModel(
        {"Cells_Intensity": [0, 1, 2], "Nuclei_Texture": [3, 4, 5, 6]},
        hidden_dim=16, attention_heads=4, attention_layers=2).double()
    x = torch.randn(6, 7, dtype=torch.float64)
    descriptor = log_profile_norm(x)
    target = torch.randn(6, GRAM_DIM, dtype=torch.float64)
    return model, x, descriptor, target


def test_full_joint_identity_initialization_sampling_and_log_density():
    model, x, descriptor, target = example()
    prediction = model(x, descriptor)
    assert prediction.mean.shape == (6, 9)
    torch.testing.assert_close(prediction.covariance_matrix,
                               torch.eye(9, dtype=x.dtype).expand(6, 9, 9), rtol=0, atol=0)
    torch.testing.assert_close(prediction.log_prob(target),
        torch.distributions.MultivariateNormal(prediction.mean,
                                               scale_tril=prediction.scale_tril).log_prob(target))
    draws = prediction.sample(17, torch.Generator().manual_seed(20))
    assert draws.shape == (17, 6, 9)
    assert not draws.requires_grad
    torch.testing.assert_close(draws, prediction.sample(17, torch.Generator().manual_seed(20)),
                               rtol=0, atol=0)
    assert prediction.rsample(2).requires_grad


def test_non_diagonal_covariance_is_learnable_and_sampling_preserves_it():
    model, x, descriptor, target = example()
    with torch.no_grad():
        model.covariance_head[-1].bias[9] = .75
        model.covariance_head[-1].bias[10:] = .10
    prediction = model(x[:1], descriptor[:1])
    covariance = prediction.covariance_matrix[0]
    assert covariance[0, 1].abs() > .1
    assert torch.linalg.eigvalsh(covariance).min() >= model.covariance_shrinkage - 1e-12
    samples = prediction.sample(25000, torch.Generator().manual_seed(11))[:, 0]
    torch.testing.assert_close(torch.cov(samples.T), covariance, atol=.035, rtol=.04)
    marginal_sum = torch.distributions.Normal(prediction.mean, prediction.stddev).log_prob(target[:1]).sum(-1)
    assert not torch.allclose(prediction.log_prob(target[:1]), marginal_sum)


def test_covariance_nll_cannot_update_encoder_or_mean_but_mean_mse_can():
    model, x, descriptor, target = example()
    mean_ids = {id(p) for p in model.mean_parameters()}
    covariance_ids = {id(p) for p in model.covariance_parameters()}
    assert not mean_ids & covariance_ids
    assert mean_ids | covariance_ids == {id(p) for p in model.parameters()}
    model.loss(x, descriptor, target)["covariance_nll"].backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.mean_parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.covariance_parameters())
    model.zero_grad(set_to_none=True)
    model.loss(x, descriptor, target)["mean_mse"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.profile_encoder.parameters())
    assert all(p.grad is None for p in model.covariance_parameters())


def test_separate_updates_preserve_mean_only_trajectory_and_have_finite_gradients():
    model, x, descriptor, target = example()
    mean_only = deepcopy(model)
    opt_m = torch.optim.AdamW(mean_only.mean_parameters(), lr=.001)
    opt_f_m = torch.optim.AdamW(model.mean_parameters(), lr=.001)
    opt_f_c = torch.optim.AdamW(model.covariance_parameters(), lr=.001)
    initial_cov = model(x, descriptor).covariance_matrix.detach().clone()
    for _ in range(3):
        opt_m.zero_grad(set_to_none=True)
        opt_f_m.zero_grad(set_to_none=True)
        opt_f_c.zero_grad(set_to_none=True)
        mean_only.loss(x, descriptor, target)["mean_mse"].backward()
        model.loss(x, descriptor, target)["loss"].backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
        torch.nn.utils.clip_grad_norm_(mean_only.mean_parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.mean_parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.covariance_parameters(), 1.)
        opt_m.step(); opt_f_m.step(); opt_f_c.step()
        torch.testing.assert_close(model(x, descriptor).mean, mean_only(x, descriptor).mean,
                                   rtol=0, atol=0)
    assert not torch.equal(initial_cov, model(x, descriptor).covariance_matrix)


def test_covariance_bounds_and_positive_definiteness_under_extreme_parameters():
    model, x, descriptor, _ = example()
    with torch.no_grad():
        model.covariance_head[-1].bias[:9] = torch.linspace(-100, 100, 9)
        model.covariance_head[-1].bias[9:] = torch.linspace(-100, 100, 36)
    prediction = model(x, descriptor)
    covariance = prediction.covariance_matrix
    assert torch.isfinite(covariance).all()
    assert torch.linalg.eigvalsh(covariance).min() >= model.covariance_shrinkage - 1e-10
    lower = (1 - model.covariance_shrinkage) * torch.exp(torch.tensor(-8.)) + model.covariance_shrinkage
    upper = (1 - model.covariance_shrinkage) * torch.exp(torch.tensor(8.)) + model.covariance_shrinkage
    assert prediction.variance.min() >= lower - 1e-6
    assert prediction.variance.max() <= upper + 1e-3


def test_reconstruction_from_config_and_all_coordinates_available():
    model, x, descriptor, _ = example()
    loaded = GramConditionalModel(**model.config).double()
    loaded.load_state_dict(model.state_dict())
    expected = model(x, descriptor)
    actual = loaded(x, descriptor)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(actual.scale_tril, expected.scale_tril, rtol=0, atol=0)
    assert model.feature_dim == 7
    x = x.detach().requires_grad_()
    model(x, descriptor).mean.square().sum().backward()
    assert torch.all(x.grad.abs().sum(0) > 0)


def test_full_3617_input_has_only_nine_coordinate_output_not_a_spectrum_decoder():
    torch.manual_seed(26)
    model = GramConditionalModel({"Cells_Intensity": list(range(1800)),
                                  "Nuclei_Texture": list(range(1800, 3617))})
    x = torch.randn(2, 3617)
    prediction = model(x, log_profile_norm(x))
    assert model.feature_dim == 3617
    assert prediction.mean.shape == (2, 9)
    assert prediction.scale_tril.shape == (2, 9, 9)
    assert torch.isfinite(prediction.covariance_matrix).all()
    assert all(layer.out_features != 3617 for layer in model.modules()
               if isinstance(layer, torch.nn.Linear))


def test_log_norm_is_observed_raw_quantity_and_invalid_values_are_rejected():
    torch.testing.assert_close(log_profile_norm(torch.tensor([[3., 4.]])),
                               torch.tensor([[5.]]).log())
    assert torch.isfinite(log_profile_norm(torch.zeros(1, 7))).all()
    model, x, descriptor, target = example()
    with pytest.raises(ValueError, match="shape"):
        model(x.unsqueeze(1), descriptor)
    with pytest.raises(ValueError, match="log_raw_norm"):
        model(x, descriptor.expand(-1, 2))
    bad = x.clone(); bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(bad, descriptor)
    target[0, 0] = float("nan")
    with pytest.raises(ValueError, match="target"):
        model.loss(x, descriptor, target)
    with pytest.raises(ValueError, match="shrinkage"):
        GramConditionalModel({"all": [0, 1]}, covariance_shrinkage=0.)
    with pytest.raises(ValueError, match="n_samples"):
        model(x, descriptor).sample(0)
