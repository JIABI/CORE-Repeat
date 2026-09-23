"""Mechanism tests on small tensors; no empirical biological claims."""
from copy import deepcopy

import pytest
import torch

from opal2.faithful import FaithfulMeasurementModel
from opal2.model import MeasurementWorldModel


def example():
    torch.manual_seed(802)
    model = MeasurementWorldModel({"Nuclei_Intensity": [0, 1, 2], "Cells_Texture": [3, 4, 5, 6]},
        4, 5, 6, hidden_dim=16, latent_rank=2, residual_rank=1,
        reference_loss_weight=0., chemical_regularization_weight=0.,
        dropout=0.).double()
    model.set_outcome_transform(torch.arange(7).double() / 10,
                                torch.arange(1, 8).double() / 2)
    b, c, t, d = 3, 2, 3, 7
    batch = {
        "context_y": torch.randn(b, c, d, dtype=torch.float64),
        "context_cond": torch.randn(b, c, 4, dtype=torch.float64),
        "context_mask": torch.ones(b, c, dtype=torch.bool),
        "context_reference": torch.randn(b, c, 3, 5, dtype=torch.float64),
        "context_reference_mask": torch.ones(b, c, 3, dtype=torch.bool),
        "context_group": torch.zeros(b, c, 3, dtype=torch.int64),
        "target_cond": torch.randn(b, t, 4, dtype=torch.float64),
        "target_reference": torch.randn(b, t, 3, 5, dtype=torch.float64),
        "target_reference_mask": torch.ones(b, t, 3, dtype=torch.bool),
        "target_group": torch.tensor([[[0, 0, 0], [0, 0, 1], [1, 0, 0]]]).expand(b, t, 3).clone(),
        "chem": torch.randn(b, 6, dtype=torch.float64),
    }
    target = torch.randn(b, t, d, dtype=torch.float64)
    return model, batch, target


def test_initial_distribution_matches_complete_base_and_reloads():
    base, batch, target = example()
    base.eval()
    original = base(batch)
    wrapper = FaithfulMeasurementModel(base).eval()
    copied = wrapper(batch)
    for name in ("mean", "diag_var", "factors", "local_factors", "environment_groups"):
        torch.testing.assert_close(getattr(copied, name), getattr(original, name), rtol=0, atol=0)
    torch.testing.assert_close(copied.joint_log_prob(target), original.joint_log_prob(target), rtol=0, atol=0)
    assert not ({id(p) for p in wrapper.mean_parameters()} & {id(p) for p in wrapper.uncertainty_parameters()})
    loaded = FaithfulMeasurementModel.from_config(wrapper.config).double().eval()
    loaded.load_state_dict(wrapper.state_dict())
    torch.testing.assert_close(loaded(batch).mean, copied.mean, rtol=0, atol=0)
    torch.testing.assert_close(loaded(batch).factors, copied.factors, rtol=0, atol=0)


def test_faithful_nll_has_no_gradient_to_any_mean_parameter():
    base, batch, target = example()
    wrapper = FaithfulMeasurementModel(base)
    result = wrapper.loss(batch, target, mode="faithful")
    result["faithful_nll"].backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in wrapper.mean_parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in wrapper.uncertainty_parameters())
    wrapper.zero_grad(set_to_none=True)
    wrapper.loss(batch, target, mode="joint")["loss"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in wrapper.mean_parameters())


def test_multiple_adam_steps_keep_M_F_mean_trajectory_identical():
    base, batch, target = example()
    mean = FaithfulMeasurementModel(deepcopy(base))
    faithful = FaithfulMeasurementModel(deepcopy(base))
    mean_parameters, faithful_mean = list(mean.mean_parameters()), list(faithful.mean_parameters())
    uncertainty = list(faithful.uncertainty_parameters())
    opt_m = torch.optim.AdamW(mean_parameters, lr=.001, weight_decay=.01)
    opt_f_m = torch.optim.AdamW(faithful_mean, lr=.001, weight_decay=.01)
    opt_f_u = torch.optim.AdamW(uncertainty, lr=.001, weight_decay=.01)
    initial_uncertainty = [p.detach().clone() for p in uncertainty]
    for step in range(4):
        opt_m.zero_grad(set_to_none=True)
        opt_f_m.zero_grad(set_to_none=True)
        opt_f_u.zero_grad(set_to_none=True)
        torch.manual_seed(40 + step)
        mean.loss(batch, target, mode="mean_only")["loss"].backward()
        torch.manual_seed(40 + step)
        faithful.loss(batch, target, mode="faithful")["loss"].backward()
        for left, right in zip(mean_parameters, faithful_mean):
            if left.grad is None:
                assert right.grad is None
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
        # A global all-parameter clip would destroy this invariant.
        torch.nn.utils.clip_grad_norm_(mean_parameters, .3)
        torch.nn.utils.clip_grad_norm_(faithful_mean, .3)
        torch.nn.utils.clip_grad_norm_(uncertainty, .3)
        opt_m.step(); opt_f_m.step(); opt_f_u.step()
        for left, right in zip(mean_parameters, faithful_mean):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(mean.exact_predictive_mean(batch), faithful.exact_predictive_mean(batch), rtol=0, atol=0)
    assert any(not torch.equal(left, right) for left, right in zip(initial_uncertainty, uncertainty))


def test_masks_physical_MSE_and_joint_density_jacobian():
    base, batch, target = example()
    wrapper = FaithfulMeasurementModel(base).eval()
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 1] = False
    mask[1, 2, 3:] = False
    target[~mask] = float("nan")
    result = wrapper.loss(batch, target, mask, mode="faithful")
    mean = wrapper.exact_predictive_mean(batch)
    expected = (((mean - target) * wrapper.outcome_scale)[mask].square()).mean()
    torch.testing.assert_close(result["mean_mse"], expected)
    assert int(result["observed_coordinates"]) == int(mask.sum())
    jacobian = wrapper.outcome_scale.log().expand_as(target)[mask].mean()
    torch.testing.assert_close(result["nll"] - result["standardized_joint_nll"], jacobian)
    result["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="observed"):
        wrapper.loss(batch, target, torch.zeros_like(mask), mode="mean_only")
    with pytest.raises(ValueError, match="boolean"):
        wrapper.loss(batch, target, mask.float(), mode="joint")


def test_covariance_updates_do_not_change_mean_and_shared_draws_survive():
    base, batch, target = example()
    wrapper = FaithfulMeasurementModel(base).eval()
    before = wrapper.exact_predictive_mean(batch).detach().clone()
    with torch.no_grad():
        for index, head in enumerate(wrapper.covariance_decoder.factor_heads):
            head.weight.zero_(); head.bias.fill_(0. if index == 0 else .4)
        wrapper.covariance_decoder.residual_head.weight.zero_()
        wrapper.covariance_decoder.residual_head.bias.zero_()
    torch.testing.assert_close(wrapper.exact_predictive_mean(batch), before, rtol=0, atol=0)
    batch["target_group"][:2] = 0
    batch["target_group"][2] = 10
    distribution = wrapper(batch)
    with torch.no_grad():
        draw = distribution.sample_joint(30000, torch.Generator().manual_seed(17))
        residual = draw[:, :, 0, 0] - distribution.mean[:, 0, 0]
    empirical = torch.cov(residual.T)
    assert abs(float(empirical[0, 1]) - .48) < .04
    assert abs(float(empirical[0, 2])) < .04


def test_auxiliary_losses_require_explicitly_disabled_configuration():
    base, _, _ = example()
    base.reference_loss_weight = .1
    with pytest.raises(ValueError, match="reference_loss_weight"):
        FaithfulMeasurementModel(base)
