import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import opal2.jepa_diagnostic_metrics as diagnostics


class FiniteLinear(nn.Linear):
    def forward(self, values):
        return super().forward(torch.nan_to_num(values))


class EvaluationLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 3
        self.teacher_encoder = FiniteLinear(3, 3, bias=False)
        self.student_encoder = FiniteLinear(3, 3, bias=False)
        self.teacher_projector = nn.Identity()
        self.student_projector = nn.Identity()
        self.register_buffer("ema_updates", torch.tensor(7))
        with torch.no_grad():
            self.teacher_encoder.weight.copy_(torch.eye(3))
            self.student_encoder.weight.copy_(2 * torch.eye(3))
        self.teacher_encoder.requires_grad_(False)

    def forward(self, batch):
        assert not torch.is_grad_enabled()
        assert not self.training
        return batch["predicted"]


def install_fixed_batch(monkeypatch, data):
    def fixed(dataset, indices, config, *, contexts, targets):
        assert dataset is data
        assert contexts == (0,) and targets == (1, 2, 3)
        ix = torch.as_tensor(indices)
        return {"predicted": data["predicted"][ix]}, data["target"][ix], data["mask"][ix]
    monkeypatch.setattr(diagnostics, "fixed_batch", fixed)


def test_geometry_low_rank_and_zero_variance():
    rank_one = torch.tensor([[-2., 0., -4.], [-1., 0., -2.], [1., 0., 2.], [2., 0., 4.]])
    stats = diagnostics._representation_statistics(rank_one)
    assert stats["covariance_participation_rank"] == pytest.approx(1)
    assert stats["covariance_entropy_rank"] == pytest.approx(1)
    assert stats["near_zero_dimension_fraction"] == pytest.approx(1 / 3)
    unequal = diagnostics._representation_statistics(torch.tensor([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]]))
    assert unequal["covariance_participation_rank"] == pytest.approx(25 / 17)
    assert unequal["covariance_entropy_rank"] == pytest.approx(np.exp(-.2 * np.log(.2) - .8 * np.log(.8)))
    two_samples = diagnostics._representation_statistics(rank_one[:2])
    assert two_samples["rank_upper_bound"] == 1
    assert two_samples["rank_limited_by_sample_count"]
    zeros = diagnostics._representation_statistics(torch.zeros(4, 3))
    assert zeros["std_mean"] == 0 and zeros["zero_covariance"]
    assert zeros["covariance_participation_rank"] == 0
    assert zeros["covariance_entropy_rank"] == 0
    paired = diagnostics._prediction_metrics(torch.ones(4, 3), torch.zeros(4, 3))
    assert paired["mse"] == 1
    assert paired["nmse"] is None and paired["zero_reference_variance"]
    assert paired["cosine_mean"] is None and paired["cosine_zero_norm_samples"] == 4
    json.dumps(paired, allow_nan=False)


def test_evaluation_pools_chunks_masks_targets_and_restores_modes(monkeypatch):
    target = torch.arange(45, dtype=torch.float32).reshape(5, 3, 3) / 10
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0] = False
    target[1, 1] = float("nan")
    target[2, 2] = torch.tensor([2., float("nan"), float("nan")])
    mask[2, 2, 0] = False  # Removing the sole finite coordinate invalidates this well.
    target[3, 1, 0] = float("nan")  # A partially available well is retained.
    data = {"target": target, "mask": mask, "predicted": torch.nan_to_num(target) + .25}
    install_fixed_batch(monkeypatch, data)
    learner = EvaluationLearner().train()
    learner.teacher_encoder.eval()
    anchor = (copy.deepcopy(learner.teacher_encoder), copy.deepcopy(learner.teacher_projector))
    anchor[0].train()
    roots = (learner,) + anchor
    modes = [(m, m.training) for root in roots for m in root.modules()]
    before = {k: v.clone() for k, v in learner.state_dict().items()}
    for parameter in learner.parameters():
        parameter.grad = torch.full_like(parameter, 3.)
    grads = [p.grad.clone() for p in learner.parameters()]
    result = diagnostics.evaluate_jepa(learner, data, np.arange(5), SimpleNamespace(), chunk_size=2, anchor=anchor)
    assert result["n_valid_target_wells"] == 12
    valid = torch.isfinite(torch.where(mask, target, float("nan"))).any(-1)
    reference = torch.nan_to_num(target[valid]).double()
    expected_var = (reference - reference.mean(0)).square().mean().item()
    assert result["alignment_mse"] == pytest.approx(.25 ** 2)
    assert result["teacher_target_variance"] == pytest.approx(expected_var)
    assert result["alignment_nmse"] == pytest.approx(.25 ** 2 / expected_var)
    assert result["anchor_target_drift"]["projected"]["mse"] == 0
    assert result["representations"]["teacher_encoder"]["n_samples"] == 12
    assert all(module.training == flag for module, flag in modes)
    assert all(torch.equal(before[k], v) for k, v in learner.state_dict().items())
    assert all(torch.equal(old, p.grad) for old, p in zip(grads, learner.parameters()))
    whole = diagnostics.evaluate_jepa(learner, data, np.arange(5), SimpleNamespace(), chunk_size=20, anchor=anchor)
    assert result == whole
    json.dumps(result, allow_nan=False)


def test_modes_restore_when_evaluation_raises(monkeypatch):
    learner = EvaluationLearner().train()
    learner.teacher_encoder.eval()
    modes = [(m, m.training) for m in learner.modules()]
    def broken(*args, **kwargs):
        raise RuntimeError("fixed batch failed")
    monkeypatch.setattr(diagnostics, "fixed_batch", broken)
    with pytest.raises(RuntimeError, match="fixed batch failed"):
        diagnostics.evaluate_jepa(learner, None, [0], SimpleNamespace())
    assert all(module.training == flag for module, flag in modes)


class GradientLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_encoder = nn.Linear(2, 1)
        self.alignment_weight = 2.
        self.variance_weight = 3.
        self.covariance_weight = 1.

    def loss(self, inputs, target_y, target_mask):
        # Bias is deliberately unused; covariance has a connected zero gradient.
        linear = self.student_encoder.weight.sum()
        return {"alignment": linear, "variance": -linear, "covariance": linear * 0}


def test_gradient_diagnostics_weights_conflict_unused_and_preserve_grad():
    learner = GradientLearner()
    for p in learner.parameters():
        p.grad = torch.full_like(p, 9.)
    before = [p.grad.clone() for p in learner.parameters()]
    result = diagnostics.component_gradient_diagnostics(learner, {"inputs": {}, "target_y": None})
    assert result["components"]["alignment"]["grad_norm"] == pytest.approx(2 * 2 ** .5)
    assert result["components"]["variance"]["grad_norm"] == pytest.approx(3 * 2 ** .5)
    assert result["components"]["alignment"]["unused_parameter_tensors"] == 1
    assert result["pairwise_cosines"]["alignment__variance"] == pytest.approx(-1)
    assert result["components"]["covariance"]["zero_norm"]
    assert result["pairwise_cosines"]["alignment__covariance"] is None
    assert all(torch.equal(old, p.grad) for old, p in zip(before, learner.parameters()))
    json.dumps(result, allow_nan=False)


def test_real_conditional_jepa_evaluation_and_component_gradients(monkeypatch):
    from opal2.jepa import ConditionalJEPA
    from opal2.model import GroupedProfileEncoder

    torch.manual_seed(31)
    torch.set_num_threads(1)
    count, dimension = 4, 6
    batch = {
        "context_y": torch.randn(count, 1, dimension),
        "context_cond": torch.randn(count, 1, 3),
        "context_mask": torch.ones(count, 1, dtype=torch.bool),
        "context_reference": torch.randn(count, 1, 3, 4),
        "context_reference_mask": torch.ones(count, 1, 3, dtype=torch.bool),
        "target_cond": torch.randn(count, 3, 3),
        "target_reference": torch.randn(count, 3, 3, 4),
        "target_reference_mask": torch.ones(count, 3, 3, dtype=torch.bool),
        "chem": torch.randn(count, 5),
    }
    target = torch.randn(count, 3, dimension)
    mask = torch.ones(count, 3, dtype=torch.bool)
    mask[0, 2] = False
    def fixed(dataset, indices, config, *, contexts, targets):
        return {k: v[indices] for k, v in batch.items()}, target[indices], mask[indices]
    monkeypatch.setattr(diagnostics, "fixed_batch", fixed)
    learner = ConditionalJEPA(GroupedProfileEncoder({"A": [0, 1, 2], "B": [3, 4, 5]}, 16), 3, 4, 5)
    learner.train()
    before = {k: v.clone() for k, v in learner.state_dict().items()}
    evaluated = diagnostics.evaluate_jepa(learner, None, np.arange(count), SimpleNamespace(), chunk_size=2)
    assert evaluated["n_valid_target_wells"] == 11
    assert evaluated["alignment_nmse"] > 0
    assert evaluated["teacher_student_same_well_drift"]["projected"]["mse"] == pytest.approx(0, abs=1e-12)
    result = diagnostics.component_gradient_diagnostics(learner, {"inputs": batch, "target_y": target, "target_mask": mask})
    assert result["components"]["alignment"]["grad_norm"] > 0
    assert result["components"]["variance"]["grad_norm"] > 0
    assert learner.training and not learner.teacher_encoder.training
    assert all(p.grad is None for p in learner.parameters())
    assert all(torch.equal(before[k], v) for k, v in learner.state_dict().items())
    json.dumps(evaluated, allow_nan=False)
    json.dumps(result, allow_nan=False)
