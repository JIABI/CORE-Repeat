import pytest
import torch

from opal2.jepa import ConditionalJEPA
from opal2.model import GroupedProfileEncoder, MeasurementWorldModel


def setup():
    torch.manual_seed(12)
    b, c, t, d = 3, 2, 2, 6
    batch = {
        "context_y": torch.randn(b, c, d), "context_cond": torch.randn(b, c, 3),
        "context_mask": torch.ones(b, c, dtype=torch.bool),
        "context_reference": torch.randn(b, c, 3, 4),
        "context_reference_mask": torch.ones(b, c, 3, dtype=torch.bool),
        "context_group": torch.zeros(b, c, 3, dtype=torch.int64),
        "target_cond": torch.randn(b, t, 3), "target_reference": torch.randn(b, t, 3, 4),
        "target_reference_mask": torch.ones(b, t, 3, dtype=torch.bool),
        "target_group": torch.zeros(b, t, 3, dtype=torch.int64), "chem": torch.randn(b, 5),
    }
    encoder = GroupedProfileEncoder({"Nuclei": [0, 1, 2], "Cells": [3, 4, 5]}, 16)
    return ConditionalJEPA(encoder, 3, 4, 5, ema_decay=0.9), batch, torch.randn(b, t, d)


def test_jepa_ema_stop_gradient_and_anticollapse():
    model, batch, target = setup()
    before = [p.detach().clone() for p in model.teacher_encoder.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = model.loss(batch, target)
    assert metrics["variance"] > 0
    assert metrics["covariance"] >= 0
    metrics["loss"].backward()
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
    assert all(p.grad is None for p in model.teacher_projector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student_encoder.parameters())
    optimizer.step()
    student = [p.detach().clone() for p in model.student_encoder.parameters()]
    model.update_teacher()
    for previous, current, source in zip(before, model.teacher_encoder.parameters(), student):
        assert torch.allclose(current, 0.9 * previous + 0.1 * source, atol=1e-7)
    assert model.ema_updates.item() == 1
    model.train()
    assert not model.teacher_encoder.training


def test_two_stage_encoder_export_and_fixed_likelihood_target():
    jepa, batch, target = setup()
    encoder = jepa.frozen_encoder()
    assert not any(p.requires_grad for p in encoder.parameters())
    assert encoder is not jepa.teacher_encoder
    world = MeasurementWorldModel(encoder.feature_groups, 3, 4, 5, hidden_dim=16,
                                   latent_rank=2, residual_rank=1)
    world.profile_encoder = encoder
    distribution = world(batch)
    assert distribution.mean.shape == target.shape
    world.loss(batch, target)["loss"].backward()
    assert all(p.grad is None for p in world.profile_encoder.parameters())
    assert world.mean_head.weight.grad.abs().sum() > 0


def test_jepa_context_permutation_and_missing_references():
    model, batch, _ = setup()
    model.eval()
    batch["target_reference_mask"][:] = False
    first = model(batch)
    batch["target_reference"][:] = float("nan")
    reverse = {k: (v.flip(1) if k.startswith("context_") else v) for k, v in batch.items()}
    second = model(reverse)
    assert torch.allclose(first, second, atol=1e-6)


def test_coordinate_mask_removing_only_finite_target_rejects_empty_teacher_set():
    model, batch, target = setup()
    target[:] = float("nan")
    target[0, 0, 0] = 1.0
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0, 0] = False
    with pytest.raises(ValueError, match="observed target"):
        model.loss(batch, target, mask)


def test_coordinate_mask_invalidated_well_matches_explicit_well_exclusion():
    model, batch, target = setup()
    target[0, 0] = float("nan")
    target[0, 0, 0] = 2.0
    coordinate_mask = torch.ones_like(target, dtype=torch.bool)
    coordinate_mask[0, 0, 0] = False
    well_mask = torch.ones(target.shape[:2], dtype=torch.bool)
    well_mask[0, 0] = False
    by_coordinates = model.loss(batch, target, coordinate_mask)
    by_well = model.loss(batch, target, well_mask)
    for key in ("loss", "alignment", "variance", "covariance", "embedding_std"):
        assert torch.allclose(by_coordinates[key], by_well[key], atol=1e-7)
