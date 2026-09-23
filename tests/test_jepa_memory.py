"""Numerical equivalence tests, not biological or B128 efficacy experiments."""
import copy

import pytest
import torch

from opal2.jepa import ConditionalJEPA
from opal2.jepa_memory import MemoryEfficientGroupedProfileEncoder
from opal2.model import GroupedProfileEncoder


def groups_for(dimension, count):
    return {f"numerical_group_{i}": list(range(i, dimension, count)) for i in range(count)}


def matching_encoders(*, dimension=24, group_count=8, hidden=32, chunk=3, dtype=torch.float64):
    torch.set_num_threads(1)
    torch.manual_seed(740)
    groups = groups_for(dimension, group_count)
    original = GroupedProfileEncoder(groups, hidden, attention_layers=2, attention_heads=4).to(dtype)
    chunked = MemoryEfficientGroupedProfileEncoder(groups, hidden, attention_layers=2,
                                                   attention_heads=4, profile_chunk_size=chunk).to(dtype)
    chunked.load_state_dict(original.state_dict(), strict=True)
    return original, chunked


def assert_parameter_gradients_match(original, chunked, *, atol=1e-10, rtol=1e-8):
    assert list(dict(original.named_parameters())) == list(dict(chunked.named_parameters()))
    for (name, a), (other, b) in zip(original.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=atol, rtol=rtol, msg=name)


def test_full_3617_coordinate_138_group_256_hidden_encoder_outputs_and_gradients_match():
    original, chunked = matching_encoders(dimension=3617, group_count=138, hidden=256, chunk=2)
    assert list(original.state_dict()) == list(chunked.state_dict())
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in chunked.parameters())
    x = torch.randn(2, 3, 3617, dtype=torch.float64)
    x[0, 0, :5] = float("nan")
    x_original, x_chunked = x.clone().requires_grad_(), x.clone().requires_grad_()
    weight = torch.randn(2, 3, 256, dtype=torch.float64)
    expected = original(x_original)
    (expected * weight).sum().backward()
    actual = chunked(x_chunked)
    (actual * weight).sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(x_original.grad, x_chunked.grad, atol=1e-10, rtol=1e-8)
    assert_parameter_gradients_match(original, chunked, atol=2e-10)


@pytest.mark.parametrize("shape", [(24,), (0, 24), (4, 0, 24), (2, 3, 4, 24)])
def test_shape_and_empty_context_compatibility(shape):
    original, chunked = matching_encoders()
    original.eval(); chunked.eval()
    y = torch.randn(shape, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(chunked(y), original(y), atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_chunk_size_rejected(bad):
    with pytest.raises(ValueError, match="positive integer"):
        MemoryEfficientGroupedProfileEncoder({"all": [0, 1]}, 16, profile_chunk_size=bad)


def test_nonreentrant_training_checkpoint_and_no_grad_teacher_chunking(monkeypatch):
    import opal2.jepa_memory as implementation
    _, encoder = matching_encoders(chunk=3)
    real_checkpoint = implementation.checkpoint
    calls = []
    def recorded(function, value, **kwargs):
        calls.append((value.shape[0], kwargs))
        return real_checkpoint(function, value, **kwargs)
    monkeypatch.setattr(implementation, "checkpoint", recorded)
    values = torch.randn(8, 24, dtype=torch.float64)  # Real measurements need no input gradients.
    encoder.train()
    encoder(values).square().mean().backward()
    assert [size for size, _ in calls] == [3, 3, 2]
    assert all(options == {"use_reentrant": False, "preserve_rng_state": True} for _, options in calls)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    calls.clear()
    with torch.no_grad():
        encoder(values)
    assert not calls
    teacher = copy.deepcopy(encoder).eval().requires_grad_(False)
    with torch.no_grad():
        torch.testing.assert_close(teacher(values), encoder.eval()(values), atol=2e-12, rtol=1e-11)
    assert not calls
    assert teacher.profile_chunk_size == 3
    assert list(teacher.state_dict()) == list(encoder.state_dict())


def jepa_batch(b, d, *, c=2, t=2, dtype=torch.float64):
    torch.manual_seed(831)
    batch = {
        "context_y": torch.randn(b, c, d, dtype=dtype),
        "context_cond": torch.randn(b, c, 5, dtype=dtype),
        "context_mask": torch.rand(b, c) > .15,
        "context_reference": torch.randn(b, c, 3, 4, dtype=dtype),
        "context_reference_mask": torch.ones(b, c, 3, dtype=torch.bool),
        "target_cond": torch.randn(b, t, 5, dtype=dtype),
        "target_reference": torch.randn(b, t, 3, 4, dtype=dtype),
        "target_reference_mask": torch.ones(b, t, 3, dtype=torch.bool),
        "chem": torch.randn(b, 7, dtype=dtype),
        "chem_mask": torch.arange(b) % 5 != 0,
    }
    # Identical physical library rows appear for many compounds; all neighbors
    # remain in attention, using the actual exact-deduplication interface.
    library_y = torch.randn(3, d, dtype=dtype)
    library_cond = torch.randn(3, 5, dtype=dtype)
    batch.update(library_y=library_y.expand(b, -1, -1).clone(),
                 library_cond=library_cond.expand(b, -1, -1).clone(),
                 library_index=torch.arange(3).expand(b, -1).clone(),
                 library_mask=torch.ones(b, 3, dtype=torch.bool),
                 library_global_mean=torch.randn(b, d, dtype=dtype),
                 library_global_variance=torch.rand(b, d, dtype=dtype),
                 library_count=torch.full((b,), 3, dtype=torch.int64),
                 library_density=torch.rand(b, 2, dtype=dtype))
    # Real panel path is exercised with numerical identity-matched controls.
    for prefix, width in (("context", c), ("target", t)):
        shape = (b, width, 3, 2)
        batch[prefix+"_panel_y"] = torch.randn(*shape, d, dtype=dtype)
        batch[prefix+"_panel_template"] = torch.randn(*shape, d, dtype=dtype)
        batch[prefix+"_panel_mask"] = torch.ones(shape, dtype=torch.bool)
        batch[prefix+"_panel_template_mask"] = torch.ones(shape, dtype=torch.bool)
        batch[prefix+"_panel_identity"] = torch.arange(2).expand(shape).clone()
    target = torch.randn(b, t, d, dtype=dtype)
    target[0, 0, 0] = float("nan")
    target_mask = torch.ones(b, t, dtype=torch.bool)
    target_mask[-1, -1] = False
    return batch, target, target_mask


@pytest.mark.parametrize("b,d,g,h,chunk", [(128, 24, 8, 32, 16), (3, 3617, 138, 256, 2)])
def test_complete_jepa_loss_global_covariance_gradients_and_ema_export_match(b, d, g, h, chunk):
    original_encoder, chunked_encoder = matching_encoders(dimension=d, group_count=g, hidden=h, chunk=chunk)
    original = ConditionalJEPA(original_encoder, 5, 4, 7, ema_decay=.91).double()
    chunked = ConditionalJEPA(chunked_encoder, 5, 4, 7, ema_decay=.91).double()
    chunked.load_state_dict(original.state_dict(), strict=True)
    assert list(original.state_dict()) == list(chunked.state_dict())
    original.train(); chunked.train()
    batch, target, mask = jepa_batch(b, d)
    expected = original.loss(batch, target, mask)
    expected_values = {k: v.detach().clone() for k, v in expected.items()}
    expected["loss"].backward()
    del expected
    actual = chunked.loss(batch, target, mask)
    for key, value in actual.items():
        torch.testing.assert_close(value, expected_values[key], atol=2e-10, rtol=1e-8, msg=key)
    actual["loss"].backward()
    assert_parameter_gradients_match(original, chunked, atol=5e-10, rtol=1e-7)
    assert all(p.grad is None for p in chunked.teacher_encoder.parameters())
    # Verify the one full-batch covariance explicitly, rather than averaging
    # covariances of independent encoder-sized chunks or compound microbatches.
    with torch.no_grad():
        context = chunked.student_projector(chunked.student_encoder(batch["context_y"]))
        targets = chunked.student_projector(chunked.student_encoder(target))
        valid = torch.isfinite(target).any(-1) & mask
        all_views = torch.cat((context[batch["context_mask"]], targets[valid]), 0)
        centered = all_views - all_views.mean(0, keepdim=True)
        covariance = centered.T @ centered / (len(all_views)-1)
        off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
        torch.testing.assert_close(actual["covariance"], off_diagonal.square().sum()/h, atol=2e-10, rtol=1e-8)
        if b == 128:
            assert len(all_views) > 128
        # Identical deterministic optimizer update, then the real EMA update.
        for model in (original, chunked):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-1e-4)
            model.update_teacher()
        exported_original, exported_chunked = original.frozen_encoder(), chunked.frozen_encoder()
        torch.testing.assert_close(exported_original(target), exported_chunked(target), atol=3e-10, rtol=1e-8)
    assert isinstance(exported_chunked, MemoryEfficientGroupedProfileEncoder)
    assert exported_chunked.profile_chunk_size == chunk
    assert not exported_chunked.training
    assert not any(p.requires_grad for p in exported_chunked.parameters())
    assert exported_chunked is not chunked.teacher_encoder


def test_checkpointing_reduces_saved_encoder_activations_without_changing_parameters():
    original, chunked = matching_encoders(chunk=4)
    x = torch.randn(32, 24, dtype=torch.float64)
    counts = []
    for encoder in (original, chunked):
        saved = []
        def pack(tensor):
            saved.append(tensor.numel())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = encoder(x)
        output.square().sum().backward()
        counts.append(sum(saved))
    assert counts[1] < counts[0] * .1
    assert_parameter_gradients_match(original, chunked, atol=2e-9, rtol=1e-7)
