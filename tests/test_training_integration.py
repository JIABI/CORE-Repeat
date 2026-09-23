"""Integration checks; optional real DEV check reads only the existing adapter.

Run real data explicitly with OPAL2_LEGACY_ROOT pointing at the previously opened
JUMP study. This is a full-coordinate software check, not a fitted experiment.
"""
from dataclasses import asdict
import os
import time

import numpy as np
import pytest
import torch

from opal2.config import TrainConfig
from opal2.data import MeasurementDataset, TrainScaler, cellprofiler_groups, make_training_batch
from opal2.model import MeasurementWorldModel
from opal2.source5 import load_source5
from opal2.splits import development_split, assert_disjoint_splits, save_split, load_split
from opal2.training import model_kwargs, fixed_batch, random_training_batch, load_model, seed_everything


def schema_fixture(d=3617, n=12, w=4):
    """Numerical shape fixture only; values are not biological observations."""
    rng = np.random.default_rng(861)
    names = np.array([f"Cells_Intensity_Feature{i}" for i in range(d)])
    gi, gn = cellprofiler_groups(names)
    return MeasurementDataset(
        Y=rng.normal(size=(n, w, d)), ids=np.array([f"fixture_{i}" for i in range(n)]),
        feature_names=names, feature_group_index=gi, feature_group_names=gn,
        cond=rng.normal(size=(n, w, 7)), reference=rng.normal(size=(n, w, 3, 5)),
        reference_mask=np.ones((n, w, 3), bool),
        groups=np.tile(np.stack([np.zeros(w), np.arange(w), np.arange(w)], -1)[None], (n, 1, 1)).astype(int),
        chem=rng.integers(0, 2, size=(n, 17)).astype(float), metadata={"fixture": True})


def test_full_coordinate_fixed_and_random_training_interfaces():
    ds = schema_fixture()
    cfg = TrainConfig(use_jepa=False, threads=1, zero_context_fraction=0)
    scaler = TrainScaler.fit(ds, np.arange(8))
    transformed = scaler.transform(ds)
    inputs, y, mask = fixed_batch(transformed, [0, 2], cfg)
    direct = make_training_batch(transformed, [0, 2])
    assert y.shape == (2, 3, 3617)
    assert torch.equal(y, direct["target_y"])
    assert torch.equal(mask, direct["target_mask"])
    for key in inputs:
        assert torch.equal(inputs[key], direct["inputs"][key])
    seen_context_sizes = set()
    rng = np.random.default_rng(112)
    for _ in range(5):
        task = random_training_batch(transformed, [0, 2, 3], rng, cfg)
        assert set(task["compound_index"].tolist()) == {0, 2, 3}
        assert task["inputs"]["context_y"].shape[-1] == 3617
        assert task["target_y"].shape[-1] == 3617
        assert "target_y" not in task["inputs"]
        seen_context_sizes.update(task["inputs"]["context_mask"].sum(1).tolist())
    assert seen_context_sizes == {1, 2, 3}


def test_compound_split_serialization_and_overlap_rejection(tmp_path):
    ds = schema_fixture(d=4)
    split = development_split(ds.ids)
    save_split(tmp_path / "split.json", ds.ids, split, evidence_scope="unit_test")
    restored, scope = load_split(tmp_path / "split.json", ds.ids)
    assert scope == "unit_test"
    for name in split:
        np.testing.assert_array_equal(split[name], restored[name])
    broken = dict(split)
    broken["evaluation"] = split["train"][:1]
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint_splits(ds.ids, broken)


@pytest.mark.parametrize("use_jepa", [False, True])
def test_checkpoint_reload_matches_forward_and_frozen_encoder(tmp_path, use_jepa):
    ds = schema_fixture(d=20)
    cfg = TrainConfig(use_jepa=use_jepa, hidden_dim=32, threads=1, use_library=False)
    seed_everything(13, 1)
    scaler = TrainScaler.fit(ds, np.arange(8))
    batch, _, _ = fixed_batch(scaler.transform(ds), [0, 1], cfg)
    kwargs = model_kwargs(ds, cfg)
    model = MeasurementWorldModel(**kwargs).eval()
    model.set_outcome_transform(scaler.y_center,scaler.y_scale)
    if use_jepa:
        model.freeze_encoder()
    original = model(batch)
    torch.save({"state_dict": model.state_dict(), "model_config": kwargs,
                "train_config": asdict(cfg), "train_ids":ds.ids[:8].tolist(),
                "validation_ids":ds.ids[8:10].tolist(),"feature_names":ds.feature_names.tolist()}, tmp_path / "best.pt")
    scaler.save(tmp_path / "scaler.json")
    restored, restored_scaler, config, _ = load_model(tmp_path)
    result = restored(batch)
    torch.testing.assert_close(result.mean, original.mean, rtol=0, atol=0)
    torch.testing.assert_close(result.diag_var, original.diag_var, rtol=0, atol=0)
    torch.testing.assert_close(result.factors, original.factors, rtol=0, atol=0)
    assert restored_scaler.to_dict() == scaler.to_dict()
    assert asdict(config) == asdict(cfg)
    assert any(p.requires_grad for p in restored.profile_encoder.parameters()) == (not use_jepa)


@pytest.mark.skipif(not os.environ.get("OPAL2_LEGACY_ROOT"), reason="Explicit existing-DEV path required")
def test_real_639_dev_full_shape_forward_marginal_backward_and_no_leakage(tmp_path):
    began = time.perf_counter()
    ds = load_source5("primary", legacy_root=os.environ["OPAL2_LEGACY_ROOT"])
    loader_seconds = time.perf_counter() - began
    assert ds.Y.shape == (639, 4, 3617)
    cfg = TrainConfig(use_jepa=False, threads=2, use_library=False)
    seed_everything(31, 2)
    splits = development_split(ds.ids, seed=31)
    scaler = TrainScaler.fit(ds, splits["train"])
    normalized = scaler.transform(ds)
    ix = splits["train"][:2]
    batch, y, mask = fixed_batch(normalized, ix, cfg)
    assert batch["context_y"].shape == (2, 1, 3617)
    assert y.shape == (2, 3, 3617)
    assert not batch["target_reference_mask"].any()
    model = MeasurementWorldModel(**model_kwargs(ds, cfg))
    model.set_outcome_transform(scaler.y_center,scaler.y_scale)
    began = time.perf_counter()
    distribution = model(batch)
    forward_seconds = time.perf_counter() - began
    assert distribution.mean.shape == y.shape
    # Exercise the existing exact per-compound marginal likelihood, not the
    # cross-compound joint objective that is owned by the model implementation.
    began = time.perf_counter()
    marginal_nll = -distribution.log_prob(y, mask).sum() / mask.sum() / ds.Y.shape[-1]
    assert torch.isfinite(marginal_nll)
    marginal_nll.backward()
    marginal_backward_seconds = time.perf_counter() - began
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    with torch.no_grad():
        prior = model(batch).mean.clone()
        normalized.Y[:, 1:] += 1e6
        after, _, _ = fixed_batch(normalized, ix, cfg)
        torch.testing.assert_close(model(after).mean, prior, rtol=0, atol=0)
    began = time.perf_counter()
    torch.save({"state_dict": model.state_dict(), "model_config": model_kwargs(ds, cfg),
                "train_config": asdict(cfg), "train_ids":ds.ids[splits["train"]].tolist(),
                "validation_ids":ds.ids[splits["validation"]].tolist(),"feature_names":ds.feature_names.tolist()}, tmp_path / "best.pt")
    scaler.save(tmp_path / "scaler.json")
    restored, recovered_scaler, recovered_config, _ = load_model(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(restored(batch).mean, prior, rtol=0, atol=0)
    checkpoint_seconds = time.perf_counter() - began
    assert recovered_scaler.to_dict() == scaler.to_dict()
    assert asdict(recovered_config) == asdict(cfg)
    print({"D": 3617, "B": 2, "context_wells": 1, "target_wells": 3,
           "parameters": sum(p.numel() for p in model.parameters()),
           "loader_seconds": round(loader_seconds, 4), "forward_seconds": round(forward_seconds, 4),
           "marginal_likelihood_backward_seconds": round(marginal_backward_seconds, 4),
           "full_checkpoint_reload_seconds": round(checkpoint_seconds, 4),
           "checkpoint_bytes": (tmp_path / "best.pt").stat().st_size})
    assert ds.metadata["final_profiles_read"] is False
    assert ds.metadata["fifth_repeat_read"] is False
