"""New full-model paths on numerical fixtures, not biological performance runs."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from test_data import records
from test_data_fidelity_r2 import panels
from opal2.config import TrainConfig
from opal2.data import TrainScaler, attach_library_context
from opal2.model import MeasurementWorldModel
from opal2.training import (fit_model, load_model, fixed_batch, model_kwargs,
                           fit_training_chemical_anchors, make_world_model_scheduler)


def chemistry_fixture():
    data = panels(records(n=20, w=4))
    chemical = data.chem.copy()
    chemical[:, -1] = 1.
    return replace(data, chem=chemical, metadata={**data.metadata, "chemical": {
        "kind": "RDKit Morgan fingerprint", "radius": 2, "bits": 16,
        "final_coordinate": "valid_SMILES_indicator"}})


def config(**changes):
    base = dict(seed=513, epochs=2, batch_size=4, hidden_dim=16, latent_rank=2,
                residual_rank=2, attention_heads=2, threads=1, use_jepa=False,
                samples=12, mc_chunk_size=4, objective="predictive_nll",
                biology_kernel_mode="structured", biology_kernel_anchors=6,
                lr_schedule="cosine", warmup_steps=1, patience=2)
    return TrainConfig(**{**base, **changes}).validate()


def split():
    return {"train": np.arange(12), "validation": np.arange(12, 15),
            "calibration": np.arange(15, 17), "evaluation": np.arange(17, 20)}


def test_optional_kernel_does_not_change_shared_initial_parameters_or_rng():
    data, cfg = chemistry_fixture(), config()
    anchors = fit_training_chemical_anchors(data, split()["train"], cfg)
    torch.manual_seed(504)
    old = MeasurementWorldModel(**model_kwargs(data, replace(cfg, biology_kernel_mode="off")))
    old_rng = torch.get_rng_state()
    torch.manual_seed(504)
    new = MeasurementWorldModel(**model_kwargs(data, cfg, chemical_anchor_data=anchors))
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, tensor in old.state_dict().items():
        assert torch.equal(tensor, new.state_dict()[name]), name
    assert new.chemistry_response_kernel is not None


@pytest.mark.parametrize("family", ["gaussian", "copula_t4"])
def test_kernel_full_training_reload_and_legal_predictions(tmp_path, family):
    data, splits, cfg = chemistry_fixture(), split(), config(observation_family=family)
    destination = tmp_path / family
    model, scaler = fit_model(data, splits, cfg, destination)
    loaded, restored, recovered, payload = load_model(destination)
    assert recovered == cfg
    normalized = attach_library_context(restored.transform(data), loaded.library_bank)
    inputs, target, mask = fixed_batch(normalized, splits["evaluation"], cfg)
    with torch.no_grad():
        a, b = model(inputs), loaded(inputs)
        assert torch.equal(a.mean, b.mean)
        assert torch.equal(a.log_prob(target, mask), b.log_prob(target, mask))
        first = a.sample_joint(12, torch.Generator().manual_seed(27))
        second = b.sample_joint(12, torch.Generator().manual_seed(27))
        assert torch.equal(first, second)
    assert payload["model_config"]["chemical_anchor_data"]["train_ids"] == data.ids[splits["train"]].tolist()
    initial = torch.load(destination / "initial_state.pt", weights_only=True)
    changed = [not torch.equal(p.detach(), initial[n]) for n, p in model.named_parameters()
               if n.startswith("chemistry_response_kernel.")]
    assert any(changed), "A saved but disconnected response kernel is not an implementation"
    original = loaded(inputs).mean.detach()
    poisoned = replace(data, Y=data.Y.copy())
    poisoned.Y[splits["evaluation"], 1:] += 10000
    bad = attach_library_context(restored.transform(poisoned), loaded.library_bank)
    legal, _, _ = fixed_batch(bad, splits["evaluation"], cfg)
    assert torch.equal(original, loaded(legal).mean.detach())
    anchors = json.loads((destination / "chemical_anchors.json").read_text())
    anchors["train_ids"][0] = str(data.ids[-1])
    (destination / "chemical_anchors.json").write_text(json.dumps(anchors))
    with pytest.raises(ValueError, match="anchors"):
        load_model(destination)


def test_cosine_schedule_is_step_based_resumable_and_reaches_floor():
    cfg = config(epochs=3, warmup_steps=2)
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=cfg.learning_rate)
    scheduler = make_world_model_scheduler(optimizer, cfg, steps_per_epoch=4)
    rates = []
    for step in range(12):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        if step == 4:
            optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
    assert rates[0] < rates[1]
    assert max(rates) == pytest.approx(cfg.learning_rate)
    assert rates[-1] == pytest.approx(cfg.min_learning_rate)
    other = torch.optim.AdamW([torch.nn.Parameter(torch.ones(()))], lr=cfg.learning_rate)
    resumed = make_world_model_scheduler(other, cfg, steps_per_epoch=4)
    other.load_state_dict(optimizer_state)
    resumed.load_state_dict(scheduler_state)
    assert other.param_groups[0]["lr"] == pytest.approx(rates[5])
    for expected in rates[5:]:
        assert other.param_groups[0]["lr"] == pytest.approx(expected)
        other.step()
        resumed.step()


def test_copula_refuses_unimplemented_elbo_and_inconsistent_config():
    with pytest.raises(ValueError, match="predictive_nll"):
        TrainConfig(observation_family="copula_t4").validate()
    with pytest.raises(ValueError, match="requires chemical"):
        TrainConfig(biology_kernel_mode="structured", use_chemistry=False).validate()


def test_old_shared_jepa_binding_receives_only_legacy_default_fields(tmp_path):
    """Compatible old encoder artifacts remain reusable; nondefault changes do not."""
    import copy
    from opal2.training import _jepa_binding, _load_shared_jepa
    data = chemistry_fixture()
    cfg = config(biology_kernel_mode="off", lr_schedule="plateau", warmup_steps=60,
                 use_jepa=True, encoder_policy="jepa_frozen")
    train = split()["train"]
    scaler = TrainScaler.fit(data, train)
    normalized = scaler.transform(data)
    kwargs = model_kwargs(normalized, cfg)
    model = MeasurementWorldModel(**kwargs)
    current = _jepa_binding(normalized, train, scaler, cfg)
    older = copy.deepcopy(current)
    for key in ("biology_kernel_mode", "biology_kernel_anchors", "observation_family",
                "lr_schedule", "warmup_steps", "min_learning_rate", "diagnostic_interval"):
        older["invariant_train_config"].pop(key)
    # Historical defaults used 64 anchors, although an off branch never fits any.
    current["invariant_train_config"]["biology_kernel_anchors"] = 64
    source, destination = tmp_path / "source", tmp_path / "restored"
    source.mkdir(); destination.mkdir()
    torch.save({"binding": older, "model_config": kwargs,
                "encoder_state_dict": model.profile_encoder.state_dict(),
                "post_jepa_torch_rng": torch.get_rng_state(), "epochs": 1}, source / "jepa.pt")
    _load_shared_jepa(model, source, destination, current, kwargs)
    assert (destination / "jepa.pt").exists()
    changed = copy.deepcopy(current)
    changed["invariant_train_config"]["lr_schedule"] = "cosine"
    with pytest.raises(ValueError, match="differs"):
        _load_shared_jepa(model, source, destination, changed, kwargs)
