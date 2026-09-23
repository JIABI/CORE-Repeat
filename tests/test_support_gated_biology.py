"""Focused opt-in gate invariants; no synthetic efficacy claims."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from opal2.dual_branch_features import NAMES
from opal2.eu_r3_adapters import EUResidualAdapter, fit_left
from opal2.support_gated_biology import (
    MaskedBiologicalBasisBranch, SupportGatedBiologyAdapter, fit_right, from_legacy,
)


def biological_rows(n=8):
    rng = np.random.default_rng(6721)
    b = rng.uniform(.1, 1., (n, len(NAMES)))
    b[:, NAMES.index("target_available")] = 1.
    b[:, NAMES.index("moa_available")] = 0.
    b[:, NAMES.index("target_confidence")] = np.linspace(0., .9, n)
    b[:, NAMES.index("moa_confidence")] = 0.
    return b


def nonzero_model(mode="structured", gate_mode="MASK_ONLY"):
    torch.manual_seed(735)
    model = SupportGatedBiologyAdapter(5, NAMES, mode=mode, gate_mode=gate_mode)
    with torch.no_grad():
        model.left.network[-1].weight.fill_(.03)
        model.right.readout.weight.fill_(.2)
        model.right.readout.bias.copy_(torch.tensor([.03, -.02]))
    return model


@pytest.mark.parametrize("mode", ["generic", "structured"])
def test_missing_relation_fields_and_parameters_cannot_change_output(mode):
    model = nonzero_model(mode)
    b = biological_rows()
    x = np.ones((len(b), 5))
    expected = model.predict_components(x, b)["right"]
    missing = [j for j, name in enumerate(NAMES) if name.startswith("moa_")]
    altered = b.copy()
    altered[:, missing] = 15.
    altered[:, NAMES.index("moa_available")] = 0.
    with torch.no_grad():
        model.right.local_coefficients[missing] = 100.
        model.right.readout.weight[:, missing] = -100.
    assert np.array_equal(expected, model.predict_components(x, altered)["right"])


def test_all_missing_and_switches_preserve_exact_left_and_core():
    model = nonzero_model()
    x, b = np.ones((8, 5)), biological_rows()
    left = model.predict_components(x, right_enabled=False)["total"]
    b[:, NAMES.index("target_available")] = 0.
    parts = model.predict_components(x, b)
    assert np.array_equal(parts["right"], np.zeros((8, 2)))
    assert np.array_equal(parts["total"], left)
    assert np.array_equal(model.predict_increment(x, enabled=False), np.zeros((8, 2)))
    assert np.array_equal(model.predict_increment(x, b, support=np.zeros(8, bool)), left)


def test_confidence_is_exact_multiplicative_output_gate_without_extra_parameters():
    mask = nonzero_model()
    confidence = from_legacy(mask, gate_mode="MASK_CONFIDENCE")
    x, b = np.ones((8, 5)), biological_rows()
    first, second = mask.predict_components(x, b), confidence.predict_components(x, b)
    gate = b[:, NAMES.index("target_confidence")]
    assert np.array_equal(second["right"], first["right"] * gate[:, None])
    assert np.array_equal(second["left"], first["left"])
    assert np.array_equal(second["right"][0], np.zeros(2))
    assert sum(p.numel() for p in confidence.right.parameters()) == 146
    b[:, NAMES.index("moa_available")] = 1
    b[:, NAMES.index("moa_confidence")] = .8
    actual = confidence.right.output_gate(torch.tensor(b)).detach().numpy()
    assert np.array_equal(actual, np.maximum(gate, .8))


def test_seeded_initialization_and_new_artifact_round_trip(tmp_path):
    torch.manual_seed(117)
    original = EUResidualAdapter(5, NAMES, mode="structured")
    torch.manual_seed(117)
    new = SupportGatedBiologyAdapter(5, NAMES, mode="structured")
    for key, value in original.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    model = nonzero_model(gate_mode="MASK_CONFIDENCE")
    path = tmp_path / "new.pt"
    model.save(path)
    restored = SupportGatedBiologyAdapter.load(path)
    x, b = np.ones((8, 5)), biological_rows()
    assert np.array_equal(model.predict_increment(x, b), restored.predict_increment(x, b))
    original.save(tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="explicit support-gated"):
        SupportGatedBiologyAdapter.load(tmp_path / "legacy.pt")


def test_full_training_recipe_freezes_left_and_missing_channel_gradients():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(342)
        x, b = rng.normal(size=(12, 5)), biological_rows(12)
        e = np.tile([5., 8.], (12, 1))
        ids = [f"ref_{i}" for i in range(12)]
        left = fit_left(x, e, ids, seed=15)
        before = deepcopy(left.state_dict())
        for gate_mode in ("MASK_ONLY", "MASK_CONFIDENCE"):
            new = fit_right(left, x, b, NAMES, np.ones(12, bool), e, ids,
                            mode="structured", seed=16, gate_mode=gate_mode)
            assert new.report["epochs"] == 60
            assert new.report["right_trainable_parameters"] == 146
            assert new.report["left_unchanged"]
            for key, value in before.items():
                if key.startswith("left.") or key.startswith("empirical_"):
                    assert torch.equal(value, new.state_dict()[key])
            assert np.array_equal(new.predict_increment(x, right_enabled=False), left.predict_increment(x))
            assert [r["epoch"] for r in new.report["history"]] == [1, 10, 20, 30, 40, 50, 60]
        branch = MaskedBiologicalBasisBranch(NAMES, "structured").double()
        with torch.no_grad():
            branch.readout.weight.fill_(.2)
        raw = torch.tensor(b)
        branch(raw, raw).sum().backward()
        moa = [j for j, name in enumerate(NAMES) if name.startswith("moa_")]
        assert torch.count_nonzero(branch.local_coefficients.grad[moa]) == 0
        assert torch.count_nonzero(branch.readout.weight.grad[:, moa]) == 0
    finally:
        torch.set_num_threads(threads)
