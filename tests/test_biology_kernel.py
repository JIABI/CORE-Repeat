"""Synthetic engineering checks for the full optional chemical kernel prior."""
import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from opal2.biology_kernel import (ChemistryResponseKernelPrior, fit_chemical_anchors,
                                  validate_anchor_data)
from opal2.kernels import SplineKANLinear


def fixture():
    rng = np.random.default_rng(91)
    bits = rng.integers(0, 2, (12, 8)).astype(float)
    bits[:, 0] = 1
    chem = np.column_stack((bits, np.ones(12)))
    ids = np.array([f"fixture_{i:02d}" for i in range(12)])
    mask = np.ones(12, dtype=bool)
    train = ids[:8].tolist()
    metadata = {"fingerprint_indices": list(range(8)), "validity_index": 8,
                "kind": "synthetic engineering binary fingerprint", "radius": 2}
    anchors = fit_chemical_anchors(chem, mask, ids, train, metadata, max_anchors=5)
    return chem, mask, ids, train, metadata, anchors


def prior_inputs(chem, mask, hidden=12, rank=3):
    torch.manual_seed(82)
    return (torch.tensor(chem, dtype=torch.float32), torch.tensor(mask),
            torch.randn(len(chem), hidden), torch.randn(len(chem), rank),
            torch.exp(torch.randn(len(chem), rank)))


def test_landmark_fit_only_reads_training_structures_and_preserves_identity_order():
    chem, mask, ids, train, metadata, anchors = fixture()
    changed = chem.copy()
    changed[8:] = np.nan  # Held-out structures must not even be validated.
    result = fit_chemical_anchors(changed, mask, ids, train, metadata, max_anchors=5)
    assert anchors == result
    assert result["train_ids"] == train
    assert set(result["anchor_ids"]).issubset(train)
    assert len(result["anchor_ids"]) == 5
    assert all(len(row) == 8 for row in result["training_fingerprints"])
    assert result["descriptor_names"][-1] == "fingerprint_bit_density"
    validate_anchor_data(json.loads(json.dumps(result, allow_nan=False)))


def test_landmark_choices_are_deterministic_under_dataset_and_training_order_permutations():
    chem, mask, ids, train, metadata, anchors = fixture()
    permutation = np.random.default_rng(3).permutation(len(ids))
    result = fit_chemical_anchors(chem[permutation], mask[permutation], ids[permutation],
                                  train, metadata, max_anchors=5)
    assert result == anchors
    reversed_result = fit_chemical_anchors(chem, mask, ids, train[::-1], metadata, max_anchors=5)
    assert reversed_result["train_ids"] == train[::-1]
    assert reversed_result["anchor_ids"] == anchors["anchor_ids"]
    assert reversed_result["anchor_fingerprints"] == anchors["anchor_fingerprints"]


def test_duplicates_have_one_landmark_and_distance_ties_use_identity():
    chem = np.column_stack((np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.ones(4)))
    names = ["z_alias", "a_alias", "b", "c"]
    result = fit_chemical_anchors(chem, np.ones(4, bool), names, names,
                                  {"bits": 3, "final_coordinate": "valid_SMILES_indicator"})
    assert result["anchor_ids"] == ["a_alias", "b", "c"]
    assert result["unique_training_fingerprints"] == 3


def test_tanimoto_excludes_validity_bit_and_structured_bank_is_exact():
    chem = np.array([[1., 0., 0., 1.], [0., 1., 0., 1.], [1., 1., 0., 1.]])
    mask = np.ones(3, bool)
    anchors = fit_chemical_anchors(chem, mask, ["a", "b", "c"], ["a", "b"],
                                  {"bits": 3, "final_coordinate": "valid_SMILES_indicator"}, max_anchors=2)
    model = ChemistryResponseKernelPrior(4, 12, 3, anchor_data=anchors)
    descriptors, active = model.descriptors(torch.tensor(chem).float(), torch.tensor(mask))
    expected = torch.tensor([[1., 0., 1/3], [0., 1., 1/3], [.5, .5, 2/3]])
    torch.testing.assert_close(descriptors, expected)
    assert active.all()
    bank = model.basis(descriptors)
    expected_bank = torch.cat((torch.ones((3, 1)), expected[:, :2], expected[:, :2]**2, expected[:, :2]**4), -1)
    torch.testing.assert_close(bank, expected_bank, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["structured", "generic", "mlp"])
def test_all_full_modes_modify_prior_and_have_finite_actual_parameter_gradients(mode):
    chem, mask, _, _, _, anchors = fixture()
    torch.manual_seed(14)
    model = ChemistryResponseKernelPrior(9, 12, 3, mode, anchors)
    inputs = prior_inputs(chem, mask)
    outputs = model(*inputs)
    for output, original in zip(outputs, inputs[2:]):
        assert output.shape == original.shape and torch.isfinite(output).all()
        assert not torch.equal(output, original)
    ratio = outputs[2] / inputs[4]
    assert (ratio >= np.exp(-2)).all() and (ratio <= np.exp(2)).all()
    loss = sum(value.square().mean() for value in outputs)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for layer in (model.token_delta, model.mean_delta, model.log_variance_delta):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    if mode != "mlp":
        assert isinstance(model.coefficients[0], SplineKANLinear)
        assert model.coefficients[0].spline_weight.grad.abs().sum() > 0
    if mode == "generic":
        assert model.centers.grad.abs().sum() > 0
    assert model.description()["prior_conditioning"].startswith("chemical structure only")


def test_all_modes_have_identical_available_descriptors():
    chem, mask, _, _, _, anchors = fixture()
    inputs = prior_inputs(chem, mask)
    descriptors = [ChemistryResponseKernelPrior(9, 12, 3, mode, anchors).descriptors(*inputs[:2])
                   for mode in ("structured", "generic", "mlp")]
    for d, m in descriptors[1:]:
        torch.testing.assert_close(d, descriptors[0][0], rtol=0, atol=0)
        torch.testing.assert_close(m, descriptors[0][1], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["structured", "generic", "mlp"])
def test_invalid_missing_and_empty_fingerprints_preserve_original_prior_exactly(mode):
    chem, mask, _, _, _, anchors = fixture()
    model = ChemistryResponseKernelPrior(9, 12, 3, mode, anchors)
    chem[0] = np.nan
    mask[0] = False
    chem[1, :8] = np.nan
    chem[1, 8] = 0  # Explicit invalid structure.
    chem[2, :8] = 0  # No active bits despite a valid-SMILES flag.
    inputs = prior_inputs(chem, mask)
    outputs = model(*inputs)
    for output, original in zip(outputs, inputs[2:]):
        torch.testing.assert_close(output[:3], original[:3], rtol=0, atol=0)
        assert torch.isfinite(output).all()
    all_missing = list(inputs)
    all_missing[1] = torch.zeros_like(inputs[1])
    result = model(*all_missing)
    assert all(a is b for a, b in zip(result, inputs[2:]))


def test_zero_available_training_anchors_means_exact_identity_not_a_random_new_prior():
    chem, mask, ids, train, metadata, _ = fixture()
    mask[:8] = False
    anchors = fit_chemical_anchors(chem, mask, ids, train, metadata)
    assert anchors["anchor_ids"] == []
    model = ChemistryResponseKernelPrior(9, 12, 3, anchor_data=anchors)
    inputs = prior_inputs(chem, np.ones(12, bool))
    result = model(*inputs)
    assert all(a is b for a, b in zip(result, inputs[2:]))


def test_no_dropout_and_batch_permutation_equivariance():
    chem, mask, _, _, _, anchors = fixture()
    model = ChemistryResponseKernelPrior(9, 12, 3, anchor_data=anchors).eval()
    inputs = prior_inputs(chem, mask)
    first = model(*inputs)
    permutation = torch.tensor([8, 2, 0, 11, 6, 10, 7, 5, 1, 4, 9, 3])
    reordered = model(*(x[permutation] for x in inputs))
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())
    for a, b in zip(first, reordered):
        torch.testing.assert_close(a[permutation], b, rtol=0, atol=0)


def test_json_metadata_and_state_roundtrip_keep_outputs_exact():
    chem, mask, _, _, _, anchors = fixture()
    model = ChemistryResponseKernelPrior(9, 12, 3, anchor_data=anchors)
    restored = ChemistryResponseKernelPrior(9, 12, 3, anchor_data=json.loads(json.dumps(anchors)))
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = prior_inputs(chem, mask)
    for original, loaded in zip(model(*inputs), restored(*inputs)):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="anchor geometry"):
        bad = copy.deepcopy(model.state_dict())
        bad["anchor_fingerprints"][0, 0] = 1 - bad["anchor_fingerprints"][0, 0]
        restored.load_state_dict(bad)


def test_provenance_validation_and_chemical_coordinate_errors_are_explicit():
    chem, mask, ids, train, metadata, anchors = fixture()
    changed = copy.deepcopy(anchors)
    changed["anchor_ids"][0] = str(ids[-1])
    with pytest.raises(ValueError, match="training chemical"):
        validate_anchor_data(changed)
    changed = copy.deepcopy(anchors)
    changed["anchor_fingerprints"][0][0] = 1 - changed["anchor_fingerprints"][0][0]
    with pytest.raises(ValueError, match="differs from"):
        validate_anchor_data(changed)
    with pytest.raises(ValueError, match="Validity indicator"):
        fit_chemical_anchors(chem, mask, ids, train, dict(metadata, validity_index=0))
    with pytest.raises(ValueError, match="explicit fingerprint"):
        fit_chemical_anchors(chem, mask, ids, train)
    with pytest.raises(ValueError, match="missing"):
        fit_chemical_anchors(chem, mask, ids, train + ["not_in_dataset"], metadata)
    with pytest.raises(ValueError, match="binary"):
        bad = chem.copy()
        bad[0, 1] = .5
        fit_chemical_anchors(bad, mask, ids, train, metadata)
    with pytest.raises(ValueError, match="mode"):
        ChemistryResponseKernelPrior(9, 12, 3, "fake", anchors)


def test_source5_512bit_defaults_and_original_metadata_exclude_the_513th_coordinate():
    chem = np.zeros((2, 513))
    chem[0, 4] = 1
    chem[1, 511] = 1
    chem[:, 512] = 1
    mask = np.ones(2, bool)
    inferred = fit_chemical_anchors(chem, mask, ["a", "b"], ["a", "b"])
    declared = fit_chemical_anchors(chem, mask, ["a", "b"], ["a", "b"],
        {"bits": 512, "final_coordinate": "valid_SMILES_indicator", "kind": "RDKit Morgan fingerprint", "radius": 2})
    assert inferred["fingerprint_indices"] == declared["fingerprint_indices"] == list(range(512))
    assert inferred["validity_index"] == declared["validity_index"] == 512
    model = ChemistryResponseKernelPrior(513, 12, 3, anchor_data=declared)
    d, _ = model.descriptors(torch.tensor(chem).float(), torch.tensor(mask))
    torch.testing.assert_close(d[:, :2], torch.eye(2), rtol=0, atol=0)
