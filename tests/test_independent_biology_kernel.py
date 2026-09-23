"""Interface, optimization and isolation checks; no real assay data are read."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2.conditional_response_kernel import LocalResponseBank
from opal2.hierarchical_geometry import RidgeResidualMean
from opal2.mechanism_response_kernel import MechanismResponseBank, MechanismResponseKernelMean
from opal2.independent_biology_kernel import IndependentBiologyKernelMean, MODES


def fixture(mode='biology', *, anchors=4, n=14, fit_n=10):
    torch.manual_seed(941)
    rng = np.random.default_rng(73)
    x = torch.tensor(rng.normal(size=(n, 7)), dtype=torch.float64)
    bits = np.asarray([[(i >> j) & 1 for j in range(9)] for i in range(1, n+1)], float)
    chem = torch.tensor(np.column_stack((bits, np.ones(n))), dtype=torch.float64)
    mask = torch.ones(n, dtype=torch.bool)
    ids = np.asarray([f'unit_{i:03d}' for i in range(n)])
    local = LocalResponseBank.fit(x.numpy(), chem.numpy(), mask.numpy(), ids, ids[:fit_n],
        dict(fingerprint_indices=list(range(9)), validity_index=9, kind='synthetic'), max_anchors=anchors)
    target = torch.tensor(rng.integers(0, 2, size=(n, 5)), dtype=torch.float64)
    moa = torch.tensor(rng.integers(0, 2, size=(n, 4)), dtype=torch.float64)
    target[:, 0] = 1; moa[:, 0] = 1
    bio = dict(target=target, target_mask=torch.ones(n, dtype=torch.bool),
               moa=moa, moa_mask=torch.ones(n, dtype=torch.bool))
    bank = MechanismResponseBank.fit(local, bio, ids, ids[:fit_n])
    hr = RidgeResidualMean(7, rng.normal(size=(7, 9))*.04, np.zeros(9), dropout=.25)
    with torch.no_grad():
        hr.network[-1].weight.normal_(0, .04)
    base = MechanismResponseKernelMean(hr, bank, mode='old_generic')
    with torch.no_grad():
        base.output.weight.normal_(0, .07)
        base.conditioner[-1].weight.normal_(0, .03)
    base.eval()
    model = IndependentBiologyKernelMean(base, mode=mode)
    packed = bank.pack_information(chem, bio)
    return model, base, x, chem, mask, bio, packed, ids


@pytest.mark.parametrize('mode', MODES)
def test_zero_start_gradient_flow_complete_frozen_base_and_exact_off(mode):
    model, base, x, chem, mask, bio, packed, ids = fixture(mode)
    baseline = base(x, packed, mask).detach()
    model.train()
    assert model.training and not model.base_a.training
    assert not any(module.training for module in model.base_a.modules())
    assert torch.equal(model(x, packed, mask), baseline)
    assert torch.equal(model.baseline_mean(x, packed, mask), baseline)
    assert torch.equal(model(x, chem, mask, bio=bio), baseline)
    frozen = {key: value.clone() for key, value in model.base_a.state_dict().items()}
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.02, weight_decay=0.)
    target = baseline+torch.linspace(.1, .3, len(x))[:, None]
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        losses = model.loss(x, packed, mask, target)
        torch.testing.assert_close(losses['loss'], losses['mean_mse']+.1*losses['incremental_mse'])
        losses['loss'].backward()
        for channel in model.channels:
            assert channel.output.weight.grad.abs().sum() > 0
            if step >= 1:
                assert channel.local_coefficients.grad.abs().sum() > 0
                assert channel.gate[-1].weight.grad.abs().sum() > 0
            if step >= 2:
                assert channel.gate[0].weight.grad.abs().sum() > 0
        optimizer.step()
        assert torch.equal(model(x, packed, mask, branch_enabled=False), baseline)
    for key, value in frozen.items():
        assert torch.equal(model.base_a.state_dict()[key], value)
    assert all(parameter.grad is None for parameter in model.base_a.parameters())
    # Caller-owned original A was copied, not modified or put into train mode.
    assert torch.equal(base(x, packed, mask), baseline)
    assert not torch.equal(model(x, packed, mask), baseline)
    details = model.diagnostics(x, packed, mask, ids=ids)
    torch.testing.assert_close(details['block_contributions'].sum(1), details['raw_increment'])
    torch.testing.assert_close(details['increment'], details['mean']-details['baseline_mean'])
    assert details['total_correction_max'] <= base.base_hr.correction_bound+1e-15
    assert details['raw_increment_max'] <= model.raw_increment_bound+1e-15


def test_modes_have_exact_parameter_capacity_and_initialization():
    _, base, *_ = fixture(n=70, fit_n=66, anchors=64)
    models = []
    for mode in MODES:
        torch.manual_seed(712)
        models.append(IndependentBiologyKernelMean(base, mode=mode))
    assert [sum(p.numel() for p in m.trainable_parameters()) for m in models] == [1794, 1794]
    for (name_a, p_a), (name_b, p_b) in zip(models[0].channels.named_parameters(), models[1].channels.named_parameters()):
        assert name_a == name_b and torch.equal(p_a, p_b)
    assert models[0].bank is models[0].base_a.bank
    assert not list(models[0].bank.parameters())
    assert models[0].channels[0].output.bias is None
    assert models[0].channels[0].output.weight.data_ptr() != models[0].channels[1].output.weight.data_ptr()
    assert all(not p.requires_grad for p in models[0].base_a.parameters())


def test_unknown_known_no_overlap_and_independent_biology_availability():
    model, _, x, chem, mask, bio, _, _ = fixture()
    bank = model.bank
    with torch.no_grad():
        bank.anchor_target.zero_(); bank.anchor_target[:, 0] = 1
        bank.anchor_moa.zero_(); bank.anchor_moa[:, 0] = 1
        for channel in model.channels:
            channel.output.weight.fill_(.8)
    clean = {k: v.clone() for k, v in bio.items()}
    clean['target'].zero_(); clean['target'][:, 0] = 1
    clean['moa'].zero_(); clean['moa'][:, 0] = 1
    # Unknown annotations may carry arbitrary placeholders and contribute zero.
    clean['target_mask'][0] = False; clean['target'][0].fill_(torch.nan)
    clean['moa_mask'][0] = False; clean['moa'][0].fill_(torch.nan)
    # A known, nonempty annotation set with no reference overlap is distinct.
    clean['target'][1].zero_(); clean['target'][1, 4] = 1
    clean['moa'][1].zero_(); clean['moa'][1, 3] = 1
    # Useful target relation remains usable despite missing chemical input.
    absent_chem = chem.clone(); absent_mask = mask.clone()
    absent_mask[2] = False; absent_chem[2].fill_(torch.nan)
    packed = bank.pack_information(absent_chem, clean)
    details = model.diagnostics(x, packed, absent_mask)
    assert not details['known'][0].any() and details['known'][1].all()
    assert not details['support'][:2].any()
    assert torch.equal(details['mean'][:2], details['baseline_mean'][:2])
    assert details['support'][2].all()
    assert details['channel_gate'][2].min() > 0
    assert not torch.equal(details['mean'][2], details['baseline_mean'][2])
    # One channel can vanish without suppressing the other channel.
    clean['target_mask'].fill_(False)
    changed = model.diagnostics(x, bank.pack_information(absent_chem, clean), absent_mask)
    assert changed['block_contributions'][:, 0].count_nonzero() == 0
    assert torch.equal(changed['block_contributions'][:, 1], details['block_contributions'][:, 1])


def test_support_aggregation_is_bounded_and_effective_count_is_explicit():
    model, *_ = fixture()
    weights = torch.tensor([[[1., 0., 0., 0.], [.5, .5, 0., 0.]],
                            [[.001, 0., 0., 0.], [0., 0., 0., 0.]]], dtype=torch.float64)
    known = torch.ones((2, 2), dtype=torch.bool)
    support = model._support(weights, known)
    torch.testing.assert_close(support['effective_neighbors'], torch.tensor([[1., 2.], [1., 0.]], dtype=torch.float64))
    torch.testing.assert_close(support['normalized_weights'].sum(-1), support['support'].to(torch.float64))
    bases = model.response_functions(weights)*support['normalized_weights'][..., None]
    assert bases.min() >= 0 and bases.sum(-2).max() <= 1.
    assert bases[1, 0].max() < .002
    assert model.response_functions(torch.zeros_like(weights)).count_nonzero() == 0
    torch.testing.assert_close(model.response_functions(torch.ones_like(weights)), torch.ones((*weights.shape, 3), dtype=torch.float64))
    assert support['support_features'][1, 1, 0] == 1
    assert support['support_features'][1, 1, 1] == 0


@pytest.mark.parametrize('mode', MODES)
def test_serialization_roundtrip_preserves_complete_baseline_and_prediction(mode, tmp_path):
    model, _, x, _, mask, _, packed, _ = fixture(mode)
    with torch.no_grad():
        for channel in model.channels:
            channel.output.weight.normal_(0, .2)
            channel.gate[-1].weight.normal_(0, .1)
    model.eval()
    expected = model(x, packed, mask)
    restored = IndependentBiologyKernelMean.from_config(json.loads(json.dumps(model.config)))
    restored.load_state_dict(model.state_dict())
    restored.eval()
    assert torch.equal(expected, restored(x, packed, mask))
    assert torch.equal(model.baseline_mean(x, packed, mask), restored.baseline_mean(x, packed, mask))
    model.save(tmp_path/'complete_model.pt')
    loaded = IndependentBiologyKernelMean.load(tmp_path/'complete_model.pt')
    assert torch.equal(expected, loaded(x, packed, mask))
    assert not loaded.base_a.training


def test_invalid_inputs_configs_and_changed_architecture_rejected():
    model, base, x, _, mask, _, packed, _ = fixture()
    for kwargs in (dict(mode='combined'), dict(support_shrinkage=0), dict(raw_increment_bound=-1),
                   dict(incremental_penalty=float('nan')), dict(hidden_dim=0)):
        with pytest.raises(ValueError): IndependentBiologyKernelMean(base, **kwargs)
    invalid = deepcopy(base); invalid.mode = 'bio_generic'
    with pytest.raises(TypeError): IndependentBiologyKernelMean(invalid)
    changed = deepcopy(model.config); changed['penalty_reference'] = 'HR'
    with pytest.raises(ValueError): IndependentBiologyKernelMean.from_config(changed)
    with pytest.raises(ValueError): model(x, packed, mask, branch_enabled=0)
    with pytest.raises(ValueError): model.loss(x, packed, mask, torch.zeros(len(x), 8, dtype=x.dtype))
    with pytest.raises(ValueError): model.response_functions(torch.tensor([float('nan')]))
