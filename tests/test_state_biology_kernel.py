"""State-conditioned kernel checks using synthetic, explicitly split inputs."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest
import torch

from opal2.hierarchical_geometry import RidgeResidualMean
from opal2.kernel_final_reference import fit_reference_bank
from opal2.mechanism_response_kernel import MechanismResponseBank, MechanismResponseKernelMean
from opal2.state_biology_kernel import MODES, StateBiologyKernelMean


def fixture(mode='state_biology', anchors=4):
    torch.manual_seed(1703)
    rng = np.random.default_rng(339)
    fit_n, extra = 18, 6
    n = anchors+fit_n+extra
    x = torch.tensor(rng.normal(size=(n, 13)), dtype=torch.float64)
    x[:, -1] = x[:, -1]*2+.7
    bits = np.asarray([[(i >> j) & 1 for j in range(9)] for i in range(1, n+1)], float)
    chem = torch.tensor(np.column_stack((bits, np.ones(n))), dtype=x.dtype)
    mask = torch.ones(n, dtype=torch.bool)
    ids = np.asarray([f'state_unit_{i:03d}' for i in range(n)])
    refs, fit = np.arange(anchors), np.arange(anchors, anchors+fit_n)
    local = fit_reference_bank(x.numpy(), chem.numpy(), mask.numpy(), ids,
        ids[:anchors+fit_n].tolist(), ids[fit].tolist(), ids[refs].tolist(),
        dict(fingerprint_indices=list(range(9)), validity_index=9, kind='synthetic'))
    target = torch.tensor(rng.integers(0, 2, size=(n, 6)), dtype=x.dtype)
    moa = torch.tensor(rng.integers(0, 2, size=(n, 5)), dtype=x.dtype)
    target[:, 0] = 1; moa[:, 0] = 1
    target[:, -1] = 0; moa[:, -1] = 0
    bio = dict(target=target, target_mask=mask.clone(), moa=moa, moa_mask=mask.clone())
    bank = MechanismResponseBank.fit(local, bio, ids, ids[fit])
    hr = RidgeResidualMean(13, rng.normal(size=(13, 9))*.02, np.zeros(9), dropout=.25)
    with torch.no_grad():
        hr.network[-1].weight.normal_(0, .03)
    base = MechanismResponseKernelMean(hr, bank, mode='old_generic')
    with torch.no_grad():
        base.output.weight.normal_(0, .04)
    base.eval()
    model = StateBiologyKernelMean(base, mode=mode)
    packed = bank.pack_information(chem, bio)
    return model, base, x, chem, mask, bio, packed, ids, fit, refs


def fit_inputs(model, x, packed, mask, ids, fit, refs):
    state = model.fit_input_state(x[fit], ids=ids[fit], reference_x=x[refs], reference_ids=ids[refs])
    scale = model.fit_aggregation_scale(x[fit], packed[fit], mask[fit], ids=ids[fit])
    return state, scale


def test_exact_train_scope_reference_identity_and_one_time_fit():
    model, _, x, _, mask, _, packed, ids, fit, refs = fixture()
    for invalid_ids in (ids[fit][:-1], np.concatenate((ids[fit][:-1], ids[refs][:1])),
                        np.concatenate((ids[fit][:-1], ids[-1:])), np.repeat(ids[fit][0], len(fit))):
        with pytest.raises(ValueError, match='commonbranch FIT'):
            model.fit_input_state(x[fit], ids=invalid_ids, reference_x=x[refs], reference_ids=ids[refs])
    for invalid_refs in (ids[refs][::-1], np.concatenate((ids[refs][:-1], ids[-1:])), np.repeat(ids[refs][0], len(refs))):
        with pytest.raises(ValueError, match='original TRAIN anchor'):
            model.fit_input_state(x[fit], ids=ids[fit], reference_x=x[refs], reference_ids=invalid_refs)
    wrong = x[refs].clone(); wrong[0, 0] += 1
    with pytest.raises(ValueError, match='anchor directions'):
        model.fit_input_state(x[fit], ids=ids[fit], reference_x=wrong, reference_ids=ids[refs])
    wrong = x[refs].clone(); wrong[0, -1] += 1
    with pytest.raises(ValueError, match='log-norm'):
        model.fit_input_state(x[fit], ids=ids[fit], reference_x=wrong, reference_ids=ids[refs])
    with pytest.raises(RuntimeError, match='before the aggregation scale'):
        model.fit_aggregation_scale(x[fit], packed[fit], mask[fit], ids=ids[fit])
    for parameter in (model.channels[0].output.weight, model.channels[1].local_coefficients):
        changed = deepcopy(model)
        key = next(name for name, p in model.named_parameters() if p is parameter)
        with torch.no_grad(): dict(changed.named_parameters())[key].add_(.1)
        with pytest.raises(RuntimeError, match='before training'):
            changed.fit_input_state(x[fit], ids[fit], x[refs], ids[refs])
    state, scale = fit_inputs(model, x, packed, mask, ids, fit, refs)
    assert state['fit_ids'] == ids[fit].tolist() and state['reference_ids'] == ids[refs].tolist()
    assert scale['fit_ids'] == state['fit_ids']
    assert torch.equal(model.state_fit_train_indices, torch.tensor(fit))
    assert torch.equal(model.state_reference_train_indices, torch.tensor(refs))
    with pytest.raises(RuntimeError, match='only be fitted once'):
        model.fit_input_state(x[fit], ids[fit], x[refs], ids[refs])
    with pytest.raises(RuntimeError, match='only be fitted once'):
        model.fit_aggregation_scale(x[fit], packed[fit], mask[fit], ids=ids[fit])
    with pytest.raises(ValueError, match='identical ordered'):
        model.fit_aggregation_scale(x[fit[:-1]], packed[fit[:-1]], mask[fit[:-1]], ids=ids[fit[:-1]])
    assert json.loads(json.dumps(state)) == state


def test_pca_is_fit_only_deterministic_standardized_and_uses_no_labels_or_rng():
    model, base, x, _, mask, _, packed, ids, fit, refs = fixture()
    torch_state = torch.get_rng_state().clone()
    model.fit_input_state(x[fit].numpy(), ids[fit], x[refs].numpy(), ids[refs])
    assert torch.equal(torch_state, torch.get_rng_state())
    state = model.input_state(x[fit])
    torch.testing.assert_close(state.mean(0), torch.zeros(9, dtype=x.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(state.std(0, unbiased=False), torch.ones(9, dtype=x.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(model.state_feature_center, x[fit, :-1].mean(0), atol=1e-14, rtol=0)
    expected_norm = (x[fit, -1]-x[fit, -1].mean())/x[fit, -1].std(unbiased=False)
    torch.testing.assert_close(state[:, -1], expected_norm)
    components = model.state_pca_components
    torch.testing.assert_close(components@components.T, torch.eye(8, dtype=x.dtype), atol=1e-14, rtol=0)
    assert (components[torch.arange(8), components.abs().argmax(1)] >= 0).all()
    other = StateBiologyKernelMean(base)
    changed = x.clone(); changed[-6:] = 999
    other.fit_input_state(changed[fit], ids[fit], changed[refs], ids[refs])
    for name, value in model.named_buffers():
        if name.startswith('state_'):
            assert torch.equal(value, dict(other.named_buffers())[name])
    assert tuple(inspect.signature(model.fit_input_state).parameters) == ('x_fit', 'ids', 'reference_x', 'reference_ids')
    assert tuple(inspect.signature(model.fit_aggregation_scale).parameters) == ('x_fit', 'packed_fit', 'mask_fit', 'ids')
    with pytest.raises(TypeError):
        model.fit_input_state(x[fit], ids[fit], x[refs], ids[refs], target=torch.zeros(len(fit), 9))
    before_aggregation = torch.get_rng_state().clone()
    model.fit_aggregation_scale(x[fit], packed[fit], mask[fit], ids=ids[fit])
    assert torch.equal(before_aggregation, torch.get_rng_state())


def test_modes_have_matched_6770_active_parameters_and_identical_initial_tensors():
    _, base, x, _, mask, _, packed, ids, fit, refs = fixture(anchors=64)
    models, random_states = [], []
    for mode in MODES:
        torch.manual_seed(289)
        model = StateBiologyKernelMean(base, mode=mode)
        before = torch.get_rng_state().clone()
        fit_inputs(model, x, packed, mask, ids, fit, refs)
        assert torch.equal(before, torch.get_rng_state())
        models.append(model); random_states.append(before)
    assert [sum(p.numel() for p in m.trainable_parameters()) for m in models] == [6770, 6770]
    assert torch.equal(*random_states)
    assert [name for name, _ in models[0].named_parameters()] == [name for name, _ in models[1].named_parameters()]
    assert all(torch.equal(a, b) for a, b in zip(models[0].parameters(), models[1].parameters()))
    assert torch.equal(models[0].state_pca_components, models[1].state_pca_components)
    assert models[0].channels[0].state_encoder[0].weight.data_ptr() != models[0].channels[1].state_encoder[0].weight.data_ptr()
    assert all(channel.output.bias is None for model in models for channel in model.channels)


@pytest.mark.parametrize('mode', MODES)
def test_zero_start_frozen_base_bounds_state_gradients_and_exact_disabled(mode):
    model, base, x, _, mask, _, packed, ids, fit, refs = fixture(mode)
    baseline = base(x, packed, mask).detach()
    assert torch.equal(model(x, packed, mask, branch_enabled=False), baseline)
    assert torch.equal(model.diagnostics(x, packed, mask, branch_enabled=False)['mean'], baseline)
    with pytest.raises(RuntimeError, match='input state'):
        model(x, packed, mask)
    model.fit_input_state(x[fit], ids[fit], x[refs], ids[refs])
    with pytest.raises(RuntimeError, match='aggregation scale'):
        model(x, packed, mask)
    model.fit_aggregation_scale(x[fit], packed[fit], mask[fit], ids=ids[fit])
    assert torch.equal(model(x, packed, mask), baseline)
    frozen = {key: value.clone() for key, value in model.base_a.state_dict().items()}
    fixed = {key: value.clone() for key, value in model.named_buffers() if not key.startswith('base_a.')}
    model.train()
    assert model.training and not any(part.training for part in model.base_a.modules())
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.02, weight_decay=0.)
    target = baseline+torch.linspace(.1, .4, len(x))[:, None]
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        model.loss(x, packed, mask, target)['loss'].backward()
        for channel in model.channels:
            assert channel.output.weight.grad.abs().sum() > 0
            if step == 0:
                assert channel.state_encoder[0].weight.grad.count_nonzero() == 0
            if step:
                assert channel.local_coefficients.grad.abs().sum() > 0
                assert channel.state_encoder[0].weight.grad.abs().sum() > 0
                assert channel.state_encoder[0].bias.grad.abs().sum() > 0
                assert channel.gate[-1].weight.grad.abs().sum() > 0
            if step >= 2:
                assert channel.gate[0].weight.grad.abs().sum() > 0
                assert channel.gate[0].weight.grad[:, 6:].abs().sum() > 0
        optimizer.step()
    assert all(torch.equal(value, model.base_a.state_dict()[key]) for key, value in frozen.items())
    assert all(torch.equal(value, dict(model.named_buffers())[key]) for key, value in fixed.items())
    assert all(p.grad is None for p in model.base_a.parameters())
    assert torch.equal(model(x, packed, mask, branch_enabled=False), baseline)
    details = model.diagnostics(x, packed, mask)
    assert details['raw_increment_max'] <= model.raw_increment_bound+1e-15
    assert details['total_correction_max'] <= base.base_hr.correction_bound+1e-15
    torch.testing.assert_close(details['state_readout_contributions'].sum(2), details['channel_readout'])
    assert details['input_state'].shape == (len(x), 9)
    assert details['state_modulation'].shape == (len(x), 2, 5)


def test_state_only_new_branch_ignores_annotations_and_chemical_availability():
    model, _, x, chem, mask, _, packed, ids, fit, refs = fixture('state_only')
    fit_inputs(model, x, packed, mask, ids, fit, refs)
    with torch.no_grad():
        for channel in model.channels: channel.output.weight.normal_(0, .1)
    baseline = model(x, packed, mask)
    changed = packed.clone(); changed[:, model.bank.chemical_dim:] = torch.nan
    assert torch.equal(model(x, changed, mask), baseline)
    assert torch.equal(model(x, chem, mask, bio={'not_an_annotation': 123}), baseline)
    detail = model.diagnostics(x, packed, mask)
    absent = model.diagnostics(x, torch.full_like(chem, torch.nan), torch.zeros_like(mask))
    assert torch.equal(absent['channel_readout'], detail['channel_readout'])
    assert torch.equal(absent['raw_increment'], detail['raw_increment'])
    assert detail['support'][:, 1].all()
    norms = torch.linalg.vector_norm(x[:, :-1], dim=-1)
    cosine = (x[:, :-1]/norms[:, None])@model.bank.descriptor_bank.anchor_directions.T
    torch.testing.assert_close(detail['relation_weights'][:, 0], cosine.clamp(0, 1))
    norm = detail['input_state'][:, -1]
    torch.testing.assert_close(detail['relation_weights'][:, 1],
                               torch.exp(-.5*(norm[:, None]-model.state_reference_log_norm[None]).square()))


def test_biology_unsupported_queries_recover_A_with_nonzero_trained_parameters():
    model, _, x, chem, mask, bio, packed, ids, fit, refs = fixture()
    fit_inputs(model, x, packed, mask, ids, fit, refs)
    with torch.no_grad():
        for channel in model.channels: channel.output.weight.normal_(0, .4)
    changed = {key: value.clone() for key, value in bio.items()}
    changed['target_mask'][0] = False; changed['moa_mask'][0] = False
    changed['target'][0] = torch.nan; changed['moa'][0] = torch.nan
    changed['target'][1] = 0; changed['target'][1, -1] = 1
    changed['moa'][1] = 0; changed['moa'][1, -1] = 1
    details = model.diagnostics(x, model.bank.pack_information(chem, changed), mask)
    assert not details['support'][:2].any()
    assert torch.equal(details['mean'][:2], details['baseline_mean'][:2])
    assert details['raw_increment'][:2].count_nonzero() == 0
    assert details['local_basis'][:2].count_nonzero() == 0


def test_same_biology_with_different_state_changes_signed_readout_direction():
    model, _, x, _, mask, _, packed, ids, fit, refs = fixture()
    fit_inputs(model, x, packed, mask, ids, fit, refs)
    with torch.no_grad():
        channel = model.channels[0]
        channel.state_encoder[0].weight.zero_(); channel.state_encoder[0].bias.zero_()
        channel.state_encoder[0].weight[0, -1] = 1.
        channel.output.weight.zero_()
        channel.output.weight[0] = .4       # constant coordinate zero
        channel.output.weight[9+1] = .5   # signed latent-zero coordinate one
    query = x[:1].repeat(2, 1)
    query[:, -1] = model.state_log_norm_center + model.state_log_norm_scale*torch.tensor([-1., 1.], dtype=x.dtype)
    same_bio = packed[:1].repeat(2, 1)
    details = model.diagnostics(query, same_bio, mask[:2])
    assert torch.equal(details['relation_weights'][0], details['relation_weights'][1])
    assert torch.equal(details['local_basis'][0], details['local_basis'][1])
    raw = details['channel_readout'][:, 0]
    assert raw[:, 0].min() > 0 and raw[0, 1] < 0 < raw[1, 1]
    assert not torch.allclose(raw[0]/raw[0].norm(), raw[1]/raw[1].norm())
    torch.testing.assert_close(raw[:, 0], raw[0, 0].expand(2))
    torch.testing.assert_close(raw[0, 1], -raw[1, 1])


@pytest.mark.parametrize('mode', MODES)
def test_serialization_roundtrip_and_definition_checks(mode, tmp_path):
    model, _, x, _, mask, _, packed, ids, fit, refs = fixture(mode)
    fit_inputs(model, x, packed, mask, ids, fit, refs)
    with torch.no_grad():
        for channel in model.channels: channel.output.weight.normal_(0, .1)
    expected = model(x, packed, mask)
    restored = StateBiologyKernelMean.from_config(json.loads(json.dumps(model.config)))
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(expected, restored(x, packed, mask))
    assert restored.input_state_metadata() == model.input_state_metadata()
    assert restored.aggregation_scale_metadata() == model.aggregation_scale_metadata()
    model.save(tmp_path/'state.pt')
    loaded = StateBiologyKernelMean.load(tmp_path/'state.pt')
    assert torch.equal(expected, loaded(x, packed, mask))
    assert not loaded.base_a.training
    with pytest.raises(RuntimeError, match='only be fitted once'):
        loaded.fit_input_state(x[fit], ids[fit], x[refs], ids[refs])
    for key, value in (('model_type', 'independent_biology_kernel'), ('state_dim', 8),
                       ('state_readout', 'positive scalar gate'), ('pca_components', 7)):
        invalid = deepcopy(model.config); invalid[key] = value
        with pytest.raises(ValueError): StateBiologyKernelMean.from_config(invalid)


def test_invalid_constructor_options():
    _, base, *_ = fixture()
    for config in (dict(mode='biology'), dict(aggregation_scaling='none'), dict(hidden_dim=0),
                   dict(scale_max_gain=0), dict(support_shrinkage=0), dict(raw_increment_bound=-1)):
        with pytest.raises(ValueError): StateBiologyKernelMean(base, **config)
