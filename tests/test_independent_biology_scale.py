"""Fixed input-only TRAIN scaling without changing schema-1 behavior."""
from copy import deepcopy
import inspect
import json
import math

import pytest
import torch

from opal2.independent_biology_kernel import IndependentBiologyKernelMean, MODES
from test_independent_biology_kernel import fixture


def scaled_fixture(mode='biology'):
    _, base, x, chem, mask, bio, packed, ids = fixture(mode)
    model = IndependentBiologyKernelMean(base, mode=mode, aggregation_scaling='train_fixed')
    return model, base, x, chem, mask, bio, packed, ids


def response_mean(weight):
    rbf = (math.exp(-2*(weight-1)**2)-math.exp(-2))/(1-math.exp(-2))
    return (weight+weight**2+rbf)/3


def test_estimator_matches_supported_row_vector_norm_hand_math(monkeypatch):
    model, _, x, _, mask, _, packed, ids = scaled_fixture()
    weights = x.new_tensor([[[1., 0., 0., 0.], [.001, 0., 0., 0.]],
                            [[.5, .5, 0., 0.], [0., 0., 0., 0.]],
                            [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                            [[0., 0., 0., 0.], [0., 0., 0., 0.]]])
    known = torch.ones((4, 2), dtype=torch.bool)
    monkeypatch.setattr(model, '_relations', lambda *args: (weights, known))
    metadata = model.fit_aggregation_scale(x[:4], packed[:4], mask[:4], ids=ids[:4])
    expected = x.new_tensor([math.sqrt((1.+2*(response_mean(.5)/2)**2)/2), response_mean(.001)])
    torch.testing.assert_close(model.aggregation_scale_raw_s, expected, rtol=1e-12, atol=1e-15)
    assert metadata['count'] == [2, 1]
    assert metadata['capped'] == [False, True]
    assert metadata['gain'] == pytest.approx([1/expected[0].item(), 32.])
    assert metadata['scale_floor'] == 1/32
    assert metadata['fit_ids'] == ids[:4].tolist()
    assert json.loads(json.dumps(metadata)) == metadata
    details = model.diagnostics(x[:4], packed[:4], mask[:4])
    torch.testing.assert_close(details['raw_initial_local_rms'], expected)
    torch.testing.assert_close(details['scaled_initial_local_rms'], expected*model.aggregation_scale_gain)
    torch.testing.assert_close(details['raw_local_rms']*model.aggregation_scale_gain,
                               details['scaled_local_rms'])
    assert details['raw_local_basis'][2:].count_nonzero() == 0
    assert details['scaled_local_basis'][2:].count_nonzero() == 0
    # Every query, anchor and response receives the same channel multiplier.
    torch.testing.assert_close(details['scaled_local_basis'],
                               details['raw_local_basis']*model.aggregation_scale_gain[None, :, None, None])
    torch.testing.assert_close(details['local_basis'], details['scaled_local_basis'])
    assert details['aggregation_scale'] == metadata


@pytest.mark.parametrize('nominal_support', [False, True])
def test_all_zero_or_unsupported_channel_has_no_boost(monkeypatch, nominal_support):
    model, _, x, _, mask, _, packed, ids = scaled_fixture()
    weights = torch.ones((3, 2, 4), dtype=x.dtype) if nominal_support else torch.zeros((3, 2, 4), dtype=x.dtype)
    known = torch.ones((3, 2), dtype=torch.bool)
    monkeypatch.setattr(model, '_relations', lambda *args: (weights, known))
    if nominal_support:
        monkeypatch.setattr(model, 'response_functions', lambda value: value.new_zeros((*value.shape, 3)))
    result = model.fit_aggregation_scale(x[:3], packed[:3], mask[:3], ids=ids[:3])
    assert result['count'] == [0, 0]
    assert result['raw_s'] == [0., 0.]
    assert result['gain'] == [1., 1.]
    assert result['capped'] == [False, False]
    assert result['fitted']


@pytest.mark.parametrize('mode', MODES)
def test_input_only_numpy_api_and_calibration_does_not_read_predictor_or_labels(mode):
    model, base, x, _, mask, _, packed, ids = scaled_fixture(mode)
    altered = deepcopy(model)
    with torch.no_grad():
        for parameter in altered.base_a.base_hr.network.parameters():
            parameter.fill_(123.)
        altered.base_a.output.weight.fill_(234.)
    actual = model.fit_aggregation_scale(x[:10].numpy(), packed[:10].numpy(), mask[:10].numpy(), ids=ids[:10])
    other = altered.fit_aggregation_scale(x[:10], packed[:10], mask[:10], ids=ids[:10])
    assert actual == other
    assert tuple(inspect.signature(model.fit_aggregation_scale).parameters) == ('x_fit', 'packed_fit', 'mask_fit', 'ids')
    with pytest.raises(TypeError):
        model.fit_aggregation_scale(x[:10], packed[:10], mask[:10], ids=ids[:10], target=x.new_zeros(10, 9))
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.base_a.state_dict()[key]) for key, value in base.state_dict().items())


def test_calibration_requires_aligned_unique_train_ids_and_initial_parameters():
    model, _, x, _, mask, _, packed, ids = scaled_fixture()
    for fit_ids in ([], ids[:2], ['unit_000']*3, ['unit_000', 'unit_001', 'unit_010']):
        with pytest.raises(ValueError):
            model.fit_aggregation_scale(x[:3], packed[:3], mask[:3], ids=fit_ids)
        assert not model.aggregation_scale_fitted
    with pytest.raises(TypeError):
        model.fit_aggregation_scale(x[:3], packed[:3], mask[:3])
    for changed in ('readout', 'coefficient', 'gradient'):
        altered = deepcopy(model)
        with torch.no_grad():
            if changed == 'readout':
                altered.channels[0].output.weight[0, 0] = .01
            elif changed == 'coefficient':
                altered.channels[1].local_coefficients[0, 0] += .01
            else:
                parameter = altered.channels[0].gate[0].weight
                parameter.grad = torch.zeros_like(parameter)
        with pytest.raises(RuntimeError, match='before training'):
            altered.fit_aggregation_scale(x[:3], packed[:3], mask[:3], ids=ids[:3])
    first = model.fit_aggregation_scale(x[:3], packed[:3], mask[:3], ids=ids[:3])
    with pytest.raises(RuntimeError, match='only be fitted once'):
        model.fit_aggregation_scale(x[3:6], packed[3:6], mask[3:6], ids=ids[3:6])
    assert model.aggregation_scale_metadata() == first


@pytest.mark.parametrize('mode', MODES)
def test_fitted_scale_preserves_baseline_zeros_bounds_and_gradient_learning(mode):
    model, base, x, chem, mask, bio, packed, ids = scaled_fixture(mode)
    baseline = base(x, packed, mask).detach()
    assert torch.equal(model(x, packed, mask, branch_enabled=False), baseline)
    assert torch.equal(model.diagnostics(x, packed, mask, branch_enabled=False)['mean'], baseline)
    with pytest.raises(RuntimeError, match='Fit the TRAIN aggregation scale'):
        model(x, packed, mask)
    with pytest.raises(RuntimeError, match='Fit the TRAIN aggregation scale'):
        model.loss(x, packed, mask, baseline)
    model.fit_aggregation_scale(x[:10], packed[:10], mask[:10], ids=ids[:10])
    calibration = model.aggregation_scale_metadata()
    assert torch.equal(model(x, packed, mask), baseline)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=.02, weight_decay=0.)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        model.loss(x, packed, mask, baseline+.2)['loss'].backward()
        for channel in model.channels:
            assert channel.output.weight.grad.abs().sum() > 0
            if step:
                assert channel.local_coefficients.grad.abs().sum() > 0
                assert channel.gate[-1].weight.grad.abs().sum() > 0
        optimizer.step()
    assert model.aggregation_scale_metadata() == calibration
    assert not torch.equal(model(x, packed, mask), baseline)
    assert torch.equal(model(x, packed, mask, branch_enabled=False), baseline)
    with torch.no_grad():
        expected = model(x, packed, mask)
        torch.testing.assert_close(torch.cat([model(x[:5], packed[:5], mask[:5]),
                                              model(x[5:], packed[5:], mask[5:])]), expected)
    absent = {key: value.clone() for key, value in bio.items()}
    absent['target_mask'][0] = False; absent['moa_mask'][0] = False
    absent_mask = mask.clone(); absent_mask[0] = False
    empty_x = x.clone(); empty_x[0, :-1] = 0
    empty_packed = model.bank.pack_information(chem, absent)
    details = model.diagnostics(empty_x, empty_packed, absent_mask)
    assert not details['support'][0].any()
    assert details['raw_local_basis'][0].count_nonzero() == 0
    assert details['scaled_local_basis'][0].count_nonzero() == 0
    assert details['local_activation'][0].count_nonzero() == 0
    assert torch.equal(details['mean'][0], details['baseline_mean'][0])
    assert details['raw_increment_max'] <= model.raw_increment_bound+1e-15
    assert details['total_correction_max'] <= base.base_hr.correction_bound+1e-15


@pytest.mark.parametrize('mode', MODES)
def test_opt_in_serialization_preserves_scale_and_predictions(mode, tmp_path):
    model, _, x, _, mask, _, packed, ids = scaled_fixture(mode)
    model.fit_aggregation_scale(x[:10], packed[:10], mask[:10], ids=ids[:10])
    with torch.no_grad():
        for channel in model.channels:
            channel.output.weight.normal_(0, .1)
    expected = model(x, packed, mask)
    config = json.loads(json.dumps(model.config))
    restored = IndependentBiologyKernelMean.from_config(config)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.aggregation_scale_metadata() == model.aggregation_scale_metadata()
    assert torch.equal(restored(x, packed, mask), expected)
    with pytest.raises(RuntimeError, match='only be fitted once'):
        restored.fit_aggregation_scale(x[:10], packed[:10], mask[:10], ids=ids[:10])
    model.save(tmp_path/'scaled.pt')
    loaded = IndependentBiologyKernelMean.load(tmp_path/'scaled.pt')
    assert torch.equal(loaded(x, packed, mask), expected)
    assert loaded.aggregation_scale_metadata() == model.aggregation_scale_metadata()
    assert all(not buffer.requires_grad for name, buffer in loaded.named_buffers() if name.startswith('aggregation_scale_'))


def test_default_schema1_strict_state_load_and_no_extra_random_draws_or_parameters():
    _, base, x, _, mask, _, packed, ids = fixture()
    variants = []
    states = []
    for scaling in ('none', 'train_fixed'):
        torch.manual_seed(614)
        variants.append(IndependentBiologyKernelMean(base, aggregation_scaling=scaling))
        states.append(torch.get_rng_state())
    plain, scaled = variants
    assert torch.equal(*states)
    assert plain.config['schema_version'] == 1 and 'aggregation_scaling' not in plain.config
    assert scaled.config['schema_version'] == 2
    assert not any(name.startswith('aggregation_scale_') for name in plain.state_dict())
    assert [name for name, _ in plain.named_parameters()] == [name for name, _ in scaled.named_parameters()]
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), scaled.parameters()))
    assert sum(p.numel() for p in plain.trainable_parameters()) == sum(p.numel() for p in scaled.trainable_parameters())
    with torch.no_grad():
        for channel in plain.channels:
            channel.output.weight.normal_(0, .1)
    restored = IndependentBiologyKernelMean.from_config(deepcopy(plain.config))
    restored.load_state_dict(plain.state_dict(), strict=True)
    assert restored.config == plain.config
    assert torch.equal(restored(x, packed, mask), plain(x, packed, mask))
    with pytest.raises(ValueError, match='requires aggregation_scaling=train_fixed'):
        plain.fit_aggregation_scale(x[:10], packed[:10], mask[:10], ids=ids[:10])


def test_invalid_scaling_configuration_is_rejected():
    model, base, *_ = scaled_fixture()
    for kwargs in (dict(aggregation_scaling='unit_norm'), dict(scale_max_gain=0),
                   dict(scale_max_gain=.5), dict(scale_max_gain=float('inf')),
                   dict(scale_max_gain=float('nan'))):
        with pytest.raises(ValueError):
            IndependentBiologyKernelMean(base, **kwargs)
    for key, value in (('schema_version', 3), ('aggregation_scaling', 'none'),
                       ('scale_floor', .5), ('aggregation_scale_application', 'center then scale')):
        changed = deepcopy(model.config); changed[key] = value
        with pytest.raises(ValueError):
            IndependentBiologyKernelMean.from_config(changed)
