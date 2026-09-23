"""Training connection tests, not experimental evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2 import conditional_response_experiment as experiment
from opal2.conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from opal2.hierarchical_geometry import RidgeResidualMean


@pytest.mark.parametrize('mode', ['static_structured', 'conditional_generic', 'conditional_structured'])
def test_actual_thirty_epoch_training_and_checkpoint(tmp_path, monkeypatch, mode):
    rng = np.random.default_rng(241)
    n, d = 20, 6
    x = rng.normal(size=(n, d))
    chem = (rng.random((n, 32)) < .25).astype(float)
    chem[:, 0] = 1
    mask = np.ones(n, dtype=bool)
    ids = np.asarray([f'unit_{i}' for i in range(n)])
    fit, valid = np.arange(15), np.arange(15, n)
    bank = LocalResponseBank.fit(x, chem, mask, ids, ids[fit],
        metadata={'fingerprint_indices': list(range(32))}, max_anchors=5)
    base = RidgeResidualMean(d, torch.zeros(d, 9, dtype=torch.float64),
        torch.zeros(9, dtype=torch.float64), hidden_dim=8).double()
    torch.manual_seed(129)
    model = ConditionalResponseKernelMean(base, bank, mode=mode, hidden_dim=8)
    target = np.tile(.1+.08*np.tanh(x[:, 0, None]), (1, 9))
    cfg = deepcopy(experiment.CONFIG)
    cfg.update(batch_size=8, threads=1)
    monkeypatch.setattr(experiment, 'CONFIG', cfg)
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()
              if key.startswith(('base_hr.', 'bank.'))}
    result = experiment.train_branch(tmp_path/mode, model, x, chem, mask,
        target, fit, valid, 321, ids)
    for key, tensor in before.items():
        assert torch.equal(result.state_dict()[key], tensor)
    saved = torch.load(tmp_path/mode/'epoch30.pt', map_location='cpu', weights_only=True)
    assert saved['actual_checkpoint_epoch'] == 30
    assert saved['optimizer_steps'] == 60
    assert saved['fit_ids'] == ids[fit].tolist()
    assert saved['validation_ids'] == ids[valid].tolist()
    assert 'optimizer_state_dict' in saved and 'torch_rng_state' in saved
    loaded = ConditionalResponseKernelMean.from_config(saved['model_config'])
    loaded.load_state_dict(saved['state_dict'])
    assert np.array_equal(experiment.predict_branch(loaded, x, chem, mask),
                          experiment.predict_branch(result, x, chem, mask))
    history = [json.loads(line) for line in (tmp_path/mode/'history.jsonl').read_text().splitlines()]
    assert len(history) == 31 and history[-1]['epoch'] == 30
    assert history[-1]['gradient_norm_mean'] > 0
    for name, parameter in result.named_parameters():
        if parameter.requires_grad:
            assert max(row['gradient_parameter_norm_mean'].get(name, 0) for row in history) > 0
    report = json.loads((tmp_path/mode/'training_complete.json').read_text())
    assert report['actual_checkpoint_epoch'] == 30 and not report['bank_changed']
    assert report['parameter_counts']['trainable'] == sum(p.numel() for p in result.trainable_parameters())
    with pytest.raises(FileExistsError):
        experiment.train_branch(tmp_path/mode, model, x, chem, mask, target, fit, valid, 321, ids)
