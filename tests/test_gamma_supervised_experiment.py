"""Changed combined-loss training/checkpoint path; no research claims."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2 import gamma_supervised_experiment as experiment
from opal2.conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from opal2.gamma_supervised_loss import JointGammaCRPS, gamma_from_raw_coordinates
from opal2.hierarchical_geometry import RidgeResidualMean


def test_gamma_training_checkpoint_monitoring_and_frozen_distribution(tmp_path, monkeypatch):
    rng = np.random.default_rng(783)
    n, d = 20, 6
    x = rng.normal(size=(n, d))
    chem = (rng.random((n, 32)) < .3).astype(float)
    chem[:, 0] = 1
    mask = np.ones(n, dtype=bool)
    ids = np.asarray([f'unit_{i}' for i in range(n)])
    fit, valid = np.arange(15), np.arange(15, n)
    bank = LocalResponseBank.fit(x, chem, mask, ids, ids[fit],
        metadata={'fingerprint_indices': list(range(32))}, max_anchors=5)
    base = RidgeResidualMean(d, torch.zeros(d, 9, dtype=torch.float64),
        torch.zeros(9, dtype=torch.float64), hidden_dim=8).double()
    torch.manual_seed(739)
    model = ConditionalResponseKernelMean(base, bank, mode='conditional_structured', hidden_dim=8)
    target = rng.normal(0, .15, (n, 9))
    gamma = gamma_from_raw_coordinates(torch.tensor(target)).numpy()
    objective = JointGammaCRPS(.04*np.eye(9), np.zeros(9), np.ones(9), np.std(gamma[fit], ddof=1))
    initial_model = deepcopy(model.state_dict())
    initial_objective = deepcopy(objective.state_dict())
    config = dict(experiment.CONFIG, batch_size=8, train_pairs=8, validation_pairs=8)
    monkeypatch.setattr(experiment, 'CONFIG', config)
    folder = tmp_path/'trained'
    result = experiment.train_supervised(folder, model, objective, x, chem, mask,
        target, gamma, fit, valid, 939, ids)
    saved = torch.load(folder/'epoch30.pt', map_location='cpu', weights_only=True)
    assert saved['epoch'] == 30 and saved['optimizer_steps'] == 60
    assert 'training_mc_rng_state' in saved and saved['gamma_weight'] == 1
    assert saved['fit_ids'] == ids[fit].tolist() and saved['validation_ids'] == ids[valid].tolist()
    for key, value in initial_model.items():
        if key.startswith(('base_hr.', 'bank.')):
            assert torch.equal(result.state_dict()[key], value)
    for key, value in initial_objective.items():
        assert torch.equal(objective.state_dict()[key], value)
    assert not torch.equal(result.state_dict()['output.weight'], initial_model['output.weight'])
    restored = ConditionalResponseKernelMean.from_config(saved['model_config'])
    restored.load_state_dict(saved['state_dict'])
    assert np.array_equal(experiment.predict_branch(restored, x, chem, mask),
                          experiment.predict_branch(result, x, chem, mask))
    rows = experiment.monitor_checkpoints(folder, objective, x, chem, mask, target,
        gamma, fit, valid, 939)
    assert [row['epoch'] for row in rows] == [0,5,10,15,20,25,30]
    assert all(np.isfinite(row['validation_gamma_crps']) for row in rows)
    assert all(row['normalized_gamma_gradient_norm'] > 0 for row in rows)
    assert all(-1 <= row['gradient_cosine'] <= 1 for row in rows)
    repeat = experiment.monitor_checkpoints(folder, objective, x, chem, mask, target,
        gamma, fit, valid, 939)
    assert repeat == rows
    complete = json.loads((folder/'training_complete.json').read_text())
    assert complete['final_epoch'] == 30 and complete['objective_buffers_changed'] is False
    with pytest.raises(FileExistsError):
        experiment.train_supervised(folder, model, objective, x, chem, mask,
            target, gamma, fit, valid, 939, ids)
