"""Changed training-reference and data-boundary checks, not assay evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2.gamma_supervised_loss import JointGammaCRPS, gamma_from_raw_coordinates
from opal2.independent_biology_kernel import IndependentBiologyKernelMean
from opal2.independent_biology_training import train_branch
from test_independent_biology_kernel import fixture
from test_kernel_gamma_continuation_training import exact


def case():
    model, base, x, _, mask, _, packed, ids = fixture()
    with torch.no_grad():
        target = base(x, packed, mask) + torch.linspace(-.1, .3, len(x))[:, None]
    gamma = gamma_from_raw_coordinates(target)
    objective = JointGammaCRPS(torch.eye(9, dtype=torch.float64)*.1,
        torch.zeros(9, dtype=torch.float64), torch.ones(9, dtype=torch.float64), .1)
    config = dict(stage_epochs=3, max_epochs=100, learning_rate=.0003,
        min_learning_rate=.000003, warmup_steps=2, weight_decay=.0001,
        batch_size=5, validation_interval=1, gradient_clip=5., gamma_weight=1.,
        incremental_penalty=.1, train_pairs=4, validation_pairs=4,
        training_mc_offset=53000, validation_mc_offset=51000)
    return model, x, packed, mask, target, gamma, objective, config, ids


def test_full_frozen_A_reference_and_validation_isolation(tmp_path):
    model, x, packed, mask, target, gamma, objective, config, ids = case()
    fit, valid = np.arange(10), np.arange(10, 14)
    saved = []
    for index in range(2):
        y, g = target.clone(), gamma.clone()
        if index:
            y[valid] += 3.; g[valid] -= .25
        result = train_branch(tmp_path/str(index), deepcopy(model), deepcopy(objective),
            x, packed, mask, y, g, fit, valid, 781, ids, config)
        payload = torch.load(tmp_path/str(index)/'epoch3.pt', weights_only=True)
        restored = IndependentBiologyKernelMean.from_config(payload['model_config'])
        restored.load_state_dict(payload['state_dict']); restored.eval()
        assert torch.equal(restored(x, packed, mask), result(x, packed, mask))
        assert torch.equal(result(x, packed, mask, branch_enabled=False), model.baseline_mean(x, packed, mask))
        completion = json.loads((tmp_path/str(index)/'training_complete.json').read_text())
        assert completion['actual_checkpoint_epoch'] == 3
        assert not completion['frozen_A_changed'] and completion['disabled_equals_A']
        assert completion['reference_A_fit_mse'] == pytest.approx(
            float((model.baseline_mean(x, packed, mask)[fit]-target[fit]).square().mean()))
        saved.append(payload)
    for key in ('state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'torch_rng_state',
                'order_rng_state', 'training_mc_rng_state', 'gamma_objective_state_dict'):
        exact(saved[0][key], saved[1][key])
    with pytest.raises(FileExistsError):
        train_branch(tmp_path/'0', model, objective, x, packed, mask, target, gamma,
                     fit, valid, 781, ids, config)


def test_no_test_rows_or_nonidentity_initialization_accepted(tmp_path):
    model, x, packed, mask, target, gamma, objective, config, ids = case()
    with pytest.raises(ValueError, match='Only explicitly supplied'):
        train_branch(tmp_path/'bad_membership', model, objective, x, packed, mask,
                     target, gamma, np.arange(9), np.arange(10, 14), 781, ids, config)
    with torch.no_grad():
        model.channels[0].output.weight.fill_(.2)
    with pytest.raises(ValueError, match='complete frozen A'):
        train_branch(tmp_path/'bad_initialization', model, objective, x, packed, mask,
                     target, gamma, np.arange(10), np.arange(10, 14), 781, ids, config)
