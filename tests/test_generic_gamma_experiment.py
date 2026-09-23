"""The new arm changes chemical response functions, not initialization or loss."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from opal2.conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from opal2.generic_gamma_experiment import check_initialization
from opal2.hierarchical_geometry import RidgeResidualMean


def test_generic_gamma_matches_initialization_without_erasing_basis_difference():
    rng = np.random.default_rng(873)
    x = rng.normal(size=(15, 6))
    chem = (rng.random((15, 32)) < .3).astype(float)
    chem[:,0] = 1
    ids = np.asarray([f'u{i}' for i in range(15)])
    mask = np.ones(15, bool)
    bank = LocalResponseBank.fit(x, chem, mask, ids, ids[:12],
        metadata={'fingerprint_indices':list(range(32))}, max_anchors=5)
    base = RidgeResidualMean(6, torch.zeros(6,9,dtype=torch.float64),
        torch.zeros(9,dtype=torch.float64), hidden_dim=8).double()
    models = {}
    for mode in ('conditional_generic','conditional_structured'):
        torch.manual_seed(733)
        models[mode] = ConditionalResponseKernelMean(base, bank, mode=mode, hidden_dim=8).double()
    generic, structured = models.values()
    old_g = dict(model_config=deepcopy(generic.config), state_dict=deepcopy(generic.state_dict()))
    old_s = dict(model_config=deepcopy(structured.config), state_dict=deepcopy(structured.state_dict()))
    result = check_initialization(generic, old_g, old_s)
    assert result['exact_historical_generic_state']
    args = (torch.tensor(x), torch.tensor(chem), torch.tensor(mask))
    assert torch.equal(generic(*args), structured(*args))
    description = bank(*args)
    assert not torch.equal(bank.basis_blocks(description,'generic')['chemical'],
        bank.basis_blocks(description,'structured')['chemical'])
    assert torch.equal(bank.basis_blocks(description,'generic')['morphology'],
        bank.basis_blocks(description,'structured')['morphology'])
    old_s['state_dict']['output.weight'][0,0] += 1
    with pytest.raises(ValueError, match='learnable initialization'):
        check_initialization(generic, old_g, old_s)
