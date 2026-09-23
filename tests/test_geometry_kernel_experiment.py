"""Critical connection tests; these synthetic cases are not research results."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2.geometry_kernel import DescriptorBank, GeometryKernelMean
from opal2.hierarchical_geometry import RidgeResidualMean
from opal2 import geometry_kernel_experiment as experiment


def fixture_model(mode):
    rng = np.random.default_rng(123)
    n, d = 20, 6
    x = rng.normal(size=(n,d))
    chem = (rng.random((n,32))<.25).astype(float)
    chem[:,0] = 1.
    mask = np.ones(n,dtype=bool)
    ids = np.array([f'unit_{i}' for i in range(n)])
    fit, valid = np.arange(15),np.arange(15,n)
    bank = DescriptorBank.fit(x,chem,mask,ids,ids[fit],
        metadata={'fingerprint_indices':list(range(32))},max_anchors=5)
    torch.manual_seed(12)
    base = RidgeResidualMean(d,torch.zeros(d,9,dtype=torch.float64),
        torch.zeros(9,dtype=torch.float64),hidden_dim=8).double()
    with torch.no_grad():
        base.network[-1].weight.fill_(.01)
    model = GeometryKernelMean(base,bank,mode=mode,hidden_dim=8,kan_hidden_dim=5)
    target = np.tile(.1+.08*np.tanh(x[:,0,None]),(1,9))
    return model,x,chem,mask,target,fit,valid,ids


@pytest.mark.parametrize('mode',['generic','structured'])
def test_stage_trains_new_branch_preserves_hr_and_reloads_actual_epoch(tmp_path,monkeypatch,mode):
    model,x,chem,mask,target,fit,valid,ids = fixture_model(mode)
    cfg = deepcopy(experiment.CONFIG)
    cfg.update(stage_epochs=2,validation_interval=1,batch_size=8,max_epochs=10,warmup_steps=1)
    monkeypatch.setattr(experiment,'CONFIG',cfg)
    base_before = deepcopy(model.base_hr.state_dict())
    folder = tmp_path/mode
    result = experiment.train_branch(folder,model,x,chem,mask,target,fit,valid,321,ids)
    for name, tensor in result.base_hr.state_dict().items():
        assert torch.equal(tensor,base_before[name])
    assert all(p.grad is None for p in result.base_hr.parameters())
    assert not result.base_hr.training
    checkpoint = torch.load(folder/'epoch2.pt',weights_only=True)
    assert checkpoint['epoch']==2
    assert checkpoint['fit_ids']==ids[fit].tolist()
    assert checkpoint['validation_ids']==ids[valid].tolist()
    assert 'optimizer_state_dict' in checkpoint and 'order_rng_state' in checkpoint
    restored = GeometryKernelMean.from_config(checkpoint['model_config'])
    restored.load_state_dict(checkpoint['state_dict'])
    assert np.array_equal(experiment.predict_branch(result,x,chem,mask),
                          experiment.predict_branch(restored,x,chem,mask))
    assert result.output.weight.abs().sum()>0
    history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    assert [line['epoch'] for line in history]==[0,1,2]
    assert history[-1]['gradient_norm_mean']>0
    completion = json.loads((folder/'training_complete.json').read_text())
    assert completion['actual_checkpoint_epoch']==2
    with pytest.raises(FileExistsError):
        experiment.train_branch(folder,model,x,chem,mask,target,fit,valid,321,ids)


def test_support_reports_train_self_exclusion_without_changing_predictions():
    model,x,chem,mask,target,fit,valid,ids = fixture_model('structured')
    dfit = experiment.support_diagnostics(model.bank,x[fit],chem[fit],mask[fit],ids[fit])
    dvalid = experiment.support_diagnostics(model.bank,x[valid],chem[valid],mask[valid],ids[valid])
    assert dfit['support_self_excluded']['mean']==1.
    assert dvalid['support_self_excluded']['mean']==0.
    assert dfit['availability']['mean']==1.
