"""Synthetic continuation equivalence; no real cohort is loaded or fitted."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2.geometry_kernel_replacement import ReplacementBasisBank,GeometryKernelReplacementMean
from opal2.geometry_kernel_continuation_training import continue_branch
from opal2.hierarchical_geometry import RidgeResidualMean
import opal2.geometry_kernel_replacement_experiment as old


def synthetic(mode):
    torch.manual_seed(74)
    rng=np.random.default_rng(17)
    x=rng.normal(size=(12,5))
    bits=np.asarray([[(i>>j)&1 for j in range(5)] for i in range(1,13)],float)
    chem=np.column_stack((bits,np.ones(12)))
    mask=np.ones(12,bool);ids=np.asarray([f'synthetic_{i}' for i in range(12)])
    fit,valid=np.arange(8),np.arange(8,12)
    metadata=dict(fingerprint_indices=list(range(5)),validity_index=5,kind='synthetic binary')
    bank=ReplacementBasisBank.fit(x,chem,mask,ids,ids[fit],metadata,max_anchors=3)
    hr=RidgeResidualMean(5,rng.normal(size=(5,9))*.03,np.zeros(9),dropout=.4)
    with torch.no_grad():
        hr.network[-1].weight.normal_(0,.05)
    model=GeometryKernelReplacementMean(hr,bank,mode=mode)
    target=rng.normal(size=(12,9))*.1
    cfg=dict(old.CONFIG,stage_epochs=4,validation_interval=2,batch_size=4,warmup_steps=2)
    return model,x,chem,mask,target,fit,valid,ids,cfg


def assert_exact(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:
            assert_exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b):
            assert_exact(left,right)
    else:
        assert a==b


@pytest.mark.parametrize('mode',['mlp','generic','structured'])
def test_resumed_and_uninterrupted_parameters_optimizers_schedules_and_rng_are_bitwise_equal(mode,tmp_path,monkeypatch):
    model,x,chem,mask,target,fit,valid,ids,cfg=synthetic(mode)
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('base_hr.','bank.'))}
    monkeypatch.setattr(old,'CONFIG',dict(cfg))
    full=old.train_branch(tmp_path/'full',deepcopy(model),x,chem,mask,target,fit,valid,313,ids)
    full_payload=torch.load(tmp_path/'full/epoch4.pt',weights_only=True)
    monkeypatch.setattr(old,'CONFIG',dict(cfg,stage_epochs=2))
    source=tmp_path/'source'
    old.train_branch(source,deepcopy(model),x,chem,mask,target,fit,valid,313,ids)
    before={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    torch.manual_seed(919);torch.randn(37)  # Unrelated runtime activity must not change continuation.
    resumed=continue_branch(tmp_path/'resumed',source,x,chem,mask,target,fit,valid,ids,cfg)
    resumed_payload=torch.load(tmp_path/'resumed/epoch4.pt',weights_only=True)
    for key in ('model_config','state_dict','optimizer_state_dict','scheduler_state_dict',
                'torch_rng_state','order_rng_state','validation_u_mse','optimizer_steps'):
        assert_exact(full_payload[key],resumed_payload[key])
    for key,value in frozen.items():
        assert torch.equal(value,resumed.state_dict()[key])
    assert all(p.grad is None for p in resumed.base_hr.parameters())
    assert not resumed.base_hr.training
    assert before=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    expected=full(torch.tensor(x),torch.tensor(chem),torch.tensor(mask))
    assert torch.equal(resumed(torch.tensor(x),torch.tensor(chem),torch.tensor(mask)),expected)
    complete=json.loads((tmp_path/'resumed/training_complete.json').read_text())
    assert complete['optimizer_steps']==8
    assert complete['resumed_from_optimizer_steps']==4
    assert complete['continuation_optimizer_steps']==4
    assert not complete['resume_audit']['warmup_restarted']
    assert not complete['fitted_bank_changed']
    history=[json.loads(line) for line in (tmp_path/'resumed/history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history]==list(range(5))
    full_history=[json.loads(line) for line in (tmp_path/'full/history.jsonl').read_text().splitlines()]
    for observed,expected in zip(history,full_history):
        for key in expected.keys()-{'elapsed_seconds'}:
            assert_exact(observed[key],expected[key])


def test_changed_ids_or_existing_output_are_rejected_before_writing(tmp_path,monkeypatch):
    model,x,chem,mask,target,fit,valid,ids,cfg=synthetic('mlp')
    monkeypatch.setattr(old,'CONFIG',dict(cfg,stage_epochs=2))
    source=tmp_path/'source'
    old.train_branch(source,model,x,chem,mask,target,fit,valid,313,ids)
    wrong=ids.copy();wrong[0]='different_identity'
    destination=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='IDs differ'):
        continue_branch(destination,source,x,chem,mask,target,fit,valid,wrong,cfg)
    assert not destination.exists()
    destination.mkdir();marker=destination/'keep.txt';marker.write_text('existing evidence')
    with pytest.raises(FileExistsError):
        continue_branch(destination,source,x,chem,mask,target,fit,valid,ids,cfg)
    assert marker.read_text()=='existing evidence'
