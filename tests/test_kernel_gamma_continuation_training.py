"""Synthetic interrupted/uninterrupted equivalence, not cohort training."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

import opal2.conditional_response_experiment as geometry_source
import opal2.gamma_supervised_experiment as gamma_source
from opal2.conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from opal2.gamma_supervised_loss import JointGammaCRPS, gamma_from_raw_coordinates
from opal2.hierarchical_geometry import RidgeResidualMean
from opal2.kernel_gamma_continuation_training import continue_branch, ARM_MODES


def synthetic(arm):
    torch.manual_seed(801)
    rng=np.random.default_rng(401)
    x=rng.normal(size=(12,5))
    bits=np.array([[(i>>j)&1 for j in range(5)] for i in range(1,13)],float)
    chem=np.column_stack((bits,np.ones(12)));mask=np.ones(12,bool)
    ids=np.asarray([f'synthetic_{i}' for i in range(12)])
    fit,valid=np.arange(8),np.arange(8,12)
    bank=LocalResponseBank.fit(x,chem,mask,ids,ids[fit],
        dict(fingerprint_indices=list(range(5)),validity_index=5,kind='synthetic'),max_anchors=3)
    hr=RidgeResidualMean(5,rng.normal(size=(5,9))*.04,np.zeros(9),dropout=.4)
    with torch.no_grad(): hr.network[-1].weight.normal_(0,.05)
    model=ConditionalResponseKernelMean(hr,bank,mode=ARM_MODES[arm][0])
    target=rng.normal(size=(12,9))*.18
    actual=gamma_from_raw_coordinates(torch.as_tensor(target)).numpy()
    covariance=torch.eye(9,dtype=torch.float64)*.02
    covariance[0,1]=covariance[1,0]=.003
    objective=JointGammaCRPS(covariance,np.zeros(9),np.ones(9),.08)
    cfg=dict(gamma_source.CONFIG,stage_epochs=4,validation_interval=2,batch_size=4,
             warmup_steps=2,train_pairs=4)
    return model,x,chem,mask,target,actual,fit,valid,ids,objective,cfg


def exact(left,right):
    if isinstance(left,torch.Tensor): assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left: exact(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right): exact(a,b)
    else: assert left==right


def source_train(folder,values,arm,cfg,monkeypatch):
    model,x,chem,mask,target,actual,fit,valid,ids,objective,_=values
    if ARM_MODES[arm][1]:
        monkeypatch.setattr(gamma_source,'CONFIG',cfg)
        result=gamma_source.train_supervised(folder,deepcopy(model),deepcopy(objective),
            x,chem,mask,target,actual,fit,valid,918,ids)
    else:
        monkeypatch.setattr(geometry_source,'CONFIG',cfg)
        result=geometry_source.train_branch(folder,deepcopy(model),x,chem,mask,target,fit,valid,918,ids)
    # The unchanged production source labels its completion as epoch30. This
    # temporary engineering fixture records the actual shorter loop endpoint;
    # no checkpoint, parameter, optimizer, RNG or training history is altered.
    metadata=json.loads((folder/'training_complete.json').read_text())
    metadata.update(epoch=cfg['stage_epochs'],actual_checkpoint_epoch=cfg['stage_epochs'],
                    final_epoch=cfg['stage_epochs'],scope='synthetic continuation regression')
    (folder/'training_complete.json').write_text(json.dumps(metadata))
    return result


@pytest.mark.parametrize('arm',list(ARM_MODES))
def test_all_four_arms_resume_bitwise_model_optimizer_schedule_rng_and_losses(arm,tmp_path,monkeypatch):
    values=synthetic(arm)
    model,x,chem,mask,target,actual,fit,valid,ids,objective,cfg=values
    frozen={key:value.clone() for key,value in model.state_dict().items() if key.startswith(('base_hr.','bank.'))}
    full=source_train(tmp_path/'full',values,arm,cfg,monkeypatch)
    source=tmp_path/'source'
    source_train(source,values,arm,dict(cfg,stage_epochs=2),monkeypatch)
    source_bytes={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    torch.manual_seed(322);torch.randn(39)
    resumed=continue_branch(tmp_path/'resumed',source,x,chem,mask,target,actual,
        fit,valid,ids,cfg,arm=arm,objective=objective,end_epoch=4)
    a=torch.load(tmp_path/'full/epoch4.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/epoch4.pt',weights_only=True)
    exact(a,b)
    assert source_bytes=={p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    for key,value in frozen.items(): assert torch.equal(resumed.state_dict()[key],value)
    assert not resumed.base_hr.training and all(p.grad is None for p in resumed.base_hr.parameters())
    with torch.no_grad():
        args=[torch.as_tensor(v) for v in (x,chem,mask)]
        assert torch.equal(resumed(*args),full(*args))
    completion=json.loads((tmp_path/'resumed/training_complete.json').read_text())
    assert completion['final_epoch']==completion['actual_checkpoint_epoch']==4
    assert completion['optimizer_steps']==8
    assert completion['source_optimizer_steps']==completion['additional_optimizer_steps']==4
    audit=completion['resume_audit']
    assert not audit['warmup_restarted'] and not audit['seed_reset']
    assert audit['training_mc_rng_restored']==ARM_MODES[arm][1]
    assert audit['scheduler_horizon_epochs']==100
    history=[json.loads(s) for s in (tmp_path/'resumed/history.jsonl').read_text().splitlines()]
    original=[json.loads(s) for s in (tmp_path/'full/history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history]==list(range(5))
    for got,wanted in zip(history,original):
        for key in wanted.keys()-{'elapsed_seconds'}: exact(got[key],wanted[key])


def test_identity_objective_and_overwrite_rejected_before_output(tmp_path,monkeypatch):
    arm='K_GEOMETRY_GAMMA_CRPS';values=synthetic(arm)
    _,x,chem,mask,target,actual,fit,valid,ids,objective,cfg=values
    source=tmp_path/'source';source_train(source,values,arm,dict(cfg,stage_epochs=2),monkeypatch)
    wrong_ids=ids.copy();wrong_ids[0]='not_the_original_object'
    destination=tmp_path/'output'
    with pytest.raises(ValueError,match='IDs differ'):
        continue_branch(destination,source,x,chem,mask,target,actual,fit,valid,wrong_ids,cfg,
                        arm=arm,objective=objective,end_epoch=4)
    assert not destination.exists()
    changed=deepcopy(objective);changed.gamma_scale.mul_(2)
    with pytest.raises(ValueError,match='objective differs'):
        continue_branch(destination,source,x,chem,mask,target,actual,fit,valid,ids,cfg,
                        arm=arm,objective=changed,end_epoch=4)
    assert not destination.exists()
    destination.mkdir();marker=destination/'keep.txt';marker.write_text('existing evidence')
    with pytest.raises(FileExistsError):
        continue_branch(destination,source,x,chem,mask,target,actual,fit,valid,ids,cfg,
                        arm=arm,objective=objective,end_epoch=4)
    assert marker.read_text()=='existing evidence'
