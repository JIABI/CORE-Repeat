"""Synthetic integration only: no LINCS profiles or experimental conclusions."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2 import lincs_biology_experiment as experiment
from opal2.baseline_policy import ACTIONS
from opal2.gamma_supervised_loss import JointGammaCRPS, gamma_from_raw_coordinates
from opal2.kernel_final_training import train_branch
from opal2.mechanism_response_kernel import MechanismResponseKernelMean
from test_mechanism_response_kernel import fixture


def test_connectivity_groups_are_wholly_disjoint_and_tested_exactly_once():
    # Distinct sample/salt identifiers share each explicit InChI14 group.
    groups=np.repeat(np.asarray([f'KEY{i:011d}' for i in range(25)]),3)
    assert all(len(v)==14 for v in groups)
    ids=np.asarray([f'{group}-SALT{salt}' for group in np.unique(groups) for salt in range(3)])
    records=experiment.grouped_folds(ids,groups,20260915)
    assert len(records)==5
    tested=np.zeros(len(ids),int);group_test={}
    for record in records:
        parts=[np.asarray(record[k],int) for k in ('fit','inner_validation','test')]
        assert len(np.unique(np.concatenate(parts)))==len(ids)
        assert sum(map(len,parts))==len(ids)
        for name,rows in zip(('fit','inner_validation','test'),parts):
            assert record[name+'_ids']==ids[rows].tolist()
        sets=[set(groups[p]) for p in parts]
        assert all(not sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2)))
        tested[parts[2]]+=1
        for group in sets[2]:
            assert group not in group_test
            group_test[group]=record['fold']
            assert set(np.flatnonzero(groups==group)).issubset(parts[2])
    assert np.array_equal(tested,np.ones(len(ids),int)) and len(group_test)==25
    assert records==experiment.grouped_folds(ids,groups,20260915)
    permutation=np.random.default_rng(33).permutation(len(ids))
    permuted=experiment.grouped_folds(ids[permutation],groups[permutation],20260915)
    for a,b in zip(records,permuted):
        for name in ('fit','inner_validation','test'):
            assert set(a[name+'_ids'])==set(b[name+'_ids'])


@pytest.mark.parametrize('mode',list(experiment.ARMS.values()))
def test_full_trainer_accepts_packed_mechanism_subclass_and_restores_actual_checkpoint(mode,tmp_path):
    # Short engineering fixture only; production remains 30 epochs/64 anchors.
    model,_,bank,_,x,chem,mask,bio,ids=fixture(mode)
    packed=bank.pack_information(chem,bio)
    fit=np.arange(10);valid=np.arange(10,14)
    generator=torch.Generator().manual_seed(918)
    target=torch.randn((len(x),9),generator=generator,dtype=torch.float64)*.1
    covariance=torch.eye(9,dtype=torch.float64)*.02
    covariance[0,1]=covariance[1,0]=.004
    objective=JointGammaCRPS(covariance,torch.zeros(9,dtype=torch.float64),
                            torch.full((9,),.2,dtype=torch.float64),.1)
    gamma=gamma_from_raw_coordinates(target*.2)
    config=dict(experiment.CONFIG,stage_epochs=2,validation_interval=1,
                batch_size=4,train_pairs=4,validation_pairs=4)
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('base_hr.','bank.'))}
    objective_before=deepcopy(objective.state_dict())
    train_branch(tmp_path/'run',model,objective,x,packed,mask,target,gamma,
                 fit,valid,782,ids,config,'weighted')
    done=json.loads((tmp_path/'run/training_complete.json').read_text())
    assert done['epoch']==done['final_epoch']==done['actual_checkpoint_epoch']==2
    assert done['optimizer_steps']==6 and not done['hidden_fallback']
    assert not done['frozen_hr_changed'] and not done['bank_changed'] and not done['objective_buffers_changed']
    payload=torch.load(tmp_path/'run/epoch2.pt',weights_only=True)
    assert payload['fit_ids']==ids[fit].tolist() and payload['validation_ids']==ids[valid].tolist()
    assert payload['model_config']['model_type']=='mechanism_response'
    restored=MechanismResponseKernelMean.from_config(payload['model_config'])
    restored.load_state_dict(payload['state_dict']);restored.eval()
    assert torch.equal(restored(x,packed,mask),model(x,packed,mask))
    for key,value in frozen.items():assert torch.equal(model.state_dict()[key],value)
    for key,value in objective_before.items():assert torch.equal(objective.state_dict()[key],value)
    assert torch.equal(model(x,packed,mask),model(x,chem,mask,bio=bio))


def test_validation_readout_does_not_change_biological_branch_training(tmp_path):
    model,_,bank,_,x,chem,mask,bio,ids=fixture('bio_structured')
    packed=bank.pack_information(chem,bio)
    fit=np.arange(10);valid=np.arange(10,14)
    target=torch.linspace(-.1,.15,len(x)*9,dtype=torch.float64).reshape(len(x),9)
    gamma=gamma_from_raw_coordinates(target*.2)
    objective=JointGammaCRPS(torch.eye(9,dtype=torch.float64)*.02,
        torch.zeros(9,dtype=torch.float64),torch.full((9,),.2,dtype=torch.float64),.1)
    config=dict(experiment.CONFIG,stage_epochs=2,validation_interval=1,
                batch_size=4,train_pairs=4,validation_pairs=4)
    altered_target=target.clone();altered_target[valid]+=2
    altered_gamma=gamma.clone();altered_gamma[valid]-=.25
    for name,t,g in [('a',target,gamma),('b',altered_target,altered_gamma)]:
        train_branch(tmp_path/name,deepcopy(model),deepcopy(objective),x,packed,mask,t,g,
                     fit,valid,782,ids,config,'weighted')
    a=torch.load(tmp_path/'a/epoch2.pt',weights_only=True)
    b=torch.load(tmp_path/'b/epoch2.pt',weights_only=True)
    for key in a['state_dict']:assert torch.equal(a['state_dict'][key],b['state_dict'][key])
    for key in ('torch_rng_state','order_rng_state','training_mc_rng_state'):
        assert torch.equal(a[key],b[key])


@pytest.mark.parametrize('arm',['HR','D_BIO_STRUCTURED'])
def test_scoring_uses_full_joint_10000_draws_same_seed_original_actions(arm,monkeypatch,tmp_path):
    mean=np.zeros((2,9));covariance=np.eye(9)
    ridge=type('Ridge',(),{'covariance':covariance})()
    target=np.zeros((2,9));grams=np.broadcast_to(np.eye(4),(2,4,4))
    ids=np.asarray(['a','b']);train_gains=np.zeros((5,3));scale=np.ones(9);stats={}
    samples,raw,draws=object(),object(),object()
    def sample(m,c,n,seed):
        assert m is mean and c is covariance and n==10000 and seed==200123
        return samples
    monkeypatch.setattr(experiment,'sample_joint_coordinates',sample)
    monkeypatch.setattr(experiment,'gaussian_coordinate_diagnostics',lambda *args: {'audit':'coordinates'})
    def restore(s,st):
        assert s is samples and st is stats
        return raw
    monkeypatch.setattr(experiment,'restore_target',restore)
    def decode(s,*,verify):
        assert s is raw and verify
        return draws,{'audit':'numerical'}
    monkeypatch.setattr(experiment,'decode_draws',decode)
    def evaluate(folder,d,g,names,**kwargs):
        assert folder==tmp_path and d is draws and g is grams and names is ids
        assert ACTIONS==('Z1','Z2','Z1Z2') and kwargs['train_actual_gains'] is train_gains
        assert kwargs['score_scale'] is scale
        return kwargs
    monkeypatch.setattr(experiment,'evaluate_and_save',evaluate)
    result=experiment.score(tmp_path,mean,ridge,stats,target,grams,ids,train_gains,scale,123,arm)
    assert result['seed']==123 and result['n_bootstrap']==result['n_random']==2000
    assert result['metadata']['formal_certificate'] is False
    assert result['metadata']['biological_information']==(arm=='D_BIO_STRUCTURED')
    assert result['metadata']['actual_checkpoint_epoch']==(30 if arm!='HR' else 'fresh validation-selected HR')
    assert experiment.CONFIG['stage_epochs']==30 and experiment.CONFIG['max_epochs']==100
    assert experiment.CONFIG['hidden_dim']==16 and experiment.CONFIG['train_pairs']==64
    assert experiment.CONFIG['validation_pairs']==128
