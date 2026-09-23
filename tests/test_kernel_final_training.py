"""Synthetic exact-reference and full-fit primal/dual regression checks."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

import opal2.gamma_supervised_experiment as original
from opal2.kernel_final_training import train_branch,constrained_objective,update_dual
from test_kernel_gamma_continuation_training import synthetic,exact


def case():
    model,x,chem,mask,target,gamma,fit,valid,ids,objective,cfg=synthetic('K_GEOMETRY_GAMMA_CRPS')
    cfg=dict(cfg,stage_epochs=3,validation_interval=1,validation_pairs=4,dual_penalty=1.,dual_step=1.)
    return model,x,chem,mask,target,gamma,fit,valid,ids,objective,cfg


def invoke(folder,values,mode,**overrides):
    model,x,chem,mask,target,gamma,fit,valid,ids,objective,cfg=values
    return train_branch(folder,deepcopy(model),deepcopy(objective),x,chem,mask,
        overrides.get('target',target),overrides.get('gamma',gamma),fit,valid,918,ids,cfg,mode)


def test_weighted_reproduces_original_parameters_optimizer_schedule_and_random_streams(tmp_path,monkeypatch):
    values=case();model,x,chem,mask,target,gamma,fit,valid,ids,objective,cfg=values
    monkeypatch.setattr(original,'CONFIG',cfg)
    original.train_supervised(tmp_path/'old',deepcopy(model),deepcopy(objective),
        x,chem,mask,target,gamma,fit,valid,918,ids)
    invoke(tmp_path/'new',values,'weighted')
    a=torch.load(tmp_path/'old/epoch3.pt',weights_only=True)
    b=torch.load(tmp_path/'new/epoch3.pt',weights_only=True)
    for key in a:
        exact(a[key],b[key])
    aa=[json.loads(x) for x in (tmp_path/'old/history.jsonl').read_text().splitlines()]
    bb=[json.loads(x) for x in (tmp_path/'new/history.jsonl').read_text().splitlines()]
    for left,right in zip(aa,bb):
        for key in left.keys()-{'elapsed_seconds'}:exact(left[key],right[key])
    assert b['dual_state']['value']==0


def test_full_fit_constraint_gradient_matches_finite_difference_and_includes_nonbatch_rows():
    mean=torch.tensor([[.1,.2],[.4,-.2],[.8,.3]],dtype=torch.float64,requires_grad=True)
    target=torch.zeros_like(mean);reference=.05
    def fn(x):
        full=(x-target).square().mean()
        return constrained_objective(x[0].sum()*.01,x[0].square().mean(),full,reference,.3)['loss']
    assert torch.autograd.gradcheck(fn,(mean,),eps=1e-6,atol=1e-6,rtol=1e-5)
    gradient=torch.autograd.grad(fn(mean),mean)[0]
    assert gradient[1:].abs().min()>0  # Full-fit rows outside the Gamma minibatch influence the primal step.
    result=constrained_objective(torch.tensor(.4),torch.tensor(.2),torch.tensor(.11),.1,.3)
    expected=.4+.1*.2+.3*.1+.5*.1**2
    assert float(result['loss'])==pytest.approx(expected)
    assert update_dual(.2,.1)==pytest.approx(.3)
    assert update_dual(.2,-.4)==0
    with pytest.raises(ValueError):update_dual(-.1,.1)


def test_constrained_records_actual_fullfit_dual_and_preserves_frozen_state(tmp_path):
    values=case();model,x,chem,mask,target,gamma,fit,valid,ids,objective,cfg=values
    before={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('base_hr.','bank.'))}
    result=invoke(tmp_path/'fit',values,'constrained')
    state=result.state_dict()
    for key,value in before.items():assert torch.equal(state[key],value)
    assert all(p.grad is None for p in result.base_hr.parameters()) and not result.base_hr.training
    history=[json.loads(x) for x in (tmp_path/'fit/history.jsonl').read_text().splitlines()]
    dual=0.
    for row in history:
        payload=torch.load(tmp_path/'fit'/f"epoch{row['epoch']}.pt",weights_only=True)
        recovered=type(model).from_config(payload['model_config']);recovered.load_state_dict(payload['state_dict'])
        with torch.no_grad():
            xx=torch.as_tensor(x[fit]);cc=torch.as_tensor(chem[fit]);mm=torch.as_tensor(mask[fit])
            actual_mse=(recovered(xx,cc,mm)-torch.as_tensor(target[fit])).square().mean().item()
        g=actual_mse/row['reference_hr_fit_mse']-1
        assert row['fit_constraint_g']==pytest.approx(g,abs=1e-14)
        assert row['dual_before']==dual
        if row['epoch']:dual=update_dual(dual,g,cfg['dual_step'])
        assert row['dual_after']==pytest.approx(dual,abs=1e-14)
        assert payload['dual_state']['value']==row['dual_after']
        exact(payload['gamma_objective_state_dict'],objective.state_dict())
    done=json.loads((tmp_path/'fit/training_complete.json').read_text())
    assert done['epoch']==done['actual_checkpoint_epoch']==done['final_epoch']==3
    assert done['optimizer_steps']==6 and not done['hidden_fallback'] and not done['feasibility_guarantee']
    assert done['geometry_feasibility']['fullfit_final_mse']==history[-1]['fit_u_mse']
    assert len((tmp_path/'fit/gamma_monitoring.jsonl').read_text().splitlines())==4


@pytest.mark.parametrize('mode',['weighted','constrained'])
def test_validation_targets_do_not_change_training_and_overwrite_rejected(mode,tmp_path):
    values=case();_,_,_,_,target,gamma,fit,valid,_,_,_=values
    invoke(tmp_path/'a',values,mode)
    new_target=target.copy();new_target[valid]+=4
    new_gamma=gamma.copy();new_gamma[valid]-=.5
    invoke(tmp_path/'b',values,mode,target=new_target,gamma=new_gamma)
    a=torch.load(tmp_path/'a/epoch3.pt',weights_only=True);b=torch.load(tmp_path/'b/epoch3.pt',weights_only=True)
    for key in ('state_dict','optimizer_state_dict','scheduler_state_dict','torch_rng_state',
                'order_rng_state','training_mc_rng_state','dual_state'):
        exact(a[key],b[key])
    with pytest.raises(FileExistsError):invoke(tmp_path/'a',values,mode)
