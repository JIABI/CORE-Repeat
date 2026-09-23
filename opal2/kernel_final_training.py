"""Matched fixed-epoch weighted and full-fit constrained kernel training.

The constrained primal step uses an exact differentiable FULL-FIT geometry
constraint, not the minibatch estimate. Gamma CRPS and the increment penalty
remain minibatch objectives. The nonnegative dual is updated once per epoch.
Finite-time feasibility is measured, never presumed or enforced by fallback.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .biology_kernel_evaluation import plain, write_json
from .conditional_response_kernel import ConditionalResponseKernelMean
from .gamma_supervised_loss import JointGammaCRPS
from .gamma_supervised_experiment import independent_noise
from .geometry_kernel_replacement_experiment import checkpoint_payload
from .gram_experiment import _schedule


def constrained_objective(normalized_gamma_crps, incremental_mse, full_fit_mse,
                          reference_mse, dual, *, incremental_penalty=.1, rho=1.):
    """Primal augmented loss; reference/dual must not carry a gradient."""
    reference=torch.as_tensor(reference_mse,dtype=full_fit_mse.dtype,device=full_fit_mse.device)
    multiplier=torch.as_tensor(dual,dtype=full_fit_mse.dtype,device=full_fit_mse.device)
    if (reference.ndim or not torch.isfinite(reference) or reference<=0 or reference.requires_grad
            or multiplier.ndim or not torch.isfinite(multiplier) or multiplier<0 or multiplier.requires_grad):
        raise ValueError('Positive fixed reference and nonnegative fixed dual scalars are required')
    if not math.isfinite(rho) or rho<=0 or not math.isfinite(incremental_penalty) or incremental_penalty<0:
        raise ValueError('Finite positive rho and nonnegative increment weight are required')
    g=full_fit_mse/reference-1
    linear=multiplier*g
    quadratic=.5*rho*torch.relu(g).square()
    loss=normalized_gamma_crps+incremental_penalty*incremental_mse+linear+quadratic
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite constrained objective')
    return dict(loss=loss,constraint_g=g,dual_term=linear,augmented_penalty=quadratic)


def update_dual(dual, full_fit_g, step=1.):
    if any(not math.isfinite(v) for v in (dual,full_fit_g,step)) or dual<0 or step<=0:
        raise ValueError('A finite nonnegative dual and positive update step are required')
    return max(0.,float(dual)+float(step)*float(full_fit_g))


def _inputs(model,objective,x,chem,mask,target,gamma,fit,valid,ids,config,loss_mode):
    if not isinstance(model,ConditionalResponseKernelMean) or not isinstance(objective,JointGammaCRPS):
        raise TypeError('Use the complete conditional response model and frozen joint Gamma objective')
    if loss_mode not in ('weighted','constrained'): raise ValueError('Unknown loss_mode')
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('This matched implementation uses the recorded CPU random streams')
    if any(p.dtype!=torch.float64 for p in model.parameters()):
        raise ValueError('The full matched model must use float64')
    if any(b.device.type!='cpu' for b in objective.buffers()) or any(p.requires_grad for p in objective.parameters()):
        raise ValueError('Joint Gamma objective must have frozen CPU buffers only')
    ids=np.asarray(ids,str);fit=np.asarray(fit);valid=np.asarray(valid)
    if ids.ndim!=1 or len(set(ids))!=len(ids): raise ValueError('Unique one-dimensional IDs required')
    n=len(ids)
    for name,indexes in (('fit',fit),('validation',valid)):
        if (indexes.ndim!=1 or not len(indexes) or not np.issubdtype(indexes.dtype,np.integer)
                or len(set(indexes))!=len(indexes) or np.any(indexes<0) or np.any(indexes>=n)):
            raise ValueError('Invalid '+name+' membership')
    if np.intersect1d(fit,valid).size: raise ValueError('Fit and validation memberships overlap')
    tensors=[torch.as_tensor(v,dtype=d) for v,d in ((x,torch.float64),(chem,torch.float64),
        (mask,torch.bool),(target,torch.float64),(gamma,torch.float64))]
    x,chem,mask,target,gamma=tensors
    if x.ndim!=2 or chem.ndim!=2 or x.shape[0]!=n or chem.shape[0]!=n or mask.shape!=(n,) or target.shape!=(n,9) or gamma.shape!=(n,):
        raise ValueError('Decision inputs and nine-coordinate/Gamma targets must align with IDs')
    if any(not torch.isfinite(t).all() for t in (x,chem,target,gamma)):
        raise ValueError('Training arrays must be finite')
    for key in ('stage_epochs','max_epochs','validation_interval','batch_size','warmup_steps','train_pairs','validation_pairs'):
        value=config[key]
        if not isinstance(value,int) or value<=0: raise ValueError('Positive integer config required: '+key)
    if config['stage_epochs']>config['max_epochs']: raise ValueError('Stage exceeds scheduler horizon')
    if config.get('gamma_weight')!=1. or not math.isclose(config['incremental_penalty'],model.incremental_penalty_weight,rel_tol=0,abs_tol=0):
        raise ValueError('Matched Gamma/increment weights differ from the model')
    for key in ('learning_rate','min_learning_rate','gradient_clip','dual_penalty','dual_step'):
        if not math.isfinite(config[key]) or config[key]<=0: raise ValueError('Positive finite config required: '+key)
    if not math.isfinite(config['weight_decay']) or config['weight_decay']<0: raise ValueError('Invalid weight decay')
    if config['min_learning_rate']>config['learning_rate']: raise ValueError('Invalid cosine learning rate floor')
    return x,chem,mask,target,gamma,fit,valid,ids


def train_branch(folder,model,objective,x,chem,mask,target,gamma,fit,valid,seed,ids,config,loss_mode):
    """Fresh fixed-stage fit, with all data supplied explicitly by the caller.

    No TEST data, path-based data loader, checkpoint selection or feasibility
    fallback exists here. ``best.pt`` is only a validation-geometry appendix;
    the returned model and ``last.pt`` always represent the fixed final epoch.
    """
    folder=Path(folder)
    if folder.exists(): raise FileExistsError('Existing training is preserved')
    config=deepcopy(dict(config))
    x,chem,mask,target,gamma,fit,valid,ids=_inputs(
        model,objective,x,chem,mask,target,gamma,fit,valid,ids,config,loss_mode)
    model.eval()
    with torch.no_grad():
        base=model.base_hr(x[fit])
        if not torch.equal(model(x[fit],chem[fit],mask[fit]),base):
            raise ValueError('Fresh branch must initially reproduce frozen HR exactly')
        reference_mse=(base-target[fit]).square().mean().detach()
        reference_validation_mse=(model.base_hr(x[valid])-target[valid]).square().mean().detach()
    if not torch.isfinite(reference_mse) or reference_mse<=0:
        raise ValueError('Frozen HR full-fit geometry MSE must be positive')
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    params=list(model.trainable_parameters())
    before={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith(('base_hr.','bank.'))}
    objective_before=deepcopy(objective.state_dict())
    optimizer=torch.optim.AdamW(params,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    total=config['max_epochs']*math.ceil(len(fit)/config['batch_size'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:_schedule(step,total,config))
    order_rng=torch.Generator().manual_seed(seed+31)
    mc_rng=torch.Generator().manual_seed(seed+config['training_mc_offset'])
    monitoring_rng=torch.Generator().manual_seed(seed+config['validation_mc_offset'])
    fit_noise=independent_noise(config['validation_pairs'],len(fit),monitoring_rng)
    valid_noise=independent_noise(config['validation_pairs'],len(valid),monitoring_rng)
    dual=0.;steps=0;best_epoch=0;best_score=float('inf');started=time.monotonic()
    write_json(folder/'training_config.json',dict(config=config,loss_mode=loss_mode,training_seed=int(seed),
        fit_ids=ids[fit].tolist(),validation_ids=ids[valid].tolist(),reference_hr_fit_mse=float(reference_mse),
        reference_hr_validation_mse=float(reference_validation_mse),
        constraint_scope='exact full-fit differentiable MSE before every constrained primal update',
        dual_update='once per epoch using post-update full-fit g; nonnegative projection',
        constraint_reference_membership='FIT only',validation_used_for_training=False,
        checkpoint_selection='fixed final epoch; best.pt is descriptive only',convergence_guarantee=False))
    for epoch in range(config['stage_epochs']+1):
        sums={k:0. for k in ('mean_mse','incremental_mse','gamma_crps','normalized_gamma_crps','total_loss','full_fit_constraint_g')}
        count=0;norms=[];dual_before=dual
        if epoch:
            model.train()
            order=fit[torch.randperm(len(fit),generator=order_rng).numpy()]
            for first in range(0,len(order),config['batch_size']):
                ii=order[first:first+config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                original=model.loss(x[ii],chem[ii],mask[ii],target[ii])
                a,b=independent_noise(config['train_pairs'],len(ii),mc_rng)
                supervised=objective(model(x[ii],chem[ii],mask[ii]),gamma[ii],a,b)
                if loss_mode=='weighted':
                    total_loss=original['loss']+config['gamma_weight']*supervised['normalized_gamma_crps']
                    g=None
                else:
                    # This graph includes all FIT objects and excludes all validation labels.
                    full_fit_mse=(model(x[fit],chem[fit],mask[fit])-target[fit]).square().mean()
                    augmented=constrained_objective(supervised['normalized_gamma_crps'],
                        original['incremental_mse'],full_fit_mse,reference_mse,dual,
                        incremental_penalty=config['incremental_penalty'],rho=config['dual_penalty'])
                    total_loss=augmented['loss'];g=augmented['constraint_g']
                if not torch.isfinite(total_loss): raise FloatingPointError('Nonfinite final-factorial objective')
                total_loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,config['gradient_clip'],error_if_nonfinite=True)
                norms.append(float(norm));optimizer.step();scheduler.step();steps+=1
                for key in ('mean_mse','incremental_mse'): sums[key]+=len(ii)*float(original[key].detach())
                for key in ('gamma_crps','normalized_gamma_crps'): sums[key]+=len(ii)*float(supervised[key].detach())
                sums['total_loss']+=len(ii)*float(total_loss.detach())
                if g is not None:sums['full_fit_constraint_g']+=len(ii)*float(g.detach())
                count+=len(ii)
        model.eval()
        with torch.no_grad():
            full_fit=model.loss(x[fit],chem[fit],mask[fit],target[fit])
            full_g=float(full_fit['mean_mse']/reference_mse-1)
        if loss_mode=='constrained' and epoch:dual=update_dual(dual,full_g,config['dual_step'])
        row=dict(epoch=epoch,optimizer_steps=steps,learning_rate=optimizer.param_groups[0]['lr'],
            **{'train_minibatch_'+key:value/count if count else None for key,value in sums.items()},
            gradient_norm_mean=float(np.mean(norms)) if norms else None,
            gradient_norm_max=float(np.max(norms)) if norms else None,
            gradient_clip_fraction=float(np.mean(np.asarray(norms)>config['gradient_clip'])) if norms else None,
            fit_u_mse=float(full_fit['mean_mse']),reference_hr_fit_mse=float(reference_mse),
            fit_constraint_g=full_g,fit_feasible=bool(full_g<=0),dual_before=dual_before,dual_after=dual,
            dual_updated=bool(loss_mode=='constrained' and epoch),elapsed_seconds=time.monotonic()-started)
        if loss_mode=='weighted': row['train_minibatch_full_fit_constraint_g']=None
        if epoch%config['validation_interval']==0 or epoch==config['stage_epochs']:
            monitoring=dict(epoch=epoch,checkpoint_readout_only=True,fixed_pairs=config['validation_pairs'],
                            fixed_full_fit_monitoring=True,fit_constraint_g=full_g,dual=dual)
            with torch.no_grad():
                for name,indexes,noise in (('fit',fit,fit_noise),('validation',valid,valid_noise)):
                    original=model.loss(x[indexes],chem[indexes],mask[indexes],target[indexes])
                    scored=objective(model(x[indexes],chem[indexes],mask[indexes]),gamma[indexes],*noise)
                    monitoring.update({name+'_u_mse':float(original['mean_mse']),
                        name+'_incremental_mse':float(original['incremental_mse']),
                        name+'_gamma_crps':float(scored['gamma_crps']),
                        name+'_normalized_gamma_crps':float(scored['normalized_gamma_crps']),
                        name+'_gamma_mse':float((scored['gamma_prediction_mean']-gamma[indexes]).square().mean())})
            row.update({k:v for k,v in monitoring.items() if k.startswith(('fit_','validation_'))})
            score=monitoring['validation_u_mse']
            payload=checkpoint_payload(model,optimizer,scheduler,order_rng,epoch,steps,score,ids[fit],ids[valid])
            payload.update(training_mc_rng_state=mc_rng.get_state(),gamma_objective_state_dict=deepcopy(objective.state_dict()),
                gamma_weight=config['gamma_weight'],gamma_train_pairs=config['train_pairs'],training_config=deepcopy(config),
                loss_mode=loss_mode,dual_state=dict(value=dual,rho=config['dual_penalty'],step=config['dual_step'],
                    last_updated_epoch=epoch if loss_mode=='constrained' else None,full_fit_constraint_g=full_g,
                    reference_hr_fit_mse=float(reference_mse)),
                monitoring_rng_state=monitoring_rng.get_state(),fixed_monitoring_seed=seed+config['validation_mc_offset'],
                fixed_monitoring_pairs=config['validation_pairs'])
            torch.save(payload,folder/f'epoch{epoch}.pt');torch.save(payload,folder/'last.pt')
            if score<best_score:
                best_score,best_epoch=score,epoch;torch.save(payload,folder/'best.pt')
            row['best_epoch']=best_epoch
            with (folder/'gamma_monitoring.jsonl').open('a') as stream:
                stream.write(json.dumps(plain(monitoring),allow_nan=False)+'\n')
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row),allow_nan=False)+'\n')
    if any(not torch.equal(model.state_dict()[k],v) for k,v in before.items()):
        raise RuntimeError('Frozen HR or fitted response bank changed')
    if any(not torch.equal(objective.state_dict()[k],v) for k,v in objective_before.items()):
        raise RuntimeError('Frozen joint covariance or objective scales changed')
    completion=dict(epoch=config['stage_epochs'],final_epoch=config['stage_epochs'],
        actual_checkpoint_epoch=config['stage_epochs'],optimizer_steps=steps,best_epoch=best_epoch,
        best_validation_u_mse=best_score,loss_mode=loss_mode,dual_final=dual,
        reference_hr_fit_mse=float(reference_mse),final_fit_u_mse=float(full_fit['mean_mse']),
        final_fit_constraint_g=full_g,fit_feasible=bool(full_g<=0),feasibility_scope='FIT only, exact at final checkpoint',
        elapsed_seconds=time.monotonic()-started,stop_reason='requested_fixed_epoch_final_factorial',
        converged_claim=False,feasibility_guarantee=False,hidden_fallback=False,
        frozen_hr_changed=False,bank_changed=False,objective_buffers_changed=False,
        trainable_parameters=sum(p.numel() for p in params),
        parameter_counts=dict(trainable=sum(p.numel() for p in params),total=sum(p.numel() for p in model.parameters())),
        geometry_feasibility=dict(fit_satisfied=float(full_fit['mean_mse']-reference_mse)<=1e-10,
            fullfit_baseline_mse=float(reference_mse),fullfit_final_mse=float(full_fit['mean_mse']),
            validation_baseline_mse=float(reference_validation_mse),validation_final_mse=row['validation_u_mse'],
            validation_satisfied=row['validation_u_mse']-float(reference_validation_mse)<=1e-10,
            numerical_tolerance=1e-10,tolerance_scope='absolute MSE roundoff only, not practical slack',
            validation_not_optimized_as_constraint=True))
    write_json(folder/'training_complete.json',completion)
    return model
