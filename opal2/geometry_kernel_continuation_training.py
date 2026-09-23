"""Exact continuation of a saved single-path kernel branch, without refitting.

The source stage is read-only. The same optimizer, full-horizon learning-rate
schedule, parameter order, and both saved random streams continue in a fresh
directory. A shorter synthetic checkpoint can exercise the same continuation
mechanics; the production orchestrator separately fixes the 10-to-30 boundary.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .biology_kernel_evaluation import plain, write_json
from .geometry_kernel_replacement import GeometryKernelReplacementMean
from .geometry_kernel_replacement_experiment import checkpoint_payload
from .gram_experiment import _schedule
from .gram_oof_experiment import event


def continue_branch(folder,source_folder,x,chem,mask,target,fit,valid,ids,config):
    """Resume exactly from the completed source endpoint, never its best epoch."""
    folder,source_folder=Path(folder),Path(source_folder)
    if folder.exists():
        raise FileExistsError('An existing continuation directory must not be overwritten')
    cfg=deepcopy(dict(config))
    completion=json.loads((source_folder/'training_complete.json').read_text())
    start=int(completion['actual_checkpoint_epoch'])
    end=int(cfg['stage_epochs'])
    interval=int(cfg['validation_interval']);batch=int(cfg['batch_size'])
    if (not 0<start<end<=int(cfg['max_epochs']) or interval<=0 or batch<=0
            or start%interval or end%interval):
        raise ValueError('Resume endpoints must be saved validation epochs within the original horizon')
    checkpoint=source_folder/f'epoch{start}.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    ids=np.asarray(ids,str);fit=np.asarray(fit);valid=np.asarray(valid)
    for indices in (fit,valid):
        if (indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer) or len(indices)==0
                or np.any(indices<0) or np.any(indices>=len(ids)) or len(np.unique(indices))!=len(indices)):
            raise ValueError('Fitting and validation index lists must be nonempty, unique and legal')
    if len(np.unique(ids))!=len(ids) or np.intersect1d(fit,valid).size:
        raise ValueError('Fitting and validation identities must be disjoint and unique')
    if payload['fit_ids']!=ids[fit].tolist() or payload['validation_ids']!=ids[valid].tolist():
        raise ValueError('Fitting or validation IDs differ from the saved checkpoint')
    batches=math.ceil(len(fit)/batch);steps=start*batches
    if (payload['epoch']!=start or payload['actual_checkpoint_epoch']!=start
            or payload['optimizer_steps']!=steps or completion['optimizer_steps']!=steps):
        raise ValueError('Saved epoch, optimizer steps and unchanged minibatch geometry disagree')
    best=torch.load(source_folder/'best.pt',map_location='cpu',weights_only=True)
    if (best['epoch']>start or best['fit_ids']!=payload['fit_ids']
            or best['validation_ids']!=payload['validation_ids']
            or completion['best_epoch']!=best['epoch']):
        raise ValueError('The source validation-best checkpoint has inconsistent provenance')
    histories=[json.loads(line) for line in (source_folder/'history.jsonl').read_text().splitlines()]
    if [row['epoch'] for row in histories]!=list(range(start+1)):
        raise ValueError('The source stage history must contain each epoch exactly once')
    model=GeometryKernelReplacementMean.from_config(payload['model_config'])
    model.load_state_dict(payload['state_dict'])
    if model.incremental_penalty_weight!=float(cfg['incremental_penalty']):
        raise ValueError('The original incremental objective must not change on continuation')
    x,chem=torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(chem,dtype=torch.float64)
    mask,target=torch.as_tensor(mask,dtype=torch.bool),torch.as_tensor(target,dtype=torch.float64)
    if any(len(value)!=len(ids) for value in (x,chem,mask,target)):
        raise ValueError('Input tensors and identities must align')
    params=list(model.trainable_parameters())
    before={key:value.detach().clone() for key,value in model.state_dict().items()
            if key.startswith(('base_hr.','bank.'))}
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    total=int(cfg['max_epochs'])*batches
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:_schedule(step,total,cfg))
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    scheduler.load_state_dict(payload['scheduler_state_dict'])
    expected_lr=cfg['learning_rate']*_schedule(steps,total,cfg)
    if (scheduler.last_epoch!=steps or scheduler.base_lrs!=[cfg['learning_rate']]
            or any(group['weight_decay']!=cfg['weight_decay'] or
                   not math.isclose(group['lr'],expected_lr,rel_tol=1e-13,abs_tol=1e-16)
                   for group in optimizer.param_groups)):
        raise ValueError('Saved optimizer or learning-rate schedule differs from the continuation configuration')
    order_rng=torch.Generator()
    order_rng.set_state(payload['order_rng_state'])
    # Construction and checkpoint reload consume initialization randomness. Restore
    # the saved stream after these operations and never restart warmup or seeding.
    torch.set_rng_state(payload['torch_rng_state'])
    best_epoch,best_score=int(best['epoch']),float(best['validation_u_mse'])
    if not math.isfinite(best_score):
        raise ValueError('The source best validation score is nonfinite')
    folder.mkdir(parents=True)
    for name in (f'epoch{start}.pt','best.pt','history.jsonl'):
        shutil.copy2(source_folder/name,folder/name)
    shutil.copy2(source_folder/'training_complete.json',folder/'source_training_complete.json')
    shutil.copy2(checkpoint,folder/'last.pt')
    first_lr=float(optimizer.param_groups[0]['lr'])
    audit=dict(source_folder=str(source_folder.resolve()),source_checkpoint=str(checkpoint.resolve()),
        resume_epoch=start,resume_optimizer_steps=steps,target_epoch=end,
        original_schedule_epochs=int(cfg['max_epochs']),original_schedule_steps=total,
        first_update_learning_rate=first_lr,optimizer_restored=True,scheduler_restored=True,
        torch_rng_restored=True,minibatch_rng_restored=True,warmup_restarted=False,
        fit_ids=ids[fit].tolist(),validation_ids=ids[valid].tolist())
    write_json(folder/'resume_audit.json',audit)
    started=time.monotonic();prior_elapsed=float(histories[-1]['elapsed_seconds'])
    event(folder,'CONTINUATION_STARTED',**audit)
    for epoch in range(start+1,end+1):
        model.train()
        order=fit[torch.randperm(len(fit),generator=order_rng).numpy()]
        weighted_mse,weighted_increment,count,grad_norms=0.,0.,0,[]
        for first in range(0,len(order),batch):
            ii=order[first:first+batch]
            optimizer.zero_grad(set_to_none=True)
            losses=model.loss(x[ii],chem[ii],mask[ii],target[ii])
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite resumed kernel objective')
            losses['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True)
            grad_norms.append(float(norm))
            optimizer.step();scheduler.step();steps+=1
            weighted_mse+=len(ii)*float(losses['mean_mse'].detach())
            weighted_increment+=len(ii)*float(losses['incremental_mse'].detach())
            count+=len(ii)
        elapsed=time.monotonic()-started
        row=dict(epoch=epoch,optimizer_steps=steps,learning_rate=optimizer.param_groups[0]['lr'],
            train_minibatch_mse=weighted_mse/count,train_minibatch_incremental_mse=weighted_increment/count,
            gradient_norm_mean=float(np.mean(grad_norms)),gradient_norm_max=float(np.max(grad_norms)),
            elapsed_seconds=prior_elapsed+elapsed,continuation_elapsed_seconds=elapsed)
        if epoch%interval==0:
            model.eval()
            with torch.no_grad():
                fl=model.loss(x[fit],chem[fit],mask[fit],target[fit])
                vl=model.loss(x[valid],chem[valid],mask[valid],target[valid])
            score=float(vl['mean_mse'])
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite continuation validation MSE')
            row.update(fit_u_mse=float(fl['mean_mse']),validation_u_mse=score,
                fit_incremental_mse=float(fl['incremental_mse']),
                validation_incremental_mse=float(vl['incremental_mse']))
            saved=checkpoint_payload(model,optimizer,scheduler,order_rng,epoch,steps,score,ids[fit],ids[valid])
            torch.save(saved,folder/f'epoch{epoch}.pt');torch.save(saved,folder/'last.pt')
            if score<best_score:
                best_score,best_epoch=score,epoch;torch.save(saved,folder/'best.pt')
            row['best_epoch']=best_epoch
            event(folder,'VALIDATED',**row)
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row),allow_nan=False)+'\n')
    model.eval()
    changed=[key for key,value in before.items() if not torch.equal(model.state_dict()[key],value)]
    if changed:
        raise RuntimeError('Frozen HR or fitted basis bank changed on continuation: '+str(changed))
    if steps!=end*batches:
        raise RuntimeError('The fixed continuation optimizer-step budget was not completed')
    result=dict(epoch=end,actual_checkpoint_epoch=end,best_epoch=best_epoch,best_validation_u_mse=best_score,
        optimizer_steps=steps,elapsed_seconds=prior_elapsed+time.monotonic()-started,
        continuation_elapsed_seconds=time.monotonic()-started,
        stop_reason=f'planned_epoch{end}_development_readout',converged_claim=False,
        frozen_hr_changed=False,fitted_bank_changed=False,
        trainable_parameters=sum(p.numel() for p in params),
        parameter_counts=dict(trainable=sum(p.numel() for p in params),total=sum(p.numel() for p in model.parameters())),
        resumed_from_epoch=start,resumed_from_optimizer_steps=start*batches,
        continuation_optimizer_steps=(end-start)*batches,resume_audit=audit)
    write_json(folder/'training_complete.json',result)
    event(folder,'STAGE_COMPLETE',**result)
    return model
