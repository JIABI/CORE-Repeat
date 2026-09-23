"""Exact CPU continuation of the four fixed kernel/Gamma-loss branches.

This module resumes the actual endpoint, not the validation-best checkpoint.
It does not refit descriptors, uncertainty or target scalers, and does not
restart the original learning-rate schedule or any random-number stream.
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
from .conditional_response_kernel import ConditionalResponseKernelMean
from .gamma_supervised_loss import JointGammaCRPS
from .geometry_kernel_replacement_experiment import checkpoint_payload
from .gram_experiment import _schedule
from .gram_oof_experiment import event


ARM_MODES = {
    'F_CONDITIONAL_GENERIC': ('conditional_generic', False),
    'J_GEOMETRY_CONTROL': ('conditional_structured', False),
    'K_GEOMETRY_GAMMA_CRPS': ('conditional_structured', True),
    'M_CONDITIONAL_GENERIC_GAMMA': ('conditional_generic', True),
}


def _identical_state(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _restore_objective(payload, provided):
    saved = payload['gamma_objective_state_dict']
    required = ('covariance', 'target_center', 'target_scale', 'gamma_scale', 'cost_two', 'scale_tril')
    if set(saved) != set(required):
        raise ValueError('Saved joint Gamma objective is incomplete or changed')
    if provided is not None:
        if not isinstance(provided, JointGammaCRPS) or not _identical_state(saved, provided.state_dict()):
            raise ValueError('Supplied frozen Gamma objective differs from the source checkpoint')
    restored = JointGammaCRPS(saved['covariance'], saved['target_center'], saved['target_scale'],
                              saved['gamma_scale'], cost_two=float(saved['cost_two']))
    # The stored factor must still be the Cholesky factor of the stored full
    # covariance. Never silently repair an inconsistent uncertainty checkpoint.
    if not _identical_state(saved, restored.state_dict()):
        raise ValueError('Stored Gamma factor or scaler does not reproduce its frozen law')
    restored.load_state_dict(saved)
    return restored


def continue_branch(folder, source_folder, x, chem, mask, target, actual_gamma,
                    fit, valid, ids, config, *, arm, objective=None, end_epoch=60, device='cpu'):
    """Resume a completed endpoint with local fit/validation arrays only.

    Production uses epoch30→60. Smaller saved endpoints exercise the identical
    mechanics in synthetic regression tests; the orchestrator fixes production
    endpoints. Only CPU is supported because the source checkpoints saved CPU
    RNG streams, not accelerator RNG streams.
    """
    folder, source_folder = Path(folder), Path(source_folder)
    if folder.exists():
        raise FileExistsError('An existing continuation directory cannot be overwritten')
    if torch.device(device).type != 'cpu':
        raise ValueError('Exact continuation retains the source CPU device and RNG semantics')
    if arm not in ARM_MODES:
        raise ValueError('Only the four declared frozen-architecture branches may continue')
    mode, supervised = ARM_MODES[arm]
    cfg = deepcopy(dict(config))
    completion = json.loads((source_folder/'training_complete.json').read_text())
    start, end = int(completion['actual_checkpoint_epoch']), int(end_epoch)
    interval, batch = int(cfg['validation_interval']), int(cfg['batch_size'])
    if (int(cfg['stage_epochs']) != end or not 0 < start < end <= int(cfg['max_epochs'])
            or interval <= 0 or batch <= 0 or start % interval or end % interval):
        raise ValueError('Continuation endpoints must remain saved epochs within the original horizon')
    checkpoint = source_folder/f'epoch{start}.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    ids, fit, valid = np.asarray(ids, str), np.asarray(fit), np.asarray(valid)
    if ids.ndim != 1 or len(set(ids.tolist())) != len(ids):
        raise ValueError('Local IDs must be unique')
    for indexes in (fit, valid):
        if (indexes.ndim != 1 or not np.issubdtype(indexes.dtype, np.integer) or not len(indexes)
                or np.any(indexes < 0) or np.any(indexes >= len(ids)) or len(np.unique(indexes)) != len(indexes)):
            raise ValueError('Local fit and validation indices must be unique, nonempty and legal')
    if not np.array_equal(np.sort(np.r_[fit,valid]), np.arange(len(ids))):
        raise ValueError('Only disjoint local fit and validation arrays may enter training')
    if payload['fit_ids'] != ids[fit].tolist() or payload['validation_ids'] != ids[valid].tolist():
        raise ValueError('Fit or validation IDs differ from the source endpoint')
    batches = math.ceil(len(fit)/batch)
    start_steps = start*batches
    if (payload['epoch'] != start or payload['actual_checkpoint_epoch'] != start
            or payload['optimizer_steps'] != start_steps or completion['optimizer_steps'] != start_steps):
        raise ValueError('The saved endpoint and unchanged minibatch step count disagree')
    if payload['model_config']['mode'] != mode:
        raise ValueError('The saved kernel mode differs from the declared arm')
    if supervised != ('training_mc_rng_state' in payload):
        raise ValueError('The source objective/random-stream family differs from the requested arm')
    if supervised and (payload.get('gamma_weight') != cfg['gamma_weight']
                       or payload.get('gamma_train_pairs') != cfg['train_pairs']):
        raise ValueError('The original Gamma weight and MC pair count must not change')
    best = torch.load(source_folder/'best.pt', map_location='cpu', weights_only=True)
    if (best['epoch'] > start or best['fit_ids'] != payload['fit_ids']
            or best['validation_ids'] != payload['validation_ids'] or best['epoch'] != completion['best_epoch']):
        raise ValueError('Source validation-best provenance is inconsistent')
    history = [json.loads(line) for line in (source_folder/'history.jsonl').read_text().splitlines() if line.strip()]
    if [row['epoch'] for row in history] != list(range(start+1)):
        raise ValueError('Source history must preserve all epochs once')

    model = ConditionalResponseKernelMean.from_config(payload['model_config']).double()
    model.load_state_dict(payload['state_dict'])
    if model.incremental_penalty_weight != float(cfg['incremental_penalty']):
        raise ValueError('The original incremental objective must remain unchanged')
    x, chem = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(chem, dtype=torch.float64)
    mask, target = torch.as_tensor(mask, dtype=torch.bool), torch.as_tensor(target, dtype=torch.float64)
    if any(len(value) != len(ids) for value in (x,chem,mask,target)) or target.shape != (len(ids),9):
        raise ValueError('Local inputs, targets and IDs must align')
    if supervised:
        gamma = torch.as_tensor(actual_gamma, dtype=torch.float64)
        if gamma.shape != (len(ids),) or not torch.isfinite(gamma).all():
            raise ValueError('Gamma supervision requires the original finite local targets')
        objective = _restore_objective(payload, objective)
        objective_before = deepcopy(objective.state_dict())
        mc_rng = torch.Generator()
        mc_rng.set_state(payload['training_mc_rng_state'])
    else:
        gamma = mc_rng = None
        objective_before = None
    params = list(model.trainable_parameters())
    frozen = {key:value.detach().clone() for key,value in model.state_dict().items()
              if key.startswith(('base_hr.', 'bank.'))}
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    defaults = {key:optimizer.param_groups[0][key] for key in ('betas','eps','amsgrad','maximize','foreach','capturable','differentiable','fused')}
    total = int(cfg['max_epochs'])*batches
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:_schedule(step,total,cfg))
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    scheduler.load_state_dict(payload['scheduler_state_dict'])
    expected_lr = cfg['learning_rate']*_schedule(start_steps,total,cfg)
    if (scheduler.last_epoch != start_steps or scheduler.base_lrs != [cfg['learning_rate']]
            or any(group['weight_decay'] != cfg['weight_decay']
                   or any(group[key] != value for key,value in defaults.items())
                   or not math.isclose(group['lr'],expected_lr,rel_tol=1e-13,abs_tol=1e-16)
                   for group in optimizer.param_groups)):
        raise ValueError('Saved optimizer or scheduler differs from the original continuation recipe')
    order_rng = torch.Generator()
    order_rng.set_state(payload['order_rng_state'])
    # Loading constructors may consume the global RNG. Restore only after all
    # constructors, before continuation; do not seed or restart any stream.
    torch.set_rng_state(payload['torch_rng_state'])
    best_epoch, best_score = int(best['epoch']), float(best['validation_u_mse'])
    if not math.isfinite(best_score): raise ValueError('Source best validation score must be finite')
    folder.mkdir(parents=True)
    for path in sorted(source_folder.glob('epoch*.pt')):
        if path.stem[5:].isdigit() and int(path.stem[5:]) <= start:
            shutil.copy2(path,folder/path.name)
    for name in ('best.pt','history.jsonl'):
        shutil.copy2(source_folder/name,folder/name)
    shutil.copy2(checkpoint,folder/'last.pt')
    shutil.copy2(source_folder/'training_complete.json',folder/'source_training_complete.json')
    audit = dict(arm=arm,source_folder=str(source_folder.resolve()),source_checkpoint=str(checkpoint.resolve()),
        start_epoch=start,source_optimizer_steps=start_steps,target_epoch=end,
        first_update_learning_rate=float(optimizer.param_groups[0]['lr']),
        scheduler_horizon_epochs=int(cfg['max_epochs']),scheduler_horizon_steps=total,
        optimizer_restored=True,scheduler_restored=True,torch_rng_restored=True,order_rng_restored=True,
        training_mc_rng_restored=supervised,objective_buffers_restored=supervised,
        geometry_arm_does_not_draw_gamma_noise=not supervised,warmup_restarted=False,seed_reset=False,
        cpu_device_preserved=True,fit_ids=ids[fit].tolist(),validation_ids=ids[valid].tolist())
    write_json(folder/'resume_audit.json',audit)
    started=time.monotonic();prior_elapsed=float(history[-1]['elapsed_seconds']);steps=start_steps
    event(folder,'CONTINUATION_STARTED',**audit)
    for epoch in range(start+1,end+1):
        model.train()
        order=fit[torch.randperm(len(fit),generator=order_rng).numpy()]
        sums=dict(mean_mse=0.,incremental_mse=0.,gamma_crps=0.,normalized_gamma_crps=0.,total_loss=0.)
        count,norms,part_norms=0,[],{}
        for first in range(0,len(order),batch):
            ii=order[first:first+batch]
            optimizer.zero_grad(set_to_none=True)
            original=model.loss(x[ii],chem[ii],mask[ii],target[ii])
            if supervised:
                shape=(int(cfg['train_pairs']),len(ii),9)
                a=torch.randn(shape,dtype=torch.float64,generator=mc_rng)
                b=torch.randn(shape,dtype=torch.float64,generator=mc_rng)
                # Same two forwards as the source Gamma loop; do not silently
                # fuse or change objective routing during continuation.
                score=objective(model(x[ii],chem[ii],mask[ii]),gamma[ii],a,b)
                loss=original['loss']+cfg['gamma_weight']*score['normalized_gamma_crps']
            else:
                score=None;loss=original['loss']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite resumed objective')
            loss.backward()
            if not supervised:
                for name,parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        part_norms.setdefault(name,[]).append(float(parameter.grad.norm()))
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip'],error_if_nonfinite=True)
            norms.append(float(norm));optimizer.step();scheduler.step();steps+=1
            for key in ('mean_mse','incremental_mse'): sums[key]+=len(ii)*float(original[key].detach())
            if supervised:
                for key in ('gamma_crps','normalized_gamma_crps'): sums[key]+=len(ii)*float(score[key].detach())
            sums['total_loss']+=len(ii)*float(loss.detach());count+=len(ii)
        elapsed=time.monotonic()-started
        row=dict(epoch=epoch,optimizer_steps=steps,learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=float(np.max(norms)),
            elapsed_seconds=prior_elapsed+elapsed,continuation_elapsed_seconds=elapsed)
        if supervised:
            row.update({'train_minibatch_'+key:value/count for key,value in sums.items()})
        else:
            row.update(train_minibatch_mse=sums['mean_mse']/count,
                train_minibatch_incremental_mse=sums['incremental_mse']/count,
                gradient_parameter_norm_mean={key:float(np.mean(value)) for key,value in part_norms.items()})
        if epoch%interval==0:
            model.eval()
            with torch.no_grad():
                fl=model.loss(x[fit],chem[fit],mask[fit],target[fit])
                vl=model.loss(x[valid],chem[valid],mask[valid],target[valid])
            value=float(vl['mean_mse'])
            if not math.isfinite(value): raise FloatingPointError('Nonfinite resumed validation MSE')
            row.update(fit_u_mse=float(fl['mean_mse']),validation_u_mse=value,
                fit_incremental_mse=float(fl['incremental_mse']),validation_incremental_mse=float(vl['incremental_mse']))
            saved=checkpoint_payload(model,optimizer,scheduler,order_rng,epoch,steps,value,ids[fit],ids[valid])
            if supervised:
                saved.update(training_mc_rng_state=mc_rng.get_state(),gamma_objective_state_dict=objective.state_dict(),
                    gamma_weight=cfg['gamma_weight'],gamma_train_pairs=cfg['train_pairs'])
            torch.save(saved,folder/f'epoch{epoch}.pt');torch.save(saved,folder/'last.pt')
            if value<best_score:
                best_score,best_epoch=value,epoch;torch.save(saved,folder/'best.pt')
            row['best_epoch']=best_epoch;event(folder,'VALIDATED',**row)
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row),allow_nan=False)+'\n')
    model.eval()
    if any(not torch.equal(model.state_dict()[key],value) for key,value in frozen.items()):
        raise RuntimeError('Frozen HR or response bank changed during continuation')
    if supervised and not _identical_state(objective_before,objective.state_dict()):
        raise RuntimeError('The frozen joint Gamma law changed during continuation')
    if steps != end*batches: raise RuntimeError('Continuation did not complete the fixed optimizer budget')
    result=dict(epoch=end,final_epoch=end,actual_checkpoint_epoch=end,start_epoch=start,
        best_epoch=best_epoch,best_validation_u_mse=best_score,optimizer_steps=steps,
        source_optimizer_steps=start_steps,additional_optimizer_steps=steps-start_steps,
        elapsed_seconds=prior_elapsed+time.monotonic()-started,
        continuation_elapsed_seconds=time.monotonic()-started,
        stop_reason=f'fixed_actual_epoch{end}_continuation',converged_claim=False,
        frozen_hr_changed=False,bank_changed=False,objective_buffers_changed=False,
        gamma_weight=cfg['gamma_weight'] if supervised else 0.,
        trainable_parameters=sum(p.numel() for p in params),
        parameter_counts=dict(trainable=sum(p.numel() for p in params),total=sum(p.numel() for p in model.parameters())),
        resume_audit=audit)
    write_json(folder/'training_complete.json',result);event(folder,'STAGE_COMPLETE',**result)
    return model
