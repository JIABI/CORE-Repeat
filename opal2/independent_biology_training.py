"""Matched 30-epoch independent corrections around a fully frozen A model.

The objective and random-stream convention match the preceding LINCS run.
Only the reference for initialization and increment regularization changes
from HR to the complete saved A model. No test arrays enter this trainer.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .biology_kernel_evaluation import plain, write_json
from .gamma_supervised_experiment import independent_noise
from .gamma_supervised_loss import JointGammaCRPS
from .geometry_kernel_replacement_experiment import checkpoint_payload
from .gram_experiment import _schedule
from .independent_biology_kernel import IndependentBiologyKernelMean


def train_branch(folder, model, objective, x, packed, mask, target, gamma,
                 fit, valid, seed, ids, config):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Existing branch results are preserved')
    if not isinstance(model, IndependentBiologyKernelMean) or not isinstance(objective, JointGammaCRPS):
        raise TypeError('Independent model and original joint Gamma objective required')
    cfg = deepcopy(config)
    if cfg['stage_epochs'] > cfg['max_epochs'] or cfg['stage_epochs'] <= 0:
        raise ValueError('Invalid declared epoch budget')
    if cfg['gamma_weight'] != 1. or cfg['incremental_penalty'] != model.incremental_penalty_weight:
        raise ValueError('Matched loss weights differ')
    fit, valid, ids = np.asarray(fit, int), np.asarray(valid, int), np.asarray(ids, str)
    if not len(fit) or not len(valid) or len(set(ids)) != len(ids) or np.intersect1d(fit, valid).size:
        raise ValueError('Disjoint nonempty FIT and validation identities required')
    if set(np.r_[fit, valid]) != set(range(len(ids))):
        raise ValueError('Only explicitly supplied FIT and validation rows are accepted')
    x, packed, target, gamma = [torch.as_tensor(v, dtype=torch.float64) for v in (x, packed, target, gamma)]
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if target.shape != (len(ids), 9) or gamma.shape != (len(ids),) or mask.shape != (len(ids),):
        raise ValueError('Misaligned targets or mask')
    if x.shape[0] != len(ids) or packed.shape[0] != len(ids):
        raise ValueError('Misaligned decision inputs')
    if any(not torch.isfinite(v).all() for v in (x, packed, target, gamma)):
        raise ValueError('Finite supplied training arrays required')
    if any(p.device.type != 'cpu' or p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError('Matched CPU float64 training required')
    model.eval()
    with torch.no_grad():
        baseline = model.baseline_mean(x, packed, mask)
        if not torch.equal(model(x, packed, mask), baseline):
            raise ValueError('A new branch must start at the complete frozen A')
        references = {name: float((baseline[ii]-target[ii]).square().mean())
                      for name, ii in (('fit', fit), ('validation', valid))}
    base_before = deepcopy(model.base_a.state_dict())
    objective_before = deepcopy(objective.state_dict())
    params = list(model.trainable_parameters())
    if not params:
        raise ValueError('No active independent correction parameters')
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    total = cfg['max_epochs']*math.ceil(len(fit)/cfg['batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, total, cfg))
    order_rng = torch.Generator().manual_seed(seed+31)
    mc_rng = torch.Generator().manual_seed(seed+cfg['training_mc_offset'])
    monitor_rng = torch.Generator().manual_seed(seed+cfg['validation_mc_offset'])
    noises = {name: independent_noise(cfg['validation_pairs'], len(ii), monitor_rng)
              for name, ii in (('fit', fit), ('validation', valid))}
    write_json(folder/'training_config.json', dict(config=cfg, training_seed=int(seed),
        fit_ids=ids[fit].tolist(), validation_ids=ids[valid].tolist(),
        baseline='complete saved A_OLD_GENERIC at epoch30, frozen', reference_A_mse=references,
        loss='geometry MSE + normalized joint Gamma CRPS + 0.1 * mean increment squared',
        checkpoint_selection='fixed epoch30 primary; validation-best including epoch0 descriptive',
        covariance='unchanged original fold RIDGE OOF covariance', test_data_supplied=False))
    started = time.monotonic()
    steps, best_epoch, best_score = 0, 0, float('inf')
    for epoch in range(cfg['stage_epochs']+1):
        sums = {k: 0. for k in ('u_mse', 'incremental_mse', 'gamma_crps', 'normalized_gamma_crps', 'loss')}
        count, norms = 0, []
        if epoch:
            model.train()
            order = fit[torch.randperm(len(fit), generator=order_rng).numpy()]
            for offset in range(0, len(order), cfg['batch_size']):
                ii = order[offset:offset+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                original = model.loss(x[ii], packed[ii], mask[ii], target[ii])
                a, b = independent_noise(cfg['train_pairs'], len(ii), mc_rng)
                scored = objective(model(x[ii], packed[ii], mask[ii]), gamma[ii], a, b)
                loss = original['loss'] + cfg['gamma_weight']*scored['normalized_gamma_crps']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite independent correction loss')
                loss.backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(params, cfg['gradient_clip'], error_if_nonfinite=True)))
                optimizer.step(); scheduler.step(); steps += 1
                values = dict(u_mse=original['mean_mse'], incremental_mse=original['incremental_mse'],
                              gamma_crps=scored['gamma_crps'], normalized_gamma_crps=scored['normalized_gamma_crps'], loss=loss)
                for key, value in values.items():
                    sums[key] += len(ii)*float(value.detach())
                count += len(ii)
        model.eval()
        row = dict(epoch=epoch, optimizer_steps=steps, learning_rate=optimizer.param_groups[0]['lr'],
                   elapsed_seconds=time.monotonic()-started,
                   gradient_norm_mean=float(np.mean(norms)) if norms else None,
                   gradient_clip_fraction=float(np.mean(np.asarray(norms)>cfg['gradient_clip'])) if norms else None,
                   **{'train_minibatch_'+k: v/count if count else None for k, v in sums.items()})
        if epoch % cfg['validation_interval'] == 0 or epoch == cfg['stage_epochs']:
            with torch.no_grad():
                for name, ii in (('fit', fit), ('validation', valid)):
                    result = model.loss(x[ii], packed[ii], mask[ii], target[ii])
                    scored = objective(model(x[ii], packed[ii], mask[ii]), gamma[ii], *noises[name])
                    row.update({name+'_u_mse': float(result['mean_mse']),
                                name+'_incremental_mse': float(result['incremental_mse']),
                                name+'_gamma_crps': float(scored['gamma_crps']),
                                name+'_normalized_gamma_crps': float(scored['normalized_gamma_crps'])})
            score = row['validation_u_mse']
            payload = checkpoint_payload(model, optimizer, scheduler, order_rng, epoch, steps, score, ids[fit], ids[valid])
            payload.update(training_config=cfg, training_mc_rng_state=mc_rng.get_state(),
                           gamma_objective_state_dict=deepcopy(objective.state_dict()),
                           monitoring_rng_state=monitor_rng.get_state(),
                           fixed_monitoring_seed=seed+cfg['validation_mc_offset'],
                           baseline_reference='A_OLD_GENERIC epoch30', loss_mode='weighted')
            torch.save(payload, folder/f'epoch{epoch}.pt')
            torch.save(payload, folder/'last.pt')
            if score < best_score:
                best_score, best_epoch = score, epoch
                torch.save(payload, folder/'best.pt')
            row['best_epoch'] = best_epoch
            with (folder/'gamma_monitoring.jsonl').open('a') as stream:
                stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
    if any(not torch.equal(v, model.base_a.state_dict()[k]) for k, v in base_before.items()):
        raise RuntimeError('Frozen A parameters or reference buffers changed')
    if any(not torch.equal(v, objective.state_dict()[k]) for k, v in objective_before.items()):
        raise RuntimeError('Joint objective buffers changed')
    with torch.no_grad():
        if not torch.equal(model(x, packed, mask, branch_enabled=False), baseline):
            raise RuntimeError('Disabling the trained branch does not exactly restore A')
    write_json(folder/'training_complete.json', dict(actual_checkpoint_epoch=cfg['stage_epochs'],
        final_epoch=cfg['stage_epochs'], optimizer_steps=steps, best_epoch=best_epoch,
        best_validation_u_mse=best_score, reference_A_fit_mse=references['fit'],
        reference_A_validation_mse=references['validation'], final_fit_u_mse=row['fit_u_mse'],
        final_validation_u_mse=row['validation_u_mse'], trainable_parameters=sum(p.numel() for p in params),
        elapsed_seconds=time.monotonic()-started, frozen_A_changed=False,
        objective_buffers_changed=False, disabled_equals_A=True, stop_reason='requested_fixed_30_epochs'))
    return model
