"""Full matched fixed-epoch training for frozen-A state/biology corrections."""
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
    folder, cfg = Path(folder), deepcopy(config)
    if folder.exists():
        raise FileExistsError('Preserve completed and partial run artifacts')
    if not isinstance(model, IndependentBiologyKernelMean) or not isinstance(objective, JointGammaCRPS):
        raise TypeError('Complete independent correction and joint Gamma objective required')
    epochs = cfg['stage_epochs']
    if not 0 < epochs <= cfg['max_epochs']:
        raise ValueError('Invalid epoch budget')
    if cfg['gamma_weight'] != 1. or cfg['incremental_penalty'] != model.incremental_penalty_weight:
        raise ValueError('Matched objective weights changed')
    fit, valid, ids = np.asarray(fit, int), np.asarray(valid, int), np.asarray(ids, str)
    if (not len(fit) or not len(valid) or len(set(ids)) != len(ids)
            or np.intersect1d(fit, valid).size or set(np.r_[fit, valid]) != set(range(len(ids)))):
        raise ValueError('Supply only disjoint declared FIT/validation objects')
    x, packed, target, gamma = [torch.as_tensor(v, dtype=torch.float64) for v in (x, packed, target, gamma)]
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if target.shape != (len(ids), 9) or gamma.shape != (len(ids),) or mask.shape != (len(ids),):
        raise ValueError('Target/scope mismatch')
    if x.shape[0] != len(ids) or packed.shape[0] != len(ids):
        raise ValueError('Decision inputs mismatch')
    if any(not torch.isfinite(v).all() for v in (x, packed, target, gamma)):
        raise ValueError('Finite declared training arrays required')
    if any(p.device.type != 'cpu' or p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError('Matched CPU float64 model required')
    model.eval()
    with torch.no_grad():
        baseline = model.baseline_mean(x, packed, mask)
        if not torch.equal(model(x, packed, mask), baseline):
            raise ValueError('Every new branch must start at complete A')
        references = {s: float((baseline[ii]-target[ii]).square().mean()) for s, ii in (('fit', fit), ('validation', valid))}
    base_before = deepcopy(model.base_a.state_dict())
    objective_before = deepcopy(objective.state_dict())
    # All pre-fitted input/PCA/scale buffers are fixed, not optimization targets.
    fixed_before = {k: v.clone() for k, v in model.named_buffers() if not k.startswith('base_a.')}
    params = list(model.trainable_parameters())
    if not params:
        raise ValueError('Missing active parameters')
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    total = cfg['max_epochs']*math.ceil(len(fit)/cfg['batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, total, cfg))
    order_rng = torch.Generator().manual_seed(seed+31)
    mc_rng = torch.Generator().manual_seed(seed+cfg['training_mc_offset'])
    monitor_rng = torch.Generator().manual_seed(seed+cfg['validation_mc_offset'])
    noises = {s: independent_noise(cfg['validation_pairs'], len(ii), monitor_rng) for s, ii in (('fit', fit), ('validation', valid))}
    write_json(folder/'training_config.json', dict(config=cfg, training_seed=int(seed), fit_ids=ids[fit].tolist(),
        validation_ids=ids[valid].tolist(), baseline='complete saved A_OLD_GENERIC epoch30, frozen',
        reference_A_mse=references, loss='geometry MSE + normalized joint Gamma CRPS + 0.1 * mean increment squared',
        checkpoint_selection=f'fixed epoch{epochs} primary; validation-best descriptive only',
        covariance='unchanged original fold RIDGE OOF covariance', test_data_supplied=False))
    started, steps, best_epoch, best_score = time.monotonic(), 0, 0, float('inf')
    for epoch in range(epochs+1):
        sums = {k: 0. for k in ('u_mse', 'incremental_mse', 'gamma_crps', 'normalized_gamma_crps', 'loss')}
        count, norms = 0, []
        if epoch:
            model.train()
            order = fit[torch.randperm(len(fit), generator=order_rng).numpy()]
            for offset in range(0, len(order), cfg['batch_size']):
                ii = order[offset:offset+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                original = model.loss(x[ii], packed[ii], mask[ii], target[ii])
                na, nb = independent_noise(cfg['train_pairs'], len(ii), mc_rng)
                scored = objective(model(x[ii], packed[ii], mask[ii]), gamma[ii], na, nb)
                loss = original['loss']+cfg['gamma_weight']*scored['normalized_gamma_crps']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite matched loss')
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
            elapsed_seconds=time.monotonic()-started, gradient_norm_mean=float(np.mean(norms)) if norms else None,
            gradient_clip_fraction=float(np.mean(np.asarray(norms)>cfg['gradient_clip'])) if norms else None,
            **{'train_minibatch_'+k: v/count if count else None for k, v in sums.items()})
        if epoch % cfg['validation_interval'] == 0 or epoch == epochs:
            with torch.no_grad():
                for name, ii in (('fit', fit), ('validation', valid)):
                    result = model.loss(x[ii], packed[ii], mask[ii], target[ii])
                    scored = objective(model(x[ii], packed[ii], mask[ii]), gamma[ii], *noises[name])
                    row.update({name+'_u_mse': float(result['mean_mse']), name+'_incremental_mse': float(result['incremental_mse']),
                        name+'_gamma_crps': float(scored['gamma_crps']), name+'_normalized_gamma_crps': float(scored['normalized_gamma_crps'])})
            score = row['validation_u_mse']
            payload = checkpoint_payload(model, optimizer, scheduler, order_rng, epoch, steps, score, ids[fit], ids[valid])
            payload.update(training_config=cfg, training_mc_rng_state=mc_rng.get_state(),
                gamma_objective_state_dict=deepcopy(objective.state_dict()), monitoring_rng_state=monitor_rng.get_state(),
                fixed_monitoring_seed=seed+cfg['validation_mc_offset'], baseline_reference='A_OLD_GENERIC epoch30', loss_mode='weighted')
            torch.save(payload, folder/f'epoch{epoch}.pt')
            torch.save(payload, folder/'last.pt')
            if score < best_score:
                best_score, best_epoch = score, epoch
                torch.save(payload, folder/'best.pt')
            row['best_epoch'] = best_epoch
            with (folder/'gamma_monitoring.jsonl').open('a') as stream:
                stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
            print(json.dumps(dict(arm=folder.name, epoch=epoch, fit_mse=row['fit_u_mse'],
                                 validation_mse=score, elapsed_seconds=row['elapsed_seconds'])), flush=True)
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
    if any(not torch.equal(v, model.base_a.state_dict()[k]) for k, v in base_before.items()):
        raise RuntimeError('Frozen A changed')
    if any(not torch.equal(v, objective.state_dict()[k]) for k, v in objective_before.items()):
        raise RuntimeError('Frozen joint objective changed')
    if any(not torch.equal(v, dict(model.named_buffers())[k]) for k, v in fixed_before.items()):
        raise RuntimeError('Fixed input transformation changed')
    with torch.no_grad():
        if not torch.equal(model(x, packed, mask, branch_enabled=False), baseline):
            raise RuntimeError('Disabled branch differs from A')
    write_json(folder/'training_complete.json', dict(actual_checkpoint_epoch=epochs, final_epoch=epochs,
        optimizer_steps=steps, best_epoch=best_epoch, best_validation_u_mse=best_score,
        reference_A_fit_mse=references['fit'], reference_A_validation_mse=references['validation'],
        final_fit_u_mse=row['fit_u_mse'], final_validation_u_mse=row['validation_u_mse'],
        trainable_parameters=sum(p.numel() for p in params), elapsed_seconds=time.monotonic()-started,
        frozen_A_changed=False, objective_buffers_changed=False, fixed_input_buffers_changed=False,
        disabled_equals_A=True, stop_reason=f'requested_fixed_{epochs}_epochs'))
    return model
