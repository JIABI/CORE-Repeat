"""Same conditional mean, original geometry loss versus added Gamma CRPS."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import plain, write_json
from .gram_experiment import _schedule
from .gram_oof_experiment import now, event
from .gram_oof_ridge import transform_input, transform_target
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import fit_score_scale
from .hierarchical_stability_experiment import validate_partition, _check_cohort
from .conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from .conditional_response_experiment import CONFIG as MODEL_CONFIG, train_branch as train_control, score_branch
from .geometry_kernel_replacement_experiment import load_frozen_fold, predict_branch, checkpoint_payload
from .gamma_supervised_loss import JointGammaCRPS


PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('A_HR', 'J_GEOMETRY_CONTROL', 'K_GEOMETRY_GAMMA_CRPS')
CONFIG = dict(MODEL_CONFIG, gamma_weight=1., train_pairs=64, validation_pairs=128,
    gamma_scale_ddof=1, training_mc_offset=53000, validation_mc_offset=51000,
    gradient_mc_offset=52000)


def prepare(output, historical):
    root, historical = Path(output).resolve(), Path(historical).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh Gamma-supervision directory')
    old = json.loads((historical/'run_manifest.json').read_text())
    if json.loads((historical/'status.json').read_text())['state'] != 'COMPLETE' or old['config'] != MODEL_CONFIG:
        raise ValueError('The completed conditional-response experiment is required')
    ds, split, _ = _load_study_data(old['data_directory'])
    _check_cohort(ds, split, old)
    for record in old['folds']:
        validate_partition(record, len(ds))
        load_frozen_fold(old['reference_run'], record)
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/'protocols/historical/GAMMA_SUPERVISION_PLAN_20260914.md', root/'PROTOCOL.md')
    manifest = deepcopy(old)
    manifest.update(created_utc=now(), source_snapshot=str(snapshot), python_executable=sys.executable,
        historical_reference_run=str(historical), config=CONFIG, arms=list(ARMS),
        architecture_changed=False, geometry_loss_changed=False, gamma_supervision_added_to_K=True,
        covariance_updated=False, checkpoint_policy='fixed epoch30; best geometry validation descriptive only',
        control_reference_arm='G_CONDITIONAL_STRUCTURED', control_exact_reproduction_required=True,
        target_changed=False, gamma_training_target='original per-object ADD_TWO realized Gamma',
        gamma_scale='TRAIN-only sample SD, ddof=1', gamma_scale_floor=None,
        lambda_selected_using_validation=False, train_pair_draws_independent=True,
        training_initialization='same historical G epoch0, not continued from its epoch30',
        monitoring='Gamma metrics and gradient components at each saved five-epoch checkpoint')
    from .gamma_supervised_summary import _validate_manifest
    _validate_manifest(manifest)
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=639, new_branch_fits=10, stage_epochs=30)
    return manifest


def load_run(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['config'] != CONFIG:
        raise ValueError('Execute the frozen experiment source and configuration')
    old = json.loads((Path(manifest['historical_reference_run'])/'run_manifest.json').read_text())
    for key in ('ids', 'folds', 'reference_run', 'data_directory'):
        if manifest[key] != old[key]:
            raise ValueError('Historical data or partition changed: '+key)
    ds, split, _ = _load_study_data(manifest['data_directory'])
    _check_cohort(ds, split, manifest)
    return root, manifest, ds


def independent_noise(pairs, n, generator):
    shape = (pairs, n, 9)
    return (torch.randn(shape, dtype=torch.float64, generator=generator),
            torch.randn(shape, dtype=torch.float64, generator=generator))


def train_supervised(folder, model, objective, x, chem, mask, target, gamma, fit, valid, seed, ids):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Existing Gamma training is preserved')
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    x, chem = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(chem, dtype=torch.float64)
    mask, target = torch.as_tensor(mask, dtype=torch.bool), torch.as_tensor(target, dtype=torch.float64)
    gamma = torch.as_tensor(gamma, dtype=torch.float64)
    params = list(model.trainable_parameters())
    before = {key: value.detach().clone() for key, value in model.state_dict().items()
              if key.startswith(('base_hr.', 'bank.'))}
    objective_before = deepcopy(objective.state_dict())
    optimizer = torch.optim.AdamW(params, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    total = CONFIG['max_epochs']*math.ceil(len(fit)/CONFIG['batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, total, CONFIG))
    order_rng = torch.Generator().manual_seed(seed+31)
    mc_rng = torch.Generator().manual_seed(seed+CONFIG['training_mc_offset'])
    steps, best_epoch, best_score = 0, 0, float('inf')
    started = time.monotonic()
    for epoch in range(CONFIG['stage_epochs']+1):
        sums = dict(mean_mse=0., incremental_mse=0., gamma_crps=0., normalized_gamma_crps=0., total_loss=0.)
        count, norms = 0, []
        if epoch:
            model.train()
            order = np.asarray(fit)[torch.randperm(len(fit), generator=order_rng).numpy()]
            for first in range(0, len(order), CONFIG['batch_size']):
                ii = order[first:first+CONFIG['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                original = model.loss(x[ii], chem[ii], mask[ii], target[ii])
                a, b = independent_noise(CONFIG['train_pairs'], len(ii), mc_rng)
                supervised = objective(model(x[ii], chem[ii], mask[ii]), gamma[ii], a, b)
                total_loss = original['loss']+CONFIG['gamma_weight']*supervised['normalized_gamma_crps']
                if not torch.isfinite(total_loss):
                    raise FloatingPointError('Nonfinite combined geometry/Gamma objective')
                total_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, CONFIG['gradient_clip'], error_if_nonfinite=True)
                norms.append(float(norm))
                optimizer.step(); scheduler.step(); steps += 1
                for key in ('mean_mse', 'incremental_mse'):
                    sums[key] += len(ii)*float(original[key].detach())
                for key in ('gamma_crps', 'normalized_gamma_crps'):
                    sums[key] += len(ii)*float(supervised[key].detach())
                sums['total_loss'] += len(ii)*float(total_loss.detach())
                count += len(ii)
        row = dict(epoch=epoch, optimizer_steps=steps, learning_rate=optimizer.param_groups[0]['lr'],
            **{'train_minibatch_'+key: value/count if count else None for key, value in sums.items()},
            gradient_norm_mean=float(np.mean(norms)) if norms else None,
            gradient_norm_max=float(np.max(norms)) if norms else None,
            elapsed_seconds=time.monotonic()-started)
        if epoch % CONFIG['validation_interval'] == 0:
            model.eval()
            with torch.no_grad():
                fl = model.loss(x[fit], chem[fit], mask[fit], target[fit])
                vl = model.loss(x[valid], chem[valid], mask[valid], target[valid])
            score = float(vl['mean_mse'])
            row.update(fit_u_mse=float(fl['mean_mse']), validation_u_mse=score,
                fit_incremental_mse=float(fl['incremental_mse']), validation_incremental_mse=float(vl['incremental_mse']))
            payload = checkpoint_payload(model, optimizer, scheduler, order_rng, epoch, steps, score, ids[fit], ids[valid])
            payload.update(training_mc_rng_state=mc_rng.get_state(), gamma_objective_state_dict=objective.state_dict(),
                gamma_weight=CONFIG['gamma_weight'], gamma_train_pairs=CONFIG['train_pairs'])
            torch.save(payload, folder/f'epoch{epoch}.pt')
            torch.save(payload, folder/'last.pt')
            if score < best_score:
                best_score, best_epoch = score, epoch
                torch.save(payload, folder/'best.pt')
            row['best_epoch'] = best_epoch
            event(folder, 'VALIDATED', **row)
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
    model.eval()
    if any(not torch.equal(model.state_dict()[key], value) for key, value in before.items()):
        raise RuntimeError('The frozen HR or response bank changed')
    if any(not torch.equal(objective.state_dict()[key], value) for key, value in objective_before.items()):
        raise RuntimeError('The frozen covariance or target scales changed')
    completion = dict(epoch=30, final_epoch=30, actual_checkpoint_epoch=30, best_epoch=best_epoch,
        best_validation_u_mse=best_score, optimizer_steps=steps, elapsed_seconds=time.monotonic()-started,
        stop_reason='requested_fixed_epoch30_loss_comparison', converged_claim=False,
        frozen_hr_changed=False, bank_changed=False, objective_buffers_changed=False,
        gamma_weight=CONFIG['gamma_weight'], trainable_parameters=sum(p.numel() for p in params),
        parameter_counts=dict(trainable=sum(p.numel() for p in params), total=sum(p.numel() for p in model.parameters())))
    write_json(folder/'training_complete.json', completion)
    event(folder, 'STAGE_COMPLETE', **completion)
    return model


def monitor_checkpoints(folder, objective, x, chem, mask, target, gamma, fit, valid, seed):
    """Evaluate saved states only; these readouts do not alter training/selection."""
    folder = Path(folder)
    x, chem = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(chem, dtype=torch.float64)
    mask, target = torch.as_tensor(mask, dtype=torch.bool), torch.as_tensor(target, dtype=torch.float64)
    gamma = torch.as_tensor(gamma, dtype=torch.float64)
    fixed_rng = torch.Generator().manual_seed(seed+CONFIG['validation_mc_offset'])
    fit_noise = independent_noise(CONFIG['validation_pairs'], len(fit), fixed_rng)
    valid_noise = independent_noise(CONFIG['validation_pairs'], len(valid), fixed_rng)
    ii = np.asarray(fit)[:CONFIG['batch_size']]
    diag_rng = torch.Generator().manual_seed(seed+CONFIG['gradient_mc_offset'])
    diag_noise = independent_noise(CONFIG['validation_pairs'], len(ii), diag_rng)
    rows = []
    for epoch in range(0, CONFIG['stage_epochs']+1, CONFIG['validation_interval']):
        saved = torch.load(folder/f'epoch{epoch}.pt', map_location='cpu', weights_only=True)
        model = ConditionalResponseKernelMean.from_config(saved['model_config'])
        model.load_state_dict(saved['state_dict']); model.eval()
        row = dict(epoch=epoch, checkpoint_readout_only=True, fixed_pairs=CONFIG['validation_pairs'])
        with torch.no_grad():
            for name, indexes, noise in (('fit', fit, fit_noise), ('validation', valid, valid_noise)):
                original = model.loss(x[indexes], chem[indexes], mask[indexes], target[indexes])
                result = objective(model(x[indexes], chem[indexes], mask[indexes]), gamma[indexes], *noise)
                row.update({name+'_u_mse': float(original['mean_mse']),
                    name+'_incremental_penalty': float(original['incremental_penalty']),
                    name+'_gamma_crps': float(result['gamma_crps']),
                    name+'_normalized_gamma_crps': float(result['normalized_gamma_crps']),
                    name+'_gamma_mse': float((result['gamma_prediction_mean']-gamma[indexes]).square().mean())})
        original = model.loss(x[ii], chem[ii], mask[ii], target[ii])
        result = objective(model(x[ii], chem[ii], mask[ii]), gamma[ii], *diag_noise)
        params = list(model.trainable_parameters())
        g = torch.autograd.grad(original['mean_mse'], params, allow_unused=True)
        h = torch.autograd.grad(result['normalized_gamma_crps'], params, allow_unused=True)
        gv = torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1) for v, p in zip(g, params)])
        hv = torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1) for v, p in zip(h, params)])
        gn, hn = float(gv.norm()), float(hv.norm())
        row.update(geometry_gradient_norm=gn, normalized_gamma_gradient_norm=hn,
            gradient_cosine=float(gv@hv)/(gn*hn) if gn*hn > 0 else None,
            gradient_scope='first 64 fit objects, same fixed 128 independent pairs; not used for weighting')
        rows.append(row)
    with (folder/'gamma_monitoring.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')
    return rows


def execute_fold(root, manifest, ds, record, grams, raw_u, gains):
    fold = record['fold']
    folder = root/'folds'/f'fold_{fold}'
    old = Path(manifest['historical_reference_run'])/'folds'/f'fold_{fold}'
    folder.mkdir(parents=True)
    fit, valid, test = validate_partition(record, len(ds))
    _, stats, ridge, hr, _ = load_frozen_fold(manifest['reference_run'], record)
    x, target = transform_input(ds.Y[:,0], stats), transform_target(raw_u, stats)
    bank = LocalResponseBank.load(old/'response_bank.pt').double()
    if bank.config['fitting_ids'] != sorted(ds.ids[fit].tolist()):
        # DescriptorBank stores a canonical sorted fit identity list.
        if set(bank.config['fitting_ids']) != set(ds.ids[fit].tolist()):
            raise ValueError('The inherited response bank fitted different compounds')
    scale = float(np.std(gains[fit,2], ddof=CONFIG['gamma_scale_ddof']))
    objective = JointGammaCRPS(ridge.covariance, stats['u_center'], stats['u_scale'], scale)
    write_json(folder/'gamma_objective.json', dict(gamma_scale=scale, scale_ddof=1,
        fitting_ids=ds.ids[fit].tolist(), train_pairs=CONFIG['train_pairs'], validation_pairs=CONFIG['validation_pairs'],
        covariance_frozen=True, gamma_weight=1., center=stats['u_center'], scale=stats['u_scale']))
    torch.save(objective.state_dict(), folder/'gamma_objective_state.pt')
    bank.save(folder/'response_bank.pt')
    shutil.copytree(old/'arms/A_HR/evaluation', folder/'arms/A_HR/evaluation')
    seed = record['seed']+CONFIG['branch_seed_offset']
    initial = torch.load(old/'arms/G_CONDITIONAL_STRUCTURED/epoch0.pt', map_location='cpu', weights_only=True)
    models = {}
    for arm in ARMS[1:]:
        torch.manual_seed(seed)
        model = ConditionalResponseKernelMean(hr, bank, mode='conditional_structured',
            incremental_penalty=CONFIG['incremental_penalty'], hidden_dim=CONFIG['hidden_dim']).double()
        if model.state_dict().keys() != initial['state_dict'].keys() or any(
                not torch.equal(value, initial['state_dict'][key]) for key, value in model.state_dict().items()):
            raise ValueError('New initial parameters differ from historical G epoch0')
        models[arm] = model
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    event(root, 'CONTROL_TRAINING', fold=fold, arm=ARMS[1], epochs=30)
    models[ARMS[1]] = train_control(folder/'arms'/ARMS[1], models[ARMS[1]], x[joined], ds.chem[joined],
        ds.chem_mask[joined], target[joined], fit_local, valid_local, seed, ds.ids[joined])
    previous = torch.load(old/'arms/G_CONDITIONAL_STRUCTURED/epoch30.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(value, previous['state_dict'][key]) for key, value in models[ARMS[1]].state_dict().items()):
        raise RuntimeError('Geometry control failed exact historical G epoch30 reproduction')
    write_json(folder/'control_reproduction.json', dict(epoch0_exact=True, epoch30_exact=True,
        all_model_tensors_compared=True, reference=str(old/'arms/G_CONDITIONAL_STRUCTURED/epoch30.pt')))
    event(root, 'GAMMA_SUPERVISED_TRAINING', fold=fold, arm=ARMS[2], epochs=30)
    models[ARMS[2]] = train_supervised(folder/'arms'/ARMS[2], models[ARMS[2]], objective,
        x[joined], ds.chem[joined], ds.chem_mask[joined], target[joined], gains[joined,2],
        fit_local, valid_local, seed, ds.ids[joined])
    metric_scale = fit_score_scale(grams[fit])
    for arm in ARMS[1:]:
        event(root, 'CHECKPOINT_MONITORING', fold=fold, arm=arm)
        monitor_checkpoints(folder/'arms'/arm, objective, x[joined], ds.chem[joined], ds.chem_mask[joined],
            target[joined], gains[joined,2], fit_local, valid_local, seed)
        event(root, 'SCORING_EPOCH30', fold=fold, arm=arm)
        model = models[arm]
        mean = predict_branch(model, x[test], ds.chem[test], ds.chem_mask[test])
        saved = torch.load(folder/'arms'/arm/'epoch30.pt', map_location='cpu', weights_only=True)
        reloaded = ConditionalResponseKernelMean.from_config(saved['model_config'])
        reloaded.load_state_dict(saved['state_dict'])
        if not np.array_equal(mean, predict_branch(reloaded, x[test], ds.chem[test], ds.chem_mask[test])):
            raise ValueError('Saved checkpoint cannot reproduce the scored mean')
        with torch.no_grad():
            d = model.diagnostics(torch.as_tensor(x[test], dtype=torch.float64),
                torch.as_tensor(ds.chem[test], dtype=torch.float64), torch.as_tensor(ds.chem_mask[test]),
                ids=ds.ids[test].tolist())
        values = {key: value.detach().cpu().numpy() for key, value in d.items() if isinstance(value, torch.Tensor)}
        values['raw'] = values['kernel_raw']
        np.savez_compressed(folder/'arms'/arm/'model_diagnostics.npz', ids=ds.ids[test],
            block_names=np.asarray(['chemical', 'morphology', 'scalar']), **values)
        counts = dict(trainable=sum(p.numel() for p in model.trainable_parameters()),
                      total=sum(p.numel() for p in model.parameters()))
        score_branch(folder/'arms'/arm/'evaluation', mean, ridge, stats, target[test], grams[test],
            ds.ids[test], gains[fit], metric_scale, record['seed'], arm, counts)
    write_json(folder/'complete.json', dict(fold=fold, test_n=len(test), actual_checkpoint_epoch=30,
        completed_utc=now(), control_exact_reproduction=True, formal_certificate=False))
    event(root, 'FOLD_COMPLETE', fold=fold, actual_checkpoint_epoch=30)


def execute(output):
    root, manifest, ds = load_run(output)
    if (root/'folds').exists():
        raise FileExistsError('This run has already started')
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root, 'RUNNING', new_branch_fits=10, epochs=30)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams = profiles_to_gram(torch.as_tensor(ds.Y, dtype=torch.float64)).numpy()
            raw_u = gram_to_coordinates(torch.as_tensor(grams)).numpy()
            gains = gram_gains(torch.as_tensor(grams)).numpy()
            for record in manifest['folds']:
                execute_fold(root, manifest, ds, record, grams, raw_u, gains)
            from .gamma_supervised_summary import summarize
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-started, stopped_at_epoch=30,
            further_training_started=False)
    except Exception as error:
        event(root, 'FAILED', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'execute', 'summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--historical', default=str(PROJECT/'runs/conditional_response_epoch30_20260914_v1'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.historical)
    elif args.mode == 'execute':
        execute(args.output)
    else:
        from .gamma_supervised_summary import summarize
        summarize(args.output)


if __name__ == '__main__':
    main()
