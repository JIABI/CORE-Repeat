"""Fixed 30-epoch conditional local-response development comparison."""
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
from .gram_evaluation import evaluate_and_save, fit_score_scale
from .gram_factor_verified import decode_draws
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import restore_target, gaussian_coordinate_diagnostics
from .hierarchical_stability_experiment import validate_partition, _check_cohort
from .geometry_kernel_replacement import ReplacementBasisBank
from .geometry_kernel_replacement_experiment import (
    CONFIG as PREVIOUS_CONFIG, load_frozen_fold, predict_branch,
    checkpoint_payload, support_diagnostics,
)


PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('A_HR', 'E_STATIC_STRUCTURED', 'F_CONDITIONAL_GENERIC', 'G_CONDITIONAL_STRUCTURED')
MODES = dict(E_STATIC_STRUCTURED='static_structured',
    F_CONDITIONAL_GENERIC='conditional_generic', G_CONDITIONAL_STRUCTURED='conditional_structured')
CONFIG = {key: value for key, value in PREVIOUS_CONFIG.items() if key != 'kan_hidden_dim'}
CONFIG.update(stage_epochs=30, hidden_dim=16)


def prepare(output, historical):
    root, historical = Path(output).resolve(), Path(historical).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh conditional-response run directory')
    old = json.loads((historical/'run_manifest.json').read_text())
    if (json.loads((historical/'status.json').read_text())['state'] != 'COMPLETE'
            or old['actual_checkpoint_epoch'] != 30):
        raise ValueError('Historical comparison must be the completed epoch30 experiment')
    ds, split, _ = _load_study_data(old['data_directory'])
    _check_cohort(ds, split, old)
    for record in old['folds']:
        validate_partition(record, len(ds))
        load_frozen_fold(old['reference_run'], record)
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/'protocols/historical/CONDITIONAL_RESPONSE_PLAN_20260914.md', root/'PROTOCOL.md')
    manifest = deepcopy(old)
    for key in ('continuation_source', 'continuation_start_epoch', 'additional_optimizer_steps',
                'continuation_decided_after_epoch10_dev'):
        manifest.pop(key, None)
    manifest.update(created_utc=now(), config=CONFIG, arms=list(ARMS),
        source_snapshot=str(snapshot), python_executable=sys.executable,
        historical_reference_run=str(historical), actual_checkpoint_epoch=30,
        checkpoint_policy='fixed actual epoch30; validation-best descriptive only',
        architecture_changed=True, optimizer_restarted=True,
        training_initialization='new branch, exact frozen HR at epoch0',
        scheduler_horizon_epochs=100, planned_total_optimizer_steps=210,
        branch_design='local response plus small conditional coefficient mixer and bias-free linear readout',
        replacement_scope='new branch only; original HR MLP remains frozen',
        common_morphology_basis='q,tanh(2q),q*abs(q)',
        common_scalar_basis='v,tanh(v),v/hypot(v,1)',
        conditional_generic_structured_parameter_matched=True,
        static_conditional_parameter_matched=False,
        conditioner_output='three basis gates per chemical, morphology and scalar block',
        no_raw_descriptor_bypass=True, no_per_object_normalization=True,
        geometry_loss_changed=False, covariance_updated=False,
        local_response_not_verified_biological_mechanism=True)
    from .conditional_response_summary import _validate_manifest
    _validate_manifest(manifest)
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=639, new_branch_fits=15, stage_epochs=30)
    return manifest


def load_run(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['config'] != CONFIG:
        raise ValueError('Execute the frozen source snapshot and recorded configuration')
    old = json.loads((Path(manifest['historical_reference_run'])/'run_manifest.json').read_text())
    for key in ('ids', 'folds', 'reference_run', 'data_directory'):
        if manifest[key] != old[key]:
            raise ValueError('The original experiment changed '+key)
    ds, split, _ = _load_study_data(manifest['data_directory'])
    _check_cohort(ds, split, manifest)
    return root, manifest, ds


def train_branch(folder, model, x, chem, mask, target, fit, valid, seed, ids):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Existing training is preserved')
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    x, chem = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(chem, dtype=torch.float64)
    mask, target = torch.as_tensor(mask, dtype=torch.bool), torch.as_tensor(target, dtype=torch.float64)
    params = list(model.trainable_parameters())
    before = {key: value.detach().clone() for key, value in model.state_dict().items()
              if key.startswith(('base_hr.', 'bank.'))}
    optimizer = torch.optim.AdamW(params, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    total = CONFIG['max_epochs']*math.ceil(len(fit)/CONFIG['batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, total, CONFIG))
    order_rng = torch.Generator().manual_seed(seed+31)
    steps, best_epoch, best_score = 0, 0, float('inf')
    started = time.monotonic()
    for epoch in range(CONFIG['stage_epochs']+1):
        total_mse, total_increment, count, norms, part_norms = 0., 0., 0, [], {}
        if epoch:
            model.train()
            order = np.asarray(fit)[torch.randperm(len(fit), generator=order_rng).numpy()]
            for first in range(0, len(order), CONFIG['batch_size']):
                ii = order[first:first+CONFIG['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                losses = model.loss(x[ii], chem[ii], mask[ii], target[ii])
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError('Nonfinite conditional-response objective')
                losses['loss'].backward()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        part_norms.setdefault(name, []).append(float(parameter.grad.norm()))
                norm = torch.nn.utils.clip_grad_norm_(params, CONFIG['gradient_clip'], error_if_nonfinite=True)
                norms.append(float(norm))
                optimizer.step(); scheduler.step(); steps += 1
                total_mse += len(ii)*float(losses['mean_mse'].detach())
                total_increment += len(ii)*float(losses['incremental_mse'].detach())
                count += len(ii)
        row = dict(epoch=epoch, optimizer_steps=steps, learning_rate=optimizer.param_groups[0]['lr'],
            train_minibatch_mse=total_mse/count if count else None,
            train_minibatch_incremental_mse=total_increment/count if count else None,
            gradient_norm_mean=float(np.mean(norms)) if norms else None,
            gradient_norm_max=float(np.max(norms)) if norms else None,
            gradient_parameter_norm_mean={key: float(np.mean(value)) for key, value in part_norms.items()},
            elapsed_seconds=time.monotonic()-started)
        if epoch % CONFIG['validation_interval'] == 0:
            model.eval()
            with torch.no_grad():
                fl = model.loss(x[fit], chem[fit], mask[fit], target[fit])
                vl = model.loss(x[valid], chem[valid], mask[valid], target[valid])
            score = float(vl['mean_mse'])
            row.update(fit_u_mse=float(fl['mean_mse']), validation_u_mse=score,
                fit_incremental_mse=float(fl['incremental_mse']),
                validation_incremental_mse=float(vl['incremental_mse']))
            payload = checkpoint_payload(model, optimizer, scheduler, order_rng,
                epoch, steps, score, ids[fit], ids[valid])
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
    changed = [name for name, tensor in model.state_dict().items()
               if name in before and not torch.equal(tensor, before[name])]
    if changed:
        raise RuntimeError('A frozen baseline or fitted descriptor changed: '+str(changed))
    completion = dict(epoch=30, actual_checkpoint_epoch=30, best_epoch=best_epoch,
        best_validation_u_mse=best_score, optimizer_steps=steps,
        elapsed_seconds=time.monotonic()-started, stop_reason='requested_fixed_epoch30_readout',
        converged_claim=False, frozen_hr_changed=False, bank_changed=False,
        trainable_parameters=sum(p.numel() for p in params),
        parameter_counts=dict(trainable=sum(p.numel() for p in params),
                              total=sum(p.numel() for p in model.parameters())))
    write_json(folder/'training_complete.json', completion)
    event(folder, 'STAGE_COMPLETE', **completion)
    return model


def score_branch(folder, mean, ridge, stats, target, actual_grams, ids,
                 train_gains, metric_scale, seed, arm, counts):
    diagnostics = gaussian_coordinate_diagnostics(folder, ids, target, mean, ridge.covariance)
    samples = sample_joint_coordinates(mean, ridge.covariance, CONFIG['samples'], seed+200000)
    draws, audit = decode_draws(restore_target(samples, stats), verify=True)
    return evaluate_and_save(folder, draws, actual_grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=30, selection='fixed epoch30',
            backbone='immutable historical HR_VALID_S0', covariance='exact RIDGE_VALID OOF covariance',
            numerics=audit, u_diagnostics=diagnostics, parameter_counts=counts,
            formal_certificate=False, physical_variance_identified=False, historical_dev=True),
        train_actual_gains=train_gains, score_scale=metric_scale, seed=seed,
        n_bootstrap=CONFIG['bootstrap'], n_random=CONFIG['random_subsets'])


def execute_fold(root, manifest, ds, record, grams, raw_u, gains):
    from .conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean

    fold = record['fold']
    folder = root/'folds'/f'fold_{fold}'
    old = Path(manifest['historical_reference_run'])/'folds'/f'fold_{fold}'
    folder.mkdir(parents=True)
    fit, valid, test = validate_partition(record, len(ds))
    _, stats, ridge, hr, payload = load_frozen_fold(manifest['reference_run'], record)
    x, target = transform_input(ds.Y[:,0], stats), transform_target(raw_u, stats)
    with torch.no_grad():
        hr_mean = hr(torch.as_tensor(x, dtype=torch.float64)).numpy()
    with np.load(old/'arms/A_HR/test/u_predictions.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['ids'], ds.ids[test]) or not np.array_equal(saved['actual_u'], target[test])
                or not np.allclose(saved['mean_u'], hr_mean[test], atol=1e-12, rtol=1e-12)
                or not np.array_equal(saved['covariance_u'],
                    np.broadcast_to(ridge.covariance, (len(test), 9, 9)))):
            raise ValueError('Frozen HR, target, covariance or held-out identities changed')
    shutil.copytree(old/'arms/A_HR/test', folder/'arms/A_HR/evaluation')
    bank = LocalResponseBank.fit(x, ds.chem, ds.chem_mask, ds.ids, ds.ids[fit].tolist(),
        metadata=ds.metadata.get('chemical') or None, max_anchors=CONFIG['max_anchors']).double()
    old_bank = ReplacementBasisBank.from_config(json.loads((old/'descriptor_config.json').read_text())).double()
    old_bank.load_state_dict(torch.load(old/'descriptor_state.pt', map_location='cpu', weights_only=True))
    before, after = old_bank.descriptor_bank.state_dict(), bank.descriptor_bank.state_dict()
    if before.keys() != after.keys() or any(not torch.equal(before[key], after[key]) for key in before):
        raise ValueError('The historical TRAIN anchors or descriptor scalers changed')
    bank.save(folder/'response_bank.pt')
    write_json(folder/'descriptor_config.json', bank.config)
    write_json(folder/'support_diagnostics.json', {name: support_diagnostics(bank,
        x[ii], ds.chem[ii], ds.chem_mask[ii], ds.ids[ii])
        for name, ii in (('fit', fit), ('validation', valid), ('test', test))})
    write_json(folder/'baseline_restore.json', dict(frozen_hr_epoch=payload['epoch'],
        original_prediction_verified=True, covariance_unchanged=True, historical_descriptors_exact=True))
    seed = record['seed']+CONFIG['branch_seed_offset']
    counts, models = {}, {}
    for arm in ARMS[1:]:
        torch.manual_seed(seed)
        model = ConditionalResponseKernelMean(hr, bank, mode=MODES[arm],
            incremental_penalty=CONFIG['incremental_penalty'], hidden_dim=CONFIG['hidden_dim']).double()
        if not np.array_equal(predict_branch(model, x, ds.chem, ds.chem_mask), hr_mean):
            raise ValueError('New branch must initially reproduce HR exactly')
        counts[arm] = dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          total=sum(p.numel() for p in model.parameters()))
        models[arm] = model
    if counts[ARMS[2]] != counts[ARMS[3]]:
        raise ValueError('Conditional generic and structured capacity differs')
    write_json(folder/'initialization.json', dict(parameter_counts=counts, zero_initialization_exact=True,
        training_seed=seed, covariance_unchanged=True, generic_structured_parameter_matched=True))
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    for arm in ARMS[1:]:
        event(root, 'BRANCH_TRAINING', fold=fold, arm=arm, stage_epochs=30)
        models[arm] = train_branch(folder/'arms'/arm, models[arm], x[joined], ds.chem[joined],
            ds.chem_mask[joined], target[joined], fit_local, valid_local, seed, ds.ids[joined])
    scale = fit_score_scale(grams[fit])
    for arm in ARMS[1:]:
        event(root, 'SCORING_EPOCH30', fold=fold, arm=arm)
        model = models[arm]
        mean = predict_branch(model, x[test], ds.chem[test], ds.chem_mask[test])
        saved = torch.load(folder/'arms'/arm/'epoch30.pt', map_location='cpu', weights_only=True)
        reloaded = ConditionalResponseKernelMean.from_config(saved['model_config'])
        reloaded.load_state_dict(saved['state_dict'])
        if not np.array_equal(mean, predict_branch(reloaded, x[test], ds.chem[test], ds.chem_mask[test])):
            raise ValueError('Reloaded checkpoint does not reproduce scored predictions')
        with torch.no_grad():
            diagnostics = model.diagnostics(torch.as_tensor(x[test], dtype=torch.float64),
                torch.as_tensor(ds.chem[test], dtype=torch.float64), torch.as_tensor(ds.chem_mask[test]),
                ids=ds.ids[test].tolist())
        values = {key: value.detach().cpu().numpy() for key, value in diagnostics.items()
                  if isinstance(value, torch.Tensor)}
        values['raw'] = values['kernel_raw']
        np.savez_compressed(folder/'arms'/arm/'model_diagnostics.npz', ids=ds.ids[test],
            block_names=np.asarray(['chemical', 'morphology', 'scalar']), **values)
        score_branch(folder/'arms'/arm/'evaluation', mean, ridge, stats, target[test], grams[test],
            ds.ids[test], gains[fit], scale, record['seed'], arm, counts[arm])
    write_json(folder/'complete.json', dict(fold=fold, test_n=len(test), actual_checkpoint_epoch=30,
        completed_utc=now(), formal_certificate=False))
    event(root, 'FOLD_COMPLETE', fold=fold, actual_checkpoint_epoch=30)


def execute(output):
    root, manifest, ds = load_run(output)
    if (root/'folds').exists():
        raise FileExistsError('Training has already started in this directory')
    torch.set_num_threads(CONFIG['threads'])
    start = time.monotonic()
    event(root, 'RUNNING', new_branch_fits=15, target_epoch=30)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams = profiles_to_gram(torch.as_tensor(ds.Y, dtype=torch.float64)).numpy()
            raw_u = gram_to_coordinates(torch.as_tensor(grams)).numpy()
            gains = gram_gains(torch.as_tensor(grams)).numpy()
            for record in manifest['folds']:
                execute_fold(root, manifest, ds, record, grams, raw_u, gains)
            from .conditional_response_summary import summarize
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-start, stopped_at_epoch=30,
            further_training_started=False)
    except Exception as error:
        event(root, 'FAILED', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic()-start)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'execute', 'summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--historical', default=str(PROJECT/'runs/geometry_kernel_replacement_epoch30_20260914_v1'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.historical)
    elif args.mode == 'execute':
        execute(args.output)
    else:
        from .conditional_response_summary import summarize
        summarize(args.output)


if __name__ == '__main__':
    main()
