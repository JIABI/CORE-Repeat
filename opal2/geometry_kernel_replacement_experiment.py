"""Frozen HR plus MLP-only or a single-path kernel replacement on opened DEV.

The first stage reports all new branches at epoch 10, not the test-best checkpoint.
The full response layer is trained; the shortened budget is explicitly an early
development readout. Original HR and its joint covariance are never optimized.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
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
from .gram_simple_models import GramSimpleGaussian
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import evaluate_and_save, fit_score_scale
from .gram_factor_verified import decode_draws
from .hierarchical_geometry import RidgeResidualMean, sample_joint_coordinates
from .hierarchical_geometry_experiment import restore_target, gaussian_coordinate_diagnostics
from .hierarchical_stability_experiment import validate_partition, _check_cohort


PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('A_HR', 'B_MLP', 'C_GENERIC', 'D_STRUCTURED')
CONFIG = dict(seed=20260914, repeat=0, folds=5, hr_training_seed_index=0,
    branch_seed_offset=7401, stage_epochs=10, max_epochs=100,
    validation_interval=5, batch_size=64, learning_rate=.0003,
    min_learning_rate=.000003, warmup_steps=10, weight_decay=.0001,
    gradient_clip=5., incremental_penalty=.1, max_anchors=64,
    hidden_dim=32, kan_hidden_dim=24, threads=2, samples=2000,
    bootstrap=2000, random_subsets=2000)


def load_frozen_fold(reference, record):
    folder = Path(reference)/'repetitions/repeat_0/folds'/f"fold_{record['fold']}"
    stats = json.loads((folder/'preprocessing.json').read_text())
    ridge = GramSimpleGaussian.load(folder/'arms/RIDGE_VALID/fit.npz')
    payload = torch.load(folder/'arms/HR_VALID_S0/best.pt', map_location='cpu', weights_only=True)
    state = payload['state_dict']
    model = RidgeResidualMean.from_config(payload['model_config'],
        coefficient=state['coefficient'], intercept=state['intercept'])
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    if (payload['fit_ids'] != record['fit_ids']
            or payload['validation_ids'] != record['inner_validation_ids']
            or not np.array_equal(model.coefficient.numpy(), ridge.coefficient)
            or not np.array_equal(model.intercept.numpy(), ridge.intercept)):
        raise ValueError('Frozen HR parameters or fitting identities differ from the reference')
    return folder, stats, ridge, model, payload


def prepare(output, reference):
    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Use a new directory for the kernel development stage')
    original = json.loads((reference/'run_manifest.json').read_text())
    ds, split, scope = _load_study_data(original['data_directory'])
    _check_cohort(ds, split, original)
    folds = original['repetitions'][CONFIG['repeat']]['folds']
    frozen_epochs = []
    for record in folds:
        validate_partition(record, len(ds))
        *_, payload = load_frozen_fold(reference, record)
        frozen_epochs.append(payload['epoch'])
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/GEOMETRY_KERNEL_REPLACEMENT_PLAN_20260914.md', root/'PROTOCOL.md')
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    manifest = dict(created_utc=now(), reference_run=str(reference),
        source_snapshot=str(snapshot), python_executable=sys.executable,
        config=CONFIG, arms=list(ARMS), folds=folds, ids=original['ids'],
        data_directory=original['data_directory'], data_shape=list(ds.Y.shape),
        feature_names=original['feature_names'], original_compound_ids=original['original_compound_ids'],
        scope=scope, frozen_hr_epochs=frozen_epochs, actual_checkpoint_epoch=10,
        checkpoint_policy='epoch10 for all three new branches; validation-best is supplementary only',
        support_policy='availability gate only; training-self-excluded support diagnostics',
        covariance_policy='unchanged full RIDGE_VALID OOF joint error covariance',
        chemical_scope='structure and observed phenotype; no verified mechanism annotations',
        replacement_scope='new branch only; frozen HR still contains its original MLP',
        branch_design='single explicit basis path to KAN mixing and zero-initialized nine-coordinate readout',
        chemical_basis_domain='same TRAIN-anchor Tanimoto values in generic and structured arms',
        generic_chemical_basis=dict(centers=[0., .5, 1.], width=.25),
        structured_chemical_basis='T,T^2,T^4',
        common_morphology_basis=dict(centers=[-1.,0.,1.],width=.5),
        common_scalar_basis='v,tanh(v),v^2/(1+v^2) on three standardized descriptors',
        basis_scale='TRAIN-only uncentered RMS separately for chemical, morphology and scalar blocks',
        no_raw_descriptor_bypass=True, generic_structured_parameter_matched=True,
        mlp_parameter_matched=False,
        historical_dev=True, independent_new_holdout=False, formal_certificate=False,
        final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False,
        original_contract_changed=False, original_split_files_changed=False,
        jepa_active=False, mechanism_annotations_active=False)
    from .geometry_kernel_replacement_summary import _validate_manifest
    _validate_manifest(manifest)
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=639, branch_fits=15, stage_epochs=10)
    return manifest


def load_run(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['config'] != CONFIG:
        raise ValueError('Execute the recorded source snapshot and configuration')
    ds, split, _ = _load_study_data(manifest['data_directory'])
    _check_cohort(ds, split, manifest)
    if tuple(manifest['arms']) != ARMS:
        raise ValueError('The four declared arms changed')
    return root, manifest, ds


def checkpoint_payload(model, optimizer, scheduler, order_rng, epoch, steps,
                       score, fit_ids, valid_ids):
    return dict(model_config=model.config, state_dict=deepcopy(model.state_dict()),
        epoch=epoch, optimizer_steps=steps, validation_u_mse=score,
        fit_ids=list(map(str, fit_ids)), validation_ids=list(map(str, valid_ids)),
        optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
        torch_rng_state=torch.get_rng_state(), order_rng_state=order_rng.get_state(),
        actual_checkpoint_epoch=epoch)


def train_branch(folder, model, x, chem, mask, target, fit, valid, seed, ids):
    """Fit the new branch only; select no epoch using outer test information."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('An existing branch stage is preserved, not restarted')
    folder.mkdir(parents=True)
    torch.manual_seed(seed)
    x, chem = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(chem, dtype=torch.float64)
    mask, target = torch.as_tensor(mask, dtype=torch.bool), torch.as_tensor(target, dtype=torch.float64)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('The kernel branch has no trainable parameters')
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(params, lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    total = CONFIG['max_epochs']*math.ceil(len(fit)/CONFIG['batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step, total, CONFIG))
    order_rng = torch.Generator().manual_seed(seed+31)
    steps, best_epoch, best_score = 0, 0, float('inf')
    started = time.monotonic()
    event(folder, 'TRAINING_STARTED', trainable_parameters=sum(p.numel() for p in params),
        fit_n=len(fit), validation_n=len(valid), target_epoch=CONFIG['stage_epochs'])
    for epoch in range(CONFIG['stage_epochs']+1):
        weighted_loss, weighted_increment, count, grad_norms = 0., 0., 0, []
        if epoch:
            model.train()
            order = np.asarray(fit)[torch.randperm(len(fit), generator=order_rng).numpy()]
            for first in range(0, len(order), CONFIG['batch_size']):
                ii = order[first:first+CONFIG['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                losses = model.loss(x[ii], chem[ii], mask[ii], target[ii])
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError('Nonfinite kernel objective')
                losses['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(params, CONFIG['gradient_clip'], error_if_nonfinite=True)
                grad_norms.append(float(norm))
                optimizer.step(); scheduler.step(); steps += 1
                weighted_loss += len(ii)*float(losses['mean_mse'].detach())
                weighted_increment += len(ii)*float(losses['incremental_mse'].detach())
                count += len(ii)
        row = dict(epoch=epoch, optimizer_steps=steps,
            learning_rate=optimizer.param_groups[0]['lr'],
            train_minibatch_mse=weighted_loss/count if count else None,
            train_minibatch_incremental_mse=weighted_increment/count if count else None,
            gradient_norm_mean=float(np.mean(grad_norms)) if grad_norms else None,
            gradient_norm_max=float(np.max(grad_norms)) if grad_norms else None,
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
    frozen_changed = [name for name, tensor in model.state_dict().items()
        if name in before and name.startswith('base_hr.') and not torch.equal(tensor, before[name])]
    if frozen_changed:
        raise RuntimeError('Frozen HR changed during kernel training: '+str(frozen_changed))
    completion = dict(epoch=CONFIG['stage_epochs'], actual_checkpoint_epoch=CONFIG['stage_epochs'],
        best_epoch=best_epoch, best_validation_u_mse=best_score, optimizer_steps=steps,
        elapsed_seconds=time.monotonic()-started, stop_reason='planned_epoch10_development_readout',
        converged_claim=False, frozen_hr_changed=False,
        trainable_parameters=sum(p.numel() for p in params),
        parameter_counts=dict(trainable=sum(p.numel() for p in params),
                              total=sum(p.numel() for p in model.parameters())))
    write_json(folder/'training_complete.json', completion)
    event(folder, 'STAGE_COMPLETE', **completion)
    return model


@torch.no_grad()
def predict_branch(model, x, chem, mask):
    model.eval()
    return model(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(chem, dtype=torch.float64),
                 torch.as_tensor(mask, dtype=torch.bool)).numpy()


def support_diagnostics(bank, x, chem, mask, ids):
    with torch.no_grad():
        d = bank(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(chem, dtype=torch.float64),
                 torch.as_tensor(mask, dtype=torch.bool), ids=ids.tolist())
    result = {'n': len(ids)}
    for name in ('availability', 'max_similarity', 'effective_support', 'similarity_mass', 'support_self_excluded'):
        value = d[name].detach().cpu().numpy()
        result[name] = dict(mean=float(value.mean()), quantiles=np.quantile(value.astype(float), [0,.25,.5,.75,1]).tolist())
    return result


def score_branch(folder, mean, ridge, stats, target, actual_grams, ids,
                 train_gains, metric_scale, seed, arm):
    diagnostics = gaussian_coordinate_diagnostics(folder, ids, target, mean, ridge.covariance)
    sampled = sample_joint_coordinates(mean, ridge.covariance, CONFIG['samples'], seed+200000)
    draws, audit = decode_draws(restore_target(sampled, stats), verify=True)
    return evaluate_and_save(folder, draws, actual_grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=10, selection='fixed epoch10 early readout',
            backbone='immutable historical HR_VALID_S0', covariance='exact RIDGE_VALID OOF covariance',
            numerics=audit, u_diagnostics=diagnostics, formal_certificate=False,
            physical_variance_identified=False, historical_dev=True),
        train_actual_gains=train_gains, score_scale=metric_scale, seed=seed,
        n_bootstrap=CONFIG['bootstrap'], n_random=CONFIG['random_subsets'])


def execute_fold(root, manifest, ds, record, grams, raw_u, gains):
    from .geometry_kernel_replacement import ReplacementBasisBank, GeometryKernelReplacementMean

    fold = record['fold']
    folder = root/'folds'/f'fold_{fold}'
    folder.mkdir(parents=True)
    fit, valid, test = validate_partition(record, len(ds))
    old, stats, ridge, hr, payload = load_frozen_fold(manifest['reference_run'], record)
    x, target = transform_input(ds.Y[:,0], stats), transform_target(raw_u, stats)
    with torch.no_grad():
        hr_mean = hr(torch.as_tensor(x, dtype=torch.float64)).numpy()
        hr_raw = hr.network(torch.as_tensor(x, dtype=torch.float64)).numpy()
    with np.load(old/'arms/HR_VALID_S0/test/u_predictions.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['ids'], ds.ids[test]) or not np.allclose(saved['mean_u'], hr_mean[test], atol=1e-12, rtol=1e-12):
            raise ValueError('Restored frozen HR differs from its recorded test prediction')
        if not np.array_equal(saved['actual_u'], target[test]):
            raise ValueError('The original target preprocessing changed')
    bank = ReplacementBasisBank.fit(x, ds.chem, ds.chem_mask, ds.ids, ds.ids[fit].tolist(),
        metadata=ds.metadata.get('chemical') or None, max_anchors=CONFIG['max_anchors']).double()
    bank_config = bank.config
    write_json(folder/'descriptor_config.json', bank_config)
    torch.save(bank.state_dict(), folder/'descriptor_state.pt')
    write_json(folder/'support_diagnostics.json', {name: support_diagnostics(bank,
        x[ii], ds.chem[ii], ds.chem_mask[ii], ds.ids[ii])
        for name, ii in (('fit',fit), ('validation',valid), ('test',test))})
    baseline_folder = folder/'arms/A_HR/test'
    baseline_folder.mkdir(parents=True)
    for name in ('predictions.npz', 'u_predictions.npz', 'u_diagnostics.json'):
        shutil.copy2(old/'arms/HR_VALID_S0/test'/name, baseline_folder/name)
    report = json.loads((old/'arms/HR_VALID_S0/test/metrics.json').read_text())
    report['model'].update(arm='A_HR', actual_checkpoint_epoch='frozen', frozen_hr_epoch=payload['epoch'],
        source_score=str(old/'arms/HR_VALID_S0/test'), source_predictions_reused=True,
        parameter_counts=dict(trainable=0, total=sum(p.numel() for p in hr.parameters())))
    write_json(baseline_folder/'metrics.json', report)
    write_json(folder/'baseline_restore.json', dict(frozen_hr_epoch=payload['epoch'],
        original_prediction_verified=True, covariance_unchanged=True,
        tanh_saturation_fraction={name: float((np.abs(np.tanh(hr_raw[ii]))>.95).mean())
            for name, ii in (('fit',fit),('validation',valid),('test',test))}))
    counts, models = {}, {}
    training_seed = record['seed']+CONFIG['branch_seed_offset']
    for arm, mode in (('B_MLP','mlp'), ('C_GENERIC','generic'), ('D_STRUCTURED','structured')):
        torch.manual_seed(training_seed)
        model = GeometryKernelReplacementMean(hr, bank, mode=mode, incremental_penalty=CONFIG['incremental_penalty'],
            hidden_dim=CONFIG['hidden_dim'], kan_hidden_dim=CONFIG['kan_hidden_dim']).double()
        initial = predict_branch(model, x, ds.chem, ds.chem_mask)
        if not np.array_equal(initial, hr_mean):
            raise ValueError('A zero-initialized kernel must exactly reproduce frozen HR')
        counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        models[arm] = model
    if counts['C_GENERIC'] != counts['D_STRUCTURED']:
        raise ValueError('The generic and structured branches must have equal trainable parameter counts')
    write_json(folder/'initialization.json', dict(parameter_counts=counts,
        zero_initialization_exact=True, training_seed=training_seed, covariance_unchanged=True,
        generic_structured_parameter_matched=True, mlp_parameter_matched=False))
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    for arm in ARMS[1:]:
        event(root, 'BRANCH_TRAINING', fold=fold, arm=arm, stage_epochs=CONFIG['stage_epochs'])
        models[arm] = train_branch(folder/'arms'/arm, models[arm], x[joined], ds.chem[joined],
            ds.chem_mask[joined], target[joined], fit_local, valid_local, training_seed, ds.ids[joined])
    metric_scale = fit_score_scale(grams[fit])
    for arm in ARMS[1:]:
        event(root, 'SCORING_EPOCH10', fold=fold, arm=arm)
        model = models[arm]
        mean = predict_branch(model, x[test], ds.chem[test], ds.chem_mask[test])
        # Validate the saved, reloadable deployment model as well as the live model.
        checkpoint = torch.load(folder/'arms'/arm/'epoch10.pt', map_location='cpu', weights_only=True)
        restored = GeometryKernelReplacementMean.from_config(checkpoint['model_config'])
        restored.load_state_dict(checkpoint['state_dict'])
        if not np.array_equal(mean, predict_branch(restored, x[test], ds.chem[test], ds.chem_mask[test])):
            raise ValueError('Restored kernel differs from the epoch10 prediction')
        with torch.no_grad():
            diagnostics = model.diagnostics(torch.as_tensor(x[test], dtype=torch.float64),
                torch.as_tensor(ds.chem[test], dtype=torch.float64), torch.as_tensor(ds.chem_mask[test]),
                ids=ds.ids[test].tolist())
        np.savez_compressed(folder/'arms'/arm/'kernel_diagnostics.npz', ids=ds.ids[test],
            **{key: value.detach().cpu().numpy() for key, value in diagnostics.items()
               if isinstance(value, torch.Tensor)})
        score_branch(folder/'arms'/arm/'test', mean, ridge, stats, target[test], grams[test],
            ds.ids[test], gains[fit], metric_scale, record['seed'], arm)
    write_json(folder/'complete.json', dict(fold=fold, test_n=len(test), actual_checkpoint_epoch=10,
        completed_utc=now(), formal_certificate=False))
    event(root, 'FOLD_COMPLETE', fold=fold, test_n=len(test), actual_checkpoint_epoch=10)


def execute(output):
    root, manifest, ds = load_run(output)
    if (root/'summary.json').exists() or (root/'folds').exists():
        raise FileExistsError('This stage has already started; existing artifacts are preserved')
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root, 'RUNNING', stage_epochs=10, branch_fits=15)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams = profiles_to_gram(torch.as_tensor(ds.Y, dtype=torch.float64)).numpy()
            raw_u = gram_to_coordinates(torch.as_tensor(grams)).numpy()
            gains = gram_gains(torch.as_tensor(grams)).numpy()
            for record in manifest['folds']:
                execute_fold(root, manifest, ds, record, grams, raw_u, gains)
            from .geometry_kernel_replacement_summary import summarize
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-started,
            stopped_at_epoch=10, further_training_started=False)
    except Exception as error:
        event(root, 'FAILED', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic()-started)
        raise


def launch(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if (root/'launch.json').exists():
        raise FileExistsError('Stage already launched')
    command = [manifest['python_executable'], '-u', '-m', 'opal2.geometry_kernel_replacement_experiment',
               'execute', '--output', str(root)]
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    with (root/'worker.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=manifest['source_snapshot'], env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    inhibitor = None
    if sys.platform == 'darwin' and Path('/usr/bin/caffeinate').is_file():
        inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True).pid
    result = dict(created_utc=now(), pid=process.pid, command=command,
        cwd=manifest['source_snapshot'], caffeinate_pid=inhibitor, automatic_restart=False)
    write_json(root/'launch.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare','execute','launch','summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', default=str(PROJECT/'runs/hierarchical_stability_20260914_v2'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.reference)
    elif args.mode == 'execute':
        execute(args.output)
    elif args.mode == 'launch':
        launch(args.output)
    else:
        from .geometry_kernel_replacement_summary import summarize
        summarize(args.output)


if __name__ == '__main__':
    main()

