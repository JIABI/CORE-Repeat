"""Unchanged four-arm geometry study continued from epoch 10 through 30."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import now, event
from .gram_oof_ridge import transform_input, transform_target
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import evaluate_and_save, fit_score_scale
from .gram_factor_verified import decode_draws
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import restore_target, gaussian_coordinate_diagnostics
from .hierarchical_stability_experiment import validate_partition, _check_cohort
from .geometry_kernel_replacement import GeometryKernelReplacementMean
from .geometry_kernel_replacement_experiment import (
    CONFIG as ORIGINAL_CONFIG, ARMS, load_frozen_fold, predict_branch,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = {**ORIGINAL_CONFIG, 'stage_epochs': 30}
NUMERIC_SOURCES = (
    'geometry_kernel_replacement.py', 'geometry_kernel.py', 'kernels.py',
    'hierarchical_geometry.py', 'geometry_kernel_replacement_experiment.py',
    'gram_experiment.py', 'gram_oof_ridge.py', 'gram_geometry.py',
    'gram_factor_verified.py', 'gram_evaluation.py', 'baseline_policy.py',
    'hierarchical_geometry_experiment.py', 'biology_kernel.py',
)


def verify_source(source):
    source = Path(source).resolve()
    manifest = json.loads((source/'run_manifest.json').read_text())
    from .geometry_kernel_replacement_summary import _validate_manifest
    _validate_manifest(manifest)
    if manifest['config'] != ORIGINAL_CONFIG or tuple(manifest['arms']) != ARMS:
        raise ValueError('The source must be the unchanged epoch-10 four-arm configuration')
    if json.loads((source/'status.json').read_text())['state'] != 'COMPLETE':
        raise ValueError('The epoch-10 stage must be complete')
    for name in NUMERIC_SOURCES:
        if (PROJECT/'opal2'/name).read_bytes() != (source/'source_snapshot/opal2'/name).read_bytes():
            raise ValueError('The numerical implementation changed since epoch 10: '+name)
    for record in manifest['folds']:
        folder = source/'folds'/f"fold_{record['fold']}"/'arms'
        for arm in ARMS[1:]:
            p = torch.load(folder/arm/'epoch10.pt', map_location='cpu', weights_only=True)
            if (p['epoch'] != 10 or p['actual_checkpoint_epoch'] != 10
                    or p['optimizer_steps'] != 70
                    or p['fit_ids'] != record['fit_ids']
                    or p['validation_ids'] != record['inner_validation_ids']):
                raise ValueError('Source checkpoint step or partition mismatch')
            for key in ('optimizer_state_dict', 'scheduler_state_dict',
                        'torch_rng_state', 'order_rng_state'):
                if key not in p:
                    raise ValueError('Missing continuation state: '+key)
    return manifest


def prepare(output, source):
    root, source = Path(output).resolve(), Path(source).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh continuation directory')
    original = verify_source(source)
    ds, split, _ = _load_study_data(original['data_directory'])
    _check_cohort(ds, split, original)
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/'protocols/historical/GEOMETRY_KERNEL_CONTINUATION_PLAN_20260914.md', root/'PROTOCOL.md')
    manifest = deepcopy(original)
    manifest.update(created_utc=now(), config=CONFIG, source_snapshot=str(snapshot),
        python_executable=sys.executable, continuation_source=str(source),
        continuation_start_epoch=10, actual_checkpoint_epoch=30,
        checkpoint_policy='fixed actual epoch30; inherited validation-best is descriptive only',
        previous_results_preserved=True, architecture_changed=False,
        optimizer_restarted=False, scheduler_horizon_epochs=100,
        planned_total_optimizer_steps=210, additional_optimizer_steps=140,
        continuation_decided_after_epoch10_dev=True)
    from .geometry_kernel_continuation_summary import _validate_manifest
    _validate_manifest(manifest)
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=639, resumed_branch_fits=15, start_epoch=10, end_epoch=30)
    return manifest


def load_run(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['config'] != CONFIG:
        raise ValueError('Execute the frozen continuation source snapshot')
    original = verify_source(manifest['continuation_source'])
    for key in ('ids', 'folds', 'data_directory', 'reference_run'):
        if manifest[key] != original[key]:
            raise ValueError('Continuation changed '+key)
    ds, split, _ = _load_study_data(manifest['data_directory'])
    _check_cohort(ds, split, manifest)
    return root, manifest, ds


def score_branch(folder, mean, ridge, stats, target, actual_grams, ids,
                 train_gains, metric_scale, seed, arm):
    diagnostics = gaussian_coordinate_diagnostics(folder, ids, target, mean, ridge.covariance)
    samples = sample_joint_coordinates(mean, ridge.covariance, CONFIG['samples'], seed+200000)
    draws, audit = decode_draws(restore_target(samples, stats), verify=True)
    return evaluate_and_save(folder, draws, actual_grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=30, selection='fixed epoch30 continuation',
            backbone='unchanged historical HR_VALID_S0', covariance='exact RIDGE_VALID OOF covariance',
            numerics=audit, u_diagnostics=diagnostics, formal_certificate=False,
            physical_variance_identified=False, historical_dev=True,
            resumed_from_epoch=10, optimizer_restarted=False),
        train_actual_gains=train_gains, score_scale=metric_scale, seed=seed,
        n_bootstrap=CONFIG['bootstrap'], n_random=CONFIG['random_subsets'])


def execute_fold(root, manifest, ds, record, grams, raw_u, gains):
    from .geometry_kernel_continuation_training import continue_branch

    fold = record['fold']
    source = Path(manifest['continuation_source'])/'folds'/f'fold_{fold}'
    folder = root/'folds'/f'fold_{fold}'
    folder.mkdir(parents=True)
    fit, valid, test = validate_partition(record, len(ds))
    _, stats, ridge, hr, _ = load_frozen_fold(manifest['reference_run'], record)
    x, target = transform_input(ds.Y[:,0], stats), transform_target(raw_u, stats)
    with torch.no_grad():
        hr_mean = hr(torch.as_tensor(x, dtype=torch.float64)).numpy()
    with np.load(source/'arms/A_HR/test/u_predictions.npz', allow_pickle=False) as a:
        if (not np.array_equal(a['ids'], ds.ids[test]) or not np.array_equal(a['actual_u'], target[test])
                or not np.allclose(a['mean_u'], hr_mean[test], atol=1e-12, rtol=1e-12)):
            raise ValueError('Frozen HR or target coordinates changed')
    shutil.copytree(source/'arms/A_HR', folder/'arms/A_HR')
    for name in ('descriptor_config.json', 'descriptor_state.pt', 'support_diagnostics.json',
                 'baseline_restore.json', 'initialization.json'):
        shutil.copy2(source/name, folder/name)
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    models = {}
    for arm in ARMS[1:]:
        event(root, 'CONTINUING', fold=fold, arm=arm, start_epoch=10, end_epoch=30)
        models[arm] = continue_branch(folder/'arms'/arm, source/'arms'/arm,
            x[joined], ds.chem[joined], ds.chem_mask[joined], target[joined],
            fit_local, valid_local, ds.ids[joined], CONFIG)
        completion = json.loads((folder/'arms'/arm/'training_complete.json').read_text())
        if completion['actual_checkpoint_epoch'] != 30 or completion['optimizer_steps'] != 210:
            raise ValueError('Continuation did not finish exactly at epoch30 / step210')
    metric_scale = fit_score_scale(grams[fit])
    for arm in ARMS[1:]:
        event(root, 'SCORING_EPOCH30', fold=fold, arm=arm)
        model = models[arm]
        mean = predict_branch(model, x[test], ds.chem[test], ds.chem_mask[test])
        saved = torch.load(folder/'arms'/arm/'epoch30.pt', map_location='cpu', weights_only=True)
        reloaded = GeometryKernelReplacementMean.from_config(saved['model_config'])
        reloaded.load_state_dict(saved['state_dict'])
        if not np.array_equal(mean, predict_branch(reloaded, x[test], ds.chem[test], ds.chem_mask[test])):
            raise ValueError('Saved epoch30 model does not reproduce the scored mean')
        with torch.no_grad():
            d = model.diagnostics(torch.as_tensor(x[test], dtype=torch.float64),
                torch.as_tensor(ds.chem[test], dtype=torch.float64), torch.as_tensor(ds.chem_mask[test]),
                ids=ds.ids[test].tolist())
        np.savez_compressed(folder/'arms'/arm/'kernel_diagnostics.npz', ids=ds.ids[test],
            **{key: value.detach().cpu().numpy() for key, value in d.items() if isinstance(value, torch.Tensor)})
        score_branch(folder/'arms'/arm/'test', mean, ridge, stats, target[test], grams[test],
            ds.ids[test], gains[fit], metric_scale, record['seed'], arm)
    write_json(folder/'complete.json', dict(fold=fold, test_n=len(test), actual_checkpoint_epoch=30,
        completed_utc=now(), formal_certificate=False))
    event(root, 'FOLD_COMPLETE', fold=fold, actual_checkpoint_epoch=30)


def execute(output):
    root, manifest, ds = load_run(output)
    if (root/'folds').exists() or (root/'summary.json').exists():
        raise FileExistsError('This continuation has already started')
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root, 'RUNNING', start_epoch=10, end_epoch=30, resumed_branch_fits=15)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams = profiles_to_gram(torch.as_tensor(ds.Y, dtype=torch.float64)).numpy()
            raw_u = gram_to_coordinates(torch.as_tensor(grams)).numpy()
            gains = gram_gains(torch.as_tensor(grams)).numpy()
            for record in manifest['folds']:
                execute_fold(root, manifest, ds, record, grams, raw_u, gains)
            from .geometry_kernel_continuation_summary import summarize
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-started,
            stopped_at_epoch=30, further_training_started=False)
    except Exception as error:
        event(root, 'FAILED', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'execute', 'summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default=str(PROJECT/'runs/geometry_kernel_replacement_epoch10_20260914_v1'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.source)
    elif args.mode == 'execute':
        execute(args.output)
    else:
        from .geometry_kernel_continuation_summary import summarize
        summarize(args.output)


if __name__ == '__main__':
    main()
