"""Selection-matched repeated DEV validation of bounded geometry corrections.

This is a full fixed-role geometry experiment. It neither opens a new cohort
nor implements an identified cross-site biological variance hierarchy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
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
from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import assign_folds, now, event, uniform_global_policy
from .gram_factor_verified import decode_draws
from .gram_oof_ridge import fit_ridge_oof, transform_input, transform_target
from .gram_simple_models import GramSimpleGaussian, fit_global
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import evaluate_and_save, fit_score_scale
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import (
    CONFIG as HR_CONFIG, train_hr, predict_hr, restore_target,
    gaussian_coordinate_diagnostics,
)
from .hierarchical_stability_ridge import fit_validation_ridge


PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('GLOBAL_GEOMETRY', 'RIDGE_TRAINCV', 'RIDGE_VALID',
        'HR_VALID_S0', 'HR_VALID_S1', 'HR_VALID_S2')
CONFIG = dict(seed=20260914, partition_seeds=[20260914, 20260915, 20260916],
    training_seed_offsets=[401, 1401, 2401], folds=5, strata=10,
    inner_validation_fraction=.2, samples=2000, bootstrap=2000,
    random_subsets=2000, threads=2, residual=deepcopy(HR_CONFIG['residual']))
SCOPE_FLAGS = ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed',
               'original_contract_changed', 'original_split_files_changed')


def validate_partition(record, n):
    arrays = [np.asarray(record[key], dtype=int) for key in ('fit', 'inner_validation', 'test')]
    if (any(a.ndim != 1 or not len(a) for a in arrays)
            or sorted(np.concatenate(arrays).tolist()) != list(range(n))):
        raise ValueError('Unique fit, selection and test indices must partition the cohort')
    return arrays


def _check_cohort(ds, split, manifest):
    if (ds.ids.tolist() != manifest['ids'] or ds.Y.shape != (639, 4, 3617)
            or ds.feature_names.tolist() != manifest['feature_names']
            or {key: ds.ids[index].tolist() for key, index in split.items()}
                != manifest['original_compound_ids']):
        raise ValueError('The opened cohort, feature order or original split changed')
    for flag in SCOPE_FLAGS:
        if manifest.get(flag) is not False:
            raise ValueError('Protected scope changed: '+flag)


def prepare(output, reference):
    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Prepare in an entirely new run directory')
    previous = json.loads((reference/'run_manifest.json').read_text())
    summary = json.loads((reference/'summary.json').read_text())
    if summary.get('complete') is not True or summary.get('n') != 639:
        raise ValueError('The completed preceding 639-object comparison is required')
    ds, split, scope = _load_study_data(previous['data_directory'])
    _check_cohort(ds, split, previous)
    repetitions, assignments = [], []
    for repetition, seed in enumerate(CONFIG['partition_seeds']):
        folds, allocation, strata = assign_folds(ds.Y[:, 0], ds.ids,
            seed=seed, n_splits=CONFIG['folds'], n_strata=CONFIG['strata'])
        for record in folds:
            validate_partition(record, len(ds))
            for key in ('fit', 'inner_validation', 'test'):
                record[key+'_ids'] = ds.ids[record[key]].tolist()
            record['training_seeds'] = [record['seed']+offset
                                       for offset in CONFIG['training_seed_offsets']]
        if repetition == 0:
            for old, new in zip(previous['folds'], folds):
                for key in ('fold', 'seed', 'fit', 'inner_validation', 'test'):
                    if old[key] != new[key]:
                        raise ValueError('The first partition must match the historical assignment')
        repetitions.append(dict(repeat=repetition, seed=seed, folds=folds))
        assignments.append((allocation, strata))
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/HIERARCHICAL_STABILITY_PLAN_20260914.md', root/'PROTOCOL.md')
    snapshot = root/'source_snapshot'
    for directory in ('opal2', 'tests'):
        shutil.copytree(PROJECT/directory, snapshot/directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    manifest = dict(created_utc=now(), source_snapshot=str(snapshot),
        python_executable=sys.executable,
        reference_run=str(reference), config=CONFIG, repetitions=repetitions,
        ids=ds.ids.tolist(), data_directory=previous['data_directory'],
        feature_names=previous['feature_names'], feature_groups=previous['feature_groups'],
        original_compound_ids=previous['original_compound_ids'], data_shape=list(ds.Y.shape),
        scope=scope, arms=list(ARMS), historical_dev=True, formal_certificate=False,
        model_scope='fixed-role nine-coordinate conditional geometry; selection-matched stability',
        independent_new_holdout=False, ensemble=False,
        **{flag: False for flag in SCOPE_FLAGS})
    write_json(root/'run_manifest.json', manifest)
    for record, (allocation, strata) in zip(repetitions, assignments):
        rep_root = root/'repetitions'/f"repeat_{record['repeat']}"
        rep_root.mkdir(parents=True)
        rep_manifest = {key: value for key, value in manifest.items() if key != 'repetitions'}
        rep_manifest.update(repeat=record['repeat'], partition_seed=record['seed'], folds=record['folds'])
        write_json(rep_root/'run_manifest.json', rep_manifest)
        np.savez_compressed(rep_root/'fold_assignment.npz', ids=ds.ids.astype(str),
            fold=allocation, x_lognorm_stratum=strata)
    event(root, 'PREPARED', n=len(ds), outer_folds=15, residual_fits=45)


def prepare_repair(output, reference):
    """Copy the stopped numerical-failure run into a fresh continuation.

    This preparation reads manifests and existing artifacts only. It does not
    load the biological data, fit a model, score a draw or change a split.
    """
    import filecmp
    from .gram_factor_verified import NUMERICAL_POLICY

    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Prepare a numerical continuation in a new directory')
    if reference == root or reference in root.parents or root in reference.parents:
        raise ValueError('Reference and continuation must be separate sibling run trees')
    original = json.loads((reference/'run_manifest.json').read_text())
    failed = json.loads((reference/'status.json').read_text())
    reason = 'The direct H=L Lᵀ is not numerically SPD; no underflow repair or jitter is permitted'
    if (failed.get('state') != 'FAILED' or failed.get('error_type') != 'ValueError'
            or failed.get('error') != reason):
        raise ValueError('This continuation handles only the recorded factor-product numerical failure')
    if original.get('config') != CONFIG or tuple(original.get('arms', ())) != ARMS:
        raise ValueError('The original full training configuration and arms must be unchanged')
    for flag in SCOPE_FLAGS:
        if original.get(flag) is not False:
            raise ValueError('The numerical continuation cannot change protected scope: '+flag)
    launched = json.loads((reference/'launch.json').read_text())
    pid = launched.get('pid')
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError('The failed worker PID is missing or invalid')
    probe = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'command='],
                           capture_output=True, text=True, check=False)
    command = probe.stdout.strip()
    if probe.returncode not in (0, 1):
        raise RuntimeError('Cannot verify that the failed worker has exited')
    if command and str(reference) in command:
        raise RuntimeError('The reference worker is still present; continuation was not prepared')
    # A recycled PID with a different command is not the original worker.
    worker_check = dict(pid=pid, original_worker_exited=True, command_at_check=command or None,
                        pid_reused=bool(command), checked_utc=now())
    protocol = PROJECT/'protocols/historical/NUMERICAL_REPAIR_PLAN_20260914.md'
    for required in (protocol, reference/'PROTOCOL.md', PROJECT/'pyproject.toml'):
        if not required.is_file():
            raise FileNotFoundError(str(required))
    source = reference/'repetitions'
    source_files = sorted(path for path in source.rglob('*') if path.is_file())
    if not source.is_dir() or any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('A complete ordinary-file repetition tree is required')
    counts = dict(completed_folds=len(list(source.glob('repeat_*/folds/fold_*/complete.json'))),
        baseline_fit_groups=len(list(source.glob('repeat_*/folds/fold_*/baseline_fits_complete.json'))),
        completed_HR_fits=len(list(source.glob('repeat_*/folds/fold_*/arms/HR_*/training_complete.json'))),
        completed_arm_scores=len(list(source.glob('repeat_*/folds/fold_*/arms/*/test/metrics.json'))))
    if counts != dict(completed_folds=9, baseline_fit_groups=10, completed_HR_fits=27,
                      completed_arm_scores=55):
        raise ValueError('The stopped artifact inventory differs from the reviewed continuation point')
    for repetition in original['repetitions']:
        rep_root = source/f"repeat_{repetition['repeat']}"
        rep_manifest = json.loads((rep_root/'run_manifest.json').read_text())
        if (rep_manifest.get('ids') != original['ids']
                or rep_manifest.get('folds') != repetition['folds']
                or rep_manifest.get('config') != CONFIG):
            raise ValueError('The recorded repeated partition or recipe changed')
        for record in repetition['folds']:
            folder = rep_root/'folds'/f"fold_{record['fold']}"
            completion_path = folder/'complete.json'
            if completion_path.exists():
                completion = json.loads(completion_path.read_text())
                if (completion.get('repeat') != repetition['repeat']
                        or completion.get('fold') != record['fold']
                        or completion.get('test_n') != len(record['test'])):
                    raise ValueError('A reused completion marker differs from its fixed partition')
                for arm in ARMS:
                    for name in ('metrics.json', 'predictions.npz', 'u_predictions.npz'):
                        if not (folder/'arms'/arm/'test'/name).is_file():
                            raise ValueError('A reused completed score is missing an artifact')

    root.mkdir(parents=True)
    shutil.copy2(reference/'PROTOCOL.md', root/'PROTOCOL.md')
    shutil.copy2(protocol, root/'NUMERICAL_REPAIR.md')
    shutil.copytree(source, root/'repetitions', copy_function=shutil.copy2)
    for path in source_files:
        copied = root/'repetitions'/path.relative_to(source)
        if not filecmp.cmp(path, copied, shallow=False):
            raise RuntimeError('A copied continuation artifact differs from its reference')
        before, after = path.stat(), copied.stat()
        if before.st_dev == after.st_dev and before.st_ino == after.st_ino:
            raise RuntimeError('Continuation artifacts must not be hard-linked to the reference')
    snapshot = root/'source_snapshot'
    for directory in ('opal2', 'tests'):
        shutil.copytree(PROJECT/directory, snapshot/directory,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    provenance = dict(kind='numerically_equivalent_factor_forward_continuation',
        reference_run=str(reference), reference_source_snapshot=original['source_snapshot'],
        reference_failed_status=failed, reference_worker_check=worker_check,
        copied_inventory=counts, ordinary_copied_files_verified=len(source_files),
        old_completed_scores_reused=True, old_scores_recomputed=False,
        reused_scores_numerical_policy='original decoder recorded in each saved score metadata',
        newly_scored_numerical_policy=NUMERICAL_POLICY,
        model_law_changed=False, fit_configuration_changed=False,
        partitions_changed=False, training_or_sampling_seeds_changed=False,
        discarded_draws=0, resampled_draws=0, original_run_modified=False)
    manifest = deepcopy(original)
    manifest.update(created_utc=now(), source_snapshot=str(snapshot),
        python_executable=sys.executable, numerical_policy=deepcopy(NUMERICAL_POLICY),
        continuation=provenance)
    write_json(root/'run_manifest.json', manifest)
    for repetition in original['repetitions']:
        path = root/'repetitions'/f"repeat_{repetition['repeat']}"/'run_manifest.json'
        rep_manifest = json.loads(path.read_text())
        rep_manifest.update(source_snapshot=str(snapshot), python_executable=sys.executable,
            numerical_policy=deepcopy(NUMERICAL_POLICY), continuation=provenance)
        write_json(path, rep_manifest)
    write_json(root/'CONTINUATION.json', provenance)
    event(root, 'PREPARED_NUMERICAL_CONTINUATION', **counts,
          reference_run=str(reference), remaining_HR_fits=45-counts['completed_HR_fits'])
    return manifest


def load_run(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if Path(manifest['source_snapshot']) != PROJECT or manifest['config'] != CONFIG:
        raise ValueError('Execute the recorded source snapshot and full configuration')
    if HR_CONFIG['residual'] != CONFIG['residual']:
        raise ValueError('The preceding bounded residual recipe changed')
    ds, split, _ = _load_study_data(manifest['data_directory'])
    _check_cohort(ds, split, manifest)
    return root, manifest, ds


def _save_or_verify_model(folder, model):
    path = folder/'fit.npz'
    if path.exists():
        raise FileExistsError('A completed fit is not overwritten')
    model.save(path)


def fit_baselines(folder, y_fit, u_fit, y_valid, u_valid, seed, fit_ids, valid_ids):
    """One common fit coordinate frame; two explicitly different selectors."""
    ready = folder/'baseline_fits_complete.json'
    if ready.exists():
        stats = json.loads((folder/'preprocessing.json').read_text())
        saved = json.loads(ready.read_text())
        if saved['fit_ids'] != fit_ids.tolist() or saved['selection_ids'] != valid_ids.tolist():
            raise ValueError('Baseline fit scope changed')
        return {name: GramSimpleGaussian.load(folder/'arms'/name/'fit.npz')
                for name in ARMS[:3]}, stats
    if any((folder/'arms'/name/'fit.npz').exists() for name in ARMS[:3]):
        raise RuntimeError('Partial baseline fitting is preserved; no silent restart')
    ridge_inner, stats = fit_ridge_oof(y_fit, u_fit, seed=seed)
    ridge_valid, valid_stats = fit_validation_ridge(y_fit, u_fit, y_valid, u_valid, seed=seed)
    if stats != valid_stats:
        raise ValueError('The two ridge selectors must share fit-only preprocessing')
    models = dict(GLOBAL_GEOMETRY=fit_global(transform_target(u_fit, stats)),
                  RIDGE_TRAINCV=ridge_inner, RIDGE_VALID=ridge_valid)
    write_json(folder/'preprocessing.json', stats)
    for name, model in models.items():
        dest = folder/'arms'/name
        dest.mkdir(parents=True, exist_ok=True)
        _save_or_verify_model(dest, model)
    write_json(ready, dict(fit_ids=fit_ids.tolist(), selection_ids=valid_ids.tolist(),
        traincv_lambda=ridge_inner.selected_lambda, valid_lambda=ridge_valid.selected_lambda,
        selection_information_matched_to_hr=True,
        equal_search_complexity_claim=False, covariance_shared_by_hr=True))
    return models, stats


def score_model(folder, ids, mean, covariance, stats, actual_u, actual_grams,
                train_gains, metric_scale, seed, metadata, *, global_model=False):
    if (folder/'metrics.json').exists():
        return
    diagnostics = gaussian_coordinate_diagnostics(folder, ids, actual_u, mean, covariance)
    if global_model:
        one = sample_joint_coordinates(mean[:1], covariance, CONFIG['samples'], seed+200000)
        coordinates = np.broadcast_to(one, (CONFIG['samples'], len(ids), 9)).copy()
    else:
        coordinates = sample_joint_coordinates(mean, covariance, CONFIG['samples'], seed+200000)
    draws, audit = decode_draws(restore_target(coordinates, stats), verify=True)
    report = evaluate_and_save(folder, draws, actual_grams, ids,
        metadata=dict(**metadata, numerics=audit, u_diagnostics=diagnostics,
            physical_variance_identified=False, independent_new_holdout=False,
            formal_certificate=False), train_actual_gains=train_gains,
        score_scale=metric_scale, seed=seed, n_bootstrap=CONFIG['bootstrap'],
        n_random=CONFIG['random_subsets'])
    if global_model:
        # A constant law cannot select compound IDs. Report exact random-subset
        # expectations, rather than the evaluator's deterministic tie subset.
        actual_gains = gram_gains(torch.as_tensor(actual_grams, dtype=torch.float64)).numpy()
        write_json(folder/'metrics.json', uniform_global_policy(report, actual_gains))


def execute_fold(root, manifest, ds, repetition, record, actual_grams, raw_u, actual):
    r, f, seed = repetition['repeat'], record['fold'], record['seed']
    folder = root/'repetitions'/f'repeat_{r}'/'folds'/f'fold_{f}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        return
    started = time.monotonic()
    fit, valid, test = validate_partition(record, len(ds))
    event(root, 'BASELINE_FIT_STARTED', repeat=r, fold=f, fit_n=len(fit),
          selection_n=len(valid), test_n=len(test))
    models, stats = fit_baselines(folder, ds.Y[fit], raw_u[fit], ds.Y[valid], raw_u[valid],
        seed, ds.ids[fit], ds.ids[valid])
    x, target = transform_input(ds.Y[:, 0], stats), transform_target(raw_u, stats)
    metric_scale = fit_score_scale(actual_grams[fit])
    for name, model in models.items():
        event(root, 'SCORING', repeat=r, fold=f, arm=name)
        score_model(folder/'arms'/name/'test', ds.ids[test], model.predict_mean(x[test]),
            model.covariance, stats, target[test], actual_grams[test], actual[fit], metric_scale,
            seed, dict(arm=name, repeat=r, fold=f, selected_lambda=model.selected_lambda,
                input='none' if name=='GLOBAL_GEOMETRY' else 'complete X and log norm',
                selection=model.metadata.get('method'),
                covariance=('fit-only' if name=='GLOBAL_GEOMETRY' else
                    'external-validation-selected OOF second moment' if name=='RIDGE_VALID' else
                    'train-only nested-CV OOF second moment')),
            global_model=name=='GLOBAL_GEOMETRY')
    ridge = models['RIDGE_VALID']
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    epochs = []
    for index, training_seed in enumerate(record['training_seeds']):
        name = f'HR_VALID_S{index}'
        event(root, 'RESIDUAL_FIT_STARTED', repeat=r, fold=f, arm=name, training_seed=training_seed)
        model, best_epoch = train_hr(folder/'arms'/name, ridge.coefficient, ridge.intercept,
            x[joined], target[joined], fit_local, valid_local, training_seed, ds.ids[joined])
        if (not np.array_equal(model.coefficient.detach().numpy(), ridge.coefficient)
                or not np.array_equal(model.intercept.detach().numpy(), ridge.intercept)):
            raise ValueError('The ridge backbone changed during neural correction')
        mean = predict_hr(model, x[test])
        epochs.append(best_epoch)
        event(root, 'SCORING', repeat=r, fold=f, arm=name, best_epoch=best_epoch)
        score_model(folder/'arms'/name/'test', ds.ids[test], mean, ridge.covariance, stats,
            target[test], actual_grams[test], actual[fit], metric_scale, seed,
            dict(arm=name, repeat=r, fold=f, training_seed=training_seed, best_epoch=best_epoch,
                input='complete X and log norm', backbone='RIDGE_VALID immutable coefficients',
                covariance='exact RIDGE_VALID OOF covariance', selection='external validation u-MSE',
                selected_ridge_lambda=ridge.selected_lambda))
        event(root, 'MODEL_COMPLETE', repeat=r, fold=f, arm=name, best_epoch=best_epoch)
    # All stochastic seeds use the same predictive covariance, without refitting it.
    for name in ARMS[3:]:
        with np.load(folder/'arms'/name/'test/u_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['covariance_u'],
                    np.broadcast_to(ridge.covariance, (len(test), 9, 9))):
                raise ValueError('Residual comparison covariance mismatch')
    completion = dict(repeat=r, fold=f, completed_utc=now(),
        elapsed_seconds=time.monotonic()-started, residual_best_epochs=epochs,
        fit_n=len(fit), selection_n=len(valid), test_n=len(test))
    write_json(folder/'complete.json', completion)
    event(root, 'FOLD_COMPLETE', **completion)


def execute(output):
    root, manifest, ds = load_run(output)
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root, 'RUNNING', repetitions=3, outer_folds=15, residual_fits=45)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            actual_grams = profiles_to_gram(torch.tensor(ds.Y, dtype=torch.float64)).numpy()
            raw_u = gram_to_coordinates(torch.tensor(actual_grams)).numpy()
            actual = gram_gains(torch.tensor(actual_grams)).numpy()
            for repetition in manifest['repetitions']:
                for record in repetition['folds']:
                    execute_fold(root, manifest, ds, repetition, record, actual_grams, raw_u, actual)
                event(root, 'REPETITION_COMPLETE', repeat=repetition['repeat'])
            from .hierarchical_stability_summary import summarize
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        event(root, 'FAILED', error_type=type(error).__name__, error=str(error),
            elapsed_seconds=time.monotonic()-started)
        raise


def launch(output):
    """Start this prepared experiment once; preserve every existing process."""
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if (root/'launch.json').exists() or (root/'summary.json').exists():
        raise FileExistsError('This experiment was already launched; no automatic restart')
    command = [manifest['python_executable'], '-u', '-m',
        'opal2.hierarchical_stability_experiment', 'execute', '--output', str(root)]
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    event(root, 'LAUNCHING')
    with (root/'worker.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=manifest['source_snapshot'], env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
    inhibitor = None
    if sys.platform == 'darwin' and Path('/usr/bin/caffeinate').is_file():
        inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True).pid
    result = dict(utc=now(), pid=process.pid, command=command,
        cwd=manifest['source_snapshot'], worker_log=str(root/'worker.log'),
        caffeinate_pid=inhibitor, automatic_restart=False)
    write_json(root/'launch.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'prepare-repair', 'execute', 'summarize', 'launch'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', default=str(PROJECT/'runs/hierarchical_geometry_20260914_v1'))
    args = parser.parse_args()
    if args.mode=='prepare':
        prepare(args.output, args.reference)
    elif args.mode=='prepare-repair':
        prepare_repair(args.output, args.reference)
    elif args.mode=='execute':
        execute(args.output)
    elif args.mode=='launch':
        launch(args.output)
    else:
        from .hierarchical_stability_summary import summarize
        summarize(args.output)


if __name__=='__main__':
    main()
