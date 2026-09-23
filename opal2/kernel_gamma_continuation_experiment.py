"""Exact epoch30-to60 continuation of the four conditional kernel arms."""
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
from .biology_kernel_evaluation import plain, write_json
from .gram_oof_experiment import now, event
from .gram_oof_ridge import transform_input, transform_target
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import evaluate_and_save, fit_score_scale
from .gram_factor_verified import decode_draws
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import restore_target, gaussian_coordinate_diagnostics
from .hierarchical_stability_experiment import validate_partition, _check_cohort
from .conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from .geometry_kernel_replacement_experiment import load_frozen_fold, predict_branch
from .gamma_supervised_experiment import CONFIG as SOURCE_CONFIG, independent_noise
from .gamma_supervised_loss import JointGammaCRPS


PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('A_HR', 'F_CONDITIONAL_GENERIC', 'J_GEOMETRY_CONTROL',
        'K_GEOMETRY_GAMMA_CRPS', 'M_CONDITIONAL_GENERIC_GAMMA')
CONFIG = dict(SOURCE_CONFIG, stage_epochs=60)
START_EPOCH = 30
NUMERICAL_SOURCES = ('conditional_response_kernel.py', 'gamma_supervised_loss.py',
    'gram_experiment.py', 'gram_geometry.py', 'gram_factor_verified.py',
    'hierarchical_geometry.py', 'gram_evaluation.py', 'gram_oof_ridge.py')


def check_source(source):
    source = Path(source).resolve()
    prior = json.loads((source/'run_manifest.json').read_text())
    if json.loads((source/'status.json').read_text())['state'] != 'COMPLETE':
        raise ValueError('The source epoch30 run is incomplete')
    if prior['config'] != SOURCE_CONFIG or tuple(prior['arms']) != ARMS:
        raise ValueError('Source arms or configuration differ')
    for name in NUMERICAL_SOURCES:
        if (PROJECT/'opal2'/name).read_bytes() != (Path(prior['source_snapshot'])/'opal2'/name).read_bytes():
            raise ValueError('Numerical implementation changed since epoch30: '+name)
    for record in prior['folds']:
        for arm in ARMS[1:]:
            folder = source/'folds'/f"fold_{record['fold']}"/'arms'/arm
            saved = torch.load(folder/'epoch30.pt', map_location='cpu', weights_only=True)
            if (saved['epoch'] != 30 or saved['actual_checkpoint_epoch'] != 30
                    or saved['optimizer_steps'] != 210 or saved['scheduler_state_dict']['last_epoch'] != 210):
                raise ValueError('Source checkpoint is not actual epoch30/step210')
            for key in ('optimizer_state_dict', 'scheduler_state_dict', 'order_rng_state', 'torch_rng_state'):
                if key not in saved:
                    raise ValueError('Missing resume state: '+key)
            if arm in ARMS[3:]:
                for key in ('training_mc_rng_state', 'gamma_objective_state_dict'):
                    if key not in saved:
                        raise ValueError('Missing Gamma resume state: '+key)
    return prior


def prepare(output, source):
    root, source = Path(output).resolve(), Path(source).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh epoch60 continuation directory')
    prior = check_source(source)
    ds, split, _ = _load_study_data(prior['data_directory'])
    _check_cohort(ds, split, prior)
    for record in prior['folds']:
        validate_partition(record, len(ds))
        load_frozen_fold(prior['reference_run'], record)
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/'protocols/historical/KERNEL_GAMMA_EPOCH60_PLAN_20260914.md', root/'PROTOCOL.md')
    manifest = deepcopy(prior)
    manifest.update(created_utc=now(), source_snapshot=str(snapshot), python_executable=sys.executable,
        source_run=str(source), config=deepcopy(CONFIG), start_epoch=30, actual_checkpoint_epoch=60,
        arms=list(ARMS), new_arms=list(ARMS[1:]), inherited_arms=['A_HR'],
        training_initialization='exact same-arm epoch30 continuation with complete optimizer and RNG state',
        optimizer_restarted=False, scheduler_restarted=False, scheduler_horizon_epochs=100,
        planned_total_optimizer_steps=420, additional_optimizer_steps=210,
        control_exact_reproduction_required=False, chemical_basis_changed_in_M=False,
        architecture_changed=False, covariance_updated=False,
        checkpoint_policy='fixed actual epoch60; epoch30 and validation trajectory are descriptive comparisons',
        continuation_decided_after_epoch30_dev=True, learning_rate_schedule_changed=False,
        monitoring='fixed-noise Gamma and geometry readout every five epochs from30 to60; not used for selection')
    from .kernel_gamma_continuation_summary import _validate_manifest
    _validate_manifest(manifest)
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=639, continuation_fits=20, start_epoch=30, end_epoch=60)
    return manifest


def monitor(folder, objective, x, chem, mask, target, gamma, fit, valid, seed):
    """Fixed-noise checkpoint diagnostics; never called within optimization."""
    x, chem, target, gamma = [torch.as_tensor(a, dtype=torch.float64) for a in (x, chem, target, gamma)]
    mask = torch.as_tensor(mask, dtype=torch.bool)
    rng = torch.Generator().manual_seed(seed+CONFIG['validation_mc_offset'])
    noises = (independent_noise(CONFIG['validation_pairs'], len(fit), rng),
              independent_noise(CONFIG['validation_pairs'], len(valid), rng))
    ii = np.asarray(fit)[:CONFIG['batch_size']]
    drng = torch.Generator().manual_seed(seed+CONFIG['gradient_mc_offset'])
    diag_noise = independent_noise(CONFIG['validation_pairs'], len(ii), drng)
    rows = []
    for epoch in range(30, 61, 5):
        saved = torch.load(folder/f'epoch{epoch}.pt', map_location='cpu', weights_only=True)
        model = ConditionalResponseKernelMean.from_config(saved['model_config'])
        model.load_state_dict(saved['state_dict']); model.eval()
        row = dict(epoch=epoch, checkpoint_readout_only=True, fixed_pairs=CONFIG['validation_pairs'])
        with torch.no_grad():
            for name, indexes, noise in zip(('fit', 'validation'), (fit, valid), noises):
                original = model.loss(x[indexes], chem[indexes], mask[indexes], target[indexes])
                result = objective(model(x[indexes], chem[indexes], mask[indexes]), gamma[indexes], *noise)
                row.update({name+'_u_mse':float(original['mean_mse']),
                    name+'_incremental_penalty':float(original['incremental_penalty']),
                    name+'_gamma_crps':float(result['gamma_crps']),
                    name+'_normalized_gamma_crps':float(result['normalized_gamma_crps']),
                    name+'_gamma_mse':float((result['gamma_prediction_mean']-gamma[indexes]).square().mean())})
        original = model.loss(x[ii], chem[ii], mask[ii], target[ii])
        result = objective(model(x[ii], chem[ii], mask[ii]), gamma[ii], *diag_noise)
        params = list(model.trainable_parameters())
        grad = torch.autograd.grad(original['mean_mse'], params, allow_unused=True)
        ggrad = torch.autograd.grad(result['normalized_gamma_crps'], params, allow_unused=True)
        g, h = [torch.cat([(v if v is not None else torch.zeros_like(p)).reshape(-1)
                          for v,p in zip(values,params)]) for values in (grad,ggrad)]
        gn, hn = float(g.norm()), float(h.norm())
        row.update(geometry_gradient_norm=gn, normalized_gamma_gradient_norm=hn,
            gradient_cosine=float(g@h)/(gn*hn) if gn*hn>0 else None,
            gradient_scope='first64fit; fixed128independentpairs; readout only')
        rows.append(row)
    with (folder/'gamma_monitoring.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(plain(row), allow_nan=False)+'\n')


def score(folder, mean, ridge, stats, target, grams, ids, train_gains, metric_scale, seed, arm, counts):
    diagnostics = gaussian_coordinate_diagnostics(folder, ids, target, mean, ridge.covariance)
    samples = sample_joint_coordinates(mean, ridge.covariance, CONFIG['samples'], seed+200000)
    draws, audit = decode_draws(restore_target(samples, stats), verify=True)
    return evaluate_and_save(folder, draws, grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=60, selection='fixed epoch60',
            backbone='immutable historical HR_VALID_S0', covariance='exact RIDGE_VALID OOF covariance',
            numerics=audit, u_diagnostics=diagnostics, parameter_counts=counts,
            formal_certificate=False, physical_variance_identified=False, historical_dev=True),
        train_actual_gains=train_gains, score_scale=metric_scale, seed=seed,
        n_bootstrap=CONFIG['bootstrap'], n_random=CONFIG['random_subsets'])


def execute_fold(root, manifest, ds, record, grams, raw_u, gains):
    from .kernel_gamma_continuation_training import continue_branch
    folder = root/'folds'/f"fold_{record['fold']}"
    old = Path(manifest['source_run'])/'folds'/f"fold_{record['fold']}"
    folder.mkdir(parents=True)
    fit, valid, test = validate_partition(record, len(ds))
    _, stats, ridge, _, _ = load_frozen_fold(manifest['reference_run'], record)
    x, target = transform_input(ds.Y[:,0], stats), transform_target(raw_u, stats)
    bank = LocalResponseBank.load(old/'response_bank.pt').double()
    if set(bank.config['fitting_ids']) != set(ds.ids[fit].tolist()):
        raise ValueError('Response bank fitting identities differ')
    for name in ('response_bank.pt', 'gamma_objective.json', 'gamma_objective_state.pt'):
        shutil.copy2(old/name, folder/name)
    objective = JointGammaCRPS(ridge.covariance, stats['u_center'], stats['u_scale'],
                              float(np.std(gains[fit,2], ddof=1)))
    saved_objective = torch.load(old/'gamma_objective_state.pt', map_location='cpu', weights_only=True)
    if objective.state_dict().keys() != saved_objective.keys() or any(
            not torch.equal(v, saved_objective[k]) for k,v in objective.state_dict().items()):
        raise ValueError('Frozen covariance or objective scaling differs')
    objective.load_state_dict(saved_objective)
    shutil.copytree(old/'arms/A_HR', folder/'arms/A_HR')
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    metric_scale = fit_score_scale(grams[fit])
    for arm in ARMS[1:]:
        af = folder/'arms'/arm
        event(root, 'CONTINUING', fold=record['fold'], arm=arm, start_epoch=30, end_epoch=60)
        model = continue_branch(af, old/'arms'/arm, x[joined], ds.chem[joined], ds.chem_mask[joined],
            target[joined], gains[joined,2], fit_local, valid_local, ds.ids[joined], CONFIG,
            arm=arm, objective=objective if arm in ARMS[3:] else None, end_epoch=60)
        event(root, 'CHECKPOINT_MONITORING', fold=record['fold'], arm=arm)
        monitor(af, objective, x[joined], ds.chem[joined], ds.chem_mask[joined], target[joined],
            gains[joined,2], fit_local, valid_local, record['seed']+CONFIG['branch_seed_offset'])
        mean = predict_branch(model, x[test], ds.chem[test], ds.chem_mask[test])
        checkpoint = torch.load(af/'epoch60.pt', map_location='cpu', weights_only=True)
        restored = ConditionalResponseKernelMean.from_config(checkpoint['model_config'])
        restored.load_state_dict(checkpoint['state_dict'])
        if not np.array_equal(mean, predict_branch(restored, x[test], ds.chem[test], ds.chem_mask[test])):
            raise ValueError('Actual epoch60 checkpoint does not reproduce its scored mean')
        with torch.no_grad():
            diagnostic = model.diagnostics(torch.as_tensor(x[test],dtype=torch.float64),
                torch.as_tensor(ds.chem[test],dtype=torch.float64), torch.as_tensor(ds.chem_mask[test]),
                ids=ds.ids[test].tolist())
        values = {k:v.detach().cpu().numpy() for k,v in diagnostic.items() if isinstance(v, torch.Tensor)}
        values['raw'] = values['kernel_raw']
        np.savez_compressed(af/'model_diagnostics.npz', ids=ds.ids[test],
                            block_names=np.asarray(['chemical','morphology','scalar']), **values)
        counts = dict(trainable=sum(p.numel() for p in model.trainable_parameters()),
                      total=sum(p.numel() for p in model.parameters()))
        event(root, 'SCORING_EPOCH60', fold=record['fold'], arm=arm)
        score(af/'evaluation', mean, ridge, stats, target[test], grams[test], ds.ids[test],
              gains[fit], metric_scale, record['seed'], arm, counts)
    write_json(folder/'complete.json', dict(fold=record['fold'], test_n=len(test),
        actual_checkpoint_epoch=60, completed_utc=now(), original_covariance_preserved=True,
        formal_certificate=False))
    event(root, 'FOLD_COMPLETE', fold=record['fold'], actual_checkpoint_epoch=60)


def execute(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['config'] != CONFIG:
        raise ValueError('Execute the frozen source snapshot and configuration')
    prior = check_source(manifest['source_run'])
    for key in ('ids','folds','reference_run','data_directory'):
        if prior[key] != manifest[key]:
            raise ValueError('Historical allocation changed: '+key)
    from .kernel_gamma_continuation_summary import _validate_manifest, summarize
    _validate_manifest(manifest)
    ds, split, _ = _load_study_data(manifest['data_directory'])
    _check_cohort(ds, split, manifest)
    if (root/'folds').exists():
        raise FileExistsError('This run has already started')
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root, 'RUNNING', continuation_fits=20, start_epoch=30, end_epoch=60)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams = profiles_to_gram(torch.as_tensor(ds.Y, dtype=torch.float64)).numpy()
            raw_u = gram_to_coordinates(torch.as_tensor(grams)).numpy()
            gains = gram_gains(torch.as_tensor(grams)).numpy()
            for record in manifest['folds']:
                execute_fold(root, manifest, ds, record, grams, raw_u, gains)
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-started,
              stopped_at_epoch=60, further_training_started=False)
    except Exception as error:
        event(root, 'FAILED', error_type=type(error).__name__, error=str(error),
              elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','execute','summarize'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--source',default=str(PROJECT/'runs/generic_gamma_epoch30_20260914_v1'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output,args.source)
    elif args.mode == 'execute':
        execute(args.output)
    else:
        from .kernel_gamma_continuation_summary import summarize
        summarize(args.output)


if __name__ == '__main__':
    main()
