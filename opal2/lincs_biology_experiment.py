"""Full four-arm pharmacology-response experiment on fresh LINCS profiles.

Reuses the complete ridge/HR, joint Gram distribution, local response model,
Gamma-CRPS trainer and predictive evaluator. Only the dataset adapter and the
predeclared biological response block are new. No JUMP loader is called.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import now, event
from .gram_oof_ridge import transform_input, transform_target
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_factor_verified import decode_draws
from .gram_evaluation import fit_score_scale, evaluate_and_save
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import train_hr, predict_hr, restore_target, gaussian_coordinate_diagnostics
from .hierarchical_stability_ridge import fit_validation_ridge
from .kernel_final_reference import reference_allocation, fit_reference_bank
from .kernel_final_training import train_branch
from .kernel_final_experiment import CONFIG as PREVIOUS_CONFIG
from .gamma_supervised_loss import JointGammaCRPS
from .mechanism_response_kernel import MechanismResponseBank, MechanismResponseKernelMean


PROJECT = Path(__file__).resolve().parents[1]
ARMS = dict(A_OLD_GENERIC='old_generic', B_OLD_STRUCTURED='old_structured',
            C_BIO_GENERIC='bio_generic', D_BIO_STRUCTURED='bio_structured')
CONFIG = dict(PREVIOUS_CONFIG, seed=20260915, folds=5, stage_epochs=30,
              samples=10000, hidden_dim=16, bootstrap=2000, random_subsets=2000)


def load_data(directory):
    root = Path(directory)
    metadata = json.loads((root/'metadata.json').read_text())
    with np.load(root/'data.npz', allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    n = len(data['ids'])
    if (data['Y'].ndim != 3 or data['Y'].shape[:2] != (n, 4)
            or len(set(data['ids'])) != n or metadata['ids'] != data['ids'].tolist()
            or data['chem'].shape != (n, 513) or not data['chem_mask'].all()):
        raise ValueError('LINCS identity, four-role measurements or chemistry do not match')
    if not all(np.isfinite(data[key]).all() for key in ('Y', 'chem', 'target', 'moa')):
        raise ValueError('Prepared arrays must be finite')
    if len(metadata['units']) != n or len(data['feature_names']) != data['Y'].shape[-1]:
        raise ValueError('Prepared feature/metadata alignment failed')
    return data, metadata


def grouped_folds(ids, groups, seed):
    """Chemistry-only grouping; no outcomes or future profiles determine folds."""
    ids, groups = np.asarray(ids, str), np.asarray(groups, str)
    unique = np.unique(groups)
    records = []
    for fold, (pool, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(unique)):
        fit, valid = train_test_split(pool, test_size=.2, random_state=seed+10000*fold+91)
        row = dict(fold=fold, seed=seed+10000*fold)
        for name, group_indices in (('fit', fit), ('inner_validation', valid), ('test', test)):
            rows = np.flatnonzero(np.isin(groups, unique[group_indices]))
            row[name] = rows.tolist()
            row[name+'_ids'] = ids[rows].tolist()
        check = [set(groups[row[k]]) for k in ('fit', 'inner_validation', 'test')]
        if any(check[a] & check[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('A chemical connectivity group crosses fitting/selection/test')
        records.append(row)
    return records


def prepare(output, data_directory):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh LINCS run directory')
    data, metadata = load_data(data_directory)
    ids = data['ids']
    folds = grouped_folds(ids, data['groups'], CONFIG['seed'])
    scopes = []
    for record in folds:
        scope = reference_allocation(ids, record['fit'], data['chem'], data['chem_mask'],
            metadata['chemical'], record['seed']+CONFIG['reference_seed_offset'], 64)
        # Independent reference roles cover the entire same-connectivity group,
        # not just an alternate sample identifier for the same structure.
        ref_groups = set(data['groups'][np.isin(ids, scope['reference_ids'])])
        fit = np.asarray(record['fit'])
        fit = fit[~np.isin(data['groups'][fit], list(ref_groups))]
        scope.update(commonbranchfit=fit.tolist(), commonbranchfit_ids=ids[fit].tolist(),
            reference_group_exclusion=True, validation_ids=record['inner_validation_ids'],
            test_ids=record['test_ids'], fold=record['fold'])
        scopes.append(scope)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/LINCS_BIOLOGY_FOUR_ARM_PLAN_20260915.md', root/'PROTOCOL.md')
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    manifest = dict(created_utc=now(), data_directory=str(Path(data_directory).resolve()),
        source_snapshot=str(snapshot), python_executable=sys.executable, config=deepcopy(CONFIG),
        ids=ids.tolist(), groups=data['groups'].tolist(), data_shape=list(data['Y'].shape),
        feature_names=data['feature_names'].tolist(), folds=folds, scopes=scopes,
        arms=['HR', *ARMS], modes=ARMS, actual_checkpoint_epoch=30,
        dataset=metadata, final_opened=False, fifth_repeat_opened=False,
        old_results_modified=False, original_contract_changed=False,
        jepa_active=False, biological_prior_kind='curated target/MoA profile relationships',
        biology_active_arms=['C_BIO_GENERIC','D_BIO_STRUCTURED'],
        biological_capacity='shared old coefficients, gates and readout; no new weights',
        covariance_updated_by_branch=False, formal_certificate=False,
        experiment_scope='new public LINCS development comparison, not protected JUMP FINAL',
        full_joint_draws=10000, fixed_epochs=30, new_branch_fits=20,
        confidence_interval_scope='paired chemistry-group bootstrap conditional on fitted predictions; shared-plate sensitivity separate')
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=len(ids), branch_fits=20, epochs=30)
    return manifest


def tensor_biology(data):
    return {key: torch.as_tensor(data[key], dtype=torch.bool if key.endswith('_mask') else torch.float64)
            for key in ('target','target_mask','moa','moa_mask')}


def score(folder, mean, ridge, stats, target, grams, ids, train_gains, scale, seed, arm):
    diagnostic = gaussian_coordinate_diagnostics(folder, ids, target, mean, ridge.covariance)
    samples = sample_joint_coordinates(mean, ridge.covariance, CONFIG['samples'], seed+200000)
    draws, numerical = decode_draws(restore_target(samples, stats), verify=True)
    return evaluate_and_save(folder, draws, grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=30 if arm!='HR' else 'fresh validation-selected HR',
            covariance='identical frozen full RIDGE_VALID OOF joint covariance',
            u_diagnostics=diagnostic, numerics=numerical, formal_certificate=False,
            biological_information=arm in ('C_BIO_GENERIC','D_BIO_STRUCTURED')),
        train_actual_gains=train_gains, score_scale=scale, seed=seed,
        n_bootstrap=CONFIG['bootstrap'], n_random=CONFIG['random_subsets'])


def execute_fold(root, manifest, data, metadata, record, scope, grams, raw_u, gains):
    folder = root/'folds'/f"fold_{record['fold']}"
    folder.mkdir(parents=True)
    write_json(folder/'scope.json', scope)
    originalfit, valid, test = [np.asarray(record[k], int) for k in ('fit','inner_validation','test')]
    fit = np.asarray(scope['commonbranchfit'], int)
    ids, y = data['ids'], data['Y']
    event(root, 'FITTING_BACKBONE', fold=record['fold'], fit_n=len(originalfit))
    ridge, stats = fit_validation_ridge(y[originalfit], raw_u[originalfit], y[valid], raw_u[valid],
        record['seed'], groups=data['groups'][originalfit])
    ridge.save(folder/'ridge.npz')
    write_json(folder/'preprocessing.json', stats)
    x, u = transform_input(y[:,0], stats), transform_target(raw_u, stats)
    joined = np.r_[originalfit, valid]
    hr, best_epoch = train_hr(folder/'HR_fit', ridge.coefficient, ridge.intercept,
        x[joined], u[joined], np.arange(len(originalfit)), np.arange(len(originalfit),len(joined)),
        record['seed']+401, ids[joined])
    hr.eval().requires_grad_(False)
    local = fit_reference_bank(x, data['chem'], data['chem_mask'], ids, scope['originalfit_ids'],
        scope['commonbranchfit_ids'], scope['reference_ids'], metadata['chemical'])
    bio_all = {k: data[k] for k in ('target','target_mask','moa','moa_mask')}
    bank = MechanismResponseBank.fit(local, bio_all, ids, scope['commonbranchfit_ids'],
        target_names=metadata['target_names'], moa_names=metadata['moa_names'])
    if any(bank.config['fitting_biological_rms'][mode]<=bank.SCALE_FLOOR for mode in ('generic','structured')):
        raise ValueError('No nonzero TRAIN biological response; do not amplify new support with an RMS floor')
    bank.save(folder/'bank.pt')
    bio = tensor_biology(data)
    packed = bank.pack_information(torch.as_tensor(data['chem']), bio).numpy()
    support = {}
    with torch.no_grad():
        values, known = bank.biological_descriptors(bio)
        for name, rows in (('fit',fit),('validation',valid),('test',test)):
            shared = values[rows] > 0
            support[name] = dict(n=len(rows), known_queries_by_channel=known[rows].any(1).sum(0).tolist(),
                shared_reference_queries_by_channel=shared.any(1).sum(0).tolist(),
                mean_nonzero_references_by_channel=shared.double().sum(1).mean(0).tolist(),
                nonzero_reference_pairs_by_channel=shared.sum((0,1)).tolist(),
                target_known=int(data['target_mask'][rows].sum()), moa_known=int(data['moa_mask'][rows].sum()),
                max_normalized_biological_response={mode:float((bank.biological_response(values[rows],known[rows],mode)
                    /getattr(bank,'biology_'+mode+'_scale')).abs().max()) for mode in ('generic','structured')})
    write_json(folder/'biological_support.json', support)
    if not any(support['test']['shared_reference_queries_by_channel']):
        raise ValueError('No held-out object shares known biology with the independent references')
    objective = JointGammaCRPS(ridge.covariance, stats['u_center'], stats['u_scale'],
                              float(np.std(gains[fit,2],ddof=1)))
    torch.save(objective.state_dict(), folder/'gamma_objective_state.pt')
    scale = fit_score_scale(grams[originalfit])
    hr_mean = predict_hr(hr, x[test])
    event(root, 'SCORING', fold=record['fold'], arm='HR', samples=CONFIG['samples'])
    score(folder/'arms/HR/evaluation', hr_mean, ridge, stats, u[test], grams[test], ids[test],
          gains[fit], scale, record['seed'], 'HR')
    training_rows = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit),len(training_rows))
    initial = None
    for arm, mode in ARMS.items():
        seed = record['seed']+CONFIG['branch_seed_offset']
        torch.manual_seed(seed)
        model = MechanismResponseKernelMean(hr, bank, mode=mode,
            incremental_penalty=CONFIG['incremental_penalty'], hidden_dim=CONFIG['hidden_dim']).double()
        parameters = {k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
        if initial is None:
            initial = parameters
        elif set(initial)!=set(parameters) or any(not torch.equal(initial[k],v) for k,v in parameters.items()):
            raise ValueError('Four-arm active parameter initialization differs')
        if sum(v.numel() for v in parameters.values()) != 3837:
            raise ValueError('Expected 3837 active branch parameters in every arm')
        with torch.no_grad():
            if not np.array_equal(model(torch.tensor(x[test]),torch.tensor(packed[test]),
                                       torch.tensor(data['chem_mask'][test])).numpy(),hr_mean):
                raise ValueError('A branch does not initialize at the identical fresh HR')
        event(root, 'TRAINING', fold=record['fold'], arm=arm, epochs=30, fit_n=len(fit))
        train_branch(folder/'arms'/arm, model, objective, x[training_rows], packed[training_rows],
            data['chem_mask'][training_rows], u[training_rows], gains[training_rows,2],
            fit_local, valid_local, seed, ids[training_rows], CONFIG, 'weighted')
        model.eval()
        with torch.no_grad():
            mean = model(torch.tensor(x[test]),torch.tensor(packed[test]),torch.tensor(data['chem_mask'][test])).numpy()
            details = model.diagnostics(torch.tensor(x[test]),torch.tensor(packed[test]),
                                        torch.tensor(data['chem_mask'][test]),ids=ids[test])
        saved = torch.load(folder/'arms'/arm/'epoch30.pt',map_location='cpu',weights_only=True)
        restored = MechanismResponseKernelMean.from_config(saved['model_config'])
        restored.load_state_dict(saved['state_dict']);restored.eval()
        with torch.no_grad():
            again = restored(torch.tensor(x[test]),torch.tensor(packed[test]),torch.tensor(data['chem_mask'][test])).numpy()
        if not np.array_equal(mean,again):
            raise ValueError('Fixed epoch30 checkpoint does not reproduce evaluated predictions')
        np.savez_compressed(folder/'arms'/arm/'model_diagnostics.npz',ids=ids[test],
            **{k:v.detach().numpy() for k,v in details.items() if isinstance(v,torch.Tensor)})
        event(root,'SCORING',fold=record['fold'],arm=arm,samples=CONFIG['samples'])
        score(folder/'arms'/arm/'evaluation',mean,ridge,stats,u[test],grams[test],ids[test],
              gains[fit],scale,record['seed'],arm)
        event(root,'ARM_COMPLETE',fold=record['fold'],arm=arm,epoch=30)
    write_json(folder/'complete.json',dict(fold=record['fold'],n=len(test),arms=['HR',*ARMS],
        branch_epochs=30,hr_validation_best_epoch=best_epoch,trainable_branch_parameters=3837))
    event(root,'FOLD_COMPLETE',fold=record['fold'])


def execute(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if Path(manifest['source_snapshot']) != PROJECT or manifest['config'] != CONFIG:
        raise ValueError('Execute the frozen source and exact prepared configuration')
    if (root/'folds').exists():
        raise FileExistsError('Existing training artifacts are not overwritten')
    data, metadata = load_data(manifest['data_directory'])
    if data['ids'].tolist()!=manifest['ids']:
        raise ValueError('Prepared LINCS cohort changed')
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root,'RUNNING',n=len(data['ids']),new_fits=20,epochs=30)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams = profiles_to_gram(torch.tensor(data['Y'])).numpy()
            raw_u = gram_to_coordinates(torch.tensor(grams)).numpy()
            gains = gram_gains(torch.tensor(grams)).numpy()
            for record,scope in zip(manifest['folds'],manifest['scopes']):
                execute_fold(root,manifest,data,metadata,record,scope,grams,raw_u,gains)
            event(root,'SUMMARIZING')
            from .lincs_biology_summary import summarize
            summarize(root)
        event(root,'COMPLETE',elapsed_seconds=time.monotonic()-started,stopped_at_epoch=30)
    except Exception as error:
        event(root,'FAILED',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','execute','summarize'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-directory')
    args = parser.parse_args()
    if args.mode=='prepare':
        if not args.data_directory:
            parser.error('--data-directory is required for preparation')
        prepare(args.output,args.data_directory)
    elif args.mode=='execute':
        execute(args.output)
    else:
        from .lincs_biology_summary import summarize
        summarize(args.output)


if __name__=='__main__':
    main()
