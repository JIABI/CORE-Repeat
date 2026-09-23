"""Full JUMP R2 comparison, retaining historical outer queries and caching every fit.

Only source5_primary_fullcontrols (639 opened objects, four roles) is read.
The complete EU architecture is reused, not replaced by a reduced surrogate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_r2_core_comparison_20260917 as comparison
from opal2.biology_kernel_evaluation import write_json
from opal2.eu_development_plan import allocate_groups
from opal2.eu_core_training import fit_complete_eu_core, predict_eu_core, choose_reference_ids
from opal2.eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from opal2.eu_core_experiment import select, extra_seed_moments, SAMPLES
from opal2.eu_r2_direct_baselines import fit_direct_baselines, evaluate_direct_distribution
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from opal2.gram_oof_ridge import transform_input, transform_target
from opal2.gram_simple_models import _fit_error_second_moment
from opal2.empirical_radial_experiment import score
from opal2.conditional_joint_error_experiment import observable_forward

SOURCE = PROJECT / 'data/source5_primary_fullcontrols'
OLD = PROJECT / 'runs/gram_oof_20260914_v1/run_manifest.json'
LEDGER = PROJECT / 'reports/new_data_qualification_20260917_v1/historical_identity_ledger.csv'
ROOT = PROJECT / 'runs/jump_r2_completion_20260918_v1'
REPORT = PROJECT / 'reports/jump_r2_completion_20260918_v1'
SEED = 20260918
ROLE_KEYS = ('TRAIN', 'VALIDATION', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL')
IDENTITY_KEYS = {'ids', 'actual'}


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_npz(path):
    return comparison.read_npz(path)


def chemical_groups(ids, ledger):
    mapping = {}
    for row in ledger:
        if row['dataset'] != 'JUMP_source5_DEV':
            continue
        oid, group = row['object_id'], row['connectivity']
        if not group or (oid in mapping and mapping[oid] != group):
            raise ValueError('Unresolved or conflicting JUMP chemistry identity')
        mapping[oid] = group
    if any(str(oid) not in mapping for oid in ids):
        raise ValueError('Identity ledger does not cover opened JUMP population')
    return np.asarray([mapping[str(oid)] for oid in ids])


def make_parts(ids, groups, layout, old_manifest):
    """Retain all historical outer query IDs; assign inner roles using metadata only."""
    if len(ids) != len(set(ids)) or len(groups) != len(ids):
        raise ValueError('Aligned unique identities required')
    np.testing.assert_array_equal(old_manifest['ids'], ids)
    all_ids = set(ids)
    result, query_count = [], np.zeros(len(ids), int)
    for f, previous in enumerate(old_manifest['folds']):
        if previous['fold'] != f:
            raise ValueError('Historical fold order changed')
        query_ids = set(previous['test_ids'])
        if not query_ids <= all_ids:
            raise ValueError('Historical query contains an unknown identity')
        q = np.flatnonzero(np.isin(ids, list(query_ids)))
        pool = np.flatnonzero(~np.isin(ids, list(query_ids)))
        if set(groups[q]) & set(groups[pool]):
            raise ValueError('Historical outer fold splits a chemical group')
        strata = {str(g): '|'.join(sorted(set(layout[pool][groups[pool] == g])))
                  for g in np.unique(groups[pool])}
        outer_role = allocate_groups(strata, ['MODEL_FIT', 'REF_FIT', 'DIST_CAL'],
                                     [.6, .2, .2], SEED + 100 + f)
        model_role = allocate_groups({g: s for g, s in strata.items() if outer_role[g] == 'MODEL_FIT'},
                                     ['TRAIN', 'VALIDATION'], [.8, .2], SEED + 1000 + f)
        part = {key: [] for key in ROLE_KEYS}
        for i in range(len(ids)):
            key = 'DEV_EVAL' if ids[i] in query_ids else model_role.get(str(groups[i]), outer_role.get(str(groups[i])))
            part[key].append(i)
        part = {key: np.asarray(rows, int) for key, rows in part.items()}
        if any(not len(rows) for rows in part.values()):
            raise ValueError('An empty role was generated')
        if any(set(groups[part[a]]) & set(groups[part[b]]) for j,a in enumerate(ROLE_KEYS) for b in ROLE_KEYS[:j]):
            raise ValueError('Chemical identities cross roles')
        query_count[part['DEV_EVAL']] += 1
        result.append(part)
    if len(result) != 5 or not np.all(query_count == 1):
        raise ValueError('Every opened object must be queried exactly once')
    return result


def prepare():
    # Do not load controls or any protected archive as an incidental array.
    with np.load(SOURCE / 'measurements.npz', allow_pickle=False) as z:
        data = {key: z[key].copy() for key in ('ids', 'chem', 'chem_mask', 'well_ids')}
    with LEDGER.open(newline='') as handle:
        data['groups'] = chemical_groups(data['ids'], list(csv.DictReader(handle)))
    data['layout'] = np.asarray(['|'.join(w.rsplit('::', 1)[0] for w in row) for row in data['well_ids']])
    old = json.loads(OLD.read_text())
    if len(data['ids']) != 639 or len(set(data['groups'])) != 639 or old['final_opened'] or old['fifth_repeat_opened']:
        raise ValueError('Opened development scope differs from the approved 639 objects')
    parts = make_parts(data['ids'], data['groups'], data['layout'], old)
    metadata = json.loads((SOURCE / 'measurements.json').read_text())['metadata']
    # Check reference availability without touching future outcomes.
    with np.load(SOURCE / 'measurements.npz', allow_pickle=False) as z:
        data['Y'] = z['Y'].copy()
    if data['Y'].shape != (639, 4, 3617) or not np.isfinite(data['Y']).all():
        raise ValueError('Unexpected JUMP profile shape or missing outcomes')
    for f, part in enumerate(parts):
        r = part['REF_FIT']
        reference = {key: data[key][r] for key in ('ids', 'groups', 'chem', 'chem_mask')}
        reference['X'] = data['Y'][r, 0]
        choose_reference_ids(reference, metadata['chemical'], SEED + f*100)
    return data, metadata, parts


def freeze_plan(data, parts):
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = dict(dataset='JUMP source_5 opened DEV', n=len(data['ids']), folds=5,
        seed=SEED, samples=SAMPLES, old_manifest=str(OLD), data=str(SOURCE),
        mean_recipe='RIDGE -> validation-best HR -> A_OLD_GENERIC30 -> STATE50',
        outer_query_membership_preserved=True, inner_roles='60/20/20 model/ref/cal; model train/valid 80/20',
        inner_role_selection='fixed-seed chemical-group allocation stratified by plate tuple; no outcome input',
        legacy_outer_stratification='Historical outer folds were stratified on first-well amplitude; unchanged',
        identity_count=len(set(data['groups'])), layout_count=len(set(data['layout'])),
        layout_interval_interpretation='Only two physical plate tuples: descriptive sensitivity, not a reliable population confidence guarantee',
        reference_bank_excluded_from_backbone=True, mean_models_refitted=True,
        additional_mc_seed_offsets=[100000, 200000],
        budget='each outer query pool: floor(.25*N) additional wells; k=budget//2',
        main_lambda=.2, control_lambda=0., selection_ties='object ID ascending',
        biology=False, representation=False, final_opened=False, fifth_repeat_opened=False,
        confirmation_opened=False, formal_certificate=False,
        parts=[{key: data['ids'][rows].tolist() for key, rows in part.items()} for part in parts])
    path = ROOT / 'run_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Cannot resume under a different population, plan or configuration')
    write_json(path, manifest)
    write_json(REPORT / 'run_manifest.json', manifest)
    save_npz(ROOT / 'identity_layout.npz', **{key: data[key] for key in ('ids', 'groups', 'layout', 'well_ids')})
    return manifest


def input_rows(data, rows, mean=None):
    out = {key: data[key][rows] for key in ('ids', 'groups', 'chem')}
    out['X'] = data['Y'][rows, 0]
    if mean is not None:
        out['mean_u'] = mean
    return out


def role_rows(data, rows, reference=False):
    out = {key: data[key][rows] for key in ('ids', 'groups', 'chem', 'chem_mask')}
    out['X' if reference else 'Y'] = data['Y'][rows, 0] if reference else data['Y'][rows]
    return out


def cache_controls(folder, data, actual, part):
    t, v, r, c, q = (part[key] for key in ROLE_KEYS)
    if (folder / 'summary.json').exists():
        return
    k = int(select(data['ids'][q], np.zeros(len(q)), np.zeros(len(q)), 0.).sum())
    rng = np.random.default_rng(SEED + int(folder.name.split('_')[-1]) + 500000)
    chosen = [rng.choice(len(q), k, replace=False) for _ in range(10000)]
    values = np.asarray([actual[q][ix].sum() for ix in chosen])
    false = np.asarray([(actual[q][ix] <= 0).sum() for ix in chosen])
    write_json(folder / 'summary.json', dict(n_query=len(q),
        random_same_budget=dict(k=k, expected_total_value=float(k*actual[q].mean()),
            total_value_quantiles=np.quantile(values,[.025,.5,.975]), null_count_quantiles=np.quantile(false,[.025,.5,.975])),
        fixed=dict(stop_total_value=0.,add_all_total_value=float(actual[q].sum()), add_all_extra_wells=2*len(q), add_all_budget_matched=False),
        costs=dict(reference_if_all_new_wells=4*len(r),reference_if_X_already_available=3*len(r),
            reference_if_reusable_new_wells=0,model_train_and_validation_wells=4*(len(t)+len(v)),
            distribution_calibration_wells=4*len(c),query_initial_wells=len(q),query_replay_future_wells=3*len(q),
            control_cost='Pre-existing preparation; not attributed to new acquisition. Report separately from reference amortization')))


def collect_cache(stores, counts, name, q, out, n):
    q = np.asarray(q)
    if q.ndim != 1 or q.dtype.kind not in 'iu' or len(set(q.tolist())) != len(q) or np.any((q<0)|(q>=n)):
        raise ValueError('Query indices must be unique, valid integer rows')
    if name in stores:
        if set(stores[name]) != set(out) or np.any(counts[name][q]):
            raise ValueError('Cached fields changed or query rows were already collected')
    validated = {}
    for key, value in out.items():
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[0] != len(q) or not np.isfinite(value).all():
            raise ValueError(f'Invalid per-object output {name}/{key}')
        if name in stores and (stores[name][key].shape[1:] != value.shape[1:] or stores[name][key].dtype != value.dtype):
            raise ValueError(f'Cached output shape or dtype changed: {name}/{key}')
        validated[key] = value
    if name not in stores:
        stores[name] = {}
        counts[name] = np.zeros(n, int)
    counts[name][q] += 1
    for key, value in validated.items():
        if key not in stores[name]:
            stores[name][key] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
        stores[name][key][q] = value


def run(prepare_only=False):
    started = time.monotonic()
    data, metadata, parts = prepare()
    manifest = freeze_plan(data, parts)
    if prepare_only:
        print(json.dumps({key: manifest[key] for key in ('n','identity_count','layout_count','samples')}, indent=2))
        print(json.dumps([{key: len(rows) for key,rows in part.items()} for part in parts], indent=2))
        return
    old_status = ROOT / 'status.json'
    if old_status.exists() and json.loads(old_status.read_text()).get('state') == 'COMPLETE':
        print(old_status.read_text()); return
    def status(stage, **extra):
        write_json(ROOT/'status.json',dict(state='RUNNING',stage=stage,elapsed_this_invocation_seconds=time.monotonic()-started,**extra))
    try:
        ids, groups = data['ids'], data['groups']
        raw_gram = profiles_to_gram(torch.as_tensor(data['Y'],dtype=torch.float64))
        raw = gram_to_coordinates(raw_gram).numpy()
        actual = gram_gains(raw_gram).numpy()[:,2]
        save_npz(ROOT/'observables.npz',ids=ids,raw_geometry=raw,actual=actual,gram=raw_gram.numpy())
        stores, counts, folds = {}, {}, np.full(len(ids),-1,int)
        for f, part in enumerate(parts):
            folder = ROOT/f'fold_{f}'; folder.mkdir(exist_ok=True)
            timing_path = folder/'component_times.json'
            timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
            def timed(label, function):
                before = time.monotonic()
                value = function()
                timings[label] = dict(wall_seconds=time.monotonic()-before,cpu_threads=1)
                write_json(timing_path,timings)
                return value
            t,v,r,c,q = (part[key] for key in ROLE_KEYS)
            folds[q] = f
            meanfolder = folder/'mean'
            if not (meanfolder/'complete.json').exists():
                if meanfolder.exists():
                    raise RuntimeError(f'Interrupted mean fit preserved at {meanfolder}; inspect its checkpoints before restarting')
                status('complete_mean_fit',fold=f,completed_folds=f)
                timed('full_mean_fit',lambda: fit_complete_eu_core(role_rows(data,t),role_rows(data,v),role_rows(data,r,True),
                    metadata,meanfolder,seed=SEED+f*100))
            stats = json.loads((meanfolder/'preprocessing.json').read_text())
            ridge, hr, state = comparison.load_means(meanfolder)
            x = transform_input(data['Y'][:,0],stats)
            target = transform_target(raw,stats)
            means = {'RIDGE_REF':ridge.predict_mean(x)}
            with torch.no_grad():
                means['HR_REF'] = hr(torch.as_tensor(x)).numpy()
            means['STATE_REF'] = predict_eu_core(state,stats,data['Y'][:,0],data['chem'],data['chem_mask'])['mean_u']
            save_npz(folder/'mean_predictions.npz',ids=ids,raw_target=raw,actual_u=target,
                     **{name:mu for name,mu in means.items()})
            specs = {}
            for name, mean in [('CORE',means['STATE_REF']), *means.items()]:
                path = folder/(name+'_distribution.joblib')
                if path.exists():
                    law_fit = joblib.load(path)
                else:
                    residual = target[r]-mean[r]
                    base = ridge.covariance if name == 'CORE' else _fit_error_second_moment(residual,include_bias=True)[0]
                    law_fit = fit_eu_distribution(input_rows(data,r),residual,input_rows(data,c),target[c]-mean[c],base,
                        float(np.std(np.log(np.linalg.norm(data['Y'][t,0],axis=1)))),
                        model_training_ids=ids[np.r_[t,v]],model_training_groups=groups[np.r_[t,v]])
                    joblib.dump(law_fit,path)
                    write_json(folder/(name+'_distribution.json'),law_fit['report'])
                distribution = predict_eu_distribution(law_fit,input_rows(data,q,mean[q]))
                save_npz(folder/(name+'_residuals.npz'),ref_ids=ids[r],cal_ids=ids[c],query_ids=ids[q],
                    ref_residual=target[r]-mean[r],cal_residual=target[c]-mean[c],query_mean=mean[q],
                    query_scatter=distribution['scatter_u'],base_query_scatter=distribution['base_scatter_u'])
                aliases = ('CORE_ORIGINAL','CORE_LOCAL_GAUSSIAN','CORE_AMP_GAUSSIAN') if name=='CORE' else (name,name+'_LOCAL_GAUSSIAN',name+'_AMP_GAUSSIAN')
                specs[aliases[0]] = dict(mean=mean[q],scatter=distribution['scatter_u'],law=distribution['law'],weights=distribution['radial_weights'])
                specs[aliases[1]] = dict(mean=mean[q],scatter=distribution['base_scatter_u'],law=None,weights=None)
                specs[aliases[2]] = dict(mean=mean[q],scatter=distribution['scatter_u'],law=None,weights=None)
            _, obs, difference, _ = observable_forward(raw[q])
            norm2 = np.square(data['Y'][q,0]).mean(1)
            absolute = np.log1p(difference*norm2[:,None])
            for name,spec in specs.items():
                cache_name = {'CORE_ORIGINAL':'AMP_EMP_LOCAL','CORE_LOCAL_GAUSSIAN':'GAUSSIAN'}.get(name,name)
                path = folder/(cache_name+'.npz')
                if not path.exists():
                    status('joint_score',fold=f,arm=name,completed_folds=f)
                    out = timed(name+'_100k_score',lambda: score(spec['mean'],spec['scatter'],target[q],stats,actual[q],obs,absolute,norm2,
                        SEED+f*100,law=spec['law'],weights=spec['weights'],samples=SAMPLES))
                    out.update(mean_u=spec['mean'],actual_u=target[q],scatter_u=spec['scatter'],brier=(out['p_null']-(actual[q]<=0))**2)
                    save_npz(path,ids=ids[q],actual=actual[q],**out)
                saved = read_npz(path)
                np.testing.assert_array_equal(saved['ids'],ids[q]); np.testing.assert_array_equal(saved['actual'],actual[q])
                collect_cache(stores,counts,name,q,{k:v for k,v in saved.items() if k not in IDENTITY_KEYS},len(ids))
                for offset in (100000,200000):
                    mpath = folder/f'{cache_name}_mc{offset}.npz'
                    if not mpath.exists():
                        status('decision_MC_sensitivity',fold=f,arm=name,seed_offset=offset,completed_folds=f)
                        moment = timed(name+f'_100k_mc{offset}',lambda: extra_seed_moments(spec['mean'],spec['scatter'],stats,law=spec['law'],weights=spec['weights'],seed=SEED+f*100+offset))
                        save_npz(mpath,ids=ids[q],**moment)
            direct_x = np.column_stack((x,data['chem']))
            for scope, training in (('ACCESS_MATCHED',np.r_[t,r]),('TRAIN_MATCHED',t)):
                fitted_path = folder/f'DIRECT_{scope}_fitted.joblib'
                if fitted_path.exists():
                    fitted = joblib.load(fitted_path)
                else:
                    status('direct_baselines',fold=f,arm=scope,completed_folds=f)
                    fitted = timed('direct_fit_'+scope,lambda: fit_direct_baselines(direct_x[training],actual[training],direct_x[v],actual[v],direct_x[c],actual[c],direct_x[q],seed=SEED+f*100))
                    joblib.dump(fitted,fitted_path)
                for family, arm in fitted.items():
                    prefix = f'DIRECT_{scope}_{family}'
                    path = folder/(prefix+'_prediction.npz')
                    if not path.exists():
                        evaluated = evaluate_direct_distribution(arm,actual[q])
                        save_npz(path,ids=ids[q],actual=actual[q],**{k:arm[k] for k in ('predicted','p_null','p_null_calibrated','gamma_residuals','gamma_distribution_mean','p_null_from_gamma')},**evaluated)
                        write_json(folder/(prefix+'.json'),dict(metadata=arm['metadata'],train_ids=ids[training],valid_ids=ids[v],cal_ids=ids[c],query_ids=ids[q]))
                    saved = read_npz(path)
                    np.testing.assert_array_equal(saved['ids'],ids[q]); np.testing.assert_array_equal(saved['actual'],actual[q])
                    common = {k:saved[k] for k in ('crps','gamma_coverage_by_level')}
                    for suffix, mk, pk in (('COHERENT','gamma_distribution_mean','p_null_from_gamma'),('CLASSIFIER_RAW','predicted','p_null'),('CLASSIFIER_CAL','predicted','p_null_calibrated')):
                        collect_cache(stores,counts,prefix+'_'+suffix,q,dict(common,predicted=saved[mk],p_null=saved[pk]),len(ids))
            values = np.sort(actual[np.r_[t,r]])
            constant = dict(predicted=np.full(len(q),values.mean()),gamma_residuals=values-values.mean(),p_null=np.full(len(q),np.mean(values<=0)),p_null_calibrated=np.full(len(q),np.mean(values<=0)))
            evaluated = evaluate_direct_distribution(constant,actual[q])
            collect_cache(stores,counts,'CONSTANT_ACCESS_MATCHED',q,dict(predicted=constant['predicted'],p_null=constant['p_null'],crps=evaluated['crps'],gamma_coverage_by_level=evaluated['gamma_coverage_by_level']),len(ids))
            cache_controls(folder,data,actual,part)
            write_json(folder/'complete.json',dict(fold=f,complete=True,elapsed_this_invocation_seconds=time.monotonic()-started))
            print(f'JUMP fold {f+1}/5 all fit/score caches complete, elapsed {time.monotonic()-started:.1f}s',flush=True)
        if np.any(folds<0) or any(not np.all(n==1) for n in counts.values()):
            raise ValueError('Incomplete or duplicate out-of-fold predictions')
        status('paired_analysis',completed_folds=5)
        comparison.ROOT=ROOT; comparison.SOURCE=ROOT
        comparison.analysis(stores,data,actual,folds,
            scope_text='Opened JUMP 639 DEV; historical outer queries, new independent REF full CORE, development diagnostics',
            population_text='JUMP source_5, 639 opened DEV objects. Full CORE fit once per fold; all 31 settings use the same query pools. FINAL and fifth repeat remain closed. Only two plate tuples: layout resampling is descriptive, not certification.')
        # Add overlap explicitly for all policies without refitting or resampling.
        overlap = {}
        base = stores['CORE_ORIGINAL']['selected_lambda_0.2']
        for name,out in stores.items():
            selected = out['selected_lambda_0.2']
            overlap[name] = dict(shared=int(np.sum(base&selected)),core_n=int(base.sum()),
                retention=float(np.sum(base&selected)/base.sum()),jaccard=float(np.sum(base&selected)/np.sum(base|selected)))
        write_json(ROOT/'selection_overlap.json',overlap)
        legacy_path = OLD.parent/'L_GRAM_oof_predictions.npz'
        legacy = read_npz(legacy_path)
        np.testing.assert_array_equal(legacy['ids'],ids)
        np.testing.assert_array_equal(legacy['fold'],folds)
        np.testing.assert_allclose(legacy['actual'][:,2],actual,atol=1e-12,rtol=1e-12)
        legacy_out = dict(predicted=legacy['predicted'][:,2],p_null=legacy['p_null'][:,2],crps=legacy['utility_crps'][:,2])
        write_json(ROOT/'historical_L_reference.json',dict(source=str(legacy_path),
            role='historical reference, not a new resource-matched fit',samples=2000,
            training_access='original larger fit pool; no new independent REF partition',
            query_membership_identical=True,metrics=comparison.summarize(legacy_out,actual,ids,folds)))
        write_json(ROOT/'status.json',dict(state='COMPLETE',completed_folds=5,arms=len(stores),elapsed_this_invocation_seconds=time.monotonic()-started,report=str(ROOT/'REPORT.md')))
    except Exception:
        write_json(ROOT/'status.json',dict(state='FAILED',elapsed_this_invocation_seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        run(args.prepare_only)
