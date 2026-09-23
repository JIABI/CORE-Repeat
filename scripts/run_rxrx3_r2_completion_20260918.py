"""Full fixed-architecture R2 on all approved RxRx3 dose-repeat conditions.

One chemical outer allocation, eight dose-specific tasks, full cached fits,
31 settings, 100k joint draws, and two additional decision-integration seeds.
No raw archive or protected-measurement API is used by this runner.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'scripts')]
import run_r2_core_comparison_20260917 as comparison
from run_jump_r2_completion_20260918 import save_npz, collect_cache, input_rows, role_rows
from opal2.biology_kernel_evaluation import write_json
from opal2.rxrx3_r2_dataset import build_dataset, grouped_parts, ROLE_KEYS, SEED
from opal2.eu_core_training import fit_complete_eu_core, predict_eu_core, choose_reference_ids
from opal2.eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from opal2.eu_core_experiment import select, extra_seed_moments, SAMPLES
from opal2.eu_r2_direct_baselines import fit_direct_baselines, evaluate_direct_distribution
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from opal2.gram_oof_ridge import transform_input, transform_target
from opal2.gram_simple_models import _fit_error_second_moment
from opal2.empirical_radial_experiment import score
from opal2.conditional_joint_error_experiment import observable_forward
from opal2.reference_information_diagnostic import bootstrap_difference

ROOT = PROJECT/'runs/rxrx3_r2_completion_20260918_v1'
REPORT = PROJECT/'reports/rxrx3_r2_completion_20260918_v1'
EXPORT = PROJECT/'data/rxrx3_r2_20260918/approved_export'
PREPARED = PROJECT/'data/rxrx3_r2_20260918/prepared_r2'
IDENTITY_KEYS = {'ids', 'actual'}


def make_manifest(data, metadata, parts, cells, outer_fold):
    return dict(dataset='RxRx3-core approved chemical development',
        n=len(data['ids']), n_chemical_groups=len(set(data['groups'])),
        outer_folds=5, cells=cells, dose_specific_models=True,
        seed=SEED, samples=SAMPLES, additional_mc_seed_offsets=[100000, 200000],
        mean_recipe='RIDGE -> validation-best HR -> A_OLD_GENERIC30 -> STATE50',
        inner_roles='global chemical 60/20/20 model/ref/cal; model 80/20 train/validation; every dose follows same roles',
        outer_assignment='metadata experiment-stratified chemical connectivity; no measurement input',
        endpoint='same-dose same-experiment ADD_TWO half-cosine gain minus 0.02',
        target_space=metadata['measurement_space'],
        control_only_assay_transform=True, train_only_model_transforms=True,
        main_estimand='available complete compound-dose conditions, not independent chemical count',
        dependence='resample connectivity across all doses; separately resample experiment sharing physical plates',
        budget='per outer-fold by dose: floor(.25*N) additional wells, k=budget//2',
        main_lambda=.2, control_lambda=0., ties='condition ID ascending',
        role_order=['X', 'Z1', 'Z2', 'V'], query_reference_condition_matched=True,
        biology_active=False, representation_active=False,
        protected_outcomes_used=False, confirmation_opened=False, formal_certificate=False,
        ids=data['ids'], groups=data['groups'], outer_fold=outer_fold,
        parts=[{key: data['ids'][rows].tolist() for key, rows in part.items()} for part in parts])


def freeze_manifest(manifest):
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    # Normalize numpy containers before comparing an existing immutable plan.
    from opal2.biology_kernel_evaluation import plain
    normalized = plain(manifest)
    path = ROOT/'run_manifest.json'
    if path.exists() and json.loads(path.read_text()) != normalized:
        raise ValueError('Existing RxRx3 run uses a different population or fixed plan')
    if not path.exists():
        write_json(path, manifest)
        write_json(REPORT/'run_manifest.json', manifest)


def cache_controls(folder, data, actual, part, cell):
    path = folder/'summary.json'
    if path.exists():
        return
    t, v, r, c, q = (part[key] for key in ROLE_KEYS)
    k = int(select(data['ids'][q], np.zeros(len(q)), np.zeros(len(q)), 0.).sum())
    rng = np.random.default_rng(SEED+500000+cell['cell'])
    draws = [rng.choice(len(q), k, replace=False) for _ in range(10000)]
    values = np.asarray([actual[q][row].sum() for row in draws])
    null = np.asarray([(actual[q][row] <= 0).sum() for row in draws])
    write_json(path, dict(n_query=len(q), cell=cell,
        random_same_budget=dict(k=k, expected_total_value=float(k*actual[q].mean()),
            total_value_quantiles=np.quantile(values, [.025, .5, .975]),
            null_count_quantiles=np.quantile(null, [.025, .5, .975])),
        fixed=dict(stop_total_value=0., add_all_total_value=float(actual[q].sum()),
            add_all_extra_wells=2*len(q), add_all_budget_matched=False),
        costs=dict(reference_if_all_new_wells=4*len(r), reference_if_X_already_available=3*len(r),
            reference_if_reusable_new_wells=0, model_train_and_validation_wells=4*(len(t)+len(v)),
            distribution_calibration_wells=4*len(c), query_initial_wells=len(q),
            query_replay_future_wells=3*len(q),
            control_cost='existing same-experiment EMPTY controls shared across doses; counts in prepared metadata, do not repurchase for every CV cell')))


def final_analysis(stores, data, actual, cells_by_row, outer_fold, cells):
    comparison.ROOT = ROOT
    comparison.SOURCE = ROOT
    comparison.analysis(stores, data, actual, cells_by_row,
        scope_text='approved RxRx3 chemical development; same-condition dose strata; grouped OOF; not confirmation',
        population_text=f"RxRx3: {len(data['ids'])} compound-dose conditions, {len(set(data['groups']))} chemical groups, {len(cells)} fold-dose cells. No protected measurements or pretrained image encoder used.")
    path = ROOT/'summary.json'
    payload = json.loads(path.read_text())
    payload.update(n_chemical_groups=len(set(data['groups'])), cells=cells,
        primary_unit='compound-dose condition', independent_identity_unit='chemical connectivity',
        direct_interval_semantics='RAW/CAL probabilities are classifier outputs; their Gamma CRPS belongs to the separate regression-residual law',
        not_cross_site_validation=True, control_resource_counts=json.loads((PREPARED/'metadata.json').read_text())['control_wells'])
    overlap, by_dose, per_identity = {}, {}, {}
    for name, out in stores.items():
        overlap[name] = {}
        for lam in (.2, 0.):
            key = f'lambda_{lam:g}'
            a, b = out['selected_'+key], stores['CORE_ORIGINAL']['selected_'+key]
            overlap[name][key] = dict(intersection=int(np.sum(a & b)),
                overlap_fraction=float(np.sum(a & b)/max(int(b.sum()), 1)),
                jaccard=float(np.sum(a & b)/max(int(np.sum(a | b)), 1)))
        # Equal identity descriptive scores: no extra fitting or sampling.
        keys = ('crps', 'brier', 'nll', 'energy')
        per_identity[name] = {key: float(np.mean([np.mean(out[key][data['groups'] == group])
            for group in np.unique(data['groups'])])) for key in keys if key in out}
    for dose in np.unique(data['dose']):
        ix = np.flatnonzero(data['dose'] == dose)
        dose_stores = {name: {key: value[ix].copy() for key, value in out.items()}
                       for name, out in stores.items()}
        metric = {name: comparison.summarize(out, actual[ix], data['ids'][ix], outer_fold[ix])
                  for name, out in dose_stores.items()}
        pairs = {}
        base = dose_stores['CORE_ORIGINAL']
        for name, out in dose_stores.items():
            if not name.startswith('DIRECT_ACCESS_MATCHED_'):
                continue
            pairs[name] = {}
            for key, left, right in (
                ('crps', out['crps'], base['crps']),
                ('brier', (out['p_null']-(actual[ix] <= 0))**2, (base['p_null']-(actual[ix] <= 0))**2),
                ('policy_value_per_candidate', actual[ix]*out['selected_lambda_0.2'], actual[ix]*base['selected_lambda_0.2'])):
                pairs[name][key] = bootstrap_difference(left, right, data['groups'][ix])
        by_dose[str(float(dose))] = dict(n_conditions=len(ix), n_chemical_groups=len(set(data['groups'][ix])),
            metrics=metric, paired_vs_CORE=pairs, scope='dose-stratified development analysis; no winner chosen')
    payload.update(selected_overlap_with_CORE=overlap, by_dose=by_dose,
        equal_chemical_weight_descriptive_scores=per_identity,
        compute_times_by_cell=[json.loads((ROOT/f"fold_{cell['cell']}"/'component_times.json').read_text()) for cell in cells])
    write_json(path, payload)
    write_json(REPORT/'summary.json', payload)
    save_npz(ROOT/'condition_metadata.npz', ids=data['ids'], groups=data['groups'],
        object_ids=data['object_ids'], dose=data['dose'], outer_fold=outer_fold,
        cell=cells_by_row, layout=data['layout'], well_ids=data['well_ids'])
    (REPORT/'REPORT.md').write_text((ROOT/'REPORT.md').read_text())


def require_complete_cell_cache(folder):
    """Aggregation may fail on a missing cache but may never silently refit."""
    required = ['complete.json', 'summary.json', 'component_times.json',
        'mean_predictions.npz', 'mean/preprocessing.json']
    names = ('CORE_ORIGINAL', 'CORE_LOCAL_GAUSSIAN', *comparison.NEW_JOINT)
    for name in names:
        cache_name = {'CORE_ORIGINAL':'AMP_EMP_LOCAL',
            'CORE_LOCAL_GAUSSIAN':'GAUSSIAN'}.get(name, name)
        required.append(cache_name+'.npz')
        required.extend(f'{cache_name}_mc{offset}.npz' for offset in (100000, 200000))
    for name in ('CORE', 'RIDGE_REF', 'HR_REF', 'STATE_REF'):
        required.append(name+'_distribution.joblib')
    for scope in ('ACCESS_MATCHED', 'TRAIN_MATCHED'):
        required.append(f'DIRECT_{scope}_fitted.joblib')
        required.extend(f'DIRECT_{scope}_{family}_prediction.npz'
            for family in ('RIDGE', 'EXTRATREES', 'HISTGB'))
    missing = [name for name in required if not (folder/name).exists()]
    if missing:
        raise RuntimeError(f'Incomplete aggregation cache at {folder}: {missing}; no refitting performed')


def run(prepare_only=False, worker_index=None, worker_count=1, aggregate_only=False):
    started = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    status_path = ROOT/('status.json' if worker_index is None else f'worker_{worker_index}_status.json')
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get('state') == 'COMPLETE' and (worker_index is None or
            previous.get('worker_count') == worker_count):
            print(status_path.read_text())
            return
    completed = 0
    def status(stage, **extra):
        write_json(status_path, dict(state='RUNNING', stage=stage, completed_cells=completed,
            elapsed_this_invocation_seconds=time.monotonic()-started, cpu_threads=1, **extra))
    try:
        status('approved_dataset_adapter')
        data, metadata = build_dataset(EXPORT, PREPARED)
        parts, cells, outer_fold = grouped_parts(data)
        freeze_manifest(make_manifest(data, metadata, parts, cells, outer_fold))
        for i, part in enumerate(parts):
            ref = role_rows(data, part['REF_FIT'], reference=True)
            choose_reference_ids(ref, metadata['chemical'], SEED+i*100+88001)
        if prepare_only or not (ROOT/'observables.npz').exists():
            # Prepared once before concurrent workers; no worker races over a
            # shared outcome cache or chooses a plan using these values.
            gram = profiles_to_gram(torch.as_tensor(data['Y'], dtype=torch.float64))
            save_npz(ROOT/'observables.npz', ids=data['ids'],
                raw_geometry=gram_to_coordinates(gram).numpy(),
                actual=gram_gains(gram).numpy()[:, 2], gram=gram.numpy())
        if prepare_only:
            write_json(status_path, dict(state='PREPARED', n=len(data['ids']),
                n_chemical_groups=len(set(data['groups'])), cells=len(cells), training_started=False))
            print(json.dumps(dict(n=len(data['ids']), groups=len(set(data['groups'])), cells=len(cells))))
            return
        ids, groups = data['ids'], data['groups']
        observed = comparison.read_npz(ROOT/'observables.npz')
        np.testing.assert_array_equal(observed['ids'], ids)
        raw, actual = observed['raw_geometry'], observed['actual']
        stores, counts, cell_by_row = {}, {}, np.full(len(ids), -1, int)
        for index, (part, cell) in enumerate(zip(parts, cells, strict=True)):
            if worker_index is not None and index % worker_count != worker_index:
                continue
            folder = ROOT/f'fold_{index}'
            folder.mkdir(exist_ok=True)
            if aggregate_only:
                require_complete_cell_cache(folder)
            t, v, r, c, q = (part[key] for key in ROLE_KEYS)
            cell_by_row[q] = index
            seed = SEED+index*100
            timing_path = folder/'component_times.json'
            timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
            def timed(label, call):
                before, cpu_before = time.monotonic(), time.process_time()
                value = call()
                if aggregate_only:
                    return value
                timing[label] = dict(wall_seconds=time.monotonic()-before,
                    process_cpu_seconds=time.process_time()-cpu_before, cpu_threads=1,
                    process_lifetime_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    memory_interpretation='process lifetime high-water mark on macOS, not isolated model peak',
                    outer_fold=cell['outer_fold'], dose_uM=cell['dose_uM'], query_rows=len(q))
                write_json(timing_path, timing)
                return value
            meanfolder = folder/'mean'
            if not (meanfolder/'complete.json').exists():
                if meanfolder.exists():
                    raise RuntimeError(f'Interrupted mean fit preserved at {meanfolder}; inspect before resuming')
                status('full_mean_fit', cell=index, total_cells=len(cells), dose_uM=cell['dose_uM'])
                timed('full_mean_fit', lambda: fit_complete_eu_core(role_rows(data,t), role_rows(data,v),
                    role_rows(data,r,True), metadata, meanfolder, seed=seed))
            stats = json.loads((meanfolder/'preprocessing.json').read_text())
            ridge, hr, state = (None, None, None) if aggregate_only else comparison.load_means(meanfolder)
            # Infer only the current dose. No need to run a dose model over other doses.
            pool = np.sort(np.concatenate(tuple(part.values())))
            x = np.empty((len(ids), len(stats['y_center'])+1), np.float64)
            target = np.empty_like(raw)
            x[pool], target[pool] = transform_input(data['Y'][pool,0], stats), transform_target(raw[pool], stats)
            def mean_prediction():
                result = {}
                result['RIDGE_REF'] = np.empty_like(raw)
                result['RIDGE_REF'][pool] = ridge.predict_mean(x[pool])
                result['HR_REF'] = np.empty_like(raw)
                with torch.no_grad():
                    result['HR_REF'][pool] = hr(torch.as_tensor(x[pool])).numpy()
                result['STATE_REF'] = np.empty_like(raw)
                result['STATE_REF'][pool] = predict_eu_core(state, stats, data['Y'][pool,0],
                    data['chem'][pool], data['chem_mask'][pool])['mean_u']
                return result
            if (folder/'mean_predictions.npz').exists():
                cached = comparison.read_npz(folder/'mean_predictions.npz')
                np.testing.assert_array_equal(cached['ids'], ids[pool])
                np.testing.assert_array_equal(cached['actual_u'], target[pool])
                means = {name: np.empty_like(raw) for name in ('RIDGE_REF','HR_REF','STATE_REF')}
                for name in means:
                    means[name][pool] = cached[name]
            else:
                means = timed('all_mean_forward', mean_prediction)
                save_npz(folder/'mean_predictions.npz', ids=ids[pool], row_indices=pool,
                    actual_u=target[pool], **{name: mean[pool] for name, mean in means.items()})
            specs = {}
            for name, mean in [('CORE', means['STATE_REF']), *means.items()]:
                law_path = folder/(name+'_distribution.joblib')
                if law_path.exists():
                    law_fit = joblib.load(law_path)
                else:
                    residual = target[r]-mean[r]
                    base = ridge.covariance if name == 'CORE' else _fit_error_second_moment(residual, include_bias=True)[0]
                    law_fit = timed(name+'_distribution_fit', lambda: fit_eu_distribution(
                        input_rows(data,r), residual, input_rows(data,c), target[c]-mean[c], base,
                        float(np.std(np.log(np.linalg.norm(data['Y'][t,0], axis=1)))),
                        model_training_ids=ids[np.r_[t,v]], model_training_groups=groups[np.r_[t,v]]))
                    joblib.dump(law_fit, law_path)
                    write_json(folder/(name+'_distribution.json'), law_fit['report'])
                distribution = timed(name+'_distribution_forward', lambda: predict_eu_distribution(law_fit, input_rows(data,q,mean[q])))
                save_npz(folder/(name+'_residuals.npz'), ref_ids=ids[r], cal_ids=ids[c], query_ids=ids[q],
                    ref_residual=target[r]-mean[r], cal_residual=target[c]-mean[c], query_mean=mean[q],
                    query_scatter=distribution['scatter_u'], base_query_scatter=distribution['base_scatter_u'],
                    ref_groups=groups[r], cal_groups=groups[c], query_groups=groups[q])
                aliases = ('CORE_ORIGINAL','CORE_LOCAL_GAUSSIAN','CORE_AMP_GAUSSIAN') if name == 'CORE' else (name,name+'_LOCAL_GAUSSIAN',name+'_AMP_GAUSSIAN')
                specs[aliases[0]] = dict(mean=mean[q], scatter=distribution['scatter_u'], law=distribution['law'], weights=distribution['radial_weights'])
                specs[aliases[1]] = dict(mean=mean[q], scatter=distribution['base_scatter_u'], law=None, weights=None)
                specs[aliases[2]] = dict(mean=mean[q], scatter=distribution['scatter_u'], law=None, weights=None)
            _, obs, difference, _ = observable_forward(raw[q])
            norm2 = np.square(data['Y'][q,0]).mean(1)
            absolute = np.log1p(difference*norm2[:,None])
            for name, spec in specs.items():
                cache_name = {'CORE_ORIGINAL':'AMP_EMP_LOCAL','CORE_LOCAL_GAUSSIAN':'GAUSSIAN'}.get(name,name)
                path = folder/(cache_name+'.npz')
                if not path.exists():
                    status('joint_score', cell=index, total_cells=len(cells), dose_uM=cell['dose_uM'], arm=name)
                    out = timed(name+'_100k_score', lambda: score(spec['mean'],spec['scatter'],target[q],stats,
                        actual[q],obs,absolute,norm2,seed,law=spec['law'],weights=spec['weights'],samples=SAMPLES))
                    out.update(mean_u=spec['mean'], actual_u=target[q], scatter_u=spec['scatter'], brier=(out['p_null']-(actual[q]<=0))**2)
                    save_npz(path, ids=ids[q], actual=actual[q], **out)
                saved = comparison.read_npz(path)
                np.testing.assert_array_equal(saved['ids'], ids[q])
                np.testing.assert_array_equal(saved['actual'], actual[q])
                collect_cache(stores,counts,name,q,{k:v for k,v in saved.items() if k not in IDENTITY_KEYS},len(ids))
                for offset in (100000,200000):
                    mpath = folder/f'{cache_name}_mc{offset}.npz'
                    if not mpath.exists():
                        status('MC_sensitivity', cell=index, total_cells=len(cells), arm=name, seed_offset=offset)
                        moment = timed(name+f'_100k_mc{offset}', lambda: extra_seed_moments(spec['mean'],spec['scatter'],stats,
                            law=spec['law'],weights=spec['weights'],seed=seed+offset))
                        save_npz(mpath, ids=ids[q], **moment)
            direct_x = np.column_stack((x[pool],data['chem'][pool]))
            inverse = np.full(len(ids), -1, int)
            inverse[pool] = np.arange(len(pool))
            for scope, training in (('ACCESS_MATCHED',np.r_[t,r]),('TRAIN_MATCHED',t)):
                fitted_path = folder/f'DIRECT_{scope}_fitted.joblib'
                if fitted_path.exists():
                    fitted = joblib.load(fitted_path)
                else:
                    status('direct_baselines', cell=index, total_cells=len(cells), arm=scope)
                    fitted = timed('direct_fit_'+scope, lambda: fit_direct_baselines(direct_x[inverse[training]],actual[training],
                        direct_x[inverse[v]],actual[v],direct_x[inverse[c]],actual[c],direct_x[inverse[q]],seed=seed))
                    joblib.dump(fitted,fitted_path)
                for family, arm in fitted.items():
                    prefix = f'DIRECT_{scope}_{family}'
                    path = folder/(prefix+'_prediction.npz')
                    if not path.exists():
                        evaluated = timed(prefix+'_scalar_distribution_score', lambda: evaluate_direct_distribution(arm,actual[q]))
                        save_npz(path,ids=ids[q],actual=actual[q],**{k:arm[k] for k in
                            ('predicted','p_null','p_null_calibrated','gamma_residuals','gamma_distribution_mean','p_null_from_gamma')},**evaluated)
                        write_json(folder/(prefix+'.json'),dict(metadata=arm['metadata'],train_ids=ids[training],valid_ids=ids[v],cal_ids=ids[c],query_ids=ids[q]))
                    saved = comparison.read_npz(path)
                    np.testing.assert_array_equal(saved['ids'],ids[q])
                    common = {k:saved[k] for k in ('crps','gamma_coverage_by_level')}
                    for suffix,mk,pk in (('COHERENT','gamma_distribution_mean','p_null_from_gamma'),('CLASSIFIER_RAW','predicted','p_null'),('CLASSIFIER_CAL','predicted','p_null_calibrated')):
                        collect_cache(stores,counts,prefix+'_'+suffix,q,dict(common,predicted=saved[mk],p_null=saved[pk]),len(ids))
            values = np.sort(actual[np.r_[t,r]])
            constant = dict(predicted=np.full(len(q),values.mean()),gamma_residuals=values-values.mean(),
                p_null=np.full(len(q),np.mean(values<=0)),p_null_calibrated=np.full(len(q),np.mean(values<=0)))
            evaluated = evaluate_direct_distribution(constant,actual[q])
            collect_cache(stores,counts,'CONSTANT_ACCESS_MATCHED',q,dict(predicted=constant['predicted'],p_null=constant['p_null'],
                crps=evaluated['crps'],gamma_coverage_by_level=evaluated['gamma_coverage_by_level']),len(ids))
            cache_controls(folder,data,actual,part,cell)
            completed += 1
            if not aggregate_only:
                write_json(folder/'complete.json',dict(cell=cell,complete=True,elapsed_this_invocation_seconds=time.monotonic()-started))
            print(f'RxRx3 {completed}/{len(cells)} complete; outer fold {cell["outer_fold"]}, dose {cell["dose_uM"]}; elapsed {time.monotonic()-started:.1f}s',flush=True)
        if worker_index is not None:
            write_json(status_path, dict(state='COMPLETE', worker_index=worker_index,
                worker_count=worker_count, completed_cells=completed,
                elapsed_this_invocation_seconds=time.monotonic()-started))
            return
        if np.any(cell_by_row < 0) or any(not np.all(n == 1) for n in counts.values()):
            raise ValueError('Missing/duplicate OOF predictions')
        status('paired_analysis', total_cells=len(cells))
        final_analysis(stores,data,actual,cell_by_row,outer_fold,cells)
        write_json(status_path,dict(state='COMPLETE',completed_cells=completed,arms=len(stores),
            elapsed_this_invocation_seconds=time.monotonic()-started,report=str(REPORT/'REPORT.md')))
        refresh = PROJECT/'scripts/summarize_r2_four_datasets_20260918.py'
        if refresh.exists():
            subprocess.run([sys.executable,str(refresh),'--refresh'],check=True)
    except Exception:
        write_json(status_path,dict(state='FAILED',completed_cells=completed,
            elapsed_this_invocation_seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        raise


def parallel_run(workers):
    """Bounded CPU workers, disjoint cell caches, then one unified analysis."""
    run(prepare_only=True)
    total_cells = len(json.loads((ROOT/'run_manifest.json').read_text())['cells'])
    launched, handles = [], []
    start = time.monotonic()
    try:
        for index in range(workers):
            log = (ROOT/f'worker_{index}.log').open('a')
            handles.append(log)
            command = [sys.executable, '-u', str(Path(__file__).resolve()),
                '--worker-index', str(index), '--worker-count', str(workers)]
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            launched.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env))
        while True:
            completed = sum((ROOT/f'fold_{i}'/'complete.json').exists() for i in range(total_cells))
            codes = [p.poll() for p in launched]
            failed = [i for i, code in enumerate(codes) if code not in (None, 0)]
            write_json(ROOT/'status.json', dict(state='WORKER_FAILURE' if failed else 'RUNNING',
                stage='full_R2_cells', workers=workers, worker_pids=[p.pid for p in launched],
                worker_returncodes=codes, failed_workers=failed,
                completed_cells=completed, total_cells=total_cells, cpu_threads_per_worker=1,
                elapsed_this_invocation_seconds=time.monotonic()-start))
            if all(code is not None for code in codes):
                break
            time.sleep(30)
        if failed:
            raise RuntimeError(f'Workers {failed} failed; completed cell caches retained')
        run(aggregate_only=True)
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--parallel-workers', type=int, choices=(1, 2))
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.parallel_workers:
            parallel_run(args.parallel_workers)
        else:
            run(args.prepare_only, args.worker_index, args.worker_count, args.aggregate_only)
