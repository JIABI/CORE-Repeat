"""Fixed-mean conditional contrast/remainder covariance experiment on LINCS."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2, spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_joint_error import fit_covariance_family
from .conditional_joint_error_experiment import (PROJECT, SEED, SAMPLES, OBSERVABLES,
    read_json, observable_forward, reference_weights, score_distribution)
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .joint_contrast_scale import (contrast_projector, projected_energy, fit_scale,
    predict_scale, rescale_components)
from .lincs_biology_experiment import load_data
from .reference_information_diagnostic import bootstrap_difference

ARMS = ('LOCAL_FIT', 'SPLIT_CONSTANT', 'AMPLITUDE_TOTAL', 'AMPLITUDE_SPLIT')
COMPARE = ('nll', 'energy', 'single_crps', 'pair_crps', 'average_crps',
           'triple_average_crps', 'absolute_pair_crps', 'crps', 'brier',
           'policy_value', 'coverage', 'joint_coverage')


def summarize(stores, actual, data, layout, folds):
    metrics = {}
    for arm, out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = out['selected']*actual
        out['mse'] = np.square(out['actual_u']-out['mean_u']).mean(1)
        chosen = out['selected'].astype(bool)
        if chosen.sum() != 146:
            raise ValueError('Query budget changed')
        row = {k: float(v.mean()) for k, v in out.items() if v.ndim == 1 and k != 'selected'}
        row.update(selected_n=int(chosen.sum()), selected_null=int((actual[chosen] <= 0).sum()),
            selected_mean=float(actual[chosen].mean()),
            selected_symmetric_difference=int(np.count_nonzero(out['selected'] != stores['LOCAL_FIT']['selected'])),
            spearman=float(spearmanr(out['predicted'], actual).statistic),
            null_auc=float(roc_auc_score(actual <= 0, out['p_null'])),
            joint_coverage_by_level={str(p): float((out['mahalanobis2'] <= chi2.ppf(p, 9)).mean())
                for p in (.5, .8, .9, .95, .99)},
            scale_quantiles={k: {str(q): float(np.quantile(out[k], q)) for q in (0, .1, .5, .9, 1)}
                for k in ('contrast_scale', 'remainder_scale')},
            observable_crps=dict(zip(OBSERVABLES, out['observable_crps'].mean(0).tolist())),
            observable_coverage=dict(zip(OBSERVABLES, out['observable_coverage'].mean(0).tolist())))
        metrics[arm] = row
    pairs = [(a, 'LOCAL_FIT') for a in ARMS[1:]]+[
        ('AMPLITUDE_SPLIT', 'AMPLITUDE_TOTAL'), ('AMPLITUDE_SPLIT', 'SPLIT_CONSTANT')]
    comparisons = {a+' minus '+b: {key: {scope: bootstrap_difference(stores[a][key], stores[b][key], labels)
        for scope, labels in (('chemistry', data['groups']), ('layout', layout))}
        for key in COMPARE} for a, b in pairs}
    by_fold = {str(f): {a: {k: float(stores[a][k][folds == f].mean()) for k in COMPARE}
        for a in ARMS} for f in np.unique(folds)}
    return metrics, comparisons, by_fold


def run(previous_run, output):
    started = time.monotonic()
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/JOINT_COMPONENT_SCALE_PLAN_20260916.md', root/'PROTOCOL.md')
    for name in ('joint_contrast_scale.py', 'joint_component_scale_experiment.py',
                 'conditional_joint_error.py', 'conditional_joint_error_experiment.py'):
        shutil.copy2(PROJECT/'opal2'/name, root/name)
    previous = read_json(Path(previous_run)/'summary.json')
    reference = Path(previous['reference_run'])
    manifest = read_json(reference/'run_manifest.json')
    data, metadata = load_data(manifest['data_directory'])
    n = len(data['ids'])
    if n != 1188 or data['ids'].tolist() != manifest['ids']:
        raise ValueError('Opened LINCS scope changed')
    lookup = {v: i for i, v in enumerate(data['ids'])}
    layout = np.array([v['layout_block'] for v in metadata['units']])
    lognorm = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    norm2_per_feature = np.square(data['Y'][:, 0]).mean(1)
    gram = profiles_to_gram(torch.tensor(data['Y']))
    raw = gram_to_coordinates(gram).numpy()
    actual, obs_actual, differences, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, gram_gains(gram).numpy()[:, 2], atol=1e-12, rtol=1e-12)
    absolute_actual = np.log1p(differences*norm2_per_feature[:, None])
    stores = {a: dict(mean_u=np.empty((n, 9)), actual_u=np.empty((n, 9)),
        covariance_u=np.empty((n, 9, 9)), selected=np.zeros(n, dtype=int),
        contrast_scale=np.empty(n), remainder_scale=np.empty(n)) for a in ARMS}
    seen, folds = np.zeros(n, int), np.full(n, -1, int)
    cells = []
    records = {r['fold']: r for r in manifest['folds']}
    write_json(root/'status.json', dict(state='RUNNING', cells_complete=0))
    for source in previous['cells']:
        fold, half = source['fold'], source['half']
        record = records[fold]
        outer, original_fit = np.array(record['test']), np.array(record['fit'])
        query, fit, cal = (np.array([lookup[v] for v in source[key]])
                          for key in ('query_ids', 'fit_ids', 'calibration_ids'))
        sets = [set(data['groups'][rows]) for rows in (original_fit, fit, cal, query)]
        if any(sets[a] & sets[b] for a in range(4) for b in range(a+1, 4)):
            raise ValueError('Chemical-group leakage')
        if set(np.r_[query, fit, cal]) != set(outer):
            raise ValueError('Prior cell changed')
        positions = {v: j for j, v in enumerate(outer)}
        fi, qi = (np.array([positions[v] for v in rows]) for rows in (fit, query))
        folder = reference/'folds'/f'fold_{fold}'
        stats = read_json(folder/'preprocessing.json')
        with np.load(folder/'arms/STATE50/evaluation/u_predictions.npz') as z:
            if not np.array_equal(z['ids'], data['ids'][outer]):
                raise ValueError('Frozen model identity changed')
            mean, target, covariance = (z[k].copy() for k in ('mean_u', 'actual_u', 'covariance_u'))
        np.testing.assert_allclose(target, (raw[outer]-stats['u_center'])/stats['u_scale'])
        np.testing.assert_allclose(covariance, np.broadcast_to(covariance[0], covariance.shape), atol=0, rtol=0)
        bandwidth = source['norm_bandwidth']
        wf, _ = reference_weights(data, fit, fit, bandwidth)
        wq, _ = reference_weights(data, query, fit, bandwidth)
        fitted = fit_covariance_family(target[fi]-mean[fi], covariance[0], wf, wq, 'LOCAL_SCALE')
        if fitted['choice'] != source['choice']:
            raise ValueError('Prior covariance fitting not reproduced')
        cf, cq = fitted['loo_covariance'], fitted['query_covariance']
        raw_f = mean[fi]*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
        raw_q = mean[qi]*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
        df = contrast_projector(raw_f, stats['u_scale'], cf)
        dq = contrast_projector(raw_q, stats['u_scale'], cq)
        energies = projected_energy(target[fi]-mean[fi], df)
        fitters = dict(
            SPLIT_CONSTANT=[fit_scale(energies[:, j], degree, lognorm[fit], conditional=False)
                            for j, degree in enumerate((3, 6))],
            AMPLITUDE_TOTAL=[fit_scale(energies.sum(1), 9, lognorm[fit], conditional=True)],
            AMPLITUDE_SPLIT=[fit_scale(energies[:, j], degree, lognorm[fit], conditional=True, penalty=degree/9)
                            for j, degree in enumerate((3, 6))])
        multipliers = dict(LOCAL_FIT=(np.ones(len(query)), np.ones(len(query))))
        for arm in ARMS[1:]:
            predictions = [predict_scale(model, lognorm[query]) for model in fitters[arm]]
            multipliers[arm] = (predictions[0], predictions[-1])
        np.savez_compressed(root/f'cell_{fold}_{half}_components.npz',
            fit_ids=data['ids'][fit], query_ids=data['ids'][query],
            fit_energy=energies, fit_log_amplitude=lognorm[fit], query_log_amplitude=lognorm[query],
            fit_projector=df['projector'], query_projector=dq['projector'],
            fit_singular=df['singular'], query_singular=dq['singular'],
            fit_residual=target[fi]-mean[fi], fit_covariance=cf, query_base_covariance=cq,
            query_contrast_energy=projected_energy(target[qi]-mean[qi], dq))
        cell = dict(fold=fold, half=half, fit_ids=data['ids'][fit].tolist(),
            calibration_ids=data['ids'][cal].tolist(), query_ids=data['ids'][query].tolist(),
            budget=source['budget'], fitters=fitters, local_choice=fitted['choice'],
            projector_rank=3, remainder_rank=6, amplitude_feature='log norm of observed X',
            calibration_used_for_fitting=False, scale_penalty=dict(total=1., contrast=3/9, remainder=6/9))
        cells.append(cell)
        write_json(root/f'cell_{fold}_{half}.json', cell)
        for arm in ARMS:
            a, b = multipliers[arm]
            cov = rescale_components(cq, dq, a, b)
            scores = score_distribution(mean[qi], cov, target[qi], stats, actual[query],
                obs_actual[query], absolute_actual[query], norm2_per_feature[query],
                seed=SEED+100*fold+half)
            out = stores[arm]
            out['mean_u'][query], out['actual_u'][query], out['covariance_u'][query] = mean[qi], target[qi], cov
            out['contrast_scale'][query], out['remainder_scale'][query] = a, b
            for k, value in scores.items():
                if k not in out:
                    out[k] = np.empty((n, *value.shape[1:]))
                out[k][query] = value
            order = np.lexsort((data['ids'][query], -scores['predicted']))
            out['selected'][query[order[:source['budget']]]] = 1
            print(f'fold={fold} half={half} arm={arm} elapsed={time.monotonic()-started:.1f}', flush=True)
        seen[query] += 1
        folds[query] = fold
        write_json(root/'status.json', dict(state='RUNNING', cells_complete=len(cells), elapsed_seconds=time.monotonic()-started))
    if not np.all(seen == 1):
        raise ValueError('Every query must be evaluated exactly once')
    with np.load(Path(previous_run)/'LOCAL_FIT.npz') as saved:
        for key in ('mean_u', 'covariance_u', 'nll', 'predicted', 'p_null', 'crps', 'selected'):
            np.testing.assert_array_equal(saved[key], stores['LOCAL_FIT'][key])
    for arm in ARMS:
        np.testing.assert_array_equal(stores[arm]['mean_u'], stores['LOCAL_FIT']['mean_u'])
    metrics, comparisons, by_fold = summarize(stores, actual, data, layout, folds)
    for arm, out in stores.items():
        np.savez_compressed(root/(arm+'.npz'), ids=data['ids'], actual=actual,
            groups=data['groups'], layout=layout, fold=folds, **out)
    summary = dict(n=n, arms=list(ARMS), samples=SAMPLES, metrics=metrics,
        comparisons=comparisons, by_fold=by_fold, cells=cells,
        reference_run=str(reference), previous_run=str(Path(previous_run).resolve()),
        data_directory=manifest['data_directory'], means_changed=False, final_opened=False,
        endpoint_changed=False, biological_relations_used=False, jepa_used=False,
        physical_shared_independent_components_identified=False,
        family='Gaussian geometry predictive error with observable-sensitive conditional scales',
        ci_scope='fixed predictions and prior split; not independent confirmation',
        reference_cost_included=False, elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json', summary)
    write_json(root/'status.json', dict(state='COMPLETE', cells_complete=len(cells), elapsed_seconds=time.monotonic()-started))
    print('COMPLETE', summary['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-run', type=Path, default=PROJECT/'runs/lincs_joint_tail_calibration_20260916_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.previous_run, args.output)
