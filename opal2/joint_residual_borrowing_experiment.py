"""Mean/covariance-matched non-Gaussian reference-block distributions."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2, norm, spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_joint_error import fit_covariance_family
from .conditional_joint_error_experiment import (
    PROJECT, SEED, SAMPLES, OBSERVABLES, SCALAR_SCORES, read_json,
    observable_forward, reference_weights, score_distribution, interval_scores)
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .joint_residual_distribution import (
    build_residual_mixture, mixture_nll, sample_mixture, select_mixture_alpha)
from .lincs_biology_experiment import load_data
from .objective_analysis import fair_crps
from .reference_information_diagnostic import bootstrap_difference, gamma_forward

ARMS = ('GAUSSIAN', 'GLOBAL_BLOCK', 'LOCAL_MATCHED', 'LOCAL_BLOCK')
MIXTURE_KEYS = ('centers', 'component_covariance', 'weights', 'covariance')
COMPARE = ('nll', 'energy', 'joint_coverage', 'coordinate_coverage', 'single_crps',
    'pair_crps', 'average_crps', 'triple_average_crps', 'absolute_pair_crps',
    'crps', 'brier', 'policy_value', 'coverage')
BANDWIDTH = .5


def subset_mixture(mixture, indices):
    return {key: np.asarray(mixture[key])[indices] for key in MIXTURE_KEYS}


def score_mixture(mean, target, mixture, alpha, stats, actual, obs_actual,
                  absolute_actual, norm2_per_feature, seed, samples=SAMPLES):
    covariance = mixture['covariance']
    n = len(mean)
    if alpha == 0:
        out = score_distribution(mean, covariance, target, stats, actual, obs_actual,
            absolute_actual, norm2_per_feature, seed, samples)
        sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        out['coordinate_lower'] = mean-norm.ppf(.975)*sd
        out['coordinate_upper'] = mean+norm.ppf(.975)*sd
        out['joint_squared_radius'] = np.full(n, chi2.ppf(.95, 9))
        return out
    if samples < 4 or samples % 2:
        raise ValueError('Independent energy pairs require an even sample count')
    out = {key: np.empty(n) for key in SCALAR_SCORES}
    for name in ('observable_crps', 'observable_coverage', 'observable_width'):
        out[name] = np.empty((n, len(OBSERVABLES)))
    for name in ('absolute_crps_by_pair', 'absolute_coverage_by_pair'):
        out[name] = np.empty((n, 3))
    out['coordinate_lower'] = np.empty((n, 9))
    out['coordinate_upper'] = np.empty((n, 9))
    out['joint_squared_radius'] = np.empty(n)
    residual = target-mean
    out['nll'] = mixture_nll(residual, mixture, alpha)
    out['mahalanobis2'] = np.einsum('ni,ni->n', residual,
        np.linalg.solve(covariance, residual[..., None])[..., 0])
    normal_rng = np.random.default_rng(seed)
    component_rng = np.random.default_rng(seed+18000)
    mixture_rng = np.random.default_rng(seed+27000)
    for start in range(0, n, 16):
        end = min(start+16, n)
        block = subset_mixture(mixture, slice(start, end))
        z = normal_rng.normal(size=(samples, end-start, 9))
        residual_draws = sample_mixture(block, alpha, z,
            component_rng.random((samples, end-start)), mixture_rng.random((samples, end-start)))
        u = mean[None, start:end]+residual_draws
        lo, hi = np.quantile(u, [.025, .975], axis=0)
        out['coordinate_lower'][start:end] = lo
        out['coordinate_upper'][start:end] = hi
        out['coordinate_coverage'][start:end] = ((target[start:end] >= lo) &
                                                (target[start:end] <= hi)).mean(1)
        inverse_chol = np.linalg.inv(np.linalg.cholesky(covariance[start:end]))
        whitened = np.einsum('nij,snj->sni', inverse_chol, residual_draws)
        radius = np.quantile(np.square(whitened).sum(-1), .95, axis=0)
        out['joint_squared_radius'][start:end] = radius
        out['joint_coverage'][start:end] = out['mahalanobis2'][start:end] <= radius
        out['energy'][start:end] = np.linalg.norm(u-target[None, start:end], axis=-1).mean(0) \
            - .5*np.linalg.norm(u[:samples//2]-u[samples//2:], axis=-1).mean(0)
        raw = u*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
        gamma, obs, differences, cosine = observable_forward(raw)
        np.testing.assert_allclose(gamma, gamma_forward(raw), atol=1e-12, rtol=1e-12)
        crps, covered, width = interval_scores(obs, obs_actual[start:end])
        out['observable_crps'][start:end] = crps
        out['observable_coverage'][start:end] = covered
        out['observable_width'][start:end] = width
        for prefix, columns in (('single', slice(0, 3)), ('pair', slice(3, 6)),
                               ('average', slice(6, 10)), ('triple_average', slice(9, 10))):
            out[prefix+'_crps'][start:end] = crps[:, columns].mean(1)
            out[prefix+'_coverage'][start:end] = covered[:, columns].mean(1)
        absolute = np.log1p(differences*norm2_per_feature[None, start:end, None])
        ac, av, _ = interval_scores(absolute, absolute_actual[start:end])
        out['absolute_crps_by_pair'][start:end] = ac
        out['absolute_coverage_by_pair'][start:end] = av
        out['absolute_pair_crps'][start:end] = ac.mean(1)
        out['absolute_pair_coverage'][start:end] = av.mean(1)
        glo, ghi = np.quantile(gamma, [.025, .975], axis=0)
        p = (gamma <= 0).mean(0)
        out['predicted'][start:end] = gamma.mean(0)
        out['p_null'][start:end] = p
        out['crps'][start:end] = fair_crps(gamma, actual[start:end])
        out['coverage'][start:end] = (actual[start:end] >= glo) & (actual[start:end] <= ghi)
        out['gamma_lower'][start:end], out['gamma_upper'][start:end] = glo, ghi
        out['gamma_mc_se'][start:end] = gamma.std(0, ddof=1)/np.sqrt(samples)
        out['null_mc_se'][start:end] = np.sqrt(p*(1-p)/samples)
        out['predicted_cos_Z1_V'][start:end] = cosine[..., 0].mean(0)
        out['predicted_cos_Z2_V'][start:end] = cosine[..., 1].mean(0)
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError('Nonfinite mixture scores; no sampled value is removed')
    return out


def summarize(stores, actual, data, layout, folds):
    metrics = {}
    for arm, out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = out['selected']*actual
        out['mse'] = np.square(out['actual_u']-out['mean_u']).mean(1)
        out['coordinate_width'] = (out['coordinate_upper']-out['coordinate_lower']).mean(1)
        out['joint_radius_factor_vs_gaussian'] = np.sqrt(out['joint_squared_radius']/chi2.ppf(.95, 9))
        chosen = out['selected'].astype(bool)
        if chosen.sum() != 146:
            raise ValueError('Query acquisition budget changed')
        row = {key: float(value.mean()) for key, value in out.items()
               if value.ndim == 1 and key != 'selected'}
        row.update(selected_n=int(chosen.sum()), selected_null=int((actual[chosen] <= 0).sum()),
            selected_mean=float(actual[chosen].mean()),
            selected_symmetric_difference=int(np.count_nonzero(
                out['selected'] != stores['GAUSSIAN']['selected'])),
            spearman=float(spearmanr(actual, out['predicted']).statistic),
            null_auc=float(roc_auc_score(actual <= 0, out['p_null'])),
            observable_crps=dict(zip(OBSERVABLES, out['observable_crps'].mean(0).tolist())),
            observable_coverage=dict(zip(OBSERVABLES, out['observable_coverage'].mean(0).tolist())),
            observable_width=dict(zip(OBSERVABLES, out['observable_width'].mean(0).tolist())))
        metrics[arm] = row
    pairs = [('GLOBAL_BLOCK', 'GAUSSIAN'), ('LOCAL_MATCHED', 'GLOBAL_BLOCK'),
             ('LOCAL_BLOCK', 'GLOBAL_BLOCK'), ('LOCAL_BLOCK', 'GAUSSIAN'),
             ('LOCAL_BLOCK', 'LOCAL_MATCHED')]
    comparisons = {a+' minus '+b: {key: {scope: bootstrap_difference(stores[a][key],
        stores[b][key], labels) for scope, labels in (('chemistry', data['groups']), ('layout', layout))}
        for key in COMPARE} for a, b in pairs}
    by_fold = {str(f): {a: {key: float(stores[a][key][folds == f].mean())
        for key in COMPARE} for a in ARMS} for f in np.unique(folds)}
    return metrics, comparisons, by_fold


def run(previous_run, output):
    started = time.monotonic()
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/JOINT_RESIDUAL_BORROWING_PLAN_20260916.md', root/'PROTOCOL.md')
    for name in ('joint_residual_distribution.py', 'joint_residual_borrowing_experiment.py',
                 'conditional_joint_error.py', 'conditional_joint_error_experiment.py'):
        shutil.copy2(PROJECT/'opal2'/name, root/name)
    previous = read_json(Path(previous_run)/'summary.json')
    reference = Path(previous['reference_run'])
    manifest = read_json(reference/'run_manifest.json')
    data, metadata = load_data(manifest['data_directory'])
    n = len(data['ids'])
    if n != 1188 or data['ids'].tolist() != manifest['ids']:
        raise ValueError('Opened dataset scope changed')
    index = {v: i for i, v in enumerate(data['ids'])}
    layout = np.array([v['layout_block'] for v in metadata['units']])
    lognorm = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    norm2_per_feature = np.square(data['Y'][:, 0]).mean(1)
    gram = profiles_to_gram(torch.tensor(data['Y']))
    raw = gram_to_coordinates(gram).numpy()
    actual, obs_actual, difference, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, gram_gains(gram).numpy()[:, 2], atol=1e-12, rtol=1e-12)
    absolute_actual = np.log1p(difference*norm2_per_feature[:, None])
    stores = {a: dict(mean_u=np.empty((n, 9)), actual_u=np.empty((n, 9)),
        covariance_u=np.empty((n, 9, 9)), selected=np.zeros(n, dtype=int)) for a in ARMS}
    seen, folds = np.zeros(n, dtype=int), np.full(n, -1, dtype=int)
    cells = []
    records = {r['fold']: r for r in manifest['folds']}
    write_json(root/'status.json', dict(state='RUNNING', cells_complete=0))
    for source in previous['cells']:
        fold, half = source['fold'], source['half']
        record = records[fold]
        outer, original_fit = np.array(record['test']), np.array(record['fit'])
        query, fit, cal = (np.array([index[v] for v in source[key]])
                           for key in ('query_ids', 'fit_ids', 'calibration_ids'))
        sets = [set(data['groups'][rows]) for rows in (original_fit, fit, cal, query)]
        if any(sets[a] & sets[b] for a in range(4) for b in range(a+1, 4)):
            raise ValueError('Chemical group leakage')
        if set(np.r_[query, fit, cal]) != set(outer):
            raise ValueError('Prior query/reference split changed')
        position = {v: j for j, v in enumerate(outer)}
        fi, ci, qi = (np.array([position[v] for v in rows]) for rows in (fit, cal, query))
        folder = reference/'folds'/f'fold_{fold}'
        stats = read_json(folder/'preprocessing.json')
        with np.load(folder/'arms/STATE50/evaluation/u_predictions.npz') as z:
            if not np.array_equal(z['ids'], data['ids'][outer]):
                raise ValueError('Frozen model identity changed')
            mean, target, covariance = (z[k].copy() for k in ('mean_u', 'actual_u', 'covariance_u'))
        np.testing.assert_allclose(target, (raw[outer]-stats['u_center'])/stats['u_scale'])
        np.testing.assert_allclose(covariance, np.broadcast_to(covariance[0], covariance.shape), atol=0, rtol=0)
        bandwidth = max(float(np.std(lognorm[original_fit])), .1)
        wf, _ = reference_weights(data, fit, fit, bandwidth)
        wcq, gcq = reference_weights(data, np.r_[cal, query], fit, bandwidth)
        fitted = fit_covariance_family(target[fi]-mean[fi], covariance[0], wf, wcq, 'LOCAL_SCALE')
        if fitted['choice'] != source['choice']:
            raise ValueError('Previous LOCAL_FIT procedure not reproduced')
        cal_cov, query_cov = np.split(fitted['query_covariance'], [len(cal)])
        libraries = {}
        moment_checks = {}
        residual = target[fi]-mean[fi]
        for name, weights in (('GLOBAL', gcq), ('LOCAL', .5*gcq+.5*wcq)):
            libraries[name] = build_residual_mixture(residual, fitted['loo_covariance'], weights,
                                                   fitted['query_covariance'], bandwidth=BANDWIDTH)
            lib = libraries[name]
            np.testing.assert_allclose(lib['mean'], 0., atol=1e-11, rtol=0)
            np.testing.assert_allclose(lib['matched_covariance'], fitted['query_covariance'],
                                       atol=1e-11, rtol=1e-11)
            moment_checks[name] = dict(max_absolute_mean=float(np.max(np.abs(lib['mean']))),
                max_absolute_covariance_error=float(np.max(np.abs(
                    lib['matched_covariance']-fitted['query_covariance']))))
        reps = np.array(source['calibration']['representative_indices'])
        if data['ids'][cal][reps].tolist() != source['calibration']['representative_ids']:
            raise ValueError('Calibration representatives changed')
        choices = {name: select_mixture_alpha(target[ci][reps]-mean[ci][reps],
                    subset_mixture(lib, reps)) for name, lib in libraries.items()}
        alphas = dict(GAUSSIAN=0., GLOBAL_BLOCK=choices['GLOBAL']['alpha'],
            LOCAL_MATCHED=choices['GLOBAL']['alpha'], LOCAL_BLOCK=choices['LOCAL']['alpha'])
        # Store full legal reference records and model components for direct replay.
        np.savez_compressed(root/f'cell_{fold}_{half}_reference.npz', fit_ids=data['ids'][fit],
            residual=residual, loo_covariance=fitted['loo_covariance'], cal_ids=data['ids'][cal],
            cal_residual=target[ci]-mean[ci], cal_covariance=cal_cov, local_weights=wcq,
            uniform_weights=gcq, query_ids=data['ids'][query], query_covariance=query_cov)
        for name, lib in libraries.items():
            np.savez_compressed(root/f'cell_{fold}_{half}_{name}_mixture.npz',
                **{k: lib[k] for k in MIXTURE_KEYS})
        cell = dict(fold=fold, half=half, query_ids=data['ids'][query].tolist(),
            fit_ids=data['ids'][fit].tolist(), calibration_ids=data['ids'][cal].tolist(),
            calibration_representative_ids=data['ids'][cal][reps].tolist(),
            budget=source['budget'], covariance_choice=fitted['choice'], choices=choices,
            moment_checks=moment_checks,
            alphas=alphas, smoothing_bandwidth=BANDWIDTH, local_weight_fraction=.5,
            local_effective_n_median=float(np.median(1/np.square((.5*gcq+.5*wcq)[len(cal):]).sum(1))))
        cells.append(cell)
        write_json(root/f'cell_{fold}_{half}.json', cell)
        for arm in ARMS:
            name = 'LOCAL' if arm.startswith('LOCAL') else 'GLOBAL'
            mixture = subset_mixture(libraries[name], slice(len(cal), None))
            np.testing.assert_array_equal(mixture['covariance'], query_cov)
            scores = score_mixture(mean[qi], target[qi], mixture, alphas[arm], stats,
                actual[query], obs_actual[query], absolute_actual[query], norm2_per_feature[query],
                seed=SEED+100*fold+half)
            out = stores[arm]
            out['mean_u'][query], out['actual_u'][query] = mean[qi], target[qi]
            out['covariance_u'][query] = query_cov
            for key, value in scores.items():
                if key not in out:
                    out[key] = np.empty((n, *value.shape[1:]))
                out[key][query] = value
            order = np.lexsort((data['ids'][query], -scores['predicted']))
            out['selected'][query[order[:cell['budget']]]] = 1
            print(json.dumps(dict(fold=fold, half=half, arm=arm, alpha=alphas[arm],
                elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
        seen[query] += 1
        folds[query] = fold
        write_json(root/'status.json', dict(state='RUNNING', cells_complete=len(cells),
            elapsed_seconds=time.monotonic()-started))
    if not np.all(seen == 1):
        raise ValueError('Every opened query must be evaluated once')
    for arm in ARMS:
        for key in ('mean_u', 'covariance_u'):
            np.testing.assert_array_equal(stores[arm][key], stores['GAUSSIAN'][key])
    with np.load(Path(previous_run)/'LOCAL_FIT.npz') as prior:
        for key in ('mean_u', 'covariance_u', 'nll', 'predicted', 'p_null', 'crps', 'selected'):
            np.testing.assert_array_equal(stores['GAUSSIAN'][key], prior[key])
    metrics, comparisons, by_fold = summarize(stores, actual, data, layout, folds)
    for arm, values in stores.items():
        np.savez_compressed(root/(arm+'.npz'), ids=data['ids'], groups=data['groups'], layout=layout,
            fold=folds, actual=actual, **values)
    result = dict(n=n, samples=SAMPLES, arms=list(ARMS), cells=cells, metrics=metrics,
        comparisons=comparisons, by_fold=by_fold, previous_run=str(Path(previous_run).resolve()),
        reference_run=str(reference), means_changed=False, covariances_changed=False,
        endpoint_changed=False, final_opened=False, biological_relations_used=False, jepa_used=False,
        reference_cost_included=False, smoothing_bandwidth=BANDWIDTH,
        law='Gaussian plus moment-matched smoothed complete residual-block mixture',
        scope='opened DEV retrospective shape comparison; shared-layout dependencies remain',
        ci_scope='fixed predictions; chemistry/layout bootstrap; no refitting or repeated-development correction',
        elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json', result)
    write_json(root/'status.json', dict(state='COMPLETE', cells_complete=len(cells),
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(state='COMPLETE', elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-run', type=Path,
        default=PROJECT/'runs/lincs_joint_tail_calibration_20260916_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.previous_run, args.output)
