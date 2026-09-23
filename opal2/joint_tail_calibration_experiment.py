"""Separate uncertainty-region calibration from changing a predictive law.

Complete frozen STATE50 means, opened LINCS objects, old query cells. The
calibration outcomes are disjoint from covariance fitting and query outcomes.
"""
from __future__ import annotations

import argparse
import json
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
from .conditional_joint_error_experiment import (
    PROJECT, SAMPLES, SEED, OBSERVABLES, read_json, observable_forward,
    reference_weights, score_distribution)
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .joint_tail_calibration import (
    split_donor_groups, fit_tail_calibration, scale_covariance)
from .lincs_biology_experiment import load_data
from .reference_information_diagnostic import bootstrap_difference

ARMS = ('LOCAL_FIT', 'REGION_ONLY', 'GAUSSIAN_Q95', 'GAUSSIAN_MLE')
CAL_SEED = 20260917
COMPARE = ('nll', 'energy', 'coordinate_coverage', 'joint_coverage',
    'single_crps', 'pair_crps', 'average_crps', 'triple_average_crps',
    'absolute_pair_crps', 'crps', 'brier', 'policy_value', 'coverage')


def region_diagnostics(covariance, mahal, threshold):
    """Geometry-region diagnostics, distinct from Gaussian probability scores."""
    dimension = covariance.shape[-1]
    nominal = chi2.ppf(.95, dimension)
    threshold = np.broadcast_to(np.asarray(threshold, float), (len(mahal),))
    if np.any(threshold <= 0) or not np.isfinite(threshold).all():
        raise ValueError('This full-law experiment requires positive finite radii')
    _, logdet = np.linalg.slogdet(covariance)
    return dict(joint_coverage=(mahal <= threshold).astype(float),
        joint_squared_radius=threshold.copy(),
        region_log_volume_without_unit_ball=.5*logdet+.5*dimension*np.log(threshold),
        region_radius_factor=np.sqrt(threshold/nominal),
        region_log_volume_ratio=.5*dimension*np.log(threshold/nominal))


def summarize(stores, actual, data, layout, folds, lognorm):
    metrics = {}
    for arm, out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = out['selected']*actual
        out['mse'] = np.square(out['actual_u']-out['mean_u']).mean(1)
        chosen = out['selected'].astype(bool)
        if chosen.sum() != 146:
            raise ValueError('Existing acquisition budget changed')
        row = {k: float(v.mean()) for k, v in out.items()
               if v.ndim == 1 and k != 'selected'}
        row.update(selected_n=int(chosen.sum()),
            selected_null=int((actual[chosen] <= 0).sum()),
            selected_mean=float(actual[chosen].mean()),
            selected_symmetric_difference=int(np.count_nonzero(
                out['selected'] != stores['LOCAL_FIT']['selected'])),
            null_auc=float(roc_auc_score(actual <= 0, out['p_null'])),
            spearman=float(spearmanr(actual, out['predicted']).statistic),
            mahalanobis_quantiles={str(q): float(np.quantile(out['mahalanobis2'], q))
                for q in (.1, .5, .9, .95, .99)},
            observable_crps=dict(zip(OBSERVABLES, out['observable_crps'].mean(0).tolist())),
            observable_coverage=dict(zip(OBSERVABLES, out['observable_coverage'].mean(0).tolist())),
            observable_width=dict(zip(OBSERVABLES, out['observable_width'].mean(0).tolist())))
        metrics[arm] = row
    pairs = [(a, 'LOCAL_FIT') for a in ARMS[1:]] + [('GAUSSIAN_Q95', 'GAUSSIAN_MLE')]
    differences = {a+' minus '+b: {key: {scope: bootstrap_difference(
        stores[a][key], stores[b][key], group) for scope, group in (
            ('chemistry', data['groups']), ('layout', layout))}
        for key in COMPARE} for a, b in pairs}
    descriptive = {}
    amp = np.digitize(lognorm, np.quantile(lognorm, [.25, .5, .75]))
    for name, labels in (('fold', folds), ('X_amplitude_quartile', amp)):
        descriptive[name] = {str(g): dict(n=int((labels == g).sum()),
            metrics={a: {k: float(stores[a][k][labels == g].mean())
                for k in COMPARE} for a in ARMS}) for g in np.unique(labels)}
    return metrics, differences, descriptive


def run(previous_run, output):
    started = time.monotonic()
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/JOINT_TAIL_CALIBRATION_PLAN_20260916.md', root/'PROTOCOL.md')
    for name in ('joint_tail_calibration.py', 'joint_tail_calibration_experiment.py',
                 'conditional_joint_error.py', 'conditional_joint_error_experiment.py'):
        shutil.copy2(PROJECT/'opal2'/name, root/name)
    previous = read_json(Path(previous_run)/'summary.json')
    reference = Path(previous['reference_run'])
    manifest = read_json(reference/'run_manifest.json')
    data, metadata = load_data(manifest['data_directory'])
    n = len(data['ids'])
    if n != 1188 or data['ids'].tolist() != manifest['ids']:
        raise ValueError('Opened LINCS scope changed')
    index = {v: i for i, v in enumerate(data['ids'])}
    layout = np.array([v['layout_block'] for v in metadata['units']])
    lognorm = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    norm2_per_feature = np.square(data['Y'][:, 0]).mean(1)
    gram = profiles_to_gram(torch.tensor(data['Y']))
    raw = gram_to_coordinates(gram).numpy()
    actual, observable_actual, difference, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, gram_gains(gram).numpy()[:, 2], atol=1e-12, rtol=1e-12)
    absolute_actual = np.log1p(difference*norm2_per_feature[:, None])
    measured_absolute = np.stack([np.log1p(np.square(data['Y'][:, a]-data['Y'][:, b]).mean(1))
                                 for a, b in ((1, 2), (1, 3), (2, 3))], axis=1)
    np.testing.assert_allclose(absolute_actual, measured_absolute, atol=1e-11, rtol=1e-11)
    stores = {a: dict(mean_u=np.empty((n, 9)), actual_u=np.empty((n, 9)),
        covariance_u=np.empty((n, 9, 9)), selected=np.zeros(n, dtype=int)) for a in ARMS}
    seen = np.zeros(n, dtype=int)
    folds = np.full(n, -1, dtype=int)
    cells = []
    records = {r['fold']: r for r in manifest['folds']}
    write_json(root/'status.json', dict(state='RUNNING', cells_complete=0))
    for source in previous['cells']:
        fold, half = source['fold'], source['half']
        record = records[fold]
        outer, original_fit = np.array(record['test']), np.array(record['fit'])
        query = np.array([index[v] for v in source['query_ids']])
        donor = np.array([index[v] for v in source['donor_ids']])
        if set(query) & set(donor) or set(query) | set(donor) != set(outer):
            raise ValueError('Existing query/reference cell changed')
        split = split_donor_groups(data['groups'][donor], seed=CAL_SEED+100*fold+half)
        fit = donor[split['fit_indices']]
        cal = donor[split['cal_indices']]
        sets = [set(data['groups'][x]) for x in (original_fit, fit, cal, query)]
        if any(sets[a] & sets[b] for a in range(4) for b in range(a+1, 4)):
            raise ValueError('Chemical group leakage across reference/calibration/query')
        positions = {v: j for j, v in enumerate(outer)}
        fi, ci, qi = (np.array([positions[v] for v in rows]) for rows in (fit, cal, query))
        folder = reference/'folds'/f'fold_{fold}'
        stats = read_json(folder/'preprocessing.json')
        with np.load(folder/'arms/STATE50/evaluation/u_predictions.npz') as z:
            if not np.array_equal(z['ids'], data['ids'][outer]):
                raise ValueError('Frozen model object order changed')
            mean, target, base_cov = (z[k].copy() for k in ('mean_u', 'actual_u', 'covariance_u'))
        np.testing.assert_allclose(target, (raw[outer]-stats['u_center'])/stats['u_scale'])
        np.testing.assert_allclose(base_cov, np.broadcast_to(base_cov[0], base_cov.shape), atol=0, rtol=0)
        bandwidth = max(float(np.std(lognorm[original_fit])), .1)
        wf, _ = reference_weights(data, fit, fit, bandwidth)
        wx, _ = reference_weights(data, np.r_[cal, query], fit, bandwidth)
        fitted = fit_covariance_family(target[fi]-mean[fi], base_cov[0], wf, wx, 'LOCAL_SCALE')
        cal_cov, query_cov = np.split(fitted['query_covariance'], [len(cal)])
        calibration = fit_tail_calibration(target[ci]-mean[ci], cal_cov,
                                           data['ids'][cal], data['groups'][cal])
        covariances = dict(LOCAL_FIT=query_cov, REGION_ONLY=query_cov,
            GAUSSIAN_Q95=scale_covariance(query_cov, calibration['full_law_scale']),
            GAUSSIAN_MLE=scale_covariance(query_cov, calibration['gaussian_mle_scale']))
        cell = dict(fold=fold, half=half, fit_ids=data['ids'][fit].tolist(),
            calibration_ids=data['ids'][cal].tolist(), query_ids=data['ids'][query].tolist(),
            donor_ids=data['ids'][donor].tolist(), budget=source['budget'],
            norm_bandwidth=bandwidth, choice=fitted['choice'], calibration=calibration,
            effective_reference_n_median=float(np.median(1/np.square(wx[len(cal):]).sum(1))))
        cells.append(cell)
        write_json(root/f'cell_{fold}_{half}.json', cell)
        cell_scores = {}
        for arm in ARMS:
            covariance = covariances[arm]
            if arm == 'REGION_ONLY':
                scores = {k: v.copy() for k, v in cell_scores['LOCAL_FIT'].items()}
            else:
                scores = score_distribution(mean[qi], covariance, target[qi], stats,
                    actual[query], observable_actual[query], absolute_actual[query],
                    norm2_per_feature[query], seed=SEED+100*fold+half)
            threshold = calibration['q'] if arm == 'REGION_ONLY' else chi2.ppf(.95, 9)
            scores.update(region_diagnostics(covariance, scores['mahalanobis2'], threshold))
            scale = (calibration['full_law_scale'] if arm in ('REGION_ONLY', 'GAUSSIAN_Q95')
                     else calibration['gaussian_mle_scale'] if arm == 'GAUSSIAN_MLE' else 1.)
            scores['radius_factor_vs_local_fit'] = np.full(len(query), np.sqrt(scale))
            scores['log_volume_ratio_vs_local_fit'] = np.full(len(query), 4.5*np.log(scale))
            cell_scores[arm] = scores
            out = stores[arm]
            out['mean_u'][query], out['actual_u'][query] = mean[qi], target[qi]
            out['covariance_u'][query] = covariance
            for key, value in scores.items():
                if key not in out:
                    out[key] = np.empty((n, *value.shape[1:]))
                out[key][query] = value
            order = np.lexsort((data['ids'][query], -scores['predicted']))
            out['selected'][query[order[:cell['budget']]]] = 1
            print(json.dumps(dict(fold=fold, half=half, arm=arm,
                elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
        np.testing.assert_array_equal(cell_scores['REGION_ONLY']['joint_coverage'],
                                      cell_scores['GAUSSIAN_Q95']['joint_coverage'])
        np.testing.assert_allclose(cell_scores['REGION_ONLY']['region_log_volume_without_unit_ball'],
            cell_scores['GAUSSIAN_Q95']['region_log_volume_without_unit_ball'], atol=1e-12, rtol=1e-12)
        seen[query] += 1
        folds[query] = fold
        write_json(root/'status.json', dict(state='RUNNING', cells_complete=len(cells),
            elapsed_seconds=time.monotonic()-started))
    if not np.all(seen == 1):
        raise ValueError('Every opened object must be evaluated exactly once')
    for arm in ARMS:
        np.testing.assert_array_equal(stores[arm]['mean_u'], stores['LOCAL_FIT']['mean_u'])
    for key in ('predicted', 'crps', 'p_null', 'selected', 'nll', 'energy', 'coverage'):
        np.testing.assert_array_equal(stores['REGION_ONLY'][key], stores['LOCAL_FIT'][key])
    metrics, comparisons, descriptive = summarize(stores, actual, data, layout, folds, lognorm)
    for arm, values in stores.items():
        np.savez_compressed(root/(arm+'.npz'), ids=data['ids'], groups=data['groups'], layout=layout,
            fold=folds, actual=actual, **values)
    result = dict(n=n, arms=list(ARMS), samples=SAMPLES, cells=cells,
        metrics=metrics, comparisons=comparisons, descriptive=descriptive,
        historical_context={k: previous['metrics'][k] for k in ('BASE', 'LOCAL_SCALE')},
        previous_run=str(Path(previous_run).resolve()), reference_run=str(reference),
        means_changed=False, endpoint_changed=False, final_opened=False,
        jepa_used=False, biological_relations_used=False, reference_cost_included=False,
        region_only_changes_density=False, claim_scope='retrospective opened DEV empirical calibration',
        ci_scope='fixed predictions; chemistry/layout cluster resampling; no refit or repeated-development correction',
        interval_guarantee_claimed=False, elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json', result)
    write_json(root/'status.json', dict(state='COMPLETE', cells_complete=len(cells),
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(state='COMPLETE', elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-run', type=Path,
        default=PROJECT/'runs/lincs_conditional_joint_error_20260916_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.previous_run, args.output)
