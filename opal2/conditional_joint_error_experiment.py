"""Frozen-mean, matched-marginal conditional error comparisons on opened LINCS.

Reference outcomes calibrate predictive errors, not causal noise components.
Every query is excluded from its own reference group. No base model is trained.
"""
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
from .conditional_joint_error import fit_covariance_family, gaussian_score
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .lincs_biology_experiment import load_data
from .objective_analysis import fair_crps
from .reference_information_diagnostic import bootstrap_difference, gamma_forward
from .reference_information_memory import (morphology_similarity, chemistry_similarity,
    normalized_topk_weights)

PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('BASE', 'GLOBAL_SCALE', 'LOCAL_SCALE', 'LOCAL_DIAG',
        'LOCAL_GLOBAL_CORR', 'LOCAL_JOINT')
SAMPLES = 10000
SEED = 20260916
OBSERVABLES = ('single_Z1', 'single_Z2', 'single_V', 'difference_Z1_Z2',
    'difference_Z1_V', 'difference_Z2_V', 'average_Z1_Z2', 'average_Z1_V',
    'average_Z2_V', 'average_Z1_Z2_V')
SCALAR_SCORES = ('nll', 'energy', 'coordinate_coverage', 'joint_coverage',
    'predicted', 'p_null', 'crps', 'coverage', 'single_crps', 'pair_crps',
    'average_crps', 'triple_average_crps', 'single_coverage', 'pair_coverage',
    'average_coverage', 'triple_average_coverage', 'absolute_pair_crps',
    'absolute_pair_coverage', 'gamma_mc_se', 'null_mc_se', 'gamma_lower',
    'gamma_upper', 'predicted_cos_Z1_V', 'predicted_cos_Z2_V')


def read_json(path):
    return json.loads(Path(path).read_text())


def observable_forward(raw):
    """Fixed-X factor-forward functionals, preserving all joint dependencies."""
    u = np.asarray(raw, dtype=np.float64)
    if u.shape[-1] != 9 or not np.isfinite(u).all():
        raise ValueError('Expected finite nine-dimensional geometry')
    rows = np.zeros((*u.shape[:-1], 4, 4))
    rows[..., 0, 0] = 1.
    rows[..., 1:, 0] = u[..., :3]
    diag = np.exp(u[..., (3, 5, 8)])
    if not np.isfinite(diag).all() or np.any(diag <= 0) or np.any(diag**2 == 0):
        raise ValueError('Invalid factor diagonal; no clipping or resampling')
    rows[..., 1, 1] = diag[..., 0]
    rows[..., 2, 1] = u[..., 4]
    rows[..., 2, 2] = diag[..., 1]
    rows[..., 3, 1] = u[..., 6]
    rows[..., 3, 2] = u[..., 7]
    rows[..., 3, 3] = diag[..., 2]
    single = np.square(rows[..., 1:, :]).sum(-1)
    pairs = ((1, 2), (1, 3), (2, 3))
    differences = np.stack([np.square(rows[..., a, :] - rows[..., b, :]).sum(-1)
                            for a, b in pairs], -1)
    means = np.stack([np.square((rows[..., a, :] + rows[..., b, :])/2).sum(-1)
                      for a, b in pairs] +
                     [np.square(rows[..., 1:, :].mean(-2)).sum(-1)], -1)
    av = rows[..., :3, :].mean(-2)
    av2 = np.square(av).sum(-1)
    if np.any(single <= 0) or np.any(av2 <= 0):
        raise ValueError('Zero acquired or validation norm')
    v = rows[..., 3, :]
    gamma = .5 * ((av*v).sum(-1)/np.sqrt(av2*single[..., 2])
                  - v[..., 0]/np.sqrt(single[..., 2])) - .02
    cosine = np.stack([(rows[..., a, :]*v).sum(-1)/
                       np.sqrt(single[..., a-1]*single[..., 2]) for a in (1, 2)], -1)
    quantities = np.concatenate((single, differences, means), -1)
    if not np.isfinite(quantities).all() or not np.isfinite(gamma).all():
        raise ValueError('Geometry overflow; no sampled value is dropped')
    return gamma, np.log1p(quantities), differences, cosine


def reference_weights(data, query, donor, bandwidth):
    allowed = data['groups'][query, None] != data['groups'][None, donor]
    direction = morphology_similarity(data['Y'][query, 0], data['Y'][donor, 0])
    chemical = chemistry_similarity(data['chem'][query, :512], data['chem'][donor, :512])
    qnorm = np.log(np.linalg.norm(data['Y'][query, 0], axis=1))
    dnorm = np.log(np.linalg.norm(data['Y'][donor, 0], axis=1))
    amplitude = np.exp(-.5*((qnorm[:, None]-dnorm[None])/bandwidth)**2)
    similarity = ((direction+chemical)/2 + amplitude)/2
    local = normalized_topk_weights(similarity, donor_ids=data['ids'][donor],
                                    top_k=16, eligible=allowed)['weights']
    uniform = allowed/allowed.sum(1, keepdims=True)
    if np.any(local[~allowed] != 0) or np.any(uniform[~allowed] != 0):
        raise ValueError('A query chemistry group entered its reference weights')
    return local, uniform


def interval_scores(draws, actual):
    lo, hi = np.quantile(draws, [.025, .975], axis=0)
    return (fair_crps(draws, actual), (actual >= lo) & (actual <= hi), hi-lo)


def score_distribution(mean, covariance, target, stats, actual, obs_actual,
                       absolute_actual, norm2_per_feature, seed, samples=SAMPLES):
    if samples < 4 or samples % 2:
        raise ValueError('Energy score requires an even number of independent draws')
    n = len(mean)
    output = {key: np.empty(n) for key in SCALAR_SCORES}
    for name in ('observable_crps', 'observable_coverage', 'observable_width'):
        output[name] = np.empty((n, len(OBSERVABLES)))
    for name in ('absolute_crps_by_pair', 'absolute_coverage_by_pair'):
        output[name] = np.empty((n, 3))
    r = target-mean
    output['nll'] = gaussian_score(r, covariance)
    sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    output['coordinate_coverage'] = (np.abs(r) <= norm.ppf(.975)*sd).mean(1)
    mahal = np.einsum('ni,ni->n', r, np.linalg.solve(covariance, r[..., None])[..., 0])
    output['joint_coverage'] = (mahal <= chi2.ppf(.95, 9)).astype(float)
    output['mahalanobis2'] = mahal
    rng = np.random.default_rng(seed)
    for start in range(0, n, 16):
        end = min(start+16, n)
        z = rng.normal(size=(samples, end-start, 9))
        u = mean[None, start:end] + np.einsum('nij,snj->sni',
            np.linalg.cholesky(covariance[start:end]), z)
        # Disjoint independent pairs: unbiased energy pair term, no O(S^2) array.
        output['energy'][start:end] = np.linalg.norm(u-target[None, start:end], axis=-1).mean(0) \
            - .5*np.linalg.norm(u[:samples//2]-u[samples//2:], axis=-1).mean(0)
        raw = u*np.asarray(stats['u_scale']) + np.asarray(stats['u_center'])
        gamma, obs, differences, cosine = observable_forward(raw)
        # Independent original implementation verifies the unchanged functional.
        if not np.allclose(gamma, gamma_forward(raw), atol=1e-12, rtol=1e-12):
            raise ValueError('Original Gamma identity changed')
        crps, cover, width = interval_scores(obs, obs_actual[start:end])
        output['observable_crps'][start:end] = crps
        output['observable_coverage'][start:end] = cover
        output['observable_width'][start:end] = width
        for prefix, cols in (('single', slice(0, 3)), ('pair', slice(3, 6)),
                             ('average', slice(6, 10)), ('triple_average', slice(9, 10))):
            output[prefix+'_crps'][start:end] = crps[:, cols].mean(1)
            output[prefix+'_coverage'][start:end] = cover[:, cols].mean(1)
        absolute = np.log1p(differences*norm2_per_feature[None, start:end, None])
        ac, av, _ = interval_scores(absolute, absolute_actual[start:end])
        output['absolute_crps_by_pair'][start:end] = ac
        output['absolute_coverage_by_pair'][start:end] = av
        output['absolute_pair_crps'][start:end] = ac.mean(1)
        output['absolute_pair_coverage'][start:end] = av.mean(1)
        lo, hi = np.quantile(gamma, [.025, .975], axis=0)
        p = (gamma <= 0).mean(0)
        output['predicted'][start:end] = gamma.mean(0)
        output['p_null'][start:end] = p
        output['crps'][start:end] = fair_crps(gamma, actual[start:end])
        output['coverage'][start:end] = (actual[start:end] >= lo) & (actual[start:end] <= hi)
        output['gamma_lower'][start:end] = lo
        output['gamma_upper'][start:end] = hi
        output['gamma_mc_se'][start:end] = gamma.std(0, ddof=1)/np.sqrt(samples)
        output['null_mc_se'][start:end] = np.sqrt(p*(1-p)/samples)
        output['predicted_cos_Z1_V'][start:end] = cosine[..., 0].mean(0)
        output['predicted_cos_Z2_V'][start:end] = cosine[..., 1].mean(0)
    if any(not np.isfinite(value).all() for value in output.values()):
        raise ValueError('Nonfinite distribution scores')
    return output


def evaluate_summary(stores, actual, data, layout, fold_index, lognorm):
    metrics = {}
    for arm, out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = out['selected']*actual
        out['gamma_mse'] = (out['predicted']-actual)**2
        out['mse'] = np.square(out['actual_u']-out['mean_u']).mean(1)
        chosen = out['selected'].astype(bool)
        if chosen.sum() != 146:
            raise ValueError('Preallocated query budget changed')
        row = {key: float(value.mean()) for key, value in out.items()
               if value.ndim == 1 and key not in ('selected',)}
        row.update(selected_n=int(chosen.sum()), selected_null=int((actual[chosen] <= 0).sum()),
            selected_mean=float(actual[chosen].mean()), actual_mean=float(actual.mean()),
            actual_null_rate=float((actual <= 0).mean()),
            spearman=float(spearmanr(actual, out['predicted']).statistic),
            null_auc=float(roc_auc_score(actual <= 0, out['p_null'])),
            observable_crps=dict(zip(OBSERVABLES, out['observable_crps'].mean(0).tolist())),
            observable_coverage=dict(zip(OBSERVABLES, out['observable_coverage'].mean(0).tolist())),
            observable_width=dict(zip(OBSERVABLES, out['observable_width'].mean(0).tolist())))
        metrics[arm] = row
    comparison_pairs = [(a, 'BASE') for a in ARMS[1:]] + [
        ('LOCAL_SCALE', 'GLOBAL_SCALE'), ('LOCAL_DIAG', 'LOCAL_SCALE'),
        ('LOCAL_GLOBAL_CORR', 'LOCAL_DIAG'), ('LOCAL_JOINT', 'LOCAL_DIAG'),
        ('LOCAL_JOINT', 'LOCAL_GLOBAL_CORR')]
    comparison_metrics = ('nll', 'energy', 'single_crps', 'pair_crps', 'average_crps',
        'triple_average_crps', 'absolute_pair_crps', 'crps', 'brier', 'policy_value')
    comparisons = {a+' minus '+b: {key: {scope: bootstrap_difference(stores[a][key],
        stores[b][key], labels) for scope, labels in (('chemistry', data['groups']), ('layout', layout))}
        for key in comparison_metrics} for a, b in comparison_pairs}
    descriptive = {}
    # These observed-X strata only describe scores and never alter fitting or policy.
    q = np.quantile(lognorm, [.25, .5, .75])
    for label, groups in (('outer_fold', fold_index), ('first_well_amplitude_quartile', np.digitize(lognorm, q))):
        descriptive[label] = {str(g): {'n': int((groups == g).sum()),
            'metrics': {a: {k: float(stores[a][k][groups == g].mean())
                for k in comparison_metrics} for a in ARMS}}
            for g in np.unique(groups)}
    return metrics, comparisons, descriptive


def run(reference_diagnostic, output):
    started = time.monotonic()
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/CONDITIONAL_JOINT_ERROR_PLAN_20260916.md', root/'PROTOCOL.md')
    for filename in ('conditional_joint_error.py', 'conditional_joint_error_experiment.py'):
        shutil.copy2(PROJECT/'opal2'/filename, root/filename)
    previous = read_json(Path(reference_diagnostic)/'summary.json')
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
    actual, obs_actual, diff, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, gram_gains(gram).numpy()[:, 2], atol=1e-12, rtol=1e-12)
    absolute_actual = np.log1p(diff*norm2_per_feature[:, None])
    measured_absolute = np.stack([np.log1p(np.square(data['Y'][:, a]-data['Y'][:, b]).mean(1))
                                 for a, b in ((1, 2), (1, 3), (2, 3))], axis=1)
    np.testing.assert_allclose(absolute_actual, measured_absolute, atol=1e-11, rtol=1e-11)
    stores = {a: {'mean_u': np.empty((n, 9)), 'actual_u': np.empty((n, 9)),
        'covariance_u': np.empty((n, 9, 9)), 'selected': np.zeros(n, dtype=int)} for a in ARMS}
    seen = np.zeros(n, dtype=int)
    fold_index = np.full(n, -1, dtype=int)
    cells = []
    records = {r['fold']: r for r in manifest['folds']}
    write_json(root/'status.json', {'state': 'RUNNING', 'stage': 'reference fitting', 'cells_complete': 0})
    for source_cell in previous['cells']:
        fold, half = source_cell['fold'], source_cell['half']
        record = records[fold]
        outer, fit = np.array(record['test']), np.array(record['fit'])
        query = np.array([index[v] for v in source_cell['query_ids']])
        donor = np.array([index[v] for v in source_cell['donor_ids']])
        if set(query) & set(donor) or set(query) | set(donor) != set(outer):
            raise ValueError('Query/reference cell changed')
        if set(data['groups'][fit]) & set(data['groups'][outer]):
            raise ValueError('Frozen model training group entered outer test')
        position = {v: i for i, v in enumerate(outer)}
        qi, di = np.array([position[v] for v in query]), np.array([position[v] for v in donor])
        folder = reference/'folds'/f'fold_{fold}'
        stats = read_json(folder/'preprocessing.json')
        with np.load(folder/'arms/STATE50/evaluation/u_predictions.npz') as z:
            if not np.array_equal(z['ids'], data['ids'][outer]):
                raise ValueError('Saved model identity mismatch')
            mean, target, base_cov = z['mean_u'].copy(), z['actual_u'].copy(), z['covariance_u'].copy()
        np.testing.assert_allclose(target, (raw[outer]-stats['u_center'])/stats['u_scale'])
        np.testing.assert_allclose(base_cov, np.broadcast_to(base_cov[0], base_cov.shape), atol=0, rtol=0)
        cov0 = base_cov[0]
        bandwidth = max(float(np.std(lognorm[fit])), .1)
        wd, gd = reference_weights(data, donor, donor, bandwidth)
        wq, gq = reference_weights(data, query, donor, bandwidth)
        residual = target[di]-mean[di]
        covariances = {'BASE': base_cov[qi].copy()}
        choices = {}
        for arm in ARMS[1:]:
            dweights, qweights = (gd, gq) if arm == 'GLOBAL_SCALE' else (wd, wq)
            kwargs = {} if arm != 'LOCAL_GLOBAL_CORR' else dict(
                correlation_loo_weights=gd, correlation_query_weights=gq)
            fitted = fit_covariance_family(residual, cov0, dweights, qweights,
                'LOCAL_JOINT' if arm == 'LOCAL_GLOBAL_CORR' else arm, **kwargs)
            covariances[arm] = fitted['query_covariance']
            choices[arm] = fitted['choice']
        for arm in ('LOCAL_GLOBAL_CORR', 'LOCAL_JOINT'):
            np.testing.assert_allclose(np.diagonal(covariances[arm], axis1=-2, axis2=-1),
                np.diagonal(covariances['LOCAL_DIAG'], axis1=-2, axis2=-1), rtol=1e-13, atol=1e-13)
        cell = dict(fold=fold, half=half, query_ids=data['ids'][query].tolist(),
            donor_ids=data['ids'][donor].tolist(), budget=source_cell['budget'],
            norm_bandwidth=bandwidth, local_neff_median=float(np.median(1/np.square(wq).sum(1))),
            choices=choices)
        cells.append(cell)
        write_json(root/f'cell_{fold}_{half}_choices.json', cell)
        for arm in ARMS:
            out = stores[arm]
            out['mean_u'][query] = mean[qi]
            out['actual_u'][query] = target[qi]
            out['covariance_u'][query] = covariances[arm]
            scores = score_distribution(mean[qi], covariances[arm], target[qi], stats,
                actual[query], obs_actual[query], absolute_actual[query], norm2_per_feature[query],
                seed=SEED+100*fold+half)
            for key, value in scores.items():
                if key not in out:
                    out[key] = np.empty((n, *value.shape[1:]))
                out[key][query] = value
            order = np.lexsort((data['ids'][query], -scores['predicted']))
            out['selected'][query[order[:cell['budget']]]] = 1
            print(json.dumps(dict(fold=fold, half=half, arm=arm, stage='scored',
                elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
        seen[query] += 1
        fold_index[query] = fold
        write_json(root/'status.json', dict(state='RUNNING', stage='joint scoring',
            cells_complete=len(cells), elapsed_seconds=time.monotonic()-started))
    if not np.all(seen == 1):
        raise ValueError('Each of 1188 queries must be evaluated once')
    for arm in ARMS:
        np.testing.assert_array_equal(stores[arm]['mean_u'], stores['BASE']['mean_u'])
    metrics, comparisons, descriptive = evaluate_summary(stores, actual, data, layout, fold_index, lognorm)
    for arm, values in stores.items():
        np.savez_compressed(root/(arm+'.npz'), ids=data['ids'], actual=actual, fold=fold_index,
                            layout=layout, groups=data['groups'], **values)
    result = dict(scope='opened LINCS frozen-mean conditional error development comparison',
        n=n, samples=SAMPLES, arms=list(ARMS), reference_run=str(reference),
        reference_diagnostic=str(Path(reference_diagnostic).resolve()),
        metrics=metrics, comparisons=comparisons, descriptive=descriptive, cells=cells,
        endpoint_changed=False, means_changed=False, final_opened=False,
        biological_relations_used=False, jepa_used=False, reference_cost_included=False,
        statistical_family='Gaussian on nine fold-standardized legal geometry coordinates',
        interval_scope='conditional fixed predictions; no refitting or repeated-development correction',
        observable_names=list(OBSERVABLES), elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json', result)
    write_json(root/'status.json', dict(state='COMPLETE', cells_complete=len(cells),
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps({'state': 'COMPLETE', 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-diagnostic', type=Path,
        default=PROJECT/'runs/lincs_reference_information_20260916_v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.reference_diagnostic, args.output)
