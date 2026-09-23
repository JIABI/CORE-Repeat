"""Finite shared-scale Gaussian mixtures at fixed conditional mean/covariance.

The scalar component belongs to a complete nine-dimensional geometry residual,
not to independent coordinates or physical wells. No covariance is fitted here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import chi2, norm, spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_joint_error import gaussian_score
from .conditional_joint_error_experiment import (
    PROJECT, SEED, SAMPLES, OBSERVABLES, SCALAR_SCORES, read_json,
    observable_forward, score_distribution, interval_scores)
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .lincs_biology_experiment import load_data
from .objective_analysis import fair_crps
from .reference_information_diagnostic import bootstrap_difference, gamma_forward

ARMS = ('GAUSSIAN', 'RADIAL_SCALE')
LEVELS = (.5, .8, .9, .95, .99)
SCALE_SEED_OFFSET = 36000
TIE_ATOL = TIE_RTOL = 1e-12
CANDIDATES = (
    dict(name='GAUSSIAN', v_small=1., p_large=0., v_large=1.),
    dict(name='SMALL_050_RARE_010', v_small=.5, p_large=.1, v_large=5.5),
    dict(name='SMALL_050_RARE_020', v_small=.5, p_large=.2, v_large=3.),
    dict(name='SMALL_075_RARE_010', v_small=.75, p_large=.1, v_large=3.25),
    dict(name='SMALL_075_RARE_020', v_small=.75, p_large=.2, v_large=2.),
)
COMPARE = ('nll', 'energy', 'joint_coverage', 'coordinate_coverage', 'single_crps',
    'pair_crps', 'average_crps', 'triple_average_crps', 'absolute_pair_crps',
    'crps', 'brier', 'policy_value', 'coverage')


def _candidate(candidate):
    small, p, large = (float(candidate[k]) for k in ('v_small', 'p_large', 'v_large'))
    if not np.isfinite([small, p, large]).all() or not (0 <= p < 1):
        raise ValueError('Finite variances and a probability in [0,1) are required')
    if small <= 0 or large <= 0:
        raise ValueError('Both component variances must be strictly positive')
    if p == 0:
        if small != 1 or large != 1:
            raise ValueError('The Gaussian candidate must have both variances equal to one')
    elif not (small < 1 < large):
        raise ValueError('A two-scale law requires v_small < 1 < v_large')
    if not np.isclose((1-p)*small+p*large, 1., atol=1e-14, rtol=1e-14):
        raise ValueError('Component variances must preserve covariance exactly')
    return small, p, large


def _covariance(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 3 or covariance.shape[1] != covariance.shape[2]
            or min(covariance.shape) < 1 or not np.isfinite(covariance).all()):
        raise ValueError('Covariance must be finite [n,d,d] with positive dimensions')
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-12, rtol=1e-12):
        raise ValueError('Covariance must be symmetric')
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Covariance must be positive definite; no clipping is applied') from exc
    return covariance, chol


def radial_nll(residual, covariance, candidate):
    """Per-object exact full-density NLL of a zero-mean covariance-matched law."""
    small, p, large = _candidate(candidate)
    covariance, chol = _covariance(covariance)
    residual = np.asarray(residual, dtype=float)
    if residual.shape != covariance.shape[:2] or not np.isfinite(residual).all():
        raise ValueError('Residuals must be finite [n,d] and match covariance')
    if p == 0:
        return gaussian_score(residual, covariance)
    d = residual.shape[1]
    whitened = np.linalg.solve(chol, residual[..., None])[..., 0]
    mahal = np.square(whitened).sum(-1)
    common = d*np.log(2*np.pi)+2*np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
    logpdf = np.stack([
        np.log1p(-p)-.5*(common+d*np.log(small)+mahal/small),
        np.log(p)-.5*(common+d*np.log(large)+mahal/large)])
    return -logsumexp(logpdf, axis=0)


def radial_quantiles(candidate, dimension=9, levels=LEVELS):
    """Analytic-law joint squared radii and standardized marginal halfwidths."""
    small, p, large = _candidate(candidate)
    levels = np.asarray(levels, dtype=float)
    if (not isinstance(dimension, (int, np.integer)) or dimension < 1 or levels.ndim != 1
            or not np.isfinite(levels).all() or np.any((levels <= 0) | (levels >= 1))):
        raise ValueError('Positive integer dimension and levels strictly inside (0,1) required')
    joint, coordinate = chi2.ppf(levels, dimension), norm.ppf((1+levels)/2)
    if p > 0:
        joint = np.array([brentq(lambda q: (1-p)*chi2.cdf(q/small, dimension)
            +p*chi2.cdf(q/large, dimension)-level, small*q0, large*q0,
            xtol=1e-13, rtol=1e-14) for level, q0 in zip(levels, joint)])
        coordinate = np.array([brentq(lambda h: (1-p)*norm.cdf(h/np.sqrt(small))
            +p*norm.cdf(h/np.sqrt(large))-(1+level)/2,
            np.sqrt(small)*h0, np.sqrt(large)*h0, xtol=1e-13, rtol=1e-14)
            for level, h0 in zip(levels, coordinate)])
    return dict(levels=levels, joint_squared_radius=joint, coordinate_halfwidth=coordinate)


def sample_radial(covariance, candidate, normal_draws, component_uniforms):
    """Use exactly one shared scalar variance for each complete residual draw."""
    small, p, large = _candidate(candidate)
    covariance, chol = _covariance(covariance)
    z, uniforms = np.asarray(normal_draws, dtype=float), np.asarray(component_uniforms, dtype=float)
    if (z.ndim != 3 or z.shape[1:] != covariance.shape[:2]
            or uniforms.shape != z.shape[:2] or not np.isfinite(z).all()
            or not np.isfinite(uniforms).all() or np.any((uniforms < 0) | (uniforms >= 1))):
        raise ValueError('Expected finite normals [s,n,d] and uniforms [s,n] in [0,1)')
    residual = np.einsum('nij,snj->sni', chol, z)
    if p == 0:
        return residual
    variance = np.where(uniforms < p, large, small)
    return residual*np.sqrt(variance)[..., None]


def select_radial_candidate(cal_residual, cal_covariance):
    """Choose solely from supplied calibration residuals; fixed Gaussian-first ties."""
    if len(cal_residual) == 0:
        raise ValueError('At least one calibration representative is required')
    rows, best, best_score = [], 0, np.inf
    for index, candidate in enumerate(CANDIDATES):
        score = float(radial_nll(cal_residual, cal_covariance, candidate).mean())
        if not np.isfinite(score):
            raise ValueError('Calibration NLL must be finite')
        rows.append(dict(candidate=candidate.copy(), cal_nll=score))
        tolerance = TIE_ATOL+TIE_RTOL*abs(best_score) if np.isfinite(best_score) else 0.
        if score < best_score-tolerance:
            best, best_score = index, score
    return dict(candidate=CANDIDATES[best].copy(), candidate_index=best, cal_nll=best_score,
        candidate_scores=rows, calibration_n=len(cal_residual),
        tie_rule='fixed candidate order, Gaussian first', tie_atol=TIE_ATOL, tie_rtol=TIE_RTOL)


def _regions(out, mean, covariance, target, candidate):
    residual = target-mean
    sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    quantiles = radial_quantiles(candidate, mean.shape[1])
    halfwidth = sd[..., None]*quantiles['coordinate_halfwidth']
    out['coordinate_coverage_by_level'] = (np.abs(residual)[..., None] <= halfwidth).mean(1)
    out['coordinate_width_by_level'] = 2*halfwidth.mean(1)
    out['joint_coverage_by_level'] = (out['mahalanobis2'][:, None]
                                    <= quantiles['joint_squared_radius']).astype(float)
    primary = LEVELS.index(.95)
    out['coordinate_lower'] = mean-halfwidth[..., primary]
    out['coordinate_upper'] = mean+halfwidth[..., primary]
    out['joint_squared_radius'] = np.full(len(mean), quantiles['joint_squared_radius'][primary])
    out['coordinate_coverage'] = out['coordinate_coverage_by_level'][:, primary]
    out['joint_coverage'] = out['joint_coverage_by_level'][:, primary]


def score_radial(mean, covariance, target, candidate, stats, actual, obs_actual,
                 absolute_actual, norm2_per_feature, seed, samples=SAMPLES):
    """Established proper scores, with only the complete residual law changed."""
    _, p, _ = _candidate(candidate)
    if p == 0:
        out = score_distribution(mean, covariance, target, stats, actual, obs_actual,
            absolute_actual, norm2_per_feature, seed, samples)
        _regions(out, mean, covariance, target, candidate)
        return out
    if samples < 4 or samples % 2:
        raise ValueError('Independent energy pairs require an even sample count')
    n = len(mean)
    out = {key: np.empty(n) for key in SCALAR_SCORES}
    for name in ('observable_crps', 'observable_coverage', 'observable_width'):
        out[name] = np.empty((n, len(OBSERVABLES)))
    for name in ('absolute_crps_by_pair', 'absolute_coverage_by_pair'):
        out[name] = np.empty((n, 3))
    residual = target-mean
    out['nll'] = radial_nll(residual, covariance, candidate)
    out['mahalanobis2'] = np.einsum('ni,ni->n', residual,
        np.linalg.solve(covariance, residual[..., None])[..., 0])
    _regions(out, mean, covariance, target, candidate)
    normal_rng, scale_rng = np.random.default_rng(seed), np.random.default_rng(seed+SCALE_SEED_OFFSET)
    for start in range(0, n, 16):
        end = min(start+16, n)
        z = normal_rng.normal(size=(samples, end-start, 9))
        u = mean[None, start:end]+sample_radial(covariance[start:end], candidate, z,
                                               scale_rng.random((samples, end-start)))
        out['energy'][start:end] = np.linalg.norm(u-target[None, start:end], axis=-1).mean(0) \
            -.5*np.linalg.norm(u[:samples//2]-u[samples//2:], axis=-1).mean(0)
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
        probability = (gamma <= 0).mean(0)
        out['predicted'][start:end] = gamma.mean(0)
        out['p_null'][start:end] = probability
        out['crps'][start:end] = fair_crps(gamma, actual[start:end])
        out['coverage'][start:end] = (actual[start:end] >= glo) & (actual[start:end] <= ghi)
        out['gamma_lower'][start:end], out['gamma_upper'][start:end] = glo, ghi
        out['gamma_mc_se'][start:end] = gamma.std(0, ddof=1)/np.sqrt(samples)
        out['null_mc_se'][start:end] = np.sqrt(probability*(1-probability)/samples)
        out['predicted_cos_Z1_V'][start:end] = cosine[..., 0].mean(0)
        out['predicted_cos_Z2_V'][start:end] = cosine[..., 1].mean(0)
    if any(not np.isfinite(value).all() for value in out.values()):
        raise ValueError('Nonfinite radial-law scores; no sampled value is removed')
    return out


def summarize(stores, actual, groups, layout, folds):
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
            observable_width=dict(zip(OBSERVABLES, out['observable_width'].mean(0).tolist())),
            regions={str(level): dict(coordinate_coverage=float(out['coordinate_coverage_by_level'][:, j].mean()),
                joint_coverage=float(out['joint_coverage_by_level'][:, j].mean()),
                coordinate_width=float(out['coordinate_width_by_level'][:, j].mean()))
                for j, level in enumerate(LEVELS)})
        metrics[arm] = row
    comparisons = {'RADIAL_SCALE minus GAUSSIAN': {key: {scope: bootstrap_difference(
        stores['RADIAL_SCALE'][key], stores['GAUSSIAN'][key], labels)
        for scope, labels in (('chemistry', groups), ('layout', layout))} for key in COMPARE}}
    by_fold = {str(f): {arm: {key: float(stores[arm][key][folds == f].mean())
        for key in COMPARE} for arm in ARMS} for f in np.unique(folds)}
    return metrics, comparisons, by_fold


def _representatives(cal_ids, cal_groups, representative_ids):
    if len(set(cal_ids.tolist())) != len(cal_ids):
        raise ValueError('Calibration identities must be unique')
    expected = {min(cal_ids[cal_groups == group]) for group in np.unique(cal_groups)}
    if len(expected) != 40 or len(representative_ids) != 40 or set(representative_ids) != expected:
        raise ValueError('The same 40 ID-min calibration group representatives are required')
    position = {value: i for i, value in enumerate(cal_ids)}
    return np.array([position[value] for value in representative_ids], dtype=int)


def run(previous_run, output):
    started = time.monotonic()
    previous_run, root = Path(previous_run).resolve(), Path(output).resolve()
    if read_json(previous_run/'status.json')['state'] != 'COMPLETE':
        raise ValueError('The saved reference run must be complete')
    if root.exists():
        raise FileExistsError(root)
    previous = read_json(previous_run/'summary.json')
    reference = Path(previous['reference_run'])
    manifest = read_json(reference/'run_manifest.json')
    data, metadata = load_data(manifest['data_directory'])
    n = len(data['ids'])
    if n != 1188 or data['ids'].tolist() != manifest['ids'] or len(previous['cells']) != 10:
        raise ValueError('Opened dataset or fixed cell scope changed')
    index = {value: i for i, value in enumerate(data['ids'])}
    layout = np.array([value['layout_block'] for value in metadata['units']])
    norm2_per_feature = np.square(data['Y'][:, 0]).mean(1)
    gram = profiles_to_gram(torch.tensor(data['Y']))
    raw = gram_to_coordinates(gram).numpy()
    actual, obs_actual, difference, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, gram_gains(gram).numpy()[:, 2], atol=1e-12, rtol=1e-12)
    absolute_actual = np.log1p(difference*norm2_per_feature[:, None])
    with np.load(previous_run/'GAUSSIAN.npz') as archive:
        prior = {key: archive[key].copy() for key in archive.files}
    for key, value in (('ids', data['ids']), ('groups', data['groups']), ('layout', layout), ('actual', actual)):
        np.testing.assert_array_equal(value, prior[key])
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/JOINT_RADIAL_SCALE_PLAN_20260916.md', root/'PROTOCOL.md')
    for name in ('joint_radial_scale_experiment.py', 'conditional_joint_error.py',
                 'conditional_joint_error_experiment.py'):
        shutil.copy2(PROJECT/'opal2'/name, root/name)
    stores = {arm: {key: prior[key].copy() for key in ('mean_u', 'actual_u', 'covariance_u')}
              for arm in ARMS}
    for out in stores.values():
        out['selected'] = np.zeros(n, dtype=int)
    seen, folds, cells = np.zeros(n, dtype=int), np.full(n, -1, dtype=int), []
    records = {record['fold']: record for record in manifest['folds']}
    write_json(root/'status.json', dict(state='RUNNING', cells_complete=0))
    for source in previous['cells']:
        fold, half = source['fold'], source['half']
        record = records[fold]
        query, fit, cal = (np.array([index[value] for value in source[key]], dtype=int)
                          for key in ('query_ids', 'fit_ids', 'calibration_ids'))
        groups = [set(data['groups'][rows]) for rows in (np.asarray(record['fit']), fit, cal, query)]
        if any(groups[a] & groups[b] for a in range(4) for b in range(a+1, 4)):
            raise ValueError('Chemical groups overlap across frozen-model FIT/reference FIT/CAL/query')
        if set(np.r_[query, fit, cal]) != set(record['test']):
            raise ValueError('Prior query/reference split changed')
        with np.load(previous_run/f'cell_{fold}_{half}_reference.npz') as archive:
            reference_arrays = {key: archive[key].copy() for key in archive.files}
        for name, rows in (('fit_ids', fit), ('cal_ids', cal), ('query_ids', query)):
            np.testing.assert_array_equal(reference_arrays[name], data['ids'][rows])
        covariance = reference_arrays['query_covariance']
        np.testing.assert_array_equal(covariance, prior['covariance_u'][query])
        reps = _representatives(data['ids'][cal], data['groups'][cal],
                                source['calibration_representative_ids'])
        choice = select_radial_candidate(reference_arrays['cal_residual'][reps],
                                          reference_arrays['cal_covariance'][reps])
        candidate = choice['candidate']
        stats = read_json(reference/'folds'/f'fold_{fold}'/'preprocessing.json')
        mean, target = prior['mean_u'][query], prior['actual_u'][query]
        np.testing.assert_allclose(target, (raw[query]-stats['u_center'])/stats['u_scale'])
        small, p, large = _candidate(candidate)
        cell = dict(fold=fold, half=half, query_ids=source['query_ids'], fit_ids=source['fit_ids'],
            calibration_ids=source['calibration_ids'],
            calibration_representative_ids=source['calibration_representative_ids'],
            calibration_representative_indices=reps.tolist(), budget=source['budget'],
            covariance_choice=source['covariance_choice'], choice=choice,
            quantiles={arm: radial_quantiles(CANDIDATES[0] if arm == 'GAUSSIAN' else candidate)
                       for arm in ARMS},
            variance_multiplier=(1-p)*small+p*large,
            squared_variance_multiplier=(1-p)*small**2+p*large**2,
            normal_seed=SEED+100*fold+half, component_seed=SEED+100*fold+half+SCALE_SEED_OFFSET)
        cells.append(cell)
        write_json(root/f'cell_{fold}_{half}.json', cell)
        np.savez_compressed(root/f'cell_{fold}_{half}_calibration.npz',
            cal_ids=data['ids'][cal], residual=reference_arrays['cal_residual'],
            covariance=reference_arrays['cal_covariance'], representative_indices=reps)
        scores_by_arm = {}
        for arm in ARMS:
            if arm == 'RADIAL_SCALE' and p == 0:
                scores = {key: value.copy() for key, value in scores_by_arm['GAUSSIAN'].items()}
            else:
                scores = score_radial(mean, covariance, target,
                    CANDIDATES[0] if arm == 'GAUSSIAN' else candidate, stats,
                    actual[query], obs_actual[query], absolute_actual[query], norm2_per_feature[query],
                    seed=SEED+100*fold+half)
            scores_by_arm[arm] = scores
            out = stores[arm]
            for key, value in scores.items():
                if key not in out:
                    out[key] = np.empty((n, *value.shape[1:]))
                out[key][query] = value
            order = np.lexsort((data['ids'][query], -scores['predicted']))
            out['selected'][query[order[:cell['budget']]]] = 1
            print(json.dumps(dict(fold=fold, half=half, arm=arm, candidate=(
                'GAUSSIAN' if arm == 'GAUSSIAN' else candidate['name']),
                elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
        cell['query_metrics'] = {arm: {key: float(scores_by_arm[arm][key].mean())
            for key in COMPARE if key not in ('brier', 'policy_value')} for arm in ARMS}
        cell['selected_symmetric_difference'] = int(np.count_nonzero(
            stores['GAUSSIAN']['selected'][query] != stores['RADIAL_SCALE']['selected'][query]))
        write_json(root/f'cell_{fold}_{half}.json', cell)
        seen[query] += 1
        folds[query] = fold
        write_json(root/'status.json', dict(state='RUNNING', cells_complete=len(cells),
            elapsed_seconds=time.monotonic()-started))
    if not np.all(seen == 1):
        raise ValueError('Every opened query must be evaluated exactly once')
    np.testing.assert_array_equal(folds, prior['fold'])
    metrics, comparisons, by_fold = summarize(stores, actual, data['groups'], layout, folds)
    replay_keys = [key for key in prior if key not in ('ids', 'groups', 'layout', 'fold', 'actual')]
    for key in replay_keys:
        np.testing.assert_array_equal(stores['GAUSSIAN'][key], prior[key], err_msg=f'Gaussian replay: {key}')
    for key in ('mean_u', 'actual_u', 'covariance_u'):
        np.testing.assert_array_equal(stores['GAUSSIAN'][key], stores['RADIAL_SCALE'][key])
    for arm, values in stores.items():
        np.savez_compressed(root/(arm+'.npz'), ids=data['ids'], groups=data['groups'], layout=layout,
            fold=folds, actual=actual, **values)
    result = dict(n=n, samples=SAMPLES, arms=list(ARMS), cells=cells, metrics=metrics,
        comparisons=comparisons, by_fold=by_fold, candidates=CANDIDATES,
        previous_run=str(previous_run), reference_run=str(reference),
        means_changed=False, covariances_changed=False, endpoint_changed=False, final_opened=False,
        biological_relations_used=False, jepa_used=False, reference_cost_included=False,
        gaussian_replay_exact=True, gaussian_replay_keys=replay_keys,
        normal_seed_formula='20260916+100*fold+half', component_seed_offset=SCALE_SEED_OFFSET,
        law='Covariance-matched finite two-scale Gaussian mixture; one scalar per complete 9D draw',
        student_t_deferred_reason='No exponential moments in log-Cholesky coordinates',
        scope='opened DEV retrospective shape comparison; shared-layout dependencies remain',
        ci_scope='fixed predictions; chemistry/layout bootstrap; no refitting or repeated-development correction',
        coverage_scope='analytic predictive-law regions, not formal coverage guarantees',
        elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json', result)
    write_json(root/'status.json', dict(state='COMPLETE', cells_complete=len(cells),
        elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(state='COMPLETE', elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-run', type=Path,
        default=PROJECT/'runs/lincs_joint_residual_borrowing_20260916_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.previous_run, args.output)
