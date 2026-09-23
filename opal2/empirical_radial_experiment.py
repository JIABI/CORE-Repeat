"""Fixed STATE50 means, calibration-only empirical radial predictive laws."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2, norm, spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import (PROJECT, SEED, SAMPLES, OBSERVABLES,
    SCALAR_SCORES, read_json, observable_forward, interval_scores, score_distribution)
from .empirical_radial import (fit_radial, reference_weights, radial_cdf, radial_ppf,
    radial_nll, draw_radial, variance_multiplier)
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .joint_contrast_scale import predict_scale
from .joint_radial_scale_experiment import _representatives
from .lincs_biology_experiment import load_data
from .objective_analysis import fair_crps
from .reference_information_diagnostic import bootstrap_difference, gamma_forward

ARMS = ('GAUSSIAN', 'AMP_GAUSSIAN', 'EMP_GLOBAL', 'AMP_EMP_GLOBAL', 'AMP_EMP_LOCAL')
LEVELS = np.array([.5, .8, .9, .95, .99])
COMPARE = ('nll', 'energy', 'single_crps', 'pair_crps', 'average_crps',
    'triple_average_crps', 'absolute_pair_crps', 'crps', 'brier', 'policy_value',
    'coverage', 'joint_coverage')
SEED_OFFSET = 47000


def interval_levels(draws, actual):
    lower = np.quantile(draws, (1-LEVELS)/2, axis=0)
    upper = np.quantile(draws, (1+LEVELS)/2, axis=0)
    return (np.moveaxis((actual[None] >= lower) & (actual[None] <= upper), 0, -1),
            np.moveaxis(upper-lower, 0, -1), lower, upper)


def score(mean, scatter, target, stats, actual, obs_actual, absolute_actual,
          norm2_per_feature, seed, *, law=None, weights=None, samples=SAMPLES):
    """Same full-vector sampling for all observables; analytic joint regions."""
    if samples < 4 or samples % 2:
        raise ValueError('Energy score needs an even number of independent samples')
    n, d = mean.shape
    out = {key: np.empty(n) for key in SCALAR_SCORES}
    for key in ('observable_crps', 'observable_coverage', 'observable_width'):
        out[key] = np.empty((n, len(OBSERVABLES)))
    for key in ('absolute_crps_by_pair', 'absolute_coverage_by_pair'):
        out[key] = np.empty((n, 3))
    for key in ('coordinate_coverage_by_level', 'coordinate_width_by_level',
                'gamma_coverage_by_level', 'gamma_width_by_level'):
        out[key] = np.empty((n, len(LEVELS)))
    for key in ('observable_coverage_by_level', 'observable_width_by_level'):
        out[key] = np.empty((n, len(OBSERVABLES), len(LEVELS)))
    residual = target-mean
    chol = np.linalg.cholesky(scatter)
    whitened = np.linalg.solve(chol, residual[..., None])[..., 0]
    radius2 = np.square(whitened).sum(-1)
    out['mahalanobis2'] = radius2
    if law is None:
        logdet = 2*np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(-1)
        out['nll'] = .5*(d*np.log(2*np.pi)+logdet+radius2)
        thresholds = np.broadcast_to(chi2.ppf(LEVELS, d), (n, len(LEVELS))).copy()
        out['radial_pit'] = chi2.cdf(radius2, d)
        multiplier = np.ones(n)
    else:
        out['nll'] = radial_nll(residual, scatter, law, weights)
        thresholds = radial_ppf(law, weights, LEVELS)**2
        out['radial_pit'] = radial_cdf(law, weights, np.sqrt(radius2))
        multiplier = variance_multiplier(law, weights)
    out['joint_squared_radius_by_level'] = thresholds
    out['joint_coverage_by_level'] = (radius2[:, None] <= thresholds).astype(float)
    out['joint_coverage'] = out['joint_coverage_by_level'][:, 3]
    out['radial_variance_multiplier'] = multiplier
    out['covariance_u'] = scatter*multiplier[:, None, None]
    out['coordinate_lower'], out['coordinate_upper'] = np.empty_like(mean), np.empty_like(mean)
    normal_rng = np.random.default_rng(seed)
    radius_rng = np.random.default_rng(seed+SEED_OFFSET)
    for start in range(0, n, 16):
        end = min(start+16, n)
        normal = normal_rng.normal(size=(samples, end-start, d))
        if law is None:
            eps = np.einsum('nij,snj->sni', chol[start:end], normal)
        else:
            eps = draw_radial(law, weights[start:end], scatter[start:end], normal,
                radius_rng.random((samples, end-start)), radius_rng.random((samples, end-start)))
        u = mean[None, start:end]+eps
        out['energy'][start:end] = np.linalg.norm(u-target[None, start:end], axis=-1).mean(0) \
            -.5*np.linalg.norm(u[:samples//2]-u[samples//2:], axis=-1).mean(0)
        cc, cw, cl, cu = interval_levels(u, target[start:end])
        if law is None:
            sd = np.sqrt(np.diagonal(scatter[start:end], axis1=-2, axis2=-1))
            h = sd[..., None]*norm.ppf((1+LEVELS)/2)
            cc = (np.abs(residual[start:end])[..., None] <= h)
            cw = 2*h
            out['coordinate_lower'][start:end] = mean[start:end]-h[..., 3]
            out['coordinate_upper'][start:end] = mean[start:end]+h[..., 3]
        else:
            out['coordinate_lower'][start:end] = cl[3]
            out['coordinate_upper'][start:end] = cu[3]
        out['coordinate_coverage_by_level'][start:end] = cc.mean(1)
        out['coordinate_width_by_level'][start:end] = cw.mean(1)
        out['coordinate_coverage'][start:end] = cc[..., 3].mean(1)
        raw = u*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
        gamma, obs, differences, cosine = observable_forward(raw)
        np.testing.assert_allclose(gamma, gamma_forward(raw), atol=1e-12, rtol=1e-12)
        crps, cover, width = interval_scores(obs, obs_actual[start:end])
        out['observable_crps'][start:end] = crps
        out['observable_coverage'][start:end] = cover
        out['observable_width'][start:end] = width
        oc, ow, _, _ = interval_levels(obs, obs_actual[start:end])
        out['observable_coverage_by_level'][start:end] = oc
        out['observable_width_by_level'][start:end] = ow
        for prefix, cols in (('single', slice(0, 3)), ('pair', slice(3, 6)),
                             ('average', slice(6, 10)), ('triple_average', slice(9, 10))):
            out[prefix+'_crps'][start:end] = crps[:, cols].mean(1)
            out[prefix+'_coverage'][start:end] = cover[:, cols].mean(1)
        absolute = np.log1p(differences*norm2_per_feature[None, start:end, None])
        ac, av, _ = interval_scores(absolute, absolute_actual[start:end])
        out['absolute_crps_by_pair'][start:end] = ac
        out['absolute_coverage_by_pair'][start:end] = av
        out['absolute_pair_crps'][start:end], out['absolute_pair_coverage'][start:end] = ac.mean(1), av.mean(1)
        gc, gw, gl, gu = interval_levels(gamma, actual[start:end])
        probability = (gamma <= 0).mean(0)
        out['predicted'][start:end] = gamma.mean(0)
        out['p_null'][start:end] = probability
        out['crps'][start:end] = fair_crps(gamma, actual[start:end])
        out['gamma_coverage_by_level'][start:end], out['gamma_width_by_level'][start:end] = gc, gw
        out['coverage'][start:end] = gc[..., 3]
        out['gamma_lower'][start:end], out['gamma_upper'][start:end] = gl[3], gu[3]
        out['gamma_mc_se'][start:end] = gamma.std(0, ddof=1)/np.sqrt(samples)
        out['null_mc_se'][start:end] = np.sqrt(probability*(1-probability)/samples)
        out['predicted_cos_Z1_V'][start:end] = cosine[..., 0].mean(0)
        out['predicted_cos_Z2_V'][start:end] = cosine[..., 1].mean(0)
    if any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError('Nonfinite outputs: no values are dropped or clipped')
    return out


def summarize(stores, actual, groups, layout, folds):
    metrics, by_fold = {}, {}
    for arm, out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = actual*out['selected']
        out['mse'] = np.square(out['actual_u']-out['mean_u']).mean(1)
        chosen = out['selected'].astype(bool)
        if chosen.sum() != 146:
            raise ValueError('Query budget changed')
        row = {key: float(v.mean()) for key, v in out.items() if v.ndim == 1 and key != 'selected'}
        row.update(selected_n=int(chosen.sum()), selected_null=int((actual[chosen] <= 0).sum()),
            selected_mean=float(actual[chosen].mean()), selected_symmetric_difference=int(
                np.count_nonzero(out['selected'] != stores['GAUSSIAN']['selected'])),
            spearman=float(spearmanr(actual, out['predicted']).statistic),
            null_auc=float(roc_auc_score(actual <= 0, out['p_null'])),
            regions={str(p): {k: float(out[k][:, j].mean()) for k in
                ('joint_coverage_by_level', 'coordinate_coverage_by_level',
                 'gamma_coverage_by_level', 'gamma_width_by_level')}
                for j, p in enumerate(LEVELS)},
            covariance_multiplier_quantiles=np.quantile(out['radial_variance_multiplier'], [0, .1, .5, .9, 1]).tolist(),
            effective_radial_references_quantiles=np.quantile(out['radial_ess'], [0, .1, .5, .9, 1]).tolist(),
            observables={name: dict(crps=float(out['observable_crps'][:, j].mean()),
                coverage_by_level=out['observable_coverage_by_level'][:, j].mean(0).tolist(),
                width_by_level=out['observable_width_by_level'][:, j].mean(0).tolist())
                for j, name in enumerate(OBSERVABLES)})
        base_error = np.abs(stores['GAUSSIAN']['joint_coverage_by_level'].mean(0)-LEVELS)
        error = np.abs(out['joint_coverage_by_level'].mean(0)-LEVELS)
        row['all_joint_levels_no_worse_than_gaussian_descriptive'] = bool(np.all(error <= base_error+1e-12))
        row['joint_absolute_coverage_error'] = error.tolist()
        row['mean_joint_absolute_coverage_error'] = float(error.mean())
        metrics[arm] = row
    pairs = [(a, 'GAUSSIAN') for a in ARMS[1:]]+[
        ('AMP_EMP_GLOBAL', 'AMP_GAUSSIAN'), ('AMP_EMP_GLOBAL', 'EMP_GLOBAL'),
        ('AMP_EMP_LOCAL', 'AMP_EMP_GLOBAL')]
    comparisons = {}
    for a, b in pairs:
        record = {key: {scope: bootstrap_difference(stores[a][key], stores[b][key], labels)
            for scope, labels in (('chemistry', groups), ('layout', layout))} for key in COMPARE}
        record['joint_coverage_by_level'] = {str(p): {scope: bootstrap_difference(
            stores[a]['joint_coverage_by_level'][:, j], stores[b]['joint_coverage_by_level'][:, j], labels)
            for scope, labels in (('chemistry', groups), ('layout', layout))} for j, p in enumerate(LEVELS)}
        comparisons[a+' minus '+b] = record
    for fold in np.unique(folds):
        take = folds == fold
        by_fold[str(fold)] = {arm: dict(**{key: float(out[key][take].mean()) for key in COMPARE},
            joint_coverage_by_level=out['joint_coverage_by_level'][take].mean(0).tolist())
            for arm, out in stores.items()}
    return metrics, comparisons, by_fold


def run(previous_run, component_run, output):
    started = time.monotonic()
    previous_run, component_run, root = map(lambda p: Path(p).resolve(), (previous_run, component_run, output))
    if root.exists():
        raise FileExistsError(root)
    previous, component = (read_json(p/'summary.json') for p in (previous_run, component_run))
    reference = Path(previous['reference_run'])
    manifest = read_json(reference/'run_manifest.json')
    data, metadata = load_data(manifest['data_directory'])
    n = len(data['ids'])
    if n != 1188 or data['ids'].tolist() != manifest['ids'] or len(previous['cells']) != 10:
        raise ValueError('Opened object scope changed')
    lookup = {v: i for i, v in enumerate(data['ids'])}
    logamp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    norm2 = np.square(data['Y'][:, 0]).mean(1)
    layout = np.array([v['layout_block'] for v in metadata['units']])
    gram = profiles_to_gram(torch.tensor(data['Y']))
    raw = gram_to_coordinates(gram).numpy()
    actual, obs, differences, _ = observable_forward(raw)
    np.testing.assert_allclose(actual, gram_gains(gram).numpy()[:, 2], atol=1e-12, rtol=1e-12)
    absolute = np.log1p(differences*norm2[:, None])
    with np.load(previous_run/'GAUSSIAN.npz') as archive:
        prior = {key: archive[key].copy() for key in archive.files}
    with np.load(component_run/'AMPLITUDE_TOTAL.npz') as archive:
        amp_prior = {key: archive[key].copy() for key in archive.files}
    np.testing.assert_array_equal(prior['ids'], data['ids'])
    np.testing.assert_array_equal(amp_prior['ids'], data['ids'])
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/EMPIRICAL_RADIAL_PLAN_20260916.md', root/'PROTOCOL.md')
    for name in ('empirical_radial.py', 'empirical_radial_experiment.py',
                 'joint_contrast_scale.py', 'conditional_joint_error_experiment.py'):
        shutil.copy2(PROJECT/'opal2'/name, root/name)
    stores = {a: {key: prior[key].copy() for key in ('mean_u', 'actual_u')} for a in ARMS}
    for out in stores.values():
        out.update(selected=np.zeros(n, int), scatter_u=np.empty((n, 9, 9)), radial_ess=np.empty(n))
    cells, seen = [], np.zeros(n, int)
    records = {r['fold']: r for r in manifest['folds']}
    c_records = {(r['fold'], r['half']): r for r in component['cells']}
    write_json(root/'status.json', dict(state='RUNNING', cells_complete=0))
    for source in previous['cells']:
        fold, half = source['fold'], source['half']
        query, fit, cal = (np.array([lookup[v] for v in source[k]]) for k in
                          ('query_ids', 'fit_ids', 'calibration_ids'))
        parts = [set(data['groups'][rows]) for rows in (records[fold]['fit'], fit, cal, query)]
        if any(parts[i] & parts[j] for i in range(4) for j in range(i+1, 4)):
            raise ValueError('Chemical identity leakage')
        if set(np.r_[fit, cal, query]) != set(records[fold]['test']):
            raise ValueError('Reference/query membership changed')
        with np.load(previous_run/f'cell_{fold}_{half}_reference.npz') as archive:
            ref = {key: archive[key].copy() for key in archive.files}
        for key, rows in (('query_ids', query), ('fit_ids', fit), ('cal_ids', cal)):
            np.testing.assert_array_equal(ref[key], data['ids'][rows])
        reps = _representatives(data['ids'][cal], data['groups'][cal], source['calibration_representative_ids'])
        mean, target = prior['mean_u'][query], prior['actual_u'][query]
        base_c = ref['query_covariance']
        np.testing.assert_array_equal(base_c, prior['covariance_u'][query])
        fitted = c_records[(fold, half)]['fitters']['AMPLITUDE_TOTAL'][0]
        fit_sd = float(logamp[fit].std())
        np.testing.assert_allclose(fit_sd, fitted['scale'], atol=0, rtol=0)
        aq, ac = predict_scale(fitted, logamp[query]), predict_scale(fitted, logamp[cal])
        amp_c = base_c*aq[:, None, None]
        np.testing.assert_allclose(amp_c, amp_prior['covariance_u'][query], rtol=2e-14, atol=2e-14)
        # Use the exact earlier matrix arrays for an exact Gaussian sampling replay.
        amp_c = amp_prior['covariance_u'][query].copy()
        cal_c = ref['cal_covariance'][reps]
        cal_amp_c = cal_c*ac[reps, None, None]
        cal_res = ref['cal_residual'][reps]
        r = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(cal_c), cal_res[..., None])[..., 0], axis=1)
        ar = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(cal_amp_c), cal_res[..., None])[..., 0], axis=1)
        laws = dict(global_law=fit_radial(r), amplitude_law=fit_radial(ar))
        uniform = reference_weights(logamp[cal][reps], logamp[query], fit_sd, conditional=False)
        local = reference_weights(logamp[cal][reps], logamp[query], fit_sd, conditional=True)
        specs = dict(GAUSSIAN=(base_c, None, None), AMP_GAUSSIAN=(amp_c, None, None),
            EMP_GLOBAL=(base_c, laws['global_law'], uniform),
            AMP_EMP_GLOBAL=(amp_c, laws['amplitude_law'], uniform),
            AMP_EMP_LOCAL=(amp_c, laws['amplitude_law'], local))
        stats = read_json(reference/'folds'/f'fold_{fold}'/'preprocessing.json')
        np.testing.assert_allclose(target, (raw[query]-stats['u_center'])/stats['u_scale'])
        cell = dict(fold=fold, half=half, query_ids=data['ids'][query].tolist(),
            fit_ids=data['ids'][fit].tolist(), calibration_ids=data['ids'][cal].tolist(),
            representative_ids=data['ids'][cal][reps].tolist(), budget=source['budget'],
            laws=laws, amplitude_fit=fitted, local_bandwidth=fit_sd,
            normal_seed=SEED+100*fold+half, radial_seed=SEED+100*fold+half+SEED_OFFSET)
        np.savez_compressed(root/f'cell_{fold}_{half}_radial.npz', cal_ids=data['ids'][cal][reps],
            query_ids=data['ids'][query], cal_residual=cal_res, cal_scatter=cal_c,
            cal_amp_scatter=cal_amp_c, radii=r, amplitude_radii=ar,
            cal_log_amplitude=logamp[cal][reps], query_log_amplitude=logamp[query],
            local_weights=local['weights'], uniform_weights=uniform['weights'],
            local_ess=local['ess'], raw_local_ess=local['local_ess'],
            amplitude_factor=aq, cal_amplitude_factor=ac[reps])
        for arm in ARMS:
            scatter, law, weighted = specs[arm]
            output_scores = score(mean, scatter, target, stats, actual[query], obs[query],
                absolute[query], norm2[query], seed=SEED+100*fold+half, law=law,
                weights=None if weighted is None else weighted['weights'])
            out = stores[arm]
            out['scatter_u'][query] = scatter
            out['radial_ess'][query] = 0 if weighted is None else weighted['ess']
            for key, value in output_scores.items():
                if key not in out:
                    out[key] = np.empty((n, *value.shape[1:]))
                out[key][query] = value
            order = np.lexsort((data['ids'][query], -output_scores['predicted']))
            out['selected'][query[order[:source['budget']]]] = 1
            print(f'fold={fold} half={half} arm={arm} elapsed={time.monotonic()-started:.1f}', flush=True)
        cells.append(cell)
        write_json(root/f'cell_{fold}_{half}.json', cell)
        seen[query] += 1
        write_json(root/'status.json', dict(state='RUNNING', cells_complete=len(cells), elapsed_seconds=time.monotonic()-started))
    if not np.all(seen == 1):
        raise ValueError('Queries must each appear exactly once')
    for arm, old in (('GAUSSIAN', prior), ('AMP_GAUSSIAN', amp_prior)):
        for key in ('mean_u', 'covariance_u', 'predicted', 'p_null', 'crps', 'energy', 'selected'):
            np.testing.assert_array_equal(stores[arm][key], old[key], err_msg=f'{arm} replay {key}')
        np.testing.assert_allclose(stores[arm]['nll'], old['nll'], rtol=1e-12, atol=1e-12)
    metrics, comparisons, by_fold = summarize(stores, actual, data['groups'], layout, prior['fold'])
    for arm, values in stores.items():
        np.testing.assert_array_equal(values['mean_u'], prior['mean_u'])
        np.savez_compressed(root/(arm+'.npz'), ids=data['ids'], groups=data['groups'], layout=layout,
            fold=prior['fold'], actual=actual, **values)
    summary = dict(n=n, samples=SAMPLES, arms=list(ARMS), levels=LEVELS, cells=cells,
        metrics=metrics, comparisons=comparisons, by_fold=by_fold, previous_run=str(previous_run),
        component_run=str(component_run), reference_run=str(reference), data_directory=manifest['data_directory'],
        final_opened=False, means_changed=False, base_local_scatter_reused=True,
        covariance_preserved=False, covariance_change='Amplitude and fitted radial second moment',
        endpoint_changed=False, biology_used=False, jepa_used=False, query_tuning=False,
        gaussian_baselines_reproduced=True, calibration_law_n_per_cell=40,
        scope='Opened DEV, fixed split, shared-layout dependencies and repeated development',
        guarantee='No distribution-free, conditional or selective probability guarantee',
        ci_scope='Fixed predictions, chemistry/layout resampling, no model refitting',
        reference_cost_included=False, elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json', summary)
    write_json(root/'status.json', dict(state='COMPLETE', cells_complete=10, elapsed_seconds=time.monotonic()-started))
    print('COMPLETE', summary['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous-run', type=Path, default=PROJECT/'runs/lincs_joint_residual_borrowing_20260916_v1')
    parser.add_argument('--component-run', type=Path, default=PROJECT/'runs/lincs_joint_component_scale_20260916_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.previous_run, args.component_run, args.output)
