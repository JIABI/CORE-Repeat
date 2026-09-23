"""Honest native-coordinate AMP_EMP_LOCAL laws from saved inner CORE means.

Each inner held-out population supplies two disjoint reference/query cells.
All means in a cell were fitted without any of its reference or query groups.
Only that cell's DIST_FIT outcomes fit scatter and amplitude; separate DIST_CAL
groups fit the empirical radius law. No mean is trained or changed here.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .biology_kernel_evaluation import write_json
from .conditional_joint_error import fit_covariance_family
from .conditional_joint_error_experiment import reference_weights as covariance_weights
from .empirical_radial import fit_radial, reference_weights, variance_multiplier
from .joint_contrast_scale import fit_scale, predict_scale
from .joint_tail_calibration import mahalanobis_scores, split_donor_groups


def plan_distribution_cells(ids, groups, error_fold, mean_records, seed):
    """Identity-only two-way query/reference split within each mean error fold."""
    ids, groups, error_fold = map(np.asarray, (ids, groups, error_fold))
    n = len(ids)
    if ids.shape != (n,) or groups.shape != (n,) or error_fold.shape != (n,):
        raise ValueError('Aligned identity, group and mean-fold vectors required')
    if len(set(ids)) != n or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Unique IDs and a nonnegative integer seed required')
    by_fold = {int(r['fold']): r for r in mean_records}
    cells = []
    for fold in sorted(set(error_fold.tolist())):
        held = np.flatnonzero(error_fold == fold)
        record = by_fold[int(fold)]
        if set(ids[held]) != set(record['heldout_ids']):
            raise ValueError('Saved held-out identities do not match residual membership')
        forbidden = set(record['fit_ids']) | set(record['inner_validation_ids']) | set(record['reference_ids'])
        forbidden_groups = set(groups[np.isin(ids, list(forbidden))])
        if set(ids[held]) & forbidden or set(groups[held]) & forbidden_groups:
            raise ValueError('An inner mean saw a distribution reference/query group')
        unique = np.unique(groups[held])
        if len(unique) < 12:
            raise ValueError('At least twelve held-out groups required for two reference cells')
        shuffled = np.random.default_rng(seed + 10000*int(fold)).permutation(unique)
        query_groups = (shuffled[:len(shuffled)//2], shuffled[len(shuffled)//2:])
        for half in range(2):
            query = held[np.isin(groups[held], query_groups[half])]
            donor = held[~np.isin(groups[held], query_groups[half])]
            local_seed = seed + 10000*int(fold) + 100*half + 17
            split = split_donor_groups(groups[donor], seed=local_seed)
            fit, cal = donor[split['fit_indices']], donor[split['cal_indices']]
            representatives = np.array([min(np.flatnonzero(groups == g), key=lambda i: ids[i])
                for g in np.unique(groups[cal])], int)
            if not set(representatives) <= set(cal):
                raise ValueError('A chemistry group crossed distribution roles')
            parts = [set(groups[x]) for x in (fit, cal, query)]
            if any(parts[a] & parts[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError('A group crosses DIST_FIT, DIST_CAL or query')
            cells.append(dict(cell=len(cells), error_fold=int(fold), half=half,
                mean_fit_ids=record['fit_ids'], mean_validation_ids=record['inner_validation_ids'],
                mean_reference_ids=record['reference_ids'], split=split,
                fit_indices=fit.tolist(), cal_indices=cal.tolist(), query_indices=query.tolist(),
                representative_indices=representatives.tolist(),
                fit_ids=ids[fit].tolist(), cal_ids=ids[cal].tolist(), query_ids=ids[query].tolist(),
                representative_ids=ids[representatives].tolist(),
                biological_reference_rule='this cell DIST_FIT only; same inner mean fold; exclude entire query chemistry group'))
    count = np.zeros(n, int)
    for c in cells:
        count[c['query_indices']] += 1
    if not np.array_equal(count, np.ones(n, int)):
        raise ValueError('Each training object must receive exactly one predictive law')
    return cells


def construct_cell(data, raw_residual, raw_covariance, cell):
    """Complete distribution recipe; query residual values are never read."""
    fit, cal, query, reps = [np.asarray(cell[k], int) for k in
        ('fit_indices', 'cal_indices', 'query_indices', 'representative_indices')]
    ids, groups = np.asarray(data['ids']), np.asarray(data['groups'])
    logamp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    if not np.isfinite(logamp).all():
        raise ValueError('Finite nonzero observed first-well norms required')
    model_fit = np.flatnonzero(np.isin(ids, cell['mean_fit_ids']))
    if len(model_fit) != len(cell['mean_fit_ids']):
        raise ValueError('Mean fitting IDs are missing from caller MODEL_FIT')
    bandwidth = max(float(logamp[model_fit].std()), .1)
    cov0 = np.asarray(raw_covariance[query[0]], float)
    held = np.r_[fit, cal, query]
    np.testing.assert_array_equal(raw_covariance[held], np.broadcast_to(cov0, (len(held), 9, 9)))
    wf, _ = covariance_weights(data, fit, fit, bandwidth)
    wcq, _ = covariance_weights(data, np.r_[cal, query], fit, bandwidth)
    if np.any(wf[groups[fit, None] == groups[None, fit]] != 0):
        raise ValueError('Reference leave-group-out weights include their own group')
    fitted = fit_covariance_family(raw_residual[fit], cov0, wf, wcq, 'LOCAL_SCALE')
    cal_cov, query_cov = np.split(fitted['query_covariance'], [len(cal)])
    energies = mahalanobis_scores(raw_residual[fit], fitted['loo_covariance'])
    amplitude = fit_scale(energies, 9, logamp[fit], conditional=True, penalty=1.)
    fit_scatter = fitted['loo_covariance']*predict_scale(amplitude, logamp[fit])[:, None, None]
    cal_scatter = cal_cov*predict_scale(amplitude, logamp[cal])[:, None, None]
    query_scatter = query_cov*predict_scale(amplitude, logamp[query])[:, None, None]
    positions = {int(v): i for i, v in enumerate(cal)}
    ri = np.asarray([positions[int(v)] for v in reps], int)
    radii = np.sqrt(mahalanobis_scores(raw_residual[reps], cal_scatter[ri]))
    law = fit_radial(radii)
    # This is the original amplitude-conditioned radial retrieval bandwidth,
    # not the model-FIT bandwidth used by the covariance reference weights.
    radial_sd = float(logamp[fit].std())
    local = reference_weights(logamp[reps], logamp[query], radial_sd, conditional=True)
    reference_local = reference_weights(logamp[reps], logamp[fit], radial_sd, conditional=True)
    multiplier = variance_multiplier(law, local['weights'])
    reference_multiplier = variance_multiplier(law, reference_local['weights'])
    arrays = dict(query_ids=ids[query], reference_ids=ids[fit], calibration_ids=ids[cal],
        representative_ids=ids[reps], query_indices=query, reference_indices=fit,
        calibration_indices=cal, representative_indices=reps,
        query_raw_scatter=query_scatter,
        query_raw_covariance=query_scatter*multiplier[:, None, None],
        query_radial_weights=local['weights'], query_radial_variance_multiplier=multiplier,
        reference_raw_residual=raw_residual[fit], reference_raw_scatter=fit_scatter,
        reference_raw_covariance=fit_scatter*reference_multiplier[:, None, None],
        reference_radial_weights=reference_local['weights'],
        calibration_raw_residual=raw_residual[reps], calibration_raw_scatter=cal_scatter[ri],
        calibration_radii=radii, fit_total_energy=energies,
        query_radial_ess=local['ess'])
    report = dict(**cell, law=law, local_covariance_choice=fitted['choice'],
        amplitude_fit=amplitude, covariance_reference_bandwidth=bandwidth,
        radial_reference_bandwidth=radial_sd,
        reference_record_warning='Use cell-specific reference scatter/covariance, never donor global OOF scatter from its other query cell',
        query_outcomes_used_for_distribution=False,
        native_coordinate_recipe='LOCAL_SCALE + AMPLITUDE_TOTAL + AMP_EMP_LOCAL',
        calibration_group_count=len(reps), mean_retrained=False)
    return arrays, report


def build_nested_core_distribution(data, metadata, nested_directory, output_directory, seed=20260917):
    """Build all native predictive laws, preserving the saved nested row order.

    ``data`` must be the original outer MODEL_FIT subset. Returns all arrays
    saved to ``distribution.npz`` plus summary, cells and artifact paths.
    ``metadata`` is accepted for consistency with the saved-mean API; this
    distribution recipe does not use biological outcome annotations.
    """
    del metadata
    nested, root = Path(nested_directory), Path(output_directory)
    if root.exists():
        raise FileExistsError('Preserve completed nested distributions; use a new output directory')
    start = time.monotonic()
    with np.load(nested/'residuals.npz', allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    plan = json.loads((nested/'plan.json').read_text())
    ids, groups = np.asarray(data['ids']), np.asarray(data['groups'])
    np.testing.assert_array_equal(ids, saved['ids'])
    np.testing.assert_array_equal(groups, saved['groups'])
    np.testing.assert_array_equal(saved['prediction_count'], np.ones(len(ids), int))
    np.testing.assert_array_equal(saved['raw_residual'], saved['raw_target']-saved['raw_mean'])
    for key in ('raw_mean', 'raw_target', 'raw_residual', 'raw_covariance'):
        if not np.isfinite(saved[key]).all():
            raise ValueError('Finite saved native geometry required: '+key)
    cells = plan_distribution_cells(ids, groups, saved['error_fold'], plan['folds'], seed)
    root.mkdir(parents=True)
    write_json(root/'plan.json', dict(seed=int(seed), source_nested_directory=str(nested.resolve()),
        cells=cells, biology_provenance='cell-specific DIST_FIT records only; no global donor OOF covariance',
        cal_split_rule='same seeded 2/3 DIST_FIT, remainder DIST_CAL group algorithm as deployed CORE'))
    n = len(ids)
    result = {k: saved[k].copy() for k in ('ids', 'groups', 'raw_mean', 'raw_target', 'raw_residual', 'error_fold')}
    result.update(raw_scatter=np.empty((n, 9, 9)), raw_covariance=np.empty((n, 9, 9)),
        construction_cell=np.full(n, -1, int), biological_reference_mask=np.zeros((n, n), bool))
    reports = []
    for cell in cells:
        arrays, report = construct_cell(data, saved['raw_residual'], saved['raw_covariance'], cell)
        q, fit = arrays['query_indices'], arrays['reference_indices']
        result['raw_scatter'][q] = arrays['query_raw_scatter']
        result['raw_covariance'][q] = arrays['query_raw_covariance']
        result['construction_cell'][q] = cell['cell']
        result['biological_reference_mask'][np.ix_(q, fit)] = True
        arrays['reference_raw_mean'] = saved['raw_mean'][fit]
        arrays['query_raw_mean'] = saved['raw_mean'][q]
        stem = f"cell_{cell['cell']:02d}"
        np.savez_compressed(root/(stem+'.npz'), **arrays)
        write_json(root/(stem+'.json'), report)
        reports.append(report)
    mask = result['biological_reference_mask']
    if np.any(mask & (groups[:, None] == groups[None, :])) or np.any(mask & (saved['error_fold'][:, None] != saved['error_fold'][None, :])):
        raise ValueError('Illegal reference relationship in completed distribution')
    for key in ('raw_scatter', 'raw_covariance'):
        if not np.isfinite(result[key]).all():
            raise ValueError('Nonfinite completed '+key)
        np.linalg.cholesky(result[key])
    if np.any(result['construction_cell'] < 0):
        raise ValueError('Missing query distribution')
    npz_path = root/'distribution.npz'
    np.savez_compressed(npz_path, **result)
    summary = dict(state='COMPLETE', n=n, cells=len(cells), elapsed_seconds=time.monotonic()-start,
        source_nested_directory=str(nested.resolve()), npz_path=str(npz_path.resolve()),
        recipe='saved complete STATE50 means + LOCAL_SCALE + AMPLITUDE_TOTAL + AMP_EMP_LOCAL',
        mean_retrained=False, mean_predictions_changed=False, query_outcomes_used_for_distribution=False,
        calibration_groups=[r['calibration_group_count'] for r in reports],
        source_mean_training_size_difference_retained=True,
        residual_interpretation='predictive error including conditional bias, not identified physical noise',
        biological_reference_rule='same cell DIST_FIT only; use cell-specific records for reference error summaries')
    write_json(root/'summary.json', summary)
    return dict(**result, summary=summary, cells=reports, npz_path=str(npz_path.resolve()))
