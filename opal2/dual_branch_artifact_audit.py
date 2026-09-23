"""Read-only replay of the saved dual-branch development experiment.

No training or Monte Carlo sampling is performed. A full audit requires a
completed run; checkpoints-only mode checks finished model fits without
examining the incomplete experiment's query performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz, scalar_metrics
from .dual_branch_biology import BOUND, DualBranchBiologyAdapter
from .dual_branch_features import biology_features, apply_increment, select_strength
from .empirical_radial import (fit_radial, reference_weights, variance_multiplier,
                              radial_nll, radial_cdf, radial_ppf)
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .lincs_biology_experiment import load_data
from .radial_mixture_evaluation import LEVELS
from .joint_contrast_scale import contrast_projector, projected_energy


FULL_ARMS = ('GELU', 'DUAL_GENERIC', 'DUAL_STRUCTURED')
ARMS = ('CORE', *FULL_ARMS, *(name+'_CAL' for name in FULL_ARMS))


def exact(left, right, description):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=False):
        raise AssertionError(description)


def close(left, right, description, atol=1e-10, rtol=1e-10):
    np.testing.assert_allclose(left, right, atol=atol, rtol=rtol, err_msg=description)


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, (float, np.floating)) or bool(np.isfinite(value))


def check_left_parameters(left, other):
    keys = [key for key in left if key.startswith('left.') or key in ('empirical_center', 'empirical_scale')]
    if not keys:
        raise AssertionError('Missing left parameters')
    for key in keys:
        if key not in other or not torch.equal(left[key], other[key]):
            raise AssertionError('Frozen shared left parameter changed: '+key)
    return len(keys)


def checkpoint_audit(root):
    """Only model structure, fit population and numerical training diagnostics."""
    root = Path(root)
    records = []
    for fold in range(5):
        folder = root/f'fold_{fold}'
        if not (folder/'fit_complete.json').exists():
            continue
        declared = read_json(folder/'fit_complete.json')
        prediction = read_npz(folder/'predictions.npz')
        models = {name: torch.load(folder/(name+'.pt'), weights_only=True, map_location='cpu')
                  for name in FULL_ARMS}
        training = {}
        for name, saved in models.items():
            report = read_json(folder/(name+'_training.json'))
            if report != saved['report'] or not finite_tree(report):
                raise AssertionError('Saved training report differs or is nonfinite')
            if report['epochs'] != 60 or report['checkpoint_selection']:
                raise AssertionError('The declared fixed-60-epoch training changed')
            exact(report['supplied_ids'], prediction['fit_ids'], 'Fitting population changed')
            exact(report['fitting_ids'], prediction['fit_ids'][prediction['fit_support']], 'Supported fitting population changed')
            left = prediction[name+'_left']
            right = prediction[name+'_right']
            exact(left+right, prediction[name], 'Saved branch sum differs')
            exact(prediction[name][~prediction['support']], np.zeros((int((~prediction['support']).sum()), 2)),
                  'Unsupported adapter does not return zero')
            if np.max(np.abs(left)) > BOUND+1e-12 or np.max(np.abs(right)) > BOUND+1e-12:
                raise AssertionError('A branch exceeds the declared log-scale bound')
            if name != 'GELU':
                check_left_parameters(models['GELU']['state_dict'], saved['state_dict'])
                exact(left, prediction['GELU'], 'Left outputs are not identical')
                count = sum(v.numel() for key, v in saved['state_dict'].items() if key.startswith('right.'))
                if count != 146 or report['right_trainable_parameters'] != 146:
                    raise AssertionError('Right-branch capacity differs from the declared 146 parameters')
            history = report['history']
            training[name] = dict(active_parameters=report['active_parameters'],
                epochs=[r['epoch'] for r in history],
                max_recorded_preclip_gradient_norm=max(r['preclip_gradient_norm_max'] for r in history),
                mean_recorded_clipped_step_fraction=float(np.mean([r['clipped_step_fraction'] for r in history])),
                first_projection_nll=history[0]['training_projection_nll'],
                last_projection_nll=history[-1]['training_projection_nll'],
                final_gradient_norm=history[-1]['preclip_gradient_norm_mean'],
                left_rms=history[-1]['left_increment_rms'], right_rms=history[-1]['right_increment_rms'])
        support_detail = {}
        names = declared['biological_names']
        for relation in ('target', 'moa'):
            detail = {}
            for split, values in (('fit', prediction['fit_features']), ('test', prediction['test_features'])):
                mask = values[:, names.index(relation+'_available')] > 0
                detail[split] = dict(n=len(values), supported=int(mask.sum()))
                for field in ('log_count', 'log_mass', 'log_ess'):
                    column = np.expm1(values[mask, names.index(relation+'_'+field)])
                    detail[split][field.removeprefix('log_')] = describe(column) if len(column) else None
            support_detail[relation] = detail
        records.append(dict(fold=fold, fit_n=declared['fit_n'], test_n=declared['test_n'],
            fit_supported=declared['fit_supported'], test_supported=declared['test_supported'],
            fit_target_supported=declared['training_queries_with_target_support'],
            fit_moa_supported=declared['training_queries_with_moa_support'],
            descriptor_count=len(declared['descriptor_names']),
            biological_descriptor_count=len(declared['biological_names']),
            identical_left_parameters=True, right_parameters_each=146,
            unsupported_zero=True, training=training, reference_support=support_detail))
    return dict(state='CHECKPOINTS_ONLY', completed_folds=len(records), folds=records,
                query_performance_inspected=False, training_run=False, monte_carlo_run=False)


def replay_calibration_covariance(scatter, residual, amplitude, groups, bandwidth):
    """Independent replay using only this cell's group-excluded residual pool."""
    scatter, residual = np.asarray(scatter), np.asarray(residual)
    chol = np.linalg.cholesky(scatter)
    radius = np.sqrt(np.sum(np.linalg.solve(chol, residual[..., None])[..., 0]**2, axis=1))
    answer = np.empty_like(scatter)
    for row, group in enumerate(groups):
        keep = np.asarray(groups) != group
        if len(np.unique(np.asarray(groups)[keep])) < 3:
            raise ValueError('Calibration has fewer than three other groups')
        local = reference_weights(np.asarray(amplitude)[keep], np.asarray(amplitude)[row:row+1], bandwidth, conditional=True)['weights']
        answer[row] = scatter[row]*variance_multiplier(fit_radial(radius[keep]), local)[0]
    return answer


def describe(values):
    a = np.asarray(values, float)
    return dict(mean=float(np.mean(a)), minimum=float(np.min(a)), maximum=float(np.max(a)),
                quantiles=np.quantile(a, [.05, .25, .5, .75, .95]).tolist())


def audit_nested_reference_mask(folder, expected_ids, id_to_group):
    nested = read_npz(folder/'nested_distribution/distribution.npz')
    exact(nested['ids'], expected_ids, 'Nested complete-CORE IDs differ')
    summary = read_json(folder/'nested_distribution/summary.json')
    if summary['state'] != 'COMPLETE' or summary['query_outcomes_used_for_distribution']:
        raise AssertionError('Nested distribution is not complete and outcome-isolated')
    ids = nested['ids']; lookup = {v: i for i, v in enumerate(ids)}
    permitted = np.zeros((len(ids), len(ids)), bool)
    seen = np.zeros(len(ids), int)
    for path in sorted((folder/'nested_distribution').glob('cell_*.json')):
        cell = read_json(path)
        queries = np.array([lookup[v] for v in cell['query_ids']])
        refs = np.array([lookup[v] for v in cell['fit_ids']])
        # The saved mask marks the same-inner-cell DIST_FIT pool. Chemical
        # identity is also excluded by the feature function before retrieval.
        permitted[np.ix_(queries, refs)] = True
        seen[queries] += 1
        if set(cell['query_ids']) & set(cell['fit_ids']):
            raise AssertionError('A nested query entered its reference pool')
        query_groups = {id_to_group[v] for v in cell['query_ids']}
        for role in ('mean_fit_ids', 'mean_validation_ids', 'mean_reference_ids', 'fit_ids', 'cal_ids'):
            if query_groups & {id_to_group[v] for v in cell[role]}:
                raise AssertionError('Nested query chemical group leaked into '+role)
    exact(seen, np.ones(len(ids), int), 'Nested targets are not each held out once')
    exact(nested['biological_reference_mask'], permitted, 'Nested donor permission differs from same-cell DIST_FIT')
    np.linalg.cholesky(nested['raw_covariance'])
    return nested


def full_audit(root):
    root = Path(root).resolve()
    if read_json(root/'status.json').get('state') != 'COMPLETE' or read_json(root/'summary.json').get('state') != 'COMPLETE':
        raise RuntimeError('Full artifact audit requires a completed run')
    audit = checkpoint_audit(root)
    if audit['completed_folds'] != 5:
        raise AssertionError('Not all five model fits are complete')
    spec = read_json(root/'run_spec.json')
    if spec['arms'] != list(ARMS) or spec['samples'] != 100000 or spec['epochs'] != 60:
        raise AssertionError('Declared experiment configuration changed')
    source, radial = Path(spec['source']), Path(spec['radial'])
    for name in ('dual_branch_biology.py', 'dual_branch_features.py'):
        if (root/name).read_bytes() != (Path(__file__).parent/name).read_bytes():
            raise AssertionError('Live implementation differs from saved run snapshot: '+name)
    runner_matches_snapshot = ((root/'dual_branch_experiment.py').read_bytes()
                              == (Path(__file__).parent/'dual_branch_experiment.py').read_bytes())
    radial_summary = read_json(radial/'summary.json')
    manifest = read_json(Path(radial_summary['reference_run'])/'run_manifest.json')
    data, metadata = load_data(radial_summary['data_directory'])
    ids, groups = data['ids'], data['groups']; n = len(ids)
    if n != 1188 or len(set(ids)) != n:
        raise AssertionError('Expected exactly 1188 unique opened development objects')
    exact(ids, manifest['ids'], 'Population differs from original folds')
    source_core, prior = read_npz(source/'CORE.npz'), read_npz(radial/'AMP_EMP_LOCAL.npz')
    stores = {arm: read_npz(root/(arm+'.npz')) for arm in ARMS}
    original = stores['CORE']
    artifact_issues = []
    incomplete_fields = {}
    for name in ARMS:
        paths = sorted(root.glob('cell_*'))
        fields = [set(read_npz(path/(name+'.npz'))) for path in paths]
        incomplete_fields[name] = sorted((set(stores[name])-set.intersection(*fields))
            - {'groups', 'layout', 'policy_value', 'policy_null', 'brier'})
        if incomplete_fields[name]:
            artifact_issues.append(dict(arm=name, issue='Aggregate fields not populated by every cell',
                                        fields=incomplete_fields[name]))
    for name, out in stores.items():
        exact(out['ids'], ids, 'Aggregate IDs differ: '+name)
        exact(out['groups'], groups, 'Aggregate chemical groups differ: '+name)
        exact(out['mean_u'], prior['mean_u'], 'Mean changed: '+name)
        exact(out['actual_u'], prior['actual_u'], 'Geometry endpoint changed: '+name)
        exact(out['actual'], original['actual'], 'Gamma endpoint changed: '+name)
        np.linalg.cholesky(out['covariance_u'])
        for key, value in out.items():
            if value.dtype.kind in 'fci' and not np.isfinite(value).all():
                if key not in incomplete_fields[name]:
                    raise AssertionError('Nonfinite saved artifact: '+name+'/'+key)
        exact(out['brier'], (out['p_null']-(out['actual'] <= 0))**2, 'Brier arithmetic differs')
    for key in (set(source_core) & set(original))-{'resource_support', 'increment'}:
        exact(original[key], source_core[key], 'CORE source field changed: '+key)
    exact(original['mean_u'], prior['mean_u'], 'Prior CORE means differ')
    lookup = {v: i for i, v in enumerate(ids)}
    id_to_group = dict(zip(ids, groups))
    records = {row['fold']: row for row in manifest['folds']}
    amplitude = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    descriptors = {}; nested = {}; predictions = {}; seen = np.zeros(n, int)
    checks = []; calibration_choices = []
    for cell in radial_summary['cells']:
        fold, half = cell['fold'], cell['half']; folder = root/f'cell_{fold}_{half}'
        q = np.array([lookup[v] for v in cell['query_ids']]); cal = np.array([lookup[v] for v in cell['representative_ids']])
        fit = np.asarray(records[fold]['fit']); test = np.asarray(records[fold]['test'])
        if set(groups[q]) & set(groups[np.r_[fit, cal]]) or set(groups[fit]) & set(groups[cal]):
            raise AssertionError('Outer chemical groups cross fit/cal/query roles')
        stats = read_json(Path(radial_summary['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
        raw_mean = prior['mean_u']*scale+center
        if fold not in descriptors:
            transformer = joblib.load(source/f'fold_{fold}/conditioners.joblib')['transformer']
            descriptors[fold] = transformer.transform(data, metadata).values
            predictions[fold] = read_npz(root/f'fold_{fold}/predictions.npz')
            exact(predictions[fold]['fit_ids'], ids[fit], 'Outer fitting IDs differ')
            exact(predictions[fold]['test_ids'], ids[test], 'Outer test IDs differ')
            nested[fold] = audit_nested_reference_mask(root/f'fold_{fold}', ids[fit], id_to_group)
            train_bio = biology_features(data, metadata, fit, fit, nested[fold]['raw_mean'],
                nested[fold]['raw_covariance'], nested[fold]['raw_residual'],
                allowed=nested[fold]['biological_reference_mask'])
            exact(train_bio['support'], predictions[fold]['fit_support'], 'Training support differs')
            close(train_bio['values'], predictions[fold]['fit_features'], 'Training reference summaries differ')
            energy = projected_energy(nested[fold]['raw_residual'], contrast_projector(
                nested[fold]['raw_mean'], np.ones(9), nested[fold]['raw_covariance']))
            close(energy, predictions[fold]['fit_energies'], 'Training geometric energies differ')
            test_bio = biology_features(data, metadata, test, fit, raw_mean[test],
                prior['covariance_u'][test]*scale[None, :, None]*scale[None, None, :], nested[fold]['raw_residual'])
            exact(test_bio['support'], predictions[fold]['support'][test], 'Outer query support differs')
            close(test_bio['values'], predictions[fold]['test_features'], 'Outer reference features differ')
            for arm in FULL_ARMS:
                model = DualBranchBiologyAdapter.load(root/f'fold_{fold}'/(arm+'.pt'))
                empirical = descriptors[fold][test]
                biological = None if arm == 'GELU' else test_bio['values']
                components = model.predict_components(empirical, biological, test_bio['support'])
                for component in ('left', 'right'):
                    exact(components[component], predictions[fold][arm+'_'+component][test], 'Checkpoint output differs')
                exact(components['total'], predictions[fold][arm][test], 'Checkpoint total differs')
                exact(model.predict_increment(empirical, biological, test_bio['support'], enabled=False),
                      np.zeros((len(test), 2)), 'Global disabled adapter differs')
                if arm != 'GELU':
                    exact(model.predict_increment(empirical, biological, test_bio['support'], right_enabled=False),
                          predictions[fold]['GELU'][test], 'Disabled biology does not return identical GELU')
        ref = read_npz(radial/f'cell_{fold}_{half}_radial.npz')
        saved_cal = read_npz(folder/'honest_calibration_inputs.npz')
        exact(saved_cal['ids'], ids[cal], 'CAL pool differs')
        exact(ref['query_ids'], ids[q], 'Query order differs')
        exact(ref['cal_ids'], ids[cal], 'Radial reference order differs')
        exact(saved_cal['scatter'], ref['cal_amp_scatter'], 'CAL inherited wrong scatter')
        cal_cov = replay_calibration_covariance(ref['cal_amp_scatter'], ref['cal_residual'],
            amplitude[cal], groups[cal], cell['local_bandwidth'])
        close(cal_cov, saved_cal['covariance'], 'CAL covariance does not replay from current-cell pool only')
        cal_bio = biology_features(data, metadata, cal, fit, raw_mean[cal],
            cal_cov*scale[None, :, None]*scale[None, None, :], nested[fold]['raw_residual'])
        close(cal_bio['values'], saved_cal['biological_features'], 'CAL reference inputs differ')
        exact(cal_bio['support'], saved_cal['support'], 'CAL support differs')
        selections = {}
        for name in FULL_ARMS:
            model = DualBranchBiologyAdapter.load(root/f'fold_{fold}'/(name+'.pt'))
            output = model.predict_increment(descriptors[fold][cal],
                None if name == 'GELU' else cal_bio['values'], cal_bio['support'])
            close(output, saved_cal[name], 'CAL checkpoint increments differ', atol=1e-11, rtol=1e-11)
            choice = select_strength(raw_mean[cal], scale, ref['cal_amp_scatter'], ref['cal_residual'],
                amplitude[cal], groups[cal], cell['local_bandwidth'], saved_cal[name])
            saved_choice = read_json(folder/(name+'_selection.json'))
            if choice['strength'] != saved_choice['strength'] or choice['best_strength'] != saved_choice['best_strength']:
                raise AssertionError('Calibration strength differs from saved current-cell inputs')
            close(choice['scores'], saved_choice['scores'], 'Calibration NLL table differs')
            selections[name] = saved_choice['strength']
        law = fit_radial(ref['amplitude_radii']); weights = ref['local_weights']
        multiplier = variance_multiplier(law, weights)
        threshold = radial_ppf(law, weights, LEVELS)**2
        minimum_eigenvalue = np.inf
        for arm in ARMS:
            out = read_npz(folder/(arm+'.npz'))
            exact(out['ids'], ids[q], 'Saved cell IDs differ')
            for key, value in out.items():
                if key in stores[arm] and value.shape[:1] == (len(q),):
                    exact(value, stores[arm][key][q], 'Cell/aggregate mismatch: '+arm+'/'+key)
            increment = np.zeros((len(q), 2)) if arm == 'CORE' else predictions[fold][arm.removesuffix('_CAL')][q]*(selections[arm.removesuffix('_CAL')] if arm.endswith('_CAL') else 1.)
            exact(out['increment'], increment, 'Applied increment does not match fitted prediction and CAL strength')
            active = np.any(increment != 0, axis=1)
            for key in ('predicted', 'p_null', 'crps', 'nll', 'energy', 'brier', 'covariance_u'):
                exact(out[key][~active], source_core[key][q][~active], 'Inactive row is not exactly CORE: '+key)
            scatter = apply_increment(raw_mean[q], scale, prior['scatter_u'][q], increment)
            minimum_eigenvalue = min(minimum_eigenvalue, float(np.linalg.eigvalsh(scatter).min()))
            close(out['covariance_u'], scatter*multiplier[:, None, None], 'Saved covariance differs from fixed-law scatter')
            close(out['radial_variance_multiplier'], multiplier, 'Radial-law second moment changed')
            close(out['joint_squared_radius_by_level'], threshold, 'Fixed radial quantile law changed')
            residual = prior['actual_u'][q]-prior['mean_u'][q]
            radius = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(scatter), residual[..., None])[..., 0], axis=1)
            close(out['nll'], radial_nll(residual, scatter, law, weights), 'Analytical density differs')
            close(out['radial_pit'], radial_cdf(law, weights, radius), 'Analytical radial CDF differs')
            exact(out['joint_coverage_by_level'], (radius[:, None]**2 <= threshold).astype(float), 'Joint coverage arithmetic differs')
            select = select_frozen_cohort_plan(ids[q], out['predicted'], out['p_null'], cell['budget'])
            exact(out['selected'], select.selected_mask, 'Fixed budget policy was not preserved')
        seen[q] += 1
        checks.append(dict(fold=fold, half=half, query_n=len(q), calibration_n=len(cal),
            supported=int(predictions[fold]['support'][q].sum()),
            min_scatter_eigenvalue=minimum_eigenvalue,
            calibration_current_cell_only=True, fixed_radial_law=True,
            reference_is_MODEL_FIT_honest_errors=True, mean_and_endpoint_unchanged=True))
        calibration_choices.append(dict(fold=fold, half=half, strengths=selections))
    exact(seen, np.ones(n, int), 'Queries are not each scored exactly once')
    if len(checks) != 10:
        raise AssertionError('Expected ten query cells')
    supported = stores['GELU']['resource_support'].astype(bool)
    summary = read_json(root/'summary.json')
    if summary['supported_n'] != int(supported.sum()):
        raise AssertionError('Aggregate support count differs')
    metrics = {}; multiplier_summaries = {}
    for arm, out in stores.items():
        exact(out['resource_support'], supported, 'Arms do not share query support')
        selected = out['selected'].astype(bool)
        row = dict(supported=scalar_metrics(out, out['actual'], supported),
            full=scalar_metrics(out, out['actual'], np.ones(n, bool)),
            selected_n=int(selected.sum()), selected_null=int((out['actual'][selected] <= 0).sum()),
            selected_mean=float(out['actual'][selected].mean()),
            changed_selected_membership=int(np.count_nonzero(selected != original['selected'])))
        declared = summary['metrics'][arm]
        for key in ('selected_n', 'selected_null', 'changed_selected_membership'):
            if row[key] != declared[key]: raise AssertionError('Summary policy count differs')
        close(row['selected_mean'], declared['selected_mean'], 'Summary selected value differs')
        for subset in ('full', 'supported'):
            for score, value in row[subset]['scores'].items():
                close(value, declared[subset]['scores'][score], 'Summary score differs: '+score)
        metrics[arm] = row
        multiplier_summaries[arm] = dict(
            contrast=describe(np.exp(out['increment'][supported, 0])),
            remainder=describe(np.exp(out['increment'][supported, 1])),
            active_n=int(np.any(out['increment'] != 0, axis=1).sum()))
    saturation = {}
    for arm in FULL_ARMS:
        rows = []
        for fold in range(5):
            pred = predictions[fold]; mask = pred['support']
            rows.append(np.stack((pred[arm+'_left'][mask], pred[arm+'_right'][mask]), axis=1))
        values = np.concatenate(rows)
        saturation[arm] = dict(left_fraction_above_95pct_bound=float(np.mean(np.abs(values[:, 0]) >= .95*BOUND)),
            right_fraction_above_95pct_bound=float(np.mean(np.abs(values[:, 1]) >= .95*BOUND)),
            left_max_abs=float(np.max(np.abs(values[:, 0]))), right_max_abs=float(np.max(np.abs(values[:, 1]))))
    audit.update(state='PASS' if not artifact_issues else 'PRIMARY_REPLAY_COMPLETE_AGGREGATE_FIELDS_INVALID',
        artifact_issues=artifact_issues, runner_matches_original_snapshot=runner_matches_snapshot,
        completed_query_cells=10, unique_query_objects=n,
        supported_n=int(supported.sum()), cells=checks, calibration_choices=calibration_choices,
        multiplier_distributions=multiplier_summaries, branch_bound_saturation=saturation,
        reproduced_metrics=metrics, source_core_exact=True, means_exact=True, endpoints_exact=True,
        inactive_scores_exact=True, conditional_covariance_positive_definite=True,
        calibration_inputs_replayed=True, training_run=False, monte_carlo_run=False,
        query_performance_inspected=True,
        interpretation='Opened development comparison; paired intervals condition on the saved folds. No independent certification.',
        copied_nested_cache_note='Nested distribution metadata may retain its original v1 path; IDs and donor masks were replayed from the current saved arrays.')
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoints-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.run).resolve()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        result = checkpoint_audit(root) if args.checkpoints_only else full_audit(root)
    if args.checkpoints_only:
        print(json.dumps(result, indent=2))
    else:
        path = root/'INDEPENDENT_ARTIFACT_AUDIT.json'
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(state=result['state'], query_objects=result['unique_query_objects'],
            supported=result['supported_n'], output=str(path), monte_carlo_run=False)))


if __name__ == '__main__':
    main()
