"""Aggregate completed RxRx3 R3 caches without fitting or drawing outcomes.

Condition-weighted scores are primary. Chemical identities stay together across
doses in paired resampling; experiment/layout blocks are a separate sensitivity
analysis. Partial reports are explicitly opt-in and never declare completion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.biology_kernel_evaluation import write_json
from opal2.eu_core_experiment import select, policy_summary
from opal2.empirical_radial_experiment import LEVELS
from scripts.run_r3_eu_modules_20260918 import ARMS

ROOT = PROJECT / 'runs/r3_rxrx3_modules_20260920_v1'
REPORT = PROJECT / 'reports/r3_rxrx3_modules_20260920_v1'
BASELINES = ('DIRECT_ACCESS_MATCHED_HISTGB_COHERENT',
             'DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL',
             'DIRECT_ACCESS_MATCHED_EXTRATREES_COHERENT',
             'DIRECT_ACCESS_MATCHED_EXTRATREES_CLASSIFIER_CAL', 'HR_REF')
PRIMARY = ('CORE', 'BIO_STRUCTURED', 'COND_REP', 'BOTH_STRUCTURED', 'CONDITIONAL')
OFFSETS = (100000, 200000)
IDENTITY = {'ids', 'actual', 'groups', 'layout', 'fold', 'outer_fold', 'dose', 'cell'}
SEED = 20260920


def read_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def _aligned(saved, ids, actual):
    np.testing.assert_array_equal(saved['ids'].astype(str), np.asarray(ids, str))
    if 'actual' in saved:
        np.testing.assert_allclose(saved['actual'], actual, rtol=0, atol=1e-12)


def completed_cells(scope, root, *, allow_partial=False):
    """Resolve every cell through the frozen R2 manifest, never new partitions."""
    data, manifest = scope['data'], scope['manifest']
    ids = np.asarray(data['ids'], str)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate global condition IDs')
    lookup = {identifier: i for i, identifier in enumerate(ids)}
    records = []
    occupied = np.zeros(len(ids), int)
    missing = []
    for index, (part, cell) in enumerate(zip(manifest['parts'], manifest['cells'], strict=True)):
        if int(cell['cell']) != index:
            raise ValueError('Manifest cell order changed')
        folder = Path(root) / f'fold_{index}'
        path = folder / 'complete.json'
        if not path.exists():
            missing.append(index)
            continue
        complete = json.loads(path.read_text())
        if complete.get('complete') is False or int(complete['fold']) != index:
            raise ValueError('Invalid completed cell: ' + str(path))
        if int(complete['outer_fold']) != int(cell['outer_fold']):
            raise ValueError('Outer fold differs from R2')
        dose = float(complete['dose_um'])
        if dose != float(cell['dose_uM']):
            raise ValueError('Dose differs from R2')
        rows = np.asarray([lookup[str(identifier)] for identifier in part['DEV_EVAL']], int)
        if len(set(rows)) != len(rows) or not len(rows):
            raise ValueError('Invalid cell query IDs')
        if not np.all(np.asarray(data['dose'])[rows] == dose):
            raise ValueError('Query conditions cross the declared dose')
        if 'query_ids' in complete:
            np.testing.assert_array_equal(complete['query_ids'], ids[rows])
        occupied[rows] += 1
        records.append(dict(index=index, rows=rows, folder=folder, complete=complete,
                            outer_fold=int(cell['outer_fold']), dose_um=dose))
    if not records or np.any(occupied > 1):
        raise ValueError('No completed cells or duplicated outer QUERY predictions')
    if missing and not allow_partial:
        raise RuntimeError(f'R3 cells incomplete: {missing}; no fitting or filling performed')
    if not missing and not np.all(occupied == 1):
        raise ValueError('Complete cells do not cover the original population')
    return records, np.flatnonzero(occupied), missing


def collect_arm(records, rows, data, actual, arm, *, offset=None):
    """Strict same-schema collection in original global condition order."""
    positions = {int(row): i for i, row in enumerate(rows)}
    result, fields = {}, None
    for cell in records:
        name = arm + (f'_mc{offset}' if offset is not None else '') + '.npz'
        saved = read_npz(cell['folder'] / name)
        take = cell['rows']
        _aligned(saved, data['ids'][take], actual[take])
        keys = {key for key, value in saved.items()
                if key not in IDENTITY and value.shape[:1] == (len(take),)}
        required = {'predicted', 'p_null'}
        if offset is None:
            required |= {'crps', 'nll', 'energy', 'mean_u', 'actual_u',
                         'increment', 'resource_support',
                         'selected_lambda_0.2', 'selected_lambda_0'}
        if not required <= keys:
            raise ValueError(f'Missing per-object fields in {name}: {required-keys}')
        if fields is not None and keys != fields:
            raise ValueError('Per-cell field schema differs for ' + name)
        fields = keys
        local = np.asarray([positions[int(row)] for row in take])
        for key in sorted(keys):
            value = saved[key]
            if value.dtype.kind in 'fci' and not np.isfinite(value).all():
                raise ValueError(f'Nonfinite {name}/{key}')
            if key not in result:
                result[key] = np.empty((len(rows), *value.shape[1:]), dtype=value.dtype)
            if result[key].shape[1:] != value.shape[1:]:
                raise ValueError('Per-object field dimensions changed')
            result[key][local] = value
    return result


def summarize(out, actual, ids, cells, *, verify_saved=True):
    """Original per-cell budgets; no pooled cross-dose or outer-fold reranking."""
    actual, ids, cells = np.asarray(actual), np.asarray(ids), np.asarray(cells)
    predicted, probability = np.asarray(out['predicted']), np.asarray(out['p_null'])
    if (predicted.shape != actual.shape or probability.shape != actual.shape
            or not np.isfinite(predicted).all() or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Invalid aligned predictions')
    null = actual <= 0
    rho = None if np.ptp(predicted) == 0 or np.ptp(actual) == 0 else float(spearmanr(actual, predicted).statistic)
    result = dict(n=len(actual), gamma_mean=float(predicted.mean()),
        gamma_mse=float(np.square(actual-predicted).mean()), gamma_spearman=rho,
        brier=float(np.square(probability-null).mean()),
        null_auc=float(roc_auc_score(null, probability)) if len(np.unique(null)) == 2 else None,
        policies={})
    for key in ('crps', 'nll', 'energy', 'single_crps', 'pair_crps', 'average_crps',
                'triple_average_crps', 'absolute_pair_crps'):
        if key in out:
            result[key] = float(out[key].mean())
    if 'mean_u' in out:
        result['geometry_mse'] = float(np.square(out['mean_u']-out['actual_u']).mean())
    for key in ('joint_coverage_by_level', 'gamma_coverage_by_level'):
        if key in out:
            result[key] = out[key].mean(0)
    if 'joint_coverage_by_level' in out:
        result['joint_mean_absolute_coverage_error_pp'] = float(
            100*np.abs(out['joint_coverage_by_level'].mean(0)-LEVELS).mean())
    for lam in (.2, 0.):
        key = f'selected_lambda_{lam:g}'
        chosen = np.zeros(len(actual), bool)
        for cell in np.unique(cells):
            ix = np.flatnonzero(cells == cell)
            chosen[ix] = select(ids[ix], predicted[ix], probability[ix], lam)
        if verify_saved and key in out:
            if out[key].dtype != bool:
                raise ValueError('Saved selection must be Boolean')
            np.testing.assert_array_equal(out[key], chosen)
        out[key] = chosen
        result['policies'][f'lambda_{lam:g}'] = dict(policy_summary(actual, chosen),
            predicted_null_count=float(probability[chosen].sum()),
            null_count_gap=float((null[chosen]-probability[chosen]).sum()),
            selected_ids=ids[chosen])
    return result


def scalar_values(out, actual):
    result = {key: out[key] for key in ('crps', 'nll', 'energy') if key in out}
    result.update(gamma_mse=np.square(out['predicted']-actual),
                  brier=np.square(out['p_null']-(actual <= 0)),
                  policy_value=actual*out['selected_lambda_0.2'],
                  policy_null=((actual <= 0)&out['selected_lambda_0.2']).astype(float))
    return result


def _cluster_differences(differences, labels, *, replicates=2000):
    """Same group bootstrap as EU, batched across all named contrasts."""
    labels = np.asarray(labels)
    groups, inverse = np.unique(labels, return_inverse=True)
    values = np.column_stack(list(differences.values()))
    sums = np.zeros((len(groups), values.shape[1]))
    np.add.at(sums, inverse, values)
    sizes = np.bincount(inverse)
    if len(groups) < 2:
        return {key: dict(difference=float(values[:, i].mean()), ci95=None,
                          blocks=len(groups), few_blocks=True)
                for i, key in enumerate(differences)}
    rng = np.random.default_rng(SEED)
    draws = np.empty((replicates, values.shape[1]))
    for start in range(0, replicates, 64):
        stop = min(start+64, replicates)
        indices = rng.integers(len(groups), size=(stop-start, len(groups)))
        weights = np.zeros((stop-start, len(groups)))
        np.add.at(weights, (np.arange(stop-start)[:, None], indices), 1.)
        draws[start:stop] = (weights@sums)/(weights@sizes)[:, None]
    return {key: dict(difference=float(values[:, i].mean()),
                      ci95=np.quantile(draws[:, i], [.025, .975]).tolist(),
                      blocks=len(groups), few_blocks=len(groups) < 10)
            for i, key in enumerate(differences)}


def paired_comparisons(stores, pairs, actual, groups, layout, support, *, replicates=2000):
    values = {name: scalar_values(out, actual) for name, out in stores.items()}
    result = {a+' minus '+b: {} for a, b in pairs}
    for scope, mask in (('all', np.ones(len(actual), bool)), ('supported', support)):
        if not mask.any():
            for record in result.values():
                record[scope] = None
            continue
        differences = {}
        for a, b in pairs:
            for metric in values[a].keys() & values[b].keys():
                if metric.startswith('policy_') and scope == 'supported':
                    continue
                differences[(a+' minus '+b, metric)] = (values[a][metric]-values[b][metric])[mask]
        for block, labels in (('chemical_group', groups), ('layout', layout)):
            computed = _cluster_differences(differences, labels[mask], replicates=replicates)
            for (comparison, metric), record in computed.items():
                destination = 'policy' if metric.startswith('policy_') else scope
                result[comparison].setdefault(destination, {}).setdefault(metric, {})[block] = record
    return result


def comparison_pairs():
    pairs = [(arm, 'CORE') for arm in ARMS if arm != 'CORE']
    attribution = [('BIO_STRUCTURED', 'BIO_GENERIC'), ('BIO_STRUCTURED', 'BIO_RANDOM'),
        ('BIO_STRUCTURED', 'GELU'), ('COND_REP', 'PCA_REP'), ('COND_REP', 'DIRECT_REP'),
        ('COND_REP', 'GELU'), ('COND_REP', 'DESCRIPTORS_HGB'),
        ('BOTH_STRUCTURED', 'BIO_STRUCTURED'), ('BOTH_STRUCTURED', 'COND_REP'),
        ('BOTH_STRUCTURED', 'BOTH_GENERIC'), ('BOTH_STRUCTURED', 'BOTH_RANDOM')]
    pairs += attribution + [(a+'_CAL', b+'_CAL') for a, b in attribution]
    return list(dict.fromkeys(pairs))


def equal_chemical_scores(out, actual, groups):
    _, inverse = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inverse)
    return {key: float(np.mean(np.bincount(inverse, weights=value)/sizes))
            for key, value in scalar_values(out, actual).items()}


def aggregate(scope, *, root=ROOT, report=REPORT, allow_partial=False, replicates=2000):
    """Aggregate saved completed cells only; no model or sampling API is called."""
    started = time.perf_counter()
    root, report = Path(root), Path(report)
    data, core = scope['data'], scope['core']
    _aligned(core, data['ids'], core['actual'])
    records, rows, missing = completed_cells(scope, root, allow_partial=allow_partial)
    actual_all = np.asarray(core['actual'])
    ids, groups, layout, dose = (np.asarray(data[key])[rows] for key in ('ids', 'groups', 'layout', 'dose'))
    actual = actual_all[rows]
    cell_by_global = np.full(len(data['ids']), -1, int)
    outer_by_global = np.full(len(data['ids']), -1, int)
    for cell in records:
        cell_by_global[cell['rows']] = cell['index']
        outer_by_global[cell['rows']] = cell['outer_fold']
    cells, outer = cell_by_global[rows], outer_by_global[rows]
    stores, metrics = {}, {}
    for arm in ARMS:
        out = collect_arm(records, rows, data, actual_all, arm)
        np.testing.assert_array_equal(out['mean_u'], core['mean_u'][rows])
        if arm == 'CORE':
            for key in ('predicted', 'p_null', 'crps', 'nll', 'energy', 'scatter_u'):
                if key in core and key in out:
                    np.testing.assert_array_equal(out[key], core[key][rows])
        metrics[arm] = summarize(out, actual, ids, cells)
        stores[arm] = out
    support = np.asarray(stores['CORE']['resource_support'], bool)
    for arm, out in stores.items():
        np.testing.assert_array_equal(np.asarray(out['resource_support'], bool), support)
        metrics[arm].update(supported_n=int(support.sum()), mean_unchanged=True,
            active_n=int(np.any(out['increment'] != 0, axis=1).sum()),
            increment_rms=float(np.sqrt(np.square(out['increment']).mean())),
            supported_scores={key: float(value[support].mean()) if support.any() else None
                              for key, value in scalar_values(out, actual).items()
                              if not key.startswith('policy_')})
    baselines, baseline_metrics = {}, {}
    source = Path(scope['source'])
    for name in BASELINES:
        saved = read_npz(source/(name+'.npz'))
        _aligned(saved, data['ids'], actual_all)
        out = {key: value[rows].copy() for key, value in saved.items()
               if key not in IDENTITY and value.shape[:1] == (len(actual_all),)}
        baseline_metrics[name] = summarize(out, actual, ids, cells)
        baselines[name] = out
    combined = dict(stores, **baselines)
    pairs = comparison_pairs()
    strong_pairs = [(arm, baseline) for arm in PRIMARY for baseline in BASELINES]
    paired = paired_comparisons(stores, pairs, actual, groups, layout, support, replicates=replicates)
    strong = paired_comparisons(combined, strong_pairs, actual, groups, layout, support, replicates=replicates)
    by_dose = {}
    for level in np.unique(dose):
        take = dose == level
        subset = {arm: {key: value[take].copy() for key, value in out.items()}
                  for arm, out in combined.items()}
        by_dose[str(float(level))] = dict(n_conditions=int(take.sum()),
            n_chemical_groups=len(set(groups[take])), deployment_cells=np.unique(cells[take]),
            supported_n=int(support[take].sum()),
            metrics={arm: summarize(out, actual[take], ids[take], cells[take])
                     for arm, out in subset.items()},
            paired=paired_comparisons(subset, pairs+strong_pairs, actual[take], groups[take],
                                      layout[take], support[take], replicates=replicates),
            scope='dose-stratified development diagnostics; no dose winner selected')
    mc = {}
    for arm in ARMS:
        mc[arm] = []
        for offset in OFFSETS:
            moment = collect_arm(records, rows, data, actual_all, arm, offset=offset)
            summary = summarize(moment, actual, ids, cells)
            for lam in (.2, 0.):
                key = f'lambda_{lam:g}'
                summary['policies'][key]['list_symmetric_difference'] = int(np.sum(
                    moment['selected_'+key] != stores[arm]['selected_'+key]))
            mc[arm].append(dict(seed_offset=offset, policies=summary['policies']))
            np.savez_compressed(root/f'{arm}_mc{offset}.npz', ids=ids, groups=groups,
                layout=layout, fold=cells, outer_fold=outer, dose=dose, actual=actual, **moment)
    deployment = []
    for cell in records:
        take = cells == cell['index']
        costs = cell['complete']['costs']
        action = {}
        for arm, out in stores.items():
            net = float(np.sum(actual[take]*out['selected_lambda_0.2'][take]))
            purchased = int(costs['reference_if_all_new_wells'])
            after_x = int(costs['reference_if_X_already_available'])
            action[arm] = dict(existing_reference_action_net_value=net,
                new_reference_net_value=net-.01*purchased,
                reference_X_available_net_value=net-.01*after_x,
                extra_reference_wells_beyond_CORE=0, training_compute_is_additional=True,
                amortization_queries_if_value_persists=None if net <= 0 else .01*purchased*int(take.sum())/net)
        deployment.append(dict(fold=cell['index'], outer_fold=cell['outer_fold'],
            dose_um=cell['dose_um'], query_n=int(take.sum()), counts=costs, arms=action))
    payload = dict(state='PARTIAL' if missing else 'COMPLETE', complete=not missing,
        n=len(ids), n_chemical_groups=len(set(groups)), completed_cells=len(records),
        expected_cells=len(scope['manifest']['cells']), missing_cells=missing,
        primary_unit='compound-dose condition', independent_identity_unit='chemical connectivity',
        metrics=metrics, paired=paired, r2_comparisons=strong, r2_baseline_metrics=baseline_metrics,
        by_dose=by_dose, supported_n=int(support.sum()),
        equal_chemical_weight_descriptive_scores={arm: equal_chemical_scores(out, actual, groups)
                                                 for arm, out in combined.items()},
        monte_carlo_sensitivity=mc, deployment_costs=deployment,
        cells=[cell['complete'] for cell in records], core_retrained=False, samples=100000,
        query_endpoints_changed=False, original_image_representation_tested=False,
        confirmation_opened=False, new_model_fits=0, new_monte_carlo_draws=0,
        evidence='RxRx3 opened well-profile development; fixed-model, fixed-selection paired diagnostics',
        interval_scope='chemical-group resampling keeps doses together; separate experiment/layout sensitivity; not a combined dependence guarantee',
        multiple_comparisons_adjusted=False, independent_confirmation=False,
        reference_cost_aggregation='per deployment cell scenarios; CV resource counts are not added as one purchase',
        scalar_baseline_semantics='CLASSIFIER_CAL p_NULL is separate from the regression-residual Gamma CRPS law',
        aggregate_wall_seconds=time.perf_counter()-started)
    for arm, out in stores.items():
        np.savez_compressed(root/(arm+'.npz'), ids=ids, groups=groups, layout=layout,
            fold=cells, outer_fold=outer, dose=dose, actual=actual, **out)
    report.mkdir(parents=True, exist_ok=True)
    write_json(root/'summary.json', payload)
    write_json(report/'summary.json', payload)
    render_report(payload, root, report)
    return payload


def render_report(payload, root, report):
    lines = ['# RxRx3 R3 well-profile modules: '+payload['state'], '',
        f"{payload['n']:,} compound-dose conditions, {payload['n_chemical_groups']:,} chemical groups; "
        f"{payload['completed_cells']}/{payload['expected_cells']} completed fold-dose cells. "
        'The condition count is not an independent compound count.', '',
        '| Arm | Gamma CRPS | NULL Brier | NULL / selected | Selected mean Gamma |',
        '|---|---:|---:|---:|---:|']
    for arm, value in payload['metrics'].items():
        policy = value['policies']['lambda_0.2']
        mean = policy['selected_mean_value']
        lines.append(f"| {arm} | {value['crps']:.8f} | {value['brier']:.8f} | "
                     f"{policy['null_selected']} / {policy['activated']} | "
                     + ('NA' if mean is None else f'{mean:.8f}')+' |')
    lines += ['', 'All 24 arms, attribution contrasts, strong saved R2 interfaces, equal-chemical scores, '
        'all available dose strata, and both saved integration seeds are retained in summary.json. '
        'No arm or dose is selected using these query results.', '',
        'Decisions retain each original fold-dose budget, endpoint and action cost. CORE means and '
        'original CORE predictions are checked against R2. Additional reference purchase scenarios '
        'are reported per deployment cell, not summed across cross-validation.', '',
        'Paired intervals condition on fitted models and selected lists. Chemical groups are resampled '
        'jointly across doses; recorded layouts are a separate few-block sensitivity analysis. '
        'These unadjusted development comparisons are not independent confirmation or finite-sample guarantees.', '',
        'This run contains well-profile representations only, not raw-image encoders, cross-dose '
        'acquisition actions or a confirmed transferable activation rule. Classifier-CAL probabilities '
        'are distinguished from the separate scalar regression-residual Gamma distribution.', '',
        f"Aggregation wall time: {payload['aggregate_wall_seconds']:.2f} seconds; no fits or new Monte Carlo draws."]
    if not payload['complete']:
        lines[2:2] = ['INCOMPLETE DEVELOPMENT SNAPSHOT. Missing cells: '+str(payload['missing_cells']), '']
    text = '\n'.join(lines)+'\n'
    (Path(root)/'REPORT.md').write_text(text)
    (Path(report)/'REPORT.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    from opal2.rxrx3_r3_cache import load_scope
    with threadpool_limits(limits=1):
        result = aggregate(load_scope(), allow_partial=args.allow_partial)
    print(json.dumps({key: result[key] for key in ('state', 'n', 'n_chemical_groups', 'completed_cells')}))


if __name__ == '__main__':
    main()
