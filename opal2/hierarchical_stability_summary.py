"""Repeated-partition/seed development summaries with compound-level pairing.

Scores and fixed policy contributions are averaged over runs WITHIN each ID
before bootstrap resampling. Nine predictions of one compound are never nine
independent observations. Seed averaging here is a performance summary, not an
ensemble predictor, and neither checkpoints nor seeds are selected here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np

from .baseline_policy import ACTIONS, NULL_THRESHOLD, POSITIVE_MARGIN
from .biology_kernel_evaluation import write_json
from .hierarchical_geometry_summary import (
    _folds, _load_arm, _action_rows, _policy_rows, _bootstrap_statistic,
    _principal_policy, _fmt,
)


ARMS = ('GLOBAL_GEOMETRY', 'RIDGE_TRAINCV', 'RIDGE_VALID',
        'HR_VALID_S0', 'HR_VALID_S1', 'HR_VALID_S2')
HR_ARMS = ARMS[3:]
OUTCOME_ROUNDOFF_ATOL = 64 * np.finfo(np.float64).eps
SCOPE = (
    'paired bootstrap of unique compound IDs after averaging within ID across '
    'partitions and HR seeds; fixed models and masks; does not include model '
    'search, shared-batch, overlapping-training or Monte Carlo integration '
    'uncertainty; reused development data, not an independent certificate'
)


def _policy_key(row):
    return tuple(row[key] for key in ('action', 'fraction', 'ranking', 'section'))


def _main_index(rows):
    key = ('Z1Z2', .25, 'expected_gain', 'common_budget')
    return next(i for i, row in enumerate(rows) if _policy_key(row) == key)


def _u_predictions(folder, records, ids, arm):
    actual, predicted = np.empty((len(ids), 9)), np.empty((len(ids), 9))
    fold_diagnostics = []
    for record in records:
        ix = np.asarray(record['test'], dtype=int)
        path = folder/'folds'/f"fold_{record['fold']}"/'arms'
        path = path/arm/'test'
        with np.load(path/'u_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Coordinate prediction IDs do not match their test fold')
            a, p = np.asarray(saved['actual_u'], float), np.asarray(saved['mean_u'], float)
            if a.shape != (len(ix), 9) or p.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
                raise ValueError('Finite aligned nine-coordinate predictions are required')
            actual[ix], predicted[ix] = a, p
        diagnostic = path/'u_diagnostics.json'
        fold_diagnostics.append(dict(fold=record['fold'], n=len(ix),
            u_mean_mse=float(np.square(a-p).mean()),
            recorded_diagnostics=json.loads(diagnostic.read_text()) if diagnostic.is_file() else None))
    return actual, predicted, fold_diagnostics


def _read_repeat(root, record, root_manifest, ids, *, output=None):
    repeat = record['repeat']
    folder = root/'repetitions'/f'repeat_{repeat}'
    manifest = json.loads((folder/'run_manifest.json').read_text())
    for flag in ('final_opened','fifth_repeat_opened','original_endpoint_changed','original_contract_changed'):
        if manifest.get(flag) is not False:
            raise ValueError('A repeated partition does not preserve '+flag)
    repeat_ids, allocation = _folds(manifest)
    if not np.array_equal(repeat_ids, ids) or list(manifest['arms']) != list(root_manifest['arms']):
        raise ValueError('Repeated runs must contain the same ordered IDs and model arms')
    planned = {item['fold']: item for item in record['folds']}
    if len(planned) != len(record['folds']) or len(planned) != len(manifest['folds']):
        raise ValueError('The root and repetition fold inventories differ')
    for fold in manifest['folds']:
        if fold['fold'] not in planned or fold['test'] != planned[fold['fold']]['test']:
            raise ValueError('A repeated partition differs from the root manifest')
        completion_path = folder/'folds'/f"fold_{fold['fold']}"/'complete.json'
        if not completion_path.is_file():
            raise ValueError('A repeated fold has no completion marker')
        completion = json.loads(completion_path.read_text())
        if (completion.get('repeat') != repeat or completion.get('fold') != fold['fold']
                or completion.get('test_n') != len(fold['test']) or not completion.get('completed_utc')):
            raise ValueError('A repeated fold completion marker does not match its manifest')
    models, data, actual_reference, u_reference = {}, {}, None, None
    for arm in root_manifest['arms']:
        arrays, fold_rows = _load_arm(folder, manifest, arm, ids)
        actual_u, mean_u, u_rows = _u_predictions(folder, manifest['folds'], ids, arm)
        if actual_reference is None:
            actual_reference, u_reference = arrays['actual'].copy(), actual_u.copy()
        elif (not np.array_equal(actual_reference, arrays['actual'])
              or not np.array_equal(u_reference, actual_u)):
            raise ValueError('Models within a partition use different Gamma or u targets')
        policies, masks = _policy_rows(arrays, ids, allocation, global_model=arm == 'GLOBAL_GEOMETRY')
        actions = _action_rows(arrays, manifest['folds'], global_model=arm == 'GLOBAL_GEOMETRY')
        arrays.update(masks=masks, actual_u=actual_u, mean_u=mean_u,
                      u_object_mse=np.square(actual_u-mean_u).mean(1))
        main = _main_index(policies)
        for fold_row, u_row, fold in zip(fold_rows, u_rows, manifest['folds']):
            ix = np.asarray(fold['test'], dtype=int)
            w, a = masks[main, ix], arrays['actual'][ix, 2]
            k = float(w.sum())
            fold_row.update(u=u_row,
                principal_policy=dict(selected_n=int(round(k)), used_wells=int(round(2*k)),
                    mean_selected_gain=float(w@a/k) if k else None,
                    net_gain_per_eligible=float(w@a/len(ix)),
                    null_count=float(w@(a <= 0)), fdp=float(w@(a <= 0)/k) if k else None,
                    fpr=float(w@(a <= 0)/(a <= 0).sum()) if np.any(a <= 0) else None))
        models[arm] = dict(actions=actions, u_mean_mse=float(arrays['u_object_mse'].mean()),
            u_scope='each test coordinate in its own fit-only fold standardized u; no mixing of native coefficient scales',
            folds=fold_rows, policies=policies,
            common_budget=[row for row in policies if row['section'] == 'common_budget'],
            within_action=[row for row in policies if row['section'] == 'within_action'],
            geometry_energy=float(arrays['geometry_energy'].mean()))
        data[arm] = arrays
        destination = root if output is None else Path(output)
        np.savez_compressed(destination/f'repeat_{repeat}_{arm}_oof_predictions.npz',
                            ids=ids, fold=allocation, **arrays)
    return dict(repeat=repeat, seed=record['seed'], n=len(ids), models=models,
                fold_test_counts=[len(fold['test']) for fold in manifest['folds']]), data, allocation


def _direction(values, *, favorable):
    values = np.asarray(values, dtype=float)
    tolerance = 1e-12
    good = values < -tolerance if favorable == 'negative' else values > tolerance
    bad = values > tolerance if favorable == 'negative' else values < -tolerance
    return dict(n=len(values), favorable=int(good.sum()), unfavorable=int(bad.sum()),
                tied=int((~good & ~bad).sum()), favorable_direction=favorable,
                numerical_zero_tolerance=tolerance)


def _average_comparison(pairs, data, reports, boot, n):
    """Equal-weight performance average over declared pairs, paired by ID."""
    crps, brier, u_mse = [], [], []
    left_crps, right_crps, left_brier, right_brier, left_u, right_u = [], [], [], [], [], []
    for repeat, left, right in pairs:
        a, b = data[repeat][left], data[repeat][right]
        if not np.array_equal(a['actual'], b['actual']):
            raise ValueError('Paired observed utilities differ')
        null = a['actual'] <= 0
        sa, sb = a['utility_crps'], b['utility_crps']
        ba, bb = (a['p_null']-null)**2, (b['p_null']-null)**2
        crps.append(sa-sb); brier.append(ba-bb)
        u_mse.append(a['u_object_mse']-b['u_object_mse'])
        left_crps.append(sa); right_crps.append(sb)
        left_brier.append(ba); right_brier.append(bb)
        left_u.append(a['u_object_mse']); right_u.append(b['u_object_mse'])
    crps, brier, u_mse = np.mean(crps, 0), np.mean(brier, 0), np.mean(u_mse, 0)
    policies, policy_contributions = [], []
    first_repeat, first_left, _ = pairs[0]
    template = reports[first_repeat]['models'][first_left]['policies']
    for j, reference_row in enumerate(template):
        action = ACTIONS.index(reference_row['action'])
        value, false, fdp, fpr = [], [], [], []
        left_value, right_value, left_fdp, right_fdp, left_fpr, right_fpr = [], [], [], [], [], []
        left_count, right_count = [], []
        for repeat, left, right in pairs:
            a, b = data[repeat][left], data[repeat][right]
            ra, rb = reports[repeat]['models'][left]['policies'][j], reports[repeat]['models'][right]['policies'][j]
            if _policy_key(ra) != _policy_key(reference_row) or _policy_key(rb) != _policy_key(reference_row):
                raise ValueError('Policy index is not consistent across repeated runs')
            wa, wb = a['masks'][j], b['masks'][j]
            ka, kb = float(wa.sum()), float(wb.sum())
            outcome = a['actual'][:, action]
            null = outcome <= 0
            nn = int(null.sum())
            value.append((wa-wb)*outcome); false.append((wa-wb)*null)
            fdp.append(n*(wa/ka-wb/kb)*null if ka and kb else np.full(n, np.nan))
            fpr.append(n*(wa-wb)*null/nn if nn else np.full(n, np.nan))
            left_value.append(float(wa@outcome/n)); right_value.append(float(wb@outcome/n))
            left_fdp.append(ra['fdp']); right_fdp.append(rb['fdp'])
            left_fpr.append(ra['fpr']); right_fpr.append(rb['fpr'])
            left_count.append(ra['selected_n']); right_count.append(rb['selected_n'])
        avg_value, avg_false = np.mean(value, 0), np.mean(false, 0)
        avg_fdp, avg_fpr = np.mean(fdp, 0), np.mean(fpr, 0)
        net_stat = _bootstrap_statistic(avg_value, boot)
        fixed_same_count = len(set(left_count+right_count)) == 1 and left_count[0] > 0
        selected_stat = None
        if fixed_same_count:
            multiplier = n/left_count[0]
            selected_stat = dict(mean=net_stat['mean']*multiplier,
                                interval95=[v*multiplier for v in net_stat['interval95']])
        row = dict(**{key: reference_row[key] for key in ('action', 'fraction', 'ranking', 'section')},
            net_gain_per_eligible=net_stat, net_gain_per_selected=selected_stat,
            false_activation_per_eligible=_bootstrap_statistic(avg_false, boot),
            fdp_difference=_bootstrap_statistic(avg_fdp, boot) if np.isfinite(avg_fdp).all() else None,
            fpr_difference=_bootstrap_statistic(avg_fpr, boot) if np.isfinite(avg_fpr).all() else None,
            left_mean_value_per_eligible=float(np.mean(left_value)),
            right_mean_value_per_eligible=float(np.mean(right_value)),
            left_mean_fdp=float(np.mean(left_fdp)) if all(v is not None for v in left_fdp) else None,
            right_mean_fdp=float(np.mean(right_fdp)) if all(v is not None for v in right_fdp) else None,
            left_mean_fpr=float(np.mean(left_fpr)) if all(v is not None for v in left_fpr) else None,
            right_mean_fpr=float(np.mean(right_fpr)) if all(v is not None for v in right_fpr) else None,
            selected_n_per_run_left=left_count, selected_n_per_run_right=right_count,
            activation_counts_pooled_for_certification=False)
        policies.append(row); policy_contributions.append(avg_value)
    result = dict(direction='left minus right', averaging='equal weight over specified runs within each compound ID',
        run_pairs=[dict(repeat=r, left=a, right=b) for r, a, b in pairs],
        unique_compounds=n, scored_run_pairs=len(pairs),
        gamma_crps=_bootstrap_statistic(crps, boot), null_brier=_bootstrap_statistic(brier, boot),
        u_mean_mse=_bootstrap_statistic(u_mse, boot),
        left_mean_gamma_crps=np.mean(left_crps, axis=(0, 1)).tolist(),
        right_mean_gamma_crps=np.mean(right_crps, axis=(0, 1)).tolist(),
        left_mean_null_brier=np.mean(left_brier, axis=(0, 1)).tolist(),
        right_mean_null_brier=np.mean(right_brier, axis=(0, 1)).tolist(),
        left_mean_u_mse=float(np.mean(left_u)), right_mean_u_mse=float(np.mean(right_u)),
        policies=policies, principal_policy=_principal_policy(policies),
        ensemble_prediction=False, best_seed_selected=False, formal_certificate=False)
    contributions = dict(gamma_crps_difference=crps, null_brier_difference=brier,
                         u_mse_difference=u_mse, policy_net_value_difference=np.asarray(policy_contributions))
    return result, contributions


def _direction_reports(repetition_records, reports):
    runs, partitions, folds = [], [], []
    for record in repetition_records:
        repeat = record['repeat']; models = reports[repeat]['models']; base = models['RIDGE_VALID']
        base_policy = _principal_policy(base['policies'])
        local = []
        for h, arm in enumerate(HR_ARMS):
            model = models[arm]; policy = _principal_policy(model['policies'])
            row = dict(repeat=repeat, partition_seed=record['seed'], hr_arm=arm, hr_seed_index=h,
                crps_difference=model['actions'][2]['gamma_crps']-base['actions'][2]['gamma_crps'],
                u_mse_difference=model['u_mean_mse']-base['u_mean_mse'],
                selected_mean_value=policy['per_selected_net_gain'],
                ridge_selected_mean_value=base_policy['per_selected_net_gain'],
                value_per_eligible_difference=policy['per_eligible_net_gain']-base_policy['per_eligible_net_gain'],
                value_per_selected_difference=policy['per_selected_net_gain']-base_policy['per_selected_net_gain'],
                selected_n=policy['selected_n'], used_wells=policy['used_wells'],
                selected_null_count=policy['selected_null_count'], fdp=policy['fdp'], fpr=policy['fpr'],
                fdp_difference=policy['fdp']-base_policy['fdp'])
            runs.append(row); local.append(row)
            base_folds = {f['fold']:f for f in base['folds']}
            planned = {f['fold']:f for f in record['folds']}
            for fold in model['folds']:
                b = base_folds[fold['fold']]; pp, bp = fold['principal_policy'], b['principal_policy']
                folds.append(dict(repeat=repeat, partition_seed=record['seed'], hr_arm=arm,
                    hr_seed_index=h, fold=fold['fold'], n=fold['n'],
                    training_seed=(planned[fold['fold']]['seed']+401+1000*h
                                   if 'seed' in planned[fold['fold']] else None),
                    crps=fold['crps'][2], ridge_crps=b['crps'][2],
                    crps_difference=fold['crps'][2]-b['crps'][2],
                    u_mse_difference=fold['u']['u_mean_mse']-b['u']['u_mean_mse'],
                    value_per_eligible_difference=pp['net_gain_per_eligible']-bp['net_gain_per_eligible'],
                    value_per_selected_difference=pp['mean_selected_gain']-bp['mean_selected_gain'],
                    selected_n=pp['selected_n'], used_wells=pp['used_wells'],
                    selected_mean_value=pp['mean_selected_gain'], ridge_selected_mean_value=bp['mean_selected_gain'],
                    null_count=pp['null_count'], ridge_null_count=bp['null_count'],
                    fdp=pp['fdp'], ridge_fdp=bp['fdp'], fpr=pp['fpr'],
                    fdp_difference=pp['fdp']-bp['fdp']))
        partitions.append(dict(repeat=repeat, partition_seed=record['seed'],
            **{key:float(np.mean([row[key] for row in local])) for key in
               ('crps_difference','u_mse_difference','value_per_eligible_difference',
                'value_per_selected_difference','fdp_difference')}))
    direction = {}
    for name, rows in (('partition_seed_means', partitions), ('individual_HR_runs', runs), ('fold_HR_results', folds)):
        direction[name] = dict(n=len(rows),
            crps=_direction([row['crps_difference'] for row in rows], favorable='negative'),
            u_mse=_direction([row['u_mse_difference'] for row in rows], favorable='negative'),
            value=_direction([row['value_per_eligible_difference'] for row in rows], favorable='positive'),
            fdp=_direction([row['fdp_difference'] for row in rows], favorable='negative'))
    return dict(partitions=partitions, runs=runs, folds=folds, direction_counts=direction,
                count_is_descriptive_not_independent_replicates=True)


def _selection_diagnostics(pairs, data, reports, ids, contribution):
    left_masks, right_masks = [], []
    right_by_repeat = {}
    for repeat, left, right in pairs:
        j = _main_index(reports[repeat]['models'][left]['policies'])
        left_masks.append(data[repeat][left]['masks'][j])
        right_masks.append(data[repeat][right]['masks'][j])
        right_by_repeat[repeat] = data[repeat][right]['masks'][j]
    left_masks, right_masks = np.asarray(left_masks), np.asarray(right_masks)
    if not np.isin(left_masks, [0, 1]).all() or not np.isin(right_masks, [0, 1]).all():
        raise ValueError('HR/RIDGE stability counts require actual selected masks')
    hr_counts = left_masks.sum(0).astype(int)
    ridge_masks = np.asarray(list(right_by_repeat.values()))
    ridge_counts = ridge_masks.sum(0).astype(int)
    overlap = (left_masks*right_masks).sum(1)
    jaccards = [float(np.sum(a*b)/np.sum((a+b)>0))
                for a, b in itertools.combinations(left_masks, 2) if np.any((a+b)>0)]
    delta = np.asarray(contribution)
    positive = np.sort(delta[delta > 0])[::-1]
    net, abs_sum = float(delta.sum()), float(np.abs(delta).sum())
    top = np.argsort(delta)[::-1][:min(5, len(ids))]
    return dict(hr_selection_count_histogram=np.bincount(hr_counts, minlength=len(left_masks)+1).tolist(),
        hr_run_count=len(left_masks), ridge_selection_count_histogram=np.bincount(
            ridge_counts, minlength=len(ridge_masks)+1).tolist(), ridge_partition_count=len(ridge_masks),
        hr_ever_selected=int((hr_counts>0).sum()), hr_selected_in_every_run=int((hr_counts==len(left_masks)).sum()),
        paired_selection_overlap_per_run=overlap.tolist(),
        between_HR_run_selection_jaccard=dict(mean=float(np.mean(jaccards)), min=float(np.min(jaccards)),
                                             max=float(np.max(jaccards))) if jaccards else None,
        net_average_total_gain_difference=net, positive_contribution_sum=float(positive.sum()),
        negative_contribution_sum=float(delta[delta<0].sum()),
        largest_absolute_share=float(np.max(np.abs(delta))/abs_sum) if abs_sum else None,
        top_positive_shares_of_net_when_net_positive={str(k):float(positive[:k].sum()/net) if net>0 else None
                                                    for k in (1,3,5)},
        top_contributing_ids=[dict(id=str(ids[i]),average_net_gain_contribution=float(delta[i])) for i in top],
        scope='descriptive fixed-policy stability; no objects excluded and no best run selected'), dict(
            hr_selection_frequency=left_masks.mean(0), ridge_selection_frequency=ridge_masks.mean(0),
            average_paired_net_gain_contribution=delta)


def check_repeated_outcomes(reference, current, ids, repeat):
    """Audit roundoff in re-evaluated bounded utility without replacing values.

    Gamma is dimensionless and bounded on approximately [-1, 1]. The absolute
    tolerance is 64 float64 eps, not a data-fit tolerance or a risk threshold.
    ID alignment has already been checked exactly. Both outcome label boundaries
    must still agree exactly, even when their numerical difference is tiny.
    """
    reference, current = np.asarray(reference), np.asarray(current)
    if (reference.shape != current.shape or reference.shape != (len(ids), 3)
            or not np.isfinite(reference).all() or not np.isfinite(current).all()):
        raise ValueError('Repeated partitions must preserve each ID original outcome: invalid shape or values')
    difference = current-reference
    null_changes = (current <= NULL_THRESHOLD) != (reference <= NULL_THRESHOLD)
    positive_changes = (current >= POSITIVE_MARGIN) != (reference >= POSITIVE_MARGIN)
    if (np.any(np.abs(difference) > OUTCOME_ROUNDOFF_ATOL)
            or null_changes.any() or positive_changes.any()):
        raise ValueError('Repeated partitions must preserve each ID original outcome: non-roundoff difference or changed label')
    changed = np.argwhere(difference != 0)
    return dict(repeat=repeat, reference_repeat=0,
        absolute_tolerance=OUTCOME_ROUNDOFF_ATOL, relative_tolerance=0.,
        max_absolute_difference=float(np.abs(difference).max()),
        changed_float_elements=len(changed), null_label_changes=int(null_changes.sum()),
        positive_label_changes=int(positive_changes.sum()), original_values_replaced=False,
        changes=[dict(id=str(ids[i]), action=ACTIONS[j], reference=float(reference[i,j]),
                      current=float(current[i,j]), difference=float(difference[i,j])) for i,j in changed])


def summarize(root, *, output=None):
    root = Path(root).resolve()
    destination = root if output is None else Path(output).resolve()
    if output is not None:
        if destination == root:
            raise ValueError('An explicit summary destination must be separate from the source run')
        destination.mkdir(parents=True, exist_ok=True)
        if (destination/'summary.json').exists() or (destination/'REPORT.md').exists():
            raise FileExistsError('A completed separate summary is not overwritten')
    manifest = json.loads((root/'run_manifest.json').read_text())
    for flag in ('final_opened','fifth_repeat_opened','original_endpoint_changed','original_contract_changed'):
        if manifest.get(flag) is not False:
            raise ValueError('Repeated development experiment does not preserve '+flag)
    if tuple(manifest['arms']) != ARMS:
        raise ValueError('The six predeclared stability arms must retain their order')
    ids = np.asarray(manifest['ids'], str)
    if not len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique compound IDs required')
    records = manifest['repetitions']
    if len(records) != 3 or len({r['repeat'] for r in records}) != 3:
        raise ValueError('Exactly three distinct predeclared partitions are required')
    reports, data, actual, outcome_audits = {}, {}, None, []
    for record in records:
        report, arrays, _ = _read_repeat(root, record, manifest, ids, output=destination)
        current = arrays['RIDGE_VALID']['actual']
        if actual is None:
            actual = current.copy()
        outcome_audits.append(check_repeated_outcomes(actual, current, ids, record['repeat']))
        reports[record['repeat']], data[record['repeat']] = report, arrays
    cfg = manifest['config']; count = int(cfg['bootstrap'])
    seed = int(cfg.get('bootstrap_seed', cfg.get('seed', 20260914)))
    if count < 2:
        raise ValueError('At least two paired-ID bootstrap replicates are required')
    boot = np.random.default_rng(seed).integers(len(ids), size=(count,len(ids)))
    primary_pairs = [(r['repeat'], arm, 'RIDGE_VALID') for r in records for arm in HR_ARMS]
    selection_pairs = [(r['repeat'],'RIDGE_VALID','RIDGE_TRAINCV') for r in records]
    primary, primary_contrib = _average_comparison(primary_pairs, data, reports, boot, len(ids))
    selection, selection_contrib = _average_comparison(selection_pairs, data, reports, boot, len(ids))
    directions = _direction_reports(records, reports)
    index = _main_index(primary['policies'])
    stability, stability_arrays = _selection_diagnostics(primary_pairs, data, reports, ids,
        primary_contrib['policy_net_value_difference'][index])
    np.savez_compressed(destination/'primary_compound_contributions.npz', ids=ids,
        **primary_contrib, **stability_arrays)
    np.savez_compressed(destination/'selection_info_compound_contributions.npz', ids=ids, **selection_contrib)
    result = dict(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(), n_unique_compounds=len(ids),
        repetitions=[reports[r['repeat']] for r in records],
        primary_HR_mean_vs_RIDGE_VALID=primary, selection_info_RIDGE_VALID_vs_TRAINCV=selection,
        direction_analysis=directions, selection_stability=stability,
        main_score='mean HR-seed ADD_TWO Gamma CRPS minus RIDGE_VALID, equal-weight partitions and seeds within each ID',
        principal_policy='ADD_TWO expected-gain ranking, 25% physical-well budget independently in each outer fold',
        interval_scope=SCOPE, bootstrap_replicates=count, bootstrap_seed=seed,
        repetitions_count=3, HR_seed_count_per_partition=3, HR_run_count=9,
        HR_fold_result_count=len(directions['folds']), bootstrap_sample_size=len(ids),
        repeated_rows_are_independent=False, best_seed_selected=False, ensemble_prediction=False,
        activation_counts_combined_for_certification=False,
        selection_information_scope='RIDGE_VALID and HR have the same validation-label availability, not identical hypothesis-search complexity',
        model_scope='fixed four-role nine-coordinate conditional geometry, not full cross-site biological world-model validation',
        samples_per_object=cfg.get('samples'), final_opened=False,fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False,
        historical_dev=True, formal_certificate=False,
        source_run=str(root), summary_destination=str(destination),
        repeated_outcome_roundoff_audit=outcome_audits,
        original_predictions_or_scores_modified=False,
        source_failure_status_reclassified=False, no_retraining=True, no_rescoring=True)
    write_json(destination/'summary.json', result)

    lines = [f'# {len(ids)} 个对象：重复划分与 HR 种子稳定性', '',
        '三个划分、每个划分三个 HR 训练种子。只汇总各次运行的评分和策略表现，不挑最好种子，也不构造集成预测器。',
        f'每个化合物先跨运行平均贡献，再对 {len(ids)} 个 ID 配对重抽；重复预测不增加独立样本数。', '',
        '## 主比较：HR 种子均值 − RIDGE_VALID', '',
        '| 指标 | 差值 | 条件式 95% 区间 |', '|---|---:|---|']
    for label, score, coordinate in [('ADD_TWO Γ-CRPS',primary['gamma_crps'],2),
                                     ('ADD_TWO NULL Brier',primary['null_brier'],2),
                                     ('标准化 u-MSE',primary['u_mean_mse'],None)]:
        mean = score['mean'] if coordinate is None else score['mean'][coordinate]
        low, high = score['interval95'] if coordinate is None else np.asarray(score['interval95'])[:,coordinate]
        lines.append(f'| {label} | {_fmt(mean)} | [{_fmt(low)}, {_fmt(high)}] |')
    pp = primary['principal_policy']
    lines += ['', '## 主预算：25% 物理孔，ADD_TWO，按期望收益选择', '',
        f"每次运行的激活对象数为 {pp['selected_n_per_run_left']}；这些计数不合并用于满足认证 guard。", '',
        '| 比较 | 每可选对象净值差 | 95% 区间 | FDP差 |', '|---|---:|---|---:|']
    for label, comparison in [('HR − RIDGE_VALID',primary),('RIDGE_VALID − RIDGE_TRAINCV',selection)]:
        policy = comparison['principal_policy']; score = policy['net_gain_per_eligible']
        fdp = policy['fdp_difference']
        lines.append(f"| {label} | {_fmt(score['mean'])} | [{_fmt(score['interval95'][0])}, {_fmt(score['interval95'][1])}] | "
                     f"{_fmt(fdp['mean'] if fdp else None)} |")
    lines += ['', '## 九次 HR 运行：不选最好一次', '',
        '| 划分 / HR种子 | Γ-CRPS差 | 每选中对象净值 | RIDGE_VALID净值 | NULL/激活 | FDP | FPR |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in directions['runs']:
        lines.append(f"| {row['repeat']} / S{row['hr_seed_index']} | {_fmt(row['crps_difference'])} | "
            f"{_fmt(row['selected_mean_value'])} | {_fmt(row['ridge_selected_mean_value'])} | "
            f"{row['selected_null_count']:g}/{row['selected_n']} | {_fmt(row['fdp'],4)} | {_fmt(row['fpr'],4)} |")
    lines += ['', '## 方向一致性（描述性，不是独立重复检验）', '',
        '| 层级 | CRPS更好/更差/持平 | 净值更好/更差/持平 | FDP更好/更差/持平 |', '|---|---|---|---|']
    for name, item in directions['direction_counts'].items():
        value = lambda key: '/'.join(str(item[key][field]) for field in ('favorable','unfavorable','tied'))
        lines.append(f"| {name} (n={item['n']}) | {value('crps')} | {value('value')} | {value('fdp')} |")
    lines += ['', '每个划分、六臂的原始评分、u-MSE、秩关联、全部36个动作/预算策略及45条 HR 折结果见 summary.json。',
        'GLOBAL 使用各折均匀子集期望；不同折的常数不参与个体排序。',
        '均值与区间反映复用 DEV 的条件式比较，不包含共享批次、训练集重叠、历史模型搜索或 Monte Carlo 积分误差。',
        'RIDGE_VALID 与 HR 的验证标签可用范围相同，不意味着搜索复杂度相同。FINAL、第五重复、原收益和七项合同均未改变。']
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result
