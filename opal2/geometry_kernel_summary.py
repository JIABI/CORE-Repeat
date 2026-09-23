"""Paired OOF summaries of the fixed-epoch HR and kernel-stage experiment.

Reads saved predictions only. It neither fits a model nor picks a checkpoint,
threshold, favorable seed, or favorable test subset. The unit of aggregation
and resampling is a unique compound, with resampling stratified by outer fold.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .baseline_policy import ACTIONS, NULL_THRESHOLD, POSITIVE_MARGIN
from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import selection_mask
from .hierarchical_geometry_summary import (
    _folds, _load_arm, _action_rows, _bootstrap_statistic, _fmt,
)


ARMS = ('A_HR', 'B_GENERIC', 'C_STRUCTURED')
COMPARISONS = (('B_GENERIC', 'A_HR'), ('C_STRUCTURED', 'A_HR'),
               ('C_STRUCTURED', 'B_GENERIC'))
INTERVAL_SCOPE = (
    '2000 paired compound bootstrap resamples within the five outer folds; '
    'fixed fitted epoch-10 predictions and policy masks; excludes training, '
    'shared-batch, historical-search and Monte Carlo integration uncertainty; '
    'development comparison, not an independent certificate'
)
U_SCOPE = ('nine coordinates standardized using each outer fit partition; '
           'paired arms have identical targets/scales within every fold')


def _validate_manifest(manifest):
    if list(manifest['arms']) != list(ARMS):
        raise ValueError('The stage must report all three declared arms in order')
    for key in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed',
                'original_contract_changed'):
        if manifest.get(key) is not False:
            raise ValueError('The recorded experiment does not preserve '+key)
    ids, allocation = _folds(manifest)
    if len(ids) != 639 or len(manifest['folds']) != 5:
        raise ValueError('This stage uses exactly 639 unique DEV IDs and five outer folds')
    for record in manifest['folds']:
        parts = [np.asarray(record[key], dtype=int)
                 for key in ('fit', 'inner_validation', 'test')]
        if any(not len(p) or len(np.unique(p)) != len(p)
               or np.any(p < 0) or np.any(p >= len(ids)) for p in parts):
            raise ValueError('Invalid fit/validation/test partition')
        joined = np.concatenate(parts)
        if not np.array_equal(np.sort(joined), np.arange(len(ids))):
            raise ValueError('Fit, inner validation and test must partition the 639 IDs')
    if int(manifest['config'].get('bootstrap', 2000)) != 2000:
        raise ValueError('The stage uses the declared 2000 bootstrap resamples')
    return ids, allocation


def _checkpoint_record(arm_folder, arm):
    metric = json.loads((arm_folder/'test'/'metrics.json').read_text())
    metadata = metric.get('model', {})
    epoch = metric.get('actual_checkpoint_epoch', metadata.get('actual_checkpoint_epoch'))
    if arm == 'A_HR':
        if epoch not in (10, 'frozen'):
            raise ValueError('A_HR must identify its frozen comparator or epoch-10 checkpoint')
    elif epoch != 10:
        raise ValueError('Kernel-stage test scores must come from the actual epoch-10 checkpoint')
    complete_path = arm_folder/'training_complete.json'
    complete = json.loads(complete_path.read_text()) if complete_path.is_file() else None
    history_path = arm_folder/'history.jsonl'
    history = ([json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
               if history_path.is_file() else [])
    # Validation records are descriptive only; no checkpoint is loaded or chosen.
    return dict(actual_checkpoint_epoch=epoch, test_metadata=metadata,
                training_complete=complete,
                validation_selection_appendix=[row for row in history if row.get('epoch') in (0, 5, 10)],
                appendix_not_used_to_choose_test_predictions=True)


def _read_coordinates(root, records, ids, arm):
    actual = np.empty((len(ids), 9), dtype=np.float64)
    mean = np.empty_like(actual)
    folds = []
    for record in records:
        ix = np.asarray(record['test'], dtype=int)
        folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
        with np.load(folder/'test'/'u_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Coordinate prediction IDs differ from their declared test fold')
            a, p = np.asarray(saved['actual_u'], float), np.asarray(saved['mean_u'], float)
            if a.shape != (len(ix), 9) or p.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
                raise ValueError('Finite aligned nine-coordinate predictions are required')
            actual[ix], mean[ix] = a, p
        folds.append(dict(fold=record['fold'], u_mse=float(np.square(a-p).mean()),
                          checkpoint=_checkpoint_record(folder, arm)))
    return actual, mean, folds


def _policy(actual, mask):
    null, positive = actual <= NULL_THRESHOLD, actual >= POSITIVE_MARGIN
    n, k = len(actual), int(mask.sum())
    false = int(mask @ null)
    total = float(mask @ actual)
    return dict(action='Z1Z2', fraction=.25, ranking='expected_gain', section='common_budget',
                eligible_n=n, selected_n=k, used_wells=2*k, total_net_gain=total,
                per_eligible_net_gain=total/n, per_selected_net_gain=total/k if k else None,
                selected_null_count=false, selected_positive_count=int(mask @ positive),
                population_null_count=int(null.sum()),
                fdp=false/k if k else None,
                fpr=false/int(null.sum()) if null.any() else None,
                sensitivity=float(mask @ positive/positive.sum()) if positive.any() else None,
                selection_rule='descending E[Gamma], lexical ID only for exact ties; within each outer fold',
                formal_certificate=False)


def _ratio_difference(left_numerator, left_denominator, right_numerator,
                      right_denominator, boot):
    """Paired ratio-statistic bootstrap of two fixed policy masks."""
    ln, ld, rn, rd = [np.asarray(x, dtype=float) for x in
                      (left_numerator, left_denominator, right_numerator, right_denominator)]
    lden, rden = ld[boot].sum(1), rd[boot].sum(1)
    valid = (lden > 0) & (rden > 0)
    samples = ln[boot].sum(1)[valid]/lden[valid] - rn[boot].sum(1)[valid]/rden[valid]
    point = float(ln.sum()/ld.sum()-rn.sum()/rd.sum()) if ld.sum() and rd.sum() else None
    return dict(mean=point,
                interval95=np.quantile(samples, [.025, .975]).tolist() if len(samples) else None,
                valid_resamples=int(valid.sum()), total_resamples=len(boot),
                denominator_scope='ratio recomputed in each paired fixed-policy resample')


def _overlap(left, right, ids):
    a, b = np.asarray(left, bool), np.asarray(right, bool)
    intersection, union = int(np.sum(a & b)), int(np.sum(a | b))
    return dict(left_selected=int(a.sum()), right_selected=int(b.sum()),
                intersection_n=intersection, union_n=union,
                jaccard=intersection/union if union else None,
                left_only_ids=ids[a & ~b].tolist(), right_only_ids=ids[b & ~a].tolist())


def _direction(rows, key, favorable):
    values = np.asarray([row[key] for row in rows], dtype=float)
    favorable_mask = values < -1e-12 if favorable == 'negative' else values > 1e-12
    unfavorable_mask = values > 1e-12 if favorable == 'negative' else values < -1e-12
    return dict(favorable=int(favorable_mask.sum()), unfavorable=int(unfavorable_mask.sum()),
                tied=int((~favorable_mask & ~unfavorable_mask).sum()),
                favorable_direction=favorable, numerical_zero_tolerance=1e-12)


def _compare(left, right, data, ids, records, boot):
    a, b = data[left], data[right]
    actual = a['actual'][:, 2]
    null = actual <= NULL_THRESHOLD
    ma, mb = a['principal_mask'], b['principal_mask']
    mse_delta = a['u_object_mse']-b['u_object_mse']
    crps_delta = a['utility_crps'][:, 2]-b['utility_crps'][:, 2]
    brier_delta = np.square(a['p_null'][:, 2]-null)-np.square(b['p_null'][:, 2]-null)
    gain_delta = (ma-mb)*actual
    false_delta = (ma-mb)*null
    fold_rows = []
    for record in records:
        ix = np.asarray(record['test'], dtype=int)
        pa, pb = _policy(actual[ix], ma[ix]), _policy(actual[ix], mb[ix])
        fold_rows.append(dict(fold=record['fold'], n=len(ix),
            u_mse=float(mse_delta[ix].mean()), gamma_crps=float(crps_delta[ix].mean()),
            null_brier=float(brier_delta[ix].mean()), net_gain_per_eligible=float(gain_delta[ix].mean()),
            fdp=pa['fdp']-pb['fdp'] if pa['fdp'] is not None and pb['fdp'] is not None else None,
            fpr=pa['fpr']-pb['fpr'] if pa['fpr'] is not None and pb['fpr'] is not None else None,
            overlap=_overlap(ma[ix], mb[ix], ids[ix])))
    return dict(left=left, right=right, direction='left minus right',
        u_mse=_bootstrap_statistic(mse_delta, boot),
        gamma_crps=_bootstrap_statistic(crps_delta, boot),
        null_brier=_bootstrap_statistic(brier_delta, boot),
        net_gain_per_eligible=_bootstrap_statistic(gain_delta, boot),
        net_gain_per_selected=_ratio_difference(ma*actual, ma, mb*actual, mb, boot),
        false_activation_per_eligible=_bootstrap_statistic(false_delta, boot),
        fdp=_ratio_difference(ma*null, ma, mb*null, mb, boot),
        fpr=_ratio_difference(ma*null, null, mb*null, null, boot),
        overlap=_overlap(ma, mb, ids), folds=fold_rows,
        fold_directions={key: _direction(fold_rows, key, direction) for key, direction in (
            ('u_mse', 'negative'), ('gamma_crps', 'negative'), ('null_brier', 'negative'),
            ('net_gain_per_eligible', 'positive'))}, formal_certificate=False)


def _interval_text(statistic):
    interval = statistic.get('interval95')
    return '未定义' if interval is None else f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"


def _tables(result):
    rows = ['| 模型 | 九维 u MSE↓ | ADD_TWO CRPS↓ | NULL Brier↓ | Γ Spearman | NULL AUC |',
            '|---|---:|---:|---:|---:|---:|']
    policy = ['| 模型 | 对象 / 孔 | 每个可选对象净值 | 每个选中对象净值 | FDP | FPR |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        model = result['models'][arm]
        a, p = model['actions'][2], model['principal_policy']
        rows.append(f"| {arm} | {_fmt(model['u_mse'])} | {_fmt(a['gamma_crps'])} | "
                    f"{_fmt(a['null_brier'])} | {_fmt(a['spearman'], 4)} | {_fmt(a['null_auc'], 4)} |")
        policy.append(f"| {arm} | {p['selected_n']} / {p['used_wells']} | "
                      f"{_fmt(p['per_eligible_net_gain'])} | {_fmt(p['per_selected_net_gain'])} | "
                      f"{_fmt(p['fdp'], 4)} | {_fmt(p['fpr'], 4)} |")
    return rows, policy


def _write_reports(root, result):
    scores, policies = _tables(result)
    intro = ['# HR 与两类 kernel：第10轮阶段结果', '',
        'B_GENERIC 与 C_STRUCTURED 均使用实际第10轮检查点；A_HR 为固定对照。',
        f"五个外层测试折合并为 {result['n']} 个唯一对象。相同对象只贡献一次折外结果。", '']
    small = intro+scores+['', '25%物理孔预算：各折按预测 ADD_TWO 净收益排序，共选择79个对象、158个孔。', '']+policies
    small += ['', '这是已反复开放 DEV 的阶段比较，不是新的独立验证或合同授权。完整配对区间和各折方向见 REPORT.md。']
    (root/'SUMMARY.md').write_text('\n'.join(small)+'\n')
    lines = intro+['## 预测与概率评分', '']+scores
    lines += ['', 'u MSE 在各折 fit-only 标准化九维坐标内计算，再按对象数合并；没有把不同折的原生坐标直接混算。',
        'Γ Spearman 与 NULL AUC 是合并折外分数的描述量；折内秩关联另存于 summary.json，未以这些结果重新选模型。',
        '', '## 同预算的价值与风险', '']+policies
    lines += ['', 'NULL 为原 Γ≤0，POSITIVE 为原 Γ≥0.005。Γ 已扣除两孔成本，不再二次扣费。',
        'FDP 分母为选中对象；FPR 分母为该测试人群的全部 NULL 对象。主规则按 E[Γ] 排序，不另外搜索阈值。',
        '', '## 三组预先声明的配对比较', '',
        '差值均为左臂减右臂；误差、CRPS、Brier 和风险负值有利，净价值正值有利。', '',
        '| 比较 | 指标 | 差值 | 条件式95%区间 |', '|---|---|---:|---|']
    labels = dict(u_mse='九维 u MSE', gamma_crps='ADD_TWO CRPS', null_brier='NULL Brier',
                  net_gain_per_eligible='每个可选对象净值', net_gain_per_selected='每个选中对象净值',
                  fdp='FDP', fpr='FPR')
    for comparison in result['comparisons'].values():
        label = comparison['left']+' − '+comparison['right']
        for key, title in labels.items():
            score = comparison[key]
            lines.append(f"| {label} | {title} | {_fmt(score['mean'])} | {_interval_text(score)} |")
    lines += ['', '## 五折方向与选择重叠', '',
        '| 比较 | u MSE 有利/不利/并列 | CRPS 有利/不利/并列 | Brier 有利/不利/并列 | 净值有利/不利/并列 | 共同选中 | Jaccard |',
        '|---|---|---|---|---|---:|---:|']
    for comparison in result['comparisons'].values():
        direction = comparison['fold_directions']
        cells = ['/'.join(str(direction[key][name]) for name in ('favorable', 'unfavorable', 'tied'))
                 for key in ('u_mse', 'gamma_crps', 'null_brier', 'net_gain_per_eligible')]
        overlap = comparison['overlap']
        lines.append('| '+comparison['left']+' − '+comparison['right']+' | '+' | '.join(cells)
                     +f" | {overlap['intersection_n']} / 79 | {_fmt(overlap['jaccard'], 4)} |")
    lines += ['', '每折的具体差值、添加/移除的对象 ID、三种动作的完整指标均保存在 summary.json；所有三臂均报告。',
        '', '## 训练记录附录与解释边界', '',
        '0/5/10轮的验证记录与训练完成信息仅作附录存入 summary.json；主表未采用 best.pt 重新选择测试结果。',
        '区间使用2000次折内对象配对复抽，条件于本次固定模型与选择掩码；比例在每次复抽中重算分母。',
        '这些区间不计入训练随机性、历史开发选择、共享批次、重叠训练集或 Monte Carlo 误差，也未作多重比较校正。',
        '本轮不改变原收益、纯度要求、七项合同或旧结果；没有打开 FINAL 或第五重复。',
        '十轮结果回答增量分支的早期行为，不能单凭一次阶段差值证明结构化 kernel 优于其他模型，或证明生物机制已被恢复。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    """Write summary.json, REPORT.md, SUMMARY.md and three OOF prediction files."""
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = _validate_manifest(manifest)
    data, models = {}, {}
    reference_actual = reference_u = None
    for arm in ARMS:
        arrays, fold_rows = _load_arm(root, manifest, arm, ids)
        actual_u, mean_u, u_rows = _read_coordinates(root, manifest['folds'], ids, arm)
        if reference_actual is None:
            reference_actual, reference_u = arrays['actual'].copy(), actual_u.copy()
        elif not np.array_equal(reference_actual, arrays['actual']) or not np.array_equal(reference_u, actual_u):
            raise ValueError('Paired arms must use identical original Gamma and nine-coordinate targets')
        mask = selection_mask(arrays['predicted'][:, 2], ids, allocation, .25, 2)
        counts = [int(mask[np.asarray(record['test'], int)].sum()) for record in manifest['folds']]
        if sorted(counts) != [15, 16, 16, 16, 16] or int(mask.sum()) != 79:
            raise ValueError('The declared foldwise 25% physical-well cap must select 79 objects')
        arrays.update(actual_u=actual_u, mean_u=mean_u,
                      u_object_mse=np.square(actual_u-mean_u).mean(1), principal_mask=mask)
        for row, u_row, record in zip(fold_rows, u_rows, manifest['folds']):
            ix = np.asarray(record['test'], int)
            row.update(u=u_row, principal_policy=_policy(arrays['actual'][ix, 2], mask[ix]))
        models[arm] = dict(u_mse=float(arrays['u_object_mse'].mean()),
            per_coordinate_u_mse=np.square(actual_u-mean_u).mean(0).tolist(), u_scope=U_SCOPE,
            actions=_action_rows(arrays, manifest['folds'], global_model=False), folds=fold_rows,
            principal_policy=_policy(arrays['actual'][:, 2], mask),
            selected_ids=ids[mask.astype(bool)].tolist(),
            geometry_energy=float(arrays['geometry_energy'].mean()))
        data[arm] = arrays
    seed = int(manifest['config'].get('bootstrap_seed', manifest['config'].get('seed', 20260914)))
    rng = np.random.default_rng(seed)
    boot = np.column_stack([rng.choice(np.flatnonzero(allocation == fold),
        size=(2000, int(np.sum(allocation == fold)))) for fold in sorted(np.unique(allocation))])
    comparisons = {left+'__minus__'+right: _compare(left, right, data, ids, manifest['folds'], boot)
                   for left, right in COMPARISONS}
    result = dict(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        n=len(ids), arms=list(ARMS), models=models, comparisons=comparisons,
        fold_test_counts=[len(record['test']) for record in manifest['folds']],
        actual_checkpoint_epoch=10, frozen_comparator='A_HR',
        principal_action='original ADD_TWO / Z1Z2', principal_budget='25% physical wells, foldwise',
        principal_ranking='expected Gamma', bootstrap_replicates=2000, bootstrap_seed=seed,
        interval_scope=INTERVAL_SCOPE, secondary_comparisons_multiplicity_adjusted=False,
        historical_dev=True, formal_certificate=False, final_opened=False,
        fifth_repeat_opened=False, original_endpoint_changed=False, original_contract_changed=False)
    for arm in ARMS:
        np.savez_compressed(root/f'{arm}_oof_predictions.npz', ids=ids, fold=allocation, **data[arm])
    write_json(root/'summary.json', result)
    _write_reports(root, result)
    return result
