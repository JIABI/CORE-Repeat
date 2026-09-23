"""Fixed epoch-30 four-arm summaries, with paired changes from saved epoch 10.

The continuation was chosen after seeing development results. No checkpoint is
selected by this module; source OOF predictions are read without modification.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .geometry_kernel_summary import U_SCOPE, _policy, _compare, _interval_text
from .geometry_kernel_replacement_summary import (
    ARMS, COMPARISONS, _validate_manifest as _validate_four_arm_manifest,
    _parameter_counts, _capacity, _score_table, _policy_table, _capacity_table,
)
from .gram_oof_experiment import selection_mask
from .hierarchical_geometry_summary import _load_arm, _action_rows, _fmt


INTERVAL_SCOPE = (
    '2000 paired compound bootstrap resamples within five outer folds; '
    'fixed epoch-30/epoch-10 predictions and selection masks; excludes training, '
    'shared-batch, historical-search and Monte Carlo uncertainty; continuation '
    'chosen after viewing DEV epoch-10 results, not independent certification'
)


def _validate_manifest(manifest):
    ids, allocation = _validate_four_arm_manifest(manifest)
    if manifest['config'].get('stage_epochs') != 30:
        raise ValueError('The continuation stage must specify config.stage_epochs=30')
    if not isinstance(manifest.get('continuation_source'), str) or not manifest['continuation_source'].strip():
        raise ValueError('continuation_source must identify the saved epoch-10 run')
    return ids, allocation


def _read_coordinates(root, records, ids, arm):
    actual = np.empty((len(ids), 9), dtype=float)
    mean = np.empty_like(actual)
    rows = []
    for record in records:
        ix = np.asarray(record['test'], int)
        folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
        with np.load(folder/'test/u_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Coordinate IDs do not match the continuation test fold')
            a, p = np.asarray(saved['actual_u'], float), np.asarray(saved['mean_u'], float)
            if a.shape != (len(ix), 9) or p.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
                raise ValueError('Finite aligned nine-coordinate predictions are required')
            actual[ix], mean[ix] = a, p
        metrics = json.loads((folder/'test/metrics.json').read_text())
        metadata = metrics.get('model', {})
        epoch = metrics.get('actual_checkpoint_epoch', metadata.get('actual_checkpoint_epoch'))
        if epoch != ('frozen' if arm == 'A_HR' else 30):
            raise ValueError('Continuation test scores must use actual epoch 30, with A_HR frozen')
        if metrics.get('samples') != 2000:
            raise ValueError('Continuation test scores must use 2000 predictive draws')
        complete_path = folder/'training_complete.json'
        complete = json.loads(complete_path.read_text()) if complete_path.is_file() else None
        history_path = folder/'history.jsonl'
        history = ([json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
                   if history_path.is_file() else [])
        checkpoint = dict(actual_checkpoint_epoch=epoch, test_metadata=metadata,
            training_complete=complete,
            validation_selection_appendix=[row for row in history if row.get('epoch') in (0, 5, 10, 15, 20, 25, 30)],
            appendix_not_used_to_choose_test_predictions=True)
        rows.append(dict(fold=record['fold'], u_mse=float(np.square(a-p).mean()), checkpoint=checkpoint))
    return actual, mean, rows


def _finish_arrays(arrays, ids, allocation):
    mask = selection_mask(arrays['predicted'][:, 2], ids, allocation, .25, 2)
    counts = [int(mask[allocation == fold].sum()) for fold in sorted(np.unique(allocation))]
    if sorted(counts) != [15, 16, 16, 16, 16] or int(mask.sum()) != 79:
        raise ValueError('The foldwise physical-well budget must select 79 objects')
    arrays['principal_mask'] = mask
    arrays['u_object_mse'] = np.square(arrays['actual_u']-arrays['mean_u']).mean(1)


def _model_scores(arrays, records, ids):
    return dict(u_mse=float(arrays['u_object_mse'].mean()), u_scope=U_SCOPE,
        per_coordinate_u_mse=np.square(arrays['actual_u']-arrays['mean_u']).mean(0).tolist(),
        actions=_action_rows(arrays, records, global_model=False),
        principal_policy=_policy(arrays['actual'][:, 2], arrays['principal_mask']),
        selected_ids=ids[arrays['principal_mask'].astype(bool)].tolist(),
        geometry_energy=float(arrays['geometry_energy'].mean()))


def _source_arrays(source, arm, ids, allocation, current):
    arrays = {}
    with np.load(source/f'{arm}_oof_predictions.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['ids'], ids) or not np.array_equal(saved['fold'], allocation):
            raise ValueError('Epoch-10 source IDs or outer-fold allocation differ')
        for key, shape in dict(actual=(639, 3), predicted=(639, 3), p_null=(639, 3),
                utility_crps=(639, 3), geometry_energy=(639,), actual_u=(639, 9), mean_u=(639, 9)).items():
            value = np.asarray(saved[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid saved epoch-10 '+key)
            arrays[key] = value.copy()
        if np.any(arrays['p_null'] < 0) or np.any(arrays['p_null'] > 1):
            raise ValueError('Epoch-10 NULL probabilities must lie in [0,1]')
        if not np.array_equal(arrays['actual'], current['actual']) or not np.array_equal(arrays['actual_u'], current['actual_u']):
            raise ValueError('Epoch-10 and epoch-30 must have identical original Gamma and nine-u targets')
        _finish_arrays(arrays, ids, allocation)
        if not np.array_equal(saved['principal_mask'], arrays['principal_mask']):
            raise ValueError('Saved epoch-10 masks do not implement the unchanged budget/ranking rule')
    return arrays


def _comparison_lines(comparisons):
    lines = ['| 比较 | 指标 | 差值 | 条件式95%区间 |', '|---|---|---:|---|']
    labels = dict(u_mse='九维 u MSE', gamma_crps='ADD_TWO CRPS', null_brier='NULL Brier',
        net_gain_per_eligible='每个可选对象净值', net_gain_per_selected='每个选中对象净值', fdp='FDP', fpr='FPR')
    for pair in comparisons.values():
        for key, title in labels.items():
            stat = pair[key]
            lines.append(f"| {pair['left']} − {pair['right']} | {title} | {_fmt(stat['mean'])} | {_interval_text(stat)} |")
    return lines


def _write_reports(root, result):
    intro = ['# 四臂续训：固定第30轮与第10轮的对照', '',
        '延长至30轮是在看过已开放 DEV 的10轮结果后决定的开发实验，不是新的预先独立认证。',
        'B_MLP、C_GENERIC、D_STRUCTURED 主表均使用实际第30轮；A_HR 保持冻结。验证集最佳轮次仅作附录。',
        '同一639对象、同一五折、同一原始 ADD_TWO 收益，各对象只贡献一次折外结果。', '']
    movement = ['| 模型：30−10 | u MSE差 | CRPS差 | Brier差 | 每可选对象净值差 | FDP差 | FPR差 | 共同选择 / 79 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, pair in result['epoch_extension_comparisons'].items():
        values = [_fmt(pair[key]['mean']) for key in ('u_mse', 'gamma_crps', 'null_brier', 'net_gain_per_eligible', 'fdp', 'fpr')]
        movement.append('| '+arm+' | '+' | '.join(values)+f" | {pair['overlap']['intersection_n']} |")
    short = intro+_score_table(result)+['', '## 同预算价值与风险', '']+_policy_table(result)
    short += ['', '## 第10轮至第30轮变化', '']+movement
    short += ['', '误差、CRPS、Brier、风险差为负有利，净值差为正有利。六组30轮臂间比较及四组续训变化的配对区间见 REPORT.md。',
        '四臂并非等容量；网络参数包含冻结 HR 网络，不含固定 ridge 或描述量缓冲值。']
    (root/'SUMMARY.md').write_text('\n'.join(short)+'\n')
    lines = intro+['## 第30轮预测与概率评分', '']+_score_table(result)
    lines += ['', 'u MSE 按各折 fit-only 标准化的九维坐标计算，再按对象合并。合并 Spearman/AUC 仅为描述性指标，折内秩关联另存 JSON。',
        '', '## 第30轮相同物理孔预算', '']+_policy_table(result)
    lines += ['', '各折按 E[Γ] 选择16/16/16/16/15对象，共79对象、158个追加孔。原 Γ 已扣成本，不再二次扣费。',
        'NULL 为 Γ≤0，POSITIVE 为 Γ≥0.005；FDP 分母为选中对象，FPR 分母为全部 NULL 对象。',
        '', '## 第30轮六组臂间比较', '',
        '所有差值为左减右；以下使用同一组2000次折内对象配对复抽。', '']+_comparison_lines(result['comparisons'])
    lines += ['', '## 第10轮至第30轮：同臂配对变化', '']+movement
    lines += ['', '直接读取旧目录的四份 OOF 文件，核对每个 ID、外折、实际 Γ 和实际九维目标完全一致。旧结果不覆盖、不重新拟合。', '']
    lines += _comparison_lines(result['epoch_extension_comparisons'])
    lines += ['', '## 五折方向与选择重叠', '',
        '| 比较 | u MSE 有利/不利/并列 | CRPS 有利/不利/并列 | Brier 有利/不利/并列 | 净值有利/不利/并列 | 共同选择 | Jaccard |',
        '|---|---|---|---|---|---:|---:|']
    for pair in [*result['comparisons'].values(), *result['epoch_extension_comparisons'].values()]:
        d = pair['fold_directions']
        cells = ['/'.join(str(d[key][name]) for name in ('favorable', 'unfavorable', 'tied'))
                 for key in ('u_mse', 'gamma_crps', 'null_brier', 'net_gain_per_eligible')]
        overlap = pair['overlap']
        lines.append('| '+pair['left']+' − '+pair['right']+' | '+' | '.join(cells)
                     +f" | {overlap['intersection_n']} / 79 | {_fmt(overlap['jaccard'], 4)} |")
    lines += ['', '## 网络参数与训练附录', '']+_capacity_table(result)
    cap = result['capacity']
    lines += ['', f"C/D 可训练网络参数相同：{cap['generic_structured_trainable_equal']}；网络参数合计相同：{cap['generic_structured_total_equal']}。",
        f"B_MLP 可训练参数较少：{cap['mlp_trainable_smaller_than_generic']}；不将四臂称为容量匹配比较。",
        '计数为 model.parameters() 中的网络参数，包含冻结 HR 网络，不含固定 ridge 系数/截距及描述量锚点/尺度缓冲值。',
        '0/5/10/15/20/25/30轮的验证记录与 training_complete 保存在 JSON 附录，未据测试表现选取最佳轮次。',
        '', '## 统计与信息边界', '',
        '区间条件于固定模型预测和选择掩码，比例在每次配对复抽中重新计算分母；未纳入训练随机性、共享批次、重叠训练集、历史选择或 Monte Carlo 误差。',
        '六组臂间比较和四组时间比较没有多重比较校正。后续训练由先前 DEV 结果触发，因此只能作开发对照。',
        '四臂与全部六组比较如实保留。原收益、七项合同、旧结果、FINAL 和第五重复均未改动。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    """Aggregate epoch 30 and pair it with the immutable saved epoch-10 OOF."""
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = _validate_manifest(manifest)
    source = Path(manifest['continuation_source']).expanduser()
    source = (source if source.is_absolute() else root/source).resolve()
    if source == root:
        raise ValueError('The continuation cannot overwrite its epoch-10 source')
    source_summary = json.loads((source/'summary.json').read_text())
    if source_summary.get('actual_checkpoint_epoch') != 10 or source_summary.get('arms') != list(ARMS):
        raise ValueError('The source summary must identify the four-arm epoch-10 run')
    data, models, previous, previous_models = {}, {}, {}, {}
    actual_reference = u_reference = None
    for arm in ARMS:
        arrays, fold_rows = _load_arm(root, manifest, arm, ids)
        actual_u, mean_u, u_rows = _read_coordinates(root, manifest['folds'], ids, arm)
        if actual_reference is None:
            actual_reference, u_reference = arrays['actual'].copy(), actual_u.copy()
        elif not np.array_equal(actual_reference, arrays['actual']) or not np.array_equal(u_reference, actual_u):
            raise ValueError('The four arms must use identical original Gamma and nine-u targets')
        arrays.update(actual_u=actual_u, mean_u=mean_u)
        _finish_arrays(arrays, ids, allocation)
        for row, u_row, record in zip(fold_rows, u_rows, manifest['folds']):
            ix = np.asarray(record['test'], int)
            row.update(u=u_row, principal_policy=_policy(arrays['actual'][ix, 2], arrays['principal_mask'][ix]),
                       parameter_counts=_parameter_counts(u_row['checkpoint'], arm))
        models[arm] = dict(**_model_scores(arrays, manifest['folds'], ids), folds=fold_rows)
        data[arm] = arrays
        previous[arm] = _source_arrays(source, arm, ids, allocation, arrays)
        previous_models[arm] = _model_scores(previous[arm], manifest['folds'], ids)
    seed = int(manifest['config'].get('bootstrap_seed', manifest['config'].get('seed', 20260914)))
    rng = np.random.default_rng(seed)
    boot = np.column_stack([rng.choice(np.flatnonzero(allocation == fold),
        size=(2000, int(np.sum(allocation == fold)))) for fold in sorted(np.unique(allocation))])
    comparisons = {left+'__minus__'+right: _compare(left, right, data, ids, manifest['folds'], boot)
                   for left, right in COMPARISONS}
    extension = {}
    for arm in ARMS:
        left, right = arm+'_epoch30', arm+'_epoch10'
        extension[arm] = _compare(left, right, {left: data[arm], right: previous[arm]}, ids, manifest['folds'], boot)
    result = dict(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        n=len(ids), arms=list(ARMS), models=models, comparisons=comparisons,
        epoch_extension_comparisons=extension, epoch10_models=previous_models, capacity=_capacity(models),
        continuation_source=str(source), extension_chosen_after_dev_epoch10=True,
        fold_test_counts=[len(record['test']) for record in manifest['folds']],
        actual_checkpoint_epoch=30, frozen_comparator='A_HR',
        principal_action='original ADD_TWO / Z1Z2', principal_budget='25% physical wells, foldwise',
        principal_ranking='expected Gamma', samples_per_object=2000, bootstrap_replicates=2000,
        bootstrap_seed=seed, interval_scope=INTERVAL_SCOPE,
        secondary_comparisons_multiplicity_adjusted=False, historical_dev=True,
        formal_certificate=False, final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False)
    for arm in ARMS:
        np.savez_compressed(root/f'{arm}_oof_predictions.npz', ids=ids, fold=allocation, **data[arm])
    write_json(root/'summary.json', result)
    _write_reports(root, result)
    return result
