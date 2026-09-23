"""Four-arm, epoch-10 replacement-kernel summaries from saved OOF results.

This extension leaves the prior three-arm summary unchanged. The comparison
unit remains one compound, not a fold, predictive draw, or training epoch.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .geometry_kernel_summary import (
    INTERVAL_SCOPE, U_SCOPE, _read_coordinates, _policy, _compare, _interval_text,
)
from .gram_oof_experiment import selection_mask
from .hierarchical_geometry_summary import _folds, _load_arm, _action_rows, _fmt


ARMS = ('A_HR', 'B_MLP', 'C_GENERIC', 'D_STRUCTURED')
COMPARISONS = (('B_MLP', 'A_HR'), ('C_GENERIC', 'A_HR'), ('D_STRUCTURED', 'A_HR'),
               ('C_GENERIC', 'B_MLP'), ('D_STRUCTURED', 'B_MLP'),
               ('D_STRUCTURED', 'C_GENERIC'))


def _validate_manifest(manifest):
    """Validate this run's four-arm identity, boundaries and unique OOF rows."""
    if list(manifest['arms']) != list(ARMS):
        raise ValueError('The replacement stage must report all four declared arms in order')
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
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(ids))):
            raise ValueError('Fit, inner validation and test must partition the 639 IDs')
    for key in ('bootstrap', 'samples'):
        if int(manifest['config'].get(key, 2000)) != 2000:
            raise ValueError('The replacement stage requires 2000 '+key)
    return ids, allocation


def _parameter_counts(checkpoint, arm):
    """Read the runner's explicit count, not checkpoint size or a guessed net."""
    metadata = checkpoint['test_metadata']
    training = checkpoint.get('training_complete') or {}
    counts = (metadata.get('parameter_counts') if arm == 'A_HR'
              else training.get('parameter_counts'))
    if counts is None:
        raise ValueError('Missing explicit parameter_counts for '+arm)
    values = {}
    for key in ('trainable', 'total'):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('Parameter counts must be nonnegative integers')
        values[key] = value
    if values['trainable'] > values['total']:
        raise ValueError('Trainable parameters cannot exceed total parameters')
    return values


def _capacity(models):
    rows = []
    for position in range(5):
        record = dict(fold=models[ARMS[0]]['folds'][position]['fold'])
        for arm in ARMS:
            record[arm] = models[arm]['folds'][position]['parameter_counts']
        rows.append(record)
    return dict(per_fold=rows,
        generic_structured_trainable_equal=all(row['C_GENERIC']['trainable'] == row['D_STRUCTURED']['trainable'] for row in rows),
        generic_structured_total_equal=all(row['C_GENERIC']['total'] == row['D_STRUCTURED']['total'] for row in rows),
        mlp_trainable_smaller_than_generic=all(row['B_MLP']['trainable'] < row['C_GENERIC']['trainable'] for row in rows),
        all_four_arms_capacity_matched=False,
        interpretation=('C_GENERIC versus D_STRUCTURED is the intended equal-parameter basis comparison; '
                        'B_MLP has a different parameter budget and is not a capacity-matched comparator; '
                        'A_HR is frozen, so zero currently trainable parameters does not mean a zero-parameter predictor; '
                        'counts are network parameters from model.parameters(), including the frozen HR network '
                        'but excluding fixed ridge coefficients/intercepts and descriptor anchor/scale buffers'))


def _score_table(result):
    lines = ['| 模型 | 九维 u MSE↓ | ADD_TWO CRPS↓ | NULL Brier↓ | Γ Spearman | NULL AUC |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = result['models'][arm]
        a = m['actions'][2]
        lines.append(f"| {arm} | {_fmt(m['u_mse'])} | {_fmt(a['gamma_crps'])} | "
                     f"{_fmt(a['null_brier'])} | {_fmt(a['spearman'], 4)} | {_fmt(a['null_auc'], 4)} |")
    return lines


def _policy_table(result):
    lines = ['| 模型 | 对象 / 孔 | 每个可选对象净值 | 每个选中对象净值 | FDP | FPR |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        p = result['models'][arm]['principal_policy']
        lines.append(f"| {arm} | {p['selected_n']} / {p['used_wells']} | "
                     f"{_fmt(p['per_eligible_net_gain'])} | {_fmt(p['per_selected_net_gain'])} | "
                     f"{_fmt(p['fdp'], 4)} | {_fmt(p['fpr'], 4)} |")
    return lines


def _capacity_table(result):
    lines = ['| 外折 | A_HR 网络参数：当前可训练 / 合计 | B_MLP | C_GENERIC | D_STRUCTURED |',
             '|---|---:|---:|---:|---:|']
    for row in result['capacity']['per_fold']:
        fields = [f"{row[arm]['trainable']:,} / {row[arm]['total']:,}" for arm in ARMS]
        lines.append(f"| {row['fold']} | "+' | '.join(fields)+' |')
    return lines


def _write_reports(root, result):
    intro = ['# HR、MLP 与两类 kernel：四臂第10轮阶段结果', '',
        'B_MLP、C_GENERIC、D_STRUCTURED 均使用实际第10轮检查点；A_HR 为固定对照。',
        '639个已开放 DEV 对象各贡献一次外折结果，五折并未扩大统计样本量。', '']
    short = intro+_score_table(result)+['', '25%物理孔预算：16/16/16/16/15个对象，共79个对象、158个追加孔。', '']+_policy_table(result)
    short += ['', '## 网络参数量', '']+_capacity_table(result)
    short += ['', '网络参数含冻结 HR 网络，不含固定 ridge 系数/截距及描述量锚点、尺度等缓冲值。',
              'B_MLP 的参数预算不同，不能将四臂称为容量匹配比较。C_GENERIC 与 D_STRUCTURED 的参数量核对见完整报告。',
              '这是开发集十轮阶段结果；六组配对区间、折间方向与选择重叠见 REPORT.md，不能代替独立合同认证。']
    (root/'SUMMARY.md').write_text('\n'.join(short)+'\n')
    lines = intro+['## 预测与概率评分', '']+_score_table(result)
    lines += ['', 'u MSE 在各外折 fit-only 标准化九维坐标中计算，按对象数加权合并。',
        '合并 Γ Spearman 与 NULL AUC 仅作描述；各折的分数来自不同拟合模型，折内秩关联另列于 summary.json。',
        '', '## 同预算价值与风险', '']+_policy_table(result)
    lines += ['', '使用原始 ADD_TWO Γ，已包含两孔成本；没有再次扣费。NULL 为 Γ≤0，POSITIVE 为 Γ≥0.005。',
        '在各折按预测 E[Γ] 排序、精确并列按 ID 排序，名额16/16/16/16/15。FDP 分母为79个选中对象，FPR 分母为全部实际 NULL 对象。',
        '', '## 网络参数数量与比较含义', '']+_capacity_table(result)
    cap = result['capacity']
    lines += ['', '这里计数的是 model.parameters() 中的网络参数：含冻结 HR 网络，不含固定 ridge 系数/截距及描述量锚点、尺度等缓冲值。',
        f"C/D 每折可训练网络参数相同：{cap['generic_structured_trainable_equal']}；网络参数合计相同：{cap['generic_structured_total_equal']}。",
        f"B_MLP 每折可训练参数少于 C_GENERIC：{cap['mlp_trainable_smaller_than_generic']}。",
        'MLP 对照回答不同函数族的表现，不单独隔离参数容量。C/D 才是所设计的等参数基函数比较；若记录不等，不能将差异仅归于基函数。',
        'A_HR 的当前可训练参数为0表示本轮冻结，不表示它是零参数模型。',
        '', '## 六组预先声明的配对比较', '',
        '差值为左臂减右臂。u MSE、CRPS、Brier、FDP/FPR 负值有利；净值正值有利。', '',
        '| 比较 | 指标 | 差值 | 条件式95%区间 |', '|---|---|---:|---|']
    labels = dict(u_mse='九维 u MSE', gamma_crps='ADD_TWO CRPS', null_brier='NULL Brier',
                  net_gain_per_eligible='每个可选对象净值', net_gain_per_selected='每个选中对象净值',
                  fdp='FDP', fpr='FPR')
    for pair in result['comparisons'].values():
        label = pair['left']+' − '+pair['right']
        for key, title in labels.items():
            statistic = pair[key]
            lines.append(f"| {label} | {title} | {_fmt(statistic['mean'])} | {_interval_text(statistic)} |")
    lines += ['', '## 折间方向与选择重叠', '',
        '| 比较 | u MSE 有利/不利/并列 | CRPS 有利/不利/并列 | Brier 有利/不利/并列 | 净值有利/不利/并列 | 共同选择 | Jaccard |',
        '|---|---|---|---|---|---:|---:|']
    for pair in result['comparisons'].values():
        d = pair['fold_directions']
        cells = ['/'.join(str(d[key][name]) for name in ('favorable', 'unfavorable', 'tied'))
                 for key in ('u_mse', 'gamma_crps', 'null_brier', 'net_gain_per_eligible')]
        o = pair['overlap']
        lines.append('| '+pair['left']+' − '+pair['right']+' | '+' | '.join(cells)
                     +f" | {o['intersection_n']} / 79 | {_fmt(o['jaccard'], 4)} |")
    lines += ['', '各折具体差值、选择新增/移除的对象 ID，以及另外两种原动作的指标均保留于 summary.json。',
        '', '## 阶段与统计范围', '',
        '训练记录中的0/5/10轮验证信息和 best_epoch 只进入附录。没有按测试结果选检查点，主结果统一为实际第10轮。',
        '2000次复抽在外折内按唯一对象配对进行，固定当前预测与选择掩码；FDP/FPR 在每次复抽中重新计算分母。',
        '区间未计入模型训练、共享批次、重叠训练集、历史开发选择或 Monte Carlo 积分误差；六组比较未进行多重比较校正。',
        '所有四臂均报告。原收益、七项合同与旧结果保留，FINAL 与第五重复未打开。',
        '十轮结果描述不同修正结构的早期表现，不据此宣称生物机制被识别，也不将一次开发集优势视为独立泛化证据。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    """Read completed saved four-arm outputs and write this run's summaries."""
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = _validate_manifest(manifest)
    data, models = {}, {}
    actual_reference = u_reference = None
    for arm in ARMS:
        arrays, fold_rows = _load_arm(root, manifest, arm, ids)
        actual_u, mean_u, u_rows = _read_coordinates(root, manifest['folds'], ids, arm)
        if actual_reference is None:
            actual_reference, u_reference = arrays['actual'].copy(), actual_u.copy()
        elif not np.array_equal(actual_reference, arrays['actual']) or not np.array_equal(u_reference, actual_u):
            raise ValueError('Four paired arms must have identical original Gamma and nine-u targets')
        mask = selection_mask(arrays['predicted'][:, 2], ids, allocation, .25, 2)
        counts = [int(mask[np.asarray(record['test'], int)].sum()) for record in manifest['folds']]
        if sorted(counts) != [15, 16, 16, 16, 16] or int(mask.sum()) != 79:
            raise ValueError('The foldwise 25% physical-well cap must select 79 objects')
        arrays.update(actual_u=actual_u, mean_u=mean_u, principal_mask=mask,
                      u_object_mse=np.square(actual_u-mean_u).mean(1))
        for row, u_row, record in zip(fold_rows, u_rows, manifest['folds']):
            ix = np.asarray(record['test'], int)
            metric_path = root/'folds'/f"fold_{record['fold']}"/'arms'/arm/'test'/'metrics.json'
            metric = json.loads(metric_path.read_text())
            if metric.get('samples') != 2000:
                raise ValueError('Saved test scores must use the declared 2000 predictive draws')
            row.update(u=u_row, principal_policy=_policy(arrays['actual'][ix, 2], mask[ix]),
                       parameter_counts=_parameter_counts(u_row['checkpoint'], arm))
        models[arm] = dict(u_mse=float(arrays['u_object_mse'].mean()), u_scope=U_SCOPE,
            per_coordinate_u_mse=np.square(actual_u-mean_u).mean(0).tolist(),
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
        n=len(ids), arms=list(ARMS), models=models, comparisons=comparisons, capacity=_capacity(models),
        fold_test_counts=[len(record['test']) for record in manifest['folds']],
        actual_checkpoint_epoch=10, frozen_comparator='A_HR',
        principal_action='original ADD_TWO / Z1Z2', principal_budget='25% physical wells, foldwise',
        principal_ranking='expected Gamma', samples_per_object=2000,
        bootstrap_replicates=2000, bootstrap_seed=seed, interval_scope=INTERVAL_SCOPE,
        secondary_comparisons_multiplicity_adjusted=False, historical_dev=True,
        formal_certificate=False, final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False)
    for arm in ARMS:
        np.savez_compressed(root/f'{arm}_oof_predictions.npz', ids=ids, fold=allocation, **data[arm])
    write_json(root/'summary.json', result)
    _write_reports(root, result)
    return result
