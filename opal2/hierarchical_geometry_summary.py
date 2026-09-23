"""Summarize fixed-role geometry refinements from saved paired OOF predictions.

This module fits no model and chooses no checkpoint or operating point. The
original four baselines are reused predictions. New comparisons remain a
development experiment on conditional nine-dimensional geometry, not a
cross-site biological world-model validation or statistical authorization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .baseline_policy import ACTIONS, WELL_COSTS, _metrics
from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import selection_mask


BASELINES = ('GLOBAL_GEOMETRY', 'RIDGE_GEOMETRY', 'G_DIRECT', 'L_GRAM')
DECLARED_COMPARISONS = {
    'primary': ('HR_OOF_COV', 'RIDGE_GEOMETRY'),
    'isolated_mean': ('HR_RIDGE_COV', 'RIDGE_GEOMETRY'),
    'isolated_G_covariance': ('G_OOF_COV', 'G_DIRECT'),
    'isolated_HR_covariance': ('HR_OOF_COV', 'HR_RIDGE_COV'),
}
INTERVAL_SCOPE = (
    'paired compound bootstrap within outer folds, fixed fitted predictions/masks; '
    'does not include model-search, Monte Carlo integration, overlapping-training '
    'or shared-batch uncertainty; not a formal certificate'
)


def _folds(manifest):
    ids = np.asarray(manifest['ids'], dtype=str)
    if not len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Each OOF compound must have a unique ID')
    allocation = np.full(len(ids), -1, dtype=int)
    counts = np.zeros(len(ids), dtype=int)
    seen = set()
    for record in manifest['folds']:
        fold = record['fold']
        ix = np.asarray(record['test'], dtype=int)
        if (fold in seen or not len(ix) or len(set(ix.tolist())) != len(ix)
                or np.any(ix < 0) or np.any(ix >= len(ids))):
            raise ValueError('Invalid or duplicate outer-test fold')
        seen.add(fold)
        allocation[ix] = fold
        counts[ix] += 1
    if not np.array_equal(counts, np.ones(len(ids), dtype=int)):
        raise ValueError('Each compound must occur in exactly one outer-test fold')
    if len(seen) != int(manifest['config']['folds']):
        raise ValueError('Fold count does not match the recorded experiment')
    return ids, allocation


def _optional_diagnostics(arm_folder):
    result = {}
    for name in ('u_diagnostics.json', 'training_diagnostics.json', 'training_complete.json'):
        for parent in (arm_folder, arm_folder/'test'):
            path = parent/name
            if path.is_file():
                result[str(path.relative_to(arm_folder))] = json.loads(path.read_text())
    return result


def _load_arm(root, manifest, arm, ids):
    arrays, fold_rows = {}, []
    for record in manifest['folds']:
        arm_folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
        folder = arm_folder/'test'
        metric = json.loads((folder/'metrics.json').read_text())
        ix = np.asarray(record['test'], dtype=int)
        with np.load(folder/'predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError(f'Prediction IDs differ in {arm}/fold {record["fold"]}')
            for key in ('actual', 'predicted', 'p_null', 'utility_crps', 'geometry_energy'):
                values = np.asarray(saved[key], dtype=np.float64)
                expected = (len(ix),) if key == 'geometry_energy' else (len(ix), len(ACTIONS))
                if values.shape != expected or not np.isfinite(values).all():
                    raise ValueError(f'Invalid saved {key} for {arm}')
                if key not in arrays:
                    arrays[key] = np.empty((len(ids), *values.shape[1:]), dtype=np.float64)
                arrays[key][ix] = values
            if np.any(saved['p_null'] < 0) or np.any(saved['p_null'] > 1):
                raise ValueError('Saved NULL probabilities must lie in [0,1]')
        fold_rows.append(dict(fold=record['fold'], n=len(ix), actions=metric['action_metrics'],
            crps=[item['crps'] for item in metric['utility']],
            diagnostics=_optional_diagnostics(arm_folder)))
    return arrays, fold_rows


def _action_rows(arrays, records, *, global_model):
    rows, n = [], len(arrays['actual'])
    for j, name in enumerate(ACTIONS):
        row = _metrics(arrays['actual'][:, j], arrays['predicted'][:, j], arrays['p_null'][:, j])
        row.update(action=name, gamma_crps=float(arrays['utility_crps'][:, j].mean()))
        rank_a, rank_p = np.zeros(n), np.zeros(n)
        for record in records:
            ix = np.asarray(record['test'], dtype=int)
            a = rankdata(arrays['actual'][ix, j])
            p = rankdata(arrays['predicted'][ix, j])
            rank_a[ix], rank_p[ix] = (a-a.mean())/len(ix), (p-p.mean())/len(ix)
        association = (float(np.corrcoef(rank_a, rank_p)[0, 1])
                       if np.std(rank_a) > 0 and np.std(rank_p) > 0 else None)
        row['within_fold_rank_association'] = association
        row['pooled_rank_scope'] = 'descriptive only; different fitted rules across outer folds'
        if global_model:
            row.update(pearson=None, spearman=None, null_auc=None, within_fold_rank_association=None)
        rows.append(row)
    return rows


def _policy_rows(arrays, ids, allocation, *, global_model):
    rows, masks, n = [], [], len(ids)
    for j, name in enumerate(ACTIONS):
        actual = arrays['actual'][:, j]
        null, positive = actual <= 0, actual >= .005
        for fraction in (.05, .10, .25):
            for ranking, score in (('expected_gain', arrays['predicted'][:, j]),
                                   ('lowest_p_null', -arrays['p_null'][:, j])):
                for section in ('common_budget', 'within_action'):
                    mask = selection_mask(score, ids, allocation, fraction, WELL_COSTS[j],
                        global_model=global_model, within_action=section == 'within_action')
                    k = float(mask.sum())
                    gain, false = float(mask@actual), float(mask@null)
                    rows.append(dict(action=name, fraction=fraction, ranking=ranking, section=section,
                        selected_n=int(round(k)), used_wells=int(round(k*WELL_COSTS[j])),
                        total_net_gain=gain, per_eligible_net_gain=gain/n,
                        per_selected_net_gain=gain/k if k else None,
                        selected_null_count=false, fdp=false/k if k else None,
                        fpr=false/int(null.sum()) if null.any() else None,
                        sensitivity=float(mask@positive/positive.sum()) if positive.any() else None,
                        selection_interpretation=('uniform-subset expectation within each fold'
                            if global_model else 'fixed within-fold selected masks'),
                        budget_scope='sum of foldwise '+('physical-well caps' if section == 'common_budget'
                                                        else 'selected-compound counts'),
                        formal_certificate=False))
                    masks.append(mask)
    return rows, np.asarray(masks)


def _bootstrap_statistic(value, boot):
    value = np.asarray(value, dtype=np.float64)
    means = np.mean(value[boot], axis=1)
    return dict(mean=np.mean(value, axis=0).tolist(),
                interval95=np.quantile(means, [.025, .975], axis=0).tolist())


def _paired(left, right, data, results, boot):
    a, b = data[left], data[right]
    if not np.array_equal(a['actual'], b['actual']):
        raise ValueError('Paired comparisons require identical actual outcomes')
    null = a['actual'] <= 0
    delta = a['utility_crps']-b['utility_crps']
    brier = (a['p_null']-null)**2-(b['p_null']-null)**2
    policies = []
    for k, row in enumerate(results[left]['policies']):
        other = results[right]['policies'][k]
        fields = ('action', 'fraction', 'ranking', 'section')
        if any(row[field] != other[field] for field in fields):
            raise ValueError('Paired policy masks have different action/budget identities')
        j = ACTIONS.index(row['action'])
        weights = a['masks'][k]-b['masks'][k]
        policies.append(dict(**{field: row[field] for field in fields},
            net_gain_per_eligible=_bootstrap_statistic(weights*a['actual'][:, j], boot),
            false_activation_per_eligible=_bootstrap_statistic(weights*null[:, j], boot)))
    return dict(left=left, right=right, direction='left minus right',
                gamma_crps=_bootstrap_statistic(delta, boot),
                null_brier=_bootstrap_statistic(brier, boot), policy=policies,
                formal_certificate=False)


def _principal_policy(rows):
    return next(row for row in rows if row['action'] == 'Z1Z2' and row['fraction'] == .25
                and row['ranking'] == 'expected_gain' and row['section'] == 'common_budget')


def _fmt(value, digits=6):
    return '—' if value is None else f'{value:.{digits}f}'


def summarize(root):
    """Read completed per-fold predictions and write this new run's summary.

    Configuration, arms and compound allocation come from the run manifest.
    Nothing is fit, resampled from a model, selected or read from another cohort.
    """
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    for flag in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed', 'original_contract_changed'):
        if manifest.get(flag) is not False:
            raise ValueError('The recorded experiment does not preserve '+flag)
    ids, allocation = _folds(manifest)
    arms = list(manifest['arms'])
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('The manifest must declare unique model arms')
    required = set(itertools.chain.from_iterable(DECLARED_COMPARISONS.values()))
    if not required.issubset(arms):
        raise ValueError('A predeclared primary/ablation arm is missing')
    data, results = {}, {}
    actual_reference = None
    for arm in arms:
        arrays, fold_rows = _load_arm(root, manifest, arm, ids)
        if actual_reference is None:
            actual_reference = arrays['actual'].copy()
        elif not np.array_equal(arrays['actual'], actual_reference):
            raise ValueError('Arms use different original outcomes')
        actions = _action_rows(arrays, manifest['folds'], global_model=arm == 'GLOBAL_GEOMETRY')
        policies, masks = _policy_rows(arrays, ids, allocation, global_model=arm == 'GLOBAL_GEOMETRY')
        arrays['masks'] = masks
        np.savez_compressed(root/f'{arm}_oof_predictions.npz', ids=ids, fold=allocation, **arrays)
        data[arm] = arrays
        results[arm] = dict(actions=actions, folds=fold_rows, policies=policies,
            common_budget=[row for row in policies if row['section'] == 'common_budget'],
            within_action=[row for row in policies if row['section'] == 'within_action'],
            geometry_energy=float(arrays['geometry_energy'].mean()),
            baseline_predictions_reused=arm in BASELINES)
    cfg = manifest['config']
    count = int(cfg['bootstrap'])
    if count < 2:
        raise ValueError('At least two bootstrap replicates are required')
    rng = np.random.default_rng(int(cfg['seed']))
    boot = np.column_stack([rng.choice(np.flatnonzero(allocation == fold),
        size=(count, int((allocation == fold).sum()))) for fold in sorted(np.unique(allocation))])
    paired = {left+'__minus__'+right: _paired(left, right, data, results, boot)
              for left, right in itertools.combinations(arms, 2)}
    declared = {name: _paired(left, right, data, results, boot)
                for name, (left, right) in DECLARED_COMPARISONS.items()}
    result = dict(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(), n=len(ids),
        arms=arms, models=results, paired=paired, declared_comparisons=declared,
        primary_comparison='HR_OOF_COV minus RIDGE_GEOMETRY; original ADD_TWO Gamma CRPS',
        principal_policy='ADD_TWO expected-gain ranking at 25% physical-well cap, applied separately in each outer fold',
        model_scope='fixed four-role nine-coordinate conditional geometry; not a full cross-site biological world model',
        original_baselines_reused=[arm for arm in arms if arm in BASELINES],
        bootstrap_replicates=count, bootstrap_seed=int(cfg['seed']),
        samples_per_object=cfg.get('samples'), interval_scope=INTERVAL_SCOPE,
        secondary_comparisons_multiplicity_adjusted=False,
        final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False,
        original_contract_changed=False, historical_dev=True, formal_certificate=False)
    write_json(root/'summary.json', result)

    lines = [f'# {len(ids)} 对象：均值修正与协方差替换的折外比较', '',
        '本轮是固定四角色、九维条件几何模型的比较，不是完整跨站点生物世界模型验证。',
        '原四臂复用既有折外预测；新臂使用同一划分和原始收益。FINAL 与第五重复未打开。', '',
        '| 模型 | ADD_TWO CRPS↓ | NULL Brier↓ | 合并相关（描述性） | 折内秩关联 |',
        '|---|---:|---:|---:|---:|']
    for arm in arms:
        row = results[arm]['actions'][2]
        lines.append(f"| {arm} | {_fmt(row['gamma_crps'])} | {_fmt(row['null_brier'])} | "
                     f"{_fmt(row['spearman'])} | {_fmt(row['within_fold_rank_association'])} |")
    lines += ['', '## 预先指定的比较', '',
        'CRPS 差为左模型减右模型；负值有利于左模型。区间为已拟合开发结果的条件式配对区间。', '',
        '| 比较 | ADD_TWO CRPS 差 | 95% 区间 |', '|---|---:|---|']
    for name, comparison in declared.items():
        score = comparison['gamma_crps']
        lines.append(f"| {name}: {comparison['left']} − {comparison['right']} | {_fmt(score['mean'][2])} | "
                     f"[{_fmt(score['interval95'][0][2])}, {_fmt(score['interval95'][1][2])}] |")
    lines += ['', '## 25% 物理孔预算：ADD_TWO，按期望收益选择', '',
              '| 模型 | 对象/孔 | 每选中对象净收益 | FDP | FPR |', '|---|---:|---:|---:|---:|']
    for arm in arms:
        row = _principal_policy(results[arm]['policies'])
        lines.append(f"| {arm} | {row['selected_n']}/{row['used_wells']} | {_fmt(row['per_selected_net_gain'])} | "
                     f"{_fmt(row['fdp'], 4)} | {_fmt(row['fpr'], 4)} |")
    lines += ['', '主预算的配对净值差（每个可选对象），并列报告风险点估计而不移除纯度要求：', '',
              '| 比较 | 每可选对象净值差 | 95% 区间 |', '|---|---:|---|']
    for name, comparison in declared.items():
        score = _principal_policy(comparison['policy'])['net_gain_per_eligible']
        lines.append(f"| {name} | {_fmt(score['mean'])} | [{_fmt(score['interval95'][0])}, {_fmt(score['interval95'][1])}] |")
    lines += ['', '所有动作、5%/10%/25% 两类预算、配对概率评分和错误激活差异见 summary.json。',
        'GLOBAL 采用折内均匀子集期望，不以折间常数差或 ID 进行个体选择。',
        'HR_RIDGE_COV 与 RIDGE 隔离均值修正；HR_OOF_COV 与 HR_RIDGE_COV 隔离协方差替换；G_OOF_COV 与 G_DIRECT 隔离 G 的协方差替换。',
        '这些区间不覆盖训练集重叠、共享批次、历史开发搜索或 Monte Carlo 积分误差；没有进行新的独立统计认证。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result
