"""Read the complete, fixed-epoch independent-biology experiment.

Only saved outer-fold predictions are read. This module cannot select a model,
checkpoint, threshold, favorable fold, or new reference bank. The two fitted
branches are compared with their unchanged, saved A comparator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .conditional_response_summary import ARRAY_SHAPES, _model_scores
from .geometry_kernel_summary import _policy
from .gram_oof_experiment import selection_mask
from .hierarchical_geometry_summary import _folds
from .lincs_biology_summary import bootstrap_weights, compare, interval


ARMS = ('A_FROZEN', 'A_PLUS_OLD', 'A_PLUS_BIO')
PAIRS = (('A_PLUS_BIO', 'A_PLUS_OLD'), ('A_PLUS_BIO', 'A_FROZEN'),
         ('A_PLUS_OLD', 'A_FROZEN'))
SAMPLES = 10000
EPOCHS = 30
BOOTSTRAP = 2000
SCOPE = ('paired bootstrap of fixed chemistry-group OOF predictions; layout groups '
         'are resampled separately as a sensitivity analysis; intervals exclude '
         'refitting, repeated development selection, and Monte Carlo uncertainty; '
         'not an independent certification or cross-batch deployment evaluation')


def _finite(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError('Expected finite aligned '+name)
    return value.copy()


def validate_manifest(manifest):
    if manifest.get('arms') != list(ARMS) or manifest.get('fixed_epochs') != EPOCHS:
        raise ValueError('Expected all three declared arms at fixed epoch30')
    ids, allocation = _folds(manifest)
    if len(manifest['folds']) != 5:
        raise ValueError('All five outer folds are required')
    groups = np.asarray(manifest['groups'], dtype=str)
    if groups.shape != ids.shape:
        raise ValueError('Chemistry groups must align with eligible IDs')
    for group in np.unique(groups):
        if len(np.unique(allocation[groups == group])) != 1:
            raise ValueError('A chemistry group crosses outer test folds')
    units = manifest['dataset']['units']
    if len(units) != len(ids) or any('layout_block' not in row for row in units):
        raise ValueError('Each eligible object needs a declared layout block')
    if manifest['config'].get('samples', SAMPLES) != SAMPLES:
        raise ValueError('The declared Monte Carlo count must be 10000')
    for key in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed',
                'original_contract_changed'):
        if manifest.get(key, False) is not False:
            raise ValueError('Protected experiment scope changed: '+key)
    return ids, allocation


def _read_diagnostics(path, expected_ids, expected_mean, *, required):
    if not path.is_file():
        if required:
            raise ValueError('Missing fitted-branch diagnostics: '+str(path))
        return {}
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['ids'], expected_ids):
            raise ValueError('Diagnostic identity/order mismatch')
        n = len(expected_ids)
        result = dict(baseline_mean=_finite(saved['baseline_mean'], (n, 9), 'baseline_mean'),
                      mean=_finite(saved['mean'], (n, 9), 'diagnostic mean'))
        if not np.array_equal(result['mean'], expected_mean):
            raise ValueError('Diagnostic mean differs from evaluated coordinates')
        if required:
            support = np.asarray(saved['support'])
            if support.shape != (n, 2) or not np.isin(support, [False, True]).all():
                raise ValueError('Two explicit binary support channels are required')
            result['support'] = support.astype(bool)
        for key in ('known', 'channel_gate', 'effective_neighbors', 'support_features',
                    'block_contributions', 'raw_increment'):
            if key in saved:
                value = np.asarray(saved[key])
                if value.ndim < 1 or len(value) != n or not np.isfinite(value).all():
                    raise ValueError('Invalid object-aligned diagnostic: '+key)
                result[key] = value.copy()
        if 'block_contributions' in result:
            if result['block_contributions'].shape != (n, 2, 9):
                raise ValueError('Two nine-coordinate branch contributions are required')
            if 'raw_increment' in result and not np.allclose(
                    result['block_contributions'].sum(1), result['raw_increment'],
                    rtol=1e-6, atol=1e-8):
                raise ValueError('Branch contributions do not sum to raw increment')
    return result


def read_arm(root, manifest, ids, allocation, arm):
    n = len(ids)
    arrays = {key: np.empty((n, *tail)) for key, tail in ARRAY_SHAPES.items()}
    arrays.update(actual_u=np.empty((n, 9)), mean_u=np.empty((n, 9)))
    diagnostics, rows = {}, []
    for record in manifest['folds']:
        ix = np.asarray(record['test'], dtype=int)
        fold = root/'folds'/f"fold_{record['fold']}"
        if not (fold/'complete.json').is_file():
            raise ValueError(f"Fold {record['fold']} is not complete")
        folder = fold/'arms'/arm
        evaluation = folder/'evaluation'
        metric = json.loads((evaluation/'metrics.json').read_text())
        if metric.get('samples') != SAMPLES:
            raise ValueError('Each saved evaluation must use 10000 samples')
        with np.load(evaluation/'predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Prediction identity/order mismatch')
            for key, tail in ARRAY_SHAPES.items():
                arrays[key][ix] = _finite(saved[key], (len(ix), *tail), key)
            draws = _finite(saved['utility_samples'], (SAMPLES, len(ix), 3), 'all utility draws')
            if not np.allclose(draws.mean(0), saved['predicted'], rtol=1e-12, atol=1e-14):
                raise ValueError('Saved predictive means do not use all 10000 draws')
            if not np.array_equal((draws <= 0).mean(0), saved['p_null']):
                raise ValueError('Saved NULL probabilities do not use all 10000 draws')
        with np.load(evaluation/'u_predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Coordinate identity/order mismatch')
            for key in ('actual_u', 'mean_u'):
                arrays[key][ix] = _finite(saved[key], (len(ix), 9), key)
        row = dict(fold=record['fold'], n=len(ix))
        if arm != 'A_FROZEN':
            completion = json.loads((folder/'training_complete.json').read_text())
            count = completion.get('trainable_parameters')
            if completion.get('actual_checkpoint_epoch') != EPOCHS:
                raise ValueError('Fitted arms must use actual epoch30, not a selected checkpoint')
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError('A fitted arm must record its actual active parameter count')
            history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()
                       if line.strip()]
            if not history or not any(entry.get('epoch') == EPOCHS for entry in history):
                raise ValueError('Fitted arm is missing its epoch30 history')
            row.update(training=completion, history=history,
                       validation_history_is_descriptive_not_checkpoint_selection=True)
        numeric = _read_diagnostics(folder/'model_diagnostics.npz', ids[ix], arrays['mean_u'][ix],
                                    required=arm != 'A_FROZEN')
        for key, value in numeric.items():
            if key not in diagnostics:
                diagnostics[key] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
            if diagnostics[key].shape[1:] != value.shape[1:]:
                raise ValueError('Diagnostic dimensions changed between folds: '+key)
            diagnostics[key][ix] = value
        rows.append(row)
    if np.any(arrays['p_null'] < 0) or np.any(arrays['p_null'] > 1):
        raise ValueError('NULL probabilities outside [0,1]')
    arrays.update(principal_mask=selection_mask(arrays['predicted'][:, 2], ids, allocation, .25, 2),
                  u_object_mse=np.square(arrays['actual_u']-arrays['mean_u']).mean(1))
    return arrays, rows, diagnostics


def _subset_scores(arrays, subset):
    if not np.any(subset):
        return dict(n=0)
    actual = arrays['actual'][subset, 2]
    return dict(n=int(subset.sum()), u_mse=float(arrays['u_object_mse'][subset].mean()),
        gamma_crps=float(arrays['utility_crps'][subset, 2].mean()),
        null_brier=float(np.square(arrays['p_null'][subset, 2]-(actual <= 0)).mean()),
        policy_restricted_from_full_cohort=_policy(actual, arrays['principal_mask'][subset]),
        selection_not_reranked_within_subset=True)


def support_checks(data, diagnostics, weights):
    baseline = data['A_FROZEN']['mean_u']
    for arm in ARMS[1:]:
        if not np.array_equal(diagnostics[arm]['baseline_mean'], baseline):
            raise ValueError('A fitted branch does not preserve the identical saved A baseline')
    bio = diagnostics['A_PLUS_BIO']
    support = bio['support'].any(1)
    unsupported = ~support
    diff = data['A_PLUS_BIO']['mean_u']-baseline
    exact = np.array_equal(data['A_PLUS_BIO']['mean_u'][unsupported], baseline[unsupported])
    if not exact:
        raise ValueError('Biology changed the baseline on objects without biological support')
    for key in ARRAY_SHAPES:
        if not np.array_equal(data['A_PLUS_BIO'][key][unsupported], data['A_FROZEN'][key][unsupported]):
            raise ValueError('Unsupported biology predictions/scores differ from the shared-sample baseline: '+key)
    subsets = {}
    for name, subset in (('supported', support), ('unsupported', unsupported)):
        scores = {arm: _subset_scores(data[arm], subset) for arm in ARMS}
        comparisons = {}
        if subset.any():
            denominator = subset.astype(float)
            for left, right in PAIRS:
                a, b = data[left], data[right]
                null = a['actual'][:, 2] <= 0
                quantities = dict(
                    u_mse=a['u_object_mse']-b['u_object_mse'],
                    gamma_crps=a['utility_crps'][:, 2]-b['utility_crps'][:, 2],
                    null_brier=np.square(a['p_null'][:, 2]-null)-np.square(b['p_null'][:, 2]-null))
                comparisons[left+'__minus__'+right] = {
                    key: interval(value*denominator, denominator, weights)
                    for key, value in quantities.items()}
        subsets[name] = dict(models=scores, comparisons=comparisons)
    return dict(supported_n=int(support.sum()), unsupported_n=int(unsupported.sum()),
        channel_supported_n=bio['support'].sum(0).astype(int).tolist(),
        exact_zero_change_without_support=exact,
        unsupported_max_absolute_mean_change=float(np.abs(diff[unsupported]).max()) if unsupported.any() else 0.,
        supported_mean_change_rms=float(np.sqrt(np.square(diff[support]).mean())) if support.any() else 0.,
        subset_definition='positive target or MoA support in the fixed TRAIN reference bank; identical BIO-defined subset for every arm',
        diagnostics_are_not_evidence_of_causal_mechanism=True, subsets=subsets)


def _fixed_and_random(actual, manifest, seed):
    fixed = dict(STOP=dict(added_wells=0, selected_n=0, mean_net_gain=0., total_net_gain=0.))
    for j, name in enumerate(('ALL_Z1', 'ALL_Z2', 'ALL_Z1Z2')):
        value = actual[:, j]
        fixed[name] = dict(added_wells=len(value)*(2 if j == 2 else 1), selected_n=len(value),
            mean_net_gain=float(value.mean()), total_net_gain=float(value.sum()),
            null_count=int((value <= 0).sum()), null_rate=float((value <= 0).mean()))
    rng = np.random.default_rng(seed)
    gains, nulls = [], []
    for _ in range(BOOTSTRAP):
        chosen = []
        for record in manifest['folds']:
            ix = np.asarray(record['test'], dtype=int)
            k = int(np.floor(.25*len(ix)))//2
            chosen.extend(rng.choice(ix, k, replace=False).tolist())
        values = actual[chosen, 2]
        gains.append(float(values.mean()))
        nulls.append(int((values <= 0).sum()))
    random = dict(repeats=BOOTSTRAP, selected_n=len(chosen), used_wells=2*len(chosen),
        mean_selected_gamma=float(np.mean(gains)),
        selected_gamma_interval95=np.quantile(gains, [.025, .975]).tolist(),
        expected_null_count=float(np.mean(nulls)), null_count_interval95=np.quantile(nulls, [.025, .975]).tolist(),
        interval_scope='random allocation distribution conditional on the observed cohort, not a confidence interval')
    return fixed, random


def _format_stat(stat):
    if stat['estimate'] is None or stat['interval95'] is None:
        return 'NA'
    low, high = stat['interval95']
    return f"{stat['estimate']:+.6f} [{low:+.6f}, {high:+.6f}]"


def _report(root, result):
    models = result['models']
    lines = ['# LINCS：独立生物分支，固定第30轮比较', '',
        f"{result['n']}个对象、五个原化学分组折；两个新分支各训练30轮，旧A直接复用保存结果。",
        'A_FROZEN：冻结HR及旧信息路径。A_PLUS_OLD：增加独立旧信息修正。A_PLUS_BIO：增加独立靶点/MoA修正。',
        '新增两臂在每折具有相同可训练参数量。共同参考库、训练预算与联合误差保持不变，以隔离分支接入的影响。', '',
        '| 模型 | 几何MSE↓ | Γ fair CRPS↓ | NULL Brier↓ | Γ Spearman | 选中对象/新增孔 | 选中实际Γ↑ | NULL数 | FDP↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        model = models[arm]
        action, policy = model['actions'][2], model['principal_policy']
        rho = 'NA' if action['spearman'] is None else f"{action['spearman']:.4f}"
        lines.append(f"|{arm}|{model['u_mse']:.6f}|{action['gamma_crps']:.6f}|{action['null_brier']:.6f}|"
            f"{rho}|{policy['selected_n']}/{policy['used_wells']}|{policy['per_selected_net_gain']:.6f}|"
            f"{policy['selected_null_count']}|{policy['fdp']:.3%}|")
    lines += ['', '预算为每折对象数25%的额外物理孔；ADD_TWO每对象使用两孔。不是选择25%的对象。',
        'Γ已包含原追加成本，不重复扣费。NULL仍为Γ≤0，原风险要求没有撤掉。', '',
        '## 配对比较', '', '差为左减右；误差下降有利、选中实际收益上升有利。', '',
        '| 对比 | 几何MSE差（95%区间） | Γ CRPS差（95%区间） | 选中Γ差（95%区间） | FDP差（95%区间） |',
        '|---|---|---|---|---|']
    for left, right in PAIRS:
        row = result['comparisons'][left+'__minus__'+right]
        cells = [_format_stat(row[key]) for key in ('u_mse', 'gamma_crps', 'net_gain_per_selected', 'fdp')]
        lines.append('|'+left+'−'+right+'|'+'|'.join(cells)+'|')
    support = result['support_analysis']
    random = result['same_budget_random']
    lines += ['', '区间跨零意味着本轮无法稳定区分，不把单个对象替换或单项几何改善称为选孔成功。', '',
        '## 独立性和支持', '',
        f"有靶点或MoA正关系支持：{support['supported_n']}/{result['n']}；无支持：{support['unsupported_n']}。",
        f"无支持对象精确恢复旧A：{support['exact_zero_change_without_support']}；均值最大变化为{support['unsupported_max_absolute_mean_change']:.3g}。",
        'supported/unsupported子集对所有模型使用同一份生物支持名单。子集选择由全队列策略限制得到，不在子集里重新排序。',
        '每个子集的几何、Γ CRPS、Brier及配对差值存于summary.json；门控或非零读出不是生物机制有效的证明。', '',
        '## 随机及固定策略', '',
        f"同预算随机：{random['selected_n']}个对象/{random['used_wells']}孔；选中Γ均值{random['mean_selected_gamma']:.6f}，平均NULL数{random['expected_null_count']:.2f}。",
        '随机范围是固定观测队列下的分配分布，不是模型泛化置信区间。', '',
        '| 固定策略 | 新增孔数 | 每对象实际净收益 | NULL数 |', '|---|---:|---:|---:|']
    for name, row in result['fixed_plans'].items():
        lines.append(f"|{name}|{row['added_wells']}|{row['mean_net_gain']:.6f}|{row.get('null_count', '不适用')}|")
    lines += ['', '这些固定计划成本不同，不冒充同预算对照；完整结果同时保存总体净值及每个对象净值。', '',
        '## 训练轨迹与解释范围', '',
        '本表使用真实第30轮，验证最优点只用于说明轨迹，不从外折结果挑检查点。各折训练/验证记录及参数量在summary.json。',
        '本轮联合误差仍冻结为原比较中的协方差。若均值修正有效，下一步才用对应完整流程的折外残差检查协方差和尾部。',
        '化学分组配对区间和板布局分组敏感性分别报告；二者都没有包含重新拟合及反复开发选择的不确定性。',
        '固定孔位、共享板和布局仍限制跨批次解释。这是已使用开发数据上的模型比较，不是新的独立认证。',
        '本轮没有更换Γ、NULL阈值、成本、原七项合同，也没有打开受保护FINAL或第五重复。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(output):
    root = Path(output)
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = validate_manifest(manifest)
    data, models, diagnostics = {}, {}, {}
    for arm in ARMS:
        data[arm], rows, diagnostics[arm] = read_arm(root, manifest, ids, allocation, arm)
        if arm != 'A_FROZEN':
            for key in ('actual', 'actual_u'):
                if not np.array_equal(data[arm][key], data['A_FROZEN'][key]):
                    raise ValueError('Arms have different realized targets: '+key)
        models[arm] = dict(_model_scores(data[arm], ids, manifest['folds']), folds=rows)
    for old, bio in zip(models['A_PLUS_OLD']['folds'], models['A_PLUS_BIO']['folds']):
        if old['training']['trainable_parameters'] != bio['training']['trainable_parameters']:
            raise ValueError('The old-information and biology controls have unequal active capacity')
    seed = int(manifest['config']['seed'])
    weights = bootstrap_weights(manifest['groups'], BOOTSTRAP, seed+701)
    comparisons = {a+'__minus__'+b: compare(a, b, data, weights, manifest['folds']) for a, b in PAIRS}
    layout = [str(unit['layout_block']) for unit in manifest['dataset']['units']]
    layout_weights = bootstrap_weights(layout, BOOTSTRAP, seed+702)
    layout_comparisons = {a+'__minus__'+b: compare(a, b, data, layout_weights, manifest['folds']) for a, b in PAIRS}
    support = support_checks(data, diagnostics, weights)
    actual = data['A_FROZEN']['actual']
    fixed, random = _fixed_and_random(actual, manifest, seed+703)
    result = dict(complete=True, n=len(ids), arms=list(ARMS), fixed_epoch=EPOCHS,
        samples=SAMPLES, models=models, comparisons=comparisons,
        layout_block_sensitivity=layout_comparisons, support_analysis=support,
        baseline_population=dict(mean_gamma=float(actual[:, 2].mean()),
            null_count=int((actual[:, 2] <= 0).sum()), null_rate=float((actual[:, 2] <= 0).mean()),
            positive_count=int((actual[:, 2] >= .005).sum())),
        fixed_plans=fixed, same_budget_random=random, uncertainty_scope=SCOPE,
        formal_certificate=False, original_endpoint_changed=False, original_contract_changed=False)
    # Nothing is written until every fold, arm, target, and identity check passes.
    for arm in ARMS:
        diagnostic = {f'diagnostic_{key}': value for key, value in diagnostics[arm].items()}
        np.savez_compressed(root/f'{arm}_oof.npz', ids=ids, fold=allocation, **data[arm], **diagnostic)
    _report(root, result)
    write_json(root/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    summarize(parser.parse_args().output)
