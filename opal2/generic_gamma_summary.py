"""Saved-output factorial comparison of generic/structured bases and Gamma loss.

Only the generic-plus-Gamma arm is newly trained. Existing four-arm predictions
must match their declared sources before any summary is written.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .conditional_response_summary import (
    INTERVAL_SCOPE, _aggregate_diagnostics, _finish, _model_scores, _read_new_arm, _table,
)
from .gamma_supervised_summary import _training_trajectory, _verify_reference
from .geometry_kernel_continuation_summary import _comparison_lines
from .geometry_kernel_summary import _compare, _policy
from .hierarchical_geometry_summary import _bootstrap_statistic, _folds, _fmt


ARMS = ('A_HR', 'F_CONDITIONAL_GENERIC', 'J_GEOMETRY_CONTROL',
        'K_GEOMETRY_GAMMA_CRPS', 'M_CONDITIONAL_GENERIC_GAMMA')
NEW_ARM = ARMS[-1]
COMPARISONS = ((ARMS[3], NEW_ARM), (NEW_ARM, ARMS[1]), (ARMS[3], ARMS[2]), (NEW_ARM, ARMS[0]))
SOURCE_KEYS = ('gamma_reference_run', 'conditional_reference_run', 'sampling_stability_run')


def _validate_manifest(manifest):
    if list(manifest['arms']) != list(ARMS) or manifest.get('new_arms') != [NEW_ARM]:
        raise ValueError('Expected five declared arms with only M newly trained')
    for key in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed', 'original_contract_changed'):
        if manifest.get(key) is not False:
            raise ValueError('Experiment does not preserve '+key)
    ids, allocation = _folds(manifest)
    if len(ids) != 639 or len(manifest['folds']) != 5:
        raise ValueError('Expected 639 unique DEV objects and five outer folds')
    for record in manifest['folds']:
        parts = [np.asarray(record[k], int) for k in ('fit', 'inner_validation', 'test')]
        if any(not len(x) or len(np.unique(x)) != len(x) or np.any(x < 0) or np.any(x >= len(ids)) for x in parts):
            raise ValueError('Invalid fit/validation/test indices')
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(ids))):
            raise ValueError('Fit/validation/test must partition all 639 objects')
    cfg = manifest['config']
    if cfg.get('stage_epochs') != 30 or any(int(cfg.get(k, 2000)) != 2000 for k in ('samples', 'bootstrap')):
        raise ValueError('Expected epoch30 and 2000 predictive/bootstrap samples')
    if any(not isinstance(manifest.get(k), str) or not manifest[k].strip() for k in SOURCE_KEYS):
        raise ValueError('Three explicit reference run paths are required')
    return ids, allocation


def _source_paths(root, manifest):
    sources = {}
    for key in SOURCE_KEYS:
        path = Path(manifest[key]).expanduser()
        path = (path if path.is_absolute() else root/path).resolve()
        if path == root:
            raise ValueError('New summary cannot overwrite a reference run')
        sources[key] = path
    return sources


def _reference_partition(source, manifest):
    old = json.loads((source/'run_manifest.json').read_text())
    if old['ids'] != manifest['ids']:
        raise ValueError('Reference manifest IDs differ from the fixed snapshot')
    keyed = {int(row['fold']): row for row in old['folds']}
    if set(keyed) != {int(row['fold']) for row in manifest['folds']}:
        raise ValueError('Reference manifest outer folds differ')
    for row in manifest['folds']:
        for key in ('fit', 'inner_validation', 'test'):
            if row[key] != keyed[int(row['fold'])][key]:
                raise ValueError('Reference manifest '+key+' differs from the frozen partition')


def _interaction(data, records, boot):
    """Descriptive paired difference of differences, not a causal effect."""
    null = data[ARMS[0]]['actual'][:, 2] <= 0
    individual = {}
    for arm in ARMS[1:]:
        arrays = data[arm]
        individual[arm] = dict(u_mse=arrays['u_object_mse'], gamma_crps=arrays['utility_crps'][:, 2],
                              null_brier=np.square(arrays['p_null'][:, 2]-null))
    keys = ('u_mse', 'gamma_crps', 'null_brier')
    delta = {k:(individual[ARMS[3]][k]-individual[ARMS[2]][k])-
               (individual[NEW_ARM][k]-individual[ARMS[1]][k]) for k in keys}
    return dict(formula='(K_GEOMETRY_GAMMA_CRPS - J_GEOMETRY_CONTROL) - '
                '(M_CONDITIONAL_GENERIC_GAMMA - F_CONDITIONAL_GENERIC)',
        favorable_direction='negative indicates a larger score reduction with the structured basis',
        statistics={k:_bootstrap_statistic(v, boot) for k,v in delta.items()},
        folds=[dict(fold=row['fold'], **{k:float(v[np.asarray(row['test'], int)].mean()) for k,v in delta.items()})
               for row in records], policy_interaction_evaluated=False, causal_interpretation=False,
        interpretation='conditional-on-fitted-predictions score interaction; not evidence of a biological mechanism')


def _write_reports(root, result):
    intro = ['# 通用基函数＋Γ监督：固定第30轮比较', '',
        '本轮仅M重新训练；A、F、J、K的保存预测逐项匹配其历史来源。M使用30轮、210步，与K具有相同可训练参数数量。',
        '同639个已开放DEV对象、固定五折，每对象一次OOF；每臂选择79对象、158孔。', '']
    short = intro+_table(result['models'], ARMS)+['',
        'K−M比较同Γ监督下的基函数；M−F比较通用基的监督目标；K−J重现结构化基的原监督对照；M−A作为基础参照。',
        '不是新的独立认证：实际第30轮固定，原Γ、纯度要求、成本和划分均不改变。']
    (root/'SUMMARY.md').write_text('\n'.join(short)+'\n')
    lines = short+['', '## 四组同预算配对比较', '',
        '差值均为左减右。MSE、CRPS、Brier和风险负值有利，实际净值正值有利。', '']
    lines += _comparison_lines(result['comparisons'])
    lines += ['', '## 监督目标与基函数的评分交互', '',
        '计算(K−J)−(M−F)。负值表示结构化基在该评分上的下降更大；不代表生物机制已被识别，也不等价于策略价值的交互。', '',
        '| 指标 | 配对交互差 | 条件式95%区间 |', '|---|---:|---|']
    for key, row in result['score_interaction']['statistics'].items():
        interval = row['interval95']
        lines.append(f"| {key} | {_fmt(row['mean'])} | [{_fmt(interval[0])}, {_fmt(interval[1])}] |")
    lines += ['', '## 各折方向与选择重叠', '',
        '| 比较 | MSE有利/不利/并列 | CRPS有利/不利/并列 | Brier有利/不利/并列 | 净值有利/不利/并列 | 共同选择/79 |',
        '|---|---|---|---|---|---:|']
    for pair in result['comparisons'].values():
        cells = ['/'.join(str(pair['fold_directions'][k][name]) for name in ('favorable','unfavorable','tied'))
                 for k in ('u_mse','gamma_crps','null_brier','net_gain_per_eligible')]
        lines.append('| '+pair['left']+' − '+pair['right']+' | '+' | '.join(cells)+f" | {pair['overlap']['intersection_n']} |")
    lines += ['', '## M的固定训练轨迹', '',
        '| 折 | epoch | fit九维MSE | validation九维MSE | validation Γ CRPS | validation Γ MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for trajectory in result['models'][NEW_ARM]['training_trajectory']:
        for row in trajectory['records']:
            cells = [_fmt(row.get(k)) for k in ('fit_u_mse','validation_u_mse','validation_gamma_crps','validation_gamma_mse')]
            lines.append(f"| {trajectory['fold']} | {row['epoch']} | "+' | '.join(cells)+' |')
    lines += ['', '## 解释范围', '',
        '只有M是本轮新训练。历史A/F/J/K的均值、概率、评分、原目标、掩码及保存的协方差逐项复核，两个历史来源的fit/validation/test名单也完全一致。',
        '3837是当前可训练网络参数数量，不包括固定ridge、描述量缓冲值及冻结HR网络；完整参数计数另存JSON。',
        '九维MSE在各折固定尺度内计算。Γ已包含原两孔成本；FDP分母为选中对象，FPR分母为全部NULL。净值和风险均保留。',
        '相关系数和AUC为描述量。gate与块贡献也只是数值诊断，不能单独说明药物机制或生物知识的作用。',
        '区间基于同一2000次折内对象配对复抽，条件于固定预测及掩码，不包含训练、共享批次、历史搜索和Monte Carlo不确定性；未校正多重比较。',
        'Monte Carlo稳定性是独立检查，来源：'+result['sampling_stability_run']+'；本报告不把该检查替代统计认证。',
        'FINAL、第五重复和旧七项合同未动；不删对象，不以本轮结果重新挑轮次或工作点。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = _validate_manifest(manifest)
    sources = _source_paths(root, manifest)
    for key in ('gamma_reference_run','conditional_reference_run'):
        _reference_partition(sources[key], manifest)
    data, models = {}, {}
    actual = actual_u = None
    for arm in ARMS:
        arrays, rows, diagnostics = _read_new_arm(root, manifest, ids, arm)
        _finish(arrays, ids, allocation)
        if actual is None:
            actual, actual_u = arrays['actual'].copy(), arrays['actual_u'].copy()
        elif not np.array_equal(actual, arrays['actual']) or not np.array_equal(actual_u, arrays['actual_u']):
            raise ValueError('All arms must use identical original Gamma and nine-u targets')
        for row, record in zip(rows, manifest['folds']):
            ix = np.asarray(record['test'], int)
            row['principal_policy'] = _policy(arrays['actual'][ix,2], arrays['principal_mask'][ix])
        models[arm] = dict(**_model_scores(arrays, ids, manifest['folds']), folds=rows,
            training_trajectory=_training_trajectory(rows, arm, root) if arm==NEW_ARM else [],
            model_diagnostics=_aggregate_diagnostics(diagnostics), newly_trained=arm==NEW_ARM)
        data[arm] = arrays
    reproduction = {}
    for arm in ARMS[:-1]:
        key = 'conditional_reference_run' if arm==ARMS[1] else 'gamma_reference_run'
        reproduction[arm] = _verify_reference(root, sources[key], manifest, ids, allocation, arm, arm, data[arm])
    for position, row in enumerate(models[NEW_ARM]['folds']):
        if row['parameter_counts']['trainable'] != 3837:
            raise ValueError('M must retain exactly 3837 trainable parameters')
        if row['parameter_counts'] != models[ARMS[3]]['folds'][position]['parameter_counts']:
            raise ValueError('K/M architecture parameter counts differ')
    seed = int(manifest['config'].get('bootstrap_seed', manifest['config'].get('seed',20260914)))
    rng = np.random.default_rng(seed)
    boot = np.column_stack([rng.choice(np.flatnonzero(allocation==fold),
        size=(2000,int(np.sum(allocation==fold)))) for fold in sorted(np.unique(allocation))])
    comparisons = {a+'__minus__'+b:_compare(a,b,data,ids,manifest['folds'],boot) for a,b in COMPARISONS}
    result = dict(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(), n=len(ids), arms=list(ARMS),
        new_arms=[NEW_ARM], models=models, comparisons=comparisons,
        score_interaction=_interaction(data, manifest['folds'], boot), historical_reproduction=reproduction,
        actual_checkpoint_epoch=30, optimizer_steps_per_new_arm=210, trainable_parameters_new_arm=3837,
        checkpoint_selection='fixed epoch30, no test-dependent checkpoint selection',
        samples_per_object=2000, bootstrap_replicates=2000, bootstrap_seed=seed, interval_scope=INTERVAL_SCOPE,
        fold_test_counts=[len(row['test']) for row in manifest['folds']],
        principal_budget='25% physical wells, 79 objects / 158 wells', original_action='ADD_TWO / Z1Z2',
        parameter_count_scope='network parameters; excludes fixed ridge and descriptor buffers',
        formal_certificate=False, historical_dev=True, secondary_comparisons_multiplicity_adjusted=False,
        final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False, original_contract_changed=False,
        **{key:str(path) for key,path in sources.items()})
    for arm in ARMS:
        np.savez_compressed(root/f'{arm}_oof_predictions.npz', ids=ids, fold=allocation, **data[arm])
    write_json(root/'summary.json',result)
    _write_reports(root,result)
    return result
