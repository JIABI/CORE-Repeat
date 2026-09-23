"""Fixed-epoch summaries for actual 30-to-60 continuation of F/J/K/M.

Reads saved outputs only, compares each continued arm with its own epoch30
predictions, and verifies that the frozen A comparator has not changed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .conditional_response_summary import (
    ARRAY_SHAPES, INTERVAL_SCOPE, _aggregate_diagnostics, _diagnostic_file,
    _finite, _finish, _model_scores, _read_coordinate_file, _read_historical, _table,
)
from .generic_gamma_summary import ARMS, _interaction, _reference_partition
from .geometry_kernel_continuation_summary import _comparison_lines
from .geometry_kernel_replacement_summary import _parameter_counts
from .geometry_kernel_summary import _compare, _policy
from .hierarchical_geometry_summary import _folds, _fmt


CONTINUED_ARMS = ARMS[1:]
COMPARISONS = ((ARMS[3], ARMS[4]), (ARMS[4], ARMS[1]), (ARMS[3], ARMS[2]), (ARMS[2], ARMS[1]))
TRAJECTORY_EPOCHS = tuple(range(30, 61, 5))


def _validate_manifest(manifest):
    if list(manifest['arms']) != list(ARMS) or manifest.get('new_arms') != list(CONTINUED_ARMS):
        raise ValueError('Expected frozen A and four continued F/J/K/M arms')
    if manifest.get('start_epoch') != 30 or manifest['config'].get('stage_epochs') != 60:
        raise ValueError('Expected actual continuation from epoch30 to epoch60')
    if not isinstance(manifest.get('source_run'), str) or not manifest['source_run'].strip():
        raise ValueError('source_run must identify the epoch30 source')
    for key in ('final_opened','fifth_repeat_opened','original_endpoint_changed','original_contract_changed'):
        if manifest.get(key) is not False:
            raise ValueError('Experiment does not preserve '+key)
    ids, allocation = _folds(manifest)
    if len(ids) != 639 or len(manifest['folds']) != 5:
        raise ValueError('Expected 639 unique DEV objects and five folds')
    for record in manifest['folds']:
        parts = [np.asarray(record[k], int) for k in ('fit','inner_validation','test')]
        if any(not len(x) or len(np.unique(x)) != len(x) or np.any(x<0) or np.any(x>=len(ids)) for x in parts):
            raise ValueError('Invalid fit/validation/test indices')
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(ids))):
            raise ValueError('Fit/validation/test must partition all 639 objects')
    if any(int(manifest['config'].get(k,2000)) != 2000 for k in ('samples','bootstrap')):
        raise ValueError('Expected 2000 predictive and bootstrap samples')
    return ids, allocation


def _read_arm(root, manifest, ids, arm):
    n = len(ids)
    arrays = {key:np.empty((n,*tail)) for key,tail in ARRAY_SHAPES.items()}
    arrays.update(actual_u=np.empty((n,9)), mean_u=np.empty((n,9)))
    rows, diagnostic_rows = [], []
    for record in manifest['folds']:
        ix = np.asarray(record['test'], int)
        folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
        evaluation = folder/'evaluation'
        metric = json.loads((evaluation/'metrics.json').read_text())
        metadata = metric.get('model', {})
        epoch = metric.get('actual_checkpoint_epoch', metadata.get('actual_checkpoint_epoch'))
        if metric.get('samples') != 2000:
            raise ValueError('Evaluation requires 2000 predictive samples')
        if arm != 'A_HR' and epoch != 60:
            raise ValueError('Continued arms must use the actual epoch60 checkpoint')
        with np.load(evaluation/'predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Evaluation IDs differ from declared outer test fold')
            for key,tail in ARRAY_SHAPES.items():
                arrays[key][ix] = _finite(saved[key], (len(ix),*tail), key)
        arrays['actual_u'][ix], arrays['mean_u'][ix] = _read_coordinate_file(evaluation/'u_predictions.npz', ids[ix])
        complete_path = folder/'training_complete.json'
        complete = json.loads(complete_path.read_text()) if complete_path.is_file() else None
        history_path = folder/'history.jsonl'
        history = ([json.loads(x) for x in history_path.read_text().splitlines() if x.strip()]
                   if history_path.is_file() else [])
        history = [x for x in history if x.get('epoch') in TRAJECTORY_EPOCHS]
        monitor_path = folder/'gamma_monitoring.jsonl'
        monitoring = ([json.loads(x) for x in monitor_path.read_text().splitlines() if x.strip()]
                      if monitor_path.is_file() else [])
        monitoring = [x for x in monitoring if x.get('epoch') in TRAJECTORY_EPOCHS]
        if arm != 'A_HR':
            final_epoch = (complete or {}).get('final_epoch', (complete or {}).get('actual_checkpoint_epoch', (complete or {}).get('epoch')))
            if final_epoch != 60 or (complete or {}).get('optimizer_steps') != 420:
                raise ValueError('Continuation must record actual epoch60 and 420 cumulative optimizer steps')
            if [x.get('epoch') for x in history] != list(TRAJECTORY_EPOCHS):
                raise ValueError('Continuation history must include fixed epochs30/35/40/45/50/55/60')
            if [x.get('epoch') for x in monitoring] != list(TRAJECTORY_EPOCHS):
                raise ValueError('Gamma monitoring must retain the seven continuation epochs')
        by_epoch = {x['epoch']:x for x in monitoring}
        trajectory = [dict(x, **{k:v for k,v in by_epoch.get(x['epoch'],{}).items() if k!='epoch'}) for x in history]
        checkpoint = dict(actual_checkpoint_epoch=epoch, test_metadata=metadata, training_complete=complete,
            original_history=history, gamma_monitoring=monitoring, continuation_trajectory=trajectory,
            validation_best_not_used_for_test=True)
        diag, numeric = _diagnostic_file(folder, ids[ix])
        diagnostic_rows.append((len(ix),diag,numeric))
        rows.append(dict(fold=record['fold'], n=len(ix), checkpoint=checkpoint,
            parameter_counts=_parameter_counts(checkpoint,arm), diagnostics=diag))
    return arrays, rows, diagnostic_rows


def _verify_fixed_uncertainty(root, source, manifest, arm):
    checked = []
    for record in manifest['folds']:
        relative = Path('folds')/f"fold_{record['fold']}"/'arms'/arm/'evaluation/u_predictions.npz'
        with np.load(root/relative, allow_pickle=False) as now, np.load(source/relative, allow_pickle=False) as old:
            if not np.array_equal(now['ids'], old['ids']):
                raise ValueError('Continuation uncertainty IDs differ from epoch30')
            keys = []
            for key in ('covariance_u','scale_tril_u'):
                if (key in now) != (key in old) or (key in now and not np.array_equal(now[key], old[key])):
                    raise ValueError('Continuation must preserve original uncertainty: '+key)
                if key in now: keys.append(key)
        checked.append(dict(fold=record['fold'],exact_arrays=keys))
    return checked


def _write_reports(root, result):
    intro = ['# Γ监督与基函数：实际第30→60轮续训', '',
        'F/J/K/M从各自第30轮继续训练到第60轮；A_HR保持冻结。主结果统一使用实际第60轮，不挑验证最佳轮次。',
        '639个已开放DEV对象各出现一次OOF，同一五折、每臂79对象/158孔预算。延长训练是在查看第30轮结果后决定的开发步骤。', '']
    small = intro+_table(result['models'],ARMS)
    small += ['', '## 各臂相对自身第30轮的变化', '',
        '| 臂 | 九维MSE变化 | Γ CRPS变化 | Brier变化 | 选中均值Γ变化 | FDP变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm, pair in result['epoch_extension_comparisons'].items():
        small.append('| '+arm+' | '+' | '.join(_fmt(pair[k]['mean']) for k in
            ('u_mse','gamma_crps','null_brier','net_gain_per_selected','fdp'))+' |')
    small += ['', '误差/风险变化负值有利，净值变化正值有利。配对区间、交互和轨迹见REPORT.md；不构成新的独立验证或合同通过。']
    (root/'SUMMARY.md').write_text('\n'.join(small)+'\n')
    lines = small+['', '## 第60轮四组比较', '']+_comparison_lines(result['comparisons'])
    lines += ['', '## 第60−30轮逐臂配对比较', '']+_comparison_lines(result['epoch_extension_comparisons'])
    lines += ['', '## 第60轮评分交互：(K−J)−(M−F)', '',
        '| 指标 | 交互差 | 条件式95%区间 |', '|---|---:|---|']
    for key,value in result['score_interaction']['statistics'].items():
        ci = value['interval95']
        lines.append(f"| {key} | {_fmt(value['mean'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] |")
    lines += ['', '负值表示结构化基在该评分上的监督改善较大，不等价于生物机制或策略价值交互。', '',
        '## 固定续训轨迹', '', '| 臂 | 折 | epoch | fit九维MSE | validation九维MSE | validation Γ CRPS | validation Γ MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in CONTINUED_ARMS:
        for fold in result['models'][arm]['folds']:
            for row in fold['checkpoint']['continuation_trajectory']:
                cells = [_fmt(row.get(k)) for k in ('fit_u_mse','validation_u_mse','validation_gamma_crps','validation_gamma_mse')]
                lines.append(f"| {arm} | {fold['fold']} | {row['epoch']} | "+' | '.join(cells)+' |')
    lines += ['', '## 比较边界', '',
        '续训检查点、优化器与调度器恢复由运行器验证；本汇总核对实际第60轮及420累计更新，追加210步。',
        '第30与60轮的对象、原Γ、九维目标和协方差逐项对齐。A的全部核心预测及选择掩码保持完全一致。',
        'Γ包含原两孔成本，FDP分母是所选对象，FPR分母是所有NULL。风险与实际净值均报告，不以较好CRPS替代纯度判断。',
        '各折方向、选择重叠、已保存检查点的loss/梯度读数保存在summary.json；逐轮训练记录保存在各臂history.jsonl。合并相关系数/AUC仅作描述。',
        '所有配对及交互采用同一2000次折内对象复抽，条件于固定拟合预测/掩码，不含训练、批次依赖、历史搜索或Monte Carlo误差；未校正多重比较。',
        '本轮没有扩大对象集合、更换收益标签或删去困难对象。FINAL、第五重复、七项合同及历史结果保持不动。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = _validate_manifest(manifest)
    source = Path(manifest['source_run']).expanduser()
    source = (source if source.is_absolute() else root/source).resolve()
    if source == root: raise ValueError('Continuation cannot overwrite its source')
    _reference_partition(source,manifest)
    source_manifest = json.loads((source/'run_manifest.json').read_text())
    if source_manifest['config'].get('stage_epochs') != 30:
        raise ValueError('The continuation source must be actual epoch30')
    data, old_data, models = {}, {}, {}
    actual = actual_u = None
    uncertainty = {}
    for arm in ARMS:
        arrays, rows, diagnostics = _read_arm(root,manifest,ids,arm)
        _finish(arrays,ids,allocation)
        old = _read_historical(source,manifest,ids,allocation,arm)
        if not np.array_equal(arrays['actual'],old['actual']) or not np.array_equal(arrays['actual_u'],old['actual_u']):
            raise ValueError('Epoch60 and epoch30 must share identical original targets')
        if actual is None: actual,actual_u=arrays['actual'].copy(),arrays['actual_u'].copy()
        elif not np.array_equal(actual,arrays['actual']) or not np.array_equal(actual_u,arrays['actual_u']):
            raise ValueError('All arms must share identical original targets')
        if arm=='A_HR':
            for key in (*ARRAY_SHAPES,'actual_u','mean_u','principal_mask'):
                if not np.array_equal(arrays[key],old[key]):
                    raise ValueError('Frozen A_HR changed: '+key)
        uncertainty[arm] = _verify_fixed_uncertainty(root,source,manifest,arm)
        for row,record in zip(rows,manifest['folds']):
            ix = np.asarray(record['test'],int)
            row['principal_policy'] = _policy(arrays['actual'][ix,2],arrays['principal_mask'][ix])
        data[arm],old_data[arm] = arrays,old
        models[arm] = dict(**_model_scores(arrays,ids,manifest['folds']),folds=rows,
            model_diagnostics=_aggregate_diagnostics(diagnostics),continued_from_epoch30=arm in CONTINUED_ARMS)
    seed = int(manifest['config'].get('bootstrap_seed',manifest['config'].get('seed',20260914)))
    rng = np.random.default_rng(seed)
    boot = np.column_stack([rng.choice(np.flatnonzero(allocation==fold),
        size=(2000,int(np.sum(allocation==fold)))) for fold in sorted(np.unique(allocation))])
    comparisons = {a+'__minus__'+b:_compare(a,b,data,ids,manifest['folds'],boot) for a,b in COMPARISONS}
    extension = {}
    for arm in ARMS:
        left,right = arm+'@epoch60',arm+'@epoch30'
        extension[arm] = _compare(left,right,{left:data[arm],right:old_data[arm]},ids,manifest['folds'],boot)
    result = dict(complete=True,completed_utc=datetime.now(timezone.utc).isoformat(),n=len(ids),arms=list(ARMS),
        continued_arms=list(CONTINUED_ARMS),source_run=str(source),actual_checkpoint_epoch=60,start_epoch=30,
        cumulative_optimizer_steps_per_continued_arm=420,additional_optimizer_steps_per_continued_arm=210,
        models=models,comparisons=comparisons,epoch_extension_comparisons=extension,
        score_interaction=_interaction(data,manifest['folds'],boot),fixed_uncertainty_checks=uncertainty,
        historical_A_exactly_matched=True,checkpoint_selection='fixed actual epoch60, not validation-best',
        extension_chosen_after_dev_epoch30=True,samples_per_object=2000,bootstrap_replicates=2000,
        bootstrap_seed=seed,interval_scope=INTERVAL_SCOPE,fold_test_counts=[len(r['test']) for r in manifest['folds']],
        principal_budget='25% physical wells, 79 objects / 158 wells',original_action='ADD_TWO / Z1Z2',
        formal_certificate=False,historical_dev=True,secondary_comparisons_multiplicity_adjusted=False,
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    for arm in ARMS:
        np.savez_compressed(root/f'{arm}_oof_predictions.npz',ids=ids,fold=allocation,**data[arm])
    write_json(root/'summary.json',result)
    _write_reports(root,result)
    return result
