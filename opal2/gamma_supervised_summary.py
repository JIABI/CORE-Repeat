"""Saved-output comparison of a geometry loss and its Gamma-CRPS extension.

No model, loss weight, checkpoint or policy is fitted here. The fresh geometry
control must reproduce the historical conditional model before scores are read.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .conditional_response_summary import (
    ARRAY_SHAPES, INTERVAL_SCOPE, _aggregate_diagnostics, _finish,
    _model_scores, _read_historical, _read_new_arm, _table,
)
from .geometry_kernel_continuation_summary import _comparison_lines
from .geometry_kernel_summary import _compare, _policy
from .hierarchical_geometry_summary import _folds, _fmt


ARMS = ('A_HR', 'J_GEOMETRY_CONTROL', 'K_GEOMETRY_GAMMA_CRPS')
COMPARISONS = (('K_GEOMETRY_GAMMA_CRPS', 'J_GEOMETRY_CONTROL'),
               ('K_GEOMETRY_GAMMA_CRPS', 'A_HR'))
HISTORICAL_MAPPING = {'A_HR': 'A_HR', 'J_GEOMETRY_CONTROL': 'G_CONDITIONAL_STRUCTURED'}
STAGE_EPOCHS = (0, 5, 10, 15, 20, 25, 30)


def _validate_manifest(manifest):
    if list(manifest['arms']) != list(ARMS):
        raise ValueError('Gamma-supervised stage requires the three declared arms in order')
    for key in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed', 'original_contract_changed'):
        if manifest.get(key) is not False:
            raise ValueError('Experiment does not preserve '+key)
    ids, allocation = _folds(manifest)
    if len(ids) != 639 or len(manifest['folds']) != 5:
        raise ValueError('Expected 639 unique DEV objects and five outer folds')
    for record in manifest['folds']:
        parts = [np.asarray(record[key], int) for key in ('fit', 'inner_validation', 'test')]
        if any(not len(x) or len(np.unique(x)) != len(x) or np.any(x < 0) or np.any(x >= len(ids)) for x in parts):
            raise ValueError('Invalid fit/validation/test indices')
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(ids))):
            raise ValueError('Fit/validation/test must partition all 639 objects')
    cfg = manifest['config']
    if cfg.get('stage_epochs') != 30 or any(int(cfg.get(k, 2000)) != 2000 for k in ('samples', 'bootstrap')):
        raise ValueError('Expected actual epoch30 and 2000 predictive/bootstrap samples')
    if not isinstance(manifest.get('historical_reference_run'), str) or not manifest['historical_reference_run'].strip():
        raise ValueError('historical_reference_run must identify the saved conditional-response run')
    return ids, allocation


def _training_trajectory(rows, arm, root):
    output = []
    if arm == 'A_HR':
        return output
    for row in rows:
        checkpoint = row['checkpoint']
        complete = checkpoint['training_complete'] or {}
        final_epoch = complete.get('final_epoch', complete.get('actual_checkpoint_epoch', complete.get('epoch')))
        if final_epoch != 30 or complete.get('optimizer_steps') != 210:
            raise ValueError('Each fresh arm must finish 30 epochs / 210 optimizer steps')
        history = checkpoint['validation_selection_appendix']
        if [item.get('epoch') for item in history] != list(STAGE_EPOCHS):
            raise ValueError('Training trajectory must retain fixed epochs 0/5/10/15/20/25/30')
        monitor_path = root/'folds'/f"fold_{row['fold']}"/'arms'/arm/'gamma_monitoring.jsonl'
        monitoring = ([json.loads(line) for line in monitor_path.read_text().splitlines() if line.strip()]
                      if monitor_path.is_file() else [])
        if monitoring and [item.get('epoch') for item in monitoring] != list(STAGE_EPOCHS):
            raise ValueError('Post-training Gamma monitor must retain the seven fixed epochs')
        by_epoch = {item['epoch']: item for item in monitoring}
        merged = [dict(item, **{k:v for k,v in by_epoch.get(item['epoch'], {}).items() if k!='epoch'}) for item in history]
        output.append(dict(fold=row['fold'], records=merged, original_history=history,
                           gamma_monitoring=monitoring, gamma_monitoring_available=bool(monitoring),
                           training_complete=complete,
                           test_checkpoint_selection='fixed epoch30, not best validation or test score'))
    return output


def _verify_reference(root, source, manifest, ids, allocation, new_arm, old_arm, data):
    historical = _read_historical(source, manifest, ids, allocation, old_arm)
    for key in (*ARRAY_SHAPES, 'actual_u', 'mean_u', 'principal_mask'):
        if not np.array_equal(data[key], historical[key]):
            raise ValueError(f'{new_arm} does not exactly reproduce historical {old_arm}: {key}')
    checked = []
    for record in manifest['folds']:
        rel = Path('folds')/f"fold_{record['fold']}"/'arms'
        new_path = root/rel/new_arm/'evaluation/u_predictions.npz'
        old_path = source/rel/old_arm/'evaluation/u_predictions.npz'
        if not old_path.is_file():
            raise ValueError('Historical per-fold evaluation is missing: '+str(old_path))
        with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
            keys = []
            if not np.array_equal(new['ids'], old['ids']):
                raise ValueError('Reference per-fold coordinate IDs differ')
            for key in ('covariance_u', 'scale_tril_u'):
                if (key in new) != (key in old):
                    raise ValueError('Reference uncertainty array missing from one side: '+key)
                if key in new:
                    if not np.array_equal(new[key], old[key]):
                        raise ValueError(f'{new_arm} reference covariance differs: {key}')
                    keys.append(key)
        checked.append(dict(fold=record['fold'], uncertainty_arrays=keys))
    return dict(historical_arm=old_arm, source=str(source), exact_core_arrays=True,
        exact_uncertainty_arrays_per_fold=checked,
        checkpoint_state_identity_scope='checked by the runner; this summary independently checks saved predictions')


def _write_reports(root, result):
    intro = ['# Γ监督：固定第30轮三臂开发比较', '',
        'A_HR沿用冻结结果；J重新训练几何损失，且逐项复现历史G；K只增加原Γ的CRPS监督。',
        '每个已开放DEV对象贡献一次外折结果，共639对象。三个模型均按同一预算选79对象、158孔。', '']
    short = intro+_table(result['models'], ARMS)+['',
        '主比较K−J隔离本轮监督目标改变，K−A作为基础模型参照。固定实际第30轮，不以test或验证最佳轮次挑选。',
        '旧目标、七项合同、FINAL及第五重复均未改；这是已反复使用DEV的开发比较，不是独立认证。']
    (root/'SUMMARY.md').write_text('\n'.join(short)+'\n')
    lines = short+['', '## 同预算配对比较', '',
        '左减右：MSE、CRPS、Brier及风险负值有利，实际净值正值有利。区间为同一2000次折内对象配对复抽。', '']
    lines += _comparison_lines(result['comparisons'])
    lines += ['', '## 各折方向与选择重叠', '',
        '| 比较 | MSE有利/不利/并列 | CRPS有利/不利/并列 | Brier有利/不利/并列 | 净值有利/不利/并列 | 共同选择/79 |',
        '|---|---|---|---|---|---:|']
    for pair in result['comparisons'].values():
        directions = pair['fold_directions']
        cells = ['/'.join(str(directions[k][name]) for name in ('favorable', 'unfavorable', 'tied'))
            for k in ('u_mse', 'gamma_crps', 'null_brier', 'net_gain_per_eligible')]
        lines.append('| '+pair['left']+' − '+pair['right']+' | '+' | '.join(cells)+f" | {pair['overlap']['intersection_n']} |")
    lines += ['', '## 固定训练轨迹（不用于重选test检查点）', '',
        '完整loss分量和梯度诊断按原字段保存在summary.json；下表列出各折固定观测时点。', '',
        '| 臂 | 折 | epoch | fit九维MSE | validation九维MSE | validation Γ CRPS | validation Γ MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS[1:]:
        for trajectory in result['models'][arm]['training_trajectory']:
            for item in trajectory['records']:
                cells = [_fmt(item.get(k)) for k in ('fit_u_mse', 'validation_u_mse',
                                                     'validation_gamma_crps', 'validation_gamma_mse')]
                lines.append(f"| {arm} | {trajectory['fold']} | {item['epoch']} | "+' | '.join(cells)+' |')
    lines += ['', '## 比较边界', '',
        'J和历史G的预测均值、Γ预测、NULL概率、CRPS、几何评分、原目标和选择掩码必须逐数组相同；A亦必须匹配历史A。',
        '双方保存的九维协方差及其因子也逐折核对。历史G因此不再重复列为第四臂，检查点参数身份另由运行器核查。',
        '九维MSE使用同一折的固定标准化尺度。Γ已包含原追加成本，未重复扣费；FDP和FPR仍分别使用选中对象与全部NULL作为分母。',
        '合并Spearman/AUC为描述量，折内秩关联另存JSON。较好CRPS不能单独证明选择价值或纯度提高，二者均完整报告。',
        '区间条件于固定拟合预测和选择掩码，不包括训练、共享批次、历史搜索或Monte Carlo不确定性；未作多重比较校正。',
        '模型诊断沿用gate/块贡献/局部系数的描述性统计，不将其解释为生物因果贡献。所有臂保留，不删难对象或更换收益标签。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    ids, allocation = _validate_manifest(manifest)
    source = Path(manifest['historical_reference_run']).expanduser()
    source = (source if source.is_absolute() else root/source).resolve()
    if source == root:
        raise ValueError('New summaries cannot overwrite their historical reference')
    data, models = {}, {}
    actual = actual_u = None
    for arm in ARMS:
        arrays, rows, diagnostics = _read_new_arm(root, manifest, ids, arm)
        _finish(arrays, ids, allocation)
        if actual is None:
            actual, actual_u = arrays['actual'].copy(), arrays['actual_u'].copy()
        elif not np.array_equal(actual, arrays['actual']) or not np.array_equal(actual_u, arrays['actual_u']):
            raise ValueError('All three arms must share identical original Gamma and nine-u targets')
        for row, record in zip(rows, manifest['folds']):
            ix = np.asarray(record['test'], int)
            row['principal_policy'] = _policy(arrays['actual'][ix, 2], arrays['principal_mask'][ix])
        models[arm] = dict(**_model_scores(arrays, ids, manifest['folds']), folds=rows,
            training_trajectory=_training_trajectory(rows, arm, root),
            model_diagnostics=_aggregate_diagnostics(diagnostics))
        data[arm] = arrays
    reproduction = {new: _verify_reference(root, source, manifest, ids, allocation, new, old, data[new])
                    for new, old in HISTORICAL_MAPPING.items()}
    for a, b in zip(models['J_GEOMETRY_CONTROL']['folds'], models['K_GEOMETRY_GAMMA_CRPS']['folds']):
        if a['parameter_counts'] != b['parameter_counts']:
            raise ValueError('J/K must retain equal architecture parameter counts')
    seed = int(manifest['config'].get('bootstrap_seed', manifest['config'].get('seed', 20260914)))
    rng = np.random.default_rng(seed)
    boot = np.column_stack([rng.choice(np.flatnonzero(allocation == fold),
        size=(2000, int(np.sum(allocation == fold)))) for fold in sorted(np.unique(allocation))])
    comparisons = {a+'__minus__'+b: _compare(a, b, data, ids, manifest['folds'], boot) for a, b in COMPARISONS}
    result = dict(complete=True, completed_utc=datetime.now(timezone.utc).isoformat(), n=len(ids),
        arms=list(ARMS), models=models, comparisons=comparisons, historical_reproduction=reproduction,
        historical_reference_run=str(source), actual_checkpoint_epoch=30, optimizer_steps_per_fresh_arm=210,
        checkpoint_selection='fixed epoch30; complete fixed validation trajectory only for diagnosis',
        samples_per_object=2000, bootstrap_replicates=2000, bootstrap_seed=seed, interval_scope=INTERVAL_SCOPE,
        fold_test_counts=[len(x['test']) for x in manifest['folds']],
        principal_budget='25% physical wells, 79 objects / 158 wells', original_action='ADD_TWO / Z1Z2',
        parameter_count_scope='network parameters including frozen HR; excludes ridge and descriptor buffers',
        equal_J_K_parameter_counts=True, formal_certificate=False, historical_dev=True,
        secondary_comparisons_multiplicity_adjusted=False,
        final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False, original_contract_changed=False)
    for arm in ARMS:
        np.savez_compressed(root/f'{arm}_oof_predictions.npz', ids=ids, fold=allocation, **data[arm])
    write_json(root/'summary.json', result)
    _write_reports(root, result)
    return result
