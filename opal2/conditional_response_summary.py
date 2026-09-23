"""Saved-prediction summary of conditional response-basis experiments.

The entry point reads the four new evaluation directories and three historical
OOF comparators. A_HR is checked against its saved historical predictions. No
model, operating point, component gate or checkpoint is fit by this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .geometry_kernel_summary import _policy, _compare, U_SCOPE
from .geometry_kernel_continuation_summary import _comparison_lines
from .geometry_kernel_replacement_summary import _parameter_counts
from .gram_oof_experiment import selection_mask
from .hierarchical_geometry_summary import _folds, _action_rows, _fmt


ARMS = ('A_HR', 'E_STATIC_STRUCTURED', 'F_CONDITIONAL_GENERIC', 'G_CONDITIONAL_STRUCTURED')
HISTORICAL_ARMS = ('B_MLP', 'C_GENERIC', 'D_STRUCTURED')
COMPARISONS = tuple(('G_CONDITIONAL_STRUCTURED', arm) for arm in
                   ('A_HR', 'E_STATIC_STRUCTURED', 'F_CONDITIONAL_GENERIC', *HISTORICAL_ARMS))
ARRAY_SHAPES = dict(actual=(3,), predicted=(3,), p_null=(3,), utility_crps=(3,), geometry_energy=())
INTERVAL_SCOPE = ('2000 paired compound bootstrap resamples within the five outer folds, fixed fitted '
    'predictions/masks; does not include shared-batch, training, historical-development-selection or '
    'Monte Carlo uncertainty; historical DEV comparison, not independent certification')


def _validate_manifest(manifest):
    if list(manifest['arms']) != list(ARMS):
        raise ValueError('The conditional response stage must report all four declared arms')
    for key in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed', 'original_contract_changed'):
        if manifest.get(key) is not False:
            raise ValueError('The experiment does not preserve '+key)
    ids, allocation = _folds(manifest)
    if len(ids) != 639 or len(manifest['folds']) != 5:
        raise ValueError('Expected 639 unique DEV IDs and five outer folds')
    for record in manifest['folds']:
        parts = [np.asarray(record[key], int) for key in ('fit', 'inner_validation', 'test')]
        if any(not len(x) or len(np.unique(x)) != len(x) or np.any(x<0) or np.any(x>=len(ids)) for x in parts):
            raise ValueError('Invalid fit, validation or test indices')
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(ids))):
            raise ValueError('Fit/validation/test must partition all 639 objects')
    cfg = manifest['config']
    if cfg.get('stage_epochs') != 30 or int(cfg.get('samples', 2000)) != 2000 or int(cfg.get('bootstrap', 2000)) != 2000:
        raise ValueError('The declared stage is epoch30 with 2000 predictive and bootstrap samples')
    if not isinstance(manifest.get('historical_reference_run'), str) or not manifest['historical_reference_run'].strip():
        raise ValueError('historical_reference_run must identify the saved four-arm epoch30 run')
    return ids, allocation


def _finite(values, shape, label):
    result = np.asarray(values, float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid finite aligned '+label)
    return result


def _read_coordinate_file(path, expected_ids):
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved['ids'], expected_ids):
            raise ValueError('Coordinate prediction IDs differ from the declared test IDs')
        return (_finite(saved['actual_u'], (len(expected_ids), 9), 'actual_u').copy(),
                _finite(saved['mean_u'], (len(expected_ids), 9), 'mean_u').copy())


def _diagnostic_file(folder, ids):
    path = folder/'model_diagnostics.npz'
    if not path.is_file():
        path = folder/'evaluation/model_diagnostics.npz'
    if not path.is_file():
        return dict(available=False), {}
    with np.load(path, allow_pickle=False) as saved:
        if 'ids' not in saved or not np.array_equal(saved['ids'], ids):
            raise ValueError('Model diagnostic IDs do not match their test fold')
        data = {}
        for key in ('gate', 'raw', 'kernel_raw', 'output_bias', 'block_contributions', 'coeff', 'coefficient', 'local',
                    'local_activation', 'local_basis'):
            if key in saved:
                value = np.asarray(saved[key], float)
                if value.ndim < 1 or len(value) != len(ids) or not np.isfinite(value).all():
                    raise ValueError('Model diagnostics must be finite and object aligned: '+key)
                data[key] = value.copy()
        coefficients = None
        if 'local_coefficients' in saved:
            coefficients = np.asarray(saved['local_coefficients'], float)
            if coefficients.ndim != 2 or coefficients.shape[1] != 3 or not np.isfinite(coefficients).all():
                raise ValueError('local_coefficients must be finite descriptor-by-three parameters')
        block_names = saved['block_names'].astype(str).tolist() if 'block_names' in saved else None
    result = dict(available=True, source=str(path), n=len(ids), fields=list(data), block_names=block_names,
                  interpretation='descriptive activations, not causal biological contribution or drop-block utility')
    if 'raw' in data and 'kernel_raw' in data and not np.array_equal(data['raw'], data['kernel_raw']):
        raise ValueError('The raw alias differs from kernel_raw')
    if coefficients is not None:
        result['local_coefficients'] = dict(shape=list(coefficients.shape), mean=float(coefficients.mean()),
            min=float(coefficients.min()), max=float(coefficients.max()), rms=float(np.sqrt(np.square(coefficients).mean())),
            rms_change_from_one_third=float(np.sqrt(np.square(coefficients-1/3).mean())),
            scope='descriptor-by-basis model parameters, not object-specific coefficients')
    if 'gate' in data:
        gate = data['gate'].reshape(len(ids), -1)
        result['gate'] = dict(shape=list(data['gate'].shape[1:]), mean=gate.mean(0).tolist(),
            within_fold_std=gate.std(0).tolist(), min=gate.min(0).tolist(), max=gate.max(0).tolist())
    if 'block_contributions' in data:
        blocks = data['block_contributions']
        if blocks.ndim != 3 or blocks.shape[-1] != 9:
            raise ValueError('block_contributions must have shape [N,block,9]')
        if block_names is not None and len(block_names) != blocks.shape[1]:
            raise ValueError('Block names do not match the contribution axis')
        result['block_contribution_rms'] = np.sqrt(np.square(blocks).mean((0, 2))).tolist()
        raw = data.get('raw', data.get('kernel_raw'))
        if raw is not None and 'output_bias' in data:
            raw = _finite(raw, (len(ids), 9), 'raw kernel output')
            bias = _finite(data['output_bias'], (len(ids), 9), 'raw output bias')
            residual = blocks.sum(1)+bias-raw
            result['sum_closure'] = dict(formula='sum(block_contributions, axis=1) + output_bias = raw',
                max_absolute_error=float(np.abs(residual).max()), rms_error=float(np.sqrt(np.square(residual).mean())),
                raw_scale=float(np.max(np.abs(raw))), bounded_final_mean_not_additive=True)
    for key in ('raw', 'kernel_raw', 'output_bias', 'coeff', 'coefficient', 'local', 'local_activation', 'local_basis'):
        if key in data:
            result[key+'_rms'] = float(np.sqrt(np.square(data[key]).mean()))
    return result, data


def _read_new_arm(root, manifest, ids, arm):
    n = len(ids)
    arrays = {key:np.empty((n,*tail)) for key,tail in ARRAY_SHAPES.items()}
    arrays.update(actual_u=np.empty((n,9)), mean_u=np.empty((n,9)))
    rows, diagnostic_rows = [], []
    for record in manifest['folds']:
        ix = np.asarray(record['test'], int)
        folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
        evaluation = folder/'evaluation'
        metric = json.loads((evaluation/'metrics.json').read_text())
        if metric.get('samples') != 2000:
            raise ValueError('Saved evaluation must use the declared 2000 predictive samples')
        metadata = metric.get('model', {})
        epoch = metric.get('actual_checkpoint_epoch', metadata.get('actual_checkpoint_epoch'))
        if arm != 'A_HR' and epoch != 30:
            raise ValueError('New conditional arms must be evaluated at actual epoch30')
        with np.load(evaluation/'predictions.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'], ids[ix]):
                raise ValueError('Evaluation IDs differ from their declared test fold')
            for key, tail in ARRAY_SHAPES.items():
                arrays[key][ix] = _finite(saved[key], (len(ix),*tail), key)
        a,p = _read_coordinate_file(evaluation/'u_predictions.npz', ids[ix])
        arrays['actual_u'][ix], arrays['mean_u'][ix] = a,p
        complete_path = folder/'training_complete.json'
        complete = json.loads(complete_path.read_text()) if complete_path.is_file() else None
        history_path = folder/'history.jsonl'
        history = ([json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
                   if history_path.is_file() else [])
        checkpoint = dict(actual_checkpoint_epoch=epoch, test_metadata=metadata, training_complete=complete,
            validation_selection_appendix=[x for x in history if x.get('epoch') in (0,5,10,15,20,25,30)],
            appendix_not_used_to_choose_test_predictions=True)
        diagnostic, numeric = _diagnostic_file(folder, ids[ix])
        diagnostic_rows.append((len(ix), diagnostic, numeric))
        rows.append(dict(fold=record['fold'], n=len(ix), checkpoint=checkpoint,
            parameter_counts=_parameter_counts(checkpoint, arm), diagnostics=diagnostic))
    if np.any(arrays['p_null']<0) or np.any(arrays['p_null']>1):
        raise ValueError('NULL probabilities must lie in [0,1]')
    return arrays, rows, diagnostic_rows


def _read_historical(source, manifest, ids, allocation, arm):
    arrays = {}
    with np.load(source/f'{arm}_oof_predictions.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['ids'], ids) or not np.array_equal(saved['fold'], allocation):
            raise ValueError('Historical comparator IDs/folds differ')
        for key,tail in ARRAY_SHAPES.items():
            arrays[key] = _finite(saved[key], (len(ids),*tail), 'historical '+key).copy()
        present = 'actual_u' in saved and 'mean_u' in saved
        if present:
            arrays['actual_u'] = _finite(saved['actual_u'], (len(ids),9), 'historical actual_u').copy()
            arrays['mean_u'] = _finite(saved['mean_u'], (len(ids),9), 'historical mean_u').copy()
        old_mask = np.asarray(saved['principal_mask'], float).copy() if 'principal_mask' in saved else None
    if not present:
        arrays.update(actual_u=np.empty((len(ids),9)), mean_u=np.empty((len(ids),9)))
        for record in manifest['folds']:
            ix=np.asarray(record['test'],int)
            path=source/'folds'/f"fold_{record['fold']}"/'arms'/arm/'test/u_predictions.npz'
            arrays['actual_u'][ix],arrays['mean_u'][ix]=_read_coordinate_file(path,ids[ix])
    _finish(arrays, ids, allocation)
    if old_mask is not None and not np.array_equal(old_mask, arrays['principal_mask']):
        raise ValueError('Historical saved mask differs from the unchanged budget rule')
    return arrays


def _finish(arrays, ids, allocation):
    if np.any(arrays['p_null']<0) or np.any(arrays['p_null']>1):
        raise ValueError('Saved NULL probabilities must lie in [0,1]')
    mask=selection_mask(arrays['predicted'][:,2],ids,allocation,.25,2)
    counts=[int(mask[allocation==fold].sum()) for fold in np.unique(allocation)]
    if sorted(counts)!=[15,16,16,16,16] or int(mask.sum())!=79:
        raise ValueError('The unchanged foldwise budget must select 79 objects / 158 wells')
    arrays.update(principal_mask=mask,u_object_mse=np.square(arrays['actual_u']-arrays['mean_u']).mean(1))


def _model_scores(arrays, ids, records):
    return dict(u_mse=float(arrays['u_object_mse'].mean()),u_scope=U_SCOPE,
        per_coordinate_u_mse=np.square(arrays['actual_u']-arrays['mean_u']).mean(0).tolist(),
        actions=_action_rows(arrays,records,global_model=False),
        principal_policy=_policy(arrays['actual'][:,2],arrays['principal_mask']),
        selected_ids=ids[arrays['principal_mask'].astype(bool)].tolist(),
        geometry_energy=float(arrays['geometry_energy'].mean()))


def _aggregate_diagnostics(rows):
    available=[(n,d,v) for n,d,v in rows if d['available']]
    result=dict(folds_available=len(available),total_folds=len(rows),
        activation_statistics_not_causal_attributions=True,drop_block_utility_not_evaluated=True)
    coefficient_rows=[d['local_coefficients'] for _,d,_ in available if 'local_coefficients' in d]
    if coefficient_rows:
        result['local_coefficients']=dict(per_fold=coefficient_rows,
            scope='separate fitted model parameters in each fold, not pooled object observations')
    gated=[(n,d,v) for n,d,v in available if 'gate' in v]
    if gated:
        shapes=[tuple(v['gate'].shape[1:]) for _,_,v in gated]
        if len(set(shapes))!=1:
            raise ValueError('Gate axes must keep the same meaning/dimensions across folds')
        all_values=np.concatenate([v['gate'].reshape(n,-1) for n,_,v in gated])
        weight=sum(n for n,_,_ in gated)
        variance=sum(n*np.asarray(d['gate']['within_fold_std'])**2 for n,d,_ in gated)/weight
        result['gate']=dict(n=weight,shape=list(shapes[0]),pooled_mean=all_values.mean(0).tolist(),
            pooled_std=all_values.std(0).tolist(),within_fold_std_rms=np.sqrt(variance).tolist(),
            interpretation='pooled variation also includes between-fold fitted-model differences')
    block_rows=[(n,d,v) for n,d,v in available if 'block_contributions' in v]
    if block_rows:
        sizes={v['block_contributions'].shape[1] for _,_,v in block_rows}
        if len(sizes)!=1:
            raise ValueError('The block axis must remain consistent across folds')
        weight=sum(n for n,_,_ in block_rows)
        squares=sum(n*np.asarray(d['block_contribution_rms'])**2 for n,d,_ in block_rows)/weight
        result['block_contributions']=dict(n=weight,block_names=block_rows[0][1]['block_names'],
            rms=np.sqrt(squares).tolist(),scope='raw/pre-final-tanh contributions, not additive final-mean or utility effects')
        closure=[d['sum_closure'] for _,d,_ in block_rows if 'sum_closure' in d]
        if closure:
            result['sum_closure']=dict(folds_checked=len(closure),formula=closure[0]['formula'],
                max_absolute_error=max(x['max_absolute_error'] for x in closure),
                diagnostic_only=True)
    return result


def _table(models, names):
    lines=['| 模型 | 九维 MSE↓ | Γ CRPS↓ | NULL Brier↓ | Γ Spearman | 选中对象均值Γ↑ | FDP | FPR | 每对象总群净值↑ |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        r=models[name];a=r['actions'][2];p=r['principal_policy']
        lines.append(f"| {name} | {_fmt(r['u_mse'])} | {_fmt(a['gamma_crps'])} | {_fmt(a['null_brier'])} | "
            f"{_fmt(a['spearman'],4)} | {_fmt(p['per_selected_net_gain'])} | {_fmt(p['fdp'],4)} | "
            f"{_fmt(p['fpr'],4)} | {_fmt(p['per_eligible_net_gain'])} |")
    return lines


def _write_reports(root,result):
    intro=['# 条件响应核：固定第30轮开发比较','',
        '新臂统一使用实际第30轮，验证集最佳轮次仅作附录。A_HR逐项匹配历史冻结结果。',
        '639个已开放DEV对象在同一五折各出现一次；每折选16/16/16/16/15对象，共79对象、158孔。','']
    short=intro+_table(result['models'],ARMS)+['','## 历史30轮参照（仅提供上下文）','']+_table(result['models'],HISTORICAL_ARMS)
    short+=['','原Γ、NULL/POSITIVE阈值、成本和预算未改。G对A/E/F及历史B/C/D的六组配对区间见完整报告。',
        '这是反复使用DEV的模型开发，不是新的独立认证；不能把gate变化或块RMS等同于生物机制贡献。']
    (root/'SUMMARY.md').write_text('\n'.join(short)+'\n')
    lines=intro+['## 新四臂结果','']+_table(result['models'],ARMS)
    lines+=['','## 历史参照','']+_table(result['models'],HISTORICAL_ARMS)
    lines+=['','九维MSE使用各折相同fit-only标准化坐标；合并Spearman/AUC为描述量，折内秩关联另存JSON。',
        'Γ已经包含原两孔成本，不重复扣费。NULL为Γ≤0，POSITIVE为Γ≥0.005。FDP以选中对象为分母；FPR以所有NULL对象为分母。',
        '选中对象平均Γ与每个可选对象的净值同时报告，不以其中一项替代纯度要求。',
        '历史B/C/D不是本轮新拟合的条件模型，输入或容量差异不能被解释成单一组件的纯消融。',
        '', '## G的六组配对比较','',
        '差为左减右；误差/CRPS/Brier/风险负值有利，净值正值有利。同一2000次折内对象配对复抽用于全部比较。','']
    lines+=_comparison_lines(result['comparisons'])
    lines+=['','## 各折方向与选择重叠','',
        '| 比较 | MSE有利/不利/并列 | CRPS有利/不利/并列 | Brier有利/不利/并列 | 净值有利/不利/并列 | 共同选择/79 |',
        '|---|---|---|---|---|---:|']
    for pair in result['comparisons'].values():
        d=pair['fold_directions'];cells=['/'.join(str(d[k][name]) for name in ('favorable','unfavorable','tied'))
            for k in ('u_mse','gamma_crps','null_brier','net_gain_per_eligible')]
        lines.append('| '+pair['left']+' − '+pair['right']+' | '+' | '.join(cells)+f" | {pair['overlap']['intersection_n']} |")
    lines+=['','## 网络参数与条件分支诊断','',
        '参数计数来自model.parameters()，含冻结HR网络，不含固定ridge及描述量缓冲值。下面逐折列出，不默认所有臂等容量。','',
        '| 模型 | 各折当前可训练 / 网络合计 |', '|---|---|']
    for arm in ARMS:
        counts=[f"{r['parameter_counts']['trainable']:,}/{r['parameter_counts']['total']:,}" for r in result['models'][arm]['folds']]
        lines.append('| '+arm+' | '+'；'.join(counts)+' |')
    for arm in ARMS[1:]:
        diag=result['models'][arm]['model_diagnostics']
        lines+=['',f'### {arm}','',f"存在诊断文件：{diag['folds_available']}/{diag['total_folds']}折。"]
        if 'gate' in diag:
            g=diag['gate'];lines.append('gate各分量折内标准差的加权RMS：'+', '.join(_fmt(v) for v in g['within_fold_std_rms'])+'。')
            lines.append('gate跨折合并标准差：'+', '.join(_fmt(v) for v in g['pooled_std'])+'；其中也包含各折拟合差异。')
        if 'block_contributions' in diag:
            block=diag['block_contributions'];names=block['block_names'] or [f'block_{i}' for i in range(len(block['rms']))]
            lines.append('各块raw修正RMS：'+'；'.join(f'{name}={_fmt(v)}' for name,v in zip(names,block['rms']))+'。')
        if 'sum_closure' in diag:
            lines.append('raw块贡献加偏置的闭合最大误差：'+f"{diag['sum_closure']['max_absolute_error']:.3g}"+'。')
        if 'local_coefficients' in diag:
            lines.append('各折局部组合系数相对初始1/3的RMS变化：'+', '.join(
                _fmt(v['rms_change_from_one_third']) for v in diag['local_coefficients']['per_fold'])+'；这是模型参数变化，不是对象间变化。')
    lines+=['','上述块统计是最终tanh之前的数值分解，不能当作最终均值或Γ的可加贡献。gate随对象变化只说明实现了条件调节，不证明调节方向正确。',
        '本次没有用这些诊断选择对象、重训gate或执行事后drop-block；相应机制消融可另外安排。',
        '', '## 解释范围','',
        '历史基线A的预测、概率、CRPS、九维均值、对象及选择掩码逐项相同；新旧原始目标也逐项匹配。',
        '区间条件于固定预测及掩码，不计入训练随机性、共享批次、训练集重叠、历史开发搜索或Monte Carlo误差；未作多重比较校正。',
        '所有臂如实保留。既有终点、七项合同和旧结果不变；FINAL与第五重复未打开。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'run_manifest.json').read_text())
    ids,allocation=_validate_manifest(manifest)
    source=Path(manifest['historical_reference_run']).expanduser()
    source=(source if source.is_absolute() else root/source).resolve()
    if source==root:
        raise ValueError('New conditional summaries must not overwrite historical results')
    data,models={},{}
    reference_actual=reference_u=None
    for arm in ARMS:
        arrays,rows,diagnostics=_read_new_arm(root,manifest,ids,arm)
        _finish(arrays,ids,allocation)
        if reference_actual is None:
            reference_actual,reference_u=arrays['actual'].copy(),arrays['actual_u'].copy()
        elif not np.array_equal(reference_actual,arrays['actual']) or not np.array_equal(reference_u,arrays['actual_u']):
            raise ValueError('New arms must share identical original Gamma and nine-u targets')
        for row,record in zip(rows,manifest['folds']):
            ix=np.asarray(record['test'],int)
            row['principal_policy']=_policy(arrays['actual'][ix,2],arrays['principal_mask'][ix])
        models[arm]=dict(**_model_scores(arrays,ids,manifest['folds']),folds=rows,
            model_diagnostics=_aggregate_diagnostics(diagnostics),historical_predictions_reused=arm=='A_HR')
        data[arm]=arrays
    historical_a=_read_historical(source,manifest,ids,allocation,'A_HR')
    for key in (*ARRAY_SHAPES,'actual_u','mean_u','principal_mask'):
        if not np.array_equal(historical_a[key],data['A_HR'][key]):
            raise ValueError('Frozen A_HR differs from the historical saved '+key)
    for arm in HISTORICAL_ARMS:
        arrays=_read_historical(source,manifest,ids,allocation,arm)
        if not np.array_equal(arrays['actual'],reference_actual) or not np.array_equal(arrays['actual_u'],reference_u):
            raise ValueError('Historical comparator uses different original Gamma or nine-u targets')
        data[arm]=arrays
        models[arm]=dict(**_model_scores(arrays,ids,manifest['folds']),historical_predictions_reused=True,
                        source_oof_file=str(source/f'{arm}_oof_predictions.npz'))
    seed=int(manifest['config'].get('bootstrap_seed',manifest['config'].get('seed',20260914)))
    rng=np.random.default_rng(seed)
    boot=np.column_stack([rng.choice(np.flatnonzero(allocation==fold),size=(2000,int(np.sum(allocation==fold))))
                          for fold in sorted(np.unique(allocation))])
    pairs={left+'__minus__'+right:_compare(left,right,data,ids,manifest['folds'],boot) for left,right in COMPARISONS}
    result=dict(complete=True,completed_utc=datetime.now(timezone.utc).isoformat(),n=len(ids),arms=list(ARMS),
        historical_comparator_arms=list(HISTORICAL_ARMS),models=models,comparisons=pairs,
        primary_comparisons=[a+'__minus__'+b for a,b in COMPARISONS[:3]],
        historical_context_comparisons=[a+'__minus__'+b for a,b in COMPARISONS[3:]],
        historical_reference_run=str(source),historical_A_exactly_matched=True,
        actual_checkpoint_epoch=30,checkpoint_selection='fixed epoch30; validation best only in appendix',
        fold_test_counts=[len(x['test']) for x in manifest['folds']],samples_per_object=2000,
        bootstrap_replicates=2000,bootstrap_seed=seed,interval_scope=INTERVAL_SCOPE,
        original_action='ADD_TWO / Z1Z2',principal_budget='25% physical wells, 79 objects / 158 wells',
        secondary_comparisons_multiplicity_adjusted=False,historical_dev=True,formal_certificate=False,
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    for arm in ARMS:
        np.savez_compressed(root/f'{arm}_oof_predictions.npz',ids=ids,fold=allocation,**data[arm])
    write_json(root/'summary.json',result)
    _write_reports(root,result)
    return result
