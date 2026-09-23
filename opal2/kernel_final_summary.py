"""Fixed-epoch 2x2x2 architecture readout from saved predictive distributions.

O/D denotes overlap/disjoint landmarks, G/S generic/structured bases and W/C
weighted/constrained geometry objectives. All eight arms remain in the report.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .biology_kernel_evaluation import write_json
from .conditional_response_summary import (
    ARRAY_SHAPES, _aggregate_diagnostics, _diagnostic_file, _finite, _finish,
    _model_scores, _read_coordinate_file, _table,
)
from .geometry_kernel_continuation_summary import _comparison_lines
from .geometry_kernel_replacement_summary import _parameter_counts
from .geometry_kernel_summary import _compare, _overlap, _policy
from .gram_oof_experiment import selection_mask
from .hierarchical_geometry_summary import _bootstrap_statistic, _folds, _fmt


ARMS = ('A_HR','O_G_W','O_S_W','O_G_C','O_S_C','D_G_W','D_S_W','D_G_C','D_S_C')
COMPARISON_GROUPS = {
    'structured_minus_generic': tuple((f'{a}_S_{c}',f'{a}_G_{c}') for a in ('O','D') for c in ('W','C')),
    'constrained_minus_weighted': tuple((f'{a}_{b}_C',f'{a}_{b}_W') for a in ('O','D') for b in ('G','S')),
    'disjoint_minus_overlap': tuple((f'D_{b}_{c}',f'O_{b}_{c}') for b in ('G','S') for c in ('W','C')),
    'new_minus_HR': tuple((a,'A_HR') for a in ARMS[1:]),
}
COMPARISONS = tuple(pair for pairs in COMPARISON_GROUPS.values() for pair in pairs)
CHUNKS = {'first2000':(0,2000),'first5000':(0,5000),'second5000':(5000,10000),'full10000':(0,10000)}


def _validate_manifest(manifest):
    if list(manifest['arms']) != list(ARMS): raise ValueError('All nine declared arms are required in order')
    for key in ('final_opened','fifth_repeat_opened','original_endpoint_changed','original_contract_changed'):
        if manifest.get(key) is not False: raise ValueError('Experiment does not preserve '+key)
    cfg=manifest['config']
    if cfg.get('stage_epochs')!=30 or cfg.get('samples')!=10000 or cfg.get('bootstrap',2000)!=2000:
        raise ValueError('Expected fixed epoch30, 10000 predictive draws and 2000 bootstrap replicates')
    ids,allocation=_folds(manifest)
    if len(ids)!=639 or len(manifest['folds'])!=5: raise ValueError('Expected 639 unique DEV objects and five folds')
    for record in manifest['folds']:
        parts=[np.asarray(record[k],int) for k in ('fit','inner_validation','test')]
        if any(not len(x) or len(np.unique(x))!=len(x) or np.any(x<0) or np.any(x>=len(ids)) for x in parts):
            raise ValueError('Invalid original fit/validation/test indices')
        if not np.array_equal(np.sort(np.concatenate(parts)),np.arange(len(ids))):
            raise ValueError('Original fit/validation/test must partition all639 objects')
    return ids,allocation


def _read_scopes(root,manifest,ids):
    result=[]
    for record in manifest['folds']:
        scope=json.loads((root/'folds'/f"fold_{record['fold']}"/'scope.json').read_text())
        for name,key in (('originalfit_ids','fit'),('validation_ids','inner_validation'),('test_ids','test')):
            if scope[name]!=ids[np.asarray(record[key],int)].tolist():
                raise ValueError('Scope differs from declared '+name)
        fit=scope['commonbranchfit_ids'];reference=scope['reference_ids'];original=scope['originalfit_ids']
        if len(fit) not in (344,345) or len(reference)!=64 or len(set(fit))!=len(fit) or len(set(reference))!=64:
            raise ValueError('Expected common344/345 branch-fit and64 reference objects')
        if set(fit)&set(reference) or set(fit)|set(reference)!=set(original):
            raise ValueError('Branch fit and reference must be a disjoint partition of original fit')
        anchors=scope['anchor_ids_by_mode']
        if set(anchors)!=set(('O','D')): raise ValueError('Anchor modes must be O and D')
        if len(anchors['O'])!=64 or len(set(anchors['O']))!=64 or not set(anchors['O'])<=set(fit):
            raise ValueError('Overlap anchors must be64 distinct common branch-fit objects')
        if len(anchors['D'])!=64 or set(anchors['D'])!=set(reference):
            raise ValueError('Disjoint anchors must equal the64 held-out reference objects')
        result.append(dict(scope,fold=record['fold']))
    return result


def _audit_learning_scope(root,scopes):
    import torch
    output=[]
    for scope in scopes:
        folder=root/'folds'/f"fold_{scope['fold']}"
        banks={mode:torch.load(folder/'banks'/f'{mode}.pt',map_location='cpu',weights_only=True)['config'] for mode in ('O','D')}
        for mode,bank in banks.items():
            descriptor=bank['descriptor_config']
            if bank['fitting_ids']!=scope['commonbranchfit_ids'] or descriptor['fitting_ids']!=scope['commonbranchfit_ids']:
                raise ValueError('Response bank scalers must fit only common branch-fit IDs')
            if descriptor['anchor_data']['anchor_ids']!=scope['anchor_ids_by_mode'][mode]:
                raise ValueError('Saved response bank anchors differ from scope')
        for arm in ARMS[1:]:
            saved=torch.load(folder/'arms'/arm/'epoch30.pt',map_location='cpu',weights_only=True)
            config=json.loads((folder/'arms'/arm/'training_config.json').read_text())
            if saved['epoch']!=30 or saved['optimizer_steps']!=180:
                raise ValueError('Saved checkpoint must record epoch30/180 optimizer steps')
            for key,expected in (('fit_ids',scope['commonbranchfit_ids']),('validation_ids',scope['validation_ids'])):
                if saved[key]!=expected or config[key]!=expected:
                    raise ValueError('Checkpoint/training configuration has different '+key)
            if saved['model_config']['bank_config']!=banks[arm[0]]:
                raise ValueError('Checkpoint bank configuration differs from frozen reference bank')
        output.append(dict(fold=scope['fold'],all_checkpoint_fit_ids_match=True,
            all_bank_scaler_fit_ids_match=True,all_declared_anchor_ids_match=True,
            checkpoint_epoch=30,optimizer_steps=180))
    return output


def _read_arm(root,manifest,ids,arm):
    n=len(ids);arrays={k:np.empty((n,*tail)) for k,tail in ARRAY_SHAPES.items()}
    arrays.update(actual_u=np.empty((n,9)),mean_u=np.empty((n,9)))
    covariance=np.empty((n,9,9));rows=[];diagnostics=[]
    chunks={name:dict(predicted=np.empty((n,3)),p_null=np.empty((n,3))) for name in CHUNKS}
    for record in manifest['folds']:
        ix=np.asarray(record['test'],int);folder=root/'folds'/f"fold_{record['fold']}"/'arms'/arm;evaluation=folder/'evaluation'
        metric=json.loads((evaluation/'metrics.json').read_text());meta=metric.get('model',{})
        epoch=metric.get('actual_checkpoint_epoch',meta.get('actual_checkpoint_epoch'))
        if metric.get('samples')!=10000: raise ValueError('Each evaluated arm requires10000 joint draws')
        if arm!='A_HR' and epoch!=30: raise ValueError('Every new arm must use its actual fixed epoch30')
        with np.load(evaluation/'predictions.npz',allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'],ids[ix]): raise ValueError('Saved predictive IDs differ')
            for key,tail in ARRAY_SHAPES.items(): arrays[key][ix]=_finite(saved[key],(len(ix),*tail),key)
            draws=_finite(saved['utility_samples'],(10000,len(ix),3),'saved utility samples')
            for name,(lo,hi) in CHUNKS.items():
                chunks[name]['predicted'][ix]=draws[lo:hi].mean(0)
                chunks[name]['p_null'][ix]=(draws[lo:hi]<=0).mean(0)
            if not np.allclose(chunks['full10000']['predicted'][ix],arrays['predicted'][ix],rtol=1e-12,atol=1e-14):
                raise ValueError('Stored means differ from all10000 saved utility draws')
            if not np.array_equal(chunks['full10000']['p_null'][ix],arrays['p_null'][ix]):
                raise ValueError('Stored NULL probabilities differ from all10000 saved draws')
        arrays['actual_u'][ix],arrays['mean_u'][ix]=_read_coordinate_file(evaluation/'u_predictions.npz',ids[ix])
        with np.load(evaluation/'u_predictions.npz',allow_pickle=False) as saved:
            covariance[ix]=_finite(saved['covariance_u'],(len(ix),9,9),'fixed covariance')
        complete_path=folder/'training_complete.json'
        complete=json.loads(complete_path.read_text()) if complete_path.is_file() else None
        history_path=folder/'history.jsonl'
        history=([json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
                 if history_path.is_file() else [])
        if arm!='A_HR':
            final=(complete or {}).get('final_epoch',(complete or {}).get('actual_checkpoint_epoch',(complete or {}).get('epoch')))
            if final!=30 or (complete or {}).get('optimizer_steps')!=180:
                raise ValueError('Training completion must identify actual epoch30/180 optimizer steps')
        checkpoint=dict(actual_checkpoint_epoch=epoch,test_metadata=meta,training_complete=complete,
            validation_trajectory=[x for x in history if x.get('epoch') in (0,5,10,15,20,25,30)],
            best_checkpoint_not_used_for_main=True)
        diagnostic,numeric=_diagnostic_file(folder,ids[ix]);diagnostics.append((len(ix),diagnostic,numeric))
        counts=_parameter_counts(checkpoint,arm)
        if arm!='A_HR' and counts['trainable']!=3837:
            raise ValueError('Each new arm must have3837 trainable parameters')
        rows.append(dict(fold=record['fold'],n=len(ix),checkpoint=checkpoint,
            parameter_counts=counts,
            geometry_feasibility=(complete or {}).get('geometry_feasibility'),diagnostics=diagnostic))
    return arrays,rows,diagnostics,chunks,covariance


def _sampling_stability(chunks,actual,ids,allocation):
    masks={name:selection_mask(values['predicted'][:,2],ids,allocation,.25,2) for name,values in chunks.items()}
    policies={name:_policy(actual,masks[name]) for name in CHUNKS}
    comparisons={}
    for left,right in (('first2000','full10000'),('first5000','full10000'),('second5000','full10000'),('first5000','second5000')):
        delta=chunks[left]['predicted'][:,2]-chunks[right]['predicted'][:,2]
        comparisons[left+'__vs__'+right]=dict(left=left,right=right,
            predicted_gamma_rms_difference=float(np.sqrt(np.mean(delta**2))),
            predicted_gamma_max_absolute_difference=float(np.abs(delta).max()),
            predicted_null_mean_absolute_difference=float(np.abs(chunks[left]['p_null'][:,2]-chunks[right]['p_null'][:,2]).mean()),
            overlap=_overlap(masks[left],masks[right],ids),
            selected_null_count_difference=policies[left]['selected_null_count']-policies[right]['selected_null_count'],
            selected_actual_value_difference=policies[left]['per_selected_net_gain']-policies[right]['per_selected_net_gain'])
    return dict(policies=policies,comparisons=comparisons,
        primary='full10000',chunk_selection_permitted=False,
        interpretation='numerical integration diagnostic conditional on frozen predictions; nested chunks are dependent; no best chunk selected')


def _feasibility(rows):
    values=[dict(fold=r['fold'],reported=r['geometry_feasibility']) for r in rows]
    def flags(key):
        return [(row['reported'].get(key) if isinstance(row['reported'],dict) and
                 isinstance(row['reported'].get(key),bool) else None) for row in values]
    statuses=flags('fit_satisfied'); validation=flags('validation_satisfied')
    return dict(per_fold=values,known_satisfied_folds=sum(x is True for x in statuses),
        known_violated_folds=sum(x is False for x in statuses),unresolved_folds=sum(x is None for x in statuses),
        validation_satisfied_folds=sum(x is True for x in validation),
        validation_violated_folds=sum(x is False for x in validation),validation_unresolved_folds=sum(x is None for x in validation),
        all_fit_constraints_satisfied=all(x is True for x in statuses),
        all_validation_geometry_noninferior=all(x is True for x in validation),default_model_declared=False)


def _factorial_interactions(data,records,boot):
    contrasts={}
    for loss in ('W','C'):
        contrasts['reference_by_basis__'+loss]=((f'D_S_{loss}',1),(f'D_G_{loss}',-1),(f'O_S_{loss}',-1),(f'O_G_{loss}',1))
    for reference in ('O','D'):
        contrasts['loss_by_basis__'+reference]=((f'{reference}_S_C',1),(f'{reference}_S_W',-1),(f'{reference}_G_C',-1),(f'{reference}_G_W',1))
    null=data['A_HR']['actual'][:,2]<=0
    scores={arm:dict(u_mse=x['u_object_mse'],gamma_crps=x['utility_crps'][:,2],
                    null_brier=(x['p_null'][:,2]-null)**2) for arm,x in data.items()}
    result={}
    for name,terms in contrasts.items():
        delta={k:sum(sign*scores[arm][k] for arm,sign in terms) for k in ('u_mse','gamma_crps','null_brier')}
        result[name]=dict(terms=[dict(arm=arm,coefficient=sign) for arm,sign in terms],
            statistics={k:_bootstrap_statistic(v,boot) for k,v in delta.items()},
            folds=[dict(fold=r['fold'],**{k:float(v[np.asarray(r['test'],int)].mean()) for k,v in delta.items()}) for r in records],
            causal_claim=False,policy_interaction_evaluated=False,
            scope='paired score difference-of-differences, exploratory fixed-model comparison')
    return result


def _write_reports(root,result):
    intro=['# 最后一轮2×2×2架构比较：固定第30轮','',
        'O/D：训练对象重叠/独立参考landmarks；G/S：通用/结构化基函数；W/C：原加权损失/Γ导向且受几何约束的目标。',
        '八个新臂共享每折344/345训练对象、相同验证和测试对象。固定第30轮全部保留，没有按test挑最佳检查点或fallback。',
        '639个已开放DEV对象各一次OOF；每个模型用10000次联合采样，原预算79对象/158孔。','']
    short=intro+_table(result['models'],ARMS)+['','## Γ均值预测与排序补充','',
        '| 臂 | Γ均值MSE↓ | NULL AUC |','|---|---:|---:|']
    for arm in ARMS:
        a=result['models'][arm]['actions'][2]
        short.append(f"| {arm} | {_fmt(a['mse'])} | {_fmt(a['null_auc'],4)} |")
    short+=['','## 内层几何约束状态','',
        '| 臂 | fit满足/违反/未明 | validation不劣/变差/未明 |','|---|---|---|']
    for arm in ARMS[1:]:
        f=result['models'][arm]['feasibility']
        short.append(f"| {arm} | {f['known_satisfied_folds']}/{f['known_violated_folds']}/{f['unresolved_folds']} | "
            f"{f['validation_satisfied_folds']}/{f['validation_violated_folds']}/{f['validation_unresolved_folds']} |")
    short+=['','约束状态为训练完成文件的原记录；有违反或未明确项时不能当作全部可行。满足几何约束也不等于收益/风险已认证。',
        '本轮尚未据此指定默认架构。所有配对、参考HR的实际价值与风险及采样稳定性见完整报告。']
    (root/'SUMMARY.md').write_text('\n'.join(short)+'\n')
    lines=short+['','## 全部20组配对比较','',
        '左减右；误差/风险负值有利，实际净值正值有利。同一2000次折内对象配对复抽用于全部比较。','']
    for group,keys in result['comparison_groups'].items():
        lines+=['',f'### {group}','']+_comparison_lines({k:result['comparisons'][k] for k in keys})
    lines+=['','## 因素交互：配对差分之差','',
        'reference_by_basis=(D_S−D_G)−(O_S−O_G)，分别固定W/C；loss_by_basis=(S_C−S_W)−(G_C−G_W)，分别固定O/D。',
        '这不是生物机制因果分解，也不包含策略价值交互。','',
        '| 对照 | 指标 | 差分之差 | 条件式95%区间 |','|---|---|---:|---|']
    for name,entry in result['factorial_interactions'].items():
        for key,stat in entry['statistics'].items():
            lo,hi=stat['interval95']
            lines.append(f"| {name} | {key} | {_fmt(stat['mean'])} | [{_fmt(lo)}, {_fmt(hi)}] |")
    lines+=['','## 联合采样数值稳定性','',
        '| 臂 | draw块 | 选中NULL/79 | 选中均值Γ | 与全10000共同选择/79 |','|---|---|---:|---:|---:|']
    for arm in ARMS:
        sampling=result['models'][arm]['sampling_stability']
        for chunk in CHUNKS:
            p=sampling['policies'][chunk]
            overlap=79 if chunk=='full10000' else sampling['comparisons'][chunk+'__vs__full10000']['overlap']['intersection_n']
            lines.append(f"| {arm} | {chunk} | {p['selected_null_count']} | {_fmt(p['per_selected_net_gain'])} | {overlap} |")
    lines+=['','各臂还保存前5000与后5000的直接对照。全部结果仍以10000为主；不按某个draw块的风险或收益选择结果。',
        '这些是冻结模型的数值积分检查，不含重新训练、换分区或新对象的不确定性。','',
        '## 解释范围','',
        '新的八臂是完整实现的单因素组合；S−G、C−W、D−O分别在其他两因素固定的四个条件下报告。所有新臂还与固定HR比较。',
        '每折预测协方差必须与A完全一致；均值和原始Γ目标也逐对象核对。landmarks与分支训练对象的关系由scope.json核实。',
        'Γ已扣除原两孔成本；FDP分母为选中对象，FPR分母为所有NULL。不能用较好几何评分或CRPS替代实际价值与风险。',
        '相关系数/AUC仅作描述。所有95%区间条件于固定模型预测和掩码，不含训练、批次依赖、反复DEV开发或Monte Carlo误差，未作多重比较校正。',
        '该轮为探索性开发比较，不是独立认证、无条件置信保证或多重检验后的架构胜出证明。FINAL、第五重复、原合同和旧结果不变。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def summarize(root):
    root=Path(root).resolve();manifest=json.loads((root/'run_manifest.json').read_text())
    ids,allocation=_validate_manifest(manifest);scopes=_read_scopes(root,manifest,ids)
    learning_scope=_audit_learning_scope(root,scopes)
    data,models={},{};actual=actual_u=covariance=None
    for arm in ARMS:
        arrays,rows,diagnostics,chunks,cov=_read_arm(root,manifest,ids,arm);_finish(arrays,ids,allocation)
        if actual is None: actual,actual_u,covariance=arrays['actual'].copy(),arrays['actual_u'].copy(),cov.copy()
        else:
            if not np.array_equal(arrays['actual'],actual) or not np.array_equal(arrays['actual_u'],actual_u):
                raise ValueError('All arms must share identical original Gamma and nine-u targets')
            if not np.array_equal(cov,covariance): raise ValueError('Every new arm must preserve the frozen HR covariance')
        for row,record in zip(rows,manifest['folds']):
            ix=np.asarray(record['test'],int)
            row['principal_policy']=_policy(arrays['actual'][ix,2],arrays['principal_mask'][ix])
        models[arm]=dict(**_model_scores(arrays,ids,manifest['folds']),folds=rows,
            feasibility=_feasibility(rows) if arm!='A_HR' else None,
            model_diagnostics=_aggregate_diagnostics(diagnostics),
            sampling_stability=_sampling_stability(chunks,arrays['actual'][:,2],ids,allocation))
        data[arm]=arrays
    seed=int(manifest['config'].get('bootstrap_seed',manifest['config'].get('seed',20260915)))
    rng=np.random.default_rng(seed)
    boot=np.column_stack([rng.choice(np.flatnonzero(allocation==fold),size=(2000,int(np.sum(allocation==fold))))
                          for fold in sorted(np.unique(allocation))])
    pairs={a+'__minus__'+b:_compare(a,b,data,ids,manifest['folds'],boot) for a,b in COMPARISONS}
    result=dict(complete=True,completed_utc=datetime.now(timezone.utc).isoformat(),n=len(ids),arms=list(ARMS),
        models=models,comparisons=pairs,comparison_groups={k:[a+'__minus__'+b for a,b in v] for k,v in COMPARISON_GROUPS.items()},
        fold_scopes=scopes,learning_scope_audit=learning_scope,factorial_interactions=_factorial_interactions(data,manifest['folds'],boot),
        actual_checkpoint_epoch=30,samples_per_object=10000,bootstrap_replicates=2000,bootstrap_seed=seed,
        covariance_identical_across_arms=True,full10000_is_only_primary_readout=True,default_model_declared=False,
        principal_budget='25% physical wells,79 objects/158 wells',original_action='ADD_TWO / Z1Z2',
        fold_test_counts=[len(r['test']) for r in manifest['folds']],
        interval_scope='fixed-prediction within-outer-fold paired compound bootstrap; repeated DEV, not formal certification; excludes MC/training/batch dependence',
        formal_certificate=False,historical_dev=True,multiplicity_adjusted=False,
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    for arm in ARMS: np.savez_compressed(root/f'{arm}_oof_predictions.npz',ids=ids,fold=allocation,**data[arm])
    write_json(root/'summary.json',result);_write_reports(root,result)
    return result
