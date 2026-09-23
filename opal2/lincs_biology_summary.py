"""Paired saved-OOF readout, with no new model/threshold/checkpoint selection."""
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


ARMS = ('HR','A_OLD_GENERIC','B_OLD_STRUCTURED','C_BIO_GENERIC','D_BIO_STRUCTURED')
PAIRS = (('B_OLD_STRUCTURED','A_OLD_GENERIC'),('C_BIO_GENERIC','A_OLD_GENERIC'),
         ('D_BIO_STRUCTURED','C_BIO_GENERIC')) + tuple((a,'HR') for a in ARMS[1:])


def bootstrap_weights(groups, repeats, seed):
    unique, inverse = np.unique(np.asarray(groups,str), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(unique), np.full(len(unique),1/len(unique)), size=repeats)
    return counts[:,inverse].astype(float)


def interval(numerator, denominator, weights):
    numerator, denominator = np.asarray(numerator,float), np.asarray(denominator,float)
    den = weights@denominator
    valid = den > 0
    samples = (weights@numerator)[valid]/den[valid]
    return dict(estimate=float(numerator.sum()/denominator.sum()) if denominator.sum()>0 else None,
        interval95=np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
        valid_resamples=int(valid.sum()))


def difference(a_num,a_den,b_num,b_den,weights):
    ad,bd = weights@a_den,weights@b_den
    valid = (ad>0)&(bd>0)
    samples = (weights@a_num)[valid]/ad[valid]-(weights@b_num)[valid]/bd[valid]
    return dict(estimate=float(a_num.sum()/a_den.sum()-b_num.sum()/b_den.sum()),
        interval95=np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
        valid_resamples=int(valid.sum()))


def read_arm(root,manifest,ids,allocation,arm):
    arrays = {key:np.empty((len(ids),*tail)) for key,tail in ARRAY_SHAPES.items()}
    arrays.update(actual_u=np.empty((len(ids),9)),mean_u=np.empty((len(ids),9)))
    rows=[]
    for record in manifest['folds']:
        ix=np.asarray(record['test'],int)
        folder=root/'folds'/f"fold_{record['fold']}"/'arms'/arm
        with np.load(folder/'evaluation/predictions.npz',allow_pickle=False) as z:
            if not np.array_equal(z['ids'],ids[ix]):
                raise ValueError('Prediction identity/order mismatch')
            for key,tail in ARRAY_SHAPES.items():
                if z[key].shape!=(len(ix),*tail) or not np.isfinite(z[key]).all():
                    raise ValueError('Nonfinite/misaligned predictive readout: '+key)
                arrays[key][ix]=z[key]
            draws=z['utility_samples']
            if draws.shape!=(10000,len(ix),3):
                raise ValueError('Expected all 10000 joint utility draws')
            if not np.allclose(draws.mean(0),z['predicted'],rtol=1e-12,atol=1e-14):
                raise ValueError('Stored predictive means do not match all draws')
            if not np.array_equal((draws<=0).mean(0),z['p_null']):
                raise ValueError('Stored NULL probabilities do not match all draws')
        with np.load(folder/'evaluation/u_predictions.npz',allow_pickle=False) as z:
            if not np.array_equal(z['ids'],ids[ix]):
                raise ValueError('Coordinate identity/order mismatch')
            if z['actual_u'].shape!=(len(ix),9) or z['mean_u'].shape!=(len(ix),9):
                raise ValueError('Coordinate predictions require one complete nine-vector per object')
            arrays['actual_u'][ix],arrays['mean_u'][ix]=z['actual_u'],z['mean_u']
        row=dict(fold=record['fold'],n=len(ix))
        if arm!='HR':
            complete=json.loads((folder/'training_complete.json').read_text())
            if complete['actual_checkpoint_epoch']!=30 or complete['trainable_parameters']!=3837:
                raise ValueError('Four arms must retain actual epoch30 and 3837 active parameters')
            history=[json.loads(s) for s in (folder/'history.jsonl').read_text().splitlines()]
            row.update(training=complete,history=history)
        rows.append(row)
    mask=selection_mask(arrays['predicted'][:,2],ids,allocation,.25,2)
    if not np.isfinite(arrays['actual_u']).all() or not np.isfinite(arrays['mean_u']).all():
        raise ValueError('Nonfinite coordinate scores')
    arrays.update(principal_mask=mask,u_object_mse=((arrays['actual_u']-arrays['mean_u'])**2).mean(1))
    return arrays,rows


def compare(left,right,data,weights,records):
    a,b=data[left],data[right]
    actual=a['actual'][:,2];null=(actual<=0).astype(float);ones=np.ones(len(actual))
    ma,mb=a['principal_mask'],b['principal_mask']
    score_deltas=dict(u_mse=a['u_object_mse']-b['u_object_mse'],
        gamma_crps=a['utility_crps'][:,2]-b['utility_crps'][:,2],
        null_brier=(a['p_null'][:,2]-null)**2-(b['p_null'][:,2]-null)**2,
        net_gain_per_eligible=(ma-mb)*actual)
    result=dict(left=left,right=right,direction='left minus right',
        **{key:interval(value,ones,weights) for key,value in score_deltas.items()},
        net_gain_per_selected=difference(ma*actual,ma,mb*actual,mb,weights),
        fdp=difference(ma*null,ma,mb*null,mb,weights),
        fpr=interval((ma-mb)*null,null,weights),
        changed_selected_objects=int(np.sum(ma!=mb)),
        selected_null_count_difference=int(ma@null-mb@null))
    result['folds']=[]
    for record in records:
        ii=np.asarray(record['test'],int)
        result['folds'].append(dict(fold=record['fold'],n=len(ii),
            **{key:float(v[ii].mean()) for key,v in score_deltas.items()},
            left_policy=_policy(actual[ii],ma[ii]),right_policy=_policy(actual[ii],mb[ii])))
    return result


def summarize(output):
    root=Path(output);manifest=json.loads((root/'run_manifest.json').read_text())
    if manifest['arms']!=list(ARMS) or manifest['fixed_epochs']!=30:
        raise ValueError('Unexpected four-arm declaration')
    ids,allocation=_folds(manifest)
    data,models={},{}
    for arm in ARMS:
        data[arm],folds=read_arm(root,manifest,ids,allocation,arm)
        if arm!='HR' and not np.array_equal(data[arm]['actual'],data['HR']['actual']):
            raise ValueError('Arms have different realized endpoints')
        models[arm]=dict(_model_scores(data[arm],ids,manifest['folds']),folds=folds)
    # Shared connectivity groups are resampled jointly. The measurements also
    # share plates; a separate layout-block sensitivity is not an i.i.d. certificate.
    weights=bootstrap_weights(manifest['groups'],2000,manifest['config']['seed']+701)
    comparisons={left+'__minus__'+right:compare(left,right,data,weights,manifest['folds']) for left,right in PAIRS}
    units=manifest['dataset']['units']
    layout=[str(u['layout_block']) for u in units]
    blockweights=bootstrap_weights(layout,2000,manifest['config']['seed']+702)
    blockcomparisons={left+'__minus__'+right:compare(left,right,data,blockweights,manifest['folds']) for left,right in PAIRS}
    actual=data['HR']['actual'][:,2]
    rng=np.random.default_rng(manifest['config']['seed']+703)
    random_values,random_null=[],[]
    for _ in range(2000):
        selected=[]
        for record in manifest['folds']:
            ix=np.asarray(record['test'],int);k=int(np.floor(.25*len(ix)/2))
            selected.extend(rng.choice(ix,k,replace=False).tolist())
        random_values.append(float(actual[selected].mean()));random_null.append(int((actual[selected]<=0).sum()))
    result=dict(complete=True,n=len(ids),arms=list(ARMS),fixed_epoch=30,models=models,
        comparisons=comparisons,layout_block_sensitivity=blockcomparisons,
        baseline_population=dict(mean_gamma=float(actual.mean()),null_count=int((actual<=0).sum()),
            null_rate=float((actual<=0).mean()),positive_count=int((actual>=.005).sum())),
        same_budget_random=dict(repeats=2000,selected_n=len(selected),used_wells=2*len(selected),
            mean_selected_gamma=float(np.mean(random_values)),
            selected_gamma_interval95=np.quantile(random_values,[.025,.975]).tolist(),
            expected_null_count=float(np.mean(random_null)),
            null_count_interval95=np.quantile(random_null,[.025,.975]).tolist()),
        uncertainty_scope='paired bootstrap conditional on fitted OOF predictions; chemistry groups and layout groups separately; no refitting or formal certification',
        shared_plates=True,original_JUMP_unchanged=True,formal_certificate=False)
    write_json(root/'summary.json',result)
    for arm,arrays in data.items():
        np.savez_compressed(root/f'{arm}_oof.npz',ids=ids,fold=allocation,**arrays)
    lines=['# LINCS：四组各30轮的比较','',
        f'{len(ids)}个化合物，五折，每个对象一次折外预测；新增分支每组3,837个可训练参数。',
        'A：原信息＋通用；B：原信息＋旧结构化；C：新增靶点/MoA＋通用；D：同C信息＋关系型生物函数。',
        'C/D保留同一个旧generic分支，只改变新增生物响应。HR为在LINCS重新拟合的共同基础模型。','',
        '| 模型 | 几何MSE↓ | Γ CRPS↓ | NULL Brier↓ | Γ Spearman | 选中对象/孔 | 选中实际Γ均值↑ | NULL个数 | FDP↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m=models[arm];a=m['actions'][2];p=m['principal_policy']
        rho='NA' if a['spearman'] is None else f"{a['spearman']:.4f}"
        lines.append(f"|{arm}|{m['u_mse']:.6f}|{a['gamma_crps']:.6f}|{a['null_brier']:.6f}|{rho}|"
            f"{p['selected_n']}/{p['used_wells']}|{p['per_selected_net_gain']:.6f}|{p['selected_null_count']}|{p['fdp']:.3%}|")
    lines+=['','预算是25%的额外物理孔，每个ADD_TWO占两孔，不是选择25%的化合物。',
        f"同预算随机选择的平均实际Γ为{np.mean(random_values):.6f}，平均NULL数为{np.mean(random_null):.2f}。",'',
        '## 三个主要配对比较','',
        '| 对比 | Γ CRPS差值（95%区间）↓ | 选中实际Γ差值（95%区间）↑ | FDP差值（95%区间）↓ |',
        '|---|---|---|---|']
    for left,right in PAIRS[:3]:
        c=comparisons[left+'__minus__'+right]
        def fmt(key):
            s=c[key];lo,hi=s['interval95'];return f"{s['estimate']:+.6f} [{lo:+.6f}, {hi:+.6f}]"
        lines.append(f"|{left}−{right}|{fmt('gamma_crps')}|{fmt('net_gain_per_selected')}|{fmt('fdp')}|")
    lines+=['','完整summary.json同时报告所有组对HR的差值、FPR、逐折方向、验证轨迹和板布局分组敏感性。',
        '配对区间跨零时，本轮不能区分两组优劣；不能把略低的NULL计数单独叫作稳定改进。',
        '', '## 解释范围','',
        '这里加入的是有来源的靶点/MoA关系，不是测得的Kd、占据率或药效方程。未知注释保留独立缺失标记。',
        '所有对象为A549、48 h、标称10 µM，实际剂量按对象记录；固定孔位与共享板仍限制外推。',
        '旧JUMP结果、FINAL、第五重复及七项合同未改。本轮是模型开发比较，不发放统计授权。',
        '', '数据与注释来源：[官方LINCS Cell Painting](https://github.com/broadinstitute/lincs-cell-painting)。']
    (root/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    summarize(parser.parse_args().output)
