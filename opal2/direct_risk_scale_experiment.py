"""Outer-isolated direct-CRPS scale calibration on opened LINCS DEV only."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz, paired_intervals
from .biology_kernel_evaluation import write_json
from .empirical_radial import draw_radial, fit_radial
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .lincs_biology_experiment import load_data
from .direct_risk_scale import centered_ecdf, fit_direct_map, scale_function
from .variance_headroom_experiment import (
    PROJECT,RADIAL,DUAL,DESCRIPTORS,evaluate_cell,summary_metrics,gamma_fast)

PLAN=PROJECT/'protocols/historical/NULL_ORACLE_DIRECT_RISK_PLAN_20260917.md'
ARMS=('CORE','DIRECT_GLOBAL','DIRECT_AMPLITUDE','DIRECT_DESCRIPTORS')
SEED=202609173
SEARCH_SAMPLES=4096
EVAL_SAMPLES=100000


def save_snapshot(root):
    sources=[PLAN,Path(__file__),PROJECT/'opal2/direct_risk_scale.py',
             PROJECT/'opal2/direct_risk_score_cache.py']
    for source in sources:
        target=root/('PROTOCOL.md' if source==PLAN else source.name)
        if target.exists() and target.read_bytes()!=source.read_bytes():
            raise ValueError(f'Changed implementation: {source.name}')
        shutil.copy2(source,target)


def prepare_calibration(fold,root,data,fit,test):
    nested=DUAL/f'fold_{fold}'/'nested_distribution'
    all_inner=read_npz(nested/'distribution.npz')
    score=read_npz(root/f'fold_{fold}'/'honest_scores.npz')
    inner_ids=all_inner['ids'];groups=all_inner['groups']
    np.testing.assert_array_equal(inner_ids,score['ids'])
    np.testing.assert_array_equal(inner_ids,data['ids'][fit])
    if set(groups)&set(data['groups'][test]):raise ValueError('Outer group leakage')
    lookup={v:i for i,v in enumerate(inner_ids)};cells=[];seen=np.zeros(len(inner_ids),int)
    for j in range(6):
        record=read_json(nested/f'cell_{j:02d}.json')
        value=read_npz(nested/f'cell_{j:02d}.npz')
        q=np.array([lookup[v] for v in value['query_ids']])
        query_groups=set(groups[q])
        used_ids=set(record['mean_fit_ids']+record['mean_validation_ids']+record['mean_reference_ids']+
                     record['fit_ids']+record['cal_ids'])
        if set(value['query_ids'])&used_ids:raise ValueError('Own outcome used for inner CORE')
        used_groups={groups[lookup[v]] for v in used_ids}
        if query_groups&used_groups:raise ValueError('Own chemistry group used for inner CORE')
        law=record['law'];law['log_centers']=np.asarray(law['log_centers'])
        rng=np.random.default_rng(SEED+10000*fold+101*j)
        m=len(q);normal=rng.normal(size=(SEARCH_SAMPLES,m,9))
        errors=draw_radial(law,value['query_radial_weights'],value['query_raw_scatter'],normal,
                          rng.random((SEARCH_SAMPLES,m)),rng.random((SEARCH_SAMPLES,m)))
        target=all_inner['raw_target'][q]
        cells.append(dict(indices=q,mean=value['query_raw_mean'],error=errors,actual=gamma_fast(target)))
        np.add.at(seen,q,1)
    np.testing.assert_array_equal(seen,np.ones(len(inner_ids),int))
    for key in ('amplitude_ecdf','descriptors_rank6_ecdf'):
        if score[key].shape!=(len(inner_ids),) or not np.isfinite(score[key]).all():
            raise ValueError(f'Invalid honest score {key}')
    return cells,inner_ids,groups,score


def fit_fold(fold,root,data,fit,test):
    directory=root/f'fold_{fold}';directory.mkdir(exist_ok=True)
    if (directory/'direct_fit.json').exists():return read_json(directory/'direct_fit.json')
    cells,ids,groups,scores=prepare_calibration(fold,root,data,fit,test)
    cache={};models={};records={};start=time.monotonic()
    for arm,key,dimension in [('DIRECT_GLOBAL','amplitude_ecdf',1),
                               ('DIRECT_AMPLITUDE','amplitude_ecdf',2),
                               ('DIRECT_DESCRIPTORS','descriptors_rank6_ecdf',2)]:
        print(f'fold {fold} fitting {arm} on {len(ids)} inner OOF objects',flush=True)
        write_json(root/'status.json',dict(state='RUNNING',stage='direct_mean_CRPS_fit',fold=fold,arm=arm))
        result=fit_direct_map(cells,groups,scores[key],dimensions=dimension,grid_cache=cache)
        for name in ('increment','candidate_scores','baseline_scores'):
            records[arm+'_'+name]=result.pop(name)
        models[arm]=result
        write_json(directory/'fit_progress.json',models)
    np.savez_compressed(directory/'calibration_fit.npz',ids=ids,groups=groups,
        amplitude_ecdf=scores['amplitude_ecdf'],descriptors_rank6_ecdf=scores['descriptors_rank6_ecdf'],**records)
    result=dict(fold=fold,models=models,fit_ids=ids.tolist(),outer_test_ids=data['ids'][test].tolist(),
        elapsed_seconds=time.monotonic()-start,outer_outcome_used=False,biological_gate_used=False)
    write_json(directory/'direct_fit.json',result)
    return result


def outer_scores(fold,data,metadata,fit,q):
    logamp=np.log(np.linalg.norm(data['Y'][:,0],axis=1))
    if not np.isfinite(logamp).all():raise ValueError('Nonfinite X amplitude')
    record=joblib.load(DESCRIPTORS/f'fold_{fold}'/'conditioners.joblib')
    model=record['conditional']['DESCRIPTORS']
    if set(model.fit_ids)!=set(data['ids'][fit]) or set(model.fit_ids)&set(data['ids'][q]):
        raise ValueError('Outer descriptor fit scope mismatch')
    descriptor=record['transformer'].transform(data,metadata).values
    prediction=model.predict(descriptor)[:,1]
    return dict(DIRECT_GLOBAL=np.zeros(len(q)),
        DIRECT_AMPLITUDE=centered_ecdf(logamp[fit],logamp[q]),
        DIRECT_DESCRIPTORS=centered_ecdf(prediction[fit],prediction[q]))


def summarize(root,data,metadata,cells,elapsed):
    ids=data['ids'];lookup={v:i for i,v in enumerate(ids)};n=len(ids)
    support=read_npz(DUAL/'GELU.npz')['resource_support'].astype(bool)
    actual=read_npz(DUAL/'CORE.npz')['actual'];layout=np.array([u['layout_block'] for u in metadata['units']])
    stores={};metrics={};comparisons={}
    for arm in ARMS:
        pieces=[];seen=np.zeros(n,int)
        for cell in cells:
            value=read_npz(root/f'cell_{cell["fold"]}_{cell["half"]}'/(arm+'.npz'))
            q=np.array([lookup[v] for v in value['ids']]);pieces.append((q,value));np.add.at(seen,q,1)
        np.testing.assert_array_equal(seen,np.ones(n,int))
        keys=set.intersection(*[{k for k,v in a.items() if k!='ids' and v.shape[:1]==(len(q),)} for q,a in pieces])
        out={k:np.empty((n,*pieces[0][1][k].shape[1:]),dtype=pieces[0][1][k].dtype) for k in keys}
        for q,v in pieces:
            for k in keys:out[k][q]=v[k]
        out['policy_value']=out['selected']*actual;out['policy_null']=out['selected']*(actual<=0)
        stores[arm]=out
        metrics[arm]=dict(full=summary_metrics(out,actual,np.ones(n,bool)),
            supported=summary_metrics(out,actual,support),active=int(np.any(out['increment']!=0,axis=1).sum()))
        np.savez_compressed(root/(arm+'.npz'),ids=ids,groups=data['groups'],layout=layout,actual=actual,support=support,**out)
    for arm in ARMS[1:]:
        comparisons[arm+'_minus_CORE']=dict(
            full=paired_intervals(stores[arm],stores['CORE'],np.ones(n,bool),data['groups'],layout,keys=('crps','brier','nll')),
            supported=paired_intervals(stores[arm],stores['CORE'],support,data['groups'],layout,keys=('crps','brier','nll')),
            policy=paired_intervals(stores[arm],stores['CORE'],np.ones(n,bool),data['groups'],layout,keys=('policy_value','policy_null')))
    comparisons['DIRECT_DESCRIPTORS_minus_DIRECT_AMPLITUDE']=paired_intervals(
        stores['DIRECT_DESCRIPTORS'],stores['DIRECT_AMPLITUDE'],np.ones(n,bool),data['groups'],layout,keys=('crps','brier','nll'))
    result=dict(state='COMPLETE',n=n,metrics=metrics,comparisons=comparisons,cells=cells,
        fits=[read_json(root/f'fold_{fold}'/'direct_fit.json') for fold in range(5)],
        elapsed_seconds=elapsed,search_samples=SEARCH_SAMPLES,evaluation_samples=EVAL_SAMPLES,
        main_rule_changed=False,mean_changed=False,final_opened=False,new_neural_training=False,
        biology_enabled=False,scope='Repeatedly opened development cohort, not independent certification')
    write_json(root/'summary.json',result)
    lines=['# Direct-risk scale calibration','','No biological branch or JEPA was enabled.','',
        '| Arm | Full Gamma CRPS | Brier | Joint NLL | Selected NULL/n | Selected mean Gamma |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,v in metrics.items():
        m=v['full'];lines.append(f'| {arm} | {m["crps"]:.8f} | {m["brier"]:.8f} | {m["nll"]:.5f} | {m["selected_null"]}/{m["selected_n"]} | {m["selected_mean"]:.8f} |')
    lines+=['','All candidate maps were fitted inside each outer MODEL_FIT using nested full-CORE distributions.',
        'The descriptive score predictor also excludes each inner query group. This is a development comparison, not a new held-out certification.',
        'A small change in NULL count must be read with Monte Carlo and cohort uncertainty, not as a standalone purity gain.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def run(output,fit_only=False):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    save_snapshot(root)
    old=read_json(RADIAL/'summary.json');manifest=read_json(Path(old['reference_run'])/'run_manifest.json')
    data,metadata=load_data(old['data_directory'])
    if len(data['ids'])!=1188 or data['ids'].tolist()!=manifest['ids']:raise ValueError('Changed cohort')
    write_json(root/'spec.json',dict(seed=SEED,search_samples=SEARCH_SAMPLES,evaluation_samples=EVAL_SAMPLES,
        arms=ARMS,scope='All 1188 objects; no biological support restriction'))
    records={r['fold']:r for r in manifest['folds']};fits={}
    for fold,r in records.items():fits[fold]=fit_fold(fold,root,data,np.asarray(r['fit']),np.asarray(r['test']))
    if fit_only:return fits
    prior=read_npz(RADIAL/'AMP_EMP_LOCAL.npz');actual=read_npz(DUAL/'CORE.npz')['actual']
    lookup={v:i for i,v in enumerate(data['ids'])}
    for number,cell in enumerate(old['cells']):
        fold,half=cell['fold'],cell['half'];directory=root/f'cell_{fold}_{half}';directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():continue
        q=np.array([lookup[v] for v in cell['query_ids']]);fit=np.asarray(records[fold]['fit'])
        scores=outer_scores(fold,data,metadata,fit,q)
        etas={'CORE':np.zeros((len(q),2))}
        for arm in ARMS[1:]:
            eta=scale_function(fits[fold]['models'][arm]['parameters'],scores[arm])
            etas[arm]=np.column_stack((eta,eta))
        stats=read_json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        ref=read_npz(RADIAL/f'cell_{fold}_{half}_radial.npz')
        np.testing.assert_array_equal(ref['query_ids'],data['ids'][q])
        write_json(root/'status.json',dict(state='RUNNING',stage='independent_outer_evaluation',cell=number+1,cells=10))
        print(f'cell {number+1}/10 independent {EVAL_SAMPLES} draw evaluation',flush=True)
        out=evaluate_cell(prior['mean_u'][q],prior['scatter_u'][q],prior['actual_u'][q],stats,actual[q],
            fit_radial(ref['amplitude_radii']),ref['local_weights'],etas,SEED+9000000+100000*number)
        for arm,z in out.items():
            chosen=select_frozen_cohort_plan(data['ids'][q],z['predicted'],z['p_null'],cell['budget'])
            z['selected']=np.asarray(chosen.selected_mask,int);z['fold']=np.full(len(q),fold,int)
            np.savez_compressed(directory/(arm+'.npz'),ids=data['ids'][q],**z)
        write_json(directory/'complete.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-start))
    result=summarize(root,data,metadata,old['cells'],time.monotonic()-start)
    write_json(root/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-start))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fit-only',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output,args.fit_only)


if __name__=='__main__':main()
