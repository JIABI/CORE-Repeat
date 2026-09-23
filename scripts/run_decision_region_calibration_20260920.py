"""Run the four-archive frozen-model decision-region calibration comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from threadpoolctl import threadpool_limits
from opal2.decision_region_calibration import (
    ARMS, VARIANTS, PENALTY, TAIL_FRACTION, VERSION, evaluate_cell,
    paired_cluster_interval, probability_rows, risk_summary, validate_cell,
)

OUTPUT = PROJECT/'runs/decision_region_calibration_20260920_v1'
REPORT = PROJECT/'reports/decision_region_calibration_20260920_v1'
DATASETS = ('EU', 'JUMP', 'LINCS', 'RxRx3')
EXPECTED_N = dict(EU=904, JUMP=639, LINCS=1188, RxRx3=10410)
R2_RUN = dict(EU='r2_core_comparison_20260917_v1',JUMP='jump_r2_completion_20260918_v1',
              LINCS='lincs_r2_completion_20260918_v1',RxRx3='rxrx3_r2_completion_20260918_v1')
SEED = 20260920
SAMPLES = 100000


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def save_npz(path, arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(temporary,**arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def save_cache(folder, cell):
    arrays={k:np.asarray(v) for k,v in cell.items() if k in (
        'cal_ids','cal_groups','cal_actual','cal_inner_fold',
        'query_ids','query_groups','query_layout','query_actual')}
    for arm,values in cell['arms'].items():
        for k,v in values.items():arrays[arm+'__'+k]=np.asarray(v)
    metadata={k:v for k,v in cell.items() if k not in arrays and k!='arms'}
    metadata['arms']=list(cell['arms'])
    save_npz(folder/'honest_calibration_predictions.npz',arrays)
    write_json(folder/'cache_metadata.json',metadata)


def load_cache(folder):
    meta=json.loads((folder/'cache_metadata.json').read_text())
    arrays=load_npz(folder/'honest_calibration_predictions.npz')
    arms=meta.pop('arms')
    cell={k:v for k,v in arrays.items() if '__' not in k}
    cell.update(meta)
    cell['arms']={a:{k.split('__',1)[1]:v for k,v in arrays.items()
                     if k.startswith(a+'__')} for a in arms}
    return cell


def adapter(dataset):
    if dataset=='LINCS':
        from opal2 import decision_region_cache_lincs as mod
    else:
        from opal2 import decision_region_cache_standard as mod
    return mod


def run_cell(dataset, cell_name, output):
    folder=Path(output)/dataset/str(cell_name)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'complete.json').exists():
            saved=json.loads((folder/'complete.json').read_text())
            if saved.get('version') != VERSION:raise ValueError('Cell version mismatch')
            return saved
        started=time.perf_counter()
        write_json(folder/'status.json',dict(state='BUILDING_HONEST_CAL',updated=now(),pid=os.getpid()))
        with threadpool_limits(limits=1):
            import torch
            torch.set_num_threads(1)
            if (folder/'cache_metadata.json').exists():
                cell=load_cache(folder)
                cache_reused=True
            else:
                cell=adapter(dataset).build_cell(dataset,str(cell_name),samples=SAMPLES,seed=SEED)
                validate_cell(cell)
                save_cache(folder,cell)
                cache_reused=False
            cache_seconds=time.perf_counter()-started
            write_json(folder/'status.json',dict(state='FITTING_OFFSETS',updated=now(),pid=os.getpid()))
            fit_started=time.perf_counter()
            arrays,metadata=evaluate_cell(cell)
            original_r2_audit(dataset,arrays)
            fit_seconds=time.perf_counter()-fit_started
        save_npz(folder/'query_predictions.npz',arrays)
        write_json(folder/'offset_models.json',metadata)
        result=dict(version=VERSION,state='COMPLETE',dataset=dataset,cell=str(cell_name),
                    n=metadata['n'],k=metadata['k'],n_cal=metadata['n_cal'],
                    cache_reused=cache_reused,cache_seconds=cache_seconds,
                    offset_fit_evaluation_seconds=fit_seconds,
                    elapsed_seconds=time.perf_counter()-started,updated=now(),
                    additional_backbone_fits=0,additional_direct_estimator_fits=0,
                    query_predictions_reused=True,protected_measurements_opened=False)
        write_json(folder/'complete.json',result)
        write_json(folder/'status.json',result)
        return result


def original_r2_audit(dataset, arrays):
    """Catch changed historical budgets/ties as well as changed probabilities."""
    for arm in ARMS:
        if arm=='CORE':filename='CORE_ORIGINAL'
        else:
            family,interface=arm.split('_',1)
            filename='DIRECT_ACCESS_MATCHED_'+family+'_'+('CLASSIFIER_CAL' if interface=='CAL' else interface)
        original=load_npz(PROJECT/'runs'/R2_RUN[dataset]/(filename+'.npz'))
        lookup={str(v):i for i,v in enumerate(original['ids'])}
        rows=np.asarray([lookup[str(v)] for v in arrays['ids']],int)
        for new_key,old_key in [(arm+'__mean','predicted'),(arm+'__ORIGINAL__p','p_null'),
                                (arm+'__ORIGINAL__selected','selected_lambda_0.2'),
                                ('actual','actual')]:
            np.testing.assert_array_equal(arrays[new_key],original[old_key][rows],
                                          err_msg=dataset+' '+arm+' original R2 '+old_key+' changed')


def write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(clean(v)) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})


def contrast_rows(dataset, arrays, name, difference, denominator=None):
    out=[]
    for block,key in [('chemical_group','groups'),('layout','layout')]:
        ci=paired_cluster_interval(difference,arrays[key],denominator=denominator,
                                   seed=SEED,replicates=2000)
        out.append(dict(dataset=dataset,contrast=name,block=block,**ci))
    return out


def dataset_analysis(dataset, arrays):
    actual=arrays['actual'];label=(actual<=0).astype(float);n=len(actual)
    all_rows=np.ones(n,bool)
    risk,policy,paired,coefficients=[],[],[],[]
    for arm in ARMS:
        original=arrays[arm+'__original_selected']
        rank=arrays[arm+'__rank']
        for variant in VARIANTS:
            prefix=arm+'__'+variant
            p=arrays[prefix+'__p'];selected=arrays[prefix+'__selected']
            regions=dict(all=all_rows,fixed_original_selected=original,
                         fixed_CORE_selected=arrays['CORE_selected'],reranked_selected=selected,
                         fixed_lambda0_selected=arrays[arm+'__lambda0_selected'])
            for lo,hi in [(0.,.05),(.05,.10),(.10,.15),(.15,.25),(.25,1.)]:
                regions['frozen_score_band_%g_%g'%(lo,hi)]=(rank>=lo)&(rank<hi)
            for region,mask in regions.items():
                summary=risk_summary(p,actual,mask)
                risk.append(dict(dataset=dataset,arm=arm,variant=variant,region=region,**summary))
            for mode,mask in [('fixed_original',original),('reranked',selected),
                              ('lambda0',arrays[arm+'__lambda0_selected'])]:
                count=int(mask.sum())
                policy.append(dict(dataset=dataset,arm=arm,variant=variant,mode=mode,n=n,k=count,
                    actual_NULL=int(label[mask].sum()),predicted_NULL=float(p[mask].sum()),
                    total_Gamma=float(actual[mask].sum()),mean_selected_Gamma=float(actual[mask].mean()),
                    value_per_candidate=float(actual[mask].sum()/n),
                    selected_NULL_per_candidate=float(label[mask].sum()/n),
                    FDP=float(label[mask].mean()),NULL_conditioned_FPR=float(label[mask].sum()/label.sum()),
                    original_intersection=int((mask&original).sum()),
                    original_replaced=int((mask&~original).sum()),additional_action_wells=2*count))
            # Signed count calibration, expressed per selected object, on the unchanged set.
            for region,mask in [('all',all_rows),('fixed_original_selected',original),
                                 ('fixed_CORE_selected',arrays['CORE_selected'])]:
                gap=(label-p)*mask
                paired += contrast_rows(dataset,arrays,prefix+':'+region+':observed_minus_predicted_NULL',gap,mask)
        for new,old in [('GLOBAL','ORIGINAL'),('REGION','ORIGINAL'),('REGION','GLOBAL')]:
            pn=arrays[arm+'__'+new+'__p'];po=arrays[arm+'__'+old+'__p']
            rn=probability_rows(pn,actual);ro=probability_rows(po,actual)
            title=arm+':'+new+' minus '+old
            for region,mask in [('all',all_rows),('fixed_original_selected',original),
                                ('fixed_CORE_selected',arrays['CORE_selected'])]:
                for score in ('brier','logloss'):
                    paired += contrast_rows(dataset,arrays,title+':'+region+':'+score,
                                            (rn[score]-ro[score])*mask,mask)
            mn=arrays[arm+'__'+new+'__selected'].astype(float)
            mo=arrays[arm+'__'+old+'__selected'].astype(float)
            paired += contrast_rows(dataset,arrays,title+':reranked:value_per_candidate',(mn-mo)*actual)
            paired += contrast_rows(dataset,arrays,title+':reranked:NULL_per_candidate',(mn-mo)*label)
    # Strong models receive the same new calibration opportunities. The selection
    # comparison uses their own lists; probability comparisons share CORE's list.
    for arm in ARMS[1:]:
        for variant in VARIANTS:
            title=arm+' minus CORE:'+variant
            pd=arrays[arm+'__'+variant+'__p'];pc=arrays['CORE__'+variant+'__p']
            rd=probability_rows(pd,actual);rc=probability_rows(pc,actual)
            for region,mask in [('all',all_rows),('fixed_CORE_selected',arrays['CORE_selected'])]:
                for score in ('brier','logloss'):
                    paired += contrast_rows(dataset,arrays,title+':'+region+':'+score,
                                            (rd[score]-rc[score])*mask,mask)
            md=arrays[arm+'__'+variant+'__selected'].astype(float)
            mc=arrays['CORE__'+variant+'__selected'].astype(float)
            paired += contrast_rows(dataset,arrays,title+':reranked:value_per_candidate',(md-mc)*actual)
            paired += contrast_rows(dataset,arrays,title+':reranked:NULL_per_candidate',(md-mc)*label)
    return risk,policy,paired


def aggregate(output, datasets, jobs):
    risk=[];policy=[];paired=[];models=[];complete=[];missing=[];mc=[]
    for dataset in datasets:
        folders=[Path(output)/d/c for d,c in jobs if d==dataset]
        if not all((f/'complete.json').exists() for f in folders):
            missing.append(dataset);continue
        blocks=[load_npz(f/'query_predictions.npz') for f in folders]
        keys=set(blocks[0])
        if any(set(b)!=keys for b in blocks):raise ValueError('Inconsistent result array keys')
        arrays={k:np.concatenate([b[k] for b in blocks]) for k in sorted(keys)}
        if len(set(arrays['ids']))!=len(arrays['ids']):raise ValueError('Duplicate QUERY condition')
        if len(arrays['ids'])!=EXPECTED_N[dataset]:raise ValueError('Incomplete dataset cannot be called complete')
        original_r2_audit(dataset,arrays)
        rr,pp,cc=dataset_analysis(dataset,arrays)
        risk.extend(rr);policy.extend(pp);paired.extend(cc)
        save_npz(Path(output)/dataset/'all_query_predictions.npz',arrays)
        for variant in VARIANTS:
            prefix='CORE__'+variant
            primary=arrays[prefix+'__selected']
            for index in (-1,0,1):
                pk=prefix+('__p' if index<0 else '__mc%d_p'%index)
                mk=prefix+('__selected' if index<0 else '__mc%d_selected'%index)
                if pk not in arrays:continue
                mask=arrays[mk];p=arrays[pk]
                mc.append(dict(dataset=dataset,variant=variant,seed='primary' if index<0 else 'offset_'+str((index+1)*100000),
                    selected=int(mask.sum()),actual_NULL=int((arrays['actual'][mask]<=0).sum()),
                    predicted_NULL=float(p[mask].sum()),mean_selected_Gamma=float(arrays['actual'][mask].mean()),
                    replaced_vs_primary=int((mask&~primary).sum()),offset_refitted=False))
        for f in folders:
            models.append(json.loads((f/'offset_models.json').read_text()))
            complete.append(json.loads((f/'complete.json').read_text()))
    write_csv(REPORT/'risk_regions.csv',risk)
    write_csv(REPORT/'allocation.csv',policy)
    write_csv(REPORT/'paired_intervals.csv',paired)
    write_csv(REPORT/'MC_seed_stability.csv',mc)
    write_json(REPORT/'offset_models.json',models)
    write_json(REPORT/'summary.json',dict(complete=not missing,datasets=list(datasets),pending=missing,
        cells=complete,risk=risk,policy=policy,paired=paired,monte_carlo=mc,updated=now(),
        method='fixed penalized risk offsets; not a Jev/RLCD reproduction',
        scope='development comparisons, fixed fitted predictions and query lists',
        gamma_distribution_unchanged=True,formal_certificate=False,protected_measurements_opened=False,
        model_selection_performed=False))
    lines=['# Decision-region calibration: four-archive development comparison','',
        'Updated: '+now(),'',
        'CORE and direct estimators are frozen. GLOBAL adds one logit offset; REGION adds a second coefficient for the top-quarter frozen-score feature. Calibration inputs are inner-group-held-out; all scores below are original outer DEV_EVAL outcomes. No BIO/representation experiment or R4 opening is part of this run.','',
        '## Fixed original selections: risk accuracy','',
        '| Dataset | Base interface | Calibration | N selected | Predicted NULL | Actual NULL | Brier |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in risk:
        if r['region']=='fixed_original_selected':
            lines.append('| %s | %s | %s | %d | %.3f | %d | %.6f |'%(r['dataset'],r['arm'],r['variant'],r['n'],r['predicted_NULL'],r['actual_NULL'],r['brier']))
    lines += ['', '## Reranking with the corrected probability','',
        '| Dataset | Base interface | Calibration | NULL / selected | Mean actual Gamma | Replaced objects |',
        '|---|---|---|---:|---:|---:|']
    for r in policy:
        if r['mode']=='reranked':
            lines.append('| %s | %s | %s | %d / %d | %.8f | %d |'%(r['dataset'],r['arm'],r['variant'],r['actual_NULL'],r['k'],r['mean_selected_Gamma'],r['original_replaced']))
    lines += ['', '## Interpretation and retained evidence','',
        'Paired intervals are in paired_intervals.csv. Fixed-original and fixed-CORE selected probability losses compare the same objects; reranked own-selected losses compare different populations. Signed NULL gaps are observed minus predicted. The shared chemical-group and layout resampling is a conditional development analysis, not an iid binomial or finite-sample certification. Few-layout intervals remain descriptive.',
        '', 'The original Gamma mean, samples and CRPS are not changed by a binary-risk offset. COHERENT in a base name identifies the source of its original probability; its recalibrated probability is a separate decision risk and no longer claimed to be induced by that unchanged Gamma law.',
        '', 'No method is automatically selected from these evaluation results. All original R2 outputs are retained. The penalty and rank feature were fixed before the new comparison; no new backbone or direct estimator was trained. Reference acquisition and action costs remain the original R2 costs; only new CAL prediction and offset fitting compute is added.']
    if missing:lines += ['', 'Pending datasets: '+', '.join(missing)]
    (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return dict(complete=not missing,completed_datasets=[d for d in datasets if d not in missing],
                pending=missing,completed_cells=len(complete),report=str(REPORT/'REPORT.md'))


def run(args):
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[(d,str(c)) for d in args.datasets for c in adapter(d).list_cells(d)]
        configuration=dict(version=VERSION,datasets=args.datasets,jobs=jobs,samples=SAMPLES,
            seed=SEED,penalty=PENALTY,tail_fraction=TAIL_FRACTION,arms=ARMS,variants=VARIANTS,
            no_new_backbone_training=True,no_new_direct_estimator_training=True,
            no_BIO_REP_runs=True,protected_measurements_opened=False,
            CAL_rank_scope='within each held-out inner fold',query_rank_scope='original deployment cell')
        manifest=OUTPUT/'run_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=clean(configuration):
            raise ValueError('Existing run configuration differs; do not overwrite')
        write_json(manifest,configuration)
        started=time.perf_counter();finished=[]
        try:
            if not args.aggregate_only:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures={pool.submit(run_cell,d,c,str(OUTPUT)):(d,c) for d,c in jobs}
                    write_json(OUTPUT/'status.json',dict(state='RUNNING',pid=os.getpid(),
                        total_cells=len(jobs),completed_cells=0,updated=now()))
                    for future in as_completed(futures):
                        result=future.result();finished.append(result)
                        status=dict(state='RUNNING',pid=os.getpid(),total_cells=len(jobs),
                            completed_cells=len(finished),last_completed=result,updated=now(),
                            elapsed_seconds=time.perf_counter()-started)
                        write_json(OUTPUT/'status.json',status)
                        print(json.dumps(clean(status)),flush=True)
            write_json(OUTPUT/'status.json',dict(state='AGGREGATING',pid=os.getpid(),updated=now()))
            result=aggregate(OUTPUT,args.datasets,jobs)
            result.update(state='COMPLETE' if result['complete'] else 'PARTIAL',updated=now(),
                          elapsed_this_invocation_seconds=time.perf_counter()-started)
            write_json(OUTPUT/'status.json',result)
            print(json.dumps(result),flush=True)
        except BaseException as error:
            write_json(OUTPUT/'status.json',dict(state='FAILED',updated=now(),pid=os.getpid(),
                error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started))
            raise


def launch(args):
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    launch_path=OUTPUT/'launch.json'
    if launch_path.exists():
        old=json.loads(launch_path.read_text())
        try:os.kill(int(old['pid']),0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Existing background launch is still alive')
    command=['/usr/bin/nice','-n','10','/usr/bin/caffeinate','-i',str(PROJECT/'.venv/bin/python'),
        '-u',str(Path(__file__).resolve()),'--workers',str(args.workers),'--datasets',*args.datasets]
    env=dict(os.environ)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
    log=OUTPUT/'background.log'
    with log.open('ab') as stream:
        child=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    result=dict(state='LAUNCHED',pid=child.pid,started=now(),command=command,log=str(log),
        compute='CPU, two workers at most, one numerical thread each, nice 10',
        assistant_API_calls=False,scheduled_assistant_monitor=False,
        sleep='idle sleep inhibited during run; display can sleep; lid-close sleep is not prevented',
        next_BIO_REP_phase='not launched; wait for this comparison and analysis')
    write_json(launch_path,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=2:parser.error('Use one or two CPU workers')
    launch(args) if args.launch else run(args)
