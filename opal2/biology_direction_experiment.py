"""Honest reference-error direction and shrunken mean correction diagnostics."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz
from .biology_kernel_evaluation import write_json
from .biology_random_reference_experiment import amplitude_bins, matched_random_weights
from .gram_geometry import profiles_to_gram, gram_to_coordinates
from .lincs_biology_experiment import load_data
from .module_switch_experiment import biology_similarity, context_mask
from .optional_radial_modules import supported_retrieval
from .reference_information_diagnostic import bootstrap_difference

PROJECT = Path(__file__).resolve().parents[1]
SEED = 2026091703
REPLICATES = 20
ALPHAS = (0., .1, .25, .5, 1.)


def safe_cosine(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite vector rows required')
    norm = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    nonzero = norm > 0
    out = np.divide(np.einsum('ni,ni->n', a, b), norm, out=np.zeros(len(a)), where=nonzero)
    return out, ~nonzero


def direction_readouts(residual, borrowed, covariance, alpha):
    """Whiten both vectors in each query's SAME frozen frame."""
    error, correction, cov = map(lambda x:np.asarray(x, float), (residual, borrowed, covariance))
    if error.ndim != 2 or correction.shape != error.shape or cov.shape != (len(error),error.shape[1],error.shape[1]):
        raise ValueError('Aligned residual, correction and covariance required')
    if not all(np.isfinite(x).all() for x in (error,correction,cov)) or not np.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('Finite inputs and bounded alpha required')
    factor = np.linalg.cholesky(cov)
    e_white = np.linalg.solve(factor,error[...,None])[...,0]
    c_white = np.linalg.solve(factor,correction[...,None])[...,0]
    cos, zero = safe_cosine(error,correction)
    white_cos, white_zero = safe_cosine(e_white,c_white)
    return dict(cosine=cos,cosine_zero=zero,whitened_cosine=white_cos,whitened_cosine_zero=white_zero,
        mse_core=np.square(error).mean(1),mse_full=np.square(error-correction).mean(1),
        mse_shrunk=np.square(error-alpha*correction).mean(1),
        whitened_mse_core=np.square(e_white).mean(1),
        whitened_mse_full=np.square(e_white-c_white).mean(1),
        whitened_mse_shrunk=np.square(e_white-alpha*c_white).mean(1),
        correction_norm=np.linalg.norm(correction,axis=1),alpha=np.full(len(error),alpha),
        borrowed_residual=correction.copy())


def choose_alpha(residual, borrowed, support, groups):
    error,pred=np.asarray(residual,float),np.asarray(borrowed,float)
    support,groups=np.asarray(support,bool),np.asarray(groups,str)
    if error.ndim!=2 or pred.shape!=error.shape or support.shape!=(len(error),) or groups.shape!=support.shape:
        raise ValueError('Calibration arrays not aligned')
    if not np.isfinite(error).all() or not np.isfinite(pred).all():
        raise ValueError('Finite calibration arrays required')
    unique=np.unique(groups[support]); records=[]
    base=np.square(error).mean(1)
    for a in ALPHAS:
        delta=np.square(error-a*pred).mean(1)-base
        grouped=np.array([delta[(groups==g)&support].mean() for g in unique])
        mean=float(grouped.mean()) if len(grouped) else 0.
        se=float(grouped.std(ddof=1)/np.sqrt(len(grouped))) if len(grouped)>1 else 0.
        eligible=a==0 or (len(unique)>=3 and mean < -se-1e-12)
        records.append(dict(alpha=a,group_mean_difference=mean,group_se=se,eligible=eligible))
    accepted=[r for r in records if r['eligible']]
    selected=min(accepted,key=lambda r:(r['group_mean_difference'],r['alpha']))
    return selected['alpha'],dict(supported_objects=int(support.sum()),supported_groups=len(unique),
        candidates=records,selected_alpha=selected['alpha'],
        reason='insufficient_supported_groups' if len(unique)<3 else 'one_group_SE_admission_then_minimum_CV_loss',
        formal_guarantee=False)


def calibration_predictions(data,metadata,cal,fit_amp,residual,*,cell_number):
    """Crossfit donor means in CAL; no query argument or query target exists."""
    groups=data['groups'][cal]
    prediction={r:np.zeros((REPLICATES+1,len(cal),9)) for r in ('TARGET','MOA')}
    support={r:np.zeros(len(cal),bool) for r in prediction}
    records=[]
    amp=np.log(np.linalg.norm(data['Y'][:,0],axis=1))
    splitter=GroupKFold(min(3,len(np.unique(groups))))
    for inner,(tr,va) in enumerate(splitter.split(cal,groups=groups)):
        donors,pseudoquery=cal[tr],cal[va]
        bins,edges=amplitude_bins(fit_amp,amp[donors])
        sims=biology_similarity(data,metadata,pseudoquery,donors)
        legal=context_mask(metadata,pseudoquery,donors)&(data['groups'][pseudoquery,None]!=data['groups'][None,donors])
        records.append(dict(inner=inner,query_ids=data['ids'][pseudoquery].tolist(),
            donor_ids=data['ids'][donors].tolist(),amplitude_edges=edges))
        for j,(relation,s) in enumerate(zip(('TARGET','MOA'),sims)):
            ch=supported_retrieval(s);support[relation][va]=ch['supported']
            prediction[relation][0,va]=ch['weights']@residual[tr]
            allowed=legal&data[relation.lower()+'_mask'][donors][None].astype(bool)
            for rep in range(REPLICATES):
                w,_=matched_random_weights(ch['weights'],allowed,bins,amp[donors],
                    seed=SEED+100000*rep+1000*cell_number+10*inner+j)
                prediction[relation][rep+1,va]=w@residual[tr]
    result={}
    for relation,pred in prediction.items():
        selections=[choose_alpha(residual,p,support[relation],groups) for p in pred]
        result[relation]=dict(prediction=pred,support=support[relation],
            alpha=np.array([a for a,_ in selections]),selection=[s for _,s in selections])
    return result,records


def intervals(left,right,mask,groups,layout):
    mask=np.asarray(mask,bool)
    return {name:bootstrap_difference(np.asarray(left)[mask],np.asarray(right)[mask],labels[mask])
            for name,labels in (('chemistry',groups),('layout',layout))}


def summarize(stores,groups,layout,support):
    result={};all_rows=np.ones(len(groups),bool)
    for relation,out in stores.items():
        mask=support[relation];stats={}
        for key in ('cosine','whitened_cosine','mse_full','mse_shrunk','whitened_mse_full','whitened_mse_shrunk'):
            values=out[key];real,random=values[0],values[1:];expected=random.mean(0)
            comparison=intervals(real,expected,mask,groups,layout)
            entry=dict(real_supported_mean=float(real[mask].mean()),expected_random_supported_mean=float(expected[mask].mean()),
                randomization_means=random[:,mask].mean(1),real_minus_expected_random=comparison)
            if 'mse' in key:
                base=out['whitened_mse_core' if key.startswith('whitened') else 'mse_core'][0]
                entry.update(core_supported_mean=float(base[mask].mean()),
                    real_minus_core_supported=intervals(real,base,mask,groups,layout),
                    random_minus_core_supported=intervals(expected,base,mask,groups,layout),
                    real_minus_core_full=intervals(real,base,all_rows,groups,layout))
            else:
                entry['real_minus_zero']=intervals(real,np.zeros(len(real)),mask,groups,layout)
            stats[key]=entry
        result[relation]=dict(supported_objects=int(mask.sum()),
            zero_cosine_objects=int(out['cosine_zero'][0,mask].sum()),
            no_correction_real_supported=int((out['alpha'][0,mask]==0).sum()),
            real_mean_selected_alpha=float(out['alpha'][0,mask].mean()),
            random_mean_selected_alpha=float(out['alpha'][1:,mask].mean()),metrics=stats)
    return result


def run(source,random_source,output):
    start=time.monotonic();source,random_source,root=map(lambda p:Path(p).resolve(),(source,random_source,output))
    if not (root/'PROTOCOL.md').exists() or (root/'summary.json').exists() or (root/'status.json').exists():
        raise ValueError('Use a fresh output directory containing the prespecified PROTOCOL.md')
    original=read_json(source/'summary.json');modules=Path(original['source'])
    radial_source=Path(read_json(modules/'summary.json')['source']);radial=read_json(radial_source/'summary.json')
    manifest=read_json(Path(radial['reference_run'])/'run_manifest.json')
    random_summary=read_json(random_source/'summary.json')
    if random_summary['state']!='COMPLETE' or random_summary['replicates']!=REPLICATES:
        raise ValueError('Completed 20-randomization control is required')
    data,metadata=load_data(radial['data_directory']);ids,groups=data['ids'],data['groups'];n=len(ids)
    if n!=1188 or ids.tolist()!=manifest['ids']:raise ValueError('Opened population changed')
    shutil.copy2(Path(__file__),root/Path(__file__).name)
    write_json(root/'run_spec.json',dict(source=str(source),random_source=str(random_source),seed=SEED,
        alpha_grid=ALPHAS,replicates=REPLICATES,started_unix=time.time()))
    write_json(root/'status.json',dict(state='RUNNING',phase='preparing'))
    prior=read_npz(radial_source/'AMP_EMP_LOCAL.npz');core=read_npz(source/'CORE.npz')
    np.testing.assert_array_equal(prior['ids'],ids);np.testing.assert_array_equal(core['ids'],ids)
    raw=gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    amp=np.log(np.linalg.norm(data['Y'][:,0],axis=1));lookup={v:i for i,v in enumerate(ids)}
    folds={f['fold']:f for f in manifest['folds']};layout=np.array([u['layout_block'] for u in metadata['units']])
    stores={r:{} for r in ('TARGET','MOA')};support={r:np.zeros(n,bool) for r in stores}
    seen=np.zeros(n,int);cells=[]
    for number,cell in enumerate(radial['cells']):
        fold,half=cell['fold'],cell['half'];folder=root/f'cell_{fold}_{half}';folder.mkdir()
        q=np.array([lookup[v] for v in cell['query_ids']]);cal=np.array([lookup[v] for v in cell['representative_ids']])
        fit=np.array(folds[fold]['fit'],int)
        if set(groups[fit])&set(groups[np.r_[q,cal]]) or set(groups[q])&set(groups[cal]):raise ValueError('Group leakage')
        stats=read_json(Path(radial['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        scale,center=np.asarray(stats['u_scale']),np.asarray(stats['u_center'])
        ref=read_npz(radial_source/f'cell_{fold}_{half}_radial.npz')
        np.testing.assert_array_equal(ref['cal_ids'],ids[cal]);np.testing.assert_array_equal(ref['query_ids'],ids[q])
        np.testing.assert_allclose(ref['cal_residual'],prior['actual_u'][cal]-prior['mean_u'][cal],rtol=1e-11,atol=1e-11)
        for rows in (q,cal):np.testing.assert_allclose(prior['actual_u'][rows]*scale+center,raw[rows],rtol=1e-11,atol=1e-11)
        cal_error=ref['cal_residual']*scale
        error=(prior['actual_u'][q]-prior['mean_u'][q])*scale
        covariance=core['covariance_u'][q]*scale[None,:,None]*scale[None,None,:]
        calibration,inner_records=calibration_predictions(data,metadata,cal,amp[fit],cal_error,cell_number=number)
        plans=read_npz(source/f'cell_{fold}_{half}'/'reference_plans.npz')
        sims=biology_similarity(data,metadata,q,cal)
        cell_report={}
        for relation,s in zip(('TARGET','MOA'),sims):
            channel=supported_retrieval(s);eligible=channel['supported'];support[relation][q]=eligible
            np.testing.assert_array_equal(channel['weights'][eligible],plans['endpoint_'+relation][eligible])
            weights=[channel['weights']]
            for rep in range(REPLICATES):
                mapped=read_npz(random_source/f'replicate_{rep:02d}'/f'cell_{fold}_{half}'/'reference_maps.npz')
                np.testing.assert_array_equal(mapped['query_ids'],ids[q]);np.testing.assert_array_equal(mapped['donor_ids'],ids[cal])
                w=mapped[relation+'_weights'].copy();w[~eligible]=0.
                np.testing.assert_array_equal(np.sort(w,axis=1),np.sort(channel['weights'],axis=1))
                weights.append(w)
            outputs=[]
            for index,w in enumerate(weights):
                borrowed=w@cal_error
                np.testing.assert_array_equal(borrowed[~eligible],np.zeros((np.sum(~eligible),9)))
                out=direction_readouts(error,borrowed,covariance,calibration[relation]['alpha'][index])
                np.testing.assert_array_equal(out['mse_shrunk'][~eligible],out['mse_core'][~eligible])
                outputs.append(out)
            for key in outputs[0]:
                values=np.stack([o[key] for o in outputs])
                if key not in stores[relation]:stores[relation][key]=np.empty((REPLICATES+1,n,*values.shape[2:]),dtype=values.dtype)
                stores[relation][key][:,q]=values
            np.savez_compressed(folder/(relation+'_calibration.npz'),ids=ids[cal],raw_residual=cal_error,
                predicted_residual=calibration[relation]['prediction'],support=calibration[relation]['support'],
                selected_alpha=calibration[relation]['alpha'])
            cell_report[relation]=dict(selections=calibration[relation]['selection'],
                query_support=int(eligible.sum()),reference_count=int(len(cal)),
                positive_donor_count=channel['positive_count'],ess=channel['ess'])
        np.savez_compressed(folder/'honest_errors.npz',query_ids=ids[q],reference_ids=ids[cal],
            query_native_error=error,reference_native_error=cal_error,query_native_covariance=covariance,
            model_fit_ids=ids[fit])
        record=dict(fold=fold,half=half,query_ids=ids[q].tolist(),reference_ids=ids[cal].tolist(),
            model_fit_ids=ids[fit].tolist(),calibration_splits=inner_records,relations=cell_report)
        write_json(folder/'cell.json',record);cells.append(record);seen[q]+=1
        write_json(root/'status.json',dict(state='RUNNING',cells_complete=len(cells),elapsed_seconds=time.monotonic()-start))
        print(f'cell={number+1}/10 direction complete elapsed={time.monotonic()-start:.1f}s',flush=True)
    np.testing.assert_array_equal(seen,np.ones(n,int))
    for relation,out in stores.items():np.savez_compressed(root/(relation+'.npz'),ids=ids,groups=groups,layout=layout,
        support=support[relation],**out)
    result=summarize(stores,groups,layout,support)
    summary=dict(state='COMPLETE',n=n,replicates=REPLICATES,source=str(source),random_source=str(random_source),
        metrics=result,cells=cells,elapsed_seconds=time.monotonic()-start,
        mean_model_retrained=False,query_used_for_alpha_selection=False,formal_certificate=False,
        space='native nine log-Cholesky geometry coordinates; secondary common-query covariance whitening',
        uncertainty='paired chemistry/layout bootstrap conditional on fixed reference banks and fitted strengths',
        endpoints_changed=False,policy_evaluated=False)
    write_json(root/'summary.json',summary)
    lines=['# Biological-reference residual direction: opened LINCS development','',
        'Native nine-coordinate geometry errors; 20 matched random controls; no world-model training.','',
        '| Relation | Supported | CORE MSE | Full BIO MSE | Shrunken BIO MSE | BIO cosine | Random cosine |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for relation,r in result.items():
        m=r['metrics'];lines.append(f"| {relation} | {r['supported_objects']} | {m['mse_full']['core_supported_mean']:.8f} | {m['mse_full']['real_supported_mean']:.8f} | {m['mse_shrunk']['real_supported_mean']:.8f} | {m['cosine']['real_supported_mean']:.6f} | {m['cosine']['expected_random_supported_mean']:.6f} |")
    lines+=['','Positive cosine alone does not establish useful mean correction. Strengths are selected only on grouped calibration CV.',
        'This diagnostic does not recalibrate the joint error law after correcting its mean and does not claim Gamma/NULL policy improvement.',
        'Full paired intervals, whitening sensitivity, support and calibration fallbacks are in summary.json.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'status.json',dict(state='COMPLETE',cells_complete=len(cells),elapsed_seconds=summary['elapsed_seconds']))
    print('COMPLETE',root,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default=str(PROJECT/'runs/biology_borrowing_diagnostic_20260916_v2'))
    parser.add_argument('--random-source',default=str(PROJECT/'runs/biology_random_reference_20260916_v1'))
    parser.add_argument('--output',default=str(PROJECT/'runs/biology_direction_20260917_v1'))
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        torch.set_num_threads(2);run(args.source,args.random_source,args.output)


if __name__=='__main__':main()
