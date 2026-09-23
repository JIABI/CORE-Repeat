"""Outcome-labelled, bounded variance-adjustment headroom on opened DEV.

Hindsight arms optimize the observed outcome; they are not deployment models.
The finite search and independently seeded Monte Carlo evaluation are separate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz, paired_intervals
from .biology_kernel_evaluation import write_json
from .dual_branch_features import apply_increment
from .conditional_residual_information import error_targets
from .empirical_radial import (fit_radial, reference_weights, radial_nll,
                              radial_ppf, draw_radial)
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .joint_contrast_scale import contrast_projector, projected_energy
from .lincs_biology_experiment import load_data
from .objective_analysis import fair_crps

PROJECT = Path(__file__).resolve().parents[1]
RADIAL = PROJECT/'runs/lincs_empirical_radial_20260916_v1'
DUAL = PROJECT/'runs/dual_branch_biology_20260917_v2'
DESCRIPTORS = PROJECT/'runs/conditional_residual_information_20260916_v1'
PLAN = PROJECT/'protocols/historical/VARIANCE_HEADROOM_PLAN_20260917.md'
ARMS = ('CORE', 'GLOBAL_CAL', 'AMPLITUDE_RANK_CAL', 'LEARNED_RANK_CAL',
        'TRUE_ENERGY_RANK_CAL', 'H1_CRPS', 'H2_CRPS', 'H1_NLL', 'H2_NLL')
HINDSIGHT = ('TRUE_ENERGY_RANK_CAL', 'H1_CRPS', 'H2_CRPS', 'H1_NLL', 'H2_NLL')
SEED = 17091701
SEARCH_SAMPLES = 4096
EVAL_SAMPLES = 100000
GRID_SIZE = 9
LEVELS = np.array([.5, .8, .9, .95, .99])


def gamma_fast(raw):
    """Exact same four-vector Gamma; omit unused observable calculations."""
    u=np.asarray(raw,float)
    if u.shape[-1]!=9 or not np.isfinite(u).all():raise ValueError('Invalid geometry')
    d1,d2,d3=np.exp(u[...,3]),np.exp(u[...,5]),np.exp(u[...,8])
    if not all(np.isfinite(d).all() and np.all(d>0) and np.all(d*d>0)
               for d in (d1,d2,d3)):
        raise ValueError('Invalid Cholesky diagonal, no sample clipping')
    a0=1+u[...,0]+u[...,1];a1=d1+u[...,4];a2=d2
    v0,v1,v2=u[...,2],u[...,6],u[...,7]
    nv=np.sqrt(v0*v0+v1*v1+v2*v2+d3*d3)
    na=np.sqrt(a0*a0+a1*a1+a2*a2)
    gamma=.5*((a0*v0+a1*v1+a2*v2)/(na*nv)-v0/nv)-.02
    if not np.isfinite(gamma).all():raise ValueError('Invalid Gamma')
    return gamma


def scalar_grid():
    return np.linspace(-np.log(4.),np.log(4.),GRID_SIZE)


def global_calibration_choice(scores,groups,grid):
    """One-standard-error choice; zero is always an admissible comparator."""
    scores=np.asarray(scores,float);groups=np.asarray(groups);grid=np.asarray(grid,float)
    if scores.shape!=(len(groups),len(grid)) or not np.isfinite(scores).all():
        raise ValueError('Invalid calibration scores')
    units=np.unique(groups);rows=np.stack([scores[groups==g].mean(0) for g in units])
    if len(units)<3:raise ValueError('Too few calibration groups')
    zero=int(np.argmin(abs(grid)))
    if grid[zero]!=0:raise ValueError('Zero must be in the family')
    best=int(np.argmin(rows.mean(0)));change=rows[:,best]-rows[:,zero]
    if best==zero or change.mean()>=-change.std(ddof=1)/np.sqrt(len(units)):
        return 0.,dict(index=zero,reason='no_one_SE_gain',means=rows.mean(0).tolist())
    admitted=[]
    for j in range(len(grid)):
        delta=rows[:,j]-rows[:,zero];gap=rows[:,j]-rows[:,best]
        if (delta.mean() < -delta.std(ddof=1)/np.sqrt(len(units))
                and gap.mean()<=gap.std(ddof=1)/np.sqrt(len(units))):admitted.append(j)
    chosen=min(admitted,key=lambda j:(abs(grid[j]),j)) if admitted else zero
    return float(grid[chosen]),dict(index=chosen,reason='group_paired_one_SE',means=rows.mean(0).tolist())


def calibrate_rank_map(cal_score,query_score,optimal_scalar):
    """A fixed monotone calibration model; query outcomes are not arguments.

    True-energy inputs must be explicitly labelled hindsight by the caller.
    Ties use mid-ranks; query scores use the calibration empirical CDF.
    """
    x,q,y=map(lambda v:np.asarray(v,float),(cal_score,query_score,optimal_scalar))
    if x.ndim!=1 or y.shape!=x.shape or q.ndim!=1 or len(x)<3:
        raise ValueError('Invalid rank-map records')
    if not all(np.isfinite(v).all() for v in (x,q,y)):raise ValueError('Nonfinite rank input')
    ranks=(rankdata(x,method='average')-.5)/len(x)
    sorted_x=np.sort(x)
    qr=(np.searchsorted(sorted_x,q,'left')+np.searchsorted(sorted_x,q,'right'))/(2*len(x))
    fitted=IsotonicRegression(increasing=True,y_min=-np.log(4.),y_max=np.log(4.),
                              out_of_bounds='clip').fit(ranks,y)
    return fitted.predict(qr),dict(calibration_n=len(x),unique_scores=int(len(np.unique(x))),
        x_thresholds=fitted.X_thresholds_.tolist(),y_thresholds=fitted.y_thresholds_.tolist(),
        target='calibration-only CRPS-optimal scalar, not query Gamma',monotone_increasing=True)


def calibration_search(prior,ref,cell,cal,groups,stats,actual,seed):
    """Honest current-cell group-LOO laws, not opposite-role cached covariance."""
    from .variance_headroom_math import evaluate_candidate_crps
    grid=scalar_grid();candidates=np.column_stack((grid,grid));rows=[]
    for j in range(len(cal)):
        take=groups[cal]!=groups[cal[j]]
        if len(np.unique(groups[cal][take]))<3:raise ValueError('Insufficient LOO calibration')
        law=fit_radial(ref['amplitude_radii'][take])
        weights=reference_weights(ref['cal_log_amplitude'][take],ref['cal_log_amplitude'][j:j+1],
                                  cell['local_bandwidth'],conditional=True)['weights']
        out=evaluate_candidate_crps(prior['mean_u'][cal[j:j+1]],ref['cal_amp_scatter'][j:j+1],
            stats,actual[cal[j:j+1]],law,weights,candidates,seed+1009*j,samples=SEARCH_SAMPLES)
        rows.append(np.asarray(out).reshape(len(grid)))
    scores=np.stack(rows)
    return scores,grid[scores.argmin(1)]


def evaluate_cell(mean,scatter,target,stats,actual,law,weights,etas,seed,samples=EVAL_SAMPLES):
    """Fresh sampling for all arms; original archives are never updated."""
    n=len(mean);scale=np.asarray(stats['u_scale']);center=np.asarray(stats['u_center'])
    raw_mean=mean*scale+center
    distributions={a:apply_increment(raw_mean,scale,scatter,e) for a,e in etas.items()}
    out={}
    for arm,cov in distributions.items():
        radius=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(cov),(target-mean)[...,None])[...,0],axis=1)
        thresholds=radial_ppf(law,weights,LEVELS)
        out[arm]=dict(nll=radial_nll(target-mean,cov,law,weights),
            joint_coverage_by_level=(radius[:,None]<=thresholds).astype(float),increment=etas[arm])
        for k in ('predicted','p_null','crps','brier','gamma_mc_se','null_mc_se','score_mc_se',
                  'mean_null_mc_covariance','crps_paired_mc_se','score_paired_mc_se'):
            out[arm][k]=np.empty(n)
        out[arm]['crps_mc_blocks']=np.empty((n,20))
    for start in range(0,n,4):
        stop=min(start+4,n);ix=slice(start,stop);m=stop-start
        # Fresh seed namespace differs from search; blocks also differ across cells.
        nr=np.random.default_rng(seed+10007*start);rr=np.random.default_rng(seed+47000+10007*start)
        normal=nr.normal(size=(samples,m,9));mix=rr.random((samples,m));kernel=rr.random((samples,m))
        draws={}
        for arm,cov in distributions.items():
            if np.all(etas[arm][ix]==0) and 'CORE' in draws:
                gamma=draws['CORE']
            else:
                error=draw_radial(law,weights[ix],cov[ix],normal,mix,kernel)
                gamma=gamma_fast((mean[None,ix]+error)*scale+center)
            draws[arm]=gamma
            null=(gamma<=0);score=gamma-.2*null;z=out[arm]
            z['predicted'][ix]=gamma.mean(0);z['p_null'][ix]=null.mean(0)
            z['crps'][ix]=fair_crps(gamma,actual[ix]);z['brier'][ix]=(null.mean(0)-(actual[ix]<=0))**2
            z['gamma_mc_se'][ix]=gamma.std(0,ddof=1)/np.sqrt(samples)
            z['null_mc_se'][ix]=null.std(0,ddof=1)/np.sqrt(samples)
            z['score_mc_se'][ix]=score.std(0,ddof=1)/np.sqrt(samples)
            z['mean_null_mc_covariance'][ix]=((gamma*null).mean(0)-gamma.mean(0)*null.mean(0))/(samples-1)
            chunks=np.split(gamma,20)
            blocks=np.stack([fair_crps(c,actual[ix]) for c in chunks],axis=1)
            z['crps_mc_blocks'][ix]=blocks
            base=draws['CORE'];paired=score-(base-.2*(base<=0))
            z['score_paired_mc_se'][ix]=paired.std(0,ddof=1)/np.sqrt(samples)
            base_blocks=out['CORE']['crps_mc_blocks'][ix]
            z['crps_paired_mc_se'][ix]=(blocks-base_blocks).std(1,ddof=1)/np.sqrt(20)
        del draws
    return out


def existing_rank_scores(fold,data,metadata,fit,cal,q):
    """Raw full/amplitude energy prediction; never infer it from capped ratios."""
    record=joblib.load(DESCRIPTORS/f'fold_{fold}/conditioners.joblib')
    for model in [record['amplitude'],record['conditional']['DESCRIPTORS']]:
        if set(model.fit_ids)!=set(data['ids'][fit]):raise ValueError('Energy predictor fit scope changed')
        if set(model.fit_ids)&set(data['ids'][np.r_[cal,q]]):raise ValueError('Calibration/query leakage')
    descriptor=record['transformer'].transform(data,metadata).values
    amp=record['amplitude'].predict(descriptor)
    full=record['conditional']['DESCRIPTORS'].predict(descriptor)
    # These are the actual existing rank-six predictors behind .390/.445.
    # Their target frame uses original RIDGE OOF covariance, not today's CORE.
    return amp[:,1],full[:,1]


def summary_metrics(out,actual,mask):
    selected=out['selected'].astype(bool)
    varied=np.std(out['predicted'][mask])>0 and np.std(actual[mask])>0
    return dict(n=int(mask.sum()),crps=float(out['crps'][mask].mean()),brier=float(out['brier'][mask].mean()),
        nll=float(out['nll'][mask].mean()),coverage=out['joint_coverage_by_level'][mask].mean(0).tolist(),
        gamma_spearman=float(spearmanr(out['predicted'][mask],actual[mask]).statistic) if varied else None,
        predicted_gamma=float(out['predicted'][mask].mean()),predicted_null=float(out['p_null'][mask].mean()),
        selected_n=int(selected.sum()),selected_null=int(((actual<=0)&selected).sum()),
        selected_mean=float(actual[selected].mean()),
        paired_crps_integration_se=float(np.sqrt(np.sum(out['crps_paired_mc_se'][mask]**2))/mask.sum()))


def summarize(root,data,metadata,cells,elapsed):
    ids=data['ids'];lookup={v:i for i,v in enumerate(ids)};n=len(ids)
    support=read_npz(DUAL/'GELU.npz')['resource_support'].astype(bool)
    actual=read_npz(DUAL/'CORE.npz')['actual'];layout=np.asarray([x['layout_block'] for x in metadata['units']])
    stores={};metrics={};comparisons={}
    for arm in ARMS:
        values=[];seen=np.zeros(n,int)
        for cell in cells:
            value=read_npz(root/f'cell_{cell["fold"]}_{cell["half"]}'/(arm+'.npz'))
            q=np.asarray([lookup[v] for v in value['ids']]);values.append((q,value));np.add.at(seen,q,1)
        np.testing.assert_array_equal(seen,np.ones(n,int))
        keys=set.intersection(*[{k for k,v in item.items() if k!='ids' and v.shape[:1]==(len(q),)} for q,item in values])
        out={k:np.empty((n,*values[0][1][k].shape[1:]),dtype=values[0][1][k].dtype) for k in keys}
        for q,value in values:
            for k in keys:out[k][q]=value[k]
        out['policy_value']=out['selected']*actual;out['policy_null']=out['selected']*(actual<=0)
        stores[arm]=out
        metrics[arm]=dict(supported=summary_metrics(out,actual,support),full=summary_metrics(out,actual,np.ones(n,bool)),
            hindsight=arm in HINDSIGHT,active=int(np.any(out['increment']!=0,axis=1).sum()))
        np.savez_compressed(root/(arm+'.npz'),ids=ids,groups=data['groups'],layout=layout,actual=actual,support=support,**out)
    for arm in ARMS[1:]:
        comparisons[arm+'_minus_CORE']=dict(
            supported=paired_intervals(stores[arm],stores['CORE'],support,data['groups'],layout,keys=('crps','brier','nll')),
            full=paired_intervals(stores[arm],stores['CORE'],np.ones(n,bool),data['groups'],layout,keys=('crps','brier','nll')),
            policy=paired_intervals(stores[arm],stores['CORE'],np.ones(n,bool),data['groups'],layout,keys=('policy_value','policy_null')))
    for a,b in [('H2_CRPS','H1_CRPS'),('H2_CRPS','H2_NLL'),('LEARNED_RANK_CAL','AMPLITUDE_RANK_CAL'),
                ('TRUE_ENERGY_RANK_CAL','LEARNED_RANK_CAL')]:
        comparisons[a+'_minus_'+b]=paired_intervals(stores[a],stores[b],support,data['groups'],layout,keys=('crps','brier','nll'))
    result=dict(state='COMPLETE',n=n,supported_n=int(support.sum()),metrics=metrics,comparisons=comparisons,
        cells=cells,search_samples=SEARCH_SAMPLES,evaluation_samples=EVAL_SAMPLES,grid_size=GRID_SIZE,
        elapsed_seconds=elapsed,mean_changed=False,main_rule_changed=False,final_opened=False,new_neural_training=False,
        scope='Finite candidate hindsight diagnostic; not a continuous optimum or information ceiling',
        mc_se_scope='20-batch paired CRPS integration approximation, not population uncertainty')
    write_json(root/'summary.json',result)
    lines=['# Bounded variance-adjustment headroom','',result['scope'],'',
        '| Arm | Supported CRPS | Supported Brier | Selected NULL/n | Selected mean Gamma |',
        '|---|---:|---:|---:|---:|']
    for arm,m in metrics.items():
        z=m['supported'];f=m['full']
        lines.append(f'| {arm} | {z["crps"]:.8f} | {z["brier"]:.8f} | {f["selected_null"]}/{f["selected_n"]} | {f["selected_mean"]:.8f} |')
    lines+=['','H1/H2 and TRUE_ENERGY_RANK use future outcomes. Their coverage and policy statistics are not deployable evidence.',
            'Calibration mappings and the GLOBAL_CAL parameter use disjoint current-cell calibration outcomes only.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def run(output):
    from .variance_headroom_math import search_hindsight
    start=time.monotonic();root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    old=read_json(RADIAL/'summary.json');manifest=read_json(Path(old['reference_run'])/'run_manifest.json')
    data,metadata=load_data(old['data_directory']);ids,groups=data['ids'],data['groups']
    if len(ids)!=1188 or ids.tolist()!=manifest['ids']:raise ValueError('Unexpected development cohort')
    spec=dict(arms=ARMS,seed=SEED,search_samples=SEARCH_SAMPLES,eval_samples=EVAL_SAMPLES,grid_size=GRID_SIZE)
    spec=json.loads(json.dumps(spec))
    if (root/'spec.json').exists() and read_json(root/'spec.json')!=spec:raise ValueError('Changed run specification')
    write_json(root/'spec.json',spec)
    for path in (PLAN,Path(__file__),PROJECT/'opal2/variance_headroom_math.py'):
        destination=root/('PROTOCOL.md' if path==PLAN else path.name)
        if destination.exists() and destination.read_bytes()!=path.read_bytes():raise ValueError('Changed implementation')
        shutil.copy2(path,destination)
    prior=read_npz(RADIAL/'AMP_EMP_LOCAL.npz');base=read_npz(DUAL/'CORE.npz')
    support=read_npz(DUAL/'GELU.npz')['resource_support'].astype(bool);actual=base['actual']
    lookup={v:i for i,v in enumerate(ids)};records={r['fold']:r for r in manifest['folds']}
    for number,cell in enumerate(old['cells']):
        fold,half=cell['fold'],cell['half'];folder=root/f'cell_{fold}_{half}';folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():continue
        q=np.asarray([lookup[v] for v in cell['query_ids']]);cal=np.asarray([lookup[v] for v in cell['representative_ids']])
        fit=np.asarray(records[fold]['fit'])
        if set(groups[q])&set(groups[cal]) or set(groups[fit])&set(groups[np.r_[q,cal]]):raise ValueError('Role overlap')
        stats=read_json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        ref=read_npz(RADIAL/f'cell_{fold}_{half}_radial.npz');np.testing.assert_array_equal(ref['query_ids'],ids[q])
        law=fit_radial(ref['amplitude_radii']);scale=np.asarray(stats['u_scale']);center=np.asarray(stats['u_center'])
        status=dict(state='RUNNING',cell=number+1,cells=10,elapsed_seconds=time.monotonic()-start)
        write_json(root/'status.json',dict(status,stage='calibration_scalar_search'))
        scores,cal_opt=calibration_search(prior,ref,cell,cal,groups,stats,actual,SEED+100000*number)
        global_eta,global_info=global_calibration_choice(scores,groups[cal],scalar_grid())
        write_json(root/'status.json',dict(status,stage='query_hindsight_search'))
        print(f'cell {number+1}/10 query hindsight search',flush=True)
        opt=search_hindsight(prior['mean_u'][q],prior['scatter_u'][q],prior['actual_u'][q],stats,
            actual[q],law,ref['local_weights'],SEED+100000*number+50000,samples=SEARCH_SAMPLES,grid_size=GRID_SIZE)
        amp,learned=existing_rank_scores(fold,data,metadata,fit,cal,q)
        # Current CORE energies are saved separately. The true-rank comparison
        # must instead match the historical rank-six predictor's own frame.
        cal_total=np.square(np.linalg.solve(np.linalg.cholesky(ref['cal_amp_scatter']),ref['cal_residual'][...,None])[...,0]).sum(1)
        query_res=prior['actual_u'][q]-prior['mean_u'][q]
        query_total=np.square(np.linalg.solve(np.linalg.cholesky(prior['scatter_u'][q]),query_res[...,None])[...,0]).sum(1)
        ridge_cov=read_npz(Path(manifest['frozen_base_reference_run'])/'folds'/f'fold_{fold}'/'ridge.npz')['covariance']
        take=np.r_[cal,q]
        reference_cov=np.broadcast_to(ridge_cov,(len(take),9,9)) if ridge_cov.shape==(9,9) else ridge_cov[take]
        old_energies=error_targets(prior['mean_u'][take]*scale+center,
            reference_cov*scale[None,:,None]*scale[None,None,:],
            (prior['actual_u'][take]-prior['mean_u'][take])*scale)
        cal_rank_energy,query_rank_energy=old_energies[:len(cal),1],old_energies[len(cal):,1]
        mappings={};etas={'CORE':np.zeros((len(q),2)),'GLOBAL_CAL':np.full((len(q),2),global_eta)}
        for arm,cal_x,query_x in [('AMPLITUDE_RANK_CAL',amp[cal],amp[q]),('LEARNED_RANK_CAL',learned[cal],learned[q]),
                                 ('TRUE_ENERGY_RANK_CAL',cal_rank_energy,query_rank_energy)]:
            e,info=calibrate_rank_map(cal_x,query_x,cal_opt);etas[arm]=np.column_stack((e,e));mappings[arm]=info
        for arm,key in [('H1_CRPS','eta_scalar_crps'),('H2_CRPS','eta_two_crps'),('H1_NLL','eta_scalar_nll'),('H2_NLL','eta_two_nll')]:
            etas[arm]=np.asarray(opt[key]).copy()
        for e in etas.values():e[~support[q]]=0
        np.savez_compressed(folder/'search.npz',ids=ids[q],cal_ids=ids[cal],cal_scores=scores,cal_optimal_scalar=cal_opt,
            cal_total_energy=cal_total,query_total_energy=query_total,cal_learned=learned[cal],query_learned=learned[q],
            cal_rank6_energy=cal_rank_energy,query_rank6_energy=query_rank_energy,
            cal_amplitude=amp[cal],query_amplitude=amp[q],
            **{k:v for k,v in opt.items() if isinstance(v,np.ndarray)})
        write_json(folder/'calibration.json',dict(global_choice=global_info,rank_mappings=mappings,
            calibration_groups=groups[cal].tolist(),query_groups=groups[q].tolist(),new_predictor_fitted=False))
        write_json(root/'status.json',dict(status,stage='independent_final_sampling'))
        print(f'cell {number+1}/10 independent {EVAL_SAMPLES} draw evaluation',flush=True)
        out=evaluate_cell(prior['mean_u'][q],prior['scatter_u'][q],prior['actual_u'][q],stats,actual[q],law,
            ref['local_weights'],etas,SEED+9000000+100000*number)
        for arm,z in out.items():
            chosen=select_frozen_cohort_plan(ids[q],z['predicted'],z['p_null'],cell['budget'])
            z['selected']=np.asarray(chosen.selected_mask,int);z['fold']=np.full(len(q),fold,int)
            np.savez_compressed(folder/(arm+'.npz'),ids=ids[q],**z)
        write_json(folder/'complete.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-start))
        print(f'cell {number+1}/10 complete elapsed {time.monotonic()-start:.1f}s',flush=True)
    result=summarize(root,data,metadata,old['cells'],time.monotonic()-start)
    write_json(root/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-start))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output)


if __name__=='__main__':main()
