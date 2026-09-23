"""Two fresh, paired, bounded cutoff Monte Carlo checks; no model refitting."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .decision_frontier_diagnostic import cell_indices,select_policy,counts_and_metrics
from .empirical_radial import draw_radial
from .reference_information_diagnostic import gamma_forward
from .policy_freeze_diagnostic import POLICIES


def moment_scores(gamma):
    n=len(gamma);indicator=gamma<=0;mean=gamma.mean(0);p=indicator.mean(0)
    cov=(np.mean(gamma*indicator,axis=0)-mean*p)/(n-1)
    return dict(predicted=mean,p_null=p,gamma_mc_se=gamma.std(0,ddof=1)/np.sqrt(n),
        null_mc_se=np.sqrt(p*(1-p)/(n-1)),mc_mean_null_covariance=cov)


def mc_score_se(mean_se,prob_se,covariance,lam):
    variance=mean_se**2+lam**2*prob_se**2-2*lam*covariance
    if np.min(variance)<-1e-15:raise ValueError('Negative score variance')
    return np.sqrt(np.maximum(variance,0))


def candidate_pool(stores,ids,cells,prior_private):
    lookup={v:i for i,v in enumerate(ids)};pool=np.zeros(len(ids),bool)
    for name in POLICIES:
        arm,l=name.split('__lambda_');lam=float(l);data=stores[arm]
        score=data['predicted']-lam*data['p_null']
        for q,k in cells:
            order=np.lexsort((ids[q],-score[q]));pool[q[order[max(0,k-5):min(len(q),k+5)]]]=True
        for c in prior_private['mc'][name]:
            for field in ('potential_removed_ids','potential_added_ids'):
                pool[[lookup[v] for v in c[field]]]=True
    return pool


def run(source,policy_directory,output,samples=50000):
    start=time.monotonic();source,policy_directory,output=map(lambda p:Path(p).resolve(),(source,policy_directory,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    summary=json.loads((source/'summary.json').read_text())
    private=json.loads((policy_directory/'private_affected_objects_and_mc_cutoffs.json').read_text())
    stores={a:dict(np.load(source/f'{a}.npz')) for a in ('GAUSSIAN','AMP_EMP_LOCAL')}
    ids=stores['GAUSSIAN']['ids'];actual=stores['GAUSSIAN']['actual'];n=len(ids)
    cells=cell_indices(ids,summary['cells']);pool=candidate_pool(stores,ids,cells,private)
    np.savez_compressed(output/'candidate_pool.npz',ids=ids,pool=pool)
    write_json(output/'PROTOCOL.json',dict(samples_per_seed=samples,seeds=[911001,921001],
        pool_rule='Union of +/-5 cutoff ranks and existing marginal-SE upper-envelope overlaps for fixed four policies; no outcomes used',
        pool_n=int(pool.sum()),unchanged_model=True,no_refit=True,no_new_lambda=True,
        outside_pool='Retain original10k-draw scores; this is bounded cutoff precision sensitivity, not full cohort100k replay'))
    moments={a:[] for a in stores};records=[]
    for repeat,offset in enumerate((911001,921001)):
        new={a:{k:v.copy() for k,v in stores[a].items() if k in ('predicted','p_null','gamma_mc_se','null_mc_se')} for a in stores}
        for a in stores:new[a]['mc_mean_null_covariance']=np.full(n,np.nan)
        for j,(q,k) in enumerate(cells):
            cell=summary['cells'][j];positions=np.flatnonzero(pool[q]);take=q[positions]
            fold,half=cell['fold'],cell['half']
            stats=json.loads((Path(summary['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json').read_text())
            with np.load(source/f'cell_{fold}_{half}_radial.npz') as z:
                np.testing.assert_array_equal(z['query_ids'],ids[q]);weights=z['local_weights'][positions]
            normal_rng=np.random.default_rng(offset+100*fold+half)
            radius_rng=np.random.default_rng(offset+47000+100*fold+half)
            for begin in range(0,len(take),4):
                end=min(begin+4,len(take));ix=take[begin:end]
                normal=normal_rng.normal(size=(samples,len(ix),9))
                mix=radius_rng.random((samples,len(ix)));kernel=radius_rng.random((samples,len(ix)))
                for arm,data in stores.items():
                    scatter=data['scatter_u'][ix]
                    if arm=='GAUSSIAN':eps=np.einsum('nij,snj->sni',np.linalg.cholesky(scatter),normal)
                    else:eps=draw_radial(cell['laws']['amplitude_law'],weights[begin:end],scatter,normal,mix,kernel)
                    u=data['mean_u'][ix][None]+eps
                    gamma=gamma_forward(u*np.asarray(stats['u_scale'])+np.asarray(stats['u_center']))
                    for key,value in moment_scores(gamma).items():new[arm][key][ix]=value
            print(f'MC seed{repeat+1} cell{j+1}/10 pool={len(take)} elapsed={time.monotonic()-start:.1f}',flush=True)
        for arm,out in new.items():
            moments[arm].append(out)
            np.savez_compressed(output/f'{arm}_seed{repeat+1}.npz',ids=ids,pool=pool,**out)
        records.append(evaluate(new,stores,ids,actual,cells,pool,f'seed{repeat+1}',samples))
    combined={}
    for arm,(one,two) in moments.items():
        out={k:v.copy() for k,v in one.items()}
        mu=(one['predicted'][pool]+two['predicted'][pool])/2
        p=(one['p_null'][pool]+two['p_null'][pool])/2
        # Pool unbiased sample moments, including between-seed mean differences.
        var_sum=(samples*(samples-1)*(one['gamma_mc_se'][pool]**2+two['gamma_mc_se'][pool]**2)
            +samples*((one['predicted'][pool]-mu)**2+(two['predicted'][pool]-mu)**2))
        out['predicted'][pool]=mu;out['p_null'][pool]=p
        out['gamma_mc_se'][pool]=np.sqrt(var_sum/((2*samples-1)*(2*samples)))
        out['null_mc_se'][pool]=np.sqrt(p*(1-p)/(2*samples-1))
        cov_sum=(samples*(samples-1)*(one['mc_mean_null_covariance'][pool]+two['mc_mean_null_covariance'][pool])
            +samples*((one['predicted'][pool]-mu)*(one['p_null'][pool]-p)+(two['predicted'][pool]-mu)*(two['p_null'][pool]-p)))
        out['mc_mean_null_covariance'][pool]=cov_sum/((2*samples-1)*(2*samples))
        combined[arm]=out
        np.savez_compressed(output/f'{arm}_combined.npz',ids=ids,pool=pool,**out)
    records.append(evaluate(combined,stores,ids,actual,cells,pool,'combined',2*samples))
    result=dict(pool_n=int(pool.sum()),records=records,samples_per_seed=samples,seeds=2,
        final_opened=False,refit=False,full_cohort_resimulation=False,
        interpretation='Fresh numerical integration sensitivity conditional on frozen measurement model, not biological sampling uncertainty or independent validation',elapsed_seconds=time.monotonic()-start)
    write_json(output/'summary.json',result)
    return result


def evaluate(new,old,ids,actual,cells,pool,name,samples):
    rows={}
    for policy in POLICIES:
        arm,lam=policy.split('__lambda_');lam=float(lam);data=new[arm]
        chosen=select_policy(data['predicted'],data['p_null'],ids,cells,lam)
        prior=select_policy(old[arm]['predicted'],old[arm]['p_null'],ids,cells,lam)
        row=counts_and_metrics(actual,data['predicted'],data['p_null'],chosen)
        row['replacements_from_original10k']=int(np.count_nonzero(chosen!=prior)//2)
        row['changed_ids']=ids[chosen!=prior].tolist();checks=[]
        for j,(q,k) in enumerate(cells):
            score=data['predicted'][q]-lam*data['p_null'][q]
            order=np.lexsort((ids[q],-score));left,right=q[order[k-1]],q[order[k]]
            checks.append(dict(cell=j,cutoff_both_in_pool=bool(pool[left] and pool[right]),
                selected_cutoff_id=str(ids[left]),unselected_cutoff_id=str(ids[right]),
                score_gap=float(score[order[k-1]]-score[order[k]])))
        row['cutoff_checks']=checks;rows[policy]=row
    return dict(name=name,samples_for_pool=samples,policies=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--policy-directory',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.policy_directory,a.output)
