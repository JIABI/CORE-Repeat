"""Honest auxiliary-reference diagnostics for the frozen LINCS STATE50 model.

The reference/query split is internal to an original outer test fold. These
are development diagnostics with extra reference measurements, not a replay
of the original acquisition contract. No neural model is retrained.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from .lincs_biology_experiment import load_data
from .reference_information_memory import (cosine_relationship, morphology_similarity,
    chemistry_similarity, normalized_topk_weights, augment_generic_weights,
    weighted_residual_moments)
from .objective_analysis import fair_crps
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .biology_kernel_evaluation import write_json

PROJECT = Path(__file__).resolve().parents[1]
SEED = 20260916
LAMBDAS = (0., .25, .5, .75, 1.)
ALPHAS = (0., .25, .5, 1.)
BETAS = (0., .25, .5, .75)
ARMS = ('BASE','POP_MEAN','GENERIC_MEAN','BIO_MEAN','COUNT_MEAN','SHUFFLE_MEAN',
        'POP_COV','GENERIC_COV','BIO_COV','GENERIC_BOTH','BIO_BOTH')


def pair_relations(data, query, donor, counts, permutation=None):
    morph = morphology_similarity(data['Y'][query,0], data['Y'][donor,0])
    chem = chemistry_similarity(data['chem'][query,:512], data['chem'][donor,:512])
    generic = (morph+chem)/2
    c = np.log1p(counts)
    scale = max(float(np.std(c[donor])), .1)
    count = np.exp(-.5*((c[query,None]-c[None,donor])/scale)**2)
    rows = np.arange(len(data['ids'])) if permutation is None else permutation
    biological = [cosine_relationship(data[key][rows[query]],data[key][rows[donor]],
        query_mask=data[key+'_mask'][rows[query]], donor_mask=data[key+'_mask'][rows[donor]])
        for key in ('target','moa')]
    return generic, (generic+count)/2, biological


def weight_set(data, query, donor, counts, mode, permutation=None):
    eligible = data['groups'][query,None] != data['groups'][None,donor]
    g, count, bio = pair_relations(data,query,donor,counts,permutation)
    if mode == 'POP':
        w = eligible/eligible.sum(1,keepdims=True)
        return [(0.,w)], np.zeros(len(query),bool), np.zeros(len(query))
    generic = normalized_topk_weights(count if mode=='COUNT' else g,top_k=16,
        donor_ids=data['ids'][donor],eligible=eligible)['weights']
    if mode not in ('BIO','SHUFFLE'):
        return [(0.,generic)], np.zeros(len(query),bool), np.zeros(len(query))
    candidates = []
    for lam in LAMBDAS:
        out = augment_generic_weights(generic,*bio,mixing_lambda=lam,eligible=eligible,shrinkage=8.)
        candidates.append((lam,out['weights']))
    return candidates,out['support'],out['neff']


def mean_select(candidates,residual,unsupported=None,fallback=None):
    best = None
    for lam,w in candidates:
        borrowed = w@residual
        for alpha in ALPHAS:
            prediction=alpha*borrowed
            if unsupported is not None: prediction[unsupported]=fallback[unsupported]
            loss = float(np.mean((residual-prediction)**2))
            key = (loss,lam,alpha)
            if best is None or key < best[0]:
                best = (key,dict(lam=lam,alpha=alpha,donor_loo_mse=loss))
    return best[1]


def get_weights(candidates,lam):
    return next(w for l,w in candidates if l==lam)


def residual_second_moment(weights,residual,correction=0.):
    moments=weighted_residual_moments(weights,residual)
    bias=moments['mean']-correction
    return moments['covariance']+np.einsum('ni,nj->nij',bias,bias)


def covariance_select(candidates,residual,cov0,score_residual=None,unsupported=None,fallback=None):
    best = None
    for lam,w in candidates:
        delta=0. if score_residual is None else residual-score_residual
        cov = residual_second_moment(w,residual,delta)
        for beta in BETAS:
            c = (1-beta)*cov0+beta*cov
            if unsupported is not None: c[unsupported]=fallback[unsupported]
            sign,ld = np.linalg.slogdet(c)
            if np.any(sign<=0):
                raise ValueError('Nonpositive covariance')
            error=residual if score_residual is None else score_residual
            loss = float(np.mean(ld+np.einsum('ni,ni->n',error,
                np.linalg.solve(c,error[...,None])[...,0])))
            key = (loss,lam,beta)
            if best is None or key<best[0]:
                best=(key,dict(lam=lam,beta=beta,donor_loo_score=loss))
    return best[1]


def gamma_forward(raw, return_contrasts=False):
    """Exact original Gamma using four virtual vectors of the joint Gram.

    This is a forward computation, not sampling independent angles. A checked
    positive triangular factor retains every cross-well dependency.
    """
    rows = np.zeros((*raw.shape[:-1],4,4))
    rows[...,0,0]=1
    rows[...,1:,0]=raw[...,:3]
    rows[...,1,1]=np.exp(raw[...,3]); rows[...,2,1]=raw[...,4]
    rows[...,2,2]=np.exp(raw[...,5]); rows[...,3,1]=raw[...,6]
    rows[...,3,2]=raw[...,7]; rows[...,3,3]=np.exp(raw[...,8])
    if not np.isfinite(rows).all() or np.any(rows[...,(1,2,3),(1,2,3)]<=0):
        raise ValueError('Invalid factor; no clipping/resampling is allowed')
    v=rows[...,3,:]; av=rows[...,:3,:].mean(-2)
    nv=np.linalg.norm(v,axis=-1)
    gamma=.5*((av*v).sum(-1)/(np.linalg.norm(av,axis=-1)*nv)-v[...,0]/nv)-.02
    if return_contrasts:
        contrast=np.stack([np.log1p(np.square(rows[...,a,:]-rows[...,b,:]).sum(-1))
            for a,b in ((1,2),(1,3),(2,3))]+
            [np.log1p(np.square(rows[...,1:,:].mean(-2)).sum(-1))],axis=-1)
        return gamma,contrast
    return gamma


def score_distribution(mean,cov,stats,actual,actual_contrasts,seed):
    n=len(mean); result={k:np.empty(n) for k in ('predicted','p_null','crps','coverage','pair_crps','average_crps')}
    rng=np.random.default_rng(seed)
    # Same draw streams in all arms. At most 16 object blocks reside in memory.
    for start in range(0,n,16):
        end=min(n,start+16)
        z=rng.normal(size=(10000,end-start,9))
        u=mean[None,start:end]+np.einsum('nij,snj->sni',np.linalg.cholesky(cov[start:end]),z)
        raw=u*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
        draws,contrasts=gamma_forward(raw,True)
        lo,hi=np.quantile(draws,[.025,.975],axis=0)
        result['predicted'][start:end]=draws.mean(0)
        result['p_null'][start:end]=(draws<=0).mean(0)
        result['crps'][start:end]=fair_crps(draws,actual[start:end])
        result['coverage'][start:end]=(actual[start:end]>=lo)&(actual[start:end]<=hi)
        cc=fair_crps(contrasts,actual_contrasts[start:end])
        result['pair_crps'][start:end]=cc[:,:3].mean(1)
        result['average_crps'][start:end]=cc[:,3]
    return result


def contrast_targets(y,fit):
    # Fixed TRAIN-only coordinates describe direction, not a new endpoint.
    pca=PCA(8,svd_solver='full').fit(y[fit,0])
    norm2=np.square(y[:,0]).sum(-1)
    energies=[]; directions=[]
    for a,b in ((1,2),(1,3),(2,3)):
        diff=y[:,a]-y[:,b]
        energy=np.square(diff).sum(-1)
        energies.append(np.log1p(energy/norm2))
        proj=diff@pca.components_.T
        # Sign-invariant directional moment; large differences cannot dominate.
        outer=np.einsum('ni,nj->nij',proj,proj)/np.maximum(energy[:,None,None],1e-20)
        directions.append(outer[:,np.triu_indices(8)[0],np.triu_indices(8)[1]])
    return np.stack(energies,1),np.concatenate(directions,1)


def auxiliary_prediction(candidates_d,candidates_q,values,donor,eligible,
                         unsupported_d=None,unsupported_q=None,fallback_d=None,fallback_q=None):
    center=values[donor].mean(0)
    v=values[donor]
    loo_center=(eligible/eligible.sum(1,keepdims=True))@v
    best=None
    for lam,w in candidates_d:
        for alpha in ALPHAS:
            pred=(1-alpha)*loo_center+alpha*(w@v)
            if unsupported_d is not None: pred[unsupported_d]=fallback_d[unsupported_d]
            key=(float(np.mean((v-pred)**2)),lam,alpha)
            if best is None or key<best:
                best=key
    _,lam,alpha=best
    pred=(1-alpha)*center+alpha*(get_weights(candidates_q,lam)@v)
    loo=(1-alpha)*loo_center+alpha*(get_weights(candidates_d,lam)@v)
    if unsupported_q is not None: pred[unsupported_q]=fallback_q[unsupported_q]
    if unsupported_d is not None: loo[unsupported_d]=fallback_d[unsupported_d]
    return pred,dict(lam=lam,alpha=alpha,donor_loo_mse=best[0]),loo


def bootstrap_difference(a,b,labels,seed=SEED):
    difference=np.asarray(a)-np.asarray(b)
    groups=np.unique(labels); sums=np.array([difference[labels==g].sum() for g in groups])
    sizes=np.array([(labels==g).sum() for g in groups])
    rng=np.random.default_rng(seed)
    ix=rng.integers(len(groups),size=(2000,len(groups)))
    samples=sums[ix].sum(1)/sizes[ix].sum(1)
    return dict(difference=float(difference.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())


def run(reference,output):
    started=time.monotonic(); root=Path(output).resolve(); ref=Path(reference).resolve()
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/REFERENCE_INFORMATION_PLAN_20260916.md',root/'PROTOCOL.md')
    for name in ('reference_information_diagnostic.py','reference_information_memory.py'):
        shutil.copy2(PROJECT/'opal2'/name,root/name)
    m=json.loads((ref/'run_manifest.json').read_text())
    data,meta=load_data(m['data_directory']); n=len(data['ids'])
    if n!=1188 or data['ids'].tolist()!=m['ids']: raise ValueError('Scope changed')
    counts=np.array([v['roles']['X']['cell_count'] for v in meta['units']],float)
    layout=np.array([v['layout_block'] for v in meta['units']])
    torch.set_num_threads(2)
    gram=profiles_to_gram(torch.tensor(data['Y'])).numpy()
    raw=gram_to_coordinates(torch.tensor(gram)).numpy()
    actual=gram_gains(torch.tensor(gram)).numpy()[:,2]
    _,actual_contrasts=gamma_forward(raw,True)
    if not np.allclose(gamma_forward(raw),actual,atol=1e-12,rtol=1e-12):
        raise ValueError('Forward Gamma identity failed')
    stores={a:{k:np.zeros(n) for k in ('mse','predicted','p_null','crps','coverage','selected','correction2','alignment','pair_crps','average_crps')}
            for a in ARMS}
    for out in stores.values():
        out.update(mean_u=np.empty((n,9)),actual_u=np.empty((n,9)),covariance_u=np.empty((n,9,9)))
    diagnostic={mode:{k:np.zeros(n) for k in ('contrast_energy_mse','contrast_direction_mse')}
                for mode in ('POP','GENERIC','BIO','COUNT')}
    cells=[]; support=np.zeros(n,bool); neff=np.zeros(n); seen=np.zeros(n,int)
    for record in m['folds']:
        fold=record['fold']; outer=np.array(record['test']); folder=ref/'folds'/f'fold_{fold}'
        stats=json.loads((folder/'preprocessing.json').read_text())
        with np.load(folder/'arms/STATE50/evaluation/u_predictions.npz') as z:
            if not np.array_equal(z['ids'],data['ids'][outer]): raise ValueError('Identity mismatch')
            mean=z['mean_u'].copy(); target=z['actual_u'].copy(); cov=z['covariance_u'].copy()
        if not np.allclose(target,(raw[outer]-stats['u_center'])/stats['u_scale']):
            raise ValueError('Target definition mismatch')
        residual=target-mean
        local={v:i for i,v in enumerate(outer)}
        rng=np.random.default_rng(SEED+fold)
        unique=np.unique(data['groups'][outer]); rng.shuffle(unique)
        halves=[outer[np.isin(data['groups'][outer],u)] for u in np.array_split(unique,2)]
        total=int(np.floor(.25*len(outer)))//2
        k0=int(np.floor(total*len(halves[0])/len(outer))); budgets=[k0,total-k0]
        energies,directions=contrast_targets(data['Y'],np.array(record['fit']))
        permutation=np.arange(n); permutation[outer]=rng.permutation(outer)
        for half in range(2):
            query,donor=halves[half],halves[1-half]
            qi=np.array([local[v] for v in query]); di=np.array([local[v] for v in donor])
            r=residual[di]; base=mean[qi]; c0=cov[qi]
            if set(data['groups'][query])&set(data['groups'][donor]): raise ValueError('Group leak')
            seen[query]+=1
            cell=dict(fold=fold,half=half,query_ids=data['ids'][query].tolist(),
                donor_ids=data['ids'][donor].tolist(),budget=budgets[half],choices={})
            means={'BASE':base}; covs={'BASE':c0}
            donor_means={}; donor_covs={}; donor_both_covs={}; aux_predictions={}
            for mode in ('POP','GENERIC','BIO','COUNT','SHUFFLE'):
                perm=permutation if mode=='SHUFFLE' else None
                wd,sup_d,_=weight_set(data,donor,donor,counts,mode,perm)
                wq,sup,eff=weight_set(data,query,donor,counts,mode,perm)
                is_bio=mode in ('BIO','SHUFFLE')
                ms=mean_select(wd,r,~sup_d if is_bio else None,donor_means.get('GENERIC'))
                correction=ms['alpha']*(get_weights(wq,ms['lam'])@r)
                donor_means[mode]=ms['alpha']*(get_weights(wd,ms['lam'])@r)
                if is_bio: donor_means[mode][~sup_d]=donor_means['GENERIC'][~sup_d]
                means[mode+'_MEAN']=base+correction; covs[mode+'_MEAN']=c0
                if mode in ('BIO','SHUFFLE'):
                    # An unsupported query must retain the entire tuned generic
                    # route, not merely identical weights with a different alpha.
                    means[mode+'_MEAN'][~sup]=means['GENERIC_MEAN'][~sup]
                cell['choices'][mode]=dict(mean=ms)
                if mode=='BIO': support[query]=sup; neff[query]=eff
                if mode in ('POP','GENERIC','BIO'):
                    cs=covariance_select(wd,r,cov[di],unsupported=~sup_d if mode=='BIO' else None,
                        fallback=donor_covs.get('GENERIC'))
                    lcov=residual_second_moment(get_weights(wq,cs['lam']),r)
                    corrected=(1-cs['beta'])*c0+cs['beta']*lcov
                    dcov=residual_second_moment(get_weights(wd,cs['lam']),r)
                    donor_covs[mode]=(1-cs['beta'])*cov[di]+cs['beta']*dcov
                    if mode=='BIO': donor_covs[mode][~sup_d]=donor_covs['GENERIC'][~sup_d]
                    if mode=='BIO': corrected[~sup]=covs['GENERIC_COV'][~sup]
                    means[mode+'_COV']=base; covs[mode+'_COV']=corrected
                    if mode!='POP':
                        both=covariance_select(wd,r,cov[di],score_residual=r-donor_means[mode],
                            unsupported=~sup_d if mode=='BIO' else None,fallback=donor_both_covs.get('GENERIC'))
                        bq=residual_second_moment(get_weights(wq,both['lam']),r,means[mode+'_MEAN']-base)
                        bd=residual_second_moment(get_weights(wd,both['lam']),r,donor_means[mode])
                        bcov=(1-both['beta'])*c0+both['beta']*bq
                        donor_both_covs[mode]=(1-both['beta'])*cov[di]+both['beta']*bd
                        if mode=='BIO':
                            bcov[~sup]=covs['GENERIC_BOTH'][~sup]
                            donor_both_covs[mode][~sup_d]=donor_both_covs['GENERIC'][~sup_d]
                        means[mode+'_BOTH']=means[mode+'_MEAN']; covs[mode+'_BOTH']=bcov
                        cell['choices'][mode]['combined_covariance']=both
                    cell['choices'][mode]['covariance']=cs
                if mode!='SHUFFLE':
                    for label,values in (('contrast_energy',energies),('contrast_direction',directions)):
                        eligible=data['groups'][donor,None]!=data['groups'][None,donor]
                        previous=aux_predictions.get(('GENERIC',label))
                        prediction,choice,loo=auxiliary_prediction(wd,wq,values,donor,eligible,
                            unsupported_d=~sup_d if mode=='BIO' else None,
                            unsupported_q=~sup if mode=='BIO' else None,
                            fallback_d=previous[1] if previous else None,
                            fallback_q=previous[0] if previous else None)
                        aux_predictions[(mode,label)]=(prediction,loo)
                        diagnostic[mode][label+'_mse'][query]=np.mean((values[query]-prediction)**2,axis=1)
                        cell['choices'][mode][label]=choice
            for arm in ARMS:
                out=stores[arm]; mu,cv=means[arm],covs[arm]
                out['mean_u'][query]=mu; out['actual_u'][query]=target[qi]; out['covariance_u'][query]=cv
                d=mu-base
                out['mse'][query]=np.mean((target[qi]-mu)**2,axis=1)
                out['correction2'][query]=np.mean(d*d,axis=1)
                out['alignment'][query]=2*np.mean(residual[qi]*d,axis=1)
                dist=score_distribution(mu,cv,stats,actual[query],actual_contrasts[query],SEED+100*fold+half)
                for key,value in dist.items(): out[key][query]=value
                order=np.lexsort((data['ids'][query],-dist['predicted']))
                out['selected'][query[order[:budgets[half]]]]=1
            cells.append(cell)
            write_json(root/'progress.json',dict(completed_cells=len(cells),total_cells=10,
                elapsed_seconds=time.monotonic()-started))
            print('CELL_COMPLETE',fold,half,round(time.monotonic()-started,1),flush=True)
    if not np.all(seen==1): raise ValueError('Each query must be scored exactly once')
    metrics={}
    for arm,out in stores.items():
        out['brier']=(out['p_null']-(actual<=0))**2
        out['policy_value']=out['selected']*actual
        selected=out['selected'].astype(bool)
        if selected.sum()!=146: raise ValueError('Budget changed')
        metrics[arm]={k:float(out[k].mean()) for k in ('mse','crps','brier','coverage','correction2','alignment','policy_value','pair_crps','average_crps')}
        metrics[arm].update(spearman=float(spearmanr(actual,out['predicted']).statistic),
            predicted_mean=float(out['predicted'].mean()),actual_mean=float(actual.mean()),
            selected_mean=float(actual[selected].mean()),selected_n=int(selected.sum()),
            selected_null=int((actual[selected]<=0).sum()),mean_null_probability=float(out['p_null'].mean()))
        np.savez_compressed(root/(arm+'.npz'),ids=data['ids'],actual=actual,**out)
    pairs=[('GENERIC_MEAN','BASE'),('BIO_MEAN','GENERIC_MEAN'),('COUNT_MEAN','GENERIC_MEAN'),
           ('SHUFFLE_MEAN','GENERIC_MEAN'),('GENERIC_COV','BASE'),('BIO_COV','GENERIC_COV'),
           ('GENERIC_BOTH','BASE'),('BIO_BOTH','GENERIC_BOTH'),('POP_COV','BASE')]
    comparisons={}
    for a,b in pairs:
        comparisons[a+' minus '+b]={metric:{scope:bootstrap_difference(stores[a][metric],stores[b][metric],lab)
            for scope,lab in (('chemistry',data['groups']),('layout',layout))}
            for metric in ('mse','crps','brier','policy_value','pair_crps','average_crps')}
    aux={mode:{k:float(v.mean()) for k,v in d.items()} for mode,d in diagnostic.items()}
    aux_comp={a+' minus '+b:{key:{scope:bootstrap_difference(diagnostic[a][key],diagnostic[b][key],lab)
                for scope,lab in (('chemistry',data['groups']),('layout',layout))}
             for key in diagnostic[a]} for a,b in (('GENERIC','POP'),('BIO','GENERIC'),('COUNT','GENERIC'))}
    np.savez_compressed(root/'contrast_diagnostics.npz',ids=data['ids'],support=support,neff=neff,
        **{m+'_'+k:v for m,d in diagnostic.items() for k,v in d.items()})
    result=dict(scope='retrospective split-reference development diagnostic',n=n,samples=10000,
        reference_run=str(ref),support_n=int(support.sum()),supported_neff_median=float(np.median(neff[support])),
        metrics=metrics,comparisons=comparisons,contrast_metrics=aux,contrast_comparisons=aux_comp,
        cells=cells,elapsed_seconds=time.monotonic()-started,final_opened=False,endpoint_changed=False,
        reference_cost_included=False,original_models_modified=False,
        interval_scope='conditional on fixed predictions; excludes retraining and repeated development selection')
    write_json(root/'summary.json',result)
    write_json(root/'status.json',dict(state='COMPLETE',elapsed_seconds=result['elapsed_seconds']))
    print('COMPLETE',round(result['elapsed_seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--reference',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2): run(args.reference,args.output)
