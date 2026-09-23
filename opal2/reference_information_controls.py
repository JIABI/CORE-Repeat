"""Post-hoc mechanism controls, explicitly separate from the first diagnostic."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from .reference_information_diagnostic import (PROJECT,SEED,load_data,pair_relations,weight_set,
    normalized_topk_weights,mean_select,get_weights,score_distribution,gamma_forward,
    profiles_to_gram,gram_to_coordinates,gram_gains,torch,auxiliary_prediction,bootstrap_difference,write_json)


def run():
    root=PROJECT/'runs/lincs_reference_information_20260916_v2'
    summary=json.loads((root/'summary.json').read_text())
    ref=Path(summary['reference_run']); manifest=json.loads((ref/'run_manifest.json').read_text())
    data,meta=load_data(manifest['data_directory']); n=len(data['ids']); index={v:i for i,v in enumerate(data['ids'])}
    counts=np.array([u['roles']['X']['cell_count'] for u in meta['units']],float)
    layout=np.array([u['layout_block'] for u in meta['units']])
    raw=gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    actual,contrasts=gamma_forward(raw,True)
    energy=contrasts[:,:3]
    lognorm=np.log(np.linalg.norm(data['Y'][:,0],axis=1))
    c=np.log1p(counts)
    out={k:np.zeros(n) for k in ('mse','predicted','p_null','crps','coverage','selected','pair_crps','average_crps')}
    errors={k:np.zeros(n) for k in ('NORM','NORM_COUNT')}; rows=[]
    for cell in summary['cells']:
        fold,half=cell['fold'],cell['half']; record=manifest['folds'][fold]
        query=np.array([index[v] for v in cell['query_ids']]); donor=np.array([index[v] for v in cell['donor_ids']])
        outer=np.array(record['test']); local={v:i for i,v in enumerate(outer)}
        qi=np.array([local[v] for v in query]); di=np.array([local[v] for v in donor])
        folder=ref/'folds'/f'fold_{fold}'
        stats=json.loads((folder/'preprocessing.json').read_text())
        with np.load(folder/'arms/STATE50/evaluation/u_predictions.npz') as z:
            means=z['mean_u'].copy(); targets=z['actual_u'].copy(); cov=z['covariance_u'].copy()
        r=targets[di]-means[di]
        wg_d,_,_=weight_set(data,donor,donor,counts,'GENERIC')
        wg_q,_,_=weight_set(data,query,donor,counts,'GENERIC')
        wb_d,sd,_=weight_set(data,donor,donor,counts,'BIO')
        wb_q,sq,_=weight_set(data,query,donor,counts,'BIO')
        gs=mean_select(wg_d,r)
        gd=gs['alpha']*(get_weights(wg_d,0.)@r); gq=gs['alpha']*(get_weights(wg_q,0.)@r)
        # Same tuning capacity on the support flag, but no biological identity in weights.
        choice=mean_select(wg_d,r,~sd,gd)
        delta=choice['alpha']*(get_weights(wg_q,0.)@r); delta[~sq]=gq[~sq]
        mu=means[qi]+delta
        out['mse'][query]=np.mean((targets[qi]-mu)**2,axis=1)
        scored=score_distribution(mu,cov[qi],stats,actual[query],contrasts[query],SEED+100*fold+half)
        for key,value in scored.items(): out[key][query]=value
        order=np.lexsort((data['ids'][query],-scored['predicted']))
        out['selected'][query[order[:cell['budget']]]]=1
        choices={}
        for mode in errors:
            candidates=[]
            for q in (donor,query):
                generic,_,_=pair_relations(data,q,donor,counts)
                norm=np.exp(-.5*((lognorm[q,None]-lognorm[None,donor])/max(np.std(lognorm[donor]),.1))**2)
                sim=(generic+norm)/2
                if mode=='NORM_COUNT':
                    count=np.exp(-.5*((c[q,None]-c[None,donor])/max(np.std(c[donor]),.1))**2)
                    sim=(generic+norm+count)/3
                eligible=data['groups'][q,None]!=data['groups'][None,donor]
                w=normalized_topk_weights(sim,donor_ids=data['ids'][donor],top_k=16,eligible=eligible)['weights']
                candidates.append([(0.,w)])
            eligible=data['groups'][donor,None]!=data['groups'][None,donor]
            prediction,selected,_=auxiliary_prediction(*candidates,energy,donor,eligible)
            errors[mode][query]=np.mean((energy[query]-prediction)**2,axis=1)
            choices[mode]=selected
        rows.append(dict(fold=fold,half=half,support_only_choice=choice,contrast_choices=choices))
    out['brier']=(out['p_null']-(actual<=0))**2
    out['policy_value']=actual*out['selected']
    np.savez_compressed(root/'SUPPORT_ONLY_MEAN_posthoc.npz',ids=data['ids'],actual=actual,**out)
    np.savez_compressed(root/'norm_count_controls_posthoc.npz',ids=data['ids'],**errors)
    bio=np.load(root/'BIO_MEAN.npz')
    comparisons={key:{scope:bootstrap_difference(bio[key],out[key],lab)
        for scope,lab in (('chemistry',data['groups']),('layout',layout))}
        for key in ('mse','crps','brier','policy_value')}
    count_comparison={scope:bootstrap_difference(errors['NORM_COUNT'],errors['NORM'],lab)
        for scope,lab in (('chemistry',data['groups']),('layout',layout))}
    result=dict(scope='post-hoc mechanism clarification after the first diagnostic, not preregistered confirmation',
        support_only_metrics={k:float(out[k].mean()) for k in ('mse','crps','brier','policy_value')},
        biological_identity_minus_support_flag=comparisons,
        norm_control_metrics={k:float(v.mean()) for k,v in errors.items()},
        count_after_norm_comparison=count_comparison,choices=rows)
    write_json(root/'posthoc_controls.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2): run()
