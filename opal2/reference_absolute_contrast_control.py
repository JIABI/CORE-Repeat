"""Separate absolute replicate spread from the known X-norm denominator."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from .reference_information_diagnostic import (PROJECT,load_data,pair_relations,weight_set,
    normalized_topk_weights,auxiliary_prediction,bootstrap_difference,write_json)


def run():
    root=PROJECT/'runs/lincs_reference_information_20260916_v2'
    summary=json.loads((root/'summary.json').read_text())
    manifest=json.loads((__import__('pathlib').Path(summary['reference_run'])/'run_manifest.json').read_text())
    data,meta=load_data(manifest['data_directory']); ids=data['ids']; n=len(ids); index={v:i for i,v in enumerate(ids)}
    counts=np.array([u['roles']['X']['cell_count'] for u in meta['units']],float)
    layout=np.array([u['layout_block'] for u in meta['units']])
    values=np.stack([np.log1p(np.square(data['Y'][:,a]-data['Y'][:,b]).mean(1))
                     for a,b in ((1,2),(1,3),(2,3))],axis=1)
    lognorm=np.log(np.linalg.norm(data['Y'][:,0],axis=1)); logcount=np.log1p(counts)
    errors={a:np.zeros(n) for a in ('POP','GENERIC','BIO','COUNT','NORM','NORM_COUNT')}
    for cell in summary['cells']:
        q=np.array([index[v] for v in cell['query_ids']]); d=np.array([index[v] for v in cell['donor_ids']])
        eligible=data['groups'][d,None]!=data['groups'][None,d]; stored={}
        for mode in errors:
            if mode in ('NORM','NORM_COUNT'):
                candidate=[]
                for query in (d,q):
                    generic,_,_=pair_relations(data,query,d,counts)
                    norm=np.exp(-.5*((lognorm[query,None]-lognorm[None,d])/max(np.std(lognorm[d]),.1))**2)
                    sim=(generic+norm)/2
                    if mode=='NORM_COUNT':
                        count=np.exp(-.5*((logcount[query,None]-logcount[None,d])/max(np.std(logcount[d]),.1))**2)
                        sim=(generic+norm+count)/3
                    allowed=data['groups'][query,None]!=data['groups'][None,d]
                    w=normalized_topk_weights(sim,donor_ids=ids[d],top_k=16,eligible=allowed)['weights']
                    candidate.append([(0.,w)])
                wd,wq=candidate; sd=sq=None
            else:
                wd,sd,_=weight_set(data,d,d,counts,mode)
                wq,sq,_=weight_set(data,q,d,counts,mode)
            prev=stored.get('GENERIC')
            pred,choice,loo=auxiliary_prediction(wd,wq,values,d,eligible,
                unsupported_d=~sd if mode=='BIO' else None,unsupported_q=~sq if mode=='BIO' else None,
                fallback_d=prev[1] if prev else None,fallback_q=prev[0] if prev else None)
            stored[mode]=(pred,loo); errors[mode][q]=np.mean((values[q]-pred)**2,axis=1)
    comparisons={a+' minus '+b:{scope:bootstrap_difference(errors[a],errors[b],lab)
        for scope,lab in (('chemistry',data['groups']),('layout',layout))}
        for a,b in (('BIO','GENERIC'),('COUNT','GENERIC'),('NORM','GENERIC'),('NORM_COUNT','NORM'))}
    result=dict(scope='post-hoc denominator diagnostic; normalized/clipped original feature units, not raw fluorescence',
        target='log1p(mean_feature((Ya-Yb)^2)); no division by X norm',
        metrics={k:float(v.mean()) for k,v in errors.items()},comparisons=comparisons)
    np.savez_compressed(root/'absolute_contrast_posthoc.npz',ids=ids,**errors)
    write_json(root/'absolute_contrast_posthoc.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2): run()
