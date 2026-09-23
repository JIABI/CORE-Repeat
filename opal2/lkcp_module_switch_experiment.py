"""Complete frozen-mean LKCP transfer with independent optional radial switches."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_joint_error import fit_covariance_family
from .conditional_joint_error_experiment import reference_weights as covariance_weights
from .conditional_joint_error_experiment import observable_forward
from .empirical_radial import fit_radial, reference_weights
from .empirical_radial_experiment import score
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .gram_geometry import profiles_to_gram, gram_to_coordinates
from .joint_contrast_scale import fit_scale, predict_scale
from .module_switch_experiment import (PROJECT, ARMS, SEED, fit_states,
    biology_similarity, summarize)
from .optional_radial_modules import (fit_radial_switch, BIO_COEFFICIENTS,
    state_similarity)


def load_conditions(root):
    records=[]; metadata=[]
    for condition in ('A549_24H','A549_48H'):
        path=Path(root)/condition
        with np.load(path/'data.npz',allow_pickle=False) as z:
            records.append({k:z[k].copy() for k in z.files})
        metadata.append(json.loads((path/'metadata.json').read_text()))
    np.testing.assert_array_equal(records[0]['feature_names'],records[1]['feature_names'])
    data={k:np.concatenate([r[k] for r in records],axis=0) for k in
        ('Y','ids','groups','chem','chem_mask','target','target_mask','moa','moa_mask')}
    data['feature_names']=records[0]['feature_names']
    if len(set(data['ids']))!=len(data['ids']): raise ValueError('Condition IDs must be unique')
    if not np.isfinite(data['Y']).all(): raise ValueError('No silent missing measurement replacement')
    n0=len(records[0]['ids'])
    return data,dict(units=metadata[0]['units']+metadata[1]['units']),n0


def grouped_roles(groups, source_count):
    """All doses/positions of each identity stay together across both conditions."""
    groups=np.asarray(groups)
    unique=np.unique(groups[:source_count])
    if set(groups[source_count:])!=set(unique): raise ValueError('Matched source/query groups required')
    records=[]
    for fold,(pool,held) in enumerate(KFold(5,shuffle=True,random_state=SEED).split(unique)):
        model,refs=train_test_split(pool,test_size=.4,random_state=SEED+100*fold+1)
        covariance,calibration=train_test_split(refs,test_size=.5,random_state=SEED+100*fold+2)
        row={'fold':fold}
        for name,selected in (('fit',model),('covfit',covariance),('calibration',calibration)):
            row[name]=np.flatnonzero(np.isin(groups[:source_count],unique[selected]))
        row['query']=source_count+np.flatnonzero(np.isin(groups[source_count:],unique[held]))
        parts=[set(groups[row[k]]) for k in ('fit','covfit','calibration','query')]
        if any(parts[i]&parts[j] for i in range(4) for j in range(i+1,4)):
            raise ValueError('Chemical group crosses module roles')
        records.append(row)
    return records


def error_recipe(data, mean, target, base_scatter, rows):
    """Unchanged LOCAL_SCALE -> amplitude -> empirical law recipe, source only."""
    fit,donor,cal,q=(rows[k] for k in ('fit','covfit','calibration','query'))
    logamp=np.log(np.linalg.norm(data['Y'][:,0],axis=1))
    bandwidth=max(float(logamp[fit].std()),.1)
    dweights,_=covariance_weights(data,donor,donor,bandwidth)
    recipients=np.r_[cal,q]
    rweights,_=covariance_weights(data,recipients,donor,bandwidth)
    residual=target[donor]-mean[donor]
    fitted=fit_covariance_family(residual,base_scatter,dweights,rweights,'LOCAL_SCALE')
    whitened=np.linalg.solve(np.linalg.cholesky(fitted['loo_covariance']),residual[...,None])[...,0]
    amplitude=fit_scale(np.square(whitened).sum(1),9,logamp[donor],conditional=True)
    scatter=fitted['query_covariance']*predict_scale(amplitude,logamp[recipients])[:,None,None]
    cal_scatter,query_scatter=scatter[:len(cal)],scatter[len(cal):]
    radii=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(cal_scatter),
        (target[cal]-mean[cal])[...,None])[...,0],axis=1)
    # The core's radial bandwidth is estimated on the error-fitting references.
    radial_bandwidth=float(logamp[donor].std())
    base=reference_weights(logamp[cal],logamp[q],radial_bandwidth,conditional=True)['weights']
    return dict(law=fit_radial(radii),radii=radii,weights=base,scatter=query_scatter,
        cal_scatter=cal_scatter,logamp=logamp,radial_bandwidth=radial_bandwidth,
        choice=fitted['choice'],amplitude=amplitude)


def run(data_root,output,state_run,epochs=60):
    from .frozen_state50_transfer import FrozenState50Transfer
    started=time.monotonic();root=Path(output).resolve()
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/LKCP_MODULE_TRANSFER_PLAN_20260916.md',root/'PROTOCOL.md')
    for name in ('lkcp_module_switch_experiment.py','optional_radial_modules.py',
                 'conditional_state_representation.py','frozen_state50_transfer.py'):
        shutil.copy2(PROJECT/'opal2'/name,root/name)
    data,metadata,n0=load_conditions(data_root)
    ids,groups=data['ids'],data['groups']; query_ids=ids[n0:];n=len(query_ids)
    roles=grouped_roles(groups,n0)
    write_json(root/'roles_before_target_scoring.json',dict(seed=SEED,roles=roles,
        ids=ids.tolist(),groups=groups.tolist(),source_count=n0,query_count=n,
        core_fold=0,primary_evaluation=False))
    core=FrozenState50Transfer.load(state_run=state_run,fold=0)
    prediction=core.predict(data['Y'][:,0],data['chem'],data['chem_mask'],
        feature_names=data['feature_names'])
    mean=prediction['mean_u'];base_scatter=prediction['base_scatter_u'][0]
    stats=core.stats
    raw=gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    target=(raw-np.asarray(stats['u_center']))/np.asarray(stats['u_scale'])
    actual,obs,difference,_=observable_forward(raw)
    norm2=np.square(data['Y'][:,0]).mean(1)
    absolute=np.log1p(difference*norm2[:,None])
    stores={arm:{} for arm in ARMS};cells=[];seen=np.zeros(n,int);folds=np.full(n,-1,int)
    for rows in roles:
        fold=rows['fold'];fit,cal,q=(rows[k] for k in ('fit','calibration','query'))
        local=q-n0;folder=root/f'fold_{fold}';folder.mkdir()
        write_json(root/'status.json',dict(state='RUNNING',fold=fold,phase='fit_source_states',
            cells_complete=len(cells),elapsed_seconds=time.monotonic()-started))
        states,training=fit_states(data,fit,folder,epochs=epochs)
        errors=error_recipe(data,mean,target,base_scatter,rows)
        base=errors['weights'];amp=errors['logamp'];radii=errors['radii']
        allowed=groups[q,None]!=groups[None,cal]
        plans={'CORE':dict(weights=base.copy(),gate=np.zeros(len(q)),
            support=np.ones(len(q),bool),selection=None)}
        bc=biology_similarity(data,metadata,cal,cal);bq=biology_similarity(data,metadata,q,cal)
        switch=fit_radial_switch(radii,amp[cal],groups[cal],bc,
            fit_amp_sd=errors['radial_bandwidth'],coefficient_grid=BIO_COEFFICIENTS)
        weights,gate,_=switch.apply(base,bq,allowed=allowed)
        plans['BIO']=dict(weights=weights,gate=gate,support=np.any(np.stack(bq)>0,axis=(0,2)),
            coefficients=switch.coefficients,selection=switch.selection,
            resource_reason='assay dose unverified and source/query exposure differs')
        for arm,(z,sd) in states.items():
            cc=state_similarity(z[cal],z[cal],scale=sd)
            cq=state_similarity(z[cal],z[q],scale=sd)
            switch=fit_radial_switch(radii,amp[cal],groups[cal],[cc],
                fit_amp_sd=errors['radial_bandwidth'])
            weights,gate,_=switch.apply(base,[cq],allowed=allowed)
            plans[arm]=dict(weights=weights,gate=gate,support=(cq>0).any(1),
                coefficients=switch.coefficients,selection=switch.selection)
        outputs={};budget=int(np.floor(.25*len(q)))//2
        for arm in ARMS:
            plan=plans[arm];changed=np.any(plan['weights']!=base,axis=1)
            if arm!='CORE' and not changed.any():
                scores={k:v.copy() for k,v in outputs['CORE'].items()}
            else:
                scores=score(mean[q],errors['scatter'],target[q],stats,actual[q],obs[q],
                    absolute[q],norm2[q],SEED+9000+100*fold,law=errors['law'],
                    weights=plan['weights'],samples=100000)
            if arm!='CORE':
                for key in scores:
                    np.testing.assert_array_equal(scores[key][~changed],outputs['CORE'][key][~changed])
            outputs[arm]=scores
            chosen=select_frozen_cohort_plan(ids[q],scores['predicted'],scores['p_null'],budget)
            scores.update(selected=np.asarray(chosen.selected_mask,int),resource_support=plan['support'],
                effective_mixing=plan['gate'],changed_weights=changed)
            for key,value in scores.items():
                if key not in stores[arm]:stores[arm][key]=np.empty((n,*value.shape[1:]),dtype=value.dtype)
                stores[arm][key][local]=value
            print(f'LKCP fold={fold} arm={arm} active={int(changed.sum())} elapsed={time.monotonic()-started:.1f}',flush=True)
        cell=dict(fold=fold,half=0,budget=budget,query_ids=ids[q].tolist(),
            model_fit_ids=ids[fit].tolist(),covfit_ids=ids[rows['covfit']].tolist(),
            calibration_ids=ids[cal].tolist(),covariance_choice=errors['choice'],
            amplitude_fit=errors['amplitude'],modules={a:{k:v for k,v in plan.items() if k!='weights'}
                for a,plan in plans.items()})
        cells.append(cell);seen[local]+=1;folds[local]=fold
        write_json(root/f'cell_{fold}.json',cell)
        np.savez_compressed(root/f'cell_{fold}_weights.npz',query_ids=ids[q],reference_ids=ids[cal],
            **{a:p['weights'] for a,p in plans.items()})
        write_json(root/'status.json',dict(state='RUNNING',cells_complete=len(cells),
            elapsed_seconds=time.monotonic()-started))
    if not np.all(seen==1):raise ValueError('Each query must appear once')
    # Three physical layout families, not individual positions mislabelled as
    # independent layouts. Three-cluster resampling is descriptive only.
    layout=np.array([u['layout'] for u in metadata['units'][n0:]])
    metrics,comparisons,refs=summarize(stores,query_ids,actual[n0:],groups[n0:],layout,folds,cells)
    for arm,out in stores.items():
        out.update(mean_u=mean[n0:].copy(),actual_u=target[n0:].copy())
        np.savez_compressed(root/(arm+'.npz'),ids=query_ids,groups=groups[n0:],layout=layout,
            fold=folds,actual=actual[n0:],**out)
    result=dict(state='COMPLETE',n=n,chemical_groups=len(np.unique(groups[n0:])),samples=100000,
        epochs_max=epochs,cells=cells,metrics=metrics,comparisons=comparisons,references=refs,
        source_condition='A54924h',query_condition='A54948h',core_fold=0,
        core_run=str(state_run),endpoint_changed=False,mean_changed=False,formal_certificate=False,
        unknown_dose_blocks_biology=True,scope='MODULE_DEV condition transfer with pre-existing core',
        original_core_overlap_possible=True,reference_cost_included=False,
        module_scope='radial reference retrieval, not residual direction learning',
        elapsed_seconds=time.monotonic()-started)
    write_json(root/'summary.json',result)
    write_json(root/'status.json',dict(state='COMPLETE',cells_complete=5,elapsed_seconds=time.monotonic()-started))
    print('LKCP COMPLETE',result['elapsed_seconds'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--state-run',type=Path,default=PROJECT/'runs/lincs_state_biology_20260916_v1')
    args=p.parse_args();torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(args.data,args.output,args.state_run)
