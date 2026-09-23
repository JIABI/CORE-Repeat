"""R2 matched error laws and direct strong baselines; reuse all saved mean fits."""
from __future__ import annotations

import json
import argparse
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.biology_kernel_evaluation import write_json
from opal2.eu_fit_dataset import read_rows
from opal2.eu_core_experiment import partitions, select, policy_summary, extra_seed_moments, SAMPLES, SEED
from opal2.eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from opal2.eu_core_training import predict_eu_core
from opal2.gram_simple_models import GramSimpleGaussian, _fit_error_second_moment
from opal2.gram_oof_ridge import transform_input, transform_target
from opal2.hierarchical_geometry import RidgeResidualMean
from opal2.state_biology_kernel import StateBiologyKernelMean
from opal2.empirical_radial_experiment import score, LEVELS
from opal2.conditional_joint_error_experiment import observable_forward
from opal2.reference_information_diagnostic import bootstrap_difference

PHASE = PROJECT/'reports/eu_core_development_20260917_v1'
DATA = PHASE/'prepared_data_cc904'
SOURCE = PROJECT/'runs/eu_core_cc904_20260917_v1'
ROOT = PROJECT/'runs/r2_core_comparison_20260917_v1'
REPORT = PROJECT/'reports/r2_core_comparison_20260917_v1'
NEW_JOINT = ('CORE_AMP_GAUSSIAN',)+tuple(
    m+s for m in ('RIDGE_REF','HR_REF','STATE_REF')
    for s in ('_LOCAL_GAUSSIAN','_AMP_GAUSSIAN',''))


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def false_activation_outcomes(actual,selected):
    """Numeric false-activation indicators for signed paired comparisons."""
    actual,selected=np.asarray(actual,float),np.asarray(selected)
    if actual.ndim!=1 or selected.shape!=actual.shape or selected.dtype!=bool:
        raise ValueError('Aligned outcomes and a Boolean selection vector required')
    return ((actual<=0)&selected).astype(np.float64)


def load_means(folder):
    ridge = GramSimpleGaussian.load(folder/'ridge.npz')
    saved = torch.load(folder/'HR_fit/best.pt',map_location='cpu',weights_only=True)
    state = saved['state_dict']
    hr = RidgeResidualMean.from_config(saved['model_config'],coefficient=state['coefficient'],intercept=state['intercept']).double()
    hr.load_state_dict(state,strict=True);hr.eval().requires_grad_(False)
    saved = torch.load(folder/'STATE50/epoch50.pt',map_location='cpu',weights_only=True)
    full = StateBiologyKernelMean.from_config(saved['model_config']).double()
    full.load_state_dict(saved['state_dict'],strict=True);full.eval().requires_grad_(False)
    return ridge,hr,full


def summarize(out,actual,ids,folds):
    null = actual<=0
    result = dict(n=len(actual),gamma_mean=float(out['predicted'].mean()),
        gamma_mse=float(np.square(actual-out['predicted']).mean()),
        gamma_spearman=None if np.ptp(out['predicted'])==0 else float(spearmanr(actual,out['predicted']).statistic),
        brier=float(np.square(out['p_null']-null).mean()),
        null_auc=float(roc_auc_score(null,out['p_null'])),policies={})
    for key in ('crps','nll','energy','single_crps','pair_crps','average_crps','triple_average_crps','absolute_pair_crps'):
        if key in out:result[key]=float(out[key].mean())
    if 'mean_u' in out:
        result['geometry_mse']=float(np.square(out['mean_u']-out['actual_u']).mean())
    for key in ('joint_coverage_by_level','gamma_coverage_by_level'):
        if key in out:result[key]=out[key].mean(0)
    if 'joint_coverage_by_level' in out:
        result['joint_mean_absolute_coverage_error_pp']=float(100*np.abs(out['joint_coverage_by_level'].mean(0)-LEVELS).mean())
    for lam in (.2,0.):
        key=f'lambda_{lam:g}';chosen=np.zeros(len(actual),bool)
        for fold in np.unique(folds):
            ix=np.flatnonzero(folds==fold)
            chosen[ix]=select(ids[ix],out['predicted'][ix],out['p_null'][ix],lam)
        out['selected_'+key]=chosen
        result['policies'][key]=dict(policy_summary(actual,chosen),
            predicted_null_count=float(out['p_null'][chosen].sum()),
            null_count_gap=float((null[chosen]-out['p_null'][chosen]).sum()),
            selected_ids=ids[chosen])
    return result


def analysis(stores,data,actual,folds,*,scope_text=None,population_text=None):
    ids,groups,layout=data['ids'],data['groups'],data['layout']
    metrics={name:summarize(out,actual,ids,folds) for name,out in stores.items()}
    null=actual<=0
    comparisons={}
    pairs=[('CORE_ORIGINAL','CORE_LOCAL_GAUSSIAN'),('CORE_AMP_GAUSSIAN','CORE_LOCAL_GAUSSIAN'),
           ('CORE_ORIGINAL','CORE_AMP_GAUSSIAN'),('HR_REF','RIDGE_REF'),('STATE_REF','HR_REF'),
           ('STATE_REF','CORE_ORIGINAL')]
    for mean in ('RIDGE_REF','HR_REF','STATE_REF'):
        pairs += [(mean+'_AMP_GAUSSIAN',mean+'_LOCAL_GAUSSIAN'),(mean,mean+'_AMP_GAUSSIAN')]
    for suffix in ('_LOCAL_GAUSSIAN','_AMP_GAUSSIAN'):
        pairs += [('HR_REF'+suffix,'RIDGE_REF'+suffix),('STATE_REF'+suffix,'HR_REF'+suffix)]
    pairs += [(name,'CORE_ORIGINAL') for name in stores if name.startswith('DIRECT_') or name=='CONSTANT_ACCESS_MATCHED']
    for a,b in pairs:
        aa,bb=stores[a],stores[b]
        values={'gamma_mse':(np.square(aa['predicted']-actual),np.square(bb['predicted']-actual)),
                'brier':(np.square(aa['p_null']-null),np.square(bb['p_null']-null)),
                'policy_value_per_candidate':(actual*aa['selected_lambda_0.2'],actual*bb['selected_lambda_0.2']),
                'false_activation_per_candidate':(false_activation_outcomes(actual,aa['selected_lambda_0.2']),
                                                   false_activation_outcomes(actual,bb['selected_lambda_0.2']))}
        for key in ('crps','nll','energy'):
            if key in aa and key in bb:values[key]=(aa[key],bb[key])
        if 'mean_u' in aa and 'mean_u' in bb:
            values['geometry_mse']=(np.square(aa['mean_u']-aa['actual_u']).mean(1),np.square(bb['mean_u']-bb['actual_u']).mean(1))
        comparisons[a+' minus '+b]={k:{scope:bootstrap_difference(x,y,labels) for scope,labels in
            (('chemical_identity',groups),('layout',layout))} for k,(x,y) in values.items()}
    # Fixed model and selected-set diagnostics, not a new calibration fit.
    calibration={}
    base_selected=stores['CORE_ORIGINAL']['selected_lambda_0.2']
    for name,out in stores.items():
        cells=[]
        for scope,take in [('all',np.ones(len(ids),bool)),('CORE_selected',base_selected),
                          ('own_selected',out['selected_lambda_0.2'])]:
            gap=(null-out['p_null'])*take
            cells.append(dict(scope=scope,n=int(take.sum()),predicted=float(out['p_null'][take].sum()),
                actual=int(null[take].sum()),brier=float(np.square(out['p_null'][take]-null[take]).mean()),
                calibration_gap_per_candidate={label:bootstrap_difference(gap,np.zeros(len(gap)),blocks)
                    for label,blocks in (('chemical_identity',groups),('layout',layout))},
                gap_interpretation='actual minus expected NULL, normalized by all candidates; fixed selected set'))
        rankbands=[]
        for lower,upper in ((0,.125),(.125,.25),(.25,.5),(.5,1.)):
            take=np.zeros(len(ids),bool)
            for f in np.unique(folds):
                ix=np.flatnonzero(folds==f)
                order=ix[np.lexsort((ids[ix],-(out['predicted'][ix]-.2*out['p_null'][ix])))]
                take[order[int(lower*len(ix)):int(upper*len(ix))]]=True
            rankbands.append(dict(lower=lower,upper=upper,n=int(take.sum()),mean_p=float(out['p_null'][take].mean()),actual_rate=float(null[take].mean())))
        calibration[name]=dict(regions=cells,score_rank_bands=rankbands,
            by_layout=[dict(layout=str(g),n=int(np.sum(layout==g)),
                selected=int(np.sum(out['selected_lambda_0.2']&(layout==g))),
                predicted_selected_null=float(out['p_null'][out['selected_lambda_0.2']&(layout==g)].sum()),
                actual_selected_null=int(null[out['selected_lambda_0.2']&(layout==g)].sum())) for g in np.unique(layout)],
            per_fold=[dict(fold=int(f),n=int(np.sum(folds==f)),**{k:float(v[folds==f].mean()) for k,v in
                {'p_null':out['p_null'],'actual_null':null,'gamma_mse':np.square(out['predicted']-actual)}.items()}) for f in np.unique(folds)])
    monte_carlo={}
    for name in ('CORE_ORIGINAL','CORE_LOCAL_GAUSSIAN',*NEW_JOINT):
        source_name={'CORE_ORIGINAL':'AMP_EMP_LOCAL','CORE_LOCAL_GAUSSIAN':'GAUSSIAN'}.get(name,name)
        source_root=SOURCE if name in ('CORE_ORIGINAL','CORE_LOCAL_GAUSSIAN') else ROOT
        records=[]
        for offset in (100000,200000):
            moment={k:np.empty(len(ids)) for k in ('predicted','p_null')}
            for f in np.unique(folds):
                ix=np.flatnonzero(folds==f)
                saved=read_npz(source_root/f'fold_{f}'/f'{source_name}_mc{offset}.npz')
                np.testing.assert_array_equal(saved['ids'],ids[ix])
                for k in moment:moment[k][ix]=saved[k]
            policies=summarize(moment,actual,ids,folds)['policies']
            for lam in (.2,0.):
                key=f'lambda_{lam:g}'
                policies[key]['list_symmetric_difference']=int(np.sum(moment['selected_'+key]!=stores[name]['selected_'+key]))
            records.append(dict(seed_offset=offset,policies=policies))
        monte_carlo[name]=records
    costs=[];fixed_controls=[]
    for f in np.unique(folds):
        source=json.loads((SOURCE/f'fold_{f}'/'summary.json').read_text())
        fixed_controls.append(dict(fold=int(f),random_same_budget=source['random_same_budget'],fixed=source['fixed']))
        cost=source['costs'];nq=int(np.sum(folds==f))
        rows={}
        for name,out in stores.items():
            sel=out['selected_lambda_0.2']&(folds==f);net=float(actual[sel].sum())
            uses_reference=not name.startswith('DIRECT_TRAIN_MATCHED')
            new=cost['reference_if_all_new_wells'] if uses_reference else 0
            missing=cost['reference_if_X_already_available'] if uses_reference else 0
            value_per_query=net/nq
            rows[name]=dict(action_net_value=net,reference_new_wells=new,
                existing_reference_total_net_value=net,new_reference_total_net_value=net-.01*new,
                reference_X_already_measured_total_net_value=net-.01*missing,
                amortization_queries_if_value_persists=None if value_per_query<=0 else float(.01*new/value_per_query))
        costs.append(dict(fold=int(f),query_n=nq,resource_counts=cost,arms=rows))
    payload=dict(complete=True,n=len(ids),metrics=metrics,paired=comparisons,calibration=calibration,
        monte_carlo_sensitivity=monte_carlo,reference_costs=costs,cached_fixed_controls=fixed_controls,
        scope=scope_text or 'opened EU FIT development, fixed fitted models and selected-set paired diagnostics',
        confirmation_opened=False,formal_certificate=False,levels=LEVELS)
    write_json(ROOT/'summary.json',payload)
    for name,out in stores.items():
        np.savez_compressed(ROOT/(name+'.npz'),ids=ids,groups=groups,layout=layout,fold=folds,actual=actual,**out)
    lines=['# R2 full comparison: completed','',
        population_text or 'Opened EU FIT, 904 objects. No CORE mean retraining, no confirmation data.','',
        '| Arm | Γ CRPS | NULL Brier | NULL AUC | Γ rho | selected NULL | selected mean Γ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        p=m['policies']['lambda_0.2']
        lines.append(f"| {name} | {m.get('crps',float('nan')):.6f} | {m['brier']:.6f} | {m['null_auc']:.4f} | {m['gamma_spearman']} | {p['null_selected']} | {p['selected_mean_value']:.6f} |")
    lines+=['','Direct scalar distributions do not provide nine-dimensional joint predictions; their NLL and joint coverage are not fabricated.',
        'CLASSIFIER_RAW/CAL strategies use separate classifier probabilities; their repeated Gamma CRPS/coverage values belong to the same regression-residual Gamma distribution, not to the classifier itself.',
        'All new query predictions, fitted models, matched residual laws and selection lists are saved for R2/R4 reuse.',
        'This is a development comparison, not independent policy certification.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run():
    from opal2.eu_r2_direct_baselines import fit_direct_baselines, evaluate_direct_distribution
    if (ROOT/'status.json').exists():
        old=json.loads((ROOT/'status.json').read_text())
        if old['state']=='COMPLETE':print(json.dumps(old));return
        raise RuntimeError('An existing partial run is preserved; inspect before resume')
    started=time.monotonic();ROOT.mkdir(parents=True,exist_ok=False)
    def status(stage,**extra):
        write_json(ROOT/'status.json',dict(state='RUNNING',stage=stage,elapsed_seconds=time.monotonic()-started,**extra))
    try:
        data=read_npz(DATA/'data.npz');metadata=json.loads((DATA/'metadata.json').read_text())
        if metadata['confirmation_data_loaded'] or len(data['ids'])!=904:raise ValueError('Opened scope changed')
        plans=read_rows(PHASE/'identity_split_plan.csv')
        parts=[partitions(data['ids'],data['groups'],plans,f,excluded_ids=metadata['excluded_incomplete_ids']) for f in range(5)]
        r1=read_npz(PROJECT/'runs/r1_completion_20260917_v1/observables.npz')
        previous=read_npz(SOURCE/'AMP_EMP_LOCAL.npz');gauss=read_npz(SOURCE/'GAUSSIAN.npz')
        np.testing.assert_array_equal(previous['ids'],data['ids'])
        np.testing.assert_array_equal(gauss['ids'],data['ids'])
        np.testing.assert_array_equal(r1['ids'],data['ids'])
        np.testing.assert_array_equal(gauss['actual'],previous['actual'])
        np.testing.assert_array_equal(gauss['fold'],previous['fold'])
        np.testing.assert_array_equal(previous['groups'],data['groups'])
        np.testing.assert_array_equal(previous['layout'],data['layout'])
        identity_keys={'ids','groups','layout','fold','actual'}
        stores={'CORE_ORIGINAL':{k:v for k,v in previous.items() if k not in identity_keys},
                'CORE_LOCAL_GAUSSIAN':{k:v for k,v in gauss.items() if k not in identity_keys}}
        actual,folds=previous['actual'],previous['fold']
        write_json(ROOT/'run_manifest.json',dict(protocol=str(REPORT/'PROTOCOL.md'),previous=str(SOURCE),data=str(DATA),
            n=904,folds=5,samples=SAMPLES,new_joint_arms=NEW_JOINT,original_CORE_retrained=False,
            original_CORE_samples_repeated=False,biology=False,representation=False,confirmation_opened=False,
            cpu_threads=1,main_lambda=.2,extra_mc_seed_offsets=[100000,200000]))
        (ROOT/'PROTOCOL.md').write_text((REPORT/'PROTOCOL.md').read_text())
        def collect(name,q,out):
            if name not in stores:stores[name]={}
            for k,v in out.items():
                v=np.asarray(v)
                if v.shape[0]!=len(q):raise ValueError('Per-object array does not align')
                if k not in stores[name]:stores[name][k]=np.empty((len(actual),*v.shape[1:]),dtype=v.dtype)
                stores[name][k][q]=v
        for f,part in enumerate(parts):
            status('fold_setup',fold=f,completed_folds=f)
            folder=ROOT/f'fold_{f}';folder.mkdir()
            source=SOURCE/f'fold_{f}';meanfolder=source/'mean'
            stats=json.loads((meanfolder/'preprocessing.json').read_text())
            ridge,hr,state=load_means(meanfolder)
            t,v,r,c,q=(part[k] for k in ('TRAIN','VALIDATION','REF_FIT','DIST_CAL','DEV_EVAL'))
            np.testing.assert_array_equal(np.flatnonzero(folds==f),q)
            x=transform_input(data['Y'][:,0],stats)
            # Native target/observables are reused from R1's exact saved Gram.
            from opal2.gram_geometry import gram_to_coordinates
            normalized_gram=r1['gram']/r1['gram'][:,0,0,None,None]
            raw=gram_to_coordinates(torch.as_tensor(normalized_gram,dtype=torch.float64)).numpy()
            target=transform_target(raw,stats)
            np.testing.assert_allclose(target[q],previous['actual_u'][q],atol=1e-11,rtol=1e-11)
            check,obs,difference,_=observable_forward(raw[q]);np.testing.assert_allclose(check,actual[q],atol=1e-10)
            norm2=np.square(data['Y'][q,0].astype(float)).mean(1)
            absolute=np.log1p(difference*norm2[:,None])
            means={'RIDGE_REF':ridge.predict_mean(x)}
            with torch.no_grad():means['HR_REF']=hr(torch.as_tensor(x)).numpy()
            means['STATE_REF']=predict_eu_core(state,stats,data['Y'][:,0],data['chem'],data['chem_mask'])['mean_u']
            np.testing.assert_allclose(means['STATE_REF'][q],previous['mean_u'][q],atol=1e-12,rtol=1e-12)
            def inputs(rows,mean=None):
                obj={k:data[k][rows] for k in ('ids','groups','chem')};obj['X']=data['Y'][rows,0]
                if mean is not None:obj['mean_u']=mean
                return obj
            oldarray=read_npz(source/'distribution_arrays.npz')
            specs={'CORE_AMP_GAUSSIAN':dict(mean=previous['mean_u'][q],scatter=oldarray['query_scatter_u'],law=None,weights=None)}
            for name,mean in means.items():
                residual=target[r]-mean[r]
                base,_,_,audit=_fit_error_second_moment(residual,include_bias=True)
                fitted=fit_eu_distribution(inputs(r),residual,inputs(c),target[c]-mean[c],base,
                    float(np.std(np.log(np.linalg.norm(data['Y'][t,0],axis=1)))),
                    model_training_ids=data['ids'][np.r_[t,v]],model_training_groups=data['groups'][np.r_[t,v]])
                distribution=predict_eu_distribution(fitted,inputs(q,mean[q]))
                specs[name]=dict(mean=mean[q],scatter=distribution['scatter_u'],law=distribution['law'],weights=distribution['radial_weights'])
                specs[name+'_LOCAL_GAUSSIAN']=dict(mean=mean[q],scatter=distribution['base_scatter_u'],law=None,weights=None)
                specs[name+'_AMP_GAUSSIAN']=dict(mean=mean[q],scatter=distribution['scatter_u'],law=None,weights=None)
                joblib.dump(fitted,folder/(name+'_distribution.joblib'))
                np.savez_compressed(folder/(name+'_residuals.npz'),ref_ids=data['ids'][r],cal_ids=data['ids'][c],
                    query_ids=data['ids'][q],ref_residual=residual,cal_residual=target[c]-mean[c],
                    base_scatter=base,query_mean=mean[q],query_scatter=distribution['scatter_u'])
                write_json(folder/(name+'_distribution.json'),dict(report=fitted['report'],base_estimation=audit,
                    base_source='mean-specific held-out REF residual second moment',mean_refitted=False))
            for name,spec in specs.items():
                status('joint_score',fold=f,arm=name,completed_folds=f)
                out=score(spec['mean'],spec['scatter'],target[q],stats,actual[q],obs,absolute,norm2,
                    SEED+f*100,law=spec['law'],weights=spec['weights'],samples=SAMPLES)
                out.update(mean_u=spec['mean'],actual_u=target[q],scatter_u=spec['scatter'],brier=(out['p_null']-(actual[q]<=0))**2)
                collect(name,q,out)
                np.savez_compressed(folder/(name+'.npz'),ids=data['ids'][q],actual=actual[q],**out)
                for offset in (100000,200000):
                    moment=extra_seed_moments(spec['mean'],spec['scatter'],stats,law=spec['law'],weights=spec['weights'],seed=SEED+f*100+offset)
                    np.savez_compressed(folder/f'{name}_mc{offset}.npz',ids=data['ids'][q],**moment)
            # Same full legal X and chemistry, with no prediction-time future input.
            direct_x=np.column_stack((x,data['chem']))
            for scope,training in (('ACCESS_MATCHED',np.r_[t,r]),('TRAIN_MATCHED',t)):
                status('direct_baselines',fold=f,arm=scope,completed_folds=f)
                fitted=fit_direct_baselines(direct_x[training],actual[training],direct_x[v],actual[v],
                    direct_x[c],actual[c],direct_x[q],seed=SEED+f*100)
                for family,arm in fitted.items():
                    joblib.dump(arm['model'],folder/f'DIRECT_{scope}_{family}.joblib')
                    write_json(folder/f'DIRECT_{scope}_{family}.json',dict(metadata=arm['metadata'],
                        train_ids=data['ids'][training],valid_ids=data['ids'][v],cal_ids=data['ids'][c],query_ids=data['ids'][q]))
                    evaluated=evaluate_direct_distribution(arm,actual[q])
                    # Each policy is explicitly identified; none selected using QUERY.
                    common={k:evaluated[k] for k in ('crps','gamma_coverage_by_level')}
                    for suffix,expected,probability in (
                        ('COHERENT',arm['gamma_distribution_mean'],arm['p_null_from_gamma']),
                        ('CLASSIFIER_RAW',arm['predicted'],arm['p_null']),
                        ('CLASSIFIER_CAL',arm['predicted'],arm['p_null_calibrated'])):
                        name=f'DIRECT_{scope}_{family}_{suffix}'
                        collect(name,q,dict(common,predicted=expected,p_null=probability))
                    np.savez_compressed(folder/f'DIRECT_{scope}_{family}_prediction.npz',ids=data['ids'][q],actual=actual[q],
                        **{k:arm[k] for k in ('predicted','p_null','p_null_calibrated','gamma_residuals','gamma_distribution_mean','p_null_from_gamma')},**evaluated)
            # Empirical unconditional scalar distribution: no neural or random model.
            values=np.sort(actual[np.r_[t,r]])
            from opal2.eu_r2_direct_baselines import evaluate_direct_distribution as eval_scalar
            constant=dict(predicted=np.full(len(q),values.mean()),gamma_residuals=values-values.mean(),
                p_null=np.full(len(q),np.mean(values<=0)),p_null_calibrated=np.full(len(q),np.mean(values<=0)))
            evaluated=eval_scalar(constant,actual[q])
            collect('CONSTANT_ACCESS_MATCHED',q,dict(predicted=constant['predicted'],p_null=constant['p_null'],
                crps=evaluated['crps'],gamma_coverage_by_level=evaluated['gamma_coverage_by_level']))
            write_json(folder/'complete.json',dict(fold=f,elapsed_seconds=time.monotonic()-started,completed=True))
            print(f'fold {f+1}/5 complete after {time.monotonic()-started:.1f}s',flush=True)
        status('paired_analysis',completed_folds=5)
        analysis(stores,data,actual,folds)
        write_json(ROOT/'status.json',dict(state='COMPLETE',completed_folds=5,elapsed_seconds=time.monotonic()-started,report=str(ROOT/'REPORT.md')))
    except Exception:
        write_json(ROOT/'status.json',dict(state='FAILED',elapsed_seconds=time.monotonic()-started,traceback=traceback.format_exc()))
        raise


def aggregate_saved():
    """Restore completed fold predictions; no fitting or sampling is called."""
    from opal2.eu_r2_direct_baselines import evaluate_direct_distribution
    old=json.loads((ROOT/'status.json').read_text())
    if old['state']=='COMPLETE':
        print(json.dumps(old));return
    started=time.monotonic()
    failure=ROOT/'aggregation_initial_failure.json'
    if not failure.exists():write_json(failure,old)
    completed=[]
    for fold in range(5):
        record=json.loads((ROOT/f'fold_{fold}'/'complete.json').read_text())
        if not record['completed'] or record['fold']!=fold:raise ValueError('A fold has not completed computation')
        completed.append(record)
    # Only opened identity/layout arrays are needed for aggregation.
    with np.load(DATA/'data.npz',allow_pickle=False) as z:
        data={k:z[k].copy() for k in ('ids','groups','layout')}
    manifest=json.loads((ROOT/'run_manifest.json').read_text())
    if manifest['confirmation_opened'] or manifest['n']!=904 or len(data['ids'])!=904:
        raise ValueError('Scope differs from completed development run')
    metadata=json.loads((DATA/'metadata.json').read_text())
    plans=read_rows(PHASE/'identity_split_plan.csv')
    previous,gauss=read_npz(SOURCE/'AMP_EMP_LOCAL.npz'),read_npz(SOURCE/'GAUSSIAN.npz')
    for cache in (previous,gauss):
        for k in ('ids','groups','layout'):np.testing.assert_array_equal(cache[k],data[k])
    for k in ('actual','fold'):np.testing.assert_array_equal(previous[k],gauss[k])
    actual,folds=previous['actual'],previous['fold']
    identity={'ids','groups','layout','fold','actual'}
    stores={'CORE_ORIGINAL':{k:v for k,v in previous.items() if k not in identity},
            'CORE_LOCAL_GAUSSIAN':{k:v for k,v in gauss.items() if k not in identity}}
    counts={};audit=[]
    def collect(name,q,out):
        if name not in stores:stores[name]={};counts[name]=np.zeros(len(actual),int)
        counts[name][q]+=1
        for key,value in out.items():
            value=np.asarray(value)
            if not value.ndim or value.shape[0]!=len(q) or not np.isfinite(value).all():
                raise ValueError('Invalid or incomplete per-object predictions: '+name+'/'+key)
            if key not in stores[name]:stores[name][key]=np.empty((len(actual),*value.shape[1:]),dtype=value.dtype)
            stores[name][key][q]=value
    for fold in range(5):
        part=partitions(data['ids'],data['groups'],plans,fold,excluded_ids=metadata['excluded_incomplete_ids'])
        q=part['DEV_EVAL'];folder=ROOT/f'fold_{fold}'
        np.testing.assert_array_equal(q,np.flatnonzero(folds==fold))
        for name in NEW_JOINT:
            saved=read_npz(folder/(name+'.npz'))
            np.testing.assert_array_equal(saved['ids'],data['ids'][q])
            np.testing.assert_array_equal(saved['actual'],actual[q])
            collect(name,q,{k:v for k,v in saved.items() if k not in ('ids','actual')})
            for offset in (100000,200000):
                mc=read_npz(folder/f'{name}_mc{offset}.npz')
                np.testing.assert_array_equal(mc['ids'],data['ids'][q])
                if not all(np.isfinite(mc[k]).all() for k in ('predicted','p_null')):
                    raise ValueError('Incomplete extra-seed predictions')
        for scope in ('ACCESS_MATCHED','TRAIN_MATCHED'):
            for family in ('RIDGE','EXTRATREES','HISTGB'):
                prefix=f'DIRECT_{scope}_{family}'
                saved=read_npz(folder/(prefix+'_prediction.npz'))
                np.testing.assert_array_equal(saved['ids'],data['ids'][q])
                np.testing.assert_array_equal(saved['actual'],actual[q])
                common={k:saved[k] for k in ('crps','gamma_coverage_by_level')}
                for suffix,mean_key,p_key in (
                    ('COHERENT','gamma_distribution_mean','p_null_from_gamma'),
                    ('CLASSIFIER_RAW','predicted','p_null'),
                    ('CLASSIFIER_CAL','predicted','p_null_calibrated')):
                    collect(prefix+'_'+suffix,q,dict(common,predicted=saved[mean_key],p_null=saved[p_key]))
        # This constant-law score was not cached before the old aggregation
        # failure. Reconstruct it algebraically, without any estimator fitting.
        values=np.sort(actual[np.r_[part['TRAIN'],part['REF_FIT']]])
        constant=dict(predicted=np.full(len(q),values.mean()),gamma_residuals=values-values.mean(),
            p_null=np.full(len(q),np.mean(values<=0)),p_null_calibrated=np.full(len(q),np.mean(values<=0)))
        evaluated=evaluate_direct_distribution(constant,actual[q])
        collect('CONSTANT_ACCESS_MATCHED',q,dict(predicted=constant['predicted'],p_null=constant['p_null'],
            crps=evaluated['crps'],gamma_coverage_by_level=evaluated['gamma_coverage_by_level']))
        audit.append(dict(fold=fold,n_query=len(q),new_joint_prediction_files=len(NEW_JOINT),
            extra_mc_files=2*len(NEW_JOINT),direct_prediction_files=6))
    if any(not np.all(count==1) for count in counts.values()):
        raise ValueError('Each object must occur exactly once per restored arm')
    write_json(ROOT/'aggregation_restore.json',dict(folds=audit,n=len(actual),arms=len(stores),
        mean_models_refitted=0,direct_models_refitted=0,joint_samples_redrawn=0,
        direct_predictions_recomputed=False,constant_law_reconstructed_algebraically=True,
        old_failure_preserved=str(failure),confirmation_opened=False))
    write_json(ROOT/'status.json',dict(state='RUNNING',stage='aggregation_only',completed_folds=5,
        fitting_and_sampling_complete=True,original_compute_seconds=completed[-1]['elapsed_seconds']))
    try:
        analysis(stores,data,actual,folds)
        write_json(ROOT/'status.json',dict(state='COMPLETE',completed_folds=5,
            original_compute_seconds=completed[-1]['elapsed_seconds'],aggregation_seconds=time.monotonic()-started,
            resumed_from_saved_predictions=True,report=str(ROOT/'REPORT.md')))
        print(json.dumps(json.loads((ROOT/'status.json').read_text()),indent=2))
    except Exception:
        write_json(ROOT/'status.json',dict(state='FAILED',stage='aggregation_only',completed_folds=5,
            fitting_and_sampling_complete=True,aggregation_seconds=time.monotonic()-started,
            traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        aggregate_saved() if args.aggregate_only else run()
