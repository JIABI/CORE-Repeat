"""Optional reference switches around the complete frozen LINCS core."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import observable_forward, OBSERVABLES
from .empirical_radial import fit_radial, reference_weights
from .empirical_radial_experiment import score, LEVELS
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .gram_geometry import profiles_to_gram, gram_to_coordinates
from .lincs_biology_experiment import load_data
from .optional_radial_modules import (fit_radial_switch, supported_retrieval,
    state_similarity, BIO_COEFFICIENTS)
from .reference_information_diagnostic import bootstrap_difference
from .reference_information_memory import cosine_relationship

PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('CORE','BIO','PCA_STATE','DIRECT_STATE','CONDITIONAL_STATE')
SEED = 20260916


def read_json(path):
    return json.loads(Path(path).read_text())


def context_mask(metadata, query, donors):
    """Exact declared cell/background/time/dose match, no outcome-based matching."""
    if metadata.get('reference_context_policy') == 'rxrx3_protocol_range_v1':
        # A separately audited nominal protocol range is not an invented exact
        # exposure time. Other datasets retain the original exact-time policy.
        from .rxrx3_r3_biology import rxrx3_context_mask
        return rxrx3_context_mask(metadata, query, donors)
    def key(i):
        unit = metadata['units'][int(i)]
        values = (unit.get('cell_line'), unit.get('exposure_hours_protocol_nominal'),
                  unit.get('actual_dose_uM'))
        if any(v is None for v in values): return None
        return str(values[0]),float(values[1]),float(values[2])
    q, r = [key(i) for i in query],[key(i) for i in donors]
    return np.array([[a is not None and b is not None and a == b for b in r] for a in q],bool)


def biology_similarity(data, metadata, query, donors):
    allowed = context_mask(metadata,query,donors)
    allowed &= data['groups'][query,None] != data['groups'][None,donors]
    output = []
    for key in ('target', 'moa'):
        values, mask = np.asarray(data[key]), np.asarray(data[key+'_mask'])
        if values.ndim == 2 and values.shape[1] == 0:
            if mask.dtype != bool or mask.shape != (len(values),) or mask.any():
                raise ValueError('An empty relation vocabulary must be explicitly missing')
            output.append(np.zeros(allowed.shape, float))
        else:
            output.append(cosine_relationship(values[query], values[donors], mask[query], mask[donors])*allowed)
    return output


def linear_states(data, rows, output):
    """Train-only shape PCA; amplitude remains available through the core."""
    x = data['Y'][:,0]
    norm = np.linalg.norm(x,axis=1)
    if np.any(norm <= 0): raise ValueError('Original zero-norm completion required')
    shape = x/norm[:,None]
    model = PCA(n_components=min(8,len(rows)-1,x.shape[1]),svd_solver='full').fit(shape[rows])
    z = model.transform(shape)
    sd = np.full(z.shape[1],max(float(np.sqrt(np.mean(np.var(z[rows],axis=0)))),1e-8))
    np.savez_compressed(output,components=model.components_,center=model.mean_,
        scale=sd,train_ids=data['ids'][rows],explained_variance=model.explained_variance_)
    return z,sd


def fit_states(data, rows, folder, *, epochs):
    from .conditional_state_representation import fit_representation
    states = {'PCA_STATE': linear_states(data,rows,folder/'pca_state.npz')}
    report = {}
    for arm,kind in (('DIRECT_STATE','direct'),('CONDITIONAL_STATE','conditional_predictive')):
        model = fit_representation(data['Y'][rows],data['groups'][rows],data['ids'][rows],
            kind=kind,seed=SEED+int(folder.name.split('_')[-1]),epochs=epochs)
        model.save(folder/(arm+'.pt'))
        full = np.asarray(model.transform(data['Y'][:,0]),float)
        # Model output explicitly appends amplitude. That coordinate already
        # conditions CORE; only the learned state is the new retrieval input.
        z = full[:,:-1]
        # Preserve relative state strengths; do not inflate near-constant
        # coordinates by independently whitening each learned dimension.
        sd = np.full(z.shape[1],max(float(np.sqrt(np.mean(np.var(z[rows],axis=0)))),1e-8))
        states[arm] = z,sd
        report[arm] = model.report
        write_json(folder/(arm+'_training.json'),model.report)
    return states,report


def summarize(stores, ids, actual, groups, layouts, folds, cells):
    metrics = {}; comparisons = {}
    for arm,out in stores.items():
        out['brier'] = (out['p_null']-(actual <= 0))**2
        out['policy_value'] = out['selected']*actual
        selected = out['selected'].astype(bool)
        metrics[arm] = dict(n=len(actual), selected_n=int(selected.sum()),
            selected_null=int((actual[selected] <= 0).sum()),
            selected_mean=float(actual[selected].mean()),
            actual_null_rate=float((actual <= 0).mean()),
            p_null_mean=float(out['p_null'].mean()),
            null_auc=float(roc_auc_score(actual <= 0,out['p_null'])),
            gamma_spearman=float(spearmanr(actual,out['predicted']).statistic),
            scores={k:float(out[k].mean()) for k in ('nll','energy','crps','brier',
                'single_crps','pair_crps','average_crps','absolute_pair_crps')},
            joint_coverage={str(p):float(out['joint_coverage_by_level'][:,j].mean())
                for j,p in enumerate(LEVELS)},
            eligible_reference_queries=int(out['resource_support'].sum()),
            effective_nonzero_queries=int((out['effective_mixing'] > 0).sum()),
            changed_weight_queries=int(out['changed_weights'].sum()),
            changed_selected_membership=int(np.count_nonzero(out['selected'] != stores['CORE']['selected'])),
            original_mean_preserved=True,
            covariance_multiplier_mean=float(out['radial_variance_multiplier'].mean()))
        if arm != 'CORE':
            comparisons[arm] = {k:{scope:bootstrap_difference(out[k],stores['CORE'][k],labels)
                for scope,labels in (('chemistry',groups),('layout',layouts))}
                for k in ('crps','brier','nll','energy','policy_value')}
    for name,a,b in (('CONDITIONAL_minus_DIRECT','CONDITIONAL_STATE','DIRECT_STATE'),
                     ('CONDITIONAL_minus_PCA','CONDITIONAL_STATE','PCA_STATE')):
        comparisons[name] = {k:{scope:bootstrap_difference(stores[a][k],stores[b][k],labels)
            for scope,labels in (('chemistry',groups),('layout',layouts))}
            for k in ('crps','brier','policy_value')}
    # Descriptive references use identical ten cell budgets; never pick a rule
    # using their observed performances.
    lambda0 = np.zeros(len(ids),bool); random_value = 0.; random_null = 0.
    lookup = {v:i for i,v in enumerate(ids)}
    for cell in cells:
        q = np.array([lookup[v] for v in cell['query_ids']]); k=cell['budget']
        order = np.lexsort((ids[q],-stores['CORE']['predicted'][q]))
        lambda0[q[order[:k]]] = True
        random_value += k*float(actual[q].mean())
        random_null += k*float((actual[q] <= 0).mean())
    total = int(stores['CORE']['selected'].sum())
    refs = dict(core_lambda0=dict(selected_n=int(lambda0.sum()),
        selected_null=int((actual[lambda0] <= 0).sum()),selected_mean=float(actual[lambda0].mean())),
        uniform_random_expectation=dict(selected_n=total,selected_null=random_null,
            selected_mean=random_value/total),
        no_additional_measurements=dict(net_value=0.),
        all_add_two_budget_unmatched=dict(selected_n=len(actual),selected_null=int((actual<=0).sum()),
            selected_mean=float(actual.mean())))
    return metrics,comparisons,refs


def run(source, output, *, epochs=60, samples=100000):
    started = time.monotonic()
    source, root = Path(source).resolve(),Path(output).resolve()
    if root.exists(): raise FileExistsError(root)
    if samples != 100000: raise ValueError('This declared comparison uses 100,000 draws; no shortened result substitution')
    old = read_json(source/'summary.json')
    manifest = read_json(Path(old['reference_run'])/'run_manifest.json')
    data,metadata = load_data(old['data_directory'])
    ids,groups = data['ids'],data['groups']; n=len(ids)
    if n != 1188 or ids.tolist() != manifest['ids']: raise ValueError('Opened LINCS scope changed')
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/MODULE_SWITCHES_PLAN_20260916.md',root/'PROTOCOL.md')
    for name in ('optional_radial_modules.py','module_switch_experiment.py','conditional_state_representation.py'):
        shutil.copy2(PROJECT/'opal2'/name,root/name)
    write_json(root/'status.json',dict(state='RUNNING',phase='preparing',started_unix=time.time()))
    with np.load(source/'AMP_EMP_LOCAL.npz',allow_pickle=False) as z:
        prior={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(prior['ids'],ids)
    layouts=np.array([u['layout_block'] for u in metadata['units']])
    raw=gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    actual,obs,diff,_=observable_forward(raw)
    np.testing.assert_allclose(actual,prior['actual'],atol=1e-12,rtol=1e-12)
    amp=np.log(np.linalg.norm(data['Y'][:,0],axis=1))
    norm2=np.square(data['Y'][:,0]).mean(1)
    absolute=np.log1p(diff*norm2[:,None])
    lookup={v:i for i,v in enumerate(ids)}
    records={r['fold']:r for r in manifest['folds']}
    states_by_fold={}; stores={a:{} for a in ARMS}; cells=[]; seen=np.zeros(n,int)
    for cell in old['cells']:
        fold,half=cell['fold'],cell['half']
        q=np.array([lookup[v] for v in cell['query_ids']])
        cal=np.array([lookup[v] for v in cell['representative_ids']])
        train=np.asarray(records[fold]['fit'],int)
        if set(groups[train]) & set(groups[np.r_[cal,q]]): raise ValueError('Representation group leakage')
        if fold not in states_by_fold:
            folder=root/f'fold_{fold}';folder.mkdir()
            write_json(root/'status.json',dict(state='RUNNING',phase='representation_fit',fold=fold,
                cells_complete=len(cells),elapsed_seconds=time.monotonic()-started))
            states_by_fold[fold],training=fit_states(data,train,folder,epochs=epochs)
        with np.load(source/f'cell_{fold}_{half}_radial.npz',allow_pickle=False) as z:
            ref={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(ref['cal_ids'],ids[cal])
        np.testing.assert_array_equal(ref['query_ids'],ids[q])
        base=ref['local_weights'];r=ref['amplitude_radii']
        law=fit_radial(r)
        stats=read_json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        allowed=groups[q,None] != groups[None,cal]
        plans={'CORE':dict(weights=base.copy(),gate=np.zeros(len(q)),support=np.ones(len(q),bool),selection=None)}
        bio_cal=biology_similarity(data,metadata,cal,cal)
        bio_query=biology_similarity(data,metadata,q,cal)
        switch=fit_radial_switch(r,amp[cal],groups[cal],bio_cal,fit_amp_sd=cell['local_bandwidth'],
            coefficient_grid=BIO_COEFFICIENTS)
        weights,gate,_=switch.apply(base,bio_query,allowed=allowed)
        support=np.any(np.stack(bio_query)>0,axis=(0,2))
        plans['BIO']=dict(weights=weights,gate=gate,support=support,selection=switch.selection,
            coefficients=switch.coefficients,
            support_by_channel={k:supported_retrieval(s,allowed=allowed) for k,s in zip(('target','moa'),bio_query)})
        for arm,(z,sd) in states_by_fold[fold].items():
            cc=state_similarity(z[cal],z[cal],scale=sd)
            cq=state_similarity(z[cal],z[q],scale=sd)
            switch=fit_radial_switch(r,amp[cal],groups[cal],[cc],fit_amp_sd=cell['local_bandwidth'])
            weights,gate,_=switch.apply(base,[cq],allowed=allowed)
            plans[arm]=dict(weights=weights,gate=gate,support=(cq>0).any(1),
                selection=switch.selection,coefficients=switch.coefficients)
        output_scores={}
        for arm in ARMS:
            plan=plans[arm];changed=np.any(plan['weights']!=base,axis=1)
            np.testing.assert_array_equal(plan['weights'][~changed],base[~changed])
            if arm != 'CORE' and not changed.any():
                scores={k:v.copy() for k,v in output_scores['CORE'].items()}
            else:
                scores=score(prior['mean_u'][q],prior['scatter_u'][q],prior['actual_u'][q],stats,
                    actual[q],obs[q],absolute[q],norm2[q],cell['normal_seed'],law=law,
                    weights=plan['weights'],samples=samples)
            if arm != 'CORE':
                for key in scores:
                    np.testing.assert_array_equal(scores[key][~changed],output_scores['CORE'][key][~changed],
                        err_msg=f'{arm} off replay {key}')
            output_scores[arm]=scores
            chosen=select_frozen_cohort_plan(ids[q],scores['predicted'],scores['p_null'],cell['budget'])
            scores.update(selected=np.asarray(chosen.selected_mask,int),resource_support=plan['support'],
                effective_mixing=plan['gate'],changed_weights=changed)
            for key,value in scores.items():
                if key not in stores[arm]: stores[arm][key]=np.empty((n,*value.shape[1:]),dtype=value.dtype)
                stores[arm][key][q]=value
            print(f'fold={fold} half={half} arm={arm} supported={int(plan["support"].sum())} active={int(changed.sum())} elapsed={time.monotonic()-started:.1f}',flush=True)
        record=dict(fold=fold,half=half,budget=cell['budget'],query_ids=ids[q].tolist(),
            calibration_ids=ids[cal].tolist(),model_fit_ids=ids[train].tolist(),
            modules={a:{k:v for k,v in p.items() if k not in ('weights','support_by_channel')}
                for a,p in plans.items()})
        cells.append(record);seen[q]+=1
        write_json(root/f'cell_{fold}_{half}.json',record)
        np.savez_compressed(root/f'cell_{fold}_{half}_weights.npz',
            query_ids=ids[q],reference_ids=ids[cal],**{a:p['weights'] for a,p in plans.items()})
        write_json(root/'status.json',dict(state='RUNNING',phase='joint_evaluation',cells_complete=len(cells),
            elapsed_seconds=time.monotonic()-started))
    if not np.all(seen == 1): raise ValueError('Incomplete/duplicate query coverage')
    for arm,out in stores.items():
        out['mean_u']=prior['mean_u'].copy();out['actual_u']=prior['actual_u'].copy()
    metrics,comparisons,refs=summarize(stores,ids,actual,groups,layouts,prior['fold'],cells)
    for arm,out in stores.items():
        np.savez_compressed(root/(arm+'.npz'),ids=ids,groups=groups,layout=layouts,fold=prior['fold'],
            actual=actual,**out)
    result=dict(state='COMPLETE',n=n,samples=samples,epochs_max=epochs,cells=cells,metrics=metrics,
        comparisons=comparisons,references=refs,source=str(source),elapsed_seconds=time.monotonic()-started,
        endpoint_changed=False,mean_changed=False,query_selection_for_modules=False,
        primary_policy='expected_gamma - .2*p_null',formal_certificate=False,
        scope='existing opened LINCS grouped development; shared layout sensitivity only',
        module_scope='radial reference retrieval; not learned residual directions',
        biology_cross_dataset_activation_validated=False)
    write_json(root/'summary.json',result)
    write_json(root/'status.json',dict(state='COMPLETE',cells_complete=10,elapsed_seconds=time.monotonic()-started))
    print('COMPLETE',time.monotonic()-started,flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=PROJECT/'runs/lincs_empirical_radial_20260916_v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60)
    args=p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2): run(args.source,args.output,epochs=args.epochs)
