"""Full five-fold shared-dose response modeling with signed REF programs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from threadpoolctl import threadpool_limits

from opal2.crossdose_response import fit_ridge_response, fit_convex_strength, apply_response_correction, profile_scores
from opal2.jointdose_program_response import (
    assert_role_isolation, fit_weighted_basis, jointdose_features, fit_component_strengths,
    apply_program_correction, matched_random_reference_weights, compact_reference_weights,
)
from opal2.reference_information_memory import cosine_relationship, morphology_similarity, normalized_topk_weights
from opal2.biology_random_reference_experiment import amplitude_bins
from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata
from opal2.biology_kernel_evaluation import write_json

ROOT = PROJECT/'runs/r3_signed_program_dose_20260921_v1/jointdose'
REPORT = PROJECT/'reports/r3_signed_program_dose_20260921_v1/jointdose'
OLD = PROJECT/'runs/r3_crossdose_response_20260920_v1'
PAIRING = PROJECT/'reports/r3_crossdose_response_20260920_v1/qualification.npz'
DATA = PROJECT/'data/rxrx3_r2_20260918/prepared_r2/data.npz'
SEED = 2026092104
TASKS = ('SAME','CROSS')
METRICS = ('profile_mse','cosine_loss','lognorm_squared_error')
FAMILIES = ('BIO_CONTEXT','TARGET','MORPH')
REPLICATES = 20


def load_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {k:file[k] for k in file.files}


def arms():
    result = ['RIDGE_RESPONSE','SHARED_SOURCE_RAW','SHARED_SOURCE_CAL','JOINTDOSE_RAW','GENERIC']
    for family in FAMILIES:
        result += [family+'_'+suffix for suffix in ('FULL_CAL','COMPONENT_CAL','PROJECTED_CAL','FULL_A025','COMPONENT_A025')]
        for rep in range(1,REPLICATES+1):
            result += [family+f'_R{rep:02d}_'+suffix for suffix in ('FULL_CAL','COMPONENT_CAL')]
    return result


CONFIG = dict(method='shared polynomial-dose response Ridge plus signed reference programs',
    tasks=list(TASKS), metrics=list(METRICS), n_folds=5, n_pairs=7, n_pair_rows=8935,
    seed=SEED, input_dimension=951, source_pca_rank=64, source_pca_standardized=True,
    amplitude_input='log Euclidean norm of observed source X, retained explicitly',
    dose_basis='1,u,u^2,u^3; u=2*log(d/.0025)/log(10/.0025)-1',
    dose_scope='seven existing adjacent pairs only; not arbitrary-dose extrapolation',
    source_only_features=66, jointdose_features=264,
    ridge_lambdas=[.001,.01,.1,1.,10.,100.], ridge_penalty='n_train*lambda',
    group_equal='TRAIN moments and coefficients; VALID lambda; CAL strengths; query summaries',
    output_basis='rank16 orthonormal PCA of TRAIN residuals of RAW joint-dose generic; not CAL fitted',
    baseline='GENERIC: CAL convex blend of saved per-pair RIDGE_RESPONSE and RAW joint-dose model, frozen before biology',
    source_baseline='same source PCA64+lognorm+intercept without dose interactions, RAW and CAL',
    families=list(FAMILIES), bio_weights='target cosine times positive source-morphology cosine',
    target_weights='target cosine only', morphology_weights='positive source-morphology cosine',
    topk=16, reference='REF only, one lexicographic source identity per chemical group and dose pair',
    legal_reference='same adjacent-dose task, different chemistry group; fixed assay context',
    random_replicates=REPLICATES,
    random_matching='same weight multiset within TRAIN source-amplitude quintile x query-donor same-source-plate status; annotation required for biological pools',
    strengths='real: full-vector scalar CAL, projected scalar CAL, component16 CAL, full and projected fixed .25; random: independent full scalar CAL and component16 CAL',
    cal_rule='analytic bounded [0,1] supported-group-equal MSE, zero for <3groups or paired gain below one SE',
    original_core_changed=False, protected_measurements_opened=False, gamma_endpoint_tested=False,
    zero_norm='epsilon=1e-12; cosine zero(loss1) if either norm<=epsilon; counts saved',
    arm_names=arms())


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'CONFIG.json'
    if path.exists():
        if json.loads(path.read_text()) != CONFIG:
            raise ValueError('Existing joint-dose configuration changed')
    else:
        write_json(path,CONFIG); write_json(REPORT/'CONFIG.json',CONFIG)
        shutil.copy2(__file__,ROOT/Path(__file__).name)
        shutil.copy2(PROJECT/'opal2/jointdose_program_response.py',ROOT/'jointdose_program_response.py')
    comparisons=[['JOINTDOSE_RAW','SHARED_SOURCE_RAW'],['GENERIC','SHARED_SOURCE_CAL'],
        ['JOINTDOSE_RAW','RIDGE_RESPONSE'],['GENERIC','RIDGE_RESPONSE']]
    for family in FAMILIES:
        for suffix in ('FULL_CAL','COMPONENT_CAL','PROJECTED_CAL','FULL_A025','COMPONENT_A025'):
            comparisons.append([family+'_'+suffix,'GENERIC'])
        for suffix in ('FULL_CAL','COMPONENT_CAL'):
            comparisons.append([family+'_'+suffix,family+'_RANDOM_MEAN_'+suffix])
            if family!='MORPH':comparisons.append([family+'_'+suffix,'MORPH_'+suffix])
        comparisons += [[family+'_COMPONENT_CAL',family+'_FULL_CAL'],
            [family+'_COMPONENT_CAL',family+'_PROJECTED_CAL']]
    comparisons += [['BIO_CONTEXT_COMPONENT_CAL','TARGET_COMPONENT_CAL'],
        ['BIO_CONTEXT_FULL_CAL','TARGET_FULL_CAL']]
    write_json(ROOT/'comparisons.json',comparisons)


def representative_rows(rows,pairing):
    picked={}
    for row in sorted(rows,key=lambda i:(int(pairing['task_index'][i]),str(pairing['object_ids'][i]))):
        picked.setdefault((int(pairing['task_index'][row]),str(pairing['groups'][row])),int(row))
    return np.asarray(list(picked.values()),int)


def old_predictions(fold, task, x, pairing, needed):
    result=np.full_like(x,np.nan)
    for pair in range(7):
        rows=np.flatnonzero((pairing['task_index']==pair)&needed)
        model=load_npz(OLD/f'unit_{fold*7+pair:02d}'/'ridge_models.npz')
        result[rows]=((x[rows]-model[task+'_input_center'])/model[task+'_input_scale'])@model[task+'_coefficient']+model[task+'_target_center']
    return result


def borrow(weights, reference_residual, query_pair, donor_pair):
    """Block multiplication uses the exact same legal weights, not approximation."""
    result=np.zeros((len(weights),reference_residual.shape[1]))
    for pair in range(7):
        a=np.flatnonzero(query_pair==pair); b=np.flatnonzero(donor_pair==pair)
        if len(a) and len(b): result[a]=weights[np.ix_(a,b)]@reference_residual[b]
    return result


def score(prediction,target):
    result=profile_scores(prediction,target)
    values=np.stack([result[k] for k in METRICS],axis=1)
    audit=dict(prediction_zero_norm=int(result['prediction_zero_norm'].sum()),target_zero_norm=int(result['target_zero_norm'].sum()))
    if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite profile score')
    return values,audit


def run_fold(fold,data,pairing,biology):
    folder=ROOT/f'fold_{fold}'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): return json.loads((folder/'complete.json').read_text())
    started=time.monotonic()
    def status(stage):
        write_json(folder/'status.json',dict(state='RUNNING',fold=fold,stage=stage,elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
        print(json.dumps(dict(fold=fold,stage=stage,seconds=time.monotonic()-started)),flush=True)
    groups=pairing['groups']; roles=pairing['outer_roles'][:,fold]
    assert_role_isolation(groups,roles)
    parts={role:np.flatnonzero(roles==role) for role in ('TRAIN','VALIDATION','REF_FIT','DIST_CAL','DEV_EVAL')}
    parts['REF_FIT']=representative_rows(parts['REF_FIT'],pairing)
    t,v,r,c,q=(parts[role] for role in ('TRAIN','VALIDATION','REF_FIT','DIST_CAL','DEV_EVAL'))
    source,target=pairing['source_rows'],pairing['target_rows']
    x=data['Y'][source,0].astype(float)
    ys=(data['Y'][source[:,None],pairing['source_roles']].mean(1),
        data['Y'][target[:,None],pairing['target_roles']].mean(1))
    status('TRAIN source PCA64')
    source_basis=fit_weighted_basis(x[t],groups[t],rank=64,standardize=True)
    features=jointdose_features(x,pairing['source_dose'],source_basis)
    plain=jointdose_features(x,pairing['source_dose'],source_basis,interacting=False)
    np.savez_compressed(folder/'source_basis.npz',**source_basis.arrays())
    cq=np.r_[c,q]; nc=len(c)
    need=np.zeros(len(x),bool); need[np.r_[r,c,q]]=True
    amplitude=np.log(np.maximum(np.linalg.norm(x,axis=1),1e-12))
    donor_bins,edges=amplitude_bins(amplitude[t],amplitude[r])
    query_pair=pairing['task_index'][cq]; donor_pair=pairing['task_index'][r]
    legal=(groups[cq,None]!=groups[None,r])&(query_pair[:,None]==donor_pair[None])
    source_plate=data['plates'][source,0]
    ann=biology['arrays']['target'][source]; ann_mask=biology['arrays']['target_mask'][source]
    target_sim=cosine_relationship(ann[cq],ann[r],ann_mask[cq],ann_mask[r])
    morph_sim=morphology_similarity(x[cq],x[r])
    plans={}
    for family,sim in zip(FAMILIES,(target_sim*morph_sim,target_sim,morph_sim)):
        retrieved=normalized_topk_weights(sim,donor_ids=pairing['source_ids'][r],top_k=16,
            eligible=legal,query_groups=groups[cq],donor_groups=groups[r])
        w,support=retrieved['weights'],retrieved['support']; w[~support]=0
        allowed=legal&(ann_mask[r][None] if family!='MORPH' else True)
        index,value=compact_reference_weights(w)
        plans[family]=dict(weights=w,support=support,random_legal=allowed,
            sparse_indices=[index],sparse_values=[value],random_audit=[])
    names=arms(); name_index={name:i for i,name in enumerate(names)}
    scores=np.empty((2,len(q),len(names),3))
    cal_reports={}; ridge_reports={}; zero_reports={}; saved={}; fitted_arrays={}
    for ti,(task,y) in enumerate(zip(TASKS,ys)):
        status(task+' shared source and dose ridge fitting')
        old=old_predictions(fold,task,x,pairing,need)
        raw_model=fit_ridge_response(features[t],y[t],features[v],y[v],training_groups=groups[t],validation_groups=groups[v])
        source_model=fit_ridge_response(plain[t],y[t],plain[v],y[v],training_groups=groups[t],validation_groups=groups[v])
        raw=raw_model.predict(features); source_prediction=source_model.predict(plain)
        ones=np.ones(len(c),bool)
        outer=fit_convex_strength(old[c],y[c],raw[c],groups[c],ones)
        source_outer=fit_convex_strength(old[c],y[c],source_prediction[c],groups[c],ones)
        generic=old.copy(); generic[need]=apply_response_correction(old[need],raw[need],np.ones(need.sum(),bool),outer['alpha'])
        source_cal=apply_response_correction(old[q],source_prediction[q],np.ones(len(q),bool),source_outer['alpha'])
        # TRAIN-only program basis remains independent of the CAL blend strength.
        response_basis=fit_weighted_basis(y[t]-raw[t],groups[t],rank=16,standardize=False)
        np.savez_compressed(folder/(task+'_response_basis.npz'),**response_basis.arrays())
        ridge_reports[task]=dict(jointdose=raw_model.report,source_only=source_model.report,
            output_basis_variance_fraction=float(response_basis.eigenvalues.sum()/response_basis.total_variance))
        cal_reports[task]=dict(generic_outer=outer,source_only_outer=source_outer)
        zero_reports[task]={}
        for label,model in [('JOINT',raw_model),('SOURCE',source_model)]:
            for field in ('input_center','input_scale','target_center','coefficient'):
                fitted_arrays[task+'_'+label+'_'+field]=getattr(model,field)
        def record(name,prediction,*,keep=False):
            values,audit=score(prediction,y[q]); scores[ti,:,name_index[name]]=values
            zero_reports[task][name]=audit
            if keep:saved[task+'_'+name]=prediction
        for name,pred in [('RIDGE_RESPONSE',old[q]),('SHARED_SOURCE_RAW',source_prediction[q]),
                          ('SHARED_SOURCE_CAL',source_cal),('JOINTDOSE_RAW',raw[q]),('GENERIC',generic[q])]:
            record(name,pred,keep=True)
        # Verify coefficient reconstruction against authoritative saved float64 scores.
        for pair in range(7):
            previous=load_npz(OLD/f'unit_{fold*7+pair:02d}'/'query_scores.npz')
            rows=np.flatnonzero(pairing['task_index'][q]==pair)
            np.testing.assert_array_equal(pairing['object_ids'][q[rows]],previous['object_ids'])
            np.testing.assert_allclose(scores[ti,rows,name_index['RIDGE_RESPONSE']],previous['scores'][ti,:,0],atol=1e-10,rtol=1e-10)
        residual=y[r]-generic[r]
        target_coeff=response_basis.signed_projection(y[c]-generic[c])
        saved[task+'_actual']=y[q]
        for family in FAMILIES:
            plan=plans[family]; support=plan['support']
            for rep in range(REPLICATES+1):
                if rep in (0,10,20): status(task+' '+family+f' signed reference draw {rep}/20')
                if rep==0: w=plan['weights']; label=family
                else:
                    w,diag=matched_random_reference_weights(plan['weights'],plan['random_legal'],donor_bins,
                        source_plate[cq],source_plate[r],seed=SEED+fold*100000+FAMILIES.index(family)*1000+rep)
                    label=family+f'_R{rep:02d}'
                    if ti==0:
                        idx,val=compact_reference_weights(w); plan['sparse_indices'].append(idx); plan['sparse_values'].append(val)
                        plan['random_audit'].append(dict(draw=rep,retained_mean=float(diag['retained_weight_mass'][support].mean()) if support.any() else 0.,
                            fixed_mean=float(diag['fixed_weight_mass'][support].mean()) if support.any() else 0.))
                delta=borrow(w,residual,query_pair,donor_pair)
                coeff=response_basis.signed_projection(delta)
                fit_full=fit_convex_strength(generic[c],y[c],generic[c]+delta[:nc],groups[c],support[:nc])
                component=fit_component_strengths(target_coeff,coeff[:nc],groups[c],support[:nc])
                full=apply_response_correction(generic[q],generic[q]+delta[nc:],support[nc:],fit_full['alpha'])
                program=apply_program_correction(generic[q],coeff[nc:],response_basis,support[nc:],component['alpha'])
                for pred in (full,program):np.testing.assert_array_equal(pred[~support[nc:]],generic[q][~support[nc:]])
                np.testing.assert_array_equal(apply_program_correction(generic[q],coeff[nc:],response_basis,support[nc:],0.),generic[q])
                record(label+'_FULL_CAL',full,keep=rep==0)
                record(label+'_COMPONENT_CAL',program,keep=rep==0)
                cal_reports[task][label]=dict(full=fit_full,components=component)
                if rep==0:
                    projected=fit_convex_strength(np.zeros_like(target_coeff),target_coeff,coeff[:nc],groups[c],support[:nc])
                    cal_reports[task][label]['projected_scalar']=projected
                    pred=apply_program_correction(generic[q],coeff[nc:],response_basis,support[nc:],projected['alpha'])
                    record(label+'_PROJECTED_CAL',pred,keep=True)
                    record(label+'_FULL_A025',apply_response_correction(generic[q],generic[q]+delta[nc:],support[nc:],.25))
                    record(label+'_COMPONENT_A025',apply_program_correction(generic[q],coeff[nc:],response_basis,support[nc:],.25))
    status('saving shared artifacts')
    np.savez_compressed(folder/'query_scores.npz',scores=scores,arm_names=np.asarray(names),metric_names=np.asarray(METRICS),
        task_names=np.asarray(TASKS),pair_rows=q,groups=groups[q],object_ids=pairing['object_ids'][q],
        source_ids=pairing['source_ids'][q],target_ids=pairing['target_ids'][q],source_dose=pairing['source_dose'][q],target_dose=pairing['target_dose'][q],
        batch=data['batches'][source[q]],source_plate=source_plate[q],layout=data['layout'][source[q]],
        target_support=plans['TARGET']['support'][nc:],bio_context_support=plans['BIO_CONTEXT']['support'][nc:],morph_support=plans['MORPH']['support'][nc:])
    np.savez_compressed(folder/'predictions.npz',source_x=x[q],**saved)
    np.savez_compressed(folder/'ridge_models.npz',**fitted_arrays)
    reference_arrays=dict(cal_rows=c,query_rows=q,reference_rows=r,reference_ids=pairing['source_ids'][r],
        reference_groups=groups[r],reference_pair=donor_pair,query_cal_pair=query_pair,
        amplitude_bins=donor_bins,amplitude_edges=edges,query_source_plate=source_plate[cq],reference_source_plate=source_plate[r])
    supports={}
    for family,plan in plans.items():
        reference_arrays[family+'_indices']=np.stack(plan['sparse_indices'])
        reference_arrays[family+'_values']=np.stack(plan['sparse_values'])
        reference_arrays[family+'_support']=plan['support']
        supports[family]=dict(cal_n=int(plan['support'][:nc].sum()),query_n=int(plan['support'][nc:].sum()),
            cal_groups=len(np.unique(groups[c][plan['support'][:nc]])),query_groups=len(np.unique(groups[q][plan['support'][nc:]])),random=plan['random_audit'])
    np.savez_compressed(folder/'references.npz',**reference_arrays)
    write_json(folder/'calibration.json',cal_reports); write_json(folder/'ridge_fitting.json',ridge_reports)
    write_json(folder/'zero_norm_audit.json',zero_reports)
    report=dict(state='COMPLETE',fold=fold,query_n=len(q),query_groups=len(np.unique(groups[q])),
        role_counts={k:len(val) for k,val in parts.items()},supports=supports,exact_fallback_verified=True,
        saved_legacy_score_reproduction=True,all_scores_finite=bool(np.isfinite(scores).all()),elapsed_seconds=time.monotonic()-started)
    write_json(folder/'complete.json',report); write_json(folder/'status.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folds',nargs='*',type=int,default=list(range(5)));parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();prepare();started=time.monotonic()
    data=load_npz(DATA);pairing=load_npz(PAIRING);biology=load_rxrx3_biology_metadata(data)
    if len(pairing['source_rows'])!=8935:raise ValueError('Paired development population changed')
    write_json(ROOT/'biology_audit.json',biology['report'])
    try:
        with threadpool_limits(limits=args.threads):
            for fold in args.folds:
                report=run_fold(fold,data,pairing,biology)
                write_json(ROOT/'status.json',dict(state='RUNNING',completed_folds=len(list(ROOT.glob('fold_*/complete.json'))),
                    total_folds=5,last_fold=fold,last_fold_seconds=report['elapsed_seconds'],elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
        count=len(list(ROOT.glob('fold_*/complete.json')))
        write_json(ROOT/'status.json',dict(state='NUMERICS_COMPLETE' if count==5 else 'PARTIAL',completed_folds=count,total_folds=5,elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
    except Exception:
        write_json(ROOT/'status.json',dict(state='FAILED',traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started,pid=os.getpid()))
        raise


if __name__=='__main__':main()
