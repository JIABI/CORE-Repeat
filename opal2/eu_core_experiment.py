"""Complete EU FIT-only CORE development run; never opens a source archive."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .eu_fit_dataset import read_rows
from .eu_core_training import fit_complete_eu_core, predict_eu_core
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_oof_ridge import transform_target, transform_input
from .empirical_radial_experiment import score, LEVELS
from .empirical_radial import draw_radial
from .conditional_joint_error_experiment import observable_forward
from .reference_information_diagnostic import gamma_forward, bootstrap_difference


SAMPLES = 100000
SEED = 20260917
ARMS = ('GAUSSIAN', 'AMP_EMP_LOCAL')


def partitions(ids, groups, plan_rows, fold, *, excluded_ids=()):
    lookup = {str(v): i for i,v in enumerate(ids)}
    if len(lookup) != len(ids): raise ValueError('Duplicate dataset identities')
    rows = [r for r in plan_rows if int(r['outer_fold']) == fold]
    excluded_ids = set(excluded_ids)
    if excluded_ids:
        if excluded_ids & set(lookup) or {r['object_id'] for r in rows} != set(lookup) | excluded_ids:
            raise ValueError('Authorized exclusion does not preserve the original population')
        rows = [r for r in rows if r['object_id'] not in excluded_ids]
    if len(rows) != len(ids) or {r['object_id'] for r in rows} != set(lookup):
        raise ValueError('Every planned identity must be present; incomplete objects cannot be silently removed')
    out = {k: [] for k in ('TRAIN','VALIDATION','REF_FIT','DIST_CAL','DEV_EVAL')}
    for row in rows:
        i = lookup[row['object_id']]
        if str(groups[i]) != row['connectivity']:
            raise ValueError('Dataset chemical group differs from frozen split')
        key = row['model_fit_subrole'] if row['phase_role'] == 'MODEL_FIT' else row['phase_role']
        if key not in out: raise ValueError('Unknown role')
        out[key].append(i)
    if any(not rows for rows in out.values()): raise ValueError('Every role requires objects')
    sets = [set(groups[rows]) for rows in out.values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError('Chemical group crosses roles')
    return {k: np.asarray(sorted(v), int) for k,v in out.items()}


def select(ids, expected, probability, lam):
    ids, expected, probability = np.asarray(ids), np.asarray(expected), np.asarray(probability)
    if expected.shape != ids.shape or probability.shape != ids.shape or not np.isfinite(expected).all():
        raise ValueError('Aligned finite predictions required')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid NULL probability')
    budget = int(np.floor(.25*len(ids)))
    k = min(len(ids), budget//2)
    chosen = np.zeros(len(ids), bool)
    chosen[np.lexsort((ids, -(expected-lam*probability)))[:k]] = True
    return chosen


def policy_summary(actual, chosen):
    null = np.asarray(actual) <= 0
    n = int(chosen.sum()); total_null = int(null.sum())
    false = int((chosen & null).sum())
    return dict(activated=n, extra_wells=2*n, null_selected=false,
        selected_mean_value=float(np.mean(actual[chosen])) if n else None,
        total_value=float(np.sum(actual[chosen])), value_per_candidate=float(np.sum(actual[chosen])/len(actual)),
        fdp=false/n if n else None, false_activation=false/total_null if total_null else None,
        no_binomial_confidence_bound=True)


def metric_summary(out, actual):
    null = actual <= 0
    row = {k: float(np.mean(out[k])) for k in ('nll','energy','crps','coordinate_coverage','joint_coverage',
        'single_crps','pair_crps','average_crps','triple_average_crps','absolute_pair_crps','coverage')}
    row.update(brier=float(np.mean((out['p_null']-null)**2)),
        null_auc=float(roc_auc_score(null, out['p_null'])) if len(set(null)) == 2 else None,
        gamma_spearman=float(spearmanr(actual, out['predicted']).statistic),
        predicted_gamma_mean=float(out['predicted'].mean()), actual_gamma_mean=float(actual.mean()),
        realized_null_rate=float(null.mean()), joint_coverage_by_level=out['joint_coverage_by_level'].mean(0),
        gamma_coverage_by_level=out['gamma_coverage_by_level'].mean(0), levels=LEVELS)
    return row


def extra_seed_moments(mean, scatter, stats, *, law, weights, seed):
    """Same whole-cohort integration budget for decision-only MC sensitivity."""
    n,d = mean.shape
    predicted, probability = np.empty(n), np.empty(n)
    normal_rng, radius_rng = np.random.default_rng(seed), np.random.default_rng(seed+47000)
    chol = np.linalg.cholesky(scatter)
    for begin in range(0,n,16):
        end = min(begin+16,n)
        normal = normal_rng.normal(size=(SAMPLES,end-begin,d))
        if law is None:
            eps = np.einsum('nij,snj->sni',chol[begin:end],normal)
        else:
            eps = draw_radial(law, weights[begin:end], scatter[begin:end], normal,
                radius_rng.random((SAMPLES,end-begin)), radius_rng.random((SAMPLES,end-begin)))
        raw = (mean[None,begin:end]+eps)*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
        gamma = gamma_forward(raw)
        predicted[begin:end], probability[begin:end] = gamma.mean(0), (gamma<=0).mean(0)
    return dict(predicted=predicted,p_null=probability)


def run(dataset, phase, output):
    # Distribution adapter is a separate, testable complete recipe.
    from .eu_core_distribution import fit_eu_distribution, predict_eu_distribution
    dataset,phase,root = map(lambda p:Path(p).resolve(),(dataset,phase,output))
    if root.exists(): raise FileExistsError(root)
    metadata = json.loads((dataset/'metadata.json').read_text())
    manifest = json.loads((phase/'phase_manifest.json').read_text())
    if metadata['confirmation_data_loaded'] or metadata['biology_active'] or metadata['representation_active']:
        raise ValueError('This run requires the declared FIT-only CORE baseline')
    with np.load(dataset/'data.npz', allow_pickle=False) as z:
        data = {k:z[k].copy() for k in z.files}
    ids, groups = data['ids'], data['groups']
    excluded = set(metadata.get('excluded_incomplete_ids',[]))
    if excluded:
        policy=json.loads((dataset/'complete_case_policy.json').read_text())
        if (policy['mode']!='development_complete_cases_keep_original_splits'
                or set(policy['excluded_compound_ids'])!=excluded or len(excluded)!=7 or len(ids)!=904):
            raise ValueError('Unexpected complete-case population or authorization')
    if (len(ids)+len(excluded)!=911 or set(ids)&excluded
            or set(ids)|excluded != set(manifest['allowed_compound_ids']) or not np.isfinite(data['Y']).all()):
        raise ValueError('Incomplete or expanded FIT-only dataset')
    plans = read_rows(phase/'identity_split_plan.csv')
    split = [partitions(ids,groups,plans,i,excluded_ids=excluded) for i in range(5)]
    qrows = np.concatenate([s['DEV_EVAL'] for s in split])
    if len(qrows)!=len(ids) or len(set(qrows))!=len(ids): raise ValueError('Every object must be queried once')
    root.mkdir(parents=True)
    started = time.monotonic()
    def status(state, **extra):
        write_json(root/'status.json',dict(state=state,elapsed_seconds=time.monotonic()-started,**extra))
    status('RUNNING',stage='mean fitting',completed_folds=0)
    write_json(root/'run_manifest.json',dict(dataset=str(dataset),phase=str(phase),seed=SEED,samples=SAMPLES,
        algorithm='full CORE recipe',mean_stages=['RIDGE','best-HR','A30','STATE50'],folds=5,
        additional_mc_seed_offsets=[100000,200000],query_once=True,confirmation_opened=False,
        original_planned_n=911,analysis_n=len(ids),excluded_incomplete_ids=sorted(excluded),
        original_split_membership_preserved=True,
        formal_certification=False,biology=False,representation=False,
        budget='per declared DEV fold: extra wells floor(.25*N); k=floor(extra_wells/2)',
        score='E[Gamma] - lambda*P(NULL)',lambdas=[.2,0.],tie_rule='ID ascending',
        dependency_note='fixed layout; seven library-plate blocks; bootstrap is diagnostic'))
    stores={a:{} for a in ARMS}; actual_all=np.full(len(ids),np.nan); folds=np.full(len(ids),-1,int)
    cells=[]
    try:
        for fold,part in enumerate(split):
            folder=root/f'fold_{fold}'; folder.mkdir()
            def role(rows,reference=False):
                d={k:data[k][rows] for k in ('ids','groups','chem','chem_mask')}
                d['X' if reference else 'Y']=data['Y'][rows,0] if reference else data['Y'][rows]
                return d
            t,v,r,c,q=(part[k] for k in ('TRAIN','VALIDATION','REF_FIT','DIST_CAL','DEV_EVAL'))
            fitted=fit_complete_eu_core(role(t),role(v),role(r,True),metadata,folder/'mean',seed=SEED+fold*100)
            stats=fitted['stats']
            def predict(rows):
                return predict_eu_core(fitted['model'],stats,data['Y'][rows,0],data['chem'][rows],data['chem_mask'][rows])
            def inputs(rows,mean=None):
                d={k:data[k][rows] for k in ('ids','groups','chem')};d['X']=data['Y'][rows,0]
                if mean is not None:d['mean_u']=mean
                return d
            def outcomes(rows):
                g=profiles_to_gram(torch.as_tensor(data['Y'][rows],dtype=torch.float64))
                raw=gram_to_coordinates(g).numpy()
                return raw,transform_target(raw,stats),gram_gains(g).numpy()[:,2]
            pr,pc,pq=predict(r),predict(c),predict(q)
            _,tr,_=outcomes(r);_,tc,_=outcomes(c)
            law_fit=fit_eu_distribution(inputs(r),tr-pr['mean_u'],inputs(c),tc-pc['mean_u'],
                fitted['base_covariance'],float(np.std(np.log(np.linalg.norm(data['Y'][t,0],axis=1)))),
                model_training_ids=ids[np.r_[t,v]],model_training_groups=groups[np.r_[t,v]])
            distribution=predict_eu_distribution(law_fit,inputs(q,pq['mean_u']))
            write_json(folder/'distribution.json',law_fit['report'])
            write_json(folder/'distribution_state.json',dict(law=distribution['law'],
                amplitude_fit=law_fit['amplitude_fit'],covariance_choice=law_fit['covariance_choice'],
                covariance_reference_bandwidth=law_fit['covariance_reference_bandwidth'],
                radial_reference_bandwidth=law_fit['radial_reference_bandwidth'],
                coordinate_space=law_fit['coordinate_space']))
            np.savez_compressed(folder/'distribution_arrays.npz',ref_ids=ids[r],cal_ids=ids[c],query_ids=ids[q],
                ref_residual=tr-pr['mean_u'],cal_residual=tc-pc['mean_u'],
                base_scatter=fitted['base_covariance'],ref_loo_weights=law_fit['reference_loo_weights'],
                ref_loo_covariance=law_fit['reference_loo_covariance'],
                cal_representative_indices=law_fit['calibration_representative_indices'],
                cal_radii=law_fit['calibration_radii'],cal_scatter_u=law_fit['calibration_scatter_u'],
                query_base_scatter_u=distribution['base_scatter_u'],query_scatter_u=distribution['scatter_u'],
                radial_weights=distribution['radial_weights'],query_reference_weights=distribution['reference_weights'],
                radial_variance_multiplier=distribution['radial_variance_multiplier'],
                query_mean_u=distribution['mean_u'])
            # Query outcomes are exposed to scoring only after means/law fixed.
            raw,target,actual=outcomes(q)
            check,obs,difference,_=observable_forward(raw)
            np.testing.assert_allclose(check,actual,rtol=1e-10,atol=1e-10)
            absolute=np.log1p(difference*pq['norm2_per_feature'][:,None])
            actual_all[q]=actual;folds[q]=fold
            ridge_mean=fitted['ridge'].predict_mean(transform_input(data['Y'][q,0],stats))
            cell=dict(fold=fold,n_query=len(q),query_ids=ids[q],reference_ids=ids[r],calibration_ids=ids[c],
                ridge_geometry_mse=float(np.square(ridge_mean-target).mean()),
                core_geometry_mse=float(np.square(pq['mean_u']-target).mean()),policies={},mc_sensitivity={},
                costs=dict(reference_if_all_new_wells=4*len(r),reference_if_X_already_available=3*len(r),
                    reference_if_reusable_new_wells=0,model_train_and_validation_wells=4*(len(t)+len(v)),
                    distribution_calibration_wells=4*len(c),query_initial_wells=len(q),
                    total_DMSO_wells_in_assay=784,query_replay_future_wells=3*len(q)))
            for arm in ARMS:
                radial=arm=='AMP_EMP_LOCAL'
                scatter=distribution['scatter_u'] if radial else distribution['base_scatter_u']
                law=distribution['law'] if radial else None
                weights=distribution['radial_weights'] if radial else None
                status('RUNNING',stage='100k joint evaluation',current_fold=fold,arm=arm,completed_folds=len(cells))
                out=score(pq['mean_u'],scatter,target,stats,actual,obs,absolute,pq['norm2_per_feature'],
                    SEED+fold*100,law=law,weights=weights,samples=SAMPLES)
                out.update(mean_u=pq['mean_u'],actual_u=target,scatter_u=scatter,brier=(out['p_null']-(actual<=0))**2)
                selections={}
                for lam in (.2,0.):
                    key=f'lambda_{lam:g}';selected=select(ids[q],out['predicted'],out['p_null'],lam)
                    out['selected_'+key]=selected;selections[key]=selected
                    cell['policies'][arm+'__'+key]=policy_summary(actual,selected)
                np.savez_compressed(folder/(arm+'.npz'),ids=ids[q],groups=groups[q],actual=actual,**out)
                cell[arm]=metric_summary(out,actual)
                for key,value in out.items():
                    if key not in stores[arm]:stores[arm][key]=np.empty((len(ids),*value.shape[1:]),dtype=value.dtype)
                    stores[arm][key][q]=value
                cell['mc_sensitivity'][arm]=[]
                for offset in (100000,200000):
                    moment=extra_seed_moments(pq['mean_u'],scatter,stats,law=law,weights=weights,seed=SEED+fold*100+offset)
                    sensitivity={}
                    for lam in (.2,0.):
                        key=f'lambda_{lam:g}';selected=select(ids[q],moment['predicted'],moment['p_null'],lam)
                        sensitivity[key]=dict(policy_summary(actual,selected),
                            symmetric_difference_vs_primary_seed=int((selected!=selections[key]).sum()),
                            selected_ids=ids[q][selected])
                    cell['mc_sensitivity'][arm].append(dict(seed_offset=offset,policies=sensitivity))
                    np.savez_compressed(folder/f'{arm}_mc{offset}.npz',ids=ids[q],**moment)
            k=int(select(ids[q],np.zeros(len(q)),np.zeros(len(q)),0.).sum())
            rng=np.random.default_rng(SEED+fold+500000)
            draws=np.array([actual[rng.choice(len(q),k,replace=False)].sum() for _ in range(10000)])
            false=np.array([(actual[rng.choice(len(q),k,replace=False)]<=0).sum() for _ in range(10000)])
            cell['random_same_budget']=dict(k=k,expected_total_value=float(k*actual.mean()),
                total_value_quantiles=np.quantile(draws,[.025,.5,.975]),null_count_quantiles=np.quantile(false,[.025,.5,.975]))
            cell['fixed']=dict(stop_total_value=0.,add_all_total_value=float(actual.sum()),
                add_all_extra_wells=2*len(q),add_all_budget_matched=False)
            write_json(folder/'summary.json',cell);cells.append(cell)
            print(f'fold {fold+1}/5 complete; seconds={time.monotonic()-started:.1f}',flush=True)
            status('RUNNING',stage='fold complete',completed_folds=len(cells))
        if np.any(folds<0) or not np.isfinite(actual_all).all():raise ValueError('Incomplete outer evaluation')
        metrics={}
        for arm,out in stores.items():
            np.savez_compressed(root/(arm+'.npz'),ids=ids,groups=groups,layout=data['layout'],fold=folds,actual=actual_all,**out)
            metrics[arm]=metric_summary(out,actual_all)
            metrics[arm]['policies']={f'lambda_{l:g}':policy_summary(actual_all,out[f'selected_lambda_{l:g}']) for l in (.2,0.)}
        paired={key:{scope:bootstrap_difference(stores['AMP_EMP_LOCAL'][key],stores['GAUSSIAN'][key],labels)
            for scope,labels in (('chemical_identity',groups),('library_plate_layout',data['layout']))}
            for key in ('crps','brier','nll','energy')}
        write_json(root/'summary.json',dict(n=len(ids),cells=cells,metrics=metrics,paired=paired,
            samples=SAMPLES,independent_confirmation=False,biology_active=False,representation_active=False,
            no_claim_of_finite_sample_certification=True,elapsed_seconds=time.monotonic()-started))
        status('COMPLETE',completed_folds=5)
    except Exception as exc:
        status('FAILED',completed_folds=len(cells),error_type=type(exc).__name__,error=str(exc))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--phase',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):run(args.dataset,args.phase,args.output)


if __name__=='__main__': main()
