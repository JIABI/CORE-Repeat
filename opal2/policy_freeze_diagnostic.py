"""Fixed lambda 0/.2 policy checks on saved opened-DEV predictions only.

The paired bootstrap conditions on already-developed models and assignments.
No acceptable noninferiority margin is chosen and no policy is certified.
No lambda outside {0,.2}, label-driven threshold search, or model fit occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .decision_frontier_diagnostic import (ClusterResamples, counts_and_metrics,
    cell_indices, interval, select_policy, SEED)


POLICIES = ('GAUSSIAN__lambda_0', 'GAUSSIAN__lambda_0.2',
    'AMP_EMP_LOCAL__lambda_0', 'AMP_EMP_LOCAL__lambda_0.2')
COMPARISONS = (
    ('GAUSSIAN__lambda_0.2', 'GAUSSIAN__lambda_0', 'rule_only_gaussian'),
    ('AMP_EMP_LOCAL__lambda_0.2', 'AMP_EMP_LOCAL__lambda_0', 'rule_only_latest'),
    ('AMP_EMP_LOCAL__lambda_0', 'GAUSSIAN__lambda_0', 'distribution_only_at_lambda0'),
    ('AMP_EMP_LOCAL__lambda_0.2', 'GAUSSIAN__lambda_0.2', 'distribution_only_at_lambda02'),
    ('AMP_EMP_LOCAL__lambda_0.2', 'GAUSSIAN__lambda_0', 'joint_distribution_and_rule'))


def minimal_margin(bound, *, higher_is_better):
    """Infer a bound-compatible margin, never an acceptable clinical margin.

For higher-is-better difference D, NI requires LCB(D)>-delta.
For lower-is-better difference D, NI requires UCB(D)<delta.
Returned threshold is an infimum: equality alone is not strict NI.
"""
    return max(0., -float(bound) if higher_is_better else float(bound))


def subset_metrics(actual, selected, keep):
    y = np.asarray(actual)[keep]
    s = np.asarray(selected)[keep]
    k = int(s.sum()); null = y <= 0; positive = y >= .005
    return dict(n=int(keep.sum()), selected_n=k, selected_null=int((s&null).sum()),
        selected_positive=int((s&positive).sum()),
        selected_mean=float(y[s].mean()) if k else None,
        all_object_value=float(np.mean(s*y)), fdp=float((s&null).sum()/k) if k else None,
        fpr=float((s&null).sum()/null.sum()) if null.any() else None,
        sensitivity=float((s&positive).sum()/positive.sum()) if positive.any() else None)


def difference_record(actual, a, b, keep):
    ma, mb = subset_metrics(actual,a,keep), subset_metrics(actual,b,keep)
    differences={key: (None if ma[key] is None or mb[key] is None else ma[key]-mb[key])
        for key in ma if key != 'n'}
    return dict(candidate=ma,baseline=mb,difference=differences)


def paired_metrics(actual, a, b, sampler):
    null = (actual <= 0).astype(float)
    value = sampler.mean((a.astype(float)-b)*actual)
    selected = sampler.ratio(a*actual,a)-sampler.ratio(b*actual,b)
    fdp = sampler.ratio(a*null,a)-sampler.ratio(b*null,b)
    fpr = sampler.ratio((a.astype(float)-b)*null,null)
    result={key:interval(x) for key,x in (('all_object_value',value),
        ('selected_mean',selected),('fdp',fdp),('fpr',fpr))}
    for key in ('all_object_value','selected_mean'):
        record=result[key]
        record['minimum_margin_infimum_from_one_sided_bound']=minimal_margin(
            record['lower_one_sided_95'],higher_is_better=True)
        record['zero_margin_strict_superiority_descriptive']=record['lower_one_sided_95']>0
    for key in ('fdp','fpr'):
        record=result[key]
        record['minimum_risk_increase_margin_infimum']=minimal_margin(
            record['upper_one_sided_95'],higher_is_better=False)
    return result


def score_uncertainty(mean, probability, mean_se, probability_se, ids, cells, lam):
    """Delta-free MC score-SE envelope from Cauchy--Schwarz only.

    Γ and 1[Γ<=0] share draws. Their covariance is nonpositive by monotonicity,
    but its magnitude is not known from saved marginal standard errors.
    sqrt(sum squares) is a lower bound, NOT the true score SE.
    Gaussian 1.96 envelopes are illustrative cutoff screens, not guarantees.
    """
    score=mean-lam*probability
    if lam < 0:raise ValueError('Only nonnegative frozen risk penalties are supported')
    se_low=np.hypot(mean_se,lam*probability_se)
    se_high=mean_se+lam*probability_se
    rows=[]
    for j,(q,k) in enumerate(cells):
        order=np.lexsort((ids[q],-score[q])); take=q[order[:k]]; leave=q[order[k:]]
        last, first=take[-1], leave[0]
        gap=float(score[last]-score[first])
        gap_se_upper=float(se_high[last]+se_high[first])
        unselected_upper=float(np.max(score[leave]+1.96*se_high[leave]))
        selected_lower=float(np.min(score[take]-1.96*se_high[take]))
        possibly_out=take[score[take]-1.96*se_high[take] <= unselected_upper]
        possibly_in=leave[score[leave]+1.96*se_high[leave] >= selected_lower]
        rows.append(dict(cell=j,n=len(q),budget=k,cutoff_gap=gap,
            cutoff_score_selected=float(score[last]),cutoff_score_unselected=float(score[first]),
            selected_score_mc_se_interval=[float(se_low[last]),float(se_high[last])],
            unselected_score_mc_se_interval=[float(se_low[first]),float(se_high[first])],
            cutoff_gap_mc_se_upper=gap_se_upper,
            gap_over_mc_se_upper=gap/gap_se_upper if gap_se_upper else None,
            gap_above_196_upper_se=bool(gap > 1.96*gap_se_upper),
            selected_with_potential_envelope_overlap=len(possibly_out),
            unselected_with_potential_envelope_overlap=len(possibly_in),
            cutoff_ids=[str(ids[last]),str(ids[first])],
            potential_removed_ids=ids[possibly_out].tolist(),
            potential_added_ids=ids[possibly_in].tolist()))
    return rows


def run(source, frontier, output):
    source,frontier,output=map(lambda p:Path(p).resolve(),(source,frontier,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    saved=json.loads((source/'summary.json').read_text())
    old=json.loads((frontier/'summary.json').read_text())
    stores={a:dict(np.load(source/f'{a}.npz')) for a in ('GAUSSIAN','AMP_EMP_LOCAL')}
    base=stores['GAUSSIAN']; ids=base['ids'];actual=base['actual'];n=len(ids)
    if n!=1188 or not np.isfinite(actual).all():raise ValueError('Opened scope changed')
    for key in ('ids','actual','groups','layout','fold'):
        np.testing.assert_array_equal(base[key],stores['AMP_EMP_LOCAL'][key])
    cells=cell_indices(ids,saved['cells'])
    with np.load(frontier/'selections.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids)
        cached={v:z['selected'][j] for j,v in enumerate(z['policy_ids']) if v in POLICIES}
    selections={}; policies={}; mc={}; private_mc={}
    for name in POLICIES:
        arm,lambda_name=name.split('__lambda_');lam=float(lambda_name);data=stores[arm]
        chosen=select_policy(data['predicted'],data['p_null'],ids,cells,lam)
        np.testing.assert_array_equal(chosen,cached[name])
        selections[name]=chosen
        row=counts_and_metrics(actual,data['predicted'],data['p_null'],chosen)
        previous=next(r for r in old['policies'] if r['selection_id']==name)
        row['previous_exact_layout_fpr_upper']=previous['bootstrap']['layout']['level']['fpr']['upper_one_sided_95']
        row['previous_iid_bca_all_object_value_lower']=previous['iid_value_bca']['lower_one_sided_95']
        policies[name]=row
        check=score_uncertainty(data['predicted'],data['p_null'],data['gamma_mc_se'],data['null_mc_se'],ids,cells,lam)
        private_mc[name]=check
        mc[name]=[{k:v for k,v in r.items() if not k.endswith('_ids')} for r in check]
    samplers={scope:ClusterResamples(base[key],5000,SEED+j) for j,(scope,key) in enumerate((('chemistry','groups'),('layout','layout')))}
    fold=base['fold'];cell_label=np.empty(n,int)
    for j,(q,_) in enumerate(cells):cell_label[q]=j
    layout_unique=np.unique(base['layout']); layout_codes={v:f'L{j+1:02}' for j,v in enumerate(layout_unique)}
    all_keep=np.ones(n,bool);comparisons={};private={}
    for candidate,reference,kind in COMPARISONS:
        a,b=selections[candidate],selections[reference]
        row=difference_record(actual,a,b,all_keep)
        row.update(candidate_id=candidate,baseline_id=reference,kind=kind,
            paired_bootstrap={scope:paired_metrics(actual,a,b,sampler) for scope,sampler in samplers.items()})
        slices={}
        for name,labels in (('fold',fold),('cell',cell_label),('layout',base['layout'])):
            records=[]
            for g in np.unique(labels):
                take=labels==g
                r=difference_record(actual,a,b,take)
                r['group']=layout_codes[g] if name=='layout' else int(g)
                records.append(r)
            slices[name]=records
        loo=[]
        for g in layout_unique:
            r=difference_record(actual,a,b,base['layout']!=g);r['omitted_layout']=layout_codes[g];loo.append(r)
        row['by_group']=slices;row['leave_one_layout_out_fixed_masks']=loo
        row['null_count_change_layouts']=[r['group'] for r in slices['layout'] if r['difference']['selected_null']!=0]
        row['layout_improvement_counts']={metric:dict(positive=sum(r['difference'][metric]>0 for r in slices['layout'] if r['difference'][metric] is not None),
            negative=sum(r['difference'][metric]<0 for r in slices['layout'] if r['difference'][metric] is not None),
            zero=sum(r['difference'][metric]==0 for r in slices['layout'] if r['difference'][metric] is not None))
            for metric in ('selected_null','all_object_value')}
        comparisons[kind]=row
        swapped=np.flatnonzero(a!=b)
        private[kind]=[dict(id=str(ids[i]),layout=str(base['layout'][i]),fold=int(fold[i]),cell=int(cell_label[i]),
            candidate_selected=bool(a[i]),baseline_selected=bool(b[i]),actual_gamma=float(actual[i]),null=bool(actual[i]<=0)) for i in swapped]
    # Reusing predictions assigned to the opposite query half as this cell's
    # CAL scores is unsafe: their radial-reference set includes our own queries.
    own_lookups={v:j for j,c in enumerate(saved['cells']) for v in c['query_ids']}
    reuse=[]
    for j,c in enumerate(saved['cells']):
        relevant=sorted(set(own_lookups[v] for v in c['calibration_ids']))
        own_queries=set(c['query_ids']); touched=set()
        for k in relevant:
            other=saved['cells'][k]
            touched.update(own_queries & (set(other['fit_ids'])|set(other['calibration_ids'])))
        reuse.append(dict(cell=j,calibration_rows=len(c['calibration_ids']),
            cached_prediction_source_cells=relevant,
            own_query_rows_used_as_reference_by_those_cached_scores=len(touched),
            safe_to_reuse_cached_query_predictions_for_cal_selection=not bool(touched)))
    result=dict(scope='Fixed opened DEV freeze diagnostics only; no new model fitting, lambda search, or FINAL access',
        n=n,policies=policies,comparisons=comparisons,mc_cutoff_screen=mc,
        mc_samples_per_object=saved['samples'],
        mc_assumptions='Only saved marginal MC SEs used; covariance of meanΓ and PNULL has nonpositive sign, but unknown magnitude. Independent-quadrature is a lowerSEbound, Cauchy sum is an upperSEbound. Upper envelopes plus normal1.96 are illustrative, not rank confidence sets. Across-object MC covariance also not assumed zero.',
        noninferiority='Reported margins are infima compatible with pointwise one-sided bounds, NOT acceptable margins. StrictNIrequiresaprespecifiedmarginlargerthantheinfimum. PositivepointestimateornonsignificantdifferenceisnotNI.',
        bootstrap='5000sharedchemistry/layoutresamples, same seed as preceding frontier, fixedpredictionsandassignments; no multiplicity/repeated-development adjustment or new-layout guarantee',
        leave_one_layout_out='Remove observed layout without refilling policies. Selectedcountsandpopulationchange. Diagnostic only, not retrained or recalibrated held-out-layout evaluation.',
        cached_calibration_reuse_audit=reuse,
        calibration_requirement='For prospective policy selection, generate own-cell FIT/CAL scores without query outcomes entering their fitted/calibrated predictor; current query-only merged arrays cannot substitute. Additional inner calibration split/crossfit must be declared before use.',
        final_opened=False,policy_frozen=False,acceptable_noninferiority_margin_chosen=False)
    write_json(output/'summary.json',result)
    write_json(output/'private_affected_objects_and_mc_cutoffs.json',dict(layout_codes=layout_codes,comparisons=private,mc=private_mc))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--frontier',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.frontier,a.output)
