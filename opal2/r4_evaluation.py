"""R4 fixed-list decisions and finite-campaign evaluation, with no fitting.

Selection accepts predictions only. Evaluation reuses the saved lists. The
separate campaign resampling repeats top-k without refitting any predictor.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .biology_kernel_evaluation import write_json
from .r4_confirmatory_metrics import (
    GAMMA_LOWER, GAMMA_UPPER, campaign_budget, paired_utility_bounds,
    risk_bounds, utility_bounds,
)


def _write_rows(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select(ids, score, eligible, k):
    ids, score, eligible = np.asarray(ids, str), np.asarray(score, float), np.asarray(eligible, bool)
    if score.shape != ids.shape or eligible.shape != ids.shape:
        raise ValueError('Decision arrays must align')
    rows = np.flatnonzero(eligible)
    if not np.isfinite(score[rows]).all() or not 0 <= k <= len(rows):
        raise ValueError('All eligible scores must be finite; k must be attainable')
    selected = np.zeros(len(ids), bool)
    # Duplicate IDs arise only in explicitly reconstructed campaigns. The
    # extra row key deterministically resolves identical copied observations.
    order = np.lexsort((rows, ids[rows], -score[rows]))
    selected[rows[order[:k]]] = True
    return selected


def freeze_selections(ids, eligible, predictors, output, *, random_seed=20260922):
    """Freeze score-based lists before any confirmation outcome is accepted.

    predictors maps names to full-N expected/p_null arrays plus optional lambda.
    Missing-X rows may contain NaNs; invalid predictions on valid X are errors.
    """
    ids, eligible = np.asarray(ids, str), np.asarray(eligible, bool)
    if ids.ndim != 1 or eligible.shape != ids.shape or len(set(ids)) != len(ids):
        raise ValueError('Unique identity-aligned eligibility required')
    root = Path(output)
    if (root/'SELECTIONS_FROZEN.json').exists():
        raise FileExistsError('Never overwrite a frozen selection')
    root.mkdir(parents=True, exist_ok=True)
    budget = campaign_budget(len(ids), int(eligible.sum()))
    k = budget['activations']
    policies = {}
    for name, prediction in predictors.items():
        mu, p = np.asarray(prediction['expected'], float), np.asarray(prediction['p_null'], float)
        if mu.shape != ids.shape or p.shape != ids.shape:
            raise ValueError('Predictor identity alignment differs')
        if (not np.isfinite(mu[eligible]).all() or not np.isfinite(p[eligible]).all()
                or np.any((p[eligible] < 0) | (p[eligible] > 1))):
            raise ValueError('An eligible prediction failed; no model-specific deletion')
        lam = float(prediction.get('lambda', .2))
        score = mu-lam*p
        policies[name] = dict(expected=mu, p_null=p, score=score,
                              selected=select(ids, score, eligible, k), risk_penalty=lam)
    if not {'CORE', 'HISTGB_CAL'} <= set(policies):
        raise ValueError('The prespecified primary pair is required')
    rng = np.random.default_rng(random_seed)
    random_mask = np.zeros(len(ids), bool)
    random_mask[rng.choice(np.flatnonzero(eligible), k, replace=False)] = True
    policies['RANDOM'] = dict(selected=random_mask)
    policies['CONSTANT_ID'] = dict(selected=select(ids, np.zeros(len(ids)), eligible, k))
    policies['STOP'] = dict(selected=np.zeros(len(ids), bool))
    policies['ALL_ELIGIBLE'] = dict(selected=eligible.copy())
    payload = {'ids':ids, 'eligible':eligible}
    for name, values in policies.items():
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                payload[name+'__'+key] = value
    np.savez_compressed(root/'selections.npz', **payload)
    _write_rows(root/'selected_lists.csv', (
        dict(policy=name, object_id=ids[i], score=values.get('score', np.full(len(ids), np.nan))[i])
        for name, values in policies.items() for i in np.flatnonzero(values['selected'])))
    record = dict(status='FROZEN', created_utc=datetime.now(timezone.utc).isoformat(),
                  budget=budget, ids=ids.tolist(), eligible_ids=ids[eligible].tolist(),
                  policies={n:dict(selected_ids=ids[v['selected']].tolist(),
                                   risk_penalty=v.get('risk_penalty')) for n,v in policies.items()},
                  random_seed=random_seed, outcomes_used=False,
                  primary_comparison='CORE minus HISTGB_CAL')
    write_json(root/'SELECTIONS_FROZEN.json', record)
    return policies, record


def load_selections(path):
    with np.load(Path(path)/'selections.npz', allow_pickle=False) as z:
        ids, eligible = z['ids'].copy(), z['eligible'].copy()
        policies = {}
        for key in z.files:
            if '__' in key:
                name, field = key.split('__',1)
                policies.setdefault(name,{})[field] = z[key].copy()
    return ids, eligible, policies


def _weighted_bounds(gamma, weights):
    known = np.isfinite(gamma)
    w = np.asarray(weights, float)
    point = float(np.dot(w[known],gamma[known]))
    missing = w[~known]
    low = point+float(np.where(missing >= 0,missing*GAMMA_LOWER,missing*GAMMA_UPPER).sum())
    high = point+float(np.where(missing >= 0,missing*GAMMA_UPPER,missing*GAMMA_LOWER).sum())
    return low/len(gamma), high/len(gamma)


def _policy_summary(gamma, values):
    chosen = values['selected']
    known = np.isfinite(gamma)
    selected_known = chosen & known
    value = utility_bounds(gamma,chosen)
    risk = risk_bounds(gamma,chosen)
    out = dict(n=len(gamma), selected_n=int(chosen.sum()),
               observed_selected_n=int(selected_known.sum()),
               unknown_selected_n=int((chosen & ~known).sum()),
               observed_selected_null=int(np.sum(gamma[selected_known] <= 0)),
               value_per_candidate_lower=value['lower'], value_per_candidate_upper=value['upper'],
               total_value=(float(gamma[chosen].sum()) if np.all(known[chosen]) else None),
               selected_mean_gamma=(float(gamma[chosen].mean()) if chosen.any() and np.all(known[chosen]) else None),
               fdp_lower=risk['fdp']['lower'], fdp_upper=risk['fdp']['upper'],
               fpr_lower=risk['fpr']['lower'], fpr_upper=risk['fpr']['upper'],
               fpr_may_be_undefined=risk['fpr']['undefined_completion_possible'],
               sensitivity_lower=risk['sensitivity']['lower'], sensitivity_upper=risk['sensitivity']['upper'])
    if 'p_null' in values:
        p, mu = values['p_null'], values['expected']
        valid = known & np.isfinite(p) & np.isfinite(mu)
        label = (gamma[valid] <= 0).astype(float)
        out.update(predicted_selected_null=float(p[chosen].sum()),
                   probability_evaluated_n=int(valid.sum()),
                   brier=float(np.mean((p[valid]-label)**2)) if valid.any() else None,
                   gamma_mse=float(np.mean((mu[valid]-gamma[valid])**2)) if valid.any() else None,
                   null_auc=float(roc_auc_score(label,p[valid])) if len(np.unique(label))==2 else None)
    return out


def _interval(values):
    values = np.asarray(values,float)
    values = values[np.isfinite(values)]
    return dict(valid_replicates=len(values),lower=float(np.quantile(values,.025)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                upper=float(np.quantile(values,.975)) if len(values) else None)


def _fdp_sensitivity_interval(lower, upper, missing_selected):
    """Percentile envelope of per-resample missing-label identification bounds.

    Unknown selected outcomes contribute [0, 1] labels, not a dropped draw or
    a point imputation. A draw with no selected objects has undefined FDP and
    is counted explicitly. The envelope is block-resampling sensitivity, not
    an iid-binomial confidence interval or a finite-sample risk guarantee.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    missing_selected = np.asarray(missing_selected, int)
    if lower.shape != upper.shape or lower.shape != missing_selected.shape:
        raise ValueError('FDP resampling arrays must align')
    defined = np.isfinite(lower) & np.isfinite(upper)
    if not np.array_equal(np.isfinite(lower), np.isfinite(upper)):
        raise ValueError('Both FDP bounds must be defined together')
    if np.any(lower[defined] > upper[defined]):
        raise ValueError('FDP lower bound exceeds upper bound')
    lo, hi = _interval(lower[defined]), _interval(upper[defined])
    return dict(
        total_replicates=len(lower), valid_replicates=int(defined.sum()),
        undefined_replicates=int((~defined).sum()),
        replicates_with_missing_selected=int(np.sum(defined & (missing_selected > 0))),
        lower=lo['lower'], upper=hi['upper'],
        median_lower=lo['median'], median_upper=hi['median'],
        lower_bound_distribution=lo, upper_bound_distribution=hi,
        interval_kind='block_percentile_envelope_of_missing_label_bounds',
        missing_outcome_handling='bound each unknown selected NULL label in [0, 1]',
        undefined_handling='no-selection FDP is undefined and counted separately',
    )


def campaign_resampling(ids, gamma, eligible, policies, labels, *, seed=20260923, repeats=10000):
    """Approximate block sensitivity for fixed-list and reconstructed-campaign targets."""
    ids, labels = np.asarray(ids,str), np.asarray(labels,str)
    _, inverse = np.unique(labels,return_inverse=True)
    blocks = [np.flatnonzero(inverse==i) for i in range(int(inverse.max())+1)]
    rng = np.random.default_rng(seed)
    names = ('CORE','HISTGB_CAL')
    rows = {mode:{key:[] for key in ('difference_low','difference_high',
            'CORE_fdp_low','CORE_fdp_high','CORE_fdp_missing',
            'HISTGB_CAL_fdp_low','HISTGB_CAL_fdp_high','HISTGB_CAL_fdp_missing',
            'CORE_random_difference_low','CORE_random_difference_high')}
            for mode in ('fixed_list','new_campaign_topk')}
    for _ in range(repeats):
        ind = np.concatenate([blocks[b] for b in rng.integers(0,len(blocks),len(blocks))])
        y, e = gamma[ind], eligible[ind]
        k = campaign_budget(len(ind),int(e.sum()))['activations']
        for mode in rows:
            masks = {name:(policies[name]['selected'][ind] if mode=='fixed_list' else
                            select(ids[ind],policies[name]['score'][ind],e,k)) for name in names}
            low, high = _weighted_bounds(y,masks['CORE'].astype(float)-masks['HISTGB_CAL'].astype(float))
            rows[mode]['difference_low'].append(low)
            rows[mode]['difference_high'].append(high)
            for name in names:
                m = masks[name]
                selected_y = y[m]
                observed = np.isfinite(selected_y)
                unknown_n = int((~observed).sum())
                null_n = int(np.sum(selected_y[observed] <= 0))
                selected_n = len(selected_y)
                rows[mode][name+'_fdp_low'].append(null_n/selected_n if selected_n else np.nan)
                rows[mode][name+'_fdp_high'].append((null_n+unknown_n)/selected_n if selected_n else np.nan)
                rows[mode][name+'_fdp_missing'].append(unknown_n)
            rk = int(masks['CORE'].sum())
            random_weights = e.astype(float)*(rk/e.sum() if e.any() else 0.)
            a,b = _weighted_bounds(y,masks['CORE'].astype(float)-random_weights)
            rows[mode]['CORE_random_difference_low'].append(a)
            rows[mode]['CORE_random_difference_high'].append(b)
    intervals = {}
    for mode, content in rows.items():
        intervals[mode] = {key:_interval(value) for key,value in content.items() if '_fdp_' not in key}
        for name in names:
            intervals[mode][name+'_fdp'] = _fdp_sensitivity_interval(
                content[name+'_fdp_low'],content[name+'_fdp_high'],content[name+'_fdp_missing'])
    return dict(blocks=len(blocks),repeats=repeats,seed=seed,fdp_schema_version=2,
                interpretation='approximate dependence sensitivity, not finite-sample certification',
                intervals=intervals)


def evaluate_campaign(selection_dir, outcome_ids, gamma, groups, layout, output, *, repeats=10000):
    ids, eligible, policies = load_selections(selection_dir)
    outcome_ids = np.asarray(outcome_ids,str)
    if not np.array_equal(ids,outcome_ids):
        raise ValueError('Outcomes must preserve the full frozen identity order')
    gamma, groups, layout = np.asarray(gamma,float), np.asarray(groups,str), np.asarray(layout,str)
    if any(v.shape != ids.shape for v in (gamma,groups,layout)):
        raise ValueError('Full-cohort outcome arrays must align')
    root = Path(output)
    if (root/'summary.json').exists():
        raise FileExistsError('Preserve the recorded confirmation evaluation')
    root.mkdir(parents=True,exist_ok=True)
    summaries = {name:_policy_summary(gamma,values) for name,values in policies.items()}
    primary = paired_utility_bounds(gamma,policies['CORE']['selected'],policies['HISTGB_CAL']['selected'])
    k = int(policies['CORE']['selected'].sum())
    random_weights = eligible.astype(float)*(k/eligible.sum() if eligible.any() else 0.)
    random_bounds = _weighted_bounds(gamma,random_weights)
    rows, regions, overlaps = [],[],[]
    for name, values in policies.items():
        rows.append(dict(policy=name,**summaries[name]))
        if 'score' not in values:
            continue
        rank = np.flatnonzero(eligible)
        rank = rank[np.lexsort((ids[rank],-values['score'][rank]))]
        percentile = np.full(len(ids),np.nan)
        percentile[rank] = (np.arange(len(rank))+.5)/len(rank) if len(rank) else []
        masks = [('all',eligible),('selected',values['selected'])]
        masks += [(f'rank_{a:g}_{b:g}',eligible & (percentile>=a)&(percentile<b))
                  for a,b in zip((0,.05,.1,.15,.25),(.05,.1,.15,.25,1.))]
        masks += [(f'layout_{block}_selected',values['selected']&(layout==block)) for block in np.unique(layout)]
        for region, mask in masks:
            known = mask & np.isfinite(gamma)
            regions.append(dict(policy=name,region=region,n=int(mask.sum()),observed_n=int(known.sum()),
                unknown_n=int((mask & ~np.isfinite(gamma)).sum()),
                predicted_null=float(values['p_null'][mask].sum()),
                predicted_null_on_observed=float(values['p_null'][known].sum()),
                observed_null=int(np.sum(gamma[known]<=0)),
                observed_rate=float(np.mean(gamma[known]<=0)) if known.any() else None,
                layout_n=len(np.unique(layout[mask]))))
    for a, pa in policies.items():
        for b, pb in policies.items():
            if a>=b: continue
            intersection=int((pa['selected']&pb['selected']).sum())
            union=int((pa['selected']|pb['selected']).sum())
            overlaps.append(dict(a=a,b=b,intersection=intersection,union=union,
                                 symmetric_difference=union-intersection,jaccard=intersection/union if union else None))
    uncertainty = {name:campaign_resampling(ids,gamma,eligible,policies,labels,repeats=repeats)
                   for name,labels in (('chemical_connectivity',groups),('library_layout',layout))}
    leave_layout = []
    for block in np.unique(layout):
        keep = layout!=block
        e, y = eligible[keep],gamma[keep]
        kk = campaign_budget(int(keep.sum()),int(e.sum()))['activations']
        masks = {name:select(ids[keep],policies[name]['score'][keep],e,kk) for name in ('CORE','HISTGB_CAL')}
        low,high=_weighted_bounds(y,masks['CORE'].astype(float)-masks['HISTGB_CAL'].astype(float))
        leave_layout.append(dict(excluded_layout=block,n=int(keep.sum()),k=kk,
                                 difference_lower=low,difference_upper=high))
    _write_rows(root/'policy_results.csv',rows)
    _write_rows(root/'selected_region_risk.csv',regions)
    _write_rows(root/'selection_overlap.csv',overlaps)
    _write_rows(root/'leave_one_layout.csv',leave_layout)
    _write_rows(root/'object_results.csv',(dict(object_id=ids[i],group=groups[i],layout=layout[i],
        eligible_x=bool(eligible[i]),gamma=gamma[i],null=int(gamma[i]<=0) if np.isfinite(gamma[i]) else '',
        **{name+'_selected':bool(v['selected'][i]) for name,v in policies.items()}) for i in range(len(ids))))
    summary=dict(status='COMPLETE',n=len(ids),eligible_x_n=int(eligible.sum()),observed_outcome_n=int(np.isfinite(gamma).sum()),
                 primary_comparison='CORE minus HISTGB_CAL',primary=primary,policies=summaries,
                 exact_random_expectation_bounds=dict(lower=random_bounds[0],upper=random_bounds[1]),
                 uncertainty=uncertainty,independence_scope='as recorded in final identity qualification',
                 selection_fixed_before_outcomes=True)
    write_json(root/'summary.json',summary)
    return summary
