"""Saved-prediction DEV value/risk frontier, not policy selection or certification.

No model is fitted. Scores use decision-time predictions only, with fixed per-cell
budgets. Outcomes evaluate the whole declared grid; no winner is adopted. All
bootstrap uncertainty conditions on the saved models and assignments. It does not
account for model fitting, repeated development, or transport to unseen layouts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json


ARMS = ('GAUSSIAN', 'AMP_GAUSSIAN', 'EMP_GLOBAL', 'AMP_EMP_GLOBAL', 'AMP_EMP_LOCAL')
LAMBDA_GRID = (0., .01, .025, .05, .1, .2, .4, .8, 1.6, 3.2, 6.4, 12.8, np.inf)
PROBABILITY_EDGES = np.array([0., .05, .1, .2, .4, .6, .8, 1.])
LOG_CLIP = 1e-6
SEED = 20260916


def cp_upper(k, n, alpha=.05):
    """One-sided 1-alpha Clopper--Pearson, not two-sided 95%."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError('Invalid binomial counts')
    return 1. if k == n else float(beta.ppf(1-alpha, k+1, n-k))


def cp_lower(k, n, alpha=.05):
    if not 0 <= k <= n or n <= 0:
        raise ValueError('Invalid binomial counts')
    return 0. if k == 0 else float(beta.ppf(alpha, k, n-k+1))


def cell_indices(ids, cells):
    lookup = {v: i for i, v in enumerate(ids.tolist())}
    if len(lookup) != len(ids):
        raise ValueError('Duplicate query identities')
    seen = np.zeros(len(ids), int)
    out = []
    for c in cells:
        q = np.array([lookup[v] for v in c['query_ids']], int)
        if not 0 < c['budget'] <= len(q):
            raise ValueError('Invalid cell budget')
        if set(c['query_ids']) & (set(c['fit_ids']) | set(c['calibration_ids'])):
            raise ValueError('Own-cell query/reference overlap')
        seen[q] += 1
        out.append((q, int(c['budget'])))
    if np.any(seen != 1):
        raise ValueError('Queries must appear exactly once')
    return out


def select_policy(mean, probability, ids, cells, lam):
    score = -probability if np.isinf(lam) else mean-lam*probability
    selected = np.zeros(len(ids), bool)
    for q, budget in cells:
        order = np.lexsort((ids[q], -score[q]))
        selected[q[order[:budget]]] = True
    return selected


def rank_band(score, ids, cells, lo=.08, hi=.16):
    """Ranks at their (j+.5)/cell_n midpoints; band definition fixed in advance."""
    mask = np.zeros(len(ids), bool)
    for q, _ in cells:
        order = np.lexsort((ids[q], -score[q]))
        rank = (np.arange(len(q))+.5)/len(q)
        mask[q[order[(rank >= lo) & (rank < hi)]]] = True
    return mask


def counts_and_metrics(actual, mean, probability, selected):
    null, positive = actual <= 0, actual >= .005
    k, false, true = int(selected.sum()), int((selected & null).sum()), int((selected & positive).sum())
    out = dict(n=len(actual), selected_n=k, extra_wells=2*k,
        selected_null=false, selected_positive=true, selected_ambiguous=k-false-true,
        null_n=int(null.sum()), positive_n=int(positive.sum()),
        predicted_selected_mean=float(mean[selected].mean()),
        predicted_all_object_value=float((selected*mean).mean()),
        predicted_fdp=float(probability[selected].mean()),
        actual_selected_mean=float(actual[selected].mean()),
        actual_all_object_value=float((selected*actual).mean()),
        actual_total_net_value=float(actual[selected].sum()),
        fdp=false/k, fpr=false/int(null.sum()), sensitivity=true/int(positive.sum()),
        coverage=k/len(actual), fdp_cp_upper=cp_upper(false,k),
        fpr_cp_upper=cp_upper(false,int(null.sum())),
        sensitivity_cp_lower=cp_lower(true,int(positive.sum())))
    out['iid_count_conditions'] = dict(fdp_35=out['fdp_cp_upper'] <= .35,
        fpr_075=out['fpr_cp_upper'] <= .075, sensitivity_05=out['sensitivity_cp_lower'] >= .05,
        activations_100=k >= 100, coverage_05=out['coverage'] >= .05,
        fdp_15_sensitivity_only=out['fdp_cp_upper'] <= .15)
    return out


class ClusterResamples:
    """Shared resamples of whole clusters; ratios retain random denominators."""
    def __init__(self, labels, n_boot=5000, seed=SEED):
        self.labels, self.inverse = np.unique(labels, return_inverse=True)
        self.g = len(self.labels)
        rng = np.random.default_rng(seed)
        self.weights = rng.multinomial(self.g, np.full(self.g, 1/self.g), size=n_boot).astype(float)
        self.sizes = self.aggregate(np.ones(len(labels)))
        self.sample_n = self.weights@self.sizes

    def aggregate(self, values):
        v = np.asarray(values, float)
        if v.ndim == 1:
            return np.bincount(self.inverse, weights=v, minlength=self.g)
        out = np.zeros((self.g, v.shape[1]))
        np.add.at(out, self.inverse, v)
        return out

    def ratio(self, numerator, denominator):
        num = self.weights@self.aggregate(numerator)
        den = self.weights@self.aggregate(denominator)
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    def mean(self, values):
        num = self.weights@self.aggregate(values)
        return num/self.sample_n if num.ndim == 1 else num/self.sample_n[:, None]


def interval(samples):
    values = np.asarray(samples)
    good = values[np.isfinite(values)]
    if not len(good):
        return dict(ci95=None, lower_one_sided_95=None, upper_one_sided_95=None, valid_n=0)
    return dict(ci95=np.quantile(good, [.025, .975]).tolist(),
        lower_one_sided_95=float(np.quantile(good, .05)),
        upper_one_sided_95=float(np.quantile(good, .95)), valid_n=len(good))


def bca_mean_lower(values, resamples, alpha=.05):
    """BCa mean lower endpoint with leave-cluster-out acceleration."""
    v = np.asarray(values, float)
    theta = float(v.mean())
    draws = resamples.mean(v)
    fraction = (np.count_nonzero(draws < theta)+.5*np.count_nonzero(draws == theta))/len(draws)
    z0 = norm.ppf(np.clip(fraction, .5/len(draws), 1-.5/len(draws)))
    sums = resamples.aggregate(v)
    jack = (v.sum()-sums)/(len(v)-resamples.sizes)
    centered = jack.mean()-jack
    denominator = 6*np.square(centered).sum()**1.5
    acceleration = 0. if denominator == 0 else float((centered**3).sum()/denominator)
    z = norm.ppf(alpha)
    adjusted = float(norm.cdf(z0+(z0+z)/(1-acceleration*(z0+z))))
    return dict(mean=theta, lower_one_sided_95=float(np.quantile(draws,adjusted)),
        z0=float(z0), acceleration=acceleration, adjusted_quantile=adjusted,
        bootstrap_n=len(draws), clusters=resamples.g)


def policy_bootstrap(actual, selected, base_selected, samplers):
    null, positive = (actual <= 0).astype(float), (actual >= .005).astype(float)
    records = {}
    for scope, sampler in samplers.items():
        vals = dict(fdp=sampler.ratio(selected*null,selected),
            fpr=sampler.ratio(selected*null,null),
            sensitivity=sampler.ratio(selected*positive,positive),
            selected_mean=sampler.ratio(selected*actual,selected),
            all_object_value=sampler.mean(selected*actual))
        base = dict(fdp=sampler.ratio(base_selected*null,base_selected),
            selected_mean=sampler.ratio(base_selected*actual,base_selected),
            all_object_value=sampler.mean(base_selected*actual))
        records[scope] = dict(level={k: interval(v) for k,v in vals.items()},
            paired_vs_own_lambda0={k:interval(vals[k]-base[k]) for k in base},
            all_object_value_bca=bca_mean_lower(selected*actual,sampler))
    return records


def calibration_record(probability, null, mask, samplers):
    n = int(mask.sum())
    if not n:
        return dict(n=0)
    p = np.clip(probability, LOG_CLIP, 1-LOG_CLIP)
    record = dict(n=n, null_n=int(null[mask].sum()), predicted_risk=float(probability[mask].mean()),
        observed_risk=float(null[mask].mean()), residual_predicted_minus_observed=float((probability-null)[mask].mean()),
        brier=float(np.square(probability-null)[mask].mean()),
        log_loss=float(-(null*np.log(p)+(1-null)*np.log1p(-p))[mask].mean()))
    record['residual_bootstrap'] = {s:interval(r.ratio(mask*(probability-null),mask)) for s,r in samplers.items()}
    return record


def diagnostics_masks(mean, probability, ids, cells):
    masks = dict(full_population=np.ones(len(ids),bool),
        expected_gain_selected=select_policy(mean,probability,ids,cells,0),
        low_null_selected=select_policy(mean,probability,ids,cells,np.inf),
        expected_gain_rank_08_16=rank_band(mean,ids,cells),
        low_null_rank_08_16=rank_band(-probability,ids,cells))
    for i,(lo,hi) in enumerate(zip(PROBABILITY_EDGES[:-1],PROBABILITY_EDGES[1:])):
        masks[f'probability_bin_{lo:g}_{hi:g}'] = (probability >= lo)&((probability < hi) if i < len(PROBABILITY_EDGES)-2 else probability <= hi)
    return masks


def run(source, output, cluster_boot=5000, iid_boot=20000):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_summary=json.loads((source/'summary.json').read_text())
    stores={a:dict(np.load(source/f'{a}.npz')) for a in ARMS}
    base=stores['GAUSSIAN']; ids=base['ids']; actual=base['actual']; null=(actual <= 0).astype(float)
    if len(ids) != 1188 or not np.isfinite(actual).all():
        raise ValueError('Expected complete opened 1188-object DEV only')
    for a,z in stores.items():
        for key in ('ids','groups','layout','fold','actual'):
            np.testing.assert_array_equal(z[key],base[key],err_msg=a+' '+key)
        if not np.isfinite(z['predicted']).all() or not np.isfinite(z['p_null']).all():
            raise ValueError('Nonfinite saved predictions')
    cells=cell_indices(ids,source_summary['cells'])
    samplers={scope:ClusterResamples(base[key],cluster_boot,SEED+i) for i,(scope,key) in enumerate((('chemistry','groups'),('layout','layout')))}
    iid=ClusterResamples(np.arange(len(ids)),iid_boot,SEED+2)
    rows=[]; masks=[]; snapshots={}; calibration={}; comparison={}
    for arm,z in stores.items():
        zero=select_policy(z['predicted'],z['p_null'],ids,cells,0)
        np.testing.assert_array_equal(zero,z['selected'].astype(bool),err_msg=arm+' lambda0 saved policy')
        for lam in LAMBDA_GRID:
            name='inf' if np.isinf(lam) else f'{lam:g}'
            chosen=select_policy(z['predicted'],z['p_null'],ids,cells,lam)
            row=counts_and_metrics(actual,z['predicted'],z['p_null'],chosen)
            row.update(arm=arm,lambda_name=name,selection_id=f'{arm}__lambda_{name}',
                replacements_vs_own_lambda0=int(np.count_nonzero(chosen != zero)//2),
                bootstrap=policy_bootstrap(actual,chosen,zero,samplers),
                iid_value_bca=bca_mean_lower(chosen*actual,iid),
                selected_ids=ids[chosen].tolist())
            row['iid_six_observed_conditions_pass'] = bool(all(v for k,v in row['iid_count_conditions'].items() if k != 'fdp_15_sensitivity_only') and row['iid_value_bca']['lower_one_sided_95'] > 0)
            rows.append(row); masks.append(chosen)
        snapshots[arm]=dict(brier=float(np.square(z['p_null']-null).mean()),
            auc=float(roc_auc_score(null,z['p_null'])), expected_gain_vs_low_null_rank_disagreement=int(np.count_nonzero(zero != masks[-1])))
        print(f'Frontier complete {arm}',flush=True)
    common=diagnostics_masks(base['predicted'],base['p_null'],ids,cells)
    for arm in ('GAUSSIAN','AMP_EMP_LOCAL'):
        z=stores[arm]
        calibration[arm]={}
        for scope,regions in (('common_gaussian_masks',common),('own_policy_masks',diagnostics_masks(z['predicted'],z['p_null'],ids,cells))):
            calibration[arm][scope]={name:calibration_record(z['p_null'],null,mask,samplers) for name,mask in regions.items()}
    new=stores['AMP_EMP_LOCAL']; p0=base['p_null']; p1=new['p_null']
    for name,mask in common.items():
        if not mask.any():
            continue
        losses={}
        for arm,p in (('GAUSSIAN',p0),('AMP_EMP_LOCAL',p1)):
            pc=np.clip(p,LOG_CLIP,1-LOG_CLIP)
            losses[arm]=dict(brier=np.square(p-null), log_loss=-null*np.log(pc)-(1-null)*np.log1p(-pc),calibration_residual=p-null)
        comparison[name]={metric:dict(difference=float((losses['AMP_EMP_LOCAL'][metric]-losses['GAUSSIAN'][metric])[mask].mean()),
            bootstrap={s:interval(r.ratio(mask*(losses['AMP_EMP_LOCAL'][metric]-losses['GAUSSIAN'][metric]),mask)) for s,r in samplers.items()}) for metric in ('brier','log_loss','calibration_residual')}
    # References are already-opened objects, but only own-cell FIT/CAL labels
    # are used to compute a deployable constant; the whole-cohort rate is descriptive.
    lookup={v:i for i,v in enumerate(ids)}; constant={}
    for scope,key in (('fit','fit_ids'),('calibration','calibration_ids')):
        p=np.empty(len(ids))
        for c,(q,_) in zip(source_summary['cells'],cells):
            ref=np.array([lookup[v] for v in c[key]])
            p[q]=null[ref].mean()
        constant[scope]=dict(brier=float(np.square(p-null).mean()),mean_predicted_probability=float(p.mean()))
    inclusion=np.empty(len(ids))
    for q,k in cells:
        inclusion[q]=k/len(q)
    random_baseline=dict(selected_n=float(inclusion.sum()),expected_null=float((inclusion*null).sum()),
        expected_selected_mean=float((inclusion*actual).sum()/inclusion.sum()),
        expected_all_object_value=float((inclusion*actual).mean()),
        expected_fdp=float((inclusion*null).sum()/inclusion.sum()),
        definition='Uniform fixed-k sampling within each unchanged query cell; analytical expectation, not a sampled policy')
    # Model-versus-model comparisons keep lambda fixed; comparisons against the
    # random-policy expectation retain the unchanged cell inclusion probabilities.
    cross_distribution={}; versus_random={}
    masks_by_key={r['selection_id']:m for r,m in zip(rows,masks)}
    for name in ('0','0.2','inf'):
        a=masks_by_key[f'AMP_EMP_LOCAL__lambda_{name}']
        b=masks_by_key[f'GAUSSIAN__lambda_{name}']
        cross_distribution[name]={}
        for scope,sampler in samplers.items():
            cross_distribution[name][scope]=dict(
                fdp=interval(sampler.ratio(a*null,a)-sampler.ratio(b*null,b)),
                selected_mean=interval(sampler.ratio(a*actual,a)-sampler.ratio(b*actual,b)),
                all_object_value=interval(sampler.mean((a.astype(float)-b)*actual)))
    for arm in ('GAUSSIAN','AMP_EMP_LOCAL'):
        selected=masks_by_key[f'{arm}__lambda_0']
        versus_random[arm]={scope:dict(
            fdp=interval(sampler.ratio(selected*null,selected)-sampler.ratio(inclusion*null,inclusion)),
            selected_mean=interval(sampler.ratio(selected*actual,selected)-sampler.ratio(inclusion*actual,inclusion)),
            all_object_value=interval(sampler.mean((selected-inclusion)*actual))) for scope,sampler in samplers.items()}
    result=dict(scope='Post-hoc opened DEV diagnostics; no new certification, no adopted lambda',
        source=str(source),n=len(ids),null_n=int(null.sum()),positive_n=int((actual >= .005).sum()),
        ambiguous_n=int(((actual > 0)&(actual < .005)).sum()),null_rate=float(null.mean()),
        total_actual_mean=float(actual.mean()),descriptive_whole_cohort_constant_brier=float(null.mean()*(1-null.mean())),
        honest_reference_constants=constant,random_same_budget=random_baseline,lambda_grid=['inf' if np.isinf(x) else x for x in LAMBDA_GRID],
        original_contract=dict(fdp_upper=.35,fpr_upper=.075,sensitivity_lower=.05,value_lower_strictly_positive=True,
            activations_min=100,coverage_min=.05,missing_max=.05),
        missingness='All1188savedoutcomesfinite; this doesnot establish source-cohort missingness or least-favourable completion',
        formal_seven_item_certification=False,policies=rows,model_snapshots=snapshots,
        paired_latest_minus_gaussian_same_lambda=cross_distribution,
        expected_gain_policy_minus_random_expectation=versus_random,
        calibration=calibration,paired_calibration_common_masks=comparison,
        bootstrap=dict(iid_n=iid_boot,cluster_n=cluster_boot,chemistry_groups=len(np.unique(base['groups'])),layout_groups=len(np.unique(base['layout'])),
            interpretation='Pointwise fixed-prediction fixed-assignment resampling; no refitting/reselection, multiplicity adjustment, or new-layout transport guarantee'),
        probability_log_clip=LOG_CLIP,probability_bins=PROBABILITY_EDGES.tolist(),
        rank_band=dict(lower=.08,upper=.16,definition='Within-cell rank midpoint (j+.5)/n'),
        final_opened=False,model_training=False,formal_policy_changed=False)
    write_json(output/'summary.json',result)
    flatkeys=['arm','lambda_name','selected_n','selected_null','selected_positive','predicted_fdp','predicted_selected_mean',
        'actual_selected_mean','actual_all_object_value','actual_total_net_value','fdp','fpr','sensitivity','fdp_cp_upper','fpr_cp_upper','sensitivity_cp_lower','replacements_vs_own_lambda0','iid_six_observed_conditions_pass']
    with (output/'frontier.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=flatkeys);writer.writeheader();writer.writerows({k:r[k] for k in flatkeys} for r in rows)
    np.savez_compressed(output/'selections.npz',ids=ids,actual=actual,groups=base['groups'],layout=base['layout'],
        policy_ids=np.array([r['selection_id'] for r in rows]),selected=np.array(masks))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cluster-boot',type=int,default=5000);parser.add_argument('--iid-boot',type=int,default=20000)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source,args.output,args.cluster_boot,args.iid_boot)


if __name__=='__main__':
    main()
