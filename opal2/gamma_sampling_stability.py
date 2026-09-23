"""MC-only reevaluation of fixed A/J/K predictions; no model or data loading.

The sample seeds are integration replicates, not independent training or
experimental replicates. All objects, outcomes, means and covariances remain
fixed. Each 2,000-draw score uses the prefix of its 10,000-draw score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from .baseline_policy import ACTIONS, WELL_COSTS, _metrics, _observed, stable_top_k
from .biology_kernel_evaluation import write_json
from .gram_factor_verified import decode_draws
from .gram_geometry import gram_gains
from .objective_analysis import fair_crps


ARMS = ('A_HR', 'J_GEOMETRY_CONTROL', 'K_GEOMETRY_GAMMA_CRPS')
CONFIG = dict(seed_offsets=[310000, 410000, 510000], sample_counts=[2000, 10000],
              historical_seed_offset=200000, decode_draw_chunk=256, threads=4,
              fractions=[.05, .10, .25], endpoint='original ADD_ONE/ADD_TWO Gamma',
              models_frozen=True, fit_performed=False, primary_action='Z1Z2',
              gate='all 3 high-draw K-J CRPS differences <0 and -mean(delta)>2*sd(delta,ddof=1)/sqrt(3)',
              gate_scope='engineering continuation screen, not statistical certification')


def now():
    return datetime.now(timezone.utc).isoformat()


def mc_continuation_gate(differences):
    """Three paired MC-replicate score differences, never 639*3 new cases."""
    values = np.asarray(differences, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Exactly three finite high-draw paired differences are required')
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd/np.sqrt(3)
    return dict(passed=bool(np.all(values < 0) and -mean > 2*se),
                differences=values.tolist(), mean_difference=mean,
                mc_replicate_sd=sd, mc_replicate_se=se, twice_mc_se=2*se,
                all_three_favor_K=bool(np.all(values < 0)),
                scope='finite Monte Carlo stability only; 3 seeds, no confidence guarantee or certificate')


def _checked_law(mean, covariance):
    mean, covariance = np.asarray(mean, float), np.asarray(covariance, float)
    if mean.ndim != 2 or mean.shape[1] != 9 or not len(mean) or not np.isfinite(mean).all():
        raise ValueError('Finite saved means [N,9] required')
    if covariance.shape == (len(mean), 9, 9):
        if not np.array_equal(covariance, np.broadcast_to(covariance[0], covariance.shape)):
            raise ValueError('This frozen experiment declared one shared covariance per fold')
        covariance = covariance[0]
    if covariance.shape != (9, 9) or not np.isfinite(covariance).all():
        raise ValueError('Finite full covariance [9,9] required')
    if not np.array_equal(covariance, covariance.T):
        raise ValueError('Saved covariance must be symmetric without numerical modification')
    return mean, np.linalg.cholesky(covariance)


def joint_gamma_samples(mean, covariance, center, scale, epsilon, *, chunk=256):
    """One common epsilon block, full covariance, original verified decoder.

    Generate epsilon for the complete [S,N,9] fold before chunking. Chunking
    only limits decoder memory; it never changes which draw belongs to an ID.
    """
    mean, chol = _checked_law(mean, covariance)
    center, scale, epsilon = [np.asarray(v, float) for v in (center, scale, epsilon)]
    if center.shape != (9,) or scale.shape != (9,) or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Nine finite native centers and positive TRAIN scales required')
    if epsilon.ndim != 3 or epsilon.shape[1:] != mean.shape or len(epsilon) < 2 or not np.isfinite(epsilon).all():
        raise ValueError('Finite complete-fold common epsilons [S>=2,N,9] required')
    if not isinstance(chunk, int) or chunk < 1:
        raise ValueError('Positive decoder chunk required')
    # Same multiplication order as the historical shared-covariance sampler.
    standardized = mean[None]+epsilon@chol.T
    native = standardized*scale+center
    gains = np.empty((*epsilon.shape[:2], 3), float)
    audit = dict(draw_object_count=0, recovered_schur_failure_count=0,
                 direct_H_refactorization_failure_count=0,
                 failed_factor_draws_high_precision_verified=0,
                 gains_max_absolute_error=0., rows_dropped=0, draws_resampled=0,
                 jitter_added=0., diagonal_floor_added=0., decode_chunks=0)
    for first in range(0, len(epsilon), chunk):
        stop = min(first+chunk, len(epsilon))
        gram, numerical = decode_draws(native[first:stop], verify=True)
        gains[first:stop] = gram_gains(torch.as_tensor(gram, dtype=torch.float64)).numpy()
        for key in ('draw_object_count', 'recovered_schur_failure_count',
                    'direct_H_refactorization_failure_count', 'failed_factor_draws_high_precision_verified'):
            audit[key] += int(numerical[key])
        audit['gains_max_absolute_error'] = max(audit['gains_max_absolute_error'],
            numerical['functional_consistency']['gains_max_absolute_error'])
        audit['decode_chunks'] += 1
    if not np.isfinite(gains).all():
        raise FloatingPointError('Original utility sample is nonfinite; no draw was omitted')
    return gains, audit


def summarize_draw_prefixes(gains, actual, counts=(2000, 10000)):
    gains, actual = np.asarray(gains, float), np.asarray(actual, float)
    if gains.ndim != 3 or gains.shape[1:] != actual.shape or actual.shape[1] != 3:
        raise ValueError('Expected original utility samples [S,N,3] and actual [N,3]')
    out = {}
    for count in counts:
        if not isinstance(count, int) or count < 2 or count > len(gains):
            raise ValueError('Each declared sample count must be an available prefix')
        selected = gains[:count]
        out[count] = dict(actual=actual.copy(), predicted=selected.mean(0),
                          p_null=(selected <= 0).mean(0), utility_crps=fair_crps(selected, actual))
    return out


def policy_summaries(predicted, p_null, actual, ids, allocation):
    """All 36 original rules, each selected separately inside its outer fold."""
    ids, allocation = np.asarray(ids, str), np.asarray(allocation, int)
    rows, masks = [], {}
    for within in (False, True):
        for action, cost in enumerate(WELL_COSTS):
            for fraction in CONFIG['fractions']:
                for ranking, scores in (('expected_gain', predicted[:, action]), ('lowest_null', -p_null[:, action])):
                    selected = np.zeros(len(ids), bool)
                    for fold in np.unique(allocation):
                        ii = np.flatnonzero(allocation == fold)
                        k = int(np.floor(fraction*len(ii)))//(1 if within else cost)
                        selected[ii] = stable_top_k(scores[ii], ids[ii], k)
                    section = 'within_action' if within else 'common_budget'
                    key = f'{section}/{ACTIONS[action]}/{fraction:g}/{ranking}'
                    masks[key] = selected
                    rows.append(dict(key=key, section=section, action=ACTIONS[action],
                        fraction=fraction, ranking=ranking,
                        **_observed(actual[:, action], selected, cost)))
    return rows, masks


PRINCIPAL = 'common_budget/Z1Z2/0.25/expected_gain'


def _score_store(store, ids, allocation):
    action_rows = []
    for j, action in enumerate(ACTIONS):
        value = _metrics(store['actual'][:, j], store['predicted'][:, j], store['p_null'][:, j])
        value.update(action=action, gamma_crps=float(store['utility_crps'][:, j].mean()))
        action_rows.append(value)
    policies, masks = policy_summaries(store['predicted'], store['p_null'], store['actual'], ids, allocation)
    folds = []
    for fold in np.unique(allocation):
        ii = np.flatnonzero(allocation == fold)
        folds.append(dict(fold=int(fold), n=len(ii), actions=[dict(action=ACTIONS[j],
            **_metrics(store['actual'][ii,j], store['predicted'][ii,j], store['p_null'][ii,j]),
            gamma_crps=float(store['utility_crps'][ii,j].mean())) for j in range(3)]))
    return dict(actions=action_rows, folds=folds, policies=policies,
                principal_policy=next(p for p in policies if p['key'] == PRINCIPAL)), masks


def _paired_comparison(left, right, lmask, rmask):
    y = left['actual']
    if not np.array_equal(y, right['actual']):
        raise ValueError('Paired original outcomes differ')
    lm, rm = lmask[PRINCIPAL], rmask[PRINCIPAL]
    null = y[:, 2] <= 0
    return dict(gamma_crps_difference=float((left['utility_crps'][:,2]-right['utility_crps'][:,2]).mean()),
        null_brier_difference=float(((left['p_null'][:,2]-null)**2-(right['p_null'][:,2]-null)**2).mean()),
        mean_prediction_difference=float((left['predicted'][:,2]-right['predicted'][:,2]).mean()),
        principal_total_value_difference=float((y[:,2]*(lm.astype(float)-rm)).sum()),
        principal_per_selected_value_difference=float(y[lm,2].mean()-y[rm,2].mean()),
        principal_fdp_difference=float(null[lm].mean()-null[rm].mean()),
        principal_fpr_difference=float((np.sum(lm&null)-np.sum(rm&null))/null.sum()),
        principal_selected_left=int(lm.sum()), principal_selected_right=int(rm.sum()),
        principal_intersection=int(np.sum(lm&rm)), principal_changed_objects=int(np.sum(lm!=rm)))


def _source_folds(source, manifest):
    ids = np.asarray(manifest['ids'], str)
    if len(ids) != 639 or len(set(ids.tolist())) != 639 or len(manifest['folds']) != 5:
        raise ValueError('This diagnostic uses the existing 639 DEV objects and five folds')
    for key in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed', 'original_contract_changed'):
        if manifest.get(key) is not False:
            raise ValueError('The historical run must preserve '+key)
    if list(manifest['arms']) != list(ARMS):
        raise ValueError('Expected precisely the frozen A/J/K arms')
    allocation, seen, folds = np.full(len(ids), -1), np.zeros(len(ids), int), []
    for record in manifest['folds']:
        f = int(record['fold']); ii = np.asarray(record['test'], int)
        folder = source/'folds'/f'fold_{f}'
        if not (folder/'complete.json').exists(): raise ValueError('Historical fold is incomplete')
        stats = json.loads((folder/'gamma_objective.json').read_text())
        models = {}
        for arm in ARMS:
            ev = folder/'arms'/arm/'evaluation'
            with np.load(ev/'u_predictions.npz', allow_pickle=False) as z:
                if not np.array_equal(z['ids'], ids[ii]): raise ValueError('Saved u IDs do not match fold')
                models[arm] = {k:z[k].copy() for k in ('mean_u', 'covariance_u', 'actual_u')}
            with np.load(ev/'predictions.npz', allow_pickle=False) as z:
                if not np.array_equal(z['ids'], ids[ii]): raise ValueError('Saved Gamma IDs do not match fold')
                models[arm].update({k:z[k].copy() for k in ('actual', 'predicted', 'p_null', 'utility_crps')})
            _checked_law(models[arm]['mean_u'], models[arm]['covariance_u'])
            if arm != ARMS[0]:
                for key in ('actual', 'actual_u', 'covariance_u'):
                    if not np.array_equal(models[arm][key], models[ARMS[0]][key]):
                        raise ValueError('Frozen arms differ in target or covariance: '+key)
        allocation[ii] = f; seen[ii] += 1
        folds.append(dict(fold=f, seed=int(record['seed']), indexes=ii, stats=stats, models=models))
    if not np.all(seen == 1): raise ValueError('Every one of the 639 IDs must appear exactly once')
    return ids, allocation, folds


def _historical_reproduction(source, output, fold):
    records = []
    n = len(fold['indexes'])
    epsilon = np.random.default_rng(fold['seed']+CONFIG['historical_seed_offset']).standard_normal((2000,n,9))
    for arm in ARMS:
        old = fold['models'][arm]
        gains, audit = joint_gamma_samples(old['mean_u'], old['covariance_u'],
            fold['stats']['center'], fold['stats']['scale'], epsilon, chunk=2000)
        recalculated = summarize_draw_prefixes(gains, old['actual'], (2000,))[2000]
        with np.load(source/'folds'/f"fold_{fold['fold']}"/'arms'/arm/'evaluation/predictions.npz') as z:
            exact_samples = np.array_equal(gains, z['utility_samples'])
            error = float(np.max(np.abs(gains-z['utility_samples'])))
        exact = {key:np.array_equal(recalculated[key], old[key]) for key in ('actual','predicted','p_null','utility_crps')}
        if not exact_samples or not all(exact.values()):
            raise ValueError(f'Historical fold0 {arm} recomputation differs: samples={error}, summary={exact}')
        records.append(dict(arm=arm, exact_utility_samples=exact_samples, exact_summary=exact,
                            max_absolute_utility_error=error, numerics=audit))
    write_json(output/'historical_reproduction.json', dict(fold=fold['fold'], records=records))


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output == source or source in output.parents:
        raise FileExistsError('Use a fresh sibling output directory; historical results are unchanged')
    manifest = json.loads((source/'run_manifest.json').read_text())
    # Record the full numerical experiment before any new predictive scoring.
    output.mkdir(parents=True)
    write_json(output/'config.json', CONFIG)
    write_json(output/'run_manifest.json', dict(source=str(source), created_utc=now(), config=CONFIG,
        ids=manifest['ids'], folds=manifest['folds'], arms=list(ARMS),
        sample_axis='3 Monte Carlo integration seeds, not training seeds',
        original_endpoint_changed=False, original_contract_changed=False,
        final_opened=False, fifth_repeat_opened=False, historical_dev=True, formal_certificate=False))
    (output/'PROTOCOL.md').write_text(
        '# Frozen-model Monte Carlo stability\n\n'
        'A/J/K means and full covariances are read from their saved epoch-30 evaluation outputs. '
        'No predictor, scaler, covariance, outcome or selection threshold is fitted. '
        'All 639 existing DEV objects retain their original five folds.\n\n'
        'For each fold, seeds are record seed +310000/+410000/+510000. Each seed generates '
        '10,000 complete-fold joint standard-normal draws, shared across the three models. '
        'The 2,000-draw setting is its exact prefix. Original verified Gram decoding, '
        'ADD_ONE/ADD_TWO net utilities, fair ensemble CRPS and foldwise budget rules are retained. '
        'Decoder chunking does not change the pre-generated draw-to-object assignments; '
        'batched floating-point contractions can differ at rounding precision. Historical '
        'reproduction uses its original full 2,000-draw contraction shape.\n\n'
        'First reproduce the original fold-0 2,000 draws and summaries exactly for all three models. '
        'The continuation screen requires all three 10,000-draw K−J mean CRPS differences to be '
        'negative and the mean improvement to exceed twice sd(delta)/sqrt(3). This is an engineering '
        'screen, not a confidence guarantee. Three integration seeds do not establish training or '
        'sampling-population stability, and the six nested settings are not six independent replicates. '
        'All action metrics, risk metrics and selections are reported, regardless of the screen.\n')
    started = time.monotonic()
    write_json(output/'status.json', dict(state='RUNNING', started_utc=now()))
    try:
        ids, allocation, folds = _source_folds(source, manifest)
        torch.set_num_threads(CONFIG['threads'])
        with threadpool_limits(limits=CONFIG['threads']):
            _historical_reproduction(source, output, next(f for f in folds if f['fold']==0))
            stores = {(offset,count,arm):{key:np.empty((len(ids),3),float)
                      for key in ('actual','predicted','p_null','utility_crps')}
                      for offset in CONFIG['seed_offsets'] for count in CONFIG['sample_counts'] for arm in ARMS}
            for fold in folds:
                for offset in CONFIG['seed_offsets']:
                    epsilon = np.random.default_rng(fold['seed']+offset).standard_normal((10000,len(fold['indexes']),9))
                    for arm in ARMS:
                        old = fold['models'][arm]
                        gains, audit = joint_gamma_samples(old['mean_u'], old['covariance_u'],
                            fold['stats']['center'], fold['stats']['scale'], epsilon,
                            chunk=CONFIG['decode_draw_chunk'])
                        prefixes = summarize_draw_prefixes(gains, old['actual'], tuple(CONFIG['sample_counts']))
                        folder = output/'folds'/f"fold_{fold['fold']}"/f'seed_offset_{offset}'/arm
                        folder.mkdir(parents=True)
                        write_json(folder/'numerical_audit.json', audit)
                        for count, values in prefixes.items():
                            for key, value in values.items(): stores[offset,count,arm][key][fold['indexes']] = value
                            np.savez_compressed(folder/f'predictions_{count}.npz',ids=ids[fold['indexes']],**values)
                        write_json(output/'status.json', dict(state='SCORING', fold=fold['fold'],
                            seed_offset=offset, arm=arm, elapsed_seconds=time.monotonic()-started))
            results, masks = {}, {}
            original = {arm:{key:np.empty((len(ids),3),float) for key in ('actual','predicted','p_null','utility_crps')} for arm in ARMS}
            for fold in folds:
                for arm in ARMS:
                    for key in original[arm]: original[arm][key][fold['indexes']]=fold['models'][arm][key]
            original_masks = {arm:_score_store(original[arm],ids,allocation)[1] for arm in ARMS}
            for offset in CONFIG['seed_offsets']:
                for count in CONFIG['sample_counts']:
                    name=f'offset_{offset}_samples_{count}'; models={}
                    for arm in ARMS:
                        store=stores[offset,count,arm]
                        models[arm], masks[offset,count,arm] = _score_store(store,ids,allocation)
                        original_mask=original_masks[arm][PRINCIPAL]; new_mask=masks[offset,count,arm][PRINCIPAL]
                        models[arm]['vs_original_2000'] = dict(principal_intersection=int(np.sum(original_mask&new_mask)),
                            principal_changed_objects=int(np.sum(original_mask!=new_mask)),
                            predicted_rank_spearman=float(spearmanr(original[arm]['predicted'][:,2],store['predicted'][:,2]).statistic),
                            predicted_gamma_max_absolute_change=float(np.max(np.abs(original[arm]['predicted'][:,2]-store['predicted'][:,2]))))
                        np.savez_compressed(output/f'{name}_{arm}.npz',ids=ids,fold=allocation,
                            **store,policy_keys=np.asarray(list(masks[offset,count,arm])),
                            policy_masks=np.stack(list(masks[offset,count,arm].values())))
                    paired={}
                    for right in ARMS[:2]:
                        left=ARMS[2]
                        paired[left+'__minus__'+right]=_paired_comparison(stores[offset,count,left],stores[offset,count,right],
                            masks[offset,count,left],masks[offset,count,right])
                    results[name]=dict(seed_offset=offset,samples=count,models=models,comparisons=paired)
            differences=[results[f'offset_{offset}_samples_10000']['comparisons'][ARMS[2]+'__minus__'+ARMS[1]]['gamma_crps_difference']
                         for offset in CONFIG['seed_offsets']]
            sampling_ranges={}
            for arm in ARMS:
                setting=[results[f'offset_{offset}_samples_10000']['models'][arm] for offset in CONFIG['seed_offsets']]
                pairwise=[]
                for i,left in enumerate(CONFIG['seed_offsets']):
                    for right in CONFIG['seed_offsets'][i+1:]:
                        lm,rm=masks[left,10000,arm][PRINCIPAL],masks[right,10000,arm][PRINCIPAL]
                        pairwise.append(dict(seed_offsets=[left,right],intersection=int(np.sum(lm&rm)),changed_objects=int(np.sum(lm!=rm))))
                sampling_ranges[arm]=dict(crps_range=[min(s['actions'][2]['gamma_crps'] for s in setting),max(s['actions'][2]['gamma_crps'] for s in setting)],
                    per_selected_value_range=[min(s['principal_policy']['per_selected_net_gain'] for s in setting),max(s['principal_policy']['per_selected_net_gain'] for s in setting)],
                    fdp_range=[min(s['principal_policy']['fdp'] for s in setting),max(s['principal_policy']['fdp'] for s in setting)],
                    principal_pairwise_selection=pairwise)
            summary=dict(complete=True,n=len(ids),settings=results,config=CONFIG,
                continuation_gate=mc_continuation_gate(differences),high_draw_ranges=sampling_ranges,
                elapsed_seconds=time.monotonic()-started,formal_certificate=False,
                scope='Only Monte Carlo stability of fixed historically reused DEV predictions')
            write_json(output/'summary.json',summary)
            write_json(output/'continuation.json', dict(source=str(source),
                proceed_to_basis_contrast=summary['continuation_gate']['passed'],
                reason=summary['continuation_gate'], config=CONFIG,
                interpretation='Monte Carlo engineering continuation only; no training-seed or original-contract pass'))
            lines=['# Frozen A/J/K Monte Carlo stability','','| Seed offset | Draws | Model | ADD_TWO CRPS | Brier | Selected value | FDP | FPR |','|---:|---:|---|---:|---:|---:|---:|---:|']
            for cell in results.values():
                for arm in ARMS:
                    model=cell['models'][arm]; action=model['actions'][2]; policy=model['principal_policy']
                    lines.append(f"| {cell['seed_offset']} | {cell['samples']} | {arm} | {action['gamma_crps']:.8f} | {action['null_brier']:.8f} | {policy['per_selected_net_gain']:.8f} | {policy['fdp']:.6f} | {policy['fpr']:.6f} |")
            lines+=['','Engineering continuation screen: '+str(summary['continuation_gate']['passed']),
                'All settings use the same 639 unique objects. ADD_TWO 25% physical budget selects 79 objects / 158 wells. '
                'This screen addresses integration error, not training-seed stability, new-data generalization or the original contract. '
                'Full action metrics, 36 policy masks, fold metrics, paired differences and numerical audits accompany this report.']
            (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
            write_json(output/'status.json',dict(state='COMPLETE',completed_utc=now(),elapsed_seconds=summary['elapsed_seconds']))
            return summary
    except Exception as exc:
        write_json(output/'status.json',dict(state='FAILED',error=repr(exc),failed_utc=now(),elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(args.source,args.output)


if __name__=='__main__': main()
