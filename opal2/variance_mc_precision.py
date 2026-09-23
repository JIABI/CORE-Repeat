"""Monte Carlo precision of frozen CORE and dual-branch policy scores.

No fitting or model/rule selection occurs. Per-object independent streams are
shared across arms. Old predictions/selected lists remain untouched. Prefix
budgets are 10k/20k/100k; two seeds independently check numerical stability.
Only score-interval overlap, never outcomes, determines boundary refinement.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz
from .biology_kernel_evaluation import write_json
from .dual_branch_features import apply_increment
from .empirical_radial import draw_radial, fit_radial
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .objective_analysis import fair_crps

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'runs/dual_branch_biology_20260917_v2'
RADIAL = PROJECT/'runs/lincs_empirical_radial_20260916_v1'
ARMS = ('CORE', 'GELU', 'DUAL_STRUCTURED')
SEEDS = (202609171, 202609172)
PREFIXES = (10000, 20000, 100000)
REFINEMENT = (500000, 1000000)
PAIR_INDEX = ((1, 0), (2, 0), (2, 1))
PAIR_NAMES = tuple(ARMS[a]+'_minus_'+ARMS[b] for a, b in PAIR_INDEX)
BLOCK = 10000
BATCHES = 20
INTERVAL_Z = 4.


def gamma_factor(raw):
    """Exact ADD_TWO Gamma without allocating four-by-four Gram arrays."""
    u = np.asarray(raw, dtype=np.float64)
    if u.shape[-1] != 9 or not np.isfinite(u).all():
        raise ValueError('Finite nine-dimensional geometry required')
    a = 1.+u[..., 0]+u[..., 1]
    first_diagonal = np.exp(u[..., 3])
    b = first_diagonal+u[..., 4]
    c = np.exp(u[..., 5])
    vd = np.exp(u[..., 8])
    av2 = a*a+b*b+c*c
    v2 = u[..., 2]**2+u[..., 6]**2+u[..., 7]**2+vd*vd
    result = .5*((a*u[..., 2]+b*u[..., 6]+c*u[..., 7])/np.sqrt(av2*v2)
                 -u[..., 2]/np.sqrt(v2))-.02
    if (not np.isfinite(result).all() or np.any(av2 <= 0) or np.any(v2 <= 0)
            or np.any(first_diagonal**2 == 0) or np.any(c*c == 0) or np.any(vd*vd == 0)):
        raise ValueError('Invalid geometry; no draw clipping or removal')
    return result


def draw_object(mean, factors, center, scale, law, weights, seed, index, samples):
    """IID sample indices; nested prefixes invariant to requested sample size."""
    if samples % BLOCK:
        raise ValueError('Draw count must use complete fixed primitive blocks')
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
    radial_rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
    result = np.empty((samples, len(factors)))
    identity = np.eye(9)[None]
    for start in range(0, samples, BLOCK):
        normal = rng.normal(size=(BLOCK, 1, 9))
        mix = radial_rng.random((BLOCK, 1))
        kernel = radial_rng.random((BLOCK, 1))
        white = draw_radial(law, weights[None], identity, normal, mix, kernel)[:, 0]
        for a, factor in enumerate(factors):
            u = mean+white@factor.T
            result[start:start+BLOCK, a] = gamma_factor(u*scale+center)
    return result


def summarize_draws(gamma, actual):
    """Joint policy-score MC variance and paired common-random-number error.

    The full fair CRPS is reported. Its numerical SE is assessed by 20
    independent fair-CRPS batches. The reported batch-jackknife SE applies
    exactly to their mean; asymptotically it estimates full-score precision.
    Both point estimators are unbiased and their difference is retained.
    """
    gamma = np.asarray(gamma, float)
    n, arms = gamma.shape
    if n % BATCHES or n//BATCHES < 2:
        raise ValueError('Twenty equal independent batches required')
    null = (gamma <= 0).astype(float)
    score = gamma-.2*null
    gamma_c = gamma-gamma.mean(0)
    null_c = null-null.mean(0)
    cov = (gamma_c*null_c).sum(0)/(n-1)
    var_g = gamma.var(0, ddof=1)
    var_i = null.var(0, ddof=1)
    joint_variance = var_g+.04*var_i-.4*cov
    # Million-draw sums in two algebraically equal orders incur roundoff.
    np.testing.assert_allclose(joint_variance, score.var(0, ddof=1), rtol=1e-10, atol=1e-14)
    chunks = gamma.reshape(BATCHES, n//BATCHES, arms)
    batch_crps = np.stack([fair_crps(g, np.full(arms, actual)) for g in chunks])
    # Delete-one-batch jackknife of the mean of independent fair batch scores.
    loo = (batch_crps.sum(0)-batch_crps)/(BATCHES-1)
    jack_se = np.sqrt((BATCHES-1)/BATCHES*np.square(loo-loo.mean(0)).sum(0))
    paired = np.stack([score[:, a]-score[:, b] for a, b in PAIR_INDEX], 1)
    paired_crps = np.stack([batch_crps[:, a]-batch_crps[:, b] for a, b in PAIR_INDEX], 1)
    return dict(predicted=gamma.mean(0), p_null=null.mean(0), score=score.mean(0),
        gamma_variance=var_g, null_variance=var_i, gamma_null_covariance=cov,
        gamma_mc_se=np.sqrt(var_g/n), null_mc_se=np.sqrt(var_i/n),
        score_mc_se=np.sqrt(joint_variance/n),
        score_paired_difference=paired.mean(0), score_paired_mc_se=paired.std(0, ddof=1)/np.sqrt(n),
        crps=fair_crps(gamma, np.full(arms, actual)),
        batch_crps=batch_crps, batch_crps_mean=batch_crps.mean(0), crps_batch_jackknife_se=jack_se,
        paired_crps_batch=paired_crps,
        paired_crps_mc_se=paired_crps.std(0, ddof=1)/np.sqrt(BATCHES))


def boundary_status(ids, scores, se, budget, z=INTERVAL_Z):
    """Use prediction intervals only; this is a numerical stability screen."""
    ids, scores, se = np.asarray(ids), np.asarray(scores), np.asarray(se)
    if scores.ndim != 1 or se.shape != scores.shape or not 0 < budget < len(ids):
        raise ValueError('Finite aligned cohort scores and interior budget required')
    if not np.isfinite(scores).all() or not np.isfinite(se).all() or np.any(se < 0):
        raise ValueError('Finite scores and nonnegative MC SE required')
    order = np.lexsort((ids, -scores)); selected = np.zeros(len(ids), bool); selected[order[:budget]] = True
    lower, upper = scores-z*se, scores+z*se
    min_selected = lower[selected].min(); max_unselected = upper[~selected].max()
    ambiguous = ((selected & (lower <= max_unselected)) |
                 (~selected & (upper >= min_selected)))
    low, high = int(order[budget-1]), int(order[budget])
    pair_se = np.hypot(se[low], se[high])  # independent per-object RNG streams
    return dict(selected=selected, ambiguous=ambiguous,
        boundary_selected_id=str(ids[low]), boundary_unselected_id=str(ids[high]),
        score_gap=float(scores[low]-scores[high]), gap_mc_se=float(pair_se),
        gap_over_se=float((scores[low]-scores[high])/pair_se) if pair_se > 0 else None,
        all_selected_lower_above_unselected_upper=bool(min_selected > max_unselected),
        ambiguous_n=int(ambiguous.sum()))


def _load():
    summary = read_json(SOURCE/'summary.json')
    old = read_json(RADIAL/'summary.json')
    prior = read_npz(RADIAL/'AMP_EMP_LOCAL.npz')
    saved = {arm: read_npz(SOURCE/(arm+'.npz')) for arm in ARMS}
    ids = saved['CORE']['ids']; lookup = {v: i for i, v in enumerate(ids)}
    np.testing.assert_array_equal(ids, prior['ids'])
    cells = []
    for number, cell in enumerate(summary['cells']):
        f, h = cell['fold'], cell['half']
        q = np.array([lookup[v] for v in cell['query_ids']])
        stats = read_json(Path(old['reference_run'])/'folds'/f'fold_{f}'/'preprocessing.json')
        scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
        raw_mean = prior['mean_u'][q]*scale+center
        matrices = [prior['scatter_u'][q]]
        for arm in ARMS[1:]:
            matrices.append(apply_increment(raw_mean, scale, prior['scatter_u'][q], saved[arm]['increment'][q]))
        factors = np.stack([np.linalg.cholesky(s) for s in matrices], 1)
        ref = read_npz(RADIAL/f'cell_{f}_{h}_radial.npz')
        np.testing.assert_array_equal(ids[q], ref['query_ids'])
        cells.append(dict(number=number, fold=f, half=h, q=q, ids=ids[q], budget=cell['budget'],
            center=center, scale=scale, mean=prior['mean_u'][q], factors=factors,
            weights=ref['local_weights'], law=fit_radial(ref['amplitude_radii'])))
    return ids, saved, cells


def _metrics(store, ids, actual, cells, old, samples):
    result = {}; boundaries = []
    masks = np.zeros((len(ids), len(ARMS)), bool)
    for c in cells:
        q = c['q']
        for a, arm in enumerate(ARMS):
            b = boundary_status(ids[q], store['score'][q, a], store['score_mc_se'][q, a], c['budget'])
            masks[q, a] = b.pop('selected'); b.pop('ambiguous')
            boundaries.append(dict(cell=c['number'], fold=c['fold'], half=c['half'], arm=arm, **b))
    for a, arm in enumerate(ARMS):
        mask = masks[:, a]
        b = store['batch_crps'][:, :, a].mean(0)
        result[arm] = dict(crps=float(store['crps'][:, a].mean()),
            crps_batch_mean=float(b.mean()), crps_batch_mc_se=float(b.std(ddof=1)/np.sqrt(BATCHES)),
            brier=float(np.square(store['p_null'][:, a]-(actual <= 0)).mean()),
            selected_n=int(mask.sum()), selected_null=int((actual[mask] <= 0).sum()),
            selected_mean=float(actual[mask].mean()),
            changed_from_original=int(np.count_nonzero(mask != old[arm]['selected'])),
            selected_ids=ids[mask].tolist(), median_score_mc_se=float(np.median(store['score_mc_se'][:, a])),
            original_crps=float(old[arm]['crps'].mean()),
            unresolved_cells=sum(x['arm']==arm and not x['all_selected_lower_above_unselected_upper'] for x in boundaries))
    contrasts = {}
    for p, name in enumerate(PAIR_NAMES):
        a, b = PAIR_INDEX[p]
        blocks = store['paired_crps_batch'][:, :, p].mean(0)
        contrasts[name] = dict(crps_difference=float((store['crps'][:, a]-store['crps'][:, b]).mean()),
            batch_paired_mc_se=float(blocks.std(ddof=1)/np.sqrt(BATCHES)),
            median_object_score_difference_mc_se=float(np.median(store['score_paired_mc_se'][:, p])))
    return dict(samples=samples, arms=result, contrasts=contrasts, boundaries=boundaries), masks


def run(output):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic(); ids, saved, cells = _load(); n = len(ids)
    actual = saved['CORE']['actual']; summaries = {}; all_masks = {}
    spec = dict(arms=ARMS, source=str(SOURCE), seed_ids=SEEDS, n=n, prefixes=PREFIXES,
        refinement=REFINEMENT, numerical_interval_z=INTERVAL_Z,
        refinement_rule='Refine union of query-score 4SE intervals overlapping each cell top-k boundary; outcomes never used',
        source_results_unchanged=True, training=False, policy_reselected_for_diagnostics_only=True,
        crps_se='Delete-one-batch jackknife for average of 20 independent fair-CRPS batch estimators; full fair score separately reported',
        rng='Independent per-object SeedSequence streams; all arms share normal/radius primitives; fixed 10000 draw blocks')
    write_json(root/'PROTOCOL.json', spec)
    for seed in SEEDS:
        stores = {count: {} for count in PREFIXES}
        for c in cells:
            path = root/f'seed_{seed}_cell_{c["number"]}.npz'
            if path.exists():
                cache = read_npz(path)
                np.testing.assert_array_equal(cache['ids'], c['ids'])
            else:
                chunks = {count: [] for count in PREFIXES}
                for j, global_index in enumerate(c['q']):
                    gamma = draw_object(c['mean'][j], c['factors'][j], c['center'], c['scale'],
                        c['law'], c['weights'][j], seed, int(global_index), max(PREFIXES))
                    for count in PREFIXES:
                        chunks[count].append(summarize_draws(gamma[:count], actual[global_index]))
                cache = {'ids': c['ids']}
                for count, records in chunks.items():
                    for key in records[0]:
                        cache[f'{count}_{key}'] = np.stack([r[key] for r in records])
                np.savez_compressed(path, **cache)
            for count in PREFIXES:
                for key, value in cache.items():
                    if not key.startswith(str(count)+'_'): continue
                    field = key.removeprefix(str(count)+'_')
                    if field not in stores[count]:
                        stores[count][field] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
                    stores[count][field][c['q']] = value
            write_json(root/'status.json', dict(state='RUNNING', stage='fixed_budgets', seed=seed,
                cell=c['number']+1, elapsed_seconds=time.monotonic()-start))
            print(f'seed={seed} cell={c["number"]+1}/10 seconds={time.monotonic()-start:.1f}', flush=True)
        seed_summary = {}
        for count, store in stores.items():
            np.savez_compressed(root/f'seed_{seed}_{count}.npz', ids=ids, actual=actual, **store)
            seed_summary[str(count)], masks = _metrics(store, ids, actual, cells, saved, count)
            all_masks[seed, count] = masks
        current = {k: v.copy() for k, v in stores[max(PREFIXES)].items()}
        sample_n = np.full(n, max(PREFIXES), int)
        refinements = []
        for count in REFINEMENT:
            needed = set()
            for c in cells:
                q = c['q']
                for a in range(len(ARMS)):
                    b = boundary_status(ids[q], current['score'][q, a], current['score_mc_se'][q, a], c['budget'])
                    needed.update(q[b['ambiguous']].tolist())
            before = len(needed)
            for c in cells:
                for j, global_index in enumerate(c['q']):
                    if int(global_index) not in needed: continue
                    path = root/f'refine_seed_{seed}_object_{global_index}_{count}.npz'
                    if path.exists():
                        record = read_npz(path)
                    else:
                        gamma = draw_object(c['mean'][j], c['factors'][j], c['center'], c['scale'],
                            c['law'], c['weights'][j], seed, int(global_index), count)
                        record = summarize_draws(gamma, actual[global_index]); np.savez_compressed(path, **record)
                    for key, value in record.items(): current[key][global_index] = value
                    sample_n[global_index] = count
            refined_summary, masks = _metrics(current, ids, actual, cells, saved, 'adaptive_to_'+str(count))
            # Post-adaptive MC SEs are descriptive only, not optional-stopping intervals.
            refined_summary['objects_refined_this_stage'] = before
            refined_summary['sample_count_frequencies'] = {str(k): int((sample_n == k).sum()) for k in np.unique(sample_n)}
            refinements.append(refined_summary)
            print(f'seed={seed} refined_to={count} objects={before}', flush=True)
            if before == 0: break
        np.savez_compressed(root/f'seed_{seed}_refined.npz', ids=ids, actual=actual, sample_n=sample_n, **current)
        all_masks[seed, 'refined'] = masks
        summaries[str(seed)] = dict(fixed=seed_summary, refinement=refinements)
    agreements = {}
    for count in (*PREFIXES, 'refined'):
        agreements[str(count)] = {arm: int(np.count_nonzero(all_masks[SEEDS[0], count][:, a] !=
                              all_masks[SEEDS[1], count][:, a])) for a, arm in enumerate(ARMS)}
    result = dict(state='COMPLETE', seeds=summaries, cross_seed_membership_changes=agreements,
        elapsed_seconds=time.monotonic()-start, **spec)
    write_json(root/'summary.json', result)
    lines = ['# Frozen policy Monte Carlo precision', '',
        'All predictors, laws and cell budgets are unchanged. These are numerical sensitivity results, not model variability or fresh biological validation.', '',
        '| Seed | Draws | Arm | Gamma CRPS | NULL Brier | Selected NULL | Selected mean | Original membership changes |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for seed, s in summaries.items():
        for count, level in s['fixed'].items():
            for arm, m in level['arms'].items():
                lines.append(f'| {seed} | {count} | {arm} | {m["crps"]:.8f} | {m["brier"]:.8f} | {m["selected_null"]}/{m["selected_n"]} | {m["selected_mean"]:.8f} | {m["changed_from_original"]} |')
    lines += ['', 'Refinement uses only predicted scores and numerical uncertainty; actual Gamma never determines additional sampling.',
        'Four-SE overlap is a numerical diagnostic, not a simultaneous statistical or optional-stopping guarantee.',
        'CRPS MC uncertainty uses paired independent batches; sampling uncertainty among compounds remains a separate question.',
        '', '## Cross-seed selected membership changes', '', '```json', json.dumps(agreements, indent=2), '```']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'status.json', dict(state='COMPLETE', elapsed_seconds=time.monotonic()-start))
    print('COMPLETE', root, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1): run(args.output)


if __name__ == '__main__': main()
