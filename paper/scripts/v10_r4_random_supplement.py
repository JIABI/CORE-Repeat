"""Complete the frozen R4 policy-minus-random sensitivity comparison.

Uses saved outcomes, scores and lists only. Reproduces the original seed,
block ordering, replicate construction, percentile method, and random-budget
matching. Writes only new v10 source-data files, never frozen source results.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def bounds(gamma, weights):
    known = np.isfinite(gamma)
    point = float(np.dot(weights[known], gamma[known]))
    missing = weights[~known]
    lower = point + float(np.where(missing >= 0, missing * -1.02, missing * .98).sum())
    upper = point + float(np.where(missing >= 0, missing * .98, missing * -1.02).sum())
    return lower / len(gamma), upper / len(gamma)


def select(ids, score, eligible, k):
    rows = np.flatnonzero(eligible)
    order = np.lexsort((rows, ids[rows], -score[rows]))
    selected = np.zeros(len(ids), dtype=bool)
    selected[rows[order[:k]]] = True
    return selected


def percentiles(values):
    return [float(x) for x in np.quantile(values, [.025, .5, .975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from release_paths import ROOT as manuscript, RESEARCH, QA
    parser.add_argument('--research-root', type=Path,
                        default=RESEARCH)
    parser.add_argument('--output', type=Path,
                        default=QA / 'r4_random_reproduction')
    args = parser.parse_args()
    report = args.research_root / 'reports/r4_execution_20260921_v1'
    selected_file = args.research_root / 'runs/r4_confirmation_20260921_v1/selections/selections.npz'
    outcomes_file = report / 'primary/campaign/object_results.csv'
    original_file = args.research_root / 'reports/r4_statistical_closure_20260921_v1/corrected_campaign_summary.json'
    before = {f: f.read_bytes() for f in (selected_file, outcomes_file, original_file)}
    original = json.loads(before[original_file])
    objects = list(csv.DictReader(outcomes_file.open()))
    saved = np.load(selected_file, allow_pickle=False)
    ids = np.array([r['object_id'] for r in objects])
    gamma = np.array([float(r['gamma']) for r in objects])
    eligible = np.array([r['eligible_x'] == 'True' for r in objects])
    np.testing.assert_array_equal(ids, saved['ids'])
    np.testing.assert_array_equal(eligible, saved['eligible'])
    names = ('CORE', 'HISTGB_CAL')
    masks = {name: saved[name + '__selected'] for name in names}
    scores = {name: saved[name + '__score'] for name in names}
    for name in names:
        np.testing.assert_array_equal(masks[name], [r[name + '_selected'] == 'True' for r in objects])
    output_rows, diagnostics, trace = [], [], {}
    args.output.mkdir(parents=True, exist_ok=True)
    for block_name, column in (('chemical_connectivity', 'group'), ('library_layout', 'layout')):
        previous = original['uncertainty'][block_name]
        seed, repeats = previous['seed'], previous['repeats']
        labels = np.array([r[column] for r in objects])
        _, inverse = np.unique(labels, return_inverse=True)
        blocks = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
        rng = np.random.default_rng(seed)
        results = {(mode, name): np.empty((repeats, 2))
                   for mode in ('fixed_list', 'new_campaign_topk') for name in names}
        fixed_propensity = {name: np.empty((repeats, 2)) for name in names}
        budget_ratio = {name: np.empty(repeats) for name in names}
        print(f'{block_name}: {repeats} original-seed replicates', flush=True)
        for b in range(repeats):
            index = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
            y, e = gamma[index], eligible[index]
            k = min((len(index) // 4) // 2, int(e.sum()))
            for name in names:
                for mode in ('fixed_list', 'new_campaign_topk'):
                    m = (masks[name][index] if mode == 'fixed_list'
                         else select(ids[index], scores[name][index], e, k))
                    random_weights = e.astype(float) * (int(m.sum()) / int(e.sum()) if e.any() else 0.)
                    results[(mode, name)][b] = bounds(y, m.astype(float) - random_weights)
                # Diagnostic alternative: freezes the original inclusion probability.
                # In a fixed-list resample this generally fails equal-budget matching.
                p_original = masks[name].sum() / eligible.sum()
                fixed_propensity[name][b] = bounds(y, masks[name][index].astype(float) - e * p_original)
                budget_ratio[name][b] = masks[name][index].sum() - e.sum() * p_original
        for (mode, name), values in results.items():
            lo, hi = percentiles(values[:, 0]), percentiles(values[:, 1])
            point = bounds(gamma, masks[name].astype(float) - eligible * masks[name].sum() / eligible.sum())
            if name == 'CORE':
                old = previous['intervals'][mode]
                for endpoint, vals in (('low', lo), ('high', hi)):
                    target = old['CORE_random_difference_' + endpoint]
                    np.testing.assert_allclose(vals, [target[x] for x in ('lower', 'median', 'upper')],
                                               atol=1e-15, rtol=0)
            output_rows.append(dict(block_unit=block_name, mode=mode, policy=name,
                blocks=len(blocks), seed=seed, repeats=repeats,
                identification_lower=point[0], identification_upper=point[1],
                lower_bound_p025=lo[0], lower_bound_p50=lo[1], lower_bound_p975=lo[2],
                upper_bound_p025=hi[0], upper_bound_p50=hi[1], upper_bound_p975=hi[2],
                sensitivity_lower=lo[0], sensitivity_upper=hi[2],
                resamples_lower_bound_positive=int((values[:, 0] > 0).sum()),
                interval_kind='block_percentile_envelope_of_paired_missing_outcome_bounds',
                random_budget='matches_each_policy_selected_count_in_each_replicate'))
            trace[block_name + '__' + mode + '__' + name] = values
        for name, values in fixed_propensity.items():
            lo, hi = percentiles(values[:, 0]), percentiles(values[:, 1])
            diagnostics.append(dict(block_unit=block_name, policy=name, seed=seed, repeats=repeats,
                mode='fixed_original_random_propensity_diagnostic_only',
                lower_bound_p025=lo[0], upper_bound_p975=hi[2],
                selected_minus_random_expected_count_min=float(budget_ratio[name].min()),
                selected_minus_random_expected_count_max=float(budget_ratio[name].max()),
                interpretation='different estimand; resampled action budgets generally unequal'))
    for filename, rows in [('policy_minus_random_intervals.csv', output_rows),
                           ('fixed_propensity_diagnostic.csv', diagnostics)]:
        with (args.output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(args.output / 'replicate_bound_endpoints.npz', **trace)
    assert all(f.read_bytes() == data for f, data in before.items())
    audit = dict(status='COMPLETE', population_n=len(ids), eligible_n=int(eligible.sum()),
        observed_gamma_n=int(np.isfinite(gamma).sum()), selected_n=192,
        sources=[str(f) for f in before], original_sources_unchanged=True,
        all_four_original_CORE_interval_pairs_reproduced=True,
        new_analysis='post-hoc completion of the prespecified random secondary comparison for HistGB',
        models_loaded_or_fitted=False, original_selections_changed=False,
        original_seed_and_replicate_definition=True,
        fixed_list_random_rule='rk = resampled frozen selected count; p_random = rk / resampled eligible count',
        reconstructed_campaign_rule='original frozen score and stable-ID top-k reapplied within each resample only',
        limitations=['Seven layout blocks give dependence sensitivity, not new-plate generalization.',
                     'Fixed original random propensity is an unequal-budget diagnostic, not the reported comparison.',
                     'The fixed-propensity diagnostic is not the equal-budget comparison.'])
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(output_rows, indent=2), flush=True)
    print('ALTERNATIVE_DIAGNOSTIC', json.dumps(diagnostics, indent=2), flush=True)


if __name__ == '__main__':
    main()
