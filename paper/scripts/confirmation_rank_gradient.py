"""Descriptive rank and baseline analyses of fixed R4 predictions and outcomes.

No predictor is trained, no frozen list is changed, and no resampling is used.
The outcome regressions below are post-hoc OLS descriptions, not new predictors
or causal adjustments. Four rank groups have sizes 192, 192, 192 and 951.
Each endpoint uses its own finite-outcome mask. Missing-NULL identification
bounds are distinct from complete-case rates and from confidence intervals.

Example (paths may be supplied independently):
  python confirmation_rank_gradient.py --project-root /path/to/opal2_measurement_model_r2 \
    --objects-path /path/to/measurement_fig4_objects.csv --output-dir /path/to/new-results

All output filenames contain 'audited'; existing output files are not replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POLICIES = (('CORE', 'CORE', 'CORE_selected'),
            ('HISTGB', 'HISTGB_CAL', 'HISTGB_CAL_selected'))
SITES = ('MEDINA', 'USC')
BUDGET = 192
PREFIX = 'confirmation_rank_'


def finite_rows(frame, *columns):
    return frame.loc[np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all(axis=1)].copy()


def null_summary(gamma):
    values = np.asarray(gamma, dtype=float)
    known = np.isfinite(values)
    n, observed = len(values), int(known.sum())
    nulls = int((values[known] <= 0).sum())
    missing = n-observed
    return dict(n_gamma_known=observed, n_gamma_missing=missing,
                gamma_mean=float(values[known].mean()), null_known=nulls,
                null_count_lower=nulls, null_count_upper=nulls+missing,
                null_rate_complete_case=nulls/observed,
                null_rate_lower=nulls/n, null_rate_upper=(nulls+missing)/n)


def load_inputs(selections_path, objects_path):
    objects = pd.read_csv(objects_path)
    with np.load(selections_path, allow_pickle=False) as saved:
        columns = dict(object_id=saved['ids'], eligible=saved['eligible'])
        for name, key, _ in POLICIES:
            columns[name+'_score'] = saved[key+'__score']
            columns[name+'_selected_saved'] = saved[key+'__selected']
        scores = pd.DataFrame(columns)
    if not objects.object_id.is_unique or not scores.object_id.is_unique:
        raise ValueError('Expected one row per saved object ID')
    if set(objects.object_id) != set(scores.object_id):
        raise ValueError('Outcome and frozen-score ID sets differ')
    data = objects.merge(scores, on='object_id', validate='one_to_one')
    if len(data) != 1539 or int(data.eligible.sum()) != 1527:
        raise ValueError('This analysis expects the 1,539-object R4 cohort, 1,527 eligible')
    np.testing.assert_array_equal(data.eligible, data.eligible_x)
    for name, _, flag in POLICIES:
        np.testing.assert_array_equal(data[flag], data[name+'_selected_saved'])
        assert int(data[flag].sum()) == BUDGET and data.loc[data[flag], 'eligible'].all()
        assert np.isfinite(data.loc[data.eligible, name+'_score']).all()
    site_complete = np.isfinite(data[[s+'_delta' for s in SITES]]).all(axis=1)
    np.testing.assert_array_equal(np.isfinite(data.delta), site_complete)
    np.testing.assert_allclose(data.loc[site_complete, 'delta'],
                               data.loc[site_complete, [s+'_delta' for s in SITES]].mean(axis=1))
    return data


def rank_table(eligible, policy, flag):
    ranked = eligible.sort_values([policy+'_score', 'object_id'], ascending=[False, True]).copy()
    top, rest = ranked.iloc[:BUDGET], ranked.iloc[BUDGET:]
    assert top[flag].all() and not rest[flag].any()
    rows = []
    edges = (0, BUDGET, 2*BUDGET, 3*BUDGET, len(ranked))
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = ranked.iloc[lo:hi]
        paired = finite_rows(part, 'delta')
        row = dict(policy=policy, tranche=f'{lo+1}-{hi}', rank_start=lo+1,
                   rank_end=hi, n=len(part), score_mean=float(part[policy+'_score'].mean()),
                   **null_summary(part.gamma), n_paired=len(paired),
                   paired_gain=float(paired.delta.mean()),
                   paired_positive_n=int((paired.delta>0).sum()),
                   paired_positive_rate=float((paired.delta>0).mean()))
        for site in SITES:
            known = finite_rows(part, site+'_delta')
            row.update({site+'_n': len(known), site+'_before': float(known[site+'_before'].mean()),
                        site+'_gain': float(known[site+'_delta'].mean())})
        rows.append(row)
    return ranked, pd.DataFrame(rows)


def correlations(eligible):
    rows = []
    for policy, _, _ in POLICIES:
        for endpoint in ('delta', 'MEDINA_delta', 'USC_delta', 'gamma'):
            pair = finite_rows(eligible, policy+'_score', endpoint)
            rho = pair[[policy+'_score', endpoint]].rank(method='average').corr().iloc[0, 1]
            rows.append(dict(policy=policy, endpoint=endpoint, n=len(pair), spearman=float(rho)))
    return pd.DataFrame(rows)


def positive_rates(data):
    # Use all 1,539 candidates for total-denominator bounds; complete-case
    # numerator/denominator remain explicitly separate in each endpoint row.
    rows = []
    for policy, _, flag in POLICIES:
        for group, part in [('selected', data.loc[data[flag]]),
                            ('unselected', data.loc[~data[flag]]), ('all', data)]:
            for endpoint in ('delta', 'MEDINA_delta', 'USC_delta'):
                observed = finite_rows(part, endpoint)
                n, n_obs = len(part), len(observed)
                positive = int((observed[endpoint]>0).sum())
                rows.append(dict(policy=policy, group=group, endpoint=endpoint,
                                 n_total=n, n_observed=n_obs, n_missing=n-n_obs,
                                 positive_n=positive, rate_complete_case=positive/n_obs,
                                 rate_lower=positive/n, rate_upper=(positive+n-n_obs)/n))
    return pd.DataFrame(rows)


def fit_descriptive_ols(before, outcome, selected):
    design = np.column_stack([np.ones(len(before)), before, np.asarray(selected, float)])
    beta, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != 3:
        raise ValueError('Baseline/selection OLS design is not full rank')
    return beta


def baseline_tables(eligible):
    strata, site_ols, paired_ols = [], [], []
    for policy, _, flag in POLICIES:
        for site in SITES:
            before, endpoint = site+'_before', site+'_delta'
            values = finite_rows(eligible, before, endpoint)
            values['quintile'] = pd.qcut(values[before], 5, labels=False)+1
            for quintile, part in values.groupby('quintile'):
                selected, other = part.loc[part[flag]], part.loc[~part[flag]]
                sm, um = float(selected[endpoint].mean()), float(other[endpoint].mean())
                strata.append(dict(policy=policy, site=site, quintile=int(quintile),
                                   baseline_min=float(part[before].min()), baseline_max=float(part[before].max()),
                                   n_selected=len(selected), n_unselected=len(other),
                                   selected_mean=sm, unselected_mean=um, difference=sm-um))
            beta = fit_descriptive_ols(values[before], values[endpoint], values[flag])
            raw = float(values.loc[values[flag], endpoint].mean()-values.loc[~values[flag], endpoint].mean())
            site_ols.append(dict(policy=policy, site=site, n=len(values),
                                 n_selected=int(values[flag].sum()), raw_difference=raw,
                                 intercept=float(beta[0]), baseline_coefficient=float(beta[1]),
                                 adjusted_difference=float(beta[2]),
                                 descriptive_relative_attenuation=(raw-float(beta[2]))/raw))
        pair = finite_rows(eligible, 'delta', 'USC_before', 'MEDINA_before')
        before = (pair.USC_before+pair.MEDINA_before)/2
        beta = fit_descriptive_ols(before, pair.delta, pair[flag])
        raw_s, raw_u = float(pair.loc[pair[flag], 'delta'].mean()), float(pair.loc[~pair[flag], 'delta'].mean())
        for reference, at in [('unselected_mean', float(before.loc[~pair[flag]].mean())),
                              ('all_complete_cases_mean', float(before.mean()))]:
            mu_u = float(beta[0]+beta[1]*at)
            mu_s = mu_u+float(beta[2])
            paired_ols.append(dict(policy=policy, endpoint='paired_delta', n=len(pair),
                                   n_selected=int(pair[flag].sum()), n_unselected=int((~pair[flag]).sum()),
                                   raw_selected_mean=raw_s, raw_unselected_mean=raw_u,
                                   raw_enrichment=raw_s/raw_u, intercept=float(beta[0]),
                                   baseline_coefficient=float(beta[1]), selected_coefficient=float(beta[2]),
                                   standardization_reference=reference, baseline_at=at,
                                   adjusted_selected_mean=mu_s, adjusted_unselected_mean=mu_u,
                                   adjusted_enrichment=mu_s/mu_u))
    return pd.DataFrame(strata), pd.DataFrame(site_ols), pd.DataFrame(paired_ols)


def nested_budget(ranked, policy):
    small = null_summary(ranked.iloc[:BUDGET].gamma)
    added = null_summary(ranked.iloc[BUDGET:2*BUDGET].gamma)
    large = null_summary(ranked.iloc[:2*BUDGET].gamma)
    # A missing object's NULL indicator is identical in both nested lists.
    # With u unknown NULLs in the original list and v in the added tranche:
    # rate_large-rate_small = (b+v-a-u)/(2*k), a/b known NULL counts.
    k, a, b = BUDGET, small['null_known'], added['null_known']
    m, n = small['n_gamma_missing'], added['n_gamma_missing']
    lower, upper = (b-a-m)/(2*k), (b-a+n)/(2*k)
    corners = [(b+v-a-u)/(2*k) for u in (0, m) for v in (0, n)]
    assert min(corners) == lower and max(corners) == upper
    row = dict(policy=policy, original_budget=k, expanded_budget=2*k,
               added_n=k, added_null_known=b, added_gamma_missing=n,
               rate_difference_lower=lower, rate_difference_upper=upper,
               rate_difference_lower_percentage_points=100*lower,
               rate_difference_upper_percentage_points=100*upper)
    for label, result in [('original', small), ('expanded', large)]:
        row.update({label+'_'+key: value for key, value in result.items()})
    return row


def analyze(data):
    eligible = data.loc[data.eligible].copy()
    outputs, nested = {}, []
    for policy, _, flag in POLICIES:
        ranked, table = rank_table(eligible, policy, flag)
        outputs[f'{PREFIX}gradient_{policy}_audited.csv'] = table
        nested.append(nested_budget(ranked, policy))
    strata, site, paired = baseline_tables(eligible)
    outputs.update({f'{PREFIX}correlations_audited.csv': correlations(eligible),
                    f'{PREFIX}positive_rates_audited.csv': positive_rates(data),
                    f'{PREFIX}baseline_strata_audited.csv': strata,
                    f'{PREFIX}site_ols_audited.csv': site,
                    f'{PREFIX}paired_ols_audited.csv': paired,
                    f'{PREFIX}nested_budget_audited.csv': pd.DataFrame(nested)})
    audit = dict(cohort_n=len(data), eligible_n=len(eligible), frozen_budget=BUDGET,
                 tranche_sizes=[192, 192, 192, 951], frozen_lists_equal_top_192=True,
                 finite_endpoint_n={col: len(finite_rows(eligible, col))
                                    for col in ['gamma', 'delta', 'MEDINA_delta', 'USC_delta']},
                 gamma_known_without_paired_site_n=int((np.isfinite(data.gamma)&~np.isfinite(data.delta)).sum()),
                 sampling='No new resampling, predictor fitting or policy-score calculation',
                 analysis='Post-hoc descriptive analysis of saved scores and observed outcomes',
                 rank_groups='First three groups of 192; final remainder of 951; not four equal-width tranches',
                 null_definition='Gamma <= 0; only finite Gamma values contribute known NULL indicators',
                 null_bounds='Known NULL / total through (known NULL + missing Gamma) / total; identification bounds, not confidence intervals',
                 masks='Each correlation/site summary uses its own finite endpoint; paired endpoint requires both sites',
                 positive_rate_denominator='Complete-case rates use endpoint-observed objects; bounds use all 1539 candidates, selected or unselected as labelled',
                 spearman='Pearson correlation of average ranks; eligible objects, finite score and endpoint',
                 site_ols='Unweighted OLS: site_delta = intercept + beta * site_before + tau * selected; own-site complete cases',
                 paired_ols='Unweighted OLS: paired_delta = intercept + beta * ((USC_before + MEDINA_before)/2) + tau * selected; paired complete cases',
                 standardization='At baseline b_ref, mu_unselected=intercept+beta*b_ref; mu_selected=mu_unselected+tau; enrichment=mu_selected/mu_unselected',
                 standardization_references=['Policy-specific unselected mean paired baseline', 'All paired-complete-case mean paired baseline'],
                 nested_rate_difference='For original k and added k: (known_added + unknown_added_NULL - known_original - unknown_original_NULL)/(2*k); optimize jointly over shared missing indicators',
                 interpretation='Conditional associations, not causal effects or a decomposition of regression to the mean; shared initial X creates mathematical coupling. No sampling intervals are computed. The 384-object analysis is a retrospective replay, not the frozen 192-object confirmation.')
    return outputs, audit


def main():
    from release_paths import DATA, RESEARCH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root', type=Path, default=RESEARCH)
    parser.add_argument('--selections-path', type=Path, help='Override the saved selections NPZ path')
    parser.add_argument('--objects-path', type=Path,
                        default=DATA/'measurement_fig4_objects.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='Calculate and check without writing outputs')
    args = parser.parse_args()
    selection = (args.selections_path or args.project_root/'runs/r4_confirmation_20260921_v1/selections/selections.npz').expanduser().resolve()
    objects = args.objects_path.expanduser().resolve()
    data = load_inputs(selection, objects)
    outputs, audit = analyze(data)
    audit['sources'] = dict(selections=str(selection), object_endpoints=str(objects), script=str(Path(__file__).resolve()))
    out = args.output_dir.expanduser().resolve()
    names = list(outputs)+[PREFIX+'audit.json', PREFIX+'calculation_notes.md']
    if not args.validate_only:
        existing = [str(out/name) for name in names if (out/name).exists()]
        if existing:
            raise FileExistsError('Choose a new output directory; refusing to replace: '+', '.join(existing))
        out.mkdir(parents=True, exist_ok=True)
        for name, table in outputs.items():
            table.to_csv(out/name, index=False, mode='x', float_format='%.17g')
        with (out/(PREFIX+'audit.json')).open('x') as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
            stream.write('\n')
        notes = '\n'.join([
            '# Confirmation rank-gradient calculations', '',
            'The tables use saved R4 predictions and outcomes. Frozen 192-object lists are unchanged.',
            'Rank groups contain 192, 192, 192 and 951 eligible objects, respectively.', '',
            '## Denominators',
            'Gamma and each site endpoint use their own finite-outcome masks. Paired-site summaries require both sites.',
            'NULL complete-case rates divide known NULLs by known Gamma outcomes. Lower/upper bounds divide known NULLs / (known NULLs + missing Gamma) by the total rank-group size. They are missing-data identification bounds, not confidence intervals.',
            'Positive-rate tables give observed numerators and denominators separately from bounds over all candidates. Correlation tables give endpoint-specific sample sizes.', '',
            '## Baseline descriptions',
            'Single-site OLS: site increment = intercept + beta * site baseline + tau * selected.',
            'Paired OLS: paired increment = intercept + beta * mean(USC baseline, MEDINA baseline) + tau * selected, using all 1,515 paired complete cases.',
            'For a common baseline b, predicted unselected mean = intercept + beta*b, and predicted selected mean adds tau. Their ratio is reported at two references: the policy-specific unselected mean baseline, and the all-complete-case mean baseline.',
            'These are separate standardization choices, not two independent estimates. No OLS uncertainty intervals are computed.', '',
            '## Nested budget replay',
            'The larger set is the first 384 objects of the same saved ranking. Its first 192 are the frozen list. A missing NULL indicator in that shared portion must take the same value in both sets.',
            'With a/b known NULLs and u/v unknown NULLs in the original/added 192 objects, the rate difference is (b+v-a-u)/384. Joint lower/upper bounds use all feasible u and v, preserving nesting.', '',
            '## Interpretation',
            'Score tranches and baseline-adjusted comparisons describe associations. They do not identify a causal selection effect, prove a smooth relationship, rule out regression to the mean, or partition its causes. The outcomes share the initial FMP profile X; adjustment for a measured baseline does not remove all mathematical coupling.',
            'The 384-object result is retrospective budget replay on this cohort, not a new prospective confirmation or a guarantee for a future budget.', '',
            'Input paths, formulas and sample counts are recorded in confirmation_rank_audit.json.',
        ])+'\n'
        with (out/(PREFIX+'calculation_notes.md')).open('x') as stream:
            stream.write(notes)
    print(json.dumps(dict(output_dir=str(out), written=not args.validate_only,
                          files=names, sample_sizes=audit['finite_endpoint_n']), indent=2))


if __name__ == '__main__':
    main()
