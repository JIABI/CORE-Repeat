"""Re-score frozen OPAL2 lists under alternative per-well action/setup costs.

No models are fitted, lists selected, predictive draws generated, or outcomes
imputed. The saved Gamma at c=0.01 shifts by 2*(0.01-c) per action. Missing
Gamma lies in [-1-2*c, 1-2*c], so equal-budget paired contrasts retain exactly
the same identification bounds. Run with --help for portable input/output paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COST = 0.01
ACTION_WELLS = 2
DEFAULT_COSTS = (0.0, 0.005, 0.01, 0.02, 0.03)
from release_paths import DATA, QA
DEFAULT_SOURCE = DATA
POLICIES = {"CORE": "CORE_selected", "HISTGB_CAL": "HISTGB_CAL_selected"}
DEV_POLICIES = {"CORE": "core_original_selected", "HistGB": "histgb_original_selected"}


def cost_shift(cost):
    return ACTION_WELLS * (BASE_COST - cost)


def boolean_column(frame, name):
    """Do not silently treat string 'False' or a missing selection as true."""
    values = frame[name]
    if values.isna().any() or not values.isin([True, False]).all():
        raise ValueError(f"{name} must contain complete boolean values")
    return values.to_numpy(dtype=bool)


def linear_bounds(gamma, weights, cost):
    """Sharp finite-cohort bounds for sum(weights * Gamma_c).

    Availability is determined ONLY by Gamma. Shared unknown outcomes use one
    coefficient in a contrast, so their contributions cancel when weights agree.
    """
    gamma = np.asarray(gamma, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if gamma.shape != weights.shape or not np.isfinite(weights).all():
        raise ValueError("Gamma and finite weights must have the same shape")
    known = np.isfinite(gamma)
    observed = float(np.dot(weights[known], gamma[known] + cost_shift(cost)))
    missing_weights = weights[~known]
    lower, upper = -1.0 - ACTION_WELLS * cost, 1.0 - ACTION_WELLS * cost
    return (
        observed + float(np.sum(np.where(missing_weights >= 0, lower, upper) * missing_weights)),
        observed + float(np.sum(np.where(missing_weights >= 0, upper, lower) * missing_weights)),
    )


def null_bounds(gamma, selected, cost):
    known = np.isfinite(gamma)
    known_null = int(np.sum(selected & known & (gamma + cost_shift(cost) <= 0)))
    unknown_n = int(np.sum(selected & ~known))
    # Normally both signs remain possible; also handle larger CLI costs exactly.
    lower = known_null + unknown_n * int(1.0 - ACTION_WELLS * cost <= 0)
    upper = known_null + unknown_n * int(-1.0 - ACTION_WELLS * cost <= 0)
    return known_null, lower, upper


def ratio_bounds(gamma, numerator_weights, denominator_weights, cost):
    """Marginal outer and sharp shared-outcome ratio bounds, without sampling.

    Only report gain ratios when the denominator is strictly positive throughout
    its identification interval. Sharp endpoints solve min/max(A-rB)=0 by
    deterministic bisection; these are missing-outcome bounds, not confidence CIs.
    """
    num = linear_bounds(gamma, numerator_weights, cost)
    den = linear_bounds(gamma, denominator_weights, cost)
    if den[0] <= 0:
        return (np.nan,) * 4 + ("not_reported_nonpositive_random_lower_bound",)
    corners = [a / b for a in num for b in den]
    outer = min(corners), max(corners)
    sharp = []
    for bound_index in (0, 1):
        left, right = outer
        for _ in range(80):
            middle = (left + right) / 2
            value = linear_bounds(
                gamma, numerator_weights - middle * denominator_weights, cost
            )[bound_index]
            if value > 0:
                left = middle
            else:
                right = middle
        sharp.append((left + right) / 2)
    return (*outer, *sharp, "positive_random_denominator")


def development_tables(dev, curves, costs):
    rows, budget_rows = [], []
    for dataset, frame in dev.groupby("dataset", sort=False):
        gamma = frame.actual_gamma.to_numpy(float)
        if not np.isfinite(gamma).all():
            raise ValueError(f"Development Gamma is incomplete for {dataset}")
        selected = {p: boolean_column(frame, col) for p, col in DEV_POLICIES.items()}
        random_weights = np.zeros(len(frame))
        # Exact expectation matches the frozen quota separately within each unit.
        for indices in frame.groupby("deployment_unit", sort=False).indices.values():
            k = int(selected["CORE"][indices].sum())
            if k != int(selected["HistGB"][indices].sum()):
                raise ValueError(f"Unequal within-unit policy quotas in {dataset}")
            random_weights[indices] = k / len(indices)
        for cost in costs:
            shifted = gamma + cost_shift(cost)
            random_total = float(np.dot(random_weights, shifted))
            for policy, mask in selected.items():
                total = float(shifted[mask].sum())
                other = selected["HistGB" if policy == "CORE" else "CORE"]
                k = int(mask.sum())
                rows.append(dict(
                    dataset=dataset, policy=policy, cost_per_well=cost,
                    n=len(frame), selected_n=k, additional_action_wells=ACTION_WELLS*k,
                    observed_null_n=int(np.sum(shifted <= 0)),
                    observed_null_fraction=float(np.mean(shifted <= 0)),
                    selected_null_n=int(np.sum(shifted[mask] <= 0)),
                    selected_null_fraction=float(np.mean(shifted[mask] <= 0)),
                    exhaustive_total=float(shifted.sum()), selected_total=total,
                    selected_total_per_candidate=total/len(frame),
                    exact_random_total=random_total,
                    selected_minus_random_total=total-random_total,
                    selected_minus_other_policy_total=total-float(shifted[other].sum()),
                    selected_over_random=total/random_total if random_total > 0 else np.nan,
                    ratio_status="positive_random_denominator" if random_total > 0 else "not_reported_nonpositive_random",
                ))
        for policy in selected:
            curve = curves[(curves.dataset == dataset) & (curves.method == policy)].sort_values("selected_n")
            if curve.empty or int(curve.selected_n.iloc[-1]) != len(frame):
                raise ValueError(f"Incomplete exhaustive budget curve for {dataset}/{policy}")
            np.testing.assert_allclose(curve.cumulative_net_gamma.iloc[-1], gamma.sum(), atol=1e-10, rtol=0)
            for cost in costs:
                cumulative = curve.cumulative_net_gamma.to_numpy(float) + curve.selected_n.to_numpy(float)*cost_shift(cost)
                exhaustive = float(cumulative[-1])
                half = exhaustive/2
                first_n = persistent_n = np.nan
                if exhaustive > 0:
                    first_n = int(curve.selected_n.iloc[np.flatnonzero(cumulative >= half)[0]])
                    suffix_min = np.minimum.accumulate(cumulative[::-1])[::-1]
                    persistent_n = int(curve.selected_n.iloc[np.flatnonzero(suffix_min >= half)[0]])
                budget_rows.append(dict(
                    dataset=dataset, policy=policy, cost_per_well=cost, n=len(frame),
                    exhaustive_total=exhaustive, target_fraction=0.5,
                    first_crossing_selected_n=first_n,
                    first_crossing_candidate_fraction=first_n/len(frame),
                    stays_above_selected_n=persistent_n,
                    stays_above_candidate_fraction=persistent_n/len(frame),
                    crossing_status="defined_positive_exhaustive_gain" if exhaustive > 0 else "not_reported_nonpositive_exhaustive_gain",
                ))
    return pd.DataFrame(rows), pd.DataFrame(budget_rows)


def confirmation_tables(frame, costs, setup_source):
    gamma = frame.gamma.to_numpy(float)
    known = np.isfinite(gamma)
    eligible = boolean_column(frame, "eligible_x")
    selected = {p: boolean_column(frame, col) for p, col in POLICIES.items()}
    sizes = {int(mask.sum()) for mask in selected.values()}
    if len(sizes) != 1 or any(np.any(mask & ~eligible) for mask in selected.values()):
        raise ValueError("Confirmation policies must have equal eligible-only frozen budgets")
    k = sizes.pop()
    n, eligible_n = len(frame), int(eligible.sum())
    random_weights = eligible.astype(float) * k / eligible_n
    weights = {p: mask.astype(float) for p, mask in selected.items()}
    weights["EXACT_RANDOM"] = random_weights
    rows, contrast_rows, setup_rows = [], [], []
    resource_columns = ["train_validation_new_wells", "reference_new_wells", "calibration_new_wells", "hypothetical_new_setup_wells"]
    scenarios = setup_source[["scenario", "amortization_campaigns", *resource_columns]].drop_duplicates()
    if scenarios.duplicated(["scenario", "amortization_campaigns"]).any():
        raise ValueError("Saved setup inventory disagrees between policies")
    for cost in costs:
        shifted = gamma + cost_shift(cost)
        for policy, w in weights.items():
            lower, upper = linear_bounds(gamma, w, cost)
            row = dict(
                policy=policy, cost_per_well=cost, n=n, eligible_n=eligible_n,
                gamma_observed_n=int(known.sum()), eligible_gamma_observed_n=int(np.sum(known & eligible)),
                selected_n=k, additional_action_wells=ACTION_WELLS*k,
                gamma_unknown_lower=-1-ACTION_WELLS*cost,
                gamma_unknown_upper=1-ACTION_WELLS*cost,
                eligible_observed_null_n=int(np.sum(known & eligible & (shifted <= 0))),
                eligible_observed_null_fraction=float(np.mean(shifted[known & eligible] <= 0)),
                total_lower=lower, total_upper=upper,
                value_per_candidate_lower=lower/n, value_per_candidate_upper=upper/n,
                value_per_selected_action_lower=lower/k, value_per_selected_action_upper=upper/k,
            )
            if policy in selected:
                mask = selected[policy]
                observed_null, null_lo, null_hi = null_bounds(gamma, mask, cost)
                outer_lo, outer_hi, sharp_lo, sharp_hi, ratio_status = ratio_bounds(gamma, w, random_weights, cost)
                row.update(
                    observed_selected_n=int(np.sum(mask & known)),
                    unknown_selected_n=int(np.sum(mask & ~known)),
                    observed_selected_sum_diagnostic=float(shifted[mask & known].sum()),
                    observed_selected_null=observed_null, selected_null_lower=null_lo,
                    selected_null_upper=null_hi, fdp_lower=null_lo/k, fdp_upper=null_hi/k,
                    ratio_to_random_outer_lower=outer_lo, ratio_to_random_outer_upper=outer_hi,
                    ratio_to_random_sharp_lower=sharp_lo, ratio_to_random_sharp_upper=sharp_hi,
                    ratio_status=ratio_status,
                )
            rows.append(row)
        for policy, comparator in (("CORE", "EXACT_RANDOM"), ("HISTGB_CAL", "EXACT_RANDOM"), ("CORE", "HISTGB_CAL")):
            difference_weights = weights[policy] - weights[comparator]
            lower, upper = linear_bounds(gamma, difference_weights, cost)
            contrast_rows.append(dict(
                policy=policy, comparator=comparator, cost_per_well=cost,
                n=n, selected_n=k, comparator_expected_actions=k,
                unknown_nonzero_contrast_weights_n=int(np.sum(~known & (difference_weights != 0))),
                total_difference_lower=lower, total_difference_upper=upper,
                difference_per_candidate_lower=lower/n, difference_per_candidate_upper=upper/n,
                bound_type="paired_finite_campaign_missing_outcome_identification",
            ))
        for policy, w in weights.items():
            if policy not in selected:
                continue
            own = linear_bounds(gamma, w, cost)
            versus_random = linear_bounds(gamma, w-random_weights, cost)
            for scenario in scenarios.to_dict("records"):
                campaigns = int(scenario["amortization_campaigns"])
                setup_cost = float(scenario["hypothetical_new_setup_wells"])*cost
                debit = setup_cost/campaigns
                setup_rows.append(dict(
                    policy=policy, cost_per_well=cost, **scenario,
                    selected_n=k, n=n, setup_cost=setup_cost,
                    amortized_setup_cost_per_campaign=debit,
                    total_net_value_lower=own[0]-debit, total_net_value_upper=own[1]-debit,
                    value_per_candidate_lower=(own[0]-debit)/n,
                    value_per_candidate_upper=(own[1]-debit)/n,
                    paired_minus_random_total_lower=versus_random[0]-debit,
                    paired_minus_random_total_upper=versus_random[1]-debit,
                    paired_minus_random_per_candidate_lower=(versus_random[0]-debit)/n,
                    paired_minus_random_per_candidate_upper=(versus_random[1]-debit)/n,
                ))
    return pd.DataFrame(rows), pd.DataFrame(contrast_rows), pd.DataFrame(setup_rows)


def verify_results(dev_results, confirmation, contrasts, setups, saved_summary, saved_setup):
    """Reproduce saved R4 at c=.01 and verify equal-budget cost invariance."""
    errors = []

    def check(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
        errors.append(float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)))))

    baseline = confirmation[confirmation.cost_per_well == BASE_COST].set_index("policy")
    for policy in POLICIES:
        expected = saved_summary["policies"][policy]
        for field in ("n", "selected_n", "observed_selected_n", "unknown_selected_n", "observed_selected_null", "value_per_candidate_lower", "value_per_candidate_upper", "fdp_lower", "fdp_upper"):
            check(baseline.loc[policy, field], expected[field])
    for side in ("lower", "upper"):
        check(baseline.loc["EXACT_RANDOM", f"value_per_candidate_{side}"], saved_summary["exact_random_expectation_bounds"][side])
        primary = contrasts[(contrasts.cost_per_well == BASE_COST) & (contrasts.policy == "CORE") & (contrasts.comparator == "HISTGB_CAL")].iloc[0]
        check(primary[f"difference_per_candidate_{side}"], saved_summary["primary"][side])
    baseline_setup = setups[setups.cost_per_well == BASE_COST]
    for row in baseline_setup.to_dict("records"):
        expected = saved_setup[(saved_setup.policy == row["policy"]) & (saved_setup.scenario == row["scenario"]) & (saved_setup.amortization_campaigns == row["amortization_campaigns"])].iloc[0]
        for field in ("amortized_setup_cost_per_campaign", "total_net_value_lower", "total_net_value_upper"):
            check(row[field], expected[field])
    invariance_errors = []
    for _, group in dev_results.groupby(["dataset", "policy"]):
        for field in ("selected_minus_random_total", "selected_minus_other_policy_total"):
            values = group[field].to_numpy()
            check(values, values[0])
            invariance_errors.append(float(np.ptp(values)))
    for _, group in contrasts.groupby(["policy", "comparator"]):
        for field in ("total_difference_lower", "total_difference_upper"):
            values = group[field].to_numpy()
            check(values, values[0])
            invariance_errors.append(float(np.ptp(values)))
    return dict(
        status="PASS", numerical_checks=len(errors),
        maximum_absolute_check_error=max(errors),
        maximum_equal_budget_invariance_range=max(invariance_errors),
        r4_baseline_reproduced_at_cost_per_well=BASE_COST,
        models_refitted=False, lists_reselected=False, sampling_performed=False,
    )


def si_table(confirmation, contrasts, setups):
    rows = []
    for cost in confirmation.cost_per_well.unique():
        c = confirmation[confirmation.cost_per_well == cost].set_index("policy")
        d = contrasts[(contrasts.cost_per_well == cost) & (contrasts.policy == "CORE")].set_index("comparator")
        setup = setups[(setups.cost_per_well == cost) & (setups.policy == "CORE") & (setups.scenario == "new_REF") & (setups.amortization_campaigns == 1)].iloc[0]
        row = dict(cost_per_well=cost, n=int(c.loc["CORE", "n"]), eligible_n=int(c.loc["CORE", "eligible_n"]), selected_actions_per_policy=int(c.loc["CORE", "selected_n"]))
        for policy in ("CORE", "HISTGB_CAL", "EXACT_RANDOM"):
            for field in ("total_lower", "total_upper"):
                row[f"{policy}_{field}"] = c.loc[policy, field]
            if policy in POLICIES:
                for field in ("selected_null_lower", "selected_null_upper", "fdp_lower", "fdp_upper"):
                    row[f"{policy}_{field}"] = c.loc[policy, field]
        for comparator in ("EXACT_RANDOM", "HISTGB_CAL"):
            for side in ("lower", "upper"):
                row[f"CORE_minus_{comparator}_paired_total_{side}"] = d.loc[comparator, f"total_difference_{side}"]
        row.update(
            CORE_new_REF_setup_debit=setup.amortized_setup_cost_per_campaign,
            CORE_new_REF_net_total_lower=setup.total_net_value_lower,
            CORE_new_REF_net_total_upper=setup.total_net_value_upper,
            CORE_new_REF_paired_minus_random_total_lower=setup.paired_minus_random_total_lower,
            CORE_new_REF_paired_minus_random_total_upper=setup.paired_minus_random_total_upper,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def calculation_note(source_dir, costs, validation):
    return f"""# Fixed-list cost sensitivity

These calculations use saved outcomes and frozen policy selections only. No training,
reselection, sampling, or new measurements are performed.

## Reproduce

Run: python cost_sensitivity_fixed_lists.py --source-dir SOURCE --out-dir OUTPUT.
The default source is the released dataset's source_data directory and
the default output is paper/qa/cost_sensitivity (or OPAL2_QA_OUT/cost_sensitivity). --costs accepts
nonnegative per-well utility costs. Costs used here: {', '.join(map(str, costs))}.
The saved c=0.01 case is always checked, even if omitted from requested outputs.

Source directory: {source_dir}. Inputs are measurement_fig2_object_predictions.csv,
measurement_fig2_budget_curves.csv, measurement_fig4_objects.csv,
r4_deployment_cost_sensitivity.csv, and r4_corrected_campaign_summary.json.
The original input files are not modified.

## Calculation and denominators

For a two-well action, Gamma(c) = Gamma(0.01) + 2(0.01-c). Availability is
isfinite(gamma), not the separate paired-site endpoint's observed flag.
R4 has 1,539 candidate objects, 1,527 eligible objects, and 1,520 known Gamma
values. Both frozen policies select 192 actions; CORE has 190 known and 2 unknown
outcomes, whereas HistGB has 191 known and 1 unknown. The NULL event is Gamma(c)<=0.
FDP bounds divide selected NULL counts by 192. Per-candidate values divide totals
by 1,539; per-selected-action values divide by 192. Known-only sums are diagnostic
partial sums and are never compared as complete policy values.

Each unknown Gamma(c) lies in [-1-2c, 1-2c]. Linear bounds select the appropriate
endpoint for each signed coefficient. For a paired policy contrast the coefficient
is the difference of selection weights, so shared unknowns cancel. Exact random
selection has weight 192/1,527 on each eligible object, not a rescaled observed-only
subset. These finite-campaign missing-outcome identification bounds are not sampling
confidence intervals. Equal total action weights imply cost-invariant paired action-only
contrasts, including the bounds, because all observed and unknown outcomes shift together.

Development random expectations match the number selected in every deployment unit.
Budget summaries report the first saved event reaching half the exhaustive gain
and the first event after which the curve stays above it. Neither is a hindsight
maximum. Half-gain crossings are not reported when exhaustive gain is nonpositive.
These are retrospective fixed-ranking replays; cost-specific risk-penalized lists
and predicted P(NULL) are not recomputed. Uniformly shifting E[Gamma] preserves its
ranking, but does not establish invariance of a score containing P(NULL).

Ratios are reported only when random's entire value interval is positive.
Marginal-interval divisions are conservative outer ratio bounds. The separately
labelled sharp ratio bounds account for shared missing outcomes by solving
min/max(A-rB)=0 using deterministic bisection. Ratios, absolute gains, and NULL
counts can change with cost even though equal-budget differences do not.

## Setup costs and outputs

The saved inventory supplies 724 new REF wells, 1,448 new REF+CAL wells, or 3,616
new wells for all fitting resources. Additional setup debit = wells*c/campaigns.
Action-only (existing_resources) has zero setup debit; the two action wells are
already included in Gamma and are not charged twice. New-REF scenarios add a
separate debit to the learned policy. Random and STOP require no fitted-resource
pool. A positive net value versus STOP is different from a positive advantage
over exact random. Setup debits can therefore remove the action-only invariance.

Six CSVs are emitted: development summaries, half-gain budget crossings,
confirmation policy bounds, confirmation paired contrasts, confirmation setup
scenarios, and a compact SI cost table. Blank numeric cells are explicitly
inapplicable (random has no realized selected list, a ratio denominator is not
strictly positive, or exhaustive gain is nonpositive). The validation JSON records
baseline reproduction and equal-budget invariance checks.

Validation: {validation['status']}; {validation['numerical_checks']} numerical checks;
largest absolute discrepancy {validation['maximum_absolute_check_error']:.3g};
largest cost-invariance range {validation['maximum_equal_budget_invariance_range']:.3g}.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--costs", type=float, nargs="+", default=list(DEFAULT_COSTS))
    args = parser.parse_args(argv)
    costs = sorted(set(args.costs))
    if not all(np.isfinite(c) and c >= 0 for c in costs):
        parser.error("Costs must be finite and nonnegative")
    source_dir = args.source_dir.expanduser().resolve()
    out_dir = (args.out_dir or QA / "cost_sensitivity").expanduser().resolve()
    # Always reproduce the saved baseline, without adding unrequested output rows.
    check_costs = sorted(set(costs + [BASE_COST]))

    def read_csv(name):
        return pd.read_csv(source_dir / name, float_precision="round_trip")

    dev = read_csv("measurement_fig2_object_predictions.csv")
    curves = read_csv("measurement_fig2_budget_curves.csv")
    confirmation_source = read_csv("measurement_fig4_objects.csv")
    setup_source = read_csv("r4_deployment_cost_sensitivity.csv")
    with (source_dir / "r4_corrected_campaign_summary.json").open() as handle:
        summary = json.load(handle)
    development, budgets = development_tables(dev, curves, check_costs)
    confirmation, contrasts, setups = confirmation_tables(confirmation_source, check_costs, setup_source)
    validation = verify_results(development, confirmation, contrasts, setups, summary, setup_source)
    validation.update(source_dir=str(source_dir), costs=costs,
                      gamma_known_but_crosssite_unobserved_n=int(np.sum(
                          np.isfinite(confirmation_source.gamma) & ~boolean_column(confirmation_source, "observed"))))
    tables = {
        "cost_sensitivity_development.csv": development,
        "cost_sensitivity_budget_half_crossings.csv": budgets,
        "cost_sensitivity_confirmation_policies.csv": confirmation,
        "cost_sensitivity_confirmation_contrasts.csv": contrasts,
        "cost_sensitivity_confirmation_setup.csv": setups,
        "cost_sensitivity_SI_table.csv": si_table(confirmation, contrasts, setups),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table[table.cost_per_well.isin(costs)].to_csv(out_dir / filename, index=False, float_format="%.17g")
    with (out_dir / "cost_sensitivity_validation.json").open("w") as handle:
        json.dump(validation, handle, indent=2)
        handle.write("\n")
    (out_dir / "cost_sensitivity_calculation_notes.md").write_text(calculation_note(source_dir, costs, validation))
    print(f"Saved {len(tables)} CSVs, calculation notes, and validation to {out_dir}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
