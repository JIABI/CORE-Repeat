"""One endpoint pass over frozen R4 predictions, selections and FMP outcomes.

This module has no fitting, integration, source-download or authorization API.
The caller supplies the separately staged X and future-outcome artifacts after
the prediction and selection freezes. All population denominators are kept.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .gram_geometry import gram_gains, profiles_to_gram
from .r4_confirmatory_metrics import campaign_budget, paired_utility_bounds, risk_bounds, utility_bounds
from .r4_evaluation import evaluate_campaign, load_selections, select, _write_rows
from .r4_final_model import (CORE_ARM, GAUSSIAN_ARM, DIRECT_ARM, SEEDS,
                             evaluate_saved_predictions)


def _artifact(value):
    if isinstance(value, Mapping):
        return {key: np.asarray(item) for key, item in value.items()}
    with np.load(Path(value), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _identity_order(actual, expected, name):
    actual, expected = np.asarray(actual, str), np.asarray(expected, str)
    if (actual.ndim != 1 or len(set(actual)) != len(actual)
            or len(actual) != len(expected) or set(actual) != set(expected)):
        raise ValueError(name+' must contain exactly the full frozen population once')
    lookup = {oid: i for i, oid in enumerate(actual)}
    return np.array([lookup[oid] for oid in expected], int)


def assemble_four_wells(query, outcomes, ids):
    """Align by immutable identity, then preserve each invalid role as NaN."""
    query, outcomes = _artifact(query), _artifact(outcomes)
    ids = np.asarray(ids, str)
    qrows = _identity_order(query['ids'], ids, 'X artifact')
    orows = _identity_order(outcomes['ids'], ids, 'Future artifact')
    x = np.asarray(query['X'], float)[qrows]
    future = np.asarray(outcomes['future'], float)[orows]
    n = len(ids)
    if x.ndim != 2 or future.shape != (n, 3, x.shape[1]):
        raise ValueError('Expected aligned X[N,D] and future[N,3,D]')
    if list(np.asarray(outcomes['role_order'], str)) != ['Z1', 'Z2', 'V']:
        raise ValueError('Future-role order must remain exactly Z1,Z2,V')
    groups, layout = np.asarray(query['groups'], str)[qrows], np.asarray(query['layout'], str)[qrows]
    for name, expected in (('groups', groups), ('layout', layout)):
        if not np.array_equal(np.asarray(outcomes[name], str)[orows], expected):
            raise ValueError('X and future metadata disagree on '+name)
    eligible = np.asarray(query['eligible'])[qrows]
    if eligible.dtype != bool or eligible.shape != (n,):
        raise ValueError('The saved common X eligibility must be a Boolean full-N vector')
    xp = np.asarray(query.get('x_present', query['eligible']), bool)[qrows]
    xv = np.asarray(query.get('x_valid', query['eligible']), bool)[qrows]
    present = np.asarray(outcomes['present'], bool)[orows]
    valid = np.asarray(outcomes['valid'], bool)[orows]
    if present.shape != (n, 3) or valid.shape != present.shape or xp.shape != (n,) or xv.shape != (n,):
        raise ValueError('Per-role present/valid flags must align with the frozen population')
    if np.any(valid & ~present) or np.any(xv & ~xp) or not np.array_equal(xv, eligible):
        raise ValueError('Missing roles cannot be valid, and X eligibility cannot change after prediction')
    finite_x = np.isfinite(x).all(1) & (np.linalg.norm(x, axis=1) > 0)
    finite_future = np.isfinite(future).all(2) & (np.linalg.norm(future, axis=2) > 0)
    if np.any(xv & ~finite_x) or np.any(valid & ~finite_future):
        raise ValueError('A staged valid role has nonfinite features or a zero norm')
    x[~xv] = np.nan
    future[~valid] = np.nan
    y = np.concatenate((x[:, None], future), axis=1)
    metadata = dict(ids=ids, groups=groups, layout=layout, eligible=eligible,
        role_present=np.column_stack((xp, present)), role_valid=np.column_stack((xv, valid)))
    if 'well_ids' in query and 'well_ids' in outcomes:
        x_wells = np.asarray(query['well_ids'], str)[qrows]
        future_wells = np.asarray(outcomes['well_ids'], str)[orows]
        if x_wells.shape != (n,) or future_wells.shape != (n, 3):
            raise ValueError('Physical well identities must have the fixed role dimensions')
        wells = np.column_stack((x_wells, future_wells))
        if np.any(wells == '') or len(set(wells.ravel())) != 4*n:
            raise ValueError('All four planned physical role keys must be nonempty and unique')
        metadata['well_ids'] = wells
    return y, metadata


def realized_gamma(y):
    """The unchanged four-well endpoint; invalid observations remain unknown."""
    y = np.asarray(y, float)
    if y.ndim != 3 or y.shape[1] != 4:
        raise ValueError('Four prepared roles in X/Z1/Z2/V order are required')
    valid = (np.isfinite(y).all((1, 2)) & (np.linalg.norm(y, axis=2) > 0).all(1)
             & (np.linalg.norm(y[:, :3].mean(1), axis=1) > 0))
    actual = np.full(len(y), np.nan)
    if valid.any():
        gram = profiles_to_gram(torch.as_tensor(y[valid], dtype=torch.float64))
        actual[valid] = gram_gains(gram).numpy()[:, 2]
    return actual


def cost_scenarios(gamma, policies, counts, *, amortization_campaigns=(1, 2, 5, 10, 20),
                   cost_per_well=.01):
    """Resource sensitivity; the two action wells are already charged in Gamma."""
    counts = {key: int(counts[key]) for key in ('TRAIN', 'VALIDATION', 'REF_FIT', 'DIST_CAL')}
    if any(value < 0 for value in counts.values()) or cost_per_well != .01:
        raise ValueError('Nonnegative frozen resource counts and original c=0.01 are required')
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
           for value in amortization_campaigns):
        raise ValueError('Positive integer amortization campaign counts are required')
    train = 4*(counts['TRAIN']+counts['VALIDATION'])
    ref, cal = 4*counts['REF_FIT'], 4*counts['DIST_CAL']
    scenarios = (
        ('existing_resources', 0, 0, 0),
        ('new_REF', 0, ref, 0),
        ('new_REF_CAL', 0, ref, cal),
        ('new_all_fitting_resources', train, ref, cal),
    )
    rows = []
    for policy, values in policies.items():
        # STOP/random/all-eligible require no fitted predictor resource pool.
        if 'p_null' not in values:
            continue
        for scenario, wt, wr, wc in scenarios:
            total_wells = wt+wr+wc
            setup = round(cost_per_well*total_wells, 2)
            for campaigns in amortization_campaigns:
                debit = setup/int(campaigns)
                result = utility_bounds(gamma, values['selected'], additional_reference_cost=debit)
                rows.append(dict(policy=policy, scenario=scenario, amortization_campaigns=int(campaigns),
                    train_validation_new_wells=wt, reference_new_wells=wr, calibration_new_wells=wc,
                    hypothetical_new_setup_wells=total_wells, setup_cost=setup,
                    amortized_setup_cost_per_campaign=debit,
                    total_net_value_lower=result['lower']*len(gamma),
                    total_net_value_upper=result['upper']*len(gamma),
                    value_per_candidate_lower=result['lower'], value_per_candidate_upper=result['upper'],
                    selected_n=result['selected_n'], missing_selected_n=result['missing_selected_n']))
    return rows


def physical_cost_inventory(metadata, policies, counts):
    """Distinct resource roles, planned evaluation wells and action union counts."""
    n = len(metadata['ids'])
    core = policies['CORE']['selected']
    direct = policies['HISTGB_CAL']['selected']
    union = core | direct
    count = {key: int(counts[key]) for key in ('TRAIN', 'VALIDATION', 'REF_FIT', 'DIST_CAL')}
    report = dict(population_n=n, eligibility_n=int(metadata['eligible'].sum()),
        budget=campaign_budget(n, int(metadata['eligible'].sum())),
        shared_development_objects=sum(count.values()), shared_development_compound_wells=4*sum(count.values()),
        development_role_objects=count, query_initial_X_planned_wells=n,
        full_cohort_future_evaluation_planned_wells=3*n,
        all_query_roles_planned_wells=4*n,
        observed_present_role_wells=int(metadata['role_present'].sum()),
        technically_valid_role_wells=int(metadata['role_valid'].sum()),
        primary_policy_action_union_objects=int(union.sum()),
        primary_policy_action_union_Z_wells=2*int(union.sum()),
        primary_pair_full_evaluation_additional_wells=3*n-2*int(union.sum()),
        single_policy={name:dict(selected_n=int(policy['selected'].sum()),
            added_action_wells=2*int(policy['selected'].sum()),
            additional_full_cohort_evaluation_wells=3*n-2*int(policy['selected'].sum()))
            for name, policy in policies.items()},
        controls='Existing frozen DMSO control measurements; no new control purchase assumed',
        initial_X_cost_boundary='Initial X is already available for the main incremental decision estimand',
        physical_counts_are_not_new_purchases=True,
        resource_pool_shared_across_CORE_and_HistGB=True,
        future_evaluation_cost_not_deducted_from_deployment_utility=True,
        action_cost_in_gamma=0.02, action_cost_deducted_again=False,
        monetary_cost='Not available; c=0.01 is a normalized utility unit per well')
    if 'well_ids' in metadata:
        wells = metadata['well_ids']
        report['physical_well_unique_count_verified'] = len(set(wells.ravel()))
        report['primary_action_union_unique_well_count_verified'] = len(set(wells[union, 1:3].ravel()))
    return report


def monte_carlo_sensitivity(prediction_dir, ids, eligible, gamma, policies):
    """Replay all three already saved seeds without choosing a new primary seed."""
    root = Path(prediction_dir)
    ids, eligible = np.asarray(ids, str), np.asarray(eligible, bool)
    erows = np.flatnonzero(eligible)
    expected_ids = ids[erows]
    k = campaign_budget(len(ids), len(erows))['activations']
    records, arrays = [], dict(ids=ids)
    for policy_name, arm in (('CORE', CORE_ARM), ('GAUSSIAN', GAUSSIAN_ARM)):
        primary_masks = {}
        for seed in SEEDS:
            filename = arm+'.npz' if seed == SEEDS[0] else f'{arm}_mc{seed-SEEDS[0]}.npz'
            out = _artifact(root/filename)
            if not np.array_equal(out['ids'].astype(str), expected_ids):
                raise ValueError('Saved MC seed identities differ from the frozen eligible order')
            mean, probability = np.asarray(out['predicted'], float), np.asarray(out['p_null'], float)
            if (mean.shape != expected_ids.shape or probability.shape != expected_ids.shape
                    or not np.isfinite(mean).all() or not np.isfinite(probability).all()
                    or np.any((probability < 0) | (probability > 1))):
                raise ValueError('Every MC seed must contain valid predictions on the same common X set')
            full_mean, full_p = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
            full_mean[erows], full_p[erows] = mean, probability
            for lam in (.2, 0.):
                selected = select(ids, full_mean-lam*full_p, eligible, k)
                saved = np.asarray(out[f'selected_lambda_{lam:g}'])
                if saved.dtype != bool or not np.array_equal(selected[erows], saved):
                    raise ValueError('MC sensitivity must reproduce saved outcome-free selections')
                if seed == SEEDS[0]:
                    primary_masks[lam] = selected
                    name = policy_name if lam == .2 else policy_name+'_LAMBDA0'
                    if name in policies:
                        np.testing.assert_array_equal(selected, policies[name]['selected'])
                base = primary_masks[lam]
                risk, value = risk_bounds(gamma, selected), utility_bounds(gamma, selected)
                delta = paired_utility_bounds(gamma, selected, base)
                observed = selected & np.isfinite(gamma)
                records.append(dict(policy=policy_name, seed=seed, risk_penalty=lam,
                    is_primary_seed=seed == SEEDS[0], selected_n=int(selected.sum()),
                    intersection_vs_primary_seed=int((selected & base).sum()),
                    symmetric_difference_vs_primary_seed=int((selected != base).sum()),
                    observed_selected_null=risk['observed_selected_null'],
                    selected_unknown=risk['selected_unknown'],
                    selected_null_count_lower=risk['observed_selected_null'],
                    selected_null_count_upper=risk['observed_selected_null']+risk['selected_unknown'],
                    predicted_selected_null=float(full_p[selected].sum()),
                    value_per_candidate_lower=value['lower'], value_per_candidate_upper=value['upper'],
                    delta_vs_primary_seed_lower=delta['lower'], delta_vs_primary_seed_upper=delta['upper'],
                    observed_selected_gamma_sum=float(gamma[observed].sum())))
                arrays[f'{policy_name}__seed{seed}__lambda{lam:g}__selected'] = selected
    aggregate = []
    for name in ('CORE', 'GAUSSIAN'):
        for lam in (.2, 0.):
            selected_rows = [r for r in records if r['policy'] == name and r['risk_penalty'] == lam]
            aggregate.append(dict(policy=name, risk_penalty=lam, seeds=list(SEEDS),
                null_count_range_lower=min(r['selected_null_count_lower'] for r in selected_rows),
                null_count_range_upper=max(r['selected_null_count_upper'] for r in selected_rows),
                max_symmetric_difference=max(r['symmetric_difference_vs_primary_seed'] for r in selected_rows),
                primary_seed_unchanged=True, seeds_not_selected_using_outcomes=True))
    return records, arrays, aggregate


def run_primary_analysis(prediction_dir, selection_dir, query, outcomes, output_dir, *,
                         model_dir=None, repeats=10000, amortization_campaigns=(1, 2, 5, 10, 20)):
    """Join once, evaluate frozen policies and caches, then report costs and MC."""
    started = time.monotonic()
    prediction_dir, selection_dir, root = map(Path, (prediction_dir, selection_dir, output_dir))
    if root.exists():
        raise FileExistsError('Preserve the previous primary endpoint analysis')
    freeze = json.loads((selection_dir/'SELECTIONS_FROZEN.json').read_text())
    manifest = json.loads((prediction_dir/'manifest.json').read_text())
    if freeze['status'] != 'FROZEN' or freeze['outcomes_used'] or manifest['state'] != 'COMPLETE':
        raise ValueError('Complete outcome-free predictions and frozen selections are required')
    if manifest.get('samples') != 100000 or tuple(manifest.get('seeds', ())) != SEEDS:
        raise ValueError('The complete frozen 100k three-seed integration is required')
    ids, eligible, policies = load_selections(selection_dir)
    if manifest['population_size'] != len(ids):
        raise ValueError('Full frozen population denominator differs from the predictions')
    if not np.array_equal(np.asarray(manifest['query_ids'], str), ids[eligible]):
        raise ValueError('The same ordered common eligible X set must serve every predictor')
    y, metadata = assemble_four_wells(query, outcomes, ids)
    if not np.array_equal(metadata['eligible'], eligible):
        raise ValueError('X eligibility changed after selection')
    model_dir = Path(manifest['model_dir'] if model_dir is None else model_dir)
    model_manifest = json.loads((model_dir/'manifest.json').read_text())
    if model_manifest['counts'] != dict(TRAIN=434, VALIDATION=108, REF_FIT=181, DIST_CAL=181):
        raise ValueError('Resource counts differ from the frozen final model')
    root.mkdir(parents=True)
    write_json(root/'status.json', dict(state='RUNNING', stage='frozen cache endpoint scoring'))
    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            gamma = realized_gamma(y)
            distribution = evaluate_saved_predictions(prediction_dir, ids[eligible], y[eligible])
            np.testing.assert_allclose(distribution['actual'], gamma[eligible], atol=1e-12, rtol=1e-12, equal_nan=True)
            geometry_valid = np.zeros(len(ids), bool)
            geometry_valid[eligible] = distribution['geometry_valid']
            np.savez_compressed(root/'outcomes.npz', **metadata, actual=gamma,
                null=np.where(np.isfinite(gamma), (gamma <= 0).astype(float), np.nan),
                gamma_valid=np.isfinite(gamma), geometry_valid=geometry_valid)
            distribution_summary = {}
            for arm, values in distribution['arms'].items():
                full_values = {}
                for key, value in values.items():
                    full = np.full((len(ids), *value.shape[1:]), np.nan)
                    full[eligible] = value
                    full_values[key] = full
                np.savez_compressed(root/(arm+'_metrics.npz'), ids=ids, **full_values)
                distribution_summary[arm] = {key:dict(
                    evaluated_count=np.isfinite(value).sum(axis=0),
                    mean=np.divide(np.nansum(value, axis=0), np.isfinite(value).sum(axis=0),
                                   out=np.full(value.shape[1:], np.nan), where=np.isfinite(value).sum(axis=0) > 0))
                    for key, value in full_values.items()}
            write_json(root/'distribution_summary.json', dict(arms=distribution_summary,
                gamma_draws_per_object=100000, gamma_cache_dtype='float64',
                core_crps='fair MC with denominator S*(S-1), all cached samples',
                direct_crps='exact finite CAL-residual law; independent of classifier risk readout',
                direct_calibrated_probability_is_not_the_scalar_law_probability=True,
                energy_and_other_observable_CRPS='Not computed in this cache-only endpoint pass',
                prediction_or_fit_repeated=False))
            write_json(root/'status.json', dict(state='RUNNING', stage='campaign evaluation and sensitivity'))
            campaign = evaluate_campaign(selection_dir, ids, gamma, metadata['groups'], metadata['layout'],
                root/'campaign', repeats=repeats)
            costs = cost_scenarios(gamma, policies, model_manifest['counts'],
                                  amortization_campaigns=amortization_campaigns)
            _write_rows(root/'deployment_cost_sensitivity.csv', costs)
            physical = physical_cost_inventory(metadata, policies, model_manifest['counts'])
            write_json(root/'physical_cost_inventory.json', physical)
            mc_rows, mc_arrays, mc_summary = monte_carlo_sensitivity(prediction_dir, ids, eligible, gamma, policies)
            _write_rows(root/'mc_seed_sensitivity.csv', mc_rows)
            np.savez_compressed(root/'mc_seed_selections.npz', **mc_arrays)
            write_json(root/'mc_seed_summary.json', mc_summary)
        summary = dict(status='COMPLETE', n=len(ids), eligible_x_n=int(eligible.sum()),
            observed_gamma_n=int(np.isfinite(gamma).sum()), geometry_evaluated_n=int(geometry_valid.sum()),
            campaign=campaign, distribution=distribution_summary, mc_sensitivity=mc_summary,
            physical_cost_inventory=physical,
            costs=dict(main='existing_resources', cost_per_well=.01,
                scenarios=['existing_resources', 'new_REF', 'new_REF_CAL', 'new_all_fitting_resources'],
                amortization_campaigns=list(amortization_campaigns),
                interpretation='Sensitivity of this realized campaign to amortized setup cost; not observations of future campaigns',
                action_cost_already_in_gamma=True, resources_shared_between_comparators=True),
            computation=dict(final_model_fit_seconds=model_manifest['elapsed_seconds'],
                frozen_scoring_seconds=manifest.get('elapsed_seconds'),
                scoring_includes_two_joint_arms_and_three_seeds=True,
                production_single_arm_single_seed_time_not_separately_measured=True,
                primary_evaluation_seconds=time.monotonic()-started),
            confirmation_predictions_or_fit_repeated=False)
        write_json(root/'summary.json', summary)
        write_json(root/'status.json', dict(state='COMPLETE', elapsed_seconds=time.monotonic()-started))
        return summary
    except Exception as exc:
        write_json(root/'status.json', dict(state='FAILED', error_type=type(exc).__name__,
            error=str(exc), elapsed_seconds=time.monotonic()-started))
        raise
