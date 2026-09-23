"""Synthetic checks for R4 artifact alignment, cost accounting and MC replay."""
import numpy as np
import pytest

from opal2.r4_primary_analysis import (assemble_four_wells, realized_gamma, cost_scenarios,
                                     physical_cost_inventory, monte_carlo_sensitivity)
from opal2.r4_final_model import CORE_ARM, GAUSSIAN_ARM, SEEDS
from opal2.r4_evaluation import select


def fixture(n=16):
    ids = np.array([f'q{i:03}' for i in range(n)])
    y = np.random.default_rng(712).normal(size=(n, 4, 12))
    groups = np.array([f'g{i:03}' for i in range(n)])
    layout = np.array([str(i % 3) for i in range(n)])
    wells = np.array([[f'{oid}::{role}' for role in ('X', 'Z1', 'Z2', 'V')] for oid in ids])
    query = dict(ids=ids, groups=groups, layout=layout, X=y[:, 0],
        eligible=np.ones(n, bool), x_valid=np.ones(n, bool), x_present=np.ones(n, bool), well_ids=wells[:, 0])
    outcomes = dict(ids=ids, groups=groups, layout=layout, future=y[:, 1:],
        role_order=np.array(['Z1', 'Z2', 'V']), present=np.ones((n, 3), bool),
        valid=np.ones((n, 3), bool), well_ids=wells[:, 1:])
    return ids, y, query, outcomes


def test_join_identity_and_missing_roles_preserve_population():
    ids, y, query, outcomes = fixture()
    outcomes = {key: value[::-1] if key != 'role_order' else value for key, value in outcomes.items()}
    assembled, metadata = assemble_four_wells(query, outcomes, ids)
    np.testing.assert_array_equal(assembled, y)
    outcomes['present'][0, 1] = False
    outcomes['valid'][0, 1] = False
    assembled, metadata = assemble_four_wells(query, outcomes, ids)
    gamma = realized_gamma(assembled)
    assert len(gamma) == len(ids) and np.isnan(gamma[-1])
    assert metadata['eligible'].all()
    assert np.isfinite(gamma[:-1]).all()
    outcomes['role_order'] = np.array(['Z2', 'Z1', 'V'])
    with pytest.raises(ValueError, match='Future-role'):
        assemble_four_wells(query, outcomes, ids)


def test_physical_well_reuse_and_metadata_mismatch_are_rejected():
    ids, _, query, outcomes = fixture()
    outcomes['well_ids'][0, 0] = query['well_ids'][0]
    with pytest.raises(ValueError, match='unique'):
        assemble_four_wells(query, outcomes, ids)
    outcomes['groups'] = outcomes['groups'].copy()
    outcomes['groups'][0] = 'other-group'
    with pytest.raises(ValueError, match='groups'):
        assemble_four_wells(query, outcomes, ids)


def test_cost_debits_once_amortizes_and_retains_missing_bounds():
    gamma = np.array([.1, .2, np.nan, .3])
    chosen = np.array([True, True, True, False])
    policies = dict(CORE=dict(selected=chosen, p_null=np.ones(4)*.2),
                    HISTGB_CAL=dict(selected=chosen, p_null=np.ones(4)*.2),
                    STOP=dict(selected=np.zeros(4, bool)))
    counts = dict(TRAIN=434, VALIDATION=108, REF_FIT=181, DIST_CAL=181)
    rows = cost_scenarios(gamma, policies, counts, amortization_campaigns=(1, 2))
    assert len(rows) == 16
    row = next(r for r in rows if r['policy'] == 'CORE' and r['scenario'] == 'existing_resources'
               and r['amortization_campaigns'] == 1)
    assert row['total_net_value_lower'] == pytest.approx(.3-1.02)
    assert row['total_net_value_upper'] == pytest.approx(.3+.98)
    new = next(r for r in rows if r['policy'] == 'CORE' and r['scenario'] == 'new_REF_CAL'
               and r['amortization_campaigns'] == 2)
    assert new['setup_cost'] == 14.48
    assert new['amortized_setup_cost_per_campaign'] == 7.24
    assert new['total_net_value_lower'] == pytest.approx(row['total_net_value_lower']-7.24)
    assert next(r for r in rows if r['scenario'] == 'new_all_fitting_resources')['setup_cost'] == 36.16


def test_mc_replays_cached_lists_without_fitting_or_sampling(tmp_path):
    ids, _, query, outcomes = fixture()
    y, metadata = assemble_four_wells(query, outcomes, ids)
    gamma = realized_gamma(y)
    eligible = metadata['eligible']
    means, probability = np.linspace(-.1, .2, len(ids)), np.full(len(ids), .2)
    primary = select(ids, means-.2*probability, eligible, 2)
    policies = dict(CORE=dict(selected=primary), HISTGB_CAL=dict(selected=primary))
    for arm in (CORE_ARM, GAUSSIAN_ARM):
        for seed in SEEDS:
            perturbed = means.copy()
            if seed != SEEDS[0]:
                perturbed[0] = .5
            file = arm+'.npz' if seed == SEEDS[0] else f'{arm}_mc{seed-SEEDS[0]}.npz'
            values = {f'selected_lambda_{lam:g}':select(ids, perturbed-lam*probability, eligible, 2)
                      for lam in (.2, 0.)}
            np.savez_compressed(tmp_path/file, ids=ids, predicted=perturbed, p_null=probability, **values)
    rows, arrays, aggregate = monte_carlo_sensitivity(tmp_path, ids, eligible, gamma, policies)
    assert len(rows) == 12 and len(aggregate) == 4
    assert max(r['symmetric_difference_vs_primary_seed'] for r in rows) == 2
    assert all(r['selected_n'] == 2 for r in rows)
    assert len(arrays) == 13
    physical = physical_cost_inventory(metadata, policies,
        dict(TRAIN=434, VALIDATION=108, REF_FIT=181, DIST_CAL=181))
    assert physical['primary_policy_action_union_Z_wells'] == 4
    assert physical['primary_pair_full_evaluation_additional_wells'] == 44
    assert physical['physical_well_unique_count_verified'] == 64


def test_primary_wrapper_joins_once_keeps_denominator_and_writes_all_reports(monkeypatch, tmp_path):
    from opal2 import r4_primary_analysis as module
    from opal2.biology_kernel_evaluation import write_json
    from opal2.r4_evaluation import freeze_selections

    ids, y, query, outcomes = fixture()
    n = len(ids)
    predictions, selections, model = (tmp_path/name for name in ('predictions', 'selections', 'model'))
    predictions.mkdir()
    model.mkdir()
    mean, probability = np.linspace(-.1, .2, n), np.full(n, .3)
    predictors = {name:dict(expected=mean, p_null=probability) for name in ('CORE', 'HISTGB_CAL', 'GAUSSIAN')}
    freeze_selections(ids, query['eligible'], predictors, selections)
    for arm in (CORE_ARM, GAUSSIAN_ARM):
        for seed in SEEDS:
            filename = arm+'.npz' if seed == SEEDS[0] else f'{arm}_mc{seed-SEEDS[0]}.npz'
            masks = {f'selected_lambda_{lam:g}':select(ids, mean-lam*probability, query['eligible'], 2)
                     for lam in (.2, 0.)}
            np.savez_compressed(predictions/filename, ids=ids, predicted=mean, p_null=probability, **masks)
    write_json(predictions/'manifest.json', dict(state='COMPLETE', samples=100000, seeds=list(SEEDS),
        population_size=n, query_ids=ids.tolist(), model_dir=str(model), elapsed_seconds=12.))
    write_json(model/'manifest.json', dict(counts=dict(TRAIN=434, VALIDATION=108, REF_FIT=181, DIST_CAL=181),
                                         elapsed_seconds=3.))
    outcomes['valid'][-1, 0] = False
    calls = []

    def fake_cached_evaluation(path, supplied_ids, supplied_y):
        calls.append(supplied_ids.copy())
        actual = realized_gamma(supplied_y)
        valid = np.isfinite(actual)
        metric = np.where(valid, .01, np.nan)
        return dict(actual=actual, valid=valid, geometry_valid=valid,
                    arms={CORE_ARM:dict(crps=metric, nll=metric)})

    monkeypatch.setattr(module, 'evaluate_saved_predictions', fake_cached_evaluation)
    result = module.run_primary_analysis(predictions, selections, query, outcomes, tmp_path/'analysis', repeats=4)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], ids)
    assert result['n'] == n and result['observed_gamma_n'] == n-1
    assert result['campaign']['policies']['CORE']['unknown_selected_n'] == 1
    assert result['campaign']['policies']['CORE']['selected_n'] == 2
    for name in ('summary.json', 'outcomes.npz', 'deployment_cost_sensitivity.csv',
                 'mc_seed_sensitivity.csv', 'mc_seed_selections.npz', 'distribution_summary.json'):
        assert (tmp_path/'analysis'/name).is_file()
    assert (tmp_path/'analysis/campaign/policy_results.csv').is_file()
    with pytest.raises(FileExistsError):
        module.run_primary_analysis(predictions, selections, query, outcomes, tmp_path/'analysis', repeats=4)
