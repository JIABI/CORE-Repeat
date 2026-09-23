"""Scope and paired-integration tests, not shortened scientific experiments."""
import json
import numpy as np
import pytest

from opal2.hierarchical_stability_experiment import validate_partition, score_model, CONFIG
from opal2.hierarchical_geometry_experiment import CONFIG as PREVIOUS_CONFIG
from opal2.hierarchical_geometry import sample_joint_coordinates
from opal2.gram_oof_experiment import assign_folds
from opal2.gram_oof_experiment import decode_draws
from opal2.gram_geometry import gram_gains
import torch


def test_repeated_partitions_keep_every_compound_and_first_matches_previous():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(150, 10))
    ids = np.array([f'c{i:03}' for i in range(150)])
    old, allocation, _ = assign_folds(x, ids)
    for repeat, seed in enumerate(CONFIG['partition_seeds']):
        folds, assigned, _ = assign_folds(x, ids, seed=seed)
        seen = np.zeros(len(ids), int)
        for record in folds:
            fit, valid, test = validate_partition(record, len(ids))
            seen[test] += 1
            assert set(fit).isdisjoint(valid) and set(valid).isdisjoint(test)
        np.testing.assert_array_equal(seen, 1)
        if repeat == 0:
            assert old == folds
            np.testing.assert_array_equal(allocation, assigned)
    with pytest.raises(ValueError):
        validate_partition(dict(fit=[0, 1], inner_validation=[1], test=[2]), 3)


def test_residual_recipe_and_covariance_noise_pairing_unchanged():
    assert CONFIG['residual'] == PREVIOUS_CONFIG['residual']
    assert CONFIG['training_seed_offsets'] == [401, 1401, 2401]
    rng = np.random.default_rng(17)
    mean = rng.normal(size=(7, 9))
    factor = rng.normal(size=(9, 9))
    covariance = factor@factor.T+np.eye(9)
    correction = .4*np.tanh(rng.normal(size=mean.shape))
    base = sample_joint_coordinates(mean, covariance, 20, 99)
    revised = sample_joint_coordinates(mean+correction, covariance, 20, 99)
    np.testing.assert_allclose(revised-base,
        np.broadcast_to(correction, base.shape), atol=1e-14)


def test_saved_global_policy_uses_uniform_subset_expectations(tmp_path, monkeypatch):
    # Small generated arrays only exercise the changed save/constant-policy path.
    for key, value in dict(samples=40, bootstrap=20, random_subsets=20).items():
        monkeypatch.setitem(CONFIG, key, value)
    rng = np.random.default_rng(13)
    actual_u = .2*rng.normal(size=(20, 9))
    actual_grams, _ = decode_draws(actual_u[None])
    actual_grams = actual_grams[0]
    gains = gram_gains(torch.tensor(actual_grams)).numpy()
    stats = dict(u_center=np.zeros(9), u_scale=np.ones(9))
    score_model(tmp_path, np.array([f'c{i}' for i in range(20)]),
        np.zeros((20, 9)), np.eye(9)*.04, stats, actual_u, actual_grams,
        gains, np.ones(9), 18, dict(arm='GLOBAL_GEOMETRY'), global_model=True)
    report = json.loads((tmp_path/'metrics.json').read_text())
    assert 'row_trace' not in report['policy']
    rows = report['policy']['common_budget']
    assert all(row['uniform_random_subset_exact_expectation'] for row in rows)
    assert all(row['selected_ids'] is None for row in rows)
    row = next(row for row in rows if row['action']=='Z1Z2' and row['fraction']==.25)
    np.testing.assert_allclose(row['total_net_gain'], row['selected_n']*gains[:, 2].mean())
