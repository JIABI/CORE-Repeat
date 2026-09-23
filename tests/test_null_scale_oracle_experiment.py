"""Null simulation is in measurement geometry, with independent re-realization."""
import numpy as np

from opal2.conditional_joint_error_experiment import observable_forward
from opal2.empirical_radial import fit_radial
from opal2.objective_analysis import fair_crps
from opal2.null_scale_oracle_experiment import (
    scores_from_sorted, synthetic_geometries, search_targets, evaluate_selected_union,
)


def fixture(n=3):
    rng = np.random.default_rng(788)
    mean = rng.normal(size=(n, 9))*.1
    factor = np.eye(9)[None]+rng.normal(size=(n, 9, 9))*.03
    scatter = .1*(factor@factor.swapaxes(-1, -2))
    stats = dict(u_scale=np.linspace(.4, .7, 9), u_center=np.linspace(-.2, .2, 9))
    law = fit_radial(np.linspace(1., 5., 13))
    weights = rng.uniform(size=(n, 13)); weights /= weights.sum(1, keepdims=True)
    return mean, scatter, stats, law, weights


def test_prefix_crps_equals_original_fair_crps_for_each_realization():
    rng = np.random.default_rng(314)
    draws = rng.normal(size=36); targets = rng.normal(size=9)
    actual = scores_from_sorted(np.sort(draws), targets)
    expected = fair_crps(np.broadcast_to(draws[:, None], (36, 9)), targets)
    np.testing.assert_allclose(actual, expected, atol=8e-16, rtol=3e-14)


def test_pseudotruth_is_original_geometry_and_second_realization_is_independent():
    mean, scatter, stats, law, weights = fixture()
    out = synthetic_geometries(mean, scatter, stats, law, weights, 3, replicates=6)
    assert out['first_geometry'].shape == out['second_geometry'].shape == (3, 6, 9)
    assert not np.array_equal(out['first_geometry'], out['second_geometry'])
    for role in ('first', 'second'):
        gamma = observable_forward(out[role+'_geometry']*stats['u_scale']+stats['u_center'])[0]
        np.testing.assert_allclose(gamma, out[role+'_gamma'], atol=0, rtol=0)
    again = synthetic_geometries(mean, scatter, stats, law, weights, 3, replicates=6)
    for key in out: np.testing.assert_array_equal(out[key], again[key])


def test_shared_grid_scalar_nested_and_zero_support_exact():
    mean, scatter, stats, law, weights = fixture()
    s = synthetic_geometries(mean, scatter, stats, law, weights, 2, replicates=4)
    gamma = np.column_stack((s['first_gamma'][:, 0], s['first_gamma']))
    result = search_targets(mean, scatter, stats, law, weights, gamma,
                            np.array([True, False, True]), 2, samples=64)
    for arm in ('CORE', 'H1', 'H2'):
        np.testing.assert_array_equal(result[arm+'_indices'][1], np.zeros(5, int))
    for i in (0, 2):
        scores = result['search_crps'][i]; t = np.arange(5)
        assert np.all(scores[result['H2_indices'][i], t] <= scores[result['H1_indices'][i], t]+1e-15)
        assert np.all(scores[result['H1_indices'][i], t] <= scores[0]+1e-15)


def test_union_evaluation_returns_exact_core_for_disabled_rows_and_no_missing_scores():
    mean, scatter, stats, law, weights = fixture(n=2)
    s = synthetic_geometries(mean, scatter, stats, law, weights, 9, replicates=3)
    gamma = np.column_stack((s['first_gamma'][:, 0], s['first_gamma']))
    geometry = np.concatenate((s['first_geometry'][:, :1], s['first_geometry']), axis=1)
    search = search_targets(mean, scatter, stats, law, weights, gamma,
                            np.array([True, False]), 9, samples=64)
    out, union = evaluate_selected_union(mean, scatter, stats, law, weights, gamma, geometry,
        s['second_gamma'], s['second_geometry'], search, 9, samples=200, mc_blocks=4)
    assert union[1] == 1
    for arm in ('CORE', 'H1', 'H2'):
        assert out[arm]['crps'].shape == (2, 4)
        assert out[arm]['second_crps'].shape == (2, 3)
        for key in out[arm]:
            assert np.isfinite(out[arm][key]).all()
            np.testing.assert_array_equal(out[arm][key][1], out['CORE'][key][1])


def test_second_outcome_cannot_change_first_fit_or_first_scores():
    mean, scatter, stats, law, weights = fixture(n=1)
    s = synthetic_geometries(mean, scatter, stats, law, weights, 14, replicates=3)
    gamma = np.column_stack((s['first_gamma'][:, 0], s['first_gamma']))
    geometry = np.concatenate((s['first_geometry'][:, :1], s['first_geometry']), axis=1)
    search = search_targets(mean, scatter, stats, law, weights, gamma, np.array([True]), 14, samples=64)
    out, _ = evaluate_selected_union(mean, scatter, stats, law, weights, gamma, geometry,
        s['second_gamma'], s['second_geometry'], search, 14, samples=200, mc_blocks=4)
    changed, _ = evaluate_selected_union(mean, scatter, stats, law, weights, gamma, geometry,
        s['second_gamma']+.1, s['second_geometry']+.3, search, 14, samples=200, mc_blocks=4)
    for arm in ('CORE', 'H1', 'H2'):
        for key in out[arm]:
            if not key.startswith('second_'):
                np.testing.assert_array_equal(out[arm][key], changed[arm][key])
        assert not np.array_equal(out[arm]['second_crps'], changed[arm]['second_crps'])


def test_candidate_union_sharing_does_not_change_existing_truth_evaluation():
    mean, scatter, stats, law, weights = fixture(n=1)
    s = synthetic_geometries(mean, scatter, stats, law, weights, 19, replicates=5)
    gamma = np.column_stack((s['first_gamma'][:, 0], s['first_gamma']))
    geometry = np.concatenate((s['first_geometry'][:, :1], s['first_geometry']), axis=1)
    large = search_targets(mean, scatter, stats, law, weights, gamma, np.array([True]), 19, samples=64)
    small = search_targets(mean, scatter, stats, law, weights, gamma[:, :2], np.array([True]), 19, samples=64)
    for arm in ('CORE', 'H1', 'H2'):
        np.testing.assert_array_equal(large[arm+'_indices'][:, :2], small[arm+'_indices'])
    out, _ = evaluate_selected_union(mean, scatter, stats, law, weights, gamma, geometry,
        s['second_gamma'], s['second_geometry'], large, 19, samples=200, mc_blocks=4)
    little, _ = evaluate_selected_union(mean, scatter, stats, law, weights, gamma[:, :2], geometry[:, :2],
        s['second_gamma'][:, :1], s['second_geometry'][:, :1], small, 19, samples=200, mc_blocks=4)
    for arm in ('CORE', 'H1', 'H2'):
        for key in out[arm]:
            t = 1 if key.startswith('second_') else 2
            np.testing.assert_array_equal(out[arm][key][:, :t], little[arm][key])
