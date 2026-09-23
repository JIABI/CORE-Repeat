"""R4 final role isolation and outcome-free Monte Carlo cache checks."""
import inspect

import numpy as np
import pytest

from opal2 import r4_final_model as module
from opal2.objective_analysis import fair_crps


def test_partitions_are_exact_and_group_disjoint():
    ids = np.array([f'D{i:04}' for i in range(904)])
    groups = np.array([f'G{i:04}' for i in range(904)])
    order = np.random.default_rng(20260921).permutation(904)
    pieces = np.split(order, [434, 542, 723])
    parts = dict(zip(module.ROLE_COUNTS, [ids[rows].tolist() for rows in pieces]))
    checked = module.normalize_partitions(ids, groups, {'partitions': parts})
    assert {key: len(rows) for key, rows in checked.items()} == module.ROLE_COUNTS
    groups[pieces[1][0]] = groups[pieces[0][0]]
    with pytest.raises(ValueError, match='Chemical group'):
        module.normalize_partitions(ids, groups, parts)
    groups = ids.copy()
    parts['DIST_CAL'][0] = 'protected-new-identity'
    with pytest.raises(ValueError, match='outside'):
        module.normalize_partitions(ids, groups, parts)


def test_query_rejects_future_fields_and_development_overlap():
    fitted = dict(manifest=dict(all_development_ids=['dev'], all_development_groups=['dev-group']))
    query = dict(ids=np.array(['q']), groups=np.array(['q-group']), X=np.ones((1, 12)),
                 chem=np.ones((1, 513)), chem_mask=np.ones(1, bool))
    module.validate_query(query, fitted)
    with pytest.raises(ValueError, match='future'):
        module.validate_query(dict(query, Y=np.ones((1, 4, 12))), fitted)
    with pytest.raises(ValueError, match='overlaps'):
        module.validate_query(dict(query, groups=np.array(['dev-group'])), fitted)
    with pytest.raises(ValueError, match='eligibility'):
        module.validate_query(dict(query, X=np.zeros((1, 12))), fitted)
    assert 'y' not in inspect.signature(module.score).parameters


def test_selection_preserves_original_population_budget_with_missing_x():
    ids = np.array(['b', 'a', 'c', 'd'])
    selected = module.select(ids, np.ones(4), np.zeros(4), population_size=16)
    assert ids[selected].tolist() == ['b', 'a']
    assert module.select(ids, np.ones(4), np.zeros(4), population_size=80).all()


def test_mc_cache_preserves_every_draw_and_exact_fair_crps(tmp_path):
    n, samples = 3, 200
    stats = dict(u_scale=np.ones(9), u_center=np.zeros(9))
    mean = np.zeros((n, 9))
    scatter = np.tile(np.eye(9)[None]*.1, (n, 1, 1))
    path = tmp_path/'gamma.npy'
    out = module.integrate(mean, scatter, stats, seed=44, samples=samples,
                           gamma_path=path, chunk_size=2)
    cache = np.load(path, mmap_mode='r')
    assert cache.shape == (n, samples) and cache.dtype == np.float64
    np.testing.assert_allclose(cache.mean(1), out['predicted'], atol=1e-16, rtol=0)
    np.testing.assert_array_equal((cache <= 0).mean(1), out['p_null'])
    np.testing.assert_array_equal(np.quantile(cache, (1-module.LEVELS)/2, axis=1).T,
                                  out['gamma_lower_by_level'])
    actual = np.array([.1, -.1, .0])
    cached = np.array([fair_crps(cache[i, :, None], actual[i:i+1])[0] for i in range(n)])
    np.testing.assert_allclose(cached, fair_crps(cache.T, actual), rtol=1e-14, atol=1e-14)
    replay = module.integrate(mean, scatter, stats, seed=44, samples=samples, chunk_size=2)
    np.testing.assert_array_equal(replay['predicted'], out['predicted'])
    with pytest.raises(FileExistsError):
        module.integrate(mean, scatter, stats, seed=44, samples=samples, gamma_path=path)


def test_later_metrics_use_frozen_cache_and_keep_singular_gamma(monkeypatch, tmp_path):
    import joblib
    from opal2.biology_kernel_evaluation import write_json
    from opal2.empirical_radial import fit_radial, radial_ppf

    n = 3
    ids = np.array(['q0', 'q1', 'q2'])
    stats = dict(u_scale=np.ones(9), u_center=np.zeros(9))
    mean = np.zeros((n, 9))
    scatter = np.tile(np.eye(9)[None]*.1, (n, 1, 1))
    law = fit_radial(np.array([1., 2., 3.]))
    weights = np.full((n, 3), 1/3)
    write_json(tmp_path/'manifest.json', dict(state='COMPLETE', query_ids=ids.tolist()))
    write_json(tmp_path/'preprocessing.json', stats)
    joblib.dump(dict(law=law, radial_weights=weights), tmp_path/'query_distribution.joblib')
    for arm in (module.CORE_ARM, module.GAUSSIAN_ARM):
        empirical = arm == module.CORE_ARM
        out = module.integrate(mean, scatter, stats, seed=33, samples=200,
            law=law if empirical else None, weights=weights if empirical else None,
            gamma_path=tmp_path/(arm+'_gamma_samples.npy'))
        out.update(mean_u=mean, scatter_u=scatter,
            joint_squared_radius_by_level=radial_ppf(law, weights, module.LEVELS)**2)
        np.savez_compressed(tmp_path/(arm+'.npz'), ids=ids, **out)
    np.savez_compressed(tmp_path/(module.DIRECT_ARM+'.npz'), ids=ids,
        predicted=np.zeros(n), gamma_residuals=np.array([-.2, .1, .2]),
        p_null_raw=np.full(n, .5), p_null=np.full(n, .4))
    y = np.random.default_rng(37).normal(size=(n, 4, 12))
    y[1] = 1.  # Defined Gamma, undefined nine-coordinate Schur geometry.
    y[2] = np.nan

    def forbidden(*args, **kwargs):
        raise AssertionError('Evaluation must not repeat integration or fitting')

    monkeypatch.setattr(module, 'integrate', forbidden)
    monkeypatch.setattr(module, 'fit_complete_eu_core', forbidden)
    evaluated = module.evaluate_saved_predictions(tmp_path, ids, y)
    assert evaluated['valid'].tolist() == [True, True, False]
    assert evaluated['geometry_valid'].tolist() == [True, False, False]
    assert evaluated['actual'][1] == pytest.approx(-.02)
    for arm in (module.CORE_ARM, module.GAUSSIAN_ARM):
        assert np.isfinite(evaluated['arms'][arm]['crps'][:2]).all()
        assert np.isnan(evaluated['arms'][arm]['nll'][1:]).all()
