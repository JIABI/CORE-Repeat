"""Synthetic aggregation checks; no development outcomes or model fits."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.report_r3_rxrx3_modules_20260920 import (
    ARMS, BASELINES, OFFSETS, aggregate, completed_cells, equal_chemical_scores,
    summarize, _cluster_differences,
)
from opal2.eu_core_experiment import select


def fixture_cache(tmp_path):
    root, source = tmp_path/'r3', tmp_path/'r2'
    root.mkdir(); source.mkdir()
    n = 32
    ids = np.array([f'id{i:03d}' for i in range(n)])
    groups = np.tile(np.array([f'g{i:02d}' for i in range(16)]), 2)
    data = dict(ids=ids, groups=groups, dose=np.repeat([.1, 1.], 16),
                layout=np.tile(np.array(['e0', 'e1']), 16))
    actual = np.tile(np.array([-.1, .2, .3, .4]), 8)
    core = dict(ids=ids, actual=actual, predicted=np.linspace(-.1, .3, n),
                p_null=np.linspace(.6, .1, n), crps=np.full(n, .1),
                nll=np.ones(n), energy=np.ones(n), mean_u=np.zeros((n, 9)),
                actual_u=np.ones((n, 9)), scatter_u=np.tile(np.eye(9), (n, 1, 1)))
    manifest = dict(parts=[], cells=[])
    for index, dose in enumerate((.1, 1.)):
        take = np.flatnonzero(data['dose'] == dose)
        manifest['parts'].append(dict(DEV_EVAL=ids[take].tolist()))
        manifest['cells'].append(dict(cell=index, outer_fold=0, dose_uM=dose))
        folder = root/f'fold_{index}'; folder.mkdir()
        complete = dict(fold=index, outer_fold=0, dose_um=dose,
            query_ids=ids[take].tolist(), costs=dict(reference_if_all_new_wells=8,
                                                  reference_if_X_already_available=6))
        (folder/'complete.json').write_text(json.dumps(complete))
        for arm in ARMS:
            out = {key: value[take].copy() for key, value in core.items()}
            out.update(increment=np.zeros((len(take), 2)),
                       resource_support=np.arange(len(take)) % 2 == 0)
            for lam in (.2, 0.):
                out[f'selected_lambda_{lam:g}'] = select(out['ids'], out['predicted'], out['p_null'], lam)
            np.savez_compressed(folder/(arm+'.npz'), **out)
            for offset in OFFSETS:
                np.savez_compressed(folder/f'{arm}_mc{offset}.npz', ids=out['ids'],
                    predicted=out['predicted'], p_null=out['p_null'])
    cell_ids = np.repeat([0, 1], 16)
    for name in BASELINES:
        saved = {key: core[key].copy() for key in ('ids', 'actual', 'predicted', 'p_null', 'crps')}
        summarize(saved, actual, ids, cell_ids)
        np.savez_compressed(source/(name+'.npz'), **saved)
    return dict(data=data, core=core, manifest=manifest, source=source), root


def test_full_synthetic_aggregation_keeps_arms_doses_budgets_and_strong_controls(tmp_path):
    scope, root = fixture_cache(tmp_path)
    result = aggregate(scope, root=root, report=tmp_path/'report', replicates=12)
    assert result['complete'] and result['n'] == 32
    assert result['n_chemical_groups'] == 16
    assert set(result['metrics']) == set(ARMS)
    assert set(result['by_dose']) == {'0.1', '1.0'}
    assert set(result['r2_baseline_metrics']) == set(BASELINES)
    for arm in ARMS:
        policy = result['metrics'][arm]['policies']['lambda_0.2']
        assert policy['activated'] == 4 and policy['extra_wells'] == 8
        assert len(result['monte_carlo_sensitivity'][arm]) == 2
        assert all(value['policies']['lambda_0.2']['list_symmetric_difference'] == 0
                   for value in result['monte_carlo_sensitivity'][arm])
    for item in result['by_dose'].values():
        assert item['n_conditions'] == 16 and item['n_chemical_groups'] == 16
        assert item['metrics']['CORE']['policies']['lambda_0.2']['activated'] == 2
    assert 'BIO_STRUCTURED minus BIO_RANDOM' in result['paired']
    assert 'BOTH_STRUCTURED minus COND_REP' in result['paired']
    assert result['new_model_fits'] == result['new_monte_carlo_draws'] == 0
    with np.load(root/'CORE.npz') as stored:
        np.testing.assert_array_equal(stored['ids'], scope['data']['ids'])
        np.testing.assert_array_equal(stored['mean_u'], scope['core']['mean_u'])
    assert (tmp_path/'report/REPORT.md').exists()


def test_incomplete_cells_require_explicit_partial_and_never_declare_complete(tmp_path):
    scope, root = fixture_cache(tmp_path)
    (root/'fold_1/complete.json').unlink()
    with pytest.raises(RuntimeError, match='incomplete'):
        completed_cells(scope, root)
    result = aggregate(scope, root=root, report=tmp_path/'report', allow_partial=True, replicates=8)
    assert result['state'] == 'PARTIAL' and not result['complete']
    assert result['n'] == 16 and result['missing_cells'] == [1]
    assert 'INCOMPLETE' in (tmp_path/'report/REPORT.md').read_text()


def test_changed_saved_selection_and_core_predictions_fail(tmp_path):
    scope, root = fixture_cache(tmp_path)
    path = root/'fold_0/CORE.npz'
    with np.load(path) as stored:
        out = {key: stored[key].copy() for key in stored.files}
    out['selected_lambda_0.2'][0] ^= True
    np.savez_compressed(path, **out)
    with pytest.raises(AssertionError):
        aggregate(scope, root=root, report=tmp_path/'report', replicates=8)


def test_equal_chemical_means_do_not_count_repeated_doses_as_independent():
    actual = np.zeros(4)
    out = dict(predicted=np.zeros(4), p_null=np.zeros(4),
               crps=np.array([1., 1., 1., 9.]), selected_lambda_0_2=None)
    out['selected_lambda_0.2'] = np.zeros(4, bool)
    result = equal_chemical_scores(out, actual, np.array(['a', 'a', 'a', 'b']))
    assert result['crps'] == 5.


def test_cluster_resampling_matches_original_group_index_bootstrap():
    labels = np.array(['a', 'a', 'b', 'c', 'c'])
    difference = np.array([.1, -.2, .3, -.4, .5])
    result = _cluster_differences({'x': difference}, labels, replicates=101)['x']
    rng = np.random.default_rng(20260920)
    index = rng.integers(3, size=(101, 3))
    sums = np.array([difference[labels == group].sum() for group in np.unique(labels)])
    sizes = np.array([np.sum(labels == group) for group in np.unique(labels)])
    expected = sums[index].sum(1)/sizes[index].sum(1)
    np.testing.assert_allclose(result['ci95'], np.quantile(expected, [.025, .975]))
    assert result['difference'] == pytest.approx(difference.mean())


def test_one_class_and_single_block_have_honest_missing_auc_interval():
    actual = np.ones(16)
    out = dict(predicted=np.ones(16), p_null=np.zeros(16))
    summary = summarize(out, actual, np.arange(16).astype(str), np.zeros(16))
    assert summary['null_auc'] is None and summary['gamma_spearman'] is None
    assert _cluster_differences({'x': np.ones(16)}, np.zeros(16))['x']['ci95'] is None
