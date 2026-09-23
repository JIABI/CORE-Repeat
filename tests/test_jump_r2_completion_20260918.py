"""Metadata, cache and aggregation checks; no assay training or protected data."""
import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'jump_r2_completion_test', PROJECT / 'scripts/run_jump_r2_completion_20260918.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def metadata_fixture(n=100):
    ids = np.asarray([f'ID{i:04}' for i in range(n)])
    groups = np.asarray([f'CHEM{i:04}' for i in range(n)])
    layout = np.asarray([f'PLATE_TUPLE_{i % 2}' for i in range(n)])
    folds = []
    for f, q in enumerate(np.array_split(np.arange(n), 5)):
        pool = np.setdiff1d(np.arange(n), q)
        folds.append(dict(fold=f, test=q.tolist(), test_ids=ids[q].tolist(),
                          fit=pool[:60].tolist(), fit_ids=ids[pool[:60]].tolist(),
                          inner_validation=pool[60:].tolist(),
                          inner_validation_ids=ids[pool[60:]].tolist()))
    return ids, groups, layout, dict(ids=ids.tolist(), folds=folds)


def assert_metadata_partition(ids, groups, parts, old):
    seen = np.zeros(len(ids), int)
    for f, part in enumerate(parts):
        assert set(part) == set(runner.ROLE_KEYS)
        assert all(len(rows) for rows in part.values())
        np.testing.assert_array_equal(np.sort(np.concatenate(list(part.values()))),
                                      np.arange(len(ids)))
        assert set(ids[part['DEV_EVAL']]) == set(old['folds'][f]['test_ids'])
        pool = set(old['folds'][f]['fit_ids']) | set(old['folds'][f]['inner_validation_ids'])
        assert set(ids[np.concatenate([part[k] for k in runner.ROLE_KEYS[:-1]])]) == pool
        for j, a in enumerate(runner.ROLE_KEYS):
            for b in runner.ROLE_KEYS[:j]:
                assert not set(groups[part[a]]) & set(groups[part[b]])
        seen[part['DEV_EVAL']] += 1
    np.testing.assert_array_equal(seen, np.ones(len(ids), int))


def test_metadata_allocation_is_deterministic_group_disjoint_and_preserves_queries():
    ids, groups, layout, old = metadata_fixture()
    parts = runner.make_parts(ids, groups, layout, old)
    assert_metadata_partition(ids, groups, parts, old)
    repeated = runner.make_parts(ids, groups, layout, deepcopy(old))
    for first, second in zip(parts, repeated):
        for role in runner.ROLE_KEYS:
            np.testing.assert_array_equal(first[role], second[role])
    # Metadata row reordering is legal when the historical manifest is aligned.
    order = np.random.default_rng(78).permutation(len(ids))
    shuffled = deepcopy(old)
    shuffled['ids'] = ids[order].tolist()
    changed = runner.make_parts(ids[order], groups[order], layout[order], shuffled)
    for first, second in zip(parts, changed):
        for role in runner.ROLE_KEYS:
            assert set(ids[first[role]]) == set(ids[order][second[role]])


def test_a_chemical_group_crossing_old_query_and_pool_is_rejected():
    ids, groups, layout, old = metadata_fixture()
    groups[25] = groups[0]
    with pytest.raises(ValueError, match='outer fold splits a chemical group'):
        runner.make_parts(ids, groups, layout, old)


def test_ledger_filters_only_jump_dev_and_rejects_missing_or_conflicting_identity():
    ids = np.array(['A', 'B'])
    ledger = [dict(dataset='JUMP_source5_DEV', object_id='A', connectivity='GROUP_A'),
              dict(dataset='JUMP_source5_DEV', object_id='B', connectivity='GROUP_B'),
              dict(dataset='UNRELATED', object_id='A', connectivity='OTHER')]
    np.testing.assert_array_equal(runner.chemical_groups(ids, ledger), ['GROUP_A', 'GROUP_B'])
    with pytest.raises(ValueError, match='does not cover'):
        runner.chemical_groups(np.array(['MISSING']), ledger)
    with pytest.raises(ValueError, match='conflicting'):
        runner.chemical_groups(ids, ledger + [dict(ledger[0], connectivity='DIFFERENT')])


@pytest.mark.skipif(not runner.SOURCE.exists(), reason='Opened local DEV metadata is unavailable')
def test_actual_jump_metadata_plan_keeps_639_old_queries_and_full_reference_capacity():
    # Access only metadata/fingerprints, never Y or any protected export.
    with np.load(runner.SOURCE / 'measurements.npz', allow_pickle=False) as z:
        ids, chem, mask, wells = (z[k].copy() for k in ('ids', 'chem', 'chem_mask', 'well_ids'))
    with runner.LEDGER.open(newline='') as handle:
        groups = runner.chemical_groups(ids, list(csv.DictReader(handle)))
    layout = np.asarray(['|'.join(w.rsplit('::', 1)[0] for w in row) for row in wells])
    old = json.loads(runner.OLD.read_text())
    meta = json.loads((runner.SOURCE / 'measurements.json').read_text())['metadata']
    parts = runner.make_parts(ids, groups, layout, old)
    assert len(ids) == len(set(groups)) == 639
    assert len(set(layout)) == 2 and chem.shape == (639, 513)
    assert_metadata_partition(ids, groups, parts, old)
    for f, part in enumerate(parts):
        expected = [246, 61, 102 if f < 4 else 103, 102, 128 if f < 4 else 127]
        assert [len(part[k]) for k in runner.ROLE_KEYS] == expected
        rows = part['REF_FIT']
        ref = dict(ids=ids[rows], groups=groups[rows], chem=chem[rows], chem_mask=mask[rows],
                   X=np.zeros((len(rows), 8)))
        chosen = runner.choose_reference_ids(ref, meta['chemical'], runner.SEED + 100*f)
        assert len(chosen) == 64 and set(chosen) <= set(ids[rows])


def test_role_adapters_do_not_expose_reference_or_prediction_future_outcomes():
    ids, groups, _, _ = metadata_fixture()
    data = dict(ids=ids, groups=groups, chem=np.ones((100, 513)),
                chem_mask=np.ones(100, bool), Y=np.arange(3200.).reshape(100, 4, 8))
    rows = np.array([2, 7])
    ref = runner.role_rows(data, rows, reference=True)
    assert set(ref) == {'ids', 'groups', 'chem', 'chem_mask', 'X'}
    prediction = runner.input_rows(data, rows, np.zeros((2, 9)))
    assert set(prediction) == {'ids', 'groups', 'chem', 'X', 'mean_u'}
    np.testing.assert_array_equal(ref['X'], data['Y'][rows, 0])
    np.testing.assert_array_equal(prediction['X'], data['Y'][rows, 0])
    assert set(runner.role_rows(data, rows)) == {'ids', 'groups', 'chem', 'chem_mask', 'Y'}


def test_collect_cache_reassembles_vectors_matrices_and_boolean_fields():
    stores, counts = {}, {}
    for q in (np.array([2, 0]), np.array([3, 1])):
        runner.collect_cache(stores, counts, 'arm', q,
                             dict(predicted=q.astype(float), covariance=np.repeat(q[:, None], 3, 1),
                                  coverage=np.repeat((q % 2 == 0)[:, None], 5, 1)), 4)
    np.testing.assert_array_equal(stores['arm']['predicted'], np.arange(4))
    np.testing.assert_array_equal(stores['arm']['covariance'], np.repeat(np.arange(4)[:, None], 3, 1))
    assert stores['arm']['coverage'].dtype == bool
    np.testing.assert_array_equal(counts['arm'], np.ones(4, int))


@pytest.mark.parametrize('value', [np.array(np.nan), np.array([1.]), np.array([1., np.inf])])
def test_collect_cache_rejects_nonfinite_scalar_or_misaligned_outputs(value):
    with pytest.raises(ValueError, match='Invalid per-object'):
        runner.collect_cache({}, {}, 'arm', np.array([0, 1]), dict(predicted=value), 4)


def test_collect_cache_rejects_changed_field_schema():
    stores, counts = {}, {}
    runner.collect_cache(stores, counts, 'arm', np.array([0, 1]),
                         dict(predicted=np.zeros(2), p_null=np.ones(2)), 4)
    with pytest.raises(ValueError):
        runner.collect_cache(stores, counts, 'arm', np.array([2, 3]), dict(predicted=np.zeros(2)), 4)


@pytest.mark.parametrize('q', [np.array([0, 0]), np.array([-1, 0])])
def test_collect_cache_rejects_duplicate_or_negative_query_rows(q):
    with pytest.raises(ValueError):
        runner.collect_cache({}, {}, 'arm', q, dict(predicted=np.zeros(2)), 4)


def test_collect_cache_rejects_repeated_query_rows_across_calls():
    stores, counts = {}, {}
    runner.collect_cache(stores, counts, 'arm', np.array([0, 1]), dict(predicted=np.zeros(2)), 4)
    with pytest.raises(ValueError):
        runner.collect_cache(stores, counts, 'arm', np.array([1, 2]), dict(predicted=np.zeros(2)), 4)


def test_atomic_npz_round_trip_keeps_direct_calibration_arrays_outside_collect(tmp_path):
    path = tmp_path / 'direct_prediction.npz'
    runner.save_npz(path, ids=np.array(['a', 'b']), actual=np.array([-.1, .2]),
                    predicted=np.array([0., .1]), p_null=np.array([.6, .4]),
                    gamma_residuals=np.arange(7.), crps=np.array([.1, .2]),
                    gamma_coverage_by_level=np.ones((2, 5), bool))
    assert not path.with_suffix('.partial.npz').exists()
    saved = runner.read_npz(path)
    common = {k: saved[k] for k in ('crps', 'gamma_coverage_by_level')}
    stores, counts = {}, {}
    runner.collect_cache(stores, counts, 'direct', np.arange(2),
                         dict(common, predicted=saved['predicted'], p_null=saved['p_null']), 2)
    assert set(stores['direct']) == {'crps', 'gamma_coverage_by_level', 'predicted', 'p_null'}


def test_all_31_cache_schemas_are_compatible_with_shared_analysis(tmp_path, monkeypatch):
    """Synthetic saved predictions exercise the real analysis, without fitting."""
    module = runner.comparison
    n = 80
    ids = np.asarray([f'Q{i:03}' for i in range(n)])
    groups = np.asarray([f'CHEM{i:03}' for i in range(n)])
    layout = np.asarray([f'PLATE{i % 2}' for i in range(n)])
    folds = np.repeat(np.arange(5), n // 5)
    actual = np.linspace(-.2, .2, n)
    # Each fold includes both NULL classes for meaningful synthetic summaries.
    actual[::2] *= -1
    predicted = actual*.7 + np.sin(np.arange(n))*.01
    probability = np.clip(.5-predicted, .01, .99)
    joint_names = ('CORE_ORIGINAL', 'CORE_LOCAL_GAUSSIAN', *module.NEW_JOINT)
    direct_names = tuple(f'DIRECT_{scope}_{family}_{suffix}'
                        for scope in ('ACCESS_MATCHED', 'TRAIN_MATCHED')
                        for family in ('RIDGE', 'EXTRATREES', 'HISTGB')
                        for suffix in ('COHERENT', 'CLASSIFIER_RAW', 'CLASSIFIER_CAL'))
    stores, counts = {}, {}
    for f in range(5):
        q = np.flatnonzero(folds == f)
        folder = tmp_path / f'fold_{f}'
        folder.mkdir()
        runner.write_json(folder/'summary.json', dict(random_same_budget={}, fixed={},
            costs=dict(reference_if_all_new_wells=64, reference_if_X_already_available=48)))
        for name in (*joint_names, *direct_names, 'CONSTANT_ACCESS_MATCHED'):
            out = dict(predicted=predicted[q], p_null=probability[q], crps=np.full(len(q), .1),
                       gamma_coverage_by_level=np.ones((len(q), 5), bool))
            if name in joint_names:
                out.update(nll=np.ones(len(q)), energy=np.ones(len(q)),
                           mean_u=np.zeros((len(q), 9)), actual_u=np.ones((len(q), 9)),
                           scatter_u=np.broadcast_to(np.eye(9), (len(q), 9, 9)).copy(),
                           joint_coverage_by_level=np.ones((len(q), 5)),
                           observable_coverage_by_level=np.ones((len(q), 10, 5)),
                           brier=np.square(probability[q]-(actual[q] <= 0)))
                saved_name = {'CORE_ORIGINAL': 'AMP_EMP_LOCAL',
                              'CORE_LOCAL_GAUSSIAN': 'GAUSSIAN'}.get(name, name)
                for offset in (100000, 200000):
                    runner.save_npz(folder/f'{saved_name}_mc{offset}.npz', ids=ids[q],
                                    predicted=predicted[q], p_null=probability[q])
            runner.collect_cache(stores, counts, name, q, out, n)
    assert len(stores) == 31 and all(np.all(value == 1) for value in counts.values())
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'SOURCE', tmp_path)
    # The bootstrap implementation has separate tests; test API wiring here.
    monkeypatch.setattr(module, 'bootstrap_difference',
                        lambda a, b, labels: dict(difference=float(np.mean(a-b))))
    module.analysis(stores, dict(ids=ids, groups=groups, layout=layout), actual, folds,
                    scope_text='SYNTHETIC TEST ONLY', population_text='Synthetic cache test, not assay results.')
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['complete'] and len(summary['metrics']) == 31
    assert set(summary['monte_carlo_sensitivity']) == set(joint_names)
    assert len(summary['cached_fixed_controls']) == 5
    for name in stores:
        saved = runner.read_npz(tmp_path/f'{name}.npz')
        np.testing.assert_array_equal(saved['ids'], ids)
        np.testing.assert_array_equal(saved['fold'], folds)
        assert int(saved['selected_lambda_0.2'].sum()) == 10
        assert 'gamma_residuals' not in saved
    assert (tmp_path/'REPORT.md').exists()
