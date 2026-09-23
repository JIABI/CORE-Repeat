"""Engineering regressions for mixed CORE-copy and active-scorer cell schemas."""
import json

import numpy as np
import pytest

from opal2.dual_branch_experiment import ARMS, collect_completed_cells


REQUIRED = ('actual', 'fold', 'mean_u', 'actual_u', 'predicted', 'p_null',
            'nll', 'energy', 'crps', 'selected', 'increment',
            'resource_support', 'joint_coverage_by_level')


def fixture_cells(root):
    ids = np.asarray([f'object_{i}' for i in range(7)])
    membership = (np.array([4, 1, 6]), np.array([0, 5]), np.array([3, 2]))
    cells, expected = [], {}
    for a, arm in enumerate(ARMS):
        marker = 1000*a+np.arange(len(ids), dtype=float)
        expected[arm] = {k: marker+j/100 for j, k in enumerate(REQUIRED)}
        expected[arm].update(
            fold=np.array([1, 0, 2, 2, 0, 1, 0]),
            mean_u=marker[:, None]+np.arange(9)[None]/10,
            actual_u=marker[:, None]+np.arange(9)[None]/20,
            increment=marker[:, None]+np.arange(2)[None]/30,
            joint_coverage_by_level=marker[:, None]+np.arange(5)[None]/40,
            selected=np.arange(len(ids)) % 2,
            resource_support=np.arange(len(ids)) % 3 != 0,
            common_optional=marker[:, None]+np.arange(3)[None]/50)
    for fold, rows in enumerate(membership):
        folder = root/f'cell_{fold}_0'
        folder.mkdir(parents=True)
        cells.append(dict(fold=fold, half=0, query_ids=ids[rows].tolist()))
        for arm in ARMS:
            payload = {k: v[rows] for k, v in expected[arm].items()}
            # CORE has this old diagnostic everywhere. A calibration fallback
            # copies it only in one cell; active scorer cells do not produce it.
            if arm == 'CORE' or fold == 1:
                payload['legacy_coordinate_interval'] = np.full((len(rows), 9), 987654.)
            if fold == 2 and arm != 'CORE':
                payload['active_only_diagnostic'] = np.full(len(rows), -123456.)
            payload['scalar_metadata'] = np.asarray(89.)
            np.savez_compressed(folder/(arm+'.npz'), ids=ids[rows], **payload)
    return ids, cells, expected


def edit_cell(root, cells, cell_index, arm, change):
    cell = cells[cell_index]
    path = root/f'cell_{cell["fold"]}_{cell["half"]}'/(arm+'.npz')
    with np.load(path) as z:
        values = {k: z[k].copy() for k in z.files}
    change(values)
    np.savez_compressed(path, **values)


def test_intersection_discards_partial_legacy_fields_and_preserves_all_primary_rows(tmp_path):
    ids, cells, expected = fixture_cells(tmp_path)
    stores = collect_completed_cells(tmp_path, ids, cells)
    for arm in ARMS:
        for key, value in expected[arm].items():
            np.testing.assert_array_equal(stores[arm][key], value)
            assert np.isfinite(stores[arm][key]).all()
        assert 'scalar_metadata' not in stores[arm]
        if arm != 'CORE':
            assert set(stores[arm]) == set(expected[arm])
            assert 'legacy_coordinate_interval' not in stores[arm]
            assert 'active_only_diagnostic' not in stores[arm]
        else:
            np.testing.assert_array_equal(stores[arm]['legacy_coordinate_interval'], np.full((7, 9), 987654.))
    audit = json.loads((tmp_path/'aggregation_fields.json').read_text())
    assert audit['CORE']['omitted_legacy_fields'] == []
    for arm in ARMS[1:]:
        assert audit[arm]['omitted_legacy_fields'] == ['active_only_diagnostic', 'legacy_coordinate_interval']
        assert audit[arm]['all_objects_assigned_once']


@pytest.mark.parametrize('key', REQUIRED)
def test_missing_required_field_fails_before_aggregation(tmp_path, key):
    ids, cells, _ = fixture_cells(tmp_path)
    edit_cell(tmp_path, cells, 1, ARMS[-1], lambda values: values.pop(key))
    with pytest.raises(ValueError, match='Missing core evaluation fields'):
        collect_completed_cells(tmp_path, ids, cells)
    assert not (tmp_path/'aggregation_fields.json').exists()


def test_missing_object_fails(tmp_path):
    ids, cells, _ = fixture_cells(tmp_path)
    cells[2]['query_ids'] = cells[2]['query_ids'][:-1]
    for arm in ARMS:
        def remove(values):
            for key, value in list(values.items()):
                if value.shape[:1] == (2,):
                    values[key] = value[:-1]
        edit_cell(tmp_path, cells, 2, arm, remove)
    with pytest.raises((AssertionError, ValueError)):
        collect_completed_cells(tmp_path, ids, cells)


def test_duplicate_object_inside_one_cell_fails_even_when_no_object_is_missing(tmp_path):
    ids, cells, _ = fixture_cells(tmp_path)
    cells[0]['query_ids'].append(cells[0]['query_ids'][0])
    for arm in ARMS:
        def duplicate(values):
            for key, value in list(values.items()):
                if value.shape[:1] == (3,):
                    values[key] = np.concatenate((value, value[:1]), axis=0)
        edit_cell(tmp_path, cells, 0, arm, duplicate)
    with pytest.raises((AssertionError, ValueError)):
        collect_completed_cells(tmp_path, ids, cells)


def test_duplicate_object_across_cells_fails(tmp_path):
    ids, cells, _ = fixture_cells(tmp_path)
    # Add an existing ID without removing any other identity, so this checks
    # duplicate assignment itself rather than accidentally relying on a gap.
    cells[2]['query_ids'].append(cells[0]['query_ids'][0])
    for arm in ARMS:
        def duplicate(values):
            for key, value in list(values.items()):
                if value.shape[:1] == (2,):
                    values[key] = np.concatenate((value, value[:1]), axis=0)
            values['ids'][-1] = cells[0]['query_ids'][0]
        edit_cell(tmp_path, cells, 2, arm, duplicate)
    with pytest.raises((AssertionError, ValueError)):
        collect_completed_cells(tmp_path, ids, cells)


@pytest.mark.parametrize('key', ('mean_u', 'increment', 'joint_coverage_by_level', 'common_optional'))
def test_common_field_shape_conflict_fails(tmp_path, key):
    ids, cells, _ = fixture_cells(tmp_path)
    edit_cell(tmp_path, cells, 2, 'GELU', lambda values: values.update({key: values[key][:, :-1]}))
    with pytest.raises(ValueError, match='Cell field shapes differ'):
        collect_completed_cells(tmp_path, ids, cells)


def test_identity_order_conflict_and_nonfinite_primary_fail(tmp_path):
    ids, cells, _ = fixture_cells(tmp_path)
    edit_cell(tmp_path, cells, 1, 'CORE', lambda values: values.update(ids=values['ids'][::-1]))
    with pytest.raises(AssertionError):
        collect_completed_cells(tmp_path, ids, cells)
    edit_cell(tmp_path, cells, 1, 'CORE', lambda values: values.update(ids=values['ids'][::-1]))
    edit_cell(tmp_path, cells, 1, 'CORE', lambda values: values['nll'].__setitem__(0, np.nan))
    with pytest.raises(ValueError, match='Nonfinite completed evaluation field'):
        collect_completed_cells(tmp_path, ids, cells)
