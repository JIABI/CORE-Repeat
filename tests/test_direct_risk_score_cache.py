"""Isolation and frame tests for expanded inner calibration score caches."""
from copy import deepcopy

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from opal2 import direct_risk_score_cache as module
from opal2.conditional_residual_information import error_targets


def test_centered_ecdf_ties_and_out_of_reference_support():
    np.testing.assert_array_equal(module.centered_ecdf([1., 2., 2., 4.], [0., 1., 2., 3., 4., 5.]),
                                  [-1., -.75, 0., .5, .75, 1.])
    with pytest.raises(ValueError):
        module.centered_ecdf([], [1.])


def fixture():
    rng = np.random.default_rng(1710)
    n, d = 150, 15
    ids = np.asarray([f'object{i:03d}' for i in range(n)])
    groups = np.asarray([f'group{i:03d}' for i in range(n)])
    y = rng.normal(size=(n, 4, d)) * rng.lognormal(0, .3, (n, 1, 1))
    data = dict(ids=ids, groups=groups, Y=y,
                feature_names=np.asarray([f'Cells_Texture_Variance_{i}' for i in range(d)]))
    metadata = dict(units=[dict(id=value, cell_line='A549', actual_dose_uM=10.,
        exposure_hours_protocol_nominal=24., roles=dict(X=dict(cell_count=100 + i, plate='p1', well='B03')))
        for i, value in enumerate(ids)])
    means = rng.normal(0, .1, (n, 9))
    residual = rng.normal(size=(n, 9))*np.exp(.2 * np.log(np.linalg.norm(y[:, 0], axis=1)))[:, None]
    nested = dict(ids=ids, groups=groups, raw_mean=means, raw_residual=residual,
                  raw_covariance=np.broadcast_to(np.eye(9), (n, 9, 9)).copy())
    cell = dict(cell=0, error_fold=0, half=0, mean_fit_ids=ids[:55].tolist(),
                mean_validation_ids=ids[55:60].tolist(), mean_reference_ids=ids[:5].tolist(),
                fit_ids=ids[60:90].tolist(), cal_ids=ids[90:110].tolist(), query_ids=ids[110:].tolist())
    return data, metadata, nested, cell


def test_query_future_poison_cannot_change_scores_or_transformer():
    data, metadata, nested, cell = fixture()
    with threadpool_limits(limits=1):
        original, report, transformer = module.fit_cell_scores(data, metadata, nested, cell, seed=99)
        changed_data, changed_nested = deepcopy(data), deepcopy(nested)
        query = original['query_indices']
        changed_data['Y'][query, 1:] = np.nan
        changed_nested['raw_residual'][query] = np.nan
        changed_nested['raw_covariance'][query] = np.nan
        changed, second, _ = module.fit_cell_scores(changed_data, metadata, changed_nested, cell, seed=99)
    for key in original:
        np.testing.assert_array_equal(original[key], changed[key])
    assert report == second
    assert transformer.fit_ids == cell['mean_fit_ids']
    assert not report['query_outcomes_used']
    assert original['query_descriptors'].shape == (40, 2)
    assert original['donor_amplitude'].shape == (50, 2)


def test_donor_frame_and_cached_transformer_reuse_are_exact():
    data, metadata, nested, cell = fixture()
    with threadpool_limits(limits=1):
        first, _, transformer = module.fit_cell_scores(data, metadata, nested, cell, seed=101)
        second, _, _ = module.fit_cell_scores(data, metadata, nested, cell, seed=101,
                                            transformer=transformer)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    donor = first['donor_indices']
    np.testing.assert_array_equal(first['donor_energy'], error_targets(
        nested['raw_mean'][donor], nested['raw_covariance'][donor], nested['raw_residual'][donor]))
    assert np.all(first['query_amplitude'] > 0)
    assert np.all(first['query_descriptors'] > 0)


@pytest.mark.parametrize('violation', ['same_group', 'mean_fit', 'outer', 'missing_id'])
def test_role_validation_rejects_leakage(violation):
    data, _, _, cell = fixture()
    excluded = []
    if violation == 'same_group':
        data['groups'][110] = data['groups'][60]
    elif violation == 'mean_fit':
        cell['mean_fit_ids'].append(cell['query_ids'][0])
    elif violation == 'outer':
        excluded = [data['ids'][0]]
    else:
        cell['query_ids'][0] = 'outside'
    with pytest.raises(ValueError):
        module.validate_cell_roles(data['ids'], data['groups'], cell, excluded)


def test_wrong_descriptor_fit_population_rejected():
    data, metadata, nested, cell = fixture()
    with threadpool_limits(limits=1):
        _, _, transformer = module.fit_cell_scores(data, metadata, nested, cell, seed=103)
    transformer.fit_ids = cell['mean_fit_ids'] + [cell['query_ids'][0]]
    with pytest.raises(ValueError, match='coordinates'):
        module.fit_cell_scores(data, metadata, nested, cell, seed=103, transformer=transformer)
