"""Synthetic full-recipe isolation tests; no assay data or neural fit."""
from copy import deepcopy

import numpy as np
import pytest

from opal2.eu_r3_reference_training import build_adapter_training, plan_reference_folds, load_adapter_training


def fixture():
    rng = np.random.default_rng(610)
    n, d = 90, 12
    x = rng.normal(size=(n, d))*np.exp(rng.normal(scale=.3, size=(n, 1)))
    y = np.full((n, 4, d), np.nan); y[:, 0] = x
    ids = np.asarray([f'id{i:03}' for i in range(n)])
    groups = np.asarray([f'g{i//2:03}' for i in range(n)])
    target = np.eye(4)[np.arange(n) % 4]
    data = dict(ids=ids, groups=groups, Y=y,
        chem=np.column_stack((rng.integers(0, 2, size=(n, 512)), np.ones(n))),
        target=target, target_mask=np.ones(n, bool), moa=target.copy(), moa_mask=np.ones(n, bool))
    units = [dict(id=i, site='FMP', platform='CP', cell_line='HepG2',
        exposure_hours_protocol_nominal=24., actual_dose_uM=10.) for i in ids]
    means = rng.normal(scale=.2, size=(n, 9))
    stats = dict(u_center=np.linspace(-.2, .2, 9).tolist(), u_scale=np.linspace(.3, .8, 9).tolist())
    raw = np.full((n, 9), np.nan)
    scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
    raw[:72] = (means[:72]+rng.normal(scale=.7, size=(72, 9)))*scale+center
    # REF order is deliberately not natural dataset order.
    rows = np.arange(72)[::-1]
    edges = np.quantile(np.log(np.linalg.norm(x[72:], axis=1)), [.2, .4, .6, .8])
    return data, dict(units=units), rows, means, raw, stats, edges


def run_fixture(folder, *, raw_change=False):
    data, meta, rows, means, raw, stats, edges = fixture()
    plans = plan_reference_folds(data['ids'], data['groups'], rows, 37)
    if raw_change:
        raw[plans[0]['heldout_rows']] += np.linspace(-.1, .15, 9)
    result = build_adapter_training(data, meta, rows, means, raw, stats, np.eye(9),
        .6, folder, 37, amplitude_edges=edges)
    return result, (data, rows, raw, stats, plans)


def test_full_recipe_order_and_native_covariance(tmp_path):
    result, (data, rows, raw, stats, plans) = run_fixture(tmp_path/'run')
    records = result['records']
    np.testing.assert_array_equal(records['ids'], data['ids'][rows])
    np.testing.assert_array_equal(records['prediction_count'], np.ones(len(rows), int))
    assert records['energies'].shape == (72, 2)
    assert records['biology_values'].shape == (72, 24)
    np.testing.assert_array_equal(records['biology_support'], records['random_biology_support'])
    scale = np.asarray(stats['u_scale'])
    np.testing.assert_allclose(records['raw_covariance'],
        records['covariance_u']*scale[None, :, None]*scale[None, None, :])
    np.testing.assert_allclose(records['raw_residual'], raw[rows]-records['raw_mean'])
    for r in result['fold_records']:
        a, b, c = [set(r[k]) for k in ('covariance_groups', 'calibration_groups', 'heldout_groups')]
        assert not a&b and not a&c and not b&c
        assert len(a) == 2*(len(a)+len(b))//3
        assert r['distribution']['recipe'] == 'LOCAL_SCALE + AMPLITUDE_TOTAL + AMP_EMP_LOCAL'
        assert r['heldout_targets_used_for_prediction'] is False
    assert result['report']['base_scatter_refitted'] is False
    loaded = load_adapter_training(tmp_path/'run', expected_ids=data['ids'][rows])
    np.testing.assert_array_equal(loaded['energies'], records['energies'])
    for field in ('values', 'support', 'support_by_relation'):
        np.testing.assert_array_equal(loaded['biology'][field], result['biology'][field])
        np.testing.assert_array_equal(loaded['random_biology'][field], result['random_biology'][field])


def test_heldout_target_changes_labels_not_own_distribution_or_features(tmp_path):
    one, _ = run_fixture(tmp_path/'one')
    two, _ = run_fixture(tmp_path/'two', raw_change=True)
    take = one['records']['fold'] == 0
    for key in ('mean_u', 'scatter_u', 'covariance_u', 'raw_mean', 'raw_covariance',
                'biology_values', 'random_biology_values', 'biology_support', 'random_biology_support'):
        np.testing.assert_array_equal(one['records'][key][take], two['records'][key][take])
    assert not np.array_equal(one['records']['energies'][take], two['records']['energies'][take])


def test_external_group_overlap_rejected_before_output(tmp_path):
    data, meta, rows, means, raw, stats, edges = fixture()
    data['groups'][72] = data['groups'][0]
    with pytest.raises(ValueError, match='crosses REF'):
        build_adapter_training(data, meta, rows, means, raw, stats, np.eye(9), .6,
            tmp_path/'bad', 37, amplitude_edges=edges)
    assert not (tmp_path/'bad').exists()
