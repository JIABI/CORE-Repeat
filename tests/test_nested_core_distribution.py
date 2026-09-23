"""Engineering checks for the full saved-mean distribution construction."""
from copy import deepcopy
import json

import numpy as np
import pytest

from opal2 import nested_core_distribution as module
from opal2.empirical_radial import variance_multiplier


def fixture(tmp_path):
    rng = np.random.default_rng(772)
    n = 180
    ids = np.asarray([f'unit{i:04d}' for i in range(n)])
    groups = np.asarray([f'group{i//2:04d}' for i in range(n)])
    fold = np.arange(n)//60
    y = rng.normal(size=(n, 4, 12)) * rng.lognormal(0, .3, (n, 1, 1))
    data = dict(ids=ids, groups=groups, Y=y,
        chem=rng.integers(0, 2, size=(n, 513)).astype(float))
    mean = rng.normal(size=(n, 9))*.1
    residual = rng.normal(size=(n, 9))*np.exp(.2*np.log(np.linalg.norm(y[:, 0], axis=1)))[:, None]
    cov = np.asarray([np.eye(9)*(1+.2*f) for f in fold])
    records = []
    for f in range(3):
        held = np.flatnonzero(fold == f)
        pool = np.flatnonzero(fold != f)
        records.append(dict(fold=f, heldout_ids=ids[held].tolist(),
            fit_ids=ids[pool[:80]].tolist(), inner_validation_ids=ids[pool[80:]].tolist(),
            reference_ids=ids[pool[:8]].tolist()))
    nested = tmp_path/'nested'
    nested.mkdir()
    np.savez_compressed(nested/'residuals.npz', ids=ids, groups=groups,
        raw_mean=mean, raw_target=mean+residual, raw_residual=(mean+residual)-mean,
        raw_covariance=cov, error_fold=fold, prediction_count=np.ones(n, int))
    (nested/'plan.json').write_text(json.dumps(dict(folds=records)))
    return data, nested, records


def test_complete_law_and_reference_provenance(tmp_path):
    data, nested, _ = fixture(tmp_path)
    out = module.build_nested_core_distribution(data, {}, nested, tmp_path/'out', seed=51)
    assert out['summary']['state'] == 'COMPLETE'
    assert out['summary']['cells'] == 6
    mask = out['biological_reference_mask']
    assert not np.any(mask & (data['groups'][:, None] == data['groups'][None, :]))
    assert not np.any(mask & (out['error_fold'][:, None] != out['error_fold'][None, :]))
    assert mask.sum(1).min() > 0
    for cell in out['cells']:
        a = np.load(tmp_path/'out'/f"cell_{cell['cell']:02d}.npz")
        q, fit = a['query_indices'], a['reference_indices']
        np.testing.assert_array_equal(mask[q], np.broadcast_to(np.isin(np.arange(len(mask)), fit), (len(q), len(mask))))
        np.testing.assert_array_equal(out['raw_scatter'][q], a['query_raw_scatter'])
        np.testing.assert_allclose(a['query_raw_covariance'],
            a['query_raw_scatter']*variance_multiplier(cell['law'], a['query_radial_weights'])[:, None, None])
        np.testing.assert_array_equal(a['reference_raw_residual'], out['raw_residual'][fit])
        assert cell['amplitude_fit']['conditional']
        assert cell['local_covariance_choice']['family'] == 'LOCAL_SCALE'
        assert cell['law']['calibration_n'] == len(a['representative_ids'])
        assert not cell['query_outcomes_used_for_distribution']
    with pytest.raises(FileExistsError):
        module.build_nested_core_distribution(data, {}, nested, tmp_path/'out', seed=51)


def test_query_outcomes_cannot_change_own_law_or_reference_records(tmp_path):
    data, nested, records = fixture(tmp_path)
    z = np.load(nested/'residuals.npz')
    cell = module.plan_distribution_cells(data['ids'], data['groups'], z['error_fold'], records, 12)[0]
    original, report = module.construct_cell(data, z['raw_residual'], z['raw_covariance'], cell)
    changed = deepcopy(data)
    q = np.asarray(cell['query_indices'])
    changed['Y'][q, 1:] *= 1000
    residual = z['raw_residual'].copy(); residual[q] *= 999
    result, second = module.construct_cell(changed, residual, z['raw_covariance'], cell)
    for key in original:
        np.testing.assert_array_equal(original[key], result[key])
    np.testing.assert_array_equal(report['law']['log_centers'], second['law']['log_centers'])
    # Confirm actual donor errors, not a placeholder covariance, drive the law.
    residual = z['raw_residual'].copy()
    residual[np.asarray(cell['fit_indices'])] *= 2
    alternative, _ = module.construct_cell(data, residual, z['raw_covariance'], cell)
    assert not np.array_equal(original['query_raw_scatter'], alternative['query_raw_scatter'])


def test_native_coordinate_equivariance(tmp_path):
    data, nested, records = fixture(tmp_path)
    z = np.load(nested/'residuals.npz')
    cell = module.plan_distribution_cells(data['ids'], data['groups'], z['error_fold'], records, 29)[0]
    base, base_report = module.construct_cell(data, z['raw_residual'], z['raw_covariance'], cell)
    scale = np.linspace(.4, 1.8, 9)
    other, report = module.construct_cell(data, z['raw_residual']*scale,
        z['raw_covariance']*scale[None, :, None]*scale[None, None, :], cell)
    for key in ('query_raw_scatter','query_raw_covariance','reference_raw_scatter','reference_raw_covariance'):
        np.testing.assert_allclose(other[key], base[key]*scale[None, :, None]*scale[None, None, :], rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(report['law']['log_centers'], base_report['law']['log_centers'], rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(other['query_radial_weights'], base['query_radial_weights'])


def test_plan_rejects_mean_or_chemistry_leakage(tmp_path):
    data, nested, records = fixture(tmp_path)
    z = np.load(nested/'residuals.npz')
    broken = deepcopy(records)
    broken[0]['fit_ids'].append(broken[0]['heldout_ids'][0])
    with pytest.raises(ValueError, match='inner mean saw'):
        module.plan_distribution_cells(data['ids'], data['groups'], z['error_fold'], broken, 29)
    changed = deepcopy(data); changed['ids'][0] = 'not-the-saved-object'
    with pytest.raises(AssertionError):
        module.build_nested_core_distribution(changed, {}, nested, tmp_path/'bad', seed=12)
    assert not (tmp_path/'bad').exists()
