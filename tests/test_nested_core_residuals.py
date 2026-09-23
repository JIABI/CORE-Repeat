"""Isolation and full-recipe checks; synthetic fixtures are engineering only."""
from copy import deepcopy
import inspect
import json

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from opal2 import nested_core_residuals as module


def fixture(n=360):
    rng = np.random.default_rng(812)
    signal = rng.normal(size=(n, 1, 12))*.4
    y = signal+rng.normal(size=(n, 4, 12))*.7
    chem = np.column_stack((rng.integers(0, 2, size=(n, 512)), np.ones(n))).astype(float)
    data = dict(ids=np.asarray([f'unit{i:04d}' for i in range(n)]),
        groups=np.asarray([f'group{i//2:04d}' for i in range(n)]), Y=y,
        chem=chem, chem_mask=np.ones(n, bool), target=np.ones((n, 2)),
        moa=np.ones((n, 3)), target_mask=np.ones(n, bool), moa_mask=np.ones(n, bool))
    metadata = dict(chemical=dict(fingerprint_indices=list(range(512)), validity_index=512,
                                   kind='synthetic engineering test'),
                    target_names=['t0', 't1'], moa_names=['m0', 'm1', 'm2'])
    return data, metadata


def test_grouped_scopes_and_reference_exclusion_and_outcome_blind_planning():
    data, metadata = fixture()
    records = module.plan_nested_core_folds(data, metadata, 51)
    count = np.zeros(len(data['ids']), int)
    for record in records:
        parts = [set(data['groups'][record[k]]) for k in ('fit', 'inner_validation', 'heldout')]
        assert all(not parts[a] & parts[b] for a, b in ((0, 1), (0, 2), (1, 2)))
        assert len(record['reference_ids']) == 64
        assert not set(data['groups'][record['references']]) & set(data['groups'][record['commonbranchfit']])
        assert set(record['reference_ids']) <= set(record['fit_ids'])
        count[record['heldout']] += 1
    assert np.array_equal(count, np.ones(len(count)))
    changed = deepcopy(data); changed['Y'] *= 100
    assert module.plan_nested_core_folds(changed, metadata, 51) == records


def test_crossfit_only_supplies_inner_fit_validation_and_decision_time_query(monkeypatch, tmp_path):
    data, metadata = fixture()
    observed = []
    def fit(train, meta, record, folder):
        names = set(train['ids'])
        assert names == set(record['fit_ids']+record['inner_validation_ids'])
        assert not names & set(record['heldout_ids'])
        assert set(record['reference_ids']) <= names
        observed.append(names)
        scales = np.arange(1, 10, dtype=float)*(record['fold']+1)
        return object(), dict(u_scale=scales.tolist()), np.eye(9), dict(fitted_ids=sorted(names))
    def predict(model, stats, first_well, chemistry, chemistry_mask, biology):
        assert first_well.ndim == 2 and first_well.shape[1] == 12
        assert set(biology) == {'target','target_mask','moa','moa_mask'}
        return np.repeat(first_well.mean(1)[:, None], 9, axis=1)
    monkeypatch.setattr(module, '_train_complete_core', fit)
    monkeypatch.setattr(module, 'predict_raw_geometry', predict)
    result = module.fit_nested_core_residuals(data, metadata, tmp_path/'run', 51)
    assert len(observed) == 3
    assert result['raw_mean'].shape == result['raw_residual'].shape == (360, 9)
    np.testing.assert_array_equal(result['raw_residual'], result['raw_target']-result['raw_mean'])
    for fold in range(3):
        expected = np.diag((np.arange(1, 10)*(fold+1))**2)
        np.testing.assert_array_equal(result['raw_covariance'][result['error_fold']==fold][0], expected)
    assert np.all(result['prediction_count'] == 1)
    assert (tmp_path/'run/residuals.npz').exists()
    with pytest.raises(FileExistsError):
        module.fit_nested_core_residuals(data, metadata, tmp_path/'run', 51)
    assert tuple(inspect.signature(module.predict_raw_geometry).parameters) == (
        'model','stats','first_well','chemistry','chemistry_mask','biology')


def test_invalid_population_rejected_before_fitting(tmp_path):
    data, metadata = fixture()
    duplicate = deepcopy(data); duplicate['ids'][1] = duplicate['ids'][0]
    with pytest.raises(ValueError, match='Unique MODEL_FIT'):
        module.fit_nested_core_residuals(duplicate, metadata, tmp_path/'bad', 51)
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError, match='two reference sets'):
        module.plan_nested_core_folds(module._take(data, np.arange(120)), metadata, 51)


def test_complete_core_keeps_exact_production_stages_and_predicts_raw(tmp_path):
    # Full production optimizers/epochs/64 anchors, small synthetic feature count.
    # This is an integration check, never an assay performance experiment.
    data, metadata = fixture()
    record = module.plan_nested_core_folds(data, metadata, 62)[0]
    train = module._take(data, np.asarray(record['fit']+record['inner_validation']))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            model, stats, covariance, report = module._train_complete_core(train, metadata, record, tmp_path/'full')
            held = np.asarray(record['heldout'])
            biology = {k: data[k][held] for k in ('target','target_mask','moa','moa_mask')}
            raw = module.predict_raw_geometry(model, stats, data['Y'][held, 0],
                data['chem'][held], data['chem_mask'][held], biology)
    finally:
        torch.set_num_threads(previous_threads)
    assert raw.shape == (len(held), 9) and np.isfinite(raw).all()
    assert covariance.shape == (9, 9) and np.linalg.eigvalsh(covariance).min() > 0
    assert not report['prior_fold_models_loaded'] and not report['heldout_outcomes_supplied']
    for name, epoch in (('A_OLD_GENERIC',30),('STATE50',50)):
        completion = json.loads((tmp_path/'full'/name/'training_complete.json').read_text())
        assert completion['actual_checkpoint_epoch'] == epoch
        assert (tmp_path/'full'/name/f'epoch{epoch}.pt').exists()
    assert (tmp_path/'full/HR_fit/training_complete.json').exists()
    assert set(report['fitted_ids']).isdisjoint(record['heldout_ids'])
    assert len(report['reference_ids']) == 64
