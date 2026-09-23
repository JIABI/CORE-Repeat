"""Engineering checks for R3 selection and cached aggregation, not results."""
import json
import numpy as np
import pytest

from scripts import run_r3_eu_modules_20260918 as run


def test_saved_query_mean_is_exact_without_accepting_model_drift():
    replay = np.ones((4, 9))
    rows = np.array([0, 2])
    saved = np.nextafter(replay[rows], 2.)
    fixed = run.reuse_saved_query_mean(replay, rows, saved)
    np.testing.assert_array_equal(fixed[rows], saved)
    np.testing.assert_array_equal(fixed[[1, 3]], replay[[1, 3]])
    np.testing.assert_array_equal(replay, np.ones((4, 9)))
    with pytest.raises(AssertionError):
        run.reuse_saved_query_mean(replay, rows, saved+.001)


def test_calibration_selects_zero_or_declared_candidate_without_query():
    groups = np.array([f'g{i}' for i in range(12)])
    def choice(delta, strength):
        return dict(strength=strength, strengths=[0., .25, .5, 1.],
                    scores=np.tile([1., 1.+delta/4, 1.+delta/2, 1.+delta], (12, 1)))
    cal = {a: choice(.1, 0.) for a in run.SWITCH_CANDIDATES[1:]}
    assert run.select_module(cal, groups)['arm'] == 'CORE'
    cal['COND_REP'] = choice(-.1, 1.)
    assert run.select_module(cal, groups)['arm'] == 'COND_REP_CAL'


def test_complete_aggregation_assigns_every_object_once_and_handles_bool(tmp_path, monkeypatch):
    root, r2 = tmp_path/'r3', tmp_path/'r2'
    root.mkdir(); r2.mkdir()
    monkeypatch.setattr(run, 'ROOT', root)
    monkeypatch.setattr(run, 'R2', r2)
    def interval(x, y, labels):
        assert len(x) == len(y) == len(labels)
        difference = np.asarray(x, float)-np.asarray(y, float)
        return dict(mean=float(difference.mean()), fixture_only=True)
    monkeypatch.setattr(run, 'bootstrap_difference', interval)
    n = 80
    ids = np.array([f'id{i:03}' for i in range(n)])
    groups = np.array([f'g{i}' for i in range(n)])
    layout = np.array([f'L{i%4}' for i in range(n)])
    folds = np.arange(n)%5
    actual = np.where(np.arange(n)%3 == 0, -.01, .1)
    mean = np.zeros((n, 9))
    data = dict(ids=ids, groups=groups, layout=layout)
    core = dict(ids=ids, actual=actual, fold=folds, mean_u=mean)
    common = dict(predicted=np.linspace(0., .2, n), p_null=np.linspace(.6, .1, n),
        crps=np.full(n, .02), nll=np.full(n, 1.), energy=np.full(n, .5),
        mean_u=mean, actual_u=np.ones((n, 9)), increment=np.zeros((n, 2)),
        resource_support=np.arange(n)%2 == 0,
        joint_coverage_by_level=np.tile(run.LEVELS, (n, 1)),
        gamma_coverage_by_level=np.tile(run.LEVELS, (n, 1)))
    cells = []
    for f in range(5):
        folder = root/f'fold_{f}'; folder.mkdir()
        rows = np.flatnonzero(folds == f)
        for arm in run.ARMS:
            np.savez_compressed(folder/(arm+'.npz'), ids=ids[rows], actual=actual[rows],
                                **{k: v[rows] for k, v in common.items()})
            for offset in (100000, 200000):
                np.savez_compressed(folder/f'{arm}_mc{offset}.npz', ids=ids[rows],
                    predicted=common['predicted'][rows], p_null=common['p_null'][rows])
        cells.append(dict(fold=f, costs=dict(reference_if_all_new_wells=40,
                                           reference_if_X_already_available=30)))
    for name in ('DIRECT_ACCESS_MATCHED_HISTGB_COHERENT',
                 'DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL', 'HR_REF'):
        np.savez_compressed(r2/(name+'.npz'), ids=ids, **common)
    result = run.collect_and_analyze(data, core, cells)
    assert result['state'] == 'COMPLETE'
    assert set(result['metrics']) == set(run.ARMS)
    assert result['supported_n'] == 40
    assert all(v['mean_unchanged'] for v in result['metrics'].values())
    assert len(result['monte_carlo_sensitivity']['CORE']) == 2
    assert len(result['deployment_costs']) == 5
    assert json.loads((root/'summary.json').read_text())['confirmation_opened'] is False
    assert (root/'REPORT.md').exists()
