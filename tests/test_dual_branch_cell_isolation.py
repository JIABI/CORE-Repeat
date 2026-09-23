"""Current-cell calibration isolation, including the actual runner block.

The optional saved-artifact integration test performs no training or predictive
evaluation. It rebuilds only calibration inputs and candidate-strength scores.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from opal2 import dual_branch_experiment as experiment


def calibration_block():
    """Execute the runner's own CAL block rather than a second implementation."""
    tree = ast.parse(inspect.getsource(experiment.run))
    cell_loop = next(node for node in tree.body[0].body if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Tuple)
                     and [getattr(x, 'id', '') for x in node.target.elts] == ['number', 'cell'])
    start = next(i for i, node in enumerate(cell_loop.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'cal_cov' for t in node.targets))
    end = next(i for i, node in enumerate(cell_loop.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'cell_out' for t in node.targets))
    block = ast.Module(body=cell_loop.body[start:end], type_ignores=[])
    return ast.fix_missing_locations(block)


def test_runner_selects_strength_from_cell_specific_calibration_predictions():
    block = calibration_block()
    calls = [n for n in ast.walk(block) if isinstance(n, ast.Call)]
    select = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'select_strength')
    assert isinstance(select.args[-1], ast.Subscript)
    assert isinstance(select.args[-1].value, ast.Name)
    assert select.args[-1].value.id == 'cal_prediction'
    frame = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'calibration_frame_covariance')
    assert ast.unparse(frame.args[0]) == "ref['cal_amp_scatter']"
    assert ast.unparse(frame.args[1]) == "ref['cal_residual']"
    bio = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'biology_features')
    assert 'cal_cov' in ast.unparse(bio.args[5])
    assert not any(isinstance(n, ast.Name) and n.id == 'prediction' for n in ast.walk(block))


def test_saved_real_current_query_poison_cannot_change_calibration(tmp_path):
    root = experiment.PROJECT/'runs/dual_branch_biology_20260917_v2'
    if not all((root/'fold_0'/(a+'.pt')).exists() for a in experiment.FULL_ARMS):
        pytest.skip('Saved declared DEV adapter checkpoints are unavailable')
    old = experiment.read_json(experiment.RADIAL/'summary.json')
    manifest = experiment.read_json(Path(old['reference_run'])/'run_manifest.json')
    data, metadata = experiment.load_data(old['data_directory'])
    ids, groups = data['ids'], data['groups']
    lookup = {v: i for i, v in enumerate(ids)}
    cell = next(c for c in old['cells'] if c['fold'] == 0 and c['half'] == 0)
    fold = 0
    q = np.asarray([lookup[v] for v in cell['query_ids']])
    cal = np.asarray([lookup[v] for v in cell['representative_ids']])
    record = next(r for r in manifest['folds'] if r['fold'] == fold)
    fit = np.asarray(record['fit'])
    stats = experiment.read_json(Path(old['reference_run'])/'folds/fold_0/preprocessing.json')
    scale, center = np.asarray(stats['u_scale']), np.asarray(stats['u_center'])
    prior = experiment.read_npz(experiment.RADIAL/'AMP_EMP_LOCAL.npz')
    raw_mean = prior['mean_u']*scale+center
    ref = experiment.read_npz(experiment.RADIAL/'cell_0_0_radial.npz')
    logamp = np.log(np.linalg.norm(data['Y'][:, 0], axis=1))
    code = compile(calibration_block(), inspect.getsourcefile(experiment), 'exec')
    def execute(name):
        folder = tmp_path/name
        folder.mkdir()
        env = dict(vars(experiment), root=root, folder=folder, data=data, metadata=metadata,
            ids=ids, groups=groups, cell=cell, fold=fold, q=q, cal=cal, fit=fit,
            scale=scale, center=center, stats=stats, prior=prior, raw_mean=raw_mean,
            ref=ref, logamp=logamp)
        with threadpool_limits(limits=1):
            exec(code, env)
        return env
    original = execute('original')
    # Poison both forbidden paths: direct future profiles of today's query,
    # and the opposite-role cached covariance previously used for CAL features.
    data['Y'][q, 1:] = np.nan
    prior['covariance_u'][cal] = np.nan
    prior['actual_u'][q] = np.nan
    poisoned = execute('poisoned')
    np.testing.assert_array_equal(original['cal_cov'], poisoned['cal_cov'])
    np.testing.assert_array_equal(original['cal_bio']['values'], poisoned['cal_bio']['values'])
    np.testing.assert_array_equal(original['cal_bio']['support'], poisoned['cal_bio']['support'])
    np.testing.assert_array_equal(original['cal_desc'], poisoned['cal_desc'])
    for arm in experiment.FULL_ARMS:
        np.testing.assert_array_equal(original['cal_prediction'][arm], poisoned['cal_prediction'][arm])
        assert original['calibration'][arm] == poisoned['calibration'][arm]
