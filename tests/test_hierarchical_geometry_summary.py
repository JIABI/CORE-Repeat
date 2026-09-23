"""Small saved-array fixtures check reporting, never scientific performance."""
import json

import numpy as np
import pytest

from opal2.baseline_policy import ACTIONS, _metrics
from opal2.hierarchical_geometry_summary import summarize


ARMS = ['GLOBAL_GEOMETRY', 'RIDGE_GEOMETRY', 'G_DIRECT', 'L_GRAM',
        'G_OOF_COV', 'HR_RIDGE_COV', 'HR_OOF_COV']


def saved_fixture(root):
    n = 40
    ids = np.asarray([f'fixture_{i:03}' for i in range(n)])
    folds = [dict(fold=f, test=list(range(f*8, (f+1)*8))) for f in range(5)]
    manifest = dict(ids=ids.tolist(), folds=folds, arms=ARMS,
        config=dict(seed=17, bootstrap=80, folds=5, samples=20),
        final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False)
    (root/'run_manifest.json').write_text(json.dumps(manifest))
    rng = np.random.default_rng(7)
    actual = rng.normal(size=(n, 3))*.08
    for ai, arm in enumerate(ARMS):
        for record in folds:
            ix = np.asarray(record['test'])
            folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm/'test'
            folder.mkdir(parents=True)
            mean = (np.full((len(ix), 3), record['fold']*.2) if ai == 0
                    else .1*actual[ix]+ai*.001)
            pnull = np.full((len(ix), 3), .5)
            crps = np.full((len(ix), 3), .05-ai*.001)
            np.savez_compressed(folder/'predictions.npz', ids=ids[ix], actual=actual[ix],
                predicted=mean, p_null=pnull, utility_crps=crps,
                geometry_energy=np.ones(len(ix)))
            metric = dict(action_metrics=[dict(action=name, **_metrics(actual[ix, j], mean[:, j], pnull[:, j]))
                                          for j, name in enumerate(ACTIONS)],
                          utility=[dict(crps=float(crps[:, j].mean())) for j in range(3)])
            (folder/'metrics.json').write_text(json.dumps(metric))
    return ids, actual, manifest


def test_all_arms_masks_primary_orientation_and_uniform_global(tmp_path):
    ids, actual, _ = saved_fixture(tmp_path)
    result = summarize(tmp_path)
    assert result['n'] == 40 and result['complete']
    assert len(result['paired']) == 21
    primary = result['declared_comparisons']['primary']
    assert primary['left'] == 'HR_OOF_COV' and primary['right'] == 'RIDGE_GEOMETRY'
    np.testing.assert_allclose(primary['gamma_crps']['mean'], -.005)
    for arm in ARMS:
        model = result['models'][arm]
        assert len(model['policies']) == 36
        assert len(model['common_budget']) == len(model['within_action']) == 18
        with np.load(tmp_path/f'{arm}_oof_predictions.npz') as saved:
            assert saved['masks'].shape == (36, 40)
            np.testing.assert_array_equal(saved['ids'], ids)
    global_result = result['models']['GLOBAL_GEOMETRY']
    assert global_result['actions'][2]['spearman'] is None
    assert global_result['actions'][2]['within_fold_rank_association'] is None
    row = next(x for x in global_result['common_budget'] if x['action'] == 'Z1Z2'
               and x['fraction'] == .25 and x['ranking'] == 'expected_gain')
    assert row['selected_n'] == 5 and row['used_wells'] == 10
    np.testing.assert_allclose(row['per_selected_net_gain'], actual[:, 2].mean())
    assert result['formal_certificate'] is False
    assert '跨站点生物世界模型' in (tmp_path/'REPORT.md').read_text()


def test_mismatched_paired_outcomes_rejected(tmp_path):
    saved_fixture(tmp_path)
    path = tmp_path/'folds/fold_0/arms/HR_OOF_COV/test/predictions.npz'
    with np.load(path) as saved:
        values = {k:saved[k].copy() for k in saved.files}
    values['actual'][0, 0] += 1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='different original outcomes'):
        summarize(tmp_path)
    assert not (tmp_path/'summary.json').exists()


def test_duplicate_outer_test_membership_rejected(tmp_path):
    _, _, manifest = saved_fixture(tmp_path)
    manifest['folds'][1]['test'][0] = 0
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='exactly one'):
        summarize(tmp_path)
