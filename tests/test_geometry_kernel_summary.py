"""Synthetic file-interface tests, not measurements or scientific evidence."""
import json

import numpy as np
import pytest

from opal2.baseline_policy import _metrics
from opal2.biology_kernel_evaluation import write_json
from opal2.geometry_kernel_summary import ARMS, _compare, _validate_manifest, summarize


def _fixture(root):
    rng = np.random.default_rng(11)
    n = 639
    ids = np.asarray([f'SYNTHETIC_{i:04d}' for i in range(n)])
    groups = np.array_split(np.arange(n), 5)
    records = []
    actual = rng.normal(0, .07, (n, 3))
    target = rng.normal(size=(n, 9))
    for f, test in enumerate(groups):
        other = np.setdiff1d(np.arange(n), test)
        records.append(dict(fold=f, fit=other[100:].tolist(),
                            inner_validation=other[:100].tolist(), test=test.tolist()))
    manifest = dict(ids=ids.tolist(), folds=records, arms=list(ARMS),
        config=dict(folds=5, seed=41, bootstrap=2000),
        final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False)
    write_json(root/'run_manifest.json', manifest)
    for record in records:
        ix = np.asarray(record['test'])
        for index, arm in enumerate(ARMS):
            folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
            (folder/'test').mkdir(parents=True)
            # B/C are exactly equal to test paired-zero and overlap handling.
            improvement = float(index > 0)
            predicted = actual[ix] * (.5+.3*improvement)
            pnull = np.where(actual[ix] <= 0, .6+.2*improvement, .4-.2*improvement)
            crps = np.full((len(ix), 3), .05-.01*improvement)
            mean_u = target[ix]+(1-.5*improvement)
            np.savez_compressed(folder/'test'/'predictions.npz', ids=ids[ix], actual=actual[ix],
                predicted=predicted, p_null=pnull, utility_crps=crps,
                geometry_energy=np.ones(len(ix)))
            np.savez_compressed(folder/'test'/'u_predictions.npz', ids=ids[ix],
                actual_u=target[ix], mean_u=mean_u)
            write_json(folder/'test'/'metrics.json', dict(
                action_metrics=[_metrics(actual[ix, j], predicted[:, j], pnull[:, j]) for j in range(3)],
                utility=[dict(crps=float(crps[:, j].mean())) for j in range(3)],
                model=dict(actual_checkpoint_epoch='frozen' if arm == 'A_HR' else 10)))
            write_json(folder/'training_complete.json', dict(best_epoch=5, final_epoch=10))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e, valid_mse=1/e if e else 2))
                                                        for e in (0, 5, 10))+'\n')
    return manifest


def test_complete_summary_unique_oof_budget_paired_scores_and_reports(tmp_path):
    manifest = _fixture(tmp_path)
    result = summarize(tmp_path)
    assert result['n'] == 639
    assert result['bootstrap_replicates'] == 2000
    assert list(result['models']) == list(ARMS)
    for arm in ARMS:
        model = result['models'][arm]
        assert model['principal_policy']['selected_n'] == 79
        assert model['principal_policy']['used_wells'] == 158
        assert [r['principal_policy']['selected_n'] for r in model['folds']] == [16, 16, 16, 16, 15]
        with np.load(tmp_path/f'{arm}_oof_predictions.npz') as saved:
            assert saved['ids'].tolist() == manifest['ids']
            assert saved['actual'].shape == (639, 3)
            assert saved['mean_u'].shape == (639, 9)
    b = result['comparisons']['B_GENERIC__minus__A_HR']
    assert b['u_mse']['mean'] == pytest.approx(-.75)
    assert b['gamma_crps']['mean'] == pytest.approx(-.01)
    assert b['null_brier']['mean'] == pytest.approx(-.12)
    assert b['fold_directions']['u_mse']['favorable'] == 5
    identical = result['comparisons']['C_STRUCTURED__minus__B_GENERIC']
    for key in ('u_mse', 'gamma_crps', 'null_brier', 'net_gain_per_eligible', 'fdp', 'fpr'):
        assert identical[key]['mean'] == 0
        assert identical[key]['interval95'] == [0, 0]
    assert identical['overlap']['intersection_n'] == 79
    assert identical['overlap']['jaccard'] == 1
    assert result['models']['B_GENERIC']['folds'][0]['u']['checkpoint']['training_complete']['best_epoch'] == 5
    assert result['actual_checkpoint_epoch'] == 10
    assert all((tmp_path/name).is_file() for name in ('summary.json', 'SUMMARY.md', 'REPORT.md'))
    assert not result['formal_certificate']


def test_wrong_stage_checkpoint_is_not_silently_selected(tmp_path):
    _fixture(tmp_path)
    path = tmp_path/'folds/fold_2/arms/C_STRUCTURED/test/metrics.json'
    metric = json.loads(path.read_text())
    metric['model']['actual_checkpoint_epoch'] = 5
    write_json(path, metric)
    with pytest.raises(ValueError, match='actual epoch-10'):
        summarize(tmp_path)
    assert not (tmp_path/'summary.json').exists()


def test_duplicate_oof_compound_or_leaking_partition_rejected(tmp_path):
    manifest = _fixture(tmp_path)
    manifest['folds'][1]['test'][0] = manifest['folds'][0]['test'][0]
    with pytest.raises(ValueError, match='exactly one'):
        _validate_manifest(manifest)
    manifest = json.loads((tmp_path/'run_manifest.json').read_text())
    manifest['folds'][0]['fit'][0] = manifest['folds'][0]['test'][0]
    with pytest.raises(ValueError, match='partition the 639'):
        _validate_manifest(manifest)


def test_paired_risk_differences_use_actual_null_and_correct_denominators():
    ids = np.asarray(['A', 'B', 'C', 'D'])
    actual = np.tile(np.array([-.2, .1, .2, -.1])[:, None], (1, 3))
    template = dict(actual=actual, p_null=np.ones((4, 3))*.5,
                    utility_crps=np.zeros((4, 3)), u_object_mse=np.zeros(4))
    left = dict(**template, principal_mask=np.array([1., 1., 0., 0.]))
    right = dict(**template, principal_mask=np.array([0., 1., 1., 0.]))
    exact_boot = np.tile(np.arange(4), (20, 1))
    comparison = _compare('L', 'R', dict(L=left, R=right), ids,
                          [dict(fold=0, test=[0, 1, 2, 3])], exact_boot)
    assert comparison['fdp']['mean'] == .5
    assert comparison['fpr']['mean'] == .5
    assert comparison['net_gain_per_eligible']['mean'] == pytest.approx(-.1)
    assert comparison['overlap']['left_only_ids'] == ['A']
    assert comparison['overlap']['right_only_ids'] == ['C']
