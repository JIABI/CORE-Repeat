"""Synthetic tests of four-arm aggregation; no scientific data or training."""
import json

import numpy as np
import pytest

from opal2.baseline_policy import _metrics
from opal2.biology_kernel_evaluation import write_json
from opal2.geometry_kernel_replacement_summary import ARMS, COMPARISONS, _validate_manifest, summarize


def _fixture(root):
    rng = np.random.default_rng(71)
    n = 639
    ids = np.asarray([f'SYNTHETIC_{i:04d}' for i in range(n)])
    folds = []
    actual, target = rng.normal(0, .07, (n, 3)), rng.normal(size=(n, 9))
    for fold, test in enumerate(np.array_split(np.arange(n), 5)):
        other = np.setdiff1d(np.arange(n), test)
        folds.append(dict(fold=fold, fit=other[100:].tolist(),
                          inner_validation=other[:100].tolist(), test=test.tolist()))
    manifest = dict(ids=ids.tolist(), folds=folds, arms=list(ARMS),
        config=dict(folds=5, seed=47, bootstrap=2000, samples=2000),
        final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False)
    write_json(root/'run_manifest.json', manifest)
    for record in folds:
        ix = np.asarray(record['test'])
        for index, arm in enumerate(ARMS):
            folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
            (folder/'test').mkdir(parents=True)
            gain = min(index, 2)  # C/D identical, despite both better than B.
            predicted = actual[ix]*(.5+.1*gain)
            pnull = np.where(actual[ix] <= 0, .6+.1*gain, .4-.1*gain)
            crps = np.full((len(ix), 3), .05-.01*gain)
            mean = target[ix]+(1-.25*gain)
            counts = dict(trainable=0 if index == 0 else 50 if index == 1 else 100,
                          total=100 if index == 0 else 50 if index == 1 else 100)
            np.savez_compressed(folder/'test/predictions.npz', ids=ids[ix], actual=actual[ix],
                predicted=predicted, p_null=pnull, utility_crps=crps, geometry_energy=np.ones(len(ix)))
            np.savez_compressed(folder/'test/u_predictions.npz', ids=ids[ix], actual_u=target[ix], mean_u=mean)
            write_json(folder/'test/metrics.json', dict(samples=2000,
                action_metrics=[_metrics(actual[ix, j], predicted[:, j], pnull[:, j]) for j in range(3)],
                utility=[dict(crps=float(crps[:, j].mean())) for j in range(3)],
                model=dict(actual_checkpoint_epoch='frozen' if arm == 'A_HR' else 10,
                           parameter_counts=counts)))
            write_json(folder/'training_complete.json', dict(best_epoch=5, final_epoch=10, parameter_counts=counts))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e, validation_mse=1/(1+e)))
                                                        for e in (0, 5, 10))+'\n')
    return manifest


def test_four_arms_six_paired_comparisons_unique_oof_budget_and_capacity(tmp_path):
    manifest = _fixture(tmp_path)
    result = summarize(tmp_path)
    assert result['n'] == 639
    assert list(result['models']) == list(ARMS)
    assert list(result['comparisons']) == [left+'__minus__'+right for left, right in COMPARISONS]
    for arm in ARMS:
        model = result['models'][arm]
        assert model['principal_policy']['selected_n'] == 79
        assert model['principal_policy']['used_wells'] == 158
        assert [row['principal_policy']['selected_n'] for row in model['folds']] == [16, 16, 16, 16, 15]
        with np.load(tmp_path/f'{arm}_oof_predictions.npz', allow_pickle=False) as saved:
            assert saved['ids'].tolist() == manifest['ids']
            assert saved['mean_u'].shape == (639, 9)
    d_a = result['comparisons']['D_STRUCTURED__minus__A_HR']
    assert d_a['u_mse']['mean'] == pytest.approx(-.75)
    assert d_a['gamma_crps']['mean'] == pytest.approx(-.02)
    assert d_a['fold_directions']['gamma_crps']['favorable'] == 5
    d_c = result['comparisons']['D_STRUCTURED__minus__C_GENERIC']
    for key in ('u_mse', 'gamma_crps', 'null_brier', 'net_gain_per_eligible', 'fdp', 'fpr'):
        assert d_c[key]['mean'] == 0
        assert d_c[key]['interval95'] == [0, 0]
    assert d_c['overlap']['intersection_n'] == 79
    assert result['capacity']['generic_structured_trainable_equal']
    assert result['capacity']['mlp_trainable_smaller_than_generic']
    assert not result['capacity']['all_four_arms_capacity_matched']
    assert result['models']['B_MLP']['folds'][0]['u']['checkpoint']['training_complete']['best_epoch'] == 5
    assert result['actual_checkpoint_epoch'] == 10
    for name in ('summary.json', 'REPORT.md', 'SUMMARY.md'):
        assert (tmp_path/name).is_file()
    assert 'B_MLP' in (tmp_path/'REPORT.md').read_text()


@pytest.mark.parametrize('field,value,match', [
    ('epoch', 5, 'actual epoch-10'),
    ('samples', 1000, '2000 predictive draws'),
    ('parameters', None, 'Missing explicit parameter_counts'),
])
def test_incorrect_stage_evidence_is_not_silently_aggregated(tmp_path, field, value, match):
    _fixture(tmp_path)
    folder = tmp_path/'folds/fold_1/arms/D_STRUCTURED'
    path = folder/'test/metrics.json'
    metrics = json.loads(path.read_text())
    if field == 'epoch':
        metrics['model']['actual_checkpoint_epoch'] = value
    elif field == 'samples':
        metrics['samples'] = value
    else:
        complete_path = folder/'training_complete.json'
        complete = json.loads(complete_path.read_text())
        complete.pop('parameter_counts')
        write_json(complete_path, complete)
    write_json(path, metrics)
    with pytest.raises(ValueError, match=match):
        summarize(tmp_path)
    assert not (tmp_path/'summary.json').exists()


def test_four_arm_identity_and_unique_test_partition_required(tmp_path):
    manifest = _fixture(tmp_path)
    manifest['arms'] = ['A_HR', 'B_GENERIC', 'C_STRUCTURED']
    with pytest.raises(ValueError, match='four declared arms'):
        _validate_manifest(manifest)
    manifest['arms'] = list(ARMS)
    manifest['folds'][1]['test'][0] = manifest['folds'][0]['test'][0]
    with pytest.raises(ValueError, match='exactly one'):
        _validate_manifest(manifest)
