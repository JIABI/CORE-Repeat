"""Synthetic continuation summaries; no biological data or model fitting."""
import json

import numpy as np
import pytest

from opal2.baseline_policy import _metrics
from opal2.biology_kernel_evaluation import write_json
from opal2.geometry_kernel_continuation_summary import ARMS, _validate_manifest, summarize
from opal2.gram_oof_experiment import selection_mask


def _fixture(root):
    source = root.parent/'synthetic_epoch10'
    source.mkdir()
    rng, n = np.random.default_rng(19), 639
    ids = np.asarray([f'SYNTHETIC_{i:04d}' for i in range(n)])
    actual, target = rng.normal(0, .07, (n, 3)), rng.normal(size=(n, 9))
    folds, allocation = [], np.empty(n, int)
    for fold, test in enumerate(np.array_split(np.arange(n), 5)):
        other = np.setdiff1d(np.arange(n), test)
        folds.append(dict(fold=fold, fit=other[100:].tolist(), inner_validation=other[:100].tolist(), test=test.tolist()))
        allocation[test] = fold
    manifest = dict(ids=ids.tolist(), arms=list(ARMS), folds=folds, continuation_source=str(source),
        config=dict(folds=5, bootstrap=2000, samples=2000, seed=47, stage_epochs=30),
        final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False, original_contract_changed=False)
    write_json(root/'run_manifest.json', manifest)
    write_json(source/'summary.json', dict(actual_checkpoint_epoch=10, arms=list(ARMS)))
    for index, arm in enumerate(ARMS):
        old = dict(actual=actual, predicted=.5*actual, p_null=np.full((n, 3), .5),
                   utility_crps=np.full((n, 3), .05), geometry_energy=np.ones(n),
                   actual_u=target, mean_u=target+1)
        old_mask = selection_mask(old['predicted'][:, 2], ids, allocation, .25, 2)
        np.savez_compressed(source/f'{arm}_oof_predictions.npz', ids=ids, fold=allocation,
                            principal_mask=old_mask, **old)
        improvement = 0 if arm == 'A_HR' else .5
        current = dict(old)
        current['mean_u'] = target+1-improvement
        current['utility_crps'] = old['utility_crps']-.02*improvement
        for record in folds:
            ix = np.asarray(record['test'])
            folder = root/'folds'/f"fold_{record['fold']}"/'arms'/arm
            (folder/'test').mkdir(parents=True)
            counts = dict(trainable=0 if index == 0 else 50 if index == 1 else 100, total=100)
            np.savez_compressed(folder/'test/predictions.npz', ids=ids[ix],
                **{k: current[k][ix] for k in ('actual', 'predicted', 'p_null', 'utility_crps', 'geometry_energy')})
            np.savez_compressed(folder/'test/u_predictions.npz', ids=ids[ix], actual_u=target[ix], mean_u=current['mean_u'][ix])
            write_json(folder/'test/metrics.json', dict(samples=2000,
                action_metrics=[_metrics(actual[ix, j], current['predicted'][ix, j], current['p_null'][ix, j]) for j in range(3)],
                utility=[dict(crps=float(current['utility_crps'][ix, j].mean())) for j in range(3)],
                model=dict(actual_checkpoint_epoch='frozen' if index == 0 else 30, parameter_counts=counts)))
            write_json(folder/'training_complete.json', dict(best_epoch=20, final_epoch=30, parameter_counts=counts))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e, validation_mse=1/(1+e)))
                                                        for e in (0, 5, 10, 15, 20, 25, 30))+'\n')
    return manifest, source


def test_epoch30_and_epoch10_are_paired_once_with_source_preserved(tmp_path):
    root = tmp_path/'continuation'
    root.mkdir()
    manifest, source = _fixture(root)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = summarize(root)
    assert result['actual_checkpoint_epoch'] == 30
    assert result['n'] == 639
    assert len(result['comparisons']) == 6
    assert set(result['epoch_extension_comparisons']) == set(ARMS)
    assert result['extension_chosen_after_dev_epoch10']
    assert not result['formal_certificate']
    for arm in ARMS:
        assert result['models'][arm]['principal_policy']['selected_n'] == 79
        assert result['models'][arm]['principal_policy']['used_wells'] == 158
        with np.load(root/f'{arm}_oof_predictions.npz', allow_pickle=False) as saved:
            assert saved['ids'].tolist() == manifest['ids']
        change = result['epoch_extension_comparisons'][arm]
        assert change['u_mse']['mean'] == pytest.approx(0 if arm == 'A_HR' else -.75)
        assert change['gamma_crps']['mean'] == pytest.approx(0 if arm == 'A_HR' else -.01)
        assert change['overlap']['intersection_n'] == 79
        assert change['fdp']['interval95'] == [0, 0]
    history = result['models']['B_MLP']['folds'][0]['u']['checkpoint']['validation_selection_appendix']
    assert [r['epoch'] for r in history] == [0, 5, 10, 15, 20, 25, 30]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert all((root/name).is_file() for name in ('summary.json', 'SUMMARY.md', 'REPORT.md'))


@pytest.mark.parametrize('mismatch', ['epoch', 'target', 'fold'])
def test_wrong_epoch_or_changed_source_identity_is_rejected(tmp_path, mismatch):
    root = tmp_path/'continuation'
    root.mkdir()
    _, source = _fixture(root)
    if mismatch == 'epoch':
        path = root/'folds/fold_1/arms/B_MLP/test/metrics.json'
        metric = json.loads(path.read_text())
        metric['model']['actual_checkpoint_epoch'] = 10
        write_json(path, metric)
        match = 'actual epoch 30'
    else:
        path = source/'B_MLP_oof_predictions.npz'
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays['actual_u' if mismatch == 'target' else 'fold'].flat[0] += 1
        np.savez_compressed(path, **arrays)
        match = 'identical original Gamma' if mismatch == 'target' else 'outer-fold allocation'
    with pytest.raises(ValueError, match=match):
        summarize(root)
    assert not (root/'summary.json').exists()


def test_manifest_declares_thirty_not_ten(tmp_path):
    root = tmp_path/'continuation'
    root.mkdir()
    manifest, _ = _fixture(root)
    manifest['config']['stage_epochs'] = 10
    with pytest.raises(ValueError, match='stage_epochs=30'):
        _validate_manifest(manifest)
