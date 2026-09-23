"""Saved-output checks for the independently gated biological increment."""
import json

import numpy as np
import pytest

from opal2 import independent_biology_summary as summary


def _write_json(path, value):
    path.write_text(json.dumps(value))


def _replace_npz(path, **changes):
    with np.load(path, allow_pickle=False) as saved:
        content = {key: saved[key].copy() for key in saved.files}
    content.update(changes)
    np.savez_compressed(path, **content)


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    # Ten objects per fold support a one-object/two-well principal policy.
    # This fixture checks output contracts; it is not an experiment result.
    monkeypatch.setattr(summary, 'BOOTSTRAP', 40)
    n = 50
    ids = np.asarray([f'object_{i:03d}' for i in range(n)])
    allocation = np.repeat(np.arange(5), 10)
    records = [dict(fold=f, test=np.flatnonzero(allocation == f).tolist()) for f in range(5)]
    manifest = dict(ids=ids.tolist(), arms=list(summary.ARMS), fixed_epochs=30,
        config=dict(folds=5, samples=10000, seed=812), folds=records,
        groups=[f'chemical_{i}' for i in range(n)],
        dataset=dict(units=[dict(layout_block=str(i % 5)) for i in range(n)]),
        final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False)
    _write_json(tmp_path/'run_manifest.json', manifest)
    actual = np.linspace(-.08, .12, n)[:, None]*np.asarray([[.4, .6, 1.]])
    actual_u = np.arange(n*9, dtype=float).reshape(n, 9)/(n*9)
    baseline = actual_u+.08*np.cos(np.arange(n)[:, None]+np.arange(9)[None])
    support = np.stack((np.arange(n) % 2 == 0, np.arange(n) % 5 == 0), 1)
    for record in records:
        ix = np.asarray(record['test'])
        fold = tmp_path/'folds'/f"fold_{record['fold']}"
        fold.mkdir(parents=True)
        _write_json(fold/'complete.json', dict(complete=True))
        for arm in summary.ARMS:
            folder = fold/'arms'/arm
            evaluation = folder/'evaluation'
            evaluation.mkdir(parents=True)
            _write_json(evaluation/'metrics.json', dict(samples=10000))
            shift = np.zeros(len(ix))
            if arm == 'A_PLUS_OLD':
                shift[:] = .002
            elif arm == 'A_PLUS_BIO':
                shift = support[ix].any(1)*.001
            draws = np.broadcast_to(actual[ix]+shift[:, None], (10000, len(ix), 3)).copy()
            draws[:2000] += .03
            draws[2000:] -= .01
            np.savez_compressed(evaluation/'predictions.npz', ids=ids[ix], actual=actual[ix],
                predicted=draws.mean(0), p_null=(draws <= 0).mean(0), utility_samples=draws,
                utility_crps=np.broadcast_to(.02+shift[:, None], (len(ix), 3)),
                geometry_energy=.7+shift)
            mean = baseline[ix]+shift[:, None]
            np.savez_compressed(evaluation/'u_predictions.npz', ids=ids[ix],
                actual_u=actual_u[ix], mean_u=mean)
            numeric = dict(ids=ids[ix], baseline_mean=baseline[ix], mean=mean)
            if arm != 'A_FROZEN':
                numeric.update(support=support[ix] if arm == 'A_PLUS_BIO' else np.ones((len(ix), 2), bool),
                    channel_gate=np.ones((len(ix), 2))*.3,
                    block_contributions=np.zeros((len(ix), 2, 9)), raw_increment=np.zeros((len(ix), 9)))
                _write_json(folder/'training_complete.json', dict(actual_checkpoint_epoch=30,
                    trainable_parameters=1794, best_validation_epoch=10))
                (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e, validation_mse=.3))
                    for e in (0, 10, 20, 30))+'\n')
            np.savez_compressed(folder/'model_diagnostics.npz', **numeric)
    return tmp_path, manifest, ids, allocation, support


def test_complete_summary_pairs_equal_capacity_and_preserves_non_support(saved_run):
    root, manifest, ids, allocation, support = saved_run
    result = summary.summarize(root)
    assert result['complete'] and result['n'] == 50
    assert result['fixed_epoch'] == 30 and result['samples'] == 10000
    assert result['formal_certificate'] is False
    assert set(result['comparisons']) == {a+'__minus__'+b for a, b in summary.PAIRS}
    assert result['support_analysis']['supported_n'] == int(support.any(1).sum())
    assert result['support_analysis']['exact_zero_change_without_support']
    assert result['support_analysis']['unsupported_max_absolute_mean_change'] == 0
    for arm in summary.ARMS:
        policy = result['models'][arm]['principal_policy']
        assert policy['selected_n'] == 5 and policy['used_wells'] == 10
        assert policy['eligible_n'] == 50
        assert (root/f'{arm}_oof.npz').is_file()
    assert result['models']['A_PLUS_BIO']['folds'][0]['training']['trainable_parameters'] == 1794
    assert result['same_budget_random']['selected_n'] == 5
    assert result['fixed_plans']['ALL_Z1']['added_wells'] == 50
    assert result['fixed_plans']['ALL_Z1Z2']['added_wells'] == 100
    assert (root/'REPORT.md').is_file() and (root/'summary.json').is_file()
    # The biology-defined subset is held constant for every comparator.
    subset = result['support_analysis']['subsets']['supported']['models']
    assert len({subset[arm]['n'] for arm in summary.ARMS}) == 1
    assert subset['A_PLUS_OLD']['selection_not_reranked_within_subset']


def test_missing_fold_completion_does_not_publish_partial_success(saved_run):
    root, *_ = saved_run
    (root/'folds/fold_4/complete.json').unlink()
    with pytest.raises(ValueError, match='not complete'):
        summary.summarize(root)
    assert not (root/'summary.json').exists()
    assert not (root/'REPORT.md').exists()


def test_epoch30_and_equal_dynamic_capacity_are_enforced(saved_run):
    root, *_ = saved_run
    path = root/'folds/fold_0/arms/A_PLUS_BIO/training_complete.json'
    _write_json(path, dict(actual_checkpoint_epoch=10, trainable_parameters=1794))
    with pytest.raises(ValueError, match='actual epoch30'):
        summary.summarize(root)
    _write_json(path, dict(actual_checkpoint_epoch=30, trainable_parameters=1795))
    with pytest.raises(ValueError, match='unequal active capacity'):
        summary.summarize(root)


def test_all_draws_not_prefix_determine_predictive_mean_and_null(saved_run):
    root, manifest, ids, allocation, _ = saved_run
    path = root/'folds/fold_0/arms/A_FROZEN/evaluation/predictions.npz'
    with np.load(path, allow_pickle=False) as saved:
        draws = saved['utility_samples'].copy()
        original_mean = saved['predicted'].copy()
    _replace_npz(path, predicted=draws[:2000].mean(0))
    with pytest.raises(ValueError, match='means'):
        summary.read_arm(root, manifest, ids, allocation, 'A_FROZEN')
    _replace_npz(path, predicted=original_mean, p_null=(draws[:2000] <= 0).mean(0))
    with pytest.raises(ValueError, match='NULL probabilities'):
        summary.read_arm(root, manifest, ids, allocation, 'A_FROZEN')


def test_targets_must_match_in_geometry_and_utility(saved_run):
    root, *_ = saved_run
    path = root/'folds/fold_0/arms/A_PLUS_BIO/evaluation/u_predictions.npz'
    with np.load(path, allow_pickle=False) as saved:
        actual = saved['actual_u'].copy()
    _replace_npz(path, actual_u=actual+.001)
    with pytest.raises(ValueError, match='different realized targets'):
        summary.summarize(root)


def test_supported_mean_and_no_support_exact_recovery_are_separate_checks(saved_run):
    root, manifest, ids, allocation, support = saved_run
    data, diagnostic = {}, {}
    for arm in summary.ARMS:
        data[arm], _, diagnostic[arm] = summary.read_arm(root, manifest, ids, allocation, arm)
    weights = summary.bootstrap_weights(manifest['groups'], 40, 77)
    row = np.flatnonzero(~support.any(1))[0]
    data['A_PLUS_BIO']['mean_u'][row, 0] += 1e-12
    with pytest.raises(ValueError, match='without biological support'):
        summary.support_checks(data, diagnostic, weights)
    data['A_PLUS_BIO']['mean_u'][row] = data['A_FROZEN']['mean_u'][row]
    diagnostic['A_PLUS_BIO']['baseline_mean'][row, 0] += 1e-12
    with pytest.raises(ValueError, match='identical saved A'):
        summary.support_checks(data, diagnostic, weights)


def test_manifest_cannot_split_chemistry_groups_or_change_scope(saved_run):
    _, manifest, *_ = saved_run
    changed = dict(manifest, groups=['same_chemical']*50)
    with pytest.raises(ValueError, match='chemistry group crosses'):
        summary.validate_manifest(changed)
    changed = dict(manifest, original_endpoint_changed=True)
    with pytest.raises(ValueError, match='scope changed'):
        summary.validate_manifest(changed)


def test_diagnostics_cannot_silently_use_different_prediction_ids(saved_run):
    root, manifest, ids, allocation, _ = saved_run
    path = root/'folds/fold_0/arms/A_PLUS_BIO/model_diagnostics.npz'
    _replace_npz(path, ids=ids[:10][::-1])
    with pytest.raises(ValueError, match='Diagnostic identity/order'):
        summary.read_arm(root, manifest, ids, allocation, 'A_PLUS_BIO')
