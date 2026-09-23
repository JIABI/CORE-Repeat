"""Focused saved-artifact contracts for the five-arm scale comparison."""
import json
import shutil

import numpy as np
import pytest

from opal2 import independent_biology_scale_summary as summary


def _json(path, value):
    path.write_text(json.dumps(value))


def _replace_npz(path, **changes):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    np.savez_compressed(path, **(arrays | changes))


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'BOOTSTRAP', 30)
    monkeypatch.setattr(summary.original, 'BOOTSTRAP', 30)
    root, source = tmp_path/'scaled', tmp_path/'original'
    root.mkdir(); source.mkdir()
    n = 50
    ids = np.asarray([f'object_{i:03d}' for i in range(n)])
    allocation = np.repeat(np.arange(5), 10)
    records = [dict(fold=f, test=np.flatnonzero(allocation == f).tolist()) for f in range(5)]
    manifest = dict(ids=ids.tolist(), groups=[f'chemical_{i // 2}' for i in range(n)],
        folds=records, config=dict(folds=5, samples=10000, seed=812), fixed_epochs=30,
        arms=list(summary.ARMS), reference_run=str(source),
        dataset=dict(units=[dict(layout_block=str(i % 5)) for i in range(n)]),
        final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False,
        original_contract_changed=False, covariance_updated_by_branch=False, formal_certificate=False)
    _json(root/'run_manifest.json', manifest)
    _json(source/'run_manifest.json', dict(manifest, arms=list(summary.original.ARMS)))
    actual = np.linspace(-.08, .12, n)[:, None]*np.asarray([[.4, .6, 1.]])
    target_u = np.arange(n*9, dtype=float).reshape(n, 9)/(n*9)
    baseline = target_u+.08*np.cos(np.arange(n)[:, None]+np.arange(9)[None])
    support = np.stack((np.arange(n) % 2 == 0, np.arange(n) % 5 == 0), 1)
    for record in records:
        ix = np.asarray(record['test'])
        fold = root/'folds'/f"fold_{record['fold']}"
        fold.mkdir(parents=True)
        _json(fold/'complete.json', dict(arms=list(summary.ARMS), branch_epochs=30))
        for a, arm in enumerate(summary.ARMS):
            folder = fold/'arms'/arm
            evaluation = folder/'evaluation'
            evaluation.mkdir(parents=True)
            _json(evaluation/'metrics.json', dict(samples=10000))
            shift = np.full(len(ix), a*.001)
            if 'BIO' in arm:
                shift *= support[ix].any(1)
            draws = np.broadcast_to(actual[ix]+shift[:, None], (10000, len(ix), 3)).copy()
            draws[:2000] += .03
            draws[2000:] -= .01
            np.savez_compressed(evaluation/'predictions.npz', ids=ids[ix], actual=actual[ix],
                predicted=draws.mean(0), p_null=(draws <= 0).mean(0), utility_samples=draws,
                utility_crps=np.broadcast_to(.02+shift[:, None], (len(ix), 3)), geometry_energy=.7+shift)
            mean = baseline[ix]+shift[:, None]
            np.savez_compressed(evaluation/'u_predictions.npz', ids=ids[ix], actual_u=target_u[ix], mean_u=mean)
            diagnostic = dict(ids=ids[ix], baseline_mean=baseline[ix], mean=mean)
            if arm != 'A_FROZEN':
                diagnostic.update(support=support[ix] if 'BIO' in arm else np.ones((len(ix), 2), bool),
                    channel_gate=np.full((len(ix), 2), .3), block_contributions=np.zeros((len(ix), 2, 9)),
                    raw_increment=np.zeros((len(ix), 9)))
                _json(folder/'training_complete.json', dict(actual_checkpoint_epoch=30, trainable_parameters=1794,
                    best_epoch=0, frozen_A_changed=False, objective_buffers_changed=False, disabled_equals_A=True))
                config = dict(config=dict(stage_epochs=30, samples=10000), training_seed=812+record['fold'],
                    fit_ids=ids[allocation != record['fold']][:30].tolist(),
                    validation_ids=ids[allocation != record['fold']][30:].tolist(),
                    loss='geometry MSE + normalized joint Gamma CRPS + 0.1 * mean increment squared',
                    covariance='unchanged original fold RIDGE OOF covariance', test_data_supplied=False)
                _json(folder/'training_config.json', config)
                history = [dict(epoch=e, fit_u_mse=.3-a*.001*e/30, fit_gamma_crps=.03,
                    fit_normalized_gamma_crps=.35, fit_incremental_mse=.0001*e/30,
                    validation_u_mse=.4+a*.001*e/30, validation_gamma_crps=.04,
                    validation_normalized_gamma_crps=.45, validation_incremental_mse=.0002*e/30) for e in (0, 30)]
                (folder/'history.jsonl').write_text('\n'.join(map(json.dumps, history))+'\n')
                if arm in summary.SCALED_ARMS:
                    _json(folder/'aggregation_scale.json', dict(aggregation_scaling='train_fixed', fitted=True,
                        fit_ids=config['fit_ids'], channels=['channel1', 'channel2'], scale_max_gain=32., scale_floor=1./32.,
                        count=[30, 10], raw_s=[.125, .025], gain=[8., 32.], capped=[False, True]))
                    diagnostic.update(raw_basis_rms=np.asarray([.1, .2]), scaled_basis_rms=np.asarray([.8, 6.4]),
                        saturation_fraction=np.asarray(.01), inherited_hr_saturation_fraction=np.asarray(.02),
                        raw_increment_max=np.asarray(.03))
            np.savez_compressed(folder/'model_diagnostics.npz', **diagnostic)
            if arm in summary.original.ARMS:
                shutil.copytree(folder, source/'folds'/f"fold_{record['fold']}"/'arms'/arm)
    return root, source, manifest, ids, allocation, support


def test_five_arms_common_support_scale_metadata_and_decisions(saved_run):
    root, source, manifest, ids, allocation, support = saved_run
    result = summary.summarize(root)
    assert result['complete'] and result['arms'] == list(summary.ARMS)
    assert result['fixed_epoch'] == 30 and result['samples'] == 10000
    assert result['certificate_status'] == 'noCERT' and result['formal_certificate'] is False
    assert result['sources']['originals_identical']
    assert set(result['comparisons']) == set(result['layout_block_sensitivity']) == {
        a+'__minus__'+b for a, b in summary.PAIRS}
    assert result['support_analysis']['supported_n'] == int(support.any(1).sum())
    assert result['support_analysis']['support_unchanged_by_scale']
    for arm in summary.ARMS:
        assert result['models'][arm]['principal_policy']['selected_n'] == 5
        assert result['models'][arm]['principal_policy']['used_wells'] == 10
        subset = result['support_analysis']['subsets']['supported']['models'][arm]
        assert subset['n'] == int(support.any(1).sum())
        assert subset['selection_not_reranked_within_subset']
        with np.load(root/f'{arm}_oof.npz') as saved:
            assert np.array_equal(saved['ids'], ids)
            assert np.array_equal(saved['fold'], allocation)
    for arm in summary.SCALED_ARMS:
        assert len(result['aggregation_scale'][arm]) == 5
        assert result['aggregation_scale'][arm][0]['metadata']['gain'] == [8., 32.]
        assert result['aggregation_scale'][arm][0]['diagnostics']['saturation_fraction'] == .01
    trajectory = result['training_trajectory']['comparisons']['A_PLUS_BIO_SCALED__minus__A_PLUS_BIO']
    assert trajectory['descriptive_equal_fold_averages']['fit_monitored_objective']['equal_fold_mean_epoch30_difference'] < 0
    assert trajectory['descriptive_equal_fold_averages']['validation_monitored_objective']['equal_fold_mean_epoch30_difference'] > 0
    pair = result['comparisons']['A_PLUS_BIO_SCALED__minus__A_PLUS_BIO']
    assert 'left_only_ids' in pair['selection_overlap']
    assert result['same_budget_random']['selected_n'] == 5
    report = (root/'REPORT.md').read_text()
    assert 'noCERT' in report and '不保证单位RMS' in report and '第30轮' in report
    assert not (source/'REPORT.md').exists()
    assert not (source/'summary.json').exists()


@pytest.mark.parametrize('fault,match', [
    ('completion', 'not complete'), ('epoch', 'actual epoch30'),
    ('capacity', 'unequal active capacity'), ('seed', 'scope, seed'),
    ('scale_ids', 'FIT identities'), ('scale_gain', 'calibration rule'),
    ('samples', '10000 samples'), ('source', 'Original copied arm artifact changed'),
    ('targets', 'different realized targets'), ('ids', 'Prediction identity/order'),
    ('support', 'changed support'), ('baseline', 'identical saved A'),
    ('unsupported', 'without biological support'), ('unsupported_score', 'scores differ from A'),
])
def test_changed_saved_contracts_do_not_publish_partial_summary(saved_run, fault, match):
    root, source, manifest, ids, allocation, support = saved_run
    folder = root/'folds/fold_0/arms/A_PLUS_BIO_SCALED'
    if fault == 'completion':
        (root/'folds/fold_4/complete.json').unlink()
    elif fault in ('epoch', 'capacity'):
        path = folder/'training_complete.json'
        saved = json.loads(path.read_text())
        saved['actual_checkpoint_epoch' if fault == 'epoch' else 'trainable_parameters'] = 10 if fault == 'epoch' else 1795
        _json(path, saved)
    elif fault == 'seed':
        path = folder/'training_config.json'
        saved = json.loads(path.read_text()); saved['training_seed'] += 1
        _json(path, saved)
    elif fault in ('scale_ids', 'scale_gain'):
        path = folder/'aggregation_scale.json'
        saved = json.loads(path.read_text())
        saved['fit_ids' if fault == 'scale_ids' else 'gain'] = ids[:30].tolist() if fault == 'scale_ids' else [8., 31.]
        _json(path, saved)
    elif fault == 'samples':
        _json(folder/'evaluation/metrics.json', dict(samples=2000))
    elif fault == 'source':
        path = root/'folds/fold_0/arms/A_PLUS_OLD/training_complete.json'
        saved = json.loads(path.read_text()); saved['trainable_parameters'] += 1
        _json(path, saved)
    elif fault == 'targets':
        path = folder/'evaluation/u_predictions.npz'
        with np.load(path) as saved:
            _replace_npz(path, actual_u=saved['actual_u']+.001)
    elif fault == 'ids':
        _replace_npz(folder/'evaluation/predictions.npz', ids=ids[:10][::-1])
    elif fault in ('support', 'baseline'):
        path = folder/'model_diagnostics.npz'
        with np.load(path) as saved:
            key = 'support' if fault == 'support' else 'baseline_mean'
            value = saved[key].copy()
        value[0, 0] = not value[0, 0] if fault == 'support' else value[0, 0]+1e-12
        _replace_npz(path, **{key: value})
    elif fault == 'unsupported':
        row = np.flatnonzero(~support[:10].any(1))[0]
        path = folder/'evaluation/u_predictions.npz'
        with np.load(path) as saved:
            value = saved['mean_u'].copy()
        value[row, 0] += 1e-12
        _replace_npz(path, mean_u=value)
        _replace_npz(folder/'model_diagnostics.npz', mean=value)
    elif fault == 'unsupported_score':
        row = np.flatnonzero(~support[:10].any(1))[0]
        path = folder/'evaluation/predictions.npz'
        with np.load(path) as saved:
            value = saved['utility_crps'].copy()
        value[row, 2] += 1e-12
        _replace_npz(path, utility_crps=value)
    with pytest.raises(ValueError, match=match):
        summary.summarize(root)
    assert not (root/'summary.json').exists()
    assert not (root/'REPORT.md').exists()
    assert not (root/'A_FROZEN_oof.npz').exists()


def test_scale_comparison_scope_and_complete_five_arm_manifest(saved_run):
    _, _, manifest, *_ = saved_run
    for flag in ('formal_certificate', 'covariance_updated_by_branch', 'reference_selection_changed'):
        with pytest.raises(ValueError, match='scope changed'):
            summary.validate_manifest(dict(manifest, **{flag: True}))
    with pytest.raises(ValueError, match='All five'):
        summary.validate_manifest(dict(manifest, arms=list(summary.ARMS[:-1])))


def test_zero_scale_channel_policy_is_retained():
    metadata = dict(aggregation_scaling='train_fixed', fitted=True, fit_ids=['fit'],
        scale_max_gain=32., scale_floor=1./32., count=[0, 1], raw_s=[0., .1],
        gain=[1., 10.], capped=[False, False])
    summary._validate_scale(metadata, ['fit'])
    with pytest.raises(ValueError, match='calibration rule'):
        summary._validate_scale(dict(metadata, gain=[32., 10.]), ['fit'])
