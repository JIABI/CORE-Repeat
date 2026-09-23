"""Small synthetic file fixtures validate reporting, not biological performance."""
import json
import shutil

import numpy as np
import pytest

from opal2 import state_biology_summary as summary


def write_json(path, value):
    path.write_text(json.dumps(value))


def replace_npz(path, **changes):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    np.savez_compressed(path, **(arrays | changes))


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, 'BOOTSTRAP', 25)
    root, source = tmp_path/'state', tmp_path/'reference'
    root.mkdir(); source.mkdir()
    n = 50
    ids = np.asarray([f'object_{i:03d}' for i in range(n)])
    allocation = np.repeat(np.arange(5), 10)
    records = [dict(fold=f, test=np.flatnonzero(allocation==f).tolist()) for f in range(5)]
    cfg = dict(folds=5, samples=10000, seed=813, stage_epochs=50, max_epochs=100)
    manifest = dict(ids=ids.tolist(), groups=[f'chemical_{i//2}' for i in range(n)],
        folds=records, config=cfg, fixed_epochs=50, actual_checkpoint_epoch=50,
        arms=list(summary.ARMS), reference_run=str(source), scopes={'test': 'original'},
        dataset=dict(units=[dict(layout_block=str(i%5)) for i in range(n)]),
        final_opened=False, fifth_repeat_opened=False, original_endpoint_changed=False,
        original_contract_changed=False, covariance_updated_by_branch=False, formal_certificate=False)
    write_json(root/'run_manifest.json', manifest)
    write_json(source/'run_manifest.json', dict(manifest, fixed_epochs=30,
        arms=['A_FROZEN', 'A_PLUS_OLD', 'A_PLUS_BIO', 'A_PLUS_OLD_SCALED', 'A_PLUS_BIO_SCALED']))
    actual = np.linspace(-.08, .12, n)[:, None]*np.asarray([[.4, .6, 1.]])
    target = np.arange(n*9).reshape(n, 9)/(n*9)
    baseline = target+.08*np.cos(np.arange(n)[:, None]+np.arange(9)[None])
    support = np.stack((np.arange(n)%2==0, np.arange(n)%5==0), axis=1)
    for record in records:
        ix = np.asarray(record['test'])
        fold = root/'folds'/f"fold_{record['fold']}"
        fold.mkdir(parents=True)
        write_json(fold/'complete.json', dict(arms=list(summary.ARMS), branch_epochs=50))
        for number, arm in enumerate(summary.ARMS):
            folder = fold/'arms'/arm
            evaluation = folder/'evaluation'
            evaluation.mkdir(parents=True)
            write_json(evaluation/'metrics.json', dict(samples=10000))
            shift = np.full(len(ix), number*.001)
            if arm in ('BIO_TEMPLATE50', 'STATE_BIO50'):
                shift *= support[ix].any(1)
            draws = np.broadcast_to(actual[ix]+shift[:, None], (10000, len(ix), 3)).copy()
            draws[:2000] += .03
            draws[2000:] -= .01
            np.savez_compressed(evaluation/'predictions.npz', ids=ids[ix], actual=actual[ix],
                predicted=draws.mean(0), p_null=(draws<=0).mean(0), utility_samples=draws,
                utility_crps=np.broadcast_to(.02+shift[:, None], (len(ix), 3)), geometry_energy=.7+shift)
            mean = baseline[ix]+shift[:, None]
            np.savez_compressed(evaluation/'u_predictions.npz', ids=ids[ix], actual_u=target[ix],
                mean_u=mean, covariance_u=np.broadcast_to(np.eye(9), (len(ix), 9, 9)))
            diagnostic = dict(ids=ids[ix], baseline_mean=baseline[ix], mean=mean)
            if arm!='A_FROZEN':
                diagnostic.update(support=support[ix] if arm!='STATE50' else np.ones((len(ix), 2), bool),
                    block_contributions=np.zeros((len(ix), 2, 9)), raw_increment=np.zeros((len(ix), 9)),
                    channel_gate=np.full((len(ix), 2), .3))
                write_json(folder/'training_complete.json', dict(actual_checkpoint_epoch=50,
                    trainable_parameters=1794 if arm=='BIO_TEMPLATE50' else 6770,
                    frozen_A_changed=False, objective_buffers_changed=False, disabled_equals_A=True))
                write_json(folder/'training_config.json', dict(config=cfg, training_seed=813+record['fold'],
                    fit_ids=ids[allocation!=record['fold']][:30].tolist(),
                    validation_ids=ids[allocation!=record['fold']][30:].tolist(), test_data_supplied=False,
                    loss='geometry MSE + normalized joint Gamma CRPS + 0.1 * mean increment squared',
                    covariance='unchanged original fold RIDGE OOF covariance'))
                (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e, fit_u_mse=.3,
                    validation_u_mse=.4)) for e in range(0, 51, 5))+'\n')
            np.savez_compressed(folder/'model_diagnostics.npz', **diagnostic)
            if arm=='A_FROZEN':
                shutil.copytree(folder, source/'folds'/f"fold_{record['fold']}"/'arms'/arm)
    return root, source, manifest, ids, allocation, support


def test_complete_fixed50_primary_equalcapacity_and_sparse_comparisons(saved_run):
    root, source, manifest, ids, allocation, support = saved_run
    result = summary.summarize(root)
    assert result['complete'] and result['fixed_epoch']==50 and result['n']==50
    assert result['primary_comparison']==['STATE_BIO50', 'STATE50']
    assert result['training_checks']['not_nested_additive_biology_ablation']
    assert result['training_checks']['folds'][0]['trainable_parameters']=={
        'BIO_TEMPLATE50': 1794, 'STATE50': 6770, 'STATE_BIO50': 6770}
    assert result['sources']['frozen_A_source_epoch']==30
    assert result['support_analysis']['supported_n']==int(support.any(1).sum())
    assert result['support_analysis']['biological_unsupported_exact_A']
    assert result['support_analysis']['state_only_can_update_biologically_unsupported']
    assert set(result['comparisons'])==set(result['layout_block_sensitivity'])
    pair=result['comparisons']['STATE_BIO50__minus__STATE50']
    assert 'selection_changes' in pair
    direction=result['direction_analysis']['comparisons']['STATE_BIO50__minus__STATE50']['all']
    assert direction['observed_mse_change']['estimate']==pytest.approx(
        direction['direction_energy']['estimate']-direction['twice_residual_alignment']['estimate'])
    for arm in summary.ARMS:
        assert result['models'][arm]['principal_policy']['selected_n']==5
        assert result['models'][arm]['principal_policy']['used_wells']==10
        with np.load(root/(arm+'_oof.npz')) as z:
            assert np.array_equal(z['ids'], ids) and np.array_equal(z['fold'], allocation)
    report=(root/'REPORT.md').read_text()
    assert '第50轮' in report and 'noCERT' in report and '不是严格' in report
    assert result['formal_certificate'] is False
    assert not (source/'REPORT.md').exists() and not (source/'summary.json').exists()


@pytest.mark.parametrize('fault,match', [
    ('partial', 'not complete'), ('epoch', 'actual epoch50'),
    ('capacity', 'unequal active capacity'), ('seed', 'scope, seed'),
    ('source', 'Frozen A copied artifact'), ('target', 'different realized targets'),
    ('covariance', 'joint covariance changed'), ('ids', 'Prediction identity/order'),
    ('support', 'Biological support differs'), ('baseline', 'identical saved A'),
    ('unsupported', 'without biological support'), ('unsupported_score', 'Unsupported biological scores'),
    ('samples', '10000 samples'), ('schedule', 'training schedule'),
])
def test_invalid_artifacts_publish_no_partial_success(saved_run, fault, match):
    root, source, manifest, ids, allocation, support=saved_run
    folder=root/'folds/fold_0/arms/STATE_BIO50'
    if fault=='partial':
        (root/'folds/fold_4/complete.json').unlink()
    elif fault in ('epoch', 'capacity'):
        path=folder/'training_complete.json'; value=json.loads(path.read_text())
        value['actual_checkpoint_epoch' if fault=='epoch' else 'trainable_parameters']=30 if fault=='epoch' else 6771
        write_json(path, value)
    elif fault in ('seed', 'schedule'):
        path=folder/'training_config.json'; value=json.loads(path.read_text())
        if fault=='seed': value['training_seed']+=1
        else: value['config']['max_epochs']=50
        write_json(path, value)
    elif fault=='source':
        (root/'folds/fold_0/arms/A_FROZEN/evaluation/metrics.json').write_text('{"samples": 10000}\n')
    elif fault in ('target', 'covariance'):
        path=folder/'evaluation/u_predictions.npz'; key='actual_u' if fault=='target' else 'covariance_u'
        with np.load(path) as z: value=z[key]+.001
        replace_npz(path, **{key:value})
    elif fault=='ids':
        replace_npz(folder/'evaluation/predictions.npz', ids=ids[:10][::-1])
    elif fault in ('support','baseline'):
        path=folder/'model_diagnostics.npz';key='support' if fault=='support' else 'baseline_mean'
        with np.load(path) as z:value=z[key].copy()
        value[0,0]=not value[0,0] if fault=='support' else value[0,0]+1e-10
        replace_npz(path, **{key:value})
    elif fault=='unsupported':
        row=np.flatnonzero(~support[:10].any(1))[0];path=folder/'evaluation/u_predictions.npz'
        with np.load(path) as z: value=z['mean_u'].copy()
        value[row,0]+=1e-10
        replace_npz(path, mean_u=value);replace_npz(folder/'model_diagnostics.npz', mean=value)
    elif fault=='unsupported_score':
        row=np.flatnonzero(~support[:10].any(1))[0];path=folder/'evaluation/predictions.npz'
        with np.load(path) as z:value=z['utility_crps'].copy()
        value[row,2]+=1e-10;replace_npz(path, utility_crps=value)
    elif fault=='samples':
        write_json(folder/'evaluation/metrics.json', dict(samples=2000))
    with pytest.raises(ValueError, match=match): summary.summarize(root)
    assert not (root/'summary.json').exists() and not (root/'REPORT.md').exists()
    assert not (root/'A_FROZEN_oof.npz').exists()


def test_no_global_epoch_override_and_template_capacity_can_differ(saved_run):
    root, _, manifest, ids, allocation, _=saved_run
    with pytest.raises(ValueError, match='actual epoch30'):
        summary.read_arm(root, manifest, ids, allocation, 'BIO_TEMPLATE50', fitted_epochs=30)
    data, rows, _=summary.read_arm(root, manifest, ids, allocation, 'BIO_TEMPLATE50', fitted_epochs=50)
    assert rows[0]['training']['trainable_parameters']==1794
    assert data['actual_u'].shape==(50,9)


def test_manifest_scope_and_grouping_cannot_change(saved_run):
    _, _, manifest, *_=saved_run
    with pytest.raises(ValueError, match='scope changed'):
        summary.validate_manifest(dict(manifest, final_opened=True))
    with pytest.raises(ValueError, match='chemistry group crosses'):
        summary.validate_manifest(dict(manifest, groups=['one']*50))
    with pytest.raises(ValueError, match='four declared arms'):
        summary.validate_manifest(dict(manifest, fixed_epochs=30))
