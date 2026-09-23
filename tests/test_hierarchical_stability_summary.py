"""Synthetic saved predictions test averaging, pairing and policy accounting."""
import json

import numpy as np
import pytest

from opal2.baseline_policy import ACTIONS, WELL_COSTS, _metrics
from opal2.hierarchical_stability_summary import ARMS, summarize, check_repeated_outcomes


def fixture(root):
    n = 40
    ids = np.asarray([f'synthetic_{i:03}' for i in range(n)])
    rng = np.random.default_rng(23)
    actual = rng.normal(0, .08, (n, 3))
    native_u = rng.normal(size=(n, 9))
    cfg = dict(seed=17, bootstrap=80, folds=5, samples=20)
    flags = dict(final_opened=False, fifth_repeat_opened=False,
                 original_endpoint_changed=False, original_contract_changed=False)
    repetitions = []
    score_delta = np.linspace(-.012, .008, n)[:, None]*np.asarray([[.7, .9, 1.]])
    for r in range(3):
        order = np.random.default_rng(91+r).permutation(n)
        folds = [dict(fold=f, test=order[f*8:(f+1)*8].tolist(), seed=101+r*10+f)
                 for f in range(5)]
        repetitions.append(dict(repeat=r, seed=91+r, folds=folds))
        folder = root/'repetitions'/f'repeat_{r}'
        folder.mkdir(parents=True)
        manifest = dict(ids=ids.tolist(), arms=list(ARMS), folds=folds, config=cfg, **flags)
        (folder/'run_manifest.json').write_text(json.dumps(manifest))
        for ai, arm in enumerate(ARMS):
            for fold in folds:
                ix = np.asarray(fold['test'])
                target = native_u[ix]+r*.1+fold['fold']*.01
                prediction = target + (1.2 if ai == 1 else 1 if ai == 2 else .7 if ai > 2 else 2.)
                base_crps = .07+np.arange(n)[:, None]*.0001+np.zeros((n, 3))
                crps = base_crps[ix].copy()
                if ai == 1:
                    crps += .002
                elif ai > 2:
                    # Across r and HR seeds these offsets average to zero.
                    crps += score_delta[ix]+(r-1)*.003+(ai-4)*.004
                mean = np.full((len(ix), 3), fold['fold']*.1) if ai == 0 else (
                    .08*actual[ix]+np.sin(ix[:, None]+ai)*.007)
                pnull = np.full((len(ix), 3), .5)
                dest = folder/'folds'/f"fold_{fold['fold']}"/'arms'/arm/'test'
                dest.mkdir(parents=True)
                np.savez_compressed(dest/'predictions.npz', ids=ids[ix], actual=actual[ix],
                    predicted=mean, p_null=pnull, utility_crps=crps, geometry_energy=np.ones(len(ix)))
                np.savez_compressed(dest/'u_predictions.npz', ids=ids[ix], actual_u=target,
                                    mean_u=prediction)
                metrics = dict(action_metrics=[dict(action=a, **_metrics(actual[ix, j], mean[:, j], pnull[:, j]))
                                               for j, a in enumerate(ACTIONS)],
                               utility=[dict(crps=float(crps[:, j].mean())) for j in range(3)])
                (dest/'metrics.json').write_text(json.dumps(metrics))
        for fold in folds:
            (folder/'folds'/f"fold_{fold['fold']}"/'complete.json').write_text(json.dumps(dict(
                repeat=r, fold=fold['fold'], completed_utc='2026-09-14T00:00:00Z', test_n=len(fold['test']))))
    (root/'run_manifest.json').write_text(json.dumps(dict(ids=ids.tolist(), arms=list(ARMS),
        repetitions=repetitions, config=cfg, **flags)))
    return ids, actual, score_delta


def test_unique_id_average_before_bootstrap_and_no_ensemble(tmp_path):
    ids, _, delta = fixture(tmp_path)
    result = summarize(tmp_path)
    primary = result['primary_HR_mean_vs_RIDGE_VALID']
    boots = np.random.default_rng(17).integers(40, size=(80, 40))
    np.testing.assert_allclose(primary['gamma_crps']['mean'], delta.mean(0))
    np.testing.assert_allclose(primary['gamma_crps']['interval95'],
                               np.quantile(delta[boots].mean(1), [.025, .975], axis=0))
    np.testing.assert_allclose(primary['u_mean_mse']['mean'], -.51)
    np.testing.assert_allclose(result['selection_info_RIDGE_VALID_vs_TRAINCV']['gamma_crps']['mean'], -.002)
    assert result['bootstrap_sample_size'] == result['n_unique_compounds'] == 40
    assert primary['scored_run_pairs'] == 9
    assert result['repeated_rows_are_independent'] is False
    assert result['best_seed_selected'] is result['ensemble_prediction'] is False
    assert result['activation_counts_combined_for_certification'] is False
    assert len(result['direction_analysis']['runs']) == 9
    assert len(result['direction_analysis']['folds']) == 45
    assert result['direction_analysis']['direction_counts']['fold_HR_results']['u_mse']['favorable'] == 45
    assert primary['principal_policy']['selected_n_per_run_left'] == [5]*9
    with np.load(tmp_path/'primary_compound_contributions.npz') as saved:
        np.testing.assert_array_equal(saved['ids'], ids)
        np.testing.assert_allclose(saved['gamma_crps_difference'], delta)
        assert saved['policy_net_value_difference'].shape == (36, 40)
    assert (tmp_path/'summary.json').is_file() and (tmp_path/'REPORT.md').is_file()


def test_global_uniform_expectation_in_every_policy(tmp_path):
    _, actual, _ = fixture(tmp_path)
    result = summarize(tmp_path)
    for report in result['repetitions']:
        model = report['models']['GLOBAL_GEOMETRY']
        assert len(model['policies']) == 36
        assert model['actions'][2]['spearman'] is None
        assert model['actions'][2]['within_fold_rank_association'] is None
        with np.load(tmp_path/f"repeat_{report['repeat']}_GLOBAL_GEOMETRY_oof_predictions.npz") as saved:
            for i, row in enumerate(model['policies']):
                weights = saved['masks'][i]
                for fold in range(5):
                    sub = weights[saved['fold'] == fold]
                    np.testing.assert_allclose(sub, sub[0])
                action = ACTIONS.index(row['action'])
                np.testing.assert_allclose(row['per_eligible_net_gain'], weights@actual[:, action]/40)
                assert row['used_wells'] == row['selected_n']*WELL_COSTS[action]
                if row['selected_n']:
                    np.testing.assert_allclose(row['per_selected_net_gain'], actual[:, action].mean())


def test_changed_repeat_target_rejected(tmp_path):
    fixture(tmp_path)
    # All six arms agree within the repeat, but one repeat changes the endpoint.
    for arm in ARMS:
        path = tmp_path/'repetitions/repeat_2/folds/fold_0/arms'/arm/'test/predictions.npz'
        with np.load(path) as saved:
            values = {k:saved[k].copy() for k in saved.files}
        values['actual'][0, 2] += 1
        np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='preserve each ID original outcome'):
        summarize(tmp_path)
    assert not (tmp_path/'summary.json').exists()


def test_incomplete_fold_rejected_even_if_predictions_exist(tmp_path):
    fixture(tmp_path)
    (tmp_path/'repetitions/repeat_0/folds/fold_0/complete.json').unlink()
    with pytest.raises(ValueError, match='no completion marker'):
        summarize(tmp_path)
    assert not (tmp_path/'summary.json').exists()


def test_roundoff_audited_without_replacing_outcomes_or_source_files(tmp_path):
    source, output = tmp_path/'source', tmp_path/'report'
    fixture(source)
    for arm in ARMS:
        path = source/'repetitions/repeat_1/folds/fold_0/arms'/arm/'test/predictions.npz'
        with np.load(path) as saved:
            values = {k:saved[k].copy() for k in saved.files}
        values['actual'][0,2] = np.nextafter(np.nextafter(values['actual'][0,2], np.inf), np.inf)
        np.savez_compressed(path, **values)
    files = {p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result = summarize(source, output=output)
    audit = result['repeated_outcome_roundoff_audit'][1]
    assert audit['changed_float_elements'] == 1
    assert audit['max_absolute_difference'] < 1e-15
    assert audit['original_values_replaced'] is False
    assert audit['null_label_changes'] == audit['positive_label_changes'] == 0
    assert (output/'summary.json').is_file() and not (source/'summary.json').exists()
    assert set(p for p in source.rglob('*') if p.is_file()) == set(files)
    for path, content in files.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize('boundary', [0., .005])
def test_even_tiny_label_boundary_crossings_rejected(boundary):
    ref = np.array([[.1, .1, boundary]])
    current = ref.copy()
    current[0,2] = np.nextafter(boundary, np.inf if boundary == 0 else -np.inf)
    with pytest.raises(ValueError, match='changed label'):
        check_repeated_outcomes(ref, current, np.array(['synthetic']), 1)
