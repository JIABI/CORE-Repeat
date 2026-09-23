"""Complete L adapter and honest matched-score tests, using synthetic profiles."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from opal2.closed_form_baseline import fit_baseline
from opal2.gram_reference import ClosedFormGramReference
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from opal2.gram_oof_ridge import fit_preprocessing
from opal2.objective_analysis import fair_crps

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('matched_l_runner', PROJECT/'scripts/run_jump_matched_l_20260918.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    rng = np.random.default_rng(721)
    y = rng.normal(size=(24, 1, 12)) + rng.normal(size=(24, 4, 12))
    model = fit_baseline(y[:18], k=5, random_state=91)
    actual_gram = profiles_to_gram(torch.from_numpy(y))
    raw = gram_to_coordinates(actual_gram).numpy()
    stats = fit_preprocessing(y[:18], raw[:18])
    return model, y, raw, gram_gains(actual_gram).numpy()[:, 2], stats


def test_existing_complete_sampler_is_used_without_dropping_residual_dimensions():
    model, y, _, _, _ = fixture()
    x = y[-2:, 0]
    actual, mean = runner.sample_original_l(model, x, samples=40, seed=17, draw_chunk=8)
    expected = ClosedFormGramReference(model).sample(x, 40, seed=17, object_chunk_size=2, draw_chunk_size=8)
    np.testing.assert_array_equal(actual, expected.grams)
    np.testing.assert_array_equal(mean, expected.mean_profile_gram)
    assert len(model.residual_var) == 12 and np.all(model.residual_var > 0)


def test_score_matches_original_endpoint_crps_and_never_invents_geometry_density():
    model, y, raw, actual, stats = fixture()
    grams, _ = runner.sample_original_l(model, y[-2:, 0], samples=128, seed=18, draw_chunk=16)
    scored, gamma = runner.score_draws(grams, raw[-2:], actual[-2:], stats, np.square(y[-2:, 0]).mean(1))
    direct_gamma = gram_gains(torch.from_numpy(grams)).numpy()[..., 2]
    np.testing.assert_allclose(gamma, direct_gamma, atol=1e-12)
    np.testing.assert_allclose(scored['crps'], fair_crps(direct_gamma, actual[-2:]), atol=1e-12)
    np.testing.assert_allclose(scored['predicted'], direct_gamma.mean(0), atol=1e-12)
    np.testing.assert_array_equal(scored['p_null'], (direct_gamma <= 0).mean(0))
    assert 'nll' not in scored and 'joint_coverage_by_level' not in scored
    assert scored['observable_crps'].shape == (2, 10)
    assert scored['gamma_coverage_by_level'].shape == (2, 5)
    u = (gram_to_coordinates(torch.from_numpy(grams)).numpy()-np.asarray(stats['u_center']))/np.asarray(stats['u_scale'])
    expected_energy = np.linalg.norm(u-scored['actual_u'][None], axis=-1).mean(0)-.5*np.linalg.norm(u[:64]-u[64:], axis=-1).mean(0)
    np.testing.assert_array_equal(scored['energy'], expected_energy)


def test_future_targets_change_scores_but_never_draws_or_decision_moments():
    model, y, raw, actual, stats = fixture()
    grams, _ = runner.sample_original_l(model, y[-2:, 0], samples=64, seed=31, draw_chunk=16)
    a, ga = runner.score_draws(grams, raw[-2:], actual[-2:], stats, np.square(y[-2:, 0]).mean(1))
    changed = y[-2:].copy()
    changed[:, 1:] += 20.
    altered_gram = profiles_to_gram(torch.from_numpy(changed))
    b, gb = runner.score_draws(grams, gram_to_coordinates(altered_gram).numpy(), gram_gains(altered_gram).numpy()[:, 2], stats, np.square(y[-2:, 0]).mean(1))
    np.testing.assert_array_equal(ga, gb)
    for key in ('predicted', 'p_null', 'mean_u'):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.allclose(a['crps'], b['crps'])


def test_resolve_parts_preserves_order_and_rejects_role_leakage():
    ids = np.asarray([f'ID{i}' for i in range(25)])
    manifest = dict(parts=[])
    for f in range(5):
        chunks = np.roll(np.arange(25).reshape(5, 5), f, axis=0)
        manifest['parts'].append({role: ids[chunk].tolist() for role, chunk in zip(runner.ROLE_KEYS, chunks)})
    parts = runner.resolve_parts(ids, manifest)
    for source, resolved in zip(manifest['parts'], parts):
        for key in runner.ROLE_KEYS:
            assert ids[resolved[key]].tolist() == source[key]
    manifest['parts'][0]['REF_FIT'][0] = manifest['parts'][0]['TRAIN'][0]
    with pytest.raises(ValueError, match='partition'):
        runner.resolve_parts(ids, manifest)


def test_resource_report_separates_available_ceiling_used_labels_and_all_new_cost():
    part = {role: np.arange(n) for role, n in zip(runner.ROLE_KEYS, (246, 61, 102, 102, 128))}
    result = runner.resource_counts(part)
    assert result['common_available_label_wells'] == 2044
    assert result['L_used_label_wells'] == 984
    assert result['L_unused_label_wells'] == 1060
    assert result['L_reference_new_wells'] == 0
    assert result['L_fully_new_fit_well_cost'] == pytest.approx(9.84)
    assert result['CORE_fully_new_fit_well_cost'] == pytest.approx(20.44)


def test_fit_adapter_is_exact_original_recipe_and_accepts_only_training_arrays(monkeypatch):
    seen = {}
    class Model:
        metadata = {}
    def fake_fit(y, **kwargs):
        seen.update(y=y.copy(), kwargs=kwargs)
        return Model()
    monkeypatch.setattr(runner, 'fit_baseline', fake_fit)
    y = np.ones((5, 4, 7))
    fitted = runner.fit_original_l(y, np.array(list('abcde')), 76)
    np.testing.assert_array_equal(seen['y'], y)
    assert seen['kwargs'] == dict(runner.L_FIT, random_state=76)
    assert fitted.metadata['train_ids'] == list('abcde')


def test_complete_block_aggregation_writes_paired_policies_resources_and_report(tmp_path, monkeypatch):
    model, y, raw, actual, stats = fixture()
    grams, _ = runner.sample_original_l(model, y[-2:, 0], samples=16, seed=18, draw_chunk=8)
    scored, gamma = runner.score_draws(grams, raw[-2:], actual[-2:], stats, np.square(y[-2:, 0]).mean(1))
    scored['profile_joint_nll'] = np.array([30., 31.])
    n = 80
    ids = np.asarray([f'ID{i:03}' for i in range(n)])
    actual = np.tile(np.array([-.05, .1]), n//2)
    folds = np.repeat(np.arange(5), 16)
    data = dict(ids=ids, groups=ids.copy(), layout=np.asarray([f'PLATE{i%2}' for i in range(n)]))
    core = {key: np.concatenate([value.copy() for _ in range(n//2)]) for key, value in scored.items()}
    core.update(ids=ids, actual=actual, fold=folds)
    parts = []
    root, report = tmp_path/'run', tmp_path/'report'
    root.mkdir(); report.mkdir()
    monkeypatch.setattr(runner, 'ROOT', root)
    monkeypatch.setattr(runner, 'REPORT', report)
    monkeypatch.setattr(runner, 'bootstrap_difference', lambda a, b, labels: {'mean': float(np.mean(a-b))})
    for f in range(5):
        q = np.flatnonzero(folds == f)
        part = {key: np.arange(3) for key in runner.ROLE_KEYS}
        part['DEV_EVAL'] = q
        parts.append(part)
        folder = root/f'fold_{f}'/'seed_0'
        folder.mkdir(parents=True)
        for start in range(0, len(q), 2):
            rows = q[start:start+2]
            path = folder/f'block_{start:04}.npz'
            runner.save_npz(path, ids=ids[rows], actual=actual[rows], gamma_samples=gamma, **scored)
            runner.write_json(path.with_suffix('.json'), {'wall_seconds': 1.})
    result = runner.analyze(data, core, {'actual': actual}, parts)
    assert result['complete'] and result['n'] == 80
    assert result['metrics'][runner.ARM]['policies']['lambda_0.2']['activated'] == 10
    assert len(result['paired_L_minus_CORE']) >= 12
    assert len(result['resources']) == 5
    assert result['common_nll_available'] is False
    assert (root/'REPORT.md').exists() and (report/'summary.json').exists()
    saved = runner.comparison.read_npz(root/(runner.ARM+'.npz'))
    assert saved['selected_lambda_0'].sum() == 10
    assert 'gamma_samples' not in saved


def test_matching_frozen_manifests_are_read_only_for_parallel_integrations(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path/'run')
    monkeypatch.setattr(runner, 'REPORT', tmp_path/'report')
    parts = [{role: np.arange(3) for role in runner.ROLE_KEYS} for _ in range(5)]
    source = {'parts': [{role: ['a', 'b', 'c'] for role in runner.ROLE_KEYS} for _ in range(5)]}
    first = runner.freeze_plan(parts, source)
    paths = (runner.ROOT/'run_manifest.json', runner.REPORT/'run_manifest.json')
    before = [path.stat().st_mtime_ns for path in paths]
    monkeypatch.setattr(runner, 'write_json', lambda *a, **k: pytest.fail('An identical manifest must not be rewritten'))
    assert runner.freeze_plan(parts, source) == first
    assert [path.stat().st_mtime_ns for path in paths] == before


def test_additional_seed_refuses_missing_fits_before_reading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load_inputs', lambda: pytest.fail('No measurement loading needed when a fit is missing'))
    with pytest.raises(ValueError, match='no refit is allowed'):
        runner.run(seed_offset=100000)
