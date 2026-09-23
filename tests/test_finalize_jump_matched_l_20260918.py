"""Finisher scheduling and saved-array MC summary tests; no real jobs launched."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('l_finisher', PROJECT/'scripts/finalize_jump_matched_l_20260918.py')
finisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finisher)


def complete(root, offset):
    suffix = '' if offset == 0 else f'_mc{offset}'
    finisher.atomic_json(root/f'status_seed_{offset}.json', {'state': 'COMPLETE'})
    finisher.atomic_json(root/('summary'+suffix+'.json'), {'complete': True, 'samples': 100000, 'MC_and_scoring_wall_seconds': 12.+offset/100000})


def fits(root):
    for fold in range(5):
        path = root/f'fold_{fold}'/'complete_original_L.npz'
        path.parent.mkdir(parents=True)
        path.touch()


def test_finisher_can_never_launch_primary_and_requires_existing_fits(tmp_path):
    with pytest.raises(ValueError, match='only the two additional'):
        finisher.launch_seed(PROJECT, tmp_path, 0)
    with pytest.raises(RuntimeError, match='no refitting'):
        finisher.launch_seed(PROJECT, tmp_path, 100000)


def test_spawn_uses_one_thread_and_explicit_no_refitting(tmp_path, monkeypatch):
    fits(tmp_path)
    captured = {}
    def spawn(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(finisher.subprocess, 'Popen', spawn)
    _, record = finisher.launch_seed(PROJECT, tmp_path, 200000)
    assert captured['command'][-3:] == ['--seed-offset', '200000', '--reuse-fitted-only']
    assert captured['kwargs']['env']['OPENBLAS_NUM_THREADS'] == '1'
    assert captured['kwargs']['start_new_session'] is True
    assert record['additional_training_runs'] == record['additional_label_wells'] == 0


def test_orchestration_refreshes_primary_then_launches_both_then_summarizes_and_refreshes(tmp_path, monkeypatch):
    fits(tmp_path); complete(tmp_path, 0)
    calls = []
    def refresh(project, root, stage):
        calls.append(stage)
        return {'returncode': 0}
    def launch(project, root, offset):
        calls.append(offset)
        complete(root, offset)
        return SimpleNamespace(pid=offset, poll=lambda: 0), {'pid': offset}
    def aggregate(root, report):
        calls.append('aggregate')
        return {'compute': {'additional_MC_only_seconds': 30.}}
    monkeypatch.setattr(finisher, 'refresh', refresh)
    monkeypatch.setattr(finisher, 'launch_seed', launch)
    monkeypatch.setattr(finisher, 'build_sensitivity', aggregate)
    finisher.run(PROJECT, tmp_path, tmp_path/'reports', poll_seconds=1)
    assert calls == ['primary_complete', 100000, 200000, 'aggregate', 'all_seeds_complete']
    assert finisher.read_json(tmp_path/'finalizer_status.json')['state'] == 'COMPLETE'


def test_saved_array_mc_summary_preserves_main_seed_and_separates_cost(tmp_path):
    ids = np.asarray([f'ID{i:03}' for i in range(80)])
    fold = np.repeat(np.arange(5), 16)
    actual = np.where(np.arange(80)%3 == 0, -.1, .1)
    for offset in (0, 100000, 200000):
        predicted = (np.arange(80)%16).astype(float)/100
        if offset:
            predicted[np.arange(80)%16 == 13] += offset/1e6
        p_null = np.full(80, .3)
        selected = np.zeros(80, bool)
        for f in range(5):
            q = np.flatnonzero(fold == f)
            order = np.lexsort((ids[q], -predicted[q]))
            selected[q[order[:2]]] = True
        suffix = '' if offset == 0 else f'_mc{offset}'
        np.savez_compressed(tmp_path/(finisher.ARM+suffix+'.npz'), ids=ids, fold=fold,
            actual=actual, predicted=predicted, p_null=p_null,
            selected_lambda_0=selected, **{'selected_lambda_0.2': selected})
        complete(tmp_path, offset)
    payload = finisher.build_sensitivity(tmp_path, tmp_path/'report')
    assert payload['complete'] and len(payload['per_seed']) == 3
    assert payload['additional_training_runs'] == payload['additional_label_wells'] == 0
    assert payload['compute']['additional_deployment_cost'] == 0
    assert payload['compute']['additional_MC_only_seconds'] == 27.
    assert payload['per_seed'][0]['policies']['lambda_0']['list_symmetric_difference_from_primary'] == 0
    assert payload['per_seed'][1]['policies']['lambda_0']['list_symmetric_difference_from_primary'] == 10
    assert payload['policy_ranges']['lambda_0']['all_seed_intersection'] == 5
    assert (tmp_path/'MC_SENSITIVITY.md').exists()
