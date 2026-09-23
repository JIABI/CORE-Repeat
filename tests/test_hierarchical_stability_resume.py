"""Synthetic artifact inventories exercise preparation without any data reads."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opal2 import hierarchical_stability_experiment as experiment


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    project, reference = tmp_path/'project', tmp_path/'v1'
    for folder in ('opal2','tests'):
        (project/folder).mkdir(parents=True)
        (project/folder/'synthetic.py').write_text('# Synthetic snapshot fixture\n')
    (project/'pyproject.toml').write_text('[project]\nname="synthetic"\n')
    (project/'protocols/historical').mkdir(parents=True)
    (project/'protocols/historical/NUMERICAL_REPAIR_PLAN_20260914.md').write_text('Synthetic numerical plan.\n')
    monkeypatch.setattr(experiment, 'PROJECT', project)
    monkeypatch.setattr(experiment.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout=''))
    cfg = deepcopy(experiment.CONFIG)
    ids = [f'synthetic_{i}' for i in range(15)]
    repetitions = [dict(repeat=r, seed=10+r, folds=[dict(fold=f, test=list(range(f*3,f*3+3)))
                                                   for f in range(5)]) for r in range(3)]
    manifest = dict(config=cfg, arms=list(experiment.ARMS), ids=ids, repetitions=repetitions,
                    source_snapshot=str(reference/'source_snapshot'),
                    **{flag:False for flag in experiment.SCOPE_FLAGS})
    _write(reference/'run_manifest.json',manifest)
    _write(reference/'status.json',dict(state='FAILED',error_type='ValueError',
        error='The direct H=L Lᵀ is not numerically SPD; no underflow repair or jitter is permitted'))
    _write(reference/'launch.json',dict(pid=12345))
    (reference/'PROTOCOL.md').write_text('Synthetic original protocol.\n')
    for rep in repetitions:
        rep_path = reference/'repetitions'/f"repeat_{rep['repeat']}"
        _write(rep_path/'run_manifest.json', dict(ids=ids, config=cfg, folds=rep['folds'],
            source_snapshot=manifest['source_snapshot'], **{flag:False for flag in experiment.SCOPE_FLAGS}))
        for fold in rep['folds']:
            position = rep['repeat']*5+fold['fold']
            dest = rep_path/'folds'/f"fold_{fold['fold']}"
            if position < 10:
                _write(dest/'baseline_fits_complete.json',dict(synthetic=True))
                for arm in experiment.ARMS[:3]:
                    fit = dest/'arms'/arm/'fit.npz'
                    fit.parent.mkdir(parents=True,exist_ok=True)
                    fit.write_bytes(b'not a real fitted model')
            for arm in experiment.ARMS:
                if position < 9 or (position == 9 and arm == 'GLOBAL_GEOMETRY'):
                    score = dest/'arms'/arm/'test'
                    _write(score/'metrics.json',dict(synthetic=True))
                    (score/'predictions.npz').write_bytes(b'synthetic prediction bytes')
                    (score/'u_predictions.npz').write_bytes(b'synthetic coordinate bytes')
                if position < 9 and arm.startswith('HR_'):
                    _write(dest/'arms'/arm/'training_complete.json',dict(synthetic=True))
            if position < 9:
                _write(dest/'complete.json',dict(repeat=rep['repeat'],fold=fold['fold'],test_n=3))
    # The failed arm has only its pre-decode coordinate diagnostic.
    _write(reference/'repetitions/repeat_1/folds/fold_4/arms/RIDGE_TRAINCV/test/u_diagnostics.json',
           dict(synthetic=True))
    return reference, tmp_path/'v2'


def test_prepare_repair_copies_without_links_or_old_status_mutation(tmp_path,monkeypatch):
    reference, output = fixture(tmp_path,monkeypatch)
    old_files={str(p.relative_to(reference)):p.read_bytes() for p in reference.rglob('*') if p.is_file()}
    manifest=experiment.prepare_repair(output,reference)
    assert manifest['config']==experiment.CONFIG
    assert manifest['continuation']['copied_inventory']==dict(completed_folds=9,baseline_fit_groups=10,
        completed_HR_fits=27,completed_arm_scores=55)
    assert manifest['source_snapshot']==str(output/'source_snapshot')
    assert not (output/'launch.json').exists() and not (output/'summary.json').exists()
    assert json.loads((output/'status.json').read_text())['state']=='PREPARED_NUMERICAL_CONTINUATION'
    for relative,value in old_files.items():
        assert (reference/relative).read_bytes()==value
        if relative.startswith('repetitions/') and not relative.endswith('run_manifest.json'):
            copied=output/relative
            assert copied.read_bytes()==value
            assert copied.stat().st_ino!=(reference/relative).stat().st_ino
    for rep in range(3):
        saved=json.loads((output/'repetitions'/f'repeat_{rep}'/'run_manifest.json').read_text())
        assert saved['source_snapshot']==str(output/'source_snapshot')
        assert saved['config']==experiment.CONFIG
    with pytest.raises(FileExistsError):
        experiment.prepare_repair(output,reference)


def test_live_reference_worker_blocks_preparation(tmp_path,monkeypatch):
    reference,output=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(experiment.subprocess,'run',lambda *a,**k:SimpleNamespace(
        returncode=0,stdout=f'python -m opal2.hierarchical_stability_experiment execute --output {reference}'))
    with pytest.raises(RuntimeError,match='still present'):
        experiment.prepare_repair(output,reference)
    assert not output.exists()


def test_unreviewed_failure_or_missing_complete_artifact_blocks(tmp_path,monkeypatch):
    reference,output=fixture(tmp_path,monkeypatch)
    path=reference/'status.json'
    original=path.read_text()
    _write(path,dict(state='FAILED',error_type='ValueError',error='some different failure'))
    with pytest.raises(ValueError,match='only the recorded'):
        experiment.prepare_repair(output,reference)
    assert not output.exists()
    path.write_text(original)
    (reference/'repetitions/repeat_0/folds/fold_0/arms/HR_VALID_S1/test/predictions.npz').unlink()
    with pytest.raises(ValueError,match='missing an artifact'):
        experiment.prepare_repair(output,reference)
    assert not output.exists()
