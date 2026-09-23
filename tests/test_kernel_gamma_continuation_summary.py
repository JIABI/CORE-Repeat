"""Synthetic saved-prediction continuation tests; no model fitting or raw data."""
import json

import numpy as np
import pytest

from opal2.biology_kernel_evaluation import write_json
from opal2.kernel_gamma_continuation_summary import ARMS, CONTINUED_ARMS, COMPARISONS, _validate_manifest, summarize
from opal2.gram_oof_experiment import selection_mask


def _fixture(root):
    source=root.parent/'epoch30';source.mkdir()
    n=639;rng=np.random.default_rng(31)
    ids=np.asarray([f'SYNTHETIC_{i:04d}' for i in range(n)])
    actual=rng.normal(0,.07,(n,3));target=rng.normal(size=(n,9))
    records=[];allocation=np.empty(n,int)
    for fold,test in enumerate(np.array_split(np.arange(n),5)):
        other=np.setdiff1d(np.arange(n),test)
        records.append(dict(fold=fold,fit=other[100:].tolist(),inner_validation=other[:100].tolist(),test=test.tolist()))
        allocation[test]=fold
    manifest=dict(ids=ids.tolist(),folds=records,arms=list(ARMS),new_arms=list(CONTINUED_ARMS),
        source_run=str(source),start_epoch=30,config=dict(folds=5,seed=24,stage_epochs=60,samples=2000,bootstrap=2000),
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    write_json(root/'run_manifest.json',manifest)
    old_manifest=dict(manifest,config=dict(manifest['config'],stage_epochs=30))
    write_json(source/'run_manifest.json',old_manifest)
    for index,arm in enumerate(ARMS):
        old=dict(actual=actual,predicted=actual*.5,p_null=np.full((n,3),.5),utility_crps=np.full((n,3),.05),
                 geometry_energy=np.ones(n),actual_u=target,mean_u=target+1)
        now=dict(old,mean_u=target+(1 if index==0 else .5),utility_crps=old['utility_crps']-(0 if index==0 else .01))
        np.savez_compressed(source/f'{arm}_oof_predictions.npz',ids=ids,fold=allocation,
            principal_mask=selection_mask(old['predicted'][:,2],ids,allocation,.25,2),**old)
        for record in records:
            ix=np.asarray(record['test']);fold=record['fold']
            folder=root/'folds'/f'fold_{fold}'/'arms'/arm;evaluation=folder/'evaluation';evaluation.mkdir(parents=True)
            old_folder=source/'folds'/f'fold_{fold}'/'arms'/arm/'evaluation';old_folder.mkdir(parents=True)
            for destination,arrays in ((evaluation,now),(old_folder,old)):
                np.savez_compressed(destination/'u_predictions.npz',ids=ids[ix],actual_u=target[ix],mean_u=arrays['mean_u'][ix],
                    covariance_u=np.broadcast_to(np.eye(9),(len(ix),9,9)),scale_tril_u=np.broadcast_to(np.eye(9),(len(ix),9,9)))
            np.savez_compressed(evaluation/'predictions.npz',ids=ids[ix],
                **{k:now[k][ix] for k in ('actual','predicted','p_null','utility_crps','geometry_energy')})
            counts=dict(trainable=0 if index==0 else 3837,total=120998)
            write_json(evaluation/'metrics.json',dict(samples=2000,model=dict(actual_checkpoint_epoch='frozen' if index==0 else 60,parameter_counts=counts)))
            write_json(folder/'training_complete.json',dict(epoch=60,optimizer_steps=420,best_epoch=35,parameter_counts=counts))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,fit_u_mse=.7,validation_u_mse=.9))
                for e in range(0,61,5))+'\n')
            (folder/'gamma_monitoring.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,validation_gamma_crps=.05,
                validation_gamma_mse=.01)) for e in range(30,61,5))+'\n')
    return manifest,source


def test_fixed60_factorial_and_same_arm30_pairing_without_source_changes(tmp_path):
    root=tmp_path/'new';root.mkdir();manifest,source=_fixture(root)
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=summarize(root)
    assert result['actual_checkpoint_epoch']==60 and result['start_epoch']==30
    assert result['historical_A_exactly_matched'] and result['extension_chosen_after_dev_epoch30']
    assert list(result['comparisons'])==[a+'__minus__'+b for a,b in COMPARISONS]
    for arm in ARMS:
        policy=result['models'][arm]['principal_policy'];assert policy['selected_n']==79 and policy['used_wells']==158
        pair=result['epoch_extension_comparisons'][arm]
        assert pair['u_mse']['mean']==pytest.approx(0 if arm=='A_HR' else -.75)
        assert pair['gamma_crps']['mean']==pytest.approx(0 if arm=='A_HR' else -.01)
        assert pair['fdp']['interval95']==[0,0]
        with np.load(root/f'{arm}_oof_predictions.npz',allow_pickle=False) as saved:
            assert saved['ids'].tolist()==manifest['ids']
    history=result['models'][ARMS[1]]['folds'][0]['checkpoint']['continuation_trajectory']
    assert [r['epoch'] for r in history]==list(range(30,61,5))
    assert history[0]['validation_gamma_crps']==.05
    assert {str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}==before
    assert all((root/name).is_file() for name in ('summary.json','SUMMARY.md','REPORT.md'))


@pytest.mark.parametrize('mismatch',['epoch','steps','frozen_A','covariance'])
def test_false_continuation_or_changed_fixed_reference_rejected(tmp_path,mismatch):
    root=tmp_path/'new';root.mkdir();_,source=_fixture(root)
    if mismatch=='epoch':
        path=root/'folds/fold_0/arms'/ARMS[1]/'evaluation/metrics.json';value=json.loads(path.read_text())
        value['model']['actual_checkpoint_epoch']=30;write_json(path,value);match='actual epoch60'
    elif mismatch=='steps':
        path=root/'folds/fold_0/arms'/ARMS[1]/'training_complete.json';value=json.loads(path.read_text())
        value['optimizer_steps']=210;write_json(path,value);match='420 cumulative'
    else:
        path=(source/'A_HR_oof_predictions.npz' if mismatch=='frozen_A' else
              root/'folds/fold_0/arms'/ARMS[2]/'evaluation/u_predictions.npz')
        with np.load(path,allow_pickle=False) as saved:value={k:saved[k].copy() for k in saved.files}
        value['mean_u' if mismatch=='frozen_A' else 'covariance_u'].flat[0]+=1
        np.savez_compressed(path,**value);match='Frozen A_HR changed' if mismatch=='frozen_A' else 'preserve original uncertainty'
    with pytest.raises(ValueError,match=match):summarize(root)
    assert not (root/'summary.json').exists()


def test_manifest_requires_true30_to60_stage(tmp_path):
    root=tmp_path/'new';root.mkdir();manifest,_=_fixture(root)
    ids,folds=_validate_manifest(manifest);assert len(ids)==len(folds)==639
    manifest['start_epoch']=0
    with pytest.raises(ValueError,match='epoch30 to epoch60'):_validate_manifest(manifest)
