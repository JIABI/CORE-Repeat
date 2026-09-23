"""Synthetic-only tests for the fixed-epoch Gamma supervision comparison."""
import json

import numpy as np
import pytest

from opal2.biology_kernel_evaluation import write_json
from opal2.gamma_supervised_summary import ARMS, COMPARISONS, _validate_manifest, summarize
from opal2.gram_oof_experiment import selection_mask


def _fixture(root):
    source = root.parent/'history'; source.mkdir()
    n=639; rng=np.random.default_rng(32)
    ids=np.asarray([f'SYNTHETIC_{i:04d}' for i in range(n)])
    actual=rng.normal(0,.07,(n,3)); actual_u=rng.normal(size=(n,9))
    records=[]; allocation=np.empty(n,int)
    for fold,test in enumerate(np.array_split(np.arange(n),5)):
        other=np.setdiff1d(np.arange(n),test)
        records.append(dict(fold=fold,fit=other[100:].tolist(),inner_validation=other[:100].tolist(),test=test.tolist()))
        allocation[test]=fold
    manifest=dict(ids=ids.tolist(),folds=records,arms=list(ARMS),historical_reference_run=str(source),
        config=dict(folds=5,seed=19,stage_epochs=30,samples=2000,bootstrap=2000),
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    write_json(root/'run_manifest.json',manifest)
    for index,arm in enumerate(ARMS):
        arrays=dict(actual=actual,predicted=.5*actual,p_null=np.full((n,3),.5),
            utility_crps=np.full((n,3),.05-(.01 if index==2 else 0)),geometry_energy=np.ones(n),
            actual_u=actual_u,mean_u=actual_u+(0.5 if index==2 else 1))
        old_arm='A_HR' if index==0 else 'G_CONDITIONAL_STRUCTURED'
        if index<2:
            np.savez_compressed(source/f'{old_arm}_oof_predictions.npz',ids=ids,fold=allocation,
                principal_mask=selection_mask(arrays['predicted'][:,2],ids,allocation,.25,2),**arrays)
        for record in records:
            ix=np.asarray(record['test']); fold=record['fold']
            folder=root/'folds'/f'fold_{fold}'/'arms'/arm
            evaluation=folder/'evaluation'; evaluation.mkdir(parents=True)
            np.savez_compressed(evaluation/'predictions.npz',ids=ids[ix],
                **{k:arrays[k][ix] for k in ('actual','predicted','p_null','utility_crps','geometry_energy')})
            coordinates=dict(ids=ids[ix],actual_u=actual_u[ix],mean_u=arrays['mean_u'][ix],
                covariance_u=np.broadcast_to(np.eye(9),(len(ix),9,9)),scale_tril_u=np.broadcast_to(np.eye(9),(len(ix),9,9)))
            np.savez_compressed(evaluation/'u_predictions.npz',**coordinates)
            if index<2:
                old_folder=source/'folds'/f'fold_{fold}'/'arms'/old_arm/'evaluation'
                old_folder.mkdir(parents=True)
                np.savez_compressed(old_folder/'u_predictions.npz',**coordinates)
            counts=dict(trainable=0 if index==0 else 100,total=200)
            write_json(evaluation/'metrics.json',dict(samples=2000,
                model=dict(actual_checkpoint_epoch='frozen' if index==0 else 30,parameter_counts=counts)))
            write_json(folder/'training_complete.json',dict(epoch=30,actual_checkpoint_epoch=30,optimizer_steps=210,
                best_epoch=5,parameter_counts=counts))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,fit_u_mse=.9,
                validation_u_mse=1.,
                train_gamma_crps_loss=.4,gradient_norm_mean=2.)) for e in (0,5,10,15,20,25,30))+'\n')
            (folder/'gamma_monitoring.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,
                validation_gamma_crps=.05,validation_gamma_mse=.01,gradient_cosine=-.2))
                for e in (0,5,10,15,20,25,30))+'\n')
    return manifest,source


def test_three_arms_exact_reference_same_budget_and_paired_results(tmp_path):
    root=tmp_path/'new';root.mkdir()
    manifest,source=_fixture(root)
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=summarize(root)
    assert result['n']==639 and result['actual_checkpoint_epoch']==30
    assert list(result['models'])==list(ARMS)
    assert list(result['comparisons'])==[a+'__minus__'+b for a,b in COMPARISONS]
    for arm in ARMS:
        assert result['models'][arm]['principal_policy']['selected_n']==79
        assert result['models'][arm]['principal_policy']['used_wells']==158
        with np.load(root/f'{arm}_oof_predictions.npz',allow_pickle=False) as saved:
            assert saved['ids'].tolist()==manifest['ids']
    assert result['historical_reproduction']['J_GEOMETRY_CONTROL']['exact_core_arrays']
    assert result['historical_reproduction']['J_GEOMETRY_CONTROL']['exact_uncertainty_arrays_per_fold'][0]['uncertainty_arrays']==['covariance_u','scale_tril_u']
    pair=result['comparisons']['K_GEOMETRY_GAMMA_CRPS__minus__J_GEOMETRY_CONTROL']
    assert pair['u_mse']['mean']==pytest.approx(-.75)
    assert pair['gamma_crps']['mean']==pytest.approx(-.01)
    assert pair['fdp']['interval95']==[0,0]
    assert pair['overlap']['intersection_n']==79
    trajectory=result['models']['K_GEOMETRY_GAMMA_CRPS']['training_trajectory'][0]
    assert [r['epoch'] for r in trajectory['records']]==[0,5,10,15,20,25,30]
    assert trajectory['records'][0]['gradient_norm_mean']==2
    assert trajectory['records'][0]['validation_gamma_crps']==.05
    assert trajectory['records'][0]['gradient_cosine']==-.2
    assert 'validation_gamma_crps' not in trajectory['original_history'][0]
    assert {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}==before
    assert all((root/name).is_file() for name in ('summary.json','REPORT.md','SUMMARY.md'))


@pytest.mark.parametrize('mismatch',['reference_prediction','reference_covariance','steps','epoch'])
def test_reference_or_fixed_training_stage_mismatch_stops_summary(tmp_path,mismatch):
    root=tmp_path/'new';root.mkdir()
    _,source=_fixture(root)
    if mismatch=='reference_prediction':
        path=source/'G_CONDITIONAL_STRUCTURED_oof_predictions.npz'
        with np.load(path,allow_pickle=False) as saved:
            a={k:saved[k].copy() for k in saved.files}
        a['mean_u'].flat[0]+=1;np.savez_compressed(path,**a);match='exactly reproduce'
    elif mismatch=='reference_covariance':
        path=source/'folds/fold_0/arms/G_CONDITIONAL_STRUCTURED/evaluation/u_predictions.npz'
        with np.load(path,allow_pickle=False) as saved:
            a={k:saved[k].copy() for k in saved.files}
        a['covariance_u'].flat[0]+=1;np.savez_compressed(path,**a);match='reference covariance differs'
    elif mismatch=='steps':
        path=root/'folds/fold_0/arms/J_GEOMETRY_CONTROL/training_complete.json'
        a=json.loads(path.read_text());a['optimizer_steps']=200;write_json(path,a);match='210 optimizer steps'
    else:
        path=root/'folds/fold_0/arms/K_GEOMETRY_GAMMA_CRPS/evaluation/metrics.json'
        a=json.loads(path.read_text());a['model']['actual_checkpoint_epoch']=5;write_json(path,a);match='actual epoch30'
    with pytest.raises(ValueError,match=match):
        summarize(root)
    assert not (root/'summary.json').exists()


def test_manifest_schema_validates_before_predictions_exist(tmp_path):
    root=tmp_path/'new';root.mkdir()
    manifest,_=_fixture(root)
    ids,folds=_validate_manifest(manifest)
    assert len(ids)==len(folds)==639
    manifest['config']['stage_epochs']=10
    with pytest.raises(ValueError,match='epoch30'):
        _validate_manifest(manifest)
