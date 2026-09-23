"""Synthetic-only tests of the generic-by-Gamma factorial saved-output summary."""
import json

import numpy as np
import pytest

from opal2.biology_kernel_evaluation import write_json
from opal2.generic_gamma_summary import ARMS, NEW_ARM, COMPARISONS, _validate_manifest, summarize
from opal2.gram_oof_experiment import selection_mask


def _fixture(root):
    gamma=root.parent/'gamma'; conditional=root.parent/'conditional'; stability=root.parent/'stability'
    for folder in (gamma,conditional,stability): folder.mkdir()
    n=639; rng=np.random.default_rng(36)
    ids=np.asarray([f'SYNTHETIC_{i:04d}' for i in range(n)])
    actual=rng.normal(0,.07,(n,3)); actual_u=rng.normal(size=(n,9))
    records=[]; allocation=np.empty(n,int)
    for fold,test in enumerate(np.array_split(np.arange(n),5)):
        other=np.setdiff1d(np.arange(n),test)
        records.append(dict(fold=fold,fit=other[100:].tolist(),inner_validation=other[:100].tolist(),test=test.tolist()))
        allocation[test]=fold
    manifest=dict(ids=ids.tolist(),folds=records,arms=list(ARMS),new_arms=[NEW_ARM],
        gamma_reference_run=str(gamma),conditional_reference_run=str(conditional),sampling_stability_run=str(stability),
        config=dict(folds=5,seed=17,stage_epochs=30,samples=2000,bootstrap=2000),
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    for folder in (root,gamma,conditional): write_json(folder/'run_manifest.json',manifest)
    for index,arm in enumerate(ARMS):
        gamma_change={3:.01,4:.02}.get(index,0)
        arrays=dict(actual=actual,predicted=actual*.5,p_null=np.full((n,3),.5),
            utility_crps=np.full((n,3),.05-gamma_change),geometry_energy=np.ones(n),
            actual_u=actual_u,mean_u=actual_u+(1-index*.1))
        source=conditional if index==1 else gamma
        if arm!=NEW_ARM:
            np.savez_compressed(source/f'{arm}_oof_predictions.npz',ids=ids,fold=allocation,
                principal_mask=selection_mask(arrays['predicted'][:,2],ids,allocation,.25,2),**arrays)
        for record in records:
            ix=np.asarray(record['test']); fold=record['fold']
            folder=root/'folds'/f'fold_{fold}'/'arms'/arm
            evaluation=folder/'evaluation';evaluation.mkdir(parents=True)
            np.savez_compressed(evaluation/'predictions.npz',ids=ids[ix],
                **{k:arrays[k][ix] for k in ('actual','predicted','p_null','utility_crps','geometry_energy')})
            coordinates=dict(ids=ids[ix],actual_u=actual_u[ix],mean_u=arrays['mean_u'][ix],
                covariance_u=np.broadcast_to(np.eye(9),(len(ix),9,9)),scale_tril_u=np.broadcast_to(np.eye(9),(len(ix),9,9)))
            np.savez_compressed(evaluation/'u_predictions.npz',**coordinates)
            if arm!=NEW_ARM:
                old=source/'folds'/f'fold_{fold}'/'arms'/arm/'evaluation';old.mkdir(parents=True)
                np.savez_compressed(old/'u_predictions.npz',**coordinates)
            counts=dict(trainable=0 if index==0 else 3837,total=120998)
            write_json(evaluation/'metrics.json',dict(samples=2000,
                model=dict(actual_checkpoint_epoch='frozen' if index==0 else 30,parameter_counts=counts)))
            write_json(folder/'training_complete.json',dict(epoch=30,optimizer_steps=210,parameter_counts=counts))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,fit_u_mse=.8,validation_u_mse=.9))
                                               for e in (0,5,10,15,20,25,30))+'\n')
    return manifest,gamma,conditional


def test_five_arms_same_budget_original_risk_and_paired_interaction(tmp_path):
    root=tmp_path/'new';root.mkdir()
    manifest,gamma,conditional=_fixture(root)
    before={str(p):p.read_bytes() for s in (gamma,conditional) for p in s.rglob('*') if p.is_file()}
    result=summarize(root)
    assert list(result['models'])==list(ARMS) and result['new_arms']==[NEW_ARM]
    assert len(result['historical_reproduction'])==4
    assert list(result['comparisons'])==[a+'__minus__'+b for a,b in COMPARISONS]
    for arm in ARMS:
        p=result['models'][arm]['principal_policy']
        assert p['selected_n']==79 and p['used_wells']==158
        assert p['fdp']==p['selected_null_count']/79
        assert p['fpr']==p['selected_null_count']/p['population_null_count']
        with np.load(root/f'{arm}_oof_predictions.npz',allow_pickle=False) as saved:
            assert saved['ids'].tolist()==manifest['ids']
    stat=result['score_interaction']['statistics']['gamma_crps']
    assert stat['mean']==pytest.approx(.01)
    assert stat['interval95']==pytest.approx([.01,.01])
    assert not result['score_interaction']['policy_interaction_evaluated']
    assert result['models'][NEW_ARM]['training_trajectory'][0]['records'][-1]['epoch']==30
    assert {str(p):p.read_bytes() for s in (gamma,conditional) for p in s.rglob('*') if p.is_file()}==before
    assert all((root/name).is_file() for name in ('summary.json','REPORT.md','SUMMARY.md'))


@pytest.mark.parametrize('mismatch',['reference_prediction','partition_snapshot','parameter_count'])
def test_changed_saved_evidence_is_rejected(tmp_path,mismatch):
    root=tmp_path/'new';root.mkdir()
    _,gamma,conditional=_fixture(root)
    if mismatch=='reference_prediction':
        path=conditional/'F_CONDITIONAL_GENERIC_oof_predictions.npz'
        with np.load(path,allow_pickle=False) as saved:a={k:saved[k].copy() for k in saved.files}
        a['mean_u'].flat[0]+=1;np.savez_compressed(path,**a);match='exactly reproduce'
    elif mismatch=='partition_snapshot':
        path=gamma/'run_manifest.json';a=json.loads(path.read_text())
        a['folds'][0]['fit'][0],a['folds'][0]['inner_validation'][0]=a['folds'][0]['inner_validation'][0],a['folds'][0]['fit'][0]
        write_json(path,a);match='frozen partition'
    else:
        path=root/'folds/fold_0/arms'/NEW_ARM/'training_complete.json';a=json.loads(path.read_text())
        a['parameter_counts']['trainable']=3838;write_json(path,a);match='3837'
    with pytest.raises(ValueError,match=match):summarize(root)
    assert not (root/'summary.json').exists()


def test_manifest_stage_and_new_arm_contract(tmp_path):
    root=tmp_path/'new';root.mkdir()
    manifest,_,_=_fixture(root)
    ids,folds=_validate_manifest(manifest);assert len(ids)==len(folds)==639
    manifest['new_arms']=[ARMS[3],NEW_ARM]
    with pytest.raises(ValueError,match='only M'):_validate_manifest(manifest)
    manifest['new_arms']=[NEW_ARM];manifest['config']['stage_epochs']=10
    with pytest.raises(ValueError,match='epoch30'):_validate_manifest(manifest)
