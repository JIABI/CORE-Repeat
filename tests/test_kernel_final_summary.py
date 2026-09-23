"""Focused output tests; synthetic fixtures never stand in for experiment arms."""
import json

import numpy as np
import pytest

from opal2.biology_kernel_evaluation import write_json
from opal2 import kernel_final_summary as summary


def _manifest(root):
    ids=np.asarray([f'SYNTHETIC_{i:04d}' for i in range(639)])
    records=[]
    for fold,test in enumerate(np.array_split(np.arange(639),5)):
        other=np.setdiff1d(np.arange(639),test);valid=other[:103];fit=other[103:]
        records.append(dict(fold=fold,fit=fit.tolist(),inner_validation=valid.tolist(),test=test.tolist()))
        scope=dict(fold=fold,originalfit_ids=ids[fit].tolist(),validation_ids=ids[valid].tolist(),test_ids=ids[test].tolist(),
            commonbranchfit_ids=ids[fit[64:]].tolist(),reference_ids=ids[fit[:64]].tolist(),
            anchor_ids_by_mode=dict(O=ids[fit[64:128]].tolist(),D=ids[fit[:64]].tolist()))
        folder=root/'folds'/f'fold_{fold}';folder.mkdir(parents=True)
        write_json(folder/'scope.json',scope)
    manifest=dict(ids=ids.tolist(),folds=records,arms=list(summary.ARMS),
        config=dict(folds=5,stage_epochs=30,samples=10000,bootstrap=2000,seed=37),
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    write_json(root/'run_manifest.json',manifest)
    return manifest,ids


def test_full_factorial_aggregate_budget_and_infeasibility_report(tmp_path,monkeypatch):
    manifest,ids=_manifest(tmp_path);rng=np.random.default_rng(54)
    actual=rng.normal(0,.08,(639,3));target=rng.normal(size=(639,9))
    def reader(root,manifest,ids,arm):
        index=summary.ARMS.index(arm)
        arrays=dict(actual=actual.copy(),predicted=actual*.5,p_null=(actual<=0).astype(float),
            utility_crps=np.full((639,3),.05-index*.001),geometry_energy=np.ones(639),
            actual_u=target.copy(),mean_u=target+1-index*.025)
        rows=[]
        for record in manifest['folds']:
            feasibility=dict(fit_satisfied=True,validation_satisfied=record['fold']!=0,
                fullfit_baseline_mse=1.,fullfit_final_mse=.9,validation_baseline_mse=1.,
                validation_final_mse=1.1 if record['fold']==0 else .9,numerical_tolerance=1e-10)
            rows.append(dict(fold=record['fold'],n=len(record['test']),checkpoint={},
                parameter_counts=dict(trainable=0 if arm=='A_HR' else 3837,total=120998),geometry_feasibility=feasibility))
        diagnostics=[(len(r['test']),dict(available=False),{}) for r in manifest['folds']]
        chunks={name:dict(predicted=arrays['predicted'].copy(),p_null=arrays['p_null'].copy()) for name in summary.CHUNKS}
        return arrays,rows,diagnostics,chunks,np.broadcast_to(np.eye(9),(639,9,9)).copy()
    monkeypatch.setattr(summary,'_read_arm',reader)
    monkeypatch.setattr(summary,'_audit_learning_scope',lambda root,scopes:[dict(fold=x['fold']) for x in scopes])
    result=summary.summarize(tmp_path)
    assert result['samples_per_object']==10000 and not result['default_model_declared']
    assert len(result['comparisons'])==20 and set(result['comparison_groups'])==set(summary.COMPARISON_GROUPS)
    assert len(result['factorial_interactions'])==4
    for entry in result['factorial_interactions'].values():
        assert entry['statistics']['gamma_crps']['mean']==pytest.approx(0,abs=1e-14)
    for arm in summary.ARMS:
        model=result['models'][arm];policy=model['principal_policy']
        assert policy['selected_n']==79 and policy['used_wells']==158
        assert policy['fdp']==policy['selected_null_count']/79
        if arm!='A_HR':
            assert model['feasibility']['known_satisfied_folds']==5
            assert model['feasibility']['validation_violated_folds']==1
            assert not model['feasibility']['all_validation_geometry_noninferior']
        assert model['sampling_stability']['comparisons']['first2000__vs__full10000']['overlap']['intersection_n']==79
        with np.load(tmp_path/f'{arm}_oof_predictions.npz',allow_pickle=False) as z:
            assert np.array_equal(z['ids'],ids) and z['actual_u'].shape==(639,9)
    assert all((tmp_path/p).is_file() for p in ('summary.json','SUMMARY.md','REPORT.md'))


def test_actual_reader_10000_draw_chunks_and_fixed_epoch(tmp_path):
    ids=np.asarray(['s0','s1','s2','s3']);manifest=dict(folds=[dict(fold=0,test=list(range(4)))])
    folder=tmp_path/'folds/fold_0/arms/O_G_W';evaluation=folder/'evaluation';evaluation.mkdir(parents=True)
    actual=np.array([[-.05]*3,[.02]*3,[.07]*3,[-.01]*3])
    samples=np.broadcast_to(actual,(10000,4,3)).copy();samples[:2000,0]+=.02
    means=samples.mean(0);probs=(samples<=0).mean(0)
    np.savez_compressed(evaluation/'predictions.npz',ids=ids,actual=actual,predicted=means,p_null=probs,
        utility_crps=np.full((4,3),.05),geometry_energy=np.ones(4),utility_samples=samples)
    np.savez_compressed(evaluation/'u_predictions.npz',ids=ids,actual_u=np.ones((4,9)),mean_u=np.zeros((4,9)),
        covariance_u=np.broadcast_to(np.eye(9),(4,9,9)))
    metric=dict(samples=10000,model=dict(actual_checkpoint_epoch=30))
    write_json(evaluation/'metrics.json',metric)
    write_json(folder/'training_complete.json',dict(epoch=30,optimizer_steps=180,parameter_counts=dict(trainable=3837,total=120998),
        geometry_feasibility=dict(fit_satisfied=False,validation_satisfied=True)))
    arrays,rows,_,chunks,_=summary._read_arm(tmp_path,manifest,ids,'O_G_W')
    assert np.array_equal(arrays['predicted'],means)
    assert chunks['first2000']['predicted'][0,0]==pytest.approx(-.03)
    assert chunks['second5000']['predicted'][0,0]==pytest.approx(-.05)
    assert summary._feasibility(rows)['known_violated_folds']==1
    metric['model']['actual_checkpoint_epoch']=25;write_json(evaluation/'metrics.json',metric)
    with pytest.raises(ValueError,match='fixed epoch30'):summary._read_arm(tmp_path,manifest,ids,'O_G_W')


def test_scope_requires_genuine_disjoint_reference_and_same_snapshot(tmp_path):
    manifest,ids=_manifest(tmp_path)
    scopes=summary._read_scopes(tmp_path,manifest,ids);assert len(scopes)==5
    path=tmp_path/'folds/fold_0/scope.json';scope=json.loads(path.read_text())
    scope['anchor_ids_by_mode']['D'][0]=scope['commonbranchfit_ids'][0];write_json(path,scope)
    with pytest.raises(ValueError,match='Disjoint anchors'):summary._read_scopes(tmp_path,manifest,ids)


def test_all_arms_and_full_draw_count_are_declared_before_readout(tmp_path):
    manifest,_=_manifest(tmp_path)
    ids,allocation=summary._validate_manifest(manifest);assert len(ids)==len(allocation)==639
    manifest['config']['samples']=2000
    with pytest.raises(ValueError,match='10000'):summary._validate_manifest(manifest)
    manifest['config']['samples']=10000;manifest['arms']=list(summary.ARMS[:-1])
    with pytest.raises(ValueError,match='nine declared'):summary._validate_manifest(manifest)


def test_saved_checkpoint_and_bank_training_membership_match_scope(tmp_path):
    import torch
    scope=dict(fold=0,commonbranchfit_ids=['fit0','fit1'],validation_ids=['val0'],
               anchor_ids_by_mode=dict(O=['fit0'],D=['ref0']))
    folder=tmp_path/'folds/fold_0';(folder/'banks').mkdir(parents=True)
    banks={}
    for mode in ('O','D'):
        bank=dict(fitting_ids=scope['commonbranchfit_ids'],descriptor_config=dict(
            fitting_ids=scope['commonbranchfit_ids'],anchor_data=dict(anchor_ids=scope['anchor_ids_by_mode'][mode])))
        banks[mode]=bank;torch.save(dict(config=bank),folder/'banks'/f'{mode}.pt')
    for arm in summary.ARMS[1:]:
        target=folder/'arms'/arm;target.mkdir(parents=True)
        saved=dict(epoch=30,optimizer_steps=180,fit_ids=scope['commonbranchfit_ids'],
            validation_ids=scope['validation_ids'],model_config=dict(bank_config=banks[arm[0]]))
        torch.save(saved,target/'epoch30.pt')
        write_json(target/'training_config.json',dict(fit_ids=scope['commonbranchfit_ids'],validation_ids=scope['validation_ids']))
    result=summary._audit_learning_scope(tmp_path,[scope])
    assert result[0]['all_checkpoint_fit_ids_match']
    path=folder/'arms/O_G_W/training_config.json'
    write_json(path,dict(fit_ids=['fit0','ref0'],validation_ids=scope['validation_ids']))
    with pytest.raises(ValueError,match='different fit_ids'):summary._audit_learning_scope(tmp_path,[scope])
