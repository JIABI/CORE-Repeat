"""Synthetic saved-output tests; no experimental data or model fitting."""
import json

import numpy as np
import pytest

from opal2.biology_kernel_evaluation import write_json
from opal2.conditional_response_summary import ARMS, HISTORICAL_ARMS, COMPARISONS, _validate_manifest, summarize
from opal2.gram_oof_experiment import selection_mask


def _fixture(root):
    source=root.parent/'history'
    source.mkdir()
    n=639; rng=np.random.default_rng(18)
    ids=np.asarray([f'SYNTHETIC_{i:04d}' for i in range(n)])
    actual=rng.normal(0,.07,(n,3)); target=rng.normal(size=(n,9))
    records=[]; allocation=np.empty(n,int)
    for fold,test in enumerate(np.array_split(np.arange(n),5)):
        other=np.setdiff1d(np.arange(n),test)
        records.append(dict(fold=fold,fit=other[100:].tolist(),inner_validation=other[:100].tolist(),test=test.tolist()))
        allocation[test]=fold
    manifest=dict(ids=ids.tolist(),folds=records,arms=list(ARMS),historical_reference_run=str(source),
        config=dict(folds=5,seed=17,bootstrap=2000,samples=2000,stage_epochs=30),
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,original_contract_changed=False)
    write_json(root/'run_manifest.json',manifest)
    old=dict(actual=actual,predicted=actual*.5,p_null=np.full((n,3),.5),utility_crps=np.full((n,3),.05),
             geometry_energy=np.ones(n),actual_u=target,mean_u=target+1)
    for arm in ('A_HR',*HISTORICAL_ARMS):
        np.savez_compressed(source/f'{arm}_oof_predictions.npz',ids=ids,fold=allocation,
            principal_mask=selection_mask(old['predicted'][:,2],ids,allocation,.25,2),**old)
    for index,arm in enumerate(ARMS):
        current=dict(old)
        improve=index*.1
        current['mean_u']=target+1-improve
        current['utility_crps']=old['utility_crps']-improve*.01
        for record in records:
            ix=np.asarray(record['test']); fold=record['fold']
            folder=root/'folds'/f'fold_{fold}'/'arms'/arm
            evaluation=folder/'evaluation';evaluation.mkdir(parents=True)
            counts=dict(trainable=0 if index==0 else 99,total=199)
            np.savez_compressed(evaluation/'predictions.npz',ids=ids[ix],
                **{k:current[k][ix] for k in ('actual','predicted','p_null','utility_crps','geometry_energy')})
            np.savez_compressed(evaluation/'u_predictions.npz',ids=ids[ix],actual_u=target[ix],mean_u=current['mean_u'][ix])
            write_json(evaluation/'metrics.json',dict(samples=2000,
                model=dict(actual_checkpoint_epoch='frozen' if index==0 else 30,parameter_counts=counts)))
            write_json(folder/'training_complete.json',dict(best_epoch=20,final_epoch=30,parameter_counts=counts))
            (folder/'history.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,validation_mse=1/(e+1)))
                for e in (0,5,10,15,20,25,30))+'\n')
            if index:
                # Foldwise constant gates: pooled variation must not be called conditioning.
                gate=np.full((len(ix),3,3),.5+.01*fold)
                block=np.full((len(ix),3,9),.02)
                raw=block.sum(1)
                np.savez_compressed(folder/'model_diagnostics.npz',ids=ids[ix],gate=gate,
                    block_contributions=block,raw=raw,kernel_raw=raw,output_bias=np.zeros_like(raw),
                    block_names=np.array(['chemical','morphology','scalar']),local_coefficients=np.full((7,3),1/3),
                    local_activation=np.ones((len(ix),7)),local_basis=np.ones((len(ix),7,3)))
    return manifest,source


def test_full_summary_historical_identity_unique_oof_budget_and_diagnostics(tmp_path):
    root=tmp_path/'new';root.mkdir()
    manifest,source=_fixture(root)
    before={p.name:p.read_bytes() for p in source.iterdir()}
    result=summarize(root)
    assert result['historical_A_exactly_matched']
    assert result['actual_checkpoint_epoch']==30 and result['n']==639
    assert list(result['models'])==list(ARMS)+list(HISTORICAL_ARMS)
    assert list(result['comparisons'])==[a+'__minus__'+b for a,b in COMPARISONS]
    for arm in ARMS:
        assert result['models'][arm]['principal_policy']['selected_n']==79
        assert result['models'][arm]['principal_policy']['used_wells']==158
        with np.load(root/f'{arm}_oof_predictions.npz',allow_pickle=False) as saved:
            assert saved['ids'].tolist()==manifest['ids']
            assert saved['mean_u'].shape==(639,9)
    comparison=result['comparisons']['G_CONDITIONAL_STRUCTURED__minus__A_HR']
    assert comparison['u_mse']['mean']==pytest.approx(-.51)
    assert comparison['gamma_crps']['mean']==pytest.approx(-.003)
    assert comparison['overlap']['intersection_n']==79
    assert comparison['fdp']['interval95']==[0,0]
    d=result['models']['G_CONDITIONAL_STRUCTURED']['model_diagnostics']
    assert max(d['gate']['within_fold_std_rms'])<1e-12
    assert min(d['gate']['pooled_std'])>.01
    assert d['sum_closure']['max_absolute_error']==0
    assert d['local_coefficients']['per_fold'][0]['shape']==[7,3]
    assert d['local_coefficients']['per_fold'][0]['rms_change_from_one_third']==0
    assert {p.name:p.read_bytes() for p in source.iterdir()}==before
    assert not (root/'B_MLP_oof_predictions.npz').exists()
    assert all((root/name).is_file() for name in ('summary.json','SUMMARY.md','REPORT.md'))


@pytest.mark.parametrize('mismatch',['epoch','historical_A','historical_target','historical_fold'])
def test_changed_checkpoint_or_historical_evidence_rejected(tmp_path,mismatch):
    root=tmp_path/'new';root.mkdir()
    _,source=_fixture(root)
    if mismatch=='epoch':
        path=root/'folds/fold_0/arms/G_CONDITIONAL_STRUCTURED/evaluation/metrics.json'
        value=json.loads(path.read_text());value['model']['actual_checkpoint_epoch']=20
        write_json(path,value);match='actual epoch30'
    else:
        arm='A_HR' if mismatch=='historical_A' else 'B_MLP'
        path=source/f'{arm}_oof_predictions.npz'
        with np.load(path,allow_pickle=False) as saved:
            value={k:saved[k].copy() for k in saved.files}
        key={'historical_A':'mean_u','historical_target':'actual_u','historical_fold':'fold'}[mismatch]
        value[key].flat[0]+=1;np.savez_compressed(path,**value)
        match={'historical_A':'Frozen A_HR','historical_target':'different original Gamma','historical_fold':'IDs/folds'}[mismatch]
    with pytest.raises(ValueError,match=match):
        summarize(root)
    assert not (root/'summary.json').exists()


def test_manifest_validation_does_not_require_finished_artifacts(tmp_path):
    root=tmp_path/'new';root.mkdir()
    manifest,_=_fixture(root)
    ids,fold=_validate_manifest(manifest)
    assert len(ids)==len(fold)==639
    manifest['config']['stage_epochs']=10
    with pytest.raises(ValueError,match='epoch30'):
        _validate_manifest(manifest)
