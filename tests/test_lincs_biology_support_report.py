"""Synthetic tests of retrospective support/integrity auditing, not experiments."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from opal2 import lincs_biology_support_report as report


def test_no_report_until_all_folds_and_actual_initial_final_checkpoints_exist(tmp_path):
    manifest=dict(arms=['HR',*report.ARMS],fixed_epochs=30,config={'folds':5},
        ids=[f'i{i}' for i in range(10)],folds=[dict(fold=f,test=[2*f,2*f+1]) for f in range(5)])
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='incomplete'):
        report.summarize(tmp_path)
    assert not (tmp_path/'BIOLOGY_SUPPORT_AND_INTEGRITY.json').exists()
    for record in manifest['folds']:
        f=tmp_path/'folds'/f"fold_{record['fold']}"
        f.mkdir(parents=True)
        (f/'complete.json').write_text(json.dumps(dict(fold=record['fold'],arms=['HR',*report.ARMS],branch_epochs=30)))
        for name in ('scope.json','biological_support.json','bank.pt','gamma_objective_state.pt','HR_fit/best.pt'):
            path=f/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
        for arm in report.ARMS:
            for name in ('training_complete.json','training_config.json','epoch0.pt','epoch30.pt','model_diagnostics.npz'):
                path=f/'arms'/arm/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
    ids,allocation=report.require_complete(tmp_path,manifest)
    assert len(ids)==10 and np.array_equal(allocation,np.repeat(np.arange(5),2))
    (tmp_path/'folds/fold_4/arms/D_BIO_STRUCTURED/epoch30.pt').unlink()
    with pytest.raises(RuntimeError,match='incomplete'):
        report.require_complete(tmp_path,manifest)


def test_capacity_and_frozen_state_audit_reject_actual_changes():
    state={'local_coefficients':torch.zeros(131,3),
        'conditioner.0.weight':torch.zeros(16,131),'conditioner.0.bias':torch.zeros(16),
        'conditioner.2.weight':torch.zeros(9,16),'conditioner.2.bias':torch.zeros(9),
        'output.weight':torch.zeros(9,131),'descriptor_block':torch.zeros(131,dtype=torch.long),
        'bank.buffer':torch.tensor([1.]),'base_hr.fixed':torch.tensor([2.])}
    active=report._active_parameters(state)
    assert len(active)==6 and sum(v.numel() for v in active.values())==3837
    report._frozen_equal(state,{'buffer':torch.tensor([1.])},'bank.')
    report._frozen_equal(state,{'fixed':torch.tensor([2.])},'base_hr.')
    with pytest.raises(ValueError,match='Frozen'):
        report._frozen_equal(state,{'buffer':torch.tensor([1.01])},'bank.')
    wrong=dict(state,unexpected_parameter=torch.zeros(1))
    with pytest.raises(ValueError,match='Unexpected'):
        report._active_parameters(wrong)
    with pytest.raises(ValueError,match='mapping'):
        report._tensor_mapping_equal({'covariance':torch.eye(2)},{'covariance':torch.eye(2)*2},'covariance')


def test_readout_rms_pools_coordinates_not_equal_weight_fold_rms():
    first=dict(biological_readout=np.ones((1,9)),old_information_readout=np.ones((1,9))*2,
        kernel_raw=np.ones((1,9))*3,increment=np.ones((1,9))*.2)
    second=dict(biological_readout=np.ones((3,9))*3,old_information_readout=np.ones((3,9))*4,
        kernel_raw=np.ones((3,9))*7,increment=np.ones((3,9))*.4)
    result=report.activation_statistics([first,second])
    assert result['objects']==4
    assert result['biological_readout']['rms']==pytest.approx(np.sqrt(7.))
    assert result['biological_readout']['rms']!=2.
    assert result['biological_readout']['nonzero_object_count']==4
    assert result['raw_decomposition_max_error']==0 and result['biological_readout_present']
    empty=dict(first,biological_readout=np.zeros((1,9)),old_information_readout=first['kernel_raw'])
    absent=report.activation_statistics([empty])
    assert absent['biological_readout_present'] is False
    assert 'causal' in absent['interpretation']


def test_normalized_biology_uses_original_modes_and_separate_channel_support():
    values=np.asarray([[[0.,.5,0.],[.5,1.,.5],[1.,1.,1.]]])
    support=np.asarray([[[False,True,False],[True,True,True],[True,False,False]]])
    structured=report.normalized_biology(values,support,.25,'structured')
    np.testing.assert_array_equal(structured,np.where(support,values,0)/.25)
    generic=report.normalized_biology(values,support,.5,'generic')
    expected=np.where(support,np.exp(-.5*((values-1)/.5)**2)-np.exp(-2),0)/.5
    np.testing.assert_array_equal(generic,expected)
    assert generic[0,0,0]==0 and generic[0,2,1]==0
    with pytest.raises(ValueError):report.normalized_biology(values,support,0.,'structured')
    with pytest.raises(ValueError):report.normalized_biology(values,support,.5,'Hill')
