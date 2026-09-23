"""Final factorial scoring preserves joint geometry, costs and random streams."""
import numpy as np
import pytest
import torch

from opal2 import kernel_final_experiment as experiment
from opal2.baseline_policy import ACTIONS
from opal2.gram_geometry import gram_gains


@pytest.mark.parametrize('arm,epoch,selection',[
    ('A_HR',None,'frozen historical HR'),
    ('D_S_C',30,'fixed epoch30'),
    ('O_G_W',30,'fixed epoch30'),
])
def test_all_final_arms_use_common_10000_draws_original_three_actions_and_metadata(
        arm,epoch,selection,monkeypatch,tmp_path):
    mean=np.zeros((2,9));covariance=np.eye(9)
    ridge=type('Ridge',(),{'covariance':covariance})()
    target=np.ones((2,9));grams=np.broadcast_to(np.eye(4),(2,4,4)).copy()
    ids=np.asarray(['a','b']);train_gains=np.arange(12,dtype=float).reshape(4,3)
    metric_scale=np.ones(9);stats={'fixed':True}
    samples,restored,draws=object(),object(),object()
    audit={'joint':True};diagnostics={'fixed_covariance':True}
    def sample(m,c,n,seed):
        assert m is mean and c is covariance
        assert n==10000 and seed==200123
        return samples
    monkeypatch.setattr(experiment,'sample_joint_coordinates',sample)
    def coordinates(folder,names,t,m,c):
        assert folder==tmp_path and names is ids and t is target and m is mean and c is covariance
        return diagnostics
    monkeypatch.setattr(experiment,'gaussian_coordinate_diagnostics',coordinates)
    def restore(s,st):
        assert s is samples and st is stats
        return restored
    monkeypatch.setattr(experiment,'restore_target',restore)
    def decode(s,*,verify):
        assert s is restored and verify is True
        return draws,audit
    monkeypatch.setattr(experiment,'decode_draws',decode)
    def evaluate(folder,d,g,names,**kwargs):
        assert folder==tmp_path and d is draws and g is grams and names is ids
        # The wrapper forwards complete four-well geometry and all three action
        # targets to the unchanged evaluator, not only the supervised ADD_TWO.
        assert ACTIONS==('Z1','Z2','Z1Z2')
        assert gram_gains(torch.as_tensor(g)).shape==(2,3)
        assert kwargs['train_actual_gains'] is train_gains
        assert kwargs['score_scale'] is metric_scale
        return kwargs
    monkeypatch.setattr(experiment,'evaluate_and_save',evaluate)
    counts={'trainable':0 if arm=='A_HR' else 3837}
    result=experiment.score(tmp_path,mean,ridge,stats,target,grams,ids,
        train_gains,metric_scale,123,arm,counts)
    assert result['seed']==123
    assert result['n_bootstrap']==result['n_random']==2000
    assert result['metadata']['arm']==arm
    assert result['metadata']['actual_checkpoint_epoch']==epoch
    assert result['metadata']['selection']==selection
    assert result['metadata']['numerics'] is audit
    assert result['metadata']['u_diagnostics'] is diagnostics
    assert result['metadata']['parameter_counts'] is counts
    assert result['metadata']['formal_certificate'] is False
    assert experiment.CONFIG['stage_epochs']==30
    assert experiment.CONFIG['max_epochs']==100
