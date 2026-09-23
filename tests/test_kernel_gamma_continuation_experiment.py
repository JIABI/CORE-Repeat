"""Epoch60 readout retains the original common sampling and physical policy."""
import numpy as np

from opal2 import kernel_gamma_continuation_experiment as experiment


def test_epoch60_scoring_changes_metadata_not_distribution_or_evaluation(monkeypatch, tmp_path):
    mean = np.zeros((2,9))
    covariance = np.eye(9)
    class Ridge:
        pass
    ridge = Ridge()
    ridge.covariance = covariance
    samples, draws = object(), object()
    seen = {}
    def sample(m, c, n, seed):
        assert m is mean and c is covariance
        assert n == 2000 and seed == 200123
        return samples
    monkeypatch.setattr(experiment,'sample_joint_coordinates',sample)
    monkeypatch.setattr(experiment,'gaussian_coordinate_diagnostics',lambda *a:{'check':True})
    monkeypatch.setattr(experiment,'restore_target',lambda s,stats:s)
    def decode(s, *, verify):
        assert s is samples and verify is True
        return draws, {'joint':True}
    monkeypatch.setattr(experiment,'decode_draws',decode)
    def evaluate(folder, d, g, ids, **kwargs):
        assert d is draws
        seen.update(kwargs)
        return kwargs
    monkeypatch.setattr(experiment,'evaluate_and_save',evaluate)
    result = experiment.score(tmp_path,mean,ridge,{},mean,np.eye(4)[None].repeat(2,0),
        np.asarray(['a','b']),np.zeros((3,3)),{},123,'M_CONDITIONAL_GENERIC_GAMMA',{'trainable':3837})
    assert result['metadata']['actual_checkpoint_epoch'] == 60
    assert result['metadata']['selection'] == 'fixed epoch60'
    assert result['seed'] == 123 and result['n_bootstrap'] == result['n_random'] == 2000
    assert experiment.CONFIG['max_epochs'] == 100
    assert experiment.CONFIG['stage_epochs'] == 60
    assert {k:v for k,v in experiment.CONFIG.items() if k != 'stage_epochs'} == {
        k:v for k,v in experiment.SOURCE_CONFIG.items() if k != 'stage_epochs'}
