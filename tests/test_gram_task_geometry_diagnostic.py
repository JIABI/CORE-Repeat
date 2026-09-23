"""Tests of the new saved-draw diagnostic paths, not biological evidence."""
from pathlib import Path

import numpy as np
import torch

from opal2.gram_geometry import gram_gains, profiles_to_gram
from opal2.gram_evaluation import fit_score_scale, paired_energy_score
from opal2.gram_task_geometry_diagnostic import (
    primitive_coordinates, validation_norm_quotient, cosine_terms, fit_task_scales,
    cancellation_decomposition, analyze_samples, _source_paths,
)
from opal2.objective_analysis import fair_crps


def fixture():
    rng=np.random.default_rng(16)
    train=profiles_to_gram(torch.from_numpy(rng.normal(size=(21,4,7)))).numpy()
    y=rng.normal(size=(7,4,7))
    draws=np.broadcast_to(y,(20,7,4,7)).copy()+rng.normal(scale=.5,size=(20,7,4,7))
    draws[:,:,0]=y[:,0]
    actual=profiles_to_gram(torch.from_numpy(y)).numpy()
    samples=profiles_to_gram(torch.from_numpy(draws)).numpy()
    scales=fit_task_scales(train,fit_score_scale(train))
    train_mean=cosine_terms(torch.from_numpy(train)).numpy().mean(0)
    return train,actual,samples,scales,train_mean


def test_quotient_drops_only_validation_length_and_retains_acquisition_sign_change():
    y=torch.eye(4,dtype=torch.float64)
    y[3]=torch.tensor([1.,1.,0.,1.],dtype=torch.float64)/np.sqrt(3)
    a=y.clone();b=y.clone();a[1]*=.1
    ga,gb=profiles_to_gram(a),profiles_to_gram(b)
    torch.testing.assert_close(primitive_coordinates(ga)[:6],primitive_coordinates(gb)[:6])
    assert gram_gains(ga)[2] < 0 < gram_gains(gb)[2]
    b[3]*=13.
    scaled=profiles_to_gram(b)
    quotient=validation_norm_quotient(scaled)
    torch.testing.assert_close(gram_gains(quotient),gram_gains(scaled),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(primitive_coordinates(quotient)[:8],primitive_coordinates(scaled)[:8])
    assert abs(float(primitive_coordinates(quotient)[8])) < 1e-15
    torch.testing.assert_close(quotient,gb,atol=1e-14,rtol=1e-14)


def test_blockwise_scores_and_original_utilities_match_dense_calculation():
    train,actual,samples,scales,train_mean=fixture()
    left,traces=analyze_samples(samples,actual,scales,train_mean,object_chunk=2)
    right,other=analyze_samples(samples,actual,scales,train_mean,object_chunk=7)
    for key in traces:
        np.testing.assert_allclose(traces[key],other[key],rtol=1e-12,atol=1e-12)
    p=primitive_coordinates(torch.from_numpy(samples)).numpy()
    a=primitive_coordinates(torch.from_numpy(actual)).numpy()
    expected=paired_energy_score(p[...,:8]/scales['task8'],a[:,:8]/scales['task8'])
    np.testing.assert_allclose(traces['energy_task8'],expected,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(traces['primitive_crps'],fair_crps(p,a),rtol=1e-13,atol=1e-13)
    np.testing.assert_array_equal(scales['task8'],scales['matched_cos_log9'][:8])
    assert left['invariance_checks']['checked_draw_objects']==140
    assert left['invariance_checks']['quotient_null_flips']==0
    assert left['invariance_checks']['max_gamma_quotient_difference'] < 1e-14
    for action in left['cancellation'].values():
        assert abs(action['mse']['identity_error']) < 1e-14
        assert abs(action['prediction_actual_covariance']['identity_error']) < 1e-14


def test_cancellation_includes_error_bias_and_all_four_cross_covariances():
    rng=np.random.default_rng(33)
    actual=rng.normal(size=(31,2))
    predicted=.5*actual+rng.normal(size=(31,2))+.7
    r=cancellation_decomposition(actual,predicted,np.array([.2,.3]))
    errors=predicted-actual
    assert abs(r['mse']['delta']-np.mean((errors[:,0]-errors[:,1])**2))<1e-14
    assert abs(r['mse']['identity_error'])<1e-14
    assert abs(r['prediction_actual_covariance']['identity_error'])<1e-14
    global_pred=np.broadcast_to(np.array([.3,.4]),actual.shape)
    g=cancellation_decomposition(actual,global_pred,np.array([.2,.3]))
    assert g['fitted_means']['covariance_cancellation_fraction'] is None
    assert g['prediction_actual_covariance']['delta_covariance']==0


def test_ridge_calibration_source_is_the_declared_forward_diagnostic():
    p,flags=_source_paths(Path('/gram'),Path('/simple'),'RIDGE_GEOMETRY','calibration')
    assert str(p)=='/simple/arms/RIDGE_GEOMETRY/calibration_forward_diagnostic'
    assert flags['ridge_calibration_forward_diagnostic']
    _,flags=_source_paths(Path('/gram'),Path('/simple'),'G_DIRECT','validation')
    assert flags['G_validation_used_for_checkpoint_selection']
