"""Integration checks for nested-error alignment and the mean-fitting runner."""
from copy import deepcopy
import json

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from opal2.gram_oof_ridge import fit_ridge_oof, transform_input, transform_target
from opal2.hierarchical_geometry_experiment import (
    CONFIG, rebuild_inner_ridge, restore_target, covariance_from_native_errors,
    train_hr, predict_hr, gaussian_coordinate_diagnostics,
)


def example(n=30,d=8):
    rng=np.random.default_rng(42)
    y=rng.normal(size=(n,4,d))+.4
    u=y[:,0]@rng.normal(size=(d,9))+.2*rng.normal(size=(n,9))
    return y,u


def test_inner_ridge_reconstructs_saved_oof_and_holdout_outcomes_cannot_change_mean():
    y,u=example()
    with threadpool_limits(limits=1):
        ridge,stats=fit_ridge_oof(y,u,seed=17)
        rec=ridge.metadata['outer_cv'][0]
        ii,jj,inside,c,b,native=rebuild_inner_ridge(y,u,ridge,rec)
        np.testing.assert_allclose(native,ridge.audit_arrays['oof_native_predictions'][jj],atol=1e-10)
        changed_y,changed_u=y.copy(),u.copy()
        changed_y[jj,1:]+=10000
        changed_u[jj]+=10000
        _,_,inside2,c2,b2,native2=rebuild_inner_ridge(changed_y,changed_u,ridge,rec)
    assert inside==inside2
    np.testing.assert_array_equal(c,c2)
    np.testing.assert_array_equal(b,b2)
    np.testing.assert_array_equal(native,native2)


def test_zero_correction_covariance_recovers_exact_original_ridge_error_fit():
    y,u=example()
    with threadpool_limits(limits=1):
        ridge,stats=fit_ridge_oof(y,u,seed=17)
        covariance,arrays,_=covariance_from_native_errors(u,ridge.audit_arrays['oof_native_predictions'],stats)
    np.testing.assert_array_equal(covariance,ridge.covariance)
    np.testing.assert_array_equal(arrays['residuals'],ridge.audit_arrays['oof_residuals'])
    np.testing.assert_allclose(restore_target(transform_target(u,stats),stats),u,atol=1e-12)


def test_training_serialization_freezes_base_and_diagnostics_keep_all_rows(tmp_path,monkeypatch):
    # This short synthetic integration budget is not a scientific model arm.
    cfg=deepcopy(CONFIG['residual'])
    cfg.update(max_epochs=2,min_epochs=1,validation_interval=1,report_interval=1,
               patience_checks=1,warmup_steps=0,batch_size=8)
    monkeypatch.setitem(CONFIG,'residual',cfg)
    torch.set_num_threads(1)
    y,u=example(n=30,d=5)
    with threadpool_limits(limits=1):
        ridge,stats=fit_ridge_oof(y[:20],u[:20],seed=8)
    x,target=transform_input(y[:,0],stats),transform_target(u,stats)
    ids=np.array([f'c{i}' for i in range(30)])
    args=(tmp_path/'fit',ridge.coefficient,ridge.intercept,x,target,np.arange(20),np.arange(20,25),99,ids)
    model,epoch=train_hr(*args)
    restored,epoch2=train_hr(*args)
    np.testing.assert_array_equal(model.coefficient.numpy(),ridge.coefficient)
    assert epoch==epoch2
    before=predict_hr(model,x[25:])
    np.testing.assert_array_equal(before,predict_hr(restored,x[25:]))
    report=gaussian_coordinate_diagnostics(tmp_path/'test',ids[25:],target[25:],before,ridge.covariance)
    assert report['n']==5 and report['formal_certificate'] is False
    with np.load(tmp_path/'test/u_predictions.npz',allow_pickle=False) as saved:
        assert saved['covariance_u'].shape==(5,9,9)
        np.testing.assert_array_equal(saved['ids'],ids[25:])
    assert len(json.loads((tmp_path/'test/u_diagnostics.json').read_text())['coverage'])==4
