import numpy as np
import pytest

from opal2.conditional_residual_information_experiment import calibrate_extended_scatter, columns_without
from opal2.conditional_residual_information import extend_scatter, fit_residual_state
from opal2.empirical_radial import fit_radial


def test_recalibration_with_identity_exact():
    rng = np.random.default_rng(765)
    nc,nq=12,7
    mc,mq=rng.normal(size=(nc,9))*.1,rng.normal(size=(nq,9))*.1
    scale=np.linspace(.8,1.2,9)
    cov=np.diag(np.linspace(.5,2.,9))
    cc=np.tile(cov,(nc,1,1)); cq=np.tile(cov,(nq,1,1))
    residual=rng.normal(size=(nc,9))
    c,q,radii,law=calibrate_extended_scatter(mc,mq,scale,cc,cq,residual,np.ones((nc,2)),np.ones((nq,2)))
    np.testing.assert_array_equal(c,cc); np.testing.assert_array_equal(q,cq)
    expected=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(cc),residual[...,None])[...,0],axis=1)
    np.testing.assert_array_equal(radii,expected)
    np.testing.assert_array_equal(law['log_centers'],fit_radial(expected)['log_centers'])
    _,newq,newradii,newlaw=calibrate_extended_scatter(mc,mq,scale,cc,cq,residual,
        np.tile([.6,1.8],(nc,1)),np.tile([.7,1.7],(nq,1)))
    assert not np.array_equal(newradii,radii)
    assert not np.array_equal(newq,cq)
    assert np.linalg.eigvalsh(newq).min()>0


def test_column_and_shape_boundaries():
    assert columns_without({'amplitude':[0,1],'cell_count':[2,3],'context':[4]},('cell_count',)).tolist()==[0,1,4]
    with pytest.raises(ValueError,match='one nine-dimensional'):
        extend_scatter(np.zeros((2,9)),np.ones(9),np.eye(9),np.ones((2,2)))
    with pytest.raises(ValueError,match='positive'):
        fit_residual_state(np.ones((2,3)),np.ones((2,4)),np.array([[0.,1.],[1.,1.]]),np.ones((2,2)),['a','b'])
