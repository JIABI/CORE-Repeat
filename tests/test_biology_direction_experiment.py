import numpy as np
import pytest

from opal2.biology_direction_experiment import safe_cosine, direction_readouts, choose_alpha


def test_zero_cosines_are_explicit_without_dropping_rows():
    c,z=safe_cosine(np.array([[1.,0],[0,0],[1,1]]),np.array([[1.,0],[1,0],[0,0]]))
    np.testing.assert_array_equal(c,[1,0,0]);np.testing.assert_array_equal(z,[False,True,True])


def test_whitening_uses_one_query_frame_for_both_vectors():
    e=np.array([[2.,1.],[1.,3.]])
    b=np.array([[1.,2.],[3.,1.]])
    cov=np.array([[[4.,0],[0,1]],[[1.,0],[0,9]]])
    out=direction_readouts(e,b,cov,.25)
    l=np.linalg.cholesky(cov)
    ew=np.stack([np.linalg.solve(l[i],e[i]) for i in range(2)])
    bw=np.stack([np.linalg.solve(l[i],b[i]) for i in range(2)])
    expected,_=safe_cosine(ew,bw)
    np.testing.assert_allclose(out['whitened_cosine'],expected)
    np.testing.assert_allclose(out['whitened_mse_shrunk'],((ew-.25*bw)**2).mean(1))


def test_off_and_unsupported_recover_original_mean_loss_exactly():
    rng=np.random.default_rng(17);e=rng.normal(size=(8,9));b=rng.normal(size=(8,9));b[1]=0
    cov=np.tile(np.eye(9),(8,1,1))
    off=direction_readouts(e,b,cov,0.)
    np.testing.assert_array_equal(off['mse_shrunk'],off['mse_core'])
    on=direction_readouts(e,b,cov,.5)
    assert on['mse_shrunk'][1]==on['mse_core'][1]


def test_calibration_strength_detects_consistent_direction_and_rejects_wrong_one():
    e=np.arange(1,7.)[:,None]*np.array([[1.,2.]])
    groups=np.array(list('abcdef'));support=np.ones(6,bool)
    a,record=choose_alpha(e,e,support,groups)
    assert a==1. and record['supported_groups']==6
    a,_=choose_alpha(e,-e,support,groups)
    assert a==0.


def test_sparse_supported_groups_force_zero_and_do_not_count_rows_as_groups():
    e=np.ones((8,2));groups=np.array(['a']*4+['b']*4)
    a,record=choose_alpha(e,e,np.ones(8,bool),groups)
    assert a==0. and record['supported_groups']==2 and record['reason']=='insufficient_supported_groups'


def test_strength_does_not_use_unsupported_objects():
    e=np.array([[1.,1.],[2,2],[3,3],[100,100.]])
    b=e.copy();b[-1]*=-100
    a,_=choose_alpha(e,b,np.array([1,1,1,0],bool),np.array(list('abcd')))
    assert a==1.


@pytest.mark.parametrize('alpha',[-1,2,np.nan])
def test_invalid_strength_fails(alpha):
    with pytest.raises(ValueError):direction_readouts(np.ones((1,2)),np.ones((1,2)),np.eye(2)[None],alpha)
