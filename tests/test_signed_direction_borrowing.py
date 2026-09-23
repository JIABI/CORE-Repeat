import numpy as np

from opal2.signed_direction_borrowing import (source_cosine,cosine_modulated_weights,
    transport_candidate,apply_borrowing,fit_direction_strength,matched_signed_random)


def test_signed_normalizes_absolute_mass_not_cancellation():
    w=np.array([[.5,.5],[.5,.5],[0.,0.]])
    cos=np.array([[1.,-1.],[-1.,-.5],[1.,1.]])
    signed,support=cosine_modulated_weights(w,cos,signed=True)
    np.testing.assert_allclose(signed,[[.5,-.5],[-2/3,-1/3],[0,0]])
    np.testing.assert_array_equal(support,[True,True,False])
    positive,ps=cosine_modulated_weights(w,cos,signed=False)
    np.testing.assert_array_equal(positive,[[1,0],[0,0],[0,0]])
    np.testing.assert_array_equal(ps,[True,False,False])


def test_source_cosine_and_signed_transport():
    x=np.array([[1.,0.],[0.,0.]])
    r=np.array([[1.,0.],[-1.,0.]])
    np.testing.assert_array_equal(source_cosine(x,r),[[1,-1],[0,0]])
    out=transport_candidate(x,r,r+[[0,2],[0,-2]],np.array([[.5,-.5],[0,0]]),np.array([True,False]))
    np.testing.assert_array_equal(out,[[1,2],[0,0]])


def test_direction_preserves_norm_and_exact_fallbacks():
    base=np.array([[3.,4.],[2.,1.],[0.,0.],[1.,-2.]])
    cand=np.array([[9.,-4.],[8.,5.],[7.,8.],[5.,6.]])
    support=np.array([True,False,True,True])
    np.testing.assert_array_equal(apply_borrowing(base,cand,support,0.,'DIRECTION'),base)
    for mode in ('FREE','DIRECTION'):
        out=apply_borrowing(base,cand,support,.5,mode)
        np.testing.assert_array_equal(out[[1,2]],base[[1,2]])
        if mode=='DIRECTION':
            np.testing.assert_allclose(np.linalg.norm(out,axis=1),np.linalg.norm(base,axis=1),atol=1e-14)
    radial=base*2
    np.testing.assert_allclose(apply_borrowing(base,radial,support,1,'DIRECTION'),base,atol=1e-14)


def test_direction_cal_selects_using_groups_and_can_choose_zero():
    base=np.tile([1.,0.],(6,1))
    cand=np.tile([1.,1.],(6,1))
    groups=np.array(['a','a','b','c','d','e'])
    support=np.ones(6,bool)
    target=apply_borrowing(base,cand,support,.1,'DIRECTION')
    fit=fit_direction_strength(base,target,cand,groups,support)
    assert fit['alpha']==.1 and fit['n_groups']==5
    assert fit_direction_strength(base,base,cand,groups,support)['alpha']==0
    assert fit_direction_strength(base,target,cand,np.array(['a']*6),support)['alpha']==0


def test_random_exact_signed_multisets_plate_bins_and_reproducibility():
    w=np.array([[.2,-.3,0,.5,0,0],[0,0,-.6,0,.4,0.]])
    legal=np.ones_like(w,bool)
    bins=np.array([0,0,0,1,1,1])
    plate=np.array([[True,False,False,False,False,True],[False,False,False,True,False,False]])
    amp=np.arange(6.)
    random,audit=matched_signed_random(w,legal,bins,plate,amp,seed=3)
    again,_=matched_signed_random(w,legal,bins,plate,amp,seed=3)
    np.testing.assert_array_equal(random,again)
    for i in range(2):
        for b in (0,1):
            for same in (False,True):
                mask=(bins==b)&(plate[i]==same)
                np.testing.assert_array_equal(np.sort(random[i,mask]),np.sort(w[i,mask]))
    assert audit['amplitude_bin_merges']==0
    assert audit['source_plate_relations_relaxed']==0
    assert audit['maximum_stratum_absolute_mass_error']<1e-14
