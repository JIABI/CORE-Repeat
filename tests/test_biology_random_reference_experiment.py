import numpy as np
import pytest

from opal2.biology_random_reference_experiment import (
    amplitude_bins, matched_random_weights, coefficient_plans,
)


def test_matched_weights_preserve_count_ess_bin_mass_and_legality():
    weights = np.array([[.2,.3,0,.5,0,0],[0,0,.7,0,.3,0],[0,0,0,0,0,0]])
    legal = np.array([[1,1,1,1,1,0],[0,1,1,0,1,1],[1,1,1,1,1,1]],bool)
    bins = np.array([0,0,0,1,1,1]); amp=np.array([1.,2.,3.,4.,5.,6.])
    random,diag=matched_random_weights(weights,legal,bins,amp,seed=7)
    np.testing.assert_array_equal(np.sort(random,axis=1),np.sort(weights,axis=1))
    np.testing.assert_array_equal(random[~legal],0)
    np.testing.assert_array_equal(random[2],weights[2])
    np.testing.assert_allclose(diag['ess'][:2],1/np.square(weights[:2]).sum(1))
    for row in range(2):
        src=np.flatnonzero(weights[row]);dest=diag['donor_mapping'][row,src]
        np.testing.assert_array_equal(bins[src],bins[dest])
        np.testing.assert_array_equal(random[row,dest],weights[row,src])
    assert diag['changed_weight_rows'][:2].any()


def test_nonexchangeable_bins_are_reported_not_relaxed():
    w=np.array([[.4,.6,0]])
    r,d=matched_random_weights(w,np.array([[1,1,0]],bool),np.array([0,1,1]),np.arange(3.),seed=9)
    np.testing.assert_array_equal(r,w)
    np.testing.assert_array_equal(d['fixed_mass'],[1.])
    np.testing.assert_array_equal(d['retained_weight_mass'],[1.])


def test_random_mapping_reproducible_and_not_outcome_dependent():
    w=np.array([[.2,.8,0,0,0,0]])
    args=(w,np.ones_like(w,bool),np.zeros(6,int),np.arange(6.))
    a,da=matched_random_weights(*args,seed=24)
    b,db=matched_random_weights(*args,seed=24)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(da['donor_mapping'],db['donor_mapping'])
    outputs={tuple(matched_random_weights(*args,seed=i)[0][0]) for i in range(20)}
    assert len(outputs)>1


@pytest.mark.parametrize('bad', [np.array([[.5,.5]]),np.array([[1.,-1.]])])
def test_illegal_original_weight_fails(bad):
    with pytest.raises(ValueError):
        matched_random_weights(bad,np.array([[1,0]],bool),np.array([0,0]),np.array([1.,2.]),seed=1)


def test_amplitude_strata_fit_only_declared_training_values():
    bins,edges=amplitude_bins(np.arange(10.),np.array([-100.,4.5,100.]))
    np.testing.assert_allclose(edges,[1.8,3.6,5.4,7.2])
    np.testing.assert_array_equal(bins,[0,2,4])


def test_coefficient_fixed_intensity_and_unsupported_exact_return():
    plans=coefficient_plans(np.array([1,0],bool),np.array([0,1],bool))
    assert len(plans)==7
    np.testing.assert_array_equal(plans['TARGET_A025'],[[.75,.25,0],[1,0,0]])
    np.testing.assert_array_equal(plans['MOA_A100'],[[1,0,0],[0,0,1]])
    for value in plans.values():np.testing.assert_array_equal(value.sum(1),1)
