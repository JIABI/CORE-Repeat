import numpy as np
import pytest

from opal2 import direct_risk_scale as m
from opal2.variance_headroom_math import fair_gamma_crps
from opal2.variance_headroom_experiment import gamma_fast


def test_ecdf_ties_and_query_independence():
    ref=np.array([1.,1.,2.,3.])
    np.testing.assert_array_equal(m.centered_ecdf(ref,np.array([0.,1.,2.,3.,4.])),[-1.,-.5,.25,.75,1.])
    assert m.centered_ecdf(ref,np.array([2.]))[0]==m.centered_ecdf(ref,np.array([999.,2.]))[1]


def test_scalar_exact_zero_and_bounded_family():
    s=np.array([-1.,0.,1.]);bound=m.ETA_BOUND
    np.testing.assert_array_equal(m.scale_function([0.,0.],s),np.zeros(3))
    np.testing.assert_allclose(m.scale_function([bound,bound],s),[0,bound,bound])
    with pytest.raises(ValueError):m.scale_function([10.],s)


def test_grid_interpolation_and_group_weights():
    table=np.array([[1.,2.,5.],[0.,2.,3.]])
    np.testing.assert_allclose(m.interpolate_scores(table,[-1.,0.,1.],[-.5,.5]),[1.5,2.5])
    w=m.group_weights(np.array(['a','a','b','c']))
    np.testing.assert_allclose(w,[1/6,1/6,1/3,1/3])


def test_direct_sampled_crps_matches_scalar_distribution():
    rng=np.random.default_rng(271)
    mean=rng.normal(0,.1,(7,9));error=rng.normal(0,.1,(200,7,9))
    target=gamma_fast(mean+rng.normal(0,.1,(7,9)))
    cells=[dict(indices=np.arange(7),mean=mean,error=error,actual=target)]
    eta=np.linspace(-.3,.3,7)
    expected=fair_gamma_crps(gamma_fast(mean[None]+error*np.exp(.5*eta)[None,:,None]),target)
    np.testing.assert_allclose(m.sampled_scores(cells,eta),expected,atol=1e-15)
    cells[0]['indices']=np.array([0,1,2,3,4,5,5])
    with pytest.raises(ValueError,match='exactly once'):m.sampled_scores(cells,eta)


def test_fit_optimizes_average_loss_not_average_oracle(monkeypatch):
    # Two groups want -.5 weakly; two want +.5 with ninefold curvature.
    # Average argmin is zero, whereas average-risk optimum is +.4.
    target=np.array([-.5,-.5,.5,.5]);curvature=np.array([1.,1.,9.,9.])
    monkeypatch.setattr(m,'sampled_scores',lambda cells,eta:curvature*(eta-target)**2)
    result=m.fit_direct_map([],np.arange(4),np.zeros(4),dimensions=1)
    assert abs(result['unshrunk_parameters'][0]-.4)<.04
    assert result['calibration_candidate_loss']<result['calibration_baseline_loss']


def test_zero_retained_when_already_optimal(monkeypatch):
    monkeypatch.setattr(m,'sampled_scores',lambda cells,eta:1.+eta**2)
    result=m.fit_direct_map([],np.arange(12),np.linspace(-1,1,12),dimensions=2)
    assert result['parameters']==[0.,0.]
    assert not result['retained']
    np.testing.assert_array_equal(result['increment'],np.zeros(12))
