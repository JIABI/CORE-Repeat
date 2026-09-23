import numpy as np
import pytest
from scipy.stats import beta

from opal2.decision_frontier_diagnostic import (select_policy, rank_band, cp_upper,
    cp_lower, counts_and_metrics, ClusterResamples, bca_mean_lower, cell_indices)


def test_one_sided_cp_not_two_sided():
    assert cp_upper(10,146)==pytest.approx(beta.ppf(.95,11,136))
    assert cp_upper(10,146)<beta.ppf(.975,11,136)
    assert cp_upper(0,10)==pytest.approx(1-.05**.1)
    assert cp_upper(10,10)==1
    assert cp_lower(0,10)==0
    assert cp_lower(10,10)==pytest.approx(.05**.1)
    with pytest.raises(ValueError):cp_upper(1,0)


def test_deterministic_endpoint_ranks_and_cell_budgets():
    ids=np.array(['c','b','a','d'])
    mean=np.array([2.,2.,1.,0.]); p=np.array([.7,.2,.2,.1])
    cells=[(np.arange(3),1),(np.array([3]),1)]
    np.testing.assert_array_equal(select_policy(mean,p,ids,cells,0),[False,True,False,True])
    np.testing.assert_array_equal(select_policy(mean,p,ids,cells,np.inf),[False,False,True,True])
    np.testing.assert_array_equal(select_policy(mean,p,ids,cells,100),[False,True,False,True])


def test_original_positive_threshold_not_not_null():
    y=np.array([-.1,0.,.003,.005,.1]); selected=np.array([1,0,1,1,0],bool)
    r=counts_and_metrics(y,np.ones(5),np.full(5,.2),selected)
    assert r['selected_n']==3 and r['selected_null']==1
    assert r['selected_positive']==1 and r['selected_ambiguous']==1
    assert r['fpr']==.5 and r['sensitivity']==.5
    assert r['actual_all_object_value']==pytest.approx((-.1+.003+.005)/5)


def test_fixed_rank_band_midpoints():
    n=100; mask=rank_band(-np.arange(n),np.arange(n).astype(str),[(np.arange(n),12)])
    np.testing.assert_array_equal(np.where(mask)[0],np.arange(8,16))


def test_cluster_ratio_and_bca_constant():
    labels=np.array(['a','a','b','c']); sampler=ClusterResamples(labels,100,42)
    np.testing.assert_allclose(sampler.mean(np.full(4,2.)),2.)
    r=bca_mean_lower(np.full(4,2.),sampler)
    assert r['lower_one_sided_95']==2.
    np.testing.assert_allclose(sampler.ratio(np.array([1,1,0,0]),np.array([2,2,0,0]))[sampler.weights[:,0]>0],.5)


def test_cell_coverage_and_own_reference_exclusion():
    ids=np.array(['a','b'])
    good=[dict(query_ids=['a'],fit_ids=['b'],calibration_ids=[],budget=1),dict(query_ids=['b'],fit_ids=['a'],calibration_ids=[],budget=1)]
    assert len(cell_indices(ids,good))==2
    bad=[dict(query_ids=['a','b'],fit_ids=['b'],calibration_ids=[],budget=1)]
    with pytest.raises(ValueError):cell_indices(ids,bad)


def test_lambda_sweep_trades_predicted_value_for_predicted_risk():
    rng=np.random.default_rng(11);n=100
    ids=np.arange(n).astype(str); mean=rng.normal(size=n); p=rng.random(n)
    cells=[(np.arange(50),7),(np.arange(50,100),6)]
    selections=[select_policy(mean,p,ids,cells,x) for x in (0,.01,.1,1,10,np.inf)]
    assert all(s.sum()==13 for s in selections)
    assert np.all(np.diff([mean[s].sum() for s in selections]) <= 1e-12)
    assert np.all(np.diff([p[s].sum() for s in selections]) <= 1e-12)
