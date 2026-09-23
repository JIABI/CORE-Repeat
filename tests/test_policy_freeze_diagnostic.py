import numpy as np
import pytest

from opal2.policy_freeze_diagnostic import (minimal_margin,subset_metrics,
    difference_record,paired_metrics,score_uncertainty)
from opal2.decision_frontier_diagnostic import ClusterResamples


def test_ni_margin_is_bound_threshold_not_absence_of_significance():
    assert minimal_margin(-.03,higher_is_better=True)==.03
    assert minimal_margin(.02,higher_is_better=True)==0
    assert minimal_margin(.04,higher_is_better=False)==.04
    assert minimal_margin(-.01,higher_is_better=False)==0


def test_subset_removal_keeps_policy_does_not_refill():
    y=np.array([-.1,.2,.003,.3]);s=np.array([True,True,False,False]);keep=np.array([False,True,True,True])
    r=subset_metrics(y,s,keep)
    assert r['n']==3 and r['selected_n']==1 and r['selected_null']==0
    assert r['all_object_value']==pytest.approx(.2/3)


def test_pair_difference_and_zero_assignment_difference():
    y=np.array([-.1,.2,.01,.3]);a=np.array([0,1,1,0],bool);b=np.array([1,1,0,0],bool)
    r=difference_record(y,a,b,np.ones(4,bool))
    assert r['difference']['selected_null']==-1
    assert r['difference']['all_object_value']==pytest.approx(.11/4)
    sampler=ClusterResamples(np.arange(4),300,12)
    same=paired_metrics(y,a,a,sampler)
    assert same['all_object_value']['ci95']==[0,0]
    assert same['all_object_value']['minimum_margin_infimum_from_one_sided_bound']==0
    assert not same['all_object_value']['zero_margin_strict_superiority_descriptive']


def test_mc_score_bounds_and_id_ties():
    mean=np.array([.3,.2,.19,.1]);p=np.full(4,.1);se=np.full(4,.01);sp=np.full(4,.02)
    result=score_uncertainty(mean,p,se,sp,np.array(['a','b','c','d']),[(np.arange(4),2)],.2)[0]
    np.testing.assert_allclose(result['selected_score_mc_se_interval'],[np.hypot(.01,.004),.014])
    assert result['cutoff_gap']==pytest.approx(.01)
    assert result['cutoff_gap_mc_se_upper']==pytest.approx(.028)
    assert result['cutoff_ids']==['b','c']
    assert not result['gap_above_196_upper_se']
