import numpy as np
import torch

from opal2.gram_oof_experiment import assign_folds, selection_mask, decode_draws, uniform_global_policy
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains


def test_outer_folds_unique_and_reproducible_with_disjoint_fit_selection_test():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(100,7))
    ids=np.array([f'c{i:03d}' for i in range(len(x))])
    records,membership,strata=assign_folds(x,ids,n_strata=4)
    again=assign_folds(x.copy(),ids.copy(),n_strata=4)
    assert records==again[0]
    np.testing.assert_array_equal(membership,again[1])
    counts=np.zeros(len(x),int)
    for r in records:
        a,b,c=[set(r[k]) for k in ('fit','inner_validation','test')]
        assert not a&b and not a&c and not b&c
        assert a|b|c==set(range(len(x)))
        counts[r['test']]+=1
    np.testing.assert_array_equal(counts,np.ones(len(x),int))
    assert set(strata)==set(range(4))


def test_foldwise_global_never_ranks_between_fold_constants():
    ids=np.array([f'c{i}' for i in range(20)])
    folds=np.repeat([0,1],[8,12])
    scores=np.where(folds==0,100.,-100.)
    weights=selection_mask(scores,ids,folds,.5,2,global_model=True)
    np.testing.assert_allclose(weights,.25)
    assert weights.sum()==5
    np.testing.assert_array_equal(weights,selection_mask(-scores,ids,folds,.5,2,global_model=True))
    conditional=selection_mask(np.arange(20),ids,folds,.5,2)
    assert conditional[folds==0].sum()==2
    assert conditional[folds==1].sum()==3
    within=selection_mask(scores,ids,folds,.5,2,global_model=True,within_action=True)
    assert within.sum()==10


def test_forward_route_preserves_all_samples_and_original_gains():
    rng=np.random.default_rng(3)
    y=torch.tensor(rng.normal(size=(3,5,4,12)),dtype=torch.float64)
    g=profiles_to_gram(y)
    u=gram_to_coordinates(g).numpy()
    reconstructed,audit=decode_draws(u,verify=True)
    np.testing.assert_allclose(gram_gains(torch.tensor(reconstructed)),gram_gains(g),atol=1e-12)
    assert reconstructed.shape==g.shape
    assert audit['recovered_schur_failure_count']==0
    assert audit['functional_consistency']['rows_dropped']==0


def test_global_policy_is_expectation_and_has_no_lexical_ids():
    actual=np.array([[-1.,0.,-1.],[1.,0.,.2],[0.,0.,.5],[0.,0.,.7]])
    report={'policy':{'statistical_scope':{},'row_trace':[{'selected_by':['arbitrary']}],
        'within_action':[{'action':'Z1Z2','selected_n':2,'selected_ids':['a','b'],'matched_random':{}}],
        'common_budget':[{'action':'Z1Z2','selected_n':1,'selected_ids':['a'],'matched_random':{}}]}}
    result=uniform_global_policy(report,actual)
    assert result['policy']['within_action'][0]['fdp']==.25
    assert result['policy']['within_action'][0]['selected_null_count']==.5
    assert result['policy']['common_budget'][0]['fpr']==.25
    assert result['policy']['common_budget'][0]['selected_ids'] is None
    assert 'row_trace' not in result['policy']
    assert report['policy']['common_budget'][0]['selected_ids']==['a']
