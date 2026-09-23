"""Small synthetic checks of variable-cohort saved-prediction summaries."""
import numpy as np
import pytest

from opal2 import lincs_biology_summary as summary
from opal2.gram_oof_experiment import selection_mask


def test_cluster_bootstrap_keeps_duplicate_structures_together_not_row_independent():
    groups=np.asarray(['g0','g0','g1','g2','g2','g2'])
    weights=summary.bootstrap_weights(groups,2000,91)
    assert weights.shape==(2000,6)
    assert np.array_equal(weights[:,0],weights[:,1])
    assert np.array_equal(weights[:,3],weights[:,4]) and np.array_equal(weights[:,4],weights[:,5])
    assert np.array_equal(weights[:,[0,2,3]].sum(1),np.full(2000,3.))
    assert np.array_equal(weights,summary.bootstrap_weights(groups,2000,91))
    # Unequal group sizes induce a variable object denominator, as required.
    assert np.unique(weights.sum(1)).size>1


def test_ratio_intervals_and_paired_difference_match_explicit_resamples():
    weights=np.asarray([[1,1,1,1],[2,2,0,0],[0,0,2,2],[0,0,0,0]],float)
    a=np.asarray([.1,-.2,.4,.3]);ad=np.asarray([1,1,1,0.])
    b=np.asarray([-.1,.2,.1,.0]);bd=np.asarray([0,1,1,1.])
    report=summary.interval(a,ad,weights)
    valid=weights@ad>0;manual=(weights@a)[valid]/(weights@ad)[valid]
    assert report['estimate']==pytest.approx(a.sum()/ad.sum())
    assert report['valid_resamples']==3
    np.testing.assert_allclose(report['interval95'],np.quantile(manual,[.025,.975]))
    difference=summary.difference(a,ad,b,bd,weights)
    valid=((weights@ad)>0)&((weights@bd)>0)
    manual=(weights@a)[valid]/(weights@ad)[valid]-(weights@b)[valid]/(weights@bd)[valid]
    assert difference['estimate']==pytest.approx(a.sum()/ad.sum()-b.sum()/bd.sum())
    np.testing.assert_allclose(difference['interval95'],np.quantile(manual,[.025,.975]))
    equal=summary.difference(a,ad,a,ad,weights)
    assert equal['estimate']==0 and equal['interval95']==[0,0]


def synthetic_arrays(n,ids,allocation,offset=0.):
    values=np.linspace(-.1,.2,n)
    actual=np.stack((values*.3,values*.6,values),1)
    predicted=actual+.02*np.cos(np.arange(n)[:,None]+np.arange(3)[None])+offset
    u=np.arange(n*9,dtype=float).reshape(n,9)/(n*9)
    arrays=dict(actual=actual,predicted=predicted,p_null=np.full((n,3),.4),
        utility_crps=np.full((n,3),.025+offset),geometry_energy=np.full(n,.7),
        actual_u=u,mean_u=u+.05+offset)
    arrays.update(u_object_mse=((arrays['actual_u']-arrays['mean_u'])**2).mean(1),
        principal_mask=selection_mask(predicted[:,2],ids,allocation,.25,2))
    return arrays


def test_variable_n_principal_budget_scores_and_paired_risk_have_no_639_or_79_assumption():
    sizes=[11,12,10,13,13];n=sum(sizes)
    ids=np.asarray([f'object_{i:03d}' for i in range(n)])
    allocation=np.repeat(np.arange(5),sizes)
    records=[dict(fold=f,test=np.flatnonzero(allocation==f).tolist()) for f in range(5)]
    a=synthetic_arrays(n,ids,allocation);b=synthetic_arrays(n,ids,allocation,.003)
    scores=summary._model_scores(a,ids,records)
    expected=sum(int(np.floor(.25*v))//2 for v in sizes)
    assert n==59 and expected==5
    assert scores['principal_policy']['eligible_n']==n
    assert scores['principal_policy']['selected_n']==expected
    assert scores['principal_policy']['used_wells']==2*expected
    assert len(scores['selected_ids'])==expected
    assert [v['action'] for v in scores['actions']]==['Z1','Z2','Z1Z2']
    assert scores['actions'][2]['gamma_crps']==pytest.approx(.025)
    weights=summary.bootstrap_weights(ids,200,88)
    comparison=summary.compare('a','b',{'a':a,'b':b},weights,records)
    assert comparison['gamma_crps']['estimate']==pytest.approx(-.003)
    assert comparison['net_gain_per_selected']['estimate']==0
    assert comparison['fdp']['estimate']==0 and comparison['fpr']['estimate']==0
    assert len(comparison['folds'])==5 and sum(v['n'] for v in comparison['folds'])==59


def test_saved_draws_are_all_used_for_means_and_null_not_only_a_prefix(tmp_path):
    ids=np.asarray(['small_a','small_b']);allocation=np.asarray([0,1])
    records=[dict(fold=f,test=[f]) for f in range(2)]
    manifest={'folds':records}
    archives=[]
    for fold in range(2):
        folder=tmp_path/'folds'/f'fold_{fold}'/'arms/HR/evaluation'
        folder.mkdir(parents=True)
        # Exact two-mass distribution with the last 8000 draws different.
        draws=np.full((10000,1,3),-.1-fold*.01);draws[:2000]=.2
        values=dict(ids=ids[[fold]],actual=np.asarray([[.01,-.02,.03]]),
            predicted=draws.mean(0),p_null=(draws<=0).mean(0),
            utility_crps=np.full((1,3),.02),geometry_energy=np.ones(1),utility_samples=draws)
        np.savez_compressed(folder/'predictions.npz',**values)
        np.savez_compressed(folder/'u_predictions.npz',ids=ids[[fold]],
            actual_u=np.zeros((1,9)),mean_u=np.ones((1,9))*.1)
        archives.append((folder,values))
    arrays,rows=summary.read_arm(tmp_path,manifest,ids,allocation,'HR')
    assert arrays['predicted'].shape==(2,3) and len(rows)==2
    np.testing.assert_allclose(arrays['p_null'],.8)
    folder,values=archives[0]
    changed=dict(values,predicted=values['utility_samples'][:2000].mean(0))
    np.savez_compressed(folder/'predictions.npz',**changed)
    with pytest.raises(ValueError,match='means'):
        summary.read_arm(tmp_path,manifest,ids,allocation,'HR')
    changed=dict(values,p_null=np.full((1,3),.1))
    np.savez_compressed(folder/'predictions.npz',**changed)
    with pytest.raises(ValueError,match='NULL'):
        summary.read_arm(tmp_path,manifest,ids,allocation,'HR')
