import numpy as np
import pytest

from opal2.decision_region_calibration import (
    ARMS, RiskOffset, evaluate_cell, fit_risk_offset, frozen_rank_feature,
    paired_cluster_interval, probability_rows, select_top_k, validate_cell,
)


def cell():
    n, c = 32, 40
    return dict(dataset='test', cell='0', query_budget=4,
                cal_ids=np.array(['c%02d'%i for i in range(c)]),
                cal_groups=np.array(['cg%02d'%i for i in range(c)]),
                cal_actual=np.linspace(-.1, .3, c), cal_inner_fold=np.arange(c)%5,
                query_ids=np.array(['q%02d'%i for i in range(n)]),
                query_groups=np.array(['qg%02d'%i for i in range(n)]),
                query_layout=np.array(['L%d'%(i%4) for i in range(n)]),
                query_actual=np.linspace(-.2, .4, n),
                arms={a: dict(cal_p=np.linspace(.1,.4,c), cal_mean=np.linspace(.01,.2,c),
                              query_p=np.linspace(.08,.5,n), query_mean=np.linspace(.1,.3,n))
                      for a in ARMS})


def test_off_identity_including_extreme_probabilities():
    p=np.array([0., .12345678, 1.])
    z=RiskOffset('REGION', np.array([0., 0.]), 20)
    assert np.array_equal(z.predict(p, np.ones(3)), p)
    on=RiskOffset('REGION', np.array([.3, -.5]), 20)
    assert np.array_equal(on.predict(p, np.ones(3), enabled=False), p)


def test_rank_orientation_ties_and_held_fold_independence():
    ids=np.array(['b','a','d','c','f','e','h','g'])
    m=np.ones(8)
    p=np.full(8,.2)
    folds=np.repeat([0,1],4)
    r,h=frozen_rank_feature(m,p,ids,folds)
    assert r[1] == .125 and h[1] == .5
    assert h[0] == 0
    changed=m.copy();changed[4:]=np.arange(4)*100
    r2,h2=frozen_rank_feature(changed,p,ids,folds)
    assert np.array_equal(r[:4],r2[:4])
    assert np.array_equal(h[:4],h2[:4])


@pytest.mark.parametrize('variant',['GLOBAL','REGION'])
@pytest.mark.parametrize('label',[0,1])
def test_one_class_fits_are_finite(variant,label):
    fit=fit_risk_offset(np.full(40,.2),np.full(40,label),np.linspace(0,1,40),variant)
    assert np.isfinite(fit.coefficient).all()
    p=fit.predict(np.array([0.,.2,1.]),np.array([0.,.5,1.]))
    assert np.all((p>0)&(p<1))


def test_global_shift_direction_and_local_region():
    p=np.full(100,.1)
    y=np.ones(100)
    fit=fit_risk_offset(p,y,np.zeros(100),'GLOBAL')
    assert fit.coefficient[0] > 0
    region=RiskOffset('REGION',np.array([0.,1.]),100)
    out=region.predict(np.full(4,.2),np.array([0.,0.,.5,1.]))
    assert np.allclose(out[:2],.2)
    assert out[-1] > out[-2] > out[0]


def test_query_labels_never_change_fit_prediction_or_selection():
    c=cell();a,info=evaluate_cell(c)
    c['query_actual']=-c['query_actual']
    b,info2=evaluate_cell(c)
    assert info == info2
    for k in a:
        if k != 'actual': assert np.array_equal(a[k],b[k]),k
    for arm in ARMS:
        assert np.array_equal(a[arm+'__mean'],c['arms'][arm]['query_mean'])
        assert np.array_equal(a[arm+'__ORIGINAL__p'],c['arms'][arm]['query_p'])


def test_group_leakage_is_rejected():
    c=cell();c['query_groups'][0]=c['cal_groups'][0]
    with pytest.raises(ValueError,match='chemistry overlap'):validate_cell(c)
    c=cell();c['cal_groups'][0]=c['cal_groups'][1]
    with pytest.raises(ValueError,match='split across'):validate_cell(c)


def test_original_half_budget_is_required_not_recomputed():
    c=cell();c['query_budget']=5
    a,info=evaluate_cell(c)
    assert info['k']==5
    assert a['CORE__ORIGINAL__selected'].sum()==5
    for invalid in (None,True,4.5,0):
        c['query_budget']=invalid
        with pytest.raises(ValueError,match='query_budget'):validate_cell(c)


def test_cached_mc_seeds_propagate_without_refitting():
    c=cell();base=c['arms']['CORE']
    base['query_seed_p']=np.stack([base['query_p'],base['query_p']+.01])
    base['query_seed_mean']=np.stack([base['query_mean'],base['query_mean']])
    a,_=evaluate_cell(c)
    assert np.array_equal(a['CORE__REGION__p'],a['CORE__REGION__mc0_p'])
    assert np.array_equal(a['CORE__ORIGINAL__mc0_selected'],a['CORE__ORIGINAL__selected'])


def test_reranking_can_change_but_lambda_zero_cannot():
    ids=np.array(['a','b']);m=np.array([.2,.19]);p=np.array([.05,.3])
    p2=np.array([.8,.05])
    assert not np.array_equal(select_top_k(ids,m,p,1),select_top_k(ids,m,p2,1))
    assert np.array_equal(select_top_k(ids,m,p,1,0),select_top_k(ids,m,p2,1,0))


def test_paired_fixed_set_interval_and_gap_sign():
    p=np.array([.1,.2,.8,.9]);g=np.array([-.1,.1,-.1,.1])
    rows=probability_rows(p,g)
    assert np.allclose(rows['null_gap'],[.9,-.2,.2,-.9])
    d=np.array([.2,0.,.4,0.]);mask=np.array([1,0,1,0])
    interval=paired_cluster_interval(d,['a','a','b','b'],denominator=mask,replicates=200)
    assert interval['difference'] == pytest.approx(.3)
    zero=paired_cluster_interval(np.zeros(4),['a','a','b','b'],replicates=100)
    assert zero['ci95'] == [0.,0.]
