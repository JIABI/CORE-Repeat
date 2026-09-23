"""Arithmetic tests only; no experimental or protected measurements are read."""
import itertools

import numpy as np
import pytest

from opal2.frozen_acquisition_policy import select_frozen_cohort_plan
from opal2.r4_confirmatory_metrics import (
    campaign_budget, utility_bounds, paired_utility_bounds, risk_bounds,
)


@pytest.mark.parametrize("n,b,k", [(1543,385,192),(1541,385,192),(904,226,113),(7,1,0)])
def test_budget(n,b,k):
    plan=campaign_budget(n)
    assert (plan['extra_well_budget'],plan['activations']) == (b,k)
    assert plan['added_action_wells']+plan['unused_action_wells'] == b


def test_missing_x_preserves_population_budget():
    p=campaign_budget(1541,10)
    assert (p['population_n'],p['extra_well_budget'],p['activations']) == (1541,385,10)
    assert campaign_budget(1541,0)['activations'] == 0


@pytest.mark.parametrize("args", [(0,),(-1,),(True,),(8,9),(8,-1),(8,1.5)])
def test_bad_budget(args):
    with pytest.raises(ValueError): campaign_budget(*args)


def test_ties_use_existing_engine_and_stable_ids():
    p=select_frozen_cohort_plan(['b','a','c'],[.1]*3,[.2]*3,2)
    q=select_frozen_cohort_plan(['c','b','a'],[.1]*3,[.2]*3,2)
    assert p.selected_ids_in_rank_order == q.selected_ids_in_rank_order == ('a','b')


def test_reference_only_cost_and_shared_missing_cancellation():
    y=[.1,np.nan,.2]
    a=[True,True,False]
    b=[False,True,True]
    u=utility_bounds(y,a,additional_reference_cost=.03)
    assert u['lower'] == pytest.approx((.1-1.02-.03)/3)
    d=paired_utility_bounds(y,a,b)
    assert d['lower'] == d['upper'] == pytest.approx(-.1/3)
    assert d['unresolved_discordant_n'] == 0


def test_paired_bounds_match_all_endpoint_completions():
    y=np.array([np.nan,np.nan,.2,np.nan])
    a=np.array([True,False,True,True]);b=np.array([False,True,False,True])
    d=paired_utility_bounds(y,a,b,reference_cost_a=.04,reference_cost_b=.01)
    values=[]
    for endpoints in itertools.product([-1.02,.98],repeat=3):
        z=y.copy();z[np.isnan(z)]=endpoints
        values.append((z[a].sum()-z[b].sum()-.04+.01)/4)
    assert d['lower'] == pytest.approx(min(values))
    assert d['upper'] == pytest.approx(max(values))


def test_all_missing_as_null_does_not_maximize_fpr():
    r=risk_bounds([-.1,-.1,np.nan,np.nan,np.nan],[True,False,True,False,False])
    assert r['fpr']['lower'] == .25
    assert r['fpr']['upper'] == pytest.approx(2/3)
    assert 2/5 < r['fpr']['upper']  # incorrectly NULL-completing everyone


def test_risk_formula_exhaustive_small_counts():
    for sn,un,sm,um in itertools.product(range(3),repeat=4):
        if sn+un+sm+um == 0: continue
        y=[-.1]*(sn+un)+[np.nan]*(sm+um)
        a=[True]*sn+[False]*un+[True]*sm+[False]*um
        r=risk_bounds(y,a)['fpr']
        ratios=[]
        for xs in range(sm+1):
            for xu in range(um+1):
                d=sn+un+xs+xu
                if d: ratios.append((sn+xs)/d)
        assert r['lower'] == (min(ratios) if ratios else None)
        assert r['upper'] == (max(ratios) if ratios else None)
        assert r['undefined_completion_possible'] == (sn+un == 0)


def test_zero_activation_and_no_null_are_undefined_not_zero_risk():
    r=risk_bounds([.2,.1],[False,False])
    assert r['fdp']['lower'] is None
    assert r['fpr']['lower'] is None
    assert r['sensitivity']['lower'] == 0


@pytest.mark.parametrize("y,a", [([2.],[True]),([np.inf],[True]),([.1],[1]),([],[])])
def test_invalid_outcomes(y,a):
    with pytest.raises(ValueError): utility_bounds(y,a)
