"""Distinct engineering checks for frozen-law MC reevaluation."""
import numpy as np
import pytest
import torch

from opal2.gamma_sampling_stability import (
    joint_gamma_samples, mc_continuation_gate, policy_summaries, summarize_draw_prefixes,
)
from opal2.gram_factor_verified import decode_draws
from opal2.gram_geometry import gram_gains
from opal2.hierarchical_geometry import sample_joint_coordinates
from opal2.objective_analysis import fair_crps


def test_complete_fold_draws_preserved_by_decoder_chunking_and_old_sampler():
    mean=np.array([[.2]*9,[-.1]*9]); covariance=np.eye(9)*.01
    covariance[0,1]=covariance[1,0]=.003
    center=np.linspace(-.05,.05,9);scale=np.linspace(.7,1.3,9)
    epsilon=np.random.default_rng(88).standard_normal((18,2,9))
    one,audit=joint_gamma_samples(mean,covariance,center,scale,epsilon,chunk=18)
    pieces,_=joint_gamma_samples(mean,covariance,center,scale,epsilon,chunk=4)
    original=sample_joint_coordinates(mean,covariance,18,88)*scale+center
    g,_=decode_draws(original)
    expected=gram_gains(torch.as_tensor(g)).numpy()
    np.testing.assert_allclose(one,pieces,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(one<=0,pieces<=0)
    np.testing.assert_array_equal(one,expected)
    assert audit['draw_object_count']==36 and audit['rows_dropped']==0


def test_prefix_summaries_are_exact_original_fair_crps_not_resampled():
    rng=np.random.default_rng(30)
    gains=rng.normal(size=(20,7,3));actual=rng.normal(size=(7,3))
    out=summarize_draw_prefixes(gains,actual,(6,20))
    np.testing.assert_array_equal(out[6]['utility_crps'],fair_crps(gains[:6],actual))
    np.testing.assert_array_equal(out[6]['predicted'],gains[:6].mean(0))
    np.testing.assert_array_equal(out[20]['p_null'],(gains<=0).mean(0))


def test_all_original_policies_and_fixed_physical_budget():
    ids=np.array([f'ID{i:04}' for i in range(639)])
    folds=np.repeat(np.arange(5),[128,128,128,128,127])
    rng=np.random.default_rng(67)
    actual=rng.normal(size=(639,3));mean=rng.normal(size=(639,3));pnull=rng.uniform(size=(639,3))
    rows,masks=policy_summaries(mean,pnull,actual,ids,folds)
    assert len(rows)==len(masks)==36
    principal=next(row for row in rows if row['key']=='common_budget/Z1Z2/0.25/expected_gain')
    assert principal['selected_n']==79 and principal['used_wells']==158
    mask=masks[principal['key']]
    assert [int(mask[folds==f].sum()) for f in range(5)]==[16,16,16,16,15]


def test_mc_gate_uses_three_replicates_and_sign_not_pseudo_samples():
    result=mc_continuation_gate([-.003,-.0029,-.0031])
    assert result['passed'] and result['all_three_favor_K']
    assert result['mc_replicate_se']==pytest.approx(np.std([-.003,-.0029,-.0031],ddof=1)/np.sqrt(3))
    assert not mc_continuation_gate([-.003,.00001,-.003])['passed']
    assert not mc_continuation_gate([-.001,-.000001,-.000001])['passed']
    with pytest.raises(ValueError): mc_continuation_gate([-.001]*639)
