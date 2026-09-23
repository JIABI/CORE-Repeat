import numpy as np

from opal2.biology_null_increment import (borrow, choose_alpha, combine_corrections,
    fixed_budget, permute_blocks, fit_generic)


def test_borrow_excludes_whole_chemical_group_and_shrinks_support():
    corr, support, neff = borrow(np.ones((2, 3)), np.array(['a','x']),
        np.array(['a','a','b']), np.array([100.,100.,1.]))
    assert support.all()
    np.testing.assert_allclose(neff, [1.,3.])
    np.testing.assert_allclose(corr[0], 1/9)


def test_unknown_relationship_is_exact_zero():
    corr, support, neff = borrow(np.zeros((2, 3)), ['a','b'], ['x','y','z'], np.ones(3))
    np.testing.assert_array_equal(corr, [0,0])
    assert not support.any() and not neff.any()


def test_calibration_can_switch_off_harmful_auxiliary_information():
    alpha, _ = choose_alpha(np.array([.1,.9]), np.array([.5,-.5]), np.array([0,1]))
    assert alpha == 0


def test_both_averages_only_supported_channels():
    out = combine_corrections([np.array([.2,.2,0]),np.array([0,.4,0])],
        [np.array([True,True,False]),np.array([False,True,False])])
    np.testing.assert_allclose(out, [.2,.3,0])


def test_fixed_budget_ties_use_ids():
    out = fixed_budget(np.array(['b','a','c']), np.array([.1,.1,.2]),
        [dict(query=np.arange(3),budget=1)])
    np.testing.assert_array_equal(out, [False,True,False])


def test_permutation_does_not_cross_blocks_or_move_unlisted_rows():
    p = permute_blocks(8,[np.array([0,2,4]),np.array([1,3])],np.random.default_rng(9))
    assert set(p[[0,2,4]]) == {0,2,4}
    assert set(p[[1,3]]) == {1,3}
    np.testing.assert_array_equal(p[[5,6,7]], [5,6,7])


def test_query_labels_never_enter_generic_readout():
    rng=np.random.default_rng(1); x=rng.normal(size=(24,8)); d=rng.normal(size=(24,5))
    y=np.arange(24)%2; fit=np.arange(16); query=np.arange(16,24)
    first=fit_generic(x,d,y,fit,query)
    y[query]=1-y[query]
    np.testing.assert_array_equal(first,fit_generic(x,d,y,fit,query))
