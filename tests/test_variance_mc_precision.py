import numpy as np
import pytest

from opal2.conditional_joint_error_experiment import observable_forward
from opal2.empirical_radial import fit_radial
from opal2.variance_mc_precision import gamma_factor, draw_object, summarize_draws, boundary_status


def test_exact_gamma_matches_original_full_observable_forward():
    rng = np.random.default_rng(591)
    raw = rng.normal(size=(2000, 7, 9))
    np.testing.assert_allclose(gamma_factor(raw), observable_forward(raw)[0], rtol=3e-13, atol=2e-15)
    raw = np.zeros((1, 9)); raw[:, 3] = -400.
    with pytest.raises(ValueError): gamma_factor(raw)
    with pytest.raises(ValueError): observable_forward(raw)


def test_joint_score_variance_and_common_random_pair_zero():
    rng = np.random.default_rng(520)
    g = rng.normal(.02, .2, (2000, 1)); draws = np.repeat(g, 3, axis=1)
    out = summarize_draws(draws, .04)
    score = draws-.2*(draws <= 0)
    np.testing.assert_allclose(out['score_mc_se'], score.std(0, ddof=1)/np.sqrt(len(score)))
    np.testing.assert_array_equal(out['score_paired_mc_se'], np.zeros(3))
    np.testing.assert_array_equal(out['paired_crps_mc_se'], np.zeros(3))
    np.testing.assert_allclose(out['crps_batch_jackknife_se'], out['batch_crps'].std(0, ddof=1)/np.sqrt(20))
    assert np.all(out['gamma_null_covariance'] < 0)


def test_nested_prefixes_reproduce_and_arms_share_draws():
    law = fit_radial(np.exp(np.linspace(0, 2, 15)))
    args = (np.zeros(9), np.repeat(np.eye(9)[None]*.15, 3, 0), np.zeros(9), np.ones(9),
            law, np.ones(15)/15, 15, 13)
    a = draw_object(*args, 10000); b = draw_object(*args, 20000)
    np.testing.assert_array_equal(a, b[:10000])
    np.testing.assert_array_equal(a[:, 0], a[:, 2])


def test_boundary_refinement_depends_only_on_score_and_se():
    names = np.array(['a', 'b', 'c', 'd'])
    b = boundary_status(names, [.4, .301, .300, .1], [.001, .005, .005, .001], 2)
    np.testing.assert_array_equal(b['ambiguous'], [False, True, True, False])
    assert not b['all_selected_lower_above_unselected_upper']
    b = boundary_status(names, [.4, .301, .300, .1], np.full(4, .00001), 2)
    assert b['all_selected_lower_above_unselected_upper']
