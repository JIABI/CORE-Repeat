"""Information-only weighting, donor-group exclusion, and joint moment checks."""
import inspect

import numpy as np
import pytest

from opal2.reference_information_memory import (
    ALPHA_GRID, LAMBDA_GRID, augment_generic_weights, chemistry_similarity,
    cosine_relationship, morphology_similarity, normalized_topk_weights,
    predict_memory, select_donor_hyperparameters, weighted_residual_moments,
)


def test_relationship_cosines_masks_zero_profiles_and_chemistry_metrics():
    query = np.array([[1., 0.], [np.nan, np.inf], [0., 0.]])
    donor = np.array([[1., 1.], [0., 1.], [np.nan, np.nan]])
    qm, dm = np.array([True, False, True]), np.array([True, True, False])
    cosine = cosine_relationship(query, donor, qm, dm)
    np.testing.assert_allclose(cosine, [[1/np.sqrt(2), 0, 0], [0, 0, 0], [0, 0, 0]])
    morphology = morphology_similarity([[1., 0.], [-1., 0.], [0., 0.]], [[1., 0.], [-1., 0.]])
    np.testing.assert_array_equal(morphology, [[1, 0], [0, 1], [0, 0]])
    np.testing.assert_allclose(chemistry_similarity(query, donor, query_mask=qm, donor_mask=dm),
                               [[.5, 0, 0], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(chemistry_similarity(query, donor, metric='cosine', query_mask=qm, donor_mask=dm), cosine)
    with pytest.raises(ValueError): cosine_relationship([[-1, 0]], [[1, 0]])
    with pytest.raises(ValueError): chemistry_similarity([[.5, 0]], [[1, 0]])
    # Bool/uint8 fingerprint products must count bits, not use boolean or
    # overflowing integer matrix multiplication.
    for dtype in (bool, np.uint8):
        bits = np.ones((1, 512), dtype=dtype)
        np.testing.assert_array_equal(chemistry_similarity(bits, bits), [[1.]])


def test_topk_ties_group_exclusion_and_uniform_fallback():
    similarity = np.array([[9., 3., 3., 1.], [1., 0., -2., 0.]])
    result = normalized_topk_weights(similarity, donor_ids=['d', 'c', 'a', 'b'], top_k=1,
                                    query_groups=['g0', 'g0'], donor_groups=['g0', 'g1', 'g2', 'g3'])
    np.testing.assert_array_equal(result['weights'][0], [0., 0., 1., 0.])
    np.testing.assert_array_equal(result['weights'][1], [0., 1/3, 1/3, 1/3])
    np.testing.assert_array_equal(result['fallback'], [False, True])
    np.testing.assert_array_equal(result['selected_count'], [1, 3])
    np.testing.assert_allclose(result['neff'], [1., 3.])
    np.testing.assert_allclose(result['weights'].sum(1), 1.)
    assert not result['weights'][~result['eligible']].any()
    with pytest.raises(ValueError, match='both'):
        normalized_topk_weights(similarity, donor_ids=['a', 'b', 'c', 'd'], query_groups=['g0', 'g1'])
    with pytest.raises(ValueError, match='eligible donor'):
        normalized_topk_weights([[1, 2]], donor_ids=['a', 'b'], query_groups=['g'], donor_groups=['g', 'g'])
    with pytest.raises(ValueError): normalized_topk_weights([[1, 2]], donor_ids=['a', 'a'])


def test_topk_and_similarity_permutation_equivariance_even_for_ties():
    rng = np.random.default_rng(703)
    query, donor = rng.normal(size=(5, 7)), rng.normal(size=(6, 7))
    order = np.array([3, 0, 5, 1, 4, 2])
    ids = np.array(['f', 'a', 'e', 'b', 'd', 'c'])
    sim = morphology_similarity(query, donor)
    sim[0] = 1
    first = normalized_topk_weights(sim, donor_ids=ids, top_k=2)
    second = normalized_topk_weights(sim[:, order], donor_ids=ids[order], top_k=2)
    np.testing.assert_array_equal(second['weights'], first['weights'][:, order])
    np.testing.assert_allclose(morphology_similarity(query, donor[order]), morphology_similarity(query, donor)[:, order])


def test_biology_mixture_hand_math_exclusion_and_exact_missing_fallback():
    generic = np.array([[1., 0., 0.], [.2, .3, .5]])
    allowed = np.ones_like(generic, dtype=bool); allowed[0, 2] = False
    target = np.array([[1., 1., 1.], [0., 0., 0.]])
    result = augment_generic_weights(generic, target, np.zeros_like(target), mixing_lambda=.5, eligible=allowed)
    np.testing.assert_array_equal(result['biology_weights'], [[.5, .5, 0.], [0, 0, 0]])
    np.testing.assert_allclose(result['neff'], [2., 0.])
    np.testing.assert_allclose(result['effective_mixing'], [.1, 0.])
    np.testing.assert_allclose(result['weights'][0], [.95, .05, 0.])
    np.testing.assert_array_equal(result['weights'][1], generic[1])
    assert result['weights'][0, 1] > 0  # biology may recruit outside generic top-k
    for mixing_lambda in LAMBDA_GRID:
        absent = augment_generic_weights(generic, mixing_lambda=mixing_lambda, eligible=allowed)
        np.testing.assert_array_equal(absent['weights'], generic)
        assert not absent['support'].any()
    zero = augment_generic_weights(generic, target, target, mixing_lambda=0., eligible=allowed)
    np.testing.assert_array_equal(zero['weights'], generic)
    with pytest.raises(ValueError): augment_generic_weights(generic, target, target, mixing_lambda=.3, eligible=allowed)
    invalid = generic.copy(); invalid[0] = [.9, 0, .1]
    with pytest.raises(ValueError, match='excluded'):
        augment_generic_weights(invalid, target, mixing_lambda=1, eligible=allowed)


def donor_case():
    ids = np.array(['d0', 'd1', 'd2', 'd3', 'd4', 'd5'])
    groups = np.array(['g0', 'g0', 'g1', 'g2', 'g3', 'g4'])
    similarity = np.array([[1., 1., .8, .4, .2, .1], [1., 1., .7, .4, .3, .2],
                           [.8, .7, 1., .2, .3, .4], [.4, .4, .2, 1., .8, .6],
                           [.2, .3, .3, .8, 1., .7], [.1, .2, .4, .6, .7, 1.]])
    residuals = np.arange(6*9).reshape(6, 9)/100+1.
    biology = np.eye(6)+.2
    biology /= biology.max()
    return ids, groups, similarity, residuals, biology


def test_donor_only_selection_excludes_entire_group_and_reproduces_grid():
    ids, groups, similarity, residuals, biology = donor_case()
    result = select_donor_hyperparameters(residuals, similarity, groups, donor_ids=ids,
        target_similarity=biology, moa_similarity=biology, top_k=2)
    excluded = groups[:, None] == groups[None]
    assert not result['donor_loo_weights'][excluded].any()
    np.testing.assert_allclose(result['donor_loo_weights'].sum(1), 1.)
    np.testing.assert_allclose(result['donor_loo_prediction'], result['alpha']*(result['donor_loo_weights']@residuals))
    np.testing.assert_allclose(result['donor_loo_mse'], np.square(residuals-result['donor_loo_prediction']).mean())
    assert len(result['candidate_scores']) == len(LAMBDA_GRID)*len(ALPHA_GRID)
    assert result['donor_loo_mse'] <= result['baseline_donor_mse']
    assert not any('query' in name for name in inspect.signature(select_donor_hyperparameters).parameters)
    with pytest.raises(TypeError):
        select_donor_hyperparameters(residuals, similarity, groups, donor_ids=ids, query_labels=residuals)
    # Each donor's leave-group-out raw prediction is independent of all
    # outcomes in its excluded group, even if selection could change globally.
    weights = result['donor_loo_weights']
    changed = residuals.copy(); changed[:2] += 1000
    np.testing.assert_array_equal((weights@changed)[:2], (weights@residuals)[:2])


def test_selection_ties_and_missing_biology_prefer_lower_lambda_alpha():
    ids, groups, sim, residuals, _ = donor_case()
    zero = select_donor_hyperparameters(np.zeros_like(residuals), sim, groups, donor_ids=ids)
    assert zero['mixing_lambda'] == 0 and zero['alpha'] == 0
    without_bio = select_donor_hyperparameters(residuals, sim, groups, donor_ids=ids)
    assert without_bio['mixing_lambda'] == 0
    assert without_bio['alpha'] == 1


def test_prediction_selection_and_moments_are_donor_permutation_invariant():
    ids, groups, sim, residuals, bio = donor_case()
    selected = select_donor_hyperparameters(residuals, sim, groups, donor_ids=ids, target_similarity=bio, top_k=2)
    query_sim = np.array([[1., 1., .2, .4, .1, .3], [0., 0., 0., 0., 0., 0.]])
    query_bio = np.array([[.4, .2, .1, .8, 0., .2], [0., 0., 0., 0., 0., 0.]])
    order = np.array([3, 1, 5, 0, 4, 2])
    kwargs = dict(query_groups=['g0', 'new'], donor_groups=groups)
    first = predict_memory(query_sim, residuals, selected, donor_ids=ids, target_similarity=query_bio, **kwargs)
    second = predict_memory(query_sim[:, order], residuals[order], selected, donor_ids=ids[order],
        target_similarity=query_bio[:, order], query_groups=['g0', 'new'], donor_groups=groups[order])
    np.testing.assert_allclose(second['prediction_mean'], first['prediction_mean'])
    np.testing.assert_allclose(second['weights'], first['weights'][:, order])
    assert not first['weights'][0, :2].any()
    changed_selection = select_donor_hyperparameters(residuals[order], sim[np.ix_(order, order)], groups[order],
        donor_ids=ids[order], target_similarity=bio[np.ix_(order, order)], top_k=2)
    assert changed_selection['mixing_lambda'] == selected['mixing_lambda']
    assert changed_selection['alpha'] == selected['alpha']
    m1 = weighted_residual_moments(first['weights'], residuals)
    m2 = weighted_residual_moments(second['weights'], residuals[order])
    np.testing.assert_allclose(m1['mean'], m2['mean'])
    np.testing.assert_allclose(m1['covariance'], m2['covariance'])
    assert 'query_labels' not in inspect.signature(predict_memory).parameters


def test_weighted_joint_covariance_is_psd_and_preserves_cross_coordinates():
    residuals = np.zeros((3, 9))
    residuals[:, 0] = [0., 1., 3.]
    residuals[:, 1] = [0., -2., -6.]
    weights = np.array([[.2, .3, .5], [0., 1., 0.]])
    moments = weighted_residual_moments(weights, residuals)
    expected_mean = weights@residuals
    expected_cov = np.stack([sum(w*np.outer(v-mean, v-mean) for w, v in zip(row, residuals))
                             for row, mean in zip(weights, expected_mean)])
    np.testing.assert_allclose(moments['mean'], expected_mean)
    np.testing.assert_allclose(moments['covariance'], expected_cov)
    assert moments['covariance'][0, 0, 1] < 0
    np.testing.assert_allclose(moments['covariance'][0, 1, 1], 4*moments['covariance'][0, 0, 0])
    assert np.linalg.eigvalsh(moments['covariance']).min() >= -1e-12
    np.testing.assert_array_equal(moments['covariance'][1], np.zeros((9, 9)))
    with pytest.raises(ValueError): weighted_residual_moments(weights, residuals[:, :8])
    with pytest.raises(ValueError): weighted_residual_moments(weights*2, residuals)
    incomplete = residuals.copy(); incomplete[1, 3] = np.nan
    with pytest.raises(ValueError, match='Complete finite'):
        weighted_residual_moments(weights, incomplete)


def test_weight_selection_apis_have_no_outcome_arguments():
    for function in (cosine_relationship, morphology_similarity, chemistry_similarity,
                     normalized_topk_weights, augment_generic_weights):
        names = inspect.signature(function).parameters
        assert not any(part in name for name in names
                       for part in ('outcome', 'residual', 'label', 'response'))
