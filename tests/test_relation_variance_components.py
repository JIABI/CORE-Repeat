import numpy as np

from opal2.relation_variance_components import (
    group_dyad_pairs, node_pair_weights, relation_grams, solve_moments,
    unit_moments,
)


def test_psd_signed_gram_and_alias_node_bootstrap():
    target = np.array([[1., 0], [1, 1], [0, 1], [1, 0]])
    source = np.array([[1., 0], [-1, 0], [0, 1], [.5, .5]])
    k = relation_grams(target, source, np.array(['a','a','b','a']),
                       np.array(['1','2','3','1']))
    for j in range(k.shape[-1]):
        assert np.linalg.eigvalsh(k[:, :, j]).min() > -1e-12
    assert k[0, 1, 2] < 0  # Signed interaction is not positive-clipped.
    groups = np.array(['A','A','B','C'])
    i, j, w = group_dyad_pairs(groups)
    assert np.all(groups[i] != groups[j])
    assert np.isclose(w.sum(), 3.)
    _, gi = np.unique(groups, return_inverse=True)
    mult = np.array([[2., 1., 0.]])
    boot = node_pair_weights(mult, gi[i], gi[j], w)
    assert np.isclose(boot.sum(), 2.)  # Both A--B alias edges share node A.


def test_intercept_residualization_recovers_signed_and_boundary_components():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(70, 3))
    y = (9. + x @ np.array([2., -3., .5]))[:, None]
    weights = rng.uniform(.2, 2., (1, len(x)))
    gram, cross, sx, sy = unit_moments(x, y, weights)
    fit = solve_moments(gram[0], cross[0, :, 0], (0, 1, 2))
    np.testing.assert_allclose(fit['raw'], [2., -3., .5], atol=1e-12)
    assert fit['nonnegative'][1] == 0.
    assert fit['identifiable']
    assert np.isclose(sy[0, 0] - sx[0] @ fit['raw'], 9.)


def test_rank_deficiency_is_reported_not_variance_identification():
    g = np.ones((2, 2))
    fit = solve_moments(g, np.array([1., 1.]), (0, 1))
    assert fit['rank'] == 1
    assert not fit['identifiable']
    assert fit['condition_number'] is None
