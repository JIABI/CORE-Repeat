"""Small synthetic checks for shared-pool radial-mixture evaluation."""
import numpy as np
import pytest

from opal2.conditional_joint_error_experiment import observable_forward
from opal2.empirical_radial import (
    fit_radial, radial_cdf, radial_nll, variance_multiplier,
)
from opal2.objective_analysis import fair_crps
from opal2.radial_mixture_evaluation import (
    evaluate_radial_mixtures, fair_cross_absolute,
    fair_mixture_crps, fair_mixture_energy,
)


def _cross_absolute_brute(left, right):
    return np.stack([
        np.abs(left[i] - right[j])
        for i in range(len(left)) for j in range(len(right)) if i != j
    ]).mean(0)


def _crps_brute(draws, coefficients, target):
    result = np.zeros_like(target, dtype=float)
    expand = (slice(None),) + (None,) * (target.ndim - 1)
    for a, left in enumerate(draws):
        wa = coefficients[:, a][expand]
        result += wa * np.abs(left - target).mean(0)
        for b, right in enumerate(draws):
            wb = coefficients[:, b][expand]
            result -= .5 * wa * wb * _cross_absolute_brute(left, right)
    return result


def _energy_brute(draws, coefficients, target):
    result = np.zeros(len(target))
    half = len(draws[0]) // 2
    for a, left in enumerate(draws):
        result += coefficients[:, a] * np.linalg.norm(left - target, axis=-1).mean(0)
        for b, right in enumerate(draws):
            # Both ordered component directions are necessary. The same-index
            # endpoint pools may be correlated; the disjoint halves are not.
            pair = np.linalg.norm(left[:half] - right[half:], axis=-1).mean(0)
            result -= .5 * coefficients[:, a] * coefficients[:, b] * pair
    return result


def test_cross_absolute_excludes_correlated_same_index_pairs():
    rng = np.random.default_rng(481)
    left = rng.normal(size=(32, 2, 3))
    right = left + np.array([.15, -.3, .7])
    expected = _cross_absolute_brute(left, right)
    actual = fair_cross_absolute(left, right)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(actual, fair_cross_absolute(right, left), atol=2e-14)
    all_pairs = np.abs(left[:, None] - right[None, :]).mean(axis=(0, 1))
    assert not np.allclose(actual, all_pairs)


def test_cross_absolute_ties_and_constant_offsets():
    left = np.zeros((32, 2))
    right = np.broadcast_to([2., -3.], left.shape).copy()
    np.testing.assert_array_equal(fair_cross_absolute(left, left), [0., 0.])
    np.testing.assert_allclose(fair_cross_absolute(left, right), [2., 3.], atol=1e-14)


@pytest.mark.parametrize('extra_shape', [(), (3,)])
def test_fair_mixture_crps_matches_full_quadratic_cross_index_formula(extra_shape):
    rng = np.random.default_rng(482)
    shared = rng.normal(size=(32, 2, *extra_shape))
    draws = [shared, .7 * shared + 1.1, -.4 * shared - .3]
    target = rng.normal(size=(2, *extra_shape))
    coefficients = np.array([[.2, .3, .5], [.6, .1, .3]])
    expected = _crps_brute(draws, coefficients, target)
    actual = fair_mixture_crps(draws, coefficients, target)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize('endpoint', [0, 1, 2])
def test_crps_endpoints_match_existing_fair_crps(endpoint):
    rng = np.random.default_rng(483)
    draws = [rng.normal(size=(32, 2, 3)) + k for k in range(3)]
    target = rng.normal(size=(2, 3))
    coefficients = np.eye(3)[endpoint]
    expected = fair_crps(draws[endpoint], target)
    np.testing.assert_allclose(
        fair_mixture_crps(draws, coefficients, target), expected,
        rtol=3e-14, atol=3e-14,
    )
    np.testing.assert_allclose(
        fair_mixture_crps(draws, np.tile(coefficients, (2, 1)), target), expected,
        rtol=3e-14, atol=3e-14,
    )


def test_energy_uses_both_ordered_cross_component_half_pairs():
    rng = np.random.default_rng(484)
    shared = rng.normal(size=(32, 2, 9))
    second = .4 * shared + 1.
    second[16:] += .7
    third = -.5 * shared - .3
    draws = [shared, second, third]
    target = rng.normal(size=(2, 9))
    coefficients = np.array([[.2, .3, .5], [.6, .1, .3]])
    expected = _energy_brute(draws, coefficients, target)
    np.testing.assert_allclose(
        fair_mixture_energy(draws, coefficients, target), expected,
        rtol=3e-14, atol=3e-14,
    )
    # Reordering endpoints together with coefficients must not select only one
    # direction of an asymmetric finite-sample cross-component half pair.
    permutation = [2, 0, 1]
    np.testing.assert_allclose(
        fair_mixture_energy([draws[k] for k in permutation],
                            coefficients[:, permutation], target), expected,
        rtol=3e-14, atol=3e-14,
    )


@pytest.mark.parametrize('endpoint', [0, 1])
def test_energy_endpoint_matches_existing_disjoint_half_estimator(endpoint):
    rng = np.random.default_rng(485)
    draws = [rng.normal(size=(32, 2, 9)) + k for k in range(2)]
    target = rng.normal(size=(2, 9))
    selected = draws[endpoint]
    expected = (np.linalg.norm(selected - target, axis=-1).mean(0)
                - .5 * np.linalg.norm(selected[:16] - selected[16:], axis=-1).mean(0))
    np.testing.assert_allclose(
        fair_mixture_energy(draws, np.eye(2)[endpoint], target), expected,
        rtol=3e-14, atol=3e-14,
    )


@pytest.mark.parametrize('bad', [
    [-.1, 1.1], [0., 0.], [.2, .2], [np.nan, 1.], [np.inf, 0.],
    [1., 0., 0.], [[1., 0.]], [[1., 0.], [0., 1.], [1., 0.]],
])
def test_mixture_scores_reject_invalid_convex_coefficients(bad):
    draws = [np.ones((32, 2, 9)), np.zeros((32, 2, 9))]
    target = np.zeros((2, 9))
    with pytest.raises(ValueError):
        fair_mixture_crps(draws, bad, target)
    with pytest.raises(ValueError):
        fair_mixture_energy(draws, bad, target)


def test_cross_absolute_rejects_misaligned_or_nonfinite_pools():
    good = np.ones((32, 2))
    for bad in [np.ones((31, 2)), np.ones((32, 3)), np.full((32, 2), np.nan)]:
        with pytest.raises(ValueError):
            fair_cross_absolute(good, bad)
    with pytest.raises(ValueError):
        fair_cross_absolute(np.ones((1, 2)), np.ones((1, 2)))


def _engine_fixture():
    mean = np.zeros((2, 9))
    mean[1] = np.linspace(-.08, .08, 9)
    factor = np.eye(9) * .2
    factor[1, 0] = .03
    scatter = np.stack([factor @ factor.T, 1.3 * factor @ factor.T])
    target = mean + np.array([
        [.05, -.04, .06, -.02, .03, .01, -.05, .04, -.03],
        [-.08, .02, .04, .03, -.06, .07, .02, -.01, .05],
    ])
    stats = dict(u_center=np.zeros(9), u_scale=np.ones(9))
    actual, observed, differences, _ = observable_forward(target)
    norm2 = np.array([.8, 1.4])
    endpoint_weights = {
        'CORE': np.full((2, 4), .25),
        'TARGET': np.array([[.7, .1, .1, .1], [.6, .2, .1, .1]]),
        'MOA': np.array([[.1, .1, .1, .7], [.1, .1, .2, .6]]),
    }
    return dict(
        mean=mean, scatter=scatter, target=target, stats=stats,
        actual=actual, obs_actual=observed,
        absolute_actual=np.log1p(differences * norm2[:, None]),
        norm2_per_feature=norm2, seed=486, law=fit_radial([.8, 1.3, 2.1, 3.]),
        endpoint_weights=endpoint_weights, samples=32, query_block_size=2,
    )


def test_small_engine_matches_analytic_radial_density_cdf_and_covariance():
    inputs = _engine_fixture()
    coefficients = {
        'CORE': np.array([1., 0., 0.]),
        'TARGET': np.array([0., 1., 0.]),
        'MOA': np.array([0., 0., 1.]),
        'MIX': np.array([[.5, .3, .2], [.25, .2, .55]]),
    }
    results = evaluate_radial_mixtures(**inputs, coefficients=coefficients)
    assert set(results) == set(coefficients)
    endpoints = np.stack(list(inputs['endpoint_weights'].values()))
    residual = inputs['target'] - inputs['mean']
    white = np.linalg.solve(np.linalg.cholesky(inputs['scatter']), residual[..., None])[..., 0]
    radii = np.linalg.norm(white, axis=-1)
    for arm, coefficient in coefficients.items():
        c = np.broadcast_to(coefficient, (2, 3))
        weights = np.einsum('nk,knm->nm', c, endpoints)
        out = results[arm]
        np.testing.assert_allclose(
            out['nll'], radial_nll(residual, inputs['scatter'], inputs['law'], weights),
            rtol=3e-13, atol=3e-13,
        )
        np.testing.assert_allclose(out['radial_pit'], radial_cdf(inputs['law'], weights, radii),
                                   rtol=3e-13, atol=3e-13)
        multiplier = variance_multiplier(inputs['law'], weights)
        np.testing.assert_allclose(out['radial_variance_multiplier'], multiplier,
                                   rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(out['covariance_u'], inputs['scatter'] * multiplier[:, None, None],
                                   rtol=3e-13, atol=3e-13)
        for value in out.values():
            array = np.asarray(value)
            assert array.shape[0] == 2 and np.isfinite(array).all()


def test_engine_coefficient_broadcast_and_saved_core_only_rows_are_exact():
    inputs = _engine_fixture()
    coefficients = {
        'BROADCAST': [.2, .3, .5],
        'MATRIX': np.tile([.2, .3, .5], (2, 1)),
        'GATED': np.array([[1., 0., 0.], [.2, .3, .5]]),
    }
    plain = evaluate_radial_mixtures(**inputs, coefficients=coefficients)
    for key in plain['BROADCAST']:
        np.testing.assert_array_equal(plain['BROADCAST'][key], plain['MATRIX'][key])
    saved = dict(nll=np.array([-71., -72.]), predicted=np.array([.123, -.234]),
                 saved_only=np.array([11., 12.]))
    restored = evaluate_radial_mixtures(**inputs, coefficients=coefficients, baseline_saved=saved)
    assert 'saved_only' not in restored['GATED']
    assert restored['GATED']['nll'][0] == saved['nll'][0]
    assert restored['GATED']['predicted'][0] == saved['predicted'][0]
    for key in plain['GATED']:
        np.testing.assert_array_equal(restored['GATED'][key][1:], plain['GATED'][key][1:])
    for key in plain['MATRIX']:
        np.testing.assert_array_equal(restored['MATRIX'][key], plain['MATRIX'][key])


@pytest.mark.parametrize('bad', [[-.1, .6, .5], [.1, .2, .3], [[1., 0., 0.]], [1., 0.]])
def test_engine_rejects_invalid_mixture_coefficients(bad):
    with pytest.raises(ValueError):
        evaluate_radial_mixtures(**_engine_fixture(), coefficients={'BAD': bad})


def test_query_targets_cannot_change_predictions_or_predictive_uncertainty():
    inputs = _engine_fixture()
    coefficients = {'CORE': [1., 0., 0.], 'MIX': [[.5, .3, .2], [.2, .4, .4]]}
    original = evaluate_radial_mixtures(**inputs, coefficients=coefficients)
    changed = dict(inputs)
    changed['target'] = inputs['target'] + np.linspace(-.2, .25, 9)
    # Deliberately change every scoring-only truth independently. None is a
    # distribution-fitting input, gate input, or sampling-stream input.
    changed['actual'] = np.array([.35, -.45])
    changed['obs_actual'] = inputs['obs_actual'] + .7
    changed['absolute_actual'] = inputs['absolute_actual'] + .4
    rescored = evaluate_radial_mixtures(**changed, coefficients=coefficients)
    unchanged = (
        'predicted', 'p_null', 'predicted_cos_Z1_V', 'predicted_cos_Z2_V',
        'radial_variance_multiplier', 'covariance_u',
        'joint_squared_radius_by_level', 'gamma_mc_se', 'null_mc_se',
    )
    for arm in coefficients:
        for key in unchanged:
            np.testing.assert_array_equal(original[arm][key], rescored[arm][key])
        # The perturbation must actually exercise the scoring path rather than
        # accidentally leave every field unchanged.
        assert not np.allclose(original[arm]['nll'], rescored[arm]['nll'])
        assert not np.allclose(original[arm]['observable_crps'], rescored[arm]['observable_crps'])


@pytest.mark.parametrize('n', [2, 17])
def test_query_subblocks_preserve_rng_alignment_and_all_outputs(n):
    inputs = _engine_fixture()
    rows = np.arange(n) % 2
    for key in ('mean', 'scatter', 'target', 'actual', 'obs_actual',
                'absolute_actual', 'norm2_per_feature'):
        inputs[key] = inputs[key][rows].copy()
    inputs['endpoint_weights'] = {
        name: values[rows].copy() for name, values in inputs['endpoint_weights'].items()
    }
    coefficients = {'CORE': [1., 0., 0.], 'MIX': [.2, .3, .5]}
    inputs['query_block_size'] = 1
    one = evaluate_radial_mixtures(**inputs, coefficients=coefficients)
    inputs['query_block_size'] = 2
    two = evaluate_radial_mixtures(**inputs, coefficients=coefficients)
    # N=17 crosses the historical 16-query random-stream boundary. The small
    # tolerance permits reduction-order roundoff, not different random draws.
    for arm in coefficients:
        assert set(one[arm]) == set(two[arm])
        for key in one[arm]:
            np.testing.assert_allclose(one[arm][key], two[arm][key], rtol=2e-13, atol=2e-13)
