"""Changed joint-dose fit/borrow paths; synthetic arrays test algebra only."""
import inspect
import numpy as np
import pytest

from opal2.jointdose_program_response import (
    assert_role_isolation, fit_weighted_basis, jointdose_features,
    reference_residual_candidate, fit_component_strengths, apply_program_correction,
    matched_random_reference_weights, compact_reference_weights,
)


def test_group_roles_and_train_basis_are_isolated_and_group_equal():
    with pytest.raises(ValueError, match='crosses roles'):
        assert_role_isolation(['a', 'a'], ['TRAIN', 'DEV_EVAL'])
    assert_role_isolation(['a', 'a', 'b'], ['TRAIN', 'TRAIN', 'DEV_EVAL'])
    rng = np.random.default_rng(41)
    x = rng.normal(size=(12, 9))
    first = fit_weighted_basis(x, np.arange(12), rank=4, standardize=True)
    duplicate = np.r_[0, 0, np.arange(12)]
    second = fit_weighted_basis(x[duplicate], np.arange(12)[duplicate], rank=4, standardize=True)
    np.testing.assert_allclose(first.center, second.center, atol=1e-14)
    np.testing.assert_allclose(first.scale, second.scale, atol=1e-14)
    np.testing.assert_allclose(first.components, second.components, atol=1e-12)
    assert set(inspect.signature(fit_weighted_basis).parameters) == {'train_values','groups','rank','standardize'}


def test_joint_dose_tensor_preserves_amplitude_and_has_declared_dimensions():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(15, 8))
    basis = fit_weighted_basis(x, np.arange(15), rank=4, standardize=True)
    low = jointdose_features(x, np.full(15, .0025), basis)
    high = jointdose_features(x, np.full(15, 10.), basis)
    plain = jointdose_features(x, np.full(15, .1), basis, interacting=False)
    assert low.shape == (15, 24) and plain.shape == (15, 6)
    np.testing.assert_allclose(plain[:, -2], np.log(np.linalg.norm(x, axis=1)))
    np.testing.assert_allclose(low.reshape(15,6,4)[:,:,0], plain)
    np.testing.assert_allclose(low.reshape(15,6,4)[:,:,1], -plain)
    np.testing.assert_allclose(high.reshape(15,6,4)[:,:,1], plain)


def test_signed_borrowing_zero_fallback_and_component_calibration():
    rng = np.random.default_rng(321)
    train = rng.normal(size=(20, 6))
    basis = fit_weighted_basis(train, np.arange(20), rank=3)
    baseline = rng.normal(size=(8, 6))
    support = np.array([True]*6+[False]*2)
    coeff = rng.normal(size=(8,3))
    target_coeff = coeff * [.2,.6,0.]
    fitted = fit_component_strengths(target_coeff, coeff, np.arange(8), support)
    np.testing.assert_allclose(fitted['alpha'], [.2,.6,0.], atol=1e-14)
    zero = apply_program_correction(baseline, coeff, basis, support, 0.)
    assert np.array_equal(zero, baseline)
    output = apply_program_correction(baseline, coeff, basis, support, fitted['alpha'])
    assert np.array_equal(output[~support], baseline[~support])
    projection = (output-baseline) @ (np.eye(6)-basis.components@basis.components.T)
    np.testing.assert_allclose(projection, 0., atol=1e-14)
    refs = rng.normal(size=(3,6)); weights = np.zeros((8,3)); weights[support,0] = 1.
    candidate = reference_residual_candidate(baseline, refs, weights, support)
    np.testing.assert_allclose(candidate[support], baseline[support]+refs[0])
    assert np.array_equal(candidate[~support], baseline[~support])


def test_matched_random_preserves_support_weights_amplitude_and_plate():
    weights = np.array([[.6,0,.4,0,0,0], [0,.2,0,0,.8,0], [0,0,0,0,0,0]])
    legal = np.ones_like(weights, bool)
    bins = np.array([0,0,0,1,1,1])
    query_plate = np.array(['a','b','a'])
    donor_plate = np.array(['a','a','b','a','b','b'])
    random, audit = matched_random_reference_weights(weights, legal, bins, query_plate, donor_plate, seed=72)
    for i in range(len(weights)):
        for b in np.unique(bins):
            for same in [False,True]:
                mask = (bins==b)&((donor_plate==query_plate[i])==same)
                np.testing.assert_allclose(random[i,mask].sum(), weights[i,mask].sum())
    assert np.array_equal(random[2], weights[2])
    indices, values = compact_reference_weights(random)
    restored = np.zeros_like(weights)
    for i in range(len(weights)):
        keep=indices[i]>=0; restored[i,indices[i,keep]]=values[i,keep]
    assert np.array_equal(restored, random)


def test_prediction_interfaces_have_no_query_target_argument():
    for function in (jointdose_features, reference_residual_candidate, apply_program_correction):
        assert all('target' not in name and 'cal' not in name for name in inspect.signature(function).parameters)
    assert 'cal_target_coefficients' in inspect.signature(fit_component_strengths).parameters
