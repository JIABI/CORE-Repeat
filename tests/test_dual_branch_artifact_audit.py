import numpy as np
import pytest
import torch

from opal2.dual_branch_artifact_audit import (check_left_parameters, exact,
    finite_tree, replay_calibration_covariance)
from opal2.dual_branch_features import calibration_frame_covariance


def test_frozen_left_requires_bitwise_identity():
    state = {'left.a': torch.tensor([1.], dtype=torch.float64), 'empirical_scale': torch.ones(2)}
    other = {key: value.clone() for key, value in state.items()}
    other['right.a'] = torch.ones(3)
    assert check_left_parameters(state, other) == 2
    other['left.a'][0] += 1e-14
    with pytest.raises(AssertionError, match='Frozen shared left'):
        check_left_parameters(state, other)


def test_exact_rejects_small_changes_and_nonfinite_values():
    exact(np.array([0., 1.]), np.array([0., 1.]), 'unchanged')
    with pytest.raises(AssertionError):
        exact([1.], [1.+1e-14], 'changed')
    with pytest.raises(AssertionError):
        exact([np.nan], [np.nan], 'nonfinite')
    assert not finite_tree({'history': [{'loss': float('inf')}]})


def test_independent_calibration_covariance_replay_excludes_group():
    rng = np.random.default_rng(45)
    n = 9
    scatter = np.broadcast_to(np.eye(9), (n, 9, 9)).copy()
    residual = rng.normal(size=(n, 9))
    groups = np.array(['a', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    amplitude = rng.normal(size=n)
    actual = replay_calibration_covariance(scatter, residual, amplitude, groups, 1.)
    expected = calibration_frame_covariance(scatter, residual, amplitude, groups, 1.)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    # Changing the own-group outcomes cannot change either own-group covariance.
    changed = residual.copy(); changed[:2] *= 100
    after = replay_calibration_covariance(scatter, changed, amplitude, groups, 1.)
    np.testing.assert_array_equal(actual[:2], after[:2])
    assert not np.array_equal(actual[2:], after[2:])


def test_calibration_replay_needs_other_groups():
    with pytest.raises(ValueError, match='three other groups'):
        replay_calibration_covariance(np.broadcast_to(np.eye(9), (3, 9, 9)),
            np.ones((3, 9)), np.arange(3), np.arange(3), 1.)
