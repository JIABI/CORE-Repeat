"""Critical new M3 paths: signed missing bounds and deterministic ranking."""
import numpy as np

from opal2.m3_amplitude_controls import bounded_total, select


def test_signed_missing_bounds_cancel_shared_objects():
    y = np.array([.1, np.nan, np.nan, np.nan])
    weights = np.array([1., 1., -1., 0.])
    result = bounded_total(y, weights, (-1.02, .98))
    assert np.isclose(result["lower"], -1.9)
    assert np.isclose(result["upper"], 2.1)
    # NULL missingness has the same cancellation but a different outcome range.
    result = bounded_total(np.array([0., np.nan, np.nan]), np.array([1., -1., 0.]), (0., 1.))
    assert result["lower"] == -1.
    assert result["upper"] == 0.


def test_scores_keep_fixed_direction_and_stable_id_ties():
    ids = np.array(["z", "b", "a", "c"])
    amp = np.array([2., 1., 1., 0.])
    np.testing.assert_array_equal(select(ids, amp, 2), [True, False, True, False])
    np.testing.assert_array_equal(select(ids, -amp, 2), [False, False, True, True])
