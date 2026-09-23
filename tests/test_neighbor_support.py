"""No response arrays or trained model are needed for a support-only audit."""
import inspect

import numpy as np
import pytest

from opal2.neighbor_support import (pairwise_profile_distances, tanimoto_distances,
                                   exponential_ess, neighbor_statistics)


def test_profile_distances_have_declared_angle_and_amplitude_definitions():
    query = np.array([[1., 0.], [0., 2.]])
    library = np.array([[2., 0.], [0., 1.]])
    np.testing.assert_allclose(pairwise_profile_distances(query, library), [[0., 1.], [1., 0.]])
    expected = np.sqrt(np.mean((query[:, None, :] - library[None, :, :])**2, axis=-1))
    np.testing.assert_allclose(pairwise_profile_distances(query, library, "rms"), expected)
    with pytest.raises(ValueError, match="Zero-norm"):
        pairwise_profile_distances(np.zeros((1, 2)), library)


def test_tanimoto_uses_binary_bits_and_unknown_is_not_absence():
    bits = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    distance = tanimoto_distances(bits, bits, [True, True, True, False], [True]*4)
    assert distance[0, 1] == pytest.approx(2/3)
    assert distance[0, 0] == 0
    assert np.isnan(distance[2]).all()
    assert np.isnan(distance[3]).all()
    with pytest.raises(ValueError, match="binary"):
        tanimoto_distances(bits + .1, bits)


def test_ess_uniform_distant_neighbors_are_not_absolute_support():
    near = exponential_ess(np.repeat(.1, 20), .5)
    far = exponential_ess(np.repeat(100., 20), .5)
    assert near["ess"] == pytest.approx(20)
    assert far["ess"] == pytest.approx(20)
    assert far["absolute_affinity_sum"] < 1e-80
    concentrated = exponential_ess(np.r_[0., np.repeat(10., 19)], .1)
    assert concentrated["ess"] == pytest.approx(1)
    with pytest.raises(ValueError, match="Bandwidth"):
        exponential_ess([1, 2], 0)


def synthetic_distances():
    ids = np.array([f"C{i}" for i in range(8)])
    distance = np.abs(np.arange(8)[:, None] - np.arange(7)[None]) / 10
    return ids, distance


def test_self_and_all_equal_ids_are_excluded_and_only_train_is_library():
    ids, distance = synthetic_distances()
    rows, fit = neighbor_statistics(distance, ids, ids[:7], np.arange(7))
    for i, row in enumerate(rows):
        assert str(ids[i]) not in row["top10_neighbor_ids"]
        assert set(row["top10_neighbor_ids"]).issubset(set(ids[:7]))
        assert "C7" not in row["top10_neighbor_ids"]
    expected = np.sort(np.where(np.eye(7), np.inf, distance[:7]), axis=1)[:, 4]
    assert fit["bandwidth"] == pytest.approx(np.median(expected))
    # Matching compound identity is excluded even when a library has duplicate
    # physical rows representing that identity.
    duplicated = np.column_stack([distance, distance[:, 0]])
    dup_ids = np.r_[ids[:7], ids[:1]]
    duplicate_rows, _ = neighbor_statistics(duplicated, ids, dup_ids, np.arange(7))
    assert "C0" not in duplicate_rows[0]["top10_neighbor_ids"]
    with pytest.raises(ValueError, match="TRAIN identities"):
        neighbor_statistics(distance, ids, ids[1:], np.arange(7))


def test_threshold_counts_exclude_self_and_bandwidth_ignores_heldout_queries():
    ids, distance = synthetic_distances()
    rows, fit = neighbor_statistics(distance, ids, ids[:7], np.arange(7), chemical=True)
    for i, row in enumerate(rows):
        eligible = ids[:7] != ids[i]
        expected = int((distance[i, eligible] <= .3 + 1e-12).sum())
        assert row["support_tanimoto_ge_0.7"] == expected
    changed = distance.copy()
    changed[-1] = 100
    _, fit_changed = neighbor_statistics(changed, ids, ids[:7], np.arange(7), chemical=True)
    assert fit_changed["bandwidth"] == fit["bandwidth"]
    assert fit_changed["train_loo_reference"] == fit["train_loo_reference"]
    missing = distance.copy(); missing[-1] = np.nan
    missing_rows, _ = neighbor_statistics(missing, ids, ids[:7], np.arange(7), chemical=True)
    assert missing_rows[-1]["support_tanimoto_ge_0.5"] is None
    assert missing_rows[-1]["ess_top20"] is None


def test_ties_are_broken_by_id_and_future_information_is_not_in_api():
    ids, distance = synthetic_distances()
    distance[-1] = .5
    row, _ = neighbor_statistics(distance, ids, ids[:7], np.arange(7))
    assert row[-1]["top10_neighbor_ids"] == sorted(ids[:7])
    for function in (pairwise_profile_distances, tanimoto_distances, exponential_ess, neighbor_statistics):
        arguments = inspect.signature(function).parameters
        assert not set(arguments) & {"Y", "Z", "V", "target", "gamma", "gains", "predictions", "labels"}
    with pytest.raises(TypeError):
        neighbor_statistics(distance, ids, ids[:7], np.arange(7), gamma=np.zeros(8))
