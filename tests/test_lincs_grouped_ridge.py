"""Synthetic checks of the new optional chemical-group covariance folds."""
import numpy as np
import pytest
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from opal2.gram_simple_models import GramSimpleGaussian, _membership
from opal2.hierarchical_stability_ridge import fit_validation_ridge


@pytest.fixture(autouse=True)
def bounded_blas():
    with threadpool_limits(limits=1):
        yield


def fixture():
    rng = np.random.default_rng(417)
    n, nv, d = 24, 9, 6
    y, vy = rng.normal(size=(n, 4, d)), rng.normal(size=(nv, 4, d))
    beta = rng.normal(size=(d, 9))
    u = y[:, 0] @ beta + rng.normal(size=(n, 9)) * .4
    vu = vy[:, 0] @ beta + rng.normal(size=(nv, 9)) * .4
    groups = np.asarray([f"chem_{i:02d}" for i in range(n)])
    groups[1], groups[6], groups[13] = groups[0], groups[5], groups[12]
    return y, u, vy, vu, groups


def test_none_keeps_legacy_row_fold_numeric_and_metadata_path():
    y, u, vy, vu, _ = fixture()
    default, default_stats = fit_validation_ridge(y, u, vy, vu, 39)
    explicit, explicit_stats = fit_validation_ridge(y, u, vy, vu, 39, groups=None)
    assert default_stats == explicit_stats
    assert default.metadata == explicit.metadata
    assert "chemistry_grouped_internal_oof" not in default.metadata
    assert set(default.audit_arrays) == set(explicit.audit_arrays)
    for key in default.audit_arrays:
        np.testing.assert_array_equal(default.audit_arrays[key], explicit.audit_arrays[key])
    np.testing.assert_array_equal(default.coefficient, explicit.coefficient)
    np.testing.assert_array_equal(default.intercept, explicit.intercept)
    np.testing.assert_array_equal(default.covariance, explicit.covariance)
    _, original_allocation = _membership(len(y), 5, 39)
    np.testing.assert_array_equal(default.audit_arrays["inner_fold_membership"], original_allocation)


def test_grouped_oof_keeps_all_members_together_and_restores_full_rows(tmp_path):
    y, u, vy, vu, groups = fixture()
    model, stats = fit_validation_ridge(y, u, vy, vu, 39, groups=groups)
    unique = np.unique(groups)
    expected = np.full(len(y), -1)
    for fold, (_, held_groups) in enumerate(KFold(5, shuffle=True, random_state=39).split(unique)):
        expected[np.isin(groups, unique[held_groups])] = fold
    np.testing.assert_array_equal(model.audit_arrays["inner_fold_membership"], expected)
    np.testing.assert_array_equal(model.audit_arrays["oof_count"], np.ones(len(y)))
    assert model.metadata["chemistry_grouped_internal_oof"] is True
    assert model.metadata["fitting_chemistry_group_count"] == len(unique)
    for record in model.metadata["internal_error_folds"]:
        fit, check = record["fit_indices"], record["error_indices"]
        assert set(groups[fit]).isdisjoint(groups[check])
        assert set(fit) | set(check) == set(range(len(y)))
        assert record["fit_chemistry_groups"] == sorted(set(groups[fit]))
        assert record["error_chemistry_groups"] == sorted(set(groups[check]))
    legacy, old_stats = fit_validation_ridge(y, u, vy, vu, 39)
    # Grouping changes only uncertainty-error estimation, not final mean fitting.
    assert stats == old_stats
    np.testing.assert_array_equal(model.coefficient, legacy.coefficient)
    np.testing.assert_array_equal(model.intercept, legacy.intercept)
    path = tmp_path / "ridge.npz"
    model.save(path)
    restored = GramSimpleGaussian.load(path)
    np.testing.assert_array_equal(restored.audit_arrays["fitting_chemistry_groups"], groups)
    assert restored.metadata == model.metadata


def test_heldout_group_targets_and_future_measurements_do_not_enter_own_prediction():
    y, u, vy, vu, groups = fixture()
    original, _ = fit_validation_ridge(y, u, vy, vu, 39, groups=groups)
    members = np.flatnonzero(groups == groups[0])
    y2, u2 = y.copy(), u.copy()
    y2[members, 1:] += 1000
    u2[members] += np.arange(9) * 100
    changed, _ = fit_validation_ridge(y2, u2, vy, vu, 39, groups=groups)
    np.testing.assert_array_equal(original.audit_arrays["oof_native_predictions"][members],
                                  changed.audit_arrays["oof_native_predictions"][members])


@pytest.mark.parametrize("groups", [[], ["one"]*24, [["a"]]*24,
                                     [None]+[str(i) for i in range(23)],
                                     [np.nan]+list(range(23))])
def test_bad_group_metadata_is_rejected(groups):
    y, u, vy, vu, _ = fixture()
    with pytest.raises(ValueError, match="groups"):
        fit_validation_ridge(y, u, vy, vu, 39, groups=groups)
