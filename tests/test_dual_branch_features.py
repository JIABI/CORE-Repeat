"""Reference permission, native geometry, and calibration isolation checks."""
from copy import deepcopy
import inspect

import numpy as np
import pytest

import opal2.dual_branch_features as features
from opal2.joint_contrast_scale import contrast_projector


def fixture_inputs():
    rng = np.random.default_rng(1709)
    y = rng.normal(size=(9, 4, 11))
    target = np.eye(3)[np.array([0, 1, 0, 2, 2, 2, 2, 2, 2])]
    moa = np.eye(3)[np.array([1, 0, 2, 2, 2, 2, 2, 2, 2])]
    data = dict(Y=y, groups=np.array([f"g{i}" for i in range(9)]),
                target=target, moa=moa, target_mask=np.ones(9, bool), moa_mask=np.ones(9, bool))
    meta = dict(units=[dict(cell_line="A549", exposure_hours_protocol_nominal=48.,
                            actual_dose_uM=10.) for _ in range(9)])
    query, donors = np.array([0, 1]), np.arange(2, 9)
    mean = rng.normal(size=(2, 9))*.2
    factor = rng.normal(size=(2, 9, 9))*.08+np.eye(9)[None]
    cov = factor@factor.transpose(0, 2, 1)
    residual = rng.normal(size=(len(donors), 9))
    return data, meta, query, donors, mean, cov, residual


def run_fixture(items, **kwargs):
    return features.biology_features(*items, **kwargs)


def test_query_future_poison_does_not_change_any_feature():
    args = fixture_inputs()
    first = run_fixture(args)
    changed = deepcopy(args)
    changed[0]["Y"][:, 1:] = np.nan
    second = run_fixture(changed)
    assert np.array_equal(first["values"], second["values"])
    assert np.array_equal(first["support"], second["support"])
    assert first["names"] == second["names"] == features.NAMES


def test_unpermitted_reference_errors_do_not_enter_means_or_se():
    args = fixture_inputs()
    permit = np.ones((2, 7), bool)
    permit[:, -2:] = False
    first = run_fixture(args, allowed=permit)
    changed = deepcopy(args)
    changed[-1][-2:] *= 1e8
    second = run_fixture(changed, allowed=permit)
    np.testing.assert_array_equal(first["values"], second["values"])


@pytest.mark.parametrize("reason", ["context", "same_group"])
def test_invalid_context_or_same_group_donor_not_used_for_single_neighbor_se(reason):
    args = list(fixture_inputs())
    if reason == "context":
        args[1]["units"][-1]["exposure_hours_protocol_nominal"] = 24.
    else:
        args[0]["groups"][-1] = args[0]["groups"][0]
    first = run_fixture(args)
    changed = deepcopy(args)
    changed[-1][-1] *= 1e8
    second = run_fixture(changed)
    # Query zero has exactly one legal target donor; excluded donors must not
    # affect the pool variance that keeps its uncertainty estimate nonzero.
    np.testing.assert_array_equal(first["values"][0], second["values"][0])


def test_target_and_moa_permission_and_zero_support_are_exact():
    args = list(fixture_inputs())
    args[1]["units"][0]["exposure_hours_protocol_nominal"] = 24.
    out = run_fixture(args)
    assert np.array_equal(out["values"], np.zeros((2, 24)))
    assert not out["support"].any()
    denied = run_fixture(fixture_inputs(), allowed=np.zeros((2, 7), bool))
    assert np.array_equal(denied["values"], np.zeros((2, 24)))
    assert not denied["support"].any()


def test_missing_moa_vocabulary_is_zero_not_copied_from_target():
    args = list(fixture_inputs())
    args[0]['moa'] = np.zeros((9, 0))
    args[0]['moa_mask'] = np.zeros(9, bool)
    result = run_fixture(args)
    assert result['support_by_relation'][:,0].any()
    assert not result['support_by_relation'][:,1].any()
    np.testing.assert_array_equal(result['values'][:,12:], np.zeros((2,12)))


def test_raw_donor_error_is_whitened_by_query_not_donor_covariance():
    args = fixture_inputs()
    _, _, _, _, mean, cov, residual = args
    out = run_fixture(args)
    dec = contrast_projector(mean, np.ones(9), cov)
    white = np.linalg.solve(dec["factor"][0], residual[0])
    pair = dec["projector"][0]@white
    expected = [np.log1p(pair@pair/3), np.log1p((white-pair)@(white-pair)/6)]
    np.testing.assert_allclose(out["values"][0, 5:7], expected, atol=1e-14)
    assert out["values"][0, 7] > 0
    assert out["values"][0, 8] > 0
    assert out["values"][0, 1] == pytest.approx(np.log(2.))
    assert out["values"][0, 3] == pytest.approx(np.log(2.))
    changed = list(deepcopy(args))
    changed[5][0] *= 4
    larger_query_scatter = run_fixture(changed)
    np.testing.assert_allclose(np.expm1(larger_query_scatter["values"][0, 5:7]),
                               np.expm1(out["values"][0, 5:7])/4, atol=1e-14)


def test_reference_order_invariance():
    args = fixture_inputs()
    first = run_fixture(args)
    changed = list(deepcopy(args))
    order = np.array([6, 2, 1, 5, 0, 4, 3])
    changed[3], changed[-1] = changed[3][order], changed[-1][order]
    np.testing.assert_allclose(first["values"], run_fixture(changed)["values"], atol=1e-14)


def test_apply_increment_exact_off_preserves_other_rows_and_spd():
    _, _, _, _, mean, cov, _ = fixture_inputs()
    original_mean, original_cov = mean.copy(), cov.copy()
    delta = np.array([[0., 0.], [np.log(2.), np.log(.5)]])
    out = features.apply_increment(mean, np.ones(9), cov, delta)
    assert np.array_equal(out[0], cov[0])
    assert np.array_equal(mean, original_mean)
    assert np.array_equal(cov, original_cov)
    assert np.linalg.eigvalsh(out).min() > 0
    np.testing.assert_allclose(out, out.transpose(0, 2, 1), atol=1e-14)
    dec = contrast_projector(mean, np.ones(9), cov)
    inv = np.linalg.inv(dec["factor"][1])
    white = inv@out[1]@inv.T
    np.testing.assert_allclose(np.linalg.eigvalsh(white), [.5]*6+[2.]*3, atol=1e-13)
    zero = features.apply_increment(mean, np.ones(9), cov, np.zeros((2, 2)))
    assert np.array_equal(zero, cov)


def calibration_arrays():
    rng = np.random.default_rng(8)
    n = 10
    mean = rng.normal(size=(n, 9))*.1
    scatter = np.repeat(np.eye(9)[None], n, axis=0)
    direction = rng.normal(size=(n, 9))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    residual = direction*np.linspace(1., 4., n)[:, None]
    logamp = np.linspace(-1., 1., n)
    groups = np.repeat(np.arange(5), 2)
    return mean, np.ones(9), scatter, residual, logamp, groups, 1., np.zeros((n, 2))


def test_calibration_radial_fit_leaves_entire_target_group_out(monkeypatch):
    args = calibration_arrays()
    calls = []
    original_fit = features.fit_radial

    def spy(radii, *a, **kw):
        calls.append(np.asarray(radii).copy())
        return original_fit(radii, *a, **kw)

    monkeypatch.setattr(features, "fit_radial", spy)
    out = features.select_strength(*args)
    assert out["strength"] == 0.
    radii = np.linalg.norm(args[3], axis=1)
    for i, used in enumerate(calls):
        np.testing.assert_allclose(used, radii[args[5] != args[5][i]], atol=1e-14)
    original_calls = deepcopy(calls)
    calls.clear()
    altered = list(deepcopy(args))
    altered[3][altered[5] == 0] *= 2
    features.select_strength(*altered)
    np.testing.assert_array_equal(calls[0], original_calls[0])
    np.testing.assert_array_equal(calls[1], original_calls[1])
    assert not np.array_equal(calls[2], original_calls[2])


def test_insufficient_calibration_groups_not_confused_with_row_count():
    args = list(calibration_arrays())
    args[5] = np.repeat([0, 1], 5)
    out = features.select_strength(*args)
    assert out["strength"] == 0.
    assert out["reason"] == "insufficient independent calibration groups"
    assert out["scores"] == []


def test_rx_radial_calibration_uses_id_min_representatives(monkeypatch):
    args = calibration_arrays()
    ids = np.array(['b', 'a', 'd', 'c', 'f', 'e', 'h', 'g', 'j', 'i'])
    calls = []
    fit = features.fit_radial
    def spy(radii, *a, **kw):
        calls.append(np.asarray(radii).copy())
        return fit(radii, *a, **kw)
    monkeypatch.setattr(features, 'fit_radial', spy)
    features.select_strength(*args, representative_ids=ids)
    radii = np.linalg.norm(args[3], axis=1)
    chosen = np.arange(1, 10, 2)
    for i, used in enumerate(calls):
        np.testing.assert_allclose(used, radii[chosen[args[5][chosen] != args[5][i]]])
    calls.clear()
    features.calibration_frame_covariance(args[2], args[3], args[4], args[5], args[6],
                                         representative_ids=ids)
    for i, used in enumerate(calls):
        np.testing.assert_allclose(used, radii[chosen[args[5][chosen] != args[5][i]]])


def test_calibration_frame_covariance_uses_only_other_calibration_groups():
    _, _, scatter, residual, logamp, groups, bandwidth, _ = calibration_arrays()
    original = features.calibration_frame_covariance(scatter, residual, logamp, groups, bandwidth)
    altered = residual.copy()
    altered[0] *= 20
    changed = features.calibration_frame_covariance(scatter, altered, logamp, groups, bandwidth)
    # The complete target group is excluded, not just the target row.
    np.testing.assert_array_equal(original[groups == groups[0]], changed[groups == groups[0]])
    assert not np.array_equal(original[groups != groups[0]], changed[groups != groups[0]])
    assert np.linalg.eigvalsh(original).min() > 0
    assert list(inspect.signature(features.calibration_frame_covariance).parameters) == [
        "scatter", "residual", "logamp", "groups", "bandwidth", "representative_ids"]
    assert inspect.signature(features.calibration_frame_covariance).parameters[
        'representative_ids'].kind == inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        features.calibration_frame_covariance(scatter, residual, logamp, groups, bandwidth,
                                               query_residual=np.ones((2, 9)))
