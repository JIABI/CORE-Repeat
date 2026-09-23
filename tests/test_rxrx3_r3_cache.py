"""Saved-role, isolation and exact-query tests; no model training or data run."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import opal2.rxrx3_r3_cache as cache
from opal2.conditional_joint_error_experiment import observable_forward


def small_scope():
    n = 7
    data = dict(ids=np.array([f"id{i}" for i in range(n)]),
        groups=np.array([f"g{i}" for i in range(n)]),
        dose=np.array([1., 2., 1., 1., 2., 1., 1.]),
        Y=np.arange(n * 4 * 7, dtype=float).reshape(n, 4, 7) + 1,
        chem=np.zeros((n, 513)), chem_mask=np.ones(n, bool),
        layout=np.array(["a"] * n), feature_names=np.array([f"f{i}" for i in range(n)]),
        target=np.arange(n * 2).reshape(n, 2), target_mask=np.ones(n, bool))
    pool = np.array([0, 2, 3, 5, 6])
    parts = {role: data["ids"][[row]].tolist() for role, row in zip(cache.ROLE_KEYS, pool)}
    info = dict(cell=0, outer_fold=0, dose_uM=1.,
                counts={key: 1 for key in parts}, group_counts={key: 1 for key in parts})
    raw = np.arange(n * 9, dtype=float).reshape(n, 9) * .001
    actual = observable_forward(raw)[0]
    return dict(data=data, manifest=dict(parts=[parts], cells=[info]),
        observables=dict(raw_geometry=raw, actual=actual), source=Path("/frozen")), pool


def adapted_fixture(monkeypatch):
    scope, pool = small_scope()
    data, part, _ = cache.remap_cell_parts(scope["data"], scope["manifest"]["parts"][0],
                                          scope["manifest"]["cells"][0])
    target = scope["observables"]["raw_geometry"][pool]
    mean = target * .5
    r, c, q = (part[key] for key in ("REF_FIT", "DIST_CAL", "DEV_EVAL"))
    scatter = np.eye(9)[None]
    original = dict(ids=data["ids"][q], actual=scope["observables"]["actual"][pool[q]],
        actual_u=target[q], mean_u=mean[q].copy(), scatter_u=scatter,
        radial_variance_multiplier=np.array([1.5]), covariance_u=scatter * 1.5,
        predicted=np.array([.125]), p_null=np.array([.25]))
    core = {key: np.zeros((len(scope["data"]["ids"]), *value.shape[1:]), dtype=value.dtype)
            for key, value in original.items()}
    for key, value in original.items():
        core[key][pool[q]] = value
    core["fold"] = np.zeros(len(scope["data"]["ids"]), int)
    scope["core"] = core
    residuals = dict(ref_residual=target[r]-mean[r], cal_residual=target[c]-mean[c],
                     query_mean=mean[q], query_scatter=scatter, base_query_scatter=scatter)
    for prefix, rows in (("ref", r), ("cal", c), ("query", q)):
        residuals[prefix+"_ids"] = data["ids"][rows]
        residuals[prefix+"_groups"] = data["groups"][rows]
    fitted = dict(ref_inputs=dict(ids=data["ids"][r], groups=data["groups"][r],
            X=data["Y"][r, 0], chem=data["chem"][r]),
        calibration_ids=data["ids"][c], calibration_groups=data["groups"][c],
        model_training_identities=dict(ids=data["ids"][:2], groups=data["groups"][:2]),
        base_scatter=np.eye(9), reference_loo_weights=np.ones((1, 1)),
        reference_loo_covariance=scatter.copy(), calibration_representative_indices=np.array([0]),
        calibration_radii=np.array([2.]), calibration_scatter_u=scatter * 2,
        law=dict(calibration_n=1), amplitude_fit={}, covariance_choice={},
        covariance_reference_bandwidth=.3, radial_reference_bandwidth=.4, coordinate_space="u")
    calls = []

    def forward(fit, inputs):
        calls.append(copy.deepcopy(inputs))
        assert set(inputs) == {"ids", "groups", "X", "chem", "mean_u"}
        assert not set(inputs["ids"]) & set(fit["calibration_ids"])
        return dict(scatter_u=scatter.copy(), base_scatter_u=scatter.copy(),
            radial_variance_multiplier=np.array([1.5]), covariance_u=scatter * 1.5,
            radial_weights=np.ones((1, 1)), reference_weights=np.ones((1, 1)))

    monkeypatch.setattr(cache, "predict_eu_distribution", forward)
    arguments = dict(mean_cache=dict(row_indices=pool, ids=data["ids"],
                                    actual_u=target.copy(), STATE_REF=mean.copy()),
        stats=dict(u_center=np.zeros(9), u_scale=np.ones(9)), fitted=fitted,
        residuals=residuals, original=original,
        mean_record=dict(model_train_ids=data["ids"][:1], model_validation_ids=data["ids"][1:2],
                         reference_fit_ids=data["ids"][r]), source=Path("/frozen/fold_0"))
    return scope, arguments, calls


def test_saved_parts_slice_complete_dose_pool_and_biology_arrays():
    scope, expected = small_scope()
    data, parts, pool = cache.remap_cell_parts(scope["data"], scope["manifest"]["parts"][0],
                                             scope["manifest"]["cells"][0])
    np.testing.assert_array_equal(pool, expected)
    for role, rows in parts.items():
        np.testing.assert_array_equal(data["ids"][rows], scope["manifest"]["parts"][0][role])
    np.testing.assert_array_equal(data["target"], scope["data"]["target"][pool])
    np.testing.assert_array_equal(data["feature_names"], scope["data"]["feature_names"])
    assert data["Y"].shape == (5, 4, 7)


def test_saved_roles_survive_global_cache_reordering():
    scope, _ = small_scope()
    permutation = np.array([6, 1, 3, 5, 0, 4, 2])
    data = {key: value.copy() if key == "feature_names" else value[permutation]
            for key, value in scope["data"].items()}
    local, parts, _ = cache.remap_cell_parts(data, scope["manifest"]["parts"][0],
                                           scope["manifest"]["cells"][0])
    for role, rows in parts.items():
        np.testing.assert_array_equal(local["ids"][rows], scope["manifest"]["parts"][0][role])


@pytest.mark.parametrize("fault", ["crossing_group", "missing_dose_row", "other_dose", "duplicate", "count"])
def test_invalid_role_assignments_are_rejected(fault):
    scope, _ = small_scope()
    data, parts, info = scope["data"], scope["manifest"]["parts"][0], scope["manifest"]["cells"][0]
    if fault == "crossing_group":
        data["groups"][2] = data["groups"][0]
    elif fault == "missing_dose_row":
        data["dose"][1] = 1.
    elif fault == "other_dose":
        parts["TRAIN"] = ["id1"]
    elif fault == "duplicate":
        parts["TRAIN"] *= 2
    else:
        info["counts"]["TRAIN"] = 2
    with pytest.raises(ValueError):
        cache.remap_cell_parts(data, parts, info)


def test_adapter_preserves_query_arrays_and_reuses_saved_cal_scatter(monkeypatch):
    scope, arguments, calls = adapted_fixture(monkeypatch)
    result = cache._adapt_cell(scope, 0, **arguments)
    q = result["part"]["DEV_EVAL"]
    np.testing.assert_array_equal(result["means"][q], arguments["original"]["mean_u"])
    np.testing.assert_array_equal(result["oldarrays"]["query_scatter_u"], arguments["original"]["scatter_u"])
    np.testing.assert_array_equal(result["oldarrays"]["cal_scatter_u"], arguments["fitted"]["calibration_scatter_u"])
    np.testing.assert_array_equal(result["original"]["p_null"], arguments["original"]["p_null"])
    assert len(calls) == 1
    assert result["audit"]["mean_fits"] == result["audit"]["distribution_fits"] == 0
    assert result["audit"]["new_mc_draws"] == 0
    json.dumps(result["audit"])


def test_approximately_equal_mean_is_restored_to_saved_exact_query(monkeypatch):
    scope, arguments, _ = adapted_fixture(monkeypatch)
    arguments["mean_cache"]["STATE_REF"][-1] += 1e-14
    result = cache._adapt_cell(scope, 0, **arguments)
    np.testing.assert_array_equal(result["means"][-1], arguments["original"]["mean_u"][0])
    assert result["audit"]["maximum_query_mean_replay_difference"] > 0


@pytest.mark.parametrize("fault", ["mean", "aggregate_probability", "ref_group", "row_index"])
def test_inconsistent_frozen_caches_fail_before_forward(monkeypatch, fault):
    scope, arguments, calls = adapted_fixture(monkeypatch)
    if fault == "mean":
        arguments["mean_cache"]["STATE_REF"][-1] += .01
    elif fault == "aggregate_probability":
        scope["core"]["p_null"][-1] += .01
    elif fault == "ref_group":
        arguments["residuals"]["ref_groups"] = np.array(["wrong"])
    else:
        arguments["mean_cache"]["row_indices"][0] = 1
    with pytest.raises(AssertionError):
        cache._adapt_cell(scope, 0, **arguments)
    assert not calls


def test_cell_name_rejects_paths_and_noncanonical_indices():
    assert cache._cell_index("fold_39") == 39
    for value in ("../fold_0", "fold_01", "fold_-1", "0", "fold_0/mean"):
        with pytest.raises(ValueError):
            cache._cell_index(value)
