"""Read-only RxRx3 R2-to-R3 adapter for the existing EU module helpers.

Only approved prepared development profiles and completed R2 caches are read.
No mean, covariance, radial law, or representation is fitted here and no Monte
Carlo draws are made. Each cell is a complete dose-specific pool: its five
saved roles are mapped to local indices without recomputing their allocation.

``load_scope`` returns the full prepared ``data``, ``metadata``, R2 ``manifest``,
``observables`` and aggregate ``core`` dictionaries, plus source paths.
``load_cell`` returns dose-local ``data``, ``part``, ``means`` (STATE_REF),
``stats``, EU-compatible ``oldarrays`` and ``state``, and ``raw``, ``observed``,
``difference``, ``actual``. It also returns the exact saved query ``original``,
``source`` Path, ``cell_info``, ``global_rows`` and JSON-compatible ``audit``.
Biological annotation enrichment remains a separate metadata-only operation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .conditional_joint_error_experiment import observable_forward
from .eu_core_distribution import predict_eu_distribution
from .gram_oof_ridge import transform_target


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "runs/rxrx3_r2_completion_20260918_v1"
PREPARED = PROJECT / "data/rxrx3_r2_20260918/prepared_r2"
ROLE_KEYS = ("TRAIN", "VALIDATION", "REF_FIT", "DIST_CAL", "DEV_EVAL")


def _json(path):
    return json.loads(Path(path).read_text())


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _cell_index(cell_name):
    match = re.fullmatch(r"fold_(0|[1-9][0-9]*)", str(cell_name))
    if match is None:
        raise ValueError("Cell name must be fold_<saved cell index>")
    return int(match[1])


def list_cells(*, scope=None):
    """List the completed R2 dose-by-outer-fold cells without profile reads."""
    source = SOURCE if scope is None else Path(scope["source"])
    manifest = _json(source / "run_manifest.json") if scope is None else scope["manifest"]
    if _json(source / "status.json").get("state") != "COMPLETE":
        raise ValueError("Frozen RxRx3 R2 run is not complete")
    result = []
    for index, info in enumerate(manifest["cells"]):
        if info["cell"] != index:
            raise ValueError("R2 cell indices are not contiguous")
        name = f"fold_{index}"
        if not (source / name / "complete.json").is_file():
            raise FileNotFoundError(source / name / "complete.json")
        result.append(name)
    return result


def load_scope():
    """Open the already approved prepared DEV cache; never open raw exports."""
    manifest = _json(SOURCE / "run_manifest.json")
    metadata = _json(PREPARED / "metadata.json")
    if (manifest.get("protected_outcomes_used") is not False
            or manifest.get("confirmation_opened") is not False
            or metadata.get("confirmation_opened") is not False):
        raise ValueError("Expected the approved development-only R2 scope")
    data = _npz(PREPARED / "data.npz")
    observables = _npz(SOURCE / "observables.npz")
    core = _npz(SOURCE / "CORE_ORIGINAL.npz")
    for saved in (manifest, observables, core):
        np.testing.assert_array_equal(saved["ids"], data["ids"])
    for saved in (manifest, core):
        np.testing.assert_array_equal(saved["groups"], data["groups"])
    np.testing.assert_array_equal(core["layout"], data["layout"])
    np.testing.assert_array_equal(core["actual"], observables["actual"])
    if (len(data["ids"]) != manifest["n"] or manifest["n"] != 10410
            or len(np.unique(data["groups"])) != manifest["n_chemical_groups"]
            or len(manifest["parts"]) != len(manifest["cells"])):
        raise ValueError("Prepared population differs from the completed R2 scope")
    scope = dict(data=data, metadata=metadata, manifest=manifest,
                 observables=observables, core=core, source=SOURCE, prepared=PREPARED)
    if len(list_cells(scope=scope)) != 40:
        raise ValueError("Expected all 40 completed dose-specific R2 cells")
    return scope


def remap_cell_parts(data, saved_parts, cell_info):
    """Return dose-local data, exact local role indices and global row indices.

    Roles retain their saved order; the complete dose pool retains the global
    prepared-cache order. Whole chemical groups cannot cross any two roles.
    This helper is outcome-independent and does not generate a new split.
    """
    ids, groups, dose = (np.asarray(data[key]) for key in ("ids", "groups", "dose"))
    if (ids.ndim != 1 or groups.shape != ids.shape or dose.shape != ids.shape
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Aligned unique prepared identities are required")
    if set(saved_parts) != set(ROLE_KEYS):
        raise ValueError("Exactly the five original R2 roles are required")
    lookup = {str(value): i for i, value in enumerate(ids)}
    global_parts, used_ids, used_groups = {}, set(), set()
    for role in ROLE_KEYS:
        requested = list(saved_parts[role])
        if not requested or len(set(requested)) != len(requested):
            raise ValueError(role + ": nonempty unique saved identities required")
        if not set(requested) <= lookup.keys():
            raise ValueError(role + ": identity is outside the approved cache")
        rows = np.asarray([lookup[value] for value in requested], int)
        role_groups = set(groups[rows])
        if used_ids.intersection(requested) or used_groups.intersection(role_groups):
            raise ValueError("Original R2 roles must be whole-group disjoint")
        if (len(rows) != cell_info["counts"][role]
                or len(role_groups) != cell_info["group_counts"][role]):
            raise ValueError(role + ": saved row/group counts do not match")
        used_ids.update(requested)
        used_groups.update(role_groups)
        global_parts[role] = rows
    pool = np.sort(np.concatenate(list(global_parts.values())))
    expected = np.flatnonzero(dose == float(cell_info["dose_uM"]))
    if not np.array_equal(pool, expected):
        raise ValueError("Roles must cover exactly the complete declared dose pool")
    inverse = {int(row): i for i, row in enumerate(pool)}
    part = {role: np.asarray([inverse[int(row)] for row in rows], int)
            for role, rows in global_parts.items()}
    local = {}
    for key, value in data.items():
        array = np.asarray(value)
        if key == "feature_names":
            local[key] = array.copy()
        elif array.ndim > 0 and len(array) == len(ids):
            local[key] = array[pool].copy()
        else:
            raise ValueError("Unrecognized non-row-aligned prepared field: " + key)
    return local, part, pool


def _adapt_cell(scope, index, *, mean_cache, stats, fitted, residuals,
                original, mean_record, source):
    info = scope["manifest"]["cells"][index]
    data, part, pool = remap_cell_parts(scope["data"], scope["manifest"]["parts"][index], info)
    t, v, r, c, q = (part[role] for role in ROLE_KEYS)
    ids, groups = data["ids"], data["groups"]
    np.testing.assert_array_equal(mean_cache["row_indices"], pool)
    np.testing.assert_array_equal(mean_cache["ids"], ids)
    for role, key in (("TRAIN", "model_train_ids"), ("VALIDATION", "model_validation_ids"),
                      ("REF_FIT", "reference_fit_ids")):
        np.testing.assert_array_equal(mean_record[key], ids[part[role]])
    for prefix, rows in (("ref", r), ("cal", c), ("query", q)):
        np.testing.assert_array_equal(residuals[prefix + "_ids"], ids[rows])
        np.testing.assert_array_equal(residuals[prefix + "_groups"], groups[rows])
    np.testing.assert_array_equal(fitted["ref_inputs"]["ids"], ids[r])
    np.testing.assert_array_equal(fitted["ref_inputs"]["groups"], groups[r])
    np.testing.assert_array_equal(fitted["ref_inputs"]["X"], data["Y"][r, 0])
    np.testing.assert_array_equal(fitted["ref_inputs"]["chem"], data["chem"][r])
    np.testing.assert_array_equal(fitted["calibration_ids"], ids[c])
    np.testing.assert_array_equal(fitted["calibration_groups"], groups[c])
    train = fitted["model_training_identities"]
    np.testing.assert_array_equal(train["ids"], ids[np.r_[t, v]])
    np.testing.assert_array_equal(train["groups"], groups[np.r_[t, v]])
    raw = scope["observables"]["raw_geometry"][pool].copy()
    actual = scope["observables"]["actual"][pool].copy()
    target = transform_target(raw, stats)
    np.testing.assert_array_equal(mean_cache["actual_u"], target)
    means = np.asarray(mean_cache["STATE_REF"], float).copy()
    if means.shape != target.shape or not np.isfinite(means).all():
        raise ValueError("Expected finite cached STATE_REF means in the saved u frame")
    np.testing.assert_allclose(residuals["ref_residual"], target[r] - means[r], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(residuals["cal_residual"], target[c] - means[c], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(original["ids"], ids[q])
    np.testing.assert_array_equal(original["actual"], actual[q])
    np.testing.assert_array_equal(original["actual_u"], target[q])
    np.testing.assert_array_equal(original["mean_u"], residuals["query_mean"])
    mean_error = float(np.max(np.abs(means[q] - original["mean_u"])))
    np.testing.assert_allclose(means[q], original["mean_u"], atol=1e-12, rtol=1e-12)
    means[q] = original["mean_u"]  # Preserve saved query values bit-for-bit.
    global_query = pool[q]
    np.testing.assert_array_equal(scope["core"]["fold"][global_query], np.full(len(q), index))
    for key, value in original.items():
        if key in scope["core"]:
            np.testing.assert_array_equal(value, scope["core"][key][global_query])
    # Only this X-only application is replayed. CAL scatter already exists in
    # the frozen fit; CAL must not be passed through the disjoint QUERY API.
    with threadpool_limits(limits=1):
        prediction = predict_eu_distribution(fitted, dict(ids=ids[q], groups=groups[q],
            X=data["Y"][q, 0], chem=data["chem"][q], mean_u=means[q]))
    np.testing.assert_array_equal(original["scatter_u"], residuals["query_scatter"])
    for actual_value, saved in ((prediction["scatter_u"], original["scatter_u"]),
                               (prediction["base_scatter_u"], residuals["base_query_scatter"]),
                               (prediction["radial_variance_multiplier"], original["radial_variance_multiplier"]),
                               (prediction["covariance_u"], original["covariance_u"])):
        np.testing.assert_allclose(actual_value, saved, atol=1e-12, rtol=1e-12)
    oldarrays = {key: residuals[key].copy() for key in
                 ("ref_ids", "cal_ids", "query_ids", "ref_residual", "cal_residual")}
    for dest, key in (("base_scatter", "base_scatter"), ("ref_loo_weights", "reference_loo_weights"),
                      ("ref_loo_covariance", "reference_loo_covariance"),
                      ("cal_representative_indices", "calibration_representative_indices"),
                      ("cal_radii", "calibration_radii"), ("cal_scatter_u", "calibration_scatter_u")):
        oldarrays[dest] = np.asarray(fitted[key]).copy()
    oldarrays.update(query_base_scatter_u=residuals["base_query_scatter"].copy(),
        query_scatter_u=original["scatter_u"].copy(), radial_weights=prediction["radial_weights"].copy(),
        query_reference_weights=prediction["reference_weights"].copy(),
        radial_variance_multiplier=original["radial_variance_multiplier"].copy(),
        query_mean_u=original["mean_u"].copy())
    state = {key: copy.deepcopy(fitted[key]) for key in
             ("law", "amplitude_fit", "covariance_choice", "covariance_reference_bandwidth",
              "radial_reference_bandwidth", "coordinate_space")}
    reconstructed, observed, difference, _ = observable_forward(raw)
    np.testing.assert_allclose(reconstructed, actual, atol=1e-12, rtol=1e-12)
    audit = dict(cell=f"fold_{index}", outer_fold=info["outer_fold"], dose_uM=info["dose_uM"],
        role_counts={role: len(rows) for role, rows in part.items()},
        role_group_counts={role: len(np.unique(groups[rows])) for role, rows in part.items()},
        role_allocation="exact saved R2 run_manifest identities; locally reindexed",
        dose_pool_rows=len(pool), source=str(source), mean_recipe="cached STATE_REF",
        maximum_query_mean_replay_difference=mean_error, query_predictions_reused_exactly=True,
        query_scatter_reused_exactly=True, query_radial_multiplier_reused_exactly=True,
        cal_scatter_source="frozen CORE_distribution.joblib calibration_scatter_u",
        query_weights_source="X-only predict_eu_distribution on the frozen R2 law",
        mean_fits=0, distribution_fits=0, new_mc_draws=0,
        raw_images_opened=False, protected_measurements_opened=False, confirmation_opened=False)
    return dict(data=data, part=part, means=means, stats=stats, oldarrays=oldarrays, state=state,
        raw=raw, observed=observed, difference=difference, actual=actual,
        original=original, source=Path(source), cell_info=copy.deepcopy(info),
        global_rows=pool, audit=audit)


def load_cell(cell_name, *, scope=None):
    """Adapt one completed cell, replaying only its frozen query-law forward pass."""
    start = time.monotonic()
    scope = load_scope() if scope is None else scope
    index = _cell_index(cell_name)
    if index >= len(scope["manifest"]["cells"]):
        raise ValueError("Cell is outside the frozen R2 manifest")
    source = Path(scope["source"]) / cell_name
    if not (source / "complete.json").is_file():
        raise FileNotFoundError(source / "complete.json")
    result = _adapt_cell(scope, index,
        mean_cache=_npz(source / "mean_predictions.npz"),
        stats=_json(source / "mean/preprocessing.json"),
        fitted=joblib.load(source / "CORE_distribution.joblib"),
        residuals=_npz(source / "CORE_residuals.npz"),
        original=_npz(source / "AMP_EMP_LOCAL.npz"),
        mean_record=_json(source / "mean/complete.json"), source=source)
    result["audit"]["load_wall_seconds"] = time.monotonic() - start
    return result
