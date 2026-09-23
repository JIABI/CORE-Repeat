"""Describe extreme coordinates in an already-open measurement export.

This module performs no fitting, exclusion, clipping, file discovery, or writes.
Exported Cell Painting profiles are not raw pixels. In the source_5 export,
``Y`` already has the declared DMSO-based endpoint normalization; the fitted
baseline's affine coordinates add a second, training-only transform.
"""
from __future__ import annotations

import numpy as np


def _number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _indices(values, ids):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Split members must be one-dimensional")
    if array.dtype.kind in "USO":
        lookup = {unit: index for index, unit in enumerate(ids.tolist())}
        try:
            array = np.asarray([lookup[str(unit)] for unit in array], dtype=int)
        except KeyError as error:
            raise ValueError("Split contains an object outside the supplied export") from error
    if array.dtype.kind not in "iu" or len(np.unique(array)) != len(array):
        raise ValueError("Split indices must be unique integers or known IDs")
    if np.any(array < 0) or np.any(array >= len(ids)):
        raise ValueError("Split indices are outside the supplied export")
    return array.astype(int, copy=False)


def audit_opened_data(dataset, baseline, split, *,
                      focus_ids=("JCP2022_107356", "JCP2022_011329"), top_coordinates=10):
    """Return JSON-serializable diagnostics without changing data or the model.

    ``split`` must partition the supplied export and identify ``train``. Members
    may be row indices or object IDs. Threshold counts are descriptive and are
    never interpreted as segmentation failures, QC rules, or exclusions.
    """
    y = np.asarray(dataset.Y, dtype=np.float64)
    ids = np.asarray(dataset.ids, dtype=str)
    names = np.asarray(dataset.feature_names, dtype=str)
    if y.ndim != 3 or ids.shape != (len(y),) or names.shape != (y.shape[-1],):
        raise ValueError("Exported values, object IDs, and feature names disagree")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Object IDs must be unique")
    if not isinstance(top_coordinates, int) or top_coordinates < 1:
        raise ValueError("top_coordinates must be a positive integer")
    center, scale = np.asarray(baseline.center), np.asarray(baseline.scale)
    if (center.shape != names.shape or scale.shape != names.shape
            or not np.isfinite(center).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("Baseline affine parameters must cover every coordinate")
    parts = {str(key): _indices(value, ids) for key, value in split.items()}
    if "train" not in parts or not len(parts["train"]):
        raise ValueError("Audit needs the original nonempty training allocation")
    members = np.concatenate(list(parts.values()))
    if len(members) != len(ids) or len(np.unique(members)) != len(ids):
        raise ValueError("Splits must partition all supplied objects without overlap")
    observed = np.asarray(dataset.observed_mask, dtype=bool)
    if observed.shape != y.shape[:2]:
        raise ValueError("Observed mask must identify supplied wells")
    if not np.isfinite(y[observed]).all():
        raise ValueError("Observed wells contain nonfinite coordinates")
    # Neither this array nor the exported Y is clipped. Missing wells remain
    # missing and are counted separately instead of being silently imputed.
    affine = (y - center) / scale
    views = {"exported_endpoint_Y": y, "L_train_affine_Y": affine}
    roles = list(dataset.metadata.get("roles", [f"well_{j}" for j in range(y.shape[1])]))
    if len(roles) != y.shape[1]:
        raise ValueError("Declared role names do not match the supplied wells")
    labels = np.empty(len(y), dtype=object)
    for label, ix in parts.items():
        labels[ix] = label
    train_values = y[parts["train"]][observed[parts["train"]]]
    if len(train_values) < 2:
        raise ValueError("At least two observed training wells are needed")
    train_sd = np.std(train_values, axis=0, ddof=0)
    train_mean = np.mean(train_values, axis=0)
    train_max = np.max(np.abs(train_values), axis=0)
    train_affine_max = np.max(np.abs((train_values - center) / scale), axis=0)

    def coordinate(row, feature):
        return {"feature_index": int(feature), "feature_name": str(names[feature]),
                "exported_Y_by_role": [_number(x) if present else None
                                       for x, present in zip(y[row, :, feature], observed[row])],
                "L_affine_Y_by_role": [_number(x) if present else None
                                       for x, present in zip(affine[row, :, feature], observed[row])],
                "L_center": float(center[feature]), "L_scale": float(scale[feature]),
                "train_exported_mean": float(train_mean[feature]),
                "train_exported_sd_ddof0": float(train_sd[feature]),
                "train_exported_max_abs": float(train_max[feature]),
                "train_L_affine_max_abs": float(train_affine_max[feature]),
                "scale_equals_train_sd": bool(np.isclose(scale[feature], train_sd[feature], rtol=1e-9, atol=1e-12))}

    def summarize(array, ix):
        value, mask = array[ix], observed[ix]
        absolute = np.where(mask[..., None], np.abs(value), -np.inf)
        if not mask.any():
            maximum = None
        else:
            row, well, feature = np.unravel_index(np.argmax(absolute), absolute.shape)
            maximum = {"absolute_value": float(absolute[row, well, feature]),
                       "signed_value": float(value[row, well, feature]),
                       "compound_id": str(ids[ix[row]]), "role": roles[well],
                       "feature_index": int(feature), "feature_name": str(names[feature])}
        thresholds = []
        for threshold in (100, 200, 1000):
            exceed = absolute > threshold
            thresholds.append({"absolute_threshold": threshold,
                               "coordinate_entries": int(exceed.sum()),
                               "physical_wells": int(exceed.any(-1).sum()),
                               "objects": int(exceed.any(axis=(1, 2)).sum())})
        return {"objects": len(ix), "observed_wells": int(mask.sum()),
                "maximum": maximum, "threshold_counts": thresholds}

    per_object = []
    for row, unit in enumerate(ids):
        entry = {"compound_id": str(unit), "partition": str(labels[row]),
                 "observed_wells": int(observed[row].sum())}
        for name, array in views.items():
            summary = summarize(array, np.asarray([row]))
            entry[name] = {"maximum": summary["maximum"],
                           "threshold_counts": summary["threshold_counts"]}
        per_object.append(entry)
    focus = []
    for unit in focus_ids:
        found = np.flatnonzero(ids == unit)
        if not len(found):
            focus.append({"compound_id": str(unit), "present_in_supplied_export": False})
            continue
        row = int(found[0])
        entry = {"compound_id": str(unit), "present_in_supplied_export": True,
                 "partition": str(labels[row]), "roles": roles,
                 "well_ids": np.asarray(dataset.well_ids[row], dtype=str).tolist(),
                 "observed_mask": observed[row].tolist()}
        for name, array in views.items():
            maximum = np.max(np.where(observed[row, :, None], np.abs(array[row]), -np.inf), axis=0)
            order = np.argsort(-maximum, kind="stable")[:min(top_coordinates, len(names))]
            entry[f"top_coordinates_by_{name}"] = [coordinate(row, feature) for feature in order]
        focus.append(entry)
    metadata = dataset.metadata
    legacy = metadata.get("legacy_info", {})
    provenance = {key: metadata.get(key) for key in (
        "dataset", "space", "scope", "outcome_normalization",
        "reference_profile_coordinates", "cell_counts_available",
        "physical_identity_verified", "roles_establish_chronology")}
    provenance["legacy_common_scale"] = legacy.get("common_scale")
    provenance["legacy_plate_center_controls"] = legacy.get("plate_center_controls")
    provenance["legacy_all_role_values_finite"] = legacy.get("all_role_values_finite")
    provenance["n_cells_records_present"] = int(np.asarray(dataset.n_cells_mask, dtype=bool).sum())
    provenance.update(
        raw_images_inspected=False, segmentation_masks_inspected=False,
        image_quality_or_segmentation_failure_labels_available=False,
        original_per_feature_DMSO_MAD_values_available_in_this_audit=False,
        input_files_opened_by_this_function=[],
        caution="Finite/observed masks denote available measurements, not passed biological or image QC. Metadata describes provenance but does not verify original segmentation or an upstream near-zero MAD.")
    return {"scope": "Descriptive audit of supplied already-open objects only",
            "shape": list(y.shape), "roles": roles,
            "coordinate_units": {
                "exported_endpoint_Y": "Previously normalized Cell Painting endpoint coordinates, not pixels or raw CellProfiler measurements",
                "L_train_affine_Y": "(exported_endpoint_Y - original TRAIN mean) / original TRAIN population SD; configured scale fallback retained"},
            "threshold_meaning": "Strict absolute-value counts for diagnosis only; not QC cutoffs, exclusions, or clipping instructions",
            "all_objects": {name: summarize(array, np.arange(len(y))) for name, array in views.items()},
            "partitions": {label: {name: summarize(array, ix) for name, array in views.items()}
                           for label, ix in parts.items()},
            "per_object": per_object, "focus_objects": focus,
            "affine_scale_min": float(scale.min()), "affine_scale_max": float(scale.max()),
            "L_center_max_abs_discrepancy_from_train_mean": float(np.abs(center - train_mean).max()),
            "L_scale_max_abs_discrepancy_from_train_sd": float(np.abs(scale - train_sd).max()),
            "provenance_evidence": provenance,
            "objects_excluded": [], "coordinates_clipped": False, "model_refitted": False}
