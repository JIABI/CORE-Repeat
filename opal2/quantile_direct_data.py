"""Read the four saved R2 development experiments for conditional direct models.

No model is loaded or refitted, no split is regenerated, and no R4 path is used.
The new comparator receives the original HistGB ACCESS_MATCHED inputs and role
order: TRAIN plus REF_FIT for fitting, VALIDATION for model selection, DIST_CAL
for distribution calibration, and DEV_EVAL for evaluation. The preprocessing
moments remain those of each saved backbone's TRAIN population.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .gram_oof_ridge import transform_input


DATASETS = ("EU", "JUMP", "LINCS", "RxRx3")
_SPEC = {
    "EU": ("runs/r2_core_comparison_20260917_v1",
           "reports/eu_core_development_20260917_v1/prepared_data_cc904/data.npz", 5),
    "JUMP": ("runs/jump_r2_completion_20260918_v1",
             "data/source5_primary_fullcontrols/measurements.npz", 5),
    "LINCS": ("runs/lincs_r2_completion_20260918_v1",
              "data/lincs_pilot1_biology_20260915/data.npz", 10),
    "RxRx3": ("runs/rxrx3_r2_completion_20260918_v1",
              "data/rxrx3_r2_20260918/prepared_r2/data.npz", 40),
}
_KEYS = ("ids", "groups", "layout", "fold", "cell", "actual", "predicted",
         "p_null", "crps", "gamma_coverage_by_level", "gamma_width_by_level",
         "selected_lambda_0.2", "selected_lambda_0")


def _json(path):
    return json.loads(Path(path).read_text())


def _dataset_name(dataset):
    names = {name.lower(): name for name in DATASETS}
    try:
        return names[str(dataset).lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown development dataset {dataset!r}") from exc


@lru_cache(maxsize=4)
def _saved(project, dataset):
    """Small outcome/identity caches; large first-well input cache is separate."""
    project = Path(project)
    run = project / _SPEC[dataset][0]
    manifest = _json(run / "run_manifest.json")
    original = {}
    for name, filename in (("core", "CORE_ORIGINAL.npz"),
                           ("histgb", "DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL.npz")):
        with np.load(run / filename, allow_pickle=False) as archive:
            original[name] = {key: archive[key].copy() for key in _KEYS if key in archive}
    core, histgb = original["core"], original["histgb"]
    for key in ("ids", "groups", "layout", "actual", "fold"):
        np.testing.assert_array_equal(core[key], histgb[key])
    ids = core["ids"].astype(str)
    lookup = {oid: index for index, oid in enumerate(ids)}
    if len(lookup) != len(ids) or not np.isfinite(core["actual"]).all():
        raise ValueError("Saved development identities/outcomes are not complete")
    return dict(run=run, manifest=manifest, original=original, ids=ids, lookup=lookup)


def _cell_record(project, dataset, cell_index):
    dataset = _dataset_name(dataset)
    project = str(Path(project).resolve())
    if not isinstance(cell_index, (int, np.integer)) or not 0 <= cell_index < _SPEC[dataset][2]:
        raise ValueError("Cell index is outside the saved experiment")
    saved = _saved(project, dataset)
    record = dict(dataset=dataset, cell_index=int(cell_index))
    if dataset == "LINCS":
        info = saved["manifest"]["cells"][cell_index]
        fold, half = info["fold"], info["half"]
        folder = saved["run"] / f"cell_{fold}_{half}"
        metadata_path = folder / "DIRECT_ACCESS_MATCHED_HISTGB_calibration.json"
        preprocessing = Path(project) / "runs/lincs_state_biology_20260916_v1/folds" / f"fold_{fold}/preprocessing.json"
        record.update(fold=fold, outer_fold=fold, half=half,
                      label=f"LINCS/fold_{fold}/half_{half}")
    else:
        folder = saved["run"] / f"fold_{cell_index}"
        metadata_path = folder / "DIRECT_ACCESS_MATCHED_HISTGB.json"
        if dataset == "EU":
            preprocessing = Path(project) / "runs/eu_core_cc904_20260917_v1" / f"fold_{cell_index}/mean/preprocessing.json"
        else:
            preprocessing = folder / "mean/preprocessing.json"
        record.update(fold=int(cell_index), outer_fold=int(cell_index),
                      label=f"{dataset}/fold_{cell_index}")
        if dataset == "RxRx3":
            info = saved["manifest"]["cells"][cell_index]
            record.update(outer_fold=info["outer_fold"], dose_uM=info["dose_uM"],
                          label=f"RxRx3/fold_{info['outer_fold']}/dose_{info['dose_uM']:g}")
    return saved, record, folder, metadata_path, preprocessing


def _roles(saved, dataset, cell_index, metadata_path):
    meta = _json(metadata_path)
    supplied = {"train": meta["train_ids"],
                "valid": meta.get("valid_ids", meta.get("validation_ids")),
                "cal": meta["cal_ids"], "query": meta["query_ids"]}
    if dataset == "LINCS":
        old_parts = saved["manifest"]["cells"][cell_index]["ids"]
    elif dataset in ("JUMP", "RxRx3"):
        old_parts = saved["manifest"]["parts"][cell_index]
    else:
        train_meta = _json(metadata_path.with_name("DIRECT_TRAIN_MATCHED_HISTGB.json"))
        train_ids = train_meta["train_ids"]
        old_parts = {"TRAIN": train_ids, "REF_FIT": supplied["train"][len(train_ids):],
                     "VALIDATION": train_meta["valid_ids"], "DIST_CAL": train_meta["cal_ids"],
                     "DEV_EVAL": train_meta["query_ids"]}
    if old_parts["TRAIN"] + old_parts["REF_FIT"] != supplied["train"]:
        raise ValueError("ACCESS_MATCHED training order differs from TRAIN plus REF_FIT")
    for role, original in (("valid", "VALIDATION"), ("cal", "DIST_CAL"), ("query", "DEV_EVAL")):
        if supplied[role] != old_parts[original]:
            raise ValueError(f"Saved direct role order differs for {role}")
    supplied.update(model_train=old_parts["TRAIN"], ref=old_parts["REF_FIT"])
    rows = {}
    for role, ids in supplied.items():
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Empty or duplicate identities in {role}")
        rows[role] = np.asarray([saved["lookup"][str(oid)] for oid in ids], dtype=int)
    groups = saved["original"]["core"]["groups"]
    five = ("model_train", "ref", "valid", "cal", "query")
    for i, left in enumerate(five):
        for right in five[:i]:
            if set(groups[rows[left]]) & set(groups[rows[right]]):
                raise ValueError(f"Chemical group crosses {left} and {right}")
    return rows, meta


def list_cells(project):
    """List and verify all 60 original deployment cells without loading profiles."""
    project = str(Path(project).resolve())
    result = []
    rx_role_by_fold = {}
    for dataset in DATASETS:
        saved = _saved(project, dataset)
        queried = np.zeros(len(saved["ids"]), dtype=int)
        for index in range(_SPEC[dataset][2]):
            _, record, _, metadata_path, _ = _cell_record(project, dataset, index)
            rows, metadata = _roles(saved, dataset, index, metadata_path)
            q = rows["query"]
            queried[q] += 1
            core = saved["original"]["core"]
            cell_key = "cell" if dataset == "LINCS" else "fold"
            if not np.all(core[cell_key][q] == index):
                raise ValueError("Query identities differ from the original cell assignment")
            selected_core = core["selected_lambda_0.2"][q]
            selected_histgb = saved["original"]["histgb"]["selected_lambda_0.2"][q]
            k = int(np.sum(selected_core))
            if k != int(np.sum(selected_histgb)):
                raise ValueError("Existing comparators used different query quotas")
            if dataset == "LINCS" and k != saved["manifest"]["cells"][index]["budget"]:
                raise ValueError("Original LINCS half-quota differs from saved policy")
            if dataset != "LINCS" and k != int(np.floor(.25 * len(q))) // 2:
                raise ValueError("Original fixed-budget quota differs from saved policy")
            record.update(query_k=k, n_query=len(q),
                          counts={role: len(rows[role]) for role in ("train", "valid", "cal", "query")},
                          full_input_dimension=int(metadata.get("full_input_dimension", metadata.get("metadata", {}).get("full_input_dimension"))))
            result.append(record)
            if dataset == "RxRx3":
                mapping = rx_role_by_fold.setdefault(record["outer_fold"], {})
                for role in ("model_train", "ref", "valid", "cal", "query"):
                    for group in core["groups"][rows[role]]:
                        if str(group) in mapping and mapping[str(group)] != role:
                            raise ValueError("RxRx3 chemical identity crosses roles across doses")
                        mapping[str(group)] = role
        np.testing.assert_array_equal(queried, np.ones(len(queried), dtype=int))
    return result


@lru_cache(maxsize=1)
def _first_well(project, dataset):
    """Retain one dataset's first-well profiles and chemistry per worker."""
    path = Path(project) / _SPEC[dataset][1]
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str)
        first = archive["Y"][:, 0].copy()
        chemistry = archive["chem"].copy()
    saved = _saved(project, dataset)
    np.testing.assert_array_equal(ids, saved["ids"])
    if not np.isfinite(first).all() or not np.isfinite(chemistry).all():
        raise ValueError("Prepared decision-time inputs contain nonfinite values")
    return first, chemistry


def clear_input_cache():
    """Release the first-well arrays when a worker is finished with a dataset."""
    _first_well.cache_clear()


def load_cell(project, dataset, cell_index):
    """Return exact saved R2 inputs, outcomes, groups, quotas and comparators.

    Output keys are ``x_ROLE``, ``y_ROLE``, ``ids_ROLE`` and ``groups_ROLE`` for
    ROLE in train/valid/cal/query. ``train`` is ACCESS_MATCHED TRAIN+REF_FIT.
    The constituent model_train/ref identities are also included. Original
    comparator fields use expected/p_null/crps, plus selected and available
    interval coverage/width arrays; histgb is the CLASSIFIER_CAL decision arm.
    """
    dataset = _dataset_name(dataset)
    project = str(Path(project).resolve())
    saved, record, folder, metadata_path, preprocessing_path = _cell_record(project, dataset, cell_index)
    rows, metadata = _roles(saved, dataset, cell_index, metadata_path)
    first, chemistry = _first_well(project, dataset)
    stats = _json(preprocessing_path)
    core = saved["original"]["core"]
    q = rows["query"]
    result = dict(record)
    result.update(query_k=int(np.sum(core["selected_lambda_0.2"][q])),
                  layout_query=core["layout"][q].copy(), query_layout=core["layout"][q].copy(),
                  source_paths={"data": str(Path(project) / _SPEC[dataset][1]),
                                "role_metadata": str(metadata_path),
                                "preprocessing": str(preprocessing_path),
                                "core": str(saved["run"] / "CORE_ORIGINAL.npz"),
                                "histgb": str(saved["run"] / "DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL.npz")},
                  metadata=metadata, originals={})
    for role, idx in rows.items():
        result["ids_" + role] = saved["ids"][idx].copy()
        result["groups_" + role] = core["groups"][idx].copy()
        if role in ("train", "valid", "cal", "query"):
            result["x_" + role] = np.column_stack((transform_input(first[idx], stats), chemistry[idx]))
            result["y_" + role] = core["actual"][idx].copy()
    expected_dim = int(metadata.get("full_input_dimension", metadata.get("metadata", {}).get("full_input_dimension")))
    if result["x_train"].shape[1] != expected_dim:
        raise ValueError("Input dimensionality differs from original HistGB")
    for name, original in saved["original"].items():
        out = {"expected": original["predicted"][q].copy(),
               "predicted": original["predicted"][q].copy(),
               "p_null": original["p_null"][q].copy(),
               "crps": original["crps"][q].copy(),
               "selected": original["selected_lambda_0.2"][q].copy()}
        for old_key, alias in (("gamma_coverage_by_level", "coverage"),
                               ("gamma_width_by_level", "width")):
            if old_key in original:
                out[old_key] = original[old_key][q].copy()
                out[alias] = out[old_key]
        result["originals"][name] = out
    prediction_path = (folder / "DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL.npz" if dataset == "LINCS"
                       else folder / "DIRECT_ACCESS_MATCHED_HISTGB_prediction.npz")
    with np.load(prediction_path, allow_pickle=False) as original_query:
        np.testing.assert_array_equal(original_query["ids"], result["ids_query"])
        np.testing.assert_array_equal(original_query["actual"], result["y_query"])
        np.testing.assert_array_equal(original_query["predicted"], result["originals"]["histgb"]["expected"])
        pk = "p_null" if dataset == "LINCS" else "p_null_calibrated"
        np.testing.assert_array_equal(original_query[pk], result["originals"]["histgb"]["p_null"])
    result["seed"] = int(metadata.get("metadata", {}).get("seed", 20260918 + 100 * record["outer_fold"] + record.get("half", 0)))
    result["full_input_dimension"] = expected_dim
    return result
