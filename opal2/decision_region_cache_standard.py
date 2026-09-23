"""Honest DIST_CAL predictions from frozen EU, JUMP and RxRx3 R2 fits.

QUERY predictions are copied from the original saved R2 arrays. Within each
deployment cell, five chemical-group folds turn DIST_CAL into honest training
data for a subsequent probability correction. The complete CORE error law is
refitted on the unchanged REF and the other CAL folds. Direct regressors and
classifiers remain frozen; only their Platt maps and empirical residual laws
are refitted. No other outer fold's fitted predictions enter this calculation.

This module does not write caches or launch a run. Its caller owns outputs.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .empirical_radial import draw_radial
from .eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from .eu_core_training import predict_eu_core
from .eu_r2_direct_baselines import (
    _bounded_prediction, _fit_platt, _probability, empirical_gamma_support,
)
from .gram_oof_ridge import transform_input
from .reference_information_diagnostic import gamma_forward
from .state_biology_kernel import StateBiologyKernelMean


PROJECT = Path(__file__).resolve().parents[1]
INNER_FOLDS = 5
ROLE_KEYS = ("TRAIN", "VALIDATION", "REF_FIT", "DIST_CAL", "DEV_EVAL")
ARM_NAMES = ("CORE", "HISTGB_CAL", "HISTGB_COHERENT",
             "EXTRATREES_CAL", "EXTRATREES_COHERENT")
_SPECS = {
    "EU": {
        "run": "runs/r2_core_comparison_20260917_v1",
        "core": "runs/eu_core_cc904_20260917_v1",
        "data": "reports/eu_core_development_20260917_v1/prepared_data_cc904/data.npz",
        "cells": 5, "n": 904,
    },
    "JUMP": {
        "run": "runs/jump_r2_completion_20260918_v1",
        "core": "runs/jump_r2_completion_20260918_v1",
        "data": "data/source5_primary_fullcontrols/measurements.npz",
        "cells": 5, "n": 639,
    },
    "RxRx3": {
        "run": "runs/rxrx3_r2_completion_20260918_v1",
        "core": "runs/rxrx3_r2_completion_20260918_v1",
        "data": "data/rxrx3_r2_20260918/prepared_r2/data.npz",
        "cells": 40, "n": 10410,
    },
}


def _dataset_name(dataset):
    names = {key.upper(): key for key in _SPECS}
    try:
        return names[str(dataset).upper()]
    except KeyError as exc:
        raise ValueError("Expected EU, JUMP or RxRx3") from exc


def _integer(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < (1 if positive else 0)):
        raise ValueError(name + " must be a " + ("positive" if positive else "nonnegative") + " integer")
    return int(value)


def _json(path):
    return json.loads(Path(path).read_text())


def _npz(path, keys=None):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in (source.files if keys is None else keys)}


def _positions(ids, selected, name):
    ids, selected = np.asarray(ids, str), np.asarray(selected, str)
    if ids.ndim != 1 or selected.ndim != 1 or len(set(ids)) != len(ids):
        raise ValueError(name + ": unique one-dimensional identities required")
    if len(set(selected)) != len(selected):
        raise ValueError(name + ": duplicate requested identities")
    index = {value: i for i, value in enumerate(ids)}
    if not set(selected) <= set(index):
        raise ValueError(name + ": saved identity is outside the approved cache")
    return np.asarray([index[value] for value in selected], int)


def grouped_calibration_folds(groups, seed, n_folds=INNER_FOLDS):
    """Outcome-independent, row-order-invariant balanced chemical-group folds."""
    seed = _integer(seed, "seed")
    n_folds = _integer(n_folds, "n_folds", positive=True)
    groups = np.asarray(groups, str)
    if groups.ndim != 1 or not len(groups) or np.any(groups == ""):
        raise ValueError("Nonempty CAL chemical groups are required")
    unique = np.unique(groups)
    count = min(n_folds, len(unique))
    if count < 2:
        raise ValueError("Honest calibration needs at least two chemical groups")
    ordered = np.random.default_rng(seed).permutation(unique)
    assignment = {group: i % count for i, group in enumerate(ordered)}
    return np.asarray([assignment[group] for group in groups], int)


def _validate_folds(groups, inner_fold):
    groups, folds = np.asarray(groups, str), np.asarray(inner_fold)
    if (groups.ndim != 1 or not len(groups) or folds.shape != groups.shape
            or folds.dtype.kind not in "iu" or np.any(folds < 0)
            or len(np.unique(folds)) < 2):
        raise ValueError("Aligned CAL groups and at least two integer folds required")
    for group in np.unique(groups):
        if len(np.unique(folds[groups == group])) != 1:
            raise ValueError("A CAL chemical group crosses inner folds")
    return folds


def integrate_core_moments(mean, scatter, stats, *, law, weights, samples, seed):
    """Use the original joint radial sampler and Gamma map, without outcomes."""
    samples, seed = _integer(samples, "samples", positive=True), _integer(seed, "seed")
    mean, scatter = np.asarray(mean, float), np.asarray(scatter, float)
    if mean.ndim != 2 or mean.shape[1] != 9 or not len(mean) or not np.isfinite(mean).all():
        raise ValueError("Finite nonempty standardized CORE [N,9] means required")
    if scatter.shape != (len(mean), 9, 9):
        raise ValueError("CORE scatter must align with means")
    center, scale = np.asarray(stats["u_center"]), np.asarray(stats["u_scale"])
    if center.shape != (9,) or scale.shape != (9,) or np.any(scale <= 0):
        raise ValueError("The saved nine-dimensional target transform is required")
    normal_rng = np.random.default_rng(seed)
    radial_rng = np.random.default_rng(seed + 47000)
    predicted, probability = np.empty(len(mean)), np.empty(len(mean))
    # Eight objects bound working memory; the sampled law and draw count are
    # unchanged. QUERY uses saved draws and is never resampled here.
    for begin in range(0, len(mean), 8):
        end = min(begin + 8, len(mean))
        normal = normal_rng.normal(size=(samples, end - begin, 9))
        if law is None:
            noise = np.einsum("nij,snj->sni", np.linalg.cholesky(scatter[begin:end]), normal)
        else:
            noise = draw_radial(law, np.asarray(weights)[begin:end], scatter[begin:end], normal,
                radial_rng.random((samples, end - begin)), radial_rng.random((samples, end - begin)))
        raw = (mean[None, begin:end] + noise) * scale + center
        gamma = gamma_forward(raw)
        predicted[begin:end] = gamma.mean(axis=0)
        probability[begin:end] = (gamma <= 0).mean(axis=0)
    return {"predicted": predicted, "p_null": probability}


def crossfit_direct_predictions(point, raw_probability, actual, groups, inner_fold, *, seed):
    """Refit only CAL-dependent Platt/residual laws; return honest CAL arrays."""
    seed = _integer(seed, "seed")
    folds = _validate_folds(groups, inner_fold)
    point, raw_probability, actual = [np.asarray(value, float)
                                     for value in (point, raw_probability, actual)]
    if any(value.shape != folds.shape or not np.isfinite(value).all()
           for value in (point, raw_probability, actual)):
        raise ValueError("Aligned finite CAL regression/probability/actual vectors required")
    if np.any((raw_probability < 0) | (raw_probability > 1)):
        raise ValueError("Invalid raw classifier probabilities")
    groups = np.asarray(groups, str)
    result = {"CAL": {"cal_p": np.empty(len(point)), "cal_mean": point.copy()},
              "COHERENT": {"cal_p": np.empty(len(point)), "cal_mean": np.empty(len(point))}}
    audit = []
    with threadpool_limits(limits=1):
        for fold in np.unique(folds):
            held, train = folds == fold, folds != fold
            platt, record = _fit_platt(raw_probability[train], (actual[train] <= 0).astype(int),
                                       seed + int(fold))
            result["CAL"]["cal_p"][held] = platt.predict(raw_probability[held])
            support = empirical_gamma_support(point[held], actual[train] - point[train])
            result["COHERENT"]["cal_mean"][held] = support.mean(axis=1)
            result["COHERENT"]["cal_p"][held] = (support <= 0).mean(axis=1)
            audit.append({"inner_fold": int(fold), "train_rows": int(train.sum()),
                "held_rows": int(held.sum()), "train_groups": int(len(np.unique(groups[train]))),
                "held_groups": int(len(np.unique(groups[held]))), "seed": seed + int(fold),
                "platt": record, "residual_law_rows": int(train.sum())})
    result["audit"] = audit
    return result


def _subset(inputs, rows, *, mean=None):
    result = {key: np.asarray(inputs[key])[rows] for key in ("ids", "groups", "X", "chem")}
    if mean is not None:
        result["mean_u"] = np.asarray(mean)[rows]
    return result


def crossfit_core_predictions(ref_inputs, ref_residual, cal_inputs, cal_residual,
                             cal_mean, base_scatter, training_log_amplitude_sd, stats,
                             inner_fold, *, model_training_ids, model_training_groups,
                             samples, seed):
    """Full frozen-mean CORE laws fit without each held CAL group's outcomes."""
    seed = _integer(seed, "seed")
    samples = _integer(samples, "samples", positive=True)
    folds = _validate_folds(cal_inputs["groups"], inner_fold)
    n = len(folds)
    residual, mean = np.asarray(cal_residual, float), np.asarray(cal_mean, float)
    if residual.shape != (n, 9) or mean.shape != (n, 9):
        raise ValueError("CAL residuals and frozen means must align")
    result = {"cal_p": np.empty(n), "cal_mean": np.empty(n)}
    audit = []
    with threadpool_limits(limits=1):
        for fold in np.unique(folds):
            train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            fit = fit_eu_distribution(ref_inputs, ref_residual,
                _subset(cal_inputs, train), residual[train], base_scatter,
                training_log_amplitude_sd, model_training_ids=model_training_ids,
                model_training_groups=model_training_groups)
            prediction = predict_eu_distribution(fit, _subset(cal_inputs, held, mean=mean))
            # Seeds depend only on declared cell/fold IDs, never residuals or labels.
            fold_seed = seed + 1009 * int(fold)
            moments = integrate_core_moments(mean[held], prediction["scatter_u"], stats,
                law=prediction["law"], weights=prediction["radial_weights"],
                samples=samples, seed=fold_seed)
            result["cal_p"][held] = moments["p_null"]
            result["cal_mean"][held] = moments["predicted"]
            audit.append({"inner_fold": int(fold), "train_rows": len(train), "held_rows": len(held),
                "train_groups": int(len(np.unique(np.asarray(cal_inputs["groups"])[train]))),
                "held_groups": int(len(np.unique(np.asarray(cal_inputs["groups"])[held]))),
                "seed": fold_seed, "radial_calibration_n": fit["law"]["calibration_n"],
                "radial_bandwidth": fit["law"]["bandwidth"],
                "radial_representative_ids": fit["representative_ids"].tolist(),
                "held_ids": np.asarray(cal_inputs["ids"])[held].tolist(),
                "recipe": fit["recipe"], "held_outcomes_used": False,
                "ref_only_covariance_choice": fit["covariance_choice"],
                "ref_only_amplitude_fit": fit["amplitude_fit"]})
    result["audit"] = audit
    return result


def list_cells(dataset):
    """List completed original R2 deployment cells, without opening measurements."""
    dataset = _dataset_name(dataset)
    spec = _SPECS[dataset]
    root = PROJECT / spec["run"]
    if _json(root / "status.json").get("state") != "COMPLETE":
        raise ValueError("The frozen R2 run is not complete")
    cells = [f"fold_{index}" for index in range(spec["cells"])]
    for cell in cells:
        if not (root / cell / "complete.json").is_file():
            raise FileNotFoundError(root / cell / "complete.json")
    return cells


@lru_cache(maxsize=3)
def _load_dataset(dataset):
    """Read only approved prepared profiles and saved R2 outcomes/identities."""
    spec = _SPECS[dataset]
    root = PROJECT / spec["run"]
    source = PROJECT / spec["data"]
    core = _npz(root / "CORE_ORIGINAL.npz",
                ("ids", "groups", "layout", "fold", "actual", "predicted", "p_null", "selected_lambda_0.2"))
    with np.load(source, allow_pickle=False) as stored:
        data = {key: stored[key].copy() for key in ("ids", "chem", "chem_mask")}
        # NumPy decompresses the approved four-role archive; only X is retained.
        # Future outcomes used here are the original cached CAL/QUERY targets.
        data["X"] = stored["Y"][:, 0].copy()
    np.testing.assert_array_equal(core["ids"], data["ids"])
    if len(data["ids"]) != spec["n"]:
        raise ValueError("The approved frozen R2 population changed")
    data.update(groups=core["groups"], layout=core["layout"], core=core)
    return data


def _role_metadata(data, mean_folder, residuals):
    mean_record = _json(mean_folder / "complete.json")
    roles = {
        "TRAIN": mean_record["model_train_ids"],
        "VALIDATION": mean_record["model_validation_ids"],
        "REF_FIT": residuals["ref_ids"],
        "DIST_CAL": residuals["cal_ids"],
        "DEV_EVAL": residuals["query_ids"],
    }
    np.testing.assert_array_equal(roles["REF_FIT"], mean_record["reference_fit_ids"])
    rows = {key: _positions(data["ids"], value, key) for key, value in roles.items()}
    for i, key in enumerate(ROLE_KEYS):
        if not len(rows[key]):
            raise ValueError("An original R2 role is empty")
        for other in ROLE_KEYS[:i]:
            if set(data["groups"][rows[key]]) & set(data["groups"][rows[other]]):
                raise ValueError("Frozen R2 roles share a chemical group")
    return rows


def _frozen_cal_mean(dataset, folder, mean_folder, data, c, stats):
    if dataset != "EU":
        cached = _npz(folder / "mean_predictions.npz", ("ids", "STATE_REF"))
        rows = _positions(cached["ids"], data["ids"][c], "cell-specific frozen CAL mean")
        return cached["STATE_REF"][rows], str(folder / "mean_predictions.npz")
    checkpoint = mean_folder / "STATE50/epoch50.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = StateBiologyKernelMean.from_config(saved["model_config"]).double()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    mean = predict_eu_core(model, stats, data["X"][c], data["chem"][c],
                           data["chem_mask"][c])["mean_u"]
    return mean, str(checkpoint)


def _assert_query_reuse(query, original, q):
    for name, key in (("ids", "ids"), ("actual", "actual"),
                      ("predicted", "predicted"), ("p_null", "p_null")):
        np.testing.assert_array_equal(query[name], original[key][q])


def build_cell(dataset, cell_name, *, samples=100000, seed=20260920):
    """Build one honest CAL cache plus exact original QUERY interfaces in memory."""
    dataset = _dataset_name(dataset)
    samples, seed = _integer(samples, "samples", positive=True), _integer(seed, "seed")
    if not isinstance(cell_name, str) or not re.fullmatch(r"fold_[0-9]+", cell_name):
        raise ValueError("cell_name must be one returned by list_cells")
    if cell_name not in list_cells(dataset):
        raise ValueError("Unknown original deployment cell")
    started = time.perf_counter()
    spec = _SPECS[dataset]
    root, source = PROJECT / spec["run"], PROJECT / spec["core"]
    folder, source_folder = root / cell_name, source / cell_name
    mean_folder = source_folder / "mean"
    cell_index = int(cell_name.removeprefix("fold_"))
    cell_seed = seed + 100003 * cell_index
    with threadpool_limits(limits=1):
        data = _load_dataset(dataset)
        residual_path = source_folder / ("distribution_arrays.npz" if dataset == "EU" else "CORE_residuals.npz")
        residual_keys = ("ref_ids", "cal_ids", "query_ids", "ref_residual", "cal_residual")
        residuals = _npz(residual_path, residual_keys + (("base_scatter",) if dataset == "EU" else ()))
        parts = _role_metadata(data, mean_folder, residuals)
        t, v, r, c, q = (parts[key] for key in ROLE_KEYS)
        np.testing.assert_array_equal(data["core"]["fold"][q], np.full(len(q), cell_index))
        stats = _json(mean_folder / "preprocessing.json")
        if dataset == "EU":
            law_state = _json(source_folder / "distribution_state.json")
            base = residuals["base_scatter"]
            amplitude_sd = law_state["covariance_reference_bandwidth"]
            law_source = source_folder / "distribution_state.json"
        else:
            law_source = folder / "CORE_distribution.joblib"
            law_state = joblib.load(law_source)
            base = np.asarray(law_state["base_scatter"])
            amplitude_sd = law_state["covariance_reference_bandwidth"]
            np.testing.assert_array_equal(law_state["calibration_ids"], data["ids"][c])
            np.testing.assert_array_equal(law_state["calibration_groups"], data["groups"][c])
        mean, mean_source = _frozen_cal_mean(dataset, folder, mean_folder, data, c, stats)
        inner_fold = grouped_calibration_folds(data["groups"][c], cell_seed)
        query_path = source_folder / "AMP_EMP_LOCAL.npz"
        query = _npz(query_path, ("ids", "actual", "predicted", "p_null"))
        _assert_query_reuse(query, data["core"], q)
        timing = {"load_seconds": time.perf_counter() - started}
        before = time.perf_counter()
        core = crossfit_core_predictions(_subset(data, r), residuals["ref_residual"],
            _subset(data, c), residuals["cal_residual"], mean, base, amplitude_sd,
            stats, inner_fold, model_training_ids=data["ids"][np.r_[t, v]],
            model_training_groups=data["groups"][np.r_[t, v]], samples=samples,
            seed=cell_seed + 50000)
        timing["core_cal_seconds"] = time.perf_counter() - before
        arms = {"CORE": {key: core[key] for key in ("cal_p", "cal_mean")}}
        arms["CORE"].update(query_p=query["p_null"].copy(), query_mean=query["predicted"].copy())
        query_seed_offsets = (100000, 200000)
        seed_cache = []
        for offset in query_seed_offsets:
            cached = _npz(source_folder / f"AMP_EMP_LOCAL_mc{offset}.npz",
                          ("ids", "predicted", "p_null"))
            np.testing.assert_array_equal(cached["ids"], query["ids"])
            seed_cache.append(cached)
        arms["CORE"]["query_seed_mean"] = np.stack([item["predicted"] for item in seed_cache])
        arms["CORE"]["query_seed_p"] = np.stack([item["p_null"] for item in seed_cache])
        before = time.perf_counter()
        direct_x = np.column_stack((transform_input(data["X"][c], stats), data["chem"][c]))
        all_direct = None
        if dataset != "EU":
            all_direct = joblib.load(folder / "DIRECT_ACCESS_MATCHED_fitted.joblib")
        direct_audit = {}
        for family in ("HISTGB", "EXTRATREES"):
            prefix = "DIRECT_ACCESS_MATCHED_" + family
            record = _json(folder / (prefix + ".json"))
            np.testing.assert_array_equal(record["cal_ids"], data["ids"][c])
            np.testing.assert_array_equal(record["query_ids"], data["ids"][q])
            np.testing.assert_array_equal(record["train_ids"], data["ids"][np.r_[t, r]])
            np.testing.assert_array_equal(record["valid_ids"], data["ids"][v])
            model_path = folder / (prefix + ".joblib" if dataset == "EU" else "DIRECT_ACCESS_MATCHED_fitted.joblib")
            model = joblib.load(model_path) if all_direct is None else all_direct[family]["model"]
            point = _bounded_prediction(model["regression"], direct_x)
            raw_probability = _probability(model["classifier"], direct_x)
            fitted = crossfit_direct_predictions(point, raw_probability, data["core"]["actual"][c],
                data["groups"][c], inner_fold, seed=cell_seed)
            prediction_path = folder / (prefix + "_prediction.npz")
            direct = _npz(prediction_path, ("ids", "actual", "predicted", "p_null_calibrated",
                                           "gamma_distribution_mean", "p_null_from_gamma"))
            np.testing.assert_array_equal(direct["ids"], data["ids"][q])
            np.testing.assert_array_equal(direct["actual"], query["actual"])
            for suffix, mean_key, p_key in (("CAL", "predicted", "p_null_calibrated"),
                    ("COHERENT", "gamma_distribution_mean", "p_null_from_gamma")):
                arm = dict(fitted[suffix], query_p=direct[p_key].copy(), query_mean=direct[mean_key].copy())
                # The cell file and publication aggregate must agree exactly.
                aggregate_suffix = "CLASSIFIER_CAL" if suffix == "CAL" else suffix
                aggregate = _npz(root / (prefix + "_" + aggregate_suffix + ".npz"),
                                 ("ids", "p_null", "predicted"))
                np.testing.assert_array_equal(aggregate["ids"][q], direct["ids"])
                np.testing.assert_array_equal(aggregate["p_null"][q], arm["query_p"])
                np.testing.assert_array_equal(aggregate["predicted"][q], arm["query_mean"])
                arms[family + "_" + suffix] = arm
            direct_audit[family] = {"model_source": str(model_path),
                "query_prediction_source": str(prediction_path), "base_estimator_refits": 0,
                "inner_folds": fitted["audit"]}
        timing["direct_cal_seconds"] = time.perf_counter() - before
    for arm in arms.values():
        for key, count in (("cal_p", len(c)), ("cal_mean", len(c)),
                           ("query_p", len(q)), ("query_mean", len(q))):
            if arm[key].shape != (count,) or not np.isfinite(arm[key]).all():
                raise ValueError("Nonfinite or misaligned interface " + key)
            if key.endswith("_p") and np.any((arm[key] < 0) | (arm[key] > 1)):
                raise ValueError("Invalid probability interface")
    audit = {"dataset": dataset, "cell": cell_name, "source_run": str(root),
        "prepared_input_source": str(PROJECT / spec["data"]), "mean_source": mean_source,
        "core_residual_source": str(residual_path), "core_distribution_source": str(law_source),
        "query_prediction_source": str(query_path), "query_arrays_exact_original": True,
        "base_estimators_refitted": 0, "mean_retrained": False, "other_outer_fold_predictions_used": False,
        "core_recipe": "LOCAL_SCALE + AMPLITUDE_TOTAL + AMP_EMP_LOCAL",
        "core_cal_dependency": "radial centers, bandwidth, representative weighting and ESS; full law refitted per inner fold",
        "fixed_core_components": "original fitted mean, preprocessing, original base scatter and REF residuals",
        "cal_folds": int(len(np.unique(inner_fold))), "cal_fold_seed": cell_seed,
        "cal_fold_rule": "sorted unique chemical groups, seeded permutation, round-robin five folds",
        "role_counts": {key: len(rows) for key, rows in parts.items()},
        "role_group_counts": {key: int(len(np.unique(data["groups"][rows]))) for key, rows in parts.items()},
        "deployment_cal_rows": len(c), "deployment_cal_groups": int(len(np.unique(data["groups"][c]))),
        "inner_cal_population_difference": "held-out CAL predictions fit the law/Platt map on other ~80% CAL; frozen deployment QUERY used all original CAL",
        "cal_predictions_conditioning": "each held CAL fold is predicted by one law fitted without that whole fold",
        "cal_samples": samples, "core_inner_folds": core["audit"], "direct": direct_audit,
        "query_seed_offsets": list(query_seed_offsets), "query_seed_samples": 100000,
        "null_definition": "Gamma <= 0", "protected_measurements_opened": False,
        "calibration_outcomes_from_original_cell": True, "query_outcomes_used_for_fitting": False,
        "biology_active": False, "representation_active": False, "formal_certificate": False}
    if dataset == "RxRx3":
        cell = _json(root / "run_manifest.json")["cells"][cell_index]
        audit.update(outer_fold=cell["outer_fold"], dose_uM=cell["dose_uM"])
    timing["total_seconds"] = time.perf_counter() - started
    return {"dataset": dataset, "cell": cell_name,
        "query_budget": int(data["core"]["selected_lambda_0.2"][q].sum()),
        "cal_ids": data["ids"][c].copy(), "cal_groups": data["groups"][c].copy(),
        "cal_inner_fold": inner_fold, "cal_layout": data["layout"][c].copy(),
        "query_ids": data["ids"][q].copy(), "query_groups": data["groups"][q].copy(),
        "query_layout": data["layout"][q].copy(), "cal_actual": data["core"]["actual"][c].copy(),
        "query_actual": query["actual"].copy(), "arms": arms, "audit": audit, "timing": timing}
