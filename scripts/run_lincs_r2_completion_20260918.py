"""Resumeable LINCS R2 completion around the saved full five-fold CORE.

The old bank anchors may have trained the RIDGE/HR backbone. Distribution
REF/CAL/QUERY are distinct chemistry groups in each original outer test fold.
No mean fit or old assay artifact is changed by this runner.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.biology_kernel_evaluation import write_json
from opal2.conditional_joint_error_experiment import observable_forward
from opal2.eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from opal2.eu_core_experiment import extra_seed_moments, policy_summary
from opal2.eu_r2_direct_baselines import (
    ARMS as DIRECT_FAMILIES, _bounded_prediction, _fit_platt, _probability,
    _select, empirical_gamma_support, evaluate_direct_distribution,
)
from opal2.empirical_radial_experiment import LEVELS, score
from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates
from opal2.gram_oof_ridge import transform_input, transform_target
from opal2.gram_simple_models import GramSimpleGaussian, _fit_error_second_moment
from opal2.lincs_biology_experiment import load_data

ROOT = PROJECT / "runs/lincs_r2_completion_20260918_v1"
REPORT = PROJECT / "reports/lincs_r2_completion_20260918_v1"
STATE = PROJECT / "runs/lincs_state_biology_20260916_v1"
BACKBONE = PROJECT / "runs/lincs_biology_four_arm_20260915_v1"
RADIAL = PROJECT / "runs/lincs_empirical_radial_20260916_v1"
REFERENCE = PROJECT / "runs/lincs_joint_residual_borrowing_20260916_v1"
SAMPLES = 100000
SEED = 20260918
OFFSETS = (100000, 200000)
THREADS = 2
VERSION = 1
CORE_ARMS = ("CORE_LOCAL_GAUSSIAN", "CORE_AMP_GAUSSIAN", "CORE_ORIGINAL")
MEANS = ("RIDGE_REF", "HR_REF", "STATE_REF")
JOINT_ARMS = CORE_ARMS + tuple(m+s for m in MEANS for s in
    ("_LOCAL_GAUSSIAN", "_AMP_GAUSSIAN", ""))
DIRECT_ARMS = tuple("DIRECT_"+scope+"_"+family+"_"+interface
    for scope in ("TRAIN_MATCHED", "ACCESS_MATCHED") for family in DIRECT_FAMILIES
    for interface in ("COHERENT", "CLASSIFIER_RAW", "CLASSIFIER_CAL"))
ARMS = JOINT_ARMS + DIRECT_ARMS + ("CONSTANT_ACCESS_MATCHED",)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.stem+".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_model(path, value):
    path = Path(path)
    temporary = path.with_name(path.stem+".tmp.joblib")
    joblib.dump(value, temporary)
    temporary.replace(path)


def rows_for(ids, supplied):
    lookup = {str(v): i for i, v in enumerate(ids)}
    if len(lookup) != len(ids) or len(set(supplied)) != len(supplied):
        raise ValueError("Duplicate identities")
    return np.asarray([lookup[str(v)] for v in supplied], int)


def validate_cell(ids, groups, record, cell):
    """Keep original object order and whole-connectivity isolation in five roles."""
    parts = {"TRAIN": rows_for(ids, record["fit_ids"]),
             "VALIDATION": rows_for(ids, record["inner_validation_ids"]),
             "REF_FIT": rows_for(ids, cell["fit_ids"]),
             "DIST_CAL": rows_for(ids, cell["calibration_ids"]),
             "DEV_EVAL": rows_for(ids, cell["query_ids"])}
    for name, rows in parts.items():
        if not len(rows):
            raise ValueError("Empty role: "+name)
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if set(groups[parts[a]]) & set(groups[parts[b]]):
                raise ValueError("Chemical identity overlap: "+a+" / "+b)
    outer = rows_for(ids, record["test_ids"])
    held = np.r_[parts["REF_FIT"], parts["DIST_CAL"], parts["DEV_EVAL"]]
    if set(outer) != set(held) or len(held) != len(outer):
        raise ValueError("Original outer test membership changed")
    if set(np.r_[parts["TRAIN"], parts["VALIDATION"], outer]) != set(range(len(ids))):
        raise ValueError("Original complete cohort changed")
    budget = cell["budget"]
    if not isinstance(budget, int) or not 0 <= budget <= len(parts["DEV_EVAL"]):
        raise ValueError("Invalid original cell acquisition count")
    return parts


def select_budget(ids, expected, probability, lam, k):
    """Fixed cell k, ID-ascending ties; preserve original half-budget remainder."""
    ids, expected, probability = np.asarray(ids), np.asarray(expected), np.asarray(probability)
    if (expected.shape != ids.shape or probability.shape != ids.shape
            or not np.isfinite(expected).all() or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))
            or not 0 <= k <= len(ids)):
        raise ValueError("Invalid aligned decision inputs")
    chosen = np.zeros(len(ids), bool)
    chosen[np.lexsort((ids, -(expected-lam*probability)))[:k]] = True
    return chosen


def calibrate_selected(models, x_cal, actual_cal, x_query, seed):
    """Use the unchanged full direct-baseline calibration; estimators stay fixed."""
    regression, classifier = models["regression"], models["classifier"]
    point = _bounded_prediction(regression, x_query)
    residual = np.sort(actual_cal-_bounded_prediction(regression, x_cal))
    probability = _probability(classifier, x_query)
    platt, meta = _fit_platt(_probability(classifier, x_cal), (actual_cal <= 0).astype(int), seed)
    support = empirical_gamma_support(point, residual)
    return dict(predicted=point, p_null=probability, p_null_calibrated=platt.predict(probability),
        gamma_residuals=residual, gamma_distribution_mean=support.mean(1),
        p_null_from_gamma=(support <= 0).mean(1), platt=platt, calibration_metadata=meta)


def fit_selected(data_x, actual, t, v, family, directory, ids, groups, seed, scope):
    """Cache each complete selected family; TRAIN_MATCHED is shared by both halves."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/(family+".joblib")
    provenance = dict(version=VERSION, family=family, scope=scope, seed=seed,
        train_ids=ids[t].tolist(), validation_ids=ids[v].tolist(),
        train_groups=np.unique(groups[t]).tolist(), validation_groups=np.unique(groups[v]).tolist(),
        full_input_dimension=data_x.shape[1], feature_truncation=False, pca_used=False,
        future_inputs_at_query=False, validation_refit=False,
        training_preprocessing="Original backbone TRAIN-only transform; full X, log-norm and Morgan-513")
    if path.exists():
        saved = joblib.load(path)
        if saved["provenance"] != provenance:
            raise ValueError("Saved direct fitting provenance changed")
        return saved
    regression, regmeta = _select(family, data_x[t], actual[t], data_x[v], actual[v], seed,
                                  classifier=False)
    classifier, clsmeta = _select(family, data_x[t], (actual[t] <= 0).astype(int),
        data_x[v], (actual[v] <= 0).astype(int), seed, classifier=True)
    result = dict(regression=regression, classifier=classifier, provenance=provenance,
                  regression_metadata=regmeta, classification_metadata=clsmeta)
    save_model(path, result)
    write_json(directory/(family+".json"), {k: val for k, val in result.items()
                                             if k not in ("regression", "classifier")})
    return result


def prepare():
    manifest = read_json(STATE/"run_manifest.json")
    prior = read_json(RADIAL/"summary.json")
    if (manifest["final_opened"] or manifest["fifth_repeat_opened"]
            or len(manifest["ids"]) != 1188 or prior["samples"] != 10000
            or Path(prior["reference_run"]) != STATE or len(prior["cells"]) != 10):
        raise ValueError("Legacy development scope or source changed")
    data, metadata = load_data(PROJECT/"data/lincs_pilot1_biology_20260915")
    ids, groups = data["ids"], data["groups"]
    if ids.tolist() != manifest["ids"] or groups.tolist() != manifest["groups"]:
        raise ValueError("Identity/chemistry order differs from saved models")
    for unit, group in zip(metadata["units"], groups):
        if unit["chemistry"]["inchikey14"] != group:
            raise ValueError("Connectivity differs from source chemistry annotation")
    records = {r["fold"]: r for r in manifest["folds"]}
    scopes = {s["fold"]: s for s in manifest["scopes"]}
    seen = np.zeros(len(ids), int)
    cells = []
    for original in prior["cells"]:
        f, h = original["fold"], original["half"]
        parts = validate_cell(ids, groups, records[f], original)
        seen[parts["DEV_EVAL"]] += 1
        anchors = scopes[f]["reference_ids"]
        if len(anchors) != 64 or not set(anchors) <= set(records[f]["fit_ids"]):
            raise ValueError("Historical bank/backbone roles changed")
        cells.append(dict(fold=f, half=h, budget=original["budget"],
            ids={name: ids[rows].tolist() for name, rows in parts.items()},
            counts={name: len(rows) for name, rows in parts.items()},
            group_counts={name: len(np.unique(groups[rows])) for name, rows in parts.items()},
            radial_representative_ids=original["representative_ids"],
            legacy_bank_anchor_ids=anchors, bank_anchors_in_backbone_fit=True,
            query_own_outcomes_used_for_any_fit=False))
    np.testing.assert_array_equal(seen, np.ones(len(ids), int))
    cfg = dict(version=VERSION, n=len(ids), chemical_groups=len(np.unique(groups)),
        data_directory=str(PROJECT/"data/lincs_pilot1_biology_20260915"),
        state_source=str(STATE), backbone_source=str(BACKBONE),
        distribution_source=str(RADIAL), reference_source=str(REFERENCE),
        samples=SAMPLES, seed=SEED, extra_mc_offsets=list(OFFSETS), threads=THREADS,
        arms=list(ARMS), cells=cells, levels=LEVELS.tolist(), lambdas=[.2, 0.],
        mean_fits_performed=0, prior_results_modified=False, protected_measurements_opened=False,
        bank_protocol="Legacy 64 anchors participated in RIDGE/HR outcomes; excluded from A/STATE branch supervision",
        error_reference_protocol="Independent outer-test REF/CAL/QUERY; whole-connectivity isolation within every cell",
        matched_reference_protocol="Mean-specific REF second moment, LOCAL_SCALE, AMPLITUDE_TOTAL, AMP_EMP_LOCAL; original fitted means",
        policy_budget="Original ten cell k retained, including original half-budget remainder; common k for all methods",
        direct_calibration="Existing fixed L2 Platt on DIST_CAL and empirical Gamma CAL residual law",
        repeated_development=True, formal_certificate=False)
    ROOT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    path = ROOT/"run_manifest.json"
    if path.exists() and read_json(path) != cfg:
        raise ValueError("Existing run configuration changed; cannot resume")
    if not path.exists():
        write_json(path, cfg)
    return data, metadata, manifest, prior, cfg


def mean_arrays(fold, data, manifest, raw, folder):
    cache = folder/"frozen_means.npz"
    record = manifest["folds"][fold]
    outer = rows_for(data["ids"], record["test_ids"])
    stats = read_json(STATE/"folds"/f"fold_{fold}"/"preprocessing.json")
    if stats != read_json(BACKBONE/"folds"/f"fold_{fold}"/"preprocessing.json"):
        raise ValueError("Backbone and STATE coordinate systems differ")
    if cache.exists():
        saved = read_npz(cache)
        np.testing.assert_array_equal(saved["ids"], data["ids"][outer])
        return saved, stats, outer
    state = read_npz(STATE/"folds"/f"fold_{fold}"/"arms/STATE50/evaluation/u_predictions.npz")
    hr = read_npz(BACKBONE/"folds"/f"fold_{fold}"/"arms/HR/evaluation/u_predictions.npz")
    for source in (state, hr):
        np.testing.assert_array_equal(source["ids"], data["ids"][outer])
        np.testing.assert_allclose(source["actual_u"], transform_target(raw[outer], stats), atol=1e-11)
    ridge = GramSimpleGaussian.load(BACKBONE/"folds"/f"fold_{fold}"/"ridge.npz")
    saved = dict(ids=data["ids"][outer], actual_u=state["actual_u"],
        RIDGE_REF=ridge.predict_mean(transform_input(data["Y"][outer, 0], stats)),
        HR_REF=hr["mean_u"], STATE_REF=state["mean_u"])
    save_npz(cache, **saved)
    write_json(folder/"mean_provenance.json", dict(STATE_REF="saved actual epoch50 predictions",
        HR_REF="saved validation-selected HR predictions", RIDGE_REF="saved ridge forward pass only",
        means_retrained=False, coordinate_source=str(STATE/"folds"/f"fold_{fold}"/"preprocessing.json")))
    return saved, stats, outer


def distribution_specs(data, part, cached, outer, stats, original, cell_folder, legacy):
    q, r, c, t, v = (part[k] for k in ("DEV_EVAL", "REF_FIT", "DIST_CAL", "TRAIN", "VALIDATION"))
    lookup = {row: i for i, row in enumerate(outer)}
    qi, ri, ci = (np.array([lookup[row] for row in rows]) for rows in (q, r, c))
    f, h = original["fold"], original["half"]
    ref = read_npz(REFERENCE/f"cell_{f}_{h}_reference.npz")
    radial = read_npz(RADIAL/f"cell_{f}_{h}_radial.npz")
    for key, rows in (("query_ids", q), ("fit_ids", r), ("cal_ids", c)):
        np.testing.assert_array_equal(ref[key], data["ids"][rows])
    np.testing.assert_array_equal(radial["query_ids"], data["ids"][q])
    np.testing.assert_array_equal(radial["cal_ids"], np.asarray(original["representative_ids"]))
    mean = cached["STATE_REF"][qi]
    np.testing.assert_array_equal(mean, legacy["AMP_EMP_LOCAL"]["mean_u"][q])
    np.testing.assert_array_equal(ref["query_covariance"], legacy["GAUSSIAN"]["scatter_u"][q])
    specs = {
        "CORE_LOCAL_GAUSSIAN": dict(mean=mean, scatter=legacy["GAUSSIAN"]["scatter_u"][q], law=None, weights=None),
        "CORE_AMP_GAUSSIAN": dict(mean=mean, scatter=legacy["AMP_GAUSSIAN"]["scatter_u"][q], law=None, weights=None),
        "CORE_ORIGINAL": dict(mean=mean, scatter=legacy["AMP_EMP_LOCAL"]["scatter_u"][q],
            law=original["laws"]["amplitude_law"], weights=radial["local_weights"]),
    }
    def inputs(rows, mu=None):
        value = {key: data[key][rows] for key in ("ids", "groups", "chem")}
        value["X"] = data["Y"][rows, 0]
        if mu is not None:
            value["mean_u"] = mu
        return value
    for name in MEANS:
        path = cell_folder/(name+"_distribution.joblib")
        mu = cached[name]
        if path.exists():
            fitted = joblib.load(path)
            np.testing.assert_array_equal(fitted["ref_inputs"]["ids"], data["ids"][r])
            np.testing.assert_array_equal(fitted["calibration_ids"], data["ids"][c])
        else:
            residual = cached["actual_u"][ri]-mu[ri]
            base, _, _, audit = _fit_error_second_moment(residual, include_bias=True)
            fitted = fit_eu_distribution(inputs(r), residual, inputs(c), cached["actual_u"][ci]-mu[ci],
                base, float(np.std(np.log(np.linalg.norm(data["Y"][t, 0], axis=1)))),
                model_training_ids=data["ids"][np.r_[t, v]],
                model_training_groups=data["groups"][np.r_[t, v]])
            save_model(path, fitted)
            write_json(cell_folder/(name+"_distribution.json"), dict(report=fitted["report"],
                base_estimation=audit, means_retrained=False,
                base_source="Mean-specific held-out REF residual second moment; distinct from original RIDGE OOF base"))
        prediction = predict_eu_distribution(fitted, inputs(q, mu[qi]))
        for suffix, scatter, law, weights in (
            ("_LOCAL_GAUSSIAN", prediction["base_scatter_u"], None, None),
            ("_AMP_GAUSSIAN", prediction["scatter_u"], None, None),
            ("", prediction["scatter_u"], prediction["law"], prediction["radial_weights"])):
            specs[name+suffix] = dict(mean=mu[qi], scatter=scatter, law=law, weights=weights)
    return specs, cached["actual_u"][qi]


def direct_cell(data, actual, stats, part, f, h, folder, update):
    t, v, r, c, q = (part[k] for k in ("TRAIN", "VALIDATION", "REF_FIT", "DIST_CAL", "DEV_EVAL"))
    ids, groups = data["ids"], data["groups"]
    x = np.column_stack((transform_input(data["Y"][:, 0], stats), data["chem"]))
    for scope, train in (("TRAIN_MATCHED", t), ("ACCESS_MATCHED", np.r_[t, r])):
        fit_dir = (ROOT/f"fold_{f}"/"direct_train_models" if scope == "TRAIN_MATCHED"
                   else folder/"direct_access_models")
        for family in DIRECT_FAMILIES:
            prefix = "DIRECT_"+scope+"_"+family
            final_paths = [folder/(prefix+"_"+suffix+".npz")
                           for suffix in ("COHERENT", "CLASSIFIER_RAW", "CLASSIFIER_CAL")]
            if all(path.exists() for path in final_paths):
                continue
            update("direct_baseline", fold=f, half=h, arm=prefix)
            models = fit_selected(x, actual, train, v, family, fit_dir, ids, groups,
                                  SEED+100*f, scope)
            bundle_path = folder/(prefix+"_calibration.joblib")
            if bundle_path.exists():
                bundle = joblib.load(bundle_path)
                np.testing.assert_array_equal(bundle["cal_ids"], ids[c])
                np.testing.assert_array_equal(bundle["query_ids"], ids[q])
            else:
                bundle = calibrate_selected(models, x[c], actual[c], x[q], SEED+100*f+h)
                bundle.update(cal_ids=ids[c], query_ids=ids[q])
                save_model(bundle_path, bundle)
                write_json(folder/(prefix+"_calibration.json"), dict(
                    calibration=bundle["calibration_metadata"], cal_ids=ids[c], query_ids=ids[q],
                    train_ids=ids[train], validation_ids=ids[v], full_input_dimension=x.shape[1],
                    classifier_is_separate_from_Gamma_distribution=True,
                    prediction_interfaces=["COHERENT", "CLASSIFIER_RAW", "CLASSIFIER_CAL"],
                    Gamma_distribution="Exact uniform CAL residual mixture, bounded to [-1.02,.98]",
                    estimator_cache=str(fit_dir/(family+".joblib"))))
            evaluated = evaluate_direct_distribution(bundle, actual[q])
            common = {key: evaluated[key] for key in ("crps", "gamma_coverage_by_level", "gamma_width_by_level")}
            for suffix, expected, probability in (
                ("COHERENT", bundle["gamma_distribution_mean"], bundle["p_null_from_gamma"]),
                ("CLASSIFIER_RAW", bundle["predicted"], bundle["p_null"]),
                ("CLASSIFIER_CAL", bundle["predicted"], bundle["p_null_calibrated"])):
                path = folder/(prefix+"_"+suffix+".npz")
                if not path.exists():
                    save_npz(path, ids=ids[q], actual=actual[q], predicted=expected, p_null=probability,
                             brier=np.square(probability-(actual[q] <= 0)), **common)
    path = folder/"CONSTANT_ACCESS_MATCHED.npz"
    if not path.exists():
        values = np.sort(actual[np.r_[t, r]])
        mu = np.full(len(q), values.mean())
        p = np.full(len(q), np.mean(values <= 0))
        evaluated = evaluate_direct_distribution(dict(predicted=mu, gamma_residuals=values-values.mean(),
                                                      p_null=p, p_null_calibrated=p), actual[q])
        save_npz(path, ids=ids[q], actual=actual[q], predicted=mu, p_null=p,
            brier=np.square(p-(actual[q] <= 0)), **{key: evaluated[key] for key in
                ("crps", "gamma_coverage_by_level", "gamma_width_by_level")})


def paired_intervals(matrix, labels, seed=SEED, replicates=2000):
    """Paired whole-block resampling, using one draw matrix for every contrast."""
    matrix = np.asarray(matrix, float)
    groups, index = np.unique(labels, return_inverse=True)
    sums = np.zeros((len(groups), matrix.shape[1]))
    np.add.at(sums, index, matrix)
    sizes = np.bincount(index, minlength=len(groups))
    rng = np.random.default_rng(seed)
    counts = np.asarray([np.bincount(rng.integers(len(groups), size=len(groups)),
                                    minlength=len(groups)) for _ in range(replicates)], float)
    samples = (counts@sums)/(counts@sizes)[:, None]
    return matrix.mean(0), np.quantile(samples, [.025, .975], axis=0).T


def resource_counts(cell, arm):
    count = cell["counts"]
    uses_ref = not arm.startswith("DIRECT_TRAIN_MATCHED")
    uses_cal = (arm != "CONSTANT_ACCESS_MATCHED" and not arm.endswith("_GAUSSIAN")
                and not arm.endswith("CLASSIFIER_RAW"))
    ref = count["REF_FIT"] if uses_ref else 0
    cal = count["DIST_CAL"] if uses_cal else 0
    return dict(backbone_training_objects=count["TRAIN"], validation_objects=count["VALIDATION"],
        bank_anchors_already_in_backbone=64, error_reference_objects=ref,
        calibration_objects=cal, common_evaluation_CAL_objects=count["DIST_CAL"],
        raw_classifier_Gamma_distribution_scoring_still_uses_CAL=arm.endswith("CLASSIFIER_RAW"),
        reference_all_new_wells=4*ref, reference_if_X_available_wells=3*ref,
        calibration_all_new_wells=4*cal, calibration_if_X_available_wells=3*cal,
        combined_setup_all_new_wells=4*(ref+cal), combined_setup_if_X_available_wells=3*(ref+cal),
        reference_outcome_rows=ref, calibration_outcome_rows=cal,
        training_scope=("TRAIN+REF" if arm.startswith("DIRECT_ACCESS_MATCHED") or arm == "CONSTANT_ACCESS_MATCHED"
            else "TRAIN" if arm.startswith("DIRECT_TRAIN_MATCHED")
            else "TRAIN; original neural branch supervised on TRAIN minus legacy bank group"))


def aggregate(data, metadata, cfg):
    ids, groups = data["ids"], data["groups"]
    layout = np.asarray([unit["layout_block"] for unit in metadata["units"]])
    n = len(ids)
    stores = {name: {} for name in ARMS}
    actual = np.empty(n)
    folds, cell_index = np.empty(n, int), np.empty(n, int)
    for number, cell in enumerate(cfg["cells"]):
        f, h = cell["fold"], cell["half"]
        folder = ROOT/f"cell_{f}_{h}"
        if not (folder/"complete.json").exists():
            raise ValueError("Cannot aggregate an unfinished cell")
        q = rows_for(ids, cell["ids"]["DEV_EVAL"])
        folds[q], cell_index[q] = f, number
        for name in ARMS:
            row = read_npz(folder/(name+".npz"))
            np.testing.assert_array_equal(row.pop("ids"), ids[q])
            values = row.pop("actual")
            if name == ARMS[0]:
                actual[q] = values
            else:
                np.testing.assert_array_equal(actual[q], values)
            for key, value in row.items():
                if value.ndim == 0 or value.shape[0] != len(q):
                    raise ValueError("Saved per-object field does not align: "+name+"/"+key)
                if key not in stores[name]:
                    stores[name][key] = np.empty((n, *value.shape[1:]), dtype=value.dtype)
                stores[name][key][q] = value
    null = actual <= 0
    for name, out in stores.items():
        for lam in (.2, 0.):
            selected = np.zeros(n, bool)
            for number, cell in enumerate(cfg["cells"]):
                q = np.flatnonzero(cell_index == number)
                selected[q] = select_budget(ids[q], out["predicted"][q], out["p_null"][q], lam, cell["budget"])
            out[f"selected_lambda_{lam:g}"] = selected
        save_npz(ROOT/(name+".npz"), ids=ids, groups=groups, layout=layout, fold=folds,
                 cell=cell_index, actual=actual, **out)
    core = stores["CORE_ORIGINAL"]
    metrics, risk, overlap, contrasts = {}, {}, {}, {}
    for name, out in stores.items():
        probability = out["p_null"]
        row = dict(n=n, gamma_mse=float(np.mean(np.square(out["predicted"]-actual))),
            gamma_spearman=float(spearmanr(actual, out["predicted"]).statistic) if np.ptp(out["predicted"]) else None,
            null_auc=float(roc_auc_score(null, probability)), brier=float(np.mean(np.square(probability-null))),
            policies={}, full_joint_available="mean_u" in out)
        for key in ("crps", "nll", "energy", "single_crps", "pair_crps", "average_crps",
                    "triple_average_crps", "absolute_pair_crps", "joint_coverage_by_level", "gamma_coverage_by_level"):
            if key in out:
                row[key] = out[key].mean(0)
        if "mean_u" in out:
            row["geometry_mse"] = float(np.square(out["mean_u"]-out["actual_u"]).mean())
        risk[name], overlap[name] = {}, {}
        for lam in (.2, 0.):
            key = f"lambda_{lam:g}"
            selected, base = out["selected_"+key], core["selected_"+key]
            row["policies"][key] = dict(policy_summary(actual, selected),
                predicted_null_count=float(probability[selected].sum()), selected_ids=ids[selected])
            intersection = int(np.sum(selected & base))
            overlap[name][key] = dict(intersection=intersection, selected=int(selected.sum()),
                overlap_fraction=intersection/max(int(base.sum()), 1),
                jaccard=intersection/max(int(np.sum(selected | base)), 1),
                symmetric_difference=int(np.sum(selected != base)))
            risk[name][key] = {}
            for scope, take in (("all", np.ones(n, bool)), ("own_selected", selected), ("CORE_selected", base)):
                risk[name][key][scope] = dict(n=int(take.sum()), predicted_null=float(probability[take].sum()),
                    actual_null=int(null[take].sum()), brier=float(np.square(probability[take]-null[take]).mean()))
                contrasts[("CALIBRATION::"+name, key+"_"+scope+"_NULL_gap_per_candidate")] = (null-probability)*take
            contrasts[(name, key+"_value_per_candidate")] = actual*(selected.astype(float)-base.astype(float))
            contrasts[(name, key+"_false_activation_per_candidate")] = null*(selected.astype(float)-base.astype(float))
        metrics[name] = row
        contrasts[(name, "gamma_mse")] = np.square(out["predicted"]-actual)-np.square(core["predicted"]-actual)
        contrasts[(name, "brier")] = np.square(probability-null)-np.square(core["p_null"]-null)
        for key in ("crps", "nll", "energy"):
            if key in out:
                contrasts[(name, key)] = out[key]-core[key]
    # Mean and error-law attribution use the same already frozen predictions.
    matched_pairs = []
    for suffix in ("_LOCAL_GAUSSIAN", "_AMP_GAUSSIAN", ""):
        matched_pairs += [("HR_REF"+suffix, "RIDGE_REF"+suffix),
                          ("STATE_REF"+suffix, "HR_REF"+suffix)]
    for mean_name in MEANS:
        matched_pairs += [(mean_name+"_AMP_GAUSSIAN", mean_name+"_LOCAL_GAUSSIAN"),
                          (mean_name, mean_name+"_AMP_GAUSSIAN")]
    matched_pairs += [("CORE_AMP_GAUSSIAN", "CORE_LOCAL_GAUSSIAN"),
                      ("CORE_ORIGINAL", "CORE_AMP_GAUSSIAN")]
    for a, b in matched_pairs:
        aa, bb = stores[a], stores[b]
        label = a+" minus "+b
        contrasts[(label, "gamma_mse")] = np.square(aa["predicted"]-actual)-np.square(bb["predicted"]-actual)
        contrasts[(label, "brier")] = np.square(aa["p_null"]-null)-np.square(bb["p_null"]-null)
        for metric in ("crps", "nll", "energy"):
            contrasts[(label, metric)] = aa[metric]-bb[metric]
        contrasts[(label, "geometry_mse")] = (np.square(aa["mean_u"]-aa["actual_u"]).mean(1)
                                              -np.square(bb["mean_u"]-bb["actual_u"]).mean(1))
        for lam in (.2, 0.):
            key = f"lambda_{lam:g}"
            delta = aa["selected_"+key].astype(float)-bb["selected_"+key].astype(float)
            contrasts[(label, key+"_value_per_candidate")] = actual*delta
            contrasts[(label, key+"_false_activation_per_candidate")] = null*delta
    keys = list(contrasts)
    matrix = np.column_stack([contrasts[key] for key in keys])
    paired = {name: {} for name in ARMS}
    for scope, labels in (("chemical_connectivity", groups), ("layout", layout)):
        mean, interval = paired_intervals(matrix, labels)
        for j, (name, metric) in enumerate(keys):
            paired.setdefault(name, {}).setdefault(metric, {})[scope] = dict(difference=mean[j], ci95=interval[j])
    for name in ARMS:
        risk[name]["absolute_gap_intervals_per_candidate"] = paired.pop("CALIBRATION::"+name)
        risk[name]["gap_definition"] = "Observed minus predicted NULL count in the named scope, divided by all 1188 candidates; absolute calibration gap, not a CORE contrast"
    monte_carlo = {}
    for name in JOINT_ARMS:
        monte_carlo[name] = []
        for offset in OFFSETS:
            moments = {key: np.empty(n) for key in ("predicted", "p_null")}
            for number, cell in enumerate(cfg["cells"]):
                q = np.flatnonzero(cell_index == number)
                value = read_npz(ROOT/f"cell_{cell['fold']}_{cell['half']}"/(name+f"_mc{offset}.npz"))
                saved_rows = rows_for(ids, value["ids"])
                if set(saved_rows) != set(q):
                    raise ValueError("MC query membership changed")
                for key in moments:
                    moments[key][saved_rows] = value[key]
            entry = dict(seed_offset=offset, samples=SAMPLES, policies={})
            for lam in (.2, 0.):
                chosen = np.zeros(n, bool)
                for number, cell in enumerate(cfg["cells"]):
                    q = np.flatnonzero(cell_index == number)
                    chosen[q] = select_budget(ids[q], moments["predicted"][q], moments["p_null"][q], lam, cell["budget"])
                key = f"lambda_{lam:g}"
                entry["policies"][key] = dict(policy_summary(actual, chosen),
                    selected_ids=ids[chosen], symmetric_difference_from_main=int(np.sum(chosen != stores[name]["selected_"+key])))
            monte_carlo[name].append(entry)
    controls = fixed_controls(ids, data["Y"][:, 0], actual, cell_index, cfg["cells"])
    costs = []
    for number, cell in enumerate(cfg["cells"]):
        q = np.flatnonzero(cell_index == number)
        arms = {}
        for name, out in stores.items():
            counts = resource_counts(cell, name)
            policies = {}
            for lam in (.2, 0.):
                selected = out[f"selected_lambda_{lam:g}"] & (cell_index == number)
                value = float(actual[selected].sum())
                policies[f"lambda_{lam:g}"] = dict(action_net_value=value,
                    acquisition_wells=2*int(selected.sum()),
                    reference_existing_total_value=value,
                    reference_all_new_total_value=value-.01*counts["reference_all_new_wells"],
                    reference_X_available_total_value=value-.01*counts["reference_if_X_available_wells"],
                    reference_plus_CAL_all_new_total_value=value-.01*counts["combined_setup_all_new_wells"],
                    reference_plus_CAL_X_available_total_value=value-.01*counts["combined_setup_if_X_available_wells"])
            arms[name] = dict(resources=counts, policies=policies)
        costs.append(dict(fold=cell["fold"], half=cell["half"], query_n=len(q), k=cell["budget"], arms=arms))
    payload = dict(complete=True, n=n, samples=SAMPLES, metrics=metrics,
        paired_vs_CORE={name:paired[name] for name in ARMS},
        paired_mean_and_law_attribution={name:values for name, values in paired.items() if name not in ARMS},
        calibration=risk, selected_overlap_with_CORE=overlap, monte_carlo=monte_carlo,
        fixed_and_random_controls=controls, costs_by_deployment_cell=costs,
        costs_note="Reference-only cost is the EU/JUMP comparable primary ledger. Calibration and combined REF+CAL setup are separate. Gaussian laws do not require CAL at deployment. Per deployed cell; repeated CV resources are not summed as a single purchase. Gamma already subtracts .02 for ADD_TWO.",
        limits="Paired block bootstrap conditional on fitted development predictions; unadjusted multiarm intervals are not independent policy certification.",
        original_mean_fits_reused=True, original_10k_predictions_preserved=True,
        legacy_bank_anchors_in_backbone=True, honest_outer_test_error_references=True,
        direct_interval_semantics="CLASSIFIER_RAW/CAL retain the regression residual Gamma CRPS/intervals; classifier probabilities are separate.",
        protected_measurements_opened=False)
    write_json(ROOT/"summary.json", payload)
    write_json(REPORT/"summary.json", payload)
    lines = ["# LINCS R2 completion", "", "1,188 development objects; frozen five-fold means and ten honest error-reference cells. All joint scores use 100,000 draws.", "",
        "| Arm | Gamma CRPS | NULL Brier | AUC | Selected NULL | Selected mean Gamma |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in metrics.items():
        policy = value["policies"]["lambda_0.2"]
        lines.append(f"| {name} | {value['crps']:.6f} | {value['brier']:.6f} | {value['null_auc']:.4f} | {policy['null_selected']} | {policy['selected_mean_value']:.6f} |")
    lines += ["", "Legacy bank anchors participated in backbone fitting; distribution REF/CAL/QUERY are disjoint outer-test chemistry groups. Mean-specific REF distributions have their own REF-estimated base scatter.",
        "", "Direct scalar methods have no fabricated nine-dimensional covariance, NLL or joint coverage. Raw/calibrated classifier probabilities and coherent Gamma-law probabilities are reported separately.",
        "", "Budgets retain each original cell's fixed k, including half-split remainder allocation. Full predictions, fitted direct estimators, calibrators, distributions, policies and sampling sensitivity are cached."]
    (REPORT/"REPORT.md").write_text("\n".join(lines)+"\n")


def fixed_controls(ids, x, actual, cell_index, cells):
    choices = {key: np.zeros(len(ids), bool) for key in ("ID_FIXED", "AMPLITUDE_FIXED")}
    random_value, random_null = np.zeros(2000), np.zeros(2000)
    for number, cell in enumerate(cells):
        q = np.flatnonzero(cell_index == number)
        k = cell["budget"]
        choices["ID_FIXED"][q[np.argsort(ids[q], kind="stable")[:k]]] = True
        choices["AMPLITUDE_FIXED"][q[np.lexsort((ids[q], -np.linalg.norm(x[q], axis=1)))[:k]]] = True
        rng = np.random.default_rng(SEED+500000+number)
        for j in range(2000):
            selected = rng.choice(q, k, replace=False)
            random_value[j] += actual[selected].sum()
            random_null[j] += (actual[selected] <= 0).sum()
    return dict(same_budget={name: dict(policy_summary(actual, chosen), selected_ids=ids[chosen])
                            for name, chosen in choices.items()},
        random_same_budget=dict(replicates=2000, fixed_k=sum(c["budget"] for c in cells),
            total_value_mean=float(random_value.mean()), total_value_interval=np.quantile(random_value, [.025, .975]),
            selected_null_mean=float(random_null.mean()), selected_null_interval=np.quantile(random_null, [.025, .975])),
        NEVER_ADD=dict(total_value=0., acquisition_wells=0),
        ALWAYS_ADD=policy_summary(actual, np.ones(len(ids), bool)))


def run(aggregate_only=False):
    started = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/"run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT/"status.json").exists() and read_json(ROOT/"status.json")["state"] == "COMPLETE":
            print("Already complete; cached result preserved.", flush=True)
            return
        def update(stage, **extras):
            payload = dict(state="RUNNING", stage=stage, elapsed_seconds=time.monotonic()-started, **extras)
            write_json(ROOT/"status.json", payload)
            print(json.dumps(payload), flush=True)
        try:
            data, metadata, manifest, prior, cfg = prepare()
            if not aggregate_only:
                raw = gram_to_coordinates(profiles_to_gram(torch.as_tensor(data["Y"], dtype=torch.float64))).numpy()
                actual, observed, difference, _ = observable_forward(raw)
                norm2 = np.square(data["Y"][:, 0]).mean(1)
                absolute = np.log1p(difference*norm2[:, None])
                legacy = {name: read_npz(RADIAL/(name+".npz")) for name in ("GAUSSIAN", "AMP_GAUSSIAN", "AMP_EMP_LOCAL")}
                for value in legacy.values():
                    np.testing.assert_array_equal(value["ids"], data["ids"])
                    np.testing.assert_allclose(value["actual"], actual, atol=1e-12, rtol=1e-12)
                for original in prior["cells"]:
                    f, h = original["fold"], original["half"]
                    folder = ROOT/f"cell_{f}_{h}"
                    folder.mkdir(exist_ok=True)
                    if (folder/"complete.json").exists():
                        continue
                    update("cell_setup", fold=f, half=h)
                    record = next(r for r in manifest["folds"] if r["fold"] == f)
                    parts = validate_cell(data["ids"], data["groups"], record, original)
                    fold_folder = ROOT/f"fold_{f}"
                    fold_folder.mkdir(exist_ok=True)
                    cached, stats, outer = mean_arrays(f, data, manifest, raw, fold_folder)
                    specs, target = distribution_specs(data, parts, cached, outer, stats, original, folder, legacy)
                    q = parts["DEV_EVAL"]
                    for name, spec in specs.items():
                        path = folder/(name+".npz")
                        seed = SEED+100*f+h
                        if not path.exists():
                            update("joint_score", fold=f, half=h, arm=name)
                            out = score(spec["mean"], spec["scatter"], target, stats, actual[q], observed[q],
                                absolute[q], norm2[q], seed, law=spec["law"], weights=spec["weights"], samples=SAMPLES)
                            out.update(mean_u=spec["mean"], actual_u=target, scatter_u=spec["scatter"],
                                       brier=np.square(out["p_null"]-(actual[q] <= 0)))
                            save_npz(path, ids=data["ids"][q], actual=actual[q], **out)
                        for offset in OFFSETS:
                            mc_path = folder/(name+f"_mc{offset}.npz")
                            if not mc_path.exists():
                                update("decision_mc", fold=f, half=h, arm=name, seed_offset=offset)
                                moments = extra_seed_moments(spec["mean"], spec["scatter"], stats,
                                    law=spec["law"], weights=spec["weights"], seed=seed+offset)
                                save_npz(mc_path, ids=data["ids"][q], **moments)
                    direct_cell(data, actual, stats, parts, f, h, folder, update)
                    write_json(folder/"complete.json", dict(complete=True, fold=f, half=h,
                        arms=list(ARMS), query_ids=data["ids"][q], samples=SAMPLES))
            update("aggregate")
            aggregate(data, metadata, cfg)
            write_json(ROOT/"status.json", dict(state="COMPLETE", elapsed_seconds=time.monotonic()-started,
                report=str(REPORT/"REPORT.md"), mean_fits_performed=0, cells=10, samples=SAMPLES))
            print("COMPLETE", flush=True)
        except Exception:
            write_json(ROOT/"status.json", dict(state="FAILED", elapsed_seconds=time.monotonic()-started,
                traceback=traceback.format_exc()))
            raise


def launch():
    prepare()
    record = REPORT/"background_execution.json"
    if record.exists():
        previous = read_json(record)
        try:
            os.kill(previous["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("A recorded background process is still alive")
    if (ROOT/"status.json").exists() and read_json(ROOT/"status.json")["state"] == "COMPLETE":
        print("Already complete; no new process.")
        return
    command = ["/usr/bin/nice", "-n", "10", "/usr/bin/caffeinate", "-i", sys.executable,
               "-u", str(Path(__file__).resolve())]
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS),
        MKL_NUM_THREADS=str(THREADS), VECLIB_MAXIMUM_THREADS=str(THREADS), NUMEXPR_NUM_THREADS=str(THREADS),
        PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    log = REPORT/"cpu_run.log"
    with log.open("ab") as stream:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    payload = dict(pid=child.pid, command=command, log=str(log), output=str(ROOT),
                   started_unix=time.time(), state="LAUNCHED", threads=THREADS, nice=10,
                   mean_fits_performed=0, resume_completed_cells=True)
    write_json(record, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(THREADS)
    with threadpool_limits(limits=THREADS):
        if arguments.launch:
            launch()
        elif arguments.prepare_only:
            _, _, _, _, configuration = prepare()
            print(json.dumps(dict(n=configuration["n"], cells=len(configuration["cells"]),
                samples=configuration["samples"], arms=len(ARMS),
                budget=sum(c["budget"] for c in configuration["cells"])), indent=2))
        else:
            run(aggregate_only=arguments.aggregate_only)
