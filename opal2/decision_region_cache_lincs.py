"""Honest decision-region inputs for the ten saved LINCS R2 cells.

No mean model or direct estimator is fitted here. QUERY predictions are copied
from each original R2 result. CAL predictions hold out entire chemical groups
inside that cell's original DIST_CAL, not another cell's outer predictions.

Original LINCS CORE differs from the R2 STATE_REF ablation: its inherited base,
LOCAL_SCALE choice and AMPLITUDE_TOTAL fit precede DIST_CAL and use fixed REF.
Only the empirical radial law and its CAL donor weights depend on DIST_CAL.
Those pieces are rebuilt for each inner fold; the original scatter is reused.
This module returns arrays and JSON-compatible provenance, and writes no files.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .empirical_radial import fit_radial, reference_weights
from .eu_r2_direct_baselines import _bounded_prediction, _probability
from .gram_oof_ridge import transform_input
from .joint_contrast_scale import predict_scale


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "runs/lincs_r2_completion_20260918_v1"
ARM_FILES = {
    "CORE": "CORE_ORIGINAL",
    "HISTGB_CAL": "DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL",
    "HISTGB_COHERENT": "DIRECT_ACCESS_MATCHED_HISTGB_COHERENT",
    "EXTRATREES_CAL": "DIRECT_ACCESS_MATCHED_EXTRATREES_CLASSIFIER_CAL",
    "EXTRATREES_COHERENT": "DIRECT_ACCESS_MATCHED_EXTRATREES_COHERENT",
}


def _json(path):
    return json.loads(Path(path).read_text())


def _npz(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (archive.files if keys is None else keys)}


def _dataset(dataset):
    if dataset != "LINCS":
        raise ValueError("This adapter accepts only dataset='LINCS'")


def _rows(ids, requested):
    ids, requested = np.asarray(ids, str), np.asarray(requested, str)
    lookup = {value: i for i, value in enumerate(ids)}
    if len(lookup) != len(ids) or len(np.unique(requested)) != len(requested):
        raise ValueError("Duplicate object identities")
    try:
        return np.asarray([lookup[value] for value in requested], dtype=int)
    except KeyError as exc:
        raise ValueError("Requested object is absent from saved scope") from exc


def _roles(ids, groups, cell):
    parts = {role: _rows(ids, cell["ids"][role]) for role in
             ("TRAIN", "VALIDATION", "REF_FIT", "DIST_CAL", "DEV_EVAL")}
    if any(len(rows) == 0 for rows in parts.values()):
        raise ValueError("Empty original LINCS role")
    if sum(map(len, parts.values())) != len(ids):
        raise ValueError("Original LINCS roles do not cover the complete population")
    names = list(parts)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if set(groups[parts[left]]) & set(groups[parts[right]]):
                raise ValueError(f"Chemical group crosses {left}/{right}")
    for role, rows in parts.items():
        if len(rows) != cell["counts"][role]:
            raise ValueError("Original role size differs from its manifest")
    return parts


def list_cells(dataset="LINCS") -> list[str]:
    """List only the completed, originally declared LINCS deployment cells."""
    _dataset(dataset)
    manifest = _json(ROOT / "run_manifest.json")
    cells = [f"cell_{cell['fold']}_{cell['half']}" for cell in manifest["cells"]]
    if len(cells) != 10 or len(set(cells)) != 10:
        raise ValueError("Expected the original ten LINCS cells")
    return cells


def _load_cell(cell_name):
    manifest = _json(ROOT / "run_manifest.json")
    matches = [cell for cell in manifest["cells"]
               if cell_name == f"cell_{cell['fold']}_{cell['half']}"]
    if len(matches) != 1:
        raise ValueError("Unknown original LINCS cell: " + str(cell_name))
    cell = matches[0]
    folder = ROOT / cell_name
    complete = _json(folder / "complete.json")
    if not complete.get("complete") or complete["samples"] != 100000:
        raise ValueError("The original 100k-query cell is not complete")
    sources = {key: Path(manifest[key]) for key in
               ("state_source", "distribution_source", "reference_source", "data_directory")}
    state = _json(sources["state_source"] / "run_manifest.json")
    if state["final_opened"] or state["fifth_repeat_opened"]:
        raise ValueError("Expected the existing development-only LINCS scope")
    data = _npz(sources["data_directory"] / "data.npz", ("ids", "groups", "Y", "chem"))
    ids, groups = data["ids"], data["groups"]
    if ids.tolist() != state["ids"] or groups.tolist() != state["groups"]:
        raise ValueError("Saved model/data chemical identity order changed")
    parts = _roles(ids, groups, cell)
    cal, query = parts["DIST_CAL"], parts["DEV_EVAL"]
    metadata = _json(sources["data_directory"] / "metadata.json")
    if metadata["ids"] != ids.tolist() or len(metadata["units"]) != len(ids):
        raise ValueError("LINCS metadata order changed")
    layout = np.asarray([unit["layout_block"] for unit in metadata["units"]], str)

    fold, half = cell["fold"], cell["half"]
    stats_path = sources["state_source"] / "folds" / f"fold_{fold}" / "preprocessing.json"
    stats = _json(stats_path)
    mean_path = ROOT / f"fold_{fold}" / "frozen_means.npz"
    means = _npz(mean_path, ("ids", "STATE_REF", "actual_u"))
    mean_rows = _rows(means["ids"], ids[cal])
    cal_mean_u = means["STATE_REF"][mean_rows]
    original = _json(sources["distribution_source"] / "summary.json")
    original = next(item for item in original["cells"]
                    if item["fold"] == fold and item["half"] == half)
    for key, role in (("fit_ids", "REF_FIT"), ("calibration_ids", "DIST_CAL"),
                      ("query_ids", "DEV_EVAL")):
        np.testing.assert_array_equal(original[key], ids[parts[role]])
    ref_path = sources["reference_source"] / f"cell_{fold}_{half}_reference.npz"
    ref = _npz(ref_path, ("fit_ids", "cal_ids", "query_ids", "cal_residual", "cal_covariance"))
    for key, role in (("fit_ids", "REF_FIT"), ("cal_ids", "DIST_CAL"), ("query_ids", "DEV_EVAL")):
        np.testing.assert_array_equal(ref[key], ids[parts[role]])
    np.testing.assert_allclose(ref["cal_residual"], means["actual_u"][mean_rows] - cal_mean_u,
                               atol=1e-12, rtol=1e-12)
    log_amplitude = np.log(np.linalg.norm(data["Y"][cal, 0], axis=1))
    scatter = ref["cal_covariance"] * predict_scale(original["amplitude_fit"], log_amplitude)[:, None, None]
    radial_path = sources["distribution_source"] / f"cell_{fold}_{half}_radial.npz"
    radial = _npz(radial_path, ("cal_ids", "cal_amp_scatter", "amplitude_radii"))
    reps = _rows(ids[cal], original["representative_ids"])
    np.testing.assert_array_equal(radial["cal_ids"], ids[cal][reps])
    np.testing.assert_array_equal(radial["cal_amp_scatter"], scatter[reps])
    radii = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(scatter[reps]),
                                         ref["cal_residual"][reps, :, None])[..., 0], axis=1)
    np.testing.assert_array_equal(radial["amplitude_radii"], radii)

    # This aggregate supplies outcomes/identities only, never pooled OOF predictions.
    actual_path = ROOT / "CORE_ORIGINAL.npz"
    outcomes = _npz(actual_path, ("ids", "groups", "actual"))
    np.testing.assert_array_equal(outcomes["ids"], ids)
    np.testing.assert_array_equal(outcomes["groups"], groups)
    original_query = {}
    for arm, filename in ARM_FILES.items():
        saved = _npz(folder / f"{filename}.npz", ("ids", "actual", "predicted", "p_null"))
        np.testing.assert_array_equal(saved["ids"], ids[query])
        np.testing.assert_array_equal(saved["actual"], outcomes["actual"][query])
        original_query[arm] = dict(query_p=saved["p_null"], query_mean=saved["predicted"])
    mc_offsets = list(manifest.get("extra_mc_offsets", []))
    mc_paths = [folder / f"CORE_ORIGINAL_mc{offset}.npz" for offset in mc_offsets]
    mc_available = len(mc_paths) == 2 and all(path.is_file() for path in mc_paths)
    if mc_available:
        mc = [_npz(path, ("ids", "predicted", "p_null")) for path in mc_paths]
        for values in mc:
            np.testing.assert_array_equal(values["ids"], ids[query])
        original_query["CORE"].update(query_seed_mean=np.stack([v["predicted"] for v in mc]),
                                      query_seed_p=np.stack([v["p_null"] for v in mc]))
    np.testing.assert_array_equal(complete["query_ids"], ids[query])
    return dict(cell=cell, folder=folder, manifest=manifest, ids=ids, groups=groups,
                parts=parts, cal_mean_u=cal_mean_u, cal_scatter=scatter,
                cal_residual=ref["cal_residual"], cal_log_amplitude=log_amplitude,
                radial_bandwidth=original["local_bandwidth"], stats=stats,
                cal_x=np.column_stack((transform_input(data["Y"][cal, 0], stats), data["chem"][cal])),
                cal_actual=outcomes["actual"][cal], query_actual=outcomes["actual"][query],
                query_layout=layout[query], original_query=original_query,
                query_mc_seed_offsets=mc_offsets if mc_available else [],
                sources=dict(manifest=str(ROOT / "run_manifest.json"), mean=str(mean_path),
                    fixed_ref_scatter=str(ref_path), original_radial=str(radial_path),
                    preprocessing=str(stats_path), actual_only=str(actual_path),
                    core_query_mc=[str(path) for path in mc_paths] if mc_available else [],
                    query_predictions={arm: str(folder / f"{name}.npz") for arm, name in ARM_FILES.items()}))


def crossfit_core_radial(ids, groups, mean_u, scatter, residual, log_amplitude,
                         radial_bandwidth, stats, inner_fold, *, samples, seed):
    """Predict CAL moments without its held fold entering the empirical law.

    ``scatter`` must already be fitted entirely outside DIST_CAL, as it is in
    original LINCS CORE. No CAL-dependent prefit scatter is accepted by intent.
    The caller audits that provenance before entering this numerical interface.
    """
    from .decision_region_cache_standard import integrate_core_moments

    ids, groups = np.asarray(ids, str), np.asarray(groups, str)
    inner_fold = np.asarray(inner_fold)
    mean_u, scatter, residual = map(np.asarray, (mean_u, scatter, residual))
    log_amplitude = np.asarray(log_amplitude, float)
    n = len(ids)
    if (len(np.unique(ids)) != n or groups.shape != (n,) or inner_fold.shape != (n,)
            or mean_u.shape != (n, 9) or residual.shape != (n, 9)
            or scatter.shape != (n, 9, 9) or log_amplitude.shape != (n,)
            or len(np.unique(inner_fold)) < 2):
        raise ValueError("Invalid aligned inner-CAL arrays")
    for group in np.unique(groups):
        if len(np.unique(inner_fold[groups == group])) != 1:
            raise ValueError("One CAL chemical group crosses inner folds")
    output = dict(cal_p=np.full(n, np.nan), cal_mean=np.full(n, np.nan))
    audit = []
    for fold in np.unique(inner_fold):
        held = np.flatnonzero(inner_fold == fold)
        donors = np.flatnonzero(inner_fold != fold)
        representatives = np.asarray([min(donors[groups[donors] == group], key=lambda i: ids[i])
                                      for group in np.unique(groups[donors])], int)
        radii = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(scatter[representatives]),
            residual[representatives, :, None])[..., 0], axis=1)
        law = fit_radial(radii, dimension=9)
        weights = reference_weights(log_amplitude[representatives], log_amplitude[held],
                                    radial_bandwidth, conditional=True)["weights"]
        moments = integrate_core_moments(mean_u[held], scatter[held], stats, law=law,
                                         weights=weights, samples=samples, seed=seed + int(fold))
        output["cal_p"][held] = moments["p_null"]
        output["cal_mean"][held] = moments["predicted"]
        audit.append(dict(inner_fold=int(fold), held_ids=ids[held].tolist(),
            held_groups=np.unique(groups[held]).tolist(), fit_ids=ids[donors].tolist(),
            fit_groups=np.unique(groups[donors]).tolist(), radial_representative_ids=ids[representatives].tolist(),
            held_rows=len(held), fit_rows=len(donors), fit_group_count=len(representatives),
            radial_bandwidth=float(law["bandwidth"]), seed=int(seed + fold)))
    if not all(np.isfinite(value).all() for value in output.values()):
        raise ValueError("Incomplete or nonfinite honest CAL moments")
    output["audit"] = audit
    return output


def build_cell(dataset, cell_name, *, samples=100000, seed=20260920) -> dict:
    """Build one cell in memory; QUERY remains exactly the stored R2 output."""
    from .decision_region_cache_standard import grouped_calibration_folds, crossfit_direct_predictions

    _dataset(dataset)
    if not isinstance(samples, (int, np.integer)) or isinstance(samples, bool) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    started = time.perf_counter()
    context = _load_cell(cell_name)
    load_seconds = time.perf_counter() - started
    cell, ids, groups, parts = (context[key] for key in ("cell", "ids", "groups", "parts"))
    cal, query = parts["DIST_CAL"], parts["DEV_EVAL"]
    cell_seed = int(seed) + 100 * int(cell["fold"]) + int(cell["half"])
    inner_fold = grouped_calibration_folds(groups[cal], cell_seed, n_folds=5)
    begun = time.perf_counter()
    with threadpool_limits(limits=1):
        core = crossfit_core_radial(ids[cal], groups[cal], context["cal_mean_u"],
            context["cal_scatter"], context["cal_residual"], context["cal_log_amplitude"],
            context["radial_bandwidth"], context["stats"], inner_fold, samples=int(samples), seed=cell_seed)
    core_seconds = time.perf_counter() - begun
    arms = {"CORE": dict(cal_p=core["cal_p"], cal_mean=core["cal_mean"],
                         **context["original_query"]["CORE"])}
    direct_audit = {}
    begun = time.perf_counter()
    with threadpool_limits(limits=1):
        for family in ("HISTGB", "EXTRATREES"):
            model_path = context["folder"] / "direct_access_models" / f"{family}.joblib"
            model = joblib.load(model_path)
            provenance = model["provenance"]
            train = np.r_[parts["TRAIN"], parts["REF_FIT"]]
            if provenance["scope"] != "ACCESS_MATCHED" or provenance["family"] != family:
                raise ValueError("Expected the saved access-matched direct family")
            np.testing.assert_array_equal(provenance["train_ids"], ids[train])
            np.testing.assert_array_equal(provenance["validation_ids"], ids[parts["VALIDATION"]])
            point = _bounded_prediction(model["regression"], context["cal_x"])
            raw = _probability(model["classifier"], context["cal_x"])
            out = crossfit_direct_predictions(point, raw, context["cal_actual"], groups[cal],
                                             inner_fold, seed=cell_seed)
            for interface in ("CAL", "COHERENT"):
                name = family + "_" + interface
                arms[name] = dict(cal_p=out[interface]["cal_p"], cal_mean=out[interface]["cal_mean"],
                                  **context["original_query"][name])
            direct_audit[family] = dict(model_path=str(model_path), estimator_refitted=False,
                                       calibration=out["audit"])
    direct_seconds = time.perf_counter() - begun
    audit = dict(dataset=dataset, cell=cell_name, sources=context["sources"], seed=cell_seed,
        cal_samples=int(samples), query_samples=int(context["manifest"]["samples"]),
        query_mc_seed_offsets=context.get("query_mc_seed_offsets", []),
        query_predictions_exact_saved_R2=True, pooled_outer_predictions_used_for_cal=False,
        cal_counts=dict(rows=len(cal), chemical_groups=len(np.unique(groups[cal])),
                        inner_folds=len(np.unique(inner_fold))),
        cal_dependency=dict(mean="Saved original STATE50 mean; unchanged, no CAL supervision",
            scatter="Original RIDGE-OOF base plus fixed-REF LOCAL_SCALE and AMPLITUDE_TOTAL; no DIST_CAL outcomes",
            radial="Refitted log-radius centers and bandwidth on other inner-CAL groups; X-only donor weights",
            direct="Saved TRAIN+REF estimators; Platt and empirical Gamma residual law exclude held CAL groups"),
        core_inner_folds=core["audit"], direct=direct_audit,
        smaller_inner_calibration="CAL OOF uses four of five chemical-group folds; saved QUERY uses full original DIST_CAL",
        query_budget=int(cell["budget"]), mean_refits=0, direct_estimator_refits=0,
        query_outcomes_used_for_calibration=False, protected_measurements_read=False)
    return dict(dataset=dataset, cell=cell_name, cal_ids=ids[cal].copy(), cal_groups=groups[cal].copy(),
        query_ids=ids[query].copy(), query_groups=groups[query].copy(), query_layout=context["query_layout"],
        cal_actual=context["cal_actual"], query_actual=context["query_actual"], cal_inner_fold=inner_fold,
        query_budget=int(cell["budget"]), arms=arms, audit=audit,
        timing=dict(load_seconds=load_seconds, core_cal_seconds=core_seconds,
                    direct_cal_seconds=direct_seconds, total_seconds=time.perf_counter() - started))
