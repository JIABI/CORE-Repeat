"""M3 amplitude controls, exact saved roles and fixed-list evaluation."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from . import quantile_direct_data as data_access
from .eu_r2_direct_baselines import _fit_platt, _probability, ConstantNullClassifier
from .quantile_direct_evaluation import paired_block_statistics, _json, _csv


NAME = "m3_amplitude_controls_20260922_v1"
ARMS = ("AMP_ASC", "AMP_DESC", "AMP_HISTGB", "DISPERSION_DESC", "CORE", "HISTGB")
NEW = ARMS[:4]
GRID = ((7, 20), (15, 10), (31, 10))
SEED = 20260922
REPEATS = 2000
GAMMA_BOUNDS = (-1.02, .98)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def select(ids, score, k):
    ids = np.asarray(ids).astype(str)
    score = np.asarray(score, float)
    if score.shape != ids.shape or not np.isfinite(score).all():
        raise ValueError("Selection needs finite aligned decision scores")
    chosen = np.lexsort((ids, -score))[:k]
    result = np.zeros(len(ids), bool)
    result[chosen] = True
    return result


def bounded_total(values, weights, bounds):
    """Bounds on a signed finite-population weighted total; shared missing cancels."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    known = np.isfinite(values)
    observed = float(np.sum(values[known] * weights[known]))
    w = weights[~known]
    lo = observed + np.sum(np.where(w >= 0, w*bounds[0], w*bounds[1]))
    hi = observed + np.sum(np.where(w >= 0, w*bounds[1], w*bounds[0]))
    return dict(observed_contribution=observed, lower=float(lo), upper=float(hi))


@lru_cache(maxsize=1)
def profile_scalars(project, dataset):
    """Prepared X amplitude and original three-future-well dispersion."""
    path = Path(project) / data_access._SPEC[dataset][1]
    with np.load(path, allow_pickle=False) as z:
        ids = z["ids"].astype(str)
        y = z["Y"].astype(float)
    norm2 = np.square(y[:, 0]).mean(1)
    future = y[:, 1:]
    w = np.square(future-future.mean(1)[:, None]).sum((1, 2))/(2*y.shape[2])
    if np.any(norm2 <= 0) or np.any(w <= 0):
        raise ValueError("Amplitude and dispersion must be positive without added floors")
    return ids, .5*np.log(norm2), np.log(w)


def fit_grid(x, y, xv, yv, seed, *, classifier=False, gamma=False):
    records, best, loss_best, best_index = [], None, np.inf, None
    for i, (leaves, minimum) in enumerate(GRID):
        parameters = dict(max_iter=200, learning_rate=.05, l2_regularization=1.,
            max_leaf_nodes=leaves, min_samples_leaf=minimum, max_bins=255,
            early_stopping=False, random_state=seed)
        start = time.monotonic()
        if classifier and len(np.unique(y)) < 2:
            model = ConstantNullClassifier(float(np.mean(y)))
        else:
            cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
            model = cls(**parameters).fit(x, y)
        fit_seconds = time.monotonic()-start
        if classifier:
            predicted = _probability(model, xv)
            loss = float(log_loss(yv, predicted, labels=[0, 1]))
        else:
            predicted = model.predict(xv)
            if gamma:
                predicted = np.clip(predicted, *GAMMA_BOUNDS)
            loss = float(np.mean(np.square(predicted-yv)))
        records.append(dict(index=i, parameters=parameters, validation_loss=loss,
                            fit_seconds=fit_seconds))
        if best is None or (loss < loss_best and not np.isclose(loss, loss_best, atol=1e-12, rtol=1e-12)):
            best, loss_best, best_index = model, loss, i
    return best, dict(selected_index=best_index, validation_loss=loss_best,
        criterion="binary log loss" if classifier else "bounded Gamma MSE" if gamma else "log W MSE",
        candidates=records, fit_scope="TRAIN+REF_FIT", selection_scope="VALIDATION")


def fit_amplitude(a, y, av, yv, ac, yc, aq, seed):
    reg, rmeta = fit_grid(a[:, None], y, av[:, None], yv, seed, gamma=True)
    clf, cmeta = fit_grid(a[:, None], y <= 0, av[:, None], yv <= 0, seed, classifier=True)
    start = time.monotonic()
    platt, pmeta = _fit_platt(_probability(clf, ac[:, None]), yc <= 0, seed)
    calibration_seconds = time.monotonic()-start
    start = time.monotonic()
    expected = np.clip(reg.predict(aq[:, None]), *GAMMA_BOUNDS)
    raw = _probability(clf, aq[:, None])
    probability = platt.predict(raw)
    prediction_seconds = time.monotonic()-start
    return dict(regression=reg, classifier=clf, platt=platt), dict(
        regression=rmeta, classifier=cmeta, platt=pmeta,
        calibration_seconds=calibration_seconds, prediction_seconds=prediction_seconds), dict(
        expected=expected, p_null_raw=raw, p_null=probability,
        score=expected-.2*probability)


def fit_cell(project, run_root, record):
    folder = Path(run_root) / record["dataset"] / f"cell_{record['cell_index']}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"complete.json").exists():
        return json.loads((folder/"complete.json").read_text())
    start = time.monotonic()
    _json(folder/"status.json", dict(state="running", started=now(), **record))
    with threadpool_limits(limits=1):
        cell = data_access.load_cell(project, record["dataset"], record["cell_index"])
        ids, amplitude, logw = profile_scalars(str(project), record["dataset"])
        index = {oid: i for i, oid in enumerate(ids)}
        rows = {role: np.asarray([index[x] for x in cell["ids_"+role]])
                for role in ("train", "valid", "cal", "query")}
        t, v, c, q = (rows[r] for r in ("train", "valid", "cal", "query"))
        amp_models, amp_meta, amp = fit_amplitude(amplitude[t], cell["y_train"],
            amplitude[v], cell["y_valid"], amplitude[c], cell["y_cal"], amplitude[q], cell["seed"])
        disp, dmeta = fit_grid(cell["x_train"], logw[t], cell["x_valid"], logw[v], cell["seed"])
        stamp = time.monotonic()
        dispersion = disp.predict(cell["x_query"])
        dmeta["prediction_seconds"] = time.monotonic()-stamp
        joblib.dump(dict(amplitude=amp_models, dispersion=disp), folder/"models.joblib")
        n, k = len(q), cell["query_k"]
        scores = dict(AMP_ASC=-amplitude[q], AMP_DESC=amplitude[q], AMP_HISTGB=amp["score"],
                      DISPERSION_DESC=dispersion)
        masks = {arm: select(cell["ids_query"], score, k) for arm, score in scores.items()}
        for arm, original in (("CORE", "core"), ("HISTGB", "histgb")):
            source = cell["originals"][original]
            masks[arm] = source["selected"]
            scores[arm] = source["expected"]-.2*source["p_null"]
        _json(folder/"selections_frozen.json", dict(frozen_at=now(), dataset=record["dataset"],
            cell=record["cell_index"], k=k, selected_ids={arm: cell["ids_query"][m].tolist()
                for arm, m in masks.items()}, query_outcomes_used_for_selection=False))
        arrays = dict(ids=cell["ids_query"], groups=cell["groups_query"], layout=cell["layout_query"],
            cell=np.full(n, record["cell_index"], int), actual=cell["y_query"], amplitude=amplitude[q],
            log_W=logw[q], predicted_log_W=dispersion, random_inclusion=np.full(n, k/n),
            AMP_HISTGB_expected=amp["expected"], AMP_HISTGB_p_null=amp["p_null"],
            AMP_HISTGB_p_null_raw=amp["p_null_raw"])
        for arm in ARMS:
            arrays[arm+"_selected"] = masks[arm]
            arrays[arm+"_score"] = scores[arm]
        for arm, original in (("CORE", "core"), ("HISTGB", "histgb")):
            arrays[arm+"_expected"] = cell["originals"][original]["expected"]
            arrays[arm+"_p_null"] = cell["originals"][original]["p_null"]
        np.savez_compressed(folder/"query_predictions.npz", **arrays)
        _json(folder/"fit_metadata.json", dict(record=record, source_paths=cell["source_paths"],
            seed=cell["seed"], amplitude=amp_meta, dispersion=dmeta,
            ids={role: cell["ids_"+role].tolist() for role in
                 ("model_train", "ref", "train", "valid", "cal", "query")},
            dimensions=dict(amplitude=1, dispersion=cell["full_input_dimension"])))
        completion = dict(dataset=record["dataset"], cell_index=record["cell_index"], state="complete",
            completed_at=now(), wall_seconds=time.monotonic()-start,
            fit_seconds=sum(x["fit_seconds"] for m in (amp_meta["regression"], amp_meta["classifier"], dmeta)
                            for x in m["candidates"]),
            prediction_seconds=amp_meta["prediction_seconds"]+dmeta["prediction_seconds"],
            peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == "darwin" else 1024),
            counts=record["counts"], action_wells=2*k,
            amplitude_setup_wells=4*sum(record["counts"][r] for r in ("train", "valid", "cal")),
            dispersion_setup_wells=4*sum(record["counts"][r] for r in ("train", "valid")))
        _json(folder/"complete.json", completion)
        _json(folder/"status.json", completion)
    return completion


def development_summary(arrays, dataset, cell="ALL"):
    y = arrays["actual"]
    null = y <= 0
    rows = []
    masks = {arm: arrays[arm+"_selected"].astype(float) for arm in ARMS}
    masks["RANDOM_EXPECTATION"] = arrays["random_inclusion"]
    for arm, selected in masks.items():
        k = float(selected.sum())
        total = float(selected@y)
        risk = float(selected@null)
        row = dict(dataset=dataset, cell=cell, arm=arm, n=len(y), selected=k,
            action_wells=2*k, total_Gamma=total, selected_mean_Gamma=total/k,
            value_per_candidate=total/len(y), NULL=risk, NULL_fraction=risk/k,
            random_total_Gamma=float(arrays["random_inclusion"]@y),
            excess_over_random=total-float(arrays["random_inclusion"]@y))
        if arm+"_expected" in arrays:
            mu, p = arrays[arm+"_expected"], arrays[arm+"_p_null"]
            row.update(gamma_mse=float(np.mean((mu-y)**2)), null_brier=float(np.mean((p-null)**2)),
                predicted_selected_NULL=float(selected@p), selected_brier=float(selected@((p-null)**2)/k))
        if arm == "DISPERSION_DESC":
            row["log_W_mse"] = float(np.mean((arrays["predicted_log_W"]-arrays["log_W"])**2))
        rows.append(row)
    return rows


def overlap_rows(arrays, dataset):
    result = []
    for left, right in itertools.combinations(ARMS, 2):
        a, b = arrays[left+"_selected"].astype(bool), arrays[right+"_selected"].astype(bool)
        result.append(dict(dataset=dataset, left=left, right=right, intersection=int((a&b).sum()),
            union=int((a|b).sum()), symmetric_difference=int((a!=b).sum()),
            fraction_of_left=float((a&b).sum()/a.sum())))
    return result


def development_contrasts(arrays, dataset):
    y, null = arrays["actual"], (arrays["actual"] <= 0).astype(float)
    masks = {arm: arrays[arm+"_selected"].astype(float) for arm in ARMS}
    masks["RANDOM_EXPECTATION"] = arrays["random_inclusion"]
    pairs = [(left, right) for left in NEW for right in ("CORE", "HISTGB", "RANDOM_EXPECTATION")]
    pairs += [("AMP_HISTGB", right) for right in ("AMP_ASC", "AMP_DESC", "DISPERSION_DESC")]
    names, contributions = [], []
    for left, right in pairs:
        weight = masks[left]-masks[right]
        for metric, actual in (("value_per_candidate", y), ("NULL_per_candidate", null)):
            names.append(dict(dataset=dataset, left=left, right=right, metric=metric))
            contributions.append(weight*actual)
    matrix = np.column_stack(contributions)
    result = []
    for block in ("groups", "layout"):
        stats = paired_block_statistics(matrix, np.ones_like(matrix), arrays[block],
                                       replicates=REPEATS, seed=SEED)
        result += [dict(**name, block="chemical_group" if block == "groups" else block,
                       **stat, fixed_lists=True) for name, stat in zip(names, stats)]
    return result


def aggregate_development(project, run_root, report_root):
    summaries, units, intervals, overlaps, costs = [], [], [], [], []
    for dataset in data_access.DATASETS:
        paths = sorted((Path(run_root)/dataset).glob("cell_*/query_predictions.npz"))
        expected = data_access._SPEC[dataset][2]
        if len(paths) != expected:
            raise ValueError(f"{dataset}: {len(paths)} cells, expected {expected}")
        parts = []
        for path in paths:
            with np.load(path) as z:
                a = {key: z[key] for key in z.files}
            parts.append(a)
            units += development_summary(a, dataset, int(a["cell"][0]))
            costs.append(json.loads((path.parent/"complete.json").read_text()))
        arrays = {key: np.concatenate([a[key] for a in parts]) for key in parts[0]}
        if len(set(arrays["ids"])) != len(arrays["ids"]):
            raise ValueError("Duplicate evaluation identity")
        root = Path(report_root)/dataset
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root/"all_query_predictions.npz", **arrays)
        per_object = pd.DataFrame({key: value for key, value in arrays.items() if value.ndim == 1})
        per_object.to_csv(root/"per_object.csv", index=False)
        current = development_summary(arrays, dataset)
        _csv(root/"summary.csv", current)
        summaries += current
        intervals += development_contrasts(arrays, dataset)
        overlaps += overlap_rows(arrays, dataset)
    _csv(Path(report_root)/"development_summary.csv", summaries)
    _csv(Path(report_root)/"deployment_units.csv", units)
    _csv(Path(report_root)/"paired_intervals.csv", intervals)
    _csv(Path(report_root)/"selection_overlap.csv", overlaps)
    _csv(Path(report_root)/"compute_and_resource_costs.csv", costs)
    return summaries


def confirmation(project, run_root, report_root):
    """New post-hoc controls; every original R4 input/output remains read-only."""
    project, run_root, report_root = map(Path, (project, run_root, report_root))
    folder = run_root/"confirmation"
    out = report_root/"confirmation"
    folder.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    freeze_path = folder/"selections_frozen.npz"
    if not freeze_path.exists():
        ids, amp, _ = profile_scalars(str(project), "EU")
        saved = data_access._saved(str(project), "EU")
        np.testing.assert_array_equal(ids, saved["ids"])
        y = saved["original"]["core"]["actual"]
        parts_path = project/"runs/r4_confirmation_20260921_v1/model/partitions.json"
        parts = json.loads(parts_path.read_text())
        lookup = {oid: i for i, oid in enumerate(ids)}
        rows = {role: np.asarray([lookup[x] for x in values], int) for role, values in parts.items()}
        t = np.r_[rows["TRAIN"], rows["REF_FIT"]]
        v, c = rows["VALIDATION"], rows["DIST_CAL"]
        qpath = project/"runs/r4_confirmation_20260921_v1/predictions/query_metadata.npz"
        with np.load(qpath) as z:
            qids, norm2 = z["ids"].astype(str), z["norm2_per_feature"]
        qa = .5*np.log(norm2)
        if len(qids) != 1527 or not np.isfinite(qa).all():
            raise ValueError("Unexpected eligible confirmation inputs")
        with threadpool_limits(limits=1):
            models, metadata, prediction = fit_amplitude(amp[t], y[t], amp[v], y[v], amp[c], y[c], qa, 20260921)
        joblib.dump(models, folder/"amplitude_models.joblib")
        scores = dict(AMP_ASC=-qa, AMP_DESC=qa, AMP_HISTGB=prediction["score"])
        arrays = dict(ids=qids, amplitude=qa, AMP_HISTGB_expected=prediction["expected"],
                      AMP_HISTGB_p_null=prediction["p_null"])
        for arm, score in scores.items():
            arrays[arm+"_score"] = score
            arrays[arm+"_selected"] = select(qids, score, 192)
        np.savez_compressed(freeze_path, **arrays)
        _json(folder/"selections_frozen.json", dict(frozen_at=now(), post_hoc=True,
            population=1539, eligible=1527, k=192, query_outcomes_used_for_fitting=False,
            source_query=str(qpath), source_partitions=str(parts_path), fitting_ids=parts,
            selected_ids={arm: qids[arrays[arm+"_selected"]].tolist() for arm in scores},
            fit_metadata=metadata))
    with np.load(freeze_path) as z:
        arrays = {key: z[key] for key in z.files}
    dispersion = confirmation_dispersion(project, folder, arrays["ids"])
    arrays.update(dispersion)
    base = project/"reports/r4_execution_20260921_v1"
    objects = pd.read_csv(base/"primary/campaign/object_results.csv")
    external = pd.read_csv(base/"external/evaluation/endpoint_table.csv")
    frame = objects[objects.eligible_x].merge(external, left_on="object_id", right_on="ids",
                                              validate="one_to_one", suffixes=("", "_external"))
    frame = frame.set_index("object_id").loc[arrays["ids"]].reset_index()
    if len(frame) != 1527:
        raise ValueError("Unexpected eligible evaluation population")
    for key in ("AMP_ASC_selected", "AMP_DESC_selected", "AMP_HISTGB_selected",
                "AMP_HISTGB_expected", "AMP_HISTGB_p_null", "amplitude",
                "DISPERSION_DESC_selected", "DISPERSION_DESC_score"):
        frame[key] = arrays[key]
    frame["HISTGB_selected"] = frame["HISTGB_CAL_selected"]
    arms = ("AMP_ASC", "AMP_DESC", "AMP_HISTGB", "DISPERSION_DESC", "CORE", "HISTGB")
    masks = {arm: frame[arm+"_selected"].to_numpy(float) for arm in arms}
    masks["RANDOM_EXPECTATION"] = np.full(len(frame), 192/len(frame))
    gamma = frame.gamma.to_numpy(float)
    null = np.where(np.isfinite(gamma), (gamma <= 0).astype(float), np.nan)
    endpoint = {"Gamma": (gamma, GAMMA_BOUNDS), "NULL": (null, (0., 1.)),
        "paired_cross_site": (frame.delta.to_numpy(float), (-2., 2.)),
        "MEDINA": (frame.MEDINA_delta.to_numpy(float), (-2., 2.)),
        "USC": (frame.USC_delta.to_numpy(float), (-2., 2.))}
    rows = []
    for arm, selected in masks.items():
        k = selected.sum()
        for name, (values, bounds) in endpoint.items():
            known = np.isfinite(values)
            observed_weight = selected[known].sum()
            result = bounded_total(values, selected, bounds)
            rows.append(dict(arm=arm, endpoint=name, population=1539, eligible=1527, selected=k,
                observed_selected=observed_weight, missing_selected=k-observed_weight,
                available_case_mean=result["observed_contribution"]/observed_weight,
                total_lower=result["lower"], total_upper=result["upper"],
                fixed_list_mean_lower=result["lower"]/k, fixed_list_mean_upper=result["upper"]/k,
                per_candidate_lower=result["lower"]/1539, per_candidate_upper=result["upper"]/1539,
                observed_sum=result["observed_contribution"], post_hoc=arm in NEW))
    _csv(out/"summary.csv", rows)
    frame.to_csv(out/"per_object.csv", index=False)
    pairs = [(left, right) for left in NEW for right in ("CORE", "HISTGB", "RANDOM_EXPECTATION")]
    pairs += [("AMP_HISTGB", "AMP_ASC"), ("AMP_HISTGB", "AMP_DESC"), ("AMP_HISTGB", "DISPERSION_DESC")]
    contrasts, names, contributions = [], [], []
    for left, right in pairs:
        weights = masks[left]-masks[right]
        for name in ("Gamma", "NULL", "paired_cross_site"):
            values, bounds = endpoint[name]
            result = bounded_total(values, weights, bounds)
            contrasts.append(dict(left=left, right=right, endpoint=name, **result,
                per_candidate_lower=result["lower"]/1539,
                per_candidate_upper=result["upper"]/1539,
                mean_selected_difference_lower=result["lower"]/192,
                mean_selected_difference_upper=result["upper"]/192))
            known = np.isfinite(values)
            for edge, lower in (("lower", True), ("upper", False)):
                fill = np.where(weights >= 0, bounds[0 if lower else 1], bounds[1 if lower else 0])
                contribution = weights*np.where(known, values, fill)
                contributions.append(contribution)
                names.append(dict(left=left, right=right, endpoint=name, bound=edge))
    _csv(out/"paired_identification_bounds.csv", contrasts)
    # Include metadata-qualified missing-X candidates as zero policy contributions.
    index = {oid: i for i, oid in enumerate(objects.object_id.astype(str))}
    all_rows = np.asarray([index[oid] for oid in frame.object_id.astype(str)])
    matrix = np.zeros((len(objects), len(contributions)))
    matrix[all_rows] = np.column_stack(contributions)
    intervals = []
    for block in ("group", "layout"):
        stats = paired_block_statistics(matrix, np.ones_like(matrix), objects[block].astype(str),
                                       replicates=REPEATS, seed=SEED)
        intervals += [dict(**name, block="chemical_group" if block == "group" else "layout", **stat,
                           fixed_lists=True, estimand="bound_endpoint_per_qualified_candidate")
                      for name, stat in zip(names, stats)]
    _csv(out/"paired_intervals.csv", intervals)
    overlap = []
    for left, right in itertools.combinations(arms, 2):
        a, b = masks[left].astype(bool), masks[right].astype(bool)
        overlap.append(dict(left=left, right=right, intersection=int((a&b).sum()),
                            symmetric_difference=int((a!=b).sum()), fraction_of_left=float((a&b).sum()/192)))
    _csv(out/"selection_overlap.csv", overlap)
    _json(out/"metadata.json", dict(completed=now(), post_hoc_controls=list(NEW),
        Gamma="0.5 cosine improvement minus 0.02; NULL=Gamma<=0",
        paired_endpoint="mean MEDINA/USC neighbourhood-Spearman improvement, both sites observed",
        random_reference="inclusion probability 192/1527 across all eligible X objects",
        intervals="identification bounds separate from block-resampling sensitivity",
        original_R4_artifacts_modified=False,
        setup_wells=dict(AMP_ASC=0, AMP_DESC=0, AMP_HISTGB=4*904, DISPERSION_DESC=4*723), action_wells=384))
    return rows


def confirmation_dispersion(project, folder, qids):
    """Declared extension using original final preprocessing and DEV roles."""
    from .gram_oof_ridge import transform_input
    project, folder = Path(project), Path(folder)
    path = folder/"dispersion_selection_frozen.npz"
    if path.exists():
        with np.load(path) as z:
            np.testing.assert_array_equal(z["ids"], qids)
            return {key: z[key] for key in z.files if key != "ids"}
    extension = project/"protocols/m3_amplitude_controls_20260922_confirmation_extension.md"
    _json(folder/"dispersion_protocol_frozen.json", dict(frozen_at=now(), protocol=extension.read_text()))
    ids, _, logw = profile_scalars(str(project), "EU")
    first, chemistry = data_access._first_well(str(project), "EU")
    base = project/"runs/r4_confirmation_20260921_v1"
    parts = json.loads((base/"model/partitions.json").read_text())
    lookup = {oid: i for i, oid in enumerate(ids)}
    t = np.asarray([lookup[x] for r in ("TRAIN", "REF_FIT") for x in parts[r]])
    v = np.asarray([lookup[x] for x in parts["VALIDATION"]])
    stats = json.loads((base/"model/mean/preprocessing.json").read_text())
    x = np.column_stack((transform_input(first, stats), chemistry))
    with np.load(base/"ingest/x/query.npz") as z:
        ix = {oid: i for i, oid in enumerate(z["ids"].astype(str))}
        rows = np.asarray([ix[oid] for oid in qids])
        if not z["eligible"][rows].all():
            raise ValueError("Dispersion query includes an ineligible first well")
        xq = np.column_stack((transform_input(z["X"][rows], stats), z["chem"][rows]))
    with threadpool_limits(limits=1):
        model, metadata = fit_grid(x[t], logw[t], x[v], logw[v], 20260921)
        score = model.predict(xq)
    mask = select(qids, score, 192)
    joblib.dump(model, folder/"dispersion_model.joblib")
    result = dict(DISPERSION_DESC_score=score, DISPERSION_DESC_selected=mask)
    np.savez_compressed(path, ids=qids, **result)
    _json(folder/"dispersion_selection_frozen.json", dict(frozen_at=now(), post_hoc=True,
        query_outcomes_used_for_fitting=False, selected_ids=qids[mask].tolist(), fit_metadata=metadata,
        input_dimension=x.shape[1], direction="descending", target="log W",
        partitions={r: parts[r] for r in ("TRAIN", "REF_FIT", "VALIDATION")},
        preprocessing=str(base/"model/mean/preprocessing.json")))
    return result
