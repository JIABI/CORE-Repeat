"""Reporting for the development-only conditional-quantile HistGB comparison.

``aggregate(project, run_root, dataset=None)`` reads completed cells at
``run_root/{dataset}/cell_*/query_predictions.npz``. Every cell must contain:
ids, groups, layout (aligned nonempty strings), actual (finite Gamma),
original_k (the saved integer quota), and raw/cal_{expected,p_null,crps}.
raw/cal_{coverage,width} have shape (n, 5), ordered at LEVELS below. CORE and
historical HistGB predictions may be provided as core/histgb_{expected,
p_null,crps,coverage,width}; when absent they are read by exact ID from the
original R2 unified outputs. Optional ``metadata.json`` and ``complete.json``
are retained as provenance. No model is fitted, no outcome is selected for
tuning, and no R4 file is accessed here.

The primary arm is the raw quantile law. CAL-offset quantiles are a separate,
prespecified secondary arm, never an automatically selected replacement.
The historical HistGB decision probability is its calibrated classifier;
its CRPS remains the original companion residual law, not that classifier.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .decision_region_calibration import select_top_k


DATASETS = ("EU", "JUMP", "LINCS", "RxRx3")
EXPECTED = {"EU": (904, 5), "JUMP": (639, 5), "LINCS": (1188, 10), "RxRx3": (10410, 40)}
R2_RUN = {"EU": "r2_core_comparison_20260917_v1", "JUMP": "jump_r2_completion_20260918_v1",
          "LINCS": "lincs_r2_completion_20260918_v1", "RxRx3": "rxrx3_r2_completion_20260918_v1"}
ARMS = ("raw", "cal", "core", "histgb")
LABELS = {"raw": "QUANTILE_RAW_PRIMARY", "cal": "QUANTILE_CAL_SECONDARY",
          "core": "CORE_ORIGINAL", "histgb": "HISTGB_CLASSIFIER_CAL"}
LEVELS = np.array([.5, .8, .9, .95, .99])
SEED = 20260922
BOOTSTRAPS = 2000
LAMBDA = .2


def _clean(value):
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_clean(v) for v in value]
    if isinstance(value, np.ndarray): return _clean(value.tolist())
    if isinstance(value, np.generic): return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(_clean(v)) if isinstance(v, (dict, list, tuple, np.ndarray))
                             else _clean(v) for k, v in row.items()})


def _npz(path, keys=None):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files if keys is None or k in keys}


def _vector(value, name, n=None):
    result = np.asarray(value, float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite nonempty vector")
    if n is not None and len(result) != n: raise ValueError(name + " is not aligned")
    return result


def validate_cell(arrays, name="cell"):
    actual = _vector(arrays["actual"], name + " actual")
    n = len(actual)
    for key in ("ids", "groups", "layout"):
        labels = np.asarray(arrays[key]).astype(str)
        if labels.shape != (n,) or np.any(labels == ""):
            raise ValueError(name + " requires aligned nonempty " + key)
    if len(set(np.asarray(arrays["ids"]).astype(str))) != n:
        raise ValueError(name + " contains duplicate condition IDs")
    k = np.asarray(arrays["original_k"])
    if k.ndim or k.dtype.kind not in "iu" or not 0 < int(k) <= n:
        raise ValueError("original_k must be the saved positive integer quota")
    for arm in ARMS:
        for field in ("expected", "p_null", "crps"):
            key = arm + "_" + field
            value = _vector(arrays[key], name + " " + key, n)
            if field == "p_null" and np.any((value < 0) | (value > 1)):
                raise ValueError(key + " lies outside [0,1]")
            if field == "crps" and np.any(value < -1e-12): raise ValueError("Negative CRPS")
        for field in ("coverage", "width"):
            key = arm + "_" + field
            if key not in arrays: continue
            value = np.asarray(arrays[key], float)
            if value.shape != (n, len(LEVELS)) or not np.isfinite(value).all():
                raise ValueError(key + " must be finite (n, 5)")
            if field == "coverage" and not np.isin(value, (0, 1)).all():
                raise ValueError(key + " must contain binary interval coverage")
            if field == "width" and np.any(value < 0): raise ValueError("Negative interval width")
        for field in ("selected", "selected_lambda0"):
            key = arm+"_"+field
            if key in arrays:
                mask = np.asarray(arrays[key])
                if mask.shape != (n,) or not np.isin(mask, (False, True)).all() or mask.sum() != int(k):
                    raise ValueError(key+" must preserve the original quota")
    return n, int(k)


def _historical(project, dataset):
    folder = Path(project) / "runs" / R2_RUN[dataset]
    keys = {"ids", "actual", "predicted", "p_null", "crps", "gamma_coverage_by_level",
            "gamma_width_by_level", "selected_lambda_0.2", "selected_lambda_0"}
    return {arm: _npz(folder / filename, keys) for arm, filename in
            (("core", "CORE_ORIGINAL.npz"),
             ("histgb", "DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL.npz"))}


def _attach_historical(arrays, historical, cell):
    ids = np.asarray(arrays["ids"]).astype(str)
    arrays = dict(arrays)
    for arm, saved in historical.items():
        old_ids = np.asarray(saved["ids"]).astype(str)
        if len(set(old_ids)) != len(old_ids): raise ValueError("Historical IDs are not unique")
        lookup = {v: i for i, v in enumerate(old_ids)}
        try: rows = np.asarray([lookup[v] for v in ids], int)
        except KeyError as error: raise ValueError("Unknown historical ID") from error
        np.testing.assert_array_equal(arrays["actual"], saved["actual"][rows],
                                      err_msg=cell + " changed historical outcomes")
        for new, old in (("expected", "predicted"), ("p_null", "p_null"), ("crps", "crps"),
                         ("coverage", "gamma_coverage_by_level"), ("width", "gamma_width_by_level")):
            key = arm + "_" + new
            if old not in saved: continue
            original = saved[old][rows]
            if key in arrays:
                np.testing.assert_array_equal(arrays[key], original,
                    err_msg=cell + " changed historical " + key)
            arrays[key] = original
        for suffix, old_key, lam in (("selected", "selected_lambda_0.2", LAMBDA),
                                     ("selected_lambda0", "selected_lambda_0", 0.)):
            if old_key not in saved: continue
            original = saved[old_key][rows].astype(bool)
            if original.sum() != int(arrays["original_k"]):
                raise ValueError(cell+" changed historical quota")
            key = arm+"_"+suffix
            if key in arrays:
                np.testing.assert_array_equal(arrays[key], original,
                                              err_msg=cell+" changed saved historical list")
            arrays[key] = original
            reconstructed = select_top_k(ids, arrays[arm+"_expected"], arrays[arm+"_p_null"],
                                         int(arrays["original_k"]), lam)
            arrays[key+"_reconstruction_mismatches"] = np.asarray(np.sum(reconstructed != original))
    return arrays


def _summary(arrays, mask, arm, dataset, cell="ALL", lam=LAMBDA):
    actual = arrays["actual"][mask]; label = actual <= 0
    expected = arrays[arm+"_expected"][mask]; p = arrays[arm+"_p_null"][mask]
    selected = arrays[arm+("_selected" if lam == LAMBDA else "_selected_lambda0")][mask]
    k = int(selected.sum()); n = len(actual)
    predicted_null = float(p[selected].sum()); actual_null = int(label[selected].sum())
    total = float(actual[selected].sum())
    # For ALL, use each original cell's exact k/n, not a global selection rate.
    random_total = float(np.sum(arrays["random_inclusion"][mask] * actual))
    random_null = float(np.sum(arrays["random_inclusion"][mask] * label))
    probability = np.clip(p, 1e-12, 1-1e-12)
    row = dict(dataset=dataset, cell=cell, arm=LABELS[arm], lambda_value=lam, n=n, selected=k,
        additional_action_wells=2*k, total_Gamma=total, mean_selected_Gamma=total/k,
        value_per_candidate=total/n, actual_NULL=actual_null, expected_NULL=predicted_null,
        FDP=actual_null/k, null_per_candidate=actual_null/n,
        selected_probability_gap=(actual_null-predicted_null)/k,
        selected_brier=float(np.mean((p[selected]-label[selected])**2)),
        brier=float(np.mean((p-label)**2)),
        null_logloss=float(np.mean(-label.astype(float)*np.log(probability)-(~label)*np.log1p(-probability))),
        null_auc=float(roc_auc_score(label, p)) if len(np.unique(label)) == 2 else None,
        gamma_mse=float(np.mean((expected-actual)**2)),
        gamma_crps=float(np.mean(arrays[arm+"_crps"][mask])),
        random_expected_total_Gamma=random_total, random_expected_NULL=random_null,
        gain_minus_random=total-random_total,
        gain_ratio_to_random=total/random_total if random_total > 0 else None)
    for field in ("coverage", "width"):
        key = arm + "_" + field
        if key in arrays:
            for i, level in enumerate(LEVELS):
                row["gamma_"+field+"_"+str(level)] = float(np.mean(arrays[key][mask, i]))
    return row


def paired_block_statistics(contributions, denominators, labels, *, replicates=BOOTSTRAPS, seed=SEED):
    """Shared whole-block bootstrap of named ratio statistics on fixed lists.

    Chemical identities retain one weight across doses and deployment cells.
    All named statistics share the same resamples. Zero denominators are
    counted rather than silently presented as valid replicates. This does not
    refit models or repeat policy selection within a reconstructed campaign.
    """
    values = np.asarray(contributions, float); denominator = np.asarray(denominators, float)
    if values.ndim != 2 or denominator.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("Unaligned finite contribution matrices required")
    if not np.isfinite(denominator).all() or np.any(denominator < 0) or np.any(denominator.sum(0) <= 0):
        raise ValueError("Finite nonnegative, nonzero denominators required")
    labels = np.asarray(labels).astype(str)
    if labels.shape != (len(values),) or np.any(labels == ""): raise ValueError("Unaligned labels")
    unique, index = np.unique(labels, return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1])); totals = np.zeros_like(sums)
    np.add.at(sums, index, values); np.add.at(totals, index, denominator)
    point = sums.sum(0)/totals.sum(0)
    if len(unique) < 2:
        return [dict(difference=float(v), ci95=None, clusters=1, valid_replicates=0,
                     requested_replicates=replicates, invalid_zero_denominator_replicates=0,
                     few_clusters=True) for v in point]
    rng = np.random.default_rng(seed)
    estimates = np.full((replicates, values.shape[1]), np.nan)
    for start in range(0, replicates, 64):
        number = min(64, replicates-start)
        weights = rng.multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=number)
        den = weights @ totals; num = weights @ sums
        estimates[start:start+number] = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    out = []
    for j in range(values.shape[1]):
        valid = estimates[np.isfinite(estimates[:, j]), j]
        out.append(dict(difference=float(point[j]), ci95=np.quantile(valid, [.025, .975]).tolist()
                        if len(valid) > 1 else None, clusters=len(unique),
                        valid_replicates=len(valid), requested_replicates=replicates,
                        invalid_zero_denominator_replicates=replicates-len(valid),
                        few_clusters=len(unique) < 10))
    return out


def _contrasts(arrays, dataset, replicates=BOOTSTRAPS):
    actual = arrays["actual"]; label = (actual <= 0).astype(float); n = len(actual)
    core_mask = arrays["core_selected"].astype(float)
    names, values, denominators = [], [], []
    for left, right in (("raw", "core"), ("cal", "core"), ("raw", "histgb"),
                        ("cal", "histgb"), ("cal", "raw")):
        a = arrays[left+"_selected"].astype(float); b = arrays[right+"_selected"].astype(float)
        lp = arrays[left+"_p_null"]; rp = arrays[right+"_p_null"]
        lb = (lp-label)**2; rb = (rp-label)**2
        metrics = {
            "gamma_crps": (arrays[left+"_crps"]-arrays[right+"_crps"], np.ones(n)),
            "gamma_mse": ((arrays[left+"_expected"]-actual)**2-(arrays[right+"_expected"]-actual)**2, np.ones(n)),
            "null_brier": (lb-rb, np.ones(n)),
            "fixed_CORE_selected_brier": ((lb-rb)*core_mask, core_mask),
            "value_per_candidate": ((a-b)*actual, np.ones(n)),
            "null_per_candidate": ((a-b)*label, np.ones(n)),
            "expected_null_per_candidate": (a*lp-b*rp, np.ones(n)),
            "lambda0_value_per_candidate": ((arrays[left+"_selected_lambda0"].astype(float)-arrays[right+"_selected_lambda0"].astype(float))*actual, np.ones(n)),
            "lambda0_null_per_candidate": ((arrays[left+"_selected_lambda0"].astype(float)-arrays[right+"_selected_lambda0"].astype(float))*label, np.ones(n)),
        }
        for metric, (value, den) in metrics.items():
            names.append(dict(dataset=dataset, left=LABELS[left], right=LABELS[right], metric=metric))
            values.append(value); denominators.append(den)
    matrix = np.column_stack(values); den = np.column_stack(denominators); result = []
    for block in ("groups", "layout"):
        statistics = paired_block_statistics(matrix, den, arrays[block], replicates=replicates)
        result += [dict(**name, block="chemical_group" if block == "groups" else block,
                        **stat, policy_lists_fixed=True, models_refitted=False,
                        multiplicity_adjusted=False, finite_sample_guarantee=False)
                   for name, stat in zip(names, statistics)]
    return result


def _risk_regions(arrays, dataset):
    result = []; actual = arrays["actual"]; label = actual <= 0
    for cell in np.unique(arrays["cell"]):
        positions = np.flatnonzero(arrays["cell"] == cell)
        for arm in ARMS:
            score = arrays[arm+"_expected"][positions]-LAMBDA*arrays[arm+"_p_null"][positions]
            order = np.lexsort((arrays["ids"][positions], -score))
            rank = np.empty(len(positions)); rank[order] = (np.arange(len(positions))+.5)/len(positions)
            masks = {"all": np.ones(len(positions), bool),
                     "own_selected": arrays[arm+"_selected"][positions],
                     "fixed_CORE_selected": arrays["core_selected"][positions]}
            for lo, hi in ((0, .05), (.05, .10), (.10, .15), (.15, .25), (.25, 1)):
                masks[f"score_rank_{lo:g}_{hi:g}"] = (rank >= lo) & (rank < hi)
            for region, mask in masks.items():
                take = positions[mask]; p = arrays[arm+"_p_null"][take]; y = label[take]
                result.append(dict(dataset=dataset, cell=str(cell), arm=LABELS[arm], region=region,
                    n=len(take), actual_NULL=int(y.sum()), expected_NULL=float(p.sum()),
                    observed_rate=float(y.mean()) if len(take) else None,
                    predicted_rate=float(p.mean()) if len(take) else None,
                    signed_gap=float((y-p).mean()) if len(take) else None,
                    brier=float(((y-p)**2).mean()) if len(take) else None))
    return result


def _budget_curves(arrays, dataset):
    result = []; total_n = len(arrays["actual"])
    cells = [np.flatnonzero(arrays["cell"] == cell) for cell in np.unique(arrays["cell"])]
    for arm, lam in ((arm, lam) for arm in ARMS for lam in (LAMBDA, 0.)):
        ordered = [q[np.lexsort((arrays["ids"][q], -(arrays[arm+"_expected"][q]-lam*arrays[arm+"_p_null"][q])))] for q in cells]
        for fraction in np.linspace(0, 1, 101):
            selected = np.zeros(total_n, bool); random_gamma = random_null = 0.
            for q, order in zip(cells, ordered):
                k = int(np.floor(fraction*len(q)+1e-10)); selected[order[:k]] = True
                random_gamma += k/len(q)*arrays["actual"][q].sum()
                random_null += k/len(q)*(arrays["actual"][q] <= 0).sum()
            k = int(selected.sum()); total = float(arrays["actual"][selected].sum())
            result.append(dict(dataset=dataset, arm=LABELS[arm], lambda_value=lam, nominal_fraction=float(fraction),
                realized_fraction=k/total_n, selected=k, total_Gamma=total,
                actual_NULL=int((arrays["actual"][selected] <= 0).sum()),
                expected_NULL=float(arrays[arm+"_p_null"][selected].sum()),
                random_expected_total_Gamma=float(random_gamma), random_expected_NULL=float(random_null),
                policy_parameter_refitted=False,
                quota_rule="floor(fraction * cell_n); separate descriptive budget replay"))
    return result


def dataset_analysis(dataset, cells, *, replicates=BOOTSTRAPS):
    """Analyze already validated cells; usable independently in tests."""
    common = set.intersection(*(set(value) for _, value in cells))
    common -= {"original_k"}
    # Quantile knots and nominal coverage levels belong to a distribution,
    # not to individual queries. Keep only fields aligned in every cell.
    row_keys = {key for key in common if all(
        np.asarray(value[key]).ndim > 0
        and np.asarray(value[key]).shape[0] == len(value["actual"])
        for _, value in cells)}
    arrays = {key: np.concatenate([np.asarray(value[key]) for _, value in cells]) for key in row_keys}
    arrays["cell"] = np.concatenate([np.repeat(name, len(value["actual"])) for name, value in cells])
    arrays["random_inclusion"] = np.concatenate([np.full(len(value["actual"]), int(value["original_k"])/len(value["actual"])) for _, value in cells])
    selection_audit = []
    for arm in ARMS:
        for suffix, lam in (("selected", LAMBDA), ("selected_lambda0", 0.)):
            masks = []
            for name, value in cells:
                computed = select_top_k(value["ids"], value[arm+"_expected"], value[arm+"_p_null"], int(value["original_k"]), lam)
                key = arm+"_"+suffix
                mask = np.asarray(value[key], bool) if arm in ("core", "histgb") and key in value else computed
                masks.append(mask)
                selection_audit.append(dict(dataset=dataset, cell=name, arm=LABELS[arm], lambda_value=lam,
                    original_saved_list_reused=arm in ("core", "histgb") and key in value,
                    reconstruction_mismatches=int(np.sum(computed != mask)), selected=int(mask.sum())))
            arrays[arm+"_"+suffix] = np.concatenate(masks)
    summaries = [_summary(arrays, np.ones(len(arrays["actual"]), bool), arm, dataset) for arm in ARMS]
    units = [_summary(arrays, arrays["cell"] == cell, arm, dataset, cell)
             for cell in np.unique(arrays["cell"]) for arm in ARMS]
    lambda0 = [_summary(arrays, np.ones(len(arrays["actual"]), bool) if cell == "ALL" else arrays["cell"] == cell,
                        arm, dataset, cell, 0.)
               for cell in ["ALL", *np.unique(arrays["cell"]).tolist()] for arm in ARMS]
    for row in summaries:
        cell_rows = [v for v in units if v["arm"] == row["arm"]]
        row["cell_absolute_probability_gap"] = sum(abs(v["actual_NULL"]-v["expected_NULL"]) for v in cell_rows)/row["selected"]
    overlaps = []
    for cell in ["ALL", *np.unique(arrays["cell"]).tolist()]:
        scope = np.ones(len(arrays["actual"]), bool) if cell == "ALL" else arrays["cell"] == cell
        for left, right in (("raw", "core"), ("cal", "core"), ("raw", "histgb"), ("cal", "histgb"), ("cal", "raw")):
            a = arrays[left+"_selected"] & scope; b = arrays[right+"_selected"] & scope
            intersection = int((a & b).sum()); union = int((a | b).sum())
            overlaps.append(dict(dataset=dataset, cell=cell, left=LABELS[left], right=LABELS[right],
                selected_left=int(a.sum()), selected_right=int(b.sum()), intersection=intersection,
                overlap_fraction=intersection/int(a.sum()), jaccard=intersection/union,
                replaced_left=int((a & ~b).sum())))
    rows = []
    for i, identifier in enumerate(arrays["ids"]):
        row = {k: arrays[k][i] for k in ("cell", "groups", "layout", "actual")}
        row.update(dataset=dataset, id=str(identifier), null=bool(arrays["actual"][i] <= 0),
                   random_inclusion=float(arrays["random_inclusion"][i]))
        for arm in ARMS:
            for field in ("expected", "p_null", "crps", "selected", "selected_lambda0"):
                row[arm+"_"+field] = arrays[arm+"_"+field][i]
        rows.append(row)
    return arrays, dict(summary=summaries, deployment_units=units, per_object=rows,
        paired_intervals=_contrasts(arrays, dataset, replicates), selection_overlap=overlaps,
        risk_regions=_risk_regions(arrays, dataset), budget_curves=_budget_curves(arrays, dataset),
        lambda0_secondary=lambda0, selection_audit=selection_audit)


def _report(path, completed, pending, tables):
    lines = ["# Conditional-quantile HistGB: development comparison", "",
        "Completed datasets: " + (", ".join(completed) or "none") + ". Pending: " + (", ".join(pending) or "none") + ".", "",
        "Raw conditional quantiles are the primary comparator; CAL-offset quantiles are secondary. "
        "Both are retained without query-based winner selection. Original CORE and classifier-calibrated HistGB predictions and lists are reused from their saved R2 outputs. Stable-rank reconstruction is audited separately; saved historical lists are never silently replaced.", "",
        "| Dataset | Arm | Gamma CRPS | Mean selected Gamma | NULL / selected | Expected NULL | Total Gamma | Exact random expectation |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in tables["summary"]:
        lines.append("| %s | %s | %.6f | %.6f | %d / %d | %.3f | %.5f | %.5f |" %
            (row["dataset"], row["arm"], row["gamma_crps"], row["mean_selected_Gamma"], row["actual_NULL"],
             row["selected"], row["expected_NULL"], row["total_Gamma"], row["random_expected_total_Gamma"]))
    lines += ["", "## Saved evidence", "",
        "Per-object predictions and selected masks, five-level interval coverage/width, deployment-unit risks, "
        "fixed-CORE-region Brier contrasts, list overlap, and descriptive budget curves are saved alongside the aggregate metrics. "
        "The four datasets are not pooled into one scientific effect estimate.", "",
        "Paired percentile intervals resample entire chemical groups or layouts with the same weights for every arm and every occurrence of a group across doses. "
        "They condition on fitted models and fixed lists; they do not establish binomial, finite-sample, or out-of-distribution guarantees. "
        "Few-layout intervals are reported separately with their number of blocks. NULL-per-candidate is not FDP or NULL-conditioned FPR.", "",
        "Historical HistGB CRPS comes from its residual-law companion, whereas its original allocation probability is its calibrated classifier. "
        "The new quantile arms obtain the mean, NULL probability, and CRPS coherently from their own conditional CDFs. "
        "Action cost is already included in actual Gamma and is not deducted again. New fitting/evaluation timing is retained from each cell separately; historical costs are not relabelled as new costs.", "",
        "No R4 measurement was opened or reanalysed. No policy is promoted automatically from this development report."]
    path.write_text("\n".join(lines)+"\n")


def aggregate(project, run_root, dataset=None):
    """Refresh complete-dataset reports, retaining an explicit 60-cell status.

    ``dataset`` may limit recalculation to one archive. Completed results for
    other archives are read from their saved reports for the combined summary.
    A dataset is never reported complete until its expected cell and row counts
    pass and every historical prediction/list agrees exactly.
    """
    project = Path(project); run_root = Path(run_root)
    if dataset is not None and dataset not in DATASETS: raise ValueError("Unknown dataset")
    report_root = project / "reports" / run_root.name
    report_root.mkdir(parents=True, exist_ok=True)
    requested = DATASETS if dataset is None else (dataset,)
    for name in requested:
        folders = sorted((run_root/name).glob("cell_*"))
        folders = [f for f in folders if f.is_dir()]
        done = [f for f in folders if (f/"complete.json").exists() and (f/"query_predictions.npz").exists()]
        if len(done) != EXPECTED[name][1]: continue
        historical = _historical(project, name); cells = []; provenance = []
        for folder in done:
            arrays = _attach_historical(_npz(folder/"query_predictions.npz"), historical, folder.name)
            validate_cell(arrays, folder.name); cells.append((folder.name, arrays))
            provenance.append(dict(cell=folder.name, complete=json.loads((folder/"complete.json").read_text()),
                metadata=json.loads((folder/"metadata.json").read_text()) if (folder/"metadata.json").exists() else None))
        ids = np.concatenate([values["ids"].astype(str) for _, values in cells])
        if len(ids) != EXPECTED[name][0] or len(set(ids)) != len(ids):
            raise ValueError(name + " does not contain the declared unique complete query cohort")
        arrays, tables = dataset_analysis(name, cells)
        output = report_root/name; output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/"all_query_predictions.npz", **arrays)
        for table, rows in tables.items(): _csv(output/(table+".csv"), rows)
        _csv(output/"compute_costs.csv", [dict(dataset=name, cell=row["cell"],
             **{key: value for key, value in row["complete"].items() if key not in ("dataset", "cell")},
             historical_training_cost_included=False, historical_cost_source="unchanged original R2 cost ledger")
             for row in provenance])
        result = dict(dataset=name, complete=True, n=len(ids), cells=len(cells), tables=tables,
                      provenance=provenance, updated=datetime.now(timezone.utc).isoformat())
        _json(output/"summary.json", result)
        _report(output/"REPORT.md", [name], [], tables)
    tables = {key: [] for key in ("summary", "deployment_units", "per_object", "paired_intervals", "selection_overlap", "risk_regions", "budget_curves", "lambda0_secondary", "selection_audit")}
    completed = []; pending = []; statuses = []
    for name in DATASETS:
        saved = report_root/name/"summary.json"
        done = list((run_root/name).glob("cell_*/complete.json"))
        statuses.append(dict(dataset=name, cells_with_completion_marker=len(done), expected_cells=EXPECTED[name][1]))
        if not saved.exists(): pending.append(name); continue
        result = json.loads(saved.read_text())
        if not result.get("complete") or result["n"] != EXPECTED[name][0] or result["cells"] != EXPECTED[name][1]:
            raise ValueError("Invalid saved complete-dataset report")
        completed.append(name)
        for key in tables: tables[key].extend(result["tables"][key])
    for key, rows in tables.items(): _csv(report_root/(key+".csv"), rows)
    status = dict(complete=not pending, completed_datasets=completed, pending_datasets=pending,
        expected_cells=60, expected_query_rows=13141, dataset_status=statuses,
        primary="QUANTILE_RAW_PRIMARY", secondary="QUANTILE_CAL_SECONDARY", lambda_value=LAMBDA,
        bootstrap_replicates=BOOTSTRAPS, bootstrap_seed=SEED, confidence_scope="fixed-model fixed-list development comparisons",
        promotion_or_query_winner_selection=False, r4_reanalysed=False,
        updated=datetime.now(timezone.utc).isoformat(), tables=tables)
    _json(report_root/"summary.json", status)
    _report(report_root/"REPORT.md", completed, pending, tables)
    return {key: value for key, value in status.items() if key != "tables"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.project, args.run_root, args.dataset), indent=2))
