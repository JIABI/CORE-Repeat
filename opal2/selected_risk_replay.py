"""CAL-only lambda tuning and fixed-policy selected-set risk forecasts.

``evaluate_cell(cell, saved_query)`` accepts the unchanged decision-region
cache and that same cell's previous ``query_predictions.npz`` mapping. It
returns ``(arrays, metadata)`` and never reads or writes files or fits models.

Common arrays retain ``cal_ids/groups/inner_fold/actual`` and
``query_ids/groups/layout/actual``; ``cal_layout`` is copied when available.
Each arm retains ``ARM__cal_p/cal_mean/query_p/query_mean`` unchanged and adds:

* ``ARM__CAL_SELECTED_GRID``: Boolean [4, n_cal], ordered by LAMBDA_GRID.
* ``ARM__FIXED_0``, ``ARM__FIXED_02``, ``ARM__CAL_TUNED``: query masks.
  Runner aliases are ``ARM__query_selected_VARIANT`` and
  ``ARM__cal_selected_L0/L01/L02/L04`` for the CAL grid rows.
* ``ARM__FIXED_02_SELECTED_QUERY_P`` and ``..._IDS``: original selected rows.
* For existing MC seeds, ``ARM__MC0_FIXED_0``, ``..._FIXED_02`` and
  ``..._CAL_TUNED`` (and MC1, etc.), plus unchanged seed p/mean arrays.
  Their runner aliases are ``ARM__query_selected_MC0_VARIANT``.

Metadata contains the integer budgets, per-arm CAL objective grid and chosen
lambda, CAL support diagnostics, and ``frozen_primary_risk`` with raw and
Jeffreys-sensitivity selected-set NULL forecasts for fixed lambda 0.2. These
forecasts are scalar summaries of a selected set: they are never substituted
for row-level probabilities or used to rerank candidates.
"""
from __future__ import annotations

import math
import numpy as np

from .decision_region_calibration import (
    ARMS, frozen_rank_feature, select_top_k, validate_cell as _validate_cell,
)


VERSION = 1
LAMBDA_GRID = (0.0, 0.1, 0.2, 0.4)
PRIMARY_LAMBDA = 0.2
OBJECTIVE_TIE_ATOL = 1e-12
VARIANTS = ("FIXED_0", "FIXED_02", "CAL_TUNED")


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _folds(value):
    folds = np.asarray(value)
    if (folds.ndim != 1 or not len(folds) or folds.dtype.kind not in "iu"
            or np.any(folds < 0) or len(np.unique(folds)) < 2):
        raise ValueError("At least two aligned nonnegative integer CAL inner folds required")
    return folds


def apportion_cal_budget(inner_fold, query_budget, n_query):
    """Round half up, then largest remainders, using sizes/IDs only.

    Remainder ties prefer larger folds, then the smaller stable integer fold
    ID. Integer arithmetic avoids ambiguous floating point half-rounding.
    """
    folds = _folds(inner_fold)
    k = _positive_integer(query_budget, "query_budget")
    n_query = _positive_integer(n_query, "n_query")
    if k > n_query:
        raise ValueError("Query budget exceeds query population")
    unique, sizes = np.unique(folds, return_counts=True)
    n_cal = len(folds)
    total = (2 * n_cal * k + n_query) // (2 * n_query)
    allocated = total * sizes // n_cal
    remainder = total * sizes % n_cal
    priority = sorted(range(len(unique)), key=lambda i: (-int(remainder[i]),
                                                         -int(sizes[i]), int(unique[i])))
    for index in priority[:int(total - allocated.sum())]:
        allocated[index] += 1
    if allocated.sum() != total or np.any(allocated > sizes):
        raise AssertionError("CAL budget allocation failed")
    return {"total_budget": int(total), "n_cal": n_cal, "query_budget": k,
            "n_query": n_query,
            "fold_sizes": {str(int(f)): int(n) for f, n in zip(unique, sizes, strict=True)},
            "fold_budgets": {str(int(f)): int(n) for f, n in zip(unique, allocated, strict=True)},
            "rounding": "round-half-up(n_cal * k_query / n_query)",
            "apportionment": "largest remainders; ties: larger fold, then smaller integer fold ID",
            "outcomes_used": False}


def validate_cell(cell):
    """Require the unchanged five-arm cache and whole-group inner folds."""
    _validate_cell(cell)
    folds = _folds(cell["cal_inner_fold"])
    for prefix in ("cal", "query"):
        ids = np.asarray(cell[prefix + "_ids"], str)
        groups = np.asarray(cell[prefix + "_groups"], str)
        if np.any(ids == "") or np.any(groups == ""):
            raise ValueError("Nonempty known CAL/QUERY identities and groups required")
    if "cal_layout" in cell and np.asarray(cell["cal_layout"]).shape != folds.shape:
        raise ValueError("CAL layout is not aligned")
    budget = apportion_cal_budget(folds, cell["query_budget"], len(cell["query_ids"]))
    if budget["total_budget"] == 0:
        raise ValueError("Rounded CAL budget is zero; selected-set tuning/risk is undefined")
    return budget


def select_cal_by_fold(ids, mean, probability, inner_fold, budget, lam):
    """Select independently in each wholly held CAL fold, never pooled scores."""
    folds = _folds(inner_fold)
    ids, mean, probability = np.asarray(ids, str), np.asarray(mean), np.asarray(probability)
    if any(value.shape != folds.shape for value in (ids, mean, probability)):
        raise ValueError("CAL scores and held-fold memberships must align")
    if len(set(ids)) != len(ids):
        raise ValueError("CAL IDs must be unique within the deployment cell")
    selected = np.zeros(len(folds), bool)
    for fold in np.unique(folds):
        rows = np.flatnonzero(folds == fold)
        count = int(budget["fold_budgets"][str(int(fold))])
        selected[rows] = select_top_k(ids[rows], mean[rows], probability[rows], count, lam)
    if int(selected.sum()) != budget["total_budget"]:
        raise AssertionError("Selected CAL counts differ from apportioned budget")
    return selected


def choose_lambda(objectives):
    """Maximize realized CAL selected mean Gamma; fixed tolerance/tie preference."""
    objectives = np.asarray(objectives, float)
    if objectives.shape != (len(LAMBDA_GRID),) or not np.isfinite(objectives).all():
        raise ValueError("One finite CAL objective for every fixed lambda candidate required")
    best = float(objectives.max())
    tied = [i for i, value in enumerate(objectives) if best - value <= OBJECTIVE_TIE_ATOL]
    winner = min(tied, key=lambda i: (LAMBDA_GRID[i] != PRIMARY_LAMBDA,
                                    abs(LAMBDA_GRID[i] - PRIMARY_LAMBDA), LAMBDA_GRID[i]))
    return float(LAMBDA_GRID[winner]), int(winner)


def _saved_rows(saved_query, ids):
    saved_ids = np.asarray(saved_query["ids"], str)
    if (saved_ids.ndim != 1 or len(saved_ids) != len(ids)
            or len(set(saved_ids)) != len(saved_ids) or set(saved_ids) != set(ids)):
        raise ValueError("Existing query predictions must contain exactly this cell's IDs")
    lookup = {value: i for i, value in enumerate(saved_ids)}
    return np.asarray([lookup[value] for value in ids], int)


def _check_saved_mask(saved_query, key, rows, mask):
    saved = np.asarray(saved_query[key])
    if saved.dtype != np.dtype(bool) or saved.shape != mask.shape:
        raise ValueError("Expected saved Boolean query mask: " + key)
    np.testing.assert_array_equal(mask, saved[rows], err_msg="Changed original query selection: " + key)


def _sum(values):
    # Fixed finite arrays; fsum also prevents input-row permutation from changing
    # objective ties through a different ordinary floating point sum order.
    return float(math.fsum(np.asarray(values, float).tolist()))


def _support(ids, groups, actual, probability, mask):
    mask = np.asarray(mask, bool)
    count = int(mask.sum())
    null = int(np.sum(np.asarray(actual)[mask] <= 0))
    total = _sum(np.asarray(probability)[mask])
    return {"n": count, "unique_ids": len(set(np.asarray(ids)[mask])),
            "unique_groups": len(set(np.asarray(groups)[mask])), "actual_NULL": null,
            "predicted_NULL": total, "observed_NULL_rate": null / count if count else None,
            "mean_original_p": total / count if count else None}


def evaluate_cell(cell, saved_query):
    """Evaluate the fixed replay; QUERY labels are copied, never used to fit/select.

    ``saved_query`` is required so both original lambda 0.2 and lambda 0 lists
    must reproduce the earlier saved experiment. The new caller may permute
    rows; old arrays are aligned by immutable query IDs before checking.
    """
    budget = validate_cell(cell)
    cids, qids = np.asarray(cell["cal_ids"], str), np.asarray(cell["query_ids"], str)
    cgroups = np.asarray(cell["cal_groups"], str)
    actual, folds = np.asarray(cell["cal_actual"], float), np.asarray(cell["cal_inner_fold"])
    k_cal, k_query = budget["total_budget"], int(cell["query_budget"])
    saved_rows = _saved_rows(saved_query, qids)
    arrays = {key: np.asarray(cell[key]).copy() for key in
        ("cal_ids", "cal_groups", "cal_inner_fold", "cal_actual",
         "query_ids", "query_groups", "query_layout", "query_actual")}
    if "cal_layout" in cell:
        arrays["cal_layout"] = np.asarray(cell["cal_layout"]).copy()
    records = {}
    for arm in ARMS:
        values = cell["arms"][arm]
        cm, cp = np.asarray(values["cal_mean"]), np.asarray(values["cal_p"])
        qm, qp = np.asarray(values["query_mean"]), np.asarray(values["query_p"])
        for name in ("cal_p", "cal_mean", "query_p", "query_mean"):
            arrays[arm + "__" + name] = np.asarray(values[name]).copy()
        np.testing.assert_array_equal(qm, np.asarray(saved_query[arm + "__mean"])[saved_rows],
                                      err_msg=arm + " original query mean changed")
        np.testing.assert_array_equal(qp, np.asarray(saved_query[arm + "__ORIGINAL__p"])[saved_rows],
                                      err_msg=arm + " original query probability changed")
        grid = np.stack([select_cal_by_fold(cids, cm, cp, folds, budget, lam) for lam in LAMBDA_GRID])
        objectives = [_sum(actual[mask]) / k_cal for mask in grid]
        chosen, chosen_index = choose_lambda(objectives)
        fixed0 = select_top_k(qids, qm, qp, k_query, 0.)
        fixed02 = select_top_k(qids, qm, qp, k_query, PRIMARY_LAMBDA)
        tuned = select_top_k(qids, qm, qp, k_query, chosen)
        _check_saved_mask(saved_query, arm + "__original_selected", saved_rows, fixed02)
        _check_saved_mask(saved_query, arm + "__lambda0_selected", saved_rows, fixed0)
        arrays[arm + "__CAL_SELECTED_GRID"] = grid
        for index, token in enumerate(("L0", "L01", "L02", "L04")):
            arrays[arm + "__cal_selected_" + token] = grid[index].copy()
        for variant, mask in zip(VARIANTS, (fixed0, fixed02, tuned), strict=True):
            arrays[arm + "__" + variant] = mask
            arrays[arm + "__query_selected_" + variant] = mask.copy()
        arrays[arm + "__FIXED_02_SELECTED_QUERY_P"] = qp[fixed02].copy()
        arrays[arm + "__FIXED_02_SELECTED_QUERY_IDS"] = qids[fixed02].copy()
        primary_mask = grid[LAMBDA_GRID.index(PRIMARY_LAMBDA)]
        cal_null = int(np.sum(actual[primary_mask] <= 0))
        q_raw, q_jeffreys = cal_null / k_cal, (cal_null + .5) / (k_cal + 1)
        rank, _ = frozen_rank_feature(cm, cp, cids, folds)
        support = {name: _support(cids, cgroups, actual, cp, mask) for name, mask in
            (("all", np.ones(len(cids), bool)), ("top_0.125", rank < .125),
             ("top_0.25", rank < .25), ("primary_budget_selected", primary_mask))}
        support["inner_folds"] = {str(int(fold)): {
            name: _support(cids, cgroups, actual, cp, mask & (folds == fold))
            for name, mask in (("all", np.ones(len(cids), bool)), ("top_0.125", rank < .125),
                              ("top_0.25", rank < .25), ("primary_budget_selected", primary_mask))}
            for fold in np.unique(folds)}
        record = {"chosen_lambda": chosen, "chosen_lambda_index": chosen_index,
            "cal_objective_grid": [{"lambda": lam, "selected_n": k_cal,
                "selected_mean_Gamma": objectives[index], "selected_total_Gamma": _sum(actual[grid[index]]),
                "selected_NULL": int(np.sum(actual[grid[index]] <= 0))}
                for index, lam in enumerate(LAMBDA_GRID)],
            "cal_support": support,
            "frozen_primary_risk": {"lambda": PRIMARY_LAMBDA, "cal_selected_n": k_cal,
                "cal_selected_NULL": cal_null, "q_raw": q_raw, "q_jeffreys": q_jeffreys,
                "query_selected_n": k_query, "raw_predicted_query_NULL": k_query * q_raw,
                "jeffreys_predicted_query_NULL": k_query * q_jeffreys,
                "base_predicted_query_NULL": _sum(qp[fixed02]),
                "scope": "selected-set NULL fraction for fixed lambda 0.2; not a row probability model",
                "primary_estimator": "raw empirical selected CAL NULL fraction",
                "sensitivity_estimator": "(0.5 + CAL selected NULL)/(CAL selected n + 1)",
                "row_probabilities_changed": False, "ranking_changed_by_risk_forecast": False},
            "existing_fixed_masks_exact": True, "query_labels_used": False}
        record.update(calibration_grid=record["cal_objective_grid"], q_empirical=q_raw,
                      q_jeffreys=q_jeffreys, n_cal_selected=k_cal)
        has_seed = ("query_seed_p" in values, "query_seed_mean" in values)
        if has_seed[0] != has_seed[1]:
            raise ValueError("Both original MC probability and mean arrays are required")
        if has_seed[0]:
            seed_p, seed_m = np.asarray(values["query_seed_p"]), np.asarray(values["query_seed_mean"])
            if (seed_p.shape != seed_m.shape or seed_p.ndim != 2 or seed_p.shape[1] != len(qids)
                    or not len(seed_p)):
                raise ValueError("Original query MC arrays are unaligned")
            arrays[arm + "__query_seed_p"] = seed_p.copy()
            arrays[arm + "__query_seed_mean"] = seed_m.copy()
            for index, (probability, mean) in enumerate(zip(seed_p, seed_m, strict=True)):
                for variant, lam in zip(VARIANTS, (0., PRIMARY_LAMBDA, chosen), strict=True):
                    mask = select_top_k(qids, mean, probability, k_query, lam)
                    arrays[f"{arm}__MC{index}_{variant}"] = mask
                    arrays[f"{arm}__query_selected_MC{index}_{variant}"] = mask.copy()
            record["existing_MC_seeds_replayed"] = len(seed_p)
            record["MC_lambda_refitted"] = False
        records[arm] = record
    metadata = {"version": VERSION, "dataset": str(cell["dataset"]), "cell": str(cell["cell"]),
        "n_query": len(qids), "query_budget": k_query, "n_cal": len(cids), "cal_budget": k_cal,
        "lambda_grid": list(LAMBDA_GRID), "primary_lambda": PRIMARY_LAMBDA,
        "objective": "maximize pooled selected realized CAL mean Gamma after separate held-fold selections",
        "lambda_tie_rule": "within absolute 1e-12 of maximum: prefer 0.2, then nearest 0.2, then smaller lambda",
        "lambda_tie_atol": OBJECTIVE_TIE_ATOL, "budget": budget, "arms": records,
        "cal_selection_scope": "within each wholly held-out inner fold; never pooled OOF ranking",
        "unique_cal_ids": len(set(cids)), "unique_cal_groups": len(set(cgroups)),
        "unique_query_ids": len(set(qids)), "unique_query_groups": len(set(cell["query_groups"])),
        "base_p_and_mean_unchanged": True, "model_fits": 0, "new_monte_carlo_draws": 0,
        "query_labels_used_for_tuning_or_forecasting": False,
        "risk_forecast_changes_row_probabilities_or_ranking": False,
        "cal_support_interpretation": "within-cell counts; cross-cell appearances are not independent identities",
        "formal_certificate": False}
    return arrays, metadata
