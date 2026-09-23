"""Auditable, fixed-budget DEV comparisons of three acquisition actions.

The inputs are already net utilities, not gross gains. Nothing in this module
fits a predictor, changes an endpoint, or issues a statistical certificate.
Compound bootstrap intervals condition on the fitted scores and shared batches;
they do not establish generalization to new batches or account for model search.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ACTIONS = ("Z1", "Z2", "Z1Z2")
WELL_COSTS = (1, 1, 2)
NULL_THRESHOLD = 0.0
POSITIVE_MARGIN = 0.005


def _matrix(value, name, n=None):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise ValueError(f"{name} must have nonempty shape (N, 3): Z1, Z2, Z1Z2")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} rows do not match IDs")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values; declare completion upstream")
    return array


def _ids(value, n):
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError("ids must contain exactly one compound ID per row")
    if not all(isinstance(x, (str, np.str_)) and x.strip() for x in raw.tolist()):
        raise ValueError("IDs must be nonempty strings")
    result = raw.astype(str)
    if len(set(result.tolist())) != n:
        raise ValueError("Repeated compound IDs are not independent comparison rows")
    return result


def stable_top_k(scores, ids, k):
    """Descending scores; exact ties are resolved by lexical compound ID."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector")
    ids = _ids(ids, len(scores))
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k <= len(ids):
        raise ValueError("k must be an integer between zero and N")
    selected = np.zeros(len(ids), dtype=bool)
    selected[np.lexsort((ids, -scores))[:k]] = True
    return selected


def _metrics(actual, mean, pnull):
    null = actual <= NULL_THRESHOLD
    pos = actual >= POSITIVE_MARGIN
    centered = actual - actual.mean()
    sst = float(centered @ centered)
    varied = np.ptp(actual) > 0 and np.ptp(mean) > 0
    return {
        "n": len(actual),
        "actual_mean": float(actual.mean()),
        "actual_sd": float(actual.std()),
        "predicted_mean": float(mean.mean()),
        "predicted_sd": float(mean.std()),
        "null_count": int(null.sum()),
        "positive_count": int(pos.sum()),
        "ambiguous_count": int((~null & ~pos).sum()),
        "null_rate": float(null.mean()),
        "positive_rate": float(pos.mean()),
        "mse": float(np.mean((actual - mean) ** 2)),
        "r2": float(1 - np.sum((actual - mean) ** 2) / sst) if sst > 0 else None,
        "pearson": float(np.corrcoef(actual, mean)[0, 1]) if varied else None,
        "spearman": float(spearmanr(actual, mean).statistic) if varied else None,
        "null_auc": float(roc_auc_score(null, pnull)) if null.any() and not null.all() else None,
        "null_brier": float(np.mean((null - pnull) ** 2)),
    }


def _observed(actual, active, cost):
    n, k = len(actual), int(active.sum())
    null, pos = actual <= NULL_THRESHOLD, actual >= POSITIVE_MARGIN
    false, hit = int((active & null).sum()), int((active & pos).sum())
    total = float(actual[active].sum())
    return {
        "eligible_n": n,
        "selected_n": k,
        "used_wells": k * cost,
        "coverage": k / n,
        "total_net_gain": total,
        "per_selected_net_gain": total / k if k else None,
        "per_eligible_net_gain": total / n,
        "selected_null_count": false,
        "selected_positive_count": hit,
        "selected_ambiguous_count": k - false - hit,
        "fdp": false / k if k else None,
        "positive_purity": hit / k if k else None,
        "fpr": false / int(null.sum()) if null.any() else None,
        "sensitivity": hit / int(pos.sum()) if pos.any() else None,
        "population_null_count": int(null.sum()),
        "population_positive_count": int(pos.sum()),
    }


def _quantiles(values, alpha):
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return [float(low), float(high)]


def _matched_random(actual, active, *, rng, n_random, n_bootstrap, alpha):
    n, k = len(actual), int(active.sum())
    q = k / n
    actual_null = actual <= NULL_THRESHOLD
    actual_pos = actual >= POSITIVE_MARGIN
    expected_total = float(k * actual.mean())
    exact = {
        "selected_n": k,
        "expected_total_net_gain": expected_total,
        "expected_per_eligible_net_gain": expected_total / n,
        "expected_per_selected_net_gain": float(actual.mean()) if k else None,
        "expected_selected_null_count": float(k * actual_null.mean()),
        "expected_selected_positive_count": float(k * actual_pos.mean()),
        "expected_fdp": float(actual_null.mean()) if k else None,
        "expected_fpr": q if actual_null.any() else None,
        "expected_sensitivity": q if actual_pos.any() else None,
    }
    # Randomization is without replacement in this finite evaluation population.
    random_total = np.empty(n_random)
    random_null = np.empty(n_random)
    for draw in range(n_random):
        pick = rng.choice(n, k, replace=False)
        random_total[draw] = actual[pick].sum()
        random_null[draw] = actual_null[pick].sum()
    finite_random = {
        "replicates": n_random,
        "total_net_gain_interval": _quantiles(random_total, alpha),
        "per_eligible_net_gain_interval": _quantiles(random_total / n, alpha),
        "per_selected_net_gain_interval": _quantiles(random_total / k, alpha) if k else None,
        "selected_null_count_interval": _quantiles(random_null, alpha),
        "fdp_interval": _quantiles(random_null / k, alpha) if k else None,
        "scope": "random subsets of these same compounds; not population confidence intervals",
    }
    # The paired estimand is E[(selected_i - q) * observed_net_gain_i].
    # q and the fitted decisions remain frozen, not retuned inside bootstrap.
    difference = (active.astype(float) - q) * actual
    means = np.empty(n_bootstrap)
    for start in range(0, n_bootstrap, 128):
        stop = min(start + 128, n_bootstrap)
        pick = rng.integers(n, size=(stop - start, n))
        means[start:stop] = difference[pick].mean(axis=1)
    paired = {
        "replicates": n_bootstrap,
        "per_eligible_difference": float(difference.mean()),
        "per_eligible_difference_interval": _quantiles(means, alpha),
        "total_difference": float(difference.sum()),
        "total_difference_interval": _quantiles(n * means, alpha),
        "per_selected_difference": float(difference.mean() / q) if k else None,
        "per_selected_difference_interval": _quantiles(means / q, alpha) if k else None,
        "resampling_unit": "compound",
        "selection_is_frozen": True,
        "scope": "conditional on fitted scores and shared fixed batches; no model-selection uncertainty",
        "formal_certificate": False,
    }
    return {"exact_expectation": exact, "finite_population_randomization": finite_random,
            "paired_bootstrap_vs_random_expectation": paired}


def evaluate_predictions(predmeans, pnull, actual, ids, *, fractions=(.05, .10, .25),
                         train_actual=None, seed=0, n_bootstrap=2000, n_random=2000,
                         alpha=.05, original_contract=None):
    """Compare precomputed predictions on complete, uniquely identified DEV rows.

    All arrays have ordered columns ``Z1, Z2, Z1Z2``. Their net utilities must
    use the same frozen endpoint and costs upstream. Fractional budgets round
    *down*. Within-action fractions mean selected compounds; across-action
    budgets are ``floor(fraction*N)`` physical wells. Both score rankings are
    declared diagnostics, not candidates from which to select an evaluation
    winner. ``train_actual`` only chooses the fixed-action comparator; it must
    come from the current training fold, never the evaluation rows.
    """
    actual = _matrix(actual, "actual")
    n = len(actual)
    predmeans = _matrix(predmeans, "predmeans", n)
    pnull = _matrix(pnull, "pnull", n)
    ids = _ids(ids, n)
    if ((pnull < 0) | (pnull > 1)).any():
        raise ValueError("pnull must be a probability in [0, 1]")
    fractions = tuple(float(x) for x in fractions)
    if not fractions or len(set(fractions)) != len(fractions) or any(not 0 < x <= 1 for x in fractions):
        raise ValueError("fractions must be distinct finite numbers in (0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    for name, count in (("n_bootstrap", n_bootstrap), ("n_random", n_random)):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError(f"{name} must be a positive integer")
    if original_contract is not None and not isinstance(original_contract, dict):
        raise ValueError("original_contract must be a declared dictionary, not an inferred contract")
    rng = np.random.default_rng(seed)
    # Canonical row order makes numerical randomization and tie handling invariant
    # to an upstream file reorder; the input row index is retained for traceability.
    order = np.argsort(ids, kind="stable")
    ids, actual, predmeans, pnull = ids[order], actual[order], predmeans[order], pnull[order]
    trace = [{"compound_id": str(ids[i]), "input_row": int(order[i]),
              "actual": dict(zip(ACTIONS, map(float, actual[i]))),
              "predicted_mean": dict(zip(ACTIONS, map(float, predmeans[i]))),
              "p_null": dict(zip(ACTIONS, map(float, pnull[i]))), "selected_by": []}
             for i in range(n)]
    result = {
        "actions": list(ACTIONS), "additional_wells": list(WELL_COSTS),
        "assumed_cost_per_well_in_supplied_net_utilities": .01,
        "null_threshold": NULL_THRESHOLD, "positive_margin": POSITIVE_MARGIN,
        "original_contract": deepcopy(original_contract), "original_contract_changed": False,
        "formal_certificate": False, "fractions": list(fractions), "interval_level": 1 - alpha,
        "action_metrics": [], "within_action": [], "common_budget": [], "fixed_policies": [],
        "row_trace": trace,
        "statistical_scope": {
            "development_diagnostic_only": True,
            "randomization_conditions_on_evaluation_population": True,
            "compound_bootstrap_conditions_on_shared_fixed_batches": True,
            "does_not_certify_new_batch_generalization": True,
            "no_multiple_comparison_correction_or_winner_claim": True,
            "score_ties": "descending score, ascending lexical compound ID",
            "spearman_ties": "scipy average ranks", "auc_ties": "sklearn standard half credit",
            "endpoint_completion": "must be declared and applied upstream; no rows dropped here",
        },
    }

    def add_row(section, j, active, label, fraction, budget, ranking):
        row = {"label": label, "action": ACTIONS[j], "ranking": ranking,
               "fraction": fraction, "budget_wells": budget,
               "budget_scope": "common physical-well cap" if section == "common_budget" else "within-action selected-compound fraction",
               **_observed(actual[:, j], active, WELL_COSTS[j]), "formal_certificate": False}
        row["unused_wells"] = budget - row["used_wells"]
        if row["unused_wells"] < 0:
            raise AssertionError("A comparison exceeded its physical well budget")
        row["matched_random"] = _matched_random(actual[:, j], active, rng=rng,
                                                 n_random=n_random, n_bootstrap=n_bootstrap, alpha=alpha)
        row["selected_ids"] = ids[active].tolist()
        result[section].append(row)
        for i in np.flatnonzero(active):
            trace[i]["selected_by"].append(label)

    for j, name in enumerate(ACTIONS):
        result["action_metrics"].append({"action": name, **_metrics(actual[:, j], predmeans[:, j], pnull[:, j])})
        result["fixed_policies"].append({"label": "ADD_ALL_" + name, "action": name,
                                         **_observed(actual[:, j], np.ones(n, dtype=bool), WELL_COSTS[j]),
                                         "budget_matched": False, "formal_certificate": False})
        for fraction in fractions:
            k = math.floor(fraction * n)
            common_budget = k
            for ranking, scores in (("expected_gain", predmeans[:, j]), ("lowest_p_null", -pnull[:, j])):
                active = stable_top_k(scores, ids, k)
                label = f"within__{name}__{ranking}__{fraction:g}"
                add_row("within_action", j, active, label, fraction, k * WELL_COSTS[j], ranking)
                same_budget_active = stable_top_k(scores, ids, common_budget // WELL_COSTS[j])
                label = f"common__{name}__{ranking}__{fraction:g}"
                add_row("common_budget", j, same_budget_active, label, fraction, common_budget, ranking)

    stop = {"label": "STOP", "action": "STOP", "eligible_n": n, "selected_n": 0,
            "used_wells": 0, "coverage": 0., "total_net_gain": 0.,
            "per_selected_net_gain": None, "per_eligible_net_gain": 0.,
            "fdp": None, "fpr": None, "sensitivity": None,
            "budget_matched": False, "formal_certificate": False}
    result["fixed_policies"].append(stop)
    result["train_selected_fixed_action"] = None
    if train_actual is not None:
        train_actual = _matrix(train_actual, "train_actual")
        train_means = train_actual.mean(axis=0)
        rates = train_means / np.asarray(WELL_COSTS)
        # STOP is first, so ties at zero do not force expenditure.
        winner = int(np.argmax(np.r_[0., rates])) - 1
        fixed = {"choice": "STOP" if winner < 0 else ACTIONS[winner],
                 "training_n": len(train_actual),
                 "training_mean_net_gain": dict(zip(ACTIONS, map(float, train_means))),
                 "training_net_gain_per_additional_well": dict(zip(ACTIONS, map(float, rates))),
                 "selection_uses_evaluation_outcomes": False, "budgets": []}
        for fraction in fractions:
            budget = math.floor(fraction * n)
            if winner < 0:
                fixed["budgets"].append({**stop, "fraction": fraction, "budget_wells": budget,
                                          "unused_wells": budget, "budget_matched": True})
                continue
            k = budget // WELL_COSTS[winner]
            null, positive = actual[:, winner] <= 0, actual[:, winner] >= POSITIVE_MARGIN
            # No evaluation information chooses compounds: exact expectation
            # under uniform random subsets, rather than a favorable random draw.
            total = float(k * actual[:, winner].mean())
            fixed["budgets"].append({"fraction": fraction, "budget_wells": budget,
                "action": ACTIONS[winner], "selected_n": k, "used_wells": k * WELL_COSTS[winner],
                "unused_wells": budget - k * WELL_COSTS[winner], "coverage": k / n,
                "expected_total_net_gain": total, "expected_per_eligible_net_gain": total / n,
                "expected_per_selected_net_gain": float(actual[:, winner].mean()) if k else None,
                "expected_fdp": float(null.mean()) if k else None,
                "expected_fpr": k / n if null.any() else None,
                "expected_sensitivity": k / n if positive.any() else None,
                "uniform_random_subset_exact_expectation": True, "formal_certificate": False})
        result["train_selected_fixed_action"] = fixed
    return result
