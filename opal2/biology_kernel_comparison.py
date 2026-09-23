"""Paired, descriptive factorial comparisons from saved full-model predictions.

Bootstrap resamples compounds, not seed-by-compound copies. Frozen selections
are reconstructed from declared scores/budgets; neither arms nor thresholds
are chosen from evaluation outcomes. Intervals are exploratory and conditional
on these fitted models and shared batches, without multiplicity guarantees.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .baseline_policy import ACTIONS, WELL_COSTS, stable_top_k
from .biology_kernel_evaluation import plain, write_json


ARM_NAMES = ("A_OFF_GAUSSIAN", "B_OFF_COPULA_T4", "C_KERNEL_GAUSSIAN", "D_KERNEL_COPULA_T4")
A, B, C, D = ARM_NAMES
CONTRASTS = {
    "B_minus_A_noise_without_kernel": {B: 1, A: -1},
    "C_minus_A_kernel_gaussian": {C: 1, A: -1},
    "D_minus_B_kernel_robust": {D: 1, B: -1},
    "interaction_D_minus_B_minus_C_plus_A": {D: 1, B: -1, C: -1, A: 1},
}


def _align(arms):
    if not arms:
        raise ValueError("At least one completed prediction arm is required")
    aligned, expected, reference_actual = {}, None, None
    for name, values in arms.items():
        if name not in ARM_NAMES:
            raise ValueError("Unknown factorial arm")
        ids = np.asarray(values["ids"]).astype(str)
        if ids.ndim != 1 or not len(ids) or len(set(ids.tolist())) != len(ids) or any(not u for u in ids):
            raise ValueError("Unique nonempty IDs required for paired comparisons")
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        if expected is None:
            expected = ids
        elif not np.array_equal(expected, ids):
            raise ValueError("Paired arms have different compound sets; do not intersect/drop rows")
        item = {"ids": ids, "feature_dim": int(values["feature_dim"])}
        if item["feature_dim"] < 1:
            raise ValueError("Positive fixed measurement dimension required")
        for key in ("actual", "predicted", "p_null", "standardized_sse", "raw_nll"):
            val = np.asarray(values[key], float)
            shape = (len(ids), 3) if key in {"actual", "predicted", "p_null"} else (len(ids),)
            if val.shape != shape or not np.isfinite(val).all():
                raise ValueError("Finite prediction/score arrays must align with all declared rows")
            item[key] = val[order]
        if np.any((item["p_null"] < 0) | (item["p_null"] > 1)):
            raise ValueError("NULL probabilities must lie in [0,1]")
        if reference_actual is None:
            reference_actual = item["actual"]
        elif not np.allclose(reference_actual, item["actual"], rtol=1e-10, atol=1e-12):
            raise ValueError("Paired arms have changed actual utility labels")
        aligned[name] = item
    if len({x["feature_dim"] for x in aligned.values()}) != 1:
        raise ValueError("Paired arms changed fixed measurement dimension")
    return aligned


def _vectors(values, fractions):
    """Per-object vectors whose means equal declared fixed-budget metrics."""
    ids, actual = values["ids"], values["actual"]
    n = len(ids)
    vectors = {"measurement/standardized_mse": values["standardized_sse"] / (3 * values["feature_dim"]),
               "measurement/original_space_marginal_nll": values["raw_nll"]}
    definitions = {name: dict(metric=name.split("/")[-1], positive_difference_means="higher error, worse") for name in vectors}
    selections = {}
    for section in ("within_action", "common_budget"):
        for j, action in enumerate(ACTIONS):
            for fraction in fractions:
                base = int(np.floor(float(fraction) * n))
                count = base if section == "within_action" else base // WELL_COSTS[j]
                budget = count * WELL_COSTS[j] if section == "within_action" else base
                for ranking, score in (("expected_gain", values["predicted"][:, j]), ("lowest_p_null", -values["p_null"][:, j])):
                    selected = stable_top_k(score, ids, count)
                    key = f"{section}/{action}/{ranking}/{fraction:g}"
                    selections[key] = dict(selected_ids=ids[selected].tolist(), selected_n=count,
                        used_wells=count * WELL_COSTS[j], budget_wells=budget)
                    q = count / n
                    valname = key + "/per_eligible_net_gain"
                    vectors[valname] = selected * actual[:, j]
                    definitions[valname] = dict(metric="per_eligible_net_gain", action=action, ranking=ranking,
                        fraction=float(fraction), budget_wells=budget, positive_difference_means="higher actual net value, better")
                    if count:
                        fdpname = key + "/fdp"
                        vectors[fdpname] = selected * (actual[:, j] <= 0) / q
                        definitions[fdpname] = dict(metric="fdp", action=action, ranking=ranking,
                            fraction=float(fraction), selected_n=count, budget_wells=budget,
                            positive_difference_means="higher selected NULL fraction, worse")
    return vectors, definitions, selections


def _summary(delta, definitions, *, seed, n_bootstrap):
    names = list(definitions)
    matrix = np.column_stack([delta[name] for name in names])
    n, width = matrix.shape
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((n_bootstrap, width))
    for first in range(0, n_bootstrap, 64):
        stop = min(first + 64, n_bootstrap)
        ix = rng.integers(n, size=(stop - first, n))
        bootstrap[first:stop] = matrix[ix].mean(1)
    low, high = np.quantile(bootstrap, [.025, .975], axis=0)
    return {name: {**definitions[name], "difference": float(matrix[:, j].mean()),
                   "paired_compound_percentile95_interval": [float(low[j]), float(high[j])]}
            for j, name in enumerate(names)}


def compare_arrays(arms, *, fractions=(.05, .10, .25), seed=20260912, n_bootstrap=2000):
    """Compare any completed pairs; unfinished contrasts remain pending."""
    fractions = tuple(float(x) for x in fractions)
    if not fractions or len(set(fractions)) != len(fractions) or any(not 0 < x <= 1 for x in fractions):
        raise ValueError("Declare unique coverage fractions in (0,1]")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("A positive bootstrap replicate count is required")
    aligned = _align(arms)
    vectors, definitions, selections = {}, None, {}
    for arm, values in aligned.items():
        vectors[arm], defs, selections[arm] = _vectors(values, fractions)
        if definitions is not None and definitions != defs:
            raise ValueError("Compared arms have mismatched metric definitions or budgets")
        definitions = defs
    ids = next(iter(aligned.values()))["ids"]
    result = dict(n=len(ids), ids=ids.tolist(), contrasts={}, pending=[], selections=selections,
        score_ties="descending score, lexical compound ID", formal_certificate=False,
        scope="paired objects conditional on fixed fitted models, fixed top-k decisions and shared batches; unadjusted exploratory intervals")
    deltas = {}
    for name, weights in CONTRASTS.items():
        if not set(weights) <= set(vectors):
            result["pending"].append(name)
            continue
        delta = {metric: sum(weight * vectors[arm][metric] for arm, weight in weights.items()) for metric in definitions}
        result["contrasts"][name] = dict(weights=weights, metrics=_summary(delta, definitions,
            seed=seed, n_bootstrap=n_bootstrap))
        deltas[name] = delta
    return result, dict(ids=ids, actual=next(iter(aligned.values()))["actual"],
                        definitions=definitions, deltas=deltas)


def summarize_factorial(root):
    """Read saved prediction artifacts only; never pool seed copies as N×3."""
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text())
    output = dict(per_seed={}, average_seed={}, formal_certificate=False,
        seed_aggregation="average of separate fitted-policy performances; not an ensemble or reranked pooled score",
        uncertainty_scope="compound resampling after averaging across the same fitted seeds; does not estimate population variability of training seeds",
        original_contract_changed=False, final_opened=False, fifth_repeat_opened=False)
    for partition in manifest["evaluation_splits"]:
        per_seed, payloads = {}, {}
        for seed in manifest["seeds"]:
            arms = {}
            for job in manifest["queue"]:
                if int(job["seed"]) != int(seed):
                    continue
                folder = root / job["directory"] / partition
                if not (folder / "metrics.json").exists():
                    continue
                metrics = json.loads((folder / "metrics.json").read_text())
                if metrics.get("formal_certificate") is not False or metrics.get("original_contract_changed") is not False:
                    raise ValueError("Only original-contract development scores may be compared")
                with np.load(folder / "predictions.npz", allow_pickle=False) as z:
                    arrays = {name: z[name] for name in ("ids", "actual", "predicted", "p_null", "standardized_sse", "raw_nll")}
                if arrays["ids"].tolist() != manifest["compound_ids"][partition] or arrays["ids"].tolist() != metrics["ids"]:
                    raise ValueError("Saved comparison rows differ from declared partition identities")
                arrays["feature_dim"] = manifest["data_shape"][-1]
                arms[job["arm"]] = arrays
            if arms:
                per_seed[str(seed)], payloads[str(seed)] = compare_arrays(arms, fractions=manifest["fractions"],
                    seed=int(seed), n_bootstrap=manifest["n_bootstrap"])
        output["per_seed"][partition] = per_seed
        averaged = dict(n=len(manifest["compound_ids"][partition]), contrasts={}, pending=[])
        for name, weights in CONTRASTS.items():
            if any(str(seed) not in payloads or name not in payloads[str(seed)]["deltas"] for seed in manifest["seeds"]):
                averaged["pending"].append(name)
                continue
            first = payloads[str(manifest["seeds"][0])]
            for seed in manifest["seeds"][1:]:
                other = payloads[str(seed)]
                if (not np.array_equal(first["ids"], other["ids"]) or
                        not np.allclose(first["actual"], other["actual"], rtol=1e-10, atol=1e-12)):
                    raise ValueError("Training seeds changed comparison populations or utility labels")
                if first["definitions"] != other["definitions"]:
                    raise ValueError("Training seeds changed comparison metric definitions")
            delta = {metric: np.mean([payloads[str(seed)]["deltas"][name][metric]
                                      for seed in manifest["seeds"]], axis=0) for metric in first["definitions"]}
            averaged["contrasts"][name] = dict(weights=weights, fitted_seeds=manifest["seeds"],
                independent_object_count=len(first["ids"]), metrics=_summary(delta, first["definitions"],
                    seed=int(manifest["seeds"][0]), n_bootstrap=manifest["n_bootstrap"]))
        output["average_seed"][partition] = averaged
    output = plain(output)
    write_json(root / "factorial_comparison.json", output)
    return output
