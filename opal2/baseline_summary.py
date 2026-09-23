"""Descriptive aggregation of saved opened-DEV baseline/probe artifacts.

This reads run outputs only, never the measurement archive. It does not fit,
reselect, relabel, authorize, or concatenate repeats as independent objects.
Incomplete phases are reported as pending rather than partially pooled.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .baseline_policy import ACTIONS, WELL_COSTS, _metrics, _observed


def _read(path):
    return json.loads(Path(path).read_text())


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _unique(ids, expected=None):
    ids = list(map(str, ids))
    if not ids or len(ids) != len(set(ids)) or any(not i for i in ids):
        raise ValueError("Saved IDs must be present and unique")
    if expected is not None and set(ids) != set(map(str, expected)):
        raise ValueError("Saved IDs differ from the declared evaluation population")
    return ids


def _forecast(folder, expected):
    report = _read(folder / "policy.json")
    if report.get("formal_certificate") is not False or report.get("original_contract_changed") is not False:
        raise ValueError("Only unchanged-contract development artifacts can be summarized")
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
        ids = _unique(z["ids"].tolist(), expected)
        arrays = {k: np.asarray(z[k], float) for k in ("actual", "predicted", "p_null", "mc_se",
                   "raw_nll", "standardized_nll", "standardized_mse", "standardized_sse")}
    for k, a in arrays.items():
        shape = (len(ids), 3) if k in {"actual", "predicted", "p_null", "mc_se"} else (len(ids),)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"Saved forecast {k} geometry/values differ")
    trace = {str(r["compound_id"]): r for r in report["row_trace"]}
    if set(trace) != set(ids) or len(report["row_trace"]) != len(ids):
        raise ValueError("Policy and forecast row identities differ")
    for i, unit in enumerate(ids):
        for j, name in enumerate(ACTIONS):
            if not np.isclose(trace[unit]["actual"][name], arrays["actual"][i, j], rtol=1e-12, atol=1e-12):
                raise ValueError("Policy and forecast actual utilities differ")
    return dict(ids=ids, arrays=arrays, report=report)


def _measurements(arrays):
    result = {k: np.mean(arrays[k], axis=0).tolist() for k in
              ("raw_nll", "standardized_nll", "standardized_mse", "mc_se")}
    sse = arrays["standardized_sse"]
    result["max_object_sse_fraction"] = float(sse.max() / sse.sum()) if sse.sum() > 0 else None
    result["median_per_object_standardized_mse"] = float(np.median(arrays["standardized_mse"]))
    return result


def _action_metrics(arrays):
    return [{"action": name, **_metrics(arrays["actual"][:, j], arrays["predicted"][:, j],
                                        arrays["p_null"][:, j])} for j, name in enumerate(ACTIONS)]


def _paired(delta, seed, replicates):
    values = np.asarray(delta, float)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for start in range(0, replicates, 128):
        stop = min(start + 128, replicates)
        means[start:stop] = values[rng.integers(len(values), size=(stop - start, len(values)))].mean(1)
    return dict(mean=float(values.mean()), percentile95_interval=np.quantile(means, [.025, .975]).tolist(),
                n=len(values), replicates=replicates,
                scope="paired compound bootstrap within this repeat, conditional on fitted fold rules and fixed batches",
                formal_certificate=False)


def _pool_baseline(parts, expected, seed, replicates):
    ids = _unique([unit for part in parts for unit in part["ids"]], expected)
    arrays = {k: np.concatenate([p["arrays"][k] for p in parts]) for k in parts[0]["arrays"]}
    result = dict(n=len(ids), ids=ids, action_metrics=_action_metrics(arrays),
                  measurement_metrics=_measurements(arrays), policies=[],
                  fold_summaries=[dict(n=len(p["ids"]), ids=p["ids"],
                      action_metrics=_action_metrics(p["arrays"]), measurement_metrics=_measurements(p["arrays"]))
                      for p in parts],
                  scope="descriptive fold-out predictions; different fitted models, not one frozen rule",
                  formal_certificate=False)
    for section in ("within_action", "common_budget"):
        keys = {(r["action"], r["ranking"], float(r["fraction"])) for r in parts[0]["report"][section]}
        for p in parts[1:]:
            if keys != {(r["action"], r["ranking"], float(r["fraction"])) for r in p["report"][section]}:
                raise ValueError("Fold policies use different declared comparisons")
        for action, ranking, fraction in sorted(keys):
            j = ACTIONS.index(action)
            selections, differences, risk_differences = [], [], []
            budget, expected_total, selected_ids = 0, 0., []
            for p in parts:
                matching = [r for r in p["report"][section] if
                            (r["action"], r["ranking"], float(r["fraction"])) == (action, ranking, fraction)]
                if len(matching) != 1:
                    raise ValueError("Duplicate or absent saved fold policy")
                row = matching[0]
                selected = list(map(str, row["selected_ids"]))
                if len(set(selected)) != len(selected) or not set(selected) <= set(p["ids"]):
                    raise ValueError("A saved policy selected an absent or repeated compound")
                active = np.array([unit in set(selected) for unit in p["ids"]])
                if int(active.sum()) != row["selected_n"]:
                    raise ValueError("Saved policy selection count mismatch")
                y = p["arrays"]["actual"][:, j]
                observed = _observed(y, active, WELL_COSTS[j])
                for field in ("total_net_gain", "used_wells", "selected_null_count"):
                    if not np.isclose(row[field], observed[field], rtol=1e-12, atol=1e-12):
                        raise ValueError("Saved policy outcome does not match selected rows")
                q = active.mean()
                selections.append(active)
                differences.append((active - q) * y)
                risk_differences.append((active - q) * (y <= 0))
                expected_total += float(active.sum() * y.mean())
                budget += int(row["budget_wells"])
                selected_ids.extend(selected)
            active = np.concatenate(selections)
            result["policies"].append(dict(section=section, action=action, ranking=ranking, fraction=fraction,
                **_observed(arrays["actual"][:, j], active, WELL_COSTS[j]), budget_wells=budget,
                selected_ids=selected_ids, selection_rule="concatenate each saved fold selection; no global OOF reranking",
                matched_fold_random_expected_total=expected_total,
                matched_fold_random_expected_per_eligible=expected_total / len(ids),
                paired_value_vs_fold_random=_paired(np.concatenate(differences), seed, replicates),
                paired_false_activation_burden_vs_fold_random=_paired(np.concatenate(risk_differences), seed, replicates),
                formal_certificate=False))
    return result


def _small_probe_metrics(metrics):
    errors = np.asarray(metrics["standardized"]["per_object_sse"], float)
    overall = metrics["standardized"]["overall"]
    width = overall["sse"] / (overall["mse"] * len(errors)) if overall["mse"] > 0 else None
    return {"max_object_standardized_sse_fraction": float(errors.max() / errors.sum()) if errors.sum() > 0 else None,
            "median_per_object_standardized_mse": float(np.median(errors) / width) if width else 0.,
            "standardized": {"overall": metrics["standardized"]["overall"],
                              "slots": metrics["standardized"]["slots"]},
            "physical": {"overall": metrics["physical"]["overall"], "slots": metrics["physical"]["slots"]},
            "r2_evaluation_mean": metrics.get("r2_evaluation_mean"),
            "slot_r2_evaluation_mean": metrics.get("slot_r2_evaluation_mean")}


def _probe(folder, expected):
    report = _read(folder / "metrics.json")
    ids = _unique(report["test_ids"], expected)
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
        stored = list(map(str, z["ids"].tolist()))
    if stored != ids:
        raise ValueError("Probe metric and prediction row order differ")
    m = report["partitions"]["all_test"]
    for key in ("standardized", "physical"):
        for field in ("per_object_sse", "per_object_baseline_sse"):
            a = np.asarray(m[key][field], float)
            if a.shape != (len(ids),) or not np.isfinite(a).all() or np.any(a < 0):
                raise ValueError("Probe error trace has invalid geometry or values")
            total_key = "sse" if field == "per_object_sse" else "baseline_sse"
            if not np.isclose(a.sum(), m[key]["overall"][total_key], rtol=1e-10, atol=1e-10):
                raise ValueError("Probe error trace and total disagree")
    return dict(ids=ids, report=report)


def _pool_probes(arms, expected, seed, replicates):
    result = dict(n=len(expected), arms={}, paired=[], formal_certificate=False,
        scope="one OOF prediction per non-pretraining compound in this repeat; fixed representations, different fold heads")
    ordered_errors = {}
    for arm, parts in arms.items():
        ids = _unique([unit for p in parts for unit in p["ids"]], expected)
        arm_result = {"input_dimension": parts[0]["report"]["input_dimension"],
                      "fold_selected_alphas": [p["report"]["selected_alphas"] for p in parts],
                      "fold_metrics": [_small_probe_metrics(p["report"]["partitions"]["all_test"]) for p in parts],
                      "metrics": {}, "r2_evaluation_mean": None,
                      "r2_evaluation_mean_unavailable_reason": "pooled evaluation-centering requires actual target means, absent from saved probe artifacts; no new data opened"}
        for space in ("standardized", "physical"):
            errors = np.concatenate([np.asarray(p["report"]["partitions"]["all_test"][space]["per_object_sse"], float) for p in parts])
            nulls = np.concatenate([np.asarray(p["report"]["partitions"]["all_test"][space]["per_object_baseline_sse"], float) for p in parts])
            total, baseline = float(errors.sum()), float(nulls.sum())
            dims = []
            for p in parts:
                metric = p["report"]["partitions"]["all_test"][space]["overall"]
                if metric["mse"] > 0:
                    dims.append(metric["sse"] / (metric["mse"] * len(p["ids"])))
            width = float(dims[0]) if dims else None
            if any(not np.isclose(d, width) for d in dims):
                raise ValueError("Probe folds changed target coordinate count")
            arm_result["metrics"][space] = dict(sse=total, baseline_sse=baseline,
                mse=total / len(ids) / width if width else None,
                r2_training_mean=1 - total / baseline if baseline > 0 else None,
                max_object_sse_fraction=float(errors.max() / total) if total > 0 else None,
                median_per_object_mse=float(np.median(errors) / width) if width else 0.,
                denominator="sum of each held-out object's own fold-training-mean squared error")
            if space == "standardized":
                by_id = dict(zip(ids, errors))
                ordered_errors[arm] = np.array([by_id[str(unit)] for unit in expected])
        result["arms"][arm] = arm_result
    for a, b in combinations(ordered_errors, 2):
        av, bv = ordered_errors[a], ordered_errors[b]
        result["paired"].append(dict(arm_a=a, arm_b=b, positive_difference_means="arm_a has lower SSE",
                                     **_paired(bv - av, seed, replicates)))
    return result


def summarize(rootPath):
    """Read saved artifacts and write ``summary.json`` without altering them."""
    root = Path(rootPath)
    cfg = _read(root / "config.json")
    if cfg.get("final_opened") is not False or cfg.get("fifth_repeat_opened") is not False:
        raise ValueError("This summary is restricted to the declared four-role opened DEV run")
    if cfg.get("biology_kernel_active") is not False or cfg.get("original_contract_changed") is not False:
        raise ValueError("Different method or contract from declared baseline comparison")
    split_ids = cfg["split_ids"]
    all_ids = _unique([unit for part in split_ids.values() for unit in part])
    pool = _unique([unit for name in ("validation", "calibration", "evaluation") for unit in split_ids[name]])
    arms = ["raw", "reliability", *cfg["jepa_arms"]]
    seed, reps = int(cfg["seed"]), int(cfg["n_bootstrap"])
    result = dict(purpose="descriptive_saved_DEV_artifact_summary", root=str(root.resolve()),
        aggregation_script=str(Path(__file__).resolve()),
        aggregation_does_not_change_frozen_fit_or_selection_code=True,
        final_opened=False, fifth_repeat_opened=False, biology_kernel_active=False,
        original_contract_changed=False, formal_certificate=False,
        no_new_measurements_read=True, fixed_baseline={}, stability_baseline=[], fixed_probes={},
        stability_probes=[], pending=[],
        repeated_cv_scope="Repeats reuse the same compounds; never pool repeats as independent samples or choose the best repeat")
    for split in ("validation", "calibration", "evaluation"):
        folder = root / "fixed_baseline" / split
        if not (folder / "policy.json").exists():
            result["pending"].append(f"fixed_baseline/{split}")
            continue
        p = _forecast(folder, split_ids[split])
        result["fixed_baseline"][split] = dict(n=len(p["ids"]), action_metrics=_action_metrics(p["arrays"]),
            measurement_metrics=_measurements(p["arrays"]), within_action=p["report"]["within_action"],
            common_budget=p["report"]["common_budget"], fixed_policies=p["report"]["fixed_policies"],
            train_selected_fixed_action=p["report"]["train_selected_fixed_action"])
    for repeat in sorted({int(f["repeat"]) for f in cfg["folds"]}):
        tasks = [f for f in cfg["folds"] if int(f["repeat"]) == repeat]
        paths = [root / "stability_baseline" / f"repeat_{repeat}_fold_{f['fold']}" for f in tasks]
        if not all((p / "policy.json").exists() for p in paths):
            result["pending"].append(f"stability_baseline/repeat_{repeat}")
            continue
        parts = [_forecast(path, f["test_ids"]) for path, f in zip(paths, tasks)]
        pooled = _pool_baseline(parts, all_ids, seed + repeat, reps)
        pooled["repeat"] = repeat
        result["stability_baseline"].append(pooled)
    fixed = root / "fixed_probes" / "fixed383"
    if all((fixed / arm / "metrics.json").exists() for arm in arms):
        for arm in arms:
            p = _probe(fixed / arm, pool)
            result["fixed_probes"][arm] = dict(input_dimension=p["report"]["input_dimension"],
                selected_alphas=p["report"]["selected_alphas"],
                partitions={key: _small_probe_metrics(value) for key, value in p["report"]["partitions"].items()})
        if (fixed / "comparisons.json").exists():
            result["fixed_probe_paired"] = _read(fixed / "comparisons.json")
    else:
        result["pending"].append("fixed_probes")
    for repeat in sorted({int(f["repeat"]) for f in cfg["probe_folds"]}):
        tasks = [f for f in cfg["probe_folds"] if int(f["repeat"]) == repeat]
        paths = [root / "stability_probes" / f"repeat_{repeat}_fold_{f['fold']}" for f in tasks]
        if not all((path / arm / "metrics.json").exists() for path in paths for arm in arms):
            result["pending"].append(f"stability_probes/repeat_{repeat}")
            continue
        parts = {arm: [_probe(path / arm, f["test_ids"]) for path, f in zip(paths, tasks)] for arm in arms}
        pooled = _pool_probes(parts, pool, seed + repeat, reps)
        pooled["repeat"] = repeat
        result["stability_probes"].append(pooled)
    result["complete"] = not result["pending"]
    result = _plain(result)
    (root / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def compact(summary):
    lines = ["Saved DEV summary (no authorization; repetitions are not new independent samples)"]
    for split, values in summary["fixed_baseline"].items():
        for metric in values["action_metrics"]:
            lines.append(f"fixed {split} {metric['action']}: Spearman={metric['spearman']} NULL_AUC={metric['null_auc']}")
    for arm, values in summary["fixed_probes"].items():
        metric = values["partitions"].get("evaluation", values["partitions"]["all_test"])
        lines.append(f"probe {arm}: eval MSE={metric['standardized']['overall']['mse']} eval-centered R2={metric['r2_evaluation_mean']}")
    lines.append(f"Complete CV repeats: baseline={len(summary['stability_baseline'])}, probes={len(summary['stability_probes'])}")
    if summary["pending"]:
        lines.append("Pending: " + ", ".join(summary["pending"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args(argv)
    print(compact(summarize(args.root)))


if __name__ == "__main__":
    main()
