"""Refresh R2 tables from saved JSON metadata; never load measurements or fit models.

Run with --refresh after a dataset runner finishes.  Missing inputs remain pending.
Only the report directory is written.  Original summaries and logs are preserved.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = "reports/r2_four_dataset_closure_20260918_v1"
INPUTS = {
    "EU": "runs/r2_core_comparison_20260917_v1/summary.json",
    "JUMP": "runs/jump_r2_completion_20260918_v1/summary.json",
    "LINCS": "reports/lincs_r2_completion_20260918_v1/summary.json",
    "RxRx3": "runs/rxrx3_r2_completion_20260918_v1/summary.json",
}
RUNS = {k: Path(v).parent for k, v in INPUTS.items()}
RUNS["LINCS"] = Path("runs/lincs_r2_completion_20260918_v1")
MAIN_ARMS = ["CORE_ORIGINAL", "RIDGE_REF", "HR_REF", "STATE_REF"] + [
    f"DIRECT_{scope}_{family}_COHERENT"
    for scope in ("TRAIN_MATCHED", "ACCESS_MATCHED")
    for family in ("RIDGE", "EXTRATREES", "HISTGB")
] + ["CONSTANT_ACCESS_MATCHED"]
RISK_PANEL_ARMS = ("CORE_ORIGINAL", "DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL")
MAIN_PAIRS = [
    ("CORE_ORIGINAL", "CORE_LOCAL_GAUSSIAN"),
    ("CORE_AMP_GAUSSIAN", "CORE_LOCAL_GAUSSIAN"),
    ("CORE_ORIGINAL", "CORE_AMP_GAUSSIAN"),
    ("HR_REF", "RIDGE_REF"), ("STATE_REF", "HR_REF"),
    ("STATE_REF", "CORE_ORIGINAL"),
] + [(mean + "_AMP_GAUSSIAN", mean + "_LOCAL_GAUSSIAN")
     for mean in ("RIDGE_REF", "HR_REF", "STATE_REF")] + [
    (mean, mean + "_AMP_GAUSSIAN") for mean in ("RIDGE_REF", "HR_REF", "STATE_REF")
] + [(a + suffix, b + suffix)
     for suffix in ("_LOCAL_GAUSSIAN", "_AMP_GAUSSIAN")
     for a, b in (("HR_REF", "RIDGE_REF"), ("STATE_REF", "HR_REF"))] + [
    (arm, "CORE_ORIGINAL") for arm in MAIN_ARMS if arm.startswith("DIRECT_")
    or arm == "CONSTANT_ACCESS_MATCHED"
]
HISTORY = {
    "EU": ["runs/eu_core_cc904_20260917_v1"],
    "JUMP": ["runs/gram_oof_20260914_v1"],
    "LINCS": ["runs/lincs_biology_four_arm_20260915_v1",
              "runs/lincs_state_biology_20260916_v1",
              "runs/lincs_empirical_radial_20260916_v1"],
    "RxRx3": [],
}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def close(a, b):
    return a is None or b is None or math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-10)


def scalar_fields(record):
    return {k: v for k, v in record.items() if not isinstance(v, (dict, list))}


def normalize_metrics(dataset, source, summary, audits):
    rows, selected = [], []
    for arm, values in summary.get("metrics", {}).items():
        n = values.get("n", summary["n"])
        for policy, p in values.get("policies", {}).items():
            ids = p.get("selected_ids")
            row = dict(dataset=dataset, arm=arm, policy=policy, n=n,
                       main_table=(arm in MAIN_ARMS or arm == "L_TRAIN_MATCHED_FULL") and policy == "lambda_0.2",
                       **{k: v for k, v in values.items() if k not in ("policies", "n")},
                       **{k: v for k, v in p.items() if k not in ("selected_ids", "false_activation")})
            row["false_activation_per_candidate"] = p["null_selected"] / n
            row["false_positive_rate_among_NULL"] = p.get("false_activation")
            row["source"] = source + ("/" if "#" in source else "#/") + "metrics/" + arm + "/policies/" + policy
            rows.append(row)
            if ids is not None:
                selected.extend(dict(dataset=dataset, arm=arm, policy=policy, candidate_id=i) for i in ids)
                audits.append(dict(dataset=dataset, check="selected_list_count", arm=arm, policy=policy,
                                   passed=len(ids) == len(set(ids)) == p["activated"],
                                   observed=len(ids), expected=p["activated"], source=row["source"]))
            for key, expected in (("value_per_candidate", p["total_value"] / n),
                                  ("fdp", p["null_selected"] / p["activated"] if p["activated"] else None),
                                  ("extra_wells", 2 * p["activated"])):
                audits.append(dict(dataset=dataset, check=key, arm=arm, policy=policy,
                                   passed=close(p.get(key), expected), observed=p.get(key),
                                   expected=expected, source=row["source"]))
    return rows, selected


def normalized_pairs(dataset, source, summary, audits):
    records = dict(summary.get("paired", {}))
    records.update(summary.get("paired_mean_and_law_attribution", {}))
    for arm, value in summary.get("paired_vs_CORE", {}).items():
        if arm != "CORE_ORIGINAL":
            records[arm + " minus CORE_ORIGINAL"] = value
    rows = []
    for contrast, metrics in records.items():
        a, b = contrast.split(" minus ", 1)
        for original_metric, blocks in metrics.items():
            policy = None
            metric = original_metric
            for key in ("lambda_0.2", "lambda_0"):
                if metric.startswith(key + "_"):
                    policy, metric = key, metric[len(key) + 1:]
                elif metric.endswith("_" + key):
                    policy, metric = key, metric[:-len(key)-1]
            if metric == "policy_value_per_candidate":
                metric, policy = "value_per_candidate", policy or "lambda_0.2"
            if metric == "false_activation_per_candidate" and policy is None:
                policy = "lambda_0.2"
            expected = None
            ma, mb = summary["metrics"].get(a, {}), summary["metrics"].get(b, {})
            if policy and ma and mb:
                pa, pb = ma["policies"][policy], mb["policies"][policy]
                if metric == "false_activation_per_candidate":
                    expected = (pa["null_selected"] - pb["null_selected"]) / summary["n"]
                elif metric == "value_per_candidate":
                    expected = pa[metric] - pb[metric]
            elif metric in ma and metric in mb and isinstance(ma[metric], (int, float)):
                expected = ma[metric] - mb[metric]
            for block, value in blocks.items():
                row = dict(dataset=dataset, contrast=contrast, arm_a=a, arm_b=b,
                           metric=metric, policy=policy, block=block,
                           main_comparison=((a, b) in MAIN_PAIRS or (b, a) in MAIN_PAIRS or a == "L_TRAIN_MATCHED_FULL") and policy in (None, "lambda_0.2"),
                           difference=value["difference"], ci95_low=value["ci95"][0],
                           ci95_high=value["ci95"][1],
                           denominator="all candidates" if "per_candidate" in metric else "metric-specific",
                           source=source, source_metric=original_metric)
                rows.append(row)
                audits.append(dict(dataset=dataset, check="paired_point_difference", contrast=contrast,
                                   metric=metric, policy=policy, block=block,
                                   passed=close(value["difference"], expected),
                                   observed=value["difference"], expected=expected, source=source))
    return rows


def canonical_main_pairs(rows):
    """Use a common contrast direction, reversing saved CIs rather than recomputing them."""
    canonical = {}
    for original in rows:
        if not original["main_comparison"]:
            continue
        row = dict(original)
        a, b = row["arm_a"], row["arm_b"]
        reverse = (a, b) not in MAIN_PAIRS and (b, a) in MAIN_PAIRS
        row["source_contrast"] = original["contrast"]
        row["sign_reversed_from_source"] = reverse
        if reverse:
            row.update(arm_a=b, arm_b=a, contrast=b+" minus "+a,
                       difference=-original["difference"], ci95_low=-original["ci95_high"],
                       ci95_high=-original["ci95_low"])
        key = (row["dataset"],row["contrast"],row["metric"],row["policy"],row["block"])
        if key not in canonical or not reverse:
            canonical[key] = row
    return list(canonical.values())


def selected_overlaps(dataset, source, summary, audits):
    rows = []
    for policy in ("lambda_0.2", "lambda_0"):
        sets = {arm: set(m["policies"][policy]["selected_ids"])
                for arm, m in summary["metrics"].items()
                if "selected_ids" in m.get("policies", {}).get(policy, {})}
        for a, b in itertools.combinations(sets, 2):
            sa, sb = sets[a], sets[b]
            rows.append(dict(dataset=dataset, policy=policy, arm_a=a, arm_b=b,
                             selected_a=len(sa), selected_b=len(sb), intersection=len(sa & sb),
                             union=len(sa | sb), jaccard=len(sa & sb) / max(len(sa | sb), 1),
                             retention_of_b=len(sa & sb) / max(len(sb), 1),
                             symmetric_difference=len(sa ^ sb),
                             main_comparison=(a, b) in MAIN_PAIRS or (b, a) in MAIN_PAIRS,
                             intersection_ids=sorted(sa & sb), only_a=sorted(sa - sb),
                             only_b=sorted(sb - sa), source=source))
        for arm, p in summary.get("selected_overlap_with_CORE", {}).items():
            if arm in sets and policy in p:
                expected = len(sets[arm] & sets["CORE_ORIGINAL"])
                audits.append(dict(dataset=dataset, check="saved_overlap", arm=arm, policy=policy,
                                   passed=p[policy]["intersection"] == expected,
                                   observed=p[policy]["intersection"], expected=expected, source=source))
    return rows


def calibration_rows(dataset, source, summary, audits):
    rows = []
    for arm, payload in summary.get("calibration", {}).items():
        if "regions" in payload:
            regions = [("lambda_0.2", r["scope"], r,
                        r.get("calibration_gap_per_candidate", {})) for r in payload["regions"]]
        else:
            regions = [(policy, scope, r,
                        payload.get("absolute_gap_intervals_per_candidate", {}).get(
                            policy + "_" + scope + "_NULL_gap_per_candidate", {}))
                       for policy in ("lambda_0.2", "lambda_0")
                       for scope, r in payload.get(policy, {}).items()]
        for policy, scope, r, intervals in regions:
            predicted = r.get("predicted", r.get("predicted_null"))
            actual = r.get("actual", r.get("actual_null"))
            gap = actual - predicted
            for block, v in (intervals or {"not_recorded": {}}).items():
                rows.append(dict(dataset=dataset, arm=arm, policy=policy, region=scope,
                                 selected_n=r["n"], predicted_NULL=predicted, actual_NULL=actual,
                                 NULL_count_gap=gap, gap_per_all_candidates=gap / summary["n"],
                                 brier=r.get("brier"), block=block,
                                 ci95_low=v.get("ci95", [None, None])[0],
                                 ci95_high=v.get("ci95", [None, None])[1], source=source))
                if v:
                    audits.append(dict(dataset=dataset, check="calibration_gap", arm=arm,
                                       policy=policy, scope=scope, block=block,
                                       passed=close(v["difference"], gap / summary["n"]),
                                       observed=v["difference"], expected=gap / summary["n"], source=source))
    return rows


def mc_rows(dataset, source, summary, audits):
    rows = []
    for arm, seeds in summary.get("monte_carlo_sensitivity", summary.get("monte_carlo", {})).items():
        for seed in seeds:
            for policy, p in seed["policies"].items():
                base = summary["metrics"][arm]["policies"][policy]
                chosen, original = set(p.get("selected_ids", [])), set(base.get("selected_ids", []))
                diff = len(chosen ^ original)
                recorded = p.get("list_symmetric_difference", p.get("symmetric_difference_from_main", p.get("list_symmetric_difference_from_primary")))
                rows.append(dict(dataset=dataset, arm=arm, policy=policy, seed_offset=seed["seed_offset"],
                                 samples=seed.get("samples", summary.get("samples", 100000)),
                                 symmetric_difference=diff, intersection=len(chosen & original),
                                 jaccard=len(chosen & original) / max(len(chosen | original), 1),
                                 null_selected=p["null_selected"], total_value=p["total_value"],
                                 null_count_change=p["null_selected"] - base["null_selected"],
                                 total_value_change=p["total_value"] - base["total_value"], source=source))
                audits.append(dict(dataset=dataset, check="MC_selected_list_difference", arm=arm,
                                   policy=policy, seed_offset=seed["seed_offset"],
                                   passed=recorded is None or recorded == diff,
                                   observed=recorded, expected=diff, source=source))
    return rows


def controls_rows(dataset, source, summary):
    rows = []
    if "cached_fixed_controls" in summary:
        for cell in summary["cached_fixed_controls"]:
            ident = {k: v for k, v in cell.items() if k not in ("fixed", "random_same_budget")}
            fixed, random = cell["fixed"], cell["random_same_budget"]
            rows += [dict(dataset=dataset, **ident, control="NEVER_ADD", total_value=fixed["stop_total_value"],
                          extra_wells=0, same_budget=False, source=source),
                     dict(dataset=dataset, **ident, control="ALWAYS_ADD", total_value=fixed["add_all_total_value"],
                          extra_wells=fixed["add_all_extra_wells"], same_budget=False, source=source),
                     dict(dataset=dataset, **ident, control="RANDOM_SAME_BUDGET", same_budget=True,
                          interval_scope="cell-specific empirical randomization; quantiles must not be summed",
                          **random, source=source)]
    else:
        for label, record in summary.get("fixed_and_random_controls", {}).items():
            if label == "same_budget":
                for name, value in record.items():
                    rows.append(dict(dataset=dataset, control=name, same_budget=True,
                                     **{k: v for k, v in value.items() if k != "selected_ids"}, source=source))
            else:
                rows.append(dict(dataset=dataset, control=label, same_budget=label == "random_same_budget",
                                 **{k: v for k, v in record.items() if k != "selected_ids"}, source=source))
    return rows


def cost_rows(project, dataset, source, summary, missing):
    rows = []
    cells = summary.get("costs_by_deployment_cell", summary.get("reference_costs", []))
    for cell in cells:
        ident = {k: v for k, v in cell.items() if k not in ("arms", "resource_counts")}
        if dataset == "RxRx3":
            condition_cell = next((c for c in summary.get("cells", []) if c.get("cell") == cell.get("fold")), {})
            ident.update({k:v for k,v in condition_cell.items() if not isinstance(v, (dict, list))})
        resources = cell.get("resource_counts", {})
        for arm, record in cell["arms"].items():
            if "resources" in record:
                counts = record["resources"]
                for policy, costs in record["policies"].items():
                    train_n = counts.get("backbone_training_objects")
                    val_n = counts.get("validation_objects")
                    rows.append(dict(dataset=dataset, arm=arm, policy=policy, **ident, **counts, **costs,
                                     model_train_and_validation_wells=4*(train_n+val_n) if train_n is not None and val_n is not None else None,
                                     training_wells_basis="4 wells per recorded TRAIN/VALIDATION object",
                                     ledger_scope="one deployment cell; do not sum CV cells as one purchase",
                                     source=source))
                continue
            # Source EU/JUMP ledger already records the REF debit. CAL is separate.
            uses_ref = not arm.startswith("DIRECT_TRAIN_MATCHED")
            ref_wells = record.get("reference_new_wells", 0)
            ref_x_wells = resources.get("reference_if_X_already_available", 0) if uses_ref else 0
            cal_common = resources.get("distribution_calibration_wells")
            cal_required = not arm.endswith(("_LOCAL_GAUSSIAN", "_AMP_GAUSSIAN", "_CLASSIFIER_RAW")) and arm != "CONSTANT_ACCESS_MATCHED"
            cal_wells = cal_common if cal_required else 0
            cal_x_wells = 0.75 * cal_wells if cal_wells is not None else None
            counts = dict(reference_all_new_wells=ref_wells, reference_if_X_available_wells=ref_x_wells,
                          calibration_all_new_wells=cal_wells, calibration_if_X_available_wells=cal_x_wells,
                          common_evaluation_CAL_wells=cal_common,
                          model_train_and_validation_wells=resources.get("model_train_and_validation_wells"),
                          raw_classifier_Gamma_distribution_scoring_still_uses_CAL=arm.endswith("_CLASSIFIER_RAW"),
                          reference_outcome_rows=ref_wells / 4,
                          calibration_outcome_rows=cal_wells / 4 if cal_wells is not None else None,
                          training_scope="TRAIN+REF" if arm.startswith("DIRECT_ACCESS_MATCHED") or arm == "CONSTANT_ACCESS_MATCHED" else "TRAIN")
            net = record["action_net_value"]
            rows.append(dict(dataset=dataset, arm=arm, policy="lambda_0.2", **ident, **counts,
                             **record, combined_setup_all_new_wells=ref_wells+cal_wells if cal_wells is not None else None,
                             reference_plus_CAL_all_new_total_value=net-.01*(ref_wells+cal_wells) if cal_wells is not None else None,
                             reference_plus_CAL_X_available_total_value=net-.01*(ref_x_wells+cal_x_wells) if cal_x_wells is not None else None,
                             source_resource_counts=resources,
                             ledger_scope="one deployment cell; do not sum CV cells as one purchase",
                             CAL_basis="recorded all-new CAL wells; 3/4 if initial X exists; zero for Gaussian/constant/raw policy interface",
                             source=source))
    if not cells:
        missing.append(dict(dataset=dataset, component="deployment_costs", status="not_recorded", action="retain missing until metadata exists"))
    return rows


def audit_costs(dataset, source, summary, rows, audits):
    for arm, metrics in summary["metrics"].items():
        for policy, p in metrics["policies"].items():
            selected = [r for r in rows if r["arm"] == arm and r["policy"] == policy]
            if selected:
                observed = sum(r["action_net_value"] for r in selected)
                audits.append(dict(dataset=dataset, check="cell_action_values_sum_to_policy_value", arm=arm,
                                   policy=policy, passed=close(observed, p["total_value"]),
                                   observed=observed, expected=p["total_value"], source=source))
    for row in rows:
        ref_key = "reference_all_new_total_value" if "reference_all_new_total_value" in row else "new_reference_total_net_value"
        if ref_key in row:
            expected = row["action_net_value"] - .01 * row["reference_all_new_wells"]
            audits.append(dict(dataset=dataset, check="reference_debit", arm=row["arm"], policy=row["policy"],
                               fold=row.get("fold"), half=row.get("half"), passed=close(row[ref_key], expected),
                               observed=row[ref_key], expected=expected, source=source))


def compute_ledger(project, dataset, run, missing):
    """Read timing metadata only; inclusive totals remain unallocated."""
    rows, seen = [], set()
    roots = [(run, "current_R2")] + [(Path(p), "historical_reused_or_reference") for p in HISTORY[dataset]]
    if dataset == "JUMP":
        roots.append((Path("runs/jump_matched_l_20260918_v1"), "matched_L_supplement"))
    filenames = {"status.json", "complete.json", "training_complete.json", "component_times.json",
                 "timings.json", "compute_times.json", "aggregation_restore.json", "fit_summary.json", "mc_sensitivity.json"}
    for root, phase in roots:
        if not (project/root).exists():
            continue
        for path in sorted((project/root).rglob("*.json")):
            timing_candidate = (path.name in filenames or path.name.startswith("DIRECT_")
                                or "direct_access_models" in path.parts or "direct_train_models" in path.parts
                                or ("matched_l" in str(root) and path.name.startswith("block_")))
            if not timing_candidate or "source_snapshot" in path.parts:
                continue
            rel = str(path.relative_to(project))
            if rel in seen:
                continue
            seen.add(rel)
            if path.name == "training_complete.json":
                # LINCS history includes unrelated alternatives; include the means actually reused.
                if "lincs_" in rel and not any(p in rel for p in ("/HR_fit/", "/A_OLD_GENERIC/", "/STATE50/")):
                    continue
                if "gram_oof_20260914" in rel:
                    continue  # G_DIRECT timing is unrelated to historical L_GRAM.
            payload = read_json(path)
            def walk(value, pointer=""):
                if not isinstance(value, dict):
                    return
                for key, item in value.items():
                    address = pointer + "/" + key
                    if isinstance(item, (float, int)) and not isinstance(item, bool) and (
                        key.endswith("seconds") or key in ("duration_s", "elapsed_s", "wall_s")):
                        grouped = path.name in ("component_times.json", "timings.json", "compute_times.json")
                        bundle = grouped and (pointer.strip("/") == "full_mean_fit"
                                             or pointer.strip("/").startswith("direct_fit_"))
                        block_total = path.name.startswith("block_") and key == "wall_seconds"
                        inclusive = path.name in ("status.json", "complete.json", "aggregation_restore.json", "mc_sensitivity.json") or bundle or block_total
                        component = str(path.parent.relative_to(project/root))
                        if grouped:
                            component += "/" + pointer.strip("/")
                        elif pointer or path.name.startswith(("DIRECT_", "block_")):
                            component += "/" + path.stem + pointer
                        if component == ".":
                            component = "run_total"
                        cpu_time = "cpu" in key.lower()
                        rows.append(dict(dataset=dataset, phase=phase, component=component or "run_total",
                                         duration_seconds=item, time_kind="process_cpu" if cpu_time else "wall",
                                         wall_seconds=None if cpu_time else item,
                                         process_cpu_seconds=item if cpu_time else None,
                                         source=rel, json_pointer=address,
                                         scope="cumulative_or_inclusive_total" if inclusive else "recorded_component",
                                         overlaps_with=("same-block full_profile_sampling_seconds" if block_total else
                                                        "nested HR/A/STATE training_complete records" if bundle and pointer.strip("/") == "full_mean_fit" else
                                                        "per-family regression/classification/prediction/calibration timers" if bundle else None),
                                         allocation="unallocated; do not add to enclosed components" if inclusive
                                         else "reported component; may include bundled models/stages",
                                         cpu_threads=value.get("cpu_threads"), outer_fold=value.get("outer_fold"),
                                         dose_uM=value.get("dose_uM"), query_rows=value.get("query_rows"),
                                         process_lifetime_max_rss_bytes=value.get("process_lifetime_max_rss_bytes"),
                                         memory_interpretation=value.get("memory_interpretation"), status="recorded"))
                    elif isinstance(item, dict):
                        walk(item, address)
            walk(payload)
    # Explicit missing entries preserve the distinction between unknown and zero.
    needs = {"RIDGE_mean_fit": r"ridge_mean_fit|mean.*ridge.*fit|ridge.*mean.*fit",
             "matched_error_law_fit": r"law_fit|distribution_fit",
             "joint_radial_calibration_fit": r"(CORE|RIDGE_REF|HR_REF|STATE_REF).*(calibration|radial)_fit",
             "direct_calibration_and_scalar_law": r"calibration_and_scalar_law_seconds",
             "direct_per_family_fit": r"regression_grid_fit|classification_grid_fit"}
    for component, pattern in needs.items():
        recorded = [r for r in rows if re.search(pattern, r["component"] + r["json_pointer"], re.I)]
        if not recorded:
            missing.append(dict(dataset=dataset, component="compute/" + component, status="not_recorded",
                                evidence="No separately identified duration in inspected timing metadata; bundled/run totals remain unallocated",
                                action="do not retrain to recover timing"))
    if not any("100k_score" in r["component"] or "sampling" in r["component"].lower() for r in rows):
        missing.append(dict(dataset=dataset, component="compute/100k_scoring_per_arm", status="not_recorded",
                            action="retain overall invocation time only; no proportional allocation"))
    if dataset == "JUMP":
        missing.append(dict(dataset=dataset, component="historical_L_GRAM_compute", status="not_recorded",
                            evidence="gram_oof fold/run totals include other arms; L-specific time unavailable",
                            action="historical all-arm totals are traceable but are not assigned to L_GRAM"))
    return rows


def group_count(project, dataset, manifest, n):
    for key in ("chemical_groups", "identity_count", "n_chemical_groups", "chemical_group_count"):
        if key in manifest:
            return manifest[key]
    if dataset == "EU":
        meta = read_json(project/"reports/eu_core_development_20260917_v1/prepared_data_cc904/metadata.json") or {}
        excluded = set(meta.get("excluded_incomplete_ids", []))
        plan = project/"reports/eu_core_development_20260917_v1/identity_split_plan.csv"
        if plan.exists():
            with plan.open(newline="") as handle:
                groups = {r["object_id"]: r["connectivity"] for r in csv.DictReader(handle)
                          if r["object_id"] not in excluded}
            if len(groups) == n:
                return len(set(groups.values()))
    return None


def append_matched_l(project, matched, tables, audits):
    source = "runs/jump_matched_l_20260918_v1/summary.json"
    arm = "L_TRAIN_MATCHED_FULL"
    normalized = dict(n=matched["n"], metrics=matched["metrics"],
                      paired={arm + " minus CORE_ORIGINAL": matched["paired_L_minus_CORE"]})
    rows, selected = normalize_metrics("JUMP", source, normalized, audits)
    tables["all_arms"].extend(r for r in rows if r["arm"] == arm)
    tables["selected_candidates"].extend(r for r in selected if r["arm"] == arm)
    tables["paired_intervals"].extend(normalized_pairs("JUMP", source, normalized, audits))
    tables["selection_overlap"].extend(selected_overlaps("JUMP", source, normalized, audits))
    calibration = {arm: {}}
    for policy, details in matched["policies"].items():
        for r in details["calibration"][arm]["regions"]:
            scope = "CORE_selected" if r["scope"] == "other_selected" else r["scope"]
            calibration[arm].setdefault(policy, {})[scope] = r
            calibration[arm].setdefault("absolute_gap_intervals_per_candidate", {})[
                policy + "_" + scope + "_NULL_gap_per_candidate"] = r["gap_per_candidate"]
    tables["selected_calibration"].extend(calibration_rows("JUMP", source, dict(n=matched["n"], calibration=calibration), audits))
    for r in matched["resources"]:
        c = r["counts"]
        tables["deployment_costs"].append(dict(dataset="JUMP", arm=arm, policy="lambda_0.2", fold=r["fold"],
                                               **c, **r["value_scenarios"][arm], source=source,
                                               ledger_scope="one deployment cell; actual used labels differ from equal available-resource ceiling"))
    mc_summary = read_json(project/"runs/jump_matched_l_20260918_v1/mc_sensitivity.json") or {}
    for offset in (100000, 200000):
        recorded = next((r for r in mc_summary.get("per_seed", []) if r.get("seed_offset") == offset), None)
        if recorded:
            seed_payload = dict(n=matched["n"], metrics=matched["metrics"], monte_carlo={arm:[
                dict(seed_offset=offset, samples=mc_summary.get("samples_per_seed",100000), policies=recorded["policies"])]})
            tables["MC_seed_stability"].extend(mc_rows("JUMP", "runs/jump_matched_l_20260918_v1/mc_sensitivity.json", seed_payload, audits))
            continue
        path = project/f"runs/jump_matched_l_20260918_v1/summary_mc{offset}.json"
        saved = read_json(path)
        if saved and saved.get("complete"):
            seed_payload = dict(n=matched["n"], metrics=matched["metrics"],
                                monte_carlo={arm:[dict(seed_offset=offset, samples=saved["samples"],
                                                      policies=saved["metrics"][arm]["policies"])]})
            tables["MC_seed_stability"].extend(mc_rows("JUMP", str(path.relative_to(project)), seed_payload, audits))


def format_number(value):
    return "—" if value is None else (str(value) if isinstance(value, str) else f"{value:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-read all cached JSON and replace derived reports")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output if args.output.is_absolute() else project/args.output
    output.mkdir(parents=True, exist_ok=True)
    audits, missing, datasets = [], [], {}
    tables = {name: [] for name in ("all_arms", "paired_intervals", "selected_candidates", "selection_overlap",
                                   "selected_calibration", "MC_seed_stability", "fixed_random_controls",
                                   "deployment_costs", "compute_costs", "rxrx3_per_dose_all_arms",
                                   "rxrx3_per_dose_main_comparisons", "rxrx3_equal_chemical_weight_scores")}
    appendices = {}
    for dataset, source in INPUTS.items():
        run = RUNS[dataset]
        status = read_json(project/run/"status.json") or {}
        manifest = read_json(project/run/"run_manifest.json") or {}
        try:
            summary = read_json(project/source)
        except json.JSONDecodeError:
            summary = None
            missing.append(dict(dataset=dataset, component="summary", status="being_written", source=source))
        if not summary or not summary.get("complete") or (status and status.get("state") != "COMPLETE"):
            datasets[dataset] = dict(status="running" if status.get("state") == "RUNNING" else "pending", source=source, run_state=status.get("state", "not_started"),
                                     n=manifest.get("n"), chemical_groups=group_count(project, dataset, manifest, manifest.get("n")),
                                     deployment_cells=len(manifest["cells"]) if isinstance(manifest.get("cells"), list) else None,
                                     counts_status="frozen metadata plan; R2 metrics pending",
                                     samples=manifest.get("samples"),
                                     stage=status.get("stage"), blocks={key:"pending" for key in
                                     ("main_comparisons", "calibration", "fixed_random_MC", "cost_ledger")})
            missing.append(dict(dataset=dataset, component="R2_results", status="pending", source=source,
                                action="refresh after runner saves complete summary"))
            continue
        arm_rows, selected = normalize_metrics(dataset, source, summary, audits)
        tables["all_arms"].extend(arm_rows)
        tables["selected_candidates"].extend(selected)
        pairs = normalized_pairs(dataset, source, summary, audits)
        tables["paired_intervals"].extend(pairs)
        tables["selection_overlap"].extend(selected_overlaps(dataset, source, summary, audits))
        tables["selected_calibration"].extend(calibration_rows(dataset, source, summary, audits))
        tables["MC_seed_stability"].extend(mc_rows(dataset, source, summary, audits))
        tables["fixed_random_controls"].extend(controls_rows(dataset, source, summary))
        costs = cost_rows(project, dataset, source, summary, missing)
        audit_costs(dataset, source, summary, costs, audits)
        tables["deployment_costs"].extend(costs)
        tables["compute_costs"].extend(compute_ledger(project, dataset, run, missing))
        if dataset == "RxRx3":
            for dose, payload in summary.get("by_dose", {}).items():
                dose_summary = dict(n=payload["n_conditions"], metrics=payload["metrics"])
                dose_audits = []
                dose_rows, _ = normalize_metrics(dataset, source+"#/by_dose/"+dose, dose_summary, dose_audits)
                for row in dose_rows:
                    row.update(dose_uM=dose, chemical_groups=payload["n_chemical_groups"],
                               estimand="condition within dose")
                audits.extend(dict(**check, dose_uM=dose) for check in dose_audits)
                tables["rxrx3_per_dose_all_arms"].extend(dose_rows)
                tables["rxrx3_per_dose_main_comparisons"].extend(r for r in dose_rows if r["main_table"])
            tables["rxrx3_equal_chemical_weight_scores"].extend(
                dict(arm=arm, estimand="equal chemical weight descriptive score", **values)
                for arm,values in summary.get("equal_chemical_weight_descriptive_scores", {}).items())
            if not summary.get("by_dose"):
                missing.append(dict(dataset=dataset, component="by_dose_results", status="not_recorded"))
        actual_pairs = {(r["arm_a"], r["arm_b"]) for r in pairs}
        missing_pairs = []
        for a, b in MAIN_PAIRS:
            if (a, b) not in actual_pairs and (b, a) not in actual_pairs:
                missing_pairs.append(a+" minus "+b)
                missing.append(dict(dataset=dataset, component="paired_interval", contrast=a+" minus "+b,
                                    status="not_recorded", action="do not infer CI from other contrasts"))
        group_n = group_count(project, dataset, manifest, summary["n"])
        if group_n is None:
            missing.append(dict(dataset=dataset, component="chemical_group_count", status="not_recorded_in_summary_or_manifest"))
        expected_joint = {"CORE_ORIGINAL", "CORE_LOCAL_GAUSSIAN", "CORE_AMP_GAUSSIAN"} | {
            mean+suffix for mean in ("RIDGE_REF", "HR_REF", "STATE_REF")
            for suffix in ("", "_LOCAL_GAUSSIAN", "_AMP_GAUSSIAN")}
        expected_arms = expected_joint | set(MAIN_ARMS) | {
            f"DIRECT_{scope}_{family}_{interface}" for scope in ("TRAIN_MATCHED", "ACCESS_MATCHED")
            for family in ("RIDGE", "EXTRATREES", "HISTGB") for interface in ("CLASSIFIER_RAW", "CLASSIFIER_CAL")}
        mc = summary.get("monte_carlo_sensitivity",summary.get("monte_carlo", {}))
        blocks = dict(
            main_comparisons="complete" if expected_arms.issubset(summary["metrics"]) and not missing_pairs else "missing",
            calibration="complete" if expected_arms.issubset(summary.get("calibration", {})) else "missing",
            fixed_random_MC="complete" if (summary.get("cached_fixed_controls") or summary.get("fixed_and_random_controls"))
                and all({100000,200000}.issubset({s.get("seed_offset") for s in mc.get(arm, [])}) for arm in expected_joint) else "missing",
            cost_ledger="recorded_resources; incomplete_component_timing" if costs else "missing")
        if dataset == "RxRx3" and not summary.get("by_dose"):
            blocks["dose_stratified_results"] = "missing"
        result_gaps = [k for k,v in blocks.items() if v == "missing"]
        for block in result_gaps:
            missing.append(dict(dataset=dataset, component="required_results/"+block, status="missing",
                                action="refresh when this saved result block is available"))
        datasets[dataset] = dict(status="complete" if not result_gaps else "incomplete_results", source=source, run_state=status.get("state"),
                                n=summary["n"], chemical_groups=group_n, arm_count=len(summary["metrics"]),
                                deployment_cells=len(summary.get("reference_costs", summary.get("costs_by_deployment_cell", []))),
                                primary_unit=summary.get("primary_unit", "candidate"),
                                dose_strata=len(summary.get("by_dose", {})) if dataset == "RxRx3" else None,
                                samples=summary.get("samples", manifest.get("samples")),
                                scope=summary.get("scope", "opened development; reused LINCS mean fits"),
                                protected_measurements_opened=summary.get("protected_measurements_opened", summary.get("confirmation_opened")),
                                blocks=blocks)
        appendices[dataset] = dict(source=source, summary=summary,
                                  manifest_context={k:v for k,v in manifest.items() if k not in ("parts", "cells")})
    matched_path = project/"runs/jump_matched_l_20260918_v1/summary.json"
    matched = read_json(matched_path)
    matched_mc = read_json(project/"runs/jump_matched_l_20260918_v1/mc_sensitivity.json")
    main_ready = bool(matched and matched.get("complete"))
    mc_ready = bool(matched_mc and matched_mc.get("complete"))
    if matched and matched.get("complete"):
        append_matched_l(project, matched, tables, audits)
    supplements = {"JUMP_matched_L": dict(status="complete" if main_ready and mc_ready else "main_complete_MC_pending" if main_ready else "pending",
                                          main_status="complete" if main_ready else "pending", mc_status="complete" if mc_ready else "pending",
                                          source=str(matched_path.relative_to(project)), summary=matched,
                                          mc_source="runs/jump_matched_l_20260918_v1/mc_sensitivity.json", mc_sensitivity=matched_mc)}
    historical = read_json(project/"runs/jump_r2_completion_20260918_v1/historical_L_reference.json")
    supplements["JUMP_historical_L"] = dict(status="available" if historical else "not_recorded", summary=historical,
                                             main_table_eligible=False,
                                             reason="historical training access and 2000-sample evaluation differ")
    if not matched or not matched.get("complete"):
        missing.append(dict(dataset="JUMP", component="matched_L_supplement", status="pending",
                            action="refresh when supplement is complete"))
    if not mc_ready:
        missing.append(dict(dataset="JUMP", component="matched_L_MC_stability", status="pending",
                            action="refresh after both cached-model 100k seed integrations and mc_sensitivity.json complete"))
    tables["main_comparisons"] = [r for r in tables["all_arms"] if r["main_table"]]
    tables["main_paired_intervals"] = canonical_main_pairs(tables["paired_intervals"])
    tables["MC_seed_ranges"] = []
    for base in tables["all_arms"]:
        seeds = [r for r in tables["MC_seed_stability"] if all(r[k] == base[k] for k in ("dataset", "arm", "policy"))]
        if seeds:
            nulls = [base["null_selected"]] + [r["null_selected"] for r in seeds]
            values = [base["total_value"]] + [r["total_value"] for r in seeds]
            tables["MC_seed_ranges"].append(dict(dataset=base["dataset"], arm=base["arm"], policy=base["policy"],
                                                primary_and_additional_seeds=1+len(seeds),
                                                selected_null_count_min=min(nulls), selected_null_count_max=max(nulls),
                                                total_value_min=min(values), total_value_max=max(values),
                                                max_list_symmetric_difference_from_main=max(r["symmetric_difference"] for r in seeds),
                                                estimand="Monte Carlo integration stability at fixed fitted models and development candidates"))
    tables["calibrated_risk_interface"] = [r for r in tables["all_arms"]
                                           if r["arm"] in RISK_PANEL_ARMS and r["policy"] == "lambda_0.2"]
    tables["calibrated_risk_interface_paired"] = [r for r in tables["paired_intervals"]
                                                  if r["arm_a"] == RISK_PANEL_ARMS[1] and r["arm_b"] == RISK_PANEL_ARMS[0]
                                                  and r["policy"] == "lambda_0.2"
                                                  and r["metric"] in ("value_per_candidate", "false_activation_per_candidate")]
    failures = [a for a in audits if not a["passed"]]
    four_complete = all(v["status"] == "complete" for v in datasets.values())
    main_comparisons_complete = four_complete and main_ready and not any("MC" not in a["check"] for a in failures)
    complete = main_comparisons_complete and mc_ready and not failures
    result = dict(version=1, refreshed_utc=datetime.now(timezone.utc).isoformat(),
                  complete=complete, all_four_datasets_complete=four_complete,
                  main_comparisons_complete=main_comparisons_complete,
                  closure_status="complete_results_with_timing_gaps" if complete else "partial_waiting_for_dataset_or_supplement",
                  metadata_only=True, protected_measurements_opened=False,
                  datasets=datasets, fixed_main_arms=MAIN_ARMS,
                  calibrated_risk_interface_arms=RISK_PANEL_ARMS,
                  required_dataset_supplements={"JUMP":"L_TRAIN_MATCHED_FULL"},
                  fixed_main_contrasts=[a+" minus "+b for a,b in MAIN_PAIRS],
                  main_policy="lambda_0.2", main_interface="direct COHERENT Gamma distribution; joint mean/law factorial reported separately",
                  estimands=dict(false_activation_per_candidate="selected NULL / all candidates",
                                 false_positive_rate_among_NULL="selected NULL / all NULL",
                                 fdp="selected NULL / selected candidates",
                                 paired_policy_intervals="fixed fitted predictions and selected sets; paired chemical-group and layout block bootstrap",
                                 cost="one deployment cell; reference, calibration and model training shown separately; Gamma already includes two-well action cost",
                                 RxRx3="pooled condition-level estimand; chemical groups cluster the same compound across doses; cells are dose x outer fold"),
                  missing=missing, audit=dict(checks=len(audits), failed=len(failures), failures=failures),
                  supplements=supplements, files={name:name+".csv" for name in tables})
    rx_metadata_path = project/"data/rxrx3_r2_20260918/prepared_r2/metadata.json"
    rx_metadata = read_json(rx_metadata_path) or {}
    result["RxRx3_population_metadata"] = {k:rx_metadata[k] for k in
        ("approved_export_conditions", "n_conditions", "n_chemical_groups", "metadata_only_unsupported_doses",
         "minimum_dose_chemical_groups", "doses_uM", "control_wells") if k in rx_metadata}
    for name, rows in tables.items():
        write_csv(output/(name+".csv"), rows)
    write_csv(output/"missing_ledger.csv", missing)
    write_json(output/"numeric_audit.json", audits)
    write_json(output/"all_arm_appendices.json", appendices)
    write_json(output/"summary.json", result)
    lines = ["# R2 four-dataset development comparison", "", "Refreshed: " + result["refreshed_utc"], "",
             "| Dataset | Results | Candidates/conditions | Chemical groups | Deployment cells | Arms |",
             "|---|---|---:|---:|---:|---:|"]
    for dataset, entry in datasets.items():
        lines.append("| " + " | ".join([dataset,entry["status"],*[format_number(entry.get(k)) for k in
                                                 ("n","chemical_groups","deployment_cells","arm_count")]]) + " |")
    if rx_metadata:
        excluded = rx_metadata.get("approved_export_conditions", 0) - rx_metadata.get("n_conditions", 0)
        lines += ["", f"RxRx3 population metadata: {rx_metadata.get('approved_export_conditions')} approved exported conditions; "
                  f"{rx_metadata.get('n_conditions')} eligible conditions, {rx_metadata.get('n_chemical_groups')} connectivity groups. "
                  f"{excluded} singleton-dose conditions are excluded by the recorded metadata-only dose rule. "
                  "The counts above describe the frozen population; a pending result status remains pending."]
    lines += ["", "The distribution-interface table uses lambda = 0.2 and COHERENT direct interfaces. A separate calibrated risk-interface panel below carries the existing access-matched HistGB comparator across datasets. The full classifier comparison remains in the all-arm appendix. Mean comparisons hold the error-law family fixed; law comparisons hold the fitted mean fixed.", "",
              "| Dataset | Arm | Γ CRPS | NULL Brier | Selected NULL / selected | Total Γ value | NULL / all candidates | NULL-conditioned FPR |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in tables["main_comparisons"]:
        lines.append("| " + " | ".join([r["dataset"],r["arm"],format_number(r.get("crps")),
                     format_number(r["brier"]),f'{r["null_selected"]} / {r["activated"]}',
                     format_number(r["total_value"]),format_number(r["false_activation_per_candidate"]),
                     format_number(r["false_positive_rate_among_NULL"])]) + " |")
    lines += ["", "Calibrated risk-interface comparison at lambda = 0.2:", "",
              "| Dataset | CORE NULL / selected | HistGB CAL NULL / selected | CORE selected mean Γ | HistGB CAL selected mean Γ |",
              "|---|---:|---:|---:|---:|"]
    for dataset in INPUTS:
        panel = {r["arm"]:r for r in tables["calibrated_risk_interface"] if r["dataset"] == dataset}
        if all(arm in panel for arm in RISK_PANEL_ARMS):
            a, b = [panel[arm] for arm in RISK_PANEL_ARMS]
            lines.append(f"| {dataset} | {a['null_selected']} / {a['activated']} | {b['null_selected']} / {b['activated']} | {a['selected_mean_value']:.8f} | {b['selected_mean_value']:.8f} |")
        else:
            lines.append(f"| {dataset} | pending | pending | — | — |")
    lines += ["", "HistGB CAL minus CORE paired contrasts, normalized by all candidates:", "",
              "| Dataset | Resampling block | Δ value / candidate [95% CI] | Δ selected NULL / candidate [95% CI] |",
              "|---|---|---:|---:|"]
    for dataset in INPUTS:
        entries = [r for r in tables["calibrated_risk_interface_paired"] if r["dataset"] == dataset]
        for block in dict.fromkeys(r["block"] for r in entries):
            mapped = {r["metric"]:r for r in entries if r["block"] == block}
            cells = []
            for metric in ("value_per_candidate", "false_activation_per_candidate"):
                r = mapped.get(metric)
                cells.append("not recorded" if r is None else f"{r['difference']:.7f} [{r['ci95_low']:.7f}, {r['ci95_high']:.7f}]")
            lines.append("| " + " | ".join([dataset, block] + cells) + " |")
    eu_panel = {r["arm"]:r for r in tables["calibrated_risk_interface"] if r["dataset"] == "EU"}
    if all(arm in eu_panel for arm in RISK_PANEL_ARMS):
        core, direct = [eu_panel[arm] for arm in RISK_PANEL_ARMS]
        text = (f"EU's calibrated HistGB interface selects {direct['null_selected']} NULLs among {direct['activated']} candidates, "
                f"compared with CORE's {core['null_selected']} among {core['activated']}; the selected mean Γ is "
                f"{direct['selected_mean_value']:.8f} versus {core['selected_mean_value']:.8f}.")
        intervals = [r for r in tables["calibrated_risk_interface_paired"] if r["dataset"] == "EU"]
        value_intervals = [r for r in intervals if r["metric"] == "value_per_candidate"]
        false_intervals = [r for r in intervals if r["metric"] == "false_activation_per_candidate"]
        if len(value_intervals) == 2 and all(r["ci95_low"] <= 0 <= r["ci95_high"] for r in value_intervals):
            text += " The value-per-candidate interval spans zero under both chemical and layout resampling."
        if len(false_intervals) == 2 and all(r["ci95_high"] < 0 for r in false_intervals):
            text += " Both false-activation-per-candidate intervals are below zero."
        text += " These unadjusted development intervals compare this existing risk interface; they do not establish a four-dataset winner."
        lines += ["", text]
    lines += ["", "Paired intervals are saved in main_paired_intervals.csv and paired_intervals.csv. Their false-activation contrast divides by all candidates; it is not a contrast of NULL-conditioned FPR. EU/JUMP saved paired policy intervals cover lambda = 0.2; LINCS also contains lambda = 0. Other saved policies remain in all_arms.csv.", "",
              "selected_candidates.csv and selection_overlap.csv preserve selected lists, intersections and differences. selected_calibration.csv records expected versus observed NULL counts in all, CORE-selected and own-selected regions. MC_seed_stability.csv records two additional 100k-draw seeds; fixed_random_controls.csv preserves each random-control interval at its original aggregation level.", "",
              "Deployment costs are per cell. Reference-only debits are the common primary accounting convention; separate columns show CAL and REF+CAL setup. Training wells are recorded separately from setup. Overlapping cross-validation cells are not summed as one deployment purchase. Historical and current compute records retain their JSON source and field; inclusive run/fold totals are not added to nested components or split among models.", "",
              "Direct CLASSIFIER_RAW/CAL rows share the regression-residual Γ distribution for CRPS and interval scoring; those scores are not a classifier-distribution score. Raw classifier policy deployment and its retained Γ-distribution evaluation have distinct CAL requirements.", "",
              "Historical JUMP L_GRAM is an appendix reference with larger training access and 2000 Monte Carlo draws. The matched-L supplement is " + supplements["JUMP_matched_L"]["status"] + ".", "",
              f"Numeric audit: {len(audits)} checks, {len(failures)} mismatches. Missing records are listed in missing_ledger.csv. These are repeated development comparisons with fixed fitted predictions, not independent policy certification.", ""]
    (output/"REPORT.md").write_text("\n".join(lines))
    compute_lines = ["# Recorded compute ledger", "",
                     "Each row in compute_costs.csv points to a recorded JSON duration. Inclusive run totals, cumulative fold times, component bundles, and nested training times overlap and have no combined grand total.", "",
                     "| Dataset | Phase | Mean component | Recorded training files | Sum of recorded seconds |",
                     "|---|---|---|---:|---:|"]
    for dataset in INPUTS:
        for phase in ("historical_reused_or_reference", "current_R2"):
            for component in ("HR_fit", "A_OLD_GENERIC", "STATE50"):
                entries = [r for r in tables["compute_costs"] if r["dataset"] == dataset and r["phase"] == phase
                           and r["source"].endswith("/"+component+"/training_complete.json")
                           and r["json_pointer"] == "/elapsed_seconds"]
                if entries:
                    compute_lines.append(f"| {dataset} | {phase} | {component} | {len(entries)} | {sum(r['wall_seconds'] for r in entries):.6f} |")
    compute_lines += ["", "Historical JUMP L_GRAM has only mixed-arm fold/run durations; those totals remain unallocated. JUMP current full_mean_fit timers include nested HR/A/STATE timing and must not be added to the table above. Direct scope timers can bundle RIDGE, ExtraTrees and HistGB; a per-family allocation is unavailable unless its own metadata records it.", "",
                      "| Dataset | Missing component | Status |", "|---|---|---|"]
    for r in missing:
        if "compute" in r["component"]:
            compute_lines.append(f"| {r['dataset']} | {r['component']} | {r['status']} |")
    (output/"COMPUTE_LEDGER.md").write_text("\n".join(compute_lines)+"\n")
    write_json(output/"claim_evidence_matrix.json", [
        dict(claim="All four R2 datasets complete", status="supported" if four_complete else "unsupported",
             evidence={k:v["status"] for k,v in datasets.items()}),
        dict(claim="Shared COHERENT main interface and fixed main-arm roster", status="supported", evidence="main_comparisons.csv; fixed_main_arms"),
        dict(claim="Paired false-activation CI is a NULL-conditioned FPR interval", status="unsupported", evidence="paired_intervals.csv denominator=all candidates"),
        dict(claim="Complete per-model compute accounting", status="unsupported", evidence="missing_ledger.csv and compute_costs.csv"),
        dict(claim="Numerical consistency of cached aggregates", status="supported" if not failures else "partially_supported", evidence="numeric_audit.json"),
    ])
    print(json.dumps(dict(output=str(output), datasets={k:v["status"] for k,v in datasets.items()},
                          complete=complete, audit_failures=len(failures), missing_records=len(missing))))


if __name__ == "__main__":
    main()
