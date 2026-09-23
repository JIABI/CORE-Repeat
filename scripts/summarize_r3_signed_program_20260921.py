"""Paired chemical-group summaries for the three-part R3 extension package.

Only saved query scores are read. No model, strength, support rule or reference
selection is fitted here. All doses/aliases are resampled together.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.biology_kernel_evaluation import write_json

OLD = PROJECT / "runs/r3_crossdose_response_20260920_v1"
ROOT = PROJECT / "runs/r3_signed_program_dose_20260921_v1"
REPORT = PROJECT / "reports/r3_signed_program_dose_20260921_v1"
METRICS = ("profile_mse", "cosine_loss", "lognorm_squared_error")
TASKS = ("SAME", "CROSS")
SEED = 2026092117
BOOTSTRAPS = 2000


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def old_population():
    population = {}
    for path in sorted(OLD.glob("unit_*/query_scores.npz")):
        with np.load(path, allow_pickle=False) as z:
            keys = ("groups", "object_ids", "source_dose", "target_dose", "batch",
                    "source_plate", "layout", "target_support", "morph_support")
            meta = {k: z[k] for k in keys}
            baseline = z["scores"][:, :, list(z["arm_names"]).index("RIDGE_RESPONSE")]
            for j, row in enumerate(z["pair_rows"]):
                if int(row) in population:
                    raise ValueError("Old query rows are repeated")
                population[int(row)] = {k: value[j] for k, value in meta.items()}
                population[int(row)]["baseline_scores"] = baseline[:, j]
    if len(population) != 8935:
        raise ValueError("Expected the full prior query population")
    return population


def compact_random(scores, names):
    """Average random-draw scores, never predictions or biological samples."""
    pools = {}
    for i, name in enumerate(names):
        key = re.sub(r"_R\d{2}(?=_|$)", "_RANDOM_MEAN", str(name))
        pools.setdefault(key, []).append(i)
    keys = list(pools)
    values, counts = [], {}
    for key, indices in pools.items():
        draws = [i for i in indices if re.search(r"_R\d{2}(?=_|$)", str(names[i]))]
        explicit = [i for i in indices if str(names[i]) == key]
        if draws:
            value = scores[:, :, draws].mean(axis=2)
            if explicit:
                np.testing.assert_allclose(scores[:, :, explicit[0]], value, rtol=1e-10, atol=1e-12)
            counts[key] = len(draws)
        else:
            if len(indices) != 1:
                raise ValueError("Duplicate non-random arm")
            value = scores[:, :, indices[0]]
            counts[key] = 1
        values.append(value)
    compact = np.stack(values, axis=2)
    return compact, keys, counts


def load_package(package, expected_units):
    paths = sorted((ROOT / package).glob("**/query_scores.npz"))
    if len(paths) != expected_units:
        raise ValueError(f"{package}: {len(paths)}/{expected_units} score files")
    fields = []
    names = None
    all_scores, all_rows = [], []
    for path in paths:
        z = read_npz(path)
        if list(z["metric_names"]) != list(METRICS) or list(z["task_names"]) != list(TASKS):
            raise ValueError(f"Task or metric order changed: {path}")
        scores, keys, counts = compact_random(z["scores"], z["arm_names"])
        if names is not None and keys != names:
            raise ValueError("Arm list changed across units")
        names = keys
        if scores.shape != (2, len(z["pair_rows"]), len(keys), 3):
            raise ValueError("Score shape mismatch")
        all_scores.append(scores)
        all_rows.append(z["pair_rows"])
        fields.append(dict(path=str(path), n_queries=len(z["pair_rows"]), random_counts=counts))
    rows = np.concatenate(all_rows)
    order = np.argsort(rows)
    rows = rows[order]
    if not np.array_equal(rows, np.arange(8935)):
        raise ValueError("Package must contain each qualified query exactly once")
    scores = np.concatenate(all_scores, axis=1)[:, order]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    return scores, names, rows, fields


def group_means(scores, groups, mask):
    labels, inverse, counts = np.unique(groups[mask], return_inverse=True, return_counts=True)
    rows = np.moveaxis(scores[:, mask], 1, 0)
    values = np.zeros((len(labels),) + rows.shape[1:])
    np.add.at(values, inverse, rows)
    values /= counts[:, None, None, None]
    return labels, values, counts


def paired_bootstrap(values, seed, n=BOOTSTRAPS):
    g = len(values)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(g, np.full(g, 1 / g), size=n)
    boot = counts @ values.reshape(g, -1) / g
    return boot.reshape((n,) + values.shape[1:])


def proposed_comparisons(names):
    """Complete labelled contrasts; no selection by score or significance."""
    pairs = []
    if "RIDGE_RESPONSE" not in names:
        raise ValueError("The original per-dose response baseline is required")
    for name in names:
        if name == "RIDGE_RESPONSE" or "RANDOM" in name:
            continue
        pairs.append((name, "RIDGE_RESPONSE"))
        candidates = []
        if name.startswith("TARGET_"):
            candidates += [name.replace("TARGET_", "TARGET_RANDOM_MEAN_", 1),
                           name.replace("TARGET_", "MORPH_", 1)]
        if name.startswith("MORPH_"):
            candidates.append(name.replace("MORPH_", "MORPH_RANDOM_MEAN_", 1))
        if "_DIRECTION_" in name:
            candidates.append(name.replace("_DIRECTION_", "_FREE_"))
        if "_SIGNED_" in name:
            candidates += [name.replace("_SIGNED_", "_POSCOS_"),
                           name.replace("_SIGNED_", "_")]
        for candidate in candidates:
            if candidate in names:
                pairs.append((name, candidate))
    # Additional precise new names can be supplied in comparisons.json by the
    # corresponding implementation before the summary is generated.
    return list(dict.fromkeys(pairs))


def contrast_record(mean, boot, ia, ib):
    result = {}
    for t, task in enumerate(TASKS):
        result[task] = {}
        for m, metric in enumerate(METRICS):
            difference = mean[t, ia, m] - mean[t, ib, m]
            delta = boot[:, t, ia, m] - boot[:, t, ib, m]
            baseline = mean[t, ib, m]
            bdraw = boot[:, t, ib, m]
            item = dict(risk_difference=float(difference),
                        difference_ci95=np.quantile(delta, [.025, .975]).tolist())
            if baseline > 0 and np.all(bdraw > 0):
                relative = 100 * (1 - mean[t, ia, m] / baseline)
                relative_boot = 100 * (1 - boot[:, t, ia, m] / bdraw)
                item.update(improvement_percent=float(relative),
                            improvement_percent_ci95=np.quantile(relative_boot, [.025, .975]).tolist())
            result[task][metric] = item
    # A significant result in CROSS and a non-significant result in SAME do
    # not establish a task interaction. Preserve the paired contrast directly.
    interaction = {}
    for m, metric in enumerate(METRICS):
        d = mean[:, ia, m] - mean[:, ib, m]
        db = boot[:, :, ia, m] - boot[:, :, ib, m]
        item = dict(risk_difference_interaction=float(d[1] - d[0]),
                    difference_ci95=np.quantile(db[:, 1] - db[:, 0], [.025, .975]).tolist())
        denom = mean[:, ib, m]
        boot_denom = boot[:, :, ib, m]
        if np.all(denom > 0) and np.all(boot_denom > 0):
            gain = 100 * (1 - mean[:, ia, m] / denom)
            gain_boot = 100 * (1 - boot[:, :, ia, m] / boot_denom)
            item.update(improvement_percentage_point_interaction=float(gain[1] - gain[0]),
                        improvement_pp_ci95=np.quantile(gain_boot[:, 1] - gain_boot[:, 0], [.025, .975]).tolist())
        interaction[metric] = item
    result["CROSS_minus_SAME"] = interaction
    return result


def simultaneous_mse_intervals(mean, boot, pairs, names):
    """Centered bootstrap max-t intervals for the displayed contrast family.

    This is a conditional-on-fit, asymptotic multiplicity sensitivity analysis,
    not a finite-sample selective guarantee. Deterministic ties are identified.
    """
    centers, draws, keys = [], [], []
    for a, b in pairs:
        ia, ib = names.index(a), names.index(b)
        for t, task in enumerate(TASKS):
            centers.append(mean[t, ia, 0] - mean[t, ib, 0])
            draws.append(boot[:, t, ia, 0] - boot[:, t, ib, 0])
            keys.append((a + " minus " + b, task))
    if not keys:
        return {}
    centers, draws = np.asarray(centers), np.stack(draws, axis=1)
    se = draws.std(0, ddof=1)
    active = se > 1e-15
    maximum = np.max(np.abs((draws[:, active] - centers[active]) / se[active]), axis=1) if active.any() else np.zeros(len(boot))
    cutoff = float(np.quantile(maximum, .95))
    result = dict(method="centered chemical-group bootstrap max-t",
                  family_size=len(keys), active_contrasts=int(active.sum()), cutoff=cutoff,
                  intervals={})
    for j, (label, task) in enumerate(keys):
        result["intervals"].setdefault(label, {})[task] = dict(
            risk_difference=float(centers[j]),
            simultaneous_ci95=[float(centers[j] - cutoff * se[j]), float(centers[j] + cutoff * se[j])],
            bootstrap_se=float(se[j]), deterministic_tie=bool(not active[j]))
    return result


def panel(scores, meta, mask, names, pairs, *, seed, simultaneous=True):
    labels, values, counts = group_means(scores, meta["groups"], mask)
    if len(labels) < 2:
        return dict(n_rows=int(mask.sum()), n_groups=len(labels), unavailable=True)
    mean = values.mean(0)
    boot = paired_bootstrap(values, seed)
    risks = {name: {task: {metric: float(mean[t, a, m]) for m, metric in enumerate(METRICS)}
                    for t, task in enumerate(TASKS)} for a, name in enumerate(names)}
    result = dict(n_rows=int(mask.sum()), n_groups=len(labels),
                  rows_per_group=[int(counts.min()), float(np.median(counts)), int(counts.max())],
                  risks=risks,
                  comparisons={a + " minus " + b: contrast_record(mean, boot, names.index(a), names.index(b)) for a, b in pairs})
    if simultaneous:
        # Calibrated real-arm MSE contrasts form the inferential family. Fixed
        # strength and local dose results are mechanistic/exploratory readouts.
        primary = [(a, b) for a, b in pairs if a.endswith("CAL") or
                   a in ("GENERIC", "JOINTDOSE_RAW", "SHARED_SOURCE_RAW")]
        result["multiplicity_sensitivity"] = simultaneous_mse_intervals(mean, boot, primary, names)
    return result


def simple_subgroups(scores, meta, names, field, mask):
    result = {}
    for key in np.unique(meta[field]):
        take = mask & (meta[field] == key)
        labels, values, _ = group_means(scores, meta["groups"], take)
        if not len(labels):
            continue
        mean = values.mean(0)
        result[str(key)] = dict(n_rows=int(take.sum()), n_groups=len(labels),
            risks={name: {task: {metric: float(mean[t, a, m]) for m, metric in enumerate(METRICS)}
                          for t, task in enumerate(TASKS)} for a, name in enumerate(names)})
    return result


def calibration_statistics(package):
    records = {}
    complete = []
    for path in sorted((ROOT / package).glob("**/calibration.json")):
        content = json.loads(path.read_text())
        def visit(value, key):
            if not isinstance(value, dict):
                return
            if "alpha" in value:
                a = np.asarray(value["alpha"], dtype=float).reshape(-1)
                records.setdefault(key, []).append(a)
                return
            for name, child in value.items():
                if not re.search(r"_R\d{2}(?=_|$)", name):
                    visit(child, key + "/" + name if key else name)
        visit(content, "")
        cp = path.parent / "complete.json"
        if cp.exists():
            complete.append(json.loads(cp.read_text()))
    result = {}
    for label, values in records.items():
        all_values = np.concatenate(values)
        nonzero = all_values[all_values > 0]
        result[label] = dict(n_units=len(values), nonzero_units=sum(np.any(a > 0) for a in values),
            n_coefficients=len(all_values), nonzero_coefficients=int(len(nonzero)),
            mean_alpha=float(all_values.mean()),
            nonzero_alpha_range=[float(nonzero.min()), float(nonzero.max())] if len(nonzero) else None)
    return dict(strengths=result, complete_units=len(complete),
                total_unit_elapsed_seconds=float(sum(c.get("elapsed_seconds", 0) for c in complete)),
                interpretation="Runtime is the complete package, including its comparisons and scoring, not the cost of one method.")


def markdown_table(package, result):
    lines = [f"## {package}", ""]
    for scope in ("all", "original_target_supported"):
        p = result["panels"][scope]
        lines += [f"### {scope}: {p['n_rows']} condition pairs, {p['n_groups']} chemical groups", "",
                  "| Arm | SAME MSE | CROSS MSE | CROSS improvement vs per-dose Ridge, % [95% CI] |",
                  "|---|---:|---:|---:|"]
        for name in result["arm_names"]:
            if "RANDOM" in name:
                continue
            r = p["risks"][name]
            label = name + " minus RIDGE_RESPONSE"
            entry = p["comparisons"].get(label, {}).get("CROSS", {}).get("profile_mse", {})
            if "improvement_percent" in entry:
                lo, hi = entry["improvement_percent_ci95"]
                improvement = f"{entry['improvement_percent']:.4f} [{lo:.4f}, {hi:.4f}]"
            else:
                improvement = "reference"
            lines.append(f"| {name} | {r['SAME']['profile_mse']:.6f} | {r['CROSS']['profile_mse']:.6f} | {improvement} |")
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", nargs="+", default=["borrowing", "jointdose"])
    args = parser.parse_args()
    population = old_population()
    meta = {k: np.asarray([population[i][k] for i in range(8935)])
            for k in population[0] if k != "baseline_scores"}
    baseline = np.stack([population[i]["baseline_scores"] for i in range(8935)], axis=1)
    report = dict(state="COMPLETE", dataset="RxRx3 approved development", n_pairs=8935,
                  n_chemical_groups=len(np.unique(meta["groups"])),
                  metrics=list(METRICS), bootstrap_replicates=BOOTSTRAPS,
                  uncertainty="Paired chemical groups, all doses/aliases together, conditional on fitted models; not independent confirmation",
                  packages={})
    lines = ["# R3 signed, directional and joint-dose response comparisons", "",
             "Lower scores are better. Relative improvements are positive when better. Pointwise intervals are chemical-group paired bootstrap intervals; JSON also includes max-t multiplicity sensitivity for calibrated comparisons. All results are development results.", ""]
    with threadpool_limits(limits=2):
        for package in args.packages:
            scores, names, rows, files = load_package(package, 35 if package == "borrowing" else 5)
            deviation = float(np.max(np.abs(scores[:, :, names.index("RIDGE_RESPONSE")] - baseline)))
            if deviation > 1e-5:
                raise ValueError(f"Saved baseline mismatch: {package} {deviation}")
            pairs = proposed_comparisons(names)
            extra_path = ROOT / package / "comparisons.json"
            if extra_path.exists():
                extra = json.loads(extra_path.read_text())
                for a, b in extra:
                    if a not in names or b not in names:
                        raise ValueError(f"Unknown explicit contrast {a}, {b}")
                    pairs.append((a, b))
            pairs = list(dict.fromkeys(pairs))
            masks = {"all": np.ones(len(rows), bool),
                     "original_target_supported": meta["target_support"].astype(bool)}
            panels = {scope: panel(scores, meta, mask, names, pairs, seed=SEED+i)
                      for i, (scope, mask) in enumerate(masks.items())}
            per_dose = {}
            for dose in np.unique(meta["source_dose"]):
                per_dose[str(dose)] = {scope: panel(scores, meta, mask & (meta["source_dose"] == dose),
                                                    names, pairs, seed=SEED+100, simultaneous=False)
                                       for scope, mask in masks.items()}
            per_batch = {scope: simple_subgroups(scores, meta, names, "batch", mask)
                         for scope, mask in masks.items()}
            result = dict(arm_names=names, score_files=files, baseline_max_abs_difference=deviation,
                          panels=panels, per_source_dose=per_dose, per_batch=per_batch,
                          calibration=calibration_statistics(package))
            report["packages"][package] = result
            lines.extend(markdown_table(package, result))
            np.savez_compressed(REPORT / f"{package}_aggregate_scores.npz", scores=scores,
                                arm_names=np.asarray(names), pair_rows=rows, **meta)
    write_json(REPORT / "summary.json", report)
    (REPORT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(state="COMPLETE", packages=args.packages, output=str(REPORT)), indent=2))


if __name__ == "__main__":
    main()
