"""Complete, paired reporting for the one-arm frozen scatter intervention."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .m4_dependence_ablation import (LEVELS, OBSERVABLES, read_json, read_npz,
                                    save_npz, write_json)

ARMS = ("CORE_ORIGINAL", "DIAGONAL_SCATTER")
FAMILIES = dict(single=slice(0, 3), difference=slice(3, 6),
                two_average=slice(6, 9), triple_average=slice(9, 10))


def metric_vectors(out):
    """Numerator and denominator contributions for identical paired resamples."""
    n = len(out["actual"])
    ones = np.ones(n)
    null = (out["actual"] <= 0).astype(float)
    numerators, denominators = {}, {}

    def add(name, vector, denominator=None):
        numerators[name] = np.asarray(vector, float)
        denominators[name] = ones if denominator is None else np.asarray(denominator, float)

    for metric in ("nll", "energy", "crps", "brier", "coordinate_coverage", "joint_coverage",
                   "single_crps", "pair_crps", "two_average_crps", "triple_average_crps",
                   "absolute_pair_crps", "gamma_mc_se", "null_mc_se"):
        add(metric, out[metric])
    add("gain_mse", (out["predicted"] - out["actual"]) ** 2)
    add("geometry_mean_mse", (out["mean_u"] - out["actual_u"]) ** 2 @ (np.ones(9) / 9))
    for prefix in ("coordinate", "gamma", "joint"):
        for j, level in enumerate(LEVELS):
            add(f"{prefix}_coverage_{level:g}", out[f"{prefix}_coverage_by_level"][:, j])
            if prefix != "joint":
                add(f"{prefix}_width_{level:g}", out[f"{prefix}_width_by_level"][:, j])
    for j, name in enumerate(OBSERVABLES):
        add(f"observable_{name}_crps", out["observable_crps"][:, j])
        for k, level in enumerate(LEVELS):
            add(f"observable_{name}_coverage_{level:g}", out["observable_coverage_by_level"][:, j, k])
            add(f"observable_{name}_width_{level:g}", out["observable_width_by_level"][:, j, k])
    for name, columns in FAMILIES.items():
        for k, level in enumerate(LEVELS):
            add(f"{name}_observable_coverage_{level:g}", out["observable_coverage_by_level"][:, columns, k].mean(1))
            add(f"{name}_observable_width_{level:g}", out["observable_width_by_level"][:, columns, k].mean(1))
    for lam in (0.2, 0):
        chosen = out[f"selected_lambda_{lam}"].astype(float)
        prefix = f"lambda_{lam}"
        add(prefix + "_value_per_candidate", chosen * out["actual"])
        add(prefix + "_null_per_candidate", chosen * null)
        add(prefix + "_selected_mean_gamma", chosen * out["actual"], chosen)
        add(prefix + "_selected_null_rate", chosen * null, chosen)
        add(prefix + "_selected_predicted_null_rate", chosen * out["p_null"], chosen)
        add(prefix + "_selected_calibration_bias", chosen * (out["p_null"] - null), chosen)
        add(prefix + "_selected_brier", chosen * out["brier"], chosen)
    return numerators, denominators


def paired_bootstrap(full, diagonal, labels, seed=20260922, replicates=2000):
    groups, index = np.unique(labels, return_inverse=True)
    full_num, full_den = metric_vectors(full)
    diag_num, diag_den = metric_vectors(diagonal)
    names = list(full_num)
    points, resamples = [], []
    rng = np.random.default_rng(seed)
    counts = np.asarray([np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
                         for _ in range(replicates)], float)
    for nums, dens in ((full_num, full_den), (diag_num, diag_den)):
        numerator, denominator = np.column_stack([nums[x] for x in names]), np.column_stack([dens[x] for x in names])
        group_num = np.zeros((len(groups), len(names)))
        group_den = np.zeros_like(group_num)
        np.add.at(group_num, index, numerator)
        np.add.at(group_den, index, denominator)
        bden = counts @ group_den
        if np.any(bden <= 0):
            raise ValueError("A fixed-policy resample has no selected observations; report cannot silently discard it")
        points.append(numerator.sum(0) / denominator.sum(0))
        resamples.append((counts @ group_num) / bden)
    difference = resamples[1] - resamples[0]
    intervals = [np.quantile(s, [.025, .975], axis=0) for s in resamples]
    delta_ci = np.quantile(difference, [.025, .975], axis=0)
    return [dict(metric=name, n_blocks=len(groups), replicates=replicates,
        core=points[0][j], diagonal=points[1][j], delta_diagonal_minus_core=points[1][j]-points[0][j],
        core_low=intervals[0][0, j], core_high=intervals[0][1, j],
        diagonal_low=intervals[1][0, j], diagonal_high=intervals[1][1, j],
        delta_low=delta_ci[0, j], delta_high=delta_ci[1, j]) for j, name in enumerate(names)]


def summary_row(dataset, arm, out):
    null = out["actual"] <= 0
    row = dict(dataset=dataset, arm=arm, n=len(null), chemical_groups=len(np.unique(out["groups"])),
               layouts=len(np.unique(out["layout"])),
               null_auc=float(roc_auc_score(null, out["p_null"])),
               gamma_spearman=float(spearmanr(out["actual"], out["predicted"]).statistic))
    nums, dens = metric_vectors(out)
    row.update({key: float(np.sum(nums[key])/np.sum(dens[key])) for key in nums
                if not key.startswith("observable_")})
    for lam in (0.2, 0):
        chosen = out[f"selected_lambda_{lam}"].astype(bool)
        prefix = f"lambda_{lam}"
        row.update({prefix+"_selected_n": int(chosen.sum()),
                    prefix+"_total_gamma": float(out["actual"][chosen].sum()),
                    prefix+"_actual_null_n": int(null[chosen].sum()),
                    prefix+"_predicted_null_n": float(out["p_null"][chosen].sum()),
                    prefix+"_action_wells": int(2*chosen.sum())})
    return row


def summarize(project, output, datasets):
    output = Path(output)
    summaries, paired, units, observables, costs, overlaps = [], [], [], [], [], []
    for dataset in datasets:
        folders = sorted((output / "cells" / dataset).glob("cell_*"))
        outcomes = {arm: [] for arm in ARMS}
        for folder in folders:
            completed = read_json(folder / "complete.json")
            record = completed["record"]
            for arm in ARMS:
                data = read_npz(folder / f"{arm}.npz")
                outcomes[arm].append(data)
                units.append(dict(cell=record["cell_index"], label=record["label"],
                                  **summary_row(dataset, arm, data)))
                costs.append(dict(dataset=dataset, cell=record["cell_index"], arm=arm,
                    new_fit_seconds=0, new_scoring_seconds=(completed["new_diagonal_score_seconds"] if arm == "DIAGONAL_SCATTER" else 0),
                    previous_fit_cost_reused=True, paired_setup_cost_difference=0,
                    paired_action_cost_difference=0, **record["costs"]))
        aggregate = {}
        for arm, cells in outcomes.items():
            keys = set.intersection(*(set(x) for x in cells))
            all_data = {key: np.concatenate([x[key] for x in cells], axis=0) for key in keys}
            if len(set(all_data["ids"])) != len(all_data["ids"]):
                raise ValueError("Duplicate development identities in aggregated output")
            aggregate[arm] = all_data
            save_npz(output / f"{dataset}_{arm}.npz", all_data)
            summaries.append(summary_row(dataset, arm, all_data))
            for j, name in enumerate(OBSERVABLES):
                for k, level in enumerate(LEVELS):
                    observables.append(dict(dataset=dataset, arm=arm, observable=name, nominal=level,
                        n=len(all_data["ids"]), crps=all_data["observable_crps"][:, j].mean(),
                        coverage=all_data["observable_coverage_by_level"][:, j, k].mean(),
                        width=all_data["observable_width_by_level"][:, j, k].mean()))
        full, diag = aggregate[ARMS[0]], aggregate[ARMS[1]]
        for key in ("ids", "actual", "groups", "layout", "mean_u", "actual_u"):
            np.testing.assert_array_equal(full[key], diag[key])
        for unit in ("groups", "layout"):
            for item in paired_bootstrap(full, diag, full[unit]):
                paired.append(dict(dataset=dataset, dependence_unit=("chemical_group" if unit=="groups" else "layout"), **item))
        for lam in (0.2, 0):
            a, b = full[f"selected_lambda_{lam}"].astype(bool), diag[f"selected_lambda_{lam}"].astype(bool)
            overlaps.append(dict(dataset=dataset, lam=lam, core_n=int(a.sum()), diagonal_n=int(b.sum()),
                intersection_n=int((a & b).sum()), replaced_n=int((a & ~b).sum()),
                intersection_over_core=float((a & b).sum()/a.sum()), jaccard=float((a & b).sum()/(a | b).sum())))
    for filename, rows in (("summary.csv", summaries), ("paired_intervals.csv", paired),
                            ("deployment_units.csv", units), ("observables.csv", observables),
                            ("costs.csv", costs), ("selection_overlap.csv", overlaps)):
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    write_json(output / "summary_metadata.json", dict(datasets=list(datasets),
        rows=sum(row["n"] for row in summaries if row["arm"]=="CORE_ORIGINAL"),
        intervention="diagonal-scatter with original shared empirical radius", samples=100000,
        paired_sign="diagonal minus full CORE", bootstrap_replicates=2000,
        bootstrap_selection="Saved cell-level lists remain fixed; no retraining or reselection",
        observations=list(OBSERVABLES), two_average_columns=[6,7,8], triple_average_column=9,
        references_and_calibration="Same frozen state and setup cost for both arms; historical fit cost not remeasured",
        R4_accessed=False))
