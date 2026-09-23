"""Matched, analysis-only comparison of the three full DEV objective arms.

This module consumes completed evaluation artifacts. It neither loads model
weights nor reads measurement pools, fits models, or changes a decision rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score, roc_auc_score


ARMS = ("A_ELBO", "B_NLL", "C_NLL_CRPS")
ACTIONS = ("stop", "add_0", "add_1", "add_0_1")
WELLS = np.array([0, 1, 1, 2])
FRACTIONS = (.05, .10, .25)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fair_crps(samples, actual):
    """Unbiased ensemble CRPS per case; sample axis 0, no quadratic allocation.

    The pair term excludes self-pairs and uses S(S-1), rather than the biased
    finite-ensemble S**2 score. Draws are independent across Monte Carlo index;
    correlations among wells/cases within a draw are retained upstream.
    """
    samples, actual = np.asarray(samples, float), np.asarray(actual, float)
    if samples.ndim < 2 or samples.shape[1:] != actual.shape or len(samples) < 2:
        raise ValueError("CRPS requires [S>=2, ...cases] and matching actual values")
    if not np.isfinite(samples).all() or not np.isfinite(actual).all():
        raise ValueError("CRPS inputs must be finite")
    count = len(samples)
    coefficients = (2 * np.arange(1, count + 1) - count - 1).reshape(
        (count,) + (1,) * actual.ndim)
    pair_half = (np.sort(samples, axis=0) * coefficients).sum(0) / (count * (count - 1))
    return np.abs(samples - actual).mean(0) - pair_half


def _metrics(actual, mean, p_null, p_positive, positive_margin=.005):
    null, positive = actual <= 0, actual >= positive_margin
    varied = len(actual) >= 2 and np.std(actual) > 0 and np.std(mean) > 0
    return dict(n=len(actual), actual_mean=actual.mean(), predicted_mean=mean.mean(),
                actual_sd=actual.std(), predicted_mean_sd=mean.std(),
                null_rate=null.mean(), positive_rate=positive.mean(),
                pearson_r=float(pearsonr(actual, mean).statistic) if varied else None,
                spearman_r=float(spearmanr(actual, mean).statistic) if varied else None,
                mse=mean_squared_error(actual, mean),
                r2=r2_score(actual, mean) if len(actual) >= 2 and np.std(actual) > 0 else None,
                null_auc=roc_auc_score(null, p_null) if len(np.unique(null)) == 2 else None,
                positive_auc=roc_auc_score(positive, p_positive) if len(np.unique(positive)) == 2 else None,
                null_brier=brier_score_loss(null, p_null),
                positive_brier=brier_score_loss(positive, p_positive))


def _close(left, right, message):
    if np.shape(left) != np.shape(right) or not np.allclose(left, right, atol=1e-11, rtol=1e-9):
        raise ValueError(message)


def _read_arm(root, arm):
    folder = root / "arms" / arm
    required = ["evaluation.json", "evaluation_predictions.tsv", "utility_draws.npz", "allocations.tsv"]
    missing = [str(folder / name) for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError("All three completed arms are required; missing: " + ", ".join(missing))
    summary = json.loads((folder / "evaluation.json").read_text())
    frame = pd.read_csv(folder / "evaluation_predictions.tsv", sep="\t", dtype={"compound_id": str})
    if "compound_id" not in frame or frame.compound_id.isna().any() or frame.compound_id.duplicated().any():
        raise ValueError(f"{arm}: evaluation IDs must be unique and present")
    ids = frame.compound_id.to_numpy(dtype=str)
    with np.load(folder / "utility_draws.npz", allow_pickle=False) as stored:
        samples = np.asarray(stored["samples"], dtype=float)
        draw_ids = np.asarray(stored["ids"], dtype=str)
        calibration_ids = np.asarray(stored["calibration_ids"], dtype=str)
        if "actions" in stored and tuple(stored["actions"].tolist()) != ACTIONS:
            raise ValueError(f"{arm}: utility draw action order differs")
    if not len(ids) or not np.array_equal(ids, draw_ids):
        raise ValueError(f"{arm}: TSV and draw compound order differ")
    if samples.ndim != 3 or samples.shape[1:] != (len(ids), len(ACTIONS)) or len(samples) < 2:
        raise ValueError(f"{arm}: expected utility draws [S>=2,N,4]")
    if not np.isfinite(samples).all():
        raise ValueError(f"{arm}: nonfinite utility draws")
    if summary.get("final_opened") is not False:
        raise ValueError(f"{arm}: this comparison requires FINAL to remain unopened")
    if summary.get("evaluation_compounds") != len(ids):
        raise ValueError(f"{arm}: evaluation count disagrees with artifacts")
    if summary.get("positive_margin") != .005 or summary.get("cost_per_optional_well") != .01:
        raise ValueError(f"{arm}: original utility label/cost definitions differ")
    if summary.get("utility") != "original fixed-space half-cosine gain":
        raise ValueError(f"{arm}: original fixed-space endpoint required")
    if set(summary.get("gain_metrics", {})) != set(ACTIONS[1:]):
        raise ValueError(f"{arm}: action metrics differ from the four-role action set")
    actual = np.column_stack([frame[action + "__actual"].to_numpy(float) for action in ACTIONS])
    if not np.isfinite(actual).all():
        raise ValueError(f"{arm}: actual endpoints must be finite")
    _close(samples[..., 0], np.zeros(samples.shape[:2]), f"{arm}: STOP draws must be zero")
    _close(actual[:, 0], np.zeros(len(ids)), f"{arm}: STOP actual endpoints must be zero")
    for index, action in enumerate(ACTIONS):
        # Older draw files omit names. Matching every saved statistic verifies
        # the documented action axis against the named prediction columns.
        _close(samples[:, :, index].mean(0), frame[action + "__predicted"],
               f"{arm}: draw action order/prediction mean mismatch for {action}")
        _close((samples[:, :, index] <= 0).mean(0), frame[action + "__p_null"],
               f"{arm}: draw action order/NULL probability mismatch for {action}")
        _close((samples[:, :, index] >= .005).mean(0), frame[action + "__p_positive"],
               f"{arm}: draw action order/POSITIVE probability mismatch for {action}")
    training_counts = None
    log = folder / "training.jsonl"
    if log.is_file():
        starts = [line for line in map(json.loads, log.read_text().splitlines())
                  if line.get("event") == "world_model_start"]
        if starts:
            training_counts = {key: starts[-1].get(key) for key in
                               ("train_compounds", "validation_compounds", "parameter_count", "trainable_parameters")}
    return dict(arm=arm, summary=summary, frame=frame, ids=ids, samples=samples,
                actual=actual, calibration_ids=calibration_ids, training_counts=training_counts,
                allocations=pd.read_csv(folder / "allocations.tsv", sep="\t"))


def _validate_matching(arms):
    first = arms[0]
    for arm in arms:
        config = arm["summary"]["model"]
        expected = "elbo" if arm["arm"] == "A_ELBO" else "predictive_nll"
        if config.get("objective") != expected:
            raise ValueError(f"{arm['arm']}: objective differs from the declared treatment")
        weight = config.get("utility_crps_weight")
        if weight is None or not np.isfinite(weight) or (weight <= 0 if arm["arm"] == "C_NLL_CRPS" else weight != 0):
            raise ValueError(f"{arm['arm']}: utility CRPS weight differs from the declared treatment")
    for current in arms[1:]:
        if not np.array_equal(first["ids"], current["ids"]):
            raise ValueError("Arm evaluation IDs/order differ; do not silently align or drop cases")
        if not np.array_equal(first["calibration_ids"], current["calibration_ids"]):
            raise ValueError("Arm calibration IDs/order differ")
        if first["samples"].shape != current["samples"].shape:
            raise ValueError("Arms must use identical Monte Carlo counts and action dimensions")
        _close(first["actual"], current["actual"], "Arm actual endpoints differ")
        for key in ("full_feature_dimension", "evaluation_compounds", "calibration_compounds"):
            if first["summary"].get(key) != current["summary"].get(key):
                raise ValueError(f"Arm {key} differs")
        model_keys = set(first["summary"]["model"]) | set(current["summary"]["model"])
        for key in model_keys - {"objective", "utility_crps_weight"}:
            if first["summary"]["model"].get(key) != current["summary"]["model"].get(key):
                raise ValueError(f"Arm model setting {key} differs outside the objective treatment")
        if first["training_counts"] != current["training_counts"]:
            raise ValueError("Arm training counts/architecture sizes differ")


def _action_rows(arm):
    rows = []
    crps = fair_crps(arm["samples"], arm["actual"])
    low, high = np.quantile(arm["samples"], [.05, .95], axis=0)
    for index, action in enumerate(ACTIONS[1:], 1):
        frame = arm["frame"]
        metrics = _metrics(arm["actual"][:, index], frame[action + "__predicted"].to_numpy(),
                           frame[action + "__p_null"].to_numpy(), frame[action + "__p_positive"].to_numpy())
        for key, value in metrics.items():
            logged = arm["summary"]["gain_metrics"][action].get(key)
            if value is not None and logged is not None:
                _close(value, logged, f"{arm['arm']}: saved metric {key} disagrees with per-case values")
        rows.append(dict(arm=arm["arm"], model="world_model", action=action, **metrics,
                         fair_utility_crps=crps[:, index].mean(),
                         raw_utility_90pct_coverage=((arm["actual"][:, index] >= low[:, index]) &
                                                  (arm["actual"][:, index] <= high[:, index])).mean(),
                         raw_utility_90pct_mean_width=(high[:, index] - low[:, index]).mean()))
    direct = arm["summary"]["same_representation_direct_heads"]
    if direct.get("n") != len(arm["ids"]):
        raise ValueError(f"{arm['arm']}: direct-head evaluation count differs")
    _close(direct["actual_mean"], arm["actual"][:, -1].mean(), "Direct-head endpoint mean differs")
    rows.append(dict(arm=arm["arm"], model="same_representation_direct_heads", action="add_0_1", **direct,
                     fair_utility_crps=None, raw_utility_90pct_coverage=None, raw_utility_90pct_mean_width=None))
    return rows, crps


def _allocation_row(actual, choices, arm, strategy, budget, seed):
    n = len(actual)
    choices = np.asarray(choices)
    if choices.shape != (n,) or not np.isfinite(choices).all() or not np.equal(choices, choices.astype(int)).all():
        raise ValueError("Allocation choices must be integer action indices, one per evaluation case")
    choices = choices.astype(int)
    if ((choices < 0) | (choices >= len(ACTIONS))).any():
        raise ValueError("Allocation action index is out of range")
    used = int(WELLS[choices].sum())
    if used > budget:
        raise ValueError("Saved plan exceeds its declared well budget")
    values = actual[np.arange(n), choices]
    active = WELLS[choices] > 0
    counts = np.bincount(choices, minlength=len(ACTIONS))
    exact_random = float(counts @ actual.mean(0) / n)
    random_fdp = float(counts[1:] @ (actual[:, 1:] <= 0).mean(0) / active.sum()) if active.any() else None
    rng = np.random.default_rng(seed)
    randomized = np.array([actual[np.arange(n), rng.permutation(choices)].mean() for _ in range(1000)])
    return dict(arm=arm, strategy=strategy, budget_wells=int(budget), used_wells=used,
                activated=int(active.sum()), action_counts=dict(zip(ACTIONS, counts.tolist())),
                population_mean_net_gain=values.mean(), total_net_gain=values.sum(),
                null_count=int(((values <= 0) & active).sum()),
                fdp=float((values[active] <= 0).mean()) if active.any() else None,
                matched_action_mix_random_mean=exact_random, matched_action_mix_random_fdp=random_fdp,
                observed_minus_matched_random_mean=values.mean() - exact_random,
                randomization_gain_p025=np.quantile(randomized, .025),
                randomization_gain_p975=np.quantile(randomized, .975),
                randomization_interval_not_sampling_confidence=True,
                action_mix_and_spent_budget_matched=True)


def _allocation_rows(arm, seed):
    rows, n = [], len(arm["ids"])
    expected = [f"planner_budget_{fraction:g}_model_null_{limit}"
                for fraction in FRACTIONS for limit in (None, .35)]
    present = {column for column in arm["frame"] if column.startswith("planner_budget_")}
    if present != set(expected):
        raise ValueError(f"{arm['arm']}: saved planner choices must cover all declared budgets and model-risk settings")
    for column in expected:
        stored = arm["allocations"].loc[arm["allocations"].strategy == column]
        if len(stored) != 1:
            raise ValueError(f"{arm['arm']}: planner allocation summary missing or duplicated: {column}")
        stored = stored.iloc[0]
        budget = stored.budget_wells
        if not np.isfinite(budget) or budget != int(budget):
            raise ValueError("Planner budget must be a finite integer")
        row = _allocation_row(arm["actual"], arm["frame"][column].to_numpy(), arm["arm"], column, int(budget), seed)
        for key in ("used_wells", "activated", "population_mean_net_gain", "null_count"):
            _close(row[key], stored[key], f"{arm['arm']}: stored plan summary disagrees with choices: {key}")
        row["kind"] = "saved_planner"
        rows.append(row)
    for fraction in FRACTIONS:
        count = int(np.ceil(fraction * n))
        for score in ("predicted", "p_positive"):
            ordering = np.lexsort((arm["ids"], -arm["frame"]["add_0_1__" + score].to_numpy()))
            choices = np.zeros(n, int)
            choices[ordering[:count]] = 3
            row = _allocation_row(arm["actual"], choices, arm["arm"],
                                  f"ADD_TWO_{score}_top_{fraction:g}", 2 * count, seed)
            row.update(kind="fixed_action_ranking", nominal_fraction=fraction,
                       actual_fraction=count / n, score=score)
            rows.append(row)
    return rows


def _paired_bootstrap(arms, crps, replicates, seed):
    if replicates == 0:
        return []
    rng, n, rows = np.random.default_rng(seed), len(arms[0]["ids"]), []
    indices = rng.integers(0, n, size=(replicates, n))
    for left, right in ((0, 1), (1, 2), (0, 2)):
        for action_index, action in enumerate(ACTIONS[1:], 1):
            for metric in ("pearson_r", "positive_auc", "fair_utility_crps"):
                differences = []
                for ix in indices:
                    values = []
                    for arm_index in (left, right):
                        arm = arms[arm_index]
                        if metric == "fair_utility_crps":
                            value = crps[arm_index][ix, action_index].mean()
                        else:
                            f = arm["frame"].iloc[ix]
                            value = _metrics(arm["actual"][ix, action_index],
                                             f[action + "__predicted"].to_numpy(),
                                             f[action + "__p_null"].to_numpy(),
                                             f[action + "__p_positive"].to_numpy())[metric]
                        values.append(value)
                    if all(value is not None and np.isfinite(value) for value in values):
                        differences.append(values[1] - values[0])
                rows.append(dict(comparison=f"{ARMS[right]} minus {ARMS[left]}", action=action, metric=metric,
                                 requested_replicates=replicates, valid_replicates=len(differences),
                                 percentile_p025=np.quantile(differences, .025) if differences else None,
                                 percentile_p975=np.quantile(differences, .975) if differences else None,
                                 scope="paired case resampling conditional on fitted rules and observed batch layout",
                                 independent_campaign_or_training_seed_uncertainty=False,
                                 formal_certificate=False))
    return rows


def _markdown_table(rows, columns):
    def show(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        return f"{value:.5g}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(show(row.get(key)) for key in columns) + " |" for row in rows)
    return "\n".join(lines)


def generate_analysis(root, *, bootstrap_replicates=0, seed=2026):
    """Write the matched comparison only when all three evaluations exist.

    Optional paired compound bootstrap is conditional descriptive uncertainty;
    it does not certify a policy under the shared-batch sampling design.
    """
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    root = Path(root)
    arms = [_read_arm(root, name) for name in ARMS]
    _validate_matching(arms)
    action_rows, crps, allocation_rows, measurement_rows = [], [], [], []
    for arm in arms:
        rows, scores = _action_rows(arm)
        action_rows.extend(rows)
        crps.append(scores)
        allocation_rows.extend(_allocation_rows(arm, seed))
        measurement = arm["summary"]["measurement"]
        measurement_rows.append(dict(arm=arm["arm"], **{key: measurement.get(key) for key in
            ("fixed_space_nll_per_coordinate", "fixed_space_mse_per_coordinate",
             "uncalibrated_gaussian_90pct_coordinate_coverage", "scored_coordinates")},
            nll_scope="compound marginal, not campaign joint",
            conformal_diagnostic=arm["summary"].get("coverage_diagnostic")))
    bootstrap = _paired_bootstrap(arms, crps, bootstrap_replicates, seed)
    output = _clean(dict(arms=list(ARMS), evaluation_ids=arms[0]["ids"],
                        calibration_ids=arms[0]["calibration_ids"], action_order=list(ACTIONS),
                        training_counts=arms[0]["training_counts"], seed=arms[0]["summary"]["model"].get("seed"),
                        arm_configurations={arm["arm"]: arm["summary"]["model"] for arm in arms},
                        checkpoint_selection="fixed-role validation predictive NLL, shared training implementation",
                        n_evaluation=len(arms[0]["ids"]), monte_carlo_samples=len(arms[0]["samples"]),
                        evidence_scope="single-seed matched DEV objective diagnostic; not efficacy certification",
                        action_metrics=action_rows, measurement_metrics=measurement_rows,
                        same_budget_comparisons=allocation_rows, paired_bootstrap=bootstrap,
                        endpoints_and_order_validated=True, final_opened=False))
    text = ["# Full DEV objective comparison", "",
            f"{output['n_evaluation']} matched evaluation compounds; seed {output['seed']}; "
            f"{output['monte_carlo_samples']} joint utility draws per arm. The original half-cosine endpoint, "
            "optional-well cost 0.01, and POSITIVE margin 0.005 are identical across arms.", "",
            "All arms use the same fixed-role validation predictive NLL for checkpoint selection. "
            "The complete objective weights and common configuration are recorded in comparison.json.", "",
            "A changes no training objective; B directly optimizes predictive NLL; C adds derived-utility CRPS. "
            "Use this single-seed DEV experiment to diagnose the training objective, not as proof of efficacy or authorization. "
            "A positive point estimate alone does not establish an improvement; shared-batch and training-seed uncertainty remain.", "",
            "## Utility prediction", "",
            _markdown_table(output["action_metrics"], ["arm", "model", "action", "pearson_r", "spearman_r", "r2", "positive_auc", "positive_brier", "null_brier"]), "",
            "## Distribution scoring", "",
            _markdown_table([r for r in output["action_metrics"] if r["model"] == "world_model"],
                            ["arm", "action", "fair_utility_crps", "raw_utility_90pct_coverage", "raw_utility_90pct_mean_width"]), "",
            "CRPS uses the fair finite-ensemble correction. Coverage is the uncalibrated central 90% interval of utility draws, "
            "not conformal coverage. Direct-head CRPS is unavailable because that comparator does not export a coherent utility distribution.", "",
            _markdown_table(output["measurement_metrics"], ["arm", "fixed_space_nll_per_coordinate", "fixed_space_mse_per_coordinate", "uncalibrated_gaussian_90pct_coordinate_coverage"]), "",
            "Measurement NLL is a compound-marginal score. Coordinates are not independent biological replicates. "
            "The saved conformal diagnostics, including unbounded intervals, remain in comparison.json and are not certificates.", "",
            "## Same-budget decisions", "",
            _markdown_table(output["same_budget_comparisons"], ["arm", "strategy", "budget_wells", "used_wells", "activated", "population_mean_net_gain", "fdp", "matched_action_mix_random_mean", "observed_minus_matched_random_mean"]), "",
            "Each comparator keeps the policy's actual spent wells and action counts fixed. The random mean is the exact "
            "uniform-permutation expectation. The reported randomization ranges in comparison.json describe selection randomization, "
            "not sampling confidence. Fixed ADD_TWO rankings report both expected-gain and POSITIVE-probability scores "
            "at all three declared fractions; ties use compound ID. No score orientation is changed after evaluation.", "",
            "## Interpretation", "",
            "Compare B with A to test exact predictive likelihood; compare C with B to test the added utility objective. "
            "Inspect ranking, distribution calibration, and same-budget realized value together. An NLL improvement alone "
            "does not show better acquisition. These analyses do not open FINAL or change the seven-criterion contract."]
    if bootstrap:
        text += ["", "## Optional paired descriptive intervals", "",
                 "Intervals resample compounds jointly across arms while holding fitted models and the observed batch layout fixed. "
                 "They omit independent-batch and training-seed variation and do not support formal certification.", "",
                 _markdown_table(output["paired_bootstrap"], ["comparison", "action", "metric", "valid_replicates", "percentile_p025", "percentile_p975"])]
    (root / "comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(output["action_metrics"]).to_csv(root / "comparison.tsv", sep="\t", index=False)
    pd.DataFrame(output["same_budget_comparisons"]).to_csv(root / "allocation_comparison.tsv", sep="\t", index=False)
    (root / "RESULT_ANALYSIS.md").write_text("\n".join(text) + "\n")
    return output
