"""Fixed-endpoint measurement and acquisition evaluation on untouched split IDs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, t as student_t
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error, r2_score

from .training import fixed_batch, batches
from .utility import enumerate_actions, cosine_utility_samples, UtilityResult, selected_outcomes
from .planning import allocate_actions
from .baselines import ConditionalResidualBootstrap, DirectRepresentationHeads
from .model import EnvironmentNoiseCache
from .batch_baseline import BatchConditionalResidualBootstrap
from .provenance import assert_evaluation_provenance
from .data import attach_library_context


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def gain_metrics(actual, mean, p_null, p_positive):
    actual, mean = np.asarray(actual), np.asarray(mean)
    null, pos = actual <= 0, actual >= .005
    varied = np.std(actual) > 0 and np.std(mean) > 0
    return {"n": len(actual), "actual_mean": actual.mean(), "predicted_mean": mean.mean(),
            "actual_sd": actual.std(), "predicted_mean_sd": mean.std(),
            "null_rate": null.mean(), "positive_rate": pos.mean(),
            "pearson_r": pearsonr(actual, mean).statistic if varied else None,
            "spearman_r": spearmanr(actual, mean).statistic if varied else None,
            "mse": mean_squared_error(actual, mean), "r2": r2_score(actual, mean),
            "null_auc": roc_auc_score(null, p_null) if len(np.unique(null)) == 2 else None,
            "positive_auc": roc_auc_score(pos, p_positive) if len(np.unique(pos)) == 2 else None,
            "null_brier": brier_score_loss(null, p_null), "positive_brier": brier_score_loss(pos, p_positive)}


@torch.no_grad()
def predict_utilities(model, scaler, dataset, indices, config):
    initial = dataset.Y[indices, 0]
    if (dataset.Y.shape[1] < 4 or not np.all(dataset.well_mask[indices, :4])
            or not np.all(dataset.observed_mask[indices, 0])
            or not np.isfinite(initial).all()):
        raise ValueError("Four-role cosine utility requires an observed finite initial X and four physical roles")
    normalized = scaler.transform(dataset)
    if hasattr(model, "library_bank"):
        normalized = attach_library_context(normalized, model.library_bank)
    model.eval()
    pieces, nll, squared_error, dimensions = [], 0.0, 0.0, 0
    max_standardized_errors, marginal_hits = [], 0
    # A block is reused across compound chunks so shared environmental draws
    # remain shared. Only S*B*A utility samples are retained, not S*N*T*D spectra.
    counts = [min(config.mc_chunk_size, config.samples - start)
              for start in range(0, config.samples, config.mc_chunk_size)]
    caches = [EnvironmentNoiseCache() for _ in counts]
    generators = [torch.Generator(device="cpu").manual_seed(config.seed + 501 + block * 100003)
                  for block in range(len(counts))]
    family = getattr(model, "observation_family", "gaussian")
    if family not in {"gaussian", "copula_t4"}:
        raise ValueError("Declare the observation family's marginal quantiles before evaluation")
    radius_90 = (1.6448536269514722 if family == "gaussian"
                 else float(student_t.ppf(.95, df=4) / np.sqrt(2.)))
    for ix in batches(indices, min(config.batch_size, 8)):
        batch, target, mask = fixed_batch(normalized, ix, config)
        distribution = model(batch)
        original_mean = scaler.inverse_y(distribution.mean).cpu().numpy()
        original_target = dataset.Y[ix, 1:4]
        error = original_mean - original_target
        observed = np.isfinite(original_target) & mask.cpu().numpy()[..., None]
        if not observed.reshape(len(ix), -1).any(1).all():
            raise ValueError("Evaluation compounds require at least one observed target coordinate")
        original_sd = np.sqrt(distribution.marginal_variance.cpu().numpy()) * scaler.y_scale
        standardized = np.where(observed, np.abs(error) / np.maximum(original_sd, 1e-12), 0)
        max_standardized_errors.extend(standardized.reshape(len(ix), -1).max(1).tolist())
        marginal_hits += int(((standardized <= radius_90) & observed).sum())
        squared_error += np.square(error[observed]).sum()
        dimensions += observed.sum()
        log_probability = distribution.log_prob(target, mask).cpu().numpy()
        # Change of coordinates back to the common fixed measurement space.
        log_jacobian = (observed * np.log(scaler.y_scale)[None, None, :]).sum((1, 2))
        nll += (-log_probability + log_jacobian).sum()
        draws = []
        for count, cache, generator in zip(counts, caches, generators):
            sampled = scaler.inverse_y(distribution.sample_joint(
                count, generator, environment_noise_cache=cache)).cpu().numpy()
            result = cosine_utility_samples(dataset.Y[ix, 0], sampled, enumerate_actions([0, 1]), 2)
            draws.append(result.samples)
        pieces.append(np.concatenate(draws, axis=0))
    result = UtilityResult.from_samples(enumerate_actions([0, 1]), np.concatenate(pieces, axis=1),
                                         np.array([0, .01, .01, .02]), utility_name="half_cosine_gain")
    return result, {"fixed_space_nll_per_coordinate": nll/dimensions,
                    "fixed_space_mse_per_coordinate": squared_error/dimensions,
                    "scored_coordinates": int(dimensions),
                    ("uncalibrated_gaussian_90pct_coordinate_coverage" if family == "gaussian"
                     else "uncalibrated_t4_90pct_coordinate_coverage"): marginal_hits/dimensions,
                    "observation_family": family,
                    "max_standardized_error_per_compound": max_standardized_errors,
                    "nll_is_compound_marginal_score_not_campaign_joint_score": True,
                    "monte_carlo_environment_latents_shared_across_all_evaluation_chunks": True,
                    "monte_carlo_samples": config.samples,
                    "monte_carlo_spectral_chunk": config.mc_chunk_size,
                    "mean_utility_mcse": result.samples.std(axis=0, ddof=1).mean(axis=0) / np.sqrt(config.samples)
                        if config.samples > 1 else [None] * len(result.actions),
                    "mcse_is_numerical_not_statistical_evidence": True}


def actual_utilities(dataset, indices, *, missing_policy="error"):
    """Original endpoint; missing completion must be explicitly requested.

    Under ``worst``, each action with an unavailable required measurement gets
    the known worst half-cosine difference (-1) minus its actual well cost.
    STOP remains zero. This option does not change any existing contract.
    """
    if missing_policy not in {"error", "worst"}:
        raise ValueError("Declare missing_policy='error' or 'worst'")
    values = dataset.Y[indices, :4]
    available = (dataset.observed_mask[indices, :4] & dataset.well_mask[indices, :4]
                 & np.isfinite(values).all(-1))
    if not available.all() and missing_policy == "error":
        raise ValueError("Missing observed endpoint roles require an explicit completion policy")
    actions = enumerate_actions([0, 1])
    # Safe fill is computational only. Every action requiring a filled role is
    # subsequently replaced with its declared worst outcome, never scored as real.
    safe = np.where(available[..., None], values, 0)
    result = cosine_utility_samples(safe[:, 0], safe[:, 1:4][None], actions, 2)
    if not available.all():
        draws = result.samples.copy()
        for j, action in enumerate(actions):
            if action.wells:
                required = [0, 3] + [int(t) + 1 for t in action.target_indices]
                valid = available[:, required].all(1)
                draws[:, ~valid, j] = -1 - result.costs[j]
        result = UtilityResult.from_samples(actions, draws, result.costs, utility_name="half_cosine_gain")
    return result


def allocation_observed(actual, choices, *, label, budget=None):
    values = selected_outcomes(actual, choices)[0]
    choices = np.asarray(choices)
    wells = np.array([a.wells for a in actual.actions])[choices]
    active = wells > 0
    return {"strategy": label, "budget_wells": budget, "used_wells": int(wells.sum()),
            "activated": int(active.sum()), "total_net_gain": values.sum(),
            "population_mean_net_gain": values.mean(),
            "active_mean_net_gain": values[active].mean() if active.any() else None,
            "null_count": int(((values <= 0) & active).sum()),
            "fdp": (values[active] <= 0).mean() if active.any() else None}


def compare_allocations(predicted, actual, ids, *, seed=0, fractions=(.05, .1, .25)):
    """Measured decisions; model risk constraints never reported as certificates."""
    rows, choices_to_save = [], {}
    n = len(ids)
    for a in range(len(actual.actions)):
        choices = np.full(n, a)
        rows.append(allocation_observed(actual, choices, label="fixed_" + actual.actions[a].name))
    rng = np.random.default_rng(seed)
    for fraction in fractions:
        budget = 2 * int(np.ceil(fraction * n))
        for limit in (None, .35):
            allocation = allocate_actions(predicted, budget, max_null_fraction=limit)
            label = f"planner_budget_{fraction:g}_model_null_{limit}"
            row = allocation_observed(actual, allocation.action_indices, label=label, budget=budget)
            row.update(predicted_total_gain=allocation.expected_total_gain,
                       predicted_null_count=allocation.expected_null_count,
                       certified=False, risk_is_model_expectation=True)
            rows.append(row)
            choices_to_save[label] = allocation.action_indices
            # Isolate which compounds receive each action from the action mix
            # and actual spent budget. Every shuffle keeps STOP/ADD_ONE/ADD_TWO
            # counts exactly equal to this planner's allocation.
            matched = []
            for _ in range(500):
                shuffled = rng.permutation(allocation.action_indices)
                matched.append(allocation_observed(actual, shuffled, label="matched_random", budget=budget))
            random_values = np.asarray([r["population_mean_net_gain"] for r in matched])
            rows.append({"strategy": "action_mix_random_for_" + label, "budget_wells": budget,
                         "used_wells": allocation.total_wells,
                         "activated": int(np.count_nonzero(allocation.action_indices)),
                         "population_mean_net_gain": random_values.mean(),
                         "fdp": np.mean([r["fdp"] for r in matched]) if np.count_nonzero(allocation.action_indices) else None,
                         "randomization_gain_p025": np.quantile(random_values, .025),
                         "randomization_gain_p975": np.quantile(random_values, .975),
                         "observed_minus_matched_random_mean": row["population_mean_net_gain"] - random_values.mean(),
                         "action_mix_and_spent_budget_matched": True,
                         "randomization_interval_not_sampling_confidence": True})
        # Fixed ADD_TWO instrument, same exactly declared budget, score only.
        number = budget // 2
        ordering = np.lexsort((np.asarray(ids, str), -predicted.p_positive[:, -1]))
        choices = np.zeros(n, dtype=int)
        choices[ordering[:number]] = len(actual.actions)-1
        rows.append(allocation_observed(actual, choices, label=f"pPOS_ADD_TWO_top_{fraction:g}", budget=budget))
        reference = []
        for _ in range(500):
            choices = np.zeros(n, dtype=int)
            choices[rng.choice(n, number, replace=False)] = len(actual.actions)-1
            reference.append(allocation_observed(actual, choices, label="random", budget=budget))
        rows.append({"strategy": f"random_ADD_TWO_{fraction:g}", "budget_wells": budget,
                     "used_wells": budget, "activated": number,
                     "population_mean_net_gain": np.mean([r["population_mean_net_gain"] for r in reference]),
                     "fdp": np.mean([r["fdp"] for r in reference]),
                     "randomization_gain_p025": np.quantile([r["population_mean_net_gain"] for r in reference], .025),
                     "randomization_gain_p975": np.quantile([r["population_mean_net_gain"] for r in reference], .975),
                     "randomization_interval_not_sampling_confidence": True})
        for action_index in (1, 2):
            count = min(budget, n)
            action_values = actual.samples[0, :, action_index]
            random_gains = np.array([action_values[rng.choice(n, count, replace=False)].sum() / n
                                    for _ in range(500)])
            rows.append({"strategy": f"random_{actual.actions[action_index].name}_budget_{fraction:g}",
                         "budget_wells": budget, "used_wells": count, "activated": count,
                         "population_mean_net_gain": count / n * action_values.mean(),
                         "fdp": (action_values <= 0).mean(),
                         "randomization_gain_p025": np.quantile(random_gains, .025),
                         "randomization_gain_p975": np.quantile(random_gains, .975),
                         "expectation_is_exact_uniform_subset_mean": True,
                         "randomization_interval_not_sampling_confidence": True})
    return rows, choices_to_save


@torch.no_grad()
def representations(model, scaler, dataset, indices, config):
    normalized = scaler.transform(dataset)
    if hasattr(model, "library_bank"):
        normalized = attach_library_context(normalized, model.library_bank)
    values = []
    for ix in batches(indices, config.batch_size):
        batch, _, _ = fixed_batch(normalized, ix, config)
        h = model.profile_encoder(batch["context_y"][:, 0])
        # The same legal conditioning information as the world model, not only
        # the representation: include known target conditions and references.
        legal = [h, batch["chem"]]
        if hasattr(model,"_state_details"):
            # Match the actual learned state, including library, reference
            # anchors and chemical posterior, not just the single-well encoder.
            state=model._state_details(batch)
            legal.extend(state[key].reshape(len(ix),-1) for key in
                         ("state","posterior_mean","posterior_var"))
        for key in ("context_cond", "context_reference", "context_reference_mask",
                    "target_cond", "target_reference", "target_reference_mask",
                    "context_mask", "context_group", "target_group"):
            legal.append(batch[key].reshape(len(ix), -1).to(h.dtype))
        values.append(torch.cat(legal, -1).cpu().numpy())
    return np.concatenate(values)


def evaluate_model(model, scaler, dataset, splits, config, directory):
    from .calibration import SplitConformalUtility
    from .splits import assert_disjoint_splits
    assert_disjoint_splits(dataset.ids, splits)
    assert_evaluation_provenance(model, scaler, dataset, splits)
    directory = Path(directory)
    evaluation, calibration = splits["evaluation"], splits["calibration"]
    prediction, measurement = predict_utilities(model, scaler, dataset, evaluation, config)
    cal_prediction, cal_measurement = predict_utilities(model, scaler, dataset, calibration, config)
    actual, cal_actual = actual_utilities(dataset, evaluation), actual_utilities(dataset, calibration)
    metrics = {a.name: gain_metrics(actual.samples[0, :, j], prediction.mean[:, j],
                                   prediction.p_null[:, j], prediction.p_positive[:, j])
               for j, a in enumerate(prediction.actions) if a.wells}
    plans, choices = compare_allocations(prediction, actual, dataset.ids[evaluation], seed=config.seed)
    frame = pd.DataFrame({"compound_id": dataset.ids[evaluation]})
    for j, action in enumerate(prediction.actions):
        frame[action.name + "__actual"] = actual.samples[0, :, j]
        frame[action.name + "__predicted"] = prediction.mean[:, j]
        frame[action.name + "__p_null"] = prediction.p_null[:, j]
        frame[action.name + "__p_positive"] = prediction.p_positive[:, j]
        frame[action.name + "__mcse"] = (prediction.samples[:, :, j].std(0, ddof=1) / np.sqrt(config.samples)
                                            if config.samples > 1 else np.nan)
    for name, value in choices.items():
        frame[name] = value
    frame.to_csv(directory / "evaluation_predictions.tsv", sep="\t", index=False)
    energy = np.log1p(np.sqrt(np.mean(dataset.Y[evaluation, 0] ** 2, axis=-1)))
    energy_diagnostics = []
    for j, action in enumerate(prediction.actions):
        if not action.wells:
            continue
        for name, values in (("predicted", prediction.mean[:, j]), ("actual", actual.samples[0, :, j])):
            energy_diagnostics.append({"action": action.name, "quantity": name,
                "spearman_with_log_rms_x": spearmanr(energy, values).statistic
                    if np.std(energy) > 0 and np.std(values) > 0 else None,
                "sd": values.std(), "n": len(values)})
    pd.DataFrame(energy_diagnostics).to_csv(directory / "norm_diagnostics.tsv", sep="\t", index=False)
    frame[["compound_id"] + [c for c in frame if c.endswith("__mcse")]].to_csv(
        directory / "mc_precision.tsv", sep="\t", index=False)
    pd.DataFrame(clean(plans)).to_csv(directory / "allocations.tsv", sep="\t", index=False)
    np.savez_compressed(directory / "utility_draws.npz", samples=prediction.samples,
                        calibration_samples=cal_prediction.samples, ids=dataset.ids[evaluation],
                        calibration_ids=dataset.ids[calibration])
    # The exchangeability/independence assumptions are not established on this
    # shared-batch DEV dataset. These are coverage diagnostics, not certificates.
    conformal = SplitConformalUtility.fit(
        cal_actual.samples[0, :, 1:], cal_prediction.mean[:, 1:], dataset.ids[calibration],
        alpha=.1, training_ids=dataset.ids[np.r_[splits["train"], splits["validation"]]])
    low, high = conformal.interval(prediction.mean[:, 1:], dataset.ids[evaluation], support=(-1.02, 1.0))
    inside = (actual.samples[0, :, 1:] >= low) & (actual.samples[0, :, 1:] <= high)
    coverage_info = {"nominal": .9, "observed_joint_action_coverage": inside.all(1).mean(),
                     "observed_per_action_coverage": inside.mean(0), "half_width": conformal.radius,
                     "calibration_n": len(calibration), "evaluation_n": len(evaluation),
                     "certificate": False, "reason": "DEV shared-batch exchangeability not established"}
    measurement_conformal = SplitConformalUtility.fit(
        np.asarray(cal_measurement["max_standardized_error_per_compound"]),
        np.zeros(len(calibration)), dataset.ids[calibration], alpha=.1,
        training_ids=dataset.ids[np.r_[splits["train"], splits["validation"]]])
    coverage_info["measurement_simultaneous_standardized_radius"] = measurement_conformal.radius
    coverage_info["measurement_all_coordinates_coverage"] = float(np.mean(
        np.asarray(measurement["max_standardized_error_per_compound"]) <= measurement_conformal.radius))
    coverage_info["measurement_coverage_is_not_conditional_distribution_calibration"] = True
    np.savez_compressed(directory / "utility_intervals.npz", lower=low, upper=high,
                        ids=dataset.ids[evaluation], calibration_ids=dataset.ids[calibration])
    train = splits["train"]
    base = ConditionalResidualBootstrap().fit(dataset.Y[train, 0], dataset.Y[train, 1:4])
    # Batch in memory so the residual baseline never materializes S*N*T*D for the full library.
    base_samples = []
    for counter, ix in enumerate(batches(evaluation, 8)):
        chunks = []
        for start in range(0, config.samples, config.mc_chunk_size):
            count = min(config.mc_chunk_size, config.samples - start)
            sampled = base.sample_joint(dataset.Y[ix, 0], count, seed=config.seed + counter + start * 100003)
            chunks.append(cosine_utility_samples(dataset.Y[ix, 0], sampled, prediction.actions, 2).samples)
        base_samples.append(np.concatenate(chunks, axis=0))
    base_result = UtilityResult.from_samples(prediction.actions, np.concatenate(base_samples, axis=1), prediction.costs)
    base_metrics = gain_metrics(actual.samples[0, :, -1], base_result.mean[:, -1],
                                base_result.p_null[:, -1], base_result.p_positive[:, -1])
    conditional = BatchConditionalResidualBootstrap().fit(
        dataset.Y[train, 0], dataset.Y[train, 1:4], dataset.groups[train, 0], dataset.groups[train, 1:4],
        train_ids=dataset.ids[train])
    conditional_samples = []
    for counter, ix in enumerate(batches(evaluation, 8)):
        chunks = []
        for start in range(0, config.samples, config.mc_chunk_size):
            count = min(config.mc_chunk_size, config.samples - start)
            draws = conditional.sample_joint(dataset.Y[ix, 0], dataset.groups[ix, 0], dataset.groups[ix, 1:4],
                                             n_samples=count, seed=config.seed + counter + start * 100003)
            chunks.append(cosine_utility_samples(dataset.Y[ix, 0], draws, prediction.actions, 2).samples)
        conditional_samples.append(np.concatenate(chunks, axis=0))
    conditional_result = UtilityResult.from_samples(prediction.actions, np.concatenate(conditional_samples, axis=1), prediction.costs)
    conditional_metrics = gain_metrics(actual.samples[0, :, -1], conditional_result.mean[:, -1],
                                       conditional_result.p_null[:, -1], conditional_result.p_positive[:, -1])
    conditional.save(directory / "batch_conditional_baseline.npz")
    conditional_diagnostics = conditional.prediction_diagnostics(dataset.groups[evaluation, 0], dataset.groups[evaluation, 1:4])
    direct = DirectRepresentationHeads(config.seed, config.threads).fit(
        representations(model, scaler, dataset, train, config), actual_utilities(dataset, train).samples[0, :, -1])
    direct_prediction = direct.predict(representations(model, scaler, dataset, evaluation, config))
    direct_metrics = gain_metrics(actual.samples[0, :, -1], **direct_prediction)
    summary = {"evidence_scope": dataset.metadata.get("evidence_scope", "DEVELOPMENT_UNCERTIFIED_PORTABLE_MEASUREMENTS"),
               "data_provenance": dataset.metadata,
               "model": as_config(config), "measurement": measurement, "gain_metrics": metrics,
               "coverage_diagnostic": coverage_info, "role_conditional_residual_baseline": base_metrics,
               "batch_conditional_residual_baseline": conditional_metrics,
               "batch_conditional_diagnostics": conditional_diagnostics,
               "same_representation_direct_heads": direct_metrics,
               "direct_heads_input": "full learned conditional state, chemical posterior, initial encoding and same ablated legal covariates",
               "utility": "original fixed-space half-cosine gain",
               "cost_per_optional_well": .01, "positive_margin": .005,
               "reference_cost": "existing shared controls treated as sunk; new-panel costs not included",
               "evaluation_compounds": len(evaluation), "calibration_compounds": len(calibration),
               "full_feature_dimension": dataset.Y.shape[-1], "final_opened": False,
               "energy_association_diagnostic": clean(energy_diagnostics),
               "energy_association_is_not_proof_of_single_dimensional_collapse": True,
               "known_limits": ["This evaluator reports development metrics, not a formal certificate",
                                "Target2 conditioning cannot be tested when those panels are not available",
                                "A compound split alone does not remove shared-batch dependence",
                                "Reference pre-decision availability follows explicit input mask"]}
    write_json(directory / "evaluation.json", summary)
    return clean(summary)


def as_config(config):
    from dataclasses import asdict
    return asdict(config)
