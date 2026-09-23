"""Fixed X-neighbor transfer of complete donor geometry, responses and contrasts.

No neural/kernel fitting or outcome-dependent neighborhood selection is done.
An empirical donor is the entire normalized four-role Gram block. The same
predeclared weights are also used for separate future-response and repeat-
contrast diagnostics. Cross-batch contrasts are not identified technical noise.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .neighbor_support import load_inputs, pairwise_profile_distances, neighbor_statistics, _write_json
from .gram_geometry import profiles_to_gram, gram_gains, gram_observables, OBSERVABLE_NAMES
from .gram_evaluation import free_entries, fit_score_scale
from .baseline_policy import evaluate_predictions, ACTIONS


ARMS = ("GLOBAL", "COSINE_TOP20", "RMS_TOP20")
PARTITIONS = ("validation", "calibration", "evaluation")
SEED = 20260914


def _weights(weights, n_donors):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.ndim != 2 or weights.shape[1] != n_donors or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.allclose(weights.sum(1), 1., rtol=1e-10, atol=1e-12)):
        raise ValueError("Finite nonnegative row-normalized donor weights required")
    return weights


def fixed_neighbor_weights(distances, query_ids, donor_ids, bandwidth, k=20):
    """Choose neighbors and weights solely from decision-time X distances."""
    distance = np.asarray(distances, dtype=float).copy()
    query_ids, donor_ids = np.asarray(query_ids, str), np.asarray(donor_ids, str)
    if distance.shape != (len(query_ids), len(donor_ids)) or not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("Finite nonnegative X-only distances matching query and donor IDs required")
    if not np.isfinite(bandwidth) or bandwidth <= 0 or not isinstance(k, int) or not 0 < k <= len(donor_ids):
        raise ValueError("Positive fixed bandwidth and valid neighbor count required")
    distance[query_ids[:, None] == donor_ids[None]] = np.inf
    lexical = np.argsort(donor_ids, kind="stable")
    order = lexical[np.argsort(distance[:, lexical], axis=1, kind="stable")][:, :k]
    nearest = np.take_along_axis(distance, order, axis=1)
    if not np.isfinite(nearest).all():
        raise ValueError("Insufficient distinct donor identities for the fixed neighborhood")
    local = np.exp(-(nearest - nearest[:, :1]) / bandwidth)
    local /= local.sum(1, keepdims=True)
    weights = np.zeros(distance.shape, dtype=np.float64)
    np.put_along_axis(weights, order, local, axis=1)
    return weights


def weighted_crps(donor_values, observed, weights):
    """Exact CRPS of the specified weighted discrete predictive distribution.

    This is E|A-y| - .5 E|A-A'| for independent draws *from the discrete law*.
    It includes the zero self-pair contributions, and must not use the fair
    finite-Monte-Carlo correction for an underlying unobserved population law.
    """
    atoms, observed = np.asarray(donor_values, float), np.asarray(observed, float)
    if atoms.ndim == 1:
        atoms, observed = atoms[:, None], observed[:, None]
    if atoms.ndim != 2 or observed.ndim != 2 or atoms.shape[1] != observed.shape[1]:
        raise ValueError("Donor atoms and observed outcomes need matching output coordinates")
    weights = _weights(weights, len(atoms))
    if len(weights) != len(observed) or not np.isfinite(atoms).all() or not np.isfinite(observed).all():
        raise ValueError("Finite donor and query outcomes with matching weight rows required")
    result = np.empty(observed.shape)
    for j in range(atoms.shape[1]):
        order = np.argsort(atoms[:, j], kind="stable")
        values, w = atoms[order, j], weights[:, order]
        before_w = np.cumsum(w, axis=1) - w
        before_wx = np.cumsum(w * values, axis=1) - w * values
        pair_half = (w * (values * before_w - before_wx)).sum(1)
        first = (weights * np.abs(atoms[None, :, j] - observed[:, None, j])).sum(1)
        result[:, j] = first - pair_half
    return result


def weighted_energy_score(donor_values, observed, weights):
    """Exact joint energy score of complete weighted multivariate donor atoms."""
    atoms, observed = np.asarray(donor_values, float), np.asarray(observed, float)
    weights = _weights(weights, len(atoms))
    if atoms.ndim != 2 or observed.ndim != 2 or atoms.shape[1] != observed.shape[1] or len(observed) != len(weights):
        raise ValueError("Matching complete multivariate donor/query blocks required")
    if not np.isfinite(atoms).all() or not np.isfinite(observed).all():
        raise ValueError("Energy-score atoms must be finite")
    donor_distance = np.linalg.norm(atoms[:, None] - atoms[None, :], axis=-1)
    observed_distance = np.linalg.norm(observed[:, None] - atoms[None, :], axis=-1)
    return ((weights * observed_distance).sum(1)
            - .5 * np.einsum("qi,ij,qj->q", weights, donor_distance, weights, optimize=True))


def weighted_quantiles(donor_values, weights, probabilities=(.025, .975)):
    atoms = np.asarray(donor_values, float)
    if atoms.ndim == 1:
        atoms = atoms[:, None]
    weights = _weights(weights, len(atoms))
    probabilities = np.asarray(probabilities, float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("Quantile probabilities must lie strictly within (0,1)")
    if not np.isfinite(atoms).all():
        raise ValueError("Quantiles require finite atoms")
    result = np.empty((len(probabilities), len(weights), atoms.shape[1]))
    for j in range(atoms.shape[1]):
        order = np.argsort(atoms[:, j], kind="stable")
        cdf = np.cumsum(weights[:, order], axis=1)
        cdf[:, -1] = 1.
        for p, probability in enumerate(probabilities):
            selected = (cdf >= probability).argmax(1)
            result[p, :, j] = atoms[order[selected], j]
    return result


def response_targets(four_profiles, feature_scale):
    """Separated observed responses and repeat contrasts, not noise components.

    The mean of future Z1/Z2/V is an observed response-profile average. Its
    difference from X is repeat-to-repeat response change, not a causal drug
    effect or a claim of actual time progression. Contrast energies retain
    fixed batch/role offsets; they are not unbiased estimates of sigma^2.
    """
    y, scale = np.asarray(four_profiles, float), np.asarray(feature_scale, float)
    if y.ndim != 3 or y.shape[1] != 4 or scale.shape != (y.shape[2],) or np.any(scale <= 0):
        raise ValueError("Four complete profile roles and their fixed TRAIN feature scales required")
    if not np.isfinite(y).all() or not np.isfinite(scale).all():
        raise ValueError("Do not omit nonfinite outcomes")
    future = y[:, 1:].mean(1)
    contrast = np.stack((y[:, 1]-y[:, 2], y[:, 1]-y[:, 3], y[:, 2]-y[:, 3]), axis=1)
    return dict(future_mean_profile=future, future_mean_minus_X=future-y[:, 0],
                repeat_contrast_energy=np.mean(contrast**2, axis=-1),
                standardized_repeat_contrast_energy=np.mean((contrast/scale)**2, axis=-1))


def _mean_error_rows(prediction, observed, global_prediction, scale=None):
    residual, baseline = prediction-observed, global_prediction-observed
    if scale is not None:
        residual, baseline = residual/scale, baseline/scale
    error, baseline_error = np.mean(residual**2, axis=-1), np.mean(baseline**2, axis=-1)
    total = error.sum()
    return dict(mse=float(error.mean()), median_object_mse=float(np.median(error)),
        max_object_sse_fraction=float(error.max()/total) if total else None,
        r2_against_global_donor_mean=float(1-error.mean()/baseline_error.mean()) if baseline_error.mean() else None), error


def _distribution_rows(donor, observed, weights, names):
    predicted = weights @ donor
    crps = weighted_crps(donor, observed, weights)
    low, high = weighted_quantiles(donor, weights)
    rows = []
    for j, name in enumerate(names):
        error = predicted[:, j]-observed[:, j]
        rows.append(dict(name=name, actual_mean=float(observed[:, j].mean()),
            predicted_mean=float(predicted[:, j].mean()), mean_bias=float(error.mean()),
            prediction_mse=float(np.mean(error**2)), median_object_squared_error=float(np.median(error**2)),
            crps=float(crps[:, j].mean()), interval95_coverage=float(((observed[:, j]>=low[:, j]) & (observed[:, j]<=high[:, j])).mean()),
            interval95_mean_width=float(np.mean(high[:, j]-low[:, j]))))
    return rows, crps


def paired_comparison(left, global_reference, seed=SEED, n_bootstrap=2000):
    rng = np.random.default_rng(seed)
    result = {}
    for name, value in left.items():
        difference = np.asarray(value)-np.asarray(global_reference[name])
        if difference.ndim == 2:
            difference = difference.mean(1)
        indices = rng.integers(len(difference), size=(n_bootstrap, len(difference)))
        interval = np.quantile(difference[indices].mean(1), [.025, .975])
        result[name] = dict(mean_difference=float(difference.mean()), interval95=interval.tolist(),
                            negative_favors_neighbor=True)
    return result


def uniform_policy_expectations(raw_policy):
    """Replace an unranked GLOBAL rule's arbitrary tied subsets by expectation.

    For a uniformly chosen size-k subset of N eligible objects, expected
    false/positive counts are k times their population fractions. FDP has
    expectation population NULL fraction because k is fixed; expected FPR and
    sensitivity equal k/N when their corresponding denominators are nonzero.
    No particular subset is claimed to have executed these fractional counts.
    """
    policy = deepcopy(raw_policy)
    metrics = {row["action"]: row for row in raw_policy["action_metrics"]}
    for section in ("within_action", "common_budget"):
        rows, seen = [], set()
        for old in raw_policy[section]:
            key = (old["action"], old["fraction"], old["budget_wells"], old["selected_n"])
            if key in seen:
                continue
            seen.add(key)
            base = metrics[old["action"]]
            n, k = int(base["n"]), int(old["selected_n"])
            if n != old["eligible_n"] or not 0 <= k <= n:
                raise ValueError("Uniform policy quota must match its scored population")
            null, positive = int(base["null_count"]), int(base["positive_count"])
            if not 0 <= null+positive <= n:
                raise ValueError("The unchanged NULL/POSITIVE classes must be disjoint")
            fraction = k/n
            row = {field: old[field] for field in ("action", "fraction", "budget_wells", "budget_scope",
                    "eligible_n", "selected_n", "used_wells", "unused_wells", "coverage")}
            row.update(label=f"{section}__{old['action']}__uniform_expected__{old['fraction']:g}",
                ranking="uniform_random_subset", evaluation_mode="exact_uniform_subset_expectation",
                selected_ids=None, expected_counts_may_be_fractional=True, realized_subset_evaluated=False,
                expected_total_net_gain=k*base["actual_mean"],
                expected_per_selected_net_gain=base["actual_mean"] if k else None,
                expected_per_eligible_net_gain=fraction*base["actual_mean"],
                expected_selected_null_count=k*null/n,
                expected_selected_positive_count=k*positive/n,
                expected_selected_ambiguous_count=k*(n-null-positive)/n,
                expected_fdp=null/n if k else None,
                expected_positive_purity=positive/n if k else None,
                expected_fpr=fraction if null else None,
                expected_sensitivity=fraction if positive else None,
                population_null_count=null, population_positive_count=positive,
                formal_certificate=False)
            rows.append(row)
        policy[section] = rows
    policy["row_trace"] = None
    policy["uniform_subset_expectation"] = dict(
        reason="GLOBAL has the same predictive distribution for every query and no individual ranking",
        retained_budgets="every distinct original action/fraction/physical-well quota",
        counts="fractional expected counts, not an executed subset",
        full_collection="fixed ADD_ALL and STOP policies retain their actual full-population values",
        interpretation="query outcomes evaluate uniform-random expectation; they do not select objects")
    policy["statistical_scope"]["score_ties"] = "GLOBAL uses exact expectation over all uniform subsets of the fixed size, not lexical-ID selection"
    return policy


def repair_global_policy_reports(output):
    """Report-only correction using saved metrics, with no dataset/model read."""
    root = Path(output).resolve()
    result = json.loads((root/"summary.json").read_text())
    if "global_policy_reporting_correction" in result:
        raise ValueError("The GLOBAL expectation reporting correction is already recorded")
    correction = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        change="GLOBAL primary budget policies use exact uniform-subset expectations",
        raw_preserved_as="policy_raw_lexical_debug", neighborhoods_changed=False,
        probabilities_or_scores_changed=False, original_protocol_preserved=True,
        model_or_data_rerun=False)
    for partition in PARTITIONS:
        path = root/"arms"/"GLOBAL"/partition/"metrics.json"
        item = json.loads(path.read_text())
        raw = item["policy"]
        item["policy_raw_lexical_debug"] = raw
        item["policy"] = uniform_policy_expectations(raw)
        item["global_policy_reporting_correction"] = correction
        _write_json(path, item)
        result["partitions"][partition]["GLOBAL"] = item
    result["global_policy_reporting_correction"] = correction
    _write_json(root/"summary.json", result)
    _write_json(root/"GLOBAL_POLICY_CORRECTION.json", correction)
    return correction


def audit(data, scaler, support_run, output):
    root, support = Path(output).resolve(), Path(support_run).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    root.mkdir(parents=True, exist_ok=True)
    # This support report contains X/chemistry distances only, no outcomes.
    support_summary = json.loads((support/"summary.json").read_text())
    bandwidths = {"COSINE_TOP20": support_summary["distance_fitting"]["morphology_cosine"]["bandwidth"],
                  "RMS_TOP20": support_summary["distance_fitting"]["morphology_rms"]["bandwidth"]}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Fixed X-neighbor support-to-response association diagnostic",
        data_scope="existing 639 four-role DEV only; fixed TRAIN383 donors and old three query partitions",
        arms=list(ARMS), top_k=20, bandwidths=bandwidths, bandwidth_source=str(support/"summary.json"),
        bandwidth_rule="unchanged TRAIN leave-one-out fifth-neighbor distance median",
        weights="GLOBAL uniform over TRAIN; neighbors exp(-distance/h), normalized within top20",
        primary_scores=["original ADD_TWO Gamma exact weighted-empirical CRPS", "NULL Brier"],
        secondary_scores=["joint normalized-Gram energy", "all original action scores", "future mean profile MSE",
            "future mean minus X profile MSE", "raw and standardized repeat-contrast energy MSE/CRPS"],
        donor_atom="complete X/Z1/Z2/V normalized-Gram block from one TRAIN compound; never splice wells",
        future_response="future Z1/Z2/V mean profile and its difference from X, not causal drug effects",
        noise_limit="Repeat contrasts include systematic role/batch differences; no identified technical-noise claim",
        exact_crps="score specified weighted discrete law; no Monte Carlo fair correction",
        selection="none; both distance rules and every original object reported regardless of outcome",
        query_outcomes_enter_neighbors=False, chemical_kernel=False, neural_training=False,
        original_endpoint_changed=False, original_contract_changed=False, original_split_changed=False,
        final_opened=False, fifth_repeat_opened=False, formal_certificate=False,
        uncertainty="paired compound bootstrap conditional on fixed shared batches/rules; no multiplicity adjustment or new-holdout claim",
        absolute_support="read previous absolute distances with truncated top20 ESS; low support is not automatically excluded",
        seed=SEED, threads=2, n_bootstrap=2000, n_random=2000)
    _write_json(root/"protocol.json", protocol)
    ids, x, _, _, splits, provenance = load_inputs(data, scaler)
    train = splits["train"]
    queries = np.concatenate([splits[name] for name in PARTITIONS])
    distances = {"COSINE_TOP20": pairwise_profile_distances(x, x[train], "cosine"),
                 "RMS_TOP20": pairwise_profile_distances(x, x[train], "rms")}
    weights = {"GLOBAL": np.full((len(queries), len(train)), 1/len(train))}
    supports = {}
    for arm in ARMS[1:]:
        rows, fit = neighbor_statistics(distances[arm], ids, ids[train], train)
        if not np.isclose(fit["bandwidth"], bandwidths[arm], rtol=1e-12, atol=1e-12):
            raise ValueError("The previously declared X-only neighborhood bandwidth changed")
        weights[arm] = fixed_neighbor_weights(distances[arm][queries], ids[queries], ids[train], bandwidths[arm])
        supports[arm] = [rows[i] for i in queries]
    # This file is written before any target statistic is computed or used.
    np.savez_compressed(root/"frozen_weights.npz", query_ids=ids[queries], donor_ids=ids[train],
                        **weights)
    _write_json(root/"input_provenance.json", provenance)
    _write_json(root/"absolute_support.json", supports)
    weights_frozen_utc = datetime.now(timezone.utc).isoformat()
    with np.load(Path(data)/"measurements.npz", allow_pickle=False) as stored:
        y = stored["Y"].copy()
        if not np.array_equal(stored["ids"].astype(str), ids):
            raise ValueError("Outcome and frozen input identity order differ")
    if y.shape != (639, 4, 3617) or not np.isfinite(y).all():
        raise ValueError("Use all existing complete objects; no outcome-based omission")
    scale = np.asarray(json.loads(Path(scaler).read_text())["y_scale"], float)
    grams = profiles_to_gram(torch.tensor(y, dtype=torch.float64)).numpy()
    gains = gram_gains(torch.tensor(grams)).numpy()
    observables = gram_observables(torch.tensor(grams)).numpy()
    response = response_targets(y, scale)
    score_scale = fit_score_scale(grams[train])
    standardized_gram = free_entries(grams)/score_scale
    del y
    query_lookup = {int(index): i for i, index in enumerate(queries)}
    results = {name: {} for name in PARTITIONS}
    comparisons = {}
    for partition in PARTITIONS:
        ii = splits[partition]
        positions = np.asarray([query_lookup[int(index)] for index in ii])
        traces = {}
        for arm in ARMS:
            w = weights[arm][positions]
            folder = root/"arms"/arm/partition
            folder.mkdir(parents=True)
            predicted, pnull = w@gains[train], w@(gains[train] <= 0)
            # Exact uniform predictions are copied across queries, so no
            # floating summation noise accidentally becomes a GLOBAL ranking.
            if arm == "GLOBAL":
                predicted[:] = gains[train].mean(0)
                pnull[:] = (gains[train] <= 0).mean(0)
            pnull = np.clip(pnull, 0., 1.)
            utility, gamma_crps = _distribution_rows(gains[train], gains[ii], w, ACTIONS)
            geometry, geometry_crps = _distribution_rows(observables[train], observables[ii], w, OBSERVABLE_NAMES)
            energy = weighted_energy_score(standardized_gram[train], standardized_gram[ii], w)
            policy = evaluate_predictions(predicted, pnull, gains[ii], ids[ii], fractions=(.05,.1,.25),
                train_actual=gains[train], seed=SEED, n_bootstrap=2000, n_random=2000)
            summary = dict(n=len(ii), donor_count=len(train), utility=utility, geometry=geometry,
                action_metrics=policy["action_metrics"], joint_geometry_energy=float(energy.mean()),
                response={}, contrast={}, policy=policy, historical_dev=True, formal_certificate=False)
            if arm == "GLOBAL":
                summary["policy_raw_lexical_debug"] = policy
                summary["policy"] = uniform_policy_expectations(policy)
            trace = dict(gamma_crps_add_two=gamma_crps[:,2],
                         null_brier_add_two=(pnull[:,2]-(gains[ii,2]<=0))**2,
                         joint_geometry_energy=energy)
            save = dict(ids=ids[ii], donor_ids=ids[train], weights=w, actual=gains[ii], predicted=predicted,
                        p_null=pnull, gamma_crps=gamma_crps, geometry_crps=geometry_crps, geometry_energy=energy)
            for target_name in ("future_mean_profile", "future_mean_minus_X"):
                target = response[target_name]
                mean = w@target[train]
                global_mean = np.broadcast_to(target[train].mean(0), mean.shape)
                summary["response"][target_name] = {}
                for coordinate, feature_scale in (("original", None), ("train_standardized", scale)):
                    report, error = _mean_error_rows(mean, target[ii], global_mean, feature_scale)
                    summary["response"][target_name][coordinate] = report
                    key = target_name+"__"+coordinate+"_mse"
                    trace[key] = error
                    save[key] = error
            for target_name in ("repeat_contrast_energy", "standardized_repeat_contrast_energy"):
                target = response[target_name]
                mean = w@target[train]
                global_mean = np.broadcast_to(target[train].mean(0), mean.shape)
                report, error = _mean_error_rows(mean, target[ii], global_mean)
                rows, crps = _distribution_rows(target[train], target[ii], w, ("Z1-Z2", "Z1-V", "Z2-V"))
                summary["contrast"][target_name] = {**report, "pairwise_distributions": rows}
                trace[target_name+"_mse"] = error
                trace[target_name+"_crps"] = crps.mean(1)
                save[target_name+"_actual"] = target[ii]
                save[target_name+"_predicted"] = mean
                save[target_name+"_crps"] = crps
            np.savez_compressed(folder/"predictions.npz", **save)
            _write_json(folder/"metrics.json", summary)
            results[partition][arm] = summary
            traces[arm] = trace
        comparisons[partition] = {arm: paired_comparison(traces[arm], traces["GLOBAL"]) for arm in ARMS[1:]}
        print(json.dumps({"state":"PARTITION_COMPLETE", "partition":partition}), flush=True)
    summary = dict(protocol=protocol, weights_frozen_utc=weights_frozen_utc,
        outcome_statistics_started_after_weights=True, input_provenance=provenance,
        partitions=results, paired_neighbor_minus_global=comparisons, score_scale=score_scale,
        completed_utc=datetime.now(timezone.utc).isoformat())
    _write_json(root/"summary.json", summary)
    lines = ["# Fixed-neighbor response and joint-geometry diagnostic", "",
        "The same 383 TRAIN compounds supply every donor distribution. All three historical non-TRAIN DEV partitions are reported; none chooses the neighborhood rule.", "",
        "| Partition | Arm | ADD_TWO Gamma CRPS | NULL Brier | Spearman | Future mean profile MSE | Mean change from X MSE | Repeat-contrast energy MSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for partition in PARTITIONS:
        for arm in ARMS:
            item = results[partition][arm]
            rho = item["action_metrics"][2]["spearman"]
            lines.append(f"| {partition} | {arm} | {item['utility'][2]['crps']:.6f} | "
                f"{item['action_metrics'][2]['null_brier']:.6f} | {rho if rho is not None else 'undefined'} | "
                f"{item['response']['future_mean_profile']['original']['mse']:.5f} | "
                f"{item['response']['future_mean_minus_X']['original']['mse']:.5f} | "
                f"{item['contrast']['repeat_contrast_energy']['mse']:.5f} |")
    lines += ["", "## Interpretation", "",
        "CRPS and energy scores are exact scores of the weighted empirical donor law, not Monte Carlo estimates. One donor contributes the complete joint four-role geometry; future wells are never independently spliced.",
        "Original half-cosine utilities and per-optional-well costs are unchanged. The empirical normalized-Gram model does not supply a fully specified original-spectrum generator or a causal biological interpretation.",
        "Future mean profiles and future-minus-X profiles are separate response diagnostics. Repeat-contrast energies include fixed role/batch differences and are not identified technical-noise components.",
        "The shared feature scaler, joint-score scale and neighborhoods use only TRAIN fitting or decision-time X. Neighbor bandwidths and top20 rules were copied from the previous support audit, and weights were saved before target statistics were computed.",
        "GLOBAL has no individual ranking. Its primary within-action and common-budget tables report exact expectations over uniformly chosen fixed-size subsets; selected_ids is null and false/positive counts can be fractional. Arbitrary lexical-tie outputs are retained only as policy_raw_lexical_debug. Same-budget random references remain available for the neighbor policies.",
        "Paired intervals condition on these fixed rules, objects and shared batches. Multiple diagnostics and historical DEV reuse preclude a new certification or a multiplicity-adjusted discovery claim.",
        "Read absolute_support.json with ESS: far-away queries retain their prescribed weights and are not removed. High top20 ESS does not establish nearby independent biological support.",
        "No structural/biological kernel, new neural model, original contract change, FINAL access or fifth-repeat access was performed.", ""]
    (root/"REPORT.md").write_text("\n".join(lines))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--support-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        result = audit(args.data, args.scaler, args.support_run, args.output)
    print(json.dumps({"state":"COMPLETE", "output":str(Path(args.output).resolve()),
                      "partitions":list(result["partitions"])}))


if __name__ == "__main__":
    main()
