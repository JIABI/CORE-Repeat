"""Full-measurement and budget evaluation for the biology-kernel DEV study.

All spectral scores retain the declared physical coordinates. Predictive
intervals are empirical joint-Monte-Carlo intervals, not conformal certificates.
The historical primary profiles' zero origin is used for descriptive response
direction/amplitude; no new normalization or biological annotation is applied.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .baseline_policy import ACTIONS, evaluate_predictions
from .data import attach_library_context
from .model import EnvironmentNoiseCache
from .representation_probe import regression_metrics
from .training import batches, fixed_batch


LEVELS = (.50, .80, .90, .95)
PAIR_NAMES = ("Z1_minus_Z2", "Z1_minus_V", "Z2_minus_V")


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cosine(a, b):
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(denom <= 0) or not np.isfinite(denom).all():
        raise ValueError("Nonfinite or zero-norm utility profile; no implicit row removal")
    return (a * b).sum(-1) / denom


def utility_draws(initial, future):
    """Net original half-cosine gains, future shape [S,N,3,D]."""
    future, initial = np.asarray(future), np.asarray(initial)
    if future.ndim != 4 or future.shape[2] != 3 or initial.shape != future.shape[1:2] + future.shape[3:]:
        raise ValueError("Utilities require X [N,D] and joint Z1/Z2/V [S,N,3,D]")
    x, z1, z2, v = initial[None], future[:, :, 0], future[:, :, 1], future[:, :, 2]
    before = cosine(x, v)
    return np.stack((.5 * (cosine((x + z1) / 2, v) - before) - .01,
                     .5 * (cosine((x + z2) / 2, v) - before) - .01,
                     .5 * (cosine((x + z1 + z2) / 3, v) - before) - .02), -1)


def actual_gains(y):
    y = np.asarray(y)
    if y.ndim != 3 or y.shape[1] != 4 or not np.isfinite(y).all():
        raise ValueError("Exactly four finite physical roles are required")
    return utility_draws(y[:, 0], y[None, :, 1:])[0]


def interval_stats(draws, target, levels=LEVELS):
    """Coordinate hits/width sums per object; all dimensions retained."""
    draws, target = np.asarray(draws), np.asarray(target)
    if draws.shape[1:] != target.shape or len(draws) < 2:
        raise ValueError("Joint interval samples and observations must align")
    if not np.isfinite(draws).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite predictive samples or observed coordinates")
    if any(not 0 < x < 1 for x in levels):
        raise ValueError("Interval levels must be in (0,1)")
    probabilities = [(1 - x) / 2 for x in levels] + [(1 + x) / 2 for x in levels]
    quantiles = np.quantile(draws, probabilities, axis=0)
    lower, upper = quantiles[:len(levels)], quantiles[len(levels):]
    dims = tuple(range(2, lower.ndim))
    hits = ((target[None] >= lower) & (target[None] <= upper)).sum(axis=dims).T
    width = (upper - lower).sum(axis=dims).T
    return hits, width, int(np.prod(target.shape[1:]))


def _small_metrics(m):
    return {key: m[key] for key in ("overall", "slots")}


@torch.no_grad()
def predict_partition(model, scaler, dataset, indices, config, *, seed,
                      object_chunk=2, levels=LEVELS, progress=None):
    """Measure full-space accuracy, marginal proper NLL and shared-draw utility."""
    ix = np.asarray(indices, int)
    if (ix.ndim != 1 or not len(ix) or len(set(ix.tolist())) != len(ix)
            or np.any(ix < 0) or np.any(ix >= len(dataset))):
        raise ValueError("Unique nonempty in-range evaluation indices required")
    if dataset.Y.shape[1] != 4 or not np.all(dataset.well_mask[ix]) or not np.all(dataset.observed_mask[ix]):
        raise ValueError("This evaluation requires the existing complete four-role cohort")
    y = dataset.Y[ix]
    if not np.isfinite(y).all() or config.samples < 2 or object_chunk < 1:
        raise ValueError("Finite outcomes, at least two joint samples and positive chunk size required")
    normalized = scaler.transform(dataset)
    if config.use_library:
        if not hasattr(model, "library_bank"):
            raise ValueError("Declared library context requires the fitted training-only bank")
        normalized = attach_library_context(normalized, model.library_bank)
    model.eval()
    n, _, d = y.shape
    predicted = np.empty((n, 3, d), np.float64)
    nll, affine_nll = np.empty(n), np.empty(n)
    samples = np.empty((config.samples, n, 3), np.float64)
    coordinate_hits = np.empty((n, len(levels)), int)
    coordinate_widths = np.empty((n, len(levels)))
    difference_hits = np.empty((n, 3, len(levels)), int)
    difference_widths = np.empty((n, 3, len(levels)))
    counts = [min(config.mc_chunk_size, config.samples - start)
              for start in range(0, config.samples, config.mc_chunk_size)]
    caches = [EnvironmentNoiseCache() for _ in counts]
    generators = [torch.Generator(device="cpu").manual_seed(seed + 501 + block * 100003)
                  for block in range(len(counts))]
    log_jacobian = 3 * np.log(np.asarray(scaler.y_scale)).sum()
    for start in range(0, n, object_chunk):
        stop = min(n, start + object_chunk)
        batch, target, mask = fixed_batch(normalized, ix[start:stop], config)
        distribution = model(batch)
        predicted[start:stop] = scaler.inverse_y(distribution.mean).cpu().numpy()
        logp = distribution.log_prob(target, mask).cpu().numpy()
        if logp.shape != (stop - start,) or not np.isfinite(logp).all():
            raise FloatingPointError("Distribution must return finite per-compound marginal log densities")
        affine_nll[start:stop] = -logp / (3 * d)
        nll[start:stop] = (-logp + log_jacobian) / (3 * d)
        chunks = []
        for count, cache, generator in zip(counts, caches, generators):
            draw = distribution.sample_joint(count, generator, environment_noise_cache=cache)
            chunks.append(scaler.inverse_y(draw).cpu().numpy())
        joint = np.concatenate(chunks)
        del chunks
        if not np.isfinite(joint).all() or joint.shape != (config.samples, stop - start, 3, d):
            raise FloatingPointError("Invalid full-coordinate joint predictive draw")
        samples[:, start:stop] = utility_draws(y[start:stop, 0], joint)
        h, w, _ = interval_stats(joint, y[start:stop, 1:], levels)
        coordinate_hits[start:stop], coordinate_widths[start:stop] = h, w
        for pair, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            h, w, _ = interval_stats(joint[:, :, a] - joint[:, :, b],
                                     y[start:stop, 1 + a] - y[start:stop, 1 + b], levels)
            difference_hits[start:stop, pair], difference_widths[start:stop, pair] = h, w
        del joint, distribution, batch, target, mask
        if progress is not None:
            progress(stop, n)
    return dict(prediction_mean=predicted, utility_samples=samples,
        predicted=samples.mean(0), predictive_sd=samples.std(0, ddof=1),
        mc_se=samples.std(0, ddof=1) / np.sqrt(config.samples), p_null=(samples <= 0).mean(0),
        p_positive=(samples >= .005).mean(0), raw_nll=nll, standardized_nll=affine_nll,
        coordinate_interval_hits=coordinate_hits, coordinate_interval_widths=coordinate_widths,
        difference_interval_hits=difference_hits, difference_interval_widths=difference_widths)


def score_measurements(actual, prediction, train_mean, scale):
    """Mean prediction and zero-origin profile-response diagnostics."""
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    train_mean, scale = np.asarray(train_mean), np.asarray(scale)
    standardized = regression_metrics(actual, prediction, train_mean, np.broadcast_to(scale, train_mean.shape))
    raw = regression_metrics(actual, prediction, train_mean)
    centered = regression_metrics(actual, prediction, actual.mean(0), np.broadcast_to(scale, train_mean.shape))
    physical_centered = regression_metrics(actual, prediction, actual.mean(0))
    zero = regression_metrics(actual, prediction, np.zeros_like(train_mean))
    nr, npred = np.linalg.norm(actual, axis=-1), np.linalg.norm(prediction, axis=-1)
    valid = (nr > 0) & (npred > 0)
    directions = np.full(nr.shape, np.nan)
    np.divide((actual * prediction).sum(-1), nr * npred, out=directions, where=valid)
    actual_rms, predicted_rms = nr / np.sqrt(actual.shape[-1]), npred / np.sqrt(actual.shape[-1])
    error = standardized["per_object_sse"]
    report = dict(physical=_small_metrics(raw), standardized=_small_metrics(standardized),
        r2_evaluation_mean=centered["overall"]["r2_training_mean"],
        slot_r2_evaluation_mean=[v["r2_training_mean"] for v in centered["slots"]],
        physical_r2_evaluation_mean=physical_centered["overall"]["r2_training_mean"],
        physical_slot_r2_evaluation_mean=[v["r2_training_mean"] for v in physical_centered["slots"]],
        zero_origin_skill=zero["overall"]["r2_training_mean"],
        median_per_object_standardized_mse=float(np.median(error) / np.prod(actual.shape[1:])),
        max_object_standardized_sse_fraction=float(error.max() / error.sum()) if error.sum() else None,
        profile_direction_mean=float(directions[valid].mean()) if valid.any() else None,
        profile_direction_defined_count=int(valid.sum()), profile_direction_total_count=int(valid.size),
        rms_amplitude_mae=float(np.mean(np.abs(predicted_rms - actual_rms))),
        actual_profile_rms_mean=float(actual_rms.mean()), predicted_mean_profile_rms_mean=float(predicted_rms.mean()),
        response_definition="unchanged dataset coordinate zero; no new centering or biological response labels",
        amplitude_definition="RMS of the predicted conditional mean, not expected RMS of a future sample")
    traces = dict(standardized_sse=error, physical_sse=raw["per_object_sse"],
                  train_mean_baseline_sse=standardized["per_object_baseline_sse"],
                  profile_direction=directions, actual_profile_rms=actual_rms,
                  predicted_mean_profile_rms=predicted_rms)
    return report, traces


def evaluate_partition(model, scaler, dataset, train_indices, indices, config, directory, *,
                       seed, fractions=(.05, .10, .25), object_chunk=2,
                       n_bootstrap=2000, n_random=2000, progress=None):
    """Save auditable per-compound evaluation without changing a trained model."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    ix, train = np.asarray(indices, int), np.asarray(train_indices, int)
    if np.intersect1d(ix, train).size:
        raise ValueError("Evaluation and training compound sets overlap")
    values = predict_partition(model, scaler, dataset, ix, config, seed=seed,
                               object_chunk=object_chunk, progress=progress)
    actual = actual_gains(dataset.Y[ix])
    metrics, traces = score_measurements(dataset.Y[ix, 1:], values["prediction_mean"],
        dataset.Y[train, 1:].mean(0), np.asarray(scaler.y_scale))
    policy = evaluate_predictions(values["predicted"], values["p_null"], actual, dataset.ids[ix],
        fractions=fractions, train_actual=actual_gains(dataset.Y[train]), seed=seed,
        n_bootstrap=n_bootstrap, n_random=n_random)
    d, n = dataset.Y.shape[-1], len(ix)
    calibration = []
    for j, level in enumerate(LEVELS):
        calibration.append(dict(level=level,
            coordinate_coverage=float(values["coordinate_interval_hits"][:, j].sum() / (n * 3 * d)),
            coordinate_mean_width=float(values["coordinate_interval_widths"][:, j].sum() / (n * 3 * d)),
            differences=[dict(pair=name,
                coverage=float(values["difference_interval_hits"][:, k, j].sum() / (n * d)),
                mean_width=float(values["difference_interval_widths"][:, k, j].sum() / (n * d)))
                for k, name in enumerate(PAIR_NAMES)]))
    from .objective_analysis import fair_crps
    utility_crps = fair_crps(values["utility_samples"], actual)
    report = dict(n=n, ids=dataset.ids[ix].tolist(), actions=list(ACTIONS),
        observation_family=getattr(config, "observation_family", "gaussian"),
        biology_kernel_mode=getattr(config, "biology_kernel_mode", "off"),
        measurement=metrics, proper_nll=dict(original_space_per_coordinate=float(values["raw_nll"].mean()),
            standardized_per_coordinate=float(values["standardized_nll"].mean()),
            per_object_max=float(values["raw_nll"].max()), per_object_median=float(np.median(values["raw_nll"])),
            scope="per-compound marginal proper score, not full-campaign joint density",
            checkpoint_score_scope="fixed validation minibatch joint density; different from marginal reporting"),
        predictive_intervals=calibration, interval_method="joint Monte Carlo coordinate quantiles; no posthoc conformal calibration",
        utility_crps_by_action=utility_crps.mean(0).tolist(),
        mean_mc_se=values["mc_se"].mean(0).tolist(), max_mc_se=values["mc_se"].max(0).tolist(),
        samples=config.samples, mc_chunk_size=config.mc_chunk_size, object_chunk=object_chunk,
        shared_environment_draws_preserved_across_object_chunks=True,
        original_endpoint_changed=False, original_contract_changed=False, formal_certificate=False,
        historical_dev=True, final_opened=False, fifth_repeat_opened=False,
        all_objects_retained=True, policy=policy)
    np.savez_compressed(path / "predictions.npz", ids=dataset.ids[ix], actual=actual,
                        utility_crps=utility_crps, **values, **traces)
    fields = ["compound_id", "raw_nll", "standardized_sse", "physical_sse"] + [
        f"{action}__{kind}" for action in ACTIONS for kind in ("actual", "predicted", "p_null", "p_positive", "mc_se")]
    with (path / "predictions.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row, unit in enumerate(dataset.ids[ix]):
            item = dict(compound_id=str(unit), raw_nll=values["raw_nll"][row],
                        standardized_sse=traces["standardized_sse"][row], physical_sse=traces["physical_sse"][row])
            for j, action in enumerate(ACTIONS):
                item[f"{action}__actual"] = actual[row, j]
                for kind in ("predicted", "p_null", "p_positive", "mc_se"):
                    item[f"{action}__{kind}"] = values[kind][row, j]
            writer.writerow(item)
    write_json(path / "metrics.json", report)
    return plain(report)
