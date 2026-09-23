"""Read-only Gaussian replicate diagnostics in the full measurement space.

These summaries do not fit a model or calibrate a threshold. Coverage is the
descriptive fraction of retained measurement coordinates inside a marginal
interval; correlated coordinates are not treated as independent trials.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import ndtri
import torch

from .model import JointGaussian


def gaussian_contrast_moments(
    distribution: JointGaussian,
    weights: np.ndarray | torch.Tensor,
    scaler_center: np.ndarray,
    scaler_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical mean/variance [N,C,D] of fixed contrasts [C,T].

    ``distribution`` is expressed in the model's affine-scaled measurement
    coordinates. The same feature scaler applies to each target well. A sum
    therefore retains ``sum(weights) * center``; a difference has no center.
    Factors are combined before squaring, preserving signed cross-well
    covariances, including environmental factors in the compound marginal.
    This returns coordinate marginal variances, not the full D-by-D covariance.
    """
    if not isinstance(distribution, JointGaussian):
        raise TypeError("Exact Gaussian contrasts require JointGaussian")
    if distribution.mean.ndim != 3 or min(distribution.mean.shape) < 1:
        raise ValueError("distribution.mean must be nonempty [N,T,D]")
    n, t, d = distribution.mean.shape
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    w = np.asarray(weights, dtype=np.float64)
    center = np.asarray(scaler_center, dtype=np.float64)
    scale = np.asarray(scaler_scale, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != t or w.shape[0] < 1:
        raise ValueError("weights must have nonempty shape [C,T]")
    if not np.isfinite(w).all() or np.any(np.all(w == 0, axis=1)):
        raise ValueError("weights must be finite and each contrast nonzero")
    if center.shape != (d,) or scale.shape != (d,):
        raise ValueError("scaler_center and scaler_scale must have shape [D]")
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Scaler center must be finite and scale finite positive")
    means, variances = [], []
    with torch.no_grad():
        for row in w:
            tw = torch.as_tensor(row, device=distribution.mean.device,
                                 dtype=distribution.mean.dtype).unsqueeze(0).expand(n, t)
            mean, diagonal, factor = distribution.weighted_moments(tw)
            # Float64 accumulation avoids avoidable precision loss when many
            # factor columns contribute. No covariance terms are discarded.
            scaled_variance = diagonal.double() + factor.double().square().sum(-1)
            mean_np = mean.detach().cpu().double().numpy()
            variance_np = scaled_variance.detach().cpu().numpy()
            means.append(mean_np * scale + row.sum() * center)
            variances.append(variance_np * np.square(scale))
    physical_mean = np.stack(means, axis=1)
    physical_variance = np.stack(variances, axis=1)
    if (not np.isfinite(physical_mean).all() or not np.isfinite(physical_variance).all()
            or np.any(physical_variance <= 0)):
        raise ValueError("Contrast means must be finite and variances finite positive")
    return physical_mean, physical_variance


def summarize_contrasts(
    actual: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    levels: Sequence[float] = (.5, .8, .9, .95),
) -> tuple[dict, dict[str, np.ndarray]]:
    """Summarize exact central Gaussian intervals of [N,C,D] contrasts.

    Return ``(summary, per_object)``. Summary values are JSON-compatible.
    Per-object coverage, width, score and tail arrays have shape [N,C,L];
    other per-object arrays have shape [N,C]. Object IDs and contrast names
    deliberately remain the caller's responsibility, preserving input order.

    RMS residual / RMS predicted SD is a scale diagnostic, NOT coverage. It
    includes mean bias, so a discrepancy is not itself a variance diagnosis.
    No interval or standard error assumes independent features or compounds.
    """
    actual = np.asarray(actual, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if actual.ndim != 3 or min(actual.shape) < 1:
        raise ValueError("actual must have nonempty shape [N,C,D]")
    if mean.shape != actual.shape or variance.shape != actual.shape:
        raise ValueError("actual, mean and variance must share shape [N,C,D]")
    if not all(np.isfinite(x).all() for x in (actual, mean, variance)):
        raise ValueError("All contrast inputs must be finite")
    if np.any(variance <= 0):
        raise ValueError("All contrast variances must be positive")
    level = np.asarray(tuple(levels), dtype=np.float64)
    if (level.ndim != 1 or level.size < 1 or not np.isfinite(level).all()
            or np.any((level <= 0) | (level >= 1))
            or np.unique(level).size != level.size):
        raise ValueError("levels must be distinct finite probabilities in (0,1)")
    residual = actual - mean
    sd = np.sqrt(variance)
    z = residual / sd
    quantiles = ndtri((1 + level) / 2)
    residual_sse = np.square(residual).sum(axis=-1)
    predicted_variance_sum = variance.sum(axis=-1)
    n, c, d = actual.shape
    per_object = {
        "residual_sse": residual_sse,
        "predicted_variance_sum": predicted_variance_sum,
        "residual_mean": residual.mean(axis=-1),
        "residual_rms": np.sqrt(residual_sse / d),
        "predicted_rms_sd": np.sqrt(predicted_variance_sum / d),
        "mean_predicted_sd": sd.mean(axis=-1),
        "residual_rms_to_predicted_rms_sd": np.sqrt(residual_sse / predicted_variance_sum),
        "z_mean": z.mean(axis=-1),
        "z_second_moment": np.square(z).mean(axis=-1),
    }
    interval_arrays = {key: [] for key in
                       ("coverage", "width", "interval_score", "lower_tail", "upper_tail")}
    for p, q in zip(level, quantiles):
        half_width = q * sd
        below = z < -q
        above = z > q
        interval_arrays["coverage"].append((~(below | above)).mean(axis=-1))
        interval_arrays["width"].append((2 * half_width).mean(axis=-1))
        interval_arrays["interval_score"].append(
            (2 * half_width + 2 / (1 - p)
             * np.maximum(np.abs(residual) - half_width, 0)).mean(axis=-1))
        interval_arrays["lower_tail"].append(below.mean(axis=-1))
        interval_arrays["upper_tail"].append(above.mean(axis=-1))
    per_object.update({key: np.stack(value, axis=-1) for key, value in interval_arrays.items()})
    contrasts = []
    for index in range(c):
        rmse = float(np.sqrt(residual_sse[:, index].sum() / (n * d)))
        predicted_rms = float(np.sqrt(predicted_variance_sum[:, index].sum() / (n * d)))
        row = {
            "contrast_index": index,
            "coordinate_mse": rmse ** 2,
            "residual_mean": float(per_object["residual_mean"][:, index].mean()),
            "rms_actual_residual": rmse,
            "rms_predicted_sd": predicted_rms,
            "residual_rms_to_predicted_rms_sd": rmse / predicted_rms,
            "mean_predicted_sd": float(per_object["mean_predicted_sd"][:, index].mean()),
            "standardized_residual_mean": float(per_object["z_mean"][:, index].mean()),
            "standardized_residual_second_moment": float(
                per_object["z_second_moment"][:, index].mean()),
        }
        for key in interval_arrays:
            row[key] = per_object[key][:, index].mean(axis=0).tolist()
        contrasts.append(row)
    summary = {
        "n_objects": n,
        "n_contrasts": c,
        "n_coordinates_per_contrast": d,
        "levels": level.tolist(),
        "contrasts": contrasts,
        "definitions": {
            "coverage": "Fraction of physical measurement coordinates in central Gaussian intervals; descriptive, not independent trials.",
            "width": "Arithmetic mean physical interval width over objects and retained coordinates.",
            "interval_score": "width + 2/(1-level) times distance outside the interval; lower is better, physical units.",
            "residual_rms_to_predicted_rms_sd": "sqrt(mean((actual-mean)^2)/mean(variance)); scale ratio, NOT coverage; includes mean bias.",
            "standardized_residual_second_moment": "mean((actual-mean)^2/variance); includes mean bias, not centered variance.",
            "aggregation": "Equal coordinate counts per object; all retained coordinates and all supplied objects, no clipping or deletion.",
        },
    }
    return summary, per_object
