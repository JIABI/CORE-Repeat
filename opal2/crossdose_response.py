"""Direct and change-transport response borrowing for a declared dose pair.

This module predicts a target-condition profile, not Gamma or uncertainty and
not the frozen CORE geometry. The caller owns chemical-group partitioning and
legal reference construction. Only TRAIN fits RIDGE_RESPONSE coefficients and
input normalization; VALID chooses its penalty; CAL chooses the bounded
borrowing strength. Query outcomes are accepted only by the scoring function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RIDGE_LAMBDAS = (.001, .01, .1, 1., 10., 100.)
EPSILON = 1e-12


def _matrix(value, name, *, minimum_rows=0):
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or result.shape[1] < 1 or len(result) < minimum_rows
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite matrix with at least {minimum_rows} rows")
    return result


def _support(value, n):
    result = np.asarray(value)
    if result.dtype != bool or result.shape != (n,):
        raise ValueError("Support must be an aligned boolean vector")
    return result


def _groups(value, n):
    result = np.asarray(value)
    if result.shape != (n,):
        raise ValueError("Groups must align with rows")
    if result.dtype.kind in "fc" and not np.isfinite(result).all():
        raise ValueError("Groups cannot contain missing identities")
    result = result.astype(str)
    if any(not x.strip() or x.strip().lower() in {"none", "nan", "<na>"} for x in result):
        raise ValueError("Groups require nonempty known identities")
    return result


def _group_mean(values, groups):
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    return unique, sums / counts if len(unique) else np.empty(0)


def response_candidates(query_source, reference_source, reference_target, weights, support):
    """Return profile candidates from known source and legal reference pairs.

    DIRECT = W @ Y_ref_target.
    TRANSPORT = X_query_source + W @ (Y_ref_target - X_ref_source).
    Supported weight rows must be nonnegative and sum to one. Unsupported rows
    must have exactly zero weights and return zero *candidate placeholders*;
    use apply_response_correction to preserve their baseline predictions.
    """
    xq = _matrix(query_source, "query_source")
    xr = _matrix(reference_source, "reference_source")
    yr = _matrix(reference_target, "reference_target")
    w = np.asarray(weights, dtype=np.float64)
    mask = _support(support, len(xq))
    if xr.shape != yr.shape or xr.shape[1] != xq.shape[1]:
        raise ValueError("Source and target reference profiles must share the query coordinate space")
    if w.shape != (len(xq), len(xr)) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Reference weights must be finite, nonnegative and aligned")
    if np.any(w[~mask] != 0):
        raise ValueError("Unsupported weight rows must be exactly zero")
    if not np.allclose(w[mask].sum(1), 1., rtol=1e-10, atol=1e-12):
        raise ValueError("Every supported reference-weight row must sum to one")
    direct, transport = np.zeros_like(xq), np.zeros_like(xq)
    if mask.any():
        direct[mask] = w[mask] @ yr
        transport[mask] = xq[mask] + w[mask] @ (yr - xr)
    if not np.isfinite(direct).all() or not np.isfinite(transport).all():
        raise FloatingPointError("Reference response calculation overflowed")
    return {"DIRECT": direct, "TRANSPORT": transport}


def apply_response_correction(baseline, candidate, support, strength):
    """Convex interpolation, not baseline plus an uncentered transport profile."""
    base = _matrix(baseline, "baseline")
    other = _matrix(candidate, "candidate")
    mask = _support(support, len(base))
    alpha = float(strength)
    if other.shape != base.shape or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Aligned candidate profiles and a strength in [0, 1] are required")
    result = base.copy()
    if alpha and mask.any():
        result[mask] = base[mask] + alpha * (other[mask] - base[mask])
    if not np.isfinite(result).all():
        raise FloatingPointError("Response interpolation overflowed")
    return result


def fit_convex_strength(baseline, target, candidate, groups, support, *, one_se=True,
                        min_groups=3):
    """CAL-only, group-equal quadratic-risk optimum with optional zero fallback.

    The continuous optimum is -mean_g<error,diff>/mean_g||diff||^2,
    clipped to [0,1]. The optional one-SE admission requires its paired
    supported-group MSE improvement to exceed one standard error. This is a
    development selection rule, not a selective or finite-sample guarantee.
    Unsupported groups do not dilute either fitting or the admission check.
    """
    base, y = _matrix(baseline, "baseline"), _matrix(target, "target")
    other = _matrix(candidate, "candidate")
    mask = _support(support, len(base))
    labels = _groups(groups, len(base))
    if y.shape != base.shape or other.shape != base.shape:
        raise ValueError("CAL baseline, target and candidate profiles must align")
    if not isinstance(min_groups, int) or min_groups < 3:
        raise ValueError("At least three supported CAL groups are required")
    unique = np.unique(labels[mask])
    report = dict(alpha=0., raw_alpha=0., clipped_alpha=0., n_groups=len(unique),
                  n_supported_rows=int(mask.sum()), n_supplied_rows=len(base),
                  paired_delta=0., se=0., one_se=bool(one_se), min_groups=min_groups,
                  criterion="CAL supported-group-equal profile MSE",
                  interpretation="development strength selection; not a statistical guarantee")
    if len(unique) < min_groups:
        return dict(report, reason="insufficient supported CAL groups", se=None)
    error, diff = base[mask] - y[mask], other[mask] - base[mask]
    _, cross = _group_mean(np.mean(error * diff, axis=1), labels[mask])
    _, quadratic = _group_mean(np.mean(diff * diff, axis=1), labels[mask])
    _, base_risk = _group_mean(np.mean(error * error, axis=1), labels[mask])
    denominator, numerator = float(quadratic.mean()), float(cross.mean())
    report.update(baseline_mse=float(base_risk.mean()), quadratic=denominator,
                  cross_term=numerator)
    if denominator <= 0:
        return dict(report, reason="candidate equals baseline on supported CAL rows",
                    selected_mse=float(base_risk.mean()), candidate_mse=float(base_risk.mean()))
    raw = -numerator / denominator
    clipped = float(np.clip(raw, 0., 1.))
    differences = 2 * clipped * cross + clipped * clipped * quadratic
    delta = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    admitted = clipped > 0 and (not one_se or delta < -se)
    alpha = clipped if admitted else 0.
    reason = ("admitted by paired one-SE rule" if admitted and one_se else
              "bounded CAL risk optimum" if admitted else
              "zero bounded optimum" if clipped == 0 else "paired improvement below one SE")
    return dict(report, alpha=alpha, raw_alpha=float(raw), clipped_alpha=clipped,
                paired_delta=delta, se=se, candidate_mse=float(base_risk.mean() + delta),
                selected_mse=float(base_risk.mean() + (delta if admitted else 0.)),
                reason=reason)


def profile_scores(prediction, target, *, epsilon=EPSILON):
    """Per-row errors in the fixed control-normalized raw profile space.

    Norms <= epsilon use epsilon for log norms. Cosine is defined as zero when
    either norm <= epsilon (loss one, including both-zero); these rows are
    flagged, never silently removed. The convention must accompany reporting.
    """
    pred, y = _matrix(prediction, "prediction"), _matrix(target, "target")
    if pred.shape != y.shape or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Aligned profiles and a positive finite epsilon are required")
    pred_norm, target_norm = np.linalg.norm(pred, axis=1), np.linalg.norm(y, axis=1)
    pred_zero, target_zero = pred_norm <= epsilon, target_norm <= epsilon
    cosine = np.zeros(len(pred))
    valid = ~(pred_zero | target_zero)
    if valid.any():
        cosine[valid] = np.sum((pred[valid] / pred_norm[valid, None]) *
                               (y[valid] / target_norm[valid, None]), axis=1)
    return dict(profile_mse=np.mean((pred - y) ** 2, axis=1),
                cosine_loss=1. - np.clip(cosine, -1., 1.),
                lognorm_squared_error=(np.log(np.maximum(pred_norm, epsilon)) -
                                       np.log(np.maximum(target_norm, epsilon))) ** 2,
                prediction_zero_norm=pred_zero, target_zero_norm=target_zero,
                metadata=dict(epsilon=float(epsilon),
                              cosine_zero_convention="cosine=0 (loss=1) if either norm<=epsilon",
                              lognorm_convention="log(max(norm,epsilon))",
                              coordinate_space="fixed control-normalized raw response profiles"))


@dataclass
class RidgeResponseModel:
    """TRAIN-fitted multioutput raw-response mean, distinct from CORE."""
    input_center: np.ndarray
    input_scale: np.ndarray
    target_center: np.ndarray
    coefficient: np.ndarray
    selected_lambda: float
    report: dict

    def predict(self, inputs):
        x = _matrix(inputs, "inputs")
        if x.shape[1] != len(self.input_center):
            raise ValueError("Response RIDGE input dimension differs")
        result = ((x - self.input_center) / self.input_scale) @ self.coefficient + self.target_center
        if not np.isfinite(result).all():
            raise FloatingPointError("Response RIDGE prediction overflowed")
        return result


def fit_ridge_response(train_x, train_y, valid_x, valid_y, *,
                       lambdas=RIDGE_LAMBDAS, training_groups=None, validation_groups=None):
    """One smaller primal/dual eigendecomposition; VALID selects n*lambda.

    No coefficient refit on VALID is performed. Input centering/scaling and the
    unpenalized target intercept use TRAIN only. Raw target coordinates are not
    standardized. Supplying training_groups makes normalization and coefficient
    fitting group equal; row weights sum to n_train to retain n_train*lambda.
    Supplying validation_groups makes penalty selection group equal.
    """
    tx, ty = _matrix(train_x, "train_x", minimum_rows=2), _matrix(train_y, "train_y", minimum_rows=2)
    vx, vy = _matrix(valid_x, "valid_x", minimum_rows=1), _matrix(valid_y, "valid_y", minimum_rows=1)
    if len(tx) != len(ty) or len(vx) != len(vy) or tx.shape[1] != vx.shape[1] or ty.shape[1] != vy.shape[1]:
        raise ValueError("TRAIN and VALID input/target dimensions must align")
    penalties = np.asarray(lambdas, dtype=float)
    if penalties.ndim != 1 or not len(penalties) or not np.isfinite(penalties).all() or np.any(penalties <= 0):
        raise ValueError("RIDGE penalties must be finite positive values")
    if len(np.unique(penalties)) != len(penalties):
        raise ValueError("RIDGE penalty grid cannot contain duplicates")
    n, p = tx.shape
    train_labels = (_groups(training_groups, n) if training_groups is not None
                    else np.asarray([str(i) for i in range(n)]))
    train_unique, inverse, counts = np.unique(train_labels, return_inverse=True, return_counts=True)
    row_weights = n / (len(train_unique) * counts[inverse])
    center = np.average(tx, axis=0, weights=row_weights)
    scale = np.sqrt(np.average((tx - center) ** 2, axis=0, weights=row_weights))
    scale = np.where(scale > 1e-8, scale, 1.)
    ym = np.average(ty, axis=0, weights=row_weights)
    v = (vx - center) / scale
    sqrt_weights = np.sqrt(row_weights)[:, None]
    x = ((tx - center) / scale) * sqrt_weights
    y = (ty - ym) * sqrt_weights
    dual = n <= p
    gram = x @ x.T if dual else x.T @ x
    eigenvalues, vectors = np.linalg.eigh((gram + gram.T) * .5)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    tolerance = 64 * np.finfo(float).eps * max(x.shape) * spectral_scale
    if eigenvalues[0] < -tolerance:
        raise FloatingPointError("Computed response RIDGE Gram is materially indefinite")
    eigenvalues = np.maximum(eigenvalues, 0.)
    if dual:
        rotated_target = vectors.T @ y
        projected_valid = (v @ x.T) @ vectors
    else:
        rotated_target = vectors.T @ (x.T @ y)
        projected_valid = v @ vectors
    labels = (_groups(validation_groups, len(vx)) if validation_groups is not None
              else np.asarray([str(i) for i in range(len(vx))]))
    records = []
    for penalty in penalties:
        prediction = projected_valid @ (rotated_target / (eigenvalues + n * penalty)[:, None]) + ym
        per_row = np.mean((prediction - vy) ** 2, axis=1)
        _, grouped = _group_mean(per_row, labels)
        if not np.isfinite(grouped).all():
            raise FloatingPointError("Response RIDGE validation risk overflowed")
        records.append(dict(lambda_value=float(penalty), penalty_n_lambda=float(n * penalty),
                            validation_mse=float(grouped.mean())))
    minimum = min(row["validation_mse"] for row in records)
    selected = max(row["lambda_value"] for row in records
                   if np.isclose(row["validation_mse"], minimum, rtol=1e-10, atol=1e-12))
    rotated = rotated_target / (eigenvalues + n * selected)[:, None]
    coefficient = x.T @ (vectors @ rotated) if dual else vectors @ rotated
    report = dict(model="RIDGE_RESPONSE", n_train=n, n_validation=len(vx),
                  n_train_groups=len(train_unique), n_validation_groups=len(np.unique(labels)),
                  input_dimension=p, target_dimension=ty.shape[1],
                  selected_lambda=selected, candidate_records=records,
                  preprocessing_fit="TRAIN only", coefficients_fit="TRAIN only",
                  training_weighting="group equal" if training_groups is not None else "row equal",
                  standardization_weighting="group equal" if training_groups is not None else "row equal",
                  selection="VALID group-equal profile MSE" if validation_groups is not None else "VALID row-equal profile MSE",
                  target_scaling="none; raw control-normalized target coordinates",
                  eigensystem="dual" if dual else "primal", penalty_convention="n_train * lambda",
                  tie_break="largest lambda within rtol=1e-10, atol=1e-12")
    return RidgeResponseModel(center, scale, ym, coefficient, selected, report)
