"""Full-input Gram ridge with preprocessing refitted inside every CV fit.

The five-fold nested error predictions first return to native nine-dimensional
Gram coordinates. Only then are they expressed in the final fit's common
coordinates for the joint error second moment. No query outcomes are accepted
by prediction. The enclosing caller owns the independent outer experiment fold.
"""
from __future__ import annotations

import numpy as np

from .gram_simple_models import (
    GRAM_DIM, REGULARIZATION_GRID, TIE_ATOL, TIE_RTOL,
    GramSimpleGaussian, _CenteredRidgePath, _fit_error_second_moment,
    _matrix, _membership, _seed,
)


SCALE_FLOOR = 1e-6


def _measurements(y):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 3 or not len(y) or y.shape[1] != 4 or y.shape[2] < 1:
        raise ValueError("Fitting measurements must be nonempty [N,4,D]")
    if not np.isfinite(y).all():
        raise ValueError("Fitting measurements must be finite")
    return y


def _lognorm(x):
    norm = np.linalg.norm(x, axis=-1)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise ValueError("Every observed X must have a finite positive norm")
    return np.log(norm)


def _moments(values):
    center = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale = np.where(scale >= SCALE_FLOOR, scale, 1.)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Preprocessing moments overflowed")
    return center, scale


def fit_preprocessing(y, raw_u):
    """Fit the six affine quantities solely on supplied fitting compounds.

    Y center/scale use all four roles, exactly TrainScaler's population-moment
    convention. Input X and its log norm are available at decision time; the
    future fitting roles are used only to estimate training preprocessing.
    """
    y = _measurements(y)
    u = _matrix(raw_u, "Native fitting u", columns=GRAM_DIM)
    if len(y) != len(u):
        raise ValueError("Y and native u must have the same ordered compounds")
    yc, ys = _moments(y.reshape(-1, y.shape[-1]))
    uc, us = _moments(u)
    nc, ns = _moments(_lognorm(y[:, 0]))
    return dict(y_center=yc.tolist(), y_scale=ys.tolist(),
                u_center=uc.tolist(), u_scale=us.tolist(),
                lognorm_center=float(nc), lognorm_scale=float(ns))


def _stats_vectors(stats, prefix, columns):
    center = np.asarray(stats[prefix+"_center"], dtype=np.float64)
    scale = np.asarray(stats[prefix+"_scale"], dtype=np.float64)
    if (center.shape != (columns,) or scale.shape != (columns,)
            or not np.isfinite(center).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("Invalid "+prefix+" preprocessing parameters")
    return center, scale


def transform_input(raw_x, stats):
    """Return complete [N,D+1] X/log-norm inputs at G's input precision."""
    x = _matrix(raw_x, "Decision-time X")
    center, scale = _stats_vectors(stats, "y", x.shape[1])
    nc, ns = float(stats["lognorm_center"]), float(stats["lognorm_scale"])
    if not np.isfinite(nc) or not np.isfinite(ns) or ns <= 0:
        raise ValueError("Invalid log-norm preprocessing parameters")
    affine = (x-center)/scale
    norm = (_lognorm(x)-nc)/ns
    # Both inputs are cast independently by the neural G caller. Retaining that
    # precision makes differences here about the estimator, not input rounding.
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.column_stack((affine.astype(np.float32).astype(np.float64),
                               norm.astype(np.float32).astype(np.float64)))
    if not np.isfinite(out).all():
        raise ValueError("Decision-time inputs overflowed float32; no clipping applied")
    return out


def transform_target(raw_u, stats):
    """Convert native u into one explicitly supplied fitting coordinate frame."""
    u = _matrix(raw_u, "Native target u", columns=GRAM_DIM)
    center, scale = _stats_vectors(stats, "u", GRAM_DIM)
    out = (u-center)/scale
    if not np.isfinite(out).all():
        raise ValueError("Target transformation overflowed")
    return out


def _restore_target(u, stats):
    center, scale = _stats_vectors(stats, "u", GRAM_DIM)
    result = np.asarray(u, dtype=np.float64)*scale+center
    if not np.isfinite(result).all():
        raise ValueError("Native target reconstruction overflowed")
    return result


def _retain_stats(audit, prefix, stats):
    for key, value in stats.items():
        audit[prefix+"__"+key] = np.asarray(value, dtype=np.float64)
    return prefix


def _select(errors, membership):
    """Pool per-object MSE computed in each CV fit's own target coordinates."""
    records = []
    for j, penalty in enumerate(REGULARIZATION_GRID):
        fold_mse = [float(errors[j, membership == fold].mean())
                    for fold in range(int(membership.max())+1)]
        records.append(dict(lambda_value=float(penalty),
                            mean_mse=float(errors[j].mean()), fold_mse=fold_mse))
    minimum = min(record["mean_mse"] for record in records)
    tied = [record for record in records if np.isclose(
        record["mean_mse"], minimum, rtol=TIE_RTOL, atol=TIE_ATOL)]
    selected = max(record["lambda_value"] for record in tied)
    return selected, dict(candidate_records=records, selected_lambda=selected,
        criterion="object-weighted MSE in each validation fold's fit-only standardized u",
        tie_rule="larger lambda among numerical ties", tie_rtol=TIE_RTOL, tie_atol=TIE_ATOL)


def fit_ridge_oof(y_fit, raw_u_fit, seed=20260914):
    """Fit complete ridge and full covariance with five-outer/four-inner CV.

    Each candidate is fitted with alpha=n_fit*lambda. Every outer error
    prediction excludes that object's complete four-role Y and u when fitting
    preprocessing, choosing lambda, and fitting the mean. Final coordinates and
    covariance intentionally use all supplied training objects, never queries.
    """
    y = _measurements(y_fit)
    raw_u = _matrix(raw_u_fit, "Native fitting u", columns=GRAM_DIM, minimum_rows=5)
    if len(y) != len(raw_u):
        raise ValueError("Y and native u must have the same ordered compounds")
    seed = _seed(seed)
    final_stats = fit_preprocessing(y, raw_u)
    x, u = transform_input(y[:, 0], final_stats), transform_target(raw_u, final_stats)
    splits, allocation = _membership(len(y), 5, seed)
    full_errors = np.empty((len(REGULARIZATION_GRID), len(y)), dtype=np.float64)
    full_native = np.empty((len(REGULARIZATION_GRID), len(y), GRAM_DIM), dtype=np.float64)
    native_oof = np.empty_like(raw_u)
    counts = np.zeros(len(y), dtype=np.int64)
    inner_allocation = np.full((5, len(y)), -1, dtype=np.int64)
    outer_records, selected_outer, audit = [], [], {}
    _retain_stats(audit, "final_preprocessing", final_stats)

    for outer, (fit, check) in enumerate(splits):
        stats = fit_preprocessing(y[fit], raw_u[fit])
        path = _CenteredRidgePath(transform_input(y[fit, 0], stats),
                                  transform_target(raw_u[fit], stats))
        candidate = path.predict_path(transform_input(y[check, 0], stats))
        check_u = transform_target(raw_u[check], stats)
        full_errors[:, check] = np.square(candidate-check_u[None]).mean(-1)
        full_native[:, check] = _restore_target(candidate, stats)
        outer_prefix = _retain_stats(audit, f"outer_{outer}_preprocessing", stats)
        inner_seed = (seed+1009*(outer+1)) % (2**32-1)
        inner_splits, inner_local = _membership(len(fit), 4, inner_seed)
        inner_allocation[outer, fit] = inner_local
        inner_errors = np.empty((len(REGULARIZATION_GRID), len(fit)), dtype=np.float64)
        inner_records = []
        for inner, (local_fit, local_check) in enumerate(inner_splits):
            ii, jj = fit[local_fit], fit[local_check]
            inner_stats = fit_preprocessing(y[ii], raw_u[ii])
            inner_path = _CenteredRidgePath(transform_input(y[ii, 0], inner_stats),
                                            transform_target(raw_u[ii], inner_stats))
            predictions = inner_path.predict_path(transform_input(y[jj, 0], inner_stats))
            target = transform_target(raw_u[jj], inner_stats)
            inner_errors[:, local_check] = np.square(predictions-target[None]).mean(-1)
            prefix = _retain_stats(audit, f"outer_{outer}_inner_{inner}_preprocessing", inner_stats)
            inner_records.append(dict(fold=inner, fit_indices=ii.tolist(),
                validation_indices=jj.tolist(), n_fit=len(ii), preprocessing_audit_prefix=prefix,
                alpha_values=[float(len(ii)*penalty) for penalty in REGULARIZATION_GRID]))
        chosen_lambda, inner_summary = _select(inner_errors, inner_local)
        chosen = REGULARIZATION_GRID.index(chosen_lambda)
        native_oof[check] = full_native[chosen, check]
        counts[check] += 1
        selected_outer.append(chosen_lambda)
        audit[f"outer_{outer}_inner_candidate_per_object_mse"] = inner_errors
        outer_records.append(dict(outer_fold=outer, outer_fit_indices=fit.tolist(),
            outer_validation_indices=check.tolist(), n_fit=len(fit), selected_lambda=chosen_lambda,
            alpha=float(len(fit)*chosen_lambda), inner_seed=inner_seed,
            preprocessing_audit_prefix=outer_prefix, inner_cv=inner_summary, inner_folds=inner_records,
            outer_fit_coordinate_mse=float(full_errors[chosen, check].mean())))

    if not np.array_equal(counts, np.ones(len(y), dtype=np.int64)):
        raise RuntimeError("Every fitting compound must have exactly one nested OOF prediction")
    if not np.isfinite(full_errors).all():
        raise ValueError("Cross-validation errors overflowed")
    selected, final_cv = _select(full_errors, allocation)
    final_path = _CenteredRidgePath(x, u)
    coefficient, intercept = final_path.coefficients(selected)
    oof = transform_target(native_oof, final_stats)
    residuals = u-oof
    covariance, centered_covariance, bias, covariance_audit = _fit_error_second_moment(
        residuals, include_bias=True)
    audit.update(training_x_mean=final_path.x_mean, training_u_mean=final_path.u_mean,
        full_cv_native_predictions_by_lambda=full_native,
        full_cv_candidate_per_object_mse=full_errors, full_cv_fold_membership=allocation.copy(),
        oof_native_predictions=native_oof, oof_predictions=oof, oof_residuals=residuals,
        residuals_for_covariance=residuals.copy(), residual_mean=bias,
        centered_residual_covariance=centered_covariance, oof_count=counts,
        outer_fold_membership=allocation, inner_fold_membership=inner_allocation,
        outer_selected_lambdas=np.asarray(selected_outer))
    metadata = dict(method="full-preprocessing nested-CV ridge Gram Gaussian",
        training_objects=len(y), feature_dimension=x.shape[1], target_coordinate_count=GRAM_DIM,
        target_space="final fit-only standardized native Gram u", seed=seed,
        regularization_grid=list(REGULARIZATION_GRID), final_alpha=float(len(y)*selected),
        penalty_convention="per-output average squared error + lambda*||B||²; sklearn alpha=n_fit*lambda",
        full_train_cv=final_cv, outer_cv=outer_records, outer_folds=5, inner_folds=4,
        kernel_eigendecompositions=26, preprocessing_fits=26,
        fold_preprocessing="all four-role Y moments, u moments and X lognorm moments refitted within every fitting fold",
        input_precision="each affine X coordinate and normalized lognorm float32 then float64, matching G input",
        oof_coordinate_alignment="each fold prediction inverse-transformed to native u, then standardized by final fit-only u moments",
        covariance_estimation=covariance_audit, covariance_residuals_out_of_fold=True,
        covariance_estimand="LedoitWolf(centered nested-OOF errors) + residual_mean outer product",
        covariance_is_full_joint=True, prediction_bias_correction=False,
        oof_mean_mse=float(np.square(residuals).mean()),
        final_fit_mean_mse=float(np.square(u-(x@coefficient+intercept)).mean()),
        monte_carlo_object_coupling="independent residual draws across query objects",
        monte_carlo_coupling_is_physical_dependence=False,
        physical_shared_independent_noise_identified=False, campaign_joint_dependence_identified=False,
        query_targets_accepted=False, formal_certificate=False,
        oof_scope="nested development error estimate; overlapping training sets and shared experimental batches remain")
    model = GramSimpleGaussian("RIDGE", intercept, covariance, coefficient=coefficient,
                               selected_lambda=selected, metadata=metadata, audit_arrays=audit)
    return model, final_stats
