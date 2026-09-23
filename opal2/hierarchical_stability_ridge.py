"""Full-input ridge selected using the residual model's external validation set.

This is a separate comparator to training-only nested-CV ridge. Validation
labels choose the penalty, never coefficients or preprocessing moments. Five
internal held-out error predictions refit preprocessing and choose their own
penalty on that same external validation set. Their aligned joint error second
moment estimates uncertainty. The caller must provide disjoint fitting,
validation and eventual query identities; this array API cannot verify IDs.
"""
from __future__ import annotations

import numpy as np

from .gram_oof_ridge import (
    _measurements, _restore_target, _retain_stats,
    fit_preprocessing, transform_input, transform_target,
)
from .gram_simple_models import (
    GRAM_DIM, REGULARIZATION_GRID, TIE_ATOL, TIE_RTOL,
    GramSimpleGaussian, _CenteredRidgePath, _fit_error_second_moment,
    _matrix, _membership, _seed,
)


def _validation_selection(path, y_valid, raw_u_valid, stats):
    """Compare the complete fixed grid in this fit's standardized u frame."""
    inputs = transform_input(y_valid[:, 0], stats)
    target = transform_target(raw_u_valid, stats)
    predictions = path.predict_path(inputs)
    errors = np.square(predictions - target[None]).mean(-1)
    if not np.isfinite(errors).all():
        raise ValueError("External validation errors overflowed")
    records = [dict(lambda_value=float(penalty), mean_mse=float(errors[j].mean()))
               for j, penalty in enumerate(REGULARIZATION_GRID)]
    minimum = min(record["mean_mse"] for record in records)
    selected = max(record["lambda_value"] for record in records if np.isclose(
        record["mean_mse"], minimum, rtol=TIE_RTOL, atol=TIE_ATOL))
    summary = dict(selected_lambda=selected, candidate_records=records,
        criterion="external validation object-weighted mean squared error over fit-standardized nine u coordinates",
        tie_rule="larger lambda among numerical ties", tie_rtol=TIE_RTOL, tie_atol=TIE_ATOL)
    return selected, errors, predictions, summary


def fit_validation_ridge(y_fit, raw_u_fit, y_valid, raw_u_valid, seed=20260914, *, groups=None):
    """Return a full joint ``GramSimpleGaussian`` and fit-only transforms.

    Measurements are [N,4,D] and native targets [N,9] in each supplied partition.
    The external validation set is the same source of labels used for HR
    checkpoint selection. Only its observed X and provided target u participate
    in this function; its other three measurement roles do not enter any fit.
    Future query measurements/labels are not accepted by the returned model.

    Each internal error holdout is excluded from preprocessing, coefficient
    fitting and penalty selection. External validation labels are deliberately
    reused across these five fits. OOF errors therefore describe development
    performance conditional on this selection design, not an independent
    calibration certificate or a physical noise decomposition.

    Optional ``groups`` are fitting-row chemical identities. When supplied,
    five shuffled folds partition unique groups before mapping back to rows;
    every member of a chemical group receives the same held-out error fold.
    The external validation partition must still be group-disjoint, which the
    caller checks. Omitting groups preserves the original row-fold procedure.
    """
    y = _measurements(y_fit)
    u_native = _matrix(raw_u_fit, "Native fitting u", columns=GRAM_DIM, minimum_rows=5)
    vy = _measurements(y_valid)
    vu_native = _matrix(raw_u_valid, "Native external validation u", columns=GRAM_DIM)
    if len(y) != len(u_native) or len(vy) != len(vu_native):
        raise ValueError("Measurements and native u must have the same ordered objects within each partition")
    if y.shape[-1] != vy.shape[-1]:
        raise ValueError("Fitting and external validation measurements must have the same feature dimension")
    seed = _seed(seed)
    stats = fit_preprocessing(y, u_native)
    x = transform_input(y[:, 0], stats)
    u = transform_target(u_native, stats)
    final_path = _CenteredRidgePath(x, u)
    selected, validation_errors, validation_predictions, final_selection = _validation_selection(
        final_path, vy, vu_native, stats)
    coefficient, intercept = final_path.coefficients(selected)

    group_values = None
    if groups is None:
        splits, allocation = _membership(len(y), 5, seed)
    else:
        supplied_groups = np.asarray(groups)
        if supplied_groups.shape != (len(y),):
            raise ValueError("Fitting chemistry groups must have one identity per fitting row")
        if supplied_groups.dtype.kind in "fc" and not np.isfinite(supplied_groups).all():
            raise ValueError("Fitting chemistry groups cannot contain missing or nonfinite identities")
        group_values = np.asarray(groups, dtype=str)
        if any(not value.strip() or value.strip().lower() in {"none", "nan"} for value in group_values):
            raise ValueError("Fitting chemistry groups require nonempty known identities")
        unique_groups, inverse = np.unique(group_values, return_inverse=True)
        if len(unique_groups) < 5:
            raise ValueError("At least five distinct fitting chemistry groups are required")
        group_splits, group_allocation = _membership(len(unique_groups), 5, seed)
        allocation = group_allocation[inverse]
        splits = [(np.flatnonzero(np.isin(inverse, fit_groups)),
                   np.flatnonzero(np.isin(inverse, error_groups)))
                  for fit_groups, error_groups in group_splits]
        if any(set(group_values[fit]) & set(group_values[check]) for fit, check in splits):
            raise RuntimeError("A fitting chemical group crosses an internal error fold")
    native_oof = np.empty_like(u_native)
    counts = np.zeros(len(y), dtype=np.int64)
    selected_inner, fold_records, audit = [], [], {}
    _retain_stats(audit, "final_preprocessing", stats)
    audit["final_validation_candidate_per_object_mse"] = validation_errors
    audit["final_validation_candidate_predictions"] = validation_predictions
    audit["final_validation_target_standardized"] = transform_target(vu_native, stats)
    audit["validation_indices"] = np.arange(len(vy), dtype=np.int64)
    audit["fit_indices"] = np.arange(len(y), dtype=np.int64)

    for fold, (fit, check) in enumerate(splits):
        inner_stats = fit_preprocessing(y[fit], u_native[fit])
        inner_x = transform_input(y[fit, 0], inner_stats)
        inner_u = transform_target(u_native[fit], inner_stats)
        path = _CenteredRidgePath(inner_x, inner_u)
        penalty, errors, predictions, choice = _validation_selection(path, vy, vu_native, inner_stats)
        selected_inner.append(penalty)
        beta, offset = path.coefficients(penalty)
        held_out_mean = transform_input(y[check, 0], inner_stats) @ beta + offset
        native_oof[check] = _restore_target(held_out_mean, inner_stats)
        counts[check] += 1
        prefix = _retain_stats(audit, f"inner_{fold}_preprocessing", inner_stats)
        audit[f"inner_{fold}_fit_indices"] = fit.copy()
        audit[f"inner_{fold}_error_indices"] = check.copy()
        audit[f"inner_{fold}_validation_candidate_per_object_mse"] = errors
        audit[f"inner_{fold}_validation_candidate_predictions"] = predictions
        fold_records.append(dict(fold=fold, fit_indices=fit.tolist(),
            error_indices=check.tolist(), validation_indices=list(range(len(vy))),
            fit_error_index_space="rows of y_fit/raw_u_fit",
            validation_index_space="rows of y_valid/raw_u_valid; a distinct caller-owned partition",
            n_fit=len(fit), n_error=len(check), n_validation=len(vy),
            selected_lambda=penalty, alpha=float(len(fit)*penalty),
            alpha_values=[float(len(fit)*value) for value in REGULARIZATION_GRID],
            preprocessing_audit_prefix=prefix, external_validation_selection=choice))

    if not np.array_equal(counts, np.ones(len(y), dtype=np.int64)):
        raise RuntimeError("Every fitting object must have exactly one held-out error prediction")
    oof = transform_target(native_oof, stats)
    residuals = u - oof
    covariance, centered_covariance, bias, covariance_audit = _fit_error_second_moment(
        residuals, include_bias=True)
    audit.update(oof_native_predictions=native_oof, oof_predictions=oof,
        oof_residuals=residuals, residuals_for_covariance=residuals.copy(),
        residual_mean=bias, centered_residual_covariance=centered_covariance,
        covariance=covariance.copy(), oof_count=counts, inner_fold_membership=allocation,
        inner_selected_lambdas=np.asarray(selected_inner),
        training_x_mean=final_path.x_mean, training_u_mean=final_path.u_mean)
    metadata = dict(method="external-validation-selected full-input ridge Gram Gaussian",
        training_objects=len(y), validation_objects=len(vy), feature_dimension=x.shape[1],
        target_coordinate_count=GRAM_DIM, target_space="final fit-only standardized native Gram u",
        seed=seed, regularization_grid=list(REGULARIZATION_GRID),
        final_alpha=float(len(y)*selected), final_selection=final_selection,
        final_fit_indices=list(range(len(y))), validation_indices=list(range(len(vy))),
        fit_index_space="rows of y_fit/raw_u_fit",
        validation_index_space="rows of y_valid/raw_u_valid; caller must ensure distinct identities",
        penalty_convention="per-output average squared error + lambda*||B||²; alpha=n_fit*lambda",
        internal_error_folds=fold_records, error_folds=5, kernel_eigendecompositions=6,
        preprocessing_fits=6, validation_used_for_preprocessing=False,
        validation_used_for_coefficient_fitting=False, validation_used_for_penalty_selection=True,
        validation_selection_shared_with_HR=True,
        selection_information_differs_from_train_only_ridge=True,
        fold_preprocessing="all four-role Y moments, u moments and X lognorm moments refitted within each coefficient-fitting subset",
        input_precision="affine X coordinates and normalized lognorm float32 then float64, matching the G/HR input convention",
        oof_coordinate_alignment="each held-out mean restored to native u, then transformed using final fit-only moments",
        covariance_estimation=covariance_audit, covariance_residuals_out_of_fold=True,
        covariance_estimand="LedoitWolf(centered held-out errors) + residual_mean outer product",
        covariance_is_full_joint=True, prediction_bias_correction=False,
        oof_mean_mse=float(np.square(residuals).mean()),
        final_fit_mean_mse=float(np.square(u-(x@coefficient+intercept)).mean()),
        oof_scope="development error estimates with the same external validation labels reused across internal fits; not an independent certificate",
        monte_carlo_object_coupling="independent residual draws across query objects",
        monte_carlo_coupling_is_physical_dependence=False,
        physical_shared_independent_noise_identified=False, campaign_joint_dependence_identified=False,
        query_targets_accepted=False, formal_certificate=False)
    if group_values is not None:
        audit.update(fitting_chemistry_groups=group_values.copy(),
                     unique_fitting_chemistry_groups=unique_groups.copy(),
                     chemistry_group_fold_membership=group_allocation.copy())
        metadata.update(chemistry_grouped_internal_oof=True,
            fitting_chemistry_group_count=len(unique_groups),
            internal_error_grouping="seeded shuffled KFold over sorted unique fitting chemistry groups; all member rows held out together",
            external_validation_group_disjointness="caller-validated; no validation group identities accepted by this function")
        for record, (fit, check) in zip(fold_records, splits):
            record.update(fit_chemistry_groups=np.unique(group_values[fit]).tolist(),
                          error_chemistry_groups=np.unique(group_values[check]).tolist())
    model = GramSimpleGaussian("RIDGE", intercept, covariance, coefficient=coefficient,
                               selected_lambda=selected, metadata=metadata, audit_arrays=audit)
    return model, stats
