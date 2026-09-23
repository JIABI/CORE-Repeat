"""Two complete Gaussian baselines in common TRAIN-standardized Gram targets.

GLOBAL fits one nine-dimensional mean and a full Ledoit--Wolf covariance.
RIDGE fits all supplied X coordinates, choosing regularization inside TRAIN.
Its covariance is a shrunk *error second moment* from nested out-of-fold mean
predictions, including any residual bias. That bias is not added to the mean.

The caller supplies X/u preprocessing fitted on the enclosing TRAIN. Internal
folds refit intercept centering, not that preprocessing. Accordingly, the OOF
records are development error estimates with shared TRAIN preprocessing, not
an unbiased validation certificate or a fitted physical noise decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold


GRAM_DIM = 9
REGULARIZATION_GRID = (.01, .1, 1., 10., 100.)
TIE_RTOL = 1e-10
TIE_ATOL = 1e-12


def _matrix(value, name, *, columns=None, minimum_rows=1):
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim != 2 or len(array) < minimum_rows or array.shape[1] < 1
            or (columns is not None and array.shape[1] != columns)):
        raise ValueError(f"{name} must be a nonempty matrix" + (f" with {columns} columns" if columns else ""))
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("An explicit nonnegative integer seed is required")
    return int(seed)


def _checked_covariance(covariance):
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (GRAM_DIM, GRAM_DIM) or not np.isfinite(covariance).all():
        raise ValueError("A finite full 9-by-9 covariance is required")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError("The joint covariance must be symmetric")
    covariance = (covariance+covariance.T)/2
    eigenvalues = np.linalg.eigvalsh(covariance)
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"Joint covariance is not numerically positive definite (minimum eigenvalue {eigenvalues[0]:.6g}); no jitter was added") from error
    if eigenvalues[0] <= 0 or not np.isfinite(factor).all():
        raise ValueError("Joint covariance failed its numerical SPD audit; no jitter was added")
    return covariance, factor, dict(minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]), condition_number=float(eigenvalues[-1]/eigenvalues[0]),
        jitter_added=0., full_joint_covariance=True)


def _fit_error_second_moment(residuals, *, include_bias):
    residuals = _matrix(residuals, "Covariance residuals", columns=GRAM_DIM, minimum_rows=2)
    bias = residuals.mean(0)
    centered = residuals-bias
    estimate = LedoitWolf(assume_centered=True).fit(centered)
    centered_covariance = np.asarray(estimate.covariance_, dtype=np.float64)
    covariance = centered_covariance+np.outer(bias, bias) if include_bias else centered_covariance
    covariance, _, audit = _checked_covariance(covariance)
    return covariance, centered_covariance, bias, dict(
        ledoit_wolf_shrinkage=float(estimate.shrinkage_), residual_count=len(residuals),
        residual_mean_added_to_covariance=bool(include_bias),
        residual_mean_added_to_prediction=False, **audit)


class _CenteredRidgePath:
    """One centered dual Gram eigendecomposition serves all five penalties."""

    def __init__(self, x, u):
        self.n_fit = len(x)
        self.x_mean, self.u_mean = x.mean(0), u.mean(0)
        self.x_centered = x-self.x_mean
        target = u-self.u_mean
        gram = self.x_centered@self.x_centered.T
        if not np.isfinite(gram).all():
            raise ValueError("The centered ridge Gram matrix overflowed")
        eigenvalues, self.eigenvectors = np.linalg.eigh((gram+gram.T)/2)
        spectral_scale = max(float(np.abs(eigenvalues).max()), np.finfo(np.float64).tiny)
        roundoff = 64*np.finfo(np.float64).eps*max(x.shape)*spectral_scale
        if eigenvalues[0] < -roundoff:
            raise ValueError("The computed X Xᵀ kernel is materially indefinite")
        # This removes only roundoff-negative eigenvalues of X Xᵀ, not a
        # covariance floor or a user-selected ridge penalty.
        self.eigenvalues = np.maximum(eigenvalues, 0.)
        self.rotated_targets = self.eigenvectors.T@target

    def predict_path(self, x, lambdas=REGULARIZATION_GRID):
        projected_cross = ((x-self.x_mean)@self.x_centered.T)@self.eigenvectors
        return np.stack([self.u_mean + projected_cross@(
            self.rotated_targets/(self.eigenvalues+self.n_fit*penalty)[:, None])
            for penalty in lambdas])

    def coefficients(self, penalty):
        dual = self.eigenvectors@(self.rotated_targets/(
            self.eigenvalues+self.n_fit*penalty)[:, None])
        coefficient = self.x_centered.T@dual
        return coefficient, self.u_mean-self.x_mean@coefficient


def _membership(n, folds, seed):
    allocation = np.full(n, -1, dtype=np.int64)
    split = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(n)))
    for fold, (_, check) in enumerate(split):
        allocation[check] = fold
    if np.any(allocation < 0):
        raise RuntimeError("Every row must receive exactly one validation fold")
    return split, allocation


def _summarize_cv(predictions, target, allocation):
    errors = np.square(predictions-target[None])
    records = []
    for index, penalty in enumerate(REGULARIZATION_GRID):
        fold_mse = [float(errors[index, allocation==fold].mean())
                    for fold in range(int(allocation.max())+1)]
        records.append(dict(lambda_value=float(penalty), mean_mse=float(errors[index].mean()),
                            fold_mse=fold_mse, unweighted_mean_fold_mse=float(np.mean(fold_mse))))
    minimum = min(record["mean_mse"] for record in records)
    tied = [record for record in records
            if np.isclose(record["mean_mse"], minimum, rtol=TIE_RTOL, atol=TIE_ATOL)]
    selected = max(tied, key=lambda record: record["lambda_value"])["lambda_value"]
    return selected, dict(candidate_records=records, selected_lambda=selected,
        selection_metric="pooled out-of-fold mean squared error over objects and nine target coordinates",
        tie_break="largest lambda within the declared numerical tie tolerance",
        tie_rtol=TIE_RTOL, tie_atol=TIE_ATOL)


@dataclass
class GramSimpleGaussian:
    """Joint Gaussian in the caller's common standardized nine-dimensional u.

    A GLOBAL shared-draw broadcast is a common-random-number integration
    convention to avoid Monte Carlo ranking of identical predictions. It is
    NOT a claim of perfect physical dependence between different compounds.
    The RIDGE sampler uses independent query-object residual draws; neither
    model provides a learned campaign-wide batch-dependence law.
    """

    kind: str
    intercept: np.ndarray
    covariance: np.ndarray
    coefficient: np.ndarray | None = None
    selected_lambda: float | None = None
    metadata: dict = field(default_factory=dict)
    audit_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    scale_tril: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.kind not in ("GLOBAL", "RIDGE"):
            raise ValueError("Unknown simple Gram model")
        self.intercept = np.asarray(self.intercept, dtype=np.float64)
        if self.intercept.shape != (GRAM_DIM,) or not np.isfinite(self.intercept).all():
            raise ValueError("The intercept must have nine finite entries")
        self.covariance, self.scale_tril, _ = _checked_covariance(self.covariance)
        if self.kind == "RIDGE":
            self.coefficient = _matrix(self.coefficient, "Ridge coefficients", columns=GRAM_DIM)
            if self.selected_lambda not in REGULARIZATION_GRID:
                raise ValueError("RIDGE must record a selected penalty from the fixed grid")
        elif self.coefficient is not None or self.selected_lambda is not None:
            raise ValueError("GLOBAL cannot have a fitted X coefficient or ridge penalty")

    @property
    def coef(self):
        return self.coefficient

    def predict_mean(self, x):
        x = _matrix(x, "Query X")
        if self.kind == "GLOBAL":
            return np.broadcast_to(self.intercept, (len(x), GRAM_DIM)).copy()
        if x.shape[1] != self.coefficient.shape[0]:
            raise ValueError("Query X must retain the complete fitted input coordinate set")
        result = x@self.coefficient+self.intercept
        if not np.isfinite(result).all():
            raise ValueError("Ridge prediction overflowed")
        return result

    def sample_coordinates(self, x, samples, seed):
        mean = self.predict_mean(x)
        if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 1:
            raise ValueError("samples must be a positive integer")
        rng = np.random.default_rng(_seed(seed))
        if self.kind == "GLOBAL":
            draws = self.intercept+rng.standard_normal((int(samples), GRAM_DIM))@self.scale_tril.T
            return np.broadcast_to(draws[:, None], (int(samples), len(mean), GRAM_DIM)).copy()
        noise = rng.standard_normal((int(samples), len(mean), GRAM_DIM))@self.scale_tril.T
        return mean[None]+noise

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError("A saved simple Gram model is not overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"schema_version": np.asarray(1, dtype=np.int64), "kind": np.asarray(self.kind),
            "intercept": self.intercept, "covariance": self.covariance,
            "scale_tril": self.scale_tril,
            "coefficient": np.empty((0, GRAM_DIM)) if self.coefficient is None else self.coefficient,
            "selected_lambda": np.asarray(np.nan if self.selected_lambda is None else self.selected_lambda),
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True, allow_nan=False))}
        for key, value in self.audit_arrays.items():
            arrays["audit__"+key] = np.asarray(value)
        # File handles preserve the exact caller-specified path suffix.
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["schema_version"]) != 1:
                raise ValueError("Unknown saved Gram baseline schema")
            kind = str(saved["kind"].item())
            result = cls(kind=kind, intercept=saved["intercept"].copy(), covariance=saved["covariance"].copy(),
                coefficient=None if kind=="GLOBAL" else saved["coefficient"].copy(),
                selected_lambda=None if kind=="GLOBAL" else float(saved["selected_lambda"]),
                metadata=json.loads(str(saved["metadata_json"].item())),
                audit_arrays={key.removeprefix("audit__"):saved[key].copy()
                              for key in saved.files if key.startswith("audit__")})
            if not np.allclose(result.scale_tril, saved["scale_tril"], rtol=1e-12, atol=1e-14):
                raise ValueError("Saved Gaussian factor does not match its full covariance")
        return result


def load(path):
    """Load either complete baseline without receiving fitting or query labels."""
    return GramSimpleGaussian.load(path)


def _common_metadata(n):
    return dict(target_coordinate_count=GRAM_DIM, target_space="common enclosing-TRAIN-standardized u",
        training_objects=int(n), formal_certificate=False,
        covariance_is_full_joint=True, hidden_jitter=0., query_targets_accepted=False,
        input_preprocessing="supplied by caller, fitted on enclosing TRAIN",
        target_preprocessing="supplied by caller, fitted on enclosing TRAIN",
        fold_preprocessing="fit-fold X and u mean centering only; shared enclosing-TRAIN scales",
        oof_scope="development error estimate, not a fully cross-fitted preprocessing certificate",
        physical_shared_independent_noise_identified=False,
        campaign_joint_dependence_identified=False)


def fit_global(u_train):
    """Fit the unconditional mean and full Ledoit--Wolf centered covariance."""
    u = _matrix(u_train, "TRAIN u", columns=GRAM_DIM, minimum_rows=2)
    mean = u.mean(0)
    residuals = u-mean
    covariance, centered_covariance, residual_mean, covariance_audit = _fit_error_second_moment(
        residuals, include_bias=False)
    return GramSimpleGaussian("GLOBAL", mean, covariance,
        metadata=dict(**_common_metadata(len(u)), covariance_estimation=covariance_audit,
            prediction="constant TRAIN mean of u", covariance_residuals_out_of_fold=False,
            monte_carlo_object_coupling="identical draws broadcast as common random numbers to avoid false ranking",
            monte_carlo_coupling_is_physical_dependence=False),
        audit_arrays=dict(training_u_mean=mean, residuals_for_covariance=residuals,
                          residual_mean=residual_mean, centered_residual_covariance=centered_covariance))


def fit_ridge(x_train, u_train, seed=20260914):
    """Full-input ridge mean plus nested-OOF full error second moment.

    For each output the fitted objective is n_fit^-1 * sum(error²) +
    lambda * ||coefficient||², with an unpenalized intercept. Thus the
    conventional unnormalized ridge parameter is alpha=n_fit*lambda.
    Five outer folds each use four inner folds to choose lambda without that
    outer fold's labels. Final lambda selection uses five-fold CV on TRAIN.
    """
    x = _matrix(x_train, "TRAIN X", minimum_rows=5)
    u = _matrix(u_train, "TRAIN u", columns=GRAM_DIM, minimum_rows=5)
    if len(x) != len(u):
        raise ValueError("TRAIN X and u must have the same ordered objects")
    seed = _seed(seed)
    outer_splits, outer_membership = _membership(len(x), 5, seed)
    full_cv_predictions = np.empty((len(REGULARIZATION_GRID), len(x), GRAM_DIM), dtype=np.float64)
    oof = np.empty_like(u)
    oof_counts = np.zeros(len(x), dtype=np.int64)
    inner_membership = np.full((5, len(x)), -1, dtype=np.int64)
    outer_lambdas, outer_records = [], []
    for outer_fold, (fit, check) in enumerate(outer_splits):
        outer_path = _CenteredRidgePath(x[fit], u[fit])
        outer_predictions = outer_path.predict_path(x[check])
        full_cv_predictions[:, check] = outer_predictions
        inner_seed = (seed+1009*(outer_fold+1)) % (2**32-1)
        inner_splits, inner_local = _membership(len(fit), 4, inner_seed)
        inner_membership[outer_fold, fit] = inner_local
        inner_predictions = np.empty((len(REGULARIZATION_GRID), len(fit), GRAM_DIM), dtype=np.float64)
        inner_records = []
        for inner_fold, (local_fit, local_check) in enumerate(inner_splits):
            path = _CenteredRidgePath(x[fit[local_fit]], u[fit[local_fit]])
            inner_predictions[:, local_check] = path.predict_path(x[fit[local_check]])
            inner_records.append(dict(fold=inner_fold, n_fit=len(local_fit),
                fit_indices=fit[local_fit].tolist(), validation_indices=fit[local_check].tolist(),
                alpha_values=[float(len(local_fit)*penalty) for penalty in REGULARIZATION_GRID]))
        selected, inner_summary = _summarize_cv(inner_predictions, u[fit], inner_local)
        chosen = REGULARIZATION_GRID.index(selected)
        oof[check] = outer_predictions[chosen]
        oof_counts[check] += 1
        outer_lambdas.append(selected)
        outer_records.append(dict(outer_fold=outer_fold, outer_fit_indices=fit.tolist(),
            outer_validation_indices=check.tolist(), n_fit=len(fit), selected_lambda=selected,
            alpha=float(len(fit)*selected), inner_seed=inner_seed,
            inner_cv=inner_summary, inner_folds=inner_records,
            outer_mse=float(np.square(oof[check]-u[check]).mean())))
    if not np.array_equal(oof_counts, np.ones(len(x), dtype=np.int64)):
        raise RuntimeError("Nested OOF prediction must cover each TRAIN row exactly once")
    selected, full_cv = _summarize_cv(full_cv_predictions, u, outer_membership)
    final_path = _CenteredRidgePath(x, u)
    coefficient, intercept = final_path.coefficients(selected)
    residuals = u-oof
    covariance, centered_covariance, bias, covariance_audit = _fit_error_second_moment(
        residuals, include_bias=True)
    return GramSimpleGaussian("RIDGE", intercept, covariance, coefficient=coefficient,
        selected_lambda=selected,
        metadata=dict(**_common_metadata(len(u)), feature_dimension=x.shape[1], seed=seed,
            regularization_grid=list(REGULARIZATION_GRID),
            penalty_convention="per-output average squared error + lambda*||B||²; sklearn alpha=n_fit*lambda",
            final_alpha=float(len(x)*selected), full_train_cv=full_cv, outer_cv=outer_records,
            outer_folds=5, inner_folds=4, kernel_eigendecompositions=26,
            covariance_estimation=covariance_audit, covariance_residuals_out_of_fold=True,
            covariance_estimand="LedoitWolf(centered nested-OOF errors) + residual_mean outer product",
            oof_mean_mse=float(np.square(residuals).mean()),
            final_fit_mean_mse=float(np.square(u-(x@coefficient+intercept)).mean()),
            prediction_bias_correction=False,
            monte_carlo_object_coupling="independent residual draws across query objects",
            monte_carlo_coupling_is_physical_dependence=False),
        audit_arrays=dict(training_x_mean=final_path.x_mean, training_u_mean=final_path.u_mean,
            full_cv_predictions_by_lambda=full_cv_predictions, full_cv_fold_membership=outer_membership.copy(),
            oof_predictions=oof, oof_residuals=residuals, residuals_for_covariance=residuals.copy(),
            residual_mean=bias, centered_residual_covariance=centered_covariance,
            oof_count=oof_counts, outer_fold_membership=outer_membership,
            inner_fold_membership=inner_membership, outer_selected_lambdas=np.asarray(outer_lambdas)))
