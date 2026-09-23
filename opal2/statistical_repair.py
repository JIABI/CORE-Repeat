"""Small, explicit repairs of a frozen full-space conditional Gaussian.

The optional ridge correction changes conditional means only, in a train-fixed
low-rank output basis. A separate global positive ``a`` multiplies the ENTIRE
conditional covariance. It is dispersion calibration, not a new estimate of
structural biological/technical variance components or a compound-specific c_i.
No future measurements are accepted at inference. The baseline's full-space
mean and positive residual variance outside the correction basis are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .closed_form_baseline import ClosedFormBaseline
from .model import JointGaussian
from .moment_repair import ClosedFormGaussianAdapter


def _finite_array(value, ndim, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with {ndim} dimensions")
    return value


@dataclass
class ResidualMeanCorrection:
    """Serializable ridge residual, including all train-fit input transforms.

    ``coefficients`` has shape [K+1,3,K] and ``intercept`` [3,K], where
    K=min(64, baseline.rank). Input coordinates are slot-centered X in the
    baseline basis, plus log1p affine-X energy. Prediction returns PHYSICAL
    full-D mean corrections, not a replacement measurement representation.
    """
    center: np.ndarray
    scale: np.ndarray
    slot0_mean: np.ndarray
    basis: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    train_ids: tuple[str, ...]
    alpha: float

    def __post_init__(self):
        for name, ndim in (("center", 1), ("scale", 1), ("slot0_mean", 1),
                           ("basis", 2), ("feature_center", 1), ("feature_scale", 1),
                           ("coefficients", 3), ("intercept", 2)):
            setattr(self, name, _finite_array(getattr(self, name), ndim, name))
        d, k = self.basis.shape
        if (not 1 <= k <= 64 or self.center.shape != (d,) or self.scale.shape != (d,)
                or self.slot0_mean.shape != (d,) or self.feature_center.shape != (k + 1,)
                or self.feature_scale.shape != (k + 1,)
                or self.coefficients.shape != (k + 1, 3, k) or self.intercept.shape != (3, k)):
            raise ValueError("Residual correction dimensions are inconsistent")
        if np.any(self.scale <= 0) or np.any(self.feature_scale <= 0):
            raise ValueError("All affine and feature scales must be positive")
        if not np.allclose(self.basis.T @ self.basis, np.eye(k), atol=1e-7):
            raise ValueError("Residual output basis must be orthonormal")
        self.alpha = float(self.alpha)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        self.train_ids = tuple(map(str, self.train_ids))
        if not self.train_ids or len(set(self.train_ids)) != len(self.train_ids):
            raise ValueError("Fit compound IDs must be nonempty and unique")

    @property
    def rank(self):
        return self.basis.shape[1]

    def _features(self, context_raw):
        x = _finite_array(context_raw, 2, "context_raw")
        if x.shape[1] != len(self.center):
            raise ValueError("context_raw must contain only the full-D initial well")
        affine = (x - self.center) / self.scale
        projected = (affine - self.slot0_mean) @ self.basis
        energy = np.log1p(np.square(affine).mean(axis=1))
        features = np.column_stack((projected, energy))
        if not np.isfinite(features).all():
            raise ValueError("Residual input feature transform overflowed")
        return features

    def predict(self, context_raw, *, weight=1.):
        weight = float(weight)
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Residual correction weight must lie in [0,1]")
        features = self._features(context_raw)
        if weight == 0:
            return np.zeros((len(features), 3, len(self.center)), dtype=np.float64)
        features = (features - self.feature_center) / self.feature_scale
        coefficient_prediction = np.einsum("np,ptk->ntk", features, self.coefficients)
        coefficient_prediction += self.intercept
        physical_delta = (coefficient_prediction @ self.basis.T) * self.scale
        return weight * physical_delta

    def save(self, path):
        arrays = {name: getattr(self, name) for name in (
            "center", "scale", "slot0_mean", "basis", "feature_center", "feature_scale",
            "coefficients", "intercept")}
        metadata = {"schema_version": 1, "alpha": self.alpha,
                    "train_ids": list(self.train_ids),
                    "ridge_objective": "sum_squared_residual_coefficients + alpha*sum_squared_coefficients; unpenalized intercept",
                    "input": "unclipped slot0 centered basis scores plus log1p(mean affineX squared)",
                    "output": "physical full-space additive mean correction; covariance unchanged"}
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **arrays, metadata=np.array(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("schema_version") != 1:
                raise ValueError("Unsupported residual correction schema")
            names = ("center", "scale", "slot0_mean", "basis", "feature_center", "feature_scale",
                     "coefficients", "intercept")
            arrays = {name: archive[name].copy() for name in names}
        return cls(**arrays, train_ids=tuple(metadata["train_ids"]), alpha=metadata["alpha"])


def fit_residual(baseline: ClosedFormBaseline, ytrain: np.ndarray,
                 ids: Sequence[str], alpha: float) -> ResidualMeanCorrection:
    """Fit only on declared training objects; no change to frozen baseline.

    The supplied role ordering is X,Z1,Z2,V. All three residual outputs use
    the same input feature matrix, with independent coefficients and an
    unpenalized intercept for each role/output coordinate. Alpha follows the
    standard ridge SUM-of-squares objective, not mean-of-squares scaling.
    """
    y = _finite_array(ytrain, 3, "ytrain")
    ids = tuple(map(str, ids))
    if (y.shape[1:] != (4, baseline.feature_dim) or len(y) < 2 or len(ids) != len(y)
            or len(set(ids)) != len(ids) or baseline.n_slots != 4):
        raise ValueError("Fit requires at least two unique training compounds with four full-D roles")
    declared = baseline.metadata.get("train_ids")
    if declared is not None and not set(ids).issubset(set(map(str, declared))):
        raise ValueError("Residual fit IDs extend outside the frozen baseline training objects")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    k = min(64, baseline.rank)
    basis = baseline.basis[:, :k].copy()
    correction = ResidualMeanCorrection(
        center=baseline.center.copy(), scale=baseline.scale.copy(),
        slot0_mean=baseline.slot_mean[0].copy(), basis=basis,
        feature_center=np.zeros(k + 1), feature_scale=np.ones(k + 1),
        coefficients=np.zeros((k + 1, 3, k)), intercept=np.zeros((3, k)),
        train_ids=ids, alpha=alpha)
    features = correction._features(y[:, 0])
    correction.feature_center = features.mean(axis=0)
    # A constant coordinate has no varying signal; retaining unit scale avoids
    # inventing large predictors while the unpenalized intercept handles means.
    raw_scale = features.std(axis=0, ddof=0)
    correction.feature_scale = np.where(raw_scale > 1e-12, raw_scale, 1.)
    features = (features - correction.feature_center) / correction.feature_scale
    conditional_mean = baseline.conditional(y[:, :1], [0], [1, 2, 3]).mean
    coefficients = ((y[:, 1:4] - conditional_mean) / baseline.scale) @ basis
    correction.intercept = coefficients.mean(axis=0)
    target = (coefficients - correction.intercept).reshape(len(y), -1)
    if alpha == 0:
        fitted = np.linalg.lstsq(features, target, rcond=None)[0]
    else:
        gram = features.T @ features + alpha * np.eye(features.shape[1])
        fitted = np.linalg.solve(gram, features.T @ target)
    correction.coefficients = fitted.reshape(k + 1, 3, k)
    correction.__post_init__()
    return correction


class AffineConditionalGaussian(JointGaussian):
    """Exact affine transport of a full joint Gaussian, preserving fast paths.

    Public covariance components are scaled consistently, but large factors
    need not be copied during sampling/density/weighted-moment evaluation.
    The one global scalar preserves all base correlations and shared draws.
    It does NOT fix a misspecified correlation structure.
    """
    def __init__(self, base: JointGaussian, mean_shift=None, a=1.):
        if not isinstance(base, JointGaussian):
            raise TypeError("Affine Gaussian transport requires JointGaussian")
        self.base = base
        self.a = float(a)
        if not np.isfinite(self.a) or self.a <= 0:
            raise ValueError("Conditional covariance multiplier a must be finite positive")
        self.sqrt_a = math.sqrt(self.a)
        if mean_shift is None:
            mean_shift = torch.zeros_like(base.mean)
        self.mean_shift = torch.as_tensor(mean_shift, device=base.mean.device, dtype=base.mean.dtype)
        if self.mean_shift.shape != base.mean.shape or not bool(torch.isfinite(self.mean_shift).all()):
            raise ValueError("Mean shift must be finite with base mean shape")
        self._zero_shift = not bool(torch.count_nonzero(self.mean_shift))
        self.mean = base.mean if self._zero_shift else base.mean + self.mean_shift
        self.latent_semantics = "frozen_full_conditional_covariance_times_global_dispersion_not_variance_component_refit"
        self.environment_groups = base.environment_groups
        self.environment_cache_namespace = base.environment_cache_namespace

    @property
    def diag_var(self):
        return self.base.diag_var if self.a == 1 else self.base.diag_var * self.a

    @property
    def factors(self):
        return self.base.factors if self.a == 1 else self.base.factors * self.sqrt_a

    @property
    def local_factors(self):
        value = self.base.local_factors
        return value if value is None or self.a == 1 else value * self.sqrt_a

    @property
    def environment_loadings(self):
        values = self.base.environment_loadings
        return values if values is None or self.a == 1 else tuple(v * self.sqrt_a for v in values)

    @property
    def marginal_variance(self):
        return self.base.marginal_variance if self.a == 1 else self.base.marginal_variance * self.a

    def sample_joint(self, n_samples, generator=None, environment_noise_cache=None):
        draw = self.base.sample_joint(n_samples, generator, environment_noise_cache)
        if self.a == 1:
            return draw if self._zero_shift else draw + self.mean_shift.unsqueeze(0)
        return self.mean.unsqueeze(0) + self.sqrt_a * (draw - self.base.mean.unsqueeze(0))

    def _base_target(self, target_y):
        if self.a == 1:
            return target_y if self._zero_shift else target_y - self.mean_shift
        return self.base.mean + (target_y - self.mean) / self.sqrt_a

    def log_prob(self, target_y, target_mask=None):
        observed = self.observed_mask(target_y, target_mask)
        result = self.base.log_prob(self._base_target(target_y), target_mask)
        return result if self.a == 1 else result - observed.sum((1, 2)).double() * (.5 * math.log(self.a))

    def joint_log_prob(self, target_y, target_mask=None):
        observed = self.observed_mask(target_y, target_mask)
        result = self.base.joint_log_prob(self._base_target(target_y), target_mask)
        return result if self.a == 1 else result - observed.sum().double() * (.5 * math.log(self.a))

    def weighted_moments(self, weights):
        mean, diagonal, factor = self.base.weighted_moments(weights)
        if not self._zero_shift:
            mean = mean + (weights.unsqueeze(-1) * self.mean_shift).sum(1)
        if self.a != 1:
            diagonal, factor = diagonal * self.a, factor * self.sqrt_a
        return mean, diagonal, factor


class StatisticalRepairAdapter(ClosedFormGaussianAdapter):
    """Existing four-role evaluator adapter; inference observes only X."""
    def __init__(self, baseline, scaler, residual=None, a=1., residual_weight=1.):
        super().__init__(baseline, scaler)
        self.a = float(a)
        self.residual_weight = float(residual_weight)
        if not np.isfinite(self.a) or self.a <= 0:
            raise ValueError("Conditional covariance multiplier a must be finite positive")
        if not np.isfinite(self.residual_weight) or not 0 <= self.residual_weight <= 1:
            raise ValueError("Residual weight must be in [0,1]")
        if residual is not None:
            if not isinstance(residual, ResidualMeanCorrection):
                raise TypeError("residual must be ResidualMeanCorrection or None")
            for left, right in ((residual.center, baseline.center), (residual.scale, baseline.scale),
                                (residual.slot0_mean, baseline.slot_mean[0]),
                                (residual.basis, baseline.basis[:, :residual.rank])):
                if not np.array_equal(left, right):
                    raise ValueError("Residual correction and frozen baseline coordinates differ")
            if not set(residual.train_ids).issubset(set(map(str, scaler.train_ids))):
                raise ValueError("Residual correction used objects outside frozen training IDs")
        self.residual = residual

    @torch.no_grad()
    def forward(self, batch):
        base = super().forward(batch)
        if self.residual is None or self.residual_weight == 0:
            if self.a == 1:
                return base
            return AffineConditionalGaussian(base, a=self.a)
        y = batch["context_y"][:, 0].detach().cpu().double().numpy()
        raw = y * self.baseline.scale + self.baseline.center
        delta = self.residual.predict(raw, weight=self.residual_weight) / self.baseline.scale
        delta = torch.as_tensor(delta, device=base.mean.device, dtype=base.mean.dtype)
        return AffineConditionalGaussian(base, mean_shift=delta, a=self.a)
