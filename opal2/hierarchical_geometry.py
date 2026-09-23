"""Fixed-ridge plus bounded nonlinear correction in nine Gram coordinates.

This is the fixed-role geometry version of a conditional measurement estimator,
not a fitted biological or source/batch/plate latent hierarchy. The caller owns
the fitting-only input/target transforms and the ridge fit. The model receives
all transformed initial-well coordinates plus its observed log norm; it neither
selects coordinates nor accepts future wells at prediction time.

Its covariance is deliberately external: the caller can estimate a full joint
error covariance from out-of-fold predictions. Such errors include mean-model
estimation error and misspecification, not just independent measurement noise.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


GRAM_DIM = 9


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class RidgeResidualMean(nn.Module):
    """Keep a fitted ridge map fixed and learn a bounded residual only.

    All outputs, including the correction bound, are in the caller's fitting-
    standardized nine-dimensional Gram-coordinate space, not native profiles.
    The constructor preserves floating coefficient precision (float32/float64).
    Inputs must use the resulting model dtype/device; ``.to(...)`` is supported
    by the usual PyTorch API and also casts the fixed ridge buffers.

    ``config`` contains JSON-serializable architectural choices. The fixed
    coefficient and intercept are persistent buffers in ``state_dict``. Restore
    with ``from_config(config, coefficient=state['coefficient'],
    intercept=state['intercept'])`` and then ``load_state_dict(state)``.
    """

    def __init__(self, input_dim, coefficient, intercept, hidden_dim=32,
                 correction_bound=.5, residual_penalty=.1, dropout=.1):
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        if not math.isfinite(correction_bound) or correction_bound <= 0:
            raise ValueError("correction_bound must be finite and positive")
        if not math.isfinite(residual_penalty) or residual_penalty < 0:
            raise ValueError("residual_penalty must be finite and nonnegative")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite in [0,1)")
        self.correction_bound = float(correction_bound)
        self.residual_penalty_weight = float(residual_penalty)
        self.dropout_probability = float(dropout)

        coefficient = torch.as_tensor(coefficient)
        if coefficient.dtype not in (torch.float32, torch.float64):
            raise ValueError("coefficient must use float32 or float64")
        intercept = torch.as_tensor(intercept, dtype=coefficient.dtype,
                                    device=coefficient.device)
        if coefficient.shape != (self.input_dim, GRAM_DIM):
            raise ValueError("coefficient must have shape [input_dim,9]")
        if intercept.shape != (GRAM_DIM,):
            raise ValueError("intercept must have shape [9]")
        if not torch.isfinite(coefficient).all() or not torch.isfinite(intercept).all():
            raise ValueError("The fixed ridge parameters must be finite")
        # Clone as well as detach: a caller mutating its original array/tensor
        # must not silently change the frozen ridge map.
        self.register_buffer("coefficient", coefficient.detach().clone())
        self.register_buffer("intercept", intercept.detach().clone())
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, GRAM_DIM),
        ).to(dtype=coefficient.dtype, device=coefficient.device)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.config = dict(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                           correction_bound=self.correction_bound,
                           residual_penalty=self.residual_penalty_weight,
                           dropout=self.dropout_probability)

    @classmethod
    def from_config(cls, config: Mapping, *, coefficient, intercept):
        return cls(coefficient=coefficient, intercept=intercept, **dict(config))

    def _check_inputs(self, inputs):
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2:
            raise ValueError("inputs must be a torch tensor [N,input_dim]")
        if inputs.shape[1] != self.input_dim or len(inputs) == 0:
            raise ValueError("inputs must contain every fitted coordinate and at least one object")
        if inputs.dtype != self.coefficient.dtype or inputs.device != self.coefficient.device:
            raise ValueError("inputs must match the model dtype and device")
        if not torch.isfinite(inputs).all():
            raise ValueError("inputs must be finite; no clipping or imputation is applied")

    def base_mean(self, inputs):
        self._check_inputs(inputs)
        result = inputs @ self.coefficient + self.intercept
        if not torch.isfinite(result).all():
            raise FloatingPointError("The fixed ridge prediction overflowed")
        return result

    def _correction(self, inputs):
        raw = self.network(inputs)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("The residual network returned nonfinite coordinates")
        return self.correction_bound * torch.tanh(raw)

    def correction(self, inputs):
        self._check_inputs(inputs)
        return self._correction(inputs)

    def forward(self, inputs):
        return self.base_mean(inputs) + self._correction(inputs)

    def loss(self, inputs, target):
        """Mean fitting objective only; covariance has no route into this loss.

        ``residual_penalty`` is the weighted correction-MSE term. Dropout is
        drawn once per call and the identical correction is used for the mean
        loss and its penalty, rather than recomputing a second stochastic path.
        """
        base = self.base_mean(inputs)
        if not isinstance(target, torch.Tensor) or target.shape != base.shape:
            raise ValueError("target must be a tensor [N,9]")
        if target.dtype != base.dtype or target.device != base.device or not torch.isfinite(target).all():
            raise ValueError("target must be finite and match the model dtype/device")
        correction = self._correction(inputs)
        prediction = base + correction
        mean_mse = F.mse_loss(prediction, target)
        correction_mse = correction.square().mean()
        penalty = self.residual_penalty_weight * correction_mse
        return dict(loss=mean_mse + penalty, mean_mse=mean_mse,
                    residual_penalty=penalty, correction_mse=correction_mse)


def sample_joint_coordinates(mean, covariance, samples, seed):
    """Sample full nine-coordinate Gaussian errors, without altering their law.

    ``mean`` is [N,9]; covariance is either global [9,9] or [N,9,9]. The returned
    float64 array is [samples,N,9]. Errors are independent across compounds and
    jointly correlated across the nine coordinates; this is not a claim of
    independent physical batches. No jitter, eigenvalue clipping, dropped rows,
    or replacement draws are permitted. The result remains in the same space as
    the supplied mean/covariance; the caller applies its fixed inverse scaler.
    """
    samples = _positive_integer(samples, "samples")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.ndim != 2 or mean.shape[1] != GRAM_DIM or not len(mean) or not np.isfinite(mean).all():
        raise ValueError("mean must be finite and have nonempty shape [N,9]")
    if covariance.shape not in ((GRAM_DIM, GRAM_DIM), (len(mean), GRAM_DIM, GRAM_DIM)):
        raise ValueError("covariance must have shape [9,9] or [N,9,9]")
    if not np.isfinite(covariance).all() or not np.allclose(
            covariance, np.swapaxes(covariance, -1, -2), rtol=1e-12, atol=1e-14):
        raise ValueError("covariance must be finite and symmetric")
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite; no jitter is added") from error
    epsilon = np.random.default_rng(int(seed)).standard_normal((samples, len(mean), GRAM_DIM))
    if factor.ndim == 2:
        result = mean[None] + epsilon @ factor.T
    else:
        result = mean[None] + np.einsum("nij,snj->sni", factor, epsilon)
    if not np.isfinite(result).all():
        raise FloatingPointError("Joint coordinate samples overflowed; no draws were replaced")
    return result
