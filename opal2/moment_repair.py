"""Explicit train-moment refit diagnostics for a frozen Gaussian predictor.

These adapters do not optimise neural weights. ``mean_only`` replaces its
conditional location; ``mean_and_within`` also substitutes the train-fitted
full-coordinate within-well covariance Q. The latter retains the original
compound and environmental factors, so it is an ablation of a specified
covariance component, not a claim that the decomposition is identified.

The fitted baseline owns its affine coordinates. Public distribution bridges
support original units too; model adapters use the exact frozen scaler and
the existing slot-0 context, slots-1/2/3 prediction protocol. No target
measurements enter construction of a predictive distribution.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import torch
from torch import nn

from .closed_form_baseline import ClosedFormBaseline, ConditionalGaussian
from .model import JointGaussian


def _tensor(value, *, dtype, device):
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)


def _noise_parameters(baseline, *, affine, dtype, device):
    scale = 1. if affine else baseline.scale
    factor = _tensor(baseline._noise_factor * np.asarray(scale)[..., None]
                     if not affine else baseline._noise_factor, dtype=dtype, device=device)
    diagonal = _tensor(baseline.residual_var * np.square(scale), dtype=dtype, device=device)
    # Factor the covariance actually represented by the returned tensors, even
    # when the inference model uses float32. Adapters cache this train-only core.
    fd, dd = factor.double(), diagonal.double()
    core = torch.eye(factor.shape[-1], dtype=torch.float64, device=device)
    cholesky = torch.linalg.cholesky(core + fd.T @ (fd / dd[:, None]))
    return factor, diagonal, cholesky


def _block_within(factor, batch, targets):
    """Give each physical well its own independent copy of the Q factor."""
    identity = torch.eye(targets, dtype=factor.dtype, device=factor.device)
    block = (identity[:, None, :, None] * factor[None, :, None, :]).reshape(
        targets, factor.shape[0], targets * factor.shape[1])
    return block.unsqueeze(0).expand(batch, -1, -1, -1)


def _environment_marginals(original):
    parts = []
    targets = original.mean.shape[1]
    for load, groups, _ in original._environment_specs():
        same = groups[:, :, None] == groups[:, None, :]
        representative = same.to(torch.int64).argmax(-1)
        assignment = torch.nn.functional.one_hot(representative, targets).to(load.dtype)
        parts.append((load.unsqueeze(-2) * assignment[:, :, None, :, None]).flatten(-2))
    return parts


class _BlockNoiseGaussian(JointGaussian):
    """JointGaussian with an exact fast path for repeated low-rank-plus-diagonal Q.

    Full factors remain available for arbitrary weighted moments and Gaussian
    conditioning. Density integrates Q per well before the smaller retained
    shared factors; sampling multiplies each within-well factor separately.
    This avoids an unnecessary factor-of-target-count in those operations.
    """

    def __init__(self, shared, within_factor, noise_cholesky):
        b, t, _ = shared.mean.shape
        within = _block_within(within_factor, b, t)
        local = shared.factors if shared.local_factors is None else shared.local_factors
        super().__init__(shared.mean, shared.diag_var,
                         torch.cat((shared.factors, within), -1),
                         torch.cat((local, within), -1),
                         shared.environment_loadings, shared.environment_groups,
                         shared.latent_semantics, shared.environment_cache_namespace)
        self._shared_distribution = shared
        self._within_factor = within_factor
        self._noise_cholesky = noise_cholesky

    def sample_joint(self, n_samples, generator=None, environment_noise_cache=None):
        draw = self._shared_distribution.sample_joint(
            n_samples, generator, environment_noise_cache)
        b, t, _ = self.mean.shape
        noise = torch.randn(n_samples, b, t, self._within_factor.shape[-1],
                            dtype=self.mean.dtype, device=self.mean.device, generator=generator)
        return draw + torch.einsum("dk,sbtk->sbtd", self._within_factor.to(draw.dtype), noise.to(draw.dtype))

    def _noise_solve(self, rhs):
        factor = self._within_factor.double()
        diagonal = self.diag_var[0, 0].double()
        direct = rhs / diagonal[:, None]
        statistic = torch.matmul(factor.T, direct)
        solved = torch.cholesky_solve(statistic, self._noise_cholesky)
        return direct - torch.matmul(factor / diagonal[:, None], solved)

    def log_prob(self, target_y, target_mask=None):
        observed = self.observed_mask(target_y, target_mask)
        if not bool(observed.all()):
            # Existing exact coordinate marginalisation handles arbitrary masks.
            return super().log_prob(target_y, target_mask)
        residual = (target_y - self.mean).double()
        factor = self._shared_distribution.factors.double()
        solved_residual = self._noise_solve(residual.unsqueeze(-1)).squeeze(-1)
        solved_factor = self._noise_solve(factor)
        rank = factor.shape[-1]
        core = torch.eye(rank, dtype=torch.float64, device=factor.device)
        core = core + torch.einsum("btdr,btds->brs", factor, solved_factor)
        cholesky = torch.linalg.cholesky(core)
        statistic = torch.einsum("btdr,btd->br", factor, solved_residual)
        solved = torch.cholesky_solve(statistic.unsqueeze(-1), cholesky).squeeze(-1)
        quadratic = (residual * solved_residual).sum((1, 2)) - (statistic * solved).sum(-1)
        t, d = self.mean.shape[1:]
        q_logdet = self.diag_var[0, 0].double().log().sum()
        q_logdet = q_logdet + 2 * self._noise_cholesky.diagonal().log().sum()
        logdet = t * q_logdet + 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -.5 * (t * d * math.log(2 * math.pi) + logdet + quadratic)


def conditional_to_joint_gaussian(conditional: ConditionalGaussian, *, affine=True,
                                 dtype=torch.float64, device="cpu", _noise=None):
    """Bridge a full-space baseline conditional to the evaluator's distribution.

    ``affine=True`` returns normalized coordinates, suitable for the existing
    scaler/evaluator. ``affine=False`` returns original measurement units.
    """
    baseline = conditional.model
    mean = ((conditional.mean - baseline.center) / baseline.scale
            if affine else conditional.mean)
    common = conditional.shared_factor * (1. if affine else baseline.scale[:, None])
    mean = _tensor(mean, dtype=dtype, device=device)
    factor = _tensor(common, dtype=dtype, device=device)
    factor = factor[None, None].expand(*mean.shape[:2], -1, -1)
    noise, diagonal, cholesky = (_noise if _noise is not None else
        _noise_parameters(baseline, affine=affine, dtype=dtype, device=device))
    shared = JointGaussian(mean, diagonal.expand_as(mean), factor, factor,
                           latent_semantics="train_fitted_full_space_closed_form_conditional")
    return _BlockNoiseGaussian(shared, noise, cholesky)


def repair_joint_gaussian(original: JointGaussian, conditional: ConditionalGaussian, *,
                          latent_rank: int, mode="mean_only", _noise=None):
    """Repair a normalized Gaussian while preserving declared shared components.

    The original distribution must use ``conditional.model`` affine coordinates.
    Adapters below verify that condition against the frozen training scaler.
    """
    if not isinstance(original, JointGaussian):
        raise TypeError("Moment repair requires an original Gaussian distribution")
    if mode not in {"mean_only", "mean_and_within"}:
        raise ValueError("Unknown moment repair mode")
    baseline = conditional.model
    mean = _tensor((conditional.mean - baseline.center) / baseline.scale,
                   dtype=original.mean.dtype, device=original.mean.device)
    if mean.shape != original.mean.shape:
        raise ValueError("Baseline conditional and original target shape differ")
    if mode == "mean_only":
        return replace(original, mean=mean)
    local = original.local_factors
    if (local is None or not isinstance(latent_rank, int) or latent_rank < 1
            or latent_rank > local.shape[-1]):
        raise ValueError("A declared compound latent rank and local factors are required")
    compound = local[..., :latent_rank]
    retained = torch.cat((compound, *_environment_marginals(original)), -1)
    noise, diagonal, cholesky = (_noise if _noise is not None else
        _noise_parameters(baseline, affine=True, dtype=mean.dtype, device=mean.device))
    shared = JointGaussian(mean, diagonal.expand_as(mean), retained, compound,
                           original.environment_loadings, original.environment_groups,
                           "frozen_compound_environment_with_train_fitted_within_covariance",
                           original.environment_cache_namespace)
    return _BlockNoiseGaussian(shared, noise, cholesky)


class ClosedFormGaussianAdapter(nn.Module):
    """Fixed slot-0 to slots-1/2/3 baseline compatible with predict_partition."""

    def __init__(self, baseline: ClosedFormBaseline, scaler):
        super().__init__()
        if (not np.array_equal(baseline.center, np.asarray(scaler.y_center))
                or not np.array_equal(baseline.scale, np.asarray(scaler.y_scale))):
            raise ValueError("Baseline and frozen scaler affine coordinates differ")
        train_ids = baseline.metadata.get("train_ids")
        if train_ids is not None and set(train_ids) != set(scaler.train_ids):
            raise ValueError("Baseline and frozen scaler training compounds differ")
        if baseline.n_slots != 4:
            raise ValueError("This adapter requires the four-role measurement protocol")
        self.baseline = baseline
        self._noise_cache = {}

    def _conditional(self, batch):
        if "target_y" in batch:
            raise ValueError("Future target measurements cannot be model inputs")
        y, mask = batch["context_y"], batch["context_mask"]
        if (y.ndim != 3 or y.shape[1:] != (1, self.baseline.feature_dim)
                or mask.shape != y.shape[:2] or mask.dtype != torch.bool
                or not bool(mask.all()) or not bool(torch.isfinite(y).all())):
            raise ValueError("Complete finite slot-0 context is required")
        for key in ("target_cond", "target_group"):
            if key in batch and batch[key].shape[:2] != (len(y), 3):
                raise ValueError("Targets must be the three declared future roles")
        raw = y.detach().cpu().double().numpy() * self.baseline.scale + self.baseline.center
        return self.baseline.conditional(raw, [0], [1, 2, 3])

    def _noise(self, reference):
        key = (reference.dtype, reference.device)
        if key not in self._noise_cache:
            self._noise_cache[key] = _noise_parameters(
                self.baseline, affine=True, dtype=reference.dtype, device=reference.device)
        return self._noise_cache[key]

    @torch.no_grad()
    def forward(self, batch):
        conditional = self._conditional(batch)
        y = batch["context_y"]
        return conditional_to_joint_gaussian(conditional, dtype=y.dtype, device=y.device,
                                            _noise=self._noise(y))


class MomentRepairModel(ClosedFormGaussianAdapter):
    """Evaluation adapter for a frozen fitted Gaussian model; no neural refit."""

    def __init__(self, base_model, baseline, scaler, mode="mean_only", latent_rank=None):
        super().__init__(baseline, scaler)
        if mode not in {"mean_only", "mean_and_within"}:
            raise ValueError("Unknown moment repair mode")
        self.base_model = base_model
        self.mode = mode
        self.latent_rank = getattr(base_model, "latent_rank", None) if latent_rank is None else latent_rank
        self.base_model.eval()

    @property
    def library_bank(self):
        return self.base_model.library_bank

    @torch.no_grad()
    def forward(self, batch):
        conditional = self._conditional(batch)
        self.base_model.eval()
        original = self.base_model(batch)
        return repair_joint_gaussian(original, conditional, latent_rank=self.latent_rank,
                                     mode=self.mode, _noise=(self._noise(original.mean)
                                     if self.mode == "mean_and_within" else None))
