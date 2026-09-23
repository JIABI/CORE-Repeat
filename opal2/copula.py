"""Normalized Student-t4 marginal / Gaussian-copula measurement distributions.

The current full-coordinate hierarchical Gaussian supplies the copula, NOT an
additive covariance decomposition of the transformed observations. Each
coordinate has a variance-matched t4 marginal. Source/batch/plate and repeat
dependence is retained by transforming the SAME joint Gaussian draw. There is
no global inverse-Gamma radius that rescales an entire batch after one outlier.

All density operations include the change-of-variables Jacobian and use stable
float64 tail arithmetic. Probe conditioning takes place in the Gaussian space
while the ORIGINAL marginal maps remain fixed. Conditional original-space
moments need numerical integration; transforming a Gaussian conditional mean
is not a valid shortcut. No truncated target or clipped NLL is introduced.
"""
from __future__ import annotations

import math
from functools import lru_cache
from numbers import Integral
from typing import Sequence

import numpy as np
import torch

from .model import JointGaussian, LazyConditionalGaussian


_LOG_TWO_PI = math.log(2 * math.pi)
_LOG_FOUR = math.log(4.)
_LOG_THREE = math.log(3.)


def _normal_logpdf(z: torch.Tensor) -> torch.Tensor:
    return -.5 * (z.square() + _LOG_TWO_PI)


def t4_logpdf(x: torch.Tensor) -> torch.Tensor:
    """Standard Student-t4 log density, avoiding x**2 overflow."""
    x = x.double()
    return math.log(3 / 8) - 2.5 * (2 * torch.hypot(x, x.new_tensor(2.)).log() - _LOG_FOUR)


def _normal_lower_quantile_logp(logp: torch.Tensor) -> torch.Tensor:
    """Normal inverse lower-tail CDF from log(p), for p <= 1/2.

    The extreme-tail branch solves log Phi(z) = log(p) directly, without
    converting an underflowing probability into zero. This is used inside a
    custom autograd map whose derivative follows the exact density ratio.
    """
    ordinary = logp >= -700
    if ordinary.all():
        return torch.special.ndtri(logp.exp())
    result = torch.empty_like(logp)
    if ordinary.any():
        result[ordinary] = torch.special.ndtri(logp[ordinary].exp())
    extreme_logp = logp[~ordinary]
    z = -torch.sqrt(-2 * extreme_logp)
    for _ in range(12):
        logcdf = torch.special.log_ndtr(z)
        derivative = torch.exp(_normal_logpdf(z) - logcdf)
        z = z - (logcdf - extreme_logp) / derivative
    result[~ordinary] = z
    return result


def _t4_to_normal_value(x: torch.Tensor) -> torch.Tensor:
    absolute = x.abs()
    central_mask = absolute <= 1
    result = torch.empty_like(x)
    if central_mask.any():
        central_x = x[central_mask]
        u = central_x / torch.hypot(central_x, central_x.new_tensor(2.))
        result[central_mask] = torch.special.ndtri(.5 + .75 * u - .25 * u.pow(3))
    if central_mask.all():
        return result
    tail_x = x[~central_mask]
    tail_absolute = absolute[~central_mask]
    hyp = torch.hypot(tail_absolute, tail_absolute.new_tensor(2.))
    # r = 1-|x|/sqrt(x*x+4), calculated without catastrophic cancellation.
    logr = _LOG_FOUR - 2 * hyp.log() - torch.log1p(tail_absolute / hyp)
    # sf(|x|) = r*r*(3-r)/4, including tails far below float64's tiny.
    logsf = 2 * logr + torch.log(3 - logr.exp()) - _LOG_FOUR
    result[~central_mask] = -tail_x.sign() * _normal_lower_quantile_logp(logsf)
    return result


def _normal_to_t4_value(z: torch.Tensor) -> torch.Tensor:
    central_mask = z.abs() <= 1
    result = torch.empty_like(z)
    if central_mask.any():
        central_z = z[central_mask]
        u = 2 * torch.sin(torch.asin(2 * torch.special.ndtr(central_z) - 1) / 3)
        result[central_mask] = 2 * u / torch.sqrt(1 - u.square())
    if central_mask.all():
        return result
    # Work with the smaller normal tail. For |z|>1 its log is finite even when
    # exp(logp) underflows; no arbitrary clipping of CDF probabilities occurs.
    tail_z = z[~central_mask]
    absolute = tail_z.abs()
    logp = torch.special.log_ndtr(-absolute)
    logr = .5 * (logp + _LOG_FOUR - _LOG_THREE)
    for _ in range(8):
        r = logr.exp()
        error = 2 * logr + torch.log(3 - r) - _LOG_FOUR - logp
        derivative = 2 - r / (3 - r)
        logr = logr - error / derivative
    r = logr.exp()
    logx = math.log(2.) + torch.log1p(-r) - .5 * (logr + torch.log(2 - r))
    result[~central_mask] = tail_z.sign() * logx.exp()
    return result


class _T4ToNormal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        z = _t4_to_normal_value(x)
        ctx.save_for_backward(x, z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, z = ctx.saved_tensors
        # dz/dx = f_t4(x) / phi(z); stable in extreme tails and at zero.
        derivative = torch.exp(t4_logpdf(x) - _normal_logpdf(z))
        return grad_output * derivative


class _NormalToT4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        x = _normal_to_t4_value(z)
        ctx.save_for_backward(x, z)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, z = ctx.saved_tensors
        return grad_output * torch.exp(_normal_logpdf(z) - t4_logpdf(x))


def normal_from_t4(x: torch.Tensor) -> torch.Tensor:
    """Phi^-1(T4(x)), with exact first-order density-ratio gradients."""
    if not torch.is_tensor(x) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("t4 quantiles must be finite floating-point tensors")
    return _T4ToNormal.apply(x.double())


def t4_from_normal(z: torch.Tensor) -> torch.Tensor:
    """T4^-1(Phi(z)); uses log tails rather than clipped probabilities."""
    if not torch.is_tensor(z) or not z.is_floating_point() or not torch.isfinite(z).all():
        raise ValueError("Normal quantiles must be finite floating-point tensors")
    value = _NormalToT4.apply(z.double())
    if not torch.isfinite(value).all():
        raise FloatingPointError("A t4 inverse quantile exceeds float64 range; it was not clipped")
    return value


@lru_cache(maxsize=8)
def _hermite_rule(order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes * math.sqrt(2.), weights / math.sqrt(math.pi)


class StudentT4GaussianCopula:
    """Full measurement distribution with variance-matched t4 marginals.

    ``base`` is the context-only joint Gaussian. Public construction normally
    takes just this base. An already conditional Gaussian can only be wrapped
    with explicitly supplied ORIGINAL marginal maps; otherwise conditioning
    would silently change the generative model.

    ``joint_log_prob`` preserves cross-object environments. ``log_prob`` is
    each object's marginal density; summing it is not the joint cohort density.
    ``mean`` and ``marginal_variance`` after a probe use converged one-dimensional
    Gaussian-Hermite integration. Cheap shape/device/dtype properties avoid
    triggering that integration merely to allocate a planning tensor.
    """
    def __init__(self, base: JointGaussian | LazyConditionalGaussian, *,
                 location: torch.Tensor | None = None,
                 variance: torch.Tensor | None = None,
                 conditional: bool = False,
                 fixed_mask: torch.Tensor | None = None,
                 fixed_values: torch.Tensor | None = None,
                 moment_order: int = 32, moment_max_order: int = 256,
                 moment_rtol: float = 1e-6, moment_atol: float = 1e-8,
                 moment_chunk_size: int = 512):
        if not isinstance(base, (JointGaussian, LazyConditionalGaussian)):
            raise TypeError("Copula base must be the complete hierarchical Gaussian distribution")
        if (location is None) != (variance is None):
            raise ValueError("Original marginal location and variance must be supplied together")
        if isinstance(base, LazyConditionalGaussian) and location is None:
            raise ValueError("A conditional Gaussian requires the original marginal maps")
        if location is None:
            if conditional:
                raise ValueError("A conditional copula requires its original marginal maps")
            location, variance = base.mean, base.marginal_variance
        if (location.shape != base.mean.shape or variance.shape != location.shape
                or location.ndim != 3 or not torch.isfinite(location).all()
                or not torch.isfinite(variance).all() or torch.any(variance <= 0)):
            raise ValueError("Original marginal maps must be finite [B,T,D] with positive variances")
        if location.device != base.mean.device or variance.device != base.mean.device:
            raise ValueError("Copula maps and base must be on the same device")
        for name, value in (("moment_order", moment_order), ("moment_max_order", moment_max_order),
                            ("moment_chunk_size", moment_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if moment_order < 8 or moment_max_order < 2 * moment_order:
            raise ValueError("Conditional moments need at least two integration resolutions")
        if not (math.isfinite(moment_rtol) and math.isfinite(moment_atol)
                and moment_rtol > 0 and moment_atol > 0):
            raise ValueError("Moment tolerances must be positive and finite")
        self.base, self.location, self.variance = base, location, variance
        self.conditional_distribution = bool(conditional or isinstance(base, LazyConditionalGaussian))
        self.moment_order, self.moment_max_order = moment_order, moment_max_order
        self.moment_rtol, self.moment_atol = moment_rtol, moment_atol
        self.moment_chunk_size = moment_chunk_size
        self._fixed_mask = torch.zeros_like(location, dtype=torch.bool) if fixed_mask is None else fixed_mask
        self._fixed_values = location if fixed_values is None else fixed_values
        if (self._fixed_mask.shape != location.shape or self._fixed_mask.dtype != torch.bool
                or self._fixed_values.shape != location.shape
                or not torch.isfinite(self._fixed_values[self._fixed_mask]).all()):
            raise ValueError("Retained observed coordinates have invalid masks or values")
        self._moments_cache = None
        self._gaussian_variance_cache = None
        self._moment_diagnostics = None
        self.latent_semantics = "student_t4_marginals_hierarchical_Gaussian_copula_not_additive_Y_variance"

    @property
    def shape(self):
        return self.location.shape

    @property
    def device(self):
        return self.location.device

    @property
    def dtype(self):
        return self.location.dtype

    @property
    def mean(self):
        return self.location if not self.conditional_distribution else self._conditional_moments()[0]

    @property
    def marginal_variance(self):
        return self.variance if not self.conditional_distribution else self._conditional_moments()[1]

    @property
    def moment_diagnostics(self):
        if not self.conditional_distribution:
            return {"method": "analytic_t4_marginal", "converged": True}
        self._conditional_moments()
        return dict(self._moment_diagnostics)

    def _raw_observed_mask(self, y, mask=None):
        if y.shape != self.shape or y.device != self.device:
            raise ValueError("Observed targets must match copula shape and device")
        observed = torch.isfinite(y)
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("Target mask must be boolean")
            if mask.shape == y.shape[:-1]:
                mask = mask.unsqueeze(-1)
            elif mask.shape != y.shape:
                raise ValueError("Target mask must index wells or coordinates")
            observed = observed & mask
        overlap = observed & self._fixed_mask
        if torch.any(overlap & (y != self._fixed_values)):
            raise ValueError("New observations conflict with retained original-space point masses")
        return observed

    def observed_mask(self, target_y, target_mask=None):
        # Point masses already conditioned upon do not contribute a Lebesgue
        # density or a second Jacobian term.
        return self._raw_observed_mask(target_y, target_mask) & ~self._fixed_mask

    def gaussianize(self, y):
        """Transform complete original measurement coordinates to the base."""
        standard_t = (y.double() - self.location.double()) / torch.sqrt(self.variance.double() / 2)
        return self.location.double() + self.variance.double().sqrt() * normal_from_t4(standard_t)

    def degaussianize(self, g):
        """Inverse map for complete targets or leading Monte Carlo dimensions."""
        z = (g.double() - self.location.double()) / self.variance.double().sqrt()
        return self.location.double() + torch.sqrt(self.variance.double() / 2) * t4_from_normal(z)

    def log_abs_det_y_to_g(self, y):
        standard_t = (y.double() - self.location.double()) / torch.sqrt(self.variance.double() / 2)
        z = normal_from_t4(standard_t)
        # sqrt(v) / sqrt(v/2) = sqrt(2); both parameters' derivatives through
        # the standardized target are retained.
        return .5 * math.log(2.) + t4_logpdf(standard_t) - _normal_logpdf(z)

    def _density_inputs(self, y, mask):
        observed = self.observed_mask(y, mask)
        safe = torch.where(observed, y, self.location)
        standard_t = (safe.double() - self.location.double()) / torch.sqrt(self.variance.double() / 2)
        z = normal_from_t4(standard_t)
        gaussian = self.location.double() + self.variance.double().sqrt() * z
        jacobian = torch.where(observed,
            .5 * math.log(2.) + t4_logpdf(standard_t) - _normal_logpdf(z), 0.)
        return observed, gaussian, jacobian

    def joint_log_prob(self, target_y, target_mask=None):
        observed, gaussian, jacobian = self._density_inputs(target_y, target_mask)
        return self.base.joint_log_prob(gaussian, observed) + jacobian.sum()

    def log_prob(self, target_y, target_mask=None):
        observed, gaussian, jacobian = self._density_inputs(target_y, target_mask)
        if hasattr(self.base, "log_prob"):
            values = self.base.log_prob(gaussian, observed)
        else:
            # LazyConditionalGaussian integrates the full original evidence.
            # A per-object mask yields its marginal *conditional on all that
            # evidence*, not a separate independently refitted posterior.
            values = []
            for i in range(self.shape[0]):
                only_i = torch.zeros_like(observed)
                only_i[i] = observed[i]
                values.append(self.base.joint_log_prob(gaussian, only_i))
            values = torch.stack(values)
        return values + jacobian.sum(dim=(1, 2))

    def sample_joint(self, n_samples, generator=None, environment_noise_cache=None):
        gaussian = self.base.sample_joint(n_samples, generator, environment_noise_cache)
        value = self.degaussianize(gaussian)
        value = torch.where(self._fixed_mask.unsqueeze(0), self._fixed_values.unsqueeze(0), value)
        # Retain double precision: extreme but finite t values must not be
        # silently rounded to float32 infinities during utility evaluation.
        return value

    def condition(self, observed_y, observed_mask, target_indices: Sequence[int], *,
                  retain_observed=False, lazy=None):
        indices = tuple(target_indices)
        if (not indices or any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices)
                or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= self.shape[1]):
            raise ValueError("Target indices must be distinct valid integer well indices")
        observed = self._raw_observed_mask(observed_y, observed_mask)
        safe = torch.where(observed, observed_y, self.location)
        gaussian = self.gaussianize(safe)
        base = self.base.condition(gaussian, observed, indices,
                                   retain_observed=retain_observed, lazy=lazy)
        index = torch.tensor(indices, dtype=torch.int64, device=self.device)
        combined_mask = observed | self._fixed_mask
        combined_values = torch.where(observed, observed_y, self._fixed_values)
        return StudentT4GaussianCopula(base,
            location=self.location.index_select(1, index),
            variance=self.variance.index_select(1, index), conditional=True,
            fixed_mask=combined_mask.index_select(1, index),
            fixed_values=combined_values.index_select(1, index),
            moment_order=self.moment_order, moment_max_order=self.moment_max_order,
            moment_rtol=self.moment_rtol, moment_atol=self.moment_atol,
            moment_chunk_size=self.moment_chunk_size)

    def _gaussian_marginal_variance(self):
        if self._gaussian_variance_cache is not None:
            return self._gaussian_variance_cache
        if isinstance(self.base, JointGaussian):
            result = self.base.marginal_variance.double()
        else:
            p = self.base
            b, t, d = self.shape
            result = p.parent.diag_var.index_select(1, p.index).double().clone()
            global_rank = 0 if not p.specs else len(p.global_mean)
            for i in range(b):
                local = p.local[i].reshape(t * d, -1).double()
                local_std = torch.linalg.solve_triangular(p.chol_a[i], local.T, upper=False).T
                value = local_std.square().sum(-1)
                if p.specs:
                    by_chunk = []
                    for start in range(0, t * d, self.moment_chunk_size):
                        end = min(t * d, start + self.moment_chunk_size)
                        rows = torch.arange(start, end, device=self.device)
                        wells, coordinates = rows // d, rows % d
                        parent_wells = p.index[wells]
                        load = torch.zeros(len(rows), global_rank, dtype=torch.float64, device=self.device)
                        offset = 0
                        for source_load, group_index, n_groups in p.specs:
                            rank = source_load.shape[-1]
                            columns = (offset + group_index[i, parent_wells].unsqueeze(-1) * rank
                                       + torch.arange(rank, device=self.device))
                            values = source_load[i, parent_wells, coordinates].double()
                            load = load.scatter_add(1, columns, values)
                            offset += n_groups * rank
                        effective = load - local[start:end] @ p.solved_cross[i]
                        std = torch.linalg.solve_triangular(p.chol_k, effective.T, upper=False).T
                        by_chunk.append(std.square().sum(-1))
                    value = value + torch.cat(by_chunk)
                result[i] = result[i] + value.reshape(t, d)
            result = result.masked_fill(self._fixed_mask, 0.)
        if not torch.isfinite(result).all() or torch.any(result < -1e-9):
            raise FloatingPointError("Invalid conditional Gaussian marginal variance")
        self._gaussian_variance_cache = result.clamp_min(0.)
        return self._gaussian_variance_cache

    def _quadrature(self, order, zmean, zsd):
        nodes, weights = _hermite_rule(order)
        nodes = torch.as_tensor(nodes, device=self.device, dtype=torch.float64)
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float64)
        m, s = zmean.reshape(-1), zsd.reshape(-1)
        first, variance = [], []
        for start in range(0, len(m), self.moment_chunk_size):
            stop = min(start + self.moment_chunk_size, len(m))
            values = t4_from_normal(m[start:stop, None] + s[start:stop, None] * nodes)
            average = (values * weights).sum(-1)
            # Centered second moments avoid subtraction of two huge moments
            # when probe observations make a marginal almost deterministic.
            var = ((values - average[:, None]).square() * weights).sum(-1)
            first.append(average)
            variance.append(var)
        return torch.cat(first).reshape(self.shape), torch.cat(variance).reshape(self.shape)

    def _conditional_moments(self):
        if self._moments_cache is not None:
            return self._moments_cache
        gvar = self._gaussian_marginal_variance()
        zmean = (self.base.mean.double() - self.location.double()) / self.variance.double().sqrt()
        zsd = torch.sqrt(gvar / self.variance.double())
        old = self._quadrature(self.moment_order, zmean, zsd)
        order = self.moment_order
        converged = False
        maximum_error = None
        while order * 2 <= self.moment_max_order:
            order *= 2
            new = self._quadrature(order, zmean, zsd)
            errors = [torch.abs(a - b) / (self.moment_atol + self.moment_rtol * (1 + b.abs()))
                      for a, b in zip(old, new)]
            maximum_error = max(float(v.detach().max()) for v in errors)
            if maximum_error <= 1:
                converged = True
                old = new
                break
            old = new
        self._moment_diagnostics = {"method": "adaptive_order_gauss_hermite",
            "order": order, "converged": converged, "maximum_normalized_error": maximum_error,
            "rtol": self.moment_rtol, "atol": self.moment_atol}
        if not converged:
            raise RuntimeError(f"Conditional copula moments did not converge: {self._moment_diagnostics}")
        scale = torch.sqrt(self.variance.double() / 2)
        mean = self.location.double() + scale * old[0]
        variance = scale.square() * old[1]
        mean = torch.where(self._fixed_mask, self._fixed_values.double(), mean)
        variance = torch.where(self._fixed_mask, 0., variance)
        self._moments_cache = mean.to(self.dtype), variance.to(self.dtype)
        return self._moments_cache

    def weighted_moments(self, weights):
        raise NotImplementedError("Copula-transformed weighted sums are not Gaussian; aggregate original-space joint samples")

    def aggregate_wells(self, weights):
        raise NotImplementedError("Copula-transformed well averages are not a Gaussian/t4 family; aggregate original-space joint samples")
