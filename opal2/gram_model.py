"""Direct conditional Gaussian for the nine Schur--Cholesky Gram coordinates.

This model predicts a *joint distribution of Gram coordinates*, not a complete
future profile. Its only inputs are the full standardized initial profile and
the log norm of that same profile before standardization. Both the input scaler
and the target-coordinate scaler must be fitted on TRAIN by the caller.

The profile encoder is trained by target-coordinate MSE alone. Covariance NLL
sees detached encoder features and a detached conditional mean, so learning a
wide uncertainty cannot reduce the gradient used to fit the mean. Covariance
is full, positive definite, and weakly shrunk towards identity in standardized
target coordinates: Sigma = (1 - shrinkage) D R D + shrinkage I.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import math
from typing import Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .model import GroupedProfileEncoder


GRAM_DIM = 9


def log_profile_norm(raw_x: torch.Tensor, norm_floor: float = 1e-12) -> torch.Tensor:
    """Natural log Euclidean norm, using only the observed raw initial well.

    Returns one trailing descriptor coordinate, i.e. ``[..., 1]``. Zero-norm
    profiles receive the declared positive floor, not an inferred target norm.
    """
    if raw_x.ndim < 2 or not raw_x.is_floating_point():
        raise ValueError("raw_x must be a floating tensor with a feature dimension")
    if not math.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError("norm_floor must be positive and finite")
    if not torch.isfinite(raw_x).all():
        raise ValueError("raw_x must be finite")
    return torch.linalg.vector_norm(raw_x, dim=-1, keepdim=True).clamp_min(norm_floor).log()


@dataclass
class GramGaussianPrediction:
    """Gaussian in TRAIN-standardized nine-dimensional Gram coordinates.

    ``sample``/``rsample`` return ``[n_samples, batch, 9]``. The caller must undo
    the TRAIN target transform before decoding Gram geometry. In particular,
    the conditional mean here is not the mean Gram matrix after nonlinear
    decoding, and evaluating a utility at this mean is not its expectation.
    """

    mean: torch.Tensor
    scale_tril: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 2 or self.mean.shape[-1] != GRAM_DIM:
            raise ValueError("mean must have shape [batch, 9]")
        if self.scale_tril.shape != (*self.mean.shape, GRAM_DIM):
            raise ValueError("scale_tril must have shape [batch, 9, 9]")
        if self.mean.device != self.scale_tril.device or self.mean.dtype != self.scale_tril.dtype:
            raise ValueError("mean and scale_tril must share dtype and device")

    @property
    def lower_cholesky(self) -> torch.Tensor:
        return self.scale_tril

    @property
    def covariance_matrix(self) -> torch.Tensor:
        return self.scale_tril @ self.scale_tril.transpose(-1, -2)

    @property
    def variance(self) -> torch.Tensor:
        return self.scale_tril.square().sum(-1)

    @property
    def stddev(self) -> torch.Tensor:
        return self.variance.sqrt()

    def log_prob(self, target: torch.Tensor, *, detach_mean: bool = False) -> torch.Tensor:
        """Exact joint log density, one scalar per object, not marginal NLLs."""
        if target.shape != self.mean.shape or not torch.isfinite(target).all():
            raise ValueError("target must be finite with shape [batch, 9]")
        mean = self.mean.detach() if detach_mean else self.mean
        centered = (target - mean).unsqueeze(-1)
        whitened = torch.linalg.solve_triangular(self.scale_tril, centered, upper=False)
        mahalanobis = whitened.squeeze(-1).square().sum(-1)
        logdet = 2 * self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -.5 * (GRAM_DIM * math.log(2 * math.pi) + logdet + mahalanobis)

    def rsample(self, n_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        epsilon = torch.randn((n_samples, *self.mean.shape), device=self.mean.device,
                              dtype=self.mean.dtype, generator=generator)
        return self.mean.unsqueeze(0) + torch.einsum("bij,sbj->sbi", self.scale_tril, epsilon)

    @torch.no_grad()
    def sample(self, n_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.rsample(n_samples, generator)


class GramConditionalModel(nn.Module):
    """Coordinate-complete initial-well encoder with mean/covariance separation.

    The output order is fixed by the geometry implementation as ``p[0:3]``
    followed by ``log L00, L10, log L11, L20, L21, log L22``. The caller provides
    these coordinates standardized using TRAIN only. No chemical, JEPA,
    reference-panel, library, future-well or learned biological-kernel inputs
    are accepted by this class.

    Marginal log standard deviations of the *unshrunk* Gaussian lie in
    ``[-log_std_bound, log_std_bound]``. A unit-diagonal lower triangular matrix
    defines a full correlation matrix by row normalization. Shrinking the full
    covariance towards identity yields an eigenvalue floor of ``shrinkage``.
    The initial conditional covariance is exactly identity for every object.
    """

    def __init__(self, feature_groups: Mapping[str, Sequence[int]], hidden_dim: int = 64,
                 attention_heads: int = 4, attention_layers: int = 2,
                 covariance_shrinkage: float = .05, log_std_bound: float = 4.):
        super().__init__()
        if not math.isfinite(covariance_shrinkage) or not 0 < covariance_shrinkage < 1:
            raise ValueError("covariance_shrinkage must be strictly between zero and one")
        if not math.isfinite(log_std_bound) or not 0 < log_std_bound <= 10:
            raise ValueError("log_std_bound must be finite in (0, 10]")
        self.profile_encoder = GroupedProfileEncoder(
            feature_groups, hidden_dim, attention_layers=attention_layers,
            attention_heads=attention_heads)
        self.feature_dim = self.profile_encoder.feature_dim
        self.hidden_dim = hidden_dim
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.log_std_bound = float(log_std_bound)
        self.mean_features = nn.Sequential(nn.Linear(hidden_dim + 1, hidden_dim),
                                           nn.GELU(), nn.LayerNorm(hidden_dim))
        self.mean_head = nn.Linear(hidden_dim, GRAM_DIM)
        self.covariance_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                             nn.LayerNorm(hidden_dim),
                                             nn.Linear(hidden_dim, GRAM_DIM * (GRAM_DIM + 1) // 2))
        nn.init.zeros_(self.covariance_head[-1].weight)
        nn.init.zeros_(self.covariance_head[-1].bias)
        lower = torch.tril_indices(GRAM_DIM, GRAM_DIM, offset=-1)
        self.register_buffer("correlation_rows", lower[0])
        self.register_buffer("correlation_cols", lower[1])
        self.register_buffer("identity", torch.eye(GRAM_DIM))
        self.config = {
            "feature_groups": self.profile_encoder.feature_groups,
            "hidden_dim": hidden_dim,
            "attention_heads": attention_heads,
            "attention_layers": attention_layers,
            "covariance_shrinkage": self.covariance_shrinkage,
            "log_std_bound": self.log_std_bound,
        }

    def mean_parameters(self) -> Iterator[nn.Parameter]:
        return chain(self.profile_encoder.parameters(), self.mean_features.parameters(),
                     self.mean_head.parameters())

    def covariance_parameters(self) -> Iterator[nn.Parameter]:
        return self.covariance_head.parameters()

    def _covariance(self, detached_features: torch.Tensor) -> torch.Tensor:
        params = self.covariance_head(detached_features)
        log_std = self.log_std_bound * torch.tanh(params[..., :GRAM_DIM])
        unit_lower = self.identity.expand(len(params), -1, -1).clone()
        unit_lower[:, self.correlation_rows, self.correlation_cols] = params[..., GRAM_DIM:]
        normalized = unit_lower / torch.linalg.vector_norm(unit_lower, dim=-1, keepdim=True)
        correlation = normalized @ normalized.transpose(-1, -2)
        std = log_std.exp()
        covariance = std.unsqueeze(-1) * correlation * std.unsqueeze(-2)
        return ((1 - self.covariance_shrinkage) * covariance
                + self.covariance_shrinkage * self.identity)

    def forward(self, x_standardized: torch.Tensor,
                log_raw_norm: torch.Tensor) -> GramGaussianPrediction:
        if x_standardized.ndim != 2 or x_standardized.shape[-1] != self.feature_dim:
            raise ValueError("x_standardized must have shape [batch, feature_dim]")
        if not x_standardized.is_floating_point() or not torch.isfinite(x_standardized).all():
            raise ValueError("x_standardized must be finite floating values")
        if log_raw_norm.shape == x_standardized.shape[:1]:
            log_raw_norm = log_raw_norm.unsqueeze(-1)
        if log_raw_norm.shape != (len(x_standardized), 1) or not torch.isfinite(log_raw_norm).all():
            raise ValueError("log_raw_norm must be finite with shape [batch, 1] or [batch]")
        if log_raw_norm.dtype != x_standardized.dtype or log_raw_norm.device != x_standardized.device:
            raise ValueError("Both inputs must share dtype and device")
        profile = self.profile_encoder(x_standardized)
        features = self.mean_features(torch.cat((profile, log_raw_norm), -1))
        mean = self.mean_head(features)
        covariance = self._covariance(features.detach())
        return GramGaussianPrediction(mean, torch.linalg.cholesky(covariance))

    def loss(self, x_standardized: torch.Tensor, log_raw_norm: torch.Tensor,
             target_standardized: torch.Tensor) -> dict[str, torch.Tensor]:
        """Faithful loss with equal per-coordinate scaling for its two terms.

        Targets are standardized ``u``, not standardized original profiles.
        The returned ``joint_nll`` is per object; ``covariance_nll`` divides it
        by nine before combining it with coordinate-average ``mean_mse``.
        If clipping gradients, clip mean and covariance parameter sets
        separately: global clipping would indirectly couple their updates.
        """
        prediction = self(x_standardized, log_raw_norm)
        joint_nll = -prediction.log_prob(target_standardized, detach_mean=True).mean()
        mean_mse = F.mse_loss(prediction.mean, target_standardized)
        covariance_nll = joint_nll / GRAM_DIM
        return {"loss": mean_mse + covariance_nll, "mean_mse": mean_mse,
                "covariance_nll": covariance_nll, "joint_nll": joint_nll}
