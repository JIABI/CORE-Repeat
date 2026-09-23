"""Structured response-basis interaction operators.

The measurement bank is an inductive bias, not an asserted biological law.
``kernel`` means response basis here, not a positive-definite RKHS kernel.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


DESCRIPTOR_NAMES = (
    "log_context_rms", "condition_distance", "reference_distance",
    "reference_availability", "observed_count_over_one_plus_count", "encoded_reference_similarity",
)

LAW_DESCRIPTOR_NAMES = (
    "model_signal_to_noise", "model_repeat_error_correlation", "observed_repeat_count",
    "observed_n_cells", "n_cells_available", "condition_distance", "reference_distance",
    "reference_pair_available", "latent_posterior_uncertainty",
)


class MeasurementLawBasis(nn.Module):
    """Explicit conditional measurement-statistics responses.

    rho and r are model-implied parameters, NOT identified estimates from a
    single well. The equicorrelated Gaussian formulas are inductive responses;
    final utility always uses the unrestricted decoded joint covariance. The
    inverse-cell-count law is used only with measured, available cell counts.
    Cost is excluded so changing a price does not retrain a measurement model.
    """
    size = 22

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        if d.shape[-1] != len(LAW_DESCRIPTOR_NAMES):
            raise ValueError("Nine named measurement-law descriptors are required")
        rho = d[..., 0].clamp_min(0)
        k = d[..., 2].clamp_min(1)
        # The full pair covariance may be negative. Its equicorrelation-bank
        # projection must remain PSD for every displayed count up to k+2;
        # this clipping affects this basis only, never the decoded covariance.
        r = torch.maximum(d[..., 1].clamp(max=1), -1/(k+1)+1e-5)
        n, nv = d[..., 3].clamp_min(1), d[..., 4].clamp(0, 1)
        cd, rd = d[..., 5].clamp_min(0), d[..., 6].clamp_min(0)
        rv, uncertainty = d[..., 7].clamp(0, 1), d[..., 8].clamp_min(0)
        def keff(count):
            return count / (1 + (count - 1) * r)
        def consistency(count):
            return rho / torch.sqrt((rho + 1 / keff(count)) * (rho + 1))
        eff = keff(k)
        c0, c1, c2 = consistency(k), consistency(k + 1), consistency(k + 2)
        gain1, gain2 = .5 * (c1 - c0), .5 * (c2 - c0)
        snr = rho / (1 + rho)
        return torch.stack((torch.ones_like(rho), snr, snr.square(), 1 - snr,
                            r, 1 - r, 1 / eff, eff / (1 + eff),
                            c0, c1, c2, gain1, gain2, gain2 - gain1,
                            nv / n, nv / n.sqrt(), nv,
                            rv * torch.exp(-rd), rv, cd / (1 + cd),
                            rd / (1 + rd), uncertainty / (1 + uncertainty)), -1)


class SplineKANLinear(nn.Module):
    """Learned univariate cubic B-splines on every input-output edge.

    Inputs are smoothly mapped to (-1, 1); a linear SiLU branch allows trends
    outside any individual localized spline response.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8,
                 degree: int = 3):
        super().__init__()
        if min(in_features, out_features, grid_size) < 1 or degree < 0:
            raise ValueError("Invalid spline dimensions")
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.degree = grid_size, degree
        knots = torch.arange(-degree, grid_size + degree + 1).float()
        self.register_buffer("knots", -1.0 + knots * (2.0 / grid_size))
        self.base = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features,
                                                       grid_size + degree))
        nn.init.normal_(self.spline_weight, std=0.03 / math.sqrt(in_features))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x).unsqueeze(-1)
        knots = self.knots.to(dtype=x.dtype)
        basis = ((x >= knots[:-1]) & (x < knots[1:])).to(x.dtype)
        for p in range(1, self.degree + 1):
            left = (x - knots[:-(p + 1)]) / (knots[p:-1] - knots[:-(p + 1)])
            right = (knots[p + 1:] - x) / (knots[p + 1:] - knots[1:-p])
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError("KAN input dimension mismatch")
        return self.base(F.silu(x)) + torch.einsum("...ik,oik->...o", self.basis(x),
                                                  self.spline_weight)


class MeasurementBasis(nn.Module):
    """Bounded responses of six observed/declared measurement descriptors.

    Columns 0--3 represent smooth observed energy trends (not signal-to-noise);
    4--7 allow similarity to vary smoothly with condition/reference distance;
    8--9 expose reference availability and n_observed/(1+n_observed);
    10--11 are encoded-reference similarity and curvature; 12--15 are gated
    interactions between availability, distances, energy and context amount.
    Exponential distance decay is an optional smoothness prior, not a known
    biological law. Covariance propagation is implemented in JointGaussian,
    without deriving an unidentified correlation parameter from these proxies.
    """
    size = 16

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        if d.shape[-1] != len(DESCRIPTOR_NAMES):
            raise ValueError("Six named measurement descriptors are required")
        energy = torch.clamp(d[..., 0], min=0)
        condition_distance = torch.clamp(d[..., 1], min=0)
        reference_distance = torch.clamp(d[..., 2], min=0)
        available = d[..., 3].clamp(0, 1)
        fraction = d[..., 4].clamp(0, 1)
        similarity = d[..., 5].clamp(-1, 1)
        e = energy / (1 + energy)
        c = condition_distance / (1 + condition_distance)
        r = reference_distance / (1 + reference_distance)
        return torch.stack((torch.ones_like(e), e, e.square(), 1 - e,
                            torch.exp(-condition_distance), c,
                            torch.exp(-reference_distance), r, available,
                            fraction, similarity, similarity.square(),
                            available * torch.exp(-reference_distance), c * r,
                            e * torch.exp(-condition_distance),
                            fraction * (1 - e)), dim=-1)


class MeasurementKernelOperator(nn.Module):
    """A vector response-bank mixture with a trainable spline/KAN mixing head.

    All modes consume identical context, query and descriptor inputs. The MLP
    and generic modes are actual architectural ablations, not matched-parameter
    claims. A learned vector lift avoids a one-scalar message bottleneck.
    """
    def __init__(self, hidden_dim: int, mode: str = "measurement", basis_size: int = 16,
                 descriptor_kind: str = "legacy"):
        super().__init__()
        if mode not in {"measurement", "generic", "mlp"}:
            raise ValueError("kernel mode must be measurement, generic or mlp")
        self.mode = mode
        if descriptor_kind not in {"legacy", "laws"}:
            raise ValueError("Unknown descriptor bank")
        self.descriptor_kind = descriptor_kind
        n_descriptors = len(LAW_DESCRIPTOR_NAMES if descriptor_kind == "laws" else DESCRIPTOR_NAMES)
        bank_class = MeasurementLawBasis if descriptor_kind == "laws" else MeasurementBasis
        inputs = 2 * hidden_dim + n_descriptors
        self.residual = nn.Sequential(nn.Linear(inputs, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        if mode != "mlp":
            self.basis_size = bank_class.size if mode == "measurement" else basis_size
            self.coefficients = nn.Sequential(
                SplineKANLinear(n_descriptors, 24), nn.Tanh(),
                SplineKANLinear(24, self.basis_size))
            self.context_coefficients = nn.Linear(2 * hidden_dim, self.basis_size)
            self.lift = nn.Linear(self.basis_size, hidden_dim, bias=False)
            if mode == "measurement":
                self.bank = bank_class()
            else:
                self.centers = nn.Parameter(torch.randn(self.basis_size,
                                                        n_descriptors) * 0.4)
                self.log_width = nn.Parameter(torch.zeros(self.basis_size))

    def forward(self, context: torch.Tensor, query: torch.Tensor,
                descriptors: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((context, query, descriptors), dim=-1)
        residual = self.residual(joined)
        if self.mode == "mlp":
            return residual
        if self.mode == "measurement":
            basis = self.bank(descriptors)
        else:
            delta = torch.tanh(descriptors).unsqueeze(-2) - self.centers
            width = F.softplus(self.log_width) + 1e-3
            basis = torch.exp(-0.5 * delta.square().sum(-1) / width.square())
        logits = self.coefficients(descriptors)
        logits = logits + self.context_coefficients(torch.cat((context, query), dim=-1))
        coefficients = torch.tanh(logits)
        coefficients = coefficients / (1 + coefficients.abs().sum(-1, keepdim=True))
        return self.lift(coefficients * basis) + residual
