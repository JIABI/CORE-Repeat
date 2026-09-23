"""Mean-faithful training routes for the complete conditional Gaussian model.

The original decision-time mean backbone is retained, including its latent
precision update and uncertainty-derived descriptors. Those quantities can
change its mean, so their parameters cannot also be updated by a purportedly
independent uncertainty loss. A separate, fully structured covariance decoder
is initialized as an exact copy of the original decoder's covariance heads.

This is an explicitly untied conditional-Gaussian variant, not an unchanged
chemical generative latent model. Its decoder remains linear in its Gaussian
random variables, so the deployed mean and joint likelihood are exact; no
target-informed variational posterior or ELBO surrogate is used here.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import JointGaussian, MeasurementWorldModel


class IndependentCovarianceDecoder(nn.Module):
    """Complete compound/source/batch/plate/within/diagonal covariance heads."""

    def __init__(self, original: MeasurementWorldModel):
        super().__init__()
        self.factor_heads = deepcopy(original.factor_heads)
        self.scale_head = deepcopy(original.scale_head)
        self.residual_head = deepcopy(original.residual_head)
        self.latent_rank = original.latent_rank
        self.residual_rank = original.residual_rank
        self.feature_dim = original.feature_dim

    def forward(self, mean, state, posterior_var, group):
        b, t, _ = state.shape
        load = self.factor_heads[0](state) / math.sqrt(self.latent_rank)
        compound = load * posterior_var.sqrt()[:, None, None, :]
        parts, environmental = [compound], []
        for level in range(3):
            factor = self.factor_heads[level + 1](state) / math.sqrt(self.latent_rank)
            environmental.append(factor)
            same = torch.ones((b, t, t), dtype=torch.bool, device=state.device)
            for ancestor in range(level + 1):
                ids = group[..., ancestor]
                same = same & (ids.unsqueeze(-1) == ids.unsqueeze(-2)) & (ids.unsqueeze(-1) >= 0)
            same = same | torch.eye(t, dtype=torch.bool, device=state.device).unsqueeze(0)
            assignment = F.one_hot(same.to(torch.int64).argmax(-1), t).to(state.dtype)
            parts.append((factor.unsqueeze(-2) * assignment[:, :, None, :, None])
                         .reshape(b, t, self.feature_dim, t * self.latent_rank))
        residual = self.residual_head(state) / math.sqrt(self.residual_rank)
        assignment = torch.eye(t, device=state.device, dtype=state.dtype)
        residual = (residual.unsqueeze(-2) * assignment[None, :, None, :, None])
        residual = residual.reshape(b, t, self.feature_dim, t * self.residual_rank)
        parts.append(residual)
        return JointGaussian(mean, self.scale_head(state).square(), torch.cat(parts, -1),
                             torch.cat((compound, residual), -1), tuple(environmental), group,
                             "untied_faithful_conditional_gaussian_hierarchical_factors")


class FaithfulMeasurementModel(nn.Module):
    """Matched J/M/F wrapper with disjoint mean and uncertainty parameters.

    J (``joint``): exact joint predictive NLL updates both branches.
    M (``mean_only``): physical-coordinate MSE updates the complete mean branch.
    F (``faithful``): the same MSE updates the mean branch; exact joint NLL sees
    detached mean, state and posterior variance and updates covariance only.

    Mean and uncertainty gradients must be clipped separately by the caller.
    Matching M/F optimizer states, minibatches, learning rates and stochastic
    streams is required for their mean trajectories to remain identical.
    All auxiliary reference/KL losses are disabled, but reference, chemistry,
    library, latent and measurement-kernel inputs remain in the full backbone.
    """

    def __init__(self, mean_model: MeasurementWorldModel):
        super().__init__()
        if not isinstance(mean_model, MeasurementWorldModel):
            raise TypeError("A complete MeasurementWorldModel is required")
        if mean_model.observation_family != "gaussian":
            raise ValueError("This declared J/M/F comparison uses joint Gaussian observations")
        if mean_model.reference_loss_weight != 0 or mean_model.chemical_regularization_weight != 0:
            raise ValueError("Declare reference_loss_weight=0 and chemical_regularization_weight=0 in all three arms")
        self.mean_model = mean_model
        self.covariance_decoder = IndependentCovarianceDecoder(mean_model)

    @property
    def config(self):
        return {"wrapper": "FaithfulMeasurementModel", "version": 1,
                "mean_model_config": deepcopy(self.mean_model.config),
                "mean_covariance_loading_tied": False,
                "reference_auxiliary_loss": 0., "chemical_kl_auxiliary_loss": 0.}

    @classmethod
    def from_config(cls, config):
        if config.get("wrapper") != "FaithfulMeasurementModel" or config.get("version") != 1:
            raise ValueError("Unknown faithful model configuration")
        return cls(MeasurementWorldModel(**config["mean_model_config"]))

    @property
    def outcome_scale(self):
        return self.mean_model.outcome_scale

    @property
    def outcome_center(self):
        return self.mean_model.outcome_center

    @property
    def feature_dim(self):
        return self.mean_model.feature_dim

    @property
    def latent_rank(self):
        return self.mean_model.latent_rank

    @property
    def observation_family(self):
        return "gaussian"

    def set_outcome_transform(self, center, scale):
        self.mean_model.set_outcome_transform(center, scale)

    def mean_parameters(self):
        return self.mean_model.parameters()

    def uncertainty_parameters(self):
        return self.covariance_decoder.parameters()

    def _mean_from_details(self, details):
        # Exact deployed mean of the existing decoder. Nonlinear functions of
        # legal context occur inside details, but the random latent is linear.
        model, state = self.mean_model, details["state"]
        mean = model.mean_head(state)
        if model.identity_transport:
            mean = mean + model.transport_scale(details["query"]) * details["raw_transport"]
        loading = model.factor_heads[0](state) / math.sqrt(model.latent_rank)
        return mean + torch.einsum("btdr,br->btd", loading, details["posterior_mean"])

    def exact_predictive_mean(self, batch):
        """Return E[Y_future|legal context] in the model's affine coordinates."""
        return self._mean_from_details(self.mean_model._state_details(batch))

    def _distribution(self, batch, details, mean, detach_covariance_inputs=False):
        state, variance = details["state"], details["posterior_var"]
        if detach_covariance_inputs:
            state, variance = state.detach(), variance.detach()
        return self.covariance_decoder(mean, state, variance, batch["target_group"])

    def forward(self, batch, detach_covariance_inputs=False):
        details = self.mean_model._state_details(batch)
        mean = self._mean_from_details(details)
        return self._distribution(batch, details, mean, detach_covariance_inputs)

    def sample_joint(self, batch, n_samples, generator=None, environment_noise_cache=None):
        return self(batch).sample_joint(n_samples, generator, environment_noise_cache)

    @staticmethod
    def _observed(mean, target_y, target_mask):
        if target_y.shape != mean.shape:
            raise ValueError("Target measurement shape mismatch")
        observed = torch.isfinite(target_y)
        if target_mask is not None:
            if target_mask.dtype != torch.bool:
                raise ValueError("Target masks must be boolean")
            if target_mask.shape == mean.shape[:-1]:
                target_mask = target_mask.unsqueeze(-1)
            elif target_mask.shape != mean.shape:
                raise ValueError("Target mask must index wells or coordinates")
            observed = observed & target_mask
        if not observed.any():
            raise ValueError("At least one observed target coordinate is required")
        return observed

    def loss(self, batch, target_y, target_mask=None, *, mode="faithful"):
        if mode not in {"joint", "mean_only", "faithful"}:
            raise ValueError("Mode must be joint, mean_only, or faithful")
        details = self.mean_model._state_details(batch)
        mean = self._mean_from_details(details)
        observed = self._observed(mean, target_y, target_mask)
        count = observed.sum()
        clean = torch.where(observed, target_y, mean.detach())
        error = torch.where(observed, (mean - clean) * self.outcome_scale,
                            torch.zeros_like(mean))
        mean_mse = error.square().sum() / count
        output = {"mean_mse": mean_mse, "physical_mean_mse": mean_mse,
                  "observed_coordinates": count, "predictive_mean": mean,
                  "reference_reconstruction_in_objective": False,
                  "chemical_kl_in_objective": False}
        if mode == "mean_only":
            # M is intentionally mean-only: no untrained uncertainty score is
            # returned as if it were part of this training objective.
            return {**output, "loss": mean_mse, "nll": None,
                    "standardized_joint_nll": None, "joint_log_prob": None}
        detach = mode == "faithful"
        distribution = self._distribution(batch, details, mean.detach() if detach else mean, detach)
        logp = distribution.joint_log_prob(target_y, target_mask)
        affine_nll = -logp / count
        log_jacobian = torch.where(observed, self.outcome_scale.log().expand_as(mean),
                                   torch.zeros_like(mean)).sum()
        physical_nll = (-logp + log_jacobian) / count
        return {**output, "loss": mean_mse + physical_nll if detach else physical_nll,
                "nll": physical_nll, "standardized_joint_nll": affine_nll,
                "joint_log_prob": logp,
                "faithful_nll": physical_nll if detach else None}
