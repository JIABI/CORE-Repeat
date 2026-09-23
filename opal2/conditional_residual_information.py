"""Predict geometric error scales, not a future realized error's sign.

The two projections are observable-sensitive geometry, not identified biological
or technical variance components. The empirical radial sampler remains external.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn

from .joint_contrast_scale import contrast_projector, projected_energy, rescale_components


DEGREES = np.array([3., 6.])
BOOST_CONFIG = dict(loss="gamma", learning_rate=.05, max_iter=150,
                    max_leaf_nodes=7, min_samples_leaf=20,
                    l2_regularization=10., early_stopping=False)


def error_targets(raw_mean, raw_covariance, raw_residual):
    """Use only the caller's honest training errors and their own fit geometry."""
    dec = contrast_projector(raw_mean, np.ones(9), raw_covariance)
    energy = projected_energy(raw_residual, dec)
    if not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError("Projection energies must be finite and positive; no target clipping")
    return energy


@dataclass
class ScalePredictor:
    models: list
    columns: np.ndarray
    fit_ids: list

    @classmethod
    def fit(cls, features, energy, columns, ids, seed=20260916):
        features, energy = np.asarray(features, float), np.asarray(energy, float)
        columns = np.asarray(columns, int)
        if features.ndim != 2 or energy.shape != (len(features), 2):
            raise ValueError("Expected aligned feature rows and two energies")
        if not np.isfinite(features).all() or not np.isfinite(energy).all() or np.any(energy <= 0):
            raise ValueError("Invalid fitting values")
        if columns.ndim != 1 or not len(columns) or len(ids) != len(features):
            raise ValueError("Feature selection or identities do not align")
        models = [HistGradientBoostingRegressor(**BOOST_CONFIG, random_state=seed+j).fit(
            features[:, columns], energy[:, j]/DEGREES[j]) for j in range(2)]
        return cls(models, columns.copy(), np.asarray(ids, str).tolist())

    def predict(self, features):
        x = np.asarray(features, float)[:, self.columns]
        result = np.column_stack([m.predict(x) for m in self.models])
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("Scale predictions must be positive and finite")
        return result


def bounded_scale_ratio(candidate, amplitude, *, enabled=True):
    candidate, amplitude = np.asarray(candidate, float), np.asarray(amplitude, float)
    if candidate.shape != amplitude.shape or candidate.ndim != 2 or candidate.shape[1] != 2:
        raise ValueError("Expected aligned N by two scale predictions")
    if not enabled:
        return np.ones_like(amplitude)
    if (not np.isfinite(candidate).all() or not np.isfinite(amplitude).all()
            or np.any(candidate <= 0) or np.any(amplitude <= 0)):
        raise ValueError("Invalid variance ratios")
    bound = np.log(4.)
    return np.exp(bound*np.tanh((np.log(candidate)-np.log(amplitude))/bound))


def extend_scatter(raw_mean, target_scale, core_scatter, ratios, *, enabled=True):
    """The disabled path is exact, with no refit, normalization or added noise."""
    if np.asarray(core_scatter).shape != (len(raw_mean), 9, 9):
        raise ValueError("Expected one nine-dimensional scatter per object")
    if not enabled:
        return np.asarray(core_scatter).copy()
    ratios = np.asarray(ratios, float)
    if ratios.shape != (len(raw_mean), 2):
        raise ValueError("Expected two variance ratios for each object")
    decomposition = contrast_projector(raw_mean, target_scale, core_scatter)
    return rescale_components(core_scatter, decomposition, ratios[:, 0], ratios[:, 1])


class ResidualStateNetwork(nn.Module):
    def __init__(self, input_dim, descriptor_dim, hidden_dim=32, latent_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, latent_dim))
        self.scale_head = nn.Linear(latent_dim+descriptor_dim, 2)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, inputs, descriptors, log_amplitude_scale):
        z = self.encoder(inputs)
        correction = 2*torch.tanh(self.scale_head(torch.cat((descriptors, z), -1))/2)
        return log_amplitude_scale+correction, z, correction


def projection_nll(log_variance, energies):
    degrees = torch.as_tensor(DEGREES, dtype=log_variance.dtype, device=log_variance.device)
    return .5*(degrees*log_variance+energies*torch.exp(-log_variance)).sum(1)


class LearnedResidualState:
    def __init__(self, network, input_center, input_scale, descriptor_center,
                 descriptor_scale, latent_center, latent_scale, report):
        self.network = network.eval().requires_grad_(False)
        self.input_center, self.input_scale = input_center, input_scale
        self.descriptor_center, self.descriptor_scale = descriptor_center, descriptor_scale
        self.latent_center, self.latent_scale = latent_center, latent_scale
        self.report = report

    def transform(self, inputs):
        x = (np.asarray(inputs, float)-self.input_center)/self.input_scale
        with torch.no_grad():
            z = self.network.encoder(torch.as_tensor(x, dtype=torch.float64)).numpy()
        return (z-self.latent_center)/self.latent_scale

    def save(self, path):
        torch.save(dict(network=self.network.state_dict(), report=self.report,
            input_center=torch.tensor(self.input_center), input_scale=torch.tensor(self.input_scale),
            descriptor_center=torch.tensor(self.descriptor_center), descriptor_scale=torch.tensor(self.descriptor_scale),
            latent_center=torch.tensor(self.latent_center), latent_scale=float(self.latent_scale)), path)


def fit_residual_state(inputs, descriptors, energies, amplitude_scale, ids, *, seed=20260916,
                       epochs=60, callback=None):
    """Fixed-length residual-distribution training; never receives query outcomes."""
    arrays = [np.asarray(v, float) for v in (inputs, descriptors, energies, amplitude_scale)]
    inputs, descriptors, energies, amplitude_scale = arrays
    n = len(inputs)
    if any(len(v) != n or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Training records must be aligned and finite")
    if (energies.shape != (n, 2) or amplitude_scale.shape != (n, 2)
            or np.any(amplitude_scale <= 0) or np.any(energies <= 0) or len(ids) != n):
        raise ValueError("Two positive amplitude scales and energies are required")
    if epochs != 60:
        raise ValueError("The recorded experiment specifies sixty encoder epochs")
    ic, isc = inputs.mean(0), inputs.std(0)
    dc, dsc = descriptors.mean(0), descriptors.std(0)
    isc, dsc = np.where(isc > 1e-8, isc, 1.), np.where(dsc > 1e-8, dsc, 1.)
    tensors = [torch.as_tensor(v, dtype=torch.float64) for v in
               ((inputs-ic)/isc, (descriptors-dc)/dsc, energies, np.log(amplitude_scale))]
    torch.manual_seed(seed)
    model = ResidualStateNetwork(inputs.shape[1], descriptors.shape[1]).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.001)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs+1):
        lr = .0003*(epoch/5 if epoch <= 5 else .5*(1+math.cos(math.pi*(epoch-5)/(epochs-5))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, 64):
            rows = order[start:start+64]
            eta, _, correction = model(tensors[0][rows], tensors[1][rows], tensors[3][rows])
            loss = projection_nll(eta, tensors[2][rows]).mean()+.05*correction.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite residual-state loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                eta, z, correction = model(tensors[0], tensors[1], tensors[3])
                row = dict(epoch=epoch, learning_rate=lr,
                    training_projection_nll=float(projection_nll(eta, tensors[2]).mean()),
                    amplitude_projection_nll=float(projection_nll(tensors[3], tensors[2]).mean()),
                    correction_rms=float(correction.square().mean().sqrt()),
                    latent_std=z.std(0).tolist())
            history.append(row)
            if callback:
                callback(row)
    with torch.no_grad():
        z = model.encoder(tensors[0]).numpy()
    latent_scale = max(float(np.sqrt(np.var(z, axis=0).mean())), 1e-8)
    report = dict(kind='conditional_residual_state', training_target='honest two-projection error likelihood',
        latent_dim=4, hidden_dim=32, epochs=epochs, batch_size=64, optimizer='AdamW',
        initial_lr=.0003, weight_decay=.001, gradient_clip=5., correction_penalty=.05,
        variance_floor_loss=False, moving_teacher=False, checkpoint_selection=False,
        fit_ids=np.asarray(ids, str).tolist(), history=history,
        parameter_count=sum(p.numel() for p in model.parameters()),
        input_dim=inputs.shape[1], descriptor_dim=descriptors.shape[1],
        latent_covariance_eigenvalues=np.linalg.eigvalsh(np.cov(z.T)).tolist())
    return LearnedResidualState(model, ic, isc, dc, dsc, z.mean(0), latent_scale, report)
