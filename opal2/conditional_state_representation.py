"""Optional, TRAIN-only, anchored conditional representation of repeat profiles.

This is a complete auxiliary representation estimator, not a replacement for the
frozen acquisition model and not a calibrated distribution.  A fixed PCA target
is fitted to INNER_FIT future profiles.  The student sees only X; role identifiers
are known target-slot conditions, not future measurements.  There is no moving
teacher, variance-floor penalty, or covariance-decorrelation loss.

``conditional_predictive`` predicts each future role through a shared conditional
predictor. ``direct`` is the matched-input ordinary nonlinear multi-output
comparator, with an unconstrained output for every role. Both use the same fixed
target, split, preprocessing, latent width, optimizer, and maximum epoch budget.
Their parameter counts are reported, not claimed to be equal.

Directions and log RMS are separate.  The fixed target comprises standardized
PCA coordinates of normalized future-profile directions plus log RMS.  Exported
embeddings contain the learned state and an explicit, unlearned X-amplitude
bypass. Normalization here defines auxiliary coordinates, never the OPAL endpoint.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from torch import nn


ARTIFACT_VERSION = 1


def _profiles(values, feature_count=None):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all()
            or (feature_count is not None and values.shape[1] != feature_count)):
        raise ValueError("Profiles must be finite, nonempty [objects, features]")
    return values


def _direction_amplitude(values):
    # A max-scaled norm avoids overflow without clipping legitimate amplitude.
    maximum = np.max(np.abs(values), axis=-1)
    scaled = values / np.where(maximum > 0, maximum, 1.)[:, None]
    scaled_rms = np.sqrt(np.mean(np.square(scaled), axis=-1))
    rms = maximum * scaled_rms
    direction = scaled / np.where(scaled_rms > 0, scaled_rms, 1.)[:, None]
    logamp = np.log(np.maximum(rms, np.finfo(np.float64).tiny))
    return direction, logamp, rms == 0


@dataclass
class _FixedCoordinates:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    coordinate_scale: np.ndarray
    amp_mean: float
    amp_scale: float

    @classmethod
    def fit(cls, values, dimension, seed):
        direction, amp, _ = _direction_amplitude(_profiles(values))
        mean, scale = direction.mean(0), direction.std(0)
        scale = np.where(scale > 1e-8, scale, 1.)
        standardized = (direction-mean)/scale
        dimensions = min(int(dimension), len(values)-1, values.shape[1])
        if dimensions < 1:
            raise ValueError("At least two INNER_FIT profile rows are required")
        pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed,
                  iterated_power=3)
        coordinates = pca.fit_transform(standardized)
        coordinate_scale = coordinates.std(0)
        # The fixed target does not inflate numerically zero PCA directions.
        coordinate_scale = np.where(coordinate_scale > 1e-7, coordinate_scale, 1.)
        amp_scale = float(amp.std())
        return cls(mean, scale, pca.mean_, pca.components_, coordinate_scale,
                   float(amp.mean()), amp_scale if amp_scale > 1e-8 else 1.)

    def transform(self, values):
        values = _profiles(values, len(self.feature_mean))
        direction, amp, _ = _direction_amplitude(values)
        shape = (((direction-self.feature_mean)/self.feature_scale-self.pca_mean)
                 @ self.components.T)/self.coordinate_scale
        return np.column_stack((shape, (amp-self.amp_mean)/self.amp_scale))

    def as_tensors(self):
        return {name: torch.as_tensor(value, dtype=torch.float64).clone()
                for name, value in vars(self).items()}

    @classmethod
    def from_tensors(cls, values):
        return cls(**{name: (float(value) if name.startswith("amp_")
                            else value.cpu().numpy().copy()) for name, value in values.items()})


class _Student(nn.Module):
    def __init__(self, input_dim, target_dim, latent_dim, hidden_dim, kind):
        super().__init__()
        self.kind = kind
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, latent_dim))
        if kind == "conditional_predictive":
            self.predictor = nn.Sequential(nn.Linear(latent_dim+1+3, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, target_dim))
        elif kind == "direct":
            self.predictor = nn.Sequential(nn.Linear(latent_dim+1, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, 3*target_dim))
        else:
            raise ValueError("kind must be 'conditional_predictive' or 'direct'")
        self.target_dim = target_dim

    def forward(self, inputs):
        latent = self.encoder(inputs)
        state = torch.cat((latent, inputs[:, -1:]), -1)
        if self.kind == "direct":
            return self.predictor(state).reshape(len(inputs), 3, self.target_dim)
        roles = torch.eye(3, dtype=inputs.dtype, device=inputs.device)
        conditional = torch.cat((state[:, None, :].expand(-1, 3, -1),
                                 roles[None].expand(len(inputs), -1, -1)), -1)
        return self.predictor(conditional)


def representation_reference_weights(reference_embedding, query_embedding, *, bandwidth,
                                     reference_ids=None, query_ids=None, reference_groups=None,
                                     query_groups=None, shrinkage_count=20.):
    """The same feature-only kernel for both arms; no reference outcomes enter.

    Pass matching IDs/groups to exclude self/chemical-group donors. In an isolated
    outer split those exclusions normally have no effect. Effective sample size
    controls shrinkage toward the eligible uniform reference distribution.
    """
    reference = _profiles(reference_embedding)
    query = _profiles(query_embedding, reference.shape[1])
    if not np.isfinite(bandwidth) or bandwidth <= 0 or not np.isfinite(shrinkage_count) or shrinkage_count < 0:
        raise ValueError("Invalid bandwidth or shrinkage_count")
    eligible = np.ones((len(query), len(reference)), dtype=bool)
    for left, right in ((reference_ids, query_ids), (reference_groups, query_groups)):
        if (left is None) != (right is None):
            raise ValueError("Reference and query identifiers must be supplied together")
        if left is not None:
            left, right = np.asarray(left).astype(str), np.asarray(right).astype(str)
            if left.shape != (len(reference),) or right.shape != (len(query),):
                raise ValueError("Identifiers must align with their profiles")
            eligible &= right[:, None] != left[None, :]
    if np.any(eligible.sum(1) == 0):
        raise ValueError("Every query needs at least one eligible reference")
    distance2 = cdist(query, reference, metric="sqeuclidean")/reference.shape[1]
    logits = np.where(eligible, -.5*distance2/bandwidth**2, -np.inf)
    local = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
    local_ess = 1/np.square(local).sum(1)
    shrinkage = local_ess/(local_ess+shrinkage_count)
    uniform = eligible/eligible.sum(1, keepdims=True)
    weights = shrinkage[:, None]*local+(1-shrinkage[:, None])*uniform
    return dict(weights=weights, local_ess=local_ess, ess=1/np.square(weights).sum(1),
                shrinkage=shrinkage, eligible_count=eligible.sum(1),
                nearest_distance=np.sqrt(np.where(eligible, distance2, np.inf).min(1)))


class ConditionalStateRepresentation:
    """Serializable optional estimator. All inference APIs accept X only."""
    def __init__(self, network, input_coordinates, target_coordinates, report,
                 embedding_center, embedding_scale, fit_embeddings, support_radius,
                 reference_bandwidth):
        self.network = network.eval().requires_grad_(False)
        self.input_coordinates, self.target_coordinates = input_coordinates, target_coordinates
        self.report = report
        self.embedding_center, self.embedding_scale = embedding_center, float(embedding_scale)
        self.fit_embeddings = fit_embeddings
        self.support_radius = float(support_radius)
        self.reference_bandwidth = float(reference_bandwidth)

    def _inputs(self, X):
        return torch.as_tensor(self.input_coordinates.transform(X), dtype=torch.float32)

    def transform(self, X):
        """Learned state (globally scaled) plus unchanged standardized log RMS."""
        inputs = self._inputs(X)
        with torch.no_grad():
            state = self.network.encoder(inputs).cpu().numpy().astype(np.float64)
        state = (state-self.embedding_center)/self.embedding_scale
        # A sqrt(latent width) block scaling gives shape and amplitude equal
        # total geometric weight; it does not individually whiten tiny latents.
        state /= np.sqrt(state.shape[1])
        return np.column_stack((state, inputs[:, -1].numpy().astype(np.float64)))

    def predict_targets(self, X):
        """Return conditional means of fixed target coordinates for three roles."""
        with torch.no_grad():
            return self.network(self._inputs(X)).cpu().numpy().astype(np.float64)

    def describe(self, X, ids=None):
        """Per-object embeddings and X-only support records (pandas not required)."""
        X = _profiles(X, len(self.input_coordinates.feature_mean))
        embedding = self.transform(X)
        names = np.arange(len(X)).astype(str) if ids is None else np.asarray(ids).astype(str)
        if names.shape != (len(X),):
            raise ValueError("ids must align with X")
        nearest = np.sqrt(cdist(embedding, self.fit_embeddings, "sqeuclidean").min(1)/embedding.shape[1])
        _, amp, zero = _direction_amplitude(X)
        fit_names = set(self.report["inner_fit_ids"])
        records = [dict(id=str(name), log_rms=float(amp[i]), zero_profile=bool(zero[i]),
                        nearest_fit_distance=float(nearest[i]),
                        within_fit_support=bool(nearest[i] <= self.support_radius),
                        is_inner_fit_object=bool(str(name) in fit_names),
                        amplitude_outside_fit_range=bool(amp[i] < self.report["fit_logamp_min"]
                                                        or amp[i] > self.report["fit_logamp_max"]))
                   for i, name in enumerate(names)]
        return dict(embedding=embedding, records=records,
                    support_definition="nearest INNER_FIT X-embedding vs 95th percentile leave-self-out FIT distances")

    def reference_weights(self, reference_X, query_X, **kwargs):
        return representation_reference_weights(self.transform(reference_X), self.transform(query_X),
                                                bandwidth=self.reference_bandwidth, **kwargs)

    def save(self, path):
        """Save tensor/primitive-only artifact; caller controls the destination."""
        payload = dict(version=ARTIFACT_VERSION, report_json=json.dumps(self.report),
                       network_state=self.network.state_dict(),
                       input_coordinates=self.input_coordinates.as_tensors(),
                       target_coordinates=self.target_coordinates.as_tensors(),
                       embedding_center=torch.as_tensor(self.embedding_center),
                       embedding_scale=self.embedding_scale,
                       fit_embeddings=torch.as_tensor(self.fit_embeddings),
                       support_radius=self.support_radius, reference_bandwidth=self.reference_bandwidth)
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload["version"] != ARTIFACT_VERSION:
            raise ValueError("Unsupported representation artifact version")
        report = json.loads(payload["report_json"])
        dims = report["dimensions"]
        # Loading must not change the caller's training RNG stream.
        with torch.random.fork_rng(devices=[]):
            network = _Student(dims["input"], dims["target"], dims["latent"], dims["hidden"], report["kind"])
        network.load_state_dict(payload["network_state"], strict=True)
        return cls(network, _FixedCoordinates.from_tensors(payload["input_coordinates"]),
                   _FixedCoordinates.from_tensors(payload["target_coordinates"]), report,
                   payload["embedding_center"].numpy(), payload["embedding_scale"],
                   payload["fit_embeddings"].numpy(), payload["support_radius"], payload["reference_bandwidth"])


def fit_representation(profiles, groups, ids, kind="conditional_predictive", seed=20260916,
                       epochs=60, *, latent_dim=8, target_dim=16, input_dim=32, hidden_dim=32,
                       batch_size=64, learning_rate=3e-4, weight_decay=1e-3, patience=10,
                       validation_fraction=.2, min_delta=1e-5):
    """Fit only the supplied TRAIN population, with an internal group holdout.

    Only INNER_FIT fits transforms and receives gradient updates. INNER_VALID
    selects the best epoch; no refit on that validation group follows. Inputs
    must already respect outer evaluation and reference-role isolation.
    All three future slots are loss targets, never student inference inputs.
    CPU training preserves the caller's torch RNG and uses local numpy RNGs.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 3 or profiles.shape[1] != 4 or min(profiles.shape) < 1 or not np.isfinite(profiles).all():
        raise ValueError("profiles must be finite [objects, 4, features]")
    groups, ids = np.asarray(groups).astype(str), np.asarray(ids).astype(str)
    if groups.shape != profiles.shape[:1] or ids.shape != profiles.shape[:1] or len(np.unique(ids)) != len(ids):
        raise ValueError("Aligned groups and unique object ids are required")
    if len(np.unique(groups)) < 4:
        raise ValueError("At least four groups are required for an internal group holdout")
    integers = (epochs, latent_dim, target_dim, input_dim, hidden_dim, batch_size, patience)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1 for v in integers):
        raise ValueError("Dimensions, epochs, batch_size and patience must be positive integers")
    if epochs > 60 or not 0 < validation_fraction < .5:
        raise ValueError("This optional estimator is bounded to 60 epochs and an internal holdout below 50%")
    if (not np.isfinite([learning_rate, weight_decay, min_delta]).all()
            or learning_rate <= 0 or weight_decay < 0 or min_delta < 0):
        raise ValueError("Invalid optimizer or early-stopping settings")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    fit, valid = next(splitter.split(profiles[:, 0], groups=groups))
    if len(fit) < 3 or len(valid) < 1:
        raise ValueError("Internal split has insufficient objects")
    input_coordinates = _FixedCoordinates.fit(profiles[fit, 0], input_dim, seed)
    target_coordinates = _FixedCoordinates.fit(profiles[fit, 1:].reshape(-1, profiles.shape[-1]), target_dim, seed)
    inputs = torch.as_tensor(input_coordinates.transform(profiles[:, 0]), dtype=torch.float32)
    target_values = target_coordinates.transform(profiles[:, 1:].reshape(-1, profiles.shape[-1]))
    targets = torch.as_tensor(target_values.reshape(len(profiles), 3, -1), dtype=torch.float32)
    dimensions = dict(input=inputs.shape[1], target=targets.shape[-1], latent=latent_dim,
                      hidden=hidden_dim, profile=profiles.shape[-1], exported=latent_dim+1)
    role_means = targets[fit].mean(0)
    constant_error = (targets[valid]-role_means).square()
    constant_validation_loss = .5*float(constant_error[..., :-1].mean()+constant_error[..., -1].mean())
    history, best_epoch, stale, best_loss = [], 0, 0, float("inf")
    generator = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        network = _Student(dimensions["input"], dimensions["target"], latent_dim, hidden_dim, kind)
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                              eta_min=learning_rate*.01)
        best_state = copy.deepcopy(network.state_dict())
        for epoch in range(1, epochs+1):
            network.train()
            order, gradient_norms = generator.permutation(fit), []
            for start in range(0, len(order), batch_size):
                indices = order[start:start+batch_size]
                optimizer.zero_grad(set_to_none=True)
                predicted = network(inputs[indices])
                # Shape and amplitude blocks have equal weight regardless of
                # PCA dimension. The same loss and target are used in both arms.
                error = (predicted-targets[indices]).square()
                loss = .5*(error[..., :-1].mean()+error[..., -1].mean())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite representation training loss")
                loss.backward()
                gradient_norms.append(float(nn.utils.clip_grad_norm_(network.parameters(), 5.)))
                optimizer.step()
            network.eval()
            with torch.no_grad():
                def evaluate(indices):
                    errors = (network(inputs[indices])-targets[indices]).square()
                    shape, amplitude = float(errors[..., :-1].mean()), float(errors[..., -1].mean())
                    return (.5*(shape+amplitude), shape, amplitude)
                train_loss, train_shape, train_amp = evaluate(fit)
                valid_loss, valid_shape, valid_amp = evaluate(valid)
            history.append(dict(epoch=epoch, learning_rate=float(optimizer.param_groups[0]["lr"]),
                                train_loss=train_loss, validation_loss=valid_loss,
                                train_shape_mse=train_shape, train_logamp_mse=train_amp,
                                validation_shape_mse=valid_shape, validation_logamp_mse=valid_amp,
                                gradient_norm_mean=float(np.mean(gradient_norms))))
            if valid_loss < best_loss-min_delta:
                best_loss, best_epoch, stale = valid_loss, epoch, 0
                best_state = copy.deepcopy(network.state_dict())
            else:
                stale += 1
            scheduler.step()
            if stale >= patience:
                break
        network.load_state_dict(best_state)
        network.eval()
        with torch.no_grad():
            fit_state = network.encoder(inputs[fit]).numpy().astype(np.float64)
    center = fit_state.mean(0)
    scale = max(float(np.sqrt(np.mean(np.square(fit_state-center)))), .01)
    fit_embeddings = np.column_stack(((fit_state-center)/scale/np.sqrt(latent_dim), inputs[fit, -1].numpy()))
    distances = pdist(fit_embeddings)/np.sqrt(latent_dim+1)
    bandwidth = max(float(np.median(distances)), .1)
    square = cdist(fit_embeddings, fit_embeddings, "sqeuclidean")/(latent_dim+1)
    np.fill_diagonal(square, np.inf)
    support = float(np.quantile(np.sqrt(square.min(1)), .95))
    _, amp, _ = _direction_amplitude(profiles[fit, 0])
    singular = np.linalg.svd(fit_state-center, compute_uv=False)
    energy = np.square(singular)
    effective_rank = float(energy.sum()**2/np.square(energy).sum()) if energy.sum() else 0.
    report = dict(kind=kind, seed=int(seed), dimensions=dimensions,
                  inner_fit_ids=ids[fit].tolist(), inner_validation_ids=ids[valid].tolist(),
                  inner_fit_groups=np.unique(groups[fit]).tolist(), inner_validation_groups=np.unique(groups[valid]).tolist(),
                  inner_fit_count=len(fit), inner_validation_count=len(valid),
                  parameter_count=sum(p.numel() for p in network.parameters()),
                  epochs_requested=int(epochs), epochs_completed=len(history), best_epoch=best_epoch,
                  best_validation_loss=best_loss, early_stopped=len(history) < epochs,
                  validation_role_mean_baseline_loss=constant_validation_loss,
                  best_validation_relative_error_reduction=(1-best_loss/constant_validation_loss
                                                            if constant_validation_loss > 0 else None),
                  stopping_patience=patience, stopping_min_delta=min_delta,
                  optimizer="AdamW", initial_learning_rate=learning_rate, weight_decay=weight_decay,
                  schedule="cosine to 1% of initial rate over declared maximum epochs",
                  history=history, fit_logamp_min=float(amp.min()), fit_logamp_max=float(amp.max()),
                  fit_latent_effective_rank=effective_rank, embedding_global_scale=scale,
                  reference_bandwidth=bandwidth, support_radius=support,
                  loss="0.5 * mean PCA-shape MSE + 0.5 * log-RMS MSE; equally weighted future roles",
                  target="fixed INNER_FIT future-profile directional PCA plus standardized log-RMS; no trainable teacher",
                  inference="X only; explicit log-RMS bypass; role slots are not batch identities",
                  data_boundary="all preprocessing and gradient updates INNER_FIT only; INNER_VALID only epoch selection",
                  status="optional representation, not a calibrated joint measurement distribution",
                  original_endpoint_changed=False)
    return ConditionalStateRepresentation(network, input_coordinates, target_coordinates, report,
                                           center, scale, fit_embeddings, support, bandwidth)
