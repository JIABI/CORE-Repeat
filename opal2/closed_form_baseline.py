"""Train-fitted, full-coordinate Gaussian baseline for repeated measurements.

This is a *fitted* closed-form statistical baseline, not a zero-parameter model
and not an information-theoretic ceiling. Robust training moments are estimated
after a train-only affine transform and optional winsorisation. Prediction is
the coherent Gaussian defined by those fitted parameters: conditioning NEVER
winsorises a new observation. The original-space utility is not redefined.

In affine coordinates, x_w = slot_mean[w] + L u + F e_w + D**.5 eta_w,
where u, e_w and eta_w are independent standard normals, and the same u is used
for all wells of one object. L = basis @ sqrt(signal_cov). F describes projected
within-object noise. D has a strictly positive entry for EVERY coordinate,
including coordinates outside the fitted PCA span. Residual signal outside
that span is deliberately approximated as independent residual noise, rather
than silently assigned zero predictive variance. Slot means are descriptive
effects of observed repeat roles; they are not identified causal batch effects.

The full covariance need never be materialised. Woodbury conditioning and
joint log densities are exact for this approximate fitted Gaussian family.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from sklearn.decomposition import PCA


def _symmetric(x: np.ndarray) -> np.ndarray:
    return (x + x.T) * .5


def _psd(x: np.ndarray, floor: float = 0.) -> np.ndarray:
    value, vector = np.linalg.eigh(_symmetric(x))
    return _symmetric((vector * np.maximum(value, floor)) @ vector.T)


def _root(x: np.ndarray) -> np.ndarray:
    value, vector = np.linalg.eigh(_symmetric(x))
    # PSD zero eigenvalues stay zero; no artificial shared signal is added.
    return vector * np.sqrt(np.maximum(value, 0.))


def _array(x: np.ndarray, ndim: int, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != ndim or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return x


def _slots(slots: Sequence[int] | int, count: int, total: int, name: str) -> np.ndarray:
    raw = np.asarray([slots] if np.isscalar(slots) else slots)
    if raw.ndim != 1 or len(raw) != count:
        raise ValueError(f"{name} must contain one slot per well")
    if raw.dtype.kind not in "iu" and raw.size:
        raise ValueError(f"{name} must contain integer slots")
    out = raw.astype(np.int64)
    if np.any(out < 0) or np.any(out >= total) or len(np.unique(out)) != len(out):
        raise ValueError(f"{name} must contain distinct in-range slots")
    return out


@dataclass
class ClosedFormBaseline:
    center: np.ndarray
    scale: np.ndarray
    slot_mean: np.ndarray
    basis: np.ndarray
    signal_cov: np.ndarray
    within_cov: np.ndarray
    residual_var: np.ndarray
    reliability_vectors: np.ndarray
    reliability_eigenvalues: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.center = _array(self.center, 1, "center")
        self.scale = _array(self.scale, 1, "scale")
        self.slot_mean = _array(self.slot_mean, 2, "slot_mean")
        self.basis = _array(self.basis, 2, "basis")
        self.signal_cov = _array(self.signal_cov, 2, "signal_cov")
        self.within_cov = _array(self.within_cov, 2, "within_cov")
        self.residual_var = _array(self.residual_var, 1, "residual_var")
        self.reliability_vectors = _array(self.reliability_vectors, 2, "reliability_vectors")
        self.reliability_eigenvalues = _array(self.reliability_eigenvalues, 1, "reliability_eigenvalues")
        d, k = self.basis.shape
        if (self.center.shape != (d,) or self.scale.shape != (d,)
                or self.residual_var.shape != (d,) or self.slot_mean.shape[1] != d
                or self.signal_cov.shape != (k, k) or self.within_cov.shape != (k, k)
                or self.reliability_vectors.shape != (k, k)
                or self.reliability_eigenvalues.shape != (k,)):
            raise ValueError("Baseline parameter dimensions are inconsistent")
        if np.any(self.scale <= 0) or np.any(self.residual_var <= 0):
            raise ValueError("Affine scales and all full-space residual variances must be positive")
        if not np.allclose(self.basis.T @ self.basis, np.eye(k), atol=1e-7):
            raise ValueError("PCA basis must be orthonormal")
        for name, covariance in (("signal", self.signal_cov), ("within", self.within_cov)):
            if (not np.allclose(covariance, covariance.T, atol=1e-10)
                    or np.linalg.eigvalsh(covariance).min() < -1e-9):
                raise ValueError(f"{name} covariance must be symmetric positive semidefinite")
        self._signal_factor = self.basis @ _root(self.signal_cov)
        self._noise_factor = self.basis @ _root(self.within_cov)
        self._noise_dinv_factor = self._noise_factor / self.residual_var[:, None]
        gram = np.eye(k) + self._noise_factor.T @ self._noise_dinv_factor
        self._noise_cholesky = cho_factor(_symmetric(gram), lower=True)
        self._noise_logdet = (np.log(self.residual_var).sum()
                              + 2 * np.log(np.diag(self._noise_cholesky[0])).sum())
        self._qinv_signal = self.noise_solve(self._signal_factor)
        self._signal_information = _symmetric(self._signal_factor.T @ self._qinv_signal)
        self._posterior_cache: dict[int, tuple] = {}
        self._density_cache: dict[tuple[int, int], tuple] = {}

    @property
    def feature_dim(self) -> int:
        return len(self.center)

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def n_slots(self) -> int:
        return len(self.slot_mean)

    @property
    def diagnostics(self) -> dict:
        return dict(self.metadata,
                    signal_min_eigenvalue=float(np.linalg.eigvalsh(self.signal_cov).min()),
                    within_min_eigenvalue=float(np.linalg.eigvalsh(self.within_cov).min()),
                    residual_variance_min=float(self.residual_var.min()),
                    residual_variance_median=float(np.median(self.residual_var)),
                    residual_variance_max=float(self.residual_var.max()),
                    reliability_eigenvalue_max=float(self.reliability_eigenvalues.max()),
                    reliability_positive_axes=int((self.reliability_eigenvalues > 1e-8).sum()))

    def transform_features(self, y: np.ndarray, slots: Sequence[int] | int,
                           clip: bool = False) -> np.ndarray:
        """Train-fitted representation transform; not Gaussian conditioning.

        2D input represents one well per object; 3D input uses a shared slot
        list. If requested, clipping precedes subtraction of the identically
        fitted clipped slot mean. ``clip=False`` is the model's affine map.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim not in (2, 3) or y.shape[-1] != self.feature_dim or not np.isfinite(y).all():
            raise ValueError("Measurement input must be finite [N,D] or [N,W,D]")
        ss = _slots(slots, 1 if y.ndim == 2 else y.shape[1], self.n_slots, "slots")
        x = (y - self.center) / self.scale
        limit = self.metadata.get("clip")
        if clip and limit is not None:
            x = np.clip(x, -float(limit), float(limit))
        return x - (self.slot_mean[ss[0]] if y.ndim == 2 else self.slot_mean[ss][None])

    def reliability_features(self, x: np.ndarray, slot: int = 0,
                             clip: bool = False) -> np.ndarray:
        """Noise-whitened, signal-ranked linear coordinates, fitted on TRAIN.

        Eigenvalues describe this fitted covariance model; no biological or
        universal reliability interpretation is asserted.
        """
        return self.transform_features(x, slot, clip=clip) @ self.basis @ self.reliability_vectors

    def noise_solve(self, rhs: np.ndarray) -> np.ndarray:
        """Apply Q^-1 to [D] or [D,M] without constructing a D-by-D matrix."""
        rhs = np.asarray(rhs, dtype=np.float64)
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != self.feature_dim:
            raise ValueError("Noise solve requires [D] or [D,M]")
        direct = rhs / self.residual_var[:, None]
        corrected = direct - self._noise_dinv_factor @ cho_solve(
            self._noise_cholesky, self._noise_factor.T @ direct)
        return corrected[:, 0] if vector else corrected

    def conditional(self, context_y: np.ndarray, context_slots: Sequence[int],
                    target_slots: Sequence[int]) -> "ConditionalGaussian":
        """Condition on actual untruncated original measurements.

        All context/target slots denote different physical wells. The
        conditional covariance is common across objects with the same number
        of observations under this homoscedastic model.
        """
        yy = _array(context_y, 3, "context_y")
        if yy.shape[-1] != self.feature_dim:
            raise ValueError("Context feature dimension differs from fitted model")
        cs = _slots(context_slots, yy.shape[1], self.n_slots, "context_slots")
        ts = _slots(target_slots, len(target_slots), self.n_slots, "target_slots")
        if len(ts) == 0 or np.intersect1d(cs, ts).size:
            raise ValueError("Targets must be nonempty physical wells distinct from context")
        centered = self.transform_features(yy, cs, clip=False)
        if len(cs) not in self._posterior_cache:
            information = np.eye(self.rank) + len(cs) * self._signal_information
            cf = cho_factor(_symmetric(information), lower=True)
            covariance = _symmetric(cho_solve(cf, np.eye(self.rank)))
            shared_factor = self._signal_factor @ _root(covariance)
            self._posterior_cache[len(cs)] = (cf, covariance, shared_factor)
        cf, covariance, shared_factor = self._posterior_cache[len(cs)]
        natural = centered.sum(axis=1) @ self._qinv_signal
        latent_mean = cho_solve(cf, natural.T).T
        common_mean = latent_mean @ self._signal_factor.T
        mean_affine = self.slot_mean[ts][None] + common_mean[:, None]
        mean = self.center + mean_affine * self.scale
        return ConditionalGaussian(self, mean, shared_factor, covariance,
                                   latent_mean, ts, len(cs))

    def save(self, path: str | Path) -> None:
        """Serialize precisely the parameters used by fit and inference."""
        payload = {key: getattr(self, key) for key in (
            "center", "scale", "slot_mean", "basis", "signal_cov", "within_cov",
            "residual_var", "reliability_vectors", "reliability_eigenvalues")}
        # File handle avoids silently changing an explicitly supplied filename.
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **payload,
                                metadata=np.array(json.dumps(self.metadata, sort_keys=True)))

    @classmethod
    def load(cls, path: str | Path) -> "ClosedFormBaseline":
        with np.load(path, allow_pickle=False) as content:
            kwargs = {key: content[key] for key in content.files if key != "metadata"}
            kwargs["metadata"] = json.loads(str(content["metadata"].item()))
        if kwargs["metadata"].get("schema_version") != 1:
            raise ValueError("Unsupported closed-form baseline schema")
        return cls(**kwargs)

    def gamma_predict(self, x: np.ndarray, *, slot: int = 0,
                      actions: Sequence[Sequence[int]] = ((1,), (2,), (1, 2)),
                      validation_slot: int = 3, n_samples: int = 512,
                      seed: int = 0, chunk_size: int = 4,
                      cost_per_well: float = .01,
                      positive_margin: float = .005) -> dict[str, np.ndarray]:
        """Monte Carlo moments of ORIGINAL cosine gain minus declared cost.

        One actual X is held fixed, not simulated. All actions use the same
        joint future draws (including V), preserving shared signal and common
        random numbers. Results are [N,A]; the utility is half the cosine
        improvement. This estimates model predictions, never a ceiling.
        Original three-state labels are retained: NULL is gain <= 0,
        POSITIVE is gain >= positive_margin (default .005), and gains strictly
        between these thresholds are AMBIGUOUS, not positive.
        """
        x = _array(x, 2, "x")
        if n_samples < 2 or chunk_size < 1 or not np.isfinite(cost_per_well) or cost_per_well < 0:
            raise ValueError("Need at least two MC samples, positive chunk size and nonnegative cost")
        if not np.isfinite(positive_margin) or positive_margin <= 0:
            raise ValueError("Positive margin must be finite and strictly positive")
        if not actions or any(len(a) == 0 for a in actions):
            raise ValueError("At least one nonempty action is required")
        action_slots = [_slots(a, len(a), self.n_slots, "action") for a in actions]
        future = sorted(set([int(validation_slot)] + [int(s) for a in action_slots for s in a]))
        if any(validation_slot in a for a in action_slots) or slot in future:
            raise ValueError("Action, validation and observed wells must be distinct")
        _slots([validation_slot], 1, self.n_slots, "validation_slot")
        lookup = {s: j for j, s in enumerate(future)}
        out = {key: np.empty((len(x), len(actions)), dtype=np.float64)
               for key in ("mean", "sd", "p_null", "p_positive", "p_ambiguous", "mc_se")}
        rng = np.random.default_rng(seed)
        for start in range(0, len(x), chunk_size):
            observed = x[start:start + chunk_size]
            conditional = self.conditional(observed[:, None], [slot], future)
            draw = conditional.sample_joint(n_samples, rng=rng)
            validation = draw[:, :, lookup[validation_slot]]
            base = _cosine(observed[None], validation)
            for j, action in enumerate(action_slots):
                after = observed[None] + draw[:, :, [lookup[int(s)] for s in action]].sum(axis=2)
                # Cosine of sum equals cosine of its positive-count average.
                gain = .5 * (_cosine(after, validation) - base) - len(action) * cost_per_well
                at = slice(start, start + len(observed))
                out["mean"][at, j] = gain.mean(axis=0)
                out["sd"][at, j] = gain.std(axis=0, ddof=1)
                out["p_null"][at, j] = (gain <= 0).mean(axis=0)
                out["p_positive"][at, j] = (gain >= positive_margin).mean(axis=0)
                out["p_ambiguous"][at, j] = ((gain > 0) & (gain < positive_margin)).mean(axis=0)
                out["mc_se"][at, j] = out["sd"][at, j] / np.sqrt(n_samples)
        return out


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    if np.any(denominator <= 0):
        raise ValueError("Zero-norm profile requires an explicitly declared failure rule")
    return (a * b).sum(axis=-1) / denominator


@dataclass
class ConditionalGaussian:
    model: ClosedFormBaseline
    mean: np.ndarray
    shared_factor: np.ndarray  # affine coordinates, shared across target wells
    latent_covariance: np.ndarray
    latent_mean: np.ndarray
    target_slots: np.ndarray
    n_context: int

    @property
    def marginal_variance(self) -> np.ndarray:
        variance = (self.model.residual_var + np.square(self.model._noise_factor).sum(axis=1)
                    + np.square(self.shared_factor).sum(axis=1)) * self.model.scale**2
        return np.broadcast_to(variance, self.mean.shape)

    def sample_joint(self, n_samples: int, seed: int | None = None,
                     rng: np.random.Generator | None = None) -> np.ndarray:
        if n_samples < 1 or (rng is not None and seed is not None):
            raise ValueError("Positive sample count and only one RNG specification are required")
        rng = np.random.default_rng(seed) if rng is None else rng
        n, t, d = self.mean.shape
        shared = (rng.standard_normal((n_samples * n, self.model.rank))
                  @ self.shared_factor.T).reshape(n_samples, n, d)
        independent = (rng.standard_normal((n_samples * n * t, self.model.rank))
                       @ self.model._noise_factor.T).reshape(n_samples, n, t, d)
        independent += (rng.standard_normal((n_samples, n, t, d))
                        * np.sqrt(self.model.residual_var))
        return self.mean[None] + (shared[:, :, None] + independent) * self.model.scale

    def iter_sample_chunks(self, n_samples: int, chunk_size: int = 64,
                           seed: int | None = None):
        """Bound peak Monte Carlo memory; use one stream across sample chunks.

        Chunk size is part of Monte Carlo reproducibility because shared and
        independent draws are interleaved differently at different sizes.
        Object chunking, seed, sample count and sample chunk size should all be
        recorded by the experiment runner.
        """
        if n_samples < 1 or chunk_size < 1:
            raise ValueError("Sample count and chunk size must be positive")
        rng = np.random.default_rng(seed)
        for start in range(0, n_samples, chunk_size):
            yield self.sample_joint(min(chunk_size, n_samples - start), rng=rng)

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        """Exact joint target log density per object in ORIGINAL coordinates."""
        y = _array(y, 3, "target_y")
        if y.shape != self.mean.shape:
            raise ValueError("Targets do not match the joint predictive distribution")
        n, t, d = y.shape
        residual = (y - self.mean) / self.model.scale
        flat = residual.reshape(-1, d)
        solved = self.model.noise_solve(flat.T).T.reshape(n, t, d)
        cache_key = (self.n_context, t)
        if cache_key not in self.model._density_cache:
            qinv_h = self.model.noise_solve(self.shared_factor)
            gram = np.eye(self.model.rank) + t * self.shared_factor.T @ qinv_h
            cf = cho_factor(_symmetric(gram), lower=True)
            self.model._density_cache[cache_key] = (cf,)
        cf, = self.model._density_cache[cache_key]
        statistic = solved.sum(axis=1) @ self.shared_factor
        correction = np.einsum("nk,nk->n", statistic, cho_solve(cf, statistic.T).T)
        quadratic = np.einsum("ntd,ntd->n", residual, solved) - correction
        logdet = (t * self.model._noise_logdet + 2 * np.log(np.diag(cf[0])).sum()
                  + 2 * t * np.log(self.model.scale).sum())
        return -.5 * (t * d * np.log(2 * np.pi) + logdet + quadratic)

    def nll(self, y: np.ndarray, per_coordinate: bool = True) -> np.ndarray:
        out = -self.log_prob(y)
        return out / np.prod(self.mean.shape[1:]) if per_coordinate else out

    def dense_covariance(self) -> np.ndarray:
        """Diagnostic only: [T*D,T*D] original-space joint covariance."""
        t = self.mean.shape[1]
        noise = self.model._noise_factor @ self.model._noise_factor.T + np.diag(self.model.residual_var)
        shared = self.shared_factor @ self.shared_factor.T
        affine = np.kron(np.eye(t), noise) + np.kron(np.ones((t, t)), shared)
        scale = np.tile(self.model.scale, t)
        return affine * scale[:, None] * scale[None]


def fit_baseline(y_train: np.ndarray, k: int = 200, clip: float | None = 8.,
                 noise_shrinkage: float = .05, variance_floor: float = 1e-6,
                 random_state: int = 0) -> ClosedFormBaseline:
    """Fit only on supplied training objects, using all declared repeat slots.

    Robust moments use clipped training coordinates, while prediction uses the
    coherent full Gaussian with no input clipping. Consequently this is robust
    moment fitting, not maximum likelihood for an untruncated Gaussian. The
    discarded-span residual model is a positive diagonal approximation, not a
    claim that discarded directions carry no signal.
    """
    y = _array(y_train, 3, "y_train")
    n, w, d = y.shape
    if n < 3 or w < 2 or d < 1 or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Fit requires at least 3 objects, 2 repeats and positive integer rank")
    if (clip is not None and (not np.isfinite(clip) or clip <= 0)):
        raise ValueError("Clipping limit must be positive or None")
    if not 0 <= noise_shrinkage <= 1 or not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("Noise shrinkage must be in [0,1] and variance floor positive")
    flattened = y.reshape(-1, d)
    # Exactly the existing TrainScaler._moments affine convention. Clipping is
    # a robust moment-fitting choice, not an additional inference transform.
    center = np.mean(flattened, axis=0, dtype=np.float64)
    scale = np.std(flattened, axis=0, dtype=np.float64)
    scale_floor = 1e-6
    scale = np.where(scale >= scale_floor, scale, 1.)
    affine = (y - center) / scale
    robust = np.clip(affine, -clip, clip) if clip is not None else affine
    slot_mean = robust.mean(axis=0)
    centered = robust - slot_mean[None]
    actual_rank = min(k, d, n * w - 1)
    pca = PCA(n_components=actual_rank,
              svd_solver="full" if actual_rank >= min(n * w, d) - 1 else "randomized",
              random_state=random_state)
    pca.fit(centered.reshape(-1, d))
    basis = pca.components_.T.copy()
    projected = centered @ basis
    object_mean = projected.mean(axis=1)
    deviations = projected - object_mean[:, None]
    # Slot means were estimated, removing one object-level degree of freedom.
    # The factor (n-1)(w-1) estimates within covariance under this fitted model.
    flattened_deviations = deviations.reshape(-1, actual_rank)
    within_empirical = (flattened_deviations.T @ flattened_deviations) / ((n - 1) * (w - 1))
    between = object_mean.T @ object_mean / (n - 1)
    unprojected_signal = _symmetric(between - within_empirical / w)
    signal = _psd(unprojected_signal)
    # Unclipped original training residuals keep heavy tails in predictive
    # dispersion even though the low-rank covariance fit is winsorised.
    untruncated_centered = affine - slot_mean[None]
    residual = untruncated_centered - (untruncated_centered @ basis) @ basis.T
    # Remaining total variance is conservatively assigned to independent noise.
    # This includes unmodelled residual signal; it must not be sold as a proven
    # decomposition of biological signal and technical noise.
    residual_var = np.square(residual).sum(axis=(0, 1)) / ((n - 1) * w)
    typical = max(float(np.mean(np.square(centered))), 1.)
    absolute_floor = variance_floor * typical
    residual_var = np.maximum(residual_var, absolute_floor)
    residual_in_basis = basis.T @ (residual_var[:, None] * basis)
    projected_noise = _psd(within_empirical - residual_in_basis)
    average_noise = max(float(np.trace(projected_noise) / actual_rank), absolute_floor)
    within = ((1 - noise_shrinkage) * projected_noise
              + noise_shrinkage * average_noise * np.eye(actual_rank))
    within = _psd(within, absolute_floor)
    total_projected_noise = _symmetric(within + residual_in_basis)
    eigenvalues, eigenvectors = eigh(signal, total_projected_noise)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = np.maximum(eigenvalues[order], 0.), eigenvectors[:, order]
    metadata = dict(schema_version=1, n_train=int(n), n_slots=int(w), feature_dim=int(d),
                    requested_rank=int(k), rank=int(actual_rank), clip=clip,
                    noise_shrinkage=float(noise_shrinkage), variance_floor=float(variance_floor),
                    scale_floor=float(scale_floor), residual_variance_floor=float(absolute_floor),
                    random_state=int(random_state),
                    negative_signal_eigenvalues=int((np.linalg.eigvalsh(unprojected_signal) < 0).sum()),
                    affine_fit="train_mean_population_std_TrainScaler_floor_1e-6_to_one",
                    slot_mean_fit="mean_of_clipped_affine_training_coordinates",
                    condition_input="unclipped_affine_original_measurement",
                    residual_model="positive_diagonal_independent_total_discarded_span_variance",
                    covariance_scope="within_object_shared_signal_independent_repeat_noise",
                    limitation="No identified batch effects, no calibrated deployment guarantee, no ceiling")
    return ClosedFormBaseline(center, scale, slot_mean, basis, signal, within,
                              residual_var, eigenvectors, eigenvalues, metadata)
