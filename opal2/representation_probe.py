"""Matched-target linear probes for existing measurement representations.

Every representation predicts the same physical measurement coordinates. No
teacher embedding is used as the regression target. Preprocessing and ridge
selection are fitted on compound-disjoint training indices; representations
that are themselves fitted must be supplied through ``feature_builder`` so
they can be refitted inside each inner fold. A pre-existing JEPA checkpoint is
an immutable representation, not an independently refitted inner-fold model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


DEFAULT_ALPHAS = (.001, .01, .1, 1., 10.)


def _finite_matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite two-dimensional array")
    return array


def _indices(value, n, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or not len(raw):
        raise ValueError(f"{name} must contain integer row indices")
    result = raw.astype(np.int64)
    if len(np.unique(result)) != len(result) or np.any(result < 0) or np.any(result >= n):
        raise ValueError(f"{name} has duplicate or out-of-range indices")
    return result


def _target_cube(value):
    array = np.asarray(value, dtype=np.float64)
    squeezed = array.ndim == 2
    if squeezed:
        array = array[:, None, :]
    if array.ndim != 3 or not all(array.shape):
        raise ValueError("targets must have shape [compound, coordinate] or [compound, slot, coordinate]")
    return array, squeezed


def _affine(values, minimum_scale):
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    active = scale >= minimum_scale
    return center, np.where(active, scale, 1.), active


@dataclass
class RidgeProbeModel:
    """Training-only affine transforms plus a dual multi-output ridge head."""

    feature_center: np.ndarray
    feature_scale: np.ndarray
    feature_active: np.ndarray
    normalized_training_features: np.ndarray
    dual_coefficients: np.ndarray
    target_center: np.ndarray
    target_scale: np.ndarray
    alphas: np.ndarray
    squeezed_target: bool = False

    def predict(self, features):
        features = _finite_matrix(features, "features")
        if features.shape[1] != len(self.feature_center):
            raise ValueError("Probe feature coordinate count changed")
        transformed = (features - self.feature_center) / self.feature_scale
        transformed[:, ~self.feature_active] = 0.
        transformed /= np.sqrt(max(1, int(self.feature_active.sum())))
        kernel = transformed @ self.normalized_training_features.T
        standardized = (kernel @ self.dual_coefficients.reshape(len(kernel.T), -1)).reshape(
            len(kernel), *self.target_center.shape)
        prediction = standardized * self.target_scale + self.target_center
        return prediction[:, 0] if self.squeezed_target else prediction


def _head_components(features, targets, minimum_scale):
    features = _finite_matrix(features, "training features")
    targets, _ = _target_cube(targets)
    if len(features) != len(targets) or not np.isfinite(targets).all():
        raise ValueError("Training features and finite targets must have matching rows")
    fc, fs, active = _affine(features, minimum_scale)
    z = (features - fc) / fs
    z[:, ~active] = 0.
    z /= np.sqrt(max(1, int(active.sum())))
    tc, ts, _ = _affine(targets, minimum_scale)
    t = (targets - tc) / ts
    eigenvalues, eigenvectors = np.linalg.eigh(z @ z.T)
    eigenvalues = np.maximum(eigenvalues, 0.)
    rotated = (eigenvectors.T @ t.reshape(len(t), -1)).reshape(t.shape)
    return fc, fs, active, z, tc, ts, eigenvalues, eigenvectors, rotated


def _fit_head(features, targets, alphas, minimum_scale=1e-8, squeezed=False):
    fc, fs, active, z, tc, ts, eigenvalues, eigenvectors, rotated = _head_components(features, targets, minimum_scale)
    alphas = np.broadcast_to(np.asarray(alphas, dtype=np.float64), (tc.shape[0],)).copy()
    if not np.isfinite(alphas).all() or np.any(alphas <= 0):
        raise ValueError("Ridge strengths must be positive and finite")
    regularized = eigenvalues[:, None] + len(z) * alphas[None, :]
    coefficients = rotated / regularized[..., None]
    dual = (eigenvectors @ coefficients.reshape(len(z), -1)).reshape(coefficients.shape)
    return RidgeProbeModel(fc, fs, active, z, dual, tc, ts, alphas, squeezed)


def regression_metrics(targets, predictions, training_target_mean, scale=None):
    """SSE, MSE and predictive R² against the same training-mean baseline.

``scale`` only defines the reported coordinate weighting. It must be fitted
on training targets, not on this evaluation partition. The returned R² uses
SSE of that fixed training mean as denominator, not evaluation-centering.
Undefined R² for a zero-error baseline is returned as None, not forced to 0.
"""
    target, _ = _target_cube(targets)
    prediction, _ = _target_cube(predictions)
    if target.shape != prediction.shape or not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("Prediction and finite target geometry must match")
    mean = np.asarray(training_target_mean, dtype=float)
    if mean.ndim == 1:
        mean = mean[None, :]
    if mean.shape != target.shape[1:] or not np.isfinite(mean).all():
        raise ValueError("Training target mean geometry must match")
    weight_scale = np.ones_like(mean) if scale is None else np.asarray(scale, dtype=float)
    if weight_scale.ndim == 1:
        weight_scale = weight_scale[None, :]
    if weight_scale.shape != mean.shape or not np.isfinite(weight_scale).all() or np.any(weight_scale <= 0):
        raise ValueError("Target scales must be positive and match target coordinates")
    squared = ((prediction - target) / weight_scale) ** 2
    baseline = ((target - mean) / weight_scale) ** 2
    sse = squared.sum(axis=-1)
    baseline_sse = baseline.sum(axis=-1)
    dimensions = target.shape[-1]

    def summarize(error, null):
        total, denom = float(error.sum()), float(null.sum())
        return {"sse": total, "baseline_sse": denom,
                "mse": total / (error.size * dimensions),
                "baseline_mse": denom / (error.size * dimensions),
                "r2_training_mean": None if denom <= 0 else 1. - total / denom}

    return {"overall": summarize(sse, baseline_sse),
            "slots": [summarize(sse[:, slot], baseline_sse[:, slot]) for slot in range(target.shape[1])],
            "per_object_slot_sse": sse,
            "per_object_slot_baseline_sse": baseline_sse,
            "per_object_sse": sse.sum(axis=1),
            "per_object_baseline_sse": baseline_sse.sum(axis=1),
            "r2_denominator": "error of the training-only slot-specific mean on evaluated targets"}


def fit_ridge_probe(features, targets, train_indices, validation_indices, *,
                    alphas=DEFAULT_ALPHAS, inner_splits=3, seed=20260912,
                    feature_builder: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                    minimum_scale=1e-8):
    """Tune one ridge strength per target slot and fit full-coordinate probes.

The callable ``feature_builder(fit_indices, query_indices)`` returns one
feature row per query index, fitting only to ``fit_indices``. It is invoked
once per inner fold and once for the final outer fit. Providing a fixed array
instead is appropriate for raw inputs or an explicitly frozen pretrained
representation, but is not nested evaluation of a data-fitted basis.

Features are train-standardized and divided by sqrt(number of active input
coordinates). Thus the training kernel has mean diagonal one across input
dimensions; the ridge penalty is n_train * alpha. All arms retain the same
full target, target scaling, alpha grid and target-slot-specific heads.
"""
    target, squeezed = _target_cube(targets)
    train = _indices(train_indices, len(target), "train_indices")
    valid = _indices(validation_indices, len(target), "validation_indices")
    if np.intersect1d(train, valid).size:
        raise ValueError("Training and evaluated compounds overlap")
    if not np.isfinite(target[train]).all() or not np.isfinite(target[valid]).all():
        raise ValueError("Selected target values must be finite; no implicit row removal is permitted")
    if isinstance(inner_splits, bool) or not isinstance(inner_splits, int) or not 2 <= inner_splits <= len(train):
        raise ValueError("At least two nonempty compound-disjoint inner folds are required")
    candidates = np.asarray(alphas, dtype=float)
    if (candidates.ndim != 1 or not len(candidates) or not np.isfinite(candidates).all()
            or np.any(candidates <= 0) or len(np.unique(candidates)) != len(candidates)):
        raise ValueError("Alpha candidates must be unique positive finite values")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive and finite")
    x = None if features is None else np.asarray(features, dtype=float)
    if feature_builder is None and (x is None or x.ndim != 2 or len(x) != len(target)):
        raise ValueError("Supply features aligned to every target row, or a feature_builder")

    def obtain(fit, heldout):
        query = np.concatenate((fit, heldout))
        rows = x[query] if feature_builder is None else feature_builder(fit.copy(), query.copy())
        rows = _finite_matrix(rows, "generated features")
        if len(rows) != len(query):
            raise ValueError("feature_builder changed the requested query row count")
        return rows[:len(fit)], rows[len(fit):]

    shuffled = np.random.default_rng(seed).permutation(train)
    folds = np.array_split(shuffled, inner_splits)
    scores = np.zeros((len(candidates), target.shape[1]), dtype=float)
    fold_scores, fold_indices = [], []
    for fold in folds:
        fitted = train[~np.isin(train, fold)]
        fx, vx = obtain(fitted, fold)
        fc, fs, active, z, tc, target_scale, eigenvalues, vectors, rotated = _head_components(fx, target[fitted], minimum_scale)
        query = (vx - fc) / fs
        query[:, ~active] = 0.
        query /= np.sqrt(max(1, int(active.sum())))
        # One eigensystem for the entire alpha path and all future slots.
        projected_query = (query @ z.T) @ vectors
        standardized_target = (target[fold] - tc) / target_scale
        this_scores = np.zeros_like(scores)
        for index, alpha in enumerate(candidates):
            coefficients = rotated / (eigenvalues + len(fitted) * alpha)[:, None, None]
            prediction = (projected_query @ coefficients.reshape(len(fitted), -1)).reshape(
                len(fold), *tc.shape)
            error = (prediction - standardized_target) ** 2
            this_scores[index] = error.sum(axis=(0, 2))
        scores += this_scores
        fold_scores.append(this_scores / (len(fold) * target.shape[-1]))
        fold_indices.append({"train_indices": fitted.tolist(), "validation_indices": fold.tolist()})
    scores /= len(train) * target.shape[-1]
    # A declared grid-order tie break uses no evaluated outcomes.
    selected = candidates[np.argmin(scores, axis=0)]
    fx, vx = obtain(train, valid)
    model = _fit_head(fx, target[train], selected, minimum_scale, squeezed)
    predictions = model.predict(vx)
    physical = regression_metrics(target[valid], predictions, model.target_center)
    standardized = regression_metrics(target[valid], predictions, model.target_center, model.target_scale)
    standardized_predictions = (np.asarray(predictions).reshape(target[valid].shape) - model.target_center) / model.target_scale
    if squeezed:
        standardized_predictions = standardized_predictions[:, 0]
    return {"model": model, "predictions": predictions,
            "standardized_predictions": standardized_predictions,
            "selected_alphas": selected, "alpha_grid": candidates,
            "cv_standardized_mse": scores, "inner_fold_mse": np.stack(fold_scores),
            "inner_folds": fold_indices, "train_indices": train, "validation_indices": valid,
            "physical_metrics": physical, "standardized_metrics": standardized,
            "input_dimension": fx.shape[1], "active_input_dimension": int(model.feature_active.sum()),
            "representation_fit": "inner_fold_refit" if feature_builder else "supplied_fixed_features_conditional_head_tuning",
            "target_definition": "unchanged full physical measurement coordinates",
            "kernel_normalization": "train z-score divided by sqrt(active input dimension); penalty n_train*alpha"}


def extract_checkpoint_features(Y, feature_names, run_directory, checkpoint_path, *,
                                batch_size=32, projector=False, expected_scaler=None):
    """Extract existing diagnostic teacher features with strict geometry binding.

This loads no dataset from disk and performs no optimization. ``Y`` contains
only caller-authorized physical input wells. Checkpoint/scaler/manifest feature
order and pretraining IDs must agree. Returned IDs identify pretraining
exposure; callers must exclude them from held-out representation comparisons.
Old diagnostic checkpoints do not embed numerical scaler values. Supplying an
independently refitted ``expected_scaler`` verifies those values against the
original training rows; without it only the saved scaler's schema/IDs bind.
"""
    import torch
    from torch import nn
    from .data import TrainScaler
    from .model import GroupedProfileEncoder

    y = np.asarray(Y)
    names = list(map(str, feature_names))
    if y.ndim < 2 or y.shape[-1] != len(names) or len(set(names)) != len(names):
        raise ValueError("Input feature geometry is invalid")
    if not np.isfinite(y).all():
        raise ValueError("Representation comparison requires finite supplied input wells")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    directory, checkpoint = Path(run_directory), Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = directory / checkpoint
    manifest = json.loads((directory / "training_diagnostic_manifest.json").read_text())
    scaler = TrainScaler.load(directory / "scaler.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("manifest") != manifest:
        raise ValueError("JEPA checkpoint does not match this diagnostic manifest")
    if names != manifest.get("feature_names") or names != scaler.feature_names:
        raise ValueError("JEPA checkpoint/scaler/input feature coordinate order differs")
    if list(scaler.train_ids) != manifest.get("train_ids"):
        raise ValueError("JEPA scaler and pretraining compound identifiers differ")
    if (len(scaler.y_center) != len(names) or len(scaler.y_scale) != len(names)
            or not np.isfinite(scaler.y_center).all() or not np.isfinite(scaler.y_scale).all()
            or np.any(np.asarray(scaler.y_scale) <= 0)):
        raise ValueError("JEPA saved measurement affine transform is invalid")
    if expected_scaler is not None:
        if (expected_scaler.feature_names != scaler.feature_names
                or expected_scaler.train_ids != scaler.train_ids
                or not np.allclose(expected_scaler.y_center, scaler.y_center, rtol=1e-10, atol=1e-10)
                or not np.allclose(expected_scaler.y_scale, scaler.y_scale, rtol=1e-10, atol=1e-10)):
            raise ValueError("JEPA saved scaler disagrees with the independently refitted training scaler")
    if len(set(manifest["train_ids"])) != len(manifest["train_ids"]):
        raise ValueError("Duplicate JEPA pretraining compound identifiers")
    config = manifest["model_config"]
    if int(config["hidden_dim"]) != int(manifest["train_config"]["hidden_dim"]):
        raise ValueError("JEPA architecture metadata is inconsistent")
    states = payload.get("state_dict", {})
    prefix = "teacher_encoder."
    state = {key[len(prefix):]: value for key, value in states.items() if key.startswith(prefix)}
    exported = payload.get("encoder_state_dict")
    if not state or exported is None or state.keys() != exported.keys() or any(
            not torch.equal(state[key], exported[key]) for key in state):
        raise ValueError("Exported JEPA encoder differs from checkpoint teacher encoder")
    with torch.random.fork_rng(devices=[]):
        encoder = GroupedProfileEncoder(config["feature_groups"], int(config["hidden_dim"]),
                    attention_layers=int(config["group_attention_layers"]),
                    attention_heads=int(config["attention_heads"]))
        encoder.load_state_dict(state, strict=True)
        encoder.eval().requires_grad_(False)
        target_projector = None
        if projector:
            width = encoder.hidden_dim
            target_projector = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            pstate = {key[len("teacher_projector."):]: value for key, value in states.items()
                      if key.startswith("teacher_projector.")}
            target_projector.load_state_dict(pstate, strict=True)
            target_projector.eval().requires_grad_(False)
        flat = y.reshape(-1, y.shape[-1])
        transformed = scaler.transform_y(flat).astype(np.float32)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(flat), batch_size):
                encoded = encoder(torch.from_numpy(transformed[start:start + batch_size]))
                if target_projector is not None:
                    encoded = target_projector(encoded)
                outputs.append(encoded.numpy())
    if not outputs:
        raise ValueError("No input wells were supplied")
    features = np.concatenate(outputs).reshape(*y.shape[:-1], encoder.hidden_dim)
    if not np.isfinite(features).all():
        raise FloatingPointError("Existing JEPA encoder produced nonfinite features")
    return features, {"checkpoint": str(checkpoint.resolve()), "run_directory": str(directory.resolve()),
            "arm": payload.get("arm"), "epoch": payload.get("epoch"),
            "representation": "teacher_projector_after_teacher_encoder" if projector else "teacher_encoder",
            "pretraining_ids": list(manifest["train_ids"]),
            "diagnostic_validation_ids": list(manifest.get("validation_ids", [])),
            "dimension": encoder.hidden_dim, "feature_names": names,
            "input_transform": "saved-run training-only affine scaler; no new clipping",
            "numerical_scaler_independently_verified": expected_scaler is not None,
            "independent_of_pretraining_only_if_test_ids_disjoint": True,
            "optimization_performed": False}
