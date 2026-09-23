"""EU R3 fixed-target, well-profile representation fits around frozen CORE.

Only TRAIN repeat blocks fit the representation estimators. All other objects
are transformed from X alone. These estimators predict future profile direction
and amplitude; they are not raw-image JEPA or residual-likelihood encoders.
The downstream error adapter, not this module, turns states into distributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .conditional_state_representation import (
    ConditionalStateRepresentation, fit_representation,
)


ARTIFACT_VERSION = 1
ARMS = ("PCA_STATE", "DIRECT_STATE", "CONDITIONAL_STATE")
NEURAL_CONFIG = dict(epochs=60, latent_dim=8, target_dim=16, input_dim=32,
    hidden_dim=32, batch_size=64, learning_rate=3e-4, weight_decay=1e-3,
    patience=10, validation_fraction=.2, min_delta=1e-5)


def _x_matrix(value, feature_count=None):
    x = np.asarray(value, dtype=np.float64)
    if (x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all()
            or (feature_count is not None and x.shape[1] != feature_count)):
        raise ValueError("X must be a finite aligned [objects, features] matrix")
    if np.any(np.linalg.norm(x, axis=1) <= 0):
        raise ValueError("Zero-norm X requires the original completion policy")
    return x


def _checked_training(data, train_rows):
    ids, groups = np.asarray(data["ids"], str), np.asarray(data["groups"], str)
    y = data["Y"]
    # Do not validate or materialize the non-TRAIN future values. They are not
    # inputs to fitting or inference, and may deliberately be inaccessible/NaN.
    if len(y.shape) != 3 or y.shape[1] != 4:
        raise ValueError("Y requires [objects, X/Z1/Z2/V, features]")
    n = y.shape[0]
    if ids.shape != (n,) or groups.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError("Unique ids and chemical groups must align with Y")
    supplied = np.asarray(train_rows)
    if supplied.dtype.kind not in "iu" or supplied.ndim != 1:
        raise ValueError("TRAIN rows must be a one-dimensional integer index")
    rows = supplied.astype(np.int64, copy=True)
    if (len(rows) < 5 or len(np.unique(rows)) != len(rows)
            or np.any(rows < 0) or np.any(rows >= n)):
        raise ValueError("At least five distinct valid TRAIN rows are required")
    other = np.ones(n, dtype=bool); other[rows] = False
    if set(groups[rows]) & set(groups[other]):
        raise ValueError("A chemical group crosses TRAIN and held-out roles")
    if len(np.unique(groups[rows])) < 4:
        raise ValueError("TRAIN needs at least four groups for internal validation")
    x = _x_matrix(y[:, 0])
    train = np.asarray(y[rows], dtype=np.float64)
    if not np.isfinite(train).all():
        raise ValueError("Only the supplied TRAIN repeat blocks must be finite")
    return ids, groups, rows, x, train


def _normalization(raw_states, rows):
    """One TRAIN-only scale, preserving relative strengths of latent axes."""
    center = raw_states[rows].mean(0)
    scale = max(float(np.sqrt(np.mean(np.var(raw_states[rows], axis=0)))), 1e-8)
    return center, np.full(raw_states.shape[1], scale)


def optional_state_features(base_features, state=None, *, enabled=True):
    """Append a fitted state; disabled input is exactly the unchanged baseline.

    This feature invariant is not a claim about a downstream trained adapter.
    Its disabled prediction must separately return the frozen CORE distribution.
    """
    base = np.asarray(base_features)
    if base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError("Base features must be a finite matrix")
    if not enabled:
        return base.copy()
    value = np.asarray(state)
    if value.ndim != 2 or len(value) != len(base) or not np.isfinite(value).all():
        raise ValueError("State must be a finite aligned matrix")
    return np.column_stack((base, value))


def fit_eu_states(data, train_rows, output, seed):
    """Fit the full PCA/direct/conditional recipe; return all X-only states.

    ``states[name]`` is ``(z, scale)``: z is centered by TRAIN and scale is
    a repeated TRAIN RMS scalar, not separate whitening of weak dimensions.
    Explicit amplitude remains in frozen CORE and the descriptor bypass.
    No reference, calibration, outer-query target, or prior encoder is fitted.
    """
    ids, groups, rows, x, train = _checked_training(data, train_rows)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("A nonnegative integer seed is required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve the existing representation fit")
    output.mkdir(parents=True)
    states, arms = {}, {}
    normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    dim = min(8, len(rows)-1, x.shape[1])
    pca = PCA(n_components=dim, svd_solver="full").fit(normalized[rows])
    raw = pca.transform(normalized)
    center, scale = _normalization(raw, rows)
    states["PCA_STATE"] = raw-center, scale
    np.savez_compressed(output/"PCA_STATE.npz", components=pca.components_,
        pca_mean=pca.mean_, state_center=center, state_scale=scale,
        explained_variance=pca.explained_variance_, fit_ids=ids[rows])
    arms["PCA_STATE"] = dict(kind="X unit-direction PCA", dimension=dim,
        model_path="PCA_STATE.npz", fit_ids=ids[rows].tolist(),
        future_targets_used=False, state_center=center.tolist(), state_scale=scale.tolist())
    for arm, kind in (("DIRECT_STATE", "direct"),
                      ("CONDITIONAL_STATE", "conditional_predictive")):
        model = fit_representation(train, groups[rows], ids[rows], kind=kind,
                                   seed=int(seed), **NEURAL_CONFIG)
        model.save(output/(arm+".pt"))
        # The final coordinate is an explicit amplitude bypass, already in CORE.
        raw = model.transform(x)[:, :-1]
        center, scale = _normalization(raw, rows)
        states[arm] = raw-center, scale
        arms[arm] = dict(kind=kind, model_path=arm+".pt", fit_ids=ids[rows].tolist(),
            state_center=center.tolist(), state_scale=scale.tolist(),
            report=model.report, amplitude_bypass_returned_as_state=False)
        fit_ids, valid_ids = set(model.report["inner_fit_ids"]), set(model.report["inner_validation_ids"])
        if fit_ids & valid_ids or fit_ids | valid_ids != set(ids[rows]):
            raise RuntimeError("Representation internal split differs from TRAIN")
        if set(model.report["inner_fit_groups"]) & set(model.report["inner_validation_groups"]):
            raise RuntimeError("Representation internal validation group leakage")
    one, two = (arms[a]["report"] for a in ("DIRECT_STATE", "CONDITIONAL_STATE"))
    if (one["inner_fit_ids"] != two["inner_fit_ids"]
            or one["inner_validation_ids"] != two["inner_validation_ids"]):
        raise RuntimeError("Direct and conditional estimators need the same inner split")
    report = dict(version=ARTIFACT_VERSION, seed=int(seed), ids=ids.tolist(),
        train_ids=ids[rows].tolist(), train_groups=np.unique(groups[rows]).tolist(),
        feature_count=x.shape[1], outer_train_count=len(rows),
        full_neural_configuration=NEURAL_CONFIG, arms=arms,
        objective="fixed future-profile directional PCA and log-RMS; not CORE residual supervision",
        raw_images_used=False, moving_teacher=False, inference="first well X only",
        state_scaling="TRAIN centering plus a single repeated RMS scale; not per-axis whitening",
        endpoint_changed=False, core_mean_changed=False,
        architecture_status="complete existing well-profile estimators; downstream scatter adapter is external")
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(output/"states.npz", ids=ids, train_ids=ids[rows],
        **{name:z for name,(z, _) in states.items()},
        **{name+"_scale":sd for name,(_,sd) in states.items()})
    (output/"complete.json").write_text(json.dumps(dict(complete=True,
        report="report.json", arms=list(ARMS), train_count=len(rows)))+"\n")
    return dict(states=states, report=report,
                model_paths={name:str(output/row["model_path"]) for name,row in arms.items()})


def load_eu_states(output, first_well):
    """Reload frozen estimators and transform X; future wells are not accepted."""
    output = Path(output)
    if not (output/"complete.json").is_file():
        raise ValueError("Representation artifacts are incomplete")
    report = json.loads((output/"report.json").read_text())
    if report["version"] != ARTIFACT_VERSION:
        raise ValueError("Unsupported EU representation artifact")
    x = _x_matrix(first_well, report["feature_count"])
    states = {}
    with np.load(output/"PCA_STATE.npz", allow_pickle=False) as pca:
        direction = x/np.linalg.norm(x, axis=1, keepdims=True)
        raw = (direction-pca["pca_mean"]) @ pca["components"].T
        states["PCA_STATE"] = raw-pca["state_center"], pca["state_scale"].copy()
    for name in ("DIRECT_STATE", "CONDITIONAL_STATE"):
        row = report["arms"][name]
        model = ConditionalStateRepresentation.load(output/row["model_path"])
        raw = model.transform(x)[:, :-1]
        states[name] = raw-np.asarray(row["state_center"]), np.asarray(row["state_scale"])
    return dict(states=states, report=report)
