"""Off-diagonal covariance moments for overlapping relation Gram kernels.

These are descriptive response-residual components, not CORE noise estimates.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


KERNEL_NAMES = ('target', 'morphology', 'target_morphology',
                'shared_batch', 'shared_source_plate')
MODELS = {
    'TARGET_ONLY': (0,),
    'TARGET_TECHNICAL': (0, 3, 4),
    'ADJUSTED': (0, 1, 3, 4),
    'SIGNED_ONLY': (1, 2, 3, 4),
    'SIGNED_JOINT': (0, 1, 2, 3, 4),
}


def normalized_rows(x):
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1)
    return np.divide(x, norms[:, None], out=np.zeros_like(x),
                     where=norms[:, None] > 1e-12)


def relation_grams(target, source_x, batch, source_plate):
    """All returned full matrices are PSD by feature-Gram/Schur construction.

    No entrywise positive clipping is applied to either signed kernel.
    """
    t, m = normalized_rows(target), normalized_rows(source_x)
    kt, km = t @ t.T, m @ m.T
    return np.stack((kt, km, kt * km,
                     np.equal.outer(batch, batch).astype(float),
                     np.equal.outer(source_plate, source_plate).astype(float)), axis=-1)


def group_dyad_pairs(groups):
    """Distinct-group edges; aliases sum to one within each group dyad."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    i, j = np.triu_indices(len(groups), 1)
    take = inverse[i] != inverse[j]
    i, j = i[take], j[take]
    w = 1. / (counts[inverse[i]] * counts[inverse[j]])
    return i, j, w


def node_pair_weights(multiplicity, left, right, weights):
    """Pigeonhole/node bootstrap: repeated nodes weight every incident edge."""
    return multiplicity[:, left] * multiplicity[:, right] * weights[None, :]


def unit_moments(x, y, weights):
    """Weighted moments after freely fitting this unit's intercept.

    Inputs weights=(replicates, edges), x=(edges,k), y=(edges,tasks).
    Each replicate/unit has mass one, retaining equal-unit estimands.
    """
    w = np.asarray(weights, float)
    mass = w.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError('A sampled unit contains no distinct-group edges')
    w = w / mass[:, None]
    sx, sy = w @ x, w @ y
    xx = (w @ (x[:, :, None] * x[:, None, :]).reshape(len(x), -1)).reshape(
        len(w), x.shape[1], x.shape[1])
    xy = (w @ (x[:, :, None] * y[:, None, :]).reshape(len(x), -1)).reshape(
        len(w), x.shape[1], y.shape[1])
    return xx - sx[:, :, None] * sx[:, None, :], xy - sx[:, :, None] * sy[:, None, :], sx, sy


def solve_moments(gram, cross, indices, *, tolerance=1e-10):
    """Unconstrained least squares plus NNLS in the same moment geometry.

    Nuisance unit intercepts have already been removed. All named kernels,
    including technical kernels, receive nonnegative coefficients in NNLS.
    Rank-deficient estimates are returned but marked non-identifiable.
    """
    g = np.asarray(gram)[np.ix_(indices, indices)]
    h = np.asarray(cross)[list(indices)]
    scale = np.sqrt(np.maximum(np.diag(g), 1e-30))
    correlation = g / scale[:, None] / scale[None, :]
    eigen, vectors = np.linalg.eigh((correlation + correlation.T) / 2)
    rank = int(np.sum(eigen > tolerance * max(float(eigen.max()), 1.)))
    raw_scaled = np.linalg.lstsq(correlation, h / scale, rcond=tolerance)[0]
    raw = raw_scaled / scale
    keep = eigen > tolerance * max(float(eigen.max()), 1.)
    if keep.any():
        a = np.sqrt(eigen[keep])[:, None] * vectors[:, keep].T
        b = (vectors[:, keep].T @ (h / scale)) / np.sqrt(eigen[keep])
        bounded = nnls(a, b, maxiter=1000)[0] / scale
    else:
        bounded = np.zeros(len(indices))
    return dict(raw=raw, nonnegative=bounded, rank=rank,
                identifiable=rank == len(indices),
                condition_number=float(eigen.max() / eigen.min()) if rank == len(indices) else None,
                eigenvalues=eigen, kernel_correlation=correlation)


def partial_design_fraction(gram, column, other_columns):
    """Unexplained fraction of a kernel's unit-demeaned design variance."""
    base = float(gram[column, column])
    if base <= 1e-20:
        return 0.
    if not other_columns:
        return 1.
    other = list(other_columns)
    residual = base - gram[column, other] @ np.linalg.pinv(
        gram[np.ix_(other, other)], rcond=1e-10) @ gram[other, column]
    return float(np.clip(residual / base, 0., 1.))
