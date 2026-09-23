"""Exact four-well geometry for the existing three half-cosine actions.

The well order is X, Z1, Z2, V. A single positive scale is applied to the
entire four-well Gram matrix, never separately to its rows. This preserves
relative well amplitudes and the original averaging actions. All leading
dimensions, including Monte Carlo sample and compound dimensions, are kept.

Log-Cholesky coordinates describe only the positive-definite Schur interior.
Singular residual geometry is a boundary, not something this module silently
repairs with a diagonal floor. Zero norms and nonfinite geometry raise errors;
the caller must use the experiment's existing failure policy.
"""
from __future__ import annotations

import math

import torch


WELL_NAMES = ("X", "Z1", "Z2", "V")
COSINE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
OBSERVABLE_NAMES = (
    "cos_X_Z1", "cos_X_Z2", "cos_X_V", "cos_Z1_Z2", "cos_Z1_V", "cos_Z2_V",
    "norm_X_over_norm_X", "norm_Z1_over_norm_X", "norm_Z2_over_norm_X", "norm_V_over_norm_X",
    "norm2_Z1_minus_Z2_over_norm2_X", "norm2_Z1_minus_V_over_norm2_X", "norm2_Z2_minus_V_over_norm2_X",
    "norm2_mean_Z1_Z2_over_norm2_X", "norm2_mean_Z1_V_over_norm2_X", "norm2_mean_Z2_V_over_norm2_X",
    "norm2_mean_X_Z1_over_norm2_X", "norm2_mean_X_Z2_over_norm2_X",
    "norm2_mean_X_Z1_Z2_over_norm2_X", "norm2_mean_Z1_Z2_V_over_norm2_X",
)
COORDINATE_NAMES = ("p_Z1", "p_Z2", "p_V", "log_L00", "L10", "log_L11", "L20", "L21", "log_L22")
ACTION_NAMES = ("ADD_ONE_Z1", "ADD_ONE_Z2", "ADD_TWO")


def _floating(value, label):
    value = value if torch.is_tensor(value) else torch.as_tensor(value)
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{label} must use real float32 or float64 coordinates")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains nonfinite values")
    return value


def _gram(value, *, normalized=False):
    value = _floating(value, "Gram matrix")
    if value.ndim < 2 or value.shape[-2:] != (4, 4):
        raise ValueError("Gram geometry requires final dimensions [4, 4] in X/Z1/Z2/V order")
    tolerance = 64 * torch.finfo(value.dtype).eps
    if not torch.allclose(value, value.transpose(-1, -2), rtol=tolerance, atol=tolerance):
        raise ValueError("The Gram matrix must be symmetric")
    if bool((value[..., 0, 0] <= 0).any()):
        raise ValueError("X has zero or negative squared norm; no epsilon replacement is permitted")
    if normalized and not torch.allclose(value[..., 0, 0], torch.ones_like(value[..., 0, 0]),
                                         rtol=tolerance, atol=tolerance):
        raise ValueError("Log-Cholesky coordinates require the whole Gram block normalized to G00=1")
    # A small eigensolver checks physical validity, not an empirical covariance
    # estimate. Allow roundoff at a PSD boundary but never alter the matrix.
    symmetric = (value+value.transpose(-1, -2))/2
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    scale = value.abs().amax(dim=(-2, -1)).clamp_min(torch.finfo(value.dtype).tiny)
    if bool((eigenvalues[..., 0] < -tolerance*scale).any()):
        raise ValueError("The Gram matrix is not positive semidefinite")
    return value


def profiles_to_gram(y, normalize_x=True):
    """Return [...,4,4] from finite profiles [...,4,D], preserving dtype/device.

    With ``normalize_x=True``, divide the entire Gram by ||X||² so G00=1.
    This is a common scale, not unit normalization of each future well. The
    unnormalized option returns physical dot products. No zero-X fallback is
    defined because the original cosine endpoint is itself then undefined.
    """
    y = _floating(y, "Profiles")
    if y.ndim < 2 or y.shape[-2] != 4 or y.shape[-1] < 1:
        raise ValueError("Profiles must have final dimensions [4, D] for the four declared roles")
    if not isinstance(normalize_x, bool):
        raise ValueError("normalize_x must be boolean")
    gram = y @ y.transpose(-1, -2)
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("Profile inner products overflowed; no implicit clipping is permitted")
    s = gram[..., 0, 0]
    if bool((s <= 0).any()):
        raise ValueError("X has zero squared norm; the original endpoint is undefined")
    if normalize_x:
        gram = gram/s[..., None, None]
        if not bool(torch.isfinite(gram).all()):
            raise ValueError("Normalization by the X norm produced nonfinite geometry")
    return gram


def gram_to_coordinates(gram):
    """Map normalized positive-definite Schur geometry to nine joint targets.

    Order: p_Z1,p_Z2,p_V,log(L00),L10,log(L11),L20,L21,log(L22),
    where p=G[1:,0] and H=G[1:,1:]-p pᵀ=L Lᵀ. These are nine
    coordinates, not a claim of nine statistically independent variables.
    A singular H cannot be represented by finite log diagonals and raises.
    """
    gram = _gram(gram, normalized=True)
    p = gram[..., 1:, 0]
    residual = gram[..., 1:, 1:]-p.unsqueeze(-1)*p.unsqueeze(-2)
    residual = (residual+residual.transpose(-1, -2))/2
    factor, info = torch.linalg.cholesky_ex(residual, check_errors=False)
    if bool((info != 0).any()) or not bool(torch.isfinite(factor).all()):
        raise ValueError("Schur residual must be numerically positive definite; singular boundaries are not floored")
    return torch.stack((p[..., 0], p[..., 1], p[..., 2],
        factor[..., 0, 0].log(), factor[..., 1, 0], factor[..., 1, 1].log(),
        factor[..., 2, 0], factor[..., 2, 1], factor[..., 2, 2].log()), dim=-1)


def coordinates_to_gram(coordinates):
    """Inverse log-Cholesky map [...,9] -> normalized Gram [...,4,4].

    Finite positive Cholesky diagonals give a mathematically positive-definite
    Gram. Extremely ill-conditioned coordinates may exceed floating-point
    resolution; no clipping or hidden diagonal regularization is performed.
    """
    u = _floating(coordinates, "Gram coordinates")
    if u.ndim < 1 or u.shape[-1] != 9:
        raise ValueError("Nine joint log-Cholesky coordinates are required")
    p, zero = u[..., :3], torch.zeros_like(u[..., 0])
    diagonal = u[..., (3, 5, 8)].exp()
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal <= 0).any()):
        raise ValueError("Log-Cholesky diagonal overflowed or underflowed to zero")
    factor = torch.stack((
        torch.stack((diagonal[..., 0], zero, zero), -1),
        torch.stack((u[..., 4], diagonal[..., 1], zero), -1),
        torch.stack((u[..., 6], u[..., 7], diagonal[..., 2]), -1)), -2)
    residual = factor @ factor.transpose(-1, -2)
    future = p.unsqueeze(-1)*p.unsqueeze(-2)+residual
    top = torch.cat((torch.ones_like(p[..., :1]), p), -1).unsqueeze(-2)
    bottom = torch.cat((p.unsqueeze(-1), future), -1)
    gram = torch.cat((top, bottom), -2)
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("Decoded Gram geometry overflowed; no implicit clipping is permitted")
    # Finite positive L diagonals alone are insufficient in floating point:
    # squaring can underflow, or adding H to an enormous p pᵀ can erase H.
    # Such output no longer belongs to the numerically invertible interior.
    recovered = future-p.unsqueeze(-1)*p.unsqueeze(-2)
    _, info = torch.linalg.cholesky_ex((recovered+recovered.transpose(-1, -2))/2,
                                      check_errors=False)
    if bool((info != 0).any()):
        raise ValueError("Decoded Schur residual reached a floating-point singular boundary; no floor is added")
    return gram


def _positive_norms(diagonal, label):
    if bool((diagonal <= 0).any()) or not bool(torch.isfinite(diagonal).all()):
        raise ValueError(f"{label} has a zero, negative or nonfinite squared norm; the cosine is undefined")


def _quadratic(gram, weights):
    weights = gram.new_tensor(weights)
    return torch.einsum("ki,...ij,kj->...k", weights, gram, weights)


def gram_observables(gram):
    """Return [...,20] in the fixed ``OBSERVABLE_NAMES`` order.

    Norms are relative to ||X||; squared norms are relative to ||X||².
    Thus normalized and physical Gram input give the same observables. These
    are measurement geometries, not identified noise variance components or
    coordinate-wise interval coverage. Mean differences can affect them too.
    """
    gram = _gram(gram)
    diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
    _positive_norms(diagonal, "A measured well")
    cosine = torch.stack([gram[..., i, j]/(diagonal[..., i]*diagonal[..., j]).sqrt()
                          for i, j in COSINE_PAIRS], -1)
    relative_norms = (diagonal/diagonal[..., :1]).sqrt()
    weights = (
        (0, 1, -1, 0), (0, 1, 0, -1), (0, 0, 1, -1),
        (0, .5, .5, 0), (0, .5, 0, .5), (0, 0, .5, .5),
        (.5, .5, 0, 0), (.5, 0, .5, 0),
        (1/3, 1/3, 1/3, 0), (0, 1/3, 1/3, 1/3),
    )
    squared = _quadratic(gram, weights)/diagonal[..., :1]
    # Keep roundoff visible, rather than adding variance or silently dropping
    # geometry. Squared norms are PSD by construction in exact arithmetic.
    return torch.cat((cosine, relative_norms, squared), -1)


def gram_gains(gram, cost_one=.01, cost_two=.02):
    """Return original ADD_ONE(Z1), ADD_ONE(Z2), ADD_TWO gains [...,3].

    Each entry is half the change in cosine with V, minus the declared cost.
    Apply this function to each JOINT Gram draw before averaging or computing
    NULL probabilities; applying it to the mean Gram is not equivalent.
    """
    gram = _gram(gram)
    if not math.isfinite(cost_one) or not math.isfinite(cost_two) or cost_one < 0 or cost_two < 0:
        raise ValueError("Action costs must be finite and nonnegative")
    x_norm, v_norm = gram[..., 0, 0], gram[..., 3, 3]
    _positive_norms(v_norm, "V")
    weights = ((.5, .5, 0, 0), (.5, 0, .5, 0), (1/3, 1/3, 1/3, 0))
    averaged_norm2 = _quadratic(gram, weights)
    _positive_norms(averaged_norm2, "An acquired average")
    cross = torch.einsum("ki,...i->...k", gram.new_tensor(weights), gram[..., :, 3])
    after = cross/(averaged_norm2*v_norm[..., None]).sqrt()
    before = gram[..., 0, 3]/(x_norm*v_norm).sqrt()
    return .5*(after-before[..., None])-gram.new_tensor((cost_one, cost_one, cost_two))
