"""Mean/covariance-matched mixtures of complete reference residual vectors.

Reference residuals and their independently fitted covariance matrices supply
shape information. The transported mixture has mean zero and the supplied
query covariance. No query outcome enters construction or sampling, no neural
model is fitted, and no eigenvalue clipping is applied. Reference and CAL
provenance/exclusions are the caller's responsibility.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .conditional_joint_error import gaussian_score


ALPHA_GRID = (0., .25, .5, .75, 1.)
TIE_RTOL = 1e-12
TIE_ATOL = 1e-12


def _matrix(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(name+' must be a nonempty finite matrix')
    return result


def _covariance(values, count, dimension, name):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (count, dimension, dimension) or not np.isfinite(result).all():
        raise ValueError(name+' must be an aligned finite covariance batch')
    if not np.allclose(result, result.swapaxes(-1, -2), rtol=1e-12, atol=1e-12):
        raise ValueError(name+' must be symmetric')
    try:
        factor = np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(name+' must be positive definite; no eigenvalue clipping is applied') from error
    return result, factor


def _weights(values, donor_count=None):
    result = _matrix(values, 'weights')
    if (donor_count is not None and result.shape[1] != donor_count):
        raise ValueError('weights must align with reference residual rows')
    if np.any(result < 0) or not np.allclose(result.sum(1), 1., rtol=0., atol=1e-12):
        raise ValueError('weights must be nonnegative and normalized by query')
    return result


def _alpha(value):
    if np.ndim(value) != 0 or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('alpha must be a finite scalar in [0,1]')
    return float(value)


def _mixture(values):
    weights = _weights(values['weights'])
    centers = np.asarray(values['centers'], dtype=np.float64)
    if (centers.ndim != 3 or centers.shape[:2] != weights.shape
            or centers.shape[2] < 1 or not np.isfinite(centers).all()):
        raise ValueError('centers must be a finite query-by-reference-by-coordinate array')
    queries, _, dimension = centers.shape
    covariance, factor = _covariance(values['covariance'], queries, dimension, 'covariance')
    component_covariance, component_factor = _covariance(
        values['component_covariance'], queries, dimension, 'component_covariance')
    return weights, centers, covariance, factor, component_covariance, component_factor


def build_residual_mixture(reference_residual, reference_covariance, weights,
                           query_covariance, bandwidth=.5):
    """Construct a complete-vector Gaussian mixture with mean zero and target C.

    For h=bandwidth, whiten each reference residual by its own Cholesky factor,
    center with the query's weights, and multiply the centers by sqrt(1-h²).
    If V is their weighted second moment plus h²I, B=chol(V), and L=chol(C),
    transport by A=L B^-1. Component centers are A a_j and their common
    covariance is h² A Aᵀ. Consequently the total mean is zero and covariance
    is C. Only 0<h<=1 is supported, so every component is a proper Gaussian.
    """
    residual = _matrix(reference_residual, 'reference_residual')
    donor_count, dimension = residual.shape
    weights = _weights(weights, donor_count)
    if np.ndim(bandwidth) != 0 or not np.isfinite(bandwidth) or not 0 < bandwidth <= 1:
        raise ValueError('bandwidth must be a finite scalar in (0,1]')
    bandwidth = float(bandwidth)
    _, reference_factor = _covariance(reference_covariance, donor_count, dimension, 'reference_covariance')
    covariance, query_factor = _covariance(query_covariance, len(weights), dimension, 'query_covariance')
    whitened = np.linalg.solve(reference_factor, residual[..., None])[..., 0]
    weighted_mean = weights@whitened
    a = np.sqrt(1-bandwidth**2)*(whitened[None]-weighted_mean[:, None])
    initial_covariance = np.einsum('qn,qni,qnj->qij', weights, a, a, optimize=True)
    initial_covariance += bandwidth**2*np.eye(dimension)
    initial_covariance = (initial_covariance+initial_covariance.swapaxes(-1, -2))*.5
    _, initial_factor = _covariance(initial_covariance, len(weights), dimension, 'initial_covariance')
    # A B=L, hence Bᵀ Aᵀ=Lᵀ. A is NOT B^-1 L when the matrices do not commute.
    transport = np.linalg.solve(initial_factor.swapaxes(-1, -2),
                                query_factor.swapaxes(-1, -2)).swapaxes(-1, -2)
    centers = np.einsum('qij,qnj->qni', transport, a, optimize=True)
    component_covariance = bandwidth**2*(transport@transport.swapaxes(-1, -2))
    component_covariance = (component_covariance+component_covariance.swapaxes(-1, -2))*.5
    _covariance(component_covariance, len(weights), dimension, 'component_covariance')
    mean = np.einsum('qn,qni->qi', weights, centers, optimize=True)
    centered = centers-mean[:, None]
    matched_covariance = (component_covariance
        +np.einsum('qn,qni,qnj->qij', weights, centered, centered, optimize=True))
    if not np.isfinite(centers).all() or not np.isfinite(matched_covariance).all():
        raise ValueError('Mixture moment transport overflowed')
    return dict(centers=centers, component_covariance=component_covariance,
        weights=weights.copy(), covariance=covariance.copy(), bandwidth=bandwidth,
        mean=mean, matched_covariance=matched_covariance,
        whitened_reference=whitened, weighted_whitened_mean=weighted_mean,
        initial_covariance=initial_covariance, transport=transport,
        interpretation='complete residual Gaussian mixture matched to fixed mean and covariance')


def mixture_nll(residual, mixture, alpha):
    """Full per-query NLL of (1-alpha)N(0,C)+alpha Σ_j w_j N(center_j,S).

    Components are complete vectors; all coordinates use the same donor index.
    Log-sum-exp preserves numerical stability and zero weights contribute zero
    mass. alpha=0 returns the existing Gaussian scorer exactly.
    """
    alpha = _alpha(alpha)
    weights, centers, covariance, _, _, component_factor = _mixture(mixture)
    residual = _matrix(residual, 'residual')
    if residual.shape != (len(weights), centers.shape[2]):
        raise ValueError('residual must align with mixture queries and coordinates')
    if alpha == 0:
        return gaussian_score(residual, covariance)
    difference = residual[:, None]-centers
    whitened = np.linalg.solve(component_factor[:, None], difference[..., None])[..., 0]
    logdet = 2*np.log(np.diagonal(component_factor, axis1=-2, axis2=-1)).sum(-1)
    component_log_density = -.5*(centers.shape[2]*np.log(2*np.pi)
        +logdet[:, None]+np.square(whitened).sum(-1))
    log_weights = np.full(weights.shape, -np.inf)
    np.log(weights, out=log_weights, where=weights > 0)
    log_density = logsumexp(log_weights+component_log_density, axis=1)
    if alpha < 1:
        log_density = np.logaddexp(np.log1p(-alpha)-gaussian_score(residual, covariance),
                                  np.log(alpha)+log_density)
    score = -log_density
    if not np.isfinite(score).all():
        raise ValueError('Mixture NLL overflowed')
    return score


def sample_mixture(mixture, alpha, normal_draws, component_uniforms, mixture_uniforms):
    """Sample complete residual vectors using only caller-supplied draw streams.

    normal_draws has shape [samples, queries, dimension]. Each uniform stream
    has shape [samples, queries] and values in [0,1). A single donor component
    and base/mixture decision are selected per complete vector, never per
    coordinate. No random-number generator is accessed internally.
    """
    alpha = _alpha(alpha)
    weights, centers, _, factor, _, component_factor = _mixture(mixture)
    draws = np.asarray(normal_draws, dtype=np.float64)
    if (draws.ndim != 3 or draws.shape[0] < 1
            or draws.shape[1:] != (len(weights), centers.shape[2]) or not np.isfinite(draws).all()):
        raise ValueError('normal_draws must be finite [samples, queries, coordinates]')
    uniforms = []
    for values, name in ((component_uniforms, 'component_uniforms'), (mixture_uniforms, 'mixture_uniforms')):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != draws.shape[:2] or not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= 1):
            raise ValueError(name+' must align with draws and contain finite values in [0,1)')
        uniforms.append(values)
    gaussian = np.einsum('qij,sqj->sqi', factor, draws, optimize=True)
    if alpha == 0:
        return gaussian
    cumulative = np.cumsum(weights, axis=1)
    # Normalize the whole CDF so a rounded-down terminal positive mass followed
    # by zero-weight donors cannot select a zero donor at u=nextafter(1,0).
    cumulative /= cumulative[:, -1, None]
    cumulative[:, -1] = 1.
    indices = np.empty(draws.shape[:2], dtype=np.intp)
    for query in range(len(weights)):
        indices[:, query] = np.searchsorted(cumulative[query], uniforms[0][:, query], side='right')
    selected_centers = centers[np.arange(len(weights))[None], indices]
    empirical = selected_centers+np.einsum('qij,sqj->sqi', component_factor, draws, optimize=True)
    if alpha == 1:
        return empirical
    return np.where((uniforms[1] < alpha)[..., None], empirical, gaussian)


def select_mixture_alpha(cal_residual, cal_mixture, grid=ALPHA_GRID):
    """Choose one blend fraction by CAL mean NLL; ties favor smaller alpha.

    No query-label argument is accepted. The reference mixture must already be
    built using reference data only; this helper changes no centers/covariances.
    """
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError('grid must be a nonempty vector of alpha candidates')
    values = sorted(set(_alpha(value) for value in values))
    candidates, best = [], None
    for alpha in values:
        nll = float(mixture_nll(cal_residual, cal_mixture, alpha).mean())
        candidates.append(dict(alpha=alpha, cal_nll=nll))
        if best is None or (nll < best['cal_nll'] and not np.isclose(
                nll, best['cal_nll'], rtol=TIE_RTOL, atol=TIE_ATOL)):
            best = dict(alpha=alpha, cal_nll=nll)
    return dict(**best, candidate_scores=candidates, grid=values,
        gaussian_cal_nll=float(mixture_nll(cal_residual, cal_mixture, 0.).mean()),
        tie_rule='smaller alpha within stated numerical tolerance',
        tie_rtol=TIE_RTOL, tie_atol=TIE_ATOL,
        selection_scope='CAL complete-residual mean NLL; no query labels')
