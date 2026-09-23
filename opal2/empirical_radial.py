"""Bounded empirical log-radius laws with a Gaussian full-support guard.

The supplied matrix is scatter, not generally covariance. Uniform spherical
directions are independent of a common radius for each complete residual.
No mean or variance normalization is applied to the empirical radius law.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammainc, gammaln, logsumexp
from scipy.stats import chi

EPSILON = .10


def fit_radial(radii, dimension=9):
    """Fit only log-radius centers and the declared robust triangular bandwidth."""
    radii = np.asarray(radii, dtype=float)
    if (radii.ndim != 1 or len(radii) == 0 or not np.isfinite(radii).all()
            or np.any(radii <= 0)):
        raise ValueError('Calibration radii must be a nonempty finite positive vector; no clipping')
    if not isinstance(dimension, (int, np.integer)) or dimension < 1:
        raise ValueError('Dimension must be a positive integer')
    centers = np.log(radii)
    sd = float(np.std(centers, ddof=1)) if len(centers) > 1 else 0.
    iqr = float(np.subtract(*np.percentile(centers, [75, 25])))
    robust_scale = min(sd, iqr/1.349) if iqr > 0 else sd
    bandwidth = max(.1, .9*robust_scale*len(centers)**(-.2))
    if not np.isfinite(bandwidth) or np.max(centers)+bandwidth >= np.log(np.finfo(float).max):
        raise ValueError('Empirical support exceeds finite numerical range')
    return dict(log_centers=centers.copy(), bandwidth=bandwidth, epsilon=EPSILON,
        dimension=int(dimension), calibration_n=len(centers), log_radius_sd=sd,
        log_radius_iqr=iqr, robust_scale=robust_scale,
        bandwidth_rule='max(.1, .9*min(sd_ddof1,IQR/1.349)*m^(-.2)); use sd if IQR=0',
        covariance_interpretation='scatter times variance_multiplier; no moment normalization')


def reference_weights(cal_logamp, query_logamp, fit_amp_sd, conditional=False):
    """Feature-only weights and actual final ESS, with local-ESS shrinkage."""
    cal, query = np.asarray(cal_logamp, dtype=float), np.asarray(query_logamp, dtype=float)
    if (cal.ndim != 1 or query.ndim != 1 or min(len(cal), len(query)) < 1
            or not np.isfinite(cal).all() or not np.isfinite(query).all()):
        raise ValueError('Calibration and query log amplitudes must be finite nonempty vectors')
    uniform = np.full((len(query), len(cal)), 1/len(cal))
    if not conditional:
        return dict(weights=uniform, ess=np.full(len(query), float(len(cal))),
                    local_ess=np.full(len(query), float(len(cal))), shrinkage=np.zeros(len(query)))
    if not np.isfinite(fit_amp_sd) or fit_amp_sd <= 0:
        raise ValueError('Conditional bandwidth fit_amp_sd must be finite and positive')
    logits = -.5*np.square((query[:, None]-cal[None, :])/fit_amp_sd)
    if not np.isfinite(logits).all():
        raise ValueError('Amplitude distances exceed finite numerical range')
    shifted = logits-logits.max(axis=1, keepdims=True)
    local = np.exp(shifted-logsumexp(shifted, axis=1, keepdims=True))
    local_ess = 1/np.square(local).sum(1)
    shrinkage = local_ess/(local_ess+20.)
    weights = (1-shrinkage[:, None])*uniform+shrinkage[:, None]*local
    return dict(weights=weights, ess=1/np.square(weights).sum(1),
                local_ess=local_ess, shrinkage=shrinkage)


def _law_weights(law, weights):
    centers = np.asarray(law['log_centers'], dtype=float)
    h, epsilon, dimension = float(law['bandwidth']), float(law['epsilon']), law['dimension']
    if (centers.ndim != 1 or len(centers) == 0 or not np.isfinite(centers).all()
            or not np.isfinite(h) or h <= 0 or epsilon != EPSILON
            or not isinstance(dimension, (int, np.integer)) or dimension < 1):
        raise ValueError('Invalid empirical radial law')
    weights = np.asarray(weights, dtype=float)
    if (weights.ndim != 2 or weights.shape[1] != len(centers) or len(weights) == 0
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.allclose(weights.sum(1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Weights must be finite nonnegative normalized [N,calibration_n]')
    return centers, h, epsilon, dimension, weights


def _scatter(scatter, count, dimension):
    scatter = np.asarray(scatter, dtype=float)
    if scatter.shape != (count, dimension, dimension) or not np.isfinite(scatter).all():
        raise ValueError('Scatter must be finite [N,d,d]')
    if not np.allclose(scatter, scatter.swapaxes(-1, -2), atol=1e-12, rtol=1e-12):
        raise ValueError('Scatter must be symmetric')
    try:
        return np.linalg.cholesky(scatter)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Scatter must be positive definite; no clipping') from exc


def radial_cdf(law, weights, radii):
    """CDF of the whitened radius, one radius per query; negative inputs give zero."""
    centers, h, epsilon, d, weights = _law_weights(law, weights)
    radii = np.asarray(radii, dtype=float)
    if radii.shape != (len(weights),) or np.isnan(radii).any():
        raise ValueError('Radii must be an aligned [N] vector without NaNs')
    result = np.zeros(len(weights))
    positive = radii > 0
    # Clipping this piecewise CDF argument implements its constant endpoint
    # pieces; it does not clip a random draw, residual, or fitted distribution.
    scaled = np.clip((np.log(radii[positive, None])-centers)/h, -1., 1.)
    triangular = np.where(scaled <= 0, .5*(scaled+1)**2, 1-.5*(1-scaled)**2)
    result[positive] = ((1-epsilon)*np.sum(weights[positive]*triangular, axis=1)
                       +epsilon*gammainc(d/2, np.square(radii[positive])/2))
    return result


def radial_ppf(law, weights, levels):
    """Vectorized bisection quantiles [N,levels]; endpoints are zero and infinity."""
    centers, h, _, d, weights = _law_weights(law, weights)
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    if levels.ndim != 1 or not np.isfinite(levels).all() or np.any((levels < 0) | (levels > 1)):
        raise ValueError('Levels must lie in [0,1]')
    result = np.empty((len(weights), len(levels)))
    upper_support = np.exp(np.max(centers)+h)
    if not np.isfinite(upper_support):
        raise ValueError('Empirical support exceeds finite numerical range')
    for column, level in enumerate(levels):
        if level == 0 or level == 1:
            result[:, column] = 0. if level == 0 else np.inf
            continue
        low = np.zeros(len(weights))
        high = np.full(len(weights), max(upper_support, chi.ppf(level, d)))
        for _ in range(1100):
            middle = low+(high-low)/2
            below = radial_cdf(law, weights, middle) < level
            low, high = np.where(below, middle, low), np.where(below, high, middle)
            if np.all(high-low <= 1e-12*np.maximum(high, np.finfo(float).tiny)):
                break
        else:
            raise ValueError('Radius quantile bisection did not converge')
        result[:, column] = low+(high-low)/2
    return result


def radial_nll(residual, scatter, law, weights):
    """Exact complete-vector density including spherical and scatter Jacobians."""
    centers, h, epsilon, d, weights = _law_weights(law, weights)
    factor = _scatter(scatter, len(weights), d)
    residual = np.asarray(residual, dtype=float)
    if residual.shape != (len(weights), d) or not np.isfinite(residual).all():
        raise ValueError('Residuals must be finite aligned [N,d]')
    whitened = np.linalg.solve(factor, residual[..., None])[..., 0]
    radius = np.linalg.norm(whitened, axis=1)
    log_guard = np.log(epsilon)-.5*(d*np.log(2*np.pi)+np.square(radius))
    log_empirical = np.full(len(weights), -np.inf)
    positive = radius > 0
    log_radius = np.log(radius[positive])
    density_log_radius = np.sum(weights[positive]
        *np.maximum(1-np.abs(log_radius[:, None]-centers)/h, 0)/h, axis=1)
    supported = density_log_radius > 0
    positions = np.flatnonzero(positive)[supported]
    log_sphere_area = np.log(2)+(d/2)*np.log(np.pi)-gammaln(d/2)
    log_empirical[positions] = (np.log1p(-epsilon)+np.log(density_log_radius[supported])
        -d*log_radius[supported]-log_sphere_area)
    log_determinant_factor = np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(1)
    return -np.logaddexp(log_guard, log_empirical)+log_determinant_factor


def draw_radial(law, weights, scatter, normal, mixtureuniforms, kerneluniforms):
    """Draw whole residual vectors with one radius and independent sphere direction.

The first uniform selects the guard or an empirical center after remapping.
The second selects triangular log-radius noise. No coordinate-wise sampling.
"""
    centers, h, epsilon, d, weights = _law_weights(law, weights)
    factor = _scatter(scatter, len(weights), d)
    normal = np.asarray(normal, dtype=float)
    mix, kernel = np.asarray(mixtureuniforms, dtype=float), np.asarray(kerneluniforms, dtype=float)
    if normal.ndim != 3 or normal.shape[1:] != (len(weights), d) or not np.isfinite(normal).all():
        raise ValueError('Normal draws must be finite [S,N,d]')
    for value in (mix, kernel):
        if value.shape != normal.shape[:2] or not np.isfinite(value).all() or np.any((value < 0) | (value >= 1)):
            raise ValueError('Uniform streams must be [S,N] with values in [0,1)')
    norm = np.linalg.norm(normal, axis=-1)
    empirical = mix >= epsilon
    if np.any(norm[empirical] <= 0):
        raise ValueError('An empirical draw requires a nonzero normal direction')
    whitened = normal.copy()
    noise = np.where(kernel < .5, h*(np.sqrt(2*kernel)-1), h*(1-np.sqrt(2-2*kernel)))
    for query, row_weights in enumerate(weights):
        selected = empirical[:, query]
        # Omit zero-weight centers so an endpoint rounding error cannot select
        # an impossible donor. Normalize cumulative sums only for inversion.
        positive = np.flatnonzero(row_weights > 0)
        cumulative = np.cumsum(row_weights[positive])
        cumulative /= cumulative[-1]
        uniform_center = (mix[selected, query]-epsilon)/(1-epsilon)
        position = np.searchsorted(cumulative, uniform_center, side='right')
        position = np.minimum(position, len(positive)-1)
        radius = np.exp(centers[positive[position]]+noise[selected, query])
        whitened[selected, query] *= (radius/norm[selected, query])[:, None]
    result = np.einsum('nij,snj->sni', factor, whitened)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite draws; no clipping or sample removal is applied')
    return result


def variance_multiplier(law, weights):
    """Exact E[R²]/d; predictive covariance equals this multiplier times scatter."""
    centers, h, epsilon, d, weights = _law_weights(law, weights)
    log_weights = np.full(weights.shape, -np.inf)
    np.log(weights, out=log_weights, where=weights > 0)
    log_sinh_h = h+np.log1p(-np.exp(-2*h))-np.log(2)
    empirical_second_moment = np.exp(logsumexp(log_weights+2*centers, axis=1)
                                     +2*(log_sinh_h-np.log(h)))
    result = epsilon+(1-epsilon)*empirical_second_moment/d
    if not np.isfinite(result).all():
        raise ValueError('The exact variance multiplier exceeds finite numerical range')
    return result
