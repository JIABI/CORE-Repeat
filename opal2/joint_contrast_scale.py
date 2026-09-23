"""Conditional scales for observable-sensitive predictive error subspaces.

The remainder subspace is NOT an identified physical shared-noise component.
All means stay fixed in the original standardized log-Cholesky coordinates.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def pair_jacobian(raw, target_scale):
    """Exact Jacobian of log1p(||future_a-future_b||^2) wrt standardized u."""
    raw = np.asarray(raw, float)
    scale = np.asarray(target_scale, float)
    if raw.ndim != 2 or raw.shape[1] != 9 or scale.shape != (9,):
        raise ValueError('Expected N by 9 raw means and nine coordinate scales')
    if not np.isfinite(raw).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid geometry or target scale')
    n = len(raw)
    vectors = np.zeros((n, 3, 4))
    vectors[:, :, 0] = raw[:, :3]
    mapping = ((3, 0, 1, True), (4, 1, 1, False), (5, 1, 2, True),
               (6, 2, 1, False), (7, 2, 2, False), (8, 2, 3, True))
    for coord, role, feature, logged in mapping:
        vectors[:, role, feature] = np.exp(raw[:, coord]) if logged else raw[:, coord]
    result = np.zeros((n, 3, 9))
    for p, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        delta = vectors[:, a]-vectors[:, b]
        denom = 1+np.square(delta).sum(1)
        result[:, p, a] = 2*delta[:, 0]/denom
        result[:, p, b] = -2*delta[:, 0]/denom
        for coord, role, feature, logged in mapping:
            sign = int(role == a)-int(role == b)
            chain = vectors[:, role, feature] if logged else 1.
            result[:, p, coord] = sign*2*delta[:, feature]*chain/denom
    result *= scale[None, None, :]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite Jacobian')
    return result


def contrast_projector(raw_mean, target_scale, covariance):
    covariance = np.asarray(covariance, float)
    L = np.linalg.cholesky(covariance)
    jac = pair_jacobian(raw_mean, target_scale)
    jac_white = jac @ L
    _, singular, vh = np.linalg.svd(jac_white, full_matrices=False)
    ranks = (singular > 1e-10*singular[:, :1]).sum(1)
    if np.any(ranks != 3):
        raise ValueError('Pair-observable derivative does not have rank three')
    projector = vh.swapaxes(-1, -2) @ vh
    return dict(projector=projector, factor=L, singular=singular, rank=ranks,
                jacobian=jac, whitened_jacobian=jac_white)


def projected_energy(residual, decomposition):
    residual = np.asarray(residual, float)
    w = np.linalg.solve(decomposition['factor'], residual[..., None])[..., 0]
    wc = np.einsum('nij,nj->ni', decomposition['projector'], w)
    wr = w-wc
    return np.column_stack((np.square(wc).sum(1), np.square(wr).sum(1)))


def fit_scale(energy, degrees, log_amplitude, *, conditional, penalty=1.):
    """Gaussian projection likelihood, not regression on noisy log energies."""
    energy, x = np.asarray(energy, float), np.asarray(log_amplitude, float)
    if energy.ndim != 1 or x.shape != energy.shape or len(x) < 3:
        raise ValueError('Scale-fitting inputs do not align')
    if np.any(energy < 0) or not np.isfinite(energy).all() or not np.isfinite(x).all():
        raise ValueError('Nonfinite or negative scale targets')
    if degrees <= 0 or penalty < 0 or energy.mean() <= 0:
        raise ValueError('Nonpositive variance information')
    center = float(x.mean())
    scale = float(x.std())
    if conditional and scale <= 0:
        raise ValueError('Conditional scale requires variable observed amplitude')
    if not conditional:
        return dict(coef=[float(np.log(energy.mean()/degrees))], center=center,
                    scale=scale, conditional=False, degrees=int(degrees), penalty=penalty,
                    success=True, iterations=0, gradient_max=0.)
    design = np.column_stack((np.ones(len(x)), (x-center)/scale))

    def objective(theta):
        eta = design @ theta
        weighted = energy*np.exp(-eta)
        value = .5*np.mean(degrees*eta+weighted)+.5*penalty*theta[1]**2
        gradient = .5*design.T @ (degrees-weighted)/len(x)
        gradient[1] += penalty*theta[1]
        return value, gradient

    result = minimize(objective, [np.log(energy.mean()/degrees), 0.], jac=True,
                      method='BFGS', options=dict(gtol=1e-8, maxiter=1000))
    gradient_max = float(np.max(np.abs(result.jac)))
    if not result.success and gradient_max > 1e-6:
        raise RuntimeError('Scale optimizer failed: '+str(result.message))
    return dict(coef=result.x.tolist(), center=center, scale=scale, conditional=True,
                degrees=int(degrees), penalty=penalty, success=True,
                iterations=int(result.nit), gradient_max=gradient_max)


def predict_scale(fit, log_amplitude):
    x = np.asarray(log_amplitude, float)
    eta = np.full(x.shape, fit['coef'][0])
    if fit['conditional']:
        eta += fit['coef'][1]*(x-fit['center'])/fit['scale']
    values = np.exp(eta)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Scale extrapolation produced an invalid variance')
    return values


def rescale_components(covariance, decomposition, contrast_scale, remainder_scale):
    covariance = np.asarray(covariance, float)
    n = len(covariance)
    a, b = np.broadcast_to(contrast_scale, (n,)), np.broadcast_to(remainder_scale, (n,))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError('Component variance multipliers must be finite and positive')
    if np.all(a == 1) and np.all(b == 1):
        return covariance.copy()
    P, L = decomposition['projector'], decomposition['factor']
    white = a[:, None, None]*P+b[:, None, None]*(np.eye(9)-P)
    result = L @ white @ L.swapaxes(-1, -2)
    result = .5*(result+result.swapaxes(-1, -2))
    np.linalg.cholesky(result)
    return result
