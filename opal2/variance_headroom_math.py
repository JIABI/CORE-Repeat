"""Finite-grid hindsight diagnostics for the frozen two-scale adapter family.

These optimizers deliberately use realized query outcomes. Their output is
an outcome-informed *finite-grid* comparator, not a deployable predictor, a
continuous optimum, or an information-theoretic ceiling. Optimization draws
must not be reused as final evaluation draws.

The rank-three contrast projector is a geometric derivative subspace. Its
complement does not identify physical shared measurement noise.
"""
from __future__ import annotations

import time

import numpy as np

from .empirical_radial import draw_radial, radial_nll
from .joint_contrast_scale import contrast_projector, projected_energy


ETA_BOUND = float(np.log(4.))


def gamma_only(raw):
    """Original ADD_TWO Gamma without allocating unrelated observables.

The mean of X/Z1/Z2 may be replaced by their sum only inside the cosine:
the common factor of three cancels, while their relative amplitudes remain.
"""
    u = np.asarray(raw, dtype=np.float64)
    if u.shape[-1] != 9 or not np.isfinite(u).all():
        raise ValueError('Expected finite nine-dimensional geometry')
    diag = np.exp(u[..., (3, 5, 8)])
    if not np.isfinite(diag).all() or np.any(diag <= 0) or np.any(diag**2 == 0):
        raise ValueError('Invalid factor diagonal; no clipping or resampling')
    a0, a1, a2 = 1+u[..., 0]+u[..., 1], diag[..., 0]+u[..., 4], diag[..., 1]
    v0, v1, v2, v3 = u[..., 2], u[..., 6], u[..., 7], diag[..., 2]
    anorm2 = a0*a0+a1*a1+a2*a2
    vnorm2 = v0*v0+v1*v1+v2*v2+v3*v3
    if np.any(anorm2 <= 0) or np.any(vnorm2 <= 0):
        raise ValueError('Zero acquired or validation norm')
    value = .5*((a0*v0+a1*v1+a2*v2)/np.sqrt(anorm2*vnorm2)
                 - v0/np.sqrt(vnorm2))-.02
    if not np.isfinite(value).all():
        raise ValueError('Geometry overflow; no sampled value is dropped')
    return value


def fair_gamma_crps(draws, actual):
    """Complete unequal-index U-statistic CRPS, O(S log S), not paired halves."""
    draws, actual = np.asarray(draws, float), np.asarray(actual, float)
    if (draws.ndim != 2 or len(draws) < 2 or actual.shape != draws.shape[1:]
            or not np.isfinite(draws).all() or not np.isfinite(actual).all()):
        raise ValueError('Need finite draws [S,N] with S>=2 and actual [N]')
    s = len(draws)
    ordered = np.sort(draws, axis=0)
    rank = (2*np.arange(s)-s+1)[:, None]
    return np.abs(draws-actual).mean(0)-(rank*ordered).sum(0)/(s*(s-1))


def adapter_grid(grid_size=9):
    """A fixed symmetric grid; ties prefer zero and then the smaller increment."""
    if (not isinstance(grid_size, (int, np.integer)) or isinstance(grid_size, bool)
            or grid_size < 3 or grid_size % 2 != 1):
        raise ValueError('grid_size must be an odd integer >=3')
    axis = np.linspace(-ETA_BOUND, ETA_BOUND, grid_size)
    axis[grid_size//2] = 0.
    pairs = np.asarray([(a, b) for a in axis for b in axis])
    order = np.lexsort((pairs[:, 1], pairs[:, 0], np.square(pairs).sum(1)))
    return pairs[order]


def _inputs(mean, scatter, target, stats, actual_gamma, law, weights):
    mean, scatter, target = map(lambda x: np.asarray(x, float), (mean, scatter, target))
    actual, weights = np.asarray(actual_gamma, float), np.asarray(weights, float)
    if (mean.ndim != 2 or mean.shape[1] != 9 or len(mean) < 1
            or target.shape != mean.shape or scatter.shape != (len(mean), 9, 9)
            or actual.shape != (len(mean),)
            or not all(np.isfinite(x).all() for x in (mean, scatter, target, actual, weights))):
        raise ValueError('Expected finite aligned geometry, scatter and actual Gamma')
    scale, center = np.asarray(stats['u_scale'], float), np.asarray(stats['u_center'], float)
    if (scale.shape != (9,) or center.shape != (9,)
            or not np.isfinite(scale).all() or not np.isfinite(center).all() or np.any(scale <= 0)):
        raise ValueError('Finite nine-dimensional center and positive scale required')
    if (law['dimension'] != 9 or weights.shape != (len(mean), len(law['log_centers']))
            or np.any(weights < 0) or not np.allclose(weights.sum(1), 1., rtol=0, atol=1e-12)):
        raise ValueError('Need aligned normalized radial weights and dimension nine')
    if not np.allclose(scatter, scatter.swapaxes(-1, -2), rtol=1e-12, atol=1e-12):
        raise ValueError('Scatter must be symmetric')
    np.linalg.cholesky(scatter)
    return mean, scatter, target, scale, center, actual, weights


def sample_components(scatter, decomposition, law, weights, normal, mixture, kernel):
    """Return exact CORE draws and projected parts using shared primitives.

For spherical W, A=L(exp(eta_c/2)P+exp(eta_r/2)(I-P)) satisfies
A A' = L(exp(eta_c)P+exp(eta_r)(I-P))L'. Consequently A W and a
Cholesky-factor draw from the updated scatter have the same law (their
factors differ by an orthogonal matrix). They need not be samplewise equal
away from zero. At eta=0, the returned CORE draw is exactly the original
draw_radial call, with the same primitive arrays and same scatter.
"""
    scatter = np.asarray(scatter, float)
    eye = np.broadcast_to(np.eye(9), scatter.shape)
    white = draw_radial(law, weights, eye, normal, mixture, kernel)
    base = np.einsum('nij,snj->sni', decomposition['factor'], white)
    projected = np.einsum('nij,snj->sni', decomposition['projector'], white)
    contrast = np.einsum('nij,snj->sni', decomposition['factor'], projected)
    return base, contrast


def residual_at_eta(base, contrast, eta):
    """Two log-variance increments; exact CORE draw when eta is zero."""
    base, contrast, eta = np.asarray(base, float), np.asarray(contrast, float), np.asarray(eta, float)
    if base.shape != contrast.shape or base.ndim != 3 or base.shape[-1] != 9:
        raise ValueError('Need aligned [S,N,9] draws')
    if eta.shape == (2,):
        eta = np.broadcast_to(eta, (base.shape[1], 2))
    if (eta.shape != (base.shape[1], 2) or not np.isfinite(eta).all()
            or np.any(np.abs(eta) > ETA_BOUND+1e-12)):
        raise ValueError('Two finite bounded increments per query required')
    # Using increments around zero both preserves the baseline and avoids a
    # P+(I-P) floating-point reconstruction in the zero arm.
    return (base+np.expm1(.5*eta[:, 0])[None, :, None]*contrast
            + np.expm1(.5*eta[:, 1])[None, :, None]*(base-contrast))


def component_nll(residual, decomposition, law, weights, eta):
    """Exact radial NLL with both scatter Jacobians, in the existing coordinates."""
    residual, eta = np.asarray(residual, float), np.asarray(eta, float)
    n = len(residual)
    if eta.shape == (2,):
        eta = np.broadcast_to(eta, (n, 2))
    if (residual.shape != (n, 9) or eta.shape != (n, 2)
            or not np.isfinite(eta).all() or np.any(np.abs(eta) > ETA_BOUND+1e-12)):
        raise ValueError('Aligned bounded increments required')
    L, P = decomposition['factor'], decomposition['projector']
    white = np.linalg.solve(L, residual[..., None])[..., 0]
    wc = np.einsum('nij,nj->ni', P, white)
    transformed = (np.exp(-eta[:, :1]/2)*wc
                   + np.exp(-eta[:, 1:]/2)*(white-wc))
    identity = np.broadcast_to(np.eye(9), (n, 9, 9))
    logdet = np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(1)
    return radial_nll(transformed, identity, law, weights)+logdet+.5*(3*eta[:, 0]+6*eta[:, 1])


def evaluate_candidate_crps(mean, scatter, stats, actual_gamma, law, weights,
                            eta_candidates, seed, samples=4096, *, query_block_size=8):
    """Shared-draw Gamma CRPS [N,K] for declared [K,2] bounded increments.

This function does not select candidates. Supplying realized outcomes makes
the scores diagnostic; any subsequent choice must obey the caller's fitting
or calibration split. It is also suitable for a single CAL object with its
own leave-group-out radial law and only nine scalar candidates.
"""
    mean, scatter, _, scale, center, actual, weights = _inputs(
        mean, scatter, mean, stats, actual_gamma, law, weights)
    if (not isinstance(samples, (int, np.integer)) or isinstance(samples, bool) or samples < 4
            or not isinstance(query_block_size, (int, np.integer)) or isinstance(query_block_size, bool)
            or query_block_size < 1):
        raise ValueError('Positive integer block size and integer samples>=4 required')
    candidates = np.asarray(eta_candidates, float)
    if (candidates.ndim != 2 or candidates.shape[1] != 2 or len(candidates) < 1
            or not np.isfinite(candidates).all() or np.any(np.abs(candidates) > ETA_BOUND+1e-12)):
        raise ValueError('Need finite bounded candidates [K,2]')
    n = len(mean)
    dec = contrast_projector(mean*scale+center, scale, scatter)
    scores = np.empty((n, len(candidates)))
    rng = np.random.default_rng(seed)
    for lo in range(0, n, query_block_size):
        hi = min(n, lo+query_block_size)
        block = slice(lo, hi)
        normal = rng.normal(size=(samples, hi-lo, 9))
        mix, kernel = rng.random((samples, hi-lo)), rng.random((samples, hi-lo))
        bdec = {key: value[block] for key, value in dec.items()}
        base, contrast = sample_components(scatter[block], bdec, law, weights[block], normal, mix, kernel)
        raw_mean = mean[block]*scale+center
        for k, eta in enumerate(candidates):
            raw = raw_mean[None]+residual_at_eta(base, contrast, eta)*scale
            scores[block, k] = fair_gamma_crps(gamma_only(raw), actual[block])
    return scores


def search_hindsight(mean, scatter, target, stats, actual_gamma, law, weights,
                     seed, samples=4096, grid_size=9, *, query_block_size=8):
    """Search a fixed two-scale grid, independently for Gamma CRPS and NLL.

All results are outcome-informed development diagnostics. Each row uses its
own actual Gamma for CRPS, and its entire realized geometry for NLL. Grid
minima use shared optimization draws; report the selected distributions
again on an independent Monte Carlo stream, not only these optimized scores.
No input arrays or fitted CORE objects are modified.
"""
    start = time.monotonic()
    mean, scatter, target, scale, center, actual, weights = _inputs(
        mean, scatter, target, stats, actual_gamma, law, weights)
    if (not isinstance(samples, (int, np.integer)) or isinstance(samples, bool) or samples < 4
            or not isinstance(query_block_size, (int, np.integer)) or isinstance(query_block_size, bool)
            or query_block_size < 1):
        raise ValueError('Positive integer block size and integer samples>=4 required')
    n = len(mean)
    grid = adapter_grid(grid_size)
    dec = contrast_projector(mean*scale+center, scale, scatter)
    energy = projected_energy(target-mean, dec)
    nll = np.stack([component_nll(target-mean, dec, law, weights, eta) for eta in grid], axis=1)
    crps = evaluate_candidate_crps(mean, scatter, stats, actual, law, weights, grid,
        seed, samples=samples, query_block_size=query_block_size)
    scalar = np.flatnonzero(grid[:, 0] == grid[:, 1])
    rows = np.arange(n)
    output = dict(eta_grid=grid, grid_eta=grid.copy(), grid_crps=crps, grid_nll=nll,
                  actual_geometry_energy=energy, actual_total_energy=energy.sum(1),
                  core_crps=crps[:, 0].copy(), core_nll=nll[:, 0].copy(),
                  optimization_samples=int(samples), grid_size=int(grid_size),
                  search_kind='outcome-informed finite-grid minimum, not continuous or information bound',
                  elapsed_seconds=float(time.monotonic()-start))
    for target_name, score in [('crps', crps), ('nll', nll)]:
        for family, choices in [('scalar', scalar), ('two', np.arange(len(grid)))]:
            index = choices[np.argmin(score[:, choices], axis=1)]
            key = family+'_'+target_name
            output['eta_'+key] = grid[index].copy()
            output['index_'+key] = index
            output['crps_'+key] = crps[rows, index]
            output['nll_'+key] = nll[rows, index]
    return output
