"""Donor-only covariance fitting around a complete frozen conditional mean.

Residual second moments retain bias relative to the frozen mean. They describe
empirical prediction error, not a centered or purely physical-noise covariance.
Weights are supplied from decision information; callers must exclude complete
chemistry groups from donor leave-group-out weights before calling this module.
There are no query-outcome arguments and no mean correction is fitted here.
"""
from __future__ import annotations

import numpy as np


MIXING_GRID = (0., .25, .5, .75)
FAMILIES = ('GLOBAL_SCALE', 'LOCAL_SCALE', 'LOCAL_DIAG', 'LOCAL_JOINT')
TIE_RTOL = 1e-12
TIE_ATOL = 1e-12


def _matrix(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(name+' must be a nonempty finite matrix')
    return result


def _weights(values, donor_count, name, *, rows=None, loo=False):
    result = _matrix(values, name)
    if result.shape[1] != donor_count or (rows is not None and len(result) != rows):
        raise ValueError(name+' must align with donor columns and query rows')
    if np.any(result < 0) or not np.allclose(result.sum(1), 1., rtol=0., atol=1e-12):
        raise ValueError(name+' must be nonnegative and normalized by row')
    if loo and np.any(np.diag(result) != 0):
        raise ValueError(name+' must exclude each donor from its own prediction')
    return result


def _covariance(values, dimension, name, *, count=None, common=False):
    result = np.asarray(values, dtype=np.float64)
    valid_shape = (result.ndim == 2 and result.shape == (dimension, dimension))
    if not common:
        valid_shape |= (result.ndim == 3 and result.shape == (count, dimension, dimension))
    if not valid_shape or not np.isfinite(result).all():
        raise ValueError(name+' must be a finite aligned covariance matrix or batch')
    if not np.allclose(result, np.swapaxes(result, -1, -2), rtol=1e-12, atol=1e-12):
        raise ValueError(name+' must be symmetric')
    try:
        factor = np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(name+' must be positive definite') from error
    return result, factor


def second_moments(weights, residual):
    """Return Σ_d w_qd r_d r_dᵀ without subtracting the residual mean.

    Complete residual vectors remain intact, preserving all cross-coordinate
    products. This is a normalized population second moment, not an unbiased
    centered covariance estimate.
    """
    residual = _matrix(residual, 'residual')
    weights = _weights(weights, len(residual), 'weights')
    result = np.einsum('qn,ni,nj->qij', weights, residual, residual, optimize=True)
    result = (result+result.swapaxes(-1, -2))*.5
    if not np.isfinite(result).all():
        raise ValueError('Residual second moments overflowed')
    return result


def gaussian_score(residual, cov):
    """Full per-object Gaussian NLL, including d log(2π), at frozen mean.

    cov may be one common d×d covariance or one covariance for each residual.
    A non-positive-definite covariance is rejected, never regularized here.
    """
    residual = _matrix(residual, 'residual')
    _, factor = _covariance(cov, residual.shape[1], 'cov', count=len(residual))
    whitened = np.linalg.solve(factor, residual[..., None])[..., 0]
    logdet = 2*np.log(np.diagonal(factor, axis1=-2, axis2=-1)).sum(-1)
    score = .5*(residual.shape[1]*np.log(2*np.pi)+logdet+np.square(whitened).sum(-1))
    if not np.isfinite(score).all():
        raise ValueError('Gaussian score overflowed')
    return score


def _correlation(covariance):
    variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    std = np.sqrt(variance)
    result = covariance/std[..., :, None]/std[..., None, :]
    indices = np.arange(result.shape[-1])
    result[..., indices, indices] = 1.
    return result


def _with_variances(correlation, variance):
    std = np.sqrt(variance)
    result = std[..., :, None]*correlation*std[..., None, :]
    result = (result+result.swapaxes(-1, -2))*.5
    indices = np.arange(result.shape[-1])
    # Set the requested diagonal directly rather than relying on sqrt(v)**2.
    result[..., indices, indices] = variance
    return result


def _diagonal_family(moment, cov0, correlation0, beta):
    if beta == 0:
        return np.broadcast_to(cov0, moment.shape).copy()
    variance = ((1-beta)*np.diag(cov0)
                +beta*np.diagonal(moment, axis1=-2, axis2=-1))
    return _with_variances(correlation0, variance)


def _select(residual, builder, *, beta=None, stage):
    """Evaluate the fixed grid in ascending order; retain the first tied fit."""
    best, candidates = None, []
    for value in MIXING_GRID:
        covariance = builder(value)
        nll = float(gaussian_score(residual, covariance).mean())
        record = dict(beta=float(value if beta is None else beta),
                      eta=float(0. if beta is None else value),
                      loo_nll=nll, stage=stage)
        candidates.append(record)
        if best is None or (nll < best[0] and not np.isclose(
                nll, best[0], rtol=TIE_RTOL, atol=TIE_ATOL)):
            best = (nll, value, covariance)
    return best, candidates


def fit_covariance_family(residual, cov0, loo_weights, query_weights, family, *,
                          correlation_loo_weights=None, correlation_query_weights=None):
    """Fit fixed-grid covariance mixing using donor leave-group-out NLL only.

    GLOBAL_SCALE expects group-excluded uniform weights; LOCAL_* expect local
    decision-information weights. The helper validates zero self-weight but
    cannot validate whole-group exclusion without group identities. The caller
    is responsible for applying those exclusions in every supplied LOO matrix.

    LOCAL_JOINT first fits exactly the LOCAL_DIAG beta, then selects correlation
    eta with those variances fixed. Its optional pair of correlation weights
    changes only the correlation moments, enabling a matched global-correlation
    control. It never changes marginal fitting or fits a mean correction.
    """
    if family not in FAMILIES:
        raise ValueError('Unknown covariance family: '+str(family))
    residual = _matrix(residual, 'residual')
    count, dimension = residual.shape
    cov0, factor0 = _covariance(cov0, dimension, 'cov0', common=True)
    loo = _weights(loo_weights, count, 'loo_weights', rows=count, loo=True)
    query = _weights(query_weights, count, 'query_weights')
    has_correlation_weights = correlation_loo_weights is not None
    if has_correlation_weights != (correlation_query_weights is not None):
        raise ValueError('Supply both correlation weight matrices or neither')
    if has_correlation_weights and family != 'LOCAL_JOINT':
        raise ValueError('Separate correlation weights are only valid for LOCAL_JOINT')
    local_loo, local_query = second_moments(loo, residual), second_moments(query, residual)
    baseline_nll = float(gaussian_score(residual, cov0).mean())
    correlation0 = _correlation(cov0)
    correlation_loo = correlation_query = None
    diagonal_nll = None

    if family in ('GLOBAL_SCALE', 'LOCAL_SCALE'):
        # Equivalent to tr(C0^-1 M)/d, evaluated as weighted squared whitened
        # residual norms to retain nonnegativity in finite precision.
        whitened = np.linalg.solve(factor0, residual.T)
        squared_norm = np.square(whitened).sum(0)/dimension
        loo_tau, query_tau = loo@squared_norm, query@squared_norm
        if not np.isfinite(loo_tau).all() or not np.isfinite(query_tau).all():
            raise ValueError('Relative residual scales overflowed')

        def scale_cov(tau, beta):
            if beta == 0:
                return np.broadcast_to(cov0, (len(tau), dimension, dimension)).copy()
            return (1-beta+beta*tau)[:, None, None]*cov0

        best, candidates = _select(residual, lambda b: scale_cov(loo_tau, b), stage='scale')
        nll, beta, loo_cov = best
        eta = 0.
        query_cov = scale_cov(query_tau, beta)
    else:
        best, candidates = _select(residual,
            lambda b: _diagonal_family(local_loo, cov0, correlation0, b), stage='diagonal')
        diagonal_nll, beta, loo_cov = best
        nll, eta = diagonal_nll, 0.
        query_cov = _diagonal_family(local_query, cov0, correlation0, beta)
        if family == 'LOCAL_JOINT':
            if has_correlation_weights:
                cw_loo = _weights(correlation_loo_weights, count, 'correlation_loo_weights',
                                  rows=count, loo=True)
                cw_query = _weights(correlation_query_weights, count, 'correlation_query_weights',
                                    rows=len(query))
                correlation_loo = second_moments(cw_loo, residual)
                correlation_query = second_moments(cw_query, residual)
            else:
                correlation_loo, correlation_query = local_loo, local_query
            r_loo = _correlation(.25*cov0+.75*correlation_loo)
            r_query = _correlation(.25*cov0+.75*correlation_query)
            loo_diagonal_cov, query_diagonal_cov = loo_cov, query_cov

            def joint_cov(diagonal_cov, local_correlation, eta):
                if eta == 0:
                    return diagonal_cov.copy()
                correlation = (1-eta)*correlation0+eta*local_correlation
                return _with_variances(correlation,
                    np.diagonal(diagonal_cov, axis1=-2, axis2=-1))

            best, correlation_candidates = _select(residual,
                lambda e: joint_cov(loo_diagonal_cov, r_loo, e), beta=beta, stage='correlation')
            nll, eta, loo_cov = best
            query_cov = joint_cov(query_diagonal_cov, r_query, eta)
            candidates += correlation_candidates

    # Query outcomes never enter selection, but every returned covariance must
    # also be positive definite for honest downstream likelihood evaluation.
    _covariance(query_cov, dimension, 'query_covariance', count=len(query))
    choice = dict(family=family, beta=float(beta), eta=float(eta), loo_nll=float(nll),
        baseline_loo_nll=baseline_nll, diagonal_loo_nll=diagonal_nll,
        candidate_scores=candidates, mixing_grid=list(MIXING_GRID),
        correlation_weights=('separate' if has_correlation_weights else 'local')
            if family == 'LOCAL_JOINT' else 'not_applicable',
        tie_rule='smaller beta, then smaller eta within stated numerical tolerance',
        tie_rtol=TIE_RTOL, tie_atol=TIE_ATOL,
        selection_scope='donor leave-group-out full Gaussian NLL around frozen mean',
        moment_definition='weighted residual outer products; residual mean not subtracted')
    return dict(query_covariance=query_cov, loo_covariance=loo_cov, choice=choice,
        loo_second_moment=local_loo, query_second_moment=local_query,
        correlation_loo_second_moment=correlation_loo,
        correlation_query_second_moment=correlation_query)
