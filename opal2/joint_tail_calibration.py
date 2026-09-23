"""Group-split joint-radius calibration for a fixed mean and covariance law.

The split and one-representative-per-group rule use identities, never outcomes.
Covariance fitting must be completed on disjoint FIT groups by the caller.
Region-only calibration changes an ellipsoid threshold, not its density. The
optional Gaussian scales are separate full-law comparisons, not conformal
guarantees about density or every object in a dependent chemistry group.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING

import numpy as np
from scipy.stats import chi2


def _identities(values, name, *, length=None, unique=False):
    result = np.asarray(values, dtype=str)
    if (result.ndim != 1 or not len(result) or np.any(result == '')
            or (length is not None and len(result) != length)
            or (unique and len(np.unique(result)) != len(result))):
        raise ValueError(name+' must contain aligned '+('unique ' if unique else '')+'nonempty identities')
    return result


def _alpha(alpha):
    if np.ndim(alpha) != 0 or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be finite and strictly between zero and one')
    return float(alpha)


def _rank(count, alpha):
    # Avoid a spurious extra rank at decimal boundaries, e.g. 10*(1-.7),
    # while applying the stated ceiling exactly to the supplied alpha value.
    adjusted = Decimal(count+1)*(Decimal(1)-Decimal(str(alpha)))
    return int(adjusted.to_integral_value(rounding=ROUND_CEILING))


def _residual(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError('residual must be a nonempty finite matrix')
    return result


def _covariance(values, *, dimension=None, count=None):
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim not in (2, 3) or min(result.shape) < 1
            or result.shape[-2] != result.shape[-1]
            or (dimension is not None and result.shape[-1] != dimension)
            or (count is not None and result.ndim == 3 and len(result) != count)
            or not np.isfinite(result).all()):
        raise ValueError('cov must be a finite aligned square matrix or batch')
    if not np.allclose(result, result.swapaxes(-1, -2), rtol=1e-12, atol=1e-12):
        raise ValueError('cov must be symmetric')
    try:
        factor = np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError('cov must be positive definite') from error
    return result, factor


def split_donor_groups(donor_groups, *, seed):
    """Return an outcome-free deterministic 2/3 FIT, 1/3 CAL group split.

    Sorted unique chemistry groups are shuffled with a private seeded generator.
    floor(2G/3) groups enter FIT and all remaining groups enter CAL; at least two
    unique groups are required. Indices refer to the supplied donor-row order.
    Row permutations do not change group membership in either partition.
    """
    groups = _identities(donor_groups, 'donor_groups')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('At least two chemistry groups are needed for a disjoint FIT/CAL split')
    shuffled = np.random.default_rng(int(seed)).permutation(unique)
    fit_count = (2*len(unique))//3
    fit_groups, cal_groups = shuffled[:fit_count], shuffled[fit_count:]
    in_fit = np.isin(groups, fit_groups)
    return dict(fit_indices=np.flatnonzero(in_fit), cal_indices=np.flatnonzero(~in_fit),
        fit_groups=fit_groups.tolist(), cal_groups=cal_groups.tolist(), seed=int(seed),
        n_groups=int(len(unique)), n_fit_groups=int(fit_count), n_cal_groups=int(len(cal_groups)),
        split_rule='seeded shuffle of sorted unique groups; floor(2G/3) FIT, remainder CAL')


def finite_sample_squared_radius(scores, *, alpha=.05):
    """Return the finite-sample split-conformal squared-radius order statistic.

    k=ceil((m+1)*(1-alpha)), using one-based ranks. If k>m, return +infinity,
    including m=0. No interpolated quantile, clamping, or tail extrapolation is
    substituted for the prespecified finite-sample rank.
    """
    alpha = _alpha(alpha)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('scores must be a finite nonnegative vector of squared radii')
    rank = _rank(len(scores), alpha)
    if rank > len(scores):
        return float('inf')
    return float(np.sort(scores)[rank-1])


def mahalanobis_scores(residual, cov):
    """Squared Mahalanobis radii around the supplied, unchanged mean."""
    residual = _residual(residual)
    _, factor = _covariance(cov, dimension=residual.shape[1], count=len(residual))
    whitened = np.linalg.solve(factor, residual[..., None])[..., 0]
    scores = np.square(whitened).sum(-1)
    if not np.isfinite(scores).all():
        raise ValueError('Mahalanobis scores overflowed')
    return scores


def fit_tail_calibration(cal_residual, cal_cov, cal_ids, cal_groups, *, alpha=.05):
    """Calibrate using only ID-min representatives of the supplied CAL groups.

    The returned q is a region-only threshold. full_law_scale=q/chi2_reference
    and gaussian_mle_scale=mean(representative_scores)/d are distinct Gaussian
    density controls. Their validity requires positive finite scales; metadata
    retains zero/infinite values rather than silently replacing them. For
    alpha=.05 the Gaussian reference is exactly chi2.ppf(.95, d).

    Rank-based coverage pertains to exchangeable group representatives under
    the supplied split construction; this does not imply simultaneous coverage
    of every object in a group or calibrated nonlinear marginal observables.
    """
    alpha = _alpha(alpha)
    residual = _residual(cal_residual)
    ids = _identities(cal_ids, 'cal_ids', length=len(residual), unique=True)
    groups = _identities(cal_groups, 'cal_groups', length=len(residual))
    scores = mahalanobis_scores(residual, cal_cov)
    unique_groups = np.unique(groups)
    representatives = np.array([min(np.flatnonzero(groups == group), key=lambda index: ids[index])
                                for group in unique_groups], dtype=np.int64)
    representative_scores = scores[representatives]
    count, dimension = len(representatives), residual.shape[1]
    rank = _rank(count, alpha)
    q = finite_sample_squared_radius(representative_scores, alpha=alpha)
    reference = float(chi2.ppf(1-alpha, dimension))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('Gaussian reference quantile must be positive and finite')
    full_law_scale = q/reference
    mle_scale = float(representative_scores.mean()/dimension)
    return dict(q=q, k=rank, m=count, alpha=alpha, dimension=int(dimension),
        chi2_reference=reference, full_law_scale=full_law_scale, gaussian_mle_scale=mle_scale,
        full_law_available=bool(np.isfinite(full_law_scale) and full_law_scale > 0),
        mle_available=bool(np.isfinite(mle_scale) and mle_scale > 0),
        raw_cal_scores=scores.tolist(), representative_indices=representatives.tolist(),
        representative_ids=ids[representatives].tolist(), representative_groups=unique_groups.tolist(),
        representative_scores=representative_scores.tolist(), cal_ids=ids.tolist(), cal_groups=groups.tolist(),
        cal_group_sizes=[int(np.count_nonzero(groups == group)) for group in unique_groups],
        representative_rule='lexicographically smallest ID in each CAL chemistry group',
        rank_rule='ceil((m+1)*(1-alpha)); one-based order statistic; infinity if k>m',
        region_only_changes_density=False,
        interpretation='held-out group-representative joint-radius calibration; not a pure-noise law')


def scale_covariance(cov, scaling):
    """Apply one positive finite scalar to the full Gaussian covariance law.

    No floor at one or epsilon is applied: shrinking and expansion are both
    allowed. Zero/infinite calibration scales cannot form a nondegenerate
    Gaussian and must remain region-only or be reported as unavailable.
    """
    if np.ndim(scaling) != 0 or not np.isfinite(scaling) or scaling <= 0:
        raise ValueError('Scaling must be positive and finite to form a nondegenerate Gaussian')
    covariance, _ = _covariance(cov)
    scaled = covariance*float(scaling)
    _covariance(scaled)
    return scaled
