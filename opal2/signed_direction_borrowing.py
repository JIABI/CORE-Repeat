"""Source-only signed response borrowing and norm-preserving direction changes.

Weights here act on signed response changes; they are not probability mixtures.
All inputs, donors and fitted Ridge models are supplied by the caller. Only the
small CAL strength selector uses calibration outcomes.
"""
from __future__ import annotations

import numpy as np

from opal2.crossdose_response import (
    EPSILON, _matrix, _support, _groups, _group_mean, apply_response_correction,
)

DIRECTION_GRID = (0., .01, .025, .05, .1, .25, .5, 1.)


def source_cosine(query_source, reference_source):
    x = _matrix(query_source, "query_source")
    r = _matrix(reference_source, "reference_source")
    if x.shape[1] != r.shape[1]:
        raise ValueError("Source profile coordinates differ")
    xn, rn = np.linalg.norm(x, axis=1), np.linalg.norm(r, axis=1)
    xu = np.divide(x, xn[:, None], out=np.zeros_like(x), where=xn[:, None] > EPSILON)
    ru = np.divide(r, rn[:, None], out=np.zeros_like(r), where=rn[:, None] > EPSILON)
    return np.clip(xu @ ru.T, -1., 1.)


def cosine_modulated_weights(weights, cosine, *, signed):
    w, cos = np.asarray(weights, float), np.asarray(cosine, float)
    if (w.ndim != 2 or cos.shape != w.shape or np.any(w < 0)
            or not np.isfinite(w).all() or not np.isfinite(cos).all()):
        raise ValueError("Aligned finite nonnegative relation weights are required")
    result = w * (cos if signed else np.maximum(cos, 0.))
    mass = np.abs(result).sum(axis=1)
    support = mass > EPSILON
    result = np.divide(result, mass[:, None], out=np.zeros_like(result),
                       where=support[:, None])
    return result, support


def transport_candidate(query_source, reference_source, reference_target, weights, support):
    x, xr, yr = (_matrix(v, name) for v, name in zip(
        (query_source, reference_source, reference_target),
        ("query_source", "reference_source", "reference_target")))
    w = np.asarray(weights, float)
    mask = _support(support, len(x))
    if xr.shape != yr.shape or xr.shape[1] != x.shape[1] or w.shape != (len(x), len(xr)):
        raise ValueError("Reference and query profile/weight shapes differ")
    if not np.isfinite(w).all() or np.any(w[~mask] != 0):
        raise ValueError("Unsupported reference weights must be exactly zero")
    if not np.allclose(np.abs(w[mask]).sum(1), 1., rtol=1e-10, atol=1e-12):
        raise ValueError("Supported response weights must have unit L1 mass")
    result = np.zeros_like(x)
    result[mask] = x[mask] + w[mask] @ (yr - xr)
    return result


def effective_support(baseline, support):
    base = _matrix(baseline, "baseline")
    return _support(support, len(base)) & (np.linalg.norm(base, axis=1) > EPSILON)


def apply_direction_correction(baseline, candidate, support, strength):
    """Project the proposal onto the Ridge tangent plane and retain its norm."""
    base, other = _matrix(baseline, "baseline"), _matrix(candidate, "candidate")
    alpha = float(strength)
    if base.shape != other.shape or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Aligned profiles and alpha in [0,1] are required")
    mask = effective_support(base, support)
    result = base.copy()
    if alpha == 0 or not mask.any():
        return result
    b, d = base[mask], other[mask] - base[mask]
    squared_norm = np.sum(b * b, axis=1)
    tangent = d - b * (np.sum(b * d, axis=1) / squared_norm)[:, None]
    moved = b + alpha * tangent
    moved_norm = np.linalg.norm(moved, axis=1)
    result[mask] = moved * (np.sqrt(squared_norm) / moved_norm)[:, None]
    return result


def apply_borrowing(baseline, candidate, support, strength, mode):
    mask = effective_support(baseline, support)
    if mode == "FREE":
        return apply_response_correction(baseline, candidate, mask, strength)
    if mode == "DIRECTION":
        return apply_direction_correction(baseline, candidate, mask, strength)
    raise ValueError("Unknown response-correction mode")


def fit_direction_strength(baseline, target, candidate, groups, support, *, min_groups=3):
    """Group-equal CAL MSE; smallest grid alpha within one SE of the best.

    Standard errors use paired per-group risk differences. If zero is within
    one SE of the grid minimum, return zero. This is a low-dimensional CAL
    selection rule, not a coverage guarantee.
    """
    base, y, other = (_matrix(v, name) for v, name in zip(
        (baseline, target, candidate), ("baseline", "target", "candidate")))
    if y.shape != base.shape or other.shape != base.shape:
        raise ValueError("CAL profiles differ")
    mask = effective_support(base, support)
    labels = _groups(groups, len(base))
    n_groups = len(np.unique(labels[mask]))
    report = dict(alpha=0., n_groups=n_groups, n_supported_rows=int(mask.sum()),
                  grid=list(DIRECTION_GRID), criterion="CAL supported-group-equal profile MSE",
                  one_se_rule="smallest alpha within paired one SE of the grid minimum")
    if n_groups < min_groups:
        return dict(report, reason="insufficient supported CAL groups")
    risks = []
    for alpha in DIRECTION_GRID:
        pred = apply_direction_correction(base, other, mask, alpha)
        _, group_risk = _group_mean(np.mean((pred[mask] - y[mask])**2, axis=1), labels[mask])
        risks.append(group_risk)
    risks = np.asarray(risks)
    mean = risks.mean(axis=1)
    best = int(np.argmin(mean))
    delta_to_best = risks - risks[best]
    se_to_best = delta_to_best.std(axis=1, ddof=1) / np.sqrt(n_groups)
    eligible = np.flatnonzero((mean - mean[best]) <= se_to_best + 1e-15)
    chosen = int(eligible[0])
    paired = risks[chosen] - risks[0]
    return dict(report, alpha=float(DIRECTION_GRID[chosen]), best_alpha=float(DIRECTION_GRID[best]),
                baseline_mse=float(mean[0]), selected_mse=float(mean[chosen]),
                grid_mse=mean.tolist(), grid_paired_se_to_best=se_to_best.tolist(),
                paired_delta=float(paired.mean()), se=float(paired.std(ddof=1)/np.sqrt(n_groups)),
                reason="zero within one SE of grid minimum" if chosen == 0 else "one-SE grid selection")


def matched_signed_random(weights, eligible, donor_bins, same_plate, donor_log_amplitude, *, seed):
    """Permute coefficient multisets within exact amplitude-bin/plate strata.

    The original donors belong to the legal pool, so exact strata are feasible;
    no bins or plate relations are relaxed. Signs and absolute weight mass are
    preserved. Sparse or fixed randomizations are explicitly measured.
    """
    w = np.asarray(weights, float)
    legal, plate = np.asarray(eligible, bool), np.asarray(same_plate, bool)
    bins, amp = np.asarray(donor_bins), np.asarray(donor_log_amplitude, float)
    if w.ndim != 2 or legal.shape != w.shape or plate.shape != w.shape:
        raise ValueError("Random-reference arrays differ")
    if bins.shape != (w.shape[1],) or amp.shape != bins.shape:
        raise ValueError("Donor descriptors differ")
    if not np.isfinite(w).all() or np.any((w != 0) & ~legal):
        raise ValueError("All original nonzero donors must be legal")
    rng, result = np.random.default_rng(seed), np.zeros_like(w)
    retained, singleton, fixed_pool, displacement = (np.zeros(len(w)) for _ in range(4))
    for i in range(len(w)):
        original = np.flatnonzero(w[i] != 0)
        for same in (False, True):
            for bin_id in np.unique(bins[original]):
                positions = original[(bins[original] == bin_id) & (plate[i, original] == same)]
                if not len(positions):
                    continue
                pool = np.flatnonzero(legal[i] & (bins == bin_id) & (plate[i] == same))
                if len(pool) < len(positions):
                    raise ValueError("Exact random stratum cannot hold the original donor count")
                chosen = rng.permutation(pool)[:len(positions)]
                coefficients = w[i, positions]
                result[i, chosen] = coefficients
                mass = np.abs(coefficients)
                displacement[i] += float(np.sum(mass * np.abs(amp[positions] - amp[chosen])))
                if len(pool) == 1:
                    singleton[i] += float(mass.sum())
                if len(pool) == len(positions):
                    fixed_pool[i] += float(mass.sum())
        retained[i] = float(np.abs(result[i, original]).sum())
    nonzero_match = np.count_nonzero(w, axis=1) == np.count_nonzero(result, axis=1)
    # These two checks jointly cover signed mass, absolute mass, and sign counts
    # within each predeclared bin and same-source-plate stratum.
    max_abs_error = max_signed_error = 0.
    for same in (False, True):
        for bin_id in np.unique(bins):
            stratum = (bins[None, :] == bin_id) & (plate == same)
            max_abs_error = max(max_abs_error, float(np.max(np.abs(
                (np.abs(w)*stratum).sum(1) - (np.abs(result)*stratum).sum(1)), initial=0)))
            max_signed_error = max(max_signed_error, float(np.max(np.abs(
                (w*stratum).sum(1) - (result*stratum).sum(1)), initial=0)))
    if not nonzero_match.all() or max(max_abs_error, max_signed_error) > 1e-12:
        raise AssertionError("Matched random reference weights changed their declared strata")
    return result, dict(retained_absolute_mass=retained, singleton_absolute_mass=singleton,
        fixed_donor_pool_absolute_mass=fixed_pool, weighted_log_amplitude_displacement=displacement,
        maximum_stratum_absolute_mass_error=max_abs_error,
        maximum_stratum_signed_mass_error=max_signed_error, amplitude_bin_merges=0,
        source_plate_relations_relaxed=0)
