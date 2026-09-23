"""Group-weighted joint-dose response features and signed program borrowing.

Inputs are the declared source well and metadata. Response bases fit on TRAIN;
all shrinkage uses CAL; prediction functions take no query outcome. This is a
response model, separate from the frozen CORE geometry/distribution model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .crossdose_response import fit_convex_strength


DOSES = np.array([.0025, .01, .025, .1, .25, 1., 2.5, 10.])


def matrix(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError(name + ' must be a finite matrix')
    return a


def group_weights(groups):
    g = np.asarray(groups, str)
    if g.ndim != 1 or not len(g) or any(x in ('', 'nan', 'None') for x in g):
        raise ValueError('Known training groups required')
    _, inv, count = np.unique(g, return_inverse=True, return_counts=True)
    return 1. / (len(count) * count[inv])


def assert_role_isolation(groups, roles):
    groups, roles = np.asarray(groups, str), np.asarray(roles, str)
    if groups.shape != roles.shape or groups.ndim != 1:
        raise ValueError('Group/role arrays must align')
    for group in np.unique(groups):
        if len(np.unique(roles[groups == group])) != 1:
            raise ValueError('Chemical group crosses roles')


@dataclass
class WeightedBasis:
    center: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    total_variance: float
    standardized: bool

    def transform(self, x):
        a = matrix(x, 'basis inputs')
        if a.shape[1] != len(self.center):
            raise ValueError('Basis coordinate mismatch')
        return ((a - self.center) / self.scale) @ self.components

    def signed_projection(self, residual):
        """Project a signed correction without adding a centering offset."""
        a = matrix(residual, 'residual')
        if self.standardized or a.shape[1] != len(self.center):
            raise ValueError('Signed response projection requires an unstandardized basis')
        return a @ self.components

    def arrays(self):
        return dict(center=self.center, scale=self.scale, components=self.components,
                    eigenvalues=self.eigenvalues, total_variance=np.array(self.total_variance))


def fit_weighted_basis(train_values, groups, *, rank, standardize=False):
    x = matrix(train_values, 'TRAIN basis values')
    if len(x) != len(groups) or min(x.shape) < rank or rank < 1:
        raise ValueError('TRAIN basis dimensions cannot support the declared rank')
    w = group_weights(groups)
    center = w @ x
    variance = w @ np.square(x - center)
    scale = np.where(variance > 1e-16, np.sqrt(variance), 1.) if standardize else np.ones(x.shape[1])
    z = (x - center) / scale
    covariance = (z * w[:, None]).T @ z
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) * .5)
    order = np.argsort(eigenvalues)[::-1][:rank]
    components = vectors[:, order]
    # Fixed sign convention prevents arbitrary sign changes across reruns.
    for j in range(rank):
        pivot = np.argmax(np.abs(components[:, j]))
        if components[pivot, j] < 0:
            components[:, j] *= -1
    return WeightedBasis(center, scale, components, np.maximum(eigenvalues[order], 0.),
                         float(np.maximum(eigenvalues, 0.).sum()), bool(standardize))


def jointdose_features(source_x, source_dose, source_basis, *, interacting=True):
    """PCA state + explicit amplitude, optionally tensored with fixed cubic dose.

    Dose maps to [-1,1] using the declared eight-dose range, not fitted extrema.
    This basis is tested only on the seven prespecified adjacent-dose tasks.
    """
    x = matrix(source_x, 'source_x')
    dose = np.asarray(source_dose, float)
    if dose.shape != (len(x),) or not np.isfinite(dose).all() or np.any(dose <= 0):
        raise ValueError('Finite positive source doses required')
    if np.any(dose < DOSES[0] - 1e-12) or np.any(dose > DOSES[-1] + 1e-12):
        raise ValueError('Dose outside declared basis range')
    amplitude = np.log(np.maximum(np.linalg.norm(x, axis=1), 1e-12))
    state = np.column_stack((source_basis.transform(x), amplitude, np.ones(len(x))))
    if not interacting:
        return state
    u = 2 * (np.log(dose) - np.log(DOSES[0])) / np.log(DOSES[-1] / DOSES[0]) - 1
    basis = np.column_stack((np.ones(len(u)), u, u*u, u*u*u))
    return (state[:, :, None] * basis[:, None, :]).reshape(len(x), -1)


def reference_residual_candidate(baseline, reference_residual, weights, support):
    """Add weighted signed REF residuals; unsupported rows exactly unchanged."""
    base, residual = matrix(baseline, 'baseline'), matrix(reference_residual, 'REF residual')
    w, mask = np.asarray(weights, float), np.asarray(support)
    if (residual.shape[1] != base.shape[1] or w.shape != (len(base), len(residual))
            or mask.dtype != bool or mask.shape != (len(base),) or not np.isfinite(w).all()
            or np.any(w < 0) or np.any(w[~mask] != 0)
            or not np.allclose(w[mask].sum(1), 1., atol=1e-12, rtol=1e-10)):
        raise ValueError('Legal normalized reference weights and support must align')
    candidate = base.copy()
    candidate[mask] += w[mask] @ residual
    return candidate


def fit_component_strengths(cal_target_coefficients, cal_reference_coefficients, groups, support):
    """Independent bounded CAL strengths in a pre-existing orthonormal basis."""
    y, ref = matrix(cal_target_coefficients, 'CAL target coefficients'), matrix(cal_reference_coefficients, 'CAL references')
    if y.shape != ref.shape:
        raise ValueError('CAL component arrays must align')
    reports = [fit_convex_strength(np.zeros((len(y), 1)), y[:, j:j+1], ref[:, j:j+1],
                                  groups, support) for j in range(y.shape[1])]
    return dict(alpha=np.array([r['alpha'] for r in reports]), component_reports=reports,
                active_components=sum(r['alpha'] > 0 for r in reports),
                criterion='CAL supported-chemical-group-equal component MSE with paired one-SE zero fallback')


def apply_program_correction(baseline, reference_coefficients, response_basis, support, alpha):
    """Keep the entire orthogonal complement and unsupported baseline exact."""
    base, coeff = matrix(baseline, 'baseline'), matrix(reference_coefficients, 'reference coefficients')
    mask = np.asarray(support)
    strengths = np.asarray(alpha, float)
    if strengths.ndim == 0:
        strengths = np.full(coeff.shape[1], float(strengths))
    if (mask.shape != (len(base),) or mask.dtype != bool or len(coeff) != len(base)
            or response_basis.components.shape != (base.shape[1], coeff.shape[1])
            or strengths.shape != (coeff.shape[1],) or not np.isfinite(strengths).all()
            or np.any(strengths < 0) or np.any(strengths > 1)):
        raise ValueError('Program coefficients, basis, support and bounded strengths must align')
    result = base.copy()
    if mask.any() and np.any(strengths > 0):
        result[mask] += (coeff[mask] * strengths) @ response_basis.components.T
    return result


def matched_random_reference_weights(weights, legal, amplitude_bins, query_plate, donor_plate, *, seed):
    """Preserve weights within amplitude quintile × same-source-plate strata."""
    w, allowed = np.asarray(weights, float), np.asarray(legal)
    bins, qp, dp = np.asarray(amplitude_bins), np.asarray(query_plate), np.asarray(donor_plate)
    if (w.ndim != 2 or allowed.dtype != bool or allowed.shape != w.shape or bins.shape != (w.shape[1],)
            or qp.shape != (len(w),) or dp.shape != bins.shape or np.any((w > 0) & ~allowed)):
        raise ValueError('Matched random reference dimensions/legal pool differ')
    random = np.zeros_like(w)
    rng = np.random.default_rng(seed)
    retained = np.zeros(len(w))
    fixed = np.zeros(len(w))
    for i in np.flatnonzero(w.sum(1) > 0):
        strata = bins * 2 + (dp == qp[i])
        for stratum in np.unique(strata[allowed[i]]):
            pool = np.flatnonzero(allowed[i] & (strata == stratum))
            destination = rng.permutation(pool)
            take = w[i, pool] > 0
            src, dst = pool[take], destination[take]
            random[i, dst] = w[i, src]
            retained[i] += np.sum(w[i, src] * (src == dst))
            if len(pool) == 1:
                fixed[i] += w[i, src].sum()
    np.testing.assert_array_equal(np.sort(w, axis=1), np.sort(random, axis=1))
    np.testing.assert_allclose((w * (qp[:, None] == dp[None])).sum(1),
                               (random * (qp[:, None] == dp[None])).sum(1), atol=1e-14, rtol=0)
    return random, dict(retained_weight_mass=retained, fixed_weight_mass=fixed)


def compact_reference_weights(weights, max_count=16):
    """Lossless fixed-width sparse export for the declared top-k retrieval."""
    w = np.asarray(weights)
    idx = np.full((len(w), max_count), -1, dtype=np.int32)
    values = np.zeros((len(w), max_count), dtype=float)
    for i in range(len(w)):
        active = np.flatnonzero(w[i] > 0)
        if len(active) > max_count:
            raise ValueError('Reference row exceeds declared sparse export width')
        idx[i, :len(active)] = active
        values[i, :len(active)] = w[i, active]
    return idx, values
