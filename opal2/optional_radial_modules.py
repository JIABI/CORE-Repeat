"""Supported optional retrieval of measured radial errors on a frozen core.

Module fitting uses calibration groups, never query outcomes. Uniform spherical
directions and the frozen core mean/scatter are unchanged. Radial reweighting can
change both covariance and tails; these are not physical variance components.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import GroupKFold

from .empirical_radial import fit_radial, reference_weights, radial_nll


BIO_COEFFICIENTS = ((0., 0.), (.25, 0.), (.5, 0.), (0., .25), (0., .5), (.25, .25))
STATE_COEFFICIENTS = ((0.,), (.25,), (.5,))


def supported_retrieval(similarity, *, allowed=None, shrinkage=20.):
    """Return actual similarity strength as well as ESS; no ESS-only activation."""
    s = np.asarray(similarity, float)
    if s.ndim != 2 or not np.isfinite(s).all() or np.any((s < 0) | (s > 1)):
        raise ValueError('Similarities must be a finite [query,reference] matrix in [0,1]')
    a = np.ones_like(s, dtype=bool) if allowed is None else np.asarray(allowed)
    if a.dtype != bool or a.shape != s.shape:
        raise ValueError('Allowed reference pairs must be an aligned Boolean matrix')
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError('Positive shrinkage is required')
    s = np.where(a, s, 0.)
    mass = s.sum(1)
    maximum = s.max(1) if s.shape[1] else np.zeros(len(s))
    scaled = np.divide(s, maximum[:, None], out=np.zeros_like(s), where=maximum[:, None] > 0)
    weights = np.divide(scaled, scaled.sum(1)[:, None], out=np.zeros_like(s),
                        where=scaled.sum(1)[:, None] > 0)
    sq = np.square(weights).sum(1)
    ess = np.divide(1., sq, out=np.zeros(len(s)), where=sq > 0)
    # Absolute strength is on a declared [0,1] scale, so a dense array of 1e-29
    # matches cannot receive a large gate merely because ESS is high.
    strength = (weights*s).sum(1)
    gate = ess/(ess+shrinkage)*strength
    return dict(weights=weights, gate=gate, supported=mass > 0, ess=ess,
                raw_mass=mass, maximum=maximum, strength=strength,
                positive_count=(s > 0).sum(1))


def blend_retrieval(base_weights, channels, coefficients, *, enabled=True):
    """Convex radial-weight update; disabled/unsupported output is bitwise core."""
    base = np.asarray(base_weights, float)
    if base.ndim != 2 or not np.isfinite(base).all() or np.any(base < 0) or not np.allclose(base.sum(1), 1., atol=1e-12, rtol=0):
        raise ValueError('Core weights must be normalized nonnegative rows')
    out = base.copy()
    if not enabled:
        return out, np.zeros(len(base))
    coeff = np.asarray(coefficients, float)
    if len(channels) != len(coeff) or np.any(coeff < 0) or coeff.sum() > 1 or not np.isfinite(coeff).all():
        raise ValueError('Channel coefficients must form a bounded nonnegative mixture')
    total = np.zeros(len(base))
    addition = np.zeros_like(base)
    for c, alpha in zip(channels, coeff):
        w, g = np.asarray(c['weights'], float), np.asarray(c['gate'], float)
        if w.shape != base.shape or g.shape != (len(base),) or not np.isfinite(w).all() or not np.isfinite(g).all() or np.any(w < 0) or np.any((g < 0) | (g > 1)):
            raise ValueError('Invalid channel weight/gate arrays')
        if not np.allclose(w[g > 0].sum(1), 1., atol=1e-12, rtol=0):
            raise ValueError('Supported channel rows must be normalized')
        amount = float(alpha)*g
        total += amount
        addition += amount[:, None]*w
    active = total > 0
    out[active] = (1-total[active, None])*base[active]+addition[active]
    if not np.allclose(out.sum(1), 1., atol=1e-12, rtol=0):
        raise ValueError('Optional mixture lost normalization')
    np.testing.assert_array_equal(out[~active], base[~active])
    return out, total


def state_similarity(reference_z, query_z, *, scale):
    """Same fixed RBF family for linear and nonlinear states, excluding amplitude.

    The amplitude channel is already present in core reference weights. The
    supplied scale must be fitted on MODEL_FIT states, not query/reference labels.
    """
    r, q, sd = map(lambda x: np.asarray(x, float), (reference_z, query_z, scale))
    if r.ndim != 2 or q.ndim != 2 or q.shape[1] != r.shape[1] or sd.shape != (r.shape[1],) or np.any(sd <= 0) or not all(np.isfinite(v).all() for v in (r,q,sd)):
        raise ValueError('States and MODEL_FIT scale must be finite and aligned')
    d2 = np.square((q[:, None]-r[None])/sd).mean(-1)
    return np.exp(-.5*d2)


@dataclass
class CalibratedRadialSwitch:
    coefficients: tuple[float, ...]
    selection: dict

    @property
    def enabled(self):
        return any(c > 0 for c in self.coefficients)

    def apply(self, base_weights, similarities, *, allowed=None, enabled=True):
        if not enabled or not self.enabled:
            return np.asarray(base_weights, float).copy(), np.zeros(len(base_weights)), []
        channels = [supported_retrieval(s, allowed=allowed) for s in similarities]
        out, gate = blend_retrieval(base_weights, channels, self.coefficients)
        return out, gate, channels


def fit_radial_switch(radii, logamp, groups, similarities, *, fit_amp_sd,
                      coefficient_grid=STATE_COEFFICIENTS, splits=3):
    """Group-heldout calibration selection; rebuild every radial law internally.

    Mean paired radial log-density improvement must exceed one group-level
    standard error. This selection heuristic is not a significance test. The
    untouched query set measures its consequences.
    """
    r, amp, g = np.asarray(radii, float), np.asarray(logamp, float), np.asarray(groups, str)
    if r.ndim != 1 or amp.shape != r.shape or g.shape != r.shape or len(r) < 2:
        raise ValueError('Aligned calibration radii, amplitude and groups required')
    if not np.isfinite(r).all() or np.any(r <= 0) or not np.isfinite(amp).all():
        raise ValueError('Invalid calibration observations')
    mats = [np.asarray(s, float) for s in similarities]
    if not mats or any(s.shape != (len(r),len(r)) or not np.isfinite(s).all() for s in mats):
        raise ValueError('Aligned calibration-to-calibration similarity matrices required')
    grid = tuple(tuple(float(v) for v in c) for c in coefficient_grid)
    if any(len(c) != len(mats) for c in grid) or grid[0] != (0.,)*len(mats):
        raise ValueError('First candidate must disable all channels')
    unique = np.unique(g)
    if len(unique) < 3:
        return CalibratedRadialSwitch(grid[0], dict(reason='insufficient_calibration_groups',
            calibrated_admission=False, group_count=len(unique)))
    loss = np.full((len(grid),len(r)), np.nan)
    for train, valid in GroupKFold(min(splits,len(unique))).split(r, groups=g):
        law = fit_radial(r[train])
        base = reference_weights(amp[train], amp[valid], fit_amp_sd, conditional=True)['weights']
        allowed = g[valid,None] != g[None,train]
        channels = [supported_retrieval(s[np.ix_(valid,train)], allowed=allowed) for s in mats]
        # Unit scatter and a radial representative yield exactly the whitened
        # density. Omitted fixed scatter Jacobians cancel in paired differences.
        residual = np.zeros((len(valid),9)); residual[:,0] = r[valid]
        scatter = np.broadcast_to(np.eye(9), (len(valid),9,9))
        for j, candidate in enumerate(grid):
            w, _ = blend_retrieval(base,channels,candidate)
            loss[j,valid] = radial_nll(residual,scatter,law,w)
    records = []
    accepted = [0]
    for j, candidate in enumerate(grid):
        delta = loss[j]-loss[0]
        grouped = np.array([delta[g == k].mean() for k in unique])
        mean = float(grouped.mean())
        se = float(grouped.std(ddof=1)/np.sqrt(len(grouped)))
        admissible = j == 0 or mean < -se-1e-12
        if admissible and j: accepted.append(j)
        records.append(dict(coefficients=candidate, mean_nll=float(loss[j].mean()),
            group_mean_paired_difference=mean, group_se=se, eligible=admissible))
    # Admission and strength selection are distinct: once a nonzero candidate
    # clears the declared paired margin, prefer the smallest total intervention.
    # Zero is the fallback, not a competing magnitude after admission succeeds.
    nonzero = [j for j in accepted if j != 0]
    best = min(nonzero, key=lambda j: (sum(grid[j]),
        records[j]['group_mean_paired_difference'],grid[j])) if nonzero else 0
    return CalibratedRadialSwitch(grid[best], dict(reason='calibration_group_cv',
        calibrated_admission=bool(best), group_count=len(unique), observations=len(r),
        candidates=records, selected_index=best,
        criterion='paired group-mean delta < -one group SE; smallest eligible nonzero total mixing, then paired loss, then coefficient tuple; zero otherwise',
        guarantee='developmental selection, not a statistical certificate'))
