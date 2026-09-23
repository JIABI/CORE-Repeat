"""Legal reference-error summaries in each query's native geometry frame."""
from __future__ import annotations

import numpy as np

from .joint_contrast_scale import contrast_projector, projected_energy, rescale_components
from .module_switch_experiment import biology_similarity, context_mask
from .empirical_radial import fit_radial, reference_weights, radial_nll, variance_multiplier


FIELDS = ('available', 'log_count', 'log_mass', 'log_ess', 'mean_similarity',
          'pair_log_energy', 'remainder_log_energy', 'pair_log_se',
          'remainder_log_se', 'log_amp_gap', 'angle_gap', 'confidence')
NAMES = [relation+'_'+field for relation in ('target', 'moa') for field in FIELDS]
STRENGTHS = (0., .25, .5, 1.)


def _calibration_representatives(groups, ids):
    """Optional ID-min group representatives, matching the Rx CORE radial law."""
    if ids is None:
        return np.ones(len(groups), bool)
    ids = np.asarray(ids, str)
    if ids.shape != groups.shape or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique calibration IDs must align with groups')
    mask = np.zeros(len(ids), bool)
    for group in np.unique(groups):
        mask[min(np.flatnonzero(groups == group), key=lambda i: ids[i])] = True
    return mask


def calibration_frame_covariance(scatter, residual, logamp, groups, bandwidth, *, representative_ids=None):
    """Current-cell, group-LOO CORE covariance for calibration-only inputs.

    Never use this object's covariance cached when it was a query in the
    opposite cell: that distribution may have used today's query outcomes.
    """
    scatter,residual=np.asarray(scatter,float),np.asarray(residual,float)
    groups,logamp=np.asarray(groups),np.asarray(logamp,float)
    radii=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(scatter),residual[...,None])[...,0],axis=1)
    result=scatter.copy()
    representatives = _calibration_representatives(groups, representative_ids)
    for i in range(len(groups)):
        take=(groups!=groups[i]) & representatives
        if len(np.unique(groups[take]))<3:
            raise ValueError('Insufficient independent calibration groups')
        law=fit_radial(radii[take])
        weights=reference_weights(logamp[take],logamp[i:i+1],bandwidth,conditional=True)['weights']
        result[i]*=variance_multiplier(law,weights)[0]
    return result


def biology_features(data, metadata, query, donors, raw_mean, raw_covariance,
                     donor_raw_residual, *, allowed=None):
    """No query outcome is read. Donor residuals must already be honest.

    A donor's own calibrated covariance is deliberately not accepted. Its
    construction could have involved the present query. All reference errors
    are instead projected in the same query-defined frame.
    """
    query, donors = np.asarray(query, int), np.asarray(donors, int)
    raw_mean, raw_covariance = np.asarray(raw_mean, float), np.asarray(raw_covariance, float)
    residual = np.asarray(donor_raw_residual, float)
    n, m = len(query), len(donors)
    if raw_mean.shape != (n, 9) or raw_covariance.shape != (n, 9, 9) or residual.shape != (m, 9):
        raise ValueError('Native geometry/reference dimensions differ')
    permit = np.ones((n,m), bool) if allowed is None else np.asarray(allowed, bool)
    if permit.shape != (n,m):
        raise ValueError('Reference permission shape differs')
    permit = permit & context_mask(metadata,query,donors)
    permit &= data['groups'][query,None] != data['groups'][None,donors]
    relations = [s*permit for s in biology_similarity(data, metadata, query, donors)]
    dec = contrast_projector(raw_mean, np.ones(9), raw_covariance)
    # Read only the initial observed hole, never any future measurement.
    x = np.asarray(data['Y'][:, 0], float)
    norm = np.linalg.norm(x, axis=1)
    if np.any(norm[np.r_[query,donors]] <= 0):
        raise ValueError('Zero-norm first holes require original completion handling')
    direction = x / norm[:,None]
    cosine = np.clip(direction[query]@direction[donors].T, -1., 1.)
    amp_gap = np.abs(np.log(norm[query,None])-np.log(norm[None,donors]))
    out = np.zeros((n,len(NAMES)))
    support_by_relation = np.zeros((n,2), bool)
    for i in range(n):
        w = np.linalg.solve(dec['factor'][i], residual.T).T
        pair = w @ dec['projector'][i]
        energies = np.column_stack((np.square(pair).sum(1)/3., np.square(w-pair).sum(1)/6.))
        pool = energies[permit[i]]
        pool_variance = np.var(pool, axis=0, ddof=1) if len(pool)>1 else np.array([2/3,2/6])
        # Single-reference support must not produce a fictitious zero SE.
        pool_variance = np.maximum(pool_variance, [2/3,2/6])
        for r, similarities in enumerate(relations):
            s = similarities[i]
            count, mass = int((s>0).sum()), float(s.sum())
            if count == 0:
                continue
            weights = s/mass
            ess = 1/float(np.square(weights).sum())
            mean = weights@energies
            denom = 1-float(np.square(weights).sum())
            variance = (weights@np.square(energies-mean))/denom if denom>1e-12 else pool_variance
            se = np.sqrt(np.maximum(variance, 0)/ess)
            confidence = ess/(ess+5.)*(-np.expm1(-mass))
            out[i,r*len(FIELDS):(r+1)*len(FIELDS)] = [
                1., np.log1p(count), np.log1p(mass), np.log1p(ess), mass/count,
                *np.log1p(mean), *np.log1p(se), float(weights@amp_gap[i]),
                float(weights@(1-cosine[i])), confidence]
            support_by_relation[i,r] = True
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite reference summaries')
    return dict(values=out, names=NAMES.copy(), support=support_by_relation.any(1),
                support_by_relation=support_by_relation)


def apply_increment(raw_mean, coordinate_scale, scatter, increment):
    """Preserve unchanged rows bitwise, including when others are modified."""
    scatter, increment = np.asarray(scatter, float), np.asarray(increment, float)
    if increment.shape != (len(scatter),2) or not np.isfinite(increment).all():
        raise ValueError('Two finite aligned increments required')
    changed = np.any(increment != 0, axis=1)
    result = scatter.copy()
    if changed.any():
        dec = contrast_projector(np.asarray(raw_mean)[changed], coordinate_scale, scatter[changed])
        factors = np.exp(increment[changed])
        result[changed] = rescale_components(scatter[changed], dec, factors[:,0], factors[:,1])
    return result


def select_strength(raw_mean, coordinate_scale, scatter, residual, logamp, groups,
                    bandwidth, increment, *, representative_ids=None):
    """Group-LOO fixed CORE radial law; candidate changes scatter only.

    This is development selection, not a conformal/independence guarantee.
    No query quantities or outcomes enter this function.
    """
    groups, logamp = np.asarray(groups), np.asarray(logamp, float)
    residual = np.asarray(residual, float)
    radii = np.linalg.norm(np.linalg.solve(np.linalg.cholesky(scatter), residual[...,None])[...,0], axis=1)
    scores = np.zeros((len(groups),len(STRENGTHS)))
    transformed = [apply_increment(raw_mean,coordinate_scale,scatter,a*increment) for a in STRENGTHS]
    representatives = _calibration_representatives(groups, representative_ids)
    for i in range(len(groups)):
        take = (groups != groups[i]) & representatives
        if len(np.unique(groups[take]))<3:
            return dict(strength=0., reason='insufficient independent calibration groups', scores=[])
        law = fit_radial(radii[take])
        weights = reference_weights(logamp[take],logamp[i:i+1],bandwidth,conditional=True)['weights']
        for j in range(len(STRENGTHS)):
            scores[i,j] = radial_nll(residual[i:i+1],transformed[j][i:i+1],law,weights)[0]
    unique = np.unique(groups)
    group_scores = np.stack([scores[groups==g].mean(0) for g in unique])
    mean = group_scores.mean(0)
    best = int(np.argmin(mean))
    delta = group_scores[:,best]-group_scores[:,0]
    se = float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.inf
    selected = 0
    if best and float(delta.mean()) < -se:
        for j in range(1,best+1):
            d0 = group_scores[:,j]-group_scores[:,0]
            sd0 = float(d0.std(ddof=1)/np.sqrt(len(d0)))
            db = group_scores[:,j]-group_scores[:,best]
            sdb = float(db.std(ddof=1)/np.sqrt(len(db)))
            if d0.mean() < -sd0 and mean[j] <= mean[best]+sdb:
                selected=j
                break
        if selected == 0:
            selected=best
    return dict(strength=STRENGTHS[selected], best_strength=STRENGTHS[best],
        n_groups=len(unique), strengths=list(STRENGTHS), mean_nll=mean.tolist(),
        paired_best_delta=float(delta.mean()),paired_best_se=se, scores=scores.tolist(),
        reason='paired one-SE admission, smaller within-one-SE strength; zero included')
