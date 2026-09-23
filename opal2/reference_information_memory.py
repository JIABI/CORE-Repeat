"""Decision-information weights and empirical complete-residual memories.

Similarity and weighting functions never accept outcomes. Hyperparameter
selection accepts donor residuals only and excludes each donor's complete
chemistry group from its prediction. Moments describe empirical residual
variation, including model error; they are not isolated physical-noise laws.
"""
from __future__ import annotations

import numpy as np


LAMBDA_GRID = (0., .25, .5, .75, 1.)
ALPHA_GRID = (0., .25, .5, 1.)
BIOLOGY_SHRINKAGE = 8.
TIE_RTOL = 1e-12
TIE_ATOL = 1e-15


def _matrix(values, name, columns=None):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1 or (columns is not None and result.shape[1] != columns):
        raise ValueError(name+' must be a nonempty aligned matrix')
    return result


def _mask(values, length, name):
    if values is None:
        return np.ones(length, dtype=bool)
    result = np.asarray(values)
    if result.shape != (length,) or result.dtype != bool:
        raise ValueError(name+' must be an aligned boolean mask')
    return result


def _profiles(query, donor, query_mask, donor_mask, *, nonnegative):
    query = _matrix(query, 'query profiles')
    donor = _matrix(donor, 'donor profiles', query.shape[1])
    qm, dm = _mask(query_mask, len(query), 'query_mask'), _mask(donor_mask, len(donor), 'donor_mask')
    if not np.isfinite(query[qm]).all() or not np.isfinite(donor[dm]).all():
        raise ValueError('Present profiles must be finite')
    if nonnegative and (np.any(query[qm] < 0) or np.any(donor[dm] < 0)):
        raise ValueError('Present relationship profiles must be nonnegative')
    return np.where(qm[:, None], query, 0.), np.where(dm[:, None], donor, 0.), qm, dm


def _cosine(query, donor, query_mask=None, donor_mask=None, *, nonnegative=True):
    q, d, qm, dm = _profiles(query, donor, query_mask, donor_mask, nonnegative=nonnegative)
    qn, dn = np.linalg.norm(q, axis=1), np.linalg.norm(d, axis=1)
    if not np.isfinite(qn).all() or not np.isfinite(dn).all():
        raise ValueError('Profile norm overflowed')
    q = q/np.where(qn > 0, qn, 1.)[:, None]
    d = d/np.where(dn > 0, dn, 1.)[:, None]
    similarity = np.clip(q@d.T, 0. if nonnegative else -1., 1.)
    present = (qm & (qn > 0))[:, None] & (dm & (dn > 0))[None]
    return np.where(present, similarity, 0.)


def cosine_relationship(query, donor, query_mask=None, donor_mask=None):
    """Target/MoA cosine; absent or zero-norm profiles contribute exact zero.

    Separate masks permit arbitrary placeholders on absent rows. Known rows
    must be finite and nonnegative. This returns only a similarity matrix.
    """
    return _cosine(query, donor, query_mask, donor_mask, nonnegative=True)


def morphology_similarity(query_morphology, donor_morphology):
    """Positive cosine of first-well morphology vectors, excluding log-norm.

    Pass the morphology block itself, e.g. standardized full X[:, :-1]. No
    query centering or fitting occurs here; zero directions have zero cosine.
    """
    return np.maximum(_cosine(query_morphology, donor_morphology, nonnegative=False), 0.)


def chemistry_similarity(query_bits, donor_bits, *, metric='tanimoto', query_mask=None, donor_mask=None):
    """Morgan binary-fingerprint Tanimoto or cosine, excluding validity flags."""
    if metric not in ('tanimoto', 'cosine'):
        raise ValueError('Chemical metric must be tanimoto or cosine')
    q, d, qm, dm = _profiles(query_bits, donor_bits, query_mask, donor_mask, nonnegative=True)
    if np.any((q != 0) & (q != 1)) or np.any((d != 0) & (d != 1)):
        raise ValueError('Morgan fingerprints must contain binary bits on present rows')
    if metric == 'cosine':
        return _cosine(q, d, qm, dm)
    intersection = q@d.T
    union = q.sum(1)[:, None]+d.sum(1)[None]-intersection
    values = intersection/np.where(union > 0, union, 1.)
    return np.where(qm[:, None] & dm[None], values, 0.)


def _names(values, length, name, *, unique=False):
    names = np.asarray(values, dtype=str)
    if names.shape != (length,) or np.any(names == '') or (unique and len(set(names.tolist())) != length):
        raise ValueError(name+' must contain aligned '+('unique ' if unique else '')+'nonempty identities')
    return names


def _eligibility(shape, eligible, query_groups, donor_groups):
    if eligible is None:
        allowed = np.ones(shape, dtype=bool)
    else:
        allowed = np.asarray(eligible)
        if allowed.shape != shape or allowed.dtype != bool:
            raise ValueError('eligible must be a query-by-donor boolean matrix')
        allowed = allowed.copy()
    if (query_groups is None) != (donor_groups is None):
        raise ValueError('Provide both query_groups and donor_groups for group exclusion')
    if query_groups is not None:
        q = _names(query_groups, shape[0], 'query_groups')
        d = _names(donor_groups, shape[1], 'donor_groups')
        allowed &= q[:, None] != d[None]
    if np.any(~allowed.any(1)):
        raise ValueError('Every query needs at least one eligible donor after exclusion')
    return allowed


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def _normalized_weights(values):
    weights = _matrix(values, 'weights')
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(1), 1., rtol=0., atol=1e-12):
        raise ValueError('Weights must be finite, nonnegative, and normalized by query')
    return weights


def normalized_topk_weights(similarity, *, donor_ids, top_k=16, eligible=None,
                            query_groups=None, donor_groups=None):
    """Normalize the top-k positive eligible values; ties use stable donor IDs.

    If there is no positive eligible similarity, use every eligible donor
    uniformly. When groups are supplied, exclusion is always enforced before
    ranking and fallback. Excluding every donor is an error, never a fallback
    that silently restores excluded donors.
    """
    similarity = _matrix(similarity, 'similarity')
    if not np.isfinite(similarity).all():
        raise ValueError('Similarities must be finite')
    top_k = _positive_integer(top_k, 'top_k')
    donor_ids = _names(donor_ids, similarity.shape[1], 'donor_ids', unique=True)
    allowed = _eligibility(similarity.shape, eligible, query_groups, donor_groups)
    weights = np.zeros_like(similarity)
    positive = allowed & (similarity > 0)
    support = positive.any(1)
    for row in range(len(similarity)):
        candidates = np.flatnonzero(positive[row])
        if len(candidates):
            order = np.lexsort((donor_ids[candidates], -similarity[row, candidates]))
            selected = candidates[order[:top_k]]
            values = similarity[row, selected]
            # Scaling by the maximum avoids overflow in the row sum while
            # retaining exactly the same normalized positive weights.
            values = values/values.max()
            weights[row, selected] = values/values.sum()
        else:
            weights[row, allowed[row]] = 1./allowed[row].sum()
    return dict(weights=weights, eligible=allowed, support=support, fallback=~support,
        neff=1./np.square(weights).sum(1), positive_eligible_count=positive.sum(1),
        selected_count=(weights > 0).sum(1), eligible_count=allowed.sum(1), top_k=top_k)


def _biology_matrix(values, shape, name):
    if values is None:
        return np.zeros(shape, dtype=float)
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape or not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError(name+' must be a finite aligned cosine matrix in [0,1]')
    return matrix


def augment_generic_weights(generic_weights, target_similarity=None, moa_similarity=None,
                            *, mixing_lambda, eligible, shrinkage=BIOLOGY_SHRINKAGE):
    """Mix generic weights with all eligible biological-overlap donors.

    B_raw=(target_cos+moa_cos)/2 includes known-zero and missing-channel zeros.
    B is its row normalization, neff=1/sum(B²), rho=neff/(neff+shrinkage), and
    final weights=(1-lambda*rho)*G+lambda*rho*B. Unknown/no-overlap rows and
    lambda=0 return the original generic weights exactly.
    """
    generic = _normalized_weights(generic_weights)
    if mixing_lambda not in LAMBDA_GRID:
        raise ValueError('mixing_lambda must belong to the prespecified lambda grid')
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError('Biology support shrinkage must be positive and finite')
    allowed = _eligibility(generic.shape, eligible, None, None)
    if np.any(generic[~allowed] != 0):
        raise ValueError('Generic weights include excluded donors')
    target = _biology_matrix(target_similarity, generic.shape, 'target_similarity')
    moa = _biology_matrix(moa_similarity, generic.shape, 'moa_similarity')
    raw = np.where(allowed, (target+moa)/2., 0.)
    mass = raw.sum(1)
    support = mass > 0
    biological = raw/np.where(support, mass, 1.)[:, None]
    squared = np.square(biological).sum(1)
    neff = np.where(support, 1./np.where(squared > 0, squared, 1.), 0.)
    factor = neff/(neff+shrinkage)
    effective = float(mixing_lambda)*factor
    weights = generic.copy()
    if mixing_lambda:
        weights[support] = ((1-effective[support, None])*generic[support]
                            +effective[support, None]*biological[support])
    return dict(weights=weights, biology_weights=biological, support=support, neff=neff,
        support_shrinkage_factor=factor, effective_mixing=effective,
        mixing_lambda=float(mixing_lambda), shrinkage=float(shrinkage), eligible=allowed)


def weighted_residual_moments(weights, donor_residuals):
    """Population moments of complete joint nine-coordinate residual blocks.

    The covariance is a nonnegative weighted sum of centered outer products.
    Coordinates are never resampled separately, and no diagonal approximation,
    covariance floor, shrinkage, or physical-noise interpretation is applied.
    """
    weights = _normalized_weights(weights)
    residuals = _matrix(donor_residuals, 'donor_residuals', 9)
    if len(residuals) != weights.shape[1] or not np.isfinite(residuals).all():
        raise ValueError('Complete finite nine-coordinate residuals must align with donor columns')
    mean = weights@residuals
    centered = residuals[None]-mean[:, None]
    covariance = np.einsum('qd,qdi,qdj->qij', weights, centered, centered, optimize=True)
    covariance = (covariance+covariance.transpose(0, 2, 1))*.5
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError('Empirical residual moments overflowed')
    return dict(mean=mean, covariance=covariance, neff=1./np.square(weights).sum(1),
        interpretation='empirical joint residual moments including model error; not pure physical noise',
        covariance_definition='normalized weighted central outer products; population second moment')


def select_donor_hyperparameters(donor_residuals, generic_similarity, donor_groups, *, donor_ids,
                                 target_similarity=None, moa_similarity=None, top_k=16,
                                 shrinkage=BIOLOGY_SHRINKAGE):
    """Choose lambda/alpha using donor leave-chemistry-group-out MSE only.

    Inputs are donor-by-donor decision-information similarities and complete
    donor residual blocks. Every prediction excludes the row donor and all
    donors in its chemistry group. Query labels cannot enter this API.
    """
    residuals = _matrix(donor_residuals, 'donor_residuals', 9)
    if not np.isfinite(residuals).all():
        raise ValueError('Donor residuals must be finite')
    similarity = _matrix(generic_similarity, 'generic_similarity', len(residuals))
    if len(similarity) != len(residuals):
        raise ValueError('Selection requires a square donor-by-donor similarity matrix')
    groups = _names(donor_groups, len(residuals), 'donor_groups')
    names = _names(donor_ids, len(residuals), 'donor_ids', unique=True)
    generic = normalized_topk_weights(similarity, donor_ids=names, top_k=top_k,
                                      query_groups=groups, donor_groups=groups)
    candidates, best = [], None
    for mixing_lambda in LAMBDA_GRID:
        weighted = augment_generic_weights(generic['weights'], target_similarity, moa_similarity,
            mixing_lambda=mixing_lambda, eligible=generic['eligible'], shrinkage=shrinkage)
        raw_mean = weighted['weights']@residuals
        for alpha in ALPHA_GRID:
            prediction = alpha*raw_mean
            mse = float(np.square(residuals-prediction).mean())
            if not np.isfinite(mse):
                raise ValueError('Donor-only selection MSE overflowed')
            candidates.append(dict(mixing_lambda=mixing_lambda, alpha=alpha, donor_loo_mse=mse))
            if best is None or (mse < best['donor_loo_mse'] and not np.isclose(
                    mse, best['donor_loo_mse'], rtol=TIE_RTOL, atol=TIE_ATOL)):
                best = dict(mixing_lambda=mixing_lambda, alpha=alpha, donor_loo_mse=mse,
                    donor_loo_prediction=prediction.copy(), donor_loo_weights=weighted['weights'].copy(),
                    donor_biology_support=weighted['support'].copy(), donor_biology_neff=weighted['neff'].copy())
    return dict(**best, candidate_scores=candidates, baseline_donor_mse=float(np.square(residuals).mean()),
        top_k=generic['top_k'], shrinkage=float(shrinkage), donor_ids=names.tolist(),
        donor_groups=groups.tolist(), lambda_grid=list(LAMBDA_GRID), alpha_grid=list(ALPHA_GRID),
        selection_scope='donor leave-chemistry-group-out nine-coordinate residual mean MSE; no query labels',
        tie_rule='lower lambda then lower alpha within the stated numerical tie tolerance',
        tie_rtol=TIE_RTOL, tie_atol=TIE_ATOL)


def predict_memory(generic_similarity, donor_residuals, selection, *, donor_ids,
                   target_similarity=None, moa_similarity=None, eligible=None,
                   query_groups=None, donor_groups=None):
    """Return a selected residual-mean correction for unlabeled query inputs.

    prediction_mean is alpha times the weighted donor residual mean. The
    caller may add it to frozen A. Covariance is deliberately not scaled here;
    call weighted_residual_moments separately for empirical joint moments.
    """
    residuals = _matrix(donor_residuals, 'donor_residuals', 9)
    names = _names(donor_ids, len(residuals), 'donor_ids', unique=True)
    if set(names.tolist()) != set(selection['donor_ids']):
        raise ValueError('Prediction donor identities must match the donor-only selection population')
    selected_groups = dict(zip(selection['donor_ids'], selection['donor_groups']))
    if donor_groups is not None and list(map(str, donor_groups)) != [selected_groups[unit] for unit in names]:
        raise ValueError('Prediction donor groups differ from donor-only selection')
    generic = normalized_topk_weights(generic_similarity, donor_ids=names, top_k=selection['top_k'],
        eligible=eligible, query_groups=query_groups, donor_groups=donor_groups)
    if generic['weights'].shape[1] != len(residuals) or not np.isfinite(residuals).all():
        raise ValueError('Prediction weights and complete finite donor residuals must align')
    if selection['alpha'] not in ALPHA_GRID:
        raise ValueError('Selected alpha is outside the prespecified grid')
    weighted = augment_generic_weights(generic['weights'], target_similarity, moa_similarity,
        mixing_lambda=selection['mixing_lambda'], eligible=generic['eligible'], shrinkage=selection['shrinkage'])
    raw_mean = weighted['weights']@residuals
    return dict(**weighted, prediction_mean=float(selection['alpha'])*raw_mean,
        raw_weighted_mean=raw_mean, alpha=float(selection['alpha']),
        generic_weights=generic['weights'], generic_support=generic['support'],
        generic_fallback=generic['fallback'], generic_neff=generic['neff'])
