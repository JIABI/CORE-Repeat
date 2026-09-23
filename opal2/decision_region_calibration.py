"""Frozen-score risk calibration, evaluated separately from allocation.

This module does not fit a measurement model or alter its Gamma distribution.
The offset's two predeclared parameters use honest, within-cell CAL predictions.
CAL ranks are formed within each wholly held-out inner fold, never by pooling
mutually label-dependent out-of-fold scores. Query ranks use the actual cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score


VERSION = 1
PENALTY = 10.0
TAIL_FRACTION = 0.25
LAMBDA = 0.2
EPS = 1e-6
VARIANTS = ("ORIGINAL", "GLOBAL", "REGION")
ARMS = ("CORE", "HISTGB_CAL", "HISTGB_COHERENT",
        "EXTRATREES_CAL", "EXTRATREES_COHERENT")


def _vector(value, name, n=None):
    x = np.asarray(value, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(name + " must be a nonempty finite vector")
    if n is not None and len(x) != n:
        raise ValueError(name + " length mismatch")
    return x


def _probability(value):
    p = _vector(value, "probability")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probability outside [0,1]")
    return p


def frozen_rank_feature(mean, probability, ids, cohorts=None):
    """Descending score rank, with the original ID tie rule, within cohorts.

    For CAL, cohorts MUST be the inner held-out fold labels. Labels/outcomes are
    intentionally not accepted here. Midranks by position include neither 0 nor
    1, and h is zero outside the top quarter of the original composite score.
    """
    p = _probability(probability)
    m = _vector(mean, "mean", len(p))
    ids = np.asarray(ids).astype(str)
    if ids.shape != p.shape or len(set(ids)) != len(ids):
        raise ValueError("Unique aligned IDs required")
    c = np.zeros(len(p), int) if cohorts is None else np.asarray(cohorts)
    if c.shape != p.shape:
        raise ValueError("Cohort labels are not aligned")
    rank = np.empty(len(p), float)
    for group in np.unique(c):
        rows = np.flatnonzero(c == group)
        order = np.lexsort((ids[rows], -(m[rows] - LAMBDA*p[rows])))
        rank[rows[order]] = (np.arange(len(rows)) + 0.5) / len(rows)
    return rank, np.maximum(0.0, 1.0-rank/TAIL_FRACTION)


@dataclass
class RiskOffset:
    variant: str
    coefficient: np.ndarray
    n_cal: int
    penalty: float = PENALTY

    def predict(self, probability, feature, *, enabled=True):
        p = _probability(probability)
        h = _vector(feature, "rank feature", len(p))
        if (not enabled or self.variant == "ORIGINAL"
                or np.all(self.coefficient == 0)):
            return p.copy()  # exact identity, including saved zero/one values
        x = np.ones((len(p), 1))
        if self.variant == "REGION":
            x = np.column_stack((x, h))
        if self.variant not in ("GLOBAL", "REGION"):
            raise ValueError("Unknown risk offset")
        return expit(logit(np.clip(p, EPS, 1-EPS)) + x@self.coefficient)

    def record(self):
        return dict(variant=self.variant, coefficient=self.coefficient.tolist(),
                    n_cal=self.n_cal, penalty=self.penalty,
                    loss="sum Bernoulli log loss + penalty/2 * squared coefficients",
                    query_labels_used=False, data_selected_hyperparameters=False)


def fit_risk_offset(probability, null_label, feature, variant):
    """One fixed penalized likelihood fit; no strength search or winner choice."""
    p = _probability(probability)
    y = _vector(null_label, "NULL label", len(p))
    h = _vector(feature, "rank feature", len(p))
    if np.any((y != 0) & (y != 1)) or np.any((h < 0) | (h > 1)):
        raise ValueError("Binary labels and [0,1] rank features required")
    if variant == "ORIGINAL":
        return RiskOffset(variant, np.zeros(1), len(p))
    if variant not in ("GLOBAL", "REGION"):
        raise ValueError("Unknown risk offset")
    x = np.ones((len(p), 1))
    if variant == "REGION":
        x = np.column_stack((x, h))
    offset = logit(np.clip(p, EPS, 1-EPS))

    def objective(beta):
        z = offset + x@beta
        loss = np.sum(np.logaddexp(0.0, z)-y*z) + PENALTY/2 * (beta@beta)
        gradient = x.T@(expit(z)-y) + PENALTY*beta
        return float(loss), gradient

    fitted = minimize(objective, np.zeros(x.shape[1]), jac=True,
                      method="L-BFGS-B", options=dict(maxiter=500, gtol=1e-9, ftol=1e-14))
    grad = np.linalg.norm(objective(fitted.x)[1], ord=np.inf)
    if not np.isfinite(fitted.x).all() or (not fitted.success and grad > 1e-5):
        raise RuntimeError("Risk offset did not converge: " + str(fitted.message))
    return RiskOffset(variant, fitted.x.copy(), len(p))


def select_top_k(ids, mean, probability, k, lam=LAMBDA):
    p = _probability(probability)
    m = _vector(mean, "mean", len(p))
    ids = np.asarray(ids).astype(str)
    if ids.shape != p.shape or len(set(ids)) != len(ids) or not 0 <= k <= len(ids):
        raise ValueError("Invalid selection inputs")
    mask = np.zeros(len(ids), bool)
    mask[np.lexsort((ids, -(m-float(lam)*p)))[:k]] = True
    return mask


def validate_cell(cell):
    """Validate the query/CAL handoff; upstream adapters validate FIT/REF roles."""
    for prefix in ("cal", "query"):
        ids = np.asarray(cell[prefix+"_ids"]).astype(str)
        groups = np.asarray(cell[prefix+"_groups"]).astype(str)
        actual = _vector(cell[prefix+"_actual"], prefix+" Gamma")
        if ids.shape != actual.shape or groups.shape != actual.shape:
            raise ValueError("Misaligned " + prefix + " roles")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs within " + prefix)
    if set(cell['cal_groups']) & set(cell['query_groups']):
        raise ValueError("CAL and QUERY chemistry overlap")
    if set(cell['cal_ids']) & set(cell['query_ids']):
        raise ValueError("CAL and QUERY identity overlap")
    f = np.asarray(cell['cal_inner_fold'])
    if f.shape != np.asarray(cell['cal_ids']).shape or len(np.unique(f)) < 2:
        raise ValueError("Honest CAL inner folds required")
    for group in set(cell['cal_groups']):
        if len(np.unique(f[np.asarray(cell['cal_groups']) == group])) != 1:
            raise ValueError("CAL chemistry split across inner folds")
    if np.asarray(cell['query_layout']).shape != np.asarray(cell['query_ids']).shape:
        raise ValueError("Query layout missing or unaligned")
    budget = cell.get('query_budget')
    if (isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer))
            or not 1 <= budget <= len(cell['query_ids'])):
        raise ValueError("Explicit original integer query_budget required")
    if set(cell['arms']) != set(ARMS):
        raise ValueError("Full predeclared arm roster required")
    for arm in ARMS:
        for prefix in ('cal', 'query'):
            n = len(cell[prefix+'_ids'])
            _vector(cell['arms'][arm][prefix+'_mean'], arm+' mean', n)
            p = _probability(cell['arms'][arm][prefix+'_p'])
            if len(p) != n:
                raise ValueError("Arm probability length mismatch")


def evaluate_cell(cell):
    validate_cell(cell)
    ids = np.asarray(cell['query_ids']).astype(str)
    y = np.asarray(cell['query_actual'], float)
    n = len(ids)
    k = int(cell['query_budget'])
    core = cell['arms']['CORE']
    core_mask = select_top_k(ids, core['query_mean'], core['query_p'], k)
    arrays = dict(ids=ids, groups=np.asarray(cell['query_groups']).astype(str),
                  layout=np.asarray(cell['query_layout']).astype(str), actual=y,
                  cell=np.full(n, str(cell['cell'])), CORE_selected=core_mask)
    fitted = {}
    for arm in ARMS:
        values = cell['arms'][arm]
        cal_rank, cal_h = frozen_rank_feature(values['cal_mean'], values['cal_p'],
                                              cell['cal_ids'], cell['cal_inner_fold'])
        query_rank, query_h = frozen_rank_feature(values['query_mean'], values['query_p'], ids)
        original = select_top_k(ids, values['query_mean'], values['query_p'], k)
        lam0 = select_top_k(ids, values['query_mean'], values['query_p'], k, 0.)
        arrays[arm+'__mean'] = np.asarray(values['query_mean'], float).copy()
        arrays[arm+'__rank'] = query_rank
        arrays[arm+'__original_selected'] = original
        arrays[arm+'__lambda0_selected'] = lam0
        fitted[arm] = {}
        for variant in VARIANTS:
            model = fit_risk_offset(values['cal_p'],
                                    (np.asarray(cell['cal_actual']) <= 0).astype(float),
                                    cal_h, variant)
            probability = model.predict(values['query_p'], query_h)
            mask = select_top_k(ids, values['query_mean'], probability, k)
            prefix = arm+'__'+variant
            arrays[prefix+'__p'] = probability
            arrays[prefix+'__selected'] = mask
            fitted[arm][variant] = model.record()
            if variant == 'ORIGINAL':
                if not np.array_equal(probability, values['query_p']):
                    raise AssertionError("Original probabilities changed")
                if not np.array_equal(mask, original):
                    raise AssertionError("Original selection changed")
            if not np.array_equal(select_top_k(ids, values['query_mean'], probability, k, 0.), lam0):
                raise AssertionError("Risk correction changed lambda=0 selection")
            if 'query_seed_p' in values:
                seed_p=np.asarray(values['query_seed_p'])
                seed_m=np.asarray(values['query_seed_mean'])
                if seed_p.shape != seed_m.shape or seed_p.ndim != 2 or seed_p.shape[1] != n:
                    raise ValueError("Original extra MC seeds are not aligned")
                for index in range(len(seed_p)):
                    _,seed_h=frozen_rank_feature(seed_m[index],seed_p[index],ids)
                    corrected=model.predict(seed_p[index],seed_h)
                    arrays[prefix+'__mc%d_p'%index]=corrected
                    arrays[prefix+'__mc%d_selected'%index]=select_top_k(ids,seed_m[index],corrected,k)
        fitted[arm]['CAL_rank_scope'] = 'within wholly held-out CAL inner fold'
        fitted[arm]['CAL_rank_range'] = [float(cal_rank.min()), float(cal_rank.max())]
    return arrays, dict(dataset=cell['dataset'], cell=cell['cell'], n=n, k=k,
                        n_cal=len(cell['cal_ids']),
                        cal_inner_sizes={str(g): int(np.sum(np.asarray(cell['cal_inner_fold']) == g))
                                         for g in np.unique(cell['cal_inner_fold'])},
                        models=fitted, gamma_mean_and_law_unchanged=True,
                        fitted_parameters_by_variant=dict(ORIGINAL=0,GLOBAL=1,REGION=2),
                        model_selection_performed=False)


def probability_rows(probability, actual):
    p = _probability(probability)
    y = (_vector(actual, 'Gamma', len(p)) <= 0).astype(float)
    clipped = np.clip(p, EPS, 1-EPS)
    return dict(brier=(p-y)**2,
                logloss=-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)),
                null_gap=y-p, null_label=y)


def risk_summary(probability, actual, mask):
    rows = probability_rows(probability, actual)
    mask = np.asarray(mask, bool)
    if mask.shape != rows['brier'].shape:
        raise ValueError("Mask shape mismatch")
    count = int(mask.sum())
    if not count:
        return dict(n=0, predicted_NULL=0., actual_NULL=0, gap=0., brier=None,
                    logloss=None, auc=None)
    p, y = np.asarray(probability)[mask], rows['null_label'][mask]
    return dict(n=count, predicted_NULL=float(p.sum()), actual_NULL=int(y.sum()),
                gap=float((y-p).sum()), brier=float(rows['brier'][mask].mean()),
                logloss=float(rows['logloss'][mask].mean()),
                auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None)


def paired_cluster_interval(difference, cluster, *, denominator=None, replicates=2000, seed=20260920):
    """Fixed-prediction, fixed-list paired cluster resampling, not certification.

    Difference and denominator are contributions per candidate. For a selected
    Brier difference, denominator is the SAME fixed selection mask. For policy
    value or false activation it is one for every candidate, matching R2.
    """
    d = _vector(difference, 'difference')
    c = np.asarray(cluster).astype(str)
    weights = np.ones(len(d)) if denominator is None else np.asarray(denominator, float)
    if c.shape != d.shape or weights.shape != d.shape or np.any(weights < 0):
        raise ValueError("Unaligned clusters or denominators")
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Positive finite total denominator required")
    unique, index = np.unique(c, return_inverse=True)
    numerator = np.bincount(index, weights=d, minlength=len(unique))
    totals = np.bincount(index, weights=weights, minlength=len(unique))
    point = float(numerator.sum()/totals.sum())
    if len(unique) < 2:
        return dict(difference=point, ci95=None, clusters=len(unique), valid_replicates=0,
                    scope='one cluster; uncertainty not estimated')
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, replicates, 64):
        take = rng.integers(0, len(unique), size=(min(64, replicates-start), len(unique)))
        den = totals[take].sum(1)
        good = den > 0
        estimates.extend((numerator[take].sum(1)[good]/den[good]).tolist())
    return dict(difference=point, ci95=np.quantile(estimates, [.025, .975]).tolist(),
                clusters=len(unique), valid_replicates=len(estimates),
                scope='paired fixed-fitted predictions and fixed candidate/list contributions',
                few_clusters=len(unique) < 10, finite_sample_guarantee=False)
