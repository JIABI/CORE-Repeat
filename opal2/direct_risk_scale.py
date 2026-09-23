"""Low-parameter scalar-scatter fitting to average CRPS, not oracle labels."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .variance_headroom_experiment import gamma_fast
from .variance_headroom_math import ETA_BOUND, fair_gamma_crps


def centered_ecdf(reference, query):
    reference, query = np.asarray(reference, float), np.asarray(query, float)
    if reference.ndim != 1 or query.ndim != 1 or not len(reference):
        raise ValueError('Need nonempty training scores and query vector')
    if not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError('Nonfinite rank inputs')
    ref = np.sort(reference)
    return (np.searchsorted(ref, query, 'left')+np.searchsorted(ref, query, 'right'))/len(ref)-1.


def scale_function(parameters, scores):
    p, s = np.asarray(parameters, float), np.asarray(scores, float)
    if p.shape not in ((1,), (2,)) or s.ndim != 1:
        raise ValueError('One or two parameters and one score per object required')
    if not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(abs(p)>ETA_BOUND+1e-10):
        raise ValueError('Invalid bounded parameters')
    return np.clip(p[0]+(p[1]*s if len(p)==2 else 0.)+np.zeros_like(s), -ETA_BOUND, ETA_BOUND)


def group_weights(groups):
    groups = np.asarray(groups)
    names, inverse, count = np.unique(groups, return_inverse=True, return_counts=True)
    if len(names)<3:
        raise ValueError('At least three chemical groups required')
    return 1./(len(names)*count[inverse])


def interpolate_scores(table, grid, eta):
    table, grid, eta = np.asarray(table,float), np.asarray(grid,float), np.asarray(eta,float)
    if table.shape != (len(eta),len(grid)) or len(grid)<3 or not np.all(np.diff(grid)>0):
        raise ValueError('Invalid aligned score grid')
    if not all(np.isfinite(a).all() for a in (table,grid,eta)):
        raise ValueError('Nonfinite score interpolation')
    if np.any(eta<grid[0]-1e-10) or np.any(eta>grid[-1]+1e-10):
        raise ValueError('No extrapolation beyond sampled family')
    j=np.clip(np.searchsorted(grid,eta,side='right')-1,0,len(grid)-2)
    f=(eta-grid[j])/(grid[j+1]-grid[j])
    rows=np.arange(len(eta))
    return (1-f)*table[rows,j]+f*table[rows,j+1]


def sampled_scores(cells, eta):
    """Exact fair CRPS for fixed MC draws, with object-specific scalar scale.

    Each cell contains fixed raw-coordinate mean and radial residual draws;
    those distributions were constructed without the calibration object's Y.
    """
    eta=np.asarray(eta,float)
    if eta.ndim!=1 or np.any(abs(eta)>ETA_BOUND+1e-10) or not np.isfinite(eta).all():
        raise ValueError('Invalid scalar increments')
    result=np.empty(len(eta));seen=np.zeros(len(eta),int)
    for cell in cells:
        ix=np.asarray(cell['indices'],int)
        if len(ix)!=len(cell['mean']) or cell['error'].shape[1:]!=(len(ix),9):
            raise ValueError('Misaligned distribution cell')
        for start in range(0,len(ix),8):
            sl=slice(start,start+8);rows=ix[sl]
            raw=cell['mean'][None,sl]+cell['error'][:,sl]*np.exp(.5*eta[rows])[None,:,None]
            result[rows]=fair_gamma_crps(gamma_fast(raw),cell['actual'][sl])
        np.add.at(seen,ix,1)
    if not np.all(seen==1):
        raise ValueError('Each calibration object must occur exactly once')
    return result


def one_se_zero_preference(candidate, baseline, groups):
    candidate,baseline,groups=np.asarray(candidate,float),np.asarray(baseline,float),np.asarray(groups)
    if candidate.shape!=baseline.shape or candidate.shape!=groups.shape:
        raise ValueError('Misaligned calibration losses')
    deltas=np.asarray([(candidate[groups==g]-baseline[groups==g]).mean() for g in np.unique(groups)])
    if len(deltas)<3 or not np.isfinite(deltas).all():
        raise ValueError('Need finite group losses')
    change=float(deltas.mean());se=float(deltas.std(ddof=1)/np.sqrt(len(deltas)))
    return bool(change < -se),dict(mean_change=change,group_se=se,groups=len(deltas),
        scope='Calibration preference for zero, not a confidence guarantee')


def fit_direct_map(cells, groups, scores, *, dimensions=2, tolerance=5e-6, grid_cache=None):
    """Fit only calibration outcomes; query outcomes are not accepted.

    A piecewise-linear score table accelerates two-dimensional search. The
    fitted function is checked against its actual sampled average loss, then
    refined or optimized directly when the declared tolerance is exceeded.
    """
    groups,scores=np.asarray(groups),np.asarray(scores,float)
    if dimensions not in (1,2) or scores.shape!=groups.shape:
        raise ValueError('Invalid low-parameter calibration family')
    weights=group_weights(groups);n=len(scores);zero=np.zeros(n)
    cache={} if grid_cache is None else grid_cache
    if 0. not in cache:cache[0.]=sampled_scores(cells,zero)
    baseline=cache[0.];history=[]
    best_parameters=np.zeros(dimensions);best_scores=baseline.copy()
    for knots in (17,33,65):
        grid=np.linspace(-ETA_BOUND,ETA_BOUND,knots);grid[knots//2]=0.
        table=[]
        for eta in grid:
            # Grid refinements share all previous dyadic knots and MC draws.
            key=round(float(eta),14)
            if key not in cache:cache[key]=sampled_scores(cells,np.full(n,eta))
            table.append(cache[key])
        table=np.column_stack(table)
        def objective(parameters):
            return float(weights@interpolate_scores(table,grid,scale_function(parameters,scores)))
        starts=[np.zeros(dimensions),best_parameters]
        if dimensions==1:
            starts += [np.array([grid[int(np.argmin(weights@table))]])]
        else:
            starts += [np.array([a,b]) for a in (-.35,0.,.35) for b in (-.35,0.,.35)]
        optimizers=[minimize(objective,s,method='Powell',bounds=[(-ETA_BOUND,ETA_BOUND)]*dimensions,
                     options=dict(xtol=1e-5,ftol=1e-10,maxiter=100)) for s in starts]
        candidates=[np.zeros(dimensions)]+[np.asarray(o.x) for o in optimizers]
        best_parameters=min(candidates,key=lambda p:(objective(p),float(np.square(p).sum())))
        eta=scale_function(best_parameters,scores)
        exact=sampled_scores(cells,eta);approximate=objective(best_parameters)
        discrepancy=float(abs(weights@exact-approximate))
        history.append(dict(knots=knots,parameters=best_parameters.tolist(),
            interpolated_loss=approximate,sampled_loss=float(weights@exact),
            discrepancy=discrepancy,optimizer_success=[bool(o.success) for o in optimizers]))
        best_scores=exact
        if discrepancy<=tolerance:break
    else:
        def direct_objective(p):return float(weights@sampled_scores(cells,scale_function(p,scores)))
        optimum=minimize(direct_objective,best_parameters,method='Powell',
            bounds=[(-ETA_BOUND,ETA_BOUND)]*dimensions,options=dict(xtol=2e-4,ftol=1e-8,maxiter=50))
        if optimum.fun < weights@best_scores:
            best_parameters=np.asarray(optimum.x);best_scores=sampled_scores(cells,scale_function(best_parameters,scores))
        history.append(dict(direct_optimization=True,success=bool(optimum.success),
                            parameters=best_parameters.tolist(),sampled_loss=float(weights@best_scores)))
    # Interpolation is exact at a grid knot even if the continuous optimum
    # lies between knots. Always refine locally against the actual loss.
    step=2*ETA_BOUND/(history[-1].get('knots',65)-1)
    local_bounds=[(max(-ETA_BOUND,float(p)-step),min(ETA_BOUND,float(p)+step)) for p in best_parameters]
    def local_objective(p):return float(weights@sampled_scores(cells,scale_function(p,scores)))
    refined=minimize(local_objective,best_parameters,method='Powell',bounds=local_bounds,
        options=dict(xtol=5e-4,ftol=1e-9,maxiter=20))
    if refined.fun < weights@best_scores:
        best_parameters=np.asarray(refined.x);best_scores=sampled_scores(cells,scale_function(best_parameters,scores))
    history.append(dict(local_direct_refinement=True,success=bool(refined.success),
        parameters=best_parameters.tolist(),sampled_loss=float(weights@best_scores),evaluations=int(refined.nfev)))
    keep,preference=one_se_zero_preference(best_scores,baseline,groups)
    fitted=best_parameters.copy() if keep else np.zeros(dimensions)
    return dict(parameters=fitted.tolist(),unshrunk_parameters=best_parameters.tolist(),
        calibration_baseline_loss=float(weights@baseline),calibration_candidate_loss=float(weights@best_scores),
        retained=keep,zero_preference=preference,n=n,dimensions=dimensions,history=history,
        fitting_target='chemistry-group-equal mean fair Gamma CRPS, not per-realization argmin',
        increment=scale_function(fitted,scores),candidate_scores=best_scores,baseline_scores=baseline)
