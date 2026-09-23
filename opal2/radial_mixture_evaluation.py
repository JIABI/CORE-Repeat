"""Stratified Monte Carlo scoring of fixed-law radial retrieval mixtures.

All endpoint laws share mean, scatter, empirical centers/bandwidth and Gaussian
guard. Convex retrieval weights therefore give an exact mixture of endpoint
distributions, including after the nonlinear Gamma/observable transformation.
Endpoint expectations are Monte Carlo estimates, not analytic Gamma integrals.

This evaluator draws S samples per endpoint with common random numbers (CRN)
and integrates the mixture coefficients, rather than resampling S mixture
labels for every arm. It changes the realized estimator, not the predictive
law or the per-endpoint sample budget. Cross-component CRPS excludes equal
draw indices; cross-component energy pairs independent halves and symmetrizes.
These are unbiased proper-score estimators despite within-index CRN dependence.

No fitting, model admission, gate learning, cohort ranking or selection occurs
here. Targets are used only to score a predeclared distribution. Caller-supplied
coefficients must already have been fixed without these query outcomes.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .conditional_joint_error_experiment import OBSERVABLES, observable_forward
from .empirical_radial import (draw_radial, radial_cdf, radial_nll, radial_ppf,
                               variance_multiplier)


LEVELS = np.array([.5, .8, .9, .95, .99])
RADIUS_SEED_OFFSET = 47000
ESTIMATOR = (
    "S draws per endpoint with shared CRN; affine endpoint expectations; "
    "fair CRPS all ordered unequal-index pairs; symmetric independent-half energy; "
    "analytic fixed-law mixture density, radial regions and covariance"
)


def _coefficients(values, n, k):
    c = np.asarray(values, dtype=np.float64)
    if c.shape == (k,):
        c = np.broadcast_to(c, (n,k)).copy()
    if (c.shape != (n,k) or not np.isfinite(c).all() or np.any(c < 0)
            or not np.allclose(c.sum(1),1.,rtol=0,atol=1e-12)):
        raise ValueError('Coefficients must be finite convex length-K or [N,K] arrays')
    return c


def _draws(values):
    arrays = tuple(np.asarray(value,dtype=np.float64) for value in values)
    if (not arrays or arrays[0].ndim < 2 or arrays[0].shape[0] < 2
            or arrays[0].shape[1] < 1 or any(a.shape != arrays[0].shape for a in arrays)
            or any(not np.isfinite(a).all() for a in arrays)):
        raise ValueError('Endpoint draws must share finite [S,N,...] shapes, with S>=2')
    return arrays


def _cross_absolute(left, right, sorted_left, sorted_right):
    """All ordered cross pairs except paired-index diagonal, O(S log S)."""
    s = left.shape[0]
    a,b = sorted_left.reshape(s,-1),sorted_right.reshape(s,-1)
    values = np.empty(a.shape[1])
    diagonal = np.abs(left-right).reshape(s,-1).sum(0)
    for column in range(a.shape[1]):
        x,y = a[:,column],b[:,column]
        prefix = np.empty(s+1)
        prefix[0]=0.
        np.cumsum(y,out=prefix[1:])
        count = np.searchsorted(y,x,side='right')
        total = np.sum(x*(2*count-s)+prefix[-1]-2*prefix[count])
        values[column]=(total-diagonal[column])/(s*(s-1))
    return values.reshape(left.shape[1:])


def fair_cross_absolute(left, right):
    """Mean |L_s-R_t| over s!=t, not same-index CRN pairs.

    The result has the input's trailing shape. A matched-index pair can be
    dependent because endpoints share primitives; only unequal indices have
    the independent product distribution required by CRPS. No S-by-S array is
    materialized. Inputs need shape [S,...], S>=2, and identical finite shapes.
    """
    a,b = np.asarray(left,dtype=float),np.asarray(right,dtype=float)
    if (a.shape != b.shape or a.ndim < 1 or a.shape[0] < 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Cross draws must have equal finite shapes and S>=2')
    return _cross_absolute(a,b,np.sort(a,axis=0),np.sort(b,axis=0))


def _crps_terms(draws, target):
    arrays = _draws(draws)
    target=np.asarray(target,dtype=float)
    if target.shape != arrays[0].shape[1:] or not np.isfinite(target).all():
        raise ValueError('Target must match finite endpoint trailing dimensions')
    k,s=len(arrays),len(arrays[0])
    first=np.stack([np.abs(a-target).mean(0) for a in arrays])
    pair=np.empty((k,k,*target.shape))
    sorted_arrays=tuple(np.sort(a,axis=0) for a in arrays)
    ranks=(2*np.arange(s)-s+1).reshape((s,)+(1,)*(arrays[0].ndim-1))
    for i in range(k):
        pair[i,i]=2*np.sum(ranks*sorted_arrays[i],axis=0)/(s*(s-1))
        for j in range(i):
            pair[i,j]=pair[j,i]=_cross_absolute(arrays[i],arrays[j],sorted_arrays[i],sorted_arrays[j])
    return first,pair


def _combine(first, pair, coefficients):
    return (np.einsum('nk,kn...->n...',coefficients,first)
            -.5*np.einsum('nk,nl,kln...->n...',coefficients,coefficients,pair))


def fair_mixture_crps(draws, coefficients, target):
    """Unbiased scalar-marginal CRPS for a convex endpoint distribution mixture.

    Each endpoint has [S,N,...] draws; target is [N,...]. Coefficients are
    length K or [N,K]. Equal coefficients do not mean averaging draw values:
    this function integrates the mixture's within/cross-component distances.
    """
    first,pair=_crps_terms(draws,target)
    c=_coefficients(coefficients,first.shape[1],first.shape[0])
    return _combine(first,pair,c)


def _energy_terms(draws,target):
    arrays=_draws(draws)
    target=np.asarray(target,dtype=float)
    if (arrays[0].ndim != 3 or arrays[0].shape[0] % 2
            or target.shape != arrays[0].shape[1:] or not np.isfinite(target).all()):
        raise ValueError('Energy requires even S, [S,N,d] draws and finite [N,d] target')
    k,h=len(arrays),len(arrays[0])//2
    first=np.stack([np.linalg.norm(a-target,axis=-1).mean(0) for a in arrays])
    pair=np.empty((k,k,len(target)))
    for i in range(k):
        pair[i,i]=np.linalg.norm(arrays[i][:h]-arrays[i][h:],axis=-1).mean(0)
        for j in range(i):
            pair[i,j]=pair[j,i]=.5*(
                np.linalg.norm(arrays[i][:h]-arrays[j][h:],axis=-1).mean(0)
                +np.linalg.norm(arrays[j][:h]-arrays[i][h:],axis=-1).mean(0))
    return first,pair


def fair_mixture_energy(draws, coefficients, target):
    """Joint energy with independent MC-index halves, symmetric across endpoints."""
    first,pair=_energy_terms(draws,target)
    return _combine(first,pair,_coefficients(coefficients,first.shape[1],first.shape[0]))


def evaluate_radial_mixtures(mean,scatter,target,stats,actual,obs_actual,
        absolute_actual,norm2_per_feature,seed,*,law,endpoint_weights,coefficients,
        samples=100000,baseline_saved=None,query_block_size=4):
    """Return per-arm arrays for predeclared convex CORE/biology mixtures.

    ``endpoint_weights`` is an insertion-ordered mapping with CORE first and
    [N,calibration_n] normalized nonnegative arrays. ``coefficients`` maps arm
    names to length-K or [N,K] convex vectors in that endpoint order. The same
    ``law``, mean and scatter apply to all endpoints and arms. Unsupported
    endpoints should be replaced by CORE or receive zero coefficient upstream.

    The original scorer's 16-query RNG chunks and random primitives are kept;
    sub-blocks bound memory without changing draw index correspondence. Mean
    Gamma and P(NULL) use endpoint integration. Their MC standard errors use
    the sample variance of the corresponding per-index CRN-weighted endpoint
    statistic, not the predictive mixture variance. MC-SE fields therefore
    describe this stratified estimator, not a newly sampled mixture ensemble.

    All joint levels and covariance are analytic for the mixed retrieval law.
    Scalar intervals are deliberately not produced. ``baseline_saved`` may
    supply the existing CORE score arrays; common fields on exactly CORE-only
    rows are copied bit-for-bit. No baseline-only fields are manufactured for
    non-core rows, and no saved artifact is changed. If all required Monte
    Carlo fields exist in that baseline, globally CORE-only queries reuse
    those saved estimates without recomputing endpoint geometry; random stream
    positions are still consumed, preserving every active query's draw indices.
    """
    mean,scatter,target=np.asarray(mean,float),np.asarray(scatter,float),np.asarray(target,float)
    if (mean.ndim != 2 or mean.shape[1] != 9 or len(mean)<1 or target.shape != mean.shape
            or scatter.shape != (len(mean),9,9) or not all(np.isfinite(v).all() for v in (mean,scatter,target))):
        raise ValueError('Expected finite mean/target [N,9] and scatter [N,9,9]')
    n,d=mean.shape
    if (not isinstance(samples,(int,np.integer)) or isinstance(samples,bool)
            or samples<4 or samples%2):
        raise ValueError('Samples must be an even integer >=4')
    if (not isinstance(query_block_size,(int,np.integer)) or isinstance(query_block_size,bool)
            or not 1<=query_block_size<=16):
        raise ValueError('query_block_size must be an integer from 1 to 16')
    if (not isinstance(endpoint_weights,Mapping) or not endpoint_weights
            or next(iter(endpoint_weights)) != 'CORE'):
        raise ValueError('Endpoint mapping must be nonempty with CORE first')
    if not isinstance(coefficients,Mapping) or not coefficients:
        raise ValueError('At least one predeclared arm is required')
    endpoints=tuple(endpoint_weights)
    weights=tuple(np.asarray(endpoint_weights[key],float) for key in endpoints)
    k=len(weights)
    expected_shape=(n,len(law['log_centers']))
    if any(w.shape != expected_shape or not np.isfinite(w).all() or np.any(w<0)
           or not np.allclose(w.sum(1),1.,rtol=0,atol=1e-12) for w in weights):
        raise ValueError('Each endpoint needs normalized finite nonnegative [N,calibration_n] weights')
    coefficients={name:_coefficients(value,n,k) for name,value in coefficients.items()}
    if not np.allclose(scatter,scatter.swapaxes(-1,-2),atol=1e-12,rtol=1e-12):
        raise ValueError('Scatter must be symmetric')
    chol=np.linalg.cholesky(scatter)
    raw_center,raw_scale=np.asarray(stats['u_center'],float),np.asarray(stats['u_scale'],float)
    if (raw_center.shape != (d,) or raw_scale.shape != (d,)
            or not np.isfinite(raw_center).all() or not np.isfinite(raw_scale).all()
            or np.any(raw_scale<=0)):
        raise ValueError('Saved target center/scale must be finite dimension-nine vectors with positive scales')
    actual,obs_actual,absolute_actual,norm2_per_feature=map(lambda v:np.asarray(v,float),
        (actual,obs_actual,absolute_actual,norm2_per_feature))
    if (actual.shape!=(n,) or obs_actual.shape!=(n,len(OBSERVABLES))
            or absolute_actual.shape!=(n,3) or norm2_per_feature.shape!=(n,)
            or not all(np.isfinite(v).all() for v in (actual,obs_actual,absolute_actual,norm2_per_feature))
            or np.any(norm2_per_feature<=0)):
        raise ValueError('Actual observables and positive first-well norm must have aligned finite shapes')
    residual=target-mean
    radius2=np.square(np.linalg.solve(chol,residual[...,None])[...,0]).sum(-1)
    stores={}
    for arm,c in coefficients.items():
        mixed=sum(c[:,j,None]*w for j,w in enumerate(weights))
        # Analytic density, radial CDF/PPF and second moment of the exact law.
        multiplier=variance_multiplier(law,mixed)
        thresholds=np.square(radial_ppf(law,mixed,LEVELS))
        stores[arm]=dict(nll=radial_nll(residual,scatter,law,mixed),
            radial_pit=radial_cdf(law,mixed,np.sqrt(radius2)),
            joint_squared_radius_by_level=thresholds,
            joint_coverage_by_level=(radius2[:,None]<=thresholds).astype(float),
            joint_coverage=(radius2<=thresholds[:,3]).astype(float),
            radial_variance_multiplier=multiplier,covariance_u=scatter*multiplier[:,None,None],
            mahalanobis2=radius2.copy(),mse=np.square(residual).mean(1))
        for key in ('predicted','p_null','crps','energy','single_crps','pair_crps','average_crps',
                    'triple_average_crps','absolute_pair_crps','gamma_mc_se','null_mc_se',
                    'predicted_cos_Z1_V','predicted_cos_Z2_V'):
            stores[arm][key]=np.empty(n)
        stores[arm]['observable_crps']=np.empty((n,len(OBSERVABLES)))
        stores[arm]['absolute_crps_by_pair']=np.empty((n,3))
    mc_keys=('predicted','p_null','crps','energy','single_crps','pair_crps','average_crps',
        'triple_average_crps','absolute_pair_crps','gamma_mc_se','null_mc_se',
        'predicted_cos_Z1_V','predicted_cos_Z2_V','observable_crps','absolute_crps_by_pair')
    saved={}
    if baseline_saved is not None:
        for key in set(next(iter(stores.values()))) & set(baseline_saved):
            value=np.asarray(baseline_saved[key])
            if value.shape != next(iter(stores.values()))[key].shape or not np.isfinite(value).all():
                raise ValueError(f'Saved CORE field {key} has incompatible shape or nonfinite values')
            saved[key]=value
    globally_core=np.zeros(n,dtype=bool)
    if all(key in saved for key in mc_keys):
        globally_core=np.logical_and.reduce([
            (c[:,0]==1)&np.all(c[:,1:]==0,axis=1) for c in coefficients.values()])
        for out in stores.values():
            for key in mc_keys:
                out[key][globally_core]=saved[key][globally_core]
    normal_rng=np.random.default_rng(seed)
    radial_rng=np.random.default_rng(seed+RADIUS_SEED_OFFSET)
    for rng_start in range(0,n,16):
        rng_end=min(rng_start+16,n)
        normal=normal_rng.normal(size=(samples,rng_end-rng_start,d))
        mix=radial_rng.random((samples,rng_end-rng_start))
        kernel=radial_rng.random((samples,rng_end-rng_start))
        active=np.flatnonzero(~globally_core[rng_start:rng_end])+rng_start
        for block in range(0,len(active),query_block_size):
            rows=active[block:block+query_block_size]
            part=rows-rng_start
            coordinates=[];marginals=[];gammas=[];cosines=[]
            for w in weights:
                error=draw_radial(law,w[rows],scatter[rows],normal[:,part],mix[:,part],kernel[:,part])
                u=mean[None,rows]+error
                gamma,observables,differences,cosine=observable_forward(u*raw_scale+raw_center)
                absolute=np.log1p(differences*norm2_per_feature[None,rows,None])
                coordinates.append(u)
                marginals.append(np.concatenate((gamma[...,None],observables,absolute),axis=-1))
                gammas.append(gamma);cosines.append(cosine)
            truth=np.concatenate((actual[rows,None],obs_actual[rows],absolute_actual[rows]),axis=-1)
            crps_first,crps_pair=_crps_terms(marginals,truth)
            energy_first,energy_pair=_energy_terms(coordinates,target[rows])
            endpoint_means=np.stack([gamma.mean(0) for gamma in gammas])
            endpoint_probs=np.stack([(gamma<=0).mean(0) for gamma in gammas])
            endpoint_cos=np.stack([cosine.mean(0) for cosine in cosines])
            for arm,full_coeff in coefficients.items():
                c=full_coeff[rows]
                out=stores[arm]
                scores=_combine(crps_first,crps_pair,c)
                out['crps'][rows]=scores[:,0]
                crps=scores[:,1:1+len(OBSERVABLES)]
                out['observable_crps'][rows]=crps
                for prefix,cols in (('single',slice(0,3)),('pair',slice(3,6)),
                                    ('average',slice(6,10)),('triple_average',slice(9,10))):
                    out[prefix+'_crps'][rows]=crps[:,cols].mean(1)
                absolute_crps=scores[:,1+len(OBSERVABLES):]
                out['absolute_crps_by_pair'][rows]=absolute_crps
                out['absolute_pair_crps'][rows]=absolute_crps.mean(1)
                out['energy'][rows]=_combine(energy_first,energy_pair,c)
                out['predicted'][rows]=np.einsum('nk,kn->n',c,endpoint_means)
                out['p_null'][rows]=np.einsum('nk,kn->n',c,endpoint_probs)
                cosine=np.einsum('nk,knj->nj',c,endpoint_cos)
                out['predicted_cos_Z1_V'][rows]=cosine[:,0]
                out['predicted_cos_Z2_V'][rows]=cosine[:,1]
                integrated_gamma=sum(c[None,:,j]*gamma for j,gamma in enumerate(gammas))
                integrated_null=sum(c[None,:,j]*(gamma<=0) for j,gamma in enumerate(gammas))
                out['gamma_mc_se'][rows]=integrated_gamma.std(0,ddof=1)/np.sqrt(samples)
                out['null_mc_se'][rows]=integrated_null.std(0,ddof=1)/np.sqrt(samples)
    for arm,out in stores.items():
        out['brier']=np.square(out['p_null']-(actual<=0))
        if baseline_saved is not None:
            core_only=(coefficients[arm][:,0]==1)&np.all(coefficients[arm][:,1:]==0,axis=1)
            for key,value in saved.items():
                out[key][core_only]=value[core_only]
            if 'brier' not in saved:
                out['brier']=np.square(out['p_null']-(actual<=0))
        if not all(np.isfinite(v).all() for v in out.values()):
            raise ValueError('Nonfinite mixture outputs; no samples or queries are discarded')
    return stores
