"""Differentiable proper scores for the declared measurement-derived utility.

Scores use the model's joint future-well distribution in fixed measurement
coordinates. They do not introduce separate gain or classification heads.
"""
from __future__ import annotations

import torch


def fair_crps(samples: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Unbiased ensemble CRPS for IID draws on the first axis, per endpoint.

E|G-y| - .5 E|G-G'| uses distinct draw pairs in the second term. Draws may
share environmental factors across compounds and across actions; it is only
the Monte Carlo draw axis that must be independent. Unlike the empirical-CDF
V-statistic, this estimator does not reward finite-ensemble underdispersion.
"""
    if samples.ndim < 1 or samples.shape[0] < 2 or samples.shape[1:] != observed.shape:
        raise ValueError("Expected at least two draws [S,...] and observed [...]")
    if not torch.isfinite(samples).all() or not torch.isfinite(observed).all():
        raise ValueError("CRPS requires finite measured and sampled endpoints")
    count=samples.shape[0]
    # Sum i<j |g_i-g_j| in O(S log S), differentiable almost everywhere.
    ordered=samples.sort(dim=0).values
    coefficient=(2*torch.arange(count,device=samples.device,dtype=samples.dtype)-count+1)
    coefficient=coefficient.reshape((count,)+(1,)*observed.ndim)
    between=(ordered*coefficient).sum(0)/(count*(count-1))
    return (samples-observed.unsqueeze(0)).abs().mean(0)-between


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    numerator=(a*b).sum(-1)
    denominator=a.norm(dim=-1)*b.norm(dim=-1)
    safe=torch.where(denominator>0,denominator,torch.ones_like(denominator))
    return torch.where(denominator>0,numerator/safe,torch.zeros_like(numerator)).clamp(-1,1)


def half_cosine_gains(known_x: torch.Tensor, future: torch.Tensor,
                      cost_per_well: float = 0.01) -> torch.Tensor:
    """Fixed X -> (Z1,Z2,V): return ADD_Z1, ADD_Z2, ADD_TWO, in that order.

known_x is [B,D], future [S,B,3,D]. Acquisition and validation wells are
jointly sampled by the caller, including all shared latent noise factors.
"""
    if (known_x.ndim != 2 or future.ndim != 4 or future.shape[2] != 3
            or future.shape[1] != known_x.shape[0] or future.shape[-1] != known_x.shape[-1]):
        raise ValueError("Expected X [B,D] and jointly drawn Z1,Z2,V [S,B,3,D]")
    if cost_per_well != 0.01:
        raise ValueError("This training comparison fixes original acquisition cost to 0.01 per well")
    v=future[:,:,2]
    x=known_x.unsqueeze(0)
    before=_cosine(x,v)
    one0=.5*(_cosine((x+future[:,:,0])/2,v)-before)-cost_per_well
    one1=.5*(_cosine((x+future[:,:,1])/2,v)-before)-cost_per_well
    two=.5*(_cosine((x+future[:,:,0]+future[:,:,1])/3,v)-before)-2*cost_per_well
    return torch.stack((one0,one1,two),-1)


def derived_utility_crps(model, inputs, target_y, target_mask=None, *, samples=16,
                         chunk_size=8, generator=None):
    """Proper score of original utilities, using only a training episode's labels.

The model sees one X and declared future conditions, not target_y. Incomplete
training endpoints cannot supply this auxiliary score and are explicitly
counted; measured target coordinates still contribute to the NLL objective.
"""
    if not isinstance(samples,int) or isinstance(samples,bool) or samples<2:
        raise ValueError("At least two draws are required")
    if not isinstance(chunk_size,int) or isinstance(chunk_size,bool) or chunk_size<1:
        raise ValueError("Positive MC chunk size required")
    if inputs["context_y"].shape[1] != 1 or target_y.shape[1] != 3:
        raise ValueError("Utility training requires fixed X0 and three future roles")
    if inputs["target_cond"].shape[1] != 3:
        raise ValueError("Future conditions must cover Z1,Z2,V")
    context=inputs["context_y"][:,0]
    valid=inputs["context_mask"][:,0] & torch.isfinite(context).all(-1)
    valid=valid & torch.isfinite(target_y).all((-1,-2))
    if target_mask is not None:
        if target_mask.shape == target_y.shape[:2]:
            valid=valid & target_mask.all(-1)
        elif target_mask.shape == target_y.shape:
            valid=valid & target_mask.all((-1,-2))
        else:
            raise ValueError("Target mask must identify measured wells or coordinates")
    if not valid.any():
        raise ValueError("No complete fixed-role training endpoints for utility CRPS")
    fixed=lambda value: value*model.outcome_scale+model.outcome_center
    observed=half_cosine_gains(fixed(context[valid]),fixed(target_y[valid]).unsqueeze(0))[0].detach()
    distribution=model(inputs)
    draws=[]
    for start in range(0,samples,chunk_size):
        # Each chunk contains complete joint draws over every well/compound.
        # Across chunks MC draws are independent; within a draw sharing is
        # retained by JointGaussian.sample_joint, without a diagonal shortcut.
        future=distribution.sample_joint(min(chunk_size,samples-start),generator=generator)
        draws.append(half_cosine_gains(fixed(context[valid]),fixed(future[:,valid])))
    utility_samples=torch.cat(draws,0)
    score=fair_crps(utility_samples,observed)
    return {"utility_crps":score.mean(),"utility_crps_by_action":score.mean(0),
            "utility_scored_compounds":valid.sum(),"utility_unscored_compounds":(~valid).sum(),
            "observed_utility_mean":observed.mean(0),"predicted_utility_mean":utility_samples.mean((0,1))}
