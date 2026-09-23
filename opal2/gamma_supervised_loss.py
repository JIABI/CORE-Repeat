"""Differentiable original ADD_TWO utility from one joint Gram distribution.

The Monte Carlo CRPS estimator requires independent standard-normal arrays A
and B, also independent across the pair axis. Reusing the same draws, pairing
antithetic draws, or otherwise coupling A and B biases the absolute-pair term.
The caller owns those random streams; this module never draws randomness.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _floating_finite(value,name):
    if not isinstance(value,torch.Tensor) or value.dtype not in (torch.float32,torch.float64):
        raise ValueError(name+' must be a real floating tensor')
    if not torch.isfinite(value).all():raise ValueError(name+' must be finite')
    return value


def gamma_from_raw_coordinates(raw_u,*,cost_two=.02):
    """Map [...,9] native log-Cholesky coordinates to original ADD_TWO Gamma.

    X=(1,0,0,0), while future rows are (p_j,L_j). Positive diagonal L makes
    these a legal joint Gram geometry. This direct factor calculation avoids
    subtraction of nearly equal Schur products; it does not clip cosines,
    jitter factors, change zero norms, or detach the differentiation graph.
    """
    u=_floating_finite(raw_u,'Native Gram coordinates')
    if u.ndim<1 or u.shape[-1]!=9 or u.numel()==0:
        raise ValueError('Native Gram coordinates must have nonempty shape [...,9]')
    if not math.isfinite(cost_two) or cost_two<0:
        raise ValueError('The declared ADD_TWO cost must be finite and nonnegative')
    diagonal=torch.exp(u[..., [3,5,8]])
    if not torch.isfinite(diagonal).all() or not (diagonal>0).all():
        raise FloatingPointError('Log-Cholesky exponent overflow/underflow; no draw is clipped or replaced')
    if (diagonal.square()==0).any():
        raise FloatingPointError('A factor diagonal squared underflowed; no floor is added')
    zero=torch.zeros_like(u[...,0]);one=torch.ones_like(zero)
    x=torch.stack((one,zero,zero,zero),-1)
    z1=torch.stack((u[...,0],diagonal[...,0],zero,zero),-1)
    z2=torch.stack((u[...,1],u[...,4],diagonal[...,1],zero),-1)
    v=torch.stack((u[...,2],u[...,6],u[...,7],diagonal[...,2]),-1)
    average=(x+z1+z2)/3
    norm_average=torch.linalg.vector_norm(average,dim=-1)
    norm_v=torch.linalg.vector_norm(v,dim=-1)
    if (not torch.isfinite(norm_average).all() or not torch.isfinite(norm_v).all()
            or not (norm_average>0).all() or not (norm_v>0).all()):
        raise FloatingPointError('A virtual-well norm is zero/nonfinite; no epsilon is inserted')
    # Normalize before multiplication so large valid projection coefficients do
    # not overflow merely when taking an unnormalized dot product.
    cosine_after=((average/norm_average[...,None])*(v/norm_v[...,None])).sum(-1)
    cosine_before=v[...,0]/norm_v
    gamma=.5*(cosine_after-cosine_before)-cost_two
    if not torch.isfinite(gamma).all():
        raise FloatingPointError('The original ADD_TWO functional is nonfinite')
    return gamma


class JointGammaCRPS(nn.Module):
    """Frozen full-joint error law with gradients only through predicted means.

    Constructor buffers use float64. ``target_center`` and ``target_scale``
    restore the original native u coordinates, and ``gamma_scale`` is the
    strictly positive fitting-only standard deviation of original Gamma.
    ``forward`` receives mean_u[B,9], actual_gamma[B], and two *independent*
    epsilon[S,B,9] arrays. The returned CRPS averages iid-pair estimates over
    S and B; its normalized value divides by gamma_scale, not its square.
    """
    def __init__(self,covariance,target_center,target_scale,gamma_scale,*,cost_two=.02):
        super().__init__()
        covariance=torch.as_tensor(covariance,dtype=torch.float64).detach().clone()
        if covariance.shape!=(9,9) or not torch.isfinite(covariance).all():
            raise ValueError('A finite full 9-by-9 covariance is required')
        if not torch.equal(covariance,covariance.T):
            raise ValueError('The frozen covariance must be symmetric; no silent symmetrization is applied')
        try:factor=torch.linalg.cholesky(covariance)
        except RuntimeError as exc:raise ValueError('The frozen covariance must be positive definite without jitter') from exc
        center=torch.as_tensor(target_center,dtype=torch.float64,device=covariance.device).detach().clone()
        scale=torch.as_tensor(target_scale,dtype=torch.float64,device=covariance.device).detach().clone()
        normalizer=torch.as_tensor(gamma_scale,dtype=torch.float64,device=covariance.device).detach().clone()
        if center.shape!=(9,) or not torch.isfinite(center).all():
            raise ValueError('Target center must contain nine finite native coordinates')
        if scale.shape!=(9,) or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('Target scale must contain nine positive finite values')
        if normalizer.ndim!=0 or not torch.isfinite(normalizer) or normalizer<=0:
            raise ValueError('TRAIN Gamma scale must be a positive finite scalar')
        if not math.isfinite(cost_two) or cost_two<0:
            raise ValueError('The declared ADD_TWO cost must be finite and nonnegative')
        self.register_buffer('covariance',covariance)
        self.register_buffer('scale_tril',factor)
        self.register_buffer('target_center',center)
        self.register_buffer('target_scale',scale)
        self.register_buffer('gamma_scale',normalizer)
        self.register_buffer('cost_two',normalizer.new_tensor(cost_two))

    def _check_mean(self,mean_u):
        _floating_finite(mean_u,'Standardized predictive mean')
        if mean_u.ndim!=2 or mean_u.shape[1]!=9 or len(mean_u)==0:
            raise ValueError('Standardized predictive mean must have nonempty shape [B,9]')
        if mean_u.dtype!=self.covariance.dtype or mean_u.device!=self.covariance.device:
            raise ValueError('Predictive inputs must match the frozen buffer dtype and device')

    def sample_standardized_coordinates(self,mean_u,epsilon):
        """Deterministic full-covariance reparameterization; no independent heads."""
        self._check_mean(mean_u);_floating_finite(epsilon,'Joint standard-normal inputs')
        if epsilon.ndim!=3 or epsilon.shape[1:]!=mean_u.shape or epsilon.shape[0]==0:
            raise ValueError('Joint standard-normal inputs must have shape [S,B,9]')
        if epsilon.dtype!=mean_u.dtype or epsilon.device!=mean_u.device:
            raise ValueError('Standard-normal inputs must match mean dtype and device')
        sample=mean_u[None]+epsilon@self.scale_tril.T
        if not torch.isfinite(sample).all():raise FloatingPointError('Joint standardized coordinates overflowed')
        return sample

    def forward(self,mean_u,actual_gamma,epsilon_a,epsilon_b):
        self._check_mean(mean_u);_floating_finite(actual_gamma,'Realized Gamma targets')
        if (actual_gamma.shape!=(len(mean_u),) or actual_gamma.dtype!=mean_u.dtype
                or actual_gamma.device!=mean_u.device):
            raise ValueError('Realized Gamma targets must have matching shape [B], dtype and device')
        a=self.sample_standardized_coordinates(mean_u,epsilon_a)
        b=self.sample_standardized_coordinates(mean_u,epsilon_b)
        if a.shape!=b.shape:raise ValueError('Both independent Monte Carlo arrays require the same shape')
        native_a=a*self.target_scale+self.target_center
        native_b=b*self.target_scale+self.target_center
        ga=gamma_from_raw_coordinates(native_a,cost_two=float(self.cost_two))
        gb=gamma_from_raw_coordinates(native_b,cost_two=float(self.cost_two))
        first=.5*((ga-actual_gamma[None]).abs()+(gb-actual_gamma[None]).abs())
        second=.5*(ga-gb).abs()
        per_object=(first-second).mean(0)
        crps=per_object.mean()
        if not torch.isfinite(crps):raise FloatingPointError('The joint Gamma CRPS is nonfinite')
        prediction_mean=.5*(ga.mean(0)+gb.mean(0))
        return dict(gamma_crps=crps,normalized_gamma_crps=crps/self.gamma_scale,
            gamma_crps_per_object=per_object,normalized_gamma_crps_per_object=per_object/self.gamma_scale,
            gamma_prediction_mean=prediction_mean,
            gamma_prediction_mse=(prediction_mean-actual_gamma).square().mean(),
            actual_gamma_mean=actual_gamma.mean(),
            gamma_samples_a=ga,gamma_samples_b=gb,
            mean_absolute_error_term=first.mean(),half_independent_pair_distance=second.mean(),
            finite=True,independent_pair_count=int(ga.shape[0]))
