"""Supplementary factor-forward checks, not a replacement production decoder.

The existing strict decoder also checks whether H can be recovered by
subtracting p pᵀ from the assembled future block. This independent diagnostic
keeps that failure visible, while checking the *directly constructed* H=L Lᵀ
and the forward utility separately. Cancellation in the subtraction need not
invalidate every forward functional, but this module does not turn an earlier
failed experiment or statistical certificate into a pass.

No clipping, diagonal floor, jitter, rejection sampling or row removal is used.
The four virtual rows [1,0,0,0] and [p,L] provide a separate vector-arithmetic
check on each observable and original half-cosine acquisition utility.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .gram_geometry import (
    _gram, ACTION_NAMES, COSINE_PAIRS, OBSERVABLE_NAMES,
    gram_gains, gram_observables,
)


def _assemble(coordinates):
    u = coordinates if torch.is_tensor(coordinates) else torch.as_tensor(coordinates)
    if u.dtype not in (torch.float32, torch.float64):
        raise ValueError("Forward diagnostic requires real float32 or float64 coordinates")
    if u.ndim < 1 or u.shape[-1] != 9 or any(size == 0 for size in u.shape):
        raise ValueError("Forward diagnostic requires nonempty [...,9] coordinates")
    if not bool(torch.isfinite(u).all()):
        raise ValueError("Forward diagnostic coordinates must be finite")
    p, zero = u[..., :3], torch.zeros_like(u[..., 0])
    diagonal = u[..., (3, 5, 8)].exp()
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal <= 0).any()):
        raise ValueError("Direct log-Cholesky diagonal overflowed or underflowed to zero")
    factor = torch.stack((
        torch.stack((diagonal[..., 0], zero, zero), -1),
        torch.stack((u[..., 4], diagonal[..., 1], zero), -1),
        torch.stack((u[..., 6], u[..., 7], diagonal[..., 2]), -1)), -2)
    direct_h = factor@factor.transpose(-1, -2)
    if not bool(torch.isfinite(direct_h).all()):
        raise ValueError("The direct H=L Lᵀ overflowed")
    direct_factor, direct_info = torch.linalg.cholesky_ex(direct_h, check_errors=False)
    if bool((direct_info != 0).any()) or not bool(torch.isfinite(direct_factor).all()):
        raise ValueError("The direct H=L Lᵀ is not numerically SPD; no underflow repair or jitter is permitted")
    future = p.unsqueeze(-1)*p.unsqueeze(-2)+direct_h
    top = torch.cat((torch.ones_like(p[..., :1]), p), -1).unsqueeze(-2)
    bottom = torch.cat((p.unsqueeze(-1), future), -1)
    gram = torch.cat((top, bottom), -2)
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("The forward Gram assembly overflowed")
    # Exactly the existing Gram validity standard, including its roundoff-only
    # PSD tolerance; do not silently introduce a different geometry criterion.
    _gram(gram, normalized=True)
    context = torch.stack((torch.ones_like(zero), zero, zero, zero), -1).unsqueeze(-2)
    virtual_rows = torch.cat((context, torch.cat((p.unsqueeze(-1), factor), -1)), -2)
    return u, p, direct_h, future, gram, virtual_rows


def coordinates_factor_forward(u):
    """Return (Gram, diagnostics) without rejecting recovered-H cancellation.

    G is assembled with exactly the same operations as the strict decoder.
    Finite factors, direct-H Cholesky and existing whole-Gram validity must all
    pass. A failed subtractive recovered-H Cholesky is instead recorded for
    every draw/object. Its NumPy boolean mask and explicit indices retain the
    caller's leading sample/object dimensions, with no objects removed.
    """
    u, p, direct_h, future, gram, _ = _assemble(u)
    recovered = future-p.unsqueeze(-1)*p.unsqueeze(-2)
    recovered = (recovered+recovered.transpose(-1, -2))/2
    recovered_factor, info = torch.linalg.cholesky_ex(recovered, check_errors=False)
    finite_factor = torch.isfinite(recovered_factor).all(dim=(-2, -1))
    failed = (info != 0) | ~finite_factor
    delta = (recovered-direct_h).abs()
    diagnostics = dict(
        purpose="supplementary equivalent factor-forward numerical diagnosis",
        primary_strict_decoder_changed=False, previous_failure_reclassified=False,
        formal_certificate=False, rows_dropped=0, draws_resampled=0,
        jitter_added=0., diagonal_floor_added=0.,
        leading_shape=list(u.shape[:-1]), draw_object_count=math.prod(u.shape[:-1]),
        direct_H_cholesky_passed=True, whole_Gram_passed_existing_PSD_standard=True,
        recovered_schur_failure_count=int(failed.sum().item()),
        recovered_schur_failed_mask=failed.detach().cpu().numpy().copy(),
        recovered_schur_failed_indices=torch.nonzero(failed, as_tuple=False).detach().cpu().tolist(),
        recovered_schur_info=info.detach().cpu().numpy().copy(),
        recovered_vs_direct_H_max_absolute_difference=float(delta.max().item()),
        forward_validity_is_not_roundtrip_invertibility=True,
    )
    return gram, diagnostics


def _positive(values, label):
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError(f"Virtual-row {label} has a zero, negative or nonfinite squared norm")


def _virtual_functionals(rows, cost_one, cost_two):
    """Direct vector arithmetic; does not form a second Gram matrix."""
    x, z1, z2, v = rows.unbind(-2)
    norm2 = rows.square().sum(-1)
    _positive(norm2, "well")
    cosines = torch.stack([(rows[..., i, :]*rows[..., j, :]).sum(-1)/(
        norm2[..., i]*norm2[..., j]).sqrt() for i, j in COSINE_PAIRS], -1)
    relative_norms = (norm2/norm2[..., :1]).sqrt()
    vectors = (z1-z2, z1-v, z2-v, (z1+z2)/2, (z1+v)/2, (z2+v)/2,
               (x+z1)/2, (x+z2)/2, (x+z1+z2)/3, (z1+z2+v)/3)
    energies = torch.stack([value.square().sum(-1)/norm2[..., 0]
                            for value in vectors], -1)
    observables = torch.cat((cosines, relative_norms, energies), -1)
    acquired = torch.stack(((x+z1)/2, (x+z2)/2, (x+z1+z2)/3), -2)
    acquired_norm2 = acquired.square().sum(-1)
    _positive(acquired_norm2, "acquired average")
    before = (x*v).sum(-1)/(norm2[..., 0]*norm2[..., 3]).sqrt()
    after = (acquired*v.unsqueeze(-2)).sum(-1)/(acquired_norm2*norm2[..., 3:]).sqrt()
    gains = .5*(after-before[..., None])-rows.new_tensor((cost_one, cost_one, cost_two))
    if not bool(torch.isfinite(observables).all()) or not bool(torch.isfinite(gains).all()):
        raise ValueError("Virtual-row observables or gains became nonfinite")
    return observables, gains, acquired_norm2


def factor_forward_consistency(u, gram=None, *, cost_one=.01, cost_two=.02):
    """Compare Gram formulas with explicit virtual-vector formulas, every draw.

    Maxima are errors, not pass thresholds. In extreme conditioning the Gram
    may lose small difference energies while preserving the action gains; both
    results must be reported. No failed/inconvenient cases are discarded.
    """
    if not math.isfinite(cost_one) or not math.isfinite(cost_two) or cost_one < 0 or cost_two < 0:
        raise ValueError("Action costs must be finite and nonnegative")
    u, _, _, _, assembled, virtual_rows = _assemble(u)
    supplied = assembled if gram is None else (
        gram if torch.is_tensor(gram) else torch.as_tensor(gram, dtype=u.dtype, device=u.device))
    if supplied.shape != assembled.shape or supplied.dtype != u.dtype or supplied.device != u.device:
        raise ValueError("Provided Gram must match the factor-forward leading shape, dtype and device")
    if not torch.equal(supplied, assembled):
        raise ValueError("The supplied Gram is not the unchanged factor-forward assembly")
    virtual_observables, virtual_gains, acquired_norm2 = _virtual_functionals(virtual_rows, cost_one, cost_two)
    actual_gains = gram_gains(supplied, cost_one=cost_one, cost_two=cost_two)
    actual_observables = gram_observables(supplied)
    gains_error = (actual_gains-virtual_gains).abs()
    observable_error = (actual_observables-virtual_observables).abs()
    gains_max = gains_error.reshape(-1, 3).amax(0).detach().cpu().numpy()
    observable_max = observable_error.reshape(-1, len(OBSERVABLE_NAMES)).amax(0).detach().cpu().numpy()
    return dict(
        reference="explicit four-dimensional virtual rows [1,0,0,0] and [p,L]; direct dot/norm/averaging arithmetic",
        reference_uses_profiles_to_gram=False, every_draw_checked=True,
        draw_object_count=math.prod(u.shape[:-1]), leading_shape=list(u.shape[:-1]),
        zero_virtual_acquired_average_count=int((acquired_norm2 <= 0).sum().item()),
        minimum_virtual_acquired_average_squared_norm=float(acquired_norm2.min().item()),
        gains_max_absolute_error=float(gains_max.max()),
        gains_max_absolute_error_by_action=dict(zip(ACTION_NAMES, map(float, gains_max))),
        observables_max_absolute_error=float(observable_max.max()),
        observables_max_absolute_error_by_name=dict(zip(OBSERVABLE_NAMES, map(float, observable_max))),
        gain_absolute_error_per_draw_object=gains_error.amax(-1).detach().cpu().numpy().copy(),
        observable_absolute_error_per_draw_object=observable_error.amax(-1).detach().cpu().numpy().copy(),
        automatic_acceptance_threshold_applied=False, primary_failure_reclassified=False,
        formal_certificate=False, rows_dropped=0, draws_resampled=0,
    )
