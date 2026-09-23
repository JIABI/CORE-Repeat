"""Unchanged factor-forward draws with checks on the reported functionals.

A positive diagonal triangular factor defines a positive-definite matrix in
exact arithmetic. Re-factorizing its rounded product is not necessary to
evaluate forward geometry. We retain that diagnostic, verify every forward
quantity against virtual vectors, and audit failed refactorizations at high
precision. No sampled value, assembled Gram entry, or model law is replaced.
The older strict/invertibility decoder remains available and unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .gram_geometry import _gram, gram_gains, gram_observables, OBSERVABLE_NAMES
from .gram_forward_diagnostic import _virtual_functionals
from .gram_high_precision import verify_factor_draw


NUMERICAL_POLICY = dict(version='factor-forward-functional-verification-v1',
    gain_absolute_tolerance=1e-9, cosine_absolute_tolerance=1e-9,
    other_observable_absolute_tolerance=1e-9, other_observable_relative_tolerance=1e-10,
    high_precision_initial_decimal_digits=80, sampled_law_changed=False,
    factor_diagonal_square_underflow_rejected=True,
    jitter=0., clipped_draws=0, resampled_draws=0, omitted_draws=0)


def _forward(coordinates):
    u = torch.as_tensor(coordinates)
    if u.dtype != torch.float64 or u.ndim < 1 or u.shape[-1] != 9 or any(n == 0 for n in u.shape):
        raise ValueError('Verified forward geometry requires nonempty float64 [...,9] coordinates')
    if not bool(torch.isfinite(u).all()):
        raise ValueError('The sampled coordinates must remain finite')
    p, zero = u[..., :3], torch.zeros_like(u[..., 0])
    diagonal = u[..., (3, 5, 8)].exp()
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal <= 0).any()):
        raise ValueError('Factor diagonal overflowed or underflowed to zero')
    if bool((diagonal.square() == 0).any()):
        raise ValueError('A factor diagonal squared underflowed; no floor is added')
    factor = torch.stack((
        torch.stack((diagonal[..., 0], zero, zero), -1),
        torch.stack((u[..., 4], diagonal[..., 1], zero), -1),
        torch.stack((u[..., 6], u[..., 7], diagonal[..., 2]), -1)), -2)
    direct_h = factor@factor.transpose(-1, -2)
    if not bool(torch.isfinite(direct_h).all()):
        raise ValueError('The direct factor product overflowed')
    direct_chol, direct_info = torch.linalg.cholesky_ex(direct_h, check_errors=False)
    direct_failed = (direct_info != 0) | ~torch.isfinite(direct_chol).all(dim=(-2, -1))
    # These operations are intentionally identical to the previous decoder.
    future = p.unsqueeze(-1)*p.unsqueeze(-2)+direct_h
    top = torch.cat((torch.ones_like(p[..., :1]), p), -1).unsqueeze(-2)
    bottom = torch.cat((p.unsqueeze(-1), future), -1)
    gram = torch.cat((top, bottom), -2)
    if not bool(torch.isfinite(gram).all()):
        raise ValueError('The assembled Gram overflowed')
    _gram(gram, normalized=True)  # Same existing roundoff-only whole-Gram PSD standard.
    context = torch.stack((torch.ones_like(zero), zero, zero, zero), -1).unsqueeze(-2)
    virtual_rows = torch.cat((context, torch.cat((p.unsqueeze(-1), factor), -1)), -2)
    return u, p, factor, direct_h, future, gram, virtual_rows, direct_failed, direct_info


def _check_functionals(gram, rows):
    reference_observables, reference_gains, acquired_norms = _virtual_functionals(rows, .01, .02)
    gains, observables = gram_gains(gram), gram_observables(gram)
    gain_error = (gains-reference_gains).abs()
    observable_error = (observables-reference_observables).abs()
    tolerance = 1e-9+1e-10*reference_observables.abs()
    tolerance[..., :6] = 1e-9
    if not bool(torch.isfinite(gain_error).all()) or bool((gain_error > 1e-9).any()):
        raise ValueError('Forward Gamma differs from direct virtual-vector evaluation beyond 1e-9')
    if not bool(torch.isfinite(observable_error).all()) or bool((observable_error > tolerance).any()):
        raise ValueError('Forward observable exceeds the predeclared absolute/relative numerical tolerance')
    audit = dict(every_draw_checked=True, gains_max_absolute_error=float(gain_error.max()),
        observables_max_absolute_error=float(observable_error.max()),
        observables_max_scaled_error=float((observable_error/tolerance).max()),
        observables_max_absolute_error_by_name=dict(zip(OBSERVABLE_NAMES,
            observable_error.reshape(-1, 20).amax(0).tolist())),
        minimum_virtual_acquired_average_squared_norm=float(acquired_norms.min()))
    return gains, observables, audit


def decode_draws(raw_u, *, verify=True):
    """Return the original assembled Gram values and a complete numerical audit.

    Verification cannot be disabled in this continuation. The seed, sampled u,
    exp(u) factors and final float64 Gram are not changed by high precision.
    """
    if verify is not True:
        raise ValueError('Functional verification is mandatory for the repaired decoder')
    u, p, factor, h, future, gram, rows, failed, info = _forward(raw_u)
    gains, observables, functional = _check_functionals(gram, rows)
    indices = torch.nonzero(failed, as_tuple=False).detach().cpu().tolist()
    high_precision = []
    for index in indices:
        key = tuple(index)
        checked = verify_factor_draw(p[key].detach().cpu().numpy(),
            factor[key].detach().cpu().numpy(), gram[key].detach().cpu().numpy(),
            gains[key].detach().cpu().numpy(), observables[key].detach().cpu().numpy())
        high_precision.append(dict(index=index, **checked))
    recovered = future-p.unsqueeze(-1)*p.unsqueeze(-2)
    recovered_chol, recovered_info = torch.linalg.cholesky_ex(
        (recovered+recovered.transpose(-1, -2))/2, check_errors=False)
    recovered_failed = (recovered_info != 0) | ~torch.isfinite(recovered_chol).all(dim=(-2, -1))
    audit = dict(numerical_policy=NUMERICAL_POLICY, leading_shape=list(u.shape[:-1]),
        draw_object_count=math.prod(u.shape[:-1]), original_gram_assembly_unchanged=True,
        direct_H_cholesky_passed=not bool(failed.any()),
        direct_H_refactorization_failure_count=int(failed.sum()),
        direct_H_failed_indices=indices,
        direct_H_failure_counts_per_object=(failed.sum(0).tolist() if failed.ndim==2 else None),
        direct_H_failed_info=info[failed].tolist(),
        failed_factor_draws_high_precision_verified=len(high_precision),
        high_precision_audits=high_precision, high_precision_used_to_modify_predictions=False,
        recovered_schur_failure_count=int(recovered_failed.sum()),
        recovered_schur_failed_indices=torch.nonzero(recovered_failed, as_tuple=False).tolist(),
        recovered_vs_direct_H_max_absolute_difference=float((recovered-h).abs().max()),
        whole_Gram_passed_existing_PSD_standard=True, functional_consistency=functional,
        acceptance_change='rounded direct-H refactorization is diagnostic; positive factors and forward/high-precision checks govern evaluation',
        old_strict_decoder_source_changed=False, old_failure_record_reclassified=False,
        sampled_distribution_changed=False, formal_certificate=False,
        rows_dropped=0, draws_resampled=0, jitter_added=0., diagonal_floor_added=0.)
    return gram.detach().cpu().numpy(), audit
