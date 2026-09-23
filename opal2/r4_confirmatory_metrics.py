"""Finite-campaign R4 arithmetic; no data access, fitting, or certification.

Gamma already deducts the ADD_TWO action cost. Missing outcomes remain bounded
unknowns. These identification bounds are not sampling confidence intervals.
The existing frozen_acquisition_policy module remains the selection engine.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np

GAMMA_LOWER = -1.02
GAMMA_UPPER = 0.98


def campaign_budget(population_n: int, eligible_x_n: int | None = None) -> dict:
    """Draft .25 added-well budget; unavailable X does not reduce population N."""
    for name, value in (("population_n", population_n), ("eligible_x_n", eligible_x_n)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral)):
            raise ValueError(f"{name} must be an integer")
    if population_n <= 0:
        raise ValueError("A nonempty metadata-qualified population is required")
    e = population_n if eligible_x_n is None else eligible_x_n
    if not 0 <= e <= population_n:
        raise ValueError("Decision-eligible count must lie in [0, N]")
    wells = int(population_n) // 4
    target = wells // 2
    k = min(target, int(e))
    return dict(population_n=int(population_n), eligible_x_n=int(e),
                extra_well_budget=wells, target_activations=target,
                activations=k, added_action_wells=2*k,
                unused_action_wells=wells-2*k)


def _arrays(gamma, selected):
    y = np.asarray(gamma, dtype=float)
    a = np.asarray(selected)
    if y.ndim != 1 or not len(y) or a.shape != y.shape or a.dtype != np.bool_:
        raise ValueError("Aligned nonempty Gamma and Boolean selection arrays required")
    if np.isinf(y).any():
        raise ValueError("Use NaN for missing Gamma; infinity is not a valid outcome")
    known = np.isfinite(y)
    if np.any((y[known] < GAMMA_LOWER-1e-12) | (y[known] > GAMMA_UPPER+1e-12)):
        raise ValueError("Gamma is outside the declared original ADD_TWO range")
    return y, a, known


def utility_bounds(gamma, selected, *, additional_reference_cost: float = 0.) -> dict:
    """Value per metadata-qualified candidate, not per selected object."""
    y, a, known = _arrays(gamma, selected)
    cost = float(additional_reference_cost)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("Incremental reference cost must be finite and nonnegative")
    observed = float(y[a & known].sum())
    missing = int((a & ~known).sum())
    return dict(lower=(observed + missing*GAMMA_LOWER-cost)/len(y),
                upper=(observed + missing*GAMMA_UPPER-cost)/len(y),
                population_n=len(y), selected_n=int(a.sum()),
                missing_selected_n=missing, additional_reference_cost=cost,
                action_cost_already_in_gamma=True,
                bound_type="finite_campaign_missing_outcome_identification")


def paired_utility_bounds(gamma, selected_a, selected_b, *, reference_cost_a=0.,
                          reference_cost_b=0.) -> dict:
    """Sharp bounded-outcome interval for A minus B, cancelling shared actions.

    Subtracting the two policies' separate worst-case completed means would
    not give a valid worst-case bound on their paired difference.
    """
    y, a, known = _arrays(gamma, selected_a)
    _, b, _ = _arrays(gamma, selected_b)
    ca, cb = float(reference_cost_a), float(reference_cost_b)
    if any(not math.isfinite(v) or v < 0 for v in (ca, cb)):
        raise ValueError("Reference costs must be finite and nonnegative")
    w = a.astype(int)-b.astype(int)
    observed = float(np.dot(w[known], y[known]))-ca+cb
    unknown_w = w[~known]
    lower = observed+float(np.where(unknown_w >= 0,
        unknown_w*GAMMA_LOWER, unknown_w*GAMMA_UPPER).sum())
    upper = observed+float(np.where(unknown_w >= 0,
        unknown_w*GAMMA_UPPER, unknown_w*GAMMA_LOWER).sum())
    return dict(lower=lower/len(y), upper=upper/len(y), population_n=len(y),
                unresolved_discordant_n=int(np.count_nonzero(unknown_w)),
                bound_type="paired_finite_campaign_missing_outcome_identification")


def _ratio_bounds(selected_known, unselected_known, selected_unknown, unselected_unknown):
    # Corners attain extrema where the ratio is defined. Some completions may
    # contain no members of the event class; report that separately, never 0.
    values = []
    for xs in (0, selected_unknown):
        for xu in (0, unselected_unknown):
            denominator = selected_known+unselected_known+xs+xu
            if denominator:
                values.append((selected_known+xs)/denominator)
    return dict(lower=min(values) if values else None,
                upper=max(values) if values else None,
                undefined_completion_possible=selected_known+unselected_known == 0)


def risk_bounds(gamma, selected) -> dict:
    """FDP, FPR and sensitivity over possible labels of missing outcomes.

    FPR denominator is all truly NULL candidates, NOT all candidates. Coding
    every missing outcome as NULL need not maximize this ratio.
    """
    y, a, known = _arrays(gamma, selected)
    null = known & (y <= 0)
    positive = known & (y > 0)
    sn, un = int((a & null).sum()), int((~a & null).sum())
    sp, up = int((a & positive).sum()), int((~a & positive).sum())
    sm, um = int((a & ~known).sum()), int((~a & ~known).sum())
    k = int(a.sum())
    return dict(selected_n=k, observed_selected_null=sn, observed_unselected_null=un,
                selected_unknown=sm, unselected_unknown=um,
                fdp=dict(lower=sn/k if k else None, upper=(sn+sm)/k if k else None),
                fpr=_ratio_bounds(sn, un, sm, um),
                sensitivity=_ratio_bounds(sp, up, sm, um),
                bound_type="finite_campaign_missing_label_identification",
                iid_binomial_certificate=False)
