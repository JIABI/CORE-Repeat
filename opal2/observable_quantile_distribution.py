"""Conditional quantile laws for nonnegative measurement observables.

The shared QuantileLaw supplies exact moments, quantiles, CDFs and CRPS for
the reconstructed piecewise-linear quantile function. Unlike the Gamma law,
these observables have no finite physical upper bound. The extrapolated p=1
knot is a numerical tail approximation, not a bound on possible observations.
"""
from __future__ import annotations

import numpy as np

from .quantile_distribution import (
    GRID, QuantileLaw, _levels, _predictions, fit_quantile_offsets,
)


class NonnegativeQuantileLaw(QuantileLaw):
    """QuantileLaw with nonnegative support and no physical upper bound."""

    def metadata(self):
        return dict(
            interior_quantile_levels=self.probabilities[1:-1].tolist(),
            bounds=[0., None],
            upper_physical_bound=None,
            crossing_rule='sort each row in increasing order, then floor at zero',
            tail_rule=('linear quantile extrapolation from adjacent extreme '
                       'knots; lower endpoint floored at zero; upper endpoint '
                       'unclipped'),
            finite_upper_endpoint=('row-specific numerical tail approximation, '
                                   'not a physical upper bound'),
            interpolation='piecewise linear quantile function; retain flat-segment atoms',
            crps='exact analytic integrated pinball loss, split at observed outcome',
            cdf='right-continuous, includes atoms at threshold',
            monte_carlo_samples=0)


def make_nonnegative_quantile_law(raw_quantiles, levels=GRID):
    """Rearrange, floor at zero and append linearly extrapolated tail knots.

    CAL offsets from ``fit_quantile_offsets`` are added to unrearranged
    predictions before this constructor, exactly as for the existing Gamma
    law. No finite upper clipping is applied to interior or extrapolated knots.
    """
    p = _levels(levels)
    raw = _predictions(raw_quantiles, len(p))
    q = np.maximum(np.sort(raw, axis=1), 0.)
    lower = q[:, 0] - p[0] * (q[:, 1] - q[:, 0]) / (p[1] - p[0])
    upper = q[:, -1] + (1 - p[-1]) * (q[:, -1] - q[:, -2]) / (p[-1] - p[-2])
    knots = np.column_stack([np.maximum(lower, 0.), q, upper])
    if not np.isfinite(knots).all():
        raise ValueError('Extrapolated observable quantile knots must remain finite')
    return NonnegativeQuantileLaw(np.r_[0., p, 1.], knots, (0., np.inf))
