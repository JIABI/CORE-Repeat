"""Deterministic bounded distributions reconstructed from conditional quantiles.

Every row defines a quantile function Q(p), linear in probability between its
knots. Flat portions are retained as point masses. Moments, CDF values and CRPS
are evaluated from that same law; none uses Monte Carlo sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GRID = np.array([
    .01, .025, .05, .10, .15, .20, .25, .35, .45, .50,
    .55, .65, .75, .80, .85, .90, .95, .975, .99,
], dtype=float)
DEFAULT_BOUNDS = (-1.02, .98)


def _levels(levels):
    p = np.asarray(levels, dtype=float)
    if (p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all()
            or p[0] <= 0 or p[-1] >= 1 or np.any(np.diff(p) <= 0)):
        raise ValueError('Quantile levels must increase strictly inside (0, 1)')
    return p


def _predictions(predictions, count):
    q = np.asarray(predictions, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    if (q.ndim != 2 or q.shape[1] != count or len(q) == 0
            or not np.isfinite(q).all()):
        raise ValueError('Predictions must be finite nonempty [N, number of levels]')
    return q


def _aligned(values, count, name, allow_infinite=False):
    x = np.asarray(values, dtype=float)
    if x.ndim == 0:
        x = np.full(count, float(x))
    if x.shape != (count,) or np.isnan(x).any():
        raise ValueError(f'{name} must be a scalar or an aligned [N] vector')
    if not allow_infinite and not np.isfinite(x).all():
        raise ValueError(f'{name} must be finite')
    return x


@dataclass(frozen=True)
class QuantileLaw:
    """One reconstructed distribution per row, including p=0 and p=1 knots."""

    probabilities: np.ndarray
    quantiles: np.ndarray
    bounds: tuple[float, float]

    def mean(self):
        """Exact E[Y] = integral_0^1 Q(p) dp, returned as an [N] vector."""
        return np.sum(
            .5 * (self.quantiles[:, :-1] + self.quantiles[:, 1:])
            * np.diff(self.probabilities)[None, :], axis=1)

    def quantile(self, probabilities):
        """Q(p): scalar p returns [N], a probability vector returns [N, K]."""
        p = np.asarray(probabilities, dtype=float)
        scalar = p.ndim == 0
        if scalar:
            p = p.reshape(1)
        if (p.ndim != 1 or not np.isfinite(p).all()
                or np.any((p < 0) | (p > 1))):
            raise ValueError('Probabilities must be a scalar or vector in [0, 1]')
        j = np.clip(np.searchsorted(self.probabilities, p, side='right') - 1,
                    0, len(self.probabilities) - 2)
        t = ((p - self.probabilities[j])
             / (self.probabilities[j + 1] - self.probabilities[j]))
        values = (self.quantiles[:, j]
                  + (self.quantiles[:, j + 1] - self.quantiles[:, j]) * t)
        return values[:, 0] if scalar else values

    def cdf(self, threshold):
        """Right-continuous P(Y <= threshold), including mass at the threshold."""
        x = _aligned(threshold, len(self.quantiles), 'Threshold', allow_infinite=True)
        # Rightmost knot <= x traverses an entire plateau, including its atom.
        j = np.sum(self.quantiles <= x[:, None], axis=1) - 1
        below = j < 0
        above = j >= self.quantiles.shape[1] - 1
        interior = ~(below | above)
        out = np.zeros(len(x), dtype=float)
        out[above] = 1.
        rows = np.flatnonzero(interior)
        cols = j[interior]
        if len(rows):
            left = self.quantiles[rows, cols]
            right = self.quantiles[rows, cols + 1]
            fraction = (x[rows] - left) / (right - left)
            out[rows] = (self.probabilities[cols]
                         + fraction * (self.probabilities[cols + 1]
                                       - self.probabilities[cols]))
        return np.clip(out, 0., 1.)

    def crps(self, outcomes):
        """Exact 2*integral pinball_p(y-Q(p)) dp for every observed outcome.

        Each segment is split where Q(p)=y. On either side the integrand is
        quadratic in the segment-local probability coordinate and is integrated
        analytically. Plateaus need no density or division by a quantile slope.
        """
        y = _aligned(outcomes, len(self.quantiles), 'Outcomes')
        a = self.probabilities[:-1][None, :]
        h = np.diff(self.probabilities)[None, :]
        q0 = self.quantiles[:, :-1]
        d = self.quantiles[:, 1:] - q0
        r0 = y[:, None] - q0
        t = np.divide(r0, d, out=np.where(r0 >= 0, 1., 0.), where=d > 0)
        t = np.clip(t, 0., 1.)

        def primitive(c, stop):
            return (c * r0 * stop + (h * r0 - c * d) * stop**2 / 2
                    - h * d * stop**3 / 3)

        positive = primitive(a, t)
        negative = primitive(a - 1., 1.) - primitive(a - 1., t)
        score = 2 * np.sum(h * (positive + negative), axis=1)
        # Only protects roundoff at exactly degenerate/zero-score cases.
        return np.maximum(score, 0.)

    def interval_metrics(self, outcomes, coverages=(.5, .8, .9, .95, .99)):
        """Central intervals; returned lower/upper/width/covered arrays are [N,K]."""
        y = _aligned(outcomes, len(self.quantiles), 'Outcomes')
        nominal = np.asarray(coverages, dtype=float)
        if (nominal.ndim != 1 or not len(nominal) or not np.isfinite(nominal).all()
                or np.any((nominal <= 0) | (nominal >= 1))):
            raise ValueError('Coverages must be a nonempty vector inside (0, 1)')
        lower = self.quantile((1 - nominal) / 2)
        upper = self.quantile((1 + nominal) / 2)
        return dict(nominal=nominal.copy(), lower=lower, upper=upper,
                    width=upper - lower,
                    covered=(y[:, None] >= lower) & (y[:, None] <= upper))

    def metadata(self):
        return dict(
            interior_quantile_levels=self.probabilities[1:-1].tolist(),
            bounds=list(self.bounds),
            crossing_rule='sort each row in increasing order, then clip to physical bounds',
            tail_rule='linear quantile extrapolation from adjacent extreme knots, bounded',
            interpolation='piecewise linear quantile function; retain flat-segment atoms',
            crps='exact analytic integrated pinball loss, split at observed outcome',
            cdf='right-continuous, includes atoms at threshold',
            monte_carlo_samples=0)


def make_quantile_law(raw_quantiles, levels=GRID, bounds=DEFAULT_BOUNDS):
    """Rearrange predictions and append bounded, linearly extrapolated tails."""
    p = _levels(levels)
    raw = _predictions(raw_quantiles, len(p))
    b = np.asarray(bounds, dtype=float)
    if b.shape != (2,) or not np.isfinite(b).all() or b[0] >= b[1]:
        raise ValueError('Bounds must be a finite increasing pair')
    q = np.clip(np.sort(raw, axis=1), b[0], b[1])
    lower = q[:, 0] - p[0] * (q[:, 1] - q[:, 0]) / (p[1] - p[0])
    upper = q[:, -1] + (1 - p[-1]) * (q[:, -1] - q[:, -2]) / (p[-1] - p[-2])
    knots = np.column_stack([np.clip(lower, b[0], b[1]), q,
                             np.clip(upper, b[0], b[1])])
    return QuantileLaw(np.r_[0., p, 1.], knots, (float(b[0]), float(b[1])))


def fit_quantile_offsets(cal_predictions, cal_outcomes, levels=GRID):
    """CAL-only offsets: empirical level-p quantile of y - q_p(x).

    Inputs are unrearranged predictions, as supplied to make_quantile_law.
    Apply the returned vector to new unrearranged predictions, then reconstruct
    the bounded monotone law. Empirical quantiles use NumPy's ``linear`` method;
    these offsets are a fitted correction, not a finite-sample coverage claim.
    """
    p = _levels(levels)
    q = _predictions(cal_predictions, len(p))
    y = _aligned(cal_outcomes, len(q), 'Calibration outcomes')
    return np.array([np.quantile(y - q[:, j], alpha, method='linear')
                     for j, alpha in enumerate(p)], dtype=float)
