"""Audit unchanged float64 factor-forward draws using higher precision.

This module never resamples, changes a factor, or returns replacement model
outputs. It lifts the already generated binary floating-point p and L exactly,
without revisiting their log-coordinate construction. High precision is used
only to verify a draw whose rounded H=L L.T cannot be Cholesky-factorized.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np


COSINE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _array(value, shape, name):
    result = np.asarray(value)
    if result.dtype != np.float64 or result.shape != shape:
        raise ValueError(f"{name} must be float64 with shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _lift_matrix(array):
    # mp.mpf(float), unlike a short decimal rendering, preserves the exact
    # float64 value at the requested working precision (at least 30 digits).
    return mp.matrix([[mp.mpf(float(value)) for value in row] for row in array])


def _dot(left, right):
    return mp.fsum(a * b for a, b in zip(left, right))


def _average(*rows):
    return [mp.fsum(values) / len(rows) for values in zip(*rows)]


def _references(p, factor):
    """Independent direct-vector functionals, without a rounded Gram input."""
    rows = [[mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(0)]]
    rows += [[p[i]] + [factor[i, j] for j in range(3)] for i in range(3)]
    norms = [_dot(row, row) for row in rows]
    if any(value <= 0 or not mp.isfinite(value) for value in norms):
        raise ValueError("High-precision virtual well norms must be positive and finite")
    gram = [[_dot(left, right) for right in rows] for left in rows]
    cosines = [gram[i][j] / mp.sqrt(norms[i] * norms[j]) for i, j in COSINE_PAIRS]
    x, z1, z2, v = rows
    differences = [[a - b for a, b in zip(left, right)]
                   for left, right in ((z1, z2), (z1, v), (z2, v))]
    averages = [_average(z1, z2), _average(z1, v), _average(z2, v),
                _average(x, z1), _average(x, z2), _average(x, z1, z2),
                _average(z1, z2, v)]
    observables = cosines + [mp.sqrt(value / norms[0]) for value in norms]
    observables += [_dot(row, row) / norms[0] for row in differences + averages]
    before = _dot(x, v) / mp.sqrt(norms[0] * norms[3])
    costs = (mp.mpf("0.01"), mp.mpf("0.01"), mp.mpf("0.02"))
    gains = []
    for acquired, cost in zip((_average(x, z1), _average(x, z2),
                                _average(x, z1, z2)), costs):
        norm = _dot(acquired, acquired)
        if norm <= 0 or not mp.isfinite(norm):
            raise ValueError("High-precision acquired average must have positive finite norm")
        after = _dot(acquired, v) / mp.sqrt(norm * norms[3])
        gains.append((after - before) / 2 - cost)
    return gram, gains, observables


def _compare(supplied, reference, absolute_only, name):
    values = np.asarray(supplied).reshape(-1)
    references = [item for row in reference for item in row] if name == "gram" else reference
    atol, rtol = mp.mpf("1e-9"), mp.mpf("1e-10")
    errors, ratios = [], []
    for index, (value, ref) in enumerate(zip(values, references)):
        if not mp.isfinite(ref):
            raise ValueError(f"Nonfinite high-precision {name} reference")
        error = abs(mp.mpf(float(value)) - ref)
        tolerance = atol if index in absolute_only else atol + rtol * abs(ref)
        ratio = error / tolerance
        errors.append(error)
        ratios.append(ratio)
        if ratio > 1:
            raise ValueError(f"High-precision {name}[{index}] mismatch: "
                             f"error={mp.nstr(error, 12)}, tolerance={mp.nstr(tolerance, 12)}")
    return dict(max_absolute_error=float(max(errors)),
                max_tolerance_ratio=float(max(ratios)))


def verify_factor_draw(p, factor, gram, gains, observables, *, dps=80):
    """Verify one original float64 draw; return audit scalars, never replacements.

    Shapes are p[3], L[3,3], G[4,4], Gamma[3], observables[20]. Observable
    order is the existing ``gram_geometry.OBSERVABLE_NAMES`` order. Cosines
    and all Gamma values use absolute tolerance 1e-9; Gram entries and other
    observables use 1e-9 + 1e-10*abs(high_precision_reference).

    Precision is increased only when Cholesky of the newly computed exact-
    lifted L L.T fails at the current precision. A functional mismatch never
    triggers a retry, alteration, clipping, or numerical repair.
    """
    p = _array(p, (3,), "p")
    factor = _array(factor, (3, 3), "factor")
    gram = _array(gram, (4, 4), "gram")
    gains = _array(gains, (3,), "gains")
    observables = _array(observables, (20,), "observables")
    if (np.any(factor[np.triu_indices(3, 1)] != 0)
            or np.any(np.diag(factor) <= 0)):
        raise ValueError("factor must be lower triangular with strictly positive diagonal")
    if isinstance(dps, (bool, np.bool_)) or not isinstance(dps, (int, np.integer)) or dps < 30:
        raise ValueError("dps must be an integer of at least 30 decimal digits")
    precisions = [int(dps)] + [value for value in (160, 320, 640) if value > dps]
    failures = []
    for precision in precisions:
        with mp.workdps(precision):
            lifted_p = [mp.mpf(float(value)) for value in p]
            lifted_factor = _lift_matrix(factor)
            direct_h = lifted_factor * lifted_factor.T
            try:
                recovered_factor = mp.cholesky(direct_h)
                if any(recovered_factor[i, i] <= 0 or not mp.isfinite(recovered_factor[i, i])
                       for i in range(3)):
                    raise ValueError("Nonpositive or nonfinite high-precision Cholesky diagonal")
            except (ValueError, ZeroDivisionError) as error:
                failures.append(dict(dps=precision, error=str(error)))
                continue
            reference_gram, reference_gains, reference_observables = _references(lifted_p, lifted_factor)
            gram_check = _compare(gram, reference_gram, set(), "gram")
            gain_check = _compare(gains, reference_gains, set(range(3)), "gains")
            observable_check = _compare(observables, reference_observables, set(range(6)), "observables")
            return dict(dps_used=precision, precision_attempts=[item["dps"] for item in failures] + [precision],
                cholesky_precision_failures=failures,
                exact_float64_lift=True, native_log_coordinates_recomputed=False,
                sylvester_pivots=[mp.nstr(recovered_factor[i, i] ** 2, precision) for i in range(3)],
                high_precision_cholesky_passed=True,
                gram_max_absolute_error=gram_check["max_absolute_error"],
                gram_max_tolerance_ratio=gram_check["max_tolerance_ratio"],
                gains_max_absolute_error=gain_check["max_absolute_error"],
                gains_max_tolerance_ratio=gain_check["max_tolerance_ratio"],
                observables_max_absolute_error=observable_check["max_absolute_error"],
                observables_max_tolerance_ratio=observable_check["max_tolerance_ratio"],
                replacement_values_returned=False, draws_resampled=0, rows_dropped=0,
                jitter_added=0., diagonal_floor_added=0.)
    raise ValueError("High-precision Cholesky verification failed at all declared precisions: "
                     + ", ".join(str(item["dps"]) for item in failures))
