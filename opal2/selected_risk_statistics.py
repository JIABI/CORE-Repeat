"""Paired CAL-to-QUERY risk forecasts with shared chemical/layout resampling.

Predictions and both selection lists are fixed. Every occurrence of a block in
CAL, QUERY and another deployment cell receives the same bootstrap weight.
Only the selected CAL empirical rate is recomputed. These are descriptive
fixed-prediction intervals, not intervals for refitted models or future cohorts.
"""
from __future__ import annotations

import numpy as np


FORECASTS = ("base", "empirical", "jeffreys")
METRICS = ("signed_gap_per_selected", "cell_absolute_gap_per_selected",
           "cell_rate_mse", "selected_brier")


def _vector(value, name, *, binary=False, probability=False):
    x = np.asarray(value, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(name + " must be a finite nonempty vector")
    if binary and not np.isin(x, (0., 1.)).all():
        raise ValueError(name + " must contain only zero and one")
    if probability and np.any((x < 0) | (x > 1)):
        raise ValueError(name + " must be in [0,1]")
    return x


def _prepare(cells, block):
    if block not in ("groups", "layout"):
        raise ValueError("block must be groups or layout")
    cells = list(cells)
    if not cells:
        raise ValueError("At least one deployment cell is required")
    rows, names, blocks = [], [], []
    for cell in cells:
        name = str(cell['cell'])
        if name in names:
            raise ValueError("Deployment cell names must be unique")
        names.append(name)
        row = {}
        for prefix in ('cal', 'query'):
            y = _vector(cell[prefix+'_null'], prefix+' NULL', binary=True)
            selected = _vector(cell[prefix+'_selected'], prefix+' selected', binary=True)
            group = np.asarray(cell[prefix+'_'+block]).astype(str)
            if selected.shape != y.shape or group.shape != y.shape or np.any(group == ''):
                raise ValueError("Unaligned or empty " + prefix + " block labels")
            if not selected.sum():
                raise ValueError("Each cell needs a nonempty selected " + prefix + " set")
            row[prefix] = (y, selected, group)
            blocks.extend(group.tolist())
        p = _vector(cell['query_probability'], 'QUERY probability', probability=True)
        if p.shape != row['query'][0].shape:
            raise ValueError("QUERY probability shape mismatch")
        row['probability'] = p
        rows.append(row)
    unique = np.unique(blocks)
    lookup = {value: index for index, value in enumerate(unique)}
    shape = (len(rows), len(unique))
    tables = {key: np.zeros(shape) for key in ('cal_n', 'cal_y', 'query_n',
              'query_y', 'base_p', 'base_brier')}
    for index, row in enumerate(rows):
        for prefix in ('cal', 'query'):
            y, selected, group = row[prefix]
            positions = np.array([lookup[value] for value in group])
            tables[prefix+'_n'][index] = np.bincount(positions, weights=selected,
                                                    minlength=len(unique))
            tables[prefix+'_y'][index] = np.bincount(positions, weights=selected*y,
                                                    minlength=len(unique))
            if prefix == 'query':
                p = row['probability']
                tables['base_p'][index] = np.bincount(positions, weights=selected*p,
                                                     minlength=len(unique))
                tables['base_brier'][index] = np.bincount(positions, weights=selected*(p-y)**2,
                                                         minlength=len(unique))
    return names, unique, tables


def _statistics(tables, weights):
    """Compute cells x replicates, with no silent cell omission."""
    total = {key: value @ weights.T for key, value in tables.items()}
    nc, yc, nq, yq = (total[key] for key in ('cal_n', 'cal_y', 'query_n', 'query_y'))
    selected = nq.sum(axis=0)
    # A cell with no QUERY contribution needs no forecast in that replicate.
    zero_cal = np.any((nc == 0) & (nq > 0), axis=0)
    zero_query = selected == 0
    valid = ~(zero_cal | zero_query)
    empirical = np.divide(yc, nc, out=np.zeros_like(yc), where=nc > 0)
    rates = {'empirical': empirical, 'jeffreys': (yc+.5)/(nc+1.)}
    result = {}
    for name in FORECASTS:
        predicted = total['base_p'] if name == 'base' else rates[name]*nq
        error = yq-predicted
        brier = (total['base_brier'] if name == 'base'
                 else rates[name]**2*nq - 2*rates[name]*yq + yq)
        result[name] = {
            'signed_gap_per_selected': np.divide(error.sum(0), selected,
                out=np.full_like(selected, np.nan), where=selected > 0),
            'cell_absolute_gap_per_selected': np.divide(np.abs(error).sum(0), selected,
                out=np.full_like(selected, np.nan), where=selected > 0),
            'cell_rate_mse': np.divide(
                np.divide(error**2, nq, out=np.zeros_like(error), where=nq > 0).sum(0),
                selected, out=np.full_like(selected, np.nan), where=selected > 0),
            'selected_brier': np.divide(brier.sum(0), selected,
                out=np.full_like(selected, np.nan), where=selected > 0),
        }
    return result, valid, zero_cal, zero_query


def summarize_risk_forecasts(cells, *, block='groups', replicates=2000, seed=20260920):
    """Compare base p, selected-CAL rate, and Jeffreys sensitivity on fixed QUERY.

    Each cell supplies ``cell``, ``cal_null``, ``cal_selected``, ``query_null``,
    ``query_selected``, ``query_probability`` and CAL/QUERY ``groups`` or
    ``layout`` arrays. NULL labels and selections are exact binary vectors.
    Selection fractions and any tuned lambda are fixed by the caller.

    Absolute gaps are summed across cells before dividing by the total selected
    QUERY count, so errors in different cells cannot cancel. Cell-rate MSE is
    the selected-QUERY-count-weighted squared cell rate error. Paired intervals use identical
    valid draws for all forecasts. A zero selected-CAL denominator invalidates
    a draw if that cell has any selected QUERY weight. Invalid-draw frequency
    and boundary CAL rates are returned; percentile intervals are conditional
    on valid draws and need not have nominal coverage.
    """
    if (isinstance(replicates, (bool, np.bool_))
            or not isinstance(replicates, (int, np.integer)) or replicates < 1):
        raise ValueError("replicates must be a positive integer")
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    names, blocks, tables = _prepare(cells, block)
    point, _, _, _ = _statistics(tables, np.ones((1, len(blocks))))
    totals = {key: value.sum(axis=1) for key, value in tables.items()}
    per_cell, boundary = [], []
    for index, name in enumerate(names):
        nc, yc, nq, yq = (float(totals[key][index])
                          for key in ('cal_n', 'cal_y', 'query_n', 'query_y'))
        if yc in (0., nc):
            boundary.append(name)
        per_cell.append(dict(cell=name, selected_cal=int(nc), cal_NULL=int(yc),
            empirical_rate=yc/nc, jeffreys_rate=(yc+.5)/(nc+1.),
            selected_query=int(nq), observed_NULL=int(yq),
            observed_query_rate=yq/nq, base_query_rate=float(totals['base_p'][index]/nq),
            base_expected_NULL=float(totals['base_p'][index]),
            empirical_expected_NULL=yc/nc*nq, jeffreys_expected_NULL=(yc+.5)/(nc+1.)*nq))
    samples = {name: {metric: [] for metric in METRICS} for name in FORECASTS}
    valid_count = zero_cal_count = zero_query_count = 0
    if len(blocks) >= 2:
        rng = np.random.default_rng(seed)
        for begin in range(0, replicates, 64):
            batch = min(64, replicates-begin)
            weights = rng.multinomial(len(blocks), np.full(len(blocks), 1./len(blocks)), size=batch)
            values, valid, zero_cal, zero_query = _statistics(tables, weights)
            valid_count += int(valid.sum())
            zero_cal_count += int(zero_cal.sum())
            zero_query_count += int(zero_query.sum())
            for name in FORECASTS:
                for metric in METRICS:
                    samples[name][metric].extend(values[name][metric][valid].tolist())
    def interval(values):
        return np.quantile(values, [.025, .975]).tolist() if len(values) >= 2 else None
    forecasts = {name: {metric: dict(estimate=float(point[name][metric][0]),
        ci95=interval(samples[name][metric])) for metric in METRICS} for name in FORECASTS}
    paired = {}
    for name in ('empirical', 'jeffreys'):
        paired[name+'_minus_base'] = {metric: dict(
            estimate=float(point[name][metric][0]-point['base'][metric][0]),
            ci95=interval(np.asarray(samples[name][metric])-np.asarray(samples['base'][metric])))
            for metric in METRICS}
    return dict(block=block, blocks=len(blocks), cells=per_cell, forecasts=forecasts,
        paired=paired, seed=int(seed), requested_replicates=int(replicates), valid_replicates=valid_count,
        invalid_zero_cal_replicates=zero_cal_count, invalid_zero_query_replicates=zero_query_count,
        invalid_fraction=(1.-valid_count/replicates) if len(blocks) >= 2 else None,
        boundary_calibration_cells=boundary,
        boundary_warning=("Resampling cannot create an unobserved CAL class; empirical rate intervals may be degenerate. "
            "Jeffreys is a smoothing sensitivity, not a dependence-adjusted confidence interval." if boundary else None),
        few_blocks=len(blocks) < 10,
        metric_definitions=dict(
            signed_gap_per_selected='sum_j(observed_NULL_j - forecast_NULL_j) / sum_j n_query_selected_j',
            cell_absolute_gap_per_selected='sum_j abs(observed_NULL_j - forecast_NULL_j) / sum_j n_query_selected_j',
            cell_rate_mse='sum_j n_query_selected_j * (observed_rate_j - forecast_rate_j)^2 / sum_j n_query_selected_j',
            selected_brier='mean selected-query squared probability error; original individual p versus cell empirical/Jeffreys constants'),
        scope='shared CAL/QUERY block weights; fixed predictions, selections and lambda; empirical CAL rate recomputed',
        interval_interpretation='descriptive percentile intervals conditional on nonzero required denominators',
        zero_cal_rule='invalidate entire paired draw when selected QUERY weight is positive; never drop a cell',
        models_refitted=False, lambda_refitted=False, finite_sample_guarantee=False)
