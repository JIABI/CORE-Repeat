"""Absolute selected-set rates under fixed-prediction block resampling.

These intervals describe observed-block sensitivity, not model-parameter
uncertainty or coverage for future cohorts. CAL forecasts use original, fixed
QUERY selection counts as cell weights and never use QUERY outcomes.
"""
from __future__ import annotations

import numpy as np

from opal2.selected_risk_statistics import _prepare


RATE_NAMES = ("observed_query_rate", "base_score_case_mix_rate",
              "empirical_cal_forecast_rate", "jeffreys_cal_forecast_rate")
INTERPRETATIONS = {
    "observed_query_rate": (
        "Observed selected-QUERY rate; its recorded value is exact for this list. "
        "The band describes block-resampling sensitivity, not uncertainty in that known value."),
    "base_score_case_mix_rate": (
        "Mean frozen original p_NULL; the band describes selected-QUERY case-mix "
        "sensitivity, not uncertainty in fitted model parameters or the fixed-list mean."),
    "empirical_cal_forecast_rate": (
        "Selected-CAL empirical risk-rate resampling sensitivity. Summary cell weights "
        "are original selected-QUERY counts; QUERY labels and resampled QUERY weights are unused."),
    "jeffreys_cal_forecast_rate": (
        "Jeffreys-smoothed CAL point-estimate sensitivity, not a posterior credible "
        "interval or dependence-adjusted coverage claim. Summary uses fixed original QUERY counts."),
}


def _rate_draws(tables, weights, fixed_query_counts):
    """Return rows [summary, each cell] with marginal invalid draws as NaN."""
    totals = {key: value @ weights.T for key, value in tables.items()
              if key in ('cal_n', 'cal_y', 'query_n', 'query_y', 'base_p')}
    nc, yc, nq, yq, bp = (totals[key] for key in
                          ('cal_n', 'cal_y', 'query_n', 'query_y', 'base_p'))

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator,
                         out=np.full_like(numerator, np.nan), where=denominator > 0)

    values = {
        'observed_query_rate': ratio(yq, nq),
        'base_score_case_mix_rate': ratio(bp, nq),
        'empirical_cal_forecast_rate': ratio(yc, nc),
        'jeffreys_cal_forecast_rate': np.where(nc > 0, (yc+.5)/(nc+1.), np.nan),
    }
    summary = {
        'observed_query_rate': ratio(yq.sum(0), nq.sum(0)),
        'base_score_case_mix_rate': ratio(bp.sum(0), nq.sum(0)),
    }
    fixed_weights = fixed_query_counts / fixed_query_counts.sum()
    for name in RATE_NAMES[2:]:
        # Every original cell has positive fixed weight. A missing CAL
        # denominator therefore invalidates the entire summary, without omission.
        summary[name] = fixed_weights @ values[name]
    return {name: np.vstack((summary[name], values[name])) for name in RATE_NAMES}


def _support(tables, prefix, index=None):
    count = tables[prefix+'_n']
    events = tables[prefix+'_y']
    if index is None:
        count, events = count.sum(0), events.sum(0)
    else:
        count, events = count[index], events[index]
    n, y = int(count.sum()), int(events.sum())
    return dict(selected_objects=n, null_events=y, non_null_events=n-y,
                active_blocks=int(np.count_nonzero(count)),
                null_bearing_blocks=int(np.count_nonzero(events)),
                zero_events=y == 0, all_events=y == n,
                boundary=('zero_events' if y == 0 else 'all_events' if y == n else None))


def _describe_rate(name, point, samples, support, *, replicates, boundary_cells):
    valid = samples[np.isfinite(samples)]
    active = support['active_blocks']
    ci = None
    if active < 2:
        status = 'insufficient_active_blocks'
    elif len(valid) < 2:
        status = 'insufficient_valid_replicates'
    else:
        ci = np.quantile(valid, [.025, .975]).tolist()
        status = 'descriptive_percentile'
    degenerate = bool(ci is not None and abs(ci[1]-ci[0]) <= 1e-15)
    boundary = bool(name != 'base_score_case_mix_rate' and support['boundary'])
    if ci is not None and boundary:
        status = 'noninformative_boundary'
    return dict(estimate=float(point), ci95=ci, active_blocks=active,
                few_blocks=active < 10, valid_replicates=int(len(valid)),
                invalid_zero_denominator_replicates=int(replicates-len(valid)),
                invalid_fraction=float(1.-len(valid)/replicates),
                degenerate=degenerate, noninformative_boundary=boundary,
                contains_boundary_cells=bool(boundary_cells),
                boundary_cells=list(boundary_cells), interval_status=status,
                interpretation=INTERPRETATIONS[name])


def summarize_selected_risk_rates(cells, *, block='groups', replicates=2000,
                                  seed=20260920):
    """Absolute selected-set rates, retaining points regardless of event count.

    Input matches ``selected_risk_statistics.summarize_risk_forecasts``. The
    same global chemical/layout block weight is used for all appearances in
    CAL, QUERY and deployment cells. A summary CAL forecast always weights
    each cell by its *original* selected-QUERY count, never its resampled count.

    ``summary`` and each entry of ``cells[*].rates`` hold four rate records:
    ``observed_query_rate``, ``base_score_case_mix_rate``,
    ``empirical_cal_forecast_rate`` and ``jeffreys_cal_forecast_rate``. Each has
    an estimate, descriptive ``ci95`` or None, support/validity diagnostics and
    an interpretation. CAL and QUERY quantities have separate validity masks;
    no CAL-empty draw is discarded from a QUERY-only rate. Support counts at
    summary level count row appearances, not independent objects.

    Fewer than two active selected blocks suppresses the interval, not the
    point. Zero/all-event empirical intervals may remain [0,0]/[1,1], explicitly
    marked noninformative about an unobserved outcome class. ``few_blocks`` is
    a descriptive warning, not an eligibility rule or a coverage guarantee.
    """
    if (isinstance(replicates, (bool, np.bool_))
            or not isinstance(replicates, (int, np.integer)) or replicates < 1):
        raise ValueError('replicates must be a positive integer')
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    names, blocks, tables = _prepare(cells, block)
    fixed_query_counts = tables['query_n'].sum(1)
    point = _rate_draws(tables, np.ones((1, len(blocks))), fixed_query_counts)
    samples = {name: np.full((len(names)+1, replicates), np.nan) for name in RATE_NAMES}
    rng = np.random.default_rng(seed)
    for begin in range(0, replicates, 64):
        batch = min(64, replicates-begin)
        weights = rng.multinomial(len(blocks), np.full(len(blocks), 1./len(blocks)), size=batch)
        values = _rate_draws(tables, weights, fixed_query_counts)
        for name in RATE_NAMES:
            samples[name][:, begin:begin+batch] = values[name]

    support = [dict(cal=_support(tables, 'cal'), query=_support(tables, 'query'))]
    support.extend(dict(cal=_support(tables, 'cal', i), query=_support(tables, 'query', i))
                   for i in range(len(names)))
    boundary_by_role = {role: [name for name, item in zip(names, support[1:])
                               if item[role]['boundary']] for role in ('cal', 'query')}
    summaries = []
    for index, item in enumerate(support):
        rates = {}
        for name in RATE_NAMES:
            role = 'cal' if name in RATE_NAMES[2:] else 'query'
            boundary_cells = ([] if name == 'base_score_case_mix_rate' else
                              boundary_by_role[role] if index == 0 else
                              [names[index-1]] if item[role]['boundary'] else [])
            rates[name] = _describe_rate(name, point[name][index, 0], samples[name][index],
                item[role], replicates=replicates, boundary_cells=boundary_cells)
        summaries.append(rates)
    per_cell = [dict(cell=name, support=support[i+1],
                     fixed_query_weight=float(fixed_query_counts[i]/fixed_query_counts.sum()),
                     rates=summaries[i+1]) for i, name in enumerate(names)]
    return dict(schema_version=1, block=block, all_blocks=len(blocks),
                summary=summaries[0], summary_support=support[0], cells=per_cell,
                seed=int(seed), requested_replicates=int(replicates),
                cal_summary_weighting='fixed original selected QUERY counts by cell',
                cal_summary_zero_denominator_rule='invalidate draw if any selected CAL cell has zero weight',
                marginal_validity='each rate uses only its own required positive denominators',
                resampling='shared global block weights across CAL, QUERY and all cells; fixed predictions and masks',
                interval_interpretation='descriptive percentile bands conditional on required nonzero denominators',
                support_count_interpretation='selected row appearances; active blocks give distinct observed support',
                boundary_warning='Resampling cannot create an unobserved class; boundary bands do not exclude unseen risk.',
                few_blocks_warning='Few active blocks produce coarse sensitivity bands; no event-count reporting cutoff.',
                separate_block_analyses='groups and layout are separate sensitivity analyses, not a combined dependence guarantee',
                models_refitted=False, lambda_refitted=False,
                parameter_uncertainty=False, future_cohort_coverage=False,
                finite_sample_guarantee=False)
