import copy

import numpy as np
import pytest

from opal2.selected_risk_rate_intervals import _rate_draws, summarize_selected_risk_rates
from opal2.selected_risk_statistics import _prepare


def make_cell(name='a', *, cal=(0, 1), query=(0, 1), probability=(.2, .8)):
    return dict(cell=name, cal_null=np.array(cal), cal_selected=np.ones(len(cal), bool),
        cal_groups=np.array(['g'+str(i) for i in range(len(cal))]),
        query_null=np.array(query), query_selected=np.ones(len(query), bool),
        query_probability=np.array(probability),
        query_groups=np.array(['g'+str(i) for i in range(len(query))]))


def test_points_and_fixed_query_count_weighting():
    left = make_cell('left', cal=(0, 1), query=(1,), probability=(.2,))
    right = make_cell('right', cal=(1, 1), query=(0, 1, 0), probability=(.4, .6, .8))
    result = summarize_selected_risk_rates([left, right], replicates=100)
    rates = result['summary']
    assert rates['observed_query_rate']['estimate'] == .5
    assert rates['base_score_case_mix_rate']['estimate'] == pytest.approx(.5)
    assert rates['empirical_cal_forecast_rate']['estimate'] == .875
    assert rates['jeffreys_cal_forecast_rate']['estimate'] == .75
    assert [cell['fixed_query_weight'] for cell in result['cells']] == [.25, .75]


def test_query_outcomes_and_probabilities_cannot_change_cal_forecast():
    cell = make_cell()
    first = summarize_selected_risk_rates([cell], replicates=150, seed=3)
    changed = copy.deepcopy(cell)
    changed['query_null'] = 1-changed['query_null']
    changed['query_probability'] = np.array([.99, .01])
    second = summarize_selected_risk_rates([changed], replicates=150, seed=3)
    for key in ('empirical_cal_forecast_rate', 'jeffreys_cal_forecast_rate'):
        assert first['summary'][key] == second['summary'][key]
        assert first['cells'][0]['rates'][key] == second['cells'][0]['rates'][key]


def test_marginal_denominators_and_fixed_weights_on_explicit_shared_draws():
    left = make_cell('left', cal=(0, 1), query=(1,), probability=(.2,))
    right = make_cell('right', cal=(1, 1), query=(0, 0, 1), probability=(.4, .4, .4))
    left['cal_groups'] = np.array(['a', 'b'])
    right['cal_groups'] = np.array(['b', 'c'])
    left['query_groups'] = np.array(['a'])
    right['query_groups'] = np.array(['c', 'c', 'c'])
    _, blocks, tables = _prepare([left, right], 'groups')
    assert blocks.tolist() == ['a', 'b', 'c']
    # Draw 1 QUERY weights favor left 2:3 instead of original 1:3. CAL
    # summary must nevertheless remain .25*(1/3)+.75*1 = 5/6.
    weights = np.array([[2., 1., 1.], [0., 1., 0.], [1., 0., 0.]])
    draws = _rate_draws(tables, weights, tables['query_n'].sum(1))
    assert draws['empirical_cal_forecast_rate'][0, 0] == pytest.approx(5/6)
    assert draws['observed_query_rate'][0, 0] == pytest.approx(3/5)
    # CAL-only draw remains valid for forecast, QUERY-only required
    # denominator invalidates QUERY rates without affecting CAL forecast.
    assert draws['empirical_cal_forecast_rate'][0, 1] == 1.
    assert np.isnan(draws['observed_query_rate'][0, 1])
    # Right CAL absent invalidates whole CAL summary, not the surviving
    # left cell or QUERY aggregate. No silent right-cell omission.
    assert np.isnan(draws['empirical_cal_forecast_rate'][0, 2])
    assert draws['empirical_cal_forecast_rate'][1, 2] == 0.
    assert draws['observed_query_rate'][0, 2] == 1.


def test_shared_blocks_preserve_identical_cal_and_query_rate_draws():
    cell = make_cell()
    _, _, tables = _prepare([cell], 'groups')
    draws = _rate_draws(tables, np.array([[2., 0.], [1., 1.], [0., 2.]]),
                        tables['query_n'].sum(1))
    np.testing.assert_equal(draws['observed_query_rate'], draws['empirical_cal_forecast_rate'])


def test_layout_and_row_order_invariance():
    cell = make_cell(cal=(0, 1, 0), query=(1, 0, 1), probability=(.2, .8, .4))
    cell.update(cal_layout=np.array(['L1', 'L2', 'L3']),
                query_layout=np.array(['L3', 'L1', 'L2']))
    first = summarize_selected_risk_rates([cell], block='layout', replicates=150, seed=9)
    reversed_cell = {key: value[::-1] if isinstance(value, np.ndarray) else value
                     for key, value in cell.items()}
    second = summarize_selected_risk_rates([reversed_cell], block='layout', replicates=150, seed=9)
    assert first == second


def test_boundary_rates_keep_point_and_explicitly_mark_degenerate_band():
    cell = make_cell(cal=(0, 0), query=(1, 1), probability=(.2, .2))
    result = summarize_selected_risk_rates([cell], replicates=100)
    empirical = result['summary']['empirical_cal_forecast_rate']
    observed = result['summary']['observed_query_rate']
    assert empirical['estimate'] == 0.
    assert empirical['ci95'] == [0., 0.]
    assert empirical['degenerate'] and empirical['noninformative_boundary']
    assert empirical['interval_status'] == 'noninformative_boundary'
    assert observed['ci95'] == [1., 1.]
    assert observed['noninformative_boundary']
    assert not result['summary']['base_score_case_mix_rate']['noninformative_boundary']
    assert result['cells'][0]['support']['cal']['null_bearing_blocks'] == 0


def test_single_active_block_suppresses_interval_even_with_many_inactive_blocks():
    cell = make_cell(cal=(0, 1, 0), query=(0, 1, 1), probability=(.2, .8, .3))
    cell['cal_selected'] = np.array([1, 0, 0])
    cell['query_selected'] = np.array([0, 1, 0])
    result = summarize_selected_risk_rates([cell], replicates=100)
    assert result['all_blocks'] == 3
    for rate in result['summary'].values():
        assert rate['active_blocks'] == 1
        assert rate['ci95'] is None
        assert rate['interval_status'] == 'insufficient_active_blocks'
        assert rate['few_blocks']
        assert rate['valid_replicates'] > 0
        assert np.isfinite(rate['estimate'])


def test_query_interval_uses_query_marginal_draws_not_cal_validity():
    cell = make_cell(cal=(0,), query=(0, 1), probability=(.1, .7))
    cell['cal_groups'] = np.array(['cal'])
    cell['query_groups'] = np.array(['q1', 'q2'])
    result = summarize_selected_risk_rates([cell], replicates=400, seed=7)
    rates = result['summary']
    assert (rates['observed_query_rate']['valid_replicates'] >
            rates['empirical_cal_forecast_rate']['valid_replicates'])
    assert rates['observed_query_rate']['ci95'] == [0., 1.]
    assert rates['empirical_cal_forecast_rate']['ci95'] is None


@pytest.mark.parametrize('kwargs', [{'replicates': 0}, {'replicates': True},
                                    {'seed': -1}, {'seed': True}])
def test_invalid_sampling_arguments(kwargs):
    with pytest.raises(ValueError):
        summarize_selected_risk_rates([make_cell()], **kwargs)
