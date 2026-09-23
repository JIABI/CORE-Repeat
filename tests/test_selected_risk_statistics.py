import numpy as np
import pytest

from opal2.selected_risk_statistics import summarize_risk_forecasts


def make_cell(name='a', *, cal=(0, 1), query=(0, 1), probability=(.2, .8)):
    return dict(cell=name, cal_null=np.array(cal), cal_selected=np.ones(len(cal), bool),
        cal_groups=np.array(['g'+str(i) for i in range(len(cal))]),
        query_null=np.array(query), query_selected=np.ones(len(query), bool),
        query_probability=np.array(probability),
        query_groups=np.array(['g'+str(i) for i in range(len(query))]))


def test_shared_cal_query_blocks_preserve_matching_empirical_counts():
    result = summarize_risk_forecasts([make_cell()], replicates=200, seed=1)
    # The same two outcomes appear in CAL and QUERY. Independent resampling of
    # those roles would incorrectly introduce count-forecast uncertainty.
    assert result['forecasts']['empirical']['signed_gap_per_selected'] == {
        'estimate': 0., 'ci95': [0., 0.]}
    assert result['forecasts']['empirical']['cell_rate_mse']['ci95'] == [0., 0.]
    assert result['valid_replicates'] == 200
    assert result['paired']['empirical_minus_base']['selected_brier']['estimate'] == pytest.approx(.21)


def test_cell_absolute_error_cannot_cancel_across_deployment_cells():
    left = make_cell('left', cal=(0, 0), query=(1, 1), probability=(.2, .2))
    right = make_cell('right', cal=(1, 1), query=(0, 0), probability=(.8, .8))
    result = summarize_risk_forecasts([left, right], replicates=100)
    empirical = result['forecasts']['empirical']
    assert empirical['signed_gap_per_selected']['estimate'] == 0.
    assert empirical['cell_absolute_gap_per_selected']['estimate'] == 1.
    assert empirical['cell_rate_mse']['estimate'] == 1.
    assert result['boundary_calibration_cells'] == ['left', 'right']


def test_zero_cal_denominator_invalidates_whole_paired_draw():
    cell = make_cell(cal=(0,), query=(1,), probability=(.3,))
    cell['cal_groups'] = np.array(['cal_only'])
    cell['query_groups'] = np.array(['query_only'])
    result = summarize_risk_forecasts([cell], replicates=400, seed=7)
    assert result['invalid_zero_cal_replicates'] > 0
    assert result['invalid_zero_query_replicates'] > 0
    assert 0 < result['valid_replicates'] < 400
    assert result['invalid_fraction'] == pytest.approx(1-result['valid_replicates']/400)
    assert result['cells'][0]['jeffreys_rate'] == .25
    assert result['forecasts']['empirical']['signed_gap_per_selected']['ci95'] == [1., 1.]
    assert result['boundary_warning']


def test_cell_rate_mse_uses_query_count_weights_not_squared_counts():
    left = make_cell('small', cal=(0, 0), query=(1,), probability=(.2,))
    right = make_cell('large', cal=(1, 1), query=(1, 1, 0), probability=(.8, .8, .8))
    result = summarize_risk_forecasts([left, right], replicates=100)
    empirical = result['forecasts']['empirical']
    assert empirical['signed_gap_per_selected']['estimate'] == 0.
    assert empirical['cell_absolute_gap_per_selected']['estimate'] == .5
    assert empirical['cell_rate_mse']['estimate'] == pytest.approx(1/3)


def test_layout_resampling_and_row_order_invariance():
    cell = make_cell()
    cell.update(cal_layout=np.array(['L1', 'L2']), query_layout=np.array(['L1', 'L2']))
    first = summarize_risk_forecasts([cell], block='layout', replicates=120, seed=11)
    shuffled = {key: value[::-1] if isinstance(value, np.ndarray) else value for key, value in cell.items()}
    second = summarize_risk_forecasts([shuffled], block='layout', replicates=120, seed=11)
    assert first == second


def test_one_block_reports_no_interval_not_spurious_precision():
    cell = make_cell()
    cell['cal_groups'][:] = 'g'
    cell['query_groups'][:] = 'g'
    result = summarize_risk_forecasts([cell], replicates=100)
    assert result['valid_replicates'] == 0
    assert result['forecasts']['base']['selected_brier']['ci95'] is None


def test_empty_selected_cal_is_not_treated_as_zero_risk():
    cell = make_cell()
    cell['cal_selected'][:] = False
    with pytest.raises(ValueError, match='nonempty selected cal'):
        summarize_risk_forecasts([cell])
