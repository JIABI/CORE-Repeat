import numpy as np
import pytest

from opal2.module_comparison_audit import audit_paired_module


def inputs():
    return dict(
        ids=['a', 'b', 'c'],
        core_predictions={'p_null': np.array([.3, .2, .4]), 'nll': np.array([1., 2., 3.])},
        module_predictions={'p_null': np.array([.1, .2, .4]), 'nll': np.array([.9, 2., 3.])},
        eligible_support=np.array([True, False, True]),
        active_mask=np.array([True, False, False]),
        core_selected=np.array([False, True, False]),
        module_selected=np.array([True, False, False]),
        actual_gamma=np.array([.1, -.1, .2]),
    )


def test_inactive_selection_spillover_is_legal_and_audit_does_not_mutate():
    data = inputs()
    original = data['module_predictions']['p_null'].copy()
    result = audit_paired_module(**data)
    assert result['exact_inactive_recovery']
    assert result['selection']['unsupported_membership_changes'] == 1
    assert result['selection']['symmetric_difference'] == 2
    assert result['selection']['total_actual_value_difference'] == pytest.approx(.2)
    assert result['selection']['actual_null_count_difference'] == -1
    assert result['paired_predictive_scores']['nll']['supported_mean_difference'] == pytest.approx(-.05)
    np.testing.assert_array_equal(data['module_predictions']['p_null'], original)
    assert not result['formal_certificate']


def test_inactive_prediction_change_is_rejected():
    data = inputs()
    data['module_predictions']['p_null'][1] = .21
    with pytest.raises(ValueError, match='exactly recover'):
        audit_paired_module(**data)


def test_support_is_not_activation_and_unsupported_cannot_activate():
    data = inputs()
    data['active_mask'][1] = True
    with pytest.raises(ValueError, match='unsupported'):
        audit_paired_module(**data)


def test_cell_quotas_are_checked_not_just_total_selected_count():
    data = inputs()
    data['cell_ids'] = np.array([0, 1, 1])
    with pytest.raises(ValueError, match='identical budgets'):
        audit_paired_module(**data)


def test_empty_support_has_explicit_none_and_exact_baseline():
    data = inputs()
    data['eligible_support'][:] = False
    data['active_mask'][:] = False
    data['module_predictions'] = {k: v.copy() for k, v in data['core_predictions'].items()}
    data['module_selected'] = data['core_selected'].copy()
    result = audit_paired_module(**data)
    assert result['active_count'] == 0
    assert result['paired_predictive_scores']['nll']['supported_mean_difference'] is None
    assert result['selection']['symmetric_difference'] == 0


def test_duplicate_episode_ids_are_rejected():
    data = inputs()
    data['ids'] = ['a', 'a', 'c']
    with pytest.raises(ValueError, match='Unique'):
        audit_paired_module(**data)
