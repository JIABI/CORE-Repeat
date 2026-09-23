import numpy as np
import pytest

from opal2.rxrx3_r2_dataset import (
    fit_negative_control_space, apply_negative_control_space, eligible_doses,
    grouped_parts, ROLE_KEYS,
)


def test_controls_define_space_and_drugs_do_not_change_it():
    rng = np.random.default_rng(42)
    controls = rng.normal(size=(32, 10))
    controls[:, -1] = 2.
    plates = np.repeat(['p0', 'p1'], 16)
    names = np.asarray([f'feature_{i}' for i in range(10)])
    space = fit_negative_control_space(controls, plates, names)
    assert len(space['selected_indices']) == 9
    y = rng.normal(size=(3, 4, 10))
    role_plates = np.tile(['p0', 'p1', 'p0', 'p1'], (3, 1))
    transformed, audit = apply_negative_control_space(y, role_plates, space)
    changed = y.copy()
    changed[2] *= 100.
    other, _ = apply_negative_control_space(changed, role_plates, space)
    np.testing.assert_array_equal(transformed[:2], other[:2])
    assert audit['drug_outcomes_used_to_fit_transform'] is False
    assert np.max(np.abs(other)) <= 10.


def test_bad_technical_well_stops_instead_of_removing_object():
    rng = np.random.default_rng(9)
    space = fit_negative_control_space(rng.normal(size=(16, 10)),
        np.repeat('p', 16), [f'f{i}' for i in range(10)])
    y = rng.normal(size=(2, 4, 10))
    y[0, 0, :2] = np.nan
    with pytest.raises(ValueError, match='no automatic object deletion'):
        apply_negative_control_space(y, np.full((2, 4), 'p'), space)


def test_rare_dose_is_not_rounded_and_eligibility_uses_chemicals():
    keep, excluded = eligible_doses(['a', 'b', 'c', 'd'], ['g0', 'g1', 'g0', 'g0'],
        [.025, .025, .026, .026], minimum_groups=2)
    np.testing.assert_array_equal(keep, [True, True, False, False])
    assert excluded[0]['dose_uM'] == .026
    assert excluded[0]['n_chemical_groups'] == 1


def test_all_doses_and_roles_share_chemical_allocation_without_outcomes():
    n = 150
    data = dict(ids=np.asarray([f'g{i}_d{d}' for i in range(n) for d in range(2)]),
        groups=np.repeat([f'g{i}' for i in range(n)], 2),
        dose=np.tile([.1, 1.], n), layout=np.repeat([f'b{i%3}' for i in range(n)], 2))
    parts, cells, outer = grouped_parts(data)
    assert len(parts) == 10
    np.testing.assert_array_equal(outer[::2], outer[1::2])
    count = np.zeros(2*n, int)
    for fold in range(5):
        two = [part for part, cell in zip(parts, cells) if cell['outer_fold'] == fold]
        for key in ROLE_KEYS:
            assert set(data['groups'][two[0][key]]) == set(data['groups'][two[1][key]])
        for part in two:
            count[part['DEV_EVAL']] += 1
    np.testing.assert_array_equal(count, 1)
