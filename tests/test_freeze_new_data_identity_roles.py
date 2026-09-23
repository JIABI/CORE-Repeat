import csv
import json

import pytest

from scripts.freeze_new_data_identity_roles_20260917 import (
    INELIGIBLE_ROLE, build_assignments, build_target_support,
)
from opal2.identity_reservations import IdentityReservations, IdentityReservationError


def identity(oid, conn, *, dataset='EU_OPENSCREEN', provisional=True):
    return dict(dataset=dataset, object_id=oid, name=oid, connectivity=conn,
                strict_new_identity_screen=('no_detected_overlap_provisional'
                                            if provisional else 'historical_overlap'),
                identity_status='normalized')


def proposals(rows, override=None):
    override = override or {}
    return {(r['dataset'], r['object_id']): {
        'proposed_use': override.get(r['object_id'], 'protected_confirmation_candidate')}
        for r in rows}


def test_reserved_union_survives_same_count_source_replacement():
    # The number remains two, but an old group disappears upstream.
    rows = [identity('A', 'A'), identity('B', 'B', provisional=False),
            identity('C', 'C'), identity('XB', 'B', dataset='RxRx3_core', provisional=False)]
    pp = proposals(rows, {'B': 'possible_dataset_specific_FIT_reference_not_new_identity_confirmation',
                         'XB': 'module_discovery_candidate'})
    assigned, state = build_assignments(rows, pp,
        previous_manifest={'reserved_connectivity_groups': ['A', 'B']})
    assert set(state['reserved_connectivity_groups']) == {'A', 'B', 'C'}
    assert state['original_reserved_connectivity_groups'] == ['A', 'B']
    byid = {r['object_id']: r for r in assigned}
    assert byid['B']['identity_role'] == 'EU_RESERVED_PENDING_REVIEW'
    assert byid['XB']['identity_role'] == 'EU_IDENTITY_RESERVED_ELSEWHERE'
    assert all(not r['measurement_access'] for r in assigned)


def test_reserved_object_survives_connectivity_renormalization():
    rows = [identity('A', 'NEW', provisional=False)]
    previous_rows = [dict(dataset='EU_OPENSCREEN', object_id='A', connectivity='OLD',
                          identity_role='EU_CONFIRMATION_RESERVED', reserved_for_EU='True')]
    pp = proposals(rows, {'A': 'possible_dataset_specific_FIT_reference_not_new_identity_confirmation'})
    assigned, state = build_assignments(rows, pp,
        previous_manifest={'reserved_connectivity_groups': ['OLD']},
        previous_assignments=previous_rows)
    assert set(state['reserved_connectivity_groups']) == {'OLD', 'NEW'}
    assert assigned[0]['reserved_for_EU']
    assert assigned[0]['identity_role'] == 'EU_RESERVED_PENDING_REVIEW'


def test_six_overlay_exclusions_stay_protected_and_leave_support_pool():
    excluded = ['EOS101092', 'EOS101189', 'EOS101386', 'EOS101484', 'EOS101551', 'EOS101571']
    rows = [identity(oid, 'GROUP_' + oid) for oid in excluded + ['eligible']]
    rows += [identity('fit', 'FIT', provisional=False)]
    pp = proposals(rows, {'fit': 'possible_dataset_specific_FIT_reference_not_new_identity_confirmation'})
    overlay = [dict(dataset='EU_OPENSCREEN', object_id=oid,
                    candidate_connectivity='GROUP_' + oid,
                    action='exclude_from_new_evaluation_conservative_identity_overlap')
               for oid in excluded for _ in range(2)]
    assigned, state = build_assignments(rows, pp, overlap_rows=overlay)
    assert len(state['reserved_connectivity_groups']) == 7
    assert len(state['EU_historical_overlap_excluded_ids']) == 6
    assert sum(r['identity_role'] == INELIGIBLE_ROLE for r in assigned) == 6
    assert sum(r['confirmation_candidate_eligible'] for r in assigned) == 1
    annotations = {r['object_id']: {'target_names': 'target'} for r in rows}
    support = build_target_support(assigned, annotations)
    assert [r['object_id'] for r in support] == ['eligible']
    assert support[0]['donor_ids'] == 'fit'
    # Removing the overlay later is not enough to reopen those identities.
    again, state2 = build_assignments(rows, pp, previous_manifest=state,
                                      previous_assignments=assigned, overlap_rows=[])
    assert sum(r['identity_role'] == INELIGIBLE_ROLE for r in again) == 6
    assert state2['reserved_connectivity_groups'] == state['reserved_connectivity_groups']


def test_disagreeing_overlay_is_not_silently_accepted():
    rows = [identity('A', 'A')]
    overlay = [dict(dataset='EU_OPENSCREEN', object_id='A', candidate_connectivity='OTHER',
                    action='exclude_from_new_evaluation_conservative_identity_overlap')]
    with pytest.raises(ValueError, match='connectivity disagrees'):
        build_assignments(rows, proposals(rows), overlap_rows=overlay)


def test_overlap_ineligible_cannot_be_released_by_role_grant(tmp_path):
    row = dict(dataset='EU_OPENSCREEN', object_id='A', connectivity='A',
               identity_role=INELIGIBLE_ROLE, reserved_for_EU=True)
    with (tmp_path / 'identities.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        identity_assignments='identities.csv', reserved_connectivity_groups=['A'],
        measurement_access_released=True,
        released_role_purposes={INELIGIBLE_ROLE: ['evaluation_x', 'evaluation_outcomes']})))
    guard = IdentityReservations(tmp_path / 'manifest.json')
    assert guard.check('EU_OPENSCREEN', 'A', 'metadata')
    for purpose in ['model_fit', 'reference_fit', 'evaluation_x', 'evaluation_outcomes']:
        with pytest.raises(IdentityReservationError, match='not eligible'):
            guard.check('EU_OPENSCREEN', 'A', purpose)
