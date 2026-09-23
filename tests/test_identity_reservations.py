import csv
import json

import pytest

from opal2.identity_reservations import IdentityReservations, IdentityReservationError


def make_manifest(tmp_path, *, released=False):
    rows=[dict(dataset='discovery',object_id='A',connectivity='RESERVED',identity_role='EU_IDENTITY_RESERVED_ELSEWHERE'),
          dict(dataset='discovery',object_id='B',connectivity='OTHER',identity_role='DEV'),
          dict(dataset='evaluation',object_id='E',connectivity='RESERVED',identity_role='EVAL')]
    with (tmp_path/'identities.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=dict(identity_assignments='identities.csv',reserved_connectivity_groups=['RESERVED'],
                  measurement_access_released=released,released_role_purposes={'DEV':['model_fit'],'EVAL':['evaluation_x']})
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    return IdentityReservations(path)


def test_current_stage_metadata_only(tmp_path):
    guard=make_manifest(tmp_path)
    assert guard.check('discovery','B','metadata')['object_id']=='B'
    for purpose in ['model_fit','reference_fit','representation_pretraining','evaluation_outcomes']:
        with pytest.raises(IdentityReservationError): guard.check('discovery','B',purpose)


def test_reserved_identity_cannot_be_released_as_reference(tmp_path):
    guard=make_manifest(tmp_path,released=True)
    for purpose in ['model_fit','reference_fit','representation_pretraining','distribution_calibration']:
        with pytest.raises(IdentityReservationError): guard.check('discovery','A',purpose)
    assert guard.check('discovery','B','model_fit')
    assert guard.check('evaluation','E','evaluation_x')
    with pytest.raises(IdentityReservationError): guard.check('evaluation','E','evaluation_outcomes')


def test_unknown_or_mismatched_identity_rejected(tmp_path):
    guard=make_manifest(tmp_path)
    with pytest.raises(IdentityReservationError): guard.check('discovery','unknown','metadata')
    with pytest.raises(IdentityReservationError): guard.check('discovery','B','metadata',connectivity='RESERVED')
    with pytest.raises(IdentityReservationError): guard.check('discovery','B','future_unspecified_purpose')
