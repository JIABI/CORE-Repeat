import json

import pytest

from opal2.module_resource_eligibility import (
    ELIGIBLE_FOR_CALIBRATION,
    NOT_APPLICABLE,
    biology_resource_eligibility,
    representation_resource_eligibility,
)


def representation_metadata():
    return dict(compatible_features=True, four_physical_repeat_roles=True,
                model_fit_groups=['f1', 'f2', 'f3', 'f4'],
                dist_cal_groups=['c1', 'c2', 'c3'],
                roles_isolated=True, x_available=True)


def biology_metadata():
    return dict(legal_annotation_available=True,
                verified_context_match_to_reference_bank=True,
                roles_isolated=True, dist_cal_groups=['c1', 'c2', 'c3'])


@pytest.mark.parametrize('function,metadata', [
    (representation_resource_eligibility, representation_metadata),
    (biology_resource_eligibility, biology_metadata),
])
def test_eligible_is_never_active_and_report_is_json(function, metadata):
    report = function(**metadata())
    assert report['status'] == ELIGIBLE_FOR_CALIBRATION
    assert report['eligible_for_calibration'] is True
    assert report['active'] is False and report['reasons'] == []
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize('function', [representation_resource_eligibility, biology_resource_eligibility])
def test_missing_resources_return_not_applicable_without_training_error(function):
    report = function()
    assert report['status'] == NOT_APPLICABLE
    assert report['active'] is False and not report['eligible_for_calibration']
    assert report['reasons']


@pytest.mark.parametrize('name', ['compatible_features', 'four_physical_repeat_roles',
                                 'roles_isolated', 'x_available'])
def test_each_representation_resource_is_required(name):
    metadata = representation_metadata()
    metadata[name] = False
    report = representation_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert any(r['code'] == name+'_not_satisfied' for r in report['reasons'])


@pytest.mark.parametrize('name', ['legal_annotation_available',
                                 'verified_context_match_to_reference_bank', 'roles_isolated'])
def test_each_biology_resource_is_required(name):
    metadata = biology_metadata()
    metadata[name] = False
    report = biology_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert any(r['code'] == name+'_not_satisfied' for r in report['reasons'])


@pytest.mark.parametrize('unverified', [None, 'false', 'true', 0, 1])
def test_truthy_or_missing_metadata_is_not_verified(unverified):
    metadata = biology_metadata()
    metadata['verified_context_match_to_reference_bank'] = unverified
    report = biology_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert report['checks']['verified_context_match_to_reference_bank'] is None


@pytest.mark.parametrize('name,groups', [
    ('model_fit_groups', ['g1']*100+['g2', 'g3']),
    ('dist_cal_groups', ['g1']*100+['g2']),
])
def test_group_threshold_counts_unique_groups_not_episodes(name, groups):
    metadata = representation_metadata()
    metadata[name] = groups
    report = representation_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert report['checks'][name]['unique_groups'] == len(set(groups))


@pytest.mark.parametrize('groups', [None, 4, 'four', [''], ['g1', None], {'g1': 1}])
def test_invalid_group_inventory_is_not_a_training_exception(groups):
    metadata = representation_metadata()
    metadata['model_fit_groups'] = groups
    report = representation_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert report['checks']['model_fit_groups']['unique_groups'] is None


def test_empty_inventory_is_a_verified_zero_count_but_ineligible():
    metadata = biology_metadata()
    metadata['dist_cal_groups'] = []
    report = biology_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert report['checks']['dist_cal_groups']['unique_groups'] == 0


def test_biology_has_no_chemical_similarity_requirement():
    report = biology_resource_eligibility(**biology_metadata())
    assert report['status'] == ELIGIBLE_FOR_CALIBRATION
    assert report['prerequisites']['chemical_similarity_threshold'] is None
    assert set(report['checks']) == {'legal_annotation_available',
        'verified_context_match_to_reference_bank', 'roles_isolated', 'dist_cal_groups'}


def test_many_annotations_cannot_override_unknown_actual_reference_context():
    metadata = biology_metadata()
    metadata['verified_context_match_to_reference_bank'] = None
    report = biology_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert report['checks']['legal_annotation_available'] is True
    assert report['active'] is False


def test_observed_group_overlap_overrides_asserted_role_isolation():
    metadata = representation_metadata()
    metadata['dist_cal_groups'] = ['f1', 'c2', 'c3']
    report = representation_resource_eligibility(**metadata)
    assert report['status'] == NOT_APPLICABLE
    assert report['checks']['model_fit_dist_cal_shared_groups'] == ['f1']
    assert any(r['code'] == 'model_fit_dist_cal_group_overlap' for r in report['reasons'])
