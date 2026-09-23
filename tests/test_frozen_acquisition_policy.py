import numpy as np
import pytest

from opal2.frozen_acquisition_policy import (select_frozen_cohort_plan,
    select_from_extra_well_budget,historical_two_half_budgets,DataRole,OutcomeStage,OutcomeUse,
    validate_reference_role_isolation,apply_optional_output_addon,
    MeasurementUse,MeasurementPurpose,account_measurements)


def roles():
    r={role.value:[role.value.lower()] for role in DataRole}
    return r,{v[0]:v[0]+'_chemical' for v in r.values()}


def test_frozen_score_ties_budget_and_no_iid_cp_certificate():
    p=select_frozen_cohort_plan(['b','a','c'],[.3,.3,.2],[.1,.1,.1],1)
    assert p.selected_ids_in_rank_order==('a',) and p.added_action_wells==2
    np.testing.assert_allclose(p.scores,[.28,.28,.18])
    assert not p.certified
    with pytest.raises(ValueError,match='cohort-dependent'):p.request_iid_clopper_pearson_certificate()


@pytest.mark.parametrize('ids,mean,p,k',[
    (['a','a'],[1,2],[.1,.2],1),(['a'],[np.nan],[.1],1),
    (['a'],[1],[1.1],1),(['a'],[1],[.1],1.5),(['a'],[1],[.1],True),
    (['a'],[1],[.1],2)])
def test_invalid_policy_inputs_rejected(ids,mean,p,k):
    with pytest.raises(ValueError):select_frozen_cohort_plan(ids,mean,p,k)


def test_historical_budget_uses_parent_fold_not_independent_half_rounding():
    assert historical_two_half_budgets(237,119)==(14,15)
    assert historical_two_half_budgets(240,122)==(15,15)
    assert 4*sum(historical_two_half_budgets(237,119))+sum(historical_two_half_budgets(240,122))==146
    assert select_frozen_cohort_plan(['a'],[1],[.1],0).selected_ids_in_rank_order==()


def test_physical_budget_converts_only_complete_actions_and_caps_population():
    p = select_from_extra_well_budget(['b','a'], [.3,.3], [.1,.1], 3)
    assert p.added_action_wells == 2 and p.selected_ids_in_rank_order == ('a',)
    assert select_from_extra_well_budget(['a'], [.3], [.1], 20).activation_budget == 1
    with pytest.raises(ValueError):
        select_from_extra_well_budget(['a'], [.3], [.1], -1)


def test_calibration_is_legal_after_frozen_reference_and_policy_after_predictor():
    r,g=roles()
    uses=[OutcomeUse(OutcomeStage.MODEL_TRAIN,('model_fit',)),
        OutcomeUse(OutcomeStage.REFERENCE_FIT,('ref_fit',)),
        OutcomeUse(OutcomeStage.DISTRIBUTION_CALIBRATION,('dist_cal',),reference_fit_frozen=True),
        OutcomeUse(OutcomeStage.POLICY_SELECTION,('policy_cal',),predictor_frozen=True)]
    result=validate_reference_role_isolation(r,g,uses)
    assert result['valid_declared_role_isolation'] and result['eval_decision_time_features_allowed']
    assert not result['formal_statistical_certification']


def test_eval_and_policy_cal_outcomes_cannot_enter_predictor_or_reference():
    r,g=roles()
    for stage,donor in ((OutcomeStage.REFERENCE_FIT,'eval'),
        (OutcomeStage.DISTRIBUTION_CALIBRATION,'policy_cal'),
        (OutcomeStage.MODEL_TRAIN,'dist_cal')):
        with pytest.raises(ValueError):validate_reference_role_isolation(r,g,[OutcomeUse(stage,(donor,),True,True)])
    with pytest.raises(ValueError,match='frozen'):
        validate_reference_role_isolation(r,g,[OutcomeUse(OutcomeStage.DISTRIBUTION_CALIBRATION,('dist_cal',))])
    with pytest.raises(ValueError,match='frozen'):
        validate_reference_role_isolation(r,g,[OutcomeUse(OutcomeStage.POLICY_SELECTION,('policy_cal',))])


def test_related_objects_cannot_cross_reference_query_roles():
    r,g=roles();g['eval']=g['ref_fit']
    with pytest.raises(ValueError,match='group identity'):validate_reference_role_isolation(r,g,[])


def test_optional_untrained_hook_preserves_exact_core_after_nonzero_use():
    base=np.array([[.1,-0.],[.2,.3]],dtype=np.float32);original=base.tobytes()
    amended=apply_optional_output_addon(base,correction=np.ones_like(base),support=np.array([True,False]),enabled=True)
    assert amended[0,0] != base[0,0]
    assert amended[1].tobytes()==base[1].tobytes()
    assert base.tobytes()==original
    def must_not_run():raise AssertionError('Disabled or unsupported hook evaluated')
    disabled=apply_optional_output_addon(base,correction=must_not_run,enabled=False)
    unsupported=apply_optional_output_addon(base,correction=must_not_run,support=np.zeros(2,bool),enabled=True)
    assert disabled.tobytes()==original and unsupported.tobytes()==original


def test_physical_well_cost_deduplicates_existing_and_new_reference_uses():
    uses=[MeasurementUse('old1','ref1',MeasurementPurpose.REFERENCE,False,1.,'well_units'),
        MeasurementUse('new1','ref2',MeasurementPurpose.REFERENCE,True,1.,'well_units'),
        MeasurementUse('new1','ref2',MeasurementPurpose.VERIFICATION,True,1.,'well_units'),
        MeasurementUse('new2','query1',MeasurementPurpose.ACTION,True,1.,'well_units')]
    r=account_measurements(uses)
    assert r['new_unique_wells']==2 and r['existing_reference_wells']==1
    assert r['new_unique_cost']==2 and r['new_non_action_cost']==1
    assert r['reused_for_multiple_purposes']==1 and not r['gamma_adjusted']
    with pytest.raises(ValueError,match='Conflicting'):
        account_measurements(uses+[MeasurementUse('new1','ref2',MeasurementPurpose.REFERENCE,False,1.,'well_units')])
    with pytest.raises(ValueError,match='Mixed'):
        account_measurements(uses+[MeasurementUse('x','q',MeasurementPurpose.ACTION,True,1.,'GBP')])
