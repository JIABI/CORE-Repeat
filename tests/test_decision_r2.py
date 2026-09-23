"""R2 decision regression and end-to-end numerical fixtures, not real results."""
from itertools import product

import numpy as np
import pytest
import torch

from opal2.calibration import RiskContract,SplitConformalUtility,evaluate_contract
from opal2.decision_workflow import (AdmissionRequirements,CampaignOutcome,
    evaluate_campaign_sequence,model_admission,select_campaign_fixed_sequence,
    run_frozen_workflow,world_model_workflow_assurance,run_model_decision_workflow,
    CampaignSequenceController,world_model_design_assurance)
from opal2.model import JointGaussian
from opal2.planning import allocate_actions
from opal2.probe import plan_selective_probe
from opal2.probe import _allocate_full_cost_recourse
from opal2.selection import BudgetCandidate,FrozenBudgetFamily,FrozenCandidateEvaluation
from opal2.utility import UtilityResult,cosine_utility_samples,enumerate_actions,selected_outcomes


def test_fractional_action_indices_and_score_labels_are_rejected():
    for invalid in ([.9,1.9],[True,1]):
        with pytest.raises(ValueError,match="integers"):enumerate_actions(invalid)
    n=100;selected=np.arange(n)<10;gain=np.ones(n)*.1;gain[0]=-.1
    null=np.zeros(n);null[0]=1;null[10:]=.01
    positive=np.zeros(n);positive[1:10]=1
    with pytest.raises(ValueError,match="population_null"):
        evaluate_contract(selected,gain,null,positive,[str(i) for i in range(n)],RiskContract(max_fpr=.075),
              gain_support=(-1.02,1),wells_if_selected=np.ones(n),assume_iid_units=True)


def test_condition_costs_and_milp_startup_match_enumeration():
    a=enumerate_actions((0,1));samples=np.array([[[0,.10,.09,.17],[0,.08,.11,.18]]])
    costs=np.array([[0,.01,.03,.04],[0,.02,.04,.06]])
    u=UtilityResult.from_samples(a,samples,costs)
    memberships={"B":np.array([[False,True,False,True],[False,True,False,True]])}
    p=allocate_actions(u,3,cost_budget=.08,setup_memberships=memberships,setup_costs={"B":.025})
    valid=[]
    for choices in product(range(4),repeat=2):
        wells=sum(a[j].wells for j in choices)
        setup=any(memberships["B"][i,j] for i,j in enumerate(choices))
        cost=sum(costs[i,j] for i,j in enumerate(choices))+.025*setup
        value=sum(samples[0,i,j] for i,j in enumerate(choices))-.025*setup
        if wells<=3 and cost<=.08+1e-12:valid.append(value)
    assert p.expected_total_gain==pytest.approx(max(valid))
    assert p.total_cost<=.08+1e-12
    assert p.setup_cost==pytest.approx(.025*bool(p.activated_setups))


def test_risk_penalty_is_prespecified_objective_and_zero_preserves_mean():
    a=enumerate_actions((0,));s=np.array([[[0,.3],[0,.05]],[[0,-.1],[0,.05]]])
    u=UtilityResult.from_samples(a,s,np.array([0,.01]))
    pure=allocate_actions(u,1);risk=allocate_actions(u,1,risk_penalty=.2)
    assert pure.action_indices.tolist()==[1,0]
    assert risk.action_indices.tolist()==[0,1]
    assert risk.expected_total_gain==pytest.approx(.05)
    assert risk.risk_adjusted_objective==pytest.approx(.05)
    with pytest.raises(ValueError):selected_outcomes(u,np.array([.1,1.]))


def _campaign(i,n=3):
    # One population NULL is not selected; two true-positive acquisitions.
    o=FrozenCandidateEvaluation(tuple(f"c{i}_{j}" for j in range(n)),
         np.array([1,1,0]),np.array([.8,.8,0]),np.array([1.,1.,0]),
         np.array([0,0,1]),np.array([1,1,0]),"fixed_add_one")
    return CampaignOutcome(f"campaign_{i}",o)


def test_all_seven_criteria_use_independent_campaign_contrasts():
    # Numerical coverage of all seven branches; these are fixture tolerances,
    # not a replacement for the unchanged historical empirical contract.
    contract=RiskContract(max_fdp=.35,max_fpr=.35,min_sensitivity=.1,min_coverage=.05,
                          min_activations=100,min_mean_net_gain=0,max_mean_wells=2)
    result=evaluate_campaign_sequence([_campaign(i) for i in range(1500)],contract,
        maximum_campaign_size=3,gain_support=(-1.02,1),assume_iid_campaigns=True,
        frozen_policy_declaration="Frozen fixture policy")
    assert result["passed"]
    assert result["stopping_campaigns"]>=2
    assert len(result["trace"][-1]["checks"])==8 # seven + independent support
    assert result["trace"][-1]["contrast_intervals"]["max_fpr"][1]<=0
    with pytest.raises(ValueError,match="counted twice"):
        evaluate_campaign_sequence([_campaign(0),_campaign(0)],contract,maximum_campaign_size=3,
             gain_support=(-1.02,1),assume_iid_campaigns=True,frozen_policy_declaration="x")


def test_one_campaign_never_gets_sequential_certificate():
    r=evaluate_campaign_sequence([_campaign(0)],RiskContract(max_mean_wells=2),maximum_campaign_size=3,
       gain_support=(-1.02,1),assume_iid_campaigns=True,frozen_policy_declaration="fixture")
    assert not r["passed"] and r["status"]=="CONTINUE"


def _family():
    return FrozenBudgetFamily((BudgetCandidate("risk0",0),BudgetCandidate("risk1",.1)),
                               "risk_penalty","Fixed fixture risk penalty order",True)


def test_campaign_ltt_stops_at_first_failure_and_workflow_blocks_failed_model():
    family=_family();outcomes={"risk0":[_campaign(i) for i in range(40)],"risk1":[_campaign(i) for i in range(40)]}
    result=select_campaign_fixed_sequence(family,outcomes,RiskContract(max_mean_wells=2),
             maximum_campaign_size=3,gain_support=(-1.02,1),assume_iid_campaigns=True)
    assert result["selected_candidate"]=="risk1"
    u=UtilityResult.from_samples(enumerate_actions((0,)),np.array([[[0,.1]]]),np.array([0,.01]))
    blocked=run_frozen_workflow(admission={"admitted":False},family=family,selection_outcomes={},
                contract=RiskContract(max_mean_wells=2),deployment_predictions=u,
                selection_design="within_independent_campaign",gain_support=(-1.02,1),
                selection_options={},planner_options={"budget":1})
    assert blocked["status"]=="MODEL_NOT_ADMITTED" and blocked["plan"] is None


def test_calibration_gate_checks_width_and_proper_score_not_coverage_only():
    n=100;ids=[f"cal{i}" for i in range(n)];admission_ids=[f"audit{i}" for i in range(n)]
    cal=SplitConformalUtility.fit(np.zeros(n),np.zeros(n),ids,alpha=.1)
    req=AdmissionRequirements(.7,2.,.2,(-1,1),(-1,1),minimum_independent_units=30,declaration="fixed fixture tolerances")
    args=dict(calibrator=cal,observed=np.zeros(n),mean=np.zeros(n),scale=np.ones(n),
              admission_ids=admission_ids,model_score=np.zeros(n),baseline_score=np.ones(n)*.2,
              requirements=req,fitting_ids=["train"],independent_units=True,
              proper_score_name="Declared bounded energy score",proper_score_is_declared_bounded=True)
    good=model_admission(**args)
    assert good["admitted"] and not good["distribution_correctness_certified"]
    args["independent_units"]=False
    assert not model_admission(**args)["admitted"]


class _IdentityScaler:
    @staticmethod
    def inverse_y(x):return x
    @staticmethod
    def transform_y(x):return x


def _joint():
    mean=torch.tensor([[[.7,.6],[.4,.9],[.4,.8]],[[.5,.6],[.9,.2],[.8,.5]]],dtype=torch.float64)
    diag=torch.full_like(mean,.1)
    factor=torch.ones((2,3,2,1),dtype=torch.float64)*.15
    return JointGaussian(mean,diag,factor)


def test_selective_probe_compares_all_candidates_without_forcing_all_P():
    torch.set_num_threads(1);dist=_joint();x=np.array([[1.,0.],[0.,1.]])
    zero=plan_selective_probe(dist,_IdentityScaler(),x,candidate_indices=(0,1),validation_index=2,
              total_budget=0,outer_samples=2,selection_samples=4,evaluation_samples=5,seed=81)
    assert zero.decision=="direct" and zero.direct_action_indices.tolist()==[0,0]
    decision=plan_selective_probe(dist,_IdentityScaler(),x,candidate_indices=(0,1),validation_index=2,
              total_budget=1,outer_samples=2,selection_samples=4,evaluation_samples=5,seed=81)
    assert len(decision.alternatives)==5 # direct summary + every compound/condition probe
    assert {(r["compound_index"],r["target_index"]) for r in decision.alternatives[1:]}=={(0,0),(0,1),(1,0),(1,1)}
    if decision.decision=="probe":assert decision.probe_target in (0,1)
    else:assert sum(enumerate_actions((0,1))[j].wells for j in decision.direct_action_indices)<=1


def test_world_model_assurance_executes_real_joint_draw_plan_contract_chain():
    torch.set_num_threads(1);dist=_joint();x=np.array([[1.,0.],[0.,1.]])
    gen=torch.Generator().manual_seed(51)
    u=cosine_utility_samples(x,dist.sample_joint(8,gen).numpy(),enumerate_actions((0,1)),2)
    r=world_model_workflow_assurance(dist,_IdentityScaler(),x,u,
           contract=RiskContract(max_mean_wells=2),planner_options={"budget":2},gain_support=(-1.02,1),
           simulations=2,campaign_counts=(2,3),seed=18)
    assert r["sample_size_unit"].startswith("Independent model-generated")
    assert len(r["results"])==2 and all(row["passes"]==2 for row in r["results"])
    assert not r["empirical_admission_overridden"]


def test_startup_cost_risk_crossing_is_not_hidden_from_allocator():
    a=enumerate_actions((0,))
    u=UtilityResult.from_samples(a,np.array([[[0,.03],[0,.02]]]),np.array([0,.01]))
    membership={"panel":np.array([[0,1],[0,1]],bool)}
    # Paying .045 across two activations makes one FULL net gain negative.
    unrestricted=allocate_actions(u,2,setup_memberships=membership,setup_costs={"panel":.045})
    assert unrestricted.action_indices.tolist()==[1,1]
    assert unrestricted.expected_total_gain==pytest.approx(.005)
    assert unrestricted.expected_null_count==1
    safe=allocate_actions(u,2,setup_memberships=membership,setup_costs={"panel":.045},max_expected_null=0)
    assert safe.action_indices.tolist()==[0,0]


def test_sequence_controller_does_not_accept_outcomes_after_stop():
    ctl=CampaignSequenceController(RiskContract(max_mean_wells=2),maximum_campaign_size=3,
          gain_support=(-1.02,1),assume_iid_campaigns=True,frozen_policy_declaration="fixture")
    assert ctl.update(_campaign(0))["status"]=="CONTINUE"
    assert ctl.update(_campaign(1))["passed"]
    with pytest.raises(RuntimeError,match="already stopped"):ctl.update(_campaign(2))


def _concrete_workflow_fixture():
    from opal2.data import MeasurementDataset,TrainScaler,cellprofiler_groups
    from opal2.model import MeasurementWorldModel
    from opal2.provenance import bind_fitting_provenance
    torch.set_num_threads(1);rng=np.random.default_rng(981);n,w,d=17,4,4
    features=[f"Cells_Intensity_F{i}" for i in range(d)];gi,gn=cellprofiler_groups(features)
    groups=np.repeat(np.arange(n)[:,None,None],w*3,axis=1).reshape(n,w,3)
    ds=MeasurementDataset(Y=rng.normal(size=(n,w,d)),ids=np.array([f"object{i}" for i in range(n)]),
             feature_names=np.array(features),feature_group_index=gi,feature_group_names=gn,
             cond=np.zeros((n,w,2)),reference=np.zeros((n,w,3,2)),reference_mask=np.zeros((n,w,3),bool),
             groups=groups,chem=np.zeros((n,3)),metadata={"fixture":True})
    scaler=TrainScaler.fit(ds,[0]);torch.manual_seed(46)
    model=MeasurementWorldModel(ds.feature_groups,2,2,3,hidden_dim=8,latent_rank=1,residual_rank=1)
    bind_fitting_provenance(model,["object0"],["object1"])
    def entries(indices,prefix):return [{"campaign_id":f"{prefix}{i}","indices":[i]} for i in indices]
    settings={"samples":8,"seed":18,"calibration_campaigns":entries(range(2,11),"cal"),
      "admission_campaigns":entries(range(11,13),"audit"),"selection_campaigns":entries(range(13,15),"select"),
      "certification_campaigns":entries(range(15,17),"test"),"assume_iid_campaigns":True,
      "conformal_alpha":.5,"maximum_campaign_size":1,"contract":{"max_mean_wells":2},
      "family":FrozenBudgetFamily((BudgetCandidate("fixture",0.),),"risk_penalty","Fixture frozen rule",True).to_dict(),
      "planner":{"budget":1},"admission_requirements":{"min_coverage":0.,"max_mean_width":2.02,
        "max_proper_score_excess":3.1,"proper_score_difference_support":[-3.1,3.1],
        "utility_support":[-1.02,1.],"minimum_independent_units":2,"declaration":"Permissive numerical plumbing test, not empirical research tolerances"}}
    return model,scaler,ds,settings


def test_concrete_real_model_workflow_and_same_source_eligibility():
    model,scaler,ds,settings=_concrete_workflow_fixture()
    output=run_model_decision_workflow(model,scaler,ds,{"samples":8},settings)
    assert output["admission"]["admitted"]
    assert output["selection"]["selected_candidate"]=="fixture"
    assert output["certification"]["passed"] and len(output["frozen_plans"])==2
    assert not output["is_prospective_certification"]
    ds.groups[:,:,0]=0
    blocked=run_model_decision_workflow(model,scaler,ds,{"samples":8},settings)
    assert blocked["status"]=="MODEL_NOT_ADMITTED" and not blocked["source_disjoint"]


@pytest.mark.parametrize("failure",["unobserved_x","padded_candidate"])
def test_workflow_four_role_boundary_rejects_missing_x_and_padded_targets(failure):
    model,scaler,ds,_=_concrete_workflow_fixture()
    if failure=="unobserved_x":
        ds.observed_mask[15,0]=False
        expected="observed initial X"
    else:
        ds.well_mask[15,1]=False
        ds.observed_mask[15,1]=False
        expected="present physical roles"
    with pytest.raises(ValueError,match=expected):
        run_model_decision_workflow(model,scaler,ds,{"samples":2},
                                   {"samples":2,"diagnostic_indices":[15],"planner":{"budget":1}})


@pytest.mark.parametrize("case,known,unseen",[
    ("absent_fitting_rows",False,False),
    ("recorded_unseen_source",True,True),
    ("recorded_overlapping_source",True,False),
    ("unknown_pretraining_source",False,False)])
def test_workflow_source_novelty_requires_complete_fitting_source_evidence(case,known,unseen):
    model,scaler,ds,_=_concrete_workflow_fixture()
    model.fitting_sources_known=False
    model.pretraining_sources_unknown=False
    if case=="absent_fitting_rows":
        # This is equivalent to an evaluation-only archive: absence of fitting
        # identities does not establish that its sources were unseen.
        model.fitting_ids=frozenset({"absent_train","absent_validation"})
    elif case.startswith("recorded_"):
        model.fitting_ids=frozenset({"absent_train","absent_validation"})
        model.fitting_sources_known=True
        model.fitting_source_groups=frozenset({2 if case=="recorded_overlapping_source" else 99})
    else:
        # Ordinary fitting rows are present, but they cannot recover the
        # unknown source coverage of the pretrained target encoder.
        model.pretraining_sources_unknown=True
    settings={"samples":2,"calibration_campaigns":[{"campaign_id":"heldout","indices":[2]}],
              "assume_iid_campaigns":True,"require_unseen_sources":True}
    result=run_model_decision_workflow(model,scaler,ds,{"samples":2},settings)
    assert result["fitting_source_provenance_known"] is known
    assert result["unseen_fitting_sources"] is unseen
    assert result["independent_campaigns_eligible"] is unseen


def test_full_design_assurance_refits_every_stage_and_rejects_insufficient_support(monkeypatch):
    import opal2.decision_workflow as workflow
    torch.set_num_threads(1);dist=_joint();x=np.array([[1.,0.],[0.,1.]])
    u=cosine_utility_samples(x,dist.sample_joint(16,torch.Generator().manual_seed(51)).numpy(),enumerate_actions((0,1)),2)
    req=AdmissionRequirements(0.,2.02,3.1,(-3.1,3.1),(-1.02,1.),minimum_independent_units=2,
                              declaration="Permissive numerical stage-connectivity fixture, not research thresholds")
    counts={"admission":0,"ltt":0,"sequential":0}
    for key,name in (("admission","model_admission"),("ltt","select_campaign_fixed_sequence"),("sequential","evaluate_campaign_sequence")):
        original=getattr(workflow,name)
        def wrapped(*args,_key=key,_method=original,**kwargs):
            counts[_key]+=1;return _method(*args,**kwargs)
        monkeypatch.setattr(workflow,name,wrapped)
    args=dict(family=_family(),contract=RiskContract(max_mean_wells=2),planner_options={"budget":2},
           admission_requirements=req,gain_support=(-1.02,1),calibration_campaigns=9,admission_campaigns=2,
           selection_campaigns=2,certification_campaign_counts=(2,),simulations=2,conformal_alpha=.5,seed=52)
    r=world_model_design_assurance(dist,_IdentityScaler(),x,u,**args)
    assert counts=={"admission":2,"ltt":2,"sequential":4}
    assert r["calibration_and_selection_refit_in_simulation"]
    assert r["results"][0]["passes"]==2
    args["admission_campaigns"]=1
    insufficient=world_model_design_assurance(dist,_IdentityScaler(),x,u,**args)
    assert insufficient["results"][0]["passes"]==0
    assert insufficient["results"][0]["stage_outcome_counts"]["model_not_admitted"]==2
    assert counts["ltt"]==2 # no selection outcomes used after admission fails


def test_fixed_empirical_energy_score_is_proper_not_split_pair_offset():
    from opal2.decision_workflow import _bounded_energy_score
    # The obsolete split pairing gave -1/3 and preferred this dispersed
    # forecast to the perfect point forecast. Exact empirical ES is 2/9.
    assert _bounded_energy_score(np.array([[-1.],[1.],[0.]]),np.array([0.]))==pytest.approx(2/9)
    assert _bounded_energy_score(np.zeros((3,1)),np.array([0.]))==0


@pytest.mark.parametrize("budget,cost_budget,penalty,total_risk",[(3,.2,0.,None),(3,.2,.1,.75),(2,.09,.02,None)])
def test_adaptive_setup_recourse_matches_full_cost_exhaustive_search(budget,cost_budget,penalty,total_risk):
    actions=enumerate_actions((0,1));purchased=np.array([[1,0,0],[0,0,0]],bool)
    costs=np.array([[.01,.02,0],[.01,.02,0]])
    memberships={"A":np.array([[1,0,0],[1,0,0]],bool),"B":np.array([[0,1,0],[0,1,0]],bool)}
    startup={"A":.03,"B":.07}
    full=np.array([[[.08,.2,.24,.3],[0,.12,.23,.28]],[[.03,.2,.14,.3],[0,.01,.09,.12]]])
    increments=UtilityResult.from_samples(actions,full-full[:,:,:1],np.array([0,.01,.02,.03]))
    allowed=np.array([[1,0,1,0],[1,1,1,1]],bool)
    result=_allocate_full_cost_recourse(increments,full,allowed,purchased,costs,total_budget=budget,
            setup_memberships=memberships,setup_costs=startup,cost_budget=cost_budget,risk_penalty=penalty,
            max_total_null_fraction=total_risk)
    feasible=[]
    for choice in product(range(4),repeat=2):
        if any(not allowed[i,j] for i,j in enumerate(choice)):continue
        bought=purchased.copy()
        for i,j in enumerate(choice):bought[i,list(actions[j].target_indices)]=True
        if bought.sum()>budget:continue
        overhead=sum(startup[k] for k,v in memberships.items() if (bought&v).any())
        financial=(bought*costs).sum()+overhead
        if financial>cost_budget+1e-10:continue
        active=bought.any(axis=1);gain=full[:,np.arange(2),choice].copy();gain[:,active]-=overhead/active.sum()
        null=(gain[:,active]<=0).sum(axis=1).mean()
        if total_risk is not None and null>total_risk*active.sum()+1e-10:continue
        feasible.append((gain.sum(axis=1).mean()-penalty*null,overhead,financial))
    assert result["score"]==pytest.approx(max(x[0] for x in feasible))
    assert result["incurred_setup_cost"]==pytest.approx(.03)
    assert result["total_cost"]<=cost_budget+1e-10
    assert result["opened_setups"].count("A")==1


def test_model_connected_probe_carries_paid_setups_and_respects_nonuniform_startup_budget():
    torch.set_num_threads(1);dist=_joint();x=np.array([[1.,0.],[0.,1.]])
    memberships={"A":np.array([[1,0,0],[1,0,0]],bool),"B":np.array([[0,1,0],[0,1,0]],bool)}
    startup={"A":.01,"B":.2};costs=np.array([.01,.02,0.])
    purchased=np.array([[1,0,0],[0,0,0]],bool);values=np.zeros((2,3,2));values[0,0]=[.5,.5]
    result=plan_selective_probe(dist,_IdentityScaler(),x,candidate_indices=(0,1),validation_index=2,
            total_budget=3,cost_budget=.04,setup_memberships=memberships,setup_costs=startup,target_costs=costs,
            purchased=purchased,observed_values=values,outer_samples=2,selection_samples=4,evaluation_samples=5,seed=221)
    assert result.incurred_setup_cost==pytest.approx(.01)
    assert result.expected_total_cost<=.04+1e-10
    direct=result.alternatives[0]
    assert direct["direct_total_setup_cost"]==pytest.approx(.01) # A is not charged again for the other compound
    assert direct["direct_total_cost"]<=.04+1e-10
    for alternative in result.alternatives[1:]:
        if alternative["target_index"]==1:
            assert alternative["feasible"] is False and "cost budget" in alternative["reason"]
