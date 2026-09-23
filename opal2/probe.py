"""Connect the finite-depth probe policy to the actual conditional world model.

Model-based planning and observed retrospective evaluation are separate entry
points. Model planning never reads stored P/Q/V values. Both entry points use
the already-declared four roles only; neither loads any dataset from disk.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Sequence

import numpy as np
import torch

from .data import MeasurementDataset, TrainScaler, make_inference_batch
from .planning import allocate_actions, finite_depth_probe_plan
from .utility import UtilityResult, cosine, cosine_utility_samples, enumerate_actions, selected_outcomes


def _distribution_geometry(distribution):
    """Cheap geometry; preserve double precision for observed probe values.

    Network locations can be float32 while a copula observation is float64
    and can lie outside float32's range. Conditioning must not round it first.
    """
    if all(hasattr(distribution, field) for field in ("shape", "device", "dtype")):
        return distribution.shape, distribution.device, torch.float64
    mean = distribution.mean
    return mean.shape, mean.device, torch.float64


@dataclass
class SelectiveProbeDecision:
    decision: str
    probe_compound: int | None
    probe_target: int | None
    direct_action_indices: np.ndarray | None
    expected_total_net_gain: float
    direct_expected_total_net_gain: float
    alternatives: list[dict]
    algorithm: str
    expected_total_cost: float = 0.
    incurred_setup_cost: float = 0.


def _state_utilities(known_x, draws, actions, validation_index, purchased, values, target_costs):
    """Exact full-plan and incremental utilities for heterogeneous bought sets."""
    x=np.asarray(known_x,float);y=np.asarray(draws,float);n,t=y.shape[1:3]
    bought=np.asarray(purchased,bool);value=np.asarray(values,float)
    costs=np.asarray(target_costs,float)
    v=y[:,:,validation_index]
    baseline=cosine(x[None],v)
    state_sum=x+np.where(bought[...,None],value,0).sum(axis=1)
    count=1+bought.sum(axis=1)
    sunk_cost=(costs*bought).sum(axis=1)
    state_profile=state_sum/count[:,None]
    sunk=.5*(cosine(state_profile[None],v)-baseline)-sunk_cost
    full=np.zeros((len(y),n,len(actions)));incremental=np.zeros_like(full)
    allowed=np.ones((n,len(actions)),bool);action_cost=np.zeros((n,len(actions)))
    for j,a in enumerate(actions):
        indices=a.target_indices
        if a.wells:
            allowed[:,j]=~bought[:,indices].any(axis=1)&(bought.sum(axis=1)+a.wells<=2)
            future_sum=y[:,:,indices,:].sum(axis=2)
            profile=(state_sum[None]+future_sum)/(count[None,:,None]+a.wells)
            action_cost[:,j]=costs[:,indices].sum(axis=1)
            full[:,:,j]=.5*(cosine(profile,v)-baseline)-sunk_cost-action_cost[:,j]
            incremental[:,:,j]=full[:,:,j]-sunk
        else:
            full[:,:,j]=sunk
    result=UtilityResult.from_samples(actions,incremental,action_cost,utility_name="incremental_half_cosine_gain")
    return result,full,allowed


def _probe_setup_state(purchased,setup_memberships,setup_costs):
    names=tuple(key for key in sorted(setup_memberships) if (purchased & setup_memberships[key]).any())
    return names,float(sum(setup_costs[key] for key in names))


def _allocate_full_cost_recourse(increments,full_before_setup,allowed,purchased,target_costs,*,
                                  total_budget,setup_memberships,setup_costs,cost_budget=None,
                                  risk_penalty=0.,max_incremental_null_fraction=None,
                                  max_total_null_fraction=None):
    """Exact sample-average static leaf with already-paid heterogeneous setups.

    Each setup state and final activation count fixes its full per-active cost.
    The leaf mean objective, full-plan risk penalty/cap, incremental risk cap,
    budget, and physical action constraints then are linear in action choices.
    STOP offsets depend on the state but are constant within each MILP and are
    restored before comparing subproblems. No sunk startup is charged twice.
    """
    full=np.asarray(full_before_setup,float);n,a=increments.mean.shape
    prior_active=purchased.any(axis=1);h=int(prior_active.sum())
    exposure=prior_active[:,None]|np.array([action.wells>0 for action in increments.actions])[None]
    opened,past_setup=_probe_setup_state(purchased,setup_memberships,setup_costs)
    past_well=float((purchased*target_costs).sum());remaining=int(total_budget-purchased.sum())
    financial_remaining=None if cost_budget is None else float(cost_budget-past_well-past_setup)
    if financial_remaining is not None and financial_remaining < -1e-10:
        raise ValueError("Already incurred probe and setup costs exceed the declared total cost budget.")
    if financial_remaining is not None:financial_remaining=max(0.,financial_remaining)
    new_names=tuple(key for key in sorted(setup_memberships) if key not in opened)
    new_members={key:np.column_stack([setup_memberships[key][:,action.target_indices].any(axis=1)
                    if action.wells else np.zeros(n,bool) for action in increments.actions]) for key in new_names}
    new_costs={key:setup_costs[key] for key in new_names}
    prior_full=full[:,:,0].copy()
    if h:prior_full[:,prior_active]-=past_setup/h
    if max_total_null_fraction is not None and (not np.isfinite(max_total_null_fraction) or not 0<=max_total_null_fraction<=1):
        raise ValueError("max_total_null_fraction must lie in [0,1].")
    best=None
    for bits in product((False,True),repeat=len(new_names)):
        new_setup=float(sum(new_costs[key] for key,on in zip(new_names,bits) if on));total_setup=past_setup+new_setup
        if financial_remaining is not None and new_setup>financial_remaining+1e-10:continue
        counts=range(max(1,h),min(n,h+remaining)+1) if total_setup>0 else (None,)
        for count in counts:
            state_full=full.copy()
            if total_setup:state_full-=total_setup/count*exposure[None]
            # Remove an action-independent row offset to retain a zero STOP
            # entry in the static allocator. Restore it in the overall score.
            centered=state_full-state_full[:,:,:1]
            result=UtilityResult.from_samples(increments.actions,centered,increments.costs,
                         positive_margin=increments.positive_margin,utility_name="full_cost_recourse_offset_utility")
            full_null=(state_full<=0).mean(axis=0)
            incremental_null=(state_full-prior_full[:,:,None]<=0).mean(axis=0)
            constraints=[]
            if count is not None:constraints.append({"weights":exposure.astype(float),"lower":count,"upper":count})
            if max_total_null_fraction is not None:
                constraints.append({"weights":(full_null-max_total_null_fraction)*exposure,"upper":0.})
            try:
                allocation=allocate_actions(result,remaining,allowed_actions=allowed,cost_budget=financial_remaining,
                    risk_penalty=risk_penalty,max_null_fraction=max_incremental_null_fraction,
                    penalty_null_probability=full_null,penalty_exposure=exposure,
                    constraint_null_probability=incremental_null,additional_constraints=constraints,
                    setup_memberships=new_members,setup_costs=new_costs,
                    _fixed_setup_state=dict(zip(new_names,bits)),_setup_included_in_utility=True)
            except RuntimeError as error:
                if "infeasible" not in str(error).lower():raise
                continue
            chosen=state_full[:,np.arange(n),allocation.action_indices]
            active=exposure[np.arange(n),allocation.action_indices]
            value=float(chosen.sum(axis=1).mean());risk=float((chosen[:,active]<=0).sum(axis=1).mean())
            score=value-risk_penalty*risk
            if best is None or score>best["score"]:
                best={"allocation":allocation,"score":score,"mean_total_gain":value,"expected_null_count":risk,
                      "setup_cost":total_setup,"incurred_setup_cost":past_setup,"active_count":int(active.sum()),
                      "opened_setups":tuple(opened)+tuple(key for key,on in zip(new_names,bits) if on),
                      "total_cost":past_well+past_setup+allocation.total_cost}
    if best is None:raise RuntimeError("No feasible full-cost static continuation for the declared state and risks.")
    return best


@torch.no_grad()
def plan_selective_probe(distribution, scaler: TrainScaler, known_x: np.ndarray, *,
                         candidate_indices: Sequence[int], validation_index: int,
                         total_budget: int, outer_samples=16, selection_samples=128,
                         evaluation_samples=128, lookahead_depth=1, seed=0,
                         target_costs: np.ndarray | None=None, cost_per_well=.01,
                         purchased: np.ndarray | None=None, observed_values: np.ndarray | None=None,
                         risk_penalty=0., max_incremental_null_fraction=None,
                         setup_memberships: dict[str,np.ndarray] | None=None,
                         setup_costs: dict[str,float] | None=None, cost_budget: float | None=None,
                         max_total_null_fraction: float | None=None) -> SelectiveProbeDecision:
    """Choose STOP/direct allocation OR which compound/condition to probe.

    Finite-depth nested Monte Carlo lookahead compares every feasible single
    information purchase with an exact static MILP continuation. Depth > 1
    recursively searches additional probes; this is a declared sample-average
    lookahead policy, NOT an exact solution of the continuous-observation POMDP.
    Each leaf selects on one draw stream and estimates its chosen value on an
    independent stream. Replanning after an actual probe reruns this function
    on the revealed state; it never chooses the nearest simulated branch.

    All compounds condition jointly on every revealed probe, so shared batch
    information propagates. The original joint distribution is reconditioned
    each time, avoiding approximate posterior re-encoding. Validation values
    are sampled only for utility evaluation and are never observed by policy.
    Costs include all probes and continuing acquisitions. This finite action
    design allows at most two added physical wells per compound.
    Optional setup memberships are Boolean [compound,target] matrices: a setup
    is charged once when its first member well is bought, even across stages.
    Full per-compound net utility shares total setup cost equally over the final
    activated set; exact leaf enumeration accounts for its NULL sign effects.
    """
    geometry, distribution_device, distribution_dtype = _distribution_geometry(distribution)
    x=np.asarray(known_x,float);b,t,d=geometry
    actions=enumerate_actions(candidate_indices)
    candidates=tuple(candidate_indices)
    if x.shape!=(b,d) or not np.isfinite(x).all() or validation_index in candidates or not isinstance(validation_index,(int,np.integer)) or not 0<=validation_index<t:
        raise ValueError("Finite X, valid candidate wells, and a distinct validation well are required.")
    if any(i>=t for i in candidates):raise ValueError("Candidate exceeds distribution target count.")
    if any(not isinstance(v,(int,np.integer)) or v<1 for v in (outer_samples,selection_samples,evaluation_samples,lookahead_depth)):
        raise ValueError("Monte Carlo counts and lookahead_depth must be positive integers.")
    if not isinstance(total_budget,(int,np.integer)) or not 0<=total_budget<=2*b:
        raise ValueError("Budget must be an integer between zero and two added wells per compound.")
    costs=np.full((b,t),cost_per_well) if target_costs is None else np.broadcast_to(np.asarray(target_costs,float),(b,t)).copy()
    if not np.isfinite(costs).all() or np.any(costs<0):raise ValueError("Finite nonnegative target costs are required.")
    memberships={key:np.asarray(value) for key,value in (setup_memberships or {}).items()}
    startup=dict(setup_costs or {})
    if set(memberships)!=set(startup):raise ValueError("Each adaptive setup needs both membership and cost.")
    for key,value in memberships.items():
        if not isinstance(key,str) or not key or value.shape!=(b,t) or not np.isin(value,[False,True]).all():
            raise ValueError("Adaptive setup memberships must be named Boolean [compound,target] matrices.")
        if value[:,validation_index].any() or any(value[:,j].any() for j in range(t) if j not in candidates):
            raise ValueError("A setup cannot be triggered by an unavailable/validation action.")
        if not np.isfinite(startup[key]) or startup[key]<0:raise ValueError("Setup costs must be finite and nonnegative.")
        memberships[key]=value.astype(bool)
    if cost_budget is not None and (not np.isfinite(cost_budget) or cost_budget<0):raise ValueError("cost_budget must be finite and nonnegative.")
    mask=np.zeros((b,t),bool) if purchased is None else np.asarray(purchased)
    if mask.shape!=(b,t) or not np.isin(mask,[False,True]).all():raise ValueError("purchased must be a Boolean [compounds,targets] mask.")
    mask=mask.astype(bool)
    values=np.zeros((b,t,d)) if observed_values is None else np.asarray(observed_values,float)
    if values.shape!=(b,t,d) or not np.isfinite(values[mask]).all() or mask[:,validation_index].any() or mask.sum()>total_budget:
        raise ValueError("Only bought candidate values may enter history, and their cost must fit the budget.")
    if np.any(mask.sum(axis=1)>2) or any(mask[:,j].any() for j in range(t) if j not in candidates):
        raise ValueError("History contains unavailable or excessive acquisitions.")
    # Never retain caller-provided values at unobserved coordinates, even if
    # they happen to contain archived future measurements.
    values=np.where(mask[...,None],values,0.)
    streams=np.random.default_rng(seed)
    def draw(dist,n):
        _, device, _ = _distribution_geometry(dist)
        generator=torch.Generator(device=device).manual_seed(int(streams.integers(0,2**63-1)))
        return scaler.inverse_y(dist.sample_joint(n,generator)).detach().cpu().double().numpy()
    def posterior(m,v):
        if not m.any():return distribution
        observed=torch.as_tensor(v,device=distribution_device,dtype=distribution_dtype)
        observed=scaler.transform_y(observed)
        return distribution.condition(observed,torch.as_tensor(m,device=distribution_device),
                                      target_indices=tuple(range(t)),retain_observed=True,lazy=True)
    def utility_draws(dist,count,m,v):
        incremental,full_blocks=[],[];last=None;allowed=None
        for start in range(0,count,32):
            sample=draw(dist,min(32,count-start))
            last,full,allowed=_state_utilities(x,sample,actions,validation_index,m,v,costs)
            incremental.append(last.samples);full_blocks.append(full)
        combined=UtilityResult.from_samples(actions,np.concatenate(incremental),last.costs,
                                             utility_name="incremental_half_cosine_gain")
        return combined,np.concatenate(full_blocks),allowed
    def search(m,v,depth):
        dist=posterior(m,v);remaining=int(total_budget-m.sum())
        increments,planning_full,allowed=utility_draws(dist,selection_samples,m,v)
        recourse=_allocate_full_cost_recourse(increments,planning_full,allowed,m,costs,total_budget=total_budget,
                    setup_memberships=memberships,setup_costs=startup,cost_budget=cost_budget,
                    risk_penalty=risk_penalty,max_incremental_null_fraction=max_incremental_null_fraction,
                    max_total_null_fraction=max_total_null_fraction)
        direct=recourse["allocation"]
        _,full,_=utility_draws(dist,evaluation_samples,m,v)
        chosen=full[:,np.arange(b),direct.action_indices]
        active=m.any(axis=1)|np.array([actions[j].wells>0 for j in direct.action_indices])
        if active.any():chosen[:,active]-=recourse["setup_cost"]/active.sum()
        direct_value=float(chosen.sum(axis=1).mean())
        direct_risk=float((chosen[:,active]<=0).sum(axis=1).mean())
        best_score=direct_value-risk_penalty*direct_risk
        best=SelectiveProbeDecision("direct",None,None,direct.action_indices,direct_value,direct_value,
                  [{"selected_risk_adjusted_value":best_score,"direct_expected_total_net_gain":direct_value,
                    "direct_total_setup_cost":recourse["setup_cost"],"direct_opened_setups":recourse["opened_setups"],
                    "direct_total_cost":recourse["total_cost"],"direct_expected_null_count":direct_risk}],
                  "Finite-depth sample-average probe lookahead; exact full-cost setup/count MILP leaf recourse; independent leaf value draws",
                  expected_total_cost=recourse["total_cost"],incurred_setup_cost=recourse["incurred_setup_cost"])
        if depth==0 or remaining==0:return best
        outer=draw(dist,outer_samples)
        for i in range(b):
            if m[i].sum()>=2:continue
            for j in candidates:
                if m[i,j]:continue
                probe_mask=m.copy();probe_mask[i,j]=True
                _,probe_setup=_probe_setup_state(probe_mask,memberships,startup)
                probe_cost=float((probe_mask*costs).sum()+probe_setup)
                if cost_budget is not None and probe_cost>cost_budget+1e-10:
                    best.alternatives.append({"compound_index":i,"target_index":j,"feasible":False,
                           "reason":"Probe plus newly triggered setup exceeds total cost budget"})
                    continue
                branch_values=[];branch_scores=[];branch_costs=[];feasible=True
                for s in range(outer_samples):
                    mm=m.copy();mm[i,j]=True;vv=v.copy();vv[i,j]=outer[s,i,j]
                    try:child=search(mm,vv,depth-1)
                    except RuntimeError as error:
                        if "No feasible full-cost static continuation" not in str(error):raise
                        feasible=False;break
                    branch_values.append(child.expected_total_net_gain)
                    branch_costs.append(child.expected_total_cost)
                    # Child already selected using the declared risk objective.
                    # The full-plan risk score is supplied in its alternatives.
                    branch_scores.append(child.alternatives[0]["selected_risk_adjusted_value"] if child.alternatives and "selected_risk_adjusted_value" in child.alternatives[0] else child.expected_total_net_gain)
                if not feasible:
                    best.alternatives.append({"compound_index":i,"target_index":j,"feasible":False,
                           "reason":"At least one sampled branch has no feasible declared-risk continuation"})
                    continue
                value=float(np.mean(branch_values));score=float(np.mean(branch_scores))
                alternative={"compound_index":i,"target_index":j,"expected_total_net_gain":value,
                             "feasible":True,"expected_total_cost":float(np.mean(branch_costs)),
                             "expected_risk_adjusted_value":score,"outer_mc_se":float(np.std(branch_values,ddof=1)/np.sqrt(outer_samples)) if outer_samples>1 else None}
                best.alternatives.append(alternative)
                if score>best_score:
                    best_score=score;best.decision="probe";best.probe_compound=i;best.probe_target=j
                    best.direct_action_indices=None;best.expected_total_net_gain=value
                    best.expected_total_cost=float(np.mean(branch_costs))
        best.alternatives[0]["selected_risk_adjusted_value"]=best_score
        return best
    return search(mask,values,min(int(lookahead_depth),int(total_budget-mask.sum())))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _validate(model, scaler, ds, indices, roles, budget, samples, allow_training_overlap, all_probes=True):
    if not isinstance(ds, MeasurementDataset) or not isinstance(scaler, TrainScaler):
        raise TypeError("A MeasurementDataset and its frozen TrainScaler are required.")
    if ds.Y.shape[1] != 4:
        raise ValueError("This integration uses exactly the four declared X/P/Q/V roles; no fifth repeat.")
    if ds.metadata.get("train_scaler_applied"):
        raise ValueError("Pass the unscaled fixed-space dataset, not a transformed copy.")
    if ds.feature_names.tolist() != scaler.feature_names:
        raise ValueError("The scaler and dataset fixed-coordinate order disagree.")
    ix = np.asarray(indices)
    if (ix.ndim != 1 or len(ix) < 1 or not np.issubdtype(ix.dtype, np.integer)
            or np.any(ix < 0) or np.any(ix >= len(ds)) or len(set(ix.tolist())) != len(ix)):
        raise ValueError("Compound indices must be nonempty, unique, in-range integers.")
    ix = ix.astype(int)
    if len(roles) != 4 or any(not isinstance(r, (int, np.integer)) for r in roles) or set(roles) != set(range(4)):
        raise ValueError("X/P/Q/V must be a permutation of the four existing role indices.")
    if any(not isinstance(s, (int, np.integer)) or s < 1 for s in samples):
        raise ValueError("Monte Carlo sample counts must be positive integers.")
    if not isinstance(budget, (int, np.integer)) or not (len(ix) if all_probes else 0) <= budget <= 2 * len(ix):
        raise ValueError("Total budget must pay all probes and cannot exceed two added wells per object.")
    if not ds.well_mask[ix][:, roles].all():
        raise ValueError("The declared four-role action design contains an absent/padded well.")
    if not ds.observed_mask[ix,roles[0]].all() or not np.isfinite(ds.Y[ix,roles[0]]).all():
        raise ValueError("The four-role utility task requires an observed finite initial X, not a missing placeholder or zero-context state.")
    for row in ds.well_ids[ix][:, roles]:
        if len(set(row)) != 4:
            raise ValueError("X, P, Q and V must be distinct physical wells.")
    fitting_ids = set(scaler.train_ids) | set(getattr(model, "fitting_ids", ()))
    overlap = sorted(set(ds.ids[ix]).intersection(fitting_ids))
    if overlap and not allow_training_overlap:
        raise ValueError("Evaluation compounds overlap model/scaler training IDs or model-validation fitting IDs; explicitly label in-sample diagnostics to allow this.")
    parameter = next(model.parameters())
    return ix, parameter.device, parameter.dtype, overlap


def _seed(seed: int, stream: int, branch: int = 0) -> int:
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    return int(np.random.SeedSequence([int(seed), stream, branch]).generate_state(1, dtype=np.uint64)[0])


@torch.no_grad()
def run_selective_model_probe(model,scaler,ds,indices,*,total_budget,lookahead_depth=1,
                              outer_samples=16,selection_samples=2000,evaluation_samples=2000,
                              roles=(0,1,2,3),reference_access="observed_only",seed=0,
                              cost_per_well=.01,target_costs=None,risk_penalty=0.,
                              max_incremental_null_fraction=None,retrospective=False,
                              allow_training_overlap=False,missing_outcome="error",
                              setup_memberships=None,setup_costs=None,cost_budget=None,
                              max_total_null_fraction=None):
    """Model-connected first-stage probe choice; optional observed DEV execution.

    Default returns only a model-based first decision. With retrospective=True
    the policy reveals exactly the chosen probe, updates all compounds jointly,
    and replans until the frozen budget is spent or direct recourse is chosen.
    The V values are read only AFTER the last action has been fixed.
    """
    ix,device,dtype,overlap=_validate(model,scaler,ds,indices,roles,total_budget,
                                     (outer_samples,selection_samples,evaluation_samples),allow_training_overlap,all_probes=False)
    if missing_outcome not in {"error","worst"}:
        raise ValueError("missing_outcome must be 'error' or a predeclared 'worst' completion.")
    was_training=model.training;model.eval()
    try:
        inputs,x=_initial_inputs(model,scaler,ds,ix,roles,reference_access,device,dtype)
        dist=model(inputs);n=len(ix);t=3;d=x.shape[-1]
        bought=np.zeros((n,t),bool);values=np.zeros((n,t,d));history=[];failed=np.zeros((n,t),bool);risk_infeasible=False
        while True:
            try:
                decision=plan_selective_probe(dist,scaler,x,candidate_indices=(0,1),validation_index=2,
                           total_budget=total_budget,outer_samples=outer_samples,selection_samples=selection_samples,
                           evaluation_samples=evaluation_samples,lookahead_depth=lookahead_depth,seed=int(seed)+len(history),
                           purchased=bought,observed_values=values,cost_per_well=cost_per_well,target_costs=target_costs,
                           risk_penalty=risk_penalty,max_incremental_null_fraction=max_incremental_null_fraction,
                           setup_memberships=setup_memberships,setup_costs=setup_costs,cost_budget=cost_budget,
                           max_total_null_fraction=max_total_null_fraction)
            except RuntimeError as error:
                if not retrospective or not bought.any() or "No feasible full-cost static continuation" not in str(error):raise
                # A finite simulated scenario tree is not a guarantee for every
                # possible actual observation. Do not spend beyond a rejection
                # or call a failed conditional risk constraint satisfied.
                risk_infeasible=True;history.append({"decision":"stop_risk_infeasible","reason":str(error)})
                break
            history.append(asdict(decision))
            if not retrospective:break
            if decision.decision=="probe":
                i,j=decision.probe_compound,decision.probe_target
                bought[i,j]=True
                available=bool(ds.observed_mask[ix[i],roles[j+1]]) and np.isfinite(ds.Y[ix[i],roles[j+1]]).all()
                if not available:
                    if missing_outcome=="error":raise ValueError("Selected probe has no observed finite measurement; a finite placeholder is not a paid observation.")
                    failed[i,j]=True
                    # Predeclared failure response: stop further acquisitions in
                    # this campaign. Do not condition on an invented profile or
                    # silently substitute a different physical probe.
                    break
                values[i,j]=ds.Y[ix[i],roles[j+1]]
            else:
                actions=enumerate_actions((0,1))
                for i,a in enumerate(decision.direct_action_indices):
                    for j in actions[a].target_indices:
                        bought[i,j]=True
                        available=bool(ds.observed_mask[ix[i],roles[j+1]]) and np.isfinite(ds.Y[ix[i],roles[j+1]]).all()
                        if not available:
                            if missing_outcome=="error":raise ValueError("Selected continuation has no observed finite measurement; a finite placeholder is not an outcome.")
                            failed[i,j]=True
                        else:values[i,j]=ds.Y[ix[i],roles[j+1]]
                break
        report={"mode":"RETROSPECTIVE_SELECTIVE_PROBE" if retrospective else "MODEL_SELECTIVE_PROBE_FIRST_DECISION",
                "compound_ids":ds.ids[ix],"decisions":history,"budget_wells":total_budget,
                "lookahead_depth":lookahead_depth,"fitting_overlap_diagnostic":overlap,
                "first_stage_forces_all_probes":False,"shared_posterior_update":True,
                "is_prospective_certification":False,"observed_result":bool(retrospective),
                "algorithm":"Finite-depth nested Monte Carlo receding-horizon information acquisition, exact MILP leaf allocation",
                "missing_outcome":missing_outcome,"failed_probe_response":"Stop further campaign acquisitions; retain all incurred costs; never condition on a failed probe",
                "model_constraint_status":"UNSAT_AFTER_PAID_PROBE" if risk_infeasible else "NO_CONDITIONAL_INFEASIBILITY_OBSERVED",
                "setup_costs":setup_costs or {},"cost_budget":cost_budget,
                "setup_cost_allocation":"Total incurred setup cost equally shared by final activated compounds",
                "max_total_null_fraction":max_total_null_fraction,"max_incremental_null_fraction":max_incremental_null_fraction,
                "scope":"Development decision experiment; neither a global POMDP optimality claim nor certification"}
        if retrospective:
            validator=ds.Y[ix,roles[3]];active=bought.any(axis=1)
            valid_v=ds.observed_mask[ix,roles[3]] & np.isfinite(validator).all(axis=1)
            if (active&~valid_v).any() and missing_outcome=="error":raise ValueError("Selected plan has no observed finite validator; a finite placeholder is not validation.")
            validator=np.where(valid_v[:,None],validator,0.)
            costs=np.full((n,t),cost_per_well) if target_costs is None else np.broadcast_to(np.asarray(target_costs,float),(n,t))
            profile=(x+np.where(bought[...,None],values,0).sum(axis=1))/(1+bought.sum(axis=1))[:,None]
            gain=.5*(cosine(profile,validator)-cosine(x,validator))-(costs*bought).sum(axis=1)
            opened,total_setup=_probe_setup_state(bought,{k:np.asarray(v,bool) for k,v in (setup_memberships or {}).items()},setup_costs or {})
            per_active_setup=total_setup/max(1,int(active.sum()))
            gain[active]-=per_active_setup
            technical_failure=active & (failed.any(axis=1)|~valid_v)
            gain[technical_failure]=-1-(costs*bought).sum(axis=1)[technical_failure]-per_active_setup
            report.update(purchased=bought,total_wells=int(bought.sum()),observed_net_gain=gain,
                          failed_acquisitions=failed,technical_failure=technical_failure,
                          observed_mean_net_gain=float(gain.mean()),observed_total_cost=float((costs*bought).sum()+total_setup),
                          incurred_setup_cost=total_setup,activated_setups=opened)
        return _jsonable(report)
    finally:model.train(was_training)


def _draw(distribution, count, seed):
    _, device, _ = _distribution_geometry(distribution)
    generator = torch.Generator(device=device).manual_seed(seed)
    return distribution.sample_joint(count, generator)


def _initial_inputs(model, scaler, ds, ix, roles, reference_access, device, dtype):
    """Only make_inference_batch's X-value slice is read, never target Y."""
    if not ds.observed_mask[ix,roles[0]].all() or not np.isfinite(ds.Y[ix,roles[0]]).all():
        raise ValueError("Probe utility requires an observed finite initial X; zero-context inference is a different task.")
    from .data import attach_library_context
    working=scaler.transform(ds)
    bank=getattr(model,"library_bank",None)
    if bank is not None:
        working=attach_library_context(working,bank)
    inputs = make_inference_batch(working, ix, context_indices=(roles[0],), target_indices=roles[1:],
                                  reference_access=reference_access, device=device, dtype=dtype)
    # Use the unrounded fixed-space X for the declared utility, after the legal
    # inference helper has established that this is the observed initial role.
    known_x = ds.Y[np.ix_(ix, [roles[0]], np.arange(ds.Y.shape[-1]))][:, 0].copy()
    return inputs, known_x


def _condition_on_probe(initial, history_fixed, scaler):
    """Construct X+P context from supplied observed/simulated history only.

    The initial target order is P,Q,V. No newly revealed controls are invented:
    P carries only references already declared available before the first
    decision. Acquiring a compound well does not silently buy a reference panel.
    """
    cy = initial["context_y"]
    history = torch.as_tensor(history_fixed, device=cy.device, dtype=cy.dtype)
    if history.shape != (cy.shape[0], 2, cy.shape[2]) or not torch.isfinite(history).all():
        raise ValueError("A complete finite X+P observed history is required.")
    output = {k:v for k,v in initial.items() if not k.startswith(("context_","target_"))}
    output["context_y"] = scaler.transform_y(history)
    suffixes=[k[len("target_"):] for k in initial if k.startswith("target_") and k!="target_mask"]
    for suffix in suffixes:
        if "context_"+suffix not in initial:continue
        output["context_" + suffix] = torch.cat(
            (initial["context_" + suffix], initial["target_" + suffix][:, :1]), dim=1)
        output["target_" + suffix] = initial["target_" + suffix][:, 1:]
    output["context_mask"] = torch.cat((initial["context_mask"], initial["target_mask"][:, :1]), dim=1)
    output["target_mask"] = initial["target_mask"][:, 1:]
    return output


def _gaussian_condition_on_probe(distribution, history_fixed, scaler):
    """Condition one fitted p(P,Q,V|X) jointly on every acquired P.

    Only the supplied probe observation is placed in the observation tensor.
    Q and V remain masked placeholders; no stored future measurement is read.
    The complete compound batch is passed together so a measured probe can
    update shared environmental latents relevant to other compounds.
    """
    shape, device, dtype = _distribution_geometry(distribution)
    history = torch.as_tensor(history_fixed, device=device, dtype=dtype)
    if (len(shape) != 3 or shape[1] != 3
            or history.shape != (shape[0], 2, shape[2])
            or not torch.isfinite(history).all()):
        raise ValueError("A complete finite X+P history and a joint P/Q/V distribution are required.")
    observed_y = torch.zeros(shape, device=device, dtype=dtype)
    observed_mask = torch.zeros(shape, device=device, dtype=torch.bool)
    observed_y[:, 0] = scaler.transform_y(history[:, 1])
    observed_mask[:, 0] = True
    return distribution.condition(observed_y, observed_mask, target_indices=(1, 2),lazy=True)


def _validate_update_mode(update_mode):
    if update_mode not in {"gaussian_condition", "reencode"}:
        raise ValueError("update_mode must be 'gaussian_condition' or 'reencode'.")


def _update_description(update_mode):
    if update_mode == "gaussian_condition":
        return "Exactly condition the fitted joint p(P,Q,V|X) on all observed P together; jointly sample Q,V"
    return "Re-run MeasurementWorldModel on X plus each observed P; jointly sample Q,V"


def _model_draws(model, inputs, scaler, count, seed):
    distribution = model(inputs)
    return scaler.inverse_y(_draw(distribution, count, seed)).detach().cpu().double().numpy()


def _predictive_summary(samples, mean_wells, label, budget, *, all_probed=True):
    values = np.asarray(samples, dtype=float)
    summary = {"strategy": label, "value_basis": "predictive_model_draws",
               "population_mean_net_gain": float(values.mean()),
               "mean_total_net_gain": float(values.sum(-1).mean()),
               "mean_used_wells": float(mean_wells), "declared_budget_wells": int(budget),
               "feasible_under_declared_budget": bool(mean_wells <= budget),
               "expected_population_null_fraction": float((values <= 0).mean()) if all_probed else None,
               "expected_population_positive_fraction": float((values >= .005).mean()) if all_probed else None}
    return summary


@torch.no_grad()
def run_model_probe(
    model, scaler: TrainScaler, ds: MeasurementDataset, indices: Sequence[int], *,
    total_budget: int, outer_samples: int = 16, selection_samples: int = 64,
    evaluation_samples: int = 64, seed: int = 0, reference_access: str = "observed_only",
    roles: tuple[int, int, int, int] = (0, 1, 2, 3), cost_per_well: float = .01,
    max_incremental_null_fraction: float | None = None,
    allow_training_overlap: bool = False, include_draws: bool = False,
    update_mode: str = "gaussian_condition",
) -> dict:
    """Predictively plan X -> simulated P -> conditional Q-or-stop with V scoring.

    All indices are evaluated in ONE batch so shared environment draws extend
    across the complete planned library. Sample counts control memory; this
    function never silently chunks compounds and destroys that dependence.
    Stored P/Q/V outcomes do not enter this function's model inputs or utility.
    Returned values are model predictions, not observed retrospective results.
    The default exactly conditions the fitted joint Gaussian, including shared
    environments. ``reencode`` retains the distinct learned-update comparator.
    """
    _validate_update_mode(update_mode)
    if not np.isfinite(cost_per_well) or cost_per_well < 0:
        raise ValueError("cost_per_well must be finite and nonnegative.")
    ix, device, dtype, overlap = _validate(model, scaler, ds, indices, roles, total_budget,
                                          (outer_samples, selection_samples, evaluation_samples), allow_training_overlap)
    was_training = model.training
    model.eval()
    after_probe_samples, all_three_samples = [], []
    try:
        initial, x = _initial_inputs(model, scaler, ds, ix, roles, reference_access, device, dtype)
        initial_distribution = model(initial)
        initial_future = scaler.inverse_y(_draw(initial_distribution, outer_samples, _seed(seed, 1)))
        probe = initial_future[:, :, 0].detach().cpu().double().numpy()
        del initial_future
        if update_mode == "reencode":
            initial_distribution = None
        conditional_branch, conditional_distribution = None, None
        def continuation_sampler(history, branch, purpose):
            nonlocal conditional_branch, conditional_distribution
            count = selection_samples if purpose == "selection" else evaluation_samples
            stream = 2 if purpose == "selection" else 3
            if update_mode == "gaussian_condition":
                if conditional_branch != branch:
                    conditional_distribution = None
                    conditional_distribution = _gaussian_condition_on_probe(initial_distribution, history, scaler)
                    conditional_branch = branch
                future = scaler.inverse_y(_draw(conditional_distribution, count, _seed(seed, stream, branch))).detach().cpu().double().numpy()
                if purpose == "evaluation":
                    conditional_distribution, conditional_branch = None, None
            else:
                inputs = _condition_on_probe(initial, history, scaler)
                future = _model_draws(model, inputs, scaler, count, _seed(seed, stream, branch))
            if purpose == "evaluation":
                current = history.mean(axis=1)
                v = future[:, :, 1]
                first = .5 * (cosine(current[None], v) - cosine(x[None], v)) - cost_per_well
                third = cosine_utility_samples(current, future, enumerate_actions([0]), 1,
                                                observed_count=2, cost_per_well=cost_per_well)
                after_probe_samples.append(first)
                all_three_samples.append(first + third.samples[:, :, 1])
            return future
        plan = finite_depth_probe_plan(x, probe, continuation_sampler, enumerate_actions([0]), 1,
                                       total_budget=total_budget, cost_per_well=cost_per_well,
                                       max_incremental_null_fraction=max_incremental_null_fraction)
        stop_after_probe = np.stack(after_probe_samples)
        all_three = np.stack(all_three_samples)
        n = len(ix)
        baselines = [
            _predictive_summary(np.zeros_like(plan.samples), 0, "K0_no_probe", total_budget, all_probed=False),
            _predictive_summary(stop_after_probe, n, "all_stop_after_probe", total_budget),
            _predictive_summary(all_three, 2*n, "all_add_probe_and_Q", total_budget),
        ]
        # Exact expectation over uniform budget-matched random subsets. NULL
        # probabilities mix outcome indicators, not the sign of averaged gains.
        probability = (total_budget - n) / n
        baselines.append({"strategy": "probe_all_uniform_random_Q", "value_basis": "predictive_model_draws_and_exact_uniform_subset_expectation",
                          "population_mean_net_gain": float(((1-probability)*stop_after_probe + probability*all_three).mean()),
                          "mean_total_net_gain": float(((1-probability)*stop_after_probe + probability*all_three).sum(-1).mean()),
                          "mean_used_wells": total_budget, "declared_budget_wells": total_budget,
                          "feasible_under_declared_budget": True,
                          "expected_population_null_fraction": float(((1-probability)*(stop_after_probe <= 0) + probability*(all_three <= 0)).mean()),
                          "expected_population_positive_fraction": float(((1-probability)*(stop_after_probe >= .005) + probability*(all_three >= .005)).mean())})
        summary = _predictive_summary(plan.samples, plan.mean_total_wells, "adaptive_after_simulated_probe", total_budget)
        report = {"mode": "MODEL_BASED_PROBE_PLANNING", "observed_result": False,
                  "prospective_certification": False, "future_outcomes_used_for_decision": False,
                  "simulated_probe_from_model_not_stored_P": True,
                  "initial_information": "Stored X and predeclared conditions/references only",
                  "update_mode": update_mode, "probe_update": _update_description(update_mode),
                  "reference_update": "No additional reference panels are revealed merely by buying P",
                  "covariance_scope": "All planned compounds sampled jointly; no compound minibatch approximation",
                  "compound_ids": ds.ids[ix], "n": n, "roles": dict(zip(("X", "P", "Q", "V"), roles)),
                  "sample_counts": {"probe": outer_samples, "selection_per_probe": selection_samples, "evaluation_per_probe": evaluation_samples},
                  "seed": seed, "reference_access": reference_access, "cost_per_added_well": cost_per_well,
                  "total_budget_wells": total_budget, "max_incremental_null_fraction": max_incremental_null_fraction,
                  "scaler_training_overlap": overlap, "policy": summary, "fixed_and_random_baselines": baselines,
                  "per_compound": [{"compound_id": str(ds.ids[ix[i]]), "predicted_total_gain": plan.mean_total_gain[i],
                                    "predicted_null_probability": plan.p_null[i], "predicted_positive_probability": plan.p_positive[i],
                                    "predicted_continuation_probability": float((plan.continuation_action_indices[:, i] > 0).mean())}
                                   for i in range(n)],
                  "branch_total_wells": plan.total_wells_by_probe_draw,
                  "branch_continuation_actions": plan.continuation_action_indices,
                  "limits": ["Model-based nested expectation, not independent empirical benefit",
                             "Risk constraint, if supplied, is incremental continuation risk; full-plan risk is reported separately",
                             ("Exact fitted-joint conditioning (through the original marginal maps for copula observations) is coherent with that model, not evidence of empirical calibration"
                              if update_mode == "gaussian_condition" else
                              "Re-encoding an observed probe is learned conditional prediction, not a proof of Bayesian consistency"),
                             "Cost excludes any references not explicitly included by the declared task"]}
        if include_draws:
            report["model_gain_draws"] = plan.samples
            report["simulated_probe_values"] = probe
        return _jsonable(report)
    finally:
        model.train(was_training)


def _observed_summary(gains, wells, label, budget):
    gains, wells = np.asarray(gains), np.asarray(wells)
    active = wells > 0
    return {"strategy": label, "value_basis": "observed_retrospective_four_role_outcomes",
            "population_mean_net_gain": float(gains.mean()), "total_net_gain": float(gains.sum()),
            "used_wells": int(wells.sum()), "declared_budget_wells": int(budget),
            "feasible_under_declared_budget": bool(wells.sum() <= budget), "activated": int(active.sum()),
            "null_count": int(((gains <= 0) & active).sum()),
            "fdp": float((gains[active] <= 0).mean()) if active.any() else None}


@torch.no_grad()
def evaluate_observed_probe(
    model, scaler: TrainScaler, ds: MeasurementDataset, indices: Sequence[int], *,
    total_budget: int, selection_samples: int = 128, evaluation_samples: int = 128,
    seed: int = 0, reference_access: str = "observed_only",
    roles: tuple[int, int, int, int] = (0, 1, 2, 3), cost_per_well: float = .01,
    max_incremental_null_fraction: float | None = None,
    allow_training_overlap: bool = False,
    update_mode: str = "gaussian_condition",
) -> dict:
    """Freeze continuation choices using actual paid X/P, then score held Q/V.

    This API explicitly reads already-open P and, only AFTER choosing actions,
    already-open Q/V. It is retrospective development evaluation, not a newly
    executed experiment or an independent certification. No future value is
    supplied to the conditional model while selecting continuation actions.
    """
    _validate_update_mode(update_mode)
    if not np.isfinite(cost_per_well) or cost_per_well < 0:
        raise ValueError("cost_per_well must be finite and nonnegative.")
    ix, device, dtype, overlap = _validate(model, scaler, ds, indices, roles, total_budget,
                                          (selection_samples, evaluation_samples), allow_training_overlap)
    was_training = model.training
    model.eval()
    try:
        initial, x = _initial_inputs(model, scaler, ds, ix, roles, reference_access, device, dtype)
        p = ds.Y[np.ix_(ix, [roles[1]], np.arange(ds.Y.shape[-1]))][:, 0].copy()
        if not ds.observed_mask[ix, roles[1]].all() or not np.isfinite(p).all():
            raise ValueError("Observed-probe evaluation requires a measured P; do not silently drop failures.")
        history = np.stack([x, p], axis=1)
        if update_mode == "gaussian_condition":
            initial_distribution = model(initial)
            conditional_distribution = _gaussian_condition_on_probe(initial_distribution, history, scaler)
            del initial_distribution
            planning_future = scaler.inverse_y(_draw(conditional_distribution, selection_samples, _seed(seed, 11))).detach().cpu().double().numpy()
        else:
            inputs = _condition_on_probe(initial, history, scaler)
            planning_future = _model_draws(model, inputs, scaler, selection_samples, _seed(seed, 11))
        current = history.mean(axis=1)
        incremental = cosine_utility_samples(current, planning_future, enumerate_actions([0]), 1,
                                              observed_count=2, cost_per_well=cost_per_well)
        allocation = allocate_actions(incremental, total_budget - len(ix),
                                      max_null_fraction=max_incremental_null_fraction)
        frozen_continue = allocation.action_indices.copy() > 0
        del planning_future, incremental
        if update_mode == "gaussian_condition":
            independent_future = scaler.inverse_y(_draw(conditional_distribution, evaluation_samples, _seed(seed, 12))).detach().cpu().double().numpy()
            del conditional_distribution
        else:
            independent_future = _model_draws(model, inputs, scaler, evaluation_samples, _seed(seed, 12))
        independent_incremental = cosine_utility_samples(current, independent_future, enumerate_actions([0]), 1,
                                                          observed_count=2, cost_per_well=cost_per_well)
        v_predictive = independent_future[:, :, 1]
        first_predictive = .5 * (cosine(current[None], v_predictive) - cosine(x[None], v_predictive)) - cost_per_well
        predicted_full = first_predictive + independent_incremental.samples[:, np.arange(len(ix)), allocation.action_indices]
        # The first read of stored Q/V happens here, AFTER continuation is fixed.
        actual_remaining = ds.Y[np.ix_(ix, [roles[2], roles[3]], np.arange(ds.Y.shape[-1]))].copy()
        if not ds.observed_mask[ix][:, [roles[2], roles[3]]].all() or not np.isfinite(actual_remaining).all():
            raise ValueError("Observed Q/V are missing; an explicit missing-outcome completion rule is required.")
        actual_future = np.concatenate([p[:, None], actual_remaining], axis=1)[None]
        actual = cosine_utility_samples(x, actual_future, enumerate_actions([0, 1]), 2,
                                        cost_per_well=cost_per_well)
        choices = np.where(frozen_continue, 3, 1)
        gain = actual.samples[0, np.arange(len(ix)), choices]
        wells = 1 + frozen_continue.astype(int)
        baselines = []
        for index, label in ((0, "K0_no_probe"), (1, "all_stop_after_probe"), (3, "all_add_probe_and_Q")):
            baselines.append(_observed_summary(actual.samples[0, :, index],
                                               np.full(len(ix), actual.actions[index].wells), label, total_budget))
        fraction = (total_budget-len(ix))/len(ix)
        random_mean = ((1-fraction)*actual.samples[0, :, 1] + fraction*actual.samples[0, :, 3]).mean()
        baselines.append({"strategy": "probe_all_uniform_random_Q", "value_basis": "exact_uniform_subset_expectation_on_observed_outcomes",
                          "population_mean_net_gain": float(random_mean), "expected_used_wells": total_budget,
                          "declared_budget_wells": total_budget, "feasible_under_declared_budget": True})
        return _jsonable({"mode": "OBSERVED_RETROSPECTIVE_PROBE_EVALUATION", "observed_result": True,
                          "prospective_certification": False, "future_Q_or_V_used_for_decision": False,
                          "decision_information": "Actual paid X/P, declared conditions, previously available references",
                          "update_mode": update_mode, "probe_update": _update_description(update_mode),
                          "evaluation_information": "Stored Q/V accessed after continuation choices fixed",
                          "compound_ids": ds.ids[ix], "n": len(ix), "roles": dict(zip(("X", "P", "Q", "V"), roles)),
                          "seed": seed, "reference_access": reference_access, "cost_per_added_well": cost_per_well,
                          "total_budget_wells": total_budget, "selection_samples": selection_samples,
                          "evaluation_samples": evaluation_samples, "scaler_training_overlap": overlap,
                          "policy": _observed_summary(gain, wells, "adaptive_after_actual_paid_probe", total_budget),
                          "model_prediction_after_actual_probe": _predictive_summary(predicted_full, wells.sum(), "frozen_continuation_prediction", total_budget),
                          "fixed_and_random_baselines": baselines,
                          "per_compound": [{"compound_id": str(ds.ids[ix[i]]), "continue_Q": bool(frozen_continue[i]),
                                            "used_wells": int(wells[i]), "observed_total_gain": float(gain[i]),
                                            "predicted_total_gain": float(predicted_full[:, i].mean())}
                                           for i in range(len(ix))],
                          "limits": ["Retrospective development evaluation, not new physical acquisitions",
                                     "The complete policy and model must be frozen before this evaluation",
                                     "Shared batches and prior exploration prevent independent-certification claims"]})
    finally:
        model.train(was_training)
