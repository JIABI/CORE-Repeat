"""Connected model admission, fixed-family planning, and campaign certification.

For cohort-coupled plans the sampling unit is a complete independent campaign,
not a compound or a paired augmentation. Risk targets are ratios of expected
campaign counts (size-weighted population rates); they are NOT E[campaign FDP].
Every inference claim below is conditional on the declared independent-campaign
sampling model. No count of correlated wells manufactures a new campaign.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .calibration import (RiskContract, SplitConformalUtility, assert_disjoint,
                          bounded_mean_confidence_sequence, clopper_pearson)
from .planning import allocate_actions
from .selection import FrozenBudgetFamily, FrozenCandidateEvaluation, select_fixed_sequence
from .utility import UtilityResult


@dataclass(frozen=True)
class AdmissionRequirements:
    """All tolerances are fixed before independent admission outcomes are read.

    ``max_proper_score_excess`` is a paired model-minus-baseline proper score
    tolerance, using lower-is-better scores. A score comparison and marginal
    conformal coverage are complementary checks, not proof that p(y|state) is
    the true conditional distribution. Calibration width is a usability limit.
    """
    min_coverage: float
    max_mean_width: float
    max_proper_score_excess: float
    proper_score_difference_support: tuple[float, float]
    utility_support: tuple[float, float]
    alpha: float = .05
    minimum_independent_units: int = 30
    declaration: str = ""

    def __post_init__(self):
        if not 0 < self.alpha < 1 or not 0 <= self.min_coverage <= 1:
            raise ValueError("Invalid admission alpha/coverage.")
        if not np.isfinite([self.max_mean_width,self.max_proper_score_excess]).all() or self.max_mean_width <= 0:
            raise ValueError("Admission width/tolerance must be finite, width positive.")
        for support in (self.proper_score_difference_support,self.utility_support):
            if len(support)!=2 or not np.isfinite(support).all() or support[0]>=support[1]:
                raise ValueError("Admission supports must be declared finite intervals.")
        if not isinstance(self.minimum_independent_units,int) or self.minimum_independent_units<2 or not self.declaration:
            raise ValueError("Declare admission criteria and at least two independent evaluation units.")


def model_admission(calibrator: SplitConformalUtility, observed, mean, scale,
                    admission_ids: Sequence[str], model_score, baseline_score,
                    requirements: AdmissionRequirements, *, fitting_ids: Sequence[str],
                    independent_units: bool, proper_score_name: str,
                    proper_score_is_declared_bounded: bool = False) -> dict:
    """Independent admission after fitting split-conformal intervals elsewhere.

    Scores must be genuinely proper on the declared outcome (e.g. bounded
    energy score on a fixed bounded measurement space). Arbitrarily clipping
    Gaussian NLL is not generally proper and must not be passed as such.
    """
    y, mu, sd = map(lambda x: np.asarray(x,dtype=float), (observed,mean,scale))
    ids = tuple(map(str,admission_ids))
    assert_disjoint(fitting_ids,ids)
    low,high = calibrator.interval(mu,ids,predicted_scale=sd,support=requirements.utility_support)
    if y.shape!=mu.shape or not np.isfinite(y).all() or np.any(y<requirements.utility_support[0]) or np.any(y>requirements.utility_support[1]):
        raise ValueError("Admission observations must match outputs and declared support.")
    score = np.asarray(model_score,float)-np.asarray(baseline_score,float)
    lo,hi = requirements.proper_score_difference_support
    if score.shape!=(len(ids),) or not np.isfinite(score).all() or np.any(score<lo) or np.any(score>hi):
        raise ValueError("One finite proper-score difference in declared support is required per independent unit.")
    if not proper_score_name or not proper_score_is_declared_bounded:
        raise ValueError("Provide a named proper bounded scoring rule, not clipped NLL.")
    covered=((y>=low)&(y<=high)).reshape(len(ids),-1).all(axis=1)
    width=(high-low).reshape(len(ids),-1).max(axis=1)
    if np.any(width<0):
        width=np.where(width<0,np.inf,width)
    sufficient=independent_units and len(ids)>=requirements.minimum_independent_units
    tail=requirements.alpha/3
    coverage_lcb=clopper_pearson(int(covered.sum()),len(ids),alpha=tail,side="lower")[0] if independent_units else None
    score_ucb=float(min(hi,score.mean()+(hi-lo)*np.sqrt(np.log(1/tail)/(2*len(ids))))) if independent_units else None
    width_support=requirements.utility_support[1]-requirements.utility_support[0]
    width_ucb=float(min(width_support,width.mean()+width_support*np.sqrt(np.log(1/tail)/(2*len(ids))))) if independent_units and np.isfinite(width).all() else None
    checks={"independent_support":bool(sufficient),
            "utility_coverage":coverage_lcb is not None and coverage_lcb>=requirements.min_coverage,
            "usable_interval_width":width_ucb is not None and width_ucb<=requirements.max_mean_width,
            "nonvacuous_intervals":bool(np.isfinite(calibrator.radius) and np.isfinite(width).all() and np.all(low<=high) and width.mean()<width_support),
            "proper_score":score_ucb is not None and score_ucb<=requirements.max_proper_score_excess}
    return {"admitted":bool(all(checks.values())),"checks":checks,"requirements":asdict(requirements),
            "admission_ids":ids,"calibration_ids":calibrator.calibration_ids,"fitting_ids":tuple(map(str,fitting_ids)),
            "coverage":float(covered.mean()),"coverage_lcb":coverage_lcb,"width_ucb":width_ucb,
            "proper_score_excess":float(score.mean()),"proper_score_excess_ucb":score_ucb,"proper_score_name":proper_score_name,
            "distribution_correctness_certified":False,
            "scope":"Independent utility-set coverage, interval usefulness, and paired proper-score gate; not conditional distribution correctness."}


@dataclass(frozen=True)
class CampaignOutcome:
    campaign_id: str
    outcome: FrozenCandidateEvaluation

    def __post_init__(self):
        if not self.campaign_id:
            raise ValueError("A campaign needs a nonempty independent sampling-unit identifier.")


def _campaign_contrasts(campaigns: Sequence[CampaignOutcome], contract: RiskContract,
                        maximum_campaign_size: int, gain_support, wells_support,
                        positive_margin=.005):
    if not campaigns or not isinstance(maximum_campaign_size,int) or maximum_campaign_size<1:
        raise ValueError("A nonempty campaign sequence and fixed positive maximum campaign size are required.")
    if len({c.campaign_id for c in campaigns})!=len(campaigns):
        raise ValueError("A campaign cannot be counted twice.")
    seen=set(); summaries=[]
    gl,gh=gain_support; wl,wh=wells_support
    if not np.isfinite([gl,gh,wl,wh]).all() or not gl<=0<gh or gl>=gh or wl!=0 or wh<=0:
        raise ValueError("Declare gain support including stop zero and nonnegative burden support.")
    for c in campaigns:
        o=c.outcome; active=o.actions!=0; n=len(o.unit_ids)
        if n>maximum_campaign_size or seen.intersection(o.unit_ids):
            raise ValueError("Campaigns exceed declared size or reuse compounds; repeated compounds are not independent campaigns here.")
        seen.update(o.unit_ids)
        if np.any(o.gain<gl) or np.any(o.gain>gh) or np.any(o.wells>wh):
            raise ValueError("Campaign outcome violates declared bounded support.")
        false=active & (o.gain<=0)
        positive=o.population_positive; null=o.population_null
        if contract.max_fpr is not None or contract.min_sensitivity is not None:
            if not o.common_endpoint:
                raise ValueError("FPR/sensitivity require a common counterfactual endpoint.")
        if contract.max_fpr is not None and (null is None or np.any(null[active]!=(o.gain[active]<=0))):
            raise ValueError("Population NULL labels disagree with selected endpoint.")
        if contract.min_sensitivity is not None and (positive is None or np.any(positive[active]!=(o.gain[active]>=positive_margin))):
            raise ValueError("Population POSITIVE labels disagree with selected endpoint.")
        summaries.append(dict(n=n,active=int(active.sum()),false=int(false.sum()),
                              null=int(null.sum()) if null is not None else 0,
                              positive=int(positive.sum()) if positive is not None else 0,
                              hit=int((active&positive).sum()) if positive is not None else 0,
                              value=float(o.gain.sum()),wells=float(o.wells.sum())))
    arr={k:np.array([r[k] for r in summaries],float) for k in summaries[0]}
    M=maximum_campaign_size; contrasts={}
    # Each contrast is oriented so E[contrast] <= 0 satisfies its criterion.
    for name,num,den,sign in (("max_fdp","false","active",1),("max_fpr","false","null",1),
                              ("min_sensitivity","hit","positive",-1),("min_coverage","active","n",-1)):
        threshold=getattr(contract,name)
        if threshold is not None:
            x=sign*(arr[num]-threshold*arr[den])/M
            support=(-threshold,1-threshold) if sign==1 else (-(1-threshold),threshold)
            contrasts[name]=(x,support,False)
    if contract.min_mean_net_gain is not None:
        threshold=contract.min_mean_net_gain
        contrasts["min_mean_net_gain"]=((threshold*arr["n"]-arr["value"])/M,
                                            (min(0,threshold-gh),max(0,threshold-gl)),True)
    if contract.max_mean_wells is not None:
        threshold=contract.max_mean_wells
        contrasts["max_mean_wells"]=((arr["wells"]-threshold*arr["n"])/M,
                                         (min(0,-threshold),max(0,wh-threshold)),False)
    return arr,contrasts


def evaluate_campaign_sequence(campaigns: Sequence[CampaignOutcome], contract: RiskContract, *,
                               maximum_campaign_size: int, gain_support: tuple[float,float],
                               wells_support=(0.,2.), assume_iid_campaigns=False,
                               minimum_campaigns=2, frozen_policy_declaration: str,
                               previously_used_ids: Sequence[str]=(), stop_on_decision=True,
                               developmental=True) -> dict:
    """Full seven-criterion anytime-valid intersection over independent campaigns.

    For example FDP targets E[F_campaign]/E[A_campaign], proved by the bounded
    contrast E[F-alpha*A] <= 0, not a binomial treatment of correlated compounds.
    Counts are divided by the predeclared maximum campaign size, not by their
    observed random size. Simultaneous alpha spending covers all criteria and
    times. A failure is established only when a lower contrast bound violates a
    threshold; otherwise the result is CONTINUE, not a disguised failure.
    """
    if not assume_iid_campaigns or not frozen_policy_declaration:
        raise ValueError("A frozen policy and explicitly independent campaigns are required.")
    if not isinstance(minimum_campaigns,int) or minimum_campaigns<2:
        raise ValueError("At least two independent campaigns must be declared; one source is not replication.")
    ids={i for c in campaigns for i in c.outcome.unit_ids}
    if ids.intersection(map(str,previously_used_ids)):
        raise ValueError("Certification campaigns overlap fitting, calibration, admission, or selection outcomes.")
    arr,contrasts=_campaign_contrasts(campaigns,contract,maximum_campaign_size,gain_support,wells_support)
    if not contrasts:
        raise ValueError("At least one population statistical criterion is required in addition to activation guards.")
    sequences={}; alpha=contract.alpha/len(contrasts)
    for name,(x,(lo,hi),strict) in contrasts.items():
        if lo==hi:
            sequences[name]={"mean":np.full(len(x),lo),"lower":np.full(len(x),lo),"upper":np.full(len(x),hi)}
        else:
            sequences[name]=bounded_mean_confidence_sequence(x,lo,hi,alpha=alpha,assume_iid_units=True)
    cumulative_active=np.cumsum(arr["active"]); trace=[]; decision="CONTINUE"; stop=len(campaigns)
    for t in range(1,len(campaigns)+1):
        checks={name:bool(seq["upper"][t-1]<0 if contrasts[name][2] else seq["upper"][t-1]<=0) for name,seq in sequences.items()}
        if contract.min_activations is not None:
            checks["min_activations"]=bool(cumulative_active[t-1]>=contract.min_activations)
        checks["independent_campaign_support"]=t>=minimum_campaigns
        violated=[name for name,seq in sequences.items() if seq["lower"][t-1]>0 or (contrasts[name][2] and seq["lower"][t-1]>=0)]
        status="PASS" if all(checks.values()) else ("FUTILITY" if violated and t>=minimum_campaigns else "CONTINUE")
        trace.append({"campaigns":t,"compounds":int(arr["n"][:t].sum()),"activations":int(cumulative_active[t-1]),
                      "checks":checks,"status":status,"violated":violated,
                      "contrast_intervals":{name:[float(seq["lower"][t-1]),float(seq["upper"][t-1])] for name,seq in sequences.items()}})
        if status!="CONTINUE" and stop_on_decision:
            decision=status;stop=t;break
        decision=status
    return {"passed":decision=="PASS","status":decision,"stopping_campaigns":stop,
            "stopping_n":int(arr["n"][:stop].sum()),"trace":trace,"alpha":contract.alpha,
            "contract":asdict(contract),"frozen_policy":frozen_policy_declaration,
            "estimand":"Ratios of expected campaign counts; mean net value/burden weighted by campaign population size",
            "assumption":"IID complete campaigns, bounded prespecified maximum size, arbitrary dependence within campaign",
            "is_prospective_certification":False,"developmental":bool(developmental),
            "scope":"Time-uniform statistical evaluation conditional on sampling assumptions; does not assert prospective execution."}


class CampaignSequenceController:
    """Consume exactly one newly revealed independent campaign per update.

    Once PASS/FUTILITY occurs, another update is rejected. Therefore a caller
    can stop without reading the remaining archived campaign outcomes. Complete
    settings, policy identity, and pre-used IDs are fixed at construction.
    """
    def __init__(self,contract:RiskContract,**options):
        if options.get("assume_iid_campaigns") is not True or not options.get("frozen_policy_declaration"):
            raise ValueError("Sequential controller needs explicit independence and a frozen policy.")
        self.contract=contract;self.options=dict(options);self.campaigns=[];self.result=None

    def update(self,campaign:CampaignOutcome):
        if self.result is not None and self.result["status"]!="CONTINUE":
            raise RuntimeError("Sequential controller already stopped; remaining outcomes must not be opened.")
        if not isinstance(campaign,CampaignOutcome):raise TypeError("update consumes one actual CampaignOutcome.")
        self.campaigns.append(campaign)
        self.result=evaluate_campaign_sequence(self.campaigns,self.contract,**self.options)
        return self.result


def select_campaign_fixed_sequence(family: FrozenBudgetFamily,
                                  outcomes: Mapping[str,Sequence[CampaignOutcome]], contract: RiskContract, *,
                                  maximum_campaign_size: int,gain_support,wells_support=(0.,2.),
                                  assume_iid_campaigns=False,previously_used_ids: Sequence[str]=()):
    """Fixed-sequence LTT using independent-campaign bounded-contrast tests.

    Each candidate is an intersection-union test at alpha; no alpha splitting
    among criteria is needed to reject the union null. Stop at the first failed
    candidate. No monotonicity of FDP along the family is assumed.
    """
    if not assume_iid_campaigns or set(outcomes)!={c.name for c in family.candidates}:
        raise ValueError("Require independent campaign outcomes for exactly the frozen family.")
    reference=[(c.campaign_id,c.outcome.unit_ids) for c in outcomes[family.candidates[0].name]]
    if any([(c.campaign_id,c.outcome.unit_ids) for c in outcomes[x.name]]!=reference for x in family.candidates):
        raise ValueError("All family candidates must use the same campaign ordering.")
    if len(reference)<2:
        raise ValueError("A single campaign cannot support the declared campaign LTT procedure.")
    if set(previously_used_ids).intersection(i for _, ids in reference for i in ids):
        raise ValueError("Selection outcomes overlap previously used outcomes.")
    tested=[];last=None
    for candidate in family.candidates:
        arr,contrasts=_campaign_contrasts(outcomes[candidate.name],contract,maximum_campaign_size,gain_support,wells_support)
        if not contrasts:
            raise ValueError("LTT requires statistical criteria, not only an activation guard.")
        tests={}
        for name,(x,(lo,hi),strict) in contrasts.items():
            margin=-float(x.mean())
            vacuous=hi<0 if strict else hi<=0
            p=0.0 if vacuous else (1.0 if margin<=0 else float(np.exp(-2*len(x)*(margin/(hi-lo))**2)))
            tests[name]={"p_value":p,"contrast_mean":float(x.mean()),"known_support":[lo,hi]}
        p=max(t["p_value"] for t in tests.values())
        guard=contract.min_activations is None or arr["active"].sum()>=contract.min_activations
        passed=p<=contract.alpha and guard
        tested.append({"candidate":asdict(candidate),"iut_p_value":p,"passed":bool(passed),"tests":tests})
        if not passed:break
        last=candidate.name
    return {"selected_candidate":last,"tested":tested,"passed":last is not None,
            "family":family.to_dict(),"method":"Fixed-sequence LTT with campaign-level bounded contrast IUT",
            "is_prospective_certification":False}


def run_frozen_workflow(*, admission: dict, family: FrozenBudgetFamily,
                        selection_outcomes: Mapping, contract: RiskContract,
                        deployment_predictions: UtilityResult,
                        selection_design: str, gain_support,
                        selection_options: dict, planner_options: dict,
                        certification_campaigns: Callable[[np.ndarray],Sequence[CampaignOutcome]] | None=None,
                        certification_options: dict | None=None) -> dict:
    """Executable admission -> LTT -> allocate -> independent certification.

    Independent outcomes enter only LTT or the post-plan certification callback,
    never the allocator. Candidate rule_parameters augment frozen planner
    options; the axis must be an allocator option (e.g. risk_penalty). Different
    deployment cohorts rerun the same selected algorithm, not a stored ranking.
    """
    if admission.get("admitted") is not True:
        return {"status":"MODEL_NOT_ADMITTED","admission":admission,"plan":None,"selection":None,"certification":None}
    used=tuple(admission["fitting_ids"])+tuple(admission["calibration_ids"])+tuple(admission["admission_ids"])
    if selection_design=="within_independent_campaign":
        selection=select_campaign_fixed_sequence(family,selection_outcomes,contract,gain_support=gain_support,
                                                 previously_used_ids=used,**selection_options)
    else:
        raise ValueError("This cohort allocator requires independent-campaign LTT; pointwise LTT cannot authorize a different, cohort-coupled deployment algorithm.")
    name=selection.get("selected_candidate")
    if name is None:
        return {"status":"NO_RULE_ADMITTED","admission":admission,"selection":selection,"plan":None,"certification":None}
    candidate=next(c for c in family.candidates if c.name==name)
    options=dict(planner_options);options.update(dict(candidate.rule_parameters));options[family.axis_name]=candidate.axis_value
    plan=allocate_actions(deployment_predictions,**options)
    certification=None
    if certification_campaigns is not None:
        if selection_design!="within_independent_campaign":
            raise ValueError("Cohort allocation certification requires independent-campaign selection, not a pointwise CP contract.")
        selection_ids=tuple(i for c in selection_outcomes[name] for i in c.outcome.unit_ids)
        campaigns=certification_campaigns(plan.action_indices.copy())
        certification=evaluate_campaign_sequence(campaigns,contract,gain_support=gain_support,
                                                   previously_used_ids=used+selection_ids,
                                                   frozen_policy_declaration=family.declaration,
                                                   **(certification_options or {}))
    return {"status":"PLAN_FROZEN" if certification is None else certification["status"],
            "admission":admission,"selection":selection,"plan":asdict(plan),"certification":certification}


def _empirical_energy_pair_expectation(draws):
    """Exact E||Z-Z'|| for a fixed uniform empirical forecast, chunked in RAM."""
    from scipy.spatial.distance import cdist
    x=np.asarray(draws,float).reshape(len(draws),-1)
    if not len(x) or not np.isfinite(x).all():raise ValueError("Empirical forecast samples must be finite and nonempty.")
    total=0.
    for start in range(0,len(x),128):
        total+=float(cdist(x[start:start+128],x,metric="euclidean").sum())
    return total/(len(x)*len(x))


def _bounded_energy_score(draws, observed, *, pair_expectation=None):
    """Proper energy score of the declared fixed empirical predictive law.

    All ordered sample pairs, including zero-distance self pairs, define the
    empirical product measure. A fixed split-pair Monte Carlo offset would NOT
    be this scoring rule and could favor a worse forecast after infinite audits.
    """
    x=np.asarray(draws,float).reshape(len(draws),-1);y=np.asarray(observed,float).reshape(-1)
    if len(x)<2 or x.shape[1]!=len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Energy scoring needs finite matching observations and two or more joint draws.")
    pair=_empirical_energy_pair_expectation(x) if pair_expectation is None else float(pair_expectation)
    if not np.isfinite(pair) or pair<0:raise ValueError("Pair expectation must be finite and nonnegative.")
    return float((np.linalg.norm(x-y,axis=1).mean()-.5*pair)/np.sqrt(len(y)))


def _observed_outcome(ids,actual,choices,actions,setup_cost=0.):
    """Selected-action truth; no invented common population endpoint."""
    choices=np.asarray(choices,int);g=actual[np.arange(len(choices)),choices].copy()
    wells=np.array([actions[j].wells for j in choices])
    active=wells>0
    if setup_cost:
        if not active.any():raise ValueError("A stop-only plan cannot incur an acquisition setup charge.")
        g[active]-=setup_cost/active.sum()
    return FrozenCandidateEvaluation(tuple(map(str,ids)),choices,g,wells)


def run_model_decision_workflow(model,scaler,ds,config,settings: dict,output_dir=None) -> dict:
    """Concrete model/data workflow used by the CLI, with no outcome-to-policy path.

    Settings declare four lists: calibration_campaigns, admission_campaigns,
    selection_campaigns, certification_campaigns. Each entry has campaign_id and
    integer indices. All lists must contain disjoint compounds AND disjoint root
    source groups to claim independent campaigns in this implementation. Actual
    data from one source produce diagnostics, never a campaign certificate.

    The model predicts from X + frozen legal contexts. Calibration and admission
    score campaign-mean action utilities as fixed vectors. Selection plans are
    generated from independent model draws before their actual outcomes are
    inspected. Certification reruns the selected frozen allocator separately in
    each independent deployment campaign. An optional model-based assurance run
    resamples joint future outcomes of frozen decision states; it cannot override
    an empirical admission failure.
    """
    import json
    from pathlib import Path
    import torch
    from .data import attach_library_context, make_inference_batch
    from .provenance import assert_unseen_outcomes
    from .selection import BudgetCandidate
    from .utility import cosine_utility_samples, enumerate_actions

    def jsonable(x):
        if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
        if isinstance(x,np.ndarray):return jsonable(x.tolist())
        if isinstance(x,np.generic):return jsonable(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    cfg=config if isinstance(config,dict) else vars(config)
    seed=int(settings.get("seed",cfg.get("seed",0)));rng=np.random.default_rng(seed)
    roles=tuple(settings.get("roles",(0,1,2,3)))
    if len(roles)!=4 or len(set(roles))!=4 or any(not isinstance(i,int) or i<0 or i>=ds.Y.shape[1] for i in roles):
        raise ValueError("Workflow requires four existing distinct X/Z1/Z2/V physical roles.")
    sample_count=int(settings.get("samples",cfg.get("samples",2000)))
    if sample_count<2:raise ValueError("At least two predictive utility draws are required.")
    working=scaler.transform(ds)
    if getattr(model,"library_bank",None) is not None:working=attach_library_context(working,model.library_bank)
    param=next(model.parameters());actions=enumerate_actions((0,1));was_training=model.training
    source_sets={};all_ids=set();stages={};cache={};disjoint_sources=True
    for stage in ("calibration","admission","selection","certification"):
        entries=settings.get(stage+"_campaigns",[]);stages[stage]=entries
        for entry in entries:
            ix=np.asarray(entry["indices"])
            if ix.ndim!=1 or not len(ix) or not np.issubdtype(ix.dtype,np.integer) or len(set(ix.tolist()))!=len(ix) or np.any(ix<0) or np.any(ix>=len(ds)):
                raise ValueError("Campaign indices must be distinct valid integers.")
            name=str(entry["campaign_id"])
            if not name or name in source_sets:raise ValueError("Campaign identifiers must be globally unique.")
            ids=set(map(str,ds.ids[ix]));assert_unseen_outcomes(model,ds.ids[ix])
            if all_ids&ids:raise ValueError("Workflow stages/campaigns reuse compounds.")
            all_ids|=ids
            sources=set(ds.groups[ix][:,roles,0].ravel().tolist())
            if any(sources&previous for previous in source_sets.values()):disjoint_sources=False
            source_sets[name]=sources
    fitting_ids=set(map(str,getattr(model,"fitting_ids",())))
    if getattr(model,"fitting_sources_known",False):
        fit_sources=set(map(int,getattr(model,"fitting_source_groups",())))
        source_provenance_known=bool(fit_sources) and not getattr(model,"pretraining_sources_unknown",False)
    elif fitting_ids and fitting_ids.issubset(set(map(str,ds.ids))):
        # A complete current dataset can reconstruct a legacy checkpoint's
        # source set, but an evaluation-only dataset cannot prove novelty by
        # failing to contain fitting rows. fitting_ids includes pretraining IDs.
        fit_rows=np.flatnonzero(np.isin(ds.ids,list(fitting_ids)))
        fit_sources=set(ds.groups[fit_rows,:,0].ravel().tolist())
        source_provenance_known=bool(fit_sources) and not getattr(model,"pretraining_sources_unknown",False)
    else:
        fit_sources=set();source_provenance_known=False
    unseen_sources=source_provenance_known and not any(values&fit_sources for values in source_sets.values())
    require_unseen_sources=bool(settings.get("require_unseen_sources",True))
    independent=bool(settings.get("assume_iid_campaigns",False)) and disjoint_sources and (unseen_sources or not require_unseen_sources)
    contract=RiskContract(**settings["contract"]) if "contract" in settings else None
    family=FrozenBudgetFamily.from_dict(settings["family"]) if "family" in settings else None
    cost=float(settings.get("cost_per_well",.01));gain_support=tuple(settings.get("gain_support",(-1.02,1.0)))
    budget=settings.get("planner",{})

    @torch.no_grad()
    def predict(entry,read_actual=True):
        name=str(entry["campaign_id"])
        if name in cache:
            if read_actual and cache[name]["actual"] is None:reveal_actual(entry,cache[name])
            return cache[name]
        raw_indices=np.asarray(entry["indices"])
        if raw_indices.ndim!=1 or not len(raw_indices) or not np.issubdtype(raw_indices.dtype,np.integer) or len(set(raw_indices.tolist()))!=len(raw_indices) or np.any(raw_indices<0) or np.any(raw_indices>=len(ds)):
            raise ValueError("Campaign/diagnostic indices must be unique, nonempty, in-range integers.")
        ix=raw_indices.astype(int)
        assert_unseen_outcomes(model,ds.ids[ix])
        if not ds.well_mask[ix][:,roles].all():
            raise ValueError("Four-role utility planning requires four present physical roles, not padded candidate wells.")
        if not ds.observed_mask[ix,roles[0]].all():
            raise ValueError("Four-role utility planning requires observed initial X; finite missing placeholders are not observations.")
        inputs=make_inference_batch(working,ix,context_indices=(roles[0],),target_indices=roles[1:],
                       reference_access=settings.get("reference_access",cfg.get("reference_access","observed_only")),device=param.device,dtype=param.dtype)
        distribution=model(inputs);x=ds.Y[ix,roles[0]].copy();blocks=[]
        if not np.isfinite(x).all():
            raise ValueError("A missing initial measurement needs an explicit zero-context task; this four-role utility workflow requires observed finite X.")
        for start in range(0,sample_count,32):
            generator=torch.Generator(device=param.device).manual_seed(int(rng.integers(0,2**63-1)))
            y=scaler.inverse_y(distribution.sample_joint(min(32,sample_count-start),generator)).cpu().double().numpy()
            blocks.append(cosine_utility_samples(x,y,actions,2,cost_per_well=cost).samples)
        predicted=UtilityResult.from_samples(actions,np.concatenate(blocks),np.array([a.wells*cost for a in actions]),utility_name="half_cosine_gain")
        result={"ids":ds.ids[ix],"prediction":predicted,"actual":None,"distribution":distribution,"x":x}
        if read_actual:reveal_actual(entry,result)
        cache[name]=result;return result
    def reveal_actual(entry,state):
        ix=np.asarray(entry["indices"],int);future=ds.Y[ix][:,roles[1:]]
        available=ds.observed_mask[ix][:,roles[1:]] & np.isfinite(future).all(axis=-1)
        missing=settings.get("missing_outcome","error")
        if missing not in ("error","worst"):raise ValueError("missing_outcome must be 'error' or predeclared 'worst'.")
        if not available.all() and missing=="error":raise ValueError("Future outcomes are missing; a worst-completion rule must be declared before evaluation.")
        safe=np.where(available[...,None],future,0.)
        actual=cosine_utility_samples(state["x"],safe[None],actions,2,cost_per_well=cost).samples[0]
        failed=np.zeros_like(actual,bool)
        for j,a in enumerate(actions):
            if a.wells:
                failed[:,j]=~available[:,2] | ~available[:,a.target_indices].all(axis=1)
                actual[failed[:,j],j]=-1-a.wells*cost
        state["actual"]=actual;state["technical_failure_mask"]=failed
        return state["actual"]
    report={"workflow_version":"R2_campaign_workflow","status":"NOT_EVALUATED","independent_campaigns_eligible":independent,
            "source_disjoint":disjoint_sources,"unseen_fitting_sources":unseen_sources,
            "fitting_source_provenance_known":source_provenance_known,
            "require_unseen_sources":require_unseen_sources,"settings":settings,"is_prospective_certification":False}
    try:
        model.eval()
        # Model admission needs both calibration and independent audit cohorts.
        if stages["calibration"] and stages["admission"] and "admission_requirements" in settings:
            cal=[predict(e) for e in stages["calibration"]];audit=[predict(e) for e in stages["admission"]]
            cy=np.stack([e["actual"].mean(axis=0)[1:] for e in cal])
            cm=np.stack([e["prediction"].mean.mean(axis=0)[1:] for e in cal])
            cs=np.stack([np.maximum(e["prediction"].samples.mean(axis=1)[:,1:].std(axis=0,ddof=1),1e-8) for e in cal])
            calibrator=SplitConformalUtility.fit(cy,cm,[e["campaign_id"] for e in stages["calibration"]],
                          alpha=float(settings.get("conformal_alpha",.1)),predicted_scale=cs,
                          training_ids=tuple(getattr(model,"fitting_ids",())))
            ay=np.stack([e["actual"].mean(axis=0)[1:] for e in audit]);am=np.stack([e["prediction"].mean.mean(axis=0)[1:] for e in audit])
            az=[e["prediction"].samples.mean(axis=1)[:,1:] for e in audit]
            asc=np.stack([np.maximum(z.std(axis=0,ddof=1),1e-8) for z in az])
            # Baseline predictive distribution is frozen from calibration-stage
            # model draws, not estimated using any independent audit outcome.
            baseline=np.concatenate([e["prediction"].samples.mean(axis=1)[:,1:] for e in cal])
            model_scores=[_bounded_energy_score(z,y) for z,y in zip(az,ay)]
            baseline_pair=_empirical_energy_pair_expectation(baseline)
            baseline_scores=[_bounded_energy_score(baseline,y,pair_expectation=baseline_pair) for y in ay]
            admission=model_admission(calibrator,ay,am,asc,[e["campaign_id"] for e in stages["admission"]],
                         model_scores,baseline_scores,AdmissionRequirements(**settings["admission_requirements"]),
                         fitting_ids=tuple(getattr(model,"fitting_ids",())),independent_units=independent,
                         proper_score_name="Energy score for joint campaign-mean action utilities",
                         proper_score_is_declared_bounded=True)
            # Retain actual compound provenance as well as statistical-unit IDs.
            admission["used_compound_ids"]=sorted(set(i for e in cal+audit for i in e["ids"]))
            report["admission"]=admission
        else:
            report["admission"]={"admitted":False,"reason":"Independent conformal-calibration and admission campaigns/requirements were not supplied."}
        if report["admission"]["admitted"] and family is not None and contract is not None and stages["selection"]:
            outcomes={c.name:[] for c in family.candidates}
            for entry in stages["selection"]:
                state=predict(entry,read_actual=False);candidate_plans={}
                for c in family.candidates:
                    options=dict(budget);options.update(dict(c.rule_parameters));options[family.axis_name]=c.axis_value
                    candidate_plans[c.name]=allocate_actions(state["prediction"],**options)
                reveal_actual(entry,state)
                for c in family.candidates:
                    p=candidate_plans[c.name]
                    o=_observed_outcome(state["ids"],state["actual"],p.action_indices,actions,p.setup_cost)
                    # FPR/sensitivity require a single declared acquisition action
                    # for ALL active objects, otherwise no coherent binary
                    # population reference has been supplied.
                    if contract.max_fpr is not None or contract.min_sensitivity is not None:
                        endpoint=settings.get("common_endpoint_action")
                        names=[a.name for a in actions]
                        if endpoint not in names or endpoint=="stop":raise ValueError("Declare a common acquisition endpoint for FPR/sensitivity.")
                        e=names.index(endpoint)
                        if any(j not in (0,e) for j in p.action_indices):raise ValueError("Variable actions do not match the declared common population endpoint.")
                        full_gain=state["actual"][:,e]-p.setup_cost/max(1,p.expected_activations)
                        o=FrozenCandidateEvaluation(o.unit_ids,o.actions,o.gain,o.wells,full_gain<=0,
                                                     full_gain>=.005,endpoint)
                    outcomes[c.name].append(CampaignOutcome(entry["campaign_id"],o))
            selection=select_campaign_fixed_sequence(family,outcomes,contract,maximum_campaign_size=int(settings["maximum_campaign_size"]),
                        gain_support=gain_support,assume_iid_campaigns=independent,
                        previously_used_ids=tuple(getattr(model,"fitting_ids",()))+tuple(report["admission"]["used_compound_ids"]))
            report["selection"]=selection;chosen=selection["selected_candidate"]
            if chosen is not None:
                c=next(c for c in family.candidates if c.name==chosen);options=dict(budget);options.update(dict(c.rule_parameters));options[family.axis_name]=c.axis_value
                evaluation=[];plans=[]
                used=tuple(getattr(model,"fitting_ids",()))+tuple(report["admission"]["used_compound_ids"])+tuple(i for x in outcomes[chosen] for i in x.outcome.unit_ids)
                controller=CampaignSequenceController(contract,maximum_campaign_size=int(settings["maximum_campaign_size"]),
                       gain_support=gain_support,assume_iid_campaigns=independent,frozen_policy_declaration=family.declaration,
                       previously_used_ids=used,minimum_campaigns=int(settings.get("minimum_campaigns",2)))
                for entry in stages["certification"]:
                    state=predict(entry,read_actual=False);p=allocate_actions(state["prediction"],**options)
                    reveal_actual(entry,state)
                    o=_observed_outcome(state["ids"],state["actual"],p.action_indices,actions,p.setup_cost)
                    if contract.max_fpr is not None or contract.min_sensitivity is not None:
                        e=[a.name for a in actions].index(settings["common_endpoint_action"])
                        if any(j not in (0,e) for j in p.action_indices):raise ValueError("Selected policy violates common endpoint restriction.")
                        full_gain=state["actual"][:,e]-p.setup_cost/max(1,p.expected_activations)
                        o=FrozenCandidateEvaluation(o.unit_ids,o.actions,o.gain,o.wells,full_gain<=0,full_gain>=.005,settings["common_endpoint_action"])
                    evaluation.append(CampaignOutcome(entry["campaign_id"],o));plans.append(asdict(p))
                    result=controller.update(evaluation[-1])
                    if result["status"]!="CONTINUE":break
                report["frozen_plans"]=plans
                if evaluation:
                    report["certification"]=controller.result
                    report["unopened_certification_campaigns"]=[e["campaign_id"] for e in stages["certification"][len(evaluation):]]
                    report["status"]=report["certification"]["status"]
                else:report["status"]="PLAN_SELECTED_NO_CERTIFICATION_CAMPAIGNS"
            else:report["status"]="NO_RULE_ADMITTED"
        else:report["status"]="MODEL_NOT_ADMITTED"
        # Separate diagnostic execution never overrides formal admission status.
        diagnostic=settings.get("diagnostic_indices")
        if diagnostic is not None:
            entry={"campaign_id":"development_diagnostic","indices":diagnostic};state=predict(entry,read_actual=False)
            options=dict(budget)
            if "budget" not in options:options["budget"]=int(np.ceil(.2*len(diagnostic)))
            p=allocate_actions(state["prediction"],**options)
            reveal_actual(entry,state)
            actual=_observed_outcome(state["ids"],state["actual"],p.action_indices,actions,p.setup_cost).gain
            report["development_diagnostic"]={"plan":asdict(p),"observed_mean_net_gain":float(actual.mean()),
                "observed_gains":actual,"compound_ids":state["ids"],"is_authorized":False,
                "technical_failures_by_action":state["technical_failure_mask"].sum(axis=0),
                "reason":"Independent non-certifying development diagnostic; admission result is unchanged."}
            if settings.get("assurance") is not None:
                report["model_conditional_assurance"]=world_model_workflow_assurance(state["distribution"],scaler,state["x"],
                         state["prediction"],contract=contract,planner_options=options,
                         gain_support=gain_support,seed=seed+10001,**settings["assurance"])
            if settings.get("design_assurance") is not None:
                if family is None or contract is None or "admission_requirements" not in settings:
                    raise ValueError("Full-design assurance needs frozen family, contract, and admission requirements.")
                report["full_design_assurance"]=world_model_design_assurance(state["distribution"],scaler,state["x"],
                         state["prediction"],family=family,contract=contract,planner_options=options,
                         admission_requirements=AdmissionRequirements(**settings["admission_requirements"]),
                         gain_support=gain_support,seed=seed+20001,**settings["design_assurance"])
        if output_dir is not None:
            path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
            (path/"decision_workflow.json").write_text(json.dumps(jsonable(report),indent=2,ensure_ascii=False)+"\n")
        return jsonable(report)
    finally:model.train(was_training)


def world_model_workflow_assurance(distribution,scaler,known_x,prediction:UtilityResult,*,
                                   contract:RiskContract|None,planner_options:dict,
                                   gain_support,seed=0,simulations=100,campaign_counts=(10,25,50),
                                   minimum_campaigns=2,target_assurance=None,common_endpoint_action=None):
    """Joint world-model -> fixed cohort allocator -> full sequential contract.

    Campaign decision states are fixed at the supplied observed X and legal
    references. Each simulated campaign independently samples all shared model
    latents together. Thus assurance is CONDITIONAL on these decision states and
    the fitted model, not assurance for new compounds/sites. A failed empirical
    model admission is not rescued by this model-conditional calculation.
    """
    import torch
    from .assurance import SimulatedCampaign,campaign_assurance
    from .utility import cosine_utility_samples
    if contract is None:raise ValueError("Model-based assurance requires the full explicitly declared contract.")
    n=len(known_x);actions=prediction.actions
    names=[a.name for a in actions];endpoint=None
    if contract.max_fpr is not None or contract.min_sensitivity is not None:
        if common_endpoint_action not in names or common_endpoint_action=="stop":
            raise ValueError("Full FPR/sensitivity assurance requires the common population acquisition endpoint.")
        endpoint=names.index(common_endpoint_action)
    def simulator(count,rng):
        outcomes=[]
        for c in range(count):
            generator=torch.Generator(device=distribution.mean.device).manual_seed(int(rng.integers(0,2**63-1)))
            with torch.no_grad():future=scaler.inverse_y(distribution.sample_joint(1,generator)).detach().cpu().double().numpy()
            actual=cosine_utility_samples(known_x,future,actions,distribution.mean.shape[1]-1).samples[0]
            # Costs can differ by action; recalculate from gross gains to keep
            # the exact frozen predictive task cost rather than hardcoding .01.
            default=np.array([a.wells*.01 for a in actions])
            actual=actual+default-np.broadcast_to(prediction.costs,actual.shape)
            outcomes.append(actual)
        return SimulatedCampaign({"prediction":prediction,"campaigns":count},outcomes)
    def policy(state):
        return [allocate_actions(state["prediction"],**planner_options) for _ in range(state["campaigns"])]
    def evaluate(outcomes,plans):
        campaigns=[]
        for i,(y,p) in enumerate(zip(outcomes,plans)):
            o=_observed_outcome([f"model_{i}_{j}" for j in range(n)],y,p.action_indices,actions,p.setup_cost)
            if endpoint is not None:
                if any(j not in (0,endpoint) for j in p.action_indices):
                    raise ValueError("Assurance policy selects actions outside the common population endpoint.")
                full_gain=y[:,endpoint]-p.setup_cost/max(1,p.expected_activations)
                o=FrozenCandidateEvaluation(o.unit_ids,o.actions,o.gain,o.wells,full_gain<=0,
                                            full_gain>=.005,common_endpoint_action)
            campaigns.append(CampaignOutcome(f"model_campaign_{i}",o))
        result=evaluate_campaign_sequence(campaigns,contract,maximum_campaign_size=n,gain_support=gain_support,
                       assume_iid_campaigns=True,minimum_campaigns=minimum_campaigns,
                       frozen_policy_declaration="Frozen model-predicted exact cohort allocator on fixed legal decision states")
        result["stopping_n"]=result["stopping_campaigns"]
        return result
    result=campaign_assurance(simulator,policy,evaluate,sample_sizes=campaign_counts,simulations=simulations,seed=seed,
              model_description="Joint fitted measurement Gaussian; independent fresh latent draws per fixed-state campaign",
              policy_declaration="Frozen cohort allocator and declared contract; no simulator outcome enters decision state",
              target_assurance=target_assurance)
    result["sample_size_unit"]="Independent model-generated campaigns, not compounds"
    result["empirical_admission_overridden"]=False
    result["calibration_and_selection_refit_in_simulation"]=False
    result["scope"]="Conditional design assurance for a frozen fitted model and frozen planner; empirical calibration/admission/selection precede this calculation"
    return result


def world_model_design_assurance(distribution,scaler,known_x,prediction:UtilityResult,*,
                                 family:FrozenBudgetFamily,contract:RiskContract,
                                 planner_options:dict,admission_requirements:AdmissionRequirements,
                                 gain_support,calibration_campaigns:int,admission_campaigns:int,
                                 selection_campaigns:int,certification_campaign_counts:Sequence[int],
                                 simulations=100,conformal_alpha=.1,minimum_campaigns=2,
                                 common_endpoint_action=None,seed=0,alpha=.05,target_assurance=None):
    """Refit calibration/admission/LTT and certify inside every model replicate.

    This is a concrete full DESIGN pipeline, not a supplied pass-flag callback:
    joint measurement draws -> action outcomes -> fresh split-conformal fit ->
    independent model admission -> fixed-family cohort plans -> campaign LTT ->
    sequential certification with unopened simulated campaigns after stopping.
    The fitted world model and observed decision states remain fixed. It does
    not resimulate model training or establish transfer to an unobserved site.
    """
    import torch
    from .utility import cosine_utility_samples
    counts=tuple(certification_campaign_counts)
    if any(not isinstance(k,(int,np.integer)) or k<1 for k in (calibration_campaigns,admission_campaigns,selection_campaigns,simulations,*counts)) or not counts or len(set(counts))!=len(counts):
        raise ValueError("All fixed stage counts and simulation counts must be positive integers.")
    if not 0<alpha<1 or not 0<conformal_alpha<1:raise ValueError("Assurance and conformal alpha must be in (0,1).")
    if target_assurance is not None and not 0<target_assurance<1:raise ValueError("target_assurance must be in (0,1).")
    if prediction.samples.shape[0]<2:raise ValueError("Joint forecast draws are required for proper-score admission.")
    n=len(known_x);actions=prediction.actions;names=[a.name for a in actions]
    endpoint=None
    if contract.max_fpr is not None or contract.min_sensitivity is not None:
        if common_endpoint_action not in names or common_endpoint_action=="stop":
            raise ValueError("Full-design FPR/sensitivity require a declared common population action.")
        endpoint=names.index(common_endpoint_action)
    active_columns=[j for j,a in enumerate(actions) if a.wells]
    mean=prediction.mean.mean(axis=0)[active_columns]
    forecast=prediction.samples.mean(axis=1)[:,active_columns]
    forecast_pair=_empirical_energy_pair_expectation(forecast)
    scale=np.maximum(forecast.std(axis=0,ddof=1),1e-8)
    stage_sizes={"calibration":int(calibration_campaigns),"admission":int(admission_campaigns),"selection":int(selection_campaigns)}
    streams=iter(np.random.SeedSequence(seed).spawn(len(counts)*simulations));rows=[]
    for maximum in counts:
        frequencies={key:0 for key in ("model_not_admitted","no_rule_admitted","certification_continue","certification_futility","pass")}
        stops=[];opened=[];replicate_reports=[]
        for repetition in range(simulations):
            rng=np.random.default_rng(next(streams));prefix=f"simulation_{maximum}_{repetition}"
            def draw_outcome():
                generator=torch.Generator(device=distribution.mean.device).manual_seed(int(rng.integers(0,2**63-1)))
                with torch.no_grad():
                    future=scaler.inverse_y(distribution.sample_joint(1,generator)).detach().cpu().double().numpy()
                actual=cosine_utility_samples(known_x,future,actions,distribution.mean.shape[1]-1).samples[0]
                return actual+np.array([a.wells*.01 for a in actions])-np.broadcast_to(prediction.costs,actual.shape)
            def campaign(stage,i,actual,plan):
                ids=[f"{prefix}_{stage}_{i}_{j}" for j in range(n)]
                o=_observed_outcome(ids,actual,plan.action_indices,actions,plan.setup_cost)
                if endpoint is not None:
                    if any(j not in (0,endpoint) for j in plan.action_indices):
                        raise ValueError("Frozen candidate plan is incompatible with its common population endpoint.")
                    y=actual[:,endpoint]-plan.setup_cost/max(1,plan.expected_activations)
                    o=FrozenCandidateEvaluation(o.unit_ids,o.actions,o.gain,o.wells,y<=0,y>=.005,common_endpoint_action)
                return CampaignOutcome(f"{prefix}_{stage}_{i}",o)
            cy=np.stack([draw_outcome().mean(axis=0)[active_columns] for _ in range(calibration_campaigns)])
            cal_ids=[f"{prefix}_conformal_{i}" for i in range(calibration_campaigns)]
            calibrator=SplitConformalUtility.fit(cy,np.broadcast_to(mean,cy.shape),cal_ids,
                            alpha=conformal_alpha,predicted_scale=np.broadcast_to(scale,cy.shape),training_ids=["fixed_fitted_world_model"])
            ay=np.stack([draw_outcome().mean(axis=0)[active_columns] for _ in range(admission_campaigns)])
            audit_ids=[f"{prefix}_admission_{i}" for i in range(admission_campaigns)]
            # Baseline is estimated only on simulated calibration outcomes.
            # Its empirical predictive distribution is frozen before admission.
            baseline=cy if len(cy)>=2 else np.repeat(cy,2,axis=0)
            baseline_pair=_empirical_energy_pair_expectation(baseline)
            admission=model_admission(calibrator,ay,np.broadcast_to(mean,ay.shape),np.broadcast_to(scale,ay.shape),audit_ids,
                         [_bounded_energy_score(forecast,y,pair_expectation=forecast_pair) for y in ay],
                         [_bounded_energy_score(baseline,y,pair_expectation=baseline_pair) for y in ay],
                         admission_requirements,fitting_ids=["fixed_fitted_world_model"],independent_units=True,
                         proper_score_name="Joint campaign-mean utility energy score",proper_score_is_declared_bounded=True)
            minimal={"replicate":repetition,"admission_passed":admission["admitted"],"selection_invoked":False,"certification_invoked":False}
            consumed=calibration_campaigns+admission_campaigns
            if not admission["admitted"]:
                frequencies["model_not_admitted"]+=1;stops.append(0);opened.append(consumed);replicate_reports.append(minimal);continue
            plans={}
            for c in family.candidates:
                options=dict(planner_options);options.update(dict(c.rule_parameters));options[family.axis_name]=c.axis_value
                plans[c.name]=allocate_actions(prediction,**options)
            selection_actual=[draw_outcome() for _ in range(selection_campaigns)];consumed+=selection_campaigns
            evaluations={name:[campaign("selection",i,y,plan) for i,y in enumerate(selection_actual)] for name,plan in plans.items()}
            minimal["selection_invoked"]=True
            if selection_campaigns<2:
                selection={"selected_candidate":None,"reason":"Fewer than two independent simulated selection campaigns"}
            else:
                selection=select_campaign_fixed_sequence(family,evaluations,contract,maximum_campaign_size=n,
                                      gain_support=gain_support,assume_iid_campaigns=True)
            chosen=selection["selected_candidate"];minimal["selected_candidate"]=chosen
            if chosen is None:
                frequencies["no_rule_admitted"]+=1;stops.append(0);opened.append(consumed);replicate_reports.append(minimal);continue
            controller=CampaignSequenceController(contract,maximum_campaign_size=n,gain_support=gain_support,
                            assume_iid_campaigns=True,minimum_campaigns=minimum_campaigns,frozen_policy_declaration=family.declaration)
            minimal["certification_invoked"]=True
            for i in range(maximum):
                # Plan is frozen before this independent actual outcome draw.
                outcome=draw_outcome();consumed+=1
                result=controller.update(campaign("certification",i,outcome,plans[chosen]))
                if result["status"]!="CONTINUE":break
            label={"PASS":"pass","FUTILITY":"certification_futility","CONTINUE":"certification_continue"}[result["status"]]
            frequencies[label]+=1;stops.append(result["stopping_campaigns"]);opened.append(consumed)
            minimal["certification_status"]=result["status"];minimal["stopping_campaigns"]=result["stopping_campaigns"]
            replicate_reports.append(minimal)
        passes=frequencies["pass"]
        interval=clopper_pearson(passes,simulations,alpha=alpha/len(counts),side="two-sided")
        rows.append({"maximum_certification_campaigns":int(maximum),"simulations":int(simulations),"passes":passes,
             "model_conditional_assurance":passes/simulations,"monte_carlo_interval":list(interval),
             "stage_outcome_counts":frequencies,"mean_opened_campaigns":float(np.mean(opened)),
             "mean_certification_campaigns":float(np.mean(stops)),"replicates":replicate_reports})
    supported=[r["maximum_certification_campaigns"] for r in rows if target_assurance is not None and r["monte_carlo_interval"][0]>=target_assurance]
    return {"mode":"FULL_MODEL_CONDITIONAL_DESIGN_ASSURANCE","stage_sizes":stage_sizes,
            "calibration_and_selection_refit_in_simulation":True,"model_training_refit_in_simulation":False,
            "results":rows,"seed":seed,"monte_carlo_familywise_alpha":alpha,"target_assurance":target_assurance,
            "model_suggested_min_certification_campaigns":min(supported) if supported else None,
            "sample_size_unit":"Independent model-generated complete campaigns with fixed observed decision states",
            "empirical_admission_overridden":False,"is_prospective_certification":False,
            "scope":"Entire calibration/admission/LTT/sequential-design pipeline under the fixed fitted joint model; not empirical certification, model-training assurance, or evidence of new-site transfer"}
