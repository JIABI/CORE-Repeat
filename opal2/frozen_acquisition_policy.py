"""Standalone contract utilities for the frozen measurement-policy recipe.

This module does not train, load, replace, or certify an experimental model.
The frozen policy consumes E[Gamma] and P(Gamma <= 0) from a declared predictor.
Gamma already includes its declared ADD_TWO action cost. Reference acquisition
costs must be recorded separately, not silently subtracted twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Iterable, Mapping

import numpy as np


FROZEN_RISK_PENALTY = .2
FROZEN_ACTION = 'ADD_TWO'


@dataclass(frozen=True)
class FrozenCohortPlan:
    ids: tuple[str, ...]
    scores: tuple[float, ...]
    selected_mask: tuple[bool, ...]
    selected_ids_in_rank_order: tuple[str, ...]
    activation_budget: int
    added_action_wells: int
    risk_penalty: float = field(default=FROZEN_RISK_PENALTY,init=False)
    assignment_type: str = field(default='cohort_dependent_fixed_budget_topk',init=False)
    tie_rule: str = field(default='ascending_unique_stable_id',init=False)
    certified: bool = field(default=False,init=False)

    def request_iid_clopper_pearson_certificate(self):
        """Block the iid-binomial shortcut for cohort-coupled top-k actions."""
        raise ValueError('A cohort-dependent top-k plan cannot be certified by '
            'plugging its pooled counts into iid Clopper--Pearson. A separately '
            'specified evaluation design and dependence-valid procedure are required.')


def _unique_ids(ids: Iterable[str]) -> tuple[str, ...]:
    values=tuple(ids)
    if not values or any(not isinstance(v,str) or not v for v in values):
        raise ValueError('A nonempty sequence of nonempty stable string IDs is required')
    if len(set(values)) != len(values):
        raise ValueError('Stable IDs must be unique')
    return values


def select_frozen_cohort_plan(ids, expected_net_gamma, p_null, activation_budget):
    """Select exactly k by m-.2p; selection uses no realized outcome.

    The integer activation budget must be declared externally. It is not a
    risk-certified budget or automatically a fixed proportion of a new cohort.
    """
    names=_unique_ids(ids);n=len(names)
    mean=np.asarray(expected_net_gamma,dtype=float);prob=np.asarray(p_null,dtype=float)
    if mean.shape!=(n,) or prob.shape!=(n,):raise ValueError('Inputs must be aligned one-dimensional arrays')
    if not np.isfinite(mean).all() or not np.isfinite(prob).all():raise ValueError('Scores must be finite; missing outcomes need a declared completion')
    if np.any((prob<0)|(prob>1)):raise ValueError('PNULL must lie in [0,1]')
    if isinstance(activation_budget,(bool,np.bool_)) or not isinstance(activation_budget,(int,np.integer)):
        raise ValueError('The activation budget must be an integer, not rounded by this utility')
    k=int(activation_budget)
    if not 0<=k<=n:raise ValueError('Activation budget outside cohort size')
    score=mean-FROZEN_RISK_PENALTY*prob
    if not np.isfinite(score).all():raise ValueError('Score arithmetic overflow')
    order=np.lexsort((np.asarray(names),-score));mask=np.zeros(n,bool);mask[order[:k]]=True
    return FrozenCohortPlan(names,tuple(map(float,score)),tuple(map(bool,mask)),
        tuple(names[i] for i in order[:k]),k,2*k)


def historical_two_half_budgets(outer_n: int, half0_n: int) -> tuple[int,int]:
    """Exact earlier DEV budget, not a newly chosen deployment fraction.

    total = floor(.25 * N_outer) // 2 additional ADD_TWO activations;
    k0 = floor(total * N_half0 / N_outer), k1 = total-k0.
    The .25 denotes the available added-well count per original outer object.
    Integer arithmetic avoids floating rounding and preserves the old14/15 split.
    """
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in (outer_n,half0_n)):
        raise ValueError('Cohort sizes must be integers')
    if not 0<half0_n<outer_n:raise ValueError('Two nonempty halves are required')
    total=(int(outer_n)//4)//2
    k0=total*int(half0_n)//int(outer_n)
    return k0,total-k0


def select_from_extra_well_budget(ids, expected_net_gamma, p_null, extra_well_budget):
    """Convert a declared physical action-well budget to complete ADD_TWO actions.

    An odd remaining well cannot fund a partial action. Surplus budget does not
    create additional objects. Reference and verification wells are separate.
    """
    names = _unique_ids(ids)
    if (isinstance(extra_well_budget, (bool, np.bool_))
            or not isinstance(extra_well_budget, (int, np.integer))
            or extra_well_budget < 0):
        raise ValueError('A nonnegative integer extra-well budget is required')
    return select_frozen_cohort_plan(names, expected_net_gamma, p_null,
        min(len(names), int(extra_well_budget)//2))


class DataRole(str,Enum):
    MODEL_FIT='MODEL_FIT'
    REF_FIT='REF_FIT'
    DIST_CAL='DIST_CAL'
    POLICY_CAL='POLICY_CAL'
    EVAL='EVAL'


class OutcomeStage(str,Enum):
    MODEL_TRAIN='MODEL_TRAIN'
    REFERENCE_FIT='REFERENCE_FIT'
    DISTRIBUTION_CALIBRATION='DISTRIBUTION_CALIBRATION'
    POLICY_SELECTION='POLICY_SELECTION'


@dataclass(frozen=True)
class OutcomeUse:
    """Document actual future-outcome donors, not ordinary decision-time X input."""
    stage: OutcomeStage
    future_outcome_ids: tuple[str,...]
    reference_fit_frozen: bool = False
    predictor_frozen: bool = False


def validate_reference_role_isolation(role_ids: Mapping[str,Iterable[str]],
    group_by_id: Mapping[str,str], outcome_uses: Iterable[OutcomeUse]):
    """Validate declared chemical-group roles and stage-specific outcome access.

    EVAL initial measurements/metadata are legal inputs; EVAL future outcomes
    are never donors. DIST_CAL outcomes may fit the distribution after reference
    fitting is frozen. POLICY_CAL may select a policy after the predictor is
    frozen, but cannot enter that predictor's model/reference/distribution fit.
    This validates a manifest, not inaccessible historical runtime provenance.
    """
    roles={DataRole(k):tuple(v) for k,v in role_ids.items()}
    if set(roles)!=set(DataRole):raise ValueError('All five roles must be explicitly declared, possibly empty')
    owner={};group_owner={}
    for role,members in roles.items():
        if len(set(members))!=len(members):raise ValueError(f'Duplicate IDs within {role.value}')
        for object_id in members:
            if not isinstance(object_id,str) or not object_id:raise ValueError('Invalid object ID')
            if object_id in owner:raise ValueError('Object ID occupies multiple roles')
            if object_id not in group_by_id or not isinstance(group_by_id[object_id],str) or not group_by_id[object_id]:
                raise ValueError('Every declared object requires a nonempty chemical/group identity')
            group=group_by_id[object_id]
            if group in group_owner and group_owner[group]!=role:raise ValueError('A chemical/group identity crosses roles')
            owner[object_id]=role;group_owner[group]=role
    allowed={OutcomeStage.MODEL_TRAIN:DataRole.MODEL_FIT,
        OutcomeStage.REFERENCE_FIT:DataRole.REF_FIT,
        OutcomeStage.DISTRIBUTION_CALIBRATION:DataRole.DIST_CAL,
        OutcomeStage.POLICY_SELECTION:DataRole.POLICY_CAL}
    uses=[]
    for declaration in outcome_uses:
        stage=OutcomeStage(declaration.stage)
        donors=tuple(declaration.future_outcome_ids)
        if len(set(donors))!=len(donors):raise ValueError('Duplicate future-outcome donor IDs in stage declaration')
        for donor in donors:
            if donor not in owner:raise ValueError('Undeclared future-outcome donor')
            if owner[donor]==DataRole.EVAL:raise ValueError('EVAL future outcomes cannot feed any fitting or policy-selection stage')
            if owner[donor]!=allowed[stage]:raise ValueError(f'{stage.value} cannot consume future outcomes assigned to {owner[donor].value}')
        if stage==OutcomeStage.DISTRIBUTION_CALIBRATION and not declaration.reference_fit_frozen:
            raise ValueError('Reference fitting must be frozen before DIST_CAL outcomes calibrate the distribution')
        if stage==OutcomeStage.POLICY_SELECTION and not declaration.predictor_frozen:
            raise ValueError('Predictor must be frozen before POLICY_CAL outcomes select a policy')
        uses.append(dict(stage=stage.value,donor_count=len(donors)))
    return dict(valid_declared_role_isolation=True,role_counts={r.value:len(x) for r,x in roles.items()},
        stages=uses,validation_scope='Declared future-outcome access and chemical/group identities only',
        eval_decision_time_features_allowed=True,formal_statistical_certification=False)


def apply_optional_output_addon(frozen_core_output, *, correction=None, support=None, enabled=False):
    """Untrained optional hook, NOT a biological model or an existing GENERIC arm.

    Always start from the supplied unchanged frozen-core output. Disabled or
    entirely unsupported paths do not even evaluate a lazy correction. There
    is no normalization, clipping, or shared-weight update in this wrapper.
    Probability or geometry validity of supported corrections needs its own
    task-specific implementation; this hook alone does not establish it.
    """
    base=np.asarray(frozen_core_output)
    if base.ndim<1 or not np.issubdtype(base.dtype,np.floating) or not np.isfinite(base).all():
        raise ValueError('Frozen core output must be a finite floating array')
    out=base.copy()
    if not enabled:return out
    mask=np.asarray(support)
    if mask.dtype!=np.bool_ or mask.shape!=(len(base),):raise ValueError('Explicit Boolean per-object support is required')
    if not mask.any():return out
    if correction is None:raise ValueError('Supported optional corrections require an explicit implementation')
    delta=np.asarray(correction() if callable(correction) else correction)
    if delta.shape!=base.shape or not np.issubdtype(delta.dtype,np.floating) or not np.isfinite(delta).all():
        raise ValueError('Correction must be a finite aligned floating array')
    out[mask]=base[mask]+delta[mask]
    if not np.isfinite(out).all():raise ValueError('Correction overflow')
    return out


class MeasurementPurpose(str,Enum):
    INITIAL='INITIAL'
    REFERENCE='REFERENCE'
    ACTION='ACTION'
    VERIFICATION='VERIFICATION'


@dataclass(frozen=True)
class MeasurementUse:
    measurement_id: str
    object_id: str
    purpose: MeasurementPurpose
    new_in_campaign: bool
    cost_amount: float
    cost_unit: str


def account_measurements(uses: Iterable[MeasurementUse]):
    """Deduplicate physical wells across uses; old references are not new spend.

    new_in_campaign describes whether a well existed BEFORE campaign start;
    it must not change to False merely because a new well is reused later.
    No currency/utility conversion or adjustment of Gamma is performed.
    """
    unique={};unit=None
    for use in uses:
        if not isinstance(use.measurement_id,str) or not use.measurement_id or not isinstance(use.object_id,str) or not use.object_id:
            raise ValueError('Physical measurement and object IDs are required')
        if not isinstance(use.new_in_campaign,(bool,np.bool_)):raise ValueError('Campaign acquisition status must be explicit Boolean')
        amount=float(use.cost_amount)
        if not np.isfinite(amount) or amount<0:raise ValueError('Costs must be finite and nonnegative')
        if not isinstance(use.cost_unit,str) or not use.cost_unit:raise ValueError('Explicit cost units required')
        if unit is not None and unit!=use.cost_unit:raise ValueError('Mixed cost units require explicit conversion before accounting')
        unit=use.cost_unit;purpose=MeasurementPurpose(use.purpose)
        identity=(use.object_id,bool(use.new_in_campaign),amount,unit)
        if use.measurement_id in unique:
            if unique[use.measurement_id]['identity']!=identity:raise ValueError('Conflicting metadata for a reused physical measurement')
            unique[use.measurement_id]['purposes'].add(purpose)
        else:unique[use.measurement_id]=dict(identity=identity,purposes={purpose})
    new=[r for r in unique.values() if r['identity'][1]]
    existing=[r for r in unique.values() if not r['identity'][1]]
    new_non_action=[r for r in new if MeasurementPurpose.ACTION not in r['purposes']]
    return dict(cost_unit=unit,new_unique_wells=len(new),existing_unique_wells=len(existing),
        new_unique_cost=sum(r['identity'][2] for r in new),
        new_non_action_cost=sum(r['identity'][2] for r in new_non_action),
        new_reference_wells=sum(MeasurementPurpose.REFERENCE in r['purposes'] for r in new),
        existing_reference_wells=sum(MeasurementPurpose.REFERENCE in r['purposes'] for r in existing),
        new_action_wells=sum(MeasurementPurpose.ACTION in r['purposes'] for r in new),
        reused_for_multiple_purposes=sum(len(r['purposes'])>1 for r in unique.values()),
        purpose_counts_may_overlap=True,gamma_adjusted=False)
