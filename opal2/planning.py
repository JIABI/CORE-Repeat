"""Finite, risk-constrained acquisition and two-stage probe planning.

Optimization uses only model distributions, never observed future outcomes.
Optimization success is not a statistical certificate of true deployment value.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .utility import Action, UtilityResult, cosine, cosine_utility_samples, selected_outcomes, strategy_summary


@dataclass
class Allocation:
    action_indices: np.ndarray
    action_names: tuple[str, ...]
    total_wells: int
    expected_total_gain: float
    expected_null_count: float
    expected_activations: int
    optimal: bool
    solver_message: str
    total_cost: float = 0.0
    setup_cost: float = 0.0
    risk_adjusted_objective: float = 0.0
    activated_setups: tuple[str, ...] = ()


def allocate_actions(
    result: UtilityResult, budget: int, *, max_null_fraction: float | None = None,
    max_expected_null: float | None = None,
    max_false_activation_rate: float | None = None,
    null_base_probability: np.ndarray | None = None,
    min_activations: int = 0, max_activations: int | None = None,
    fixed_actions: dict[int, str] | None = None, time_limit: float | None = None,
    risk_penalty: float = 0.0, cost_budget: float | None = None,
    setup_memberships: dict[str, np.ndarray] | None = None,
    setup_costs: dict[str, float] | None = None,
    allowed_actions: np.ndarray | None = None,
    penalty_null_probability: np.ndarray | None = None,
    penalty_exposure: np.ndarray | None = None,
    constraint_null_probability: np.ndarray | None = None,
    additional_constraints: Sequence[dict] = (),
    _fixed_setup_state: dict[str,bool] | None = None,
    _setup_included_in_utility: bool = False,
) -> Allocation:
    """Exact multiple-choice integer allocation, including the stop action.

    ``max_null_fraction`` constrains the model-expected NULL count divided by
    the number of selected compounds (a ratio of model expectations, not a
    certificate of realized FDP).  FPR requires an explicitly supplied
    population-NULL probability for a *common declared action/endpoint*;
    the selected action's NULL event must be a subset of that population event.
    Only a necessary probability implication is checked numerically; the caller
    must establish the event relation in the declared task.
    """
    n, a = result.mean.shape
    if not np.isfinite(risk_penalty) or risk_penalty < 0:
        raise ValueError("risk_penalty must be finite and nonnegative; zero preserves pure expected net value.")
    if cost_budget is not None and (not np.isfinite(cost_budget) or cost_budget < 0):
        raise ValueError("cost_budget must be finite and nonnegative.")
    memberships = setup_memberships or {}
    setup_costs = setup_costs or {}
    if set(memberships) != set(setup_costs):
        raise ValueError("Every declared setup needs both action membership and cost.")
    setup_names = tuple(sorted(memberships))
    for key in setup_names:
        m = np.asarray(memberships[key])
        if m.shape != (n, a) or not np.isin(m, [False, True]).all():
            raise ValueError("Setup membership must be a Boolean [compounds,actions] matrix.")
        if not np.isfinite(setup_costs[key]) or setup_costs[key] < 0:
            raise ValueError("Setup costs must be finite and nonnegative.")
    setups = np.asarray([setup_costs[key] for key in setup_names], dtype=float)
    costs = np.broadcast_to(result.costs, (n, a))
    if _fixed_setup_state is not None and set(_fixed_setup_state)!=set(setup_names):
        raise ValueError("Internal fixed-setup state must enumerate every declared setup.")
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("Budget must be a nonnegative number of wells.")
    if not isinstance(min_activations, (int, np.integer)) or not 0 <= min_activations <= n:
        raise ValueError("Invalid min_activations.")
    max_activations = n if max_activations is None else max_activations
    if not isinstance(max_activations, (int, np.integer)) or not min_activations <= max_activations <= n:
        raise ValueError("Invalid max_activations.")
    stop = [j for j, action in enumerate(result.actions) if action.wells == 0]
    if len(stop) != 1 or not np.allclose(result.samples[:, :, stop[0]], 0):
        raise ValueError("Allocation requires exactly one zero-utility stop action.")
    wells = np.array([action.wells for action in result.actions], dtype=float)
    active = wells > 0
    if any(np.asarray(memberships[key])[:, ~active].any() for key in setup_names):
        raise ValueError("A stop action cannot trigger a setup.")
    activated = np.broadcast_to(active, (n, a)).astype(float)
    null = result.p_null * activated
    if constraint_null_probability is not None:
        probability=np.asarray(constraint_null_probability,float)
        if probability.shape!=(n,a) or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise ValueError("Constraint NULL probabilities must be finite [compounds,actions] values in [0,1].")
        if setups.sum()>0 and not (_fixed_setup_state is not None and _setup_included_in_utility):
            raise ValueError("External constraint probabilities with startup cost require a fixed setup/count utility state.")
        null=probability*activated
    penalty_null=null
    if penalty_null_probability is not None or penalty_exposure is not None:
        probability=np.asarray(penalty_null_probability,float);exposure=np.asarray(penalty_exposure)
        if probability.shape!=(n,a) or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)) or exposure.shape!=(n,a) or not np.isin(exposure,[0,1]).all():
            raise ValueError("Full-plan penalty needs probabilities and Boolean exposure for every compound/action, including previously bought STOP states.")
        if setups.sum()>0 and not (_fixed_setup_state is not None and _setup_included_in_utility):
            raise ValueError("Externally supplied full-plan risk scores must incorporate setup/count coupling; use the ordinary setup-aware cohort risk interface instead.")
        penalty_null=probability*exposure
    risk_uses_null=(risk_penalty>0 or any(v is not None for v in (max_null_fraction,max_expected_null,max_false_activation_rate)))
    if setups.sum()>0 and _fixed_setup_state is None and risk_uses_null:
        # A shared setup cost changes the sign of per-compound FULL net gains.
        # Enumerating setup state and activation count fixes the declared equal
        # charge per activated compound. Each resulting subproblem is a true
        # MILP; taking the best proven optimum is exact over this finite union.
        # This can be exponential in the number of setup types, deliberately
        # never approximated or silently ignored when a risk cap is requested.
        best=None
        max_count=min(n,budget,max_activations)
        for bits in product((False,True),repeat=len(setup_names)):
            overhead=float(setups@np.asarray(bits))
            for count in range(min_activations,max_count+1):
                if count==0 and any(bits):continue
                shifted=result.samples.copy()
                if count:
                    shifted[:,:,active]-=overhead/count
                bracket=UtilityResult.from_samples(result.actions,shifted,result.costs,
                                                    positive_margin=result.positive_margin,utility_name=result.utility_name)
                try:
                    candidate=allocate_actions(bracket,budget,max_null_fraction=max_null_fraction,
                       max_expected_null=max_expected_null,max_false_activation_rate=max_false_activation_rate,
                       null_base_probability=null_base_probability,min_activations=count,max_activations=count,
                       fixed_actions=fixed_actions,time_limit=time_limit,risk_penalty=risk_penalty,cost_budget=cost_budget,
                       setup_memberships=memberships,setup_costs=setup_costs,allowed_actions=allowed_actions,
                       additional_constraints=additional_constraints,
                       _fixed_setup_state=dict(zip(setup_names,bits)),_setup_included_in_utility=True)
                except RuntimeError as error:
                    if "infeasible" not in str(error).lower():raise
                    continue
                if best is None or candidate.risk_adjusted_objective>best.risk_adjusted_objective:
                    best=candidate
        if best is None:raise RuntimeError("No proven optimal feasible allocation across setup/count states.")
        best.solver_message="Exact finite setup-state/activation-count enumeration; "+best.solver_message
        return best
    # Sparse equality block avoids quadratic memory for a large compound library.
    rows = [sparse.kron(sparse.eye(n, format="csr"), np.ones((1, a)), format="csr")]
    lower, upper = [1.0] * n, [1.0] * n
    rows.append(np.broadcast_to(wells, (n, a)).ravel())
    lower.append(0.0); upper.append(float(budget))
    rows.append(activated.ravel())
    lower.append(float(min_activations)); upper.append(float(max_activations))
    for constraint in additional_constraints:
        weights=np.asarray(constraint["weights"],float)
        lo=float(constraint.get("lower",-np.inf));hi=float(constraint.get("upper",np.inf))
        if weights.shape!=(n,a) or not np.isfinite(weights).all() or np.isnan(lo) or np.isnan(hi) or lo>hi:
            raise ValueError("Additional linear constraints need finite [compound,action] weights and ordered bounds.")
        rows.append(weights.ravel());lower.append(lo);upper.append(hi)
    if max_null_fraction is not None:
        if not 0 <= max_null_fraction <= 1:
            raise ValueError("max_null_fraction must lie in [0,1].")
        rows.append((null - max_null_fraction * activated).ravel())
        lower.append(-np.inf); upper.append(0.0)
    if max_expected_null is not None:
        if not np.isfinite(max_expected_null) or max_expected_null < 0:
            raise ValueError("max_expected_null must be finite and nonnegative.")
        rows.append(null.ravel()); lower.append(0.0); upper.append(float(max_expected_null))
    if max_false_activation_rate is not None:
        if not 0 <= max_false_activation_rate <= 1 or null_base_probability is None:
            raise ValueError("FPR requires a rate in [0,1] and declared population NULL probabilities.")
        population = np.asarray(null_base_probability, dtype=float)
        if population.shape != (n,) or not np.isfinite(population).all() or np.any((population < 0) | (population > 1)):
            raise ValueError("Invalid population NULL probabilities.")
        if np.any(null > population[:, None] + 1e-12):
            raise ValueError("Selected NULL events are incompatible with the declared population NULL event.")
        rows.append(null.ravel()); lower.append(0.0)
        upper.append(float(max_false_activation_rate * population.sum()))
    lb, ub = np.zeros((n, a)), np.ones((n, a))
    if allowed_actions is not None:
        allowed = np.asarray(allowed_actions)
        if allowed.shape != (n,a) or not np.isin(allowed,[False,True]).all() or not allowed[:,stop[0]].all():
            raise ValueError("allowed_actions must be Boolean [compounds,actions], with stop allowed for every compound.")
        ub = allowed.astype(float)
    names = [action.name for action in result.actions]
    for index, name in (fixed_actions or {}).items():
        if not isinstance(index,(int,np.integer)) or index < 0 or index >= n or name not in names:
            raise ValueError("Invalid fixed compound/action.")
        if ub[index,names.index(name)]==0:
            raise ValueError("Fixed action is forbidden by the declared state/action boundary.")
        lb[index, names.index(name)] = 1.0
        ub[index] = 0.0
        ub[index, names.index(name)] = 1.0
    options = {"mip_rel_gap": 0.0}
    if time_limit is not None:
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be positive.")
        options["time_limit"] = float(time_limit)
    base = sparse.vstack([sparse.csr_matrix(row) for row in rows], format="csr")
    extended = [sparse.hstack([base, sparse.csr_matrix((base.shape[0], len(setups)))])]
    # Binary setup y is the OR of all chosen member actions: x_ij <= y and
    # y <= sum(member x). This charges each batch/panel startup exactly once.
    for s, key in enumerate(setup_names):
        member = np.flatnonzero(np.asarray(memberships[key], dtype=bool).ravel())
        if not len(member):
            row = sparse.csr_matrix(([1.0], ([0], [n*a+s])), shape=(1, n*a+len(setups)))
            extended.append(row); lower.append(0.0); upper.append(0.0)
            continue
        rr = np.repeat(np.arange(len(member)), 2)
        cc = np.column_stack([member, np.full(len(member), n*a+s)]).ravel()
        vv = np.tile([1.0, -1.0], len(member))
        extended.append(sparse.csr_matrix((vv, (rr, cc)), shape=(len(member), n*a+len(setups))))
        lower.extend([-np.inf]*len(member)); upper.extend([0.0]*len(member))
        row = np.zeros(n*a+len(setups)); row[member] = -1; row[n*a+s] = 1
        extended.append(sparse.csr_matrix(row)); lower.append(-np.inf); upper.append(0.0)
    if cost_budget is not None:
        extended.append(sparse.csr_matrix(np.r_[costs.ravel(), setups]))
        lower.append(0.0); upper.append(float(cost_budget))
    constraint_matrix = sparse.vstack(extended, format="csc")
    score = result.mean - risk_penalty * penalty_null
    objective = np.r_[-score.ravel(), np.zeros_like(setups) if _setup_included_in_utility else setups]
    setup_lb=np.zeros(len(setups));setup_ub=np.ones(len(setups))
    if _fixed_setup_state is not None:
        setup_lb=np.asarray([_fixed_setup_state[key] for key in setup_names],float);setup_ub=setup_lb.copy()
    solution = milp(objective, integrality=np.ones(n * a+len(setups)),
                    bounds=Bounds(np.r_[lb.ravel(), setup_lb], np.r_[ub.ravel(), setup_ub]),
                    constraints=LinearConstraint(constraint_matrix, lower, upper), options=options)
    if not solution.success or solution.x is None:
        raise RuntimeError(f"No proven optimal feasible allocation: {solution.message}")
    matrix = solution.x[:n*a].reshape(n, a)
    choices = np.argmax(matrix, axis=1)
    discrete = np.zeros_like(matrix)
    discrete[np.arange(n), choices] = 1.0
    used = np.asarray([bool((discrete * np.asarray(memberships[key])).any()) for key in setup_names])
    check = constraint_matrix @ np.r_[discrete.ravel(), used.astype(float)]
    if np.any(check < np.asarray(lower) - 1e-7) or np.any(check > np.asarray(upper) + 1e-7):
        raise RuntimeError("Rounded integer solution violates an allocation constraint.")
    summary = strategy_summary(result, choices)
    overhead = float(setups @ used)
    subtract_overhead=0. if _setup_included_in_utility else overhead
    selected_gain=selected_outcomes(result,choices)
    selected_active=np.array([result.actions[j].wells>0 for j in choices])
    if selected_active.any() and subtract_overhead:
        selected_gain=selected_gain.copy();selected_gain[:,selected_active]-=subtract_overhead/selected_active.sum()
    null_count=float((selected_gain[:,selected_active]<=0).sum(axis=1).mean())
    return Allocation(choices, tuple(names[j] for j in choices), summary["total_wells"],
                      float((result.mean * discrete).sum() - subtract_overhead), null_count,
                      summary["activations"], True, str(solution.message),
                      float((costs * discrete).sum() + overhead), overhead,
                      float((score * discrete).sum() - subtract_overhead), tuple(k for k,u in zip(setup_names,used) if u))


@dataclass
class ProbePlanResult:
    samples: np.ndarray  # [probe draws, independent continuation draws, compounds]
    mean_total_gain: np.ndarray
    p_null: np.ndarray
    p_positive: np.ndarray
    continuation_action_indices: np.ndarray  # [probe draws, compounds]
    total_wells_by_probe_draw: np.ndarray
    mean_total_wells: float
    description: str


def finite_depth_probe_plan(
    known_x: np.ndarray, probe_samples: np.ndarray,
    continuation_sampler: Callable[[np.ndarray, int, str], np.ndarray],
    continuation_actions: Sequence[Action], validation_index: int, *,
    total_budget: int, cost_per_well: float = 0.01,
    max_total_additions_per_compound: int = 2,
    max_incremental_null_fraction: float | None = None,
    positive_margin: float = 0.005,
) -> ProbePlanResult:
    """Evaluate a genuine finite-depth contingent policy by nested expectation.

    Every object first buys one probe. For EACH possible probe observation, the
    callback receives the observed history [B,2,D], outer-draw index, and purpose
    (``selection`` or ``evaluation``). It must update the conditional posterior
    and jointly sample the remaining candidate wells and independent validator.
    The planner then optimizes the continuation, charging all probe costs.

    Evaluation calls require independent Monte Carlo randomness, reducing reuse
    of the optimizer's Monte Carlo noise. No observed future validation value is
    ever passed into action selection. This computes a model-based policy value;
    the caller must separately compare with no-probe fixed/random strategies.
    Optional risk limits concern INCREMENTAL continuation gains, not full-plan
    FDP. Full-plan risks are returned separately rather than falsely certified.
    """
    x, probe = np.asarray(known_x, dtype=float), np.asarray(probe_samples, dtype=float)
    actions = tuple(continuation_actions)
    if x.ndim != 2 or probe.ndim != 3 or probe.shape[1:] != x.shape or probe.shape[0] < 1:
        raise ValueError("Expected known_x [B,D] and probe_samples [outer draws,B,D].")
    if not np.isfinite(x).all() or not np.isfinite(probe).all():
        raise ValueError("Probe histories must be finite.")
    if total_budget < len(x) or int(total_budget) != total_budget:
        raise ValueError("The full budget must pay one initial probe for every object.")
    if max_total_additions_per_compound < 1 or any(1 + a.wells > max_total_additions_per_compound for a in actions):
        raise ValueError("Continuation actions exceed the declared finite-depth action budget.")
    branch_samples, branch_actions, branch_wells = [], [], []
    for outer, p in enumerate(probe):
        history = np.stack([x, p], axis=1)
        planning_draws = np.asarray(continuation_sampler(history.copy(), outer, "selection"), dtype=float)
        current = 0.5 * (x + p)
        incremental = cosine_utility_samples(current, planning_draws, actions, validation_index,
                                              cost_per_well=cost_per_well, observed_count=2,
                                              positive_margin=positive_margin)
        allocation = allocate_actions(incremental, int(total_budget) - len(x),
                                      max_null_fraction=max_incremental_null_fraction)
        # Release full measurement draws before the next conditional callback.
        del planning_draws, incremental
        evaluation_draws = np.asarray(continuation_sampler(history.copy(), outer, "evaluation"), dtype=float)
        independent = cosine_utility_samples(current, evaluation_draws, actions, validation_index,
                                              cost_per_well=cost_per_well, observed_count=2,
                                              positive_margin=positive_margin)
        validator = evaluation_draws[:, :, validation_index, :]
        probe_gain = 0.5 * (cosine(current[None], validator) - cosine(x[None], validator)) - cost_per_well
        full = probe_gain + selected_outcomes(independent, allocation.action_indices)
        if branch_samples and full.shape != branch_samples[0].shape:
            raise ValueError("Every probe branch needs the same number of evaluation draws.")
        branch_samples.append(full)
        branch_actions.append(allocation.action_indices)
        branch_wells.append(len(x) + allocation.total_wells)
        del evaluation_draws, validator, independent, probe_gain, history, current, full, allocation
    samples = np.stack(branch_samples)
    return ProbePlanResult(samples, samples.mean(axis=(0, 1)), (samples <= 0).mean(axis=(0, 1)),
                           (samples >= positive_margin).mean(axis=(0, 1)), np.stack(branch_actions),
                           np.asarray(branch_wells), float(np.mean(branch_wells)),
                           "Predictive nested-Monte-Carlo two-stage policy; all probes charged; not certification.")
