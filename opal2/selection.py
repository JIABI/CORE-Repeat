"""Fixed-sequence Learn-then-Test selection for a predeclared 1-D policy family.

Each candidate is frozen before calibration outcomes are inspected. Candidate
order must be specified in advance, not sorted by calibration p-values. For a
candidate, the maximum criterion p-value is an intersection-union test: all
declared risks must pass. A fixed sequence tests each candidate at the full alpha
and stops at the FIRST failure; risk need not be monotone along the budget axis.

Exact compound-level binomial FDP/FPR tests require IID common-Bernoulli units
under a frozen pointwise rule. Freezing a top-k or whole-cohort knapsack does NOT
by itself establish that condition. Cohort-coupled allocations are accepted only
as independent campaigns/clusters, with their aggregate bounded net value tested;
this module never turns clustered compound FDP into an exact binomial guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import binom

from .calibration import RiskContract, clopper_pearson


@dataclass(frozen=True)
class BudgetCandidate:
    name: str
    axis_value: float
    rule_parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        if not self.name or not np.isfinite(self.axis_value):
            raise ValueError("Candidate needs a name and a finite predeclared axis value")
        parameters = tuple((str(k), float(v)) for k, v in self.rule_parameters)
        if len({k for k, _ in parameters}) != len(parameters) or not all(np.isfinite(v) for _, v in parameters):
            raise ValueError("Rule thresholds must have unique names and finite values")
        object.__setattr__(self, "rule_parameters", parameters)


@dataclass(frozen=True)
class FrozenBudgetFamily:
    candidates: tuple[BudgetCandidate, ...]
    axis_name: str
    declaration: str
    frozen_before_calibration: bool = False

    def __post_init__(self):
        candidates = tuple(self.candidates)
        if not candidates or len({c.name for c in candidates}) != len(candidates):
            raise ValueError("A fixed sequence needs nonempty unique candidate names")
        values = np.asarray([c.axis_value for c in candidates])
        difference = np.diff(values)
        if len(values) > 1 and not (np.all(difference > 0) or np.all(difference < 0)):
            raise ValueError("The one-dimensional family must be strictly ordered before calibration")
        if not self.axis_name or not self.declaration or not self.frozen_before_calibration:
            raise ValueError("Explicit pre-calibration family/order declaration is required")
        object.__setattr__(self, "candidates", candidates)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["candidates"] = tuple(BudgetCandidate(**c) for c in value["candidates"])
        return cls(**value)


@dataclass(frozen=True)
class FrozenCandidateEvaluation:
    """Frozen per-unit actions and their independent calibration outcomes.

    Action 0 denotes stop. ``gain`` is realized NET gain of the chosen action
    for active units and must be exactly zero for stop. For FPR/sensitivity,
    explicit population labels must describe a common declared counterfactual
    endpoint, not stop's mathematical zero. Constructor copying prevents later
    accidental mutation; it cannot retrospectively prove when a rule was chosen.
    """
    unit_ids: tuple[str, ...]
    actions: np.ndarray
    gain: np.ndarray
    wells: np.ndarray
    population_null: np.ndarray | None = None
    population_positive: np.ndarray | None = None
    common_endpoint: str | None = None

    def __post_init__(self):
        ids = tuple(map(str, self.unit_ids))
        if not ids or any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("Calibration unit IDs must be unique and nonempty")
        object.__setattr__(self, "unit_ids", ids)
        for name, dtype in (("actions", np.int64), ("gain", float), ("wells", float),
                            ("population_null", bool), ("population_positive", bool)):
            original = getattr(self, name)
            if original is None:
                continue
            array = np.array(original, dtype=dtype, copy=True)
            if array.shape != (len(ids),):
                raise ValueError(f"{name} must align with calibration unit IDs")
            if name == "actions" and not np.array_equal(array, np.asarray(original)):
                raise ValueError("Action indices must be integers")
            if name in {"population_null", "population_positive"} and not np.isin(np.asarray(original), [False, True]).all():
                raise ValueError("Population labels must be Boolean or zero/one, not scores")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if np.any(self.actions < 0) or not np.isfinite(self.gain).all():
            raise ValueError("Actions must be nonnegative and gains finite")
        if not np.isfinite(self.wells).all() or np.any(self.wells < 0):
            raise ValueError("Well burdens must be finite and nonnegative")
        active = self.actions != 0
        if np.any(self.gain[~active] != 0) or np.any(self.wells[~active] != 0) or np.any(self.wells[active] <= 0):
            raise ValueError("Stop has zero realized gain/burden; active actions buy positive wells")
        if self.population_null is not None and self.population_positive is not None and np.any(self.population_null & self.population_positive):
            raise ValueError("NULL and POSITIVE population labels must be disjoint")


def _binomial_pvalue(k, n, threshold, *, direction):
    if n == 0:
        return 1.0
    if direction == "upper":
        return 0.0 if threshold == 1 else float(binom.cdf(k, n, threshold))
    return 0.0 if threshold == 0 else float(binom.sf(k - 1, n, threshold))


def _hoeffding_test(values, lower, upper, threshold, *, direction, alpha):
    x = np.asarray(values, dtype=float)
    if (x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not np.isfinite([lower, upper, threshold]).all()
            or lower >= upper or np.any(x < lower) or np.any(x > upper)):
        raise ValueError("Outcomes must lie within declared finite known bounds")
    mean = float(x.mean())
    margin = mean - threshold if direction == "lower" else threshold - mean
    # A known support can itself prove a weak upper-burden constraint. The
    # net-value criterion is strict (mean > threshold), so lower equality is
    # not vacuous and still requires evidence.
    vacuous = (direction == "upper" and threshold >= upper) or (direction == "lower" and threshold < lower)
    pvalue = 0.0 if vacuous else (1.0 if margin <= 0 else float(np.exp(-2 * len(x) * (margin / (upper - lower)) ** 2)))
    radius = (upper - lower) * np.sqrt(np.log(1 / alpha) / (2 * len(x)))
    bound = max(lower, mean - radius) if direction == "lower" else min(upper, mean + radius)
    return {"p_value": pvalue, "bound": float(bound), "estimate": mean,
            "trials": len(x), "threshold": float(threshold), "method": "one-sided bounded Hoeffding",
            "known_support": [float(lower), float(upper)]}


def evaluate_ltt_candidate(outcome: FrozenCandidateEvaluation, contract: RiskContract, *,
                           gain_support: tuple[float, float], assume_iid_units=False,
                           selection_design="pointwise", cluster_ids: Sequence[str] | None = None,
                           assume_iid_clusters=False, wells_support=(0.0, 2.0), positive_margin=.005):
    """Compute one IUT p-value; alpha is not split across intersection criteria."""
    if not np.isfinite(positive_margin) or positive_margin <= 0:
        raise ValueError("POSITIVE margin must be finite and strictly above the NULL cutoff zero")
    alpha = contract.alpha
    active = outcome.actions != 0
    n, m = len(active), int(active.sum())
    false = int(((outcome.gain <= 0) & active).sum())
    tests, guards = {}, {}
    if selection_design == "pointwise":
        if not assume_iid_units or cluster_ids is not None:
            raise ValueError("Exact compound tests require explicit IID pointwise units, without cluster substitution")
        values, burden = outcome.gain, outcome.wells
        scope = "IID pointwise compound policy; common-Bernoulli conditional rate assumptions"
    elif selection_design == "within_independent_cluster":
        if not assume_iid_clusters or cluster_ids is None:
            raise ValueError("Coupled policies require explicitly independent campaigns/clusters")
        if any(getattr(contract, name) is not None for name in ("max_fdp", "max_fpr", "min_sensitivity", "min_coverage")):
            raise ValueError("No compound-level exact FDP/FPR/coverage certificate for cluster-coupled selection")
        clusters = np.asarray(cluster_ids, dtype=str)
        if clusters.shape != (n,) or any(not x for x in clusters):
            raise ValueError("One nonempty cluster identifier is required per row")
        unique = np.unique(clusters)
        values = np.array([outcome.gain[clusters == c].mean() for c in unique])
        burden = np.array([outcome.wells[clusters == c].mean() for c in unique])
        scope = "IID campaigns/clusters; equally weighted cluster-mean per-compound net value, not compound FDP"
    else:
        raise ValueError("Unknown selection design; whole-cohort coupling is not IID pointwise selection")
    lo, hi = gain_support
    if not np.isfinite([lo, hi]).all() or lo >= hi or not lo <= 0 <= hi or np.any(outcome.gain < lo) or np.any(outcome.gain > hi):
        raise ValueError("All realized net outcomes, including stop, must obey the declared gain support")
    if contract.max_fdp is not None:
        p = _binomial_pvalue(false, m, contract.max_fdp, direction="upper")
        tests["max_fdp"] = {"p_value": p, "bound": clopper_pearson(false, m, alpha=alpha, side="upper")[1],
                             "successes": false, "trials": m, "threshold": contract.max_fdp,
                             "method": "one-sided binomial / Clopper-Pearson"}
    if contract.max_fpr is not None or contract.min_sensitivity is not None:
        if not outcome.common_endpoint:
            raise ValueError("FPR/sensitivity require a declared common population counterfactual endpoint")
    if contract.max_fpr is not None:
        null = outcome.population_null
        if null is None or np.any(null[active] != (outcome.gain[active] <= 0)):
            raise ValueError("Common NULL labels are absent or incompatible with the selected action")
        trials = int(null.sum())
        tests["max_fpr"] = {"p_value": _binomial_pvalue(false, trials, contract.max_fpr, direction="upper"),
                             "bound": clopper_pearson(false, trials, alpha=alpha, side="upper")[1],
                             "successes": false, "trials": trials, "threshold": contract.max_fpr,
                             "method": "one-sided binomial / Clopper-Pearson"}
    if contract.min_sensitivity is not None:
        positive = outcome.population_positive
        if positive is None or np.any(positive[active] != (outcome.gain[active] >= positive_margin)):
            raise ValueError("Common POSITIVE labels are absent or incompatible with the selected action")
        hit, trials = int((active & positive).sum()), int(positive.sum())
        tests["min_sensitivity"] = {"p_value": _binomial_pvalue(hit, trials, contract.min_sensitivity, direction="lower"),
                                     "bound": clopper_pearson(hit, trials, alpha=alpha, side="lower")[0],
                                     "successes": hit, "trials": trials, "threshold": contract.min_sensitivity,
                                     "method": "one-sided binomial / Clopper-Pearson"}
    if contract.min_coverage is not None:
        tests["min_coverage"] = {"p_value": _binomial_pvalue(m, n, contract.min_coverage, direction="lower"),
                                  "bound": clopper_pearson(m, n, alpha=alpha, side="lower")[0],
                                  "successes": m, "trials": n, "threshold": contract.min_coverage,
                                  "method": "one-sided binomial / Clopper-Pearson"}
    if contract.min_mean_net_gain is not None:
        tests["min_mean_net_gain"] = _hoeffding_test(values, lo, hi, contract.min_mean_net_gain,
                                                     direction="lower", alpha=alpha)
    if contract.max_mean_wells is not None:
        tests["max_mean_wells"] = _hoeffding_test(burden, *wells_support, contract.max_mean_wells,
                                                  direction="upper", alpha=alpha)
    if contract.min_activations is not None:
        guards["min_activations"] = m >= contract.min_activations
    # A purely deterministic guard may reject but does not certify a population
    # risk; callers must declare a genuine statistical criterion for LTT.
    if not tests:
        raise ValueError("LTT requires at least one statistical criterion")
    pvalue = max(test["p_value"] for test in tests.values())
    passed = pvalue <= alpha and all(guards.values())
    return {"passed": bool(passed), "iut_p_value": float(pvalue), "alpha": alpha,
            "tests": tests, "guards": guards, "calibration_compounds": n,
            "statistical_units": len(values), "activations": m, "false_activations": false,
            "scope": scope, "is_prospective_certification": False,
            "interpretation": "Population-risk test conditional on declared sampling/design assumptions; not a distribution-calibration certificate"}


def select_fixed_sequence(family: FrozenBudgetFamily,
                          outcomes: Mapping[str, FrozenCandidateEvaluation], contract: RiskContract, *,
                          gain_support: tuple[float, float], assume_iid_units=False,
                          selection_design="pointwise", cluster_ids: Sequence[str] | None = None,
                          assume_iid_clusters=False, training_ids: Sequence[str] | None = None,
                          **criterion_options):
    """Test in declared order, stop on failure, return the last passed candidate.

    All candidates use the same calibration unit ordering. Reusing that set
    across candidates is allowed by fixed-sequence FWER control; independence
    between candidate p-values is not required. This does not permit changing
    the family or retrying a new order after inspecting these outcomes.
    """
    names = [candidate.name for candidate in family.candidates]
    if set(outcomes) != set(names):
        raise ValueError("Supply exactly the outcomes for the predeclared family")
    ids = outcomes[names[0]].unit_ids
    if any(outcomes[name].unit_ids != ids for name in names):
        raise ValueError("Candidate calibration units/order differ")
    if training_ids is not None and set(map(str, training_ids)) & set(ids):
        raise ValueError("Training and LTT calibration units overlap")
    tested, passed, stopped_at = [], [], None
    for candidate in family.candidates:
        result = evaluate_ltt_candidate(outcomes[candidate.name], contract, gain_support=gain_support,
                                         assume_iid_units=assume_iid_units, selection_design=selection_design,
                                         cluster_ids=cluster_ids, assume_iid_clusters=assume_iid_clusters,
                                         **criterion_options)
        tested.append({"candidate": asdict(candidate), **result})
        if not result["passed"]:
            stopped_at = candidate.name
            break
        passed.append(candidate.name)
    return {"status": "selected_last_passed" if passed else "no_certified_candidate",
            "selected_candidate": passed[-1] if passed else None, "passed_candidates": passed,
            "stopped_at": stopped_at, "tested": tested,
            "not_tested": names[len(tested):], "family": family.to_dict(), "contract": asdict(contract),
            "familywise_alpha": contract.alpha, "method": "fixed sequence of candidate-level intersection-union tests",
            "scope": tested[0]["scope"], "calibration_units": len(ids),
            "guarantee_scope": "Under declared valid p-value assumptions, probability of certifying any violating candidate in this fixed family is at most alpha",
            "is_prospective_certification": False}
