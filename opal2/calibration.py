"""Independent calibration and explicit-assumption fixed/sequential evaluation.

No function asserts that correlated wells, paired augmentations, pooled
cross-validation, or one source are independent deployment units. Callers must
declare the sampling assumption; development output is never called a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import beta


def _ids(values: Sequence[str], expected: int | None = None) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("Pass a sequence of unit IDs, not one string.")
    result = tuple(str(x) for x in values)
    if not result or any(not x for x in result) or len(set(result)) != len(result):
        raise ValueError("Unit IDs must be nonempty and unique; repeated wells are not new units.")
    if expected is not None and len(result) != expected:
        raise ValueError("Unit count and ID count disagree.")
    return result


def assert_disjoint(calibration_ids: Sequence[str], evaluation_ids: Sequence[str]) -> None:
    c, e = _ids(calibration_ids), _ids(evaluation_ids)
    overlap = set(c).intersection(e)
    if overlap:
        raise ValueError(f"Calibration/evaluation unit overlap ({len(overlap)} IDs).")


def _alpha(alpha: float) -> float:
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1).")
    return float(alpha)


def strict_binary(values, name="labels") -> np.ndarray:
    """Reject probabilities/fractional labels before conversion, including NaN."""
    original = np.asarray(values)
    if not np.isin(original, (0, 1)).all():
        raise ValueError(f"{name} must contain Boolean or exact zero/one values, not scores.")
    return original.astype(bool)


@dataclass
class SplitConformalUtility:
    alpha: float
    radius: float
    calibration_ids: tuple[str, ...]
    calibration_scores: np.ndarray
    output_shape: tuple[int, ...]
    assumption: str = "Exchangeable calibration/test units; marginal set coverage, not conditional-distribution correctness."

    @classmethod
    def fit(cls, observed: np.ndarray, predicted_mean: np.ndarray,
            calibration_ids: Sequence[str], *, alpha: float = 0.05,
            predicted_scale: np.ndarray | None = None,
            training_ids: Sequence[str] | None = None) -> "SplitConformalUtility":
        """Split-conformal absolute standardized residual scores.

        Scalars [N] are supported. For a fixed collection of utilities [N,A],
        the maximum standardized residual yields simultaneous coverage for the
        collection within one test unit. This is NOT per-subgroup conditional
        coverage. The predictive model and any scale must be fitted before these
        calibration outcomes; optional training IDs enforce that declared split.
        """
        y, mean = np.asarray(observed, dtype=float), np.asarray(predicted_mean, dtype=float)
        if y.ndim < 1 or mean.shape != y.shape or y.shape[0] < 1:
            raise ValueError("Observed and predicted utilities must have matching [N,...] shapes.")
        ids = _ids(calibration_ids, y.shape[0])
        if training_ids is not None:
            assert_disjoint(training_ids, ids)
        scale = np.ones_like(y) if predicted_scale is None else np.asarray(predicted_scale, dtype=float)
        if scale.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Calibration values must be finite and scales strictly positive.")
        alpha = _alpha(alpha)
        scores = (np.abs(y - mean) / scale).reshape(len(y), -1).max(axis=1)
        rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
        # The extra +infinity score is essential for exact finite-sample validity.
        radius = np.inf if rank > len(scores) else float(np.partition(scores, rank - 1)[rank - 1])
        return cls(alpha, radius, ids, scores.copy(), y.shape[1:])

    def interval(self, predicted_mean: np.ndarray, evaluation_ids: Sequence[str], *,
                 predicted_scale: np.ndarray | None = None,
                 support: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(predicted_mean, dtype=float)
        if mean.ndim < 1 or mean.shape[1:] != self.output_shape:
            raise ValueError("Evaluation outputs must have the calibrated fixed shape.")
        ids = _ids(evaluation_ids, len(mean))
        assert_disjoint(self.calibration_ids, ids)
        scale = np.ones_like(mean) if predicted_scale is None else np.asarray(predicted_scale, dtype=float)
        if scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Evaluation predictions/scales must be finite and scales positive.")
        low, high = mean - self.radius * scale, mean + self.radius * scale
        if support is not None:
            lo, hi = support
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError("Invalid known outcome support.")
            # Intersection may be empty for badly extrapolated predictions.
            low, high = np.maximum(low, lo), np.minimum(high, hi)
        return low, high


def clopper_pearson(successes: int, trials: int, *, alpha: float = 0.05,
                    side: str = "two-sided") -> tuple[float, float]:
    """Exact binomial bounds under the declared common-Bernoulli sampling model."""
    alpha = _alpha(alpha)
    if (not isinstance(successes, (int, np.integer)) or not isinstance(trials, (int, np.integer))
            or not 0 <= successes <= trials):
        raise ValueError("Binomial counts must be integers with 0 <= successes <= trials.")
    if side not in {"two-sided", "upper", "lower"}:
        raise ValueError("side must be two-sided, upper, or lower.")
    if trials == 0:
        return 0.0, 1.0
    tail = alpha / 2 if side == "two-sided" else alpha
    lower = 0.0 if successes == 0 or side == "upper" else float(beta.ppf(tail, successes, trials - successes + 1))
    upper = 1.0 if successes == trials or side == "lower" else float(beta.ppf(1 - tail, successes + 1, trials - successes))
    return lower, upper


def _bounded(values: np.ndarray, lower: float, upper: float, assume_iid_units: bool) -> np.ndarray:
    if not assume_iid_units:
        raise ValueError("These bounds require an explicit IID-unit assumption; shared batches do not establish it.")
    x = np.asarray(values, dtype=float)
    if (x.ndim != 1 or not len(x) or not np.isfinite(x).all()
            or not np.isfinite([lower, upper]).all() or lower >= upper
            or np.any(x < lower) or np.any(x > upper)):
        raise ValueError("Values must be a nonempty finite vector within the declared known bounds.")
    return x


def bounded_mean_interval(values: np.ndarray, lower: float, upper: float, *,
                          alpha: float = 0.05, assume_iid_units: bool = False,
                          side: str = "two-sided") -> tuple[float, float]:
    x = _bounded(values, lower, upper, assume_iid_units)
    alpha = _alpha(alpha)
    if side not in {"two-sided", "upper", "lower"}:
        raise ValueError("Invalid interval side.")
    radius = (upper - lower) * np.sqrt(np.log((2 if side == "two-sided" else 1) / alpha) / (2 * len(x)))
    lo = lower if side == "upper" else max(lower, float(x.mean() - radius))
    hi = upper if side == "lower" else min(upper, float(x.mean() + radius))
    return lo, hi


def bounded_mean_confidence_sequence(values: np.ndarray, lower: float, upper: float, *,
                                     alpha: float = 0.05, assume_iid_units: bool = False) -> dict:
    """Conservative anytime-valid Hoeffding sequence via alpha/[t(t+1)].

    The spending sums to alpha over all positive integers; a union bound over
    two-sided fixed-time Hoeffding intervals proves time-uniform coverage. This
    does not create additional samples, remove clustering, or license feedback
    that changes the evaluated frozen policy using its evaluation outcomes.
    """
    x = _bounded(values, lower, upper, assume_iid_units)
    alpha = _alpha(alpha)
    t = np.arange(1, len(x) + 1)
    alpha_t = alpha / (t * (t + 1))
    mean = np.cumsum(x) / t
    radius = (upper - lower) * np.sqrt(np.log(2 / alpha_t) / (2 * t))
    return {"time": t, "mean": mean, "lower": np.maximum(lower, mean - radius),
            "upper": np.minimum(upper, mean + radius), "alpha_spent_at_time": alpha_t,
            "method": "Hoeffding with summable alpha/(t*(t+1))", "assumption": "IID bounded evaluation units"}


def bernoulli_confidence_sequence(values: np.ndarray, *, alpha: float = 0.05,
                                  assume_iid_units: bool = False) -> dict:
    x = _bounded(values, 0.0, 1.0, assume_iid_units)
    if np.any((x != 0) & (x != 1)):
        raise ValueError("Bernoulli observations must be zero or one.")
    alpha = _alpha(alpha)
    t, totals = np.arange(1, len(x) + 1), np.cumsum(x).astype(int)
    alpha_t = alpha / (t * (t + 1))
    intervals = np.array([clopper_pearson(int(k), int(n), alpha=float(a))
                          for k, n, a in zip(totals, t, alpha_t)])
    return {"time": t, "mean": totals / t, "lower": intervals[:, 0], "upper": intervals[:, 1],
            "alpha_spent_at_time": alpha_t, "method": "Clopper-Pearson with summable alpha spending",
            "assumption": "IID Bernoulli evaluation units"}


@dataclass(frozen=True)
class RiskContract:
    max_fdp: float | None = None
    max_fpr: float | None = None
    min_sensitivity: float | None = None
    min_coverage: float | None = None
    min_activations: int | None = None
    min_mean_net_gain: float | None = None
    max_mean_wells: float | None = None
    alpha: float = 0.05

    def __post_init__(self) -> None:
        _alpha(self.alpha)
        for name in ("max_fdp", "max_fpr", "min_sensitivity", "min_coverage"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be in [0,1].")
        if self.min_activations is not None and (not isinstance(self.min_activations, (int, np.integer)) or self.min_activations < 0):
            raise ValueError("min_activations must be a nonnegative integer.")
        if self.min_mean_net_gain is not None and not np.isfinite(self.min_mean_net_gain):
            raise ValueError("min_mean_net_gain must be finite.")
        if self.max_mean_wells is not None and (not np.isfinite(self.max_mean_wells) or self.max_mean_wells < 0):
            raise ValueError("max_mean_wells must be nonnegative.")
        if all(getattr(self, f) is None for f in ("max_fdp", "max_fpr", "min_sensitivity", "min_coverage", "min_activations", "min_mean_net_gain", "max_mean_wells")):
            raise ValueError("Declare at least one acceptance criterion.")


def evaluate_contract(
    selected: np.ndarray, gain_if_selected: np.ndarray, population_null: np.ndarray,
    population_positive: np.ndarray, unit_ids: Sequence[str], contract: RiskContract, *,
    gain_support: tuple[float, float], wells_if_selected: np.ndarray,
    calibration_ids: Sequence[str] | None = None,
    training_ids: Sequence[str] | None = None,
    assume_iid_units: bool = False, developmental: bool = True,
    selection_design: str = "pointwise",
    positive_margin: float = 0.005,
) -> dict:
    """Evaluate a frozen pointwise policy on explicitly independent units.

    ``selection_design='pointwise'`` preserves the existing API but declares
    that one unit's action does not depend on other evaluation units. A frozen
    whole-cohort knapsack, top-k selector, or all-cohort probe update does not
    satisfy this condition merely because input compounds were independently
    sampled. Such designs are rejected here. For independently sampled complete
    campaigns use ``selection.evaluate_ltt_candidate`` with its explicit
    ``within_independent_cluster`` design and permitted aggregate criteria.

    ``population_null/positive`` must refer to a declared common counterfactual
    acquisition endpoint, also for unselected units. They are not inferred from
    stop's zero utility. Variable-action policies require an explicit coherent
    population reference before using FPR/sensitivity. Observed selected gains
    must agree with these labels. All bounds are conservatively Bonferroni
    simultaneous across declared statistical criteria. Coverage is population
    coverage, not only empirical sample coverage.
    """
    if selection_design != "pointwise":
        raise ValueError(
            "Compound-level contract bounds require selection_design='pointwise'; "
            "cohort-coupled allocation cannot use IID compound CP bounds. For "
            "independent campaigns use selection.evaluate_ltt_candidate with "
            "selection_design='within_independent_cluster'.")
    choose = strict_binary(selected, "selected")
    gain = np.asarray(gain_if_selected, dtype=float)
    null, positive = strict_binary(population_null, "population_null"), strict_binary(population_positive, "population_positive")
    wells = np.asarray(wells_if_selected, dtype=float)
    if choose.ndim != 1 or not len(choose) or any(x.shape != choose.shape for x in (gain, null, positive, wells)):
        raise ValueError("All policy arrays must be matching nonempty vectors.")
    ids = _ids(unit_ids, len(choose))
    for prior in (calibration_ids, training_ids):
        if prior is not None:
            assert_disjoint(prior, ids)
    if not np.isfinite(gain).all() or not np.isfinite(wells).all() or np.any(wells < 0) or np.any(wells > 2):
        raise ValueError("Gains must be finite; action burden is between zero and two wells.")
    if np.any(null & positive) or np.any(null[choose] != (gain[choose] <= 0)) or np.any(positive[choose] != (gain[choose] >= positive_margin)):
        raise ValueError("Declared population labels conflict with the selected action's realized endpoint.")
    if np.any(choose & (wells == 0)):
        raise ValueError("Activated actions must buy at least one well.")
    outcome = np.where(choose, gain, 0.0)
    _bounded(outcome, *gain_support, assume_iid_units)
    statistical = [name for name in ("max_fdp", "max_fpr", "min_sensitivity", "min_coverage", "min_mean_net_gain", "max_mean_wells") if getattr(contract, name) is not None]
    a = contract.alpha / max(1, len(statistical))
    n, m = len(choose), int(choose.sum())
    false = int((choose & null).sum())
    hit = int((choose & positive).sum())
    bounds = {
        "fdp_upper": clopper_pearson(false, m, alpha=a, side="upper")[1],
        "fpr_upper": clopper_pearson(false, int(null.sum()), alpha=a, side="upper")[1],
        "sensitivity_lower": clopper_pearson(hit, int(positive.sum()), alpha=a, side="lower")[0],
        "coverage_lower": clopper_pearson(m, n, alpha=a, side="lower")[0],
        "net_gain_lower": bounded_mean_interval(outcome, *gain_support, alpha=a, assume_iid_units=True, side="lower")[0],
        "mean_wells_upper": bounded_mean_interval(np.where(choose, wells, 0.0), 0.0, 2.0, alpha=a, assume_iid_units=True, side="upper")[1],
    }
    checks = {}
    for name, key in (("max_fdp", "fdp_upper"), ("max_fpr", "fpr_upper"), ("max_mean_wells", "mean_wells_upper")):
        if getattr(contract, name) is not None:
            checks[name] = bool(bounds[key] <= getattr(contract, name))
    for name, key in (("min_sensitivity", "sensitivity_lower"), ("min_coverage", "coverage_lower")):
        if getattr(contract, name) is not None:
            checks[name] = bool(bounds[key] >= getattr(contract, name))
    if contract.min_mean_net_gain is not None:
        checks["min_mean_net_gain"] = bool(bounds["net_gain_lower"] > contract.min_mean_net_gain)
    if contract.min_activations is not None:
        checks["min_activations"] = m >= contract.min_activations
    passed = all(checks.values())
    prefix = "DEVELOPMENT_DIAGNOSTIC" if developmental else "INDEPENDENT_FROZEN_POLICY_EVALUATION"
    return {"status": prefix + ("_PASS" if passed else "_NOT_PASSED"), "passed": passed,
            "checks": checks, "bounds": bounds, "n": n, "activations": m, "false_activations": false,
            "positive_activations": hit, "observed_mean_net_gain": float(outcome.mean()),
            "observed_mean_wells": float(np.where(choose, wells, 0.0).mean()),
            "alpha_per_declared_statistical_criterion": a,
            "selection_design": selection_design,
            "assumption": "IID evaluation units, frozen pointwise policy, declared common population endpoint",
            "is_prospective_certification": False,
            "scope_note": "A numerical pass does not establish prospective execution or unseen-source generalization."}
