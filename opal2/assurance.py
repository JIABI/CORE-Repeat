"""Model-conditional finite-campaign assurance via an entire frozen workflow.

Monte Carlo uncertainty concerns the simulated pass probability. It does not
certify that the simulator is correct or guarantee an actual study will pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration import clopper_pearson


@dataclass(frozen=True)
class SimulatedCampaign:
    decision_state: Any
    evaluation_outcomes: Any


def campaign_assurance(simulator: Callable[[int, np.random.Generator], SimulatedCampaign],
                       frozen_policy: Callable[[Any], Any],
                       contract_evaluator: Callable[[Any, Any], Mapping[str, Any]], *,
                       sample_sizes: Sequence[int], simulations: int, seed: int,
                       model_description: str, policy_declaration: str,
                       alpha=.05, target_assurance: float | None = None):
    """Simulate data -> frozen policy(state only) -> full contract(outcomes,plan).

    The simulator must include all modeled technical, biological and missing-data
    mechanisms plus calibration/reference costs relevant to the declared study.
    A sequential evaluator may report stopping_n. Every simulation reruns the
    whole frozen decision/contract pipeline rather than sampling a pass flag.
    Different trials receive independent RNG streams. Per-size Monte Carlo
    intervals use Bonferroni alpha over the predeclared size grid.
    Policy and evaluator should be deterministic conditional on their inputs;
    any additional randomized quantities belong in the seeded simulator state.
    """
    sizes = tuple(sample_sizes)
    if (not sizes or any(not isinstance(n, (int, np.integer)) or n < 1 for n in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("Sample sizes must be unique positive integers")
    if not isinstance(simulations, (int, np.integer)) or simulations < 1:
        raise ValueError("simulations must be a positive integer")
    if not 0 < alpha < 1 or not np.isfinite(alpha) or not model_description or not policy_declaration:
        raise ValueError("Explicit model/frozen-policy description and valid alpha are required")
    if target_assurance is not None and (not np.isfinite(target_assurance) or not 0 < target_assurance < 1):
        raise ValueError("target_assurance must be strictly between zero and one")
    streams = iter(np.random.SeedSequence(seed).spawn(len(sizes) * simulations))
    rows = []
    for n in sizes:
        passes, stopping = 0, []
        for _ in range(simulations):
            campaign = simulator(int(n), np.random.default_rng(next(streams)))
            if not isinstance(campaign, SimulatedCampaign):
                raise TypeError("Simulator must separate decision_state from evaluation_outcomes")
            plan = frozen_policy(campaign.decision_state)
            result = contract_evaluator(campaign.evaluation_outcomes, plan)
            if not isinstance(result, Mapping) or not isinstance(result.get("passed"), (bool, np.bool_)):
                raise ValueError("Full contract evaluator must return a boolean passed field")
            passes += int(result["passed"])
            stop = result.get("stopping_n", int(n))
            if not isinstance(stop, (int, np.integer)) or not 1 <= stop <= n:
                raise ValueError("Reported stopping_n must be an actual sample count within the campaign")
            stopping.append(int(stop))
        interval = clopper_pearson(passes, simulations, alpha=alpha / len(sizes), side="two-sided")
        rows.append({"planned_n": int(n), "simulations": int(simulations), "passes": passes,
                     "model_conditional_assurance": passes / simulations,
                     "monte_carlo_interval": list(interval),
                     "mean_stopping_n": float(np.mean(stopping)),
                     "stopping_n_quantiles": np.quantile(stopping, [.1, .5, .9]).tolist()})
    supported = [r["planned_n"] for r in rows if target_assurance is not None and r["monte_carlo_interval"][0] >= target_assurance]
    return {"model": model_description, "frozen_policy": policy_declaration, "seed": int(seed),
            "results": rows, "target_assurance": target_assurance,
            "model_suggested_min_n": min(supported) if supported else None,
            "monte_carlo_familywise_alpha": alpha,
            "scope": "Model-conditional assurance of complete frozen workflows; not empirical certification or a guarantee of real-world success",
            "simulator_validity_certified": False, "is_prospective_certification": False}
