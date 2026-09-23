"""Utilities from the same joint measurement draws, not independent score heads.

Every outcome is an explicitly declared predictive Monte Carlo quantity.  These
functions do not turn model draws into observed biological evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Action:
    name: str
    target_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(i, (int, np.integer)) or isinstance(i,(bool,np.bool_)) for i in self.target_indices):
            raise ValueError("Target indices must be integers.")
        object.__setattr__(self, "target_indices", tuple(int(i) for i in self.target_indices))
        if not self.name or len(set(self.target_indices)) != len(self.target_indices):
            raise ValueError("Actions need a name and distinct target indices.")
        if len(self.target_indices) > 2 or any(i < 0 for i in self.target_indices):
            raise ValueError("An action may add zero, one, or two future wells.")

    @property
    def wells(self) -> int:
        return len(self.target_indices)


def enumerate_actions(candidate_indices: Sequence[int]) -> tuple[Action, ...]:
    """Enumerate the complete finite stop/add-one/add-two action set."""
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) for i in candidate_indices):
        raise ValueError("Candidate indices must be integers, not fractional indices or Boolean flags.")
    indices = tuple(int(i) for i in candidate_indices)
    if len(set(indices)) != len(indices) or any(i < 0 for i in indices):
        raise ValueError("Candidate indices must be distinct and nonnegative.")
    actions = [Action("stop")]
    for k in (1, 2):
        for group in combinations(indices, k):
            actions.append(Action("add_" + "_".join(map(str, group)), group))
    return tuple(actions)


@dataclass
class UtilityResult:
    actions: tuple[Action, ...]
    samples: np.ndarray  # [Monte Carlo draws, compounds, actions], cost included
    mean: np.ndarray  # [compounds, actions]
    p_null: np.ndarray
    p_positive: np.ndarray
    costs: np.ndarray  # [actions] or [compounds, actions], condition-specific monetary/utility cost
    positive_margin: float
    utility_name: str

    @classmethod
    def from_samples(
        cls, actions: Sequence[Action], samples: np.ndarray, costs: np.ndarray,
        *, positive_margin: float = 0.005, utility_name: str = "declared_utility",
    ) -> "UtilityResult":
        actions = tuple(actions)
        x = np.asarray(samples, dtype=float)
        cost = np.asarray(costs, dtype=float)
        if (x.ndim != 3 or min(x.shape) < 1 or x.shape[2] != len(actions)
                or cost.shape not in {(len(actions),), x.shape[1:]}):
            raise ValueError("Expected samples [draws, compounds, actions] and action or compound/action costs.")
        if not np.isfinite(x).all() or not np.isfinite(cost).all() or np.any(cost < 0):
            raise ValueError("Utility samples and nonnegative costs must be finite.")
        if not np.isfinite(positive_margin) or positive_margin <= 0:
            raise ValueError("The positive margin must be positive; NULL is gain <= 0.")
        if len({a.name for a in actions}) != len(actions):
            raise ValueError("Action names must be unique.")
        return cls(actions, x, x.mean(axis=0), (x <= 0).mean(axis=0),
                   (x >= positive_margin).mean(axis=0), cost,
                   float(positive_margin), utility_name)

    @property
    def p_ambiguous(self) -> np.ndarray:
        return ((self.samples > 0) & (self.samples < self.positive_margin)).mean(axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine on the final axis; a zero-norm vector has cosine zero."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape[-1] != b.shape[-1] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cosine requires finite vectors with matching dimensions.")
    numerator = np.sum(a * b, axis=-1)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    output = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return np.clip(output, -1.0, 1.0)


def _inputs(known_x: np.ndarray, future_y: np.ndarray, actions: Sequence[Action],
            cost_per_well: float, observed_count: int) -> tuple[np.ndarray, np.ndarray, tuple[Action, ...], np.ndarray]:
    x, y = np.asarray(known_x, dtype=float), np.asarray(future_y, dtype=float)
    actions = tuple(actions)
    if (x.ndim != 2 or y.ndim != 4 or y.shape[1] != x.shape[0]
            or y.shape[3] != x.shape[1] or min(y.shape) < 1):
        raise ValueError("Expected known_x [B,D] and joint future_y [samples,B,T,D].")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite measurements require an explicit external missing-outcome rule.")
    if not isinstance(observed_count, (int, np.integer)) or observed_count < 1:
        raise ValueError("observed_count must describe the number of wells averaged in known_x.")
    if not np.isfinite(cost_per_well) or cost_per_well < 0:
        raise ValueError("cost_per_well must be finite and nonnegative.")
    if not actions or any(i >= y.shape[2] for a in actions for i in a.target_indices):
        raise ValueError("Action indices exceed the available future draws.")
    return x, y, actions, np.array([a.wells * cost_per_well for a in actions])


def cosine_utility_samples(
    known_x: np.ndarray, future_y: np.ndarray, actions: Sequence[Action],
    validation_index: int, *, cost_per_well: float = 0.01,
    observed_count: int = 1, positive_margin: float = 0.005,
    target_costs: np.ndarray | None = None,
) -> UtilityResult:
    """Original half-cosine difference minus acquisition cost.

    The validation well and acquired wells must be jointly drawn by the caller.
    The validation well cannot be acquired.  ``known_x`` may be an average of
    ``observed_count`` already-observed wells, allowing exact incremental utility.
    Stop has gain zero and hence mathematical NULL probability one; allocators
    correctly exclude unactivated stop actions from acquisition-risk counts.
    """
    x, y, actions, costs = _inputs(known_x, future_y, actions, cost_per_well, observed_count)
    if target_costs is not None:
        tc = np.asarray(target_costs, dtype=float)
        if tc.shape not in {(y.shape[2],), y.shape[1:3]} or not np.isfinite(tc).all() or np.any(tc < 0):
            raise ValueError("target_costs must be nonnegative [targets] or [compounds,targets].")
        tc = np.broadcast_to(tc, y.shape[1:3])
        costs = np.column_stack([tc[:, a.target_indices].sum(axis=1) for a in actions])
    if not isinstance(validation_index,(int,np.integer)) or isinstance(validation_index,(bool,np.bool_)) or validation_index < 0 or validation_index >= y.shape[2]:
        raise ValueError("Invalid validation_index.")
    if any(validation_index in a.target_indices for a in actions):
        raise ValueError("The validation well cannot be reused in an acquired profile.")
    v = y[:, :, validation_index, :]
    baseline = cosine(x[None], v)
    outcomes = np.zeros((y.shape[0], y.shape[1], len(actions)))
    for j, action in enumerate(actions):
        if action.wells:
            profile = (observed_count * x[None] + y[:, :, action.target_indices, :].sum(axis=2))
            profile /= observed_count + action.wells
            action_cost = costs[j] if costs.ndim == 1 else costs[:, j]
            outcomes[:, :, j] = 0.5 * (cosine(profile, v) - baseline) - action_cost
    return UtilityResult.from_samples(actions, outcomes, costs,
                                     positive_margin=positive_margin, utility_name="half_cosine_gain")


def average_precision(query: np.ndarray, gallery: np.ndarray, relevant: np.ndarray,
                      gallery_ids: Sequence[str]) -> float:
    """Exact AP with deterministic measurement-ID tie breaking."""
    gallery = np.asarray(gallery, dtype=float)
    labels=np.asarray(relevant)
    if not np.isin(labels,[False,True]).all():
        raise ValueError("Relevance labels must be exact Boolean/zero-one values, not scores.")
    relevant = labels.astype(bool)
    ids = np.asarray(gallery_ids, dtype=str)
    if gallery.ndim != 2 or relevant.shape != (len(gallery),) or ids.shape != relevant.shape:
        raise ValueError("Invalid gallery, relevance labels, or gallery IDs.")
    if len(set(ids)) != len(ids) or not relevant.any():
        raise ValueError("Gallery IDs must be unique and each query needs a relevant independent repeat.")
    similarity = cosine(np.asarray(query)[None], gallery)
    order = np.lexsort((ids, -similarity))
    hits = relevant[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].sum() / relevant.sum())


def retrieval_utility_samples(
    known_x: np.ndarray, future_y: np.ndarray, actions: Sequence[Action], *,
    gallery: np.ndarray, gallery_compound_ids: Sequence[str],
    gallery_measurement_ids: Sequence[str], query_compound_ids: Sequence[str],
    known_measurement_ids: Sequence[Sequence[str]], future_measurement_ids: np.ndarray,
    cost_per_well: float = 0.01, observed_count: int = 1,
    positive_margin: float = 0.005, target_costs: np.ndarray | None = None,
) -> UtilityResult:
    """AP improvement against an unchanged, independent gallery minus cost.

    All acquired/initial measurement IDs are forbidden in the gallery.  Same
    compound IDs are expected: they define independent repeat retrieval.  This
    is a technical retrieval endpoint, not an assertion of biological activity.
    """
    x, y, actions, costs = _inputs(known_x, future_y, actions, cost_per_well, observed_count)
    if target_costs is not None:
        tc=np.asarray(target_costs,float)
        if tc.shape not in {(y.shape[2],),y.shape[1:3]} or not np.isfinite(tc).all() or np.any(tc<0):
            raise ValueError("target_costs must be nonnegative [targets] or [compounds,targets].")
        tc=np.broadcast_to(tc,y.shape[1:3])
        costs=np.column_stack([tc[:,a.target_indices].sum(axis=1) for a in actions])
    g = np.asarray(gallery, dtype=float)
    gc, gm = np.asarray(gallery_compound_ids, dtype=str), np.asarray(gallery_measurement_ids, dtype=str)
    qc, fm = np.asarray(query_compound_ids, dtype=str), np.asarray(future_measurement_ids, dtype=str)
    if (g.ndim != 2 or g.shape[1] != x.shape[1] or gc.shape != (len(g),)
            or gm.shape != gc.shape or qc.shape != (len(x),) or fm.shape != y.shape[1:3]
            or len(known_measurement_ids) != len(x) or not np.isfinite(g).all()):
        raise ValueError("Gallery/query measurements and identifiers must match their arrays.")
    if len(set(qc)) != len(qc) or len(set(gm)) != len(gm):
        raise ValueError("Query compound IDs and gallery measurement IDs must be unique.")
    used = set()
    candidate_indices = sorted({i for a in actions for i in a.target_indices})
    for i, known_ids in enumerate(known_measurement_ids):
        if isinstance(known_ids, str) or len(known_ids) != observed_count:
            raise ValueError("Supply one actual measurement ID for every already-observed well.")
        observed = tuple(str(v) for v in known_ids)
        future = tuple(str(v) for v in fm[i, candidate_indices])
        if len(set(observed + future)) != len(observed) + len(future):
            raise ValueError("A measured well cannot be acquired twice.")
        if used.intersection(observed + future):
            raise ValueError("A physical measurement cannot belong to two query compounds.")
        used.update(observed + future)
    if used.intersection(gm):
        raise ValueError("Self-reuse: an initial or acquired measurement occurs in the gallery.")
    outcomes = np.zeros((y.shape[0], y.shape[1], len(actions)))
    for i in range(len(x)):
        relevance = gc == qc[i]
        baseline = average_precision(x[i], g, relevance, gm)
        for j, action in enumerate(actions):
            if action.wells:
                profiles = (observed_count * x[i] + y[:, i, action.target_indices, :].sum(axis=1))
                profiles /= observed_count + action.wells
                action_cost=costs[j] if costs.ndim==1 else costs[i,j]
                outcomes[:, i, j] = [average_precision(p, g, relevance, gm) - baseline - action_cost
                                     for p in profiles]
    return UtilityResult.from_samples(actions, outcomes, costs,
                                     positive_margin=positive_margin, utility_name="fixed_gallery_ap_gain")


def selected_outcomes(result: UtilityResult, action_indices: np.ndarray) -> np.ndarray:
    original = np.asarray(action_indices)
    if not np.issubdtype(original.dtype, np.integer) or np.issubdtype(original.dtype, np.bool_):
        raise ValueError("Action indices must have integer dtype.")
    choice = original.astype(int)
    if choice.shape != (result.mean.shape[0],) or np.any(choice < 0) or np.any(choice >= len(result.actions)):
        raise ValueError("One valid action index is required per compound.")
    return result.samples[:, np.arange(len(choice)), choice]


def fixed_strategy(result: UtilityResult, action_name: str) -> dict:
    names = [a.name for a in result.actions]
    if action_name not in names:
        raise ValueError("Unknown fixed action.")
    indices = np.full(result.mean.shape[0], names.index(action_name), dtype=int)
    return strategy_summary(result, indices)


def strategy_summary(result: UtilityResult, action_indices: np.ndarray) -> dict:
    samples = selected_outcomes(result, action_indices)
    wells = np.array([a.wells for a in result.actions])[action_indices]
    active = wells > 0
    return {"action_indices": np.asarray(action_indices).copy(), "total_wells": int(wells.sum()),
            "activations": int(active.sum()), "mean_net_gain": float(samples.mean()),
            "expected_null_count": float((samples[:, active] <= 0).sum(axis=1).mean()),
            "expected_fdp": float((samples[:, active] <= 0).mean()) if active.any() else 0.0,
            "per_compound_mean": samples.mean(axis=0),
            "draw_mean_net_gain": samples.mean(axis=1), "basis": "predictive_model_draws"}


def random_strategy(result: UtilityResult, action_name: str, n_selected: int,
                    rng: np.random.Generator) -> dict:
    """Uniformly choose compounds without consulting their realized/model utility."""
    names = [a.name for a in result.actions]
    stops = [j for j, a in enumerate(result.actions) if a.wells == 0]
    if len(stops) != 1 or action_name not in names or not isinstance(n_selected,(int,np.integer)) or not 0 <= n_selected <= result.mean.shape[0]:
        raise ValueError("Random comparator needs one stop, a valid action, and a valid count.")
    index = names.index(action_name)
    if result.actions[index].wells == 0:
        raise ValueError("The selected random action must acquire at least one well.")
    choices = np.full(result.mean.shape[0], stops[0], dtype=int)
    choices[rng.choice(len(choices), size=n_selected, replace=False)] = index
    return strategy_summary(result, choices)
