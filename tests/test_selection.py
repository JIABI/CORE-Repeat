"""Statistical software fixtures only; none are biological experiment results."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import binom

from opal2.assurance import SimulatedCampaign, campaign_assurance
from opal2.calibration import RiskContract, clopper_pearson
from opal2.selection import (BudgetCandidate, FrozenBudgetFamily,
                             FrozenCandidateEvaluation, evaluate_ltt_candidate,
                             select_fixed_sequence)


def family(*names):
    return FrozenBudgetFamily(tuple(BudgetCandidate(name, i / 10, (("threshold", .9 - i / 10),))
                                    for i, name in enumerate(names)),
                              "predeclared_budget", "Software fixture, frozen before calibration", True)


def outcome(n=100, false=0, active=None, gain=.2):
    active = n if active is None else active
    actions = np.r_[np.ones(active), np.zeros(n - active)].astype(int)
    values = np.where(actions, gain, 0.)
    values[:false] = -.1
    return FrozenCandidateEvaluation(tuple(f"c{i}" for i in range(n)), actions, values, actions)


def test_family_order_and_json_roundtrip():
    spec = family("a", "b", "c")
    assert FrozenBudgetFamily.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    with pytest.raises(ValueError, match="pre-calibration"):
        FrozenBudgetFamily((BudgetCandidate("a", .1),), "q", "No freeze")
    with pytest.raises(ValueError, match="strictly ordered"):
        FrozenBudgetFamily((BudgetCandidate("a", .1), BudgetCandidate("b", .3), BudgetCandidate("c", .2)),
                           "q", "Frozen declaration", True)
    with pytest.raises(ValueError, match="unique candidate"):
        family("a", "a")


def test_candidate_copies_and_freezes_choices():
    actions = np.array([1, 0]); gains = np.array([.1, 0.]); wells = actions.copy()
    item = FrozenCandidateEvaluation(("a", "b"), actions, gains, wells)
    gains[0] = 9
    actions[0] = 0
    assert item.gain[0] == .1 and item.actions[0] == 1
    with pytest.raises(ValueError, match="read-only"):
        item.gain[0] = .2
    with pytest.raises(ValueError, match="Stop has zero"):
        FrozenCandidateEvaluation(("a",), np.array([0]), np.array([.1]), np.array([0]))
    with pytest.raises(ValueError, match="integers"):
        FrozenCandidateEvaluation(("a",), np.array([.5]), np.array([.1]), np.array([1]))
    with pytest.raises(ValueError, match="not scores"):
        FrozenCandidateEvaluation(("a",), np.array([1]), np.array([.1]), np.array([1]), np.array([.6]))


def test_exact_binomial_iut_full_alpha_and_guard():
    item = outcome(n=100, false=3, active=50)
    contract = RiskContract(max_fdp=.2, min_coverage=.3, min_activations=40, alpha=.05)
    result = evaluate_ltt_candidate(item, contract, gain_support=(-1, 1), assume_iid_units=True)
    tests = result["tests"]
    assert tests["max_fdp"]["p_value"] == pytest.approx(binom.cdf(3, 50, .2))
    assert tests["min_coverage"]["p_value"] == pytest.approx(binom.sf(49, 100, .3))
    assert tests["max_fdp"]["bound"] == pytest.approx(clopper_pearson(3, 50, alpha=.05, side="upper")[1])
    assert result["iut_p_value"] == max(t["p_value"] for t in tests.values())
    assert result["passed"]
    stricter_guard = RiskContract(max_fdp=.2, min_activations=51)
    failed = evaluate_ltt_candidate(item, stricter_guard, gain_support=(-1, 1), assume_iid_units=True)
    assert not failed["passed"] and failed["iut_p_value"] < .05
    empty = evaluate_ltt_candidate(outcome(active=0), RiskContract(max_fdp=.35),
                                   gain_support=(-1, 1), assume_iid_units=True)
    assert not empty["passed"] and empty["tests"]["max_fdp"]["p_value"] == 1


def test_fixed_sequence_stops_first_failure_never_reorders():
    # Third candidate would pass but must remain untested after the second fails.
    outcomes = {"third": outcome(false=0), "second": outcome(false=60), "first": outcome(false=0)}
    result = select_fixed_sequence(family("first", "second", "third"), outcomes,
                                   RiskContract(max_fdp=.35), gain_support=(-1, 1), assume_iid_units=True)
    assert result["selected_candidate"] == "first"
    assert result["stopped_at"] == "second" and result["not_tested"] == ["third"]
    assert [r["candidate"]["name"] for r in result["tested"]] == ["first", "second"]
    assert result["familywise_alpha"] == .05
    assert not result["is_prospective_certification"]
    outcomes["first"] = outcome(false=60)
    failed = select_fixed_sequence(family("first", "second", "third"), outcomes,
                                   RiskContract(max_fdp=.35), gain_support=(-1, 1), assume_iid_units=True)
    assert failed["status"] == "no_certified_candidate" and failed["selected_candidate"] is None
    assert failed["not_tested"] == ["second", "third"]


def test_calibration_unit_alignment_and_training_overlap():
    spec = family("a", "b")
    a = outcome()
    b = FrozenCandidateEvaluation(tuple(reversed(a.unit_ids)), a.actions, a.gain, a.wells)
    with pytest.raises(ValueError, match="units/order"):
        select_fixed_sequence(spec, {"a": a, "b": b}, RiskContract(max_fdp=.35),
                              gain_support=(-1, 1), assume_iid_units=True)
    with pytest.raises(ValueError, match="overlap"):
        select_fixed_sequence(spec, {"a": a, "b": a}, RiskContract(max_fdp=.35),
                              gain_support=(-1, 1), assume_iid_units=True, training_ids=["c0"])


def test_common_endpoint_fpr_sensitivity():
    base = outcome(100, false=1, active=30)
    null = np.zeros(100, bool); null[0] = True; null[50:] = True
    positive = ~null
    item = FrozenCandidateEvaluation(base.unit_ids, base.actions, base.gain, base.wells,
                                     null, positive, "Fixed ADD_ONE net gain at cost .01")
    result = evaluate_ltt_candidate(item, RiskContract(max_fpr=.2, min_sensitivity=.2),
                                    gain_support=(-1, 1), assume_iid_units=True)
    assert result["tests"]["max_fpr"]["trials"] == 51
    assert result["tests"]["max_fpr"]["successes"] == 1
    assert result["tests"]["min_sensitivity"]["successes"] == 29
    assert result["tests"]["min_sensitivity"]["trials"] == 49
    assert result["passed"]
    with pytest.raises(ValueError, match="common population"):
        evaluate_ltt_candidate(base, RiskContract(max_fpr=.2), gain_support=(-1, 1), assume_iid_units=True)
    incompatible = FrozenCandidateEvaluation(base.unit_ids, base.actions, base.gain, base.wells,
                                             np.zeros(100, bool), None, "Declared")
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_ltt_candidate(incompatible, RiskContract(max_fpr=.2), gain_support=(-1, 1), assume_iid_units=True)


def test_iid_assumption_explicit_and_cohort_topk_not_compound_cp():
    item = outcome()
    with pytest.raises(ValueError, match="explicit IID"):
        evaluate_ltt_candidate(item, RiskContract(max_fdp=.35), gain_support=(-1, 1))
    with pytest.raises(ValueError, match="whole-cohort"):
        evaluate_ltt_candidate(item, RiskContract(max_fdp=.35), gain_support=(-1, 1),
                               assume_iid_units=True, selection_design="top_k")
    with pytest.raises(ValueError, match="No compound-level exact"):
        evaluate_ltt_candidate(item, RiskContract(max_fdp=.35), gain_support=(-1, 1),
                               selection_design="within_independent_cluster", assume_iid_clusters=True,
                               cluster_ids=[f"campaign{i // 10}" for i in range(100)])


def test_cluster_value_uses_independent_campaign_count_equal_weight():
    # Unequal campaign sizes: correct estimand is mean of .1 and .9, not .82.
    gains = np.r_[.1, np.full(9, .9)]
    item = FrozenCandidateEvaluation(tuple(f"c{i}" for i in range(10)), np.ones(10), gains, np.ones(10))
    result = evaluate_ltt_candidate(item, RiskContract(min_mean_net_gain=0), gain_support=(-1, 1),
                                    selection_design="within_independent_cluster", assume_iid_clusters=True,
                                    cluster_ids=["a"] + ["b"] * 9)
    value = result["tests"]["min_mean_net_gain"]
    assert result["statistical_units"] == value["trials"] == 2
    assert result["calibration_compounds"] == 10
    assert value["estimate"] == pytest.approx(.5)
    assert value["p_value"] == pytest.approx(np.exp(-2 * 2 * (.5 / 2)**2))
    assert "not compound FDP" in result["scope"] and not result["passed"]


def test_hoeffding_bounded_value_and_burden():
    item = outcome(200, active=100, gain=.5)
    result = evaluate_ltt_candidate(item, RiskContract(min_mean_net_gain=0, max_mean_wells=1),
                                    gain_support=(-1, 1), wells_support=(0, 1), assume_iid_units=True)
    value = result["tests"]["min_mean_net_gain"]
    assert value["estimate"] == .25
    assert value["p_value"] == pytest.approx(np.exp(-2 * 200 * (.25 / 2)**2))
    assert value["bound"] == pytest.approx(.25 - 2 * np.sqrt(np.log(20) / 400))
    assert result["passed"]
    with pytest.raises(ValueError, match="declared gain support"):
        evaluate_ltt_candidate(item, RiskContract(min_mean_net_gain=0), gain_support=(-.1, .1), assume_iid_units=True)
    with pytest.raises(ValueError, match="at least one statistical"):
        evaluate_ltt_candidate(item, RiskContract(min_activations=5), gain_support=(-1, 1), assume_iid_units=True)
    # A known one-well action cap already implies mean burden <= 1.
    capped = evaluate_ltt_candidate(outcome(), RiskContract(max_mean_wells=1),
                                    gain_support=(-1, 1), wells_support=(0, 1), assume_iid_units=True)
    assert capped["passed"] and capped["tests"]["max_mean_wells"]["p_value"] == 0


def test_assurance_runs_separated_entire_pipeline_and_reproduces():
    counts = {"simulation": 0, "policy": 0, "contract": 0}
    def simulate(n, rng):
        counts["simulation"] += 1
        return SimulatedCampaign({"available": rng.normal(size=n)}, {"future": rng.normal(size=n)})
    def policy(state):
        counts["policy"] += 1
        assert set(state) == {"available"}
        return state["available"] > 0
    def evaluate(realized, plan):
        counts["contract"] += 1
        assert set(realized) == {"future"}
        return {"passed": bool(np.mean(realized["future"] * plan) > 0), "stopping_n": len(plan)}
    args = dict(sample_sizes=(10, 20), simulations=30, seed=14, model_description="Test Gaussian simulator",
                policy_declaration="Fixed positive-state rule")
    a = campaign_assurance(simulate, policy, evaluate, **args)
    b = campaign_assurance(simulate, policy, evaluate, **args)
    assert a == b
    assert counts == {"simulation": 120, "policy": 120, "contract": 120}
    for row in a["results"]:
        assert row["monte_carlo_interval"] == list(clopper_pearson(row["passes"], 30, alpha=.025))
        assert row["mean_stopping_n"] == row["planned_n"]
    assert "Model-conditional" in a["scope"]
    assert not a["simulator_validity_certified"] and not a["is_prospective_certification"]


def test_assurance_size_choice_uses_mc_lower_bound_and_stopping():
    def simulate(n, rng):
        return SimulatedCampaign({"n": n}, np.ones(n))
    result = campaign_assurance(simulate, lambda state: state["n"],
                                lambda observed, plan: {"passed": plan >= 20, "stopping_n": min(plan, 5)},
                                sample_sizes=(30, 10, 20), simulations=100, seed=4,
                                model_description="Deliberate software fixture", policy_declaration="Frozen fixture",
                                target_assurance=.9)
    assert result["model_suggested_min_n"] == 20
    assert all(row["mean_stopping_n"] == 5 for row in result["results"])


@pytest.mark.parametrize("violation", ["state", "passed", "stopping"])
def test_assurance_rejects_incomplete_callback_contract(violation):
    def simulate(n, rng):
        return np.ones(n) if violation == "state" else SimulatedCampaign(n, np.ones(n))
    def evaluate(outcomes, plan):
        return {"passed": 1 if violation == "passed" else True,
                "stopping_n": 100 if violation == "stopping" else 1}
    with pytest.raises((ValueError, TypeError)):
        campaign_assurance(simulate, lambda state: state, evaluate, sample_sizes=(10,), simulations=1,
                            seed=2, model_description="Fixture", policy_declaration="Declared fixture")
