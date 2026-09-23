import numpy as np
import pytest

from opal2.biology_delta_gate import (ALPHAS, FEATURE_NAMES, BiologyDeltaGates,
    _candidate_deltas, _context, _fit_gate_set, _fit_predictor, _make_oof,
    _mix, fit_biology_delta_gates)
from opal2.empirical_radial import fit_radial, reference_weights


def population(n=24):
    rng = np.random.default_rng(14)
    amp = np.linspace(-2, 2, n)
    r = np.exp(.5 + .2 * amp + rng.normal(0, .35, n))
    groups = np.arange(n).astype(str)
    ids = np.array([f"r{i}" for i in range(n)])
    category = np.arange(n) % 3
    sim = (category[:, None] == category[None, :]).astype(float)
    return r, amp, groups, ids, sim


def test_delta_density_definition_and_direct_alpha_no_second_gate():
    r, amp, groups, ids, sim = population()
    ctx = _context(r[:16], amp[:16], groups[:16], amp[16:], groups[16:], sim[16:, :16], .8)
    weights, alpha = _mix(ctx, np.ones(8))
    np.testing.assert_array_equal(weights, ctx["relation"])
    assert np.all(alpha == 1)
    delta = _candidate_deltas(r[16:], fit_radial(r[:16]), ctx)
    np.testing.assert_array_equal(delta[:, 0], 0)
    assert delta.shape == (8, 4) and np.isfinite(delta).all()
    # Every mixture density is the same convex combination of the endpoint
    # densities, including their shared Gaussian guard.
    for j, a in enumerate(ALPHAS):
        np.testing.assert_allclose(delta[:, j], np.logaddexp(np.log1p(-a), np.log(a)+delta[:, -1])
                                   if 0 < a < 1 else (0 if a == 0 else delta[:, -1]), atol=1e-12)


def test_nested_roles_and_query_independence(tmp_path):
    r, amp, groups, ids, sim = population()
    fitted = fit_biology_delta_gates(r, amp, groups, {"target": sim}, ids, fit_amp_sd=.8)
    report = fitted.report["channels"]["target"]
    for outer in report["outer_ledger"]:
        assert not set(outer["fit_groups"]) & set(outer["query_groups"])
        for inner in outer["inner_oof"]:
            assert set(inner["fit_ids"] + inner["query_ids"]) <= set(outer["fit_ids"])
            assert not set(inner["fit_groups"]) & set(inner["query_groups"])
            assert not set(inner["fit_ids"] + inner["query_ids"]) & set(outer["query_ids"])
    query_amp = [.2, .7]
    query_sim = {"target": sim[:2].copy()}
    baseline = reference_weights(amp, query_amp, .8, conditional=True)["weights"]
    out = fitted.apply(query_amp, ["q0", "q1"], query_sim, ["Q0", "Q1"], base_weights=baseline)
    assert set(out["plans"]) == {"TARGET_FIXED_SELECTED", "TARGET_DELTA_RIDGE", "TARGET_DELTA_BOOST"}
    for plan in out["plans"].values():
        np.testing.assert_allclose(plan["weights"].sum(1), 1)
        assert plan["features"].shape == (2, len(FEATURE_NAMES))
    path = tmp_path / "gates.joblib"
    fitted.save(path)
    with pytest.raises(FileExistsError): fitted.save(path)
    loaded = BiologyDeltaGates.load(path)
    repeated = loaded.apply(query_amp, ["q0", "q1"], query_sim, ["Q0", "Q1"], base_weights=baseline)
    for name in out["plans"]:
        np.testing.assert_array_equal(out["plans"][name]["weights"], repeated["plans"][name]["weights"])
    with pytest.raises(ValueError, match="isolated"):
        fitted.apply([.2], [groups[0]], {"target": sim[:1]}, ["unseen_id"])


def test_own_future_radius_never_enters_own_oof_features():
    r, amp, groups, ids, sim = population()
    sim = np.ones_like(sim)
    first = _make_oof(r, amp, groups, ids, sim, .8)
    index = int(np.argmax(np.abs(first["delta"][:, -1])))
    assert abs(first["delta"][index, -1]) > 0
    changed = r.copy(); changed[index] *= 30
    second = _make_oof(changed, amp, groups, ids, sim, .8)
    np.testing.assert_array_equal(first["features"][index], second["features"][index])
    assert not np.array_equal(first["delta"][index], second["delta"][index])
    own_fold = next(x for x in first["ledger"] if ids[index] in x["query_ids"])
    assert ids[index] not in own_fold["fit_ids"]


def test_unsupported_and_ties_exactly_off():
    r, amp, groups, ids, sim = population(12)
    fitted = fit_biology_delta_gates(r, amp, groups, {"target": np.zeros_like(sim), "moa": np.zeros_like(sim)},
                                     ids, fit_amp_sd=.8)
    out = fitted.apply([0., 1.], ["Q0", "Q1"], {"target": np.zeros((2,12)), "moa": np.zeros((2,12))}, ["Q0", "Q1"])
    for plan in out["plans"].values():
        np.testing.assert_array_equal(plan["weights"], out["base_weights"])
        np.testing.assert_array_equal(plan["alpha"], 0)
        np.testing.assert_array_equal(plan["predicted_delta"], 0)
    for channel in fitted.report["channels"].values():
        assert channel["final_fixed_alpha"] == 0
        assert channel["final_boost"]["fallback_reason"] is not None
        assert channel["nested_diagnostics"]["DELTA_RIDGE"]["prediction"]["supported"]["r2"] is None


def test_whole_gate_outer_query_radius_does_not_change_prediction():
    r, amp, groups, ids, sim = population(18)
    first = fit_biology_delta_gates(r, amp, groups, {"target": sim}, ids, fit_amp_sd=.8)
    changed = r.copy(); changed[4] *= 20
    second = fit_biology_delta_gates(changed, amp, groups, {"target": sim}, ids, fit_amp_sd=.8)
    for strategy in ("FIXED_SELECTED", "DELTA_RIDGE", "DELTA_BOOST"):
        a = first.report["channels"]["target"]["nested_records"]["strategies"][strategy]
        b = second.report["channels"]["target"]["nested_records"]["strategies"][strategy]
        np.testing.assert_array_equal(a["predicted_delta"][4], b["predicted_delta"][4])
        assert a["alpha"][4] == b["alpha"][4]


def test_boost_is_real_nonlinear_and_ridge_is_regularized():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(48, len(FEATURE_NAMES)))
    signal = .3 * (x[:, 0] > 0) + .1 * x[:, 1]**2
    delta = np.column_stack((np.zeros(len(x)), signal*.25, signal*.5, signal))
    oof = dict(features=x, delta=delta, supported=np.ones(len(x), bool))
    groups = np.arange(len(x)).astype(str)
    boosted = _fit_predictor(oof, groups, "boost", 13)
    ridge = _fit_predictor(oof, groups, "ridge", 13)
    assert min(boosted.diagnostics["total_actual_splits"]) > 0
    assert ridge.estimators[0].alpha == 20
    assert np.std(boosted.predict(x, oof["supported"])[:, -1]) > .01
    gates = _fit_gate_set(oof, groups, 13)
    assert gates["fixed_alpha"] == 1


def test_bad_inputs_and_group_eligibility():
    r, amp, groups, ids, sim = population(12)
    with pytest.raises(ValueError, match="positive"):
        fit_biology_delta_gates(r, amp, groups, {"target": sim}, ids, fit_amp_sd=0)
    with pytest.raises(ValueError, match="groups"):
        fit_biology_delta_gates(r, amp, np.repeat("x",12), {"target": sim}, ids, fit_amp_sd=1)
    bad = sim.copy(); bad[0, 1] = 1.1
    with pytest.raises(ValueError, match="Similarity"):
        fit_biology_delta_gates(r, amp, groups, {"target": bad}, ids, fit_amp_sd=1)
    with pytest.raises(ValueError, match="six groups"):
        fit_biology_delta_gates(r, amp, np.arange(12)%5, {"target": sim}, ids, fit_amp_sd=1)
    ctx = _context(r, amp, groups, np.array([.2]), np.array([groups[0]]), sim[:1], 1.)
    assert ctx["base"][0,0] == 0 and ctx["relation"][0,0] == 0
