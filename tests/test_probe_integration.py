"""Integration checks with the real MeasurementWorldModel, not a fake sampler.

The small numerical fixtures are unit-test arrays, not biological evidence.
"""
import copy
import json

import numpy as np
import pytest
import torch

from opal2.data import MeasurementDataset, TrainScaler, cellprofiler_groups
from opal2.model import MeasurementWorldModel
from opal2.probe import (run_model_probe, evaluate_observed_probe, _seed,
                        _initial_inputs, _gaussian_condition_on_probe,run_selective_model_probe)
from opal2.utility import cosine


@pytest.fixture
def setup():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rng = np.random.default_rng(281)
    n, w, d = 6, 4, 6
    names = [f"Nuclei_Intensity_Feature{i}" for i in range(3)] + [f"Cells_AreaShape_Feature{i}" for i in range(3)]
    gi, gn = cellprofiler_groups(names)
    groups = np.zeros((n, w, 3), dtype=int)
    groups[:, :, 1] = np.arange(w)
    groups[:, :, 2] = np.arange(w)
    reference_mask = np.zeros((n, w, 3), dtype=bool)
    reference_mask[:, 0, 2] = True
    ds = MeasurementDataset(Y=2. + rng.normal(size=(n, w, d)), ids=np.array([f"unit{i}" for i in range(n)]),
                            feature_names=np.array(names), feature_group_index=gi, feature_group_names=gn,
                            cond=rng.normal(size=(n, w, 2)), reference=rng.normal(size=(n, w, 3, 3)),
                            reference_mask=reference_mask, groups=groups, chem=rng.normal(size=(n, 4)),
                            metadata={"fixture":True})
    scaler = TrainScaler.fit(ds, [0, 1, 2])
    torch.manual_seed(41)
    model = MeasurementWorldModel(ds.feature_groups, 2, 3, 4, hidden_dim=8, latent_rank=1, residual_rank=1)
    yield model, scaler, ds, np.array([3, 4])
    torch.set_num_threads(old_threads)


def options():
    return dict(total_budget=3, outer_samples=3, selection_samples=4, evaluation_samples=5,
                seed=983, update_mode="reencode")


def test_real_model_based_probe_runs_reconditions_and_is_jsonable(setup):
    model, scaler, ds, ix = setup
    model.train()
    histories = []
    hook = model.register_forward_pre_hook(lambda module, args: histories.append(args[0]["context_y"].detach().clone()))
    report = run_model_probe(model, scaler, ds, ix, include_draws=True, **options())
    hook.remove()
    json.dumps(report, allow_nan=False)
    assert model.training  # inference restores the caller's mode
    assert report["mode"] == "MODEL_BASED_PROBE_PLANNING" and report["observed_result"] is False
    assert report["update_mode"] == "reencode"
    assert report["future_outcomes_used_for_decision"] is False
    assert len(histories) == 1 + 2*3
    assert histories[0].shape == (2, 1, 6)
    probe = np.asarray(report["simulated_probe_values"])
    for branch in range(3):
        for purpose in range(2):
            h = histories[1 + 2*branch + purpose]
            assert h.shape == (2, 2, 6)
            observed = scaler.inverse_y(h).numpy()
            np.testing.assert_allclose(observed[:, 0], ds.Y[ix, 0], rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(observed[:, 1], probe[branch], rtol=2e-6, atol=2e-6)
    assert all(2 <= value <= 3 for value in report["branch_total_wells"])
    for row in report["per_compound"]:
        assert 0 <= row["predicted_null_probability"] + row["predicted_positive_probability"] <= 1 + 1e-12


def test_simulation_does_not_read_real_probe_or_other_future_values(setup):
    model, scaler, ds, ix = setup
    original = run_model_probe(model, scaler, ds, ix, **options())
    unavailable_future = copy.deepcopy(ds)
    unavailable_future.Y[ix, 1:] = np.nan
    unavailable_future.observed_mask[ix, 1:] = False
    # All physical future roles remain planned/available, but no measured value
    # exists. A model-based probe can still be planned from X.
    later = run_model_probe(model, scaler, unavailable_future, ix, **options())
    assert later == original


@pytest.mark.parametrize("update_mode", ["reencode", "gaussian_condition"])
def test_selection_and_evaluation_random_streams_are_distinct(setup, monkeypatch, update_mode):
    from opal2.model import JointGaussian,LazyConditionalGaussian
    model, scaler, ds, ix = setup
    seeds = []
    method = JointGaussian.sample_joint
    def recorded(distribution, n_samples, generator=None):
        seeds.append(generator.initial_seed())
        return method(distribution, n_samples, generator)
    monkeypatch.setattr(JointGaussian, "sample_joint", recorded)
    lazy_method=LazyConditionalGaussian.sample_joint
    def recorded_lazy(distribution,n_samples,generator=None):
        seeds.append(generator.initial_seed())
        return lazy_method(distribution,n_samples,generator)
    monkeypatch.setattr(LazyConditionalGaussian,"sample_joint",recorded_lazy)
    cfg = options(); cfg["update_mode"] = update_mode
    run_model_probe(model, scaler, ds, ix, **cfg)
    assert len(set(seeds)) == len(seeds) == 7
    assert seeds[0] == _seed(983, 1)
    assert seeds[1:3] == [_seed(983, 2, 0), _seed(983, 3, 0)]


def test_budget_equal_probe_count_forces_stop_and_charges_every_probe(setup):
    model, scaler, ds, ix = setup
    cfg = options(); cfg["total_budget"] = len(ix)
    report = run_model_probe(model, scaler, ds, ix, **cfg)
    assert np.asarray(report["branch_continuation_actions"]).sum() == 0
    assert report["branch_total_wells"] == [2, 2, 2]
    fixed = next(row for row in report["fixed_and_random_baselines"] if row["strategy"] == "all_stop_after_probe")
    np.testing.assert_allclose(report["policy"]["population_mean_net_gain"], fixed["population_mean_net_gain"])
    assert report["policy"]["mean_used_wells"] == 2


@pytest.mark.parametrize("update_mode", ["reencode", "gaussian_condition"])
def test_actual_probe_evaluation_decisions_ignore_Q_and_V_but_scores_use_them(setup, update_mode):
    model, scaler, ds, ix = setup
    cfg = dict(total_budget=3, selection_samples=5, evaluation_samples=6, seed=23, update_mode=update_mode)
    first = evaluate_observed_probe(model, scaler, ds, ix, **cfg)
    changed = copy.deepcopy(ds)
    changed.Y[ix, 2:] = -5. * ds.Y[ix, 2:]
    second = evaluate_observed_probe(model, scaler, changed, ix, **cfg)
    json.dumps(first, allow_nan=False)
    assert first["mode"] == "OBSERVED_RETROSPECTIVE_PROBE_EVALUATION"
    assert first["observed_result"] and not first["prospective_certification"]
    assert first["future_Q_or_V_used_for_decision"] is False
    assert [row["continue_Q"] for row in first["per_compound"]] == [row["continue_Q"] for row in second["per_compound"]]
    assert first["model_prediction_after_actual_probe"] == second["model_prediction_after_actual_probe"]
    assert first["policy"]["population_mean_net_gain"] != second["policy"]["population_mean_net_gain"]
    x, p, q, v = [ds.Y[ix, role] for role in range(4)]
    stop = .5 * (cosine((x+p)/2, v) - cosine(x, v)) - .01
    both = .5 * (cosine((x+p+q)/3, v) - cosine(x, v)) - .02
    for i, row in enumerate(first["per_compound"]):
        np.testing.assert_allclose(row["observed_total_gain"], both[i] if row["continue_Q"] else stop[i])
    assert first["policy"]["used_wells"] <= 3


def test_actual_probe_is_the_paid_observation_in_context(setup):
    model, scaler, ds, ix = setup
    histories = []
    hook = model.register_forward_pre_hook(lambda module, args: histories.append(args[0]["context_y"].detach().clone()))
    evaluate_observed_probe(model, scaler, ds, ix, total_budget=2, selection_samples=3,
                            evaluation_samples=3, update_mode="reencode")
    hook.remove()
    assert len(histories) == 2
    for normalized in histories:
        observed = scaler.inverse_y(normalized).numpy()
        np.testing.assert_allclose(observed, ds.Y[ix, :2], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("update_mode", ["reencode", "gaussian_condition"])
def test_roles_can_be_swapped_without_opening_extra_repeats(setup, update_mode):
    model, scaler, ds, ix = setup
    cfg = options(); cfg["update_mode"] = update_mode
    report = run_model_probe(model, scaler, ds, ix, roles=(0, 2, 1, 3), **cfg)
    assert report["roles"] == {"X": 0, "P": 2, "Q": 1, "V": 3}
    assert report["n"] == 2


def test_bad_split_double_scaling_missing_P_and_roles_rejected(setup):
    model, scaler, ds, ix = setup
    with pytest.raises(ValueError, match="training IDs"):
        run_model_probe(model, scaler, ds, [0, 1], **options())
    with pytest.raises(ValueError, match="unscaled"):
        run_model_probe(model, scaler, scaler.transform(ds), ix, **options())
    with pytest.raises(ValueError, match="permutation"):
        run_model_probe(model, scaler, ds, ix, roles=(0, 1, 1, 3), **options())
    invalid = options(); invalid["update_mode"] = "undeclared_update"
    with pytest.raises(ValueError, match="update_mode"):
        run_model_probe(model, scaler, ds, ix, **invalid)
    missing = copy.deepcopy(ds)
    missing.Y[ix, 1] = np.nan
    missing.observed_mask[ix, 1] = False
    with pytest.raises(ValueError, match="measured P"):
        evaluate_observed_probe(model, scaler, missing, ix, total_budget=3)


def test_default_exact_update_conditions_only_joint_probe_once_per_branch(setup, monkeypatch):
    from opal2.model import JointGaussian
    model, scaler, ds, ix = setup
    histories, observed = [], []
    hook = model.register_forward_pre_hook(
        lambda module, args: histories.append(args[0]["context_y"].detach().clone()))
    method = JointGaussian.condition
    def record(distribution, observed_y, observed_mask, target_indices,**kwargs):
        observed.append((observed_y.detach().clone(), observed_mask.detach().clone(), target_indices))
        assert kwargs.get("lazy") is True
        return method(distribution, observed_y, observed_mask, target_indices=target_indices,**kwargs)
    monkeypatch.setattr(JointGaussian, "condition", record)
    cfg = options(); cfg.pop("update_mode")
    report = run_model_probe(model, scaler, ds, ix, include_draws=True, **cfg)
    hook.remove()
    json.dumps(report, allow_nan=False)
    assert report["update_mode"] == "gaussian_condition"
    assert len(histories) == 1 and histories[0].shape == (2, 1, 6)
    assert len(observed) == cfg["outer_samples"]  # selection/evaluation share one posterior
    probe = np.asarray(report["simulated_probe_values"])
    for branch, (values, mask, targets) in enumerate(observed):
        assert values.shape == mask.shape == (2, 3, 6)
        assert mask[:, 0].all() and not mask[:, 1:].any()
        assert targets == (1, 2) and torch.count_nonzero(values[:, 1:]) == 0
        np.testing.assert_allclose(scaler.inverse_y(values[:, 0]).numpy(), probe[branch],
                                   rtol=2e-6, atol=2e-6)


def test_exact_model_probe_never_reads_stored_future_values(setup):
    model, scaler, ds, ix = setup
    cfg = options(); cfg["update_mode"] = "gaussian_condition"
    original = run_model_probe(model, scaler, ds, ix, **cfg)
    unavailable = copy.deepcopy(ds)
    unavailable.Y[ix, 1:] = np.nan
    unavailable.observed_mask[ix, 1:] = False
    assert run_model_probe(model, scaler, unavailable, ix, **cfg) == original


def test_exact_observed_probe_conditions_actual_P_without_reencoding(setup, monkeypatch):
    from opal2.model import JointGaussian
    model, scaler, ds, ix = setup
    contexts, probes = [], []
    hook = model.register_forward_pre_hook(
        lambda module, args: contexts.append(args[0]["context_y"].detach().clone()))
    method = JointGaussian.condition
    def record(distribution, observed_y, observed_mask, target_indices,**kwargs):
        probes.append((observed_y.detach().clone(), observed_mask.detach().clone()))
        assert kwargs.get("lazy") is True
        return method(distribution, observed_y, observed_mask, target_indices=target_indices,**kwargs)
    monkeypatch.setattr(JointGaussian, "condition", record)
    report = evaluate_observed_probe(model, scaler, ds, ix, total_budget=3,
                                     selection_samples=3, evaluation_samples=4)
    hook.remove()
    assert report["update_mode"] == "gaussian_condition"
    assert len(contexts) == len(probes) == 1 and contexts[0].shape == (2, 1, 6)
    values, mask = probes[0]
    np.testing.assert_allclose(scaler.inverse_y(values[:, 0]).numpy(), ds.Y[ix, 1],
                               rtol=2e-6, atol=2e-6)
    assert mask[:, 0].all() and not mask[:, 1:].any()


def test_exact_probe_of_one_compound_updates_other_shared_environment_compound(setup):
    model, scaler, ds, ix = setup
    model.eval()
    parameter = next(model.parameters())
    with torch.no_grad():
        inputs, x = _initial_inputs(model, scaler, ds, ix, (0, 1, 2, 3), "observed_only",
                                    parameter.device, parameter.dtype)
        joint = model(inputs)
        p = scaler.inverse_y(joint.mean[:, 0]).numpy()
        history = np.stack([x, p], axis=1)
        baseline = _gaussian_condition_on_probe(joint, history, scaler)
        perturbed = history.copy()
        perturbed[0, 1] += 2.
        updated = _gaussian_condition_on_probe(joint, perturbed, scaler)
        assert torch.max(torch.abs(updated.mean[1] - baseline.mean[1])) > 1e-7
        # The second compound's own observed X/P stayed fixed. Its update is
        # transmitted by shared environmental covariance, not re-encoding.
        np.testing.assert_array_equal(perturbed[1], history[1])


def test_selective_probe_initial_decision_never_reads_any_future(setup):
    model,scaler,ds,ix=setup
    cfg=dict(total_budget=1,lookahead_depth=1,outer_samples=2,selection_samples=4,evaluation_samples=5,seed=672)
    report=run_selective_model_probe(model,scaler,ds,ix,**cfg)
    changed=copy.deepcopy(ds);changed.Y[ix,1:]=np.nan;changed.observed_mask[ix,1:]=False
    assert run_selective_model_probe(model,scaler,changed,ix,**cfg)==report
    assert not report["first_stage_forces_all_probes"]
    assert report["budget_wells"]<len(ix)


def test_selective_retrospective_actions_do_not_use_validator(setup):
    model,scaler,ds,ix=setup
    cfg=dict(total_budget=2,lookahead_depth=1,outer_samples=2,selection_samples=4,evaluation_samples=5,seed=829,retrospective=True)
    report=run_selective_model_probe(model,scaler,ds,ix,**cfg)
    changed=copy.deepcopy(ds);changed.Y[ix,3]*=-7
    later=run_selective_model_probe(model,scaler,changed,ix,**cfg)
    assert later["decisions"]==report["decisions"]
    assert later["purchased"]==report["purchased"]
    assert report["total_wells"]<=2


@pytest.mark.parametrize("wrapper",[run_model_probe,run_selective_model_probe,evaluate_observed_probe])
def test_probe_four_role_tasks_reject_finite_unobserved_initial_X(setup,wrapper):
    model,scaler,ds,ix=setup
    ds.observed_mask[ix[0],0]=False
    with pytest.raises(ValueError,match="observed finite initial X"):
        wrapper(model,scaler,ds,ix,total_budget=3)


@pytest.mark.parametrize("failed_role",[1,3])
def test_selective_missing_finite_placeholders_error_or_worst_with_setup_cost(setup,monkeypatch,failed_role):
    import opal2.probe as probe_module
    model,scaler,ds,ix=setup
    ds.observed_mask[ix[0],failed_role]=False
    # Deterministic test decision isolates execution masks; the planner itself
    # is exercised by the exhaustive/numerical tests in test_decision_r2.
    def fixed(*args,**kwargs):
        return probe_module.SelectiveProbeDecision("direct",None,None,np.array([1,0]),.1,.1,[],"fixture")
    monkeypatch.setattr(probe_module,"plan_selective_probe",fixed)
    membership={"batch":np.array([[1,0,0],[1,0,0]],bool)}
    cfg=dict(total_budget=1,retrospective=True,setup_memberships=membership,setup_costs={"batch":.03})
    with pytest.raises(ValueError,match="observed finite"):
        run_selective_model_probe(model,scaler,ds,ix,**cfg)
    report=run_selective_model_probe(model,scaler,ds,ix,missing_outcome="worst",**cfg)
    assert report["observed_net_gain"][0]==pytest.approx(-1.04)
    assert report["observed_net_gain"][1]==0
    assert report["observed_total_cost"]==pytest.approx(.04)
    assert report["technical_failure"]==[True,False]


def test_failed_probe_is_not_conditioned_on_and_stops_further_acquisition(setup,monkeypatch):
    import opal2.probe as probe_module
    model,scaler,ds,ix=setup;ds.observed_mask[ix[0],1]=False;calls=[]
    def first_probe(*args,**kwargs):
        calls.append(kwargs["purchased"].copy())
        return probe_module.SelectiveProbeDecision("probe",0,0,None,.1,0.,[],"fixture")
    monkeypatch.setattr(probe_module,"plan_selective_probe",first_probe)
    report=run_selective_model_probe(model,scaler,ds,ix,total_budget=3,retrospective=True,missing_outcome="worst")
    assert len(calls)==1 and report["total_wells"]==1
    assert report["observed_net_gain"][0]==pytest.approx(-1.01)
