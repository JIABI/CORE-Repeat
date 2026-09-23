"""Focused mathematical and pipeline tests for the predeclared objective arms."""
from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from opal2.config import TrainConfig
from opal2.probabilistic_scores import fair_crps, half_cosine_gains, derived_utility_crps
from opal2.training import fit_model, objective_random_stream, load_model
from opal2.utility import enumerate_actions, cosine_utility_samples
from test_model import make_batch, new_model
from test_training_integration import schema_fixture


def test_fair_crps_equals_explicit_off_diagonal_statistic_and_has_gradients():
    torch.manual_seed(118)
    draws=torch.randn(7,3,3,dtype=torch.float64,requires_grad=True)
    truth=torch.randn(3,3,dtype=torch.float64)
    off_diagonal=~torch.eye(7,dtype=torch.bool)
    pair=(draws[:,None]-draws[None,:]).abs()[off_diagonal].mean(0)
    expected=(draws-truth).abs().mean(0)-.5*pair
    actual=fair_crps(draws,truth)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    actual.mean().backward()
    assert draws.grad is not None and torch.isfinite(draws.grad).all() and draws.grad.abs().sum()>0
    # Finite-ensemble V-statistic differs: the fair estimator excludes i=j.
    biased=(draws-truth).abs().mean(0)-.5*(draws[:,None]-draws[None,:]).abs().mean((0,1))
    assert not torch.allclose(actual,biased)


def test_utility_torch_exactly_matches_original_numpy_roles_costs_and_zero_norm():
    torch.manual_seed(42)
    x=torch.randn(4,9,dtype=torch.float64)
    x[0]=0
    future=torch.randn(8,4,3,9,dtype=torch.float64,requires_grad=True)
    expected=cosine_utility_samples(x.numpy(),future.detach().numpy(),enumerate_actions((0,1)),2)
    actual=half_cosine_gains(x,future)
    np.testing.assert_allclose(actual.detach().numpy(),expected.samples[:,:,1:],rtol=1e-12,atol=1e-12)
    actual.mean().backward()
    assert torch.isfinite(future.grad).all()
    assert (future.grad.abs().sum((0,1,3))>0).all()  # both added wells AND V


def test_predictive_nll_is_exact_and_never_constructs_target_variational_posterior():
    model=new_model().eval()
    inputs=make_batch(c=1)
    y=torch.randn(3,3,7)
    with patch.object(model,"_update_gaussian",wraps=model._update_gaussian) as updates:
        result=model.loss(inputs,y,objective="predictive_nll")
        assert updates.call_count==1
    expected=-model(inputs).joint_log_prob(y)/y.numel()
    torch.testing.assert_close(result["nll"],expected)
    assert "elbo_loss" not in result and "latent_kl" not in result
    expected_total=(result["nll"]+model.reference_loss_weight*result["reference_reconstruction"]
                    +model.chemical_regularization_weight*result["chemical_kl_regularizer"])
    torch.testing.assert_close(result["loss"],expected_total)
    result["loss"].backward()
    assert model.evidence_head.weight.grad.abs().sum()>0
    assert model.scale_head.feature_log_scale.grad.abs().sum()>0


def test_default_elbo_and_explicit_elbo_are_bitwise_identical():
    model=new_model().eval(); inputs=make_batch(c=1); y=torch.randn(3,3,7)
    torch.manual_seed(903)
    default=model.loss(inputs,y)
    torch.manual_seed(903)
    explicit=model.loss(inputs,y,objective="elbo")
    for key in default:
        assert torch.equal(default[key],explicit[key])
    with patch.object(model,"_update_gaussian",wraps=model._update_gaussian) as updates:
        model.loss(inputs,y)
        assert updates.call_count==2


def test_no_evidence_update_preserves_exact_prior_for_empty_and_masked_context():
    model=new_model()
    for context_count in (0,2):
        inputs=make_batch(c=context_count)
        inputs["context_mask"][:]=False
        details=model._state_details(inputs)
        assert torch.equal(details["prior_mean"],details["posterior_mean"])
        assert torch.equal(details["prior_var"],details["posterior_var"])


def test_derived_score_uses_fixed_space_joint_draws_and_backpropagates():
    model=new_model().eval(); inputs=make_batch(c=1); y=torch.randn(3,3,7)
    center=torch.arange(7,dtype=torch.float32)-2
    scale=torch.arange(7,dtype=torch.float32)+.5
    model.set_outcome_transform(center,scale)
    dist=model(inputs)
    generator=torch.Generator().manual_seed(5)
    values=torch.cat([dist.sample_joint(4,generator=generator),dist.sample_joint(4,generator=generator)],0)
    target=half_cosine_gains(inputs["context_y"][:,0]*scale+center,(y*scale+center)[None])[0]
    utilities=half_cosine_gains(inputs["context_y"][:,0]*scale+center,values*scale+center)
    expected=fair_crps(utilities,target).mean()
    generator=torch.Generator().manual_seed(5)
    result=derived_utility_crps(model,inputs,y,samples=8,chunk_size=4,generator=generator)
    torch.testing.assert_close(result["utility_crps"],expected)
    assert result["utility_scored_compounds"]==3
    result["utility_crps"].backward()
    assert model.mean_head.weight.grad.abs().sum()>0
    assert model.factor_heads[0].weight.grad.abs().sum()>0
    assert model.factor_heads[1].weight.grad.abs().sum()>0


def test_paired_stream_preserves_later_minibatches_despite_extra_random_calls():
    cfg=TrainConfig(paired_objective_rng=True)
    torch.manual_seed(62)
    state=torch.get_rng_state().clone()
    with objective_random_stream(cfg,0,0):
        first=torch.rand(7)
        torch.rand(300)
    assert torch.equal(state,torch.get_rng_state())
    with objective_random_stream(cfg,0,0):
        torch.testing.assert_close(torch.rand(7),first,rtol=0,atol=0)
    with objective_random_stream(cfg,0,0,auxiliary=True):
        auxiliary=torch.rand(7)
    assert not torch.equal(auxiliary,first)
    with objective_random_stream(cfg,0,1):
        following=torch.rand(7)
    with objective_random_stream(cfg,0,0,auxiliary=True):
        torch.rand(912)
    with objective_random_stream(cfg,0,1):
        torch.testing.assert_close(torch.rand(7),following,rtol=0,atol=0)


@pytest.mark.parametrize("change",[
    {"objective":"unknown"},{"utility_crps_samples":1},{"utility_crps_samples":True},
    {"utility_crps_weight":-1},{"utility_crps_weight":float("nan")},
    {"utility_crps_weight":1},{"paired_objective_rng":1},
])
def test_objective_config_rejects_undeclared_or_invalid_scoring(change):
    with pytest.raises(ValueError):
        replace(TrainConfig(),**change).validate()


def test_full_three_arm_training_interfaces_share_jepa_initialization_and_fixed_selection(tmp_path):
    dataset=schema_fixture(d=9,n=12)
    splits={"train":np.arange(8),"validation":np.array([8,9]),
            "calibration":np.array([10]),"evaluation":np.array([11])}
    config=TrainConfig(seed=711,epochs=2,jepa_epochs=1,batch_size=4,hidden_dim=16,
                       latent_rank=2,residual_rank=1,threads=1,use_library=False,
                       paired_objective_rng=True,utility_crps_samples=4)
    initial=[]
    for arm,overrides in (("A",{}),("B",{"objective":"predictive_nll"}),
                          ("C",{"objective":"predictive_nll","utility_crps_weight":1.0})):
        cfg=replace(config,**overrides)
        model,scaler=fit_model(dataset,splits,cfg,tmp_path/arm,
                              shared_jepa_directory=None if arm=="A" else tmp_path/"A")
        payload=torch.load(tmp_path/arm/"best.pt",weights_only=True)
        initial.append(torch.load(tmp_path/arm/"initial_state.pt",weights_only=True))
        assert payload["checkpoint_selection"]=="fixed_role_predictive_nll"
        assert payload["train_ids"]==dataset.ids[:8].tolist()
        reloaded,_,loaded_config,_=load_model(tmp_path/arm)
        assert loaded_config.objective==cfg.objective
        records=[json.loads(line) for line in (tmp_path/arm/"training.jsonl").read_text().splitlines()]
        epochs=[record for record in records if record["event"]=="world_model_epoch"]
        assert len(epochs)==2
        if arm=="C":
            assert all(record["train_utility_crps_scored_compounds"]==8 for record in epochs)
            assert all(record["train_utility_crps"] is not None for record in epochs)
            for record in epochs:
                assert record["train_objective"]==pytest.approx(record["train_base_objective"]+record["train_utility_crps"])
        else:
            assert all(record["train_utility_crps_contribution"]==0 for record in epochs)
    for other in initial[1:]:
        assert initial[0].keys()==other.keys()
        assert all(torch.equal(value,other[key]) for key,value in initial[0].items())
    changed=replace(config,learning_rate=config.learning_rate*2,objective="predictive_nll")
    with pytest.raises(ValueError,match="Shared JEPA"):
        fit_model(dataset,splits,changed,tmp_path/"bad_shared",shared_jepa_directory=tmp_path/"A")
