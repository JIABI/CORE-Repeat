"""Fidelity/numerical regression checks for the complete R2 model paths."""
import copy
import math
import pytest
import torch

from opal2.model import (MeasurementWorldModel, IdentityMatchedReferenceEncoder,
                        GroupedProfileEncoder, JointGaussian, LazyConditionalGaussian, encode_library_profiles)
from opal2.kernels import MeasurementLawBasis, MeasurementKernelOperator
from opal2.jepa import ConditionalJEPA
from test_model import make_batch, new_model, dense_full_covariance


def enrich(batch):
    b,c,d=batch["context_y"].shape
    t=batch["target_cond"].shape[1]
    k=batch["context_cond"].shape[-1]
    batch["chem_mask"]=torch.ones(b,dtype=torch.bool)
    for prefix,w in (("context",c),("target",t)):
        batch[prefix+"_panel_y"]=torch.randn(b,w,3,3,d)
        batch[prefix+"_panel_template"]=torch.randn(b,w,3,3,d)
        batch[prefix+"_panel_mask"]=torch.ones(b,w,3,3,dtype=torch.bool)
        batch[prefix+"_panel_template_mask"]=torch.ones(b,w,3,3,dtype=torch.bool)
    batch["library_y"]=torch.randn(b,4,d)
    batch["library_cond"]=torch.randn(b,4,k)
    batch["library_mask"]=torch.ones(b,4,dtype=torch.bool)
    batch["library_global_mean"]=torch.randn(b,d)
    batch["library_global_variance"]=torch.rand(b,d)
    batch["library_density"]=torch.rand(b,2)
    batch["library_count"]=torch.full((b,),100.)
    batch["context_n_cells"]=torch.full((b,c),300.)
    batch["context_n_cells_mask"]=torch.ones(b,c,dtype=torch.bool)
    return batch


def test_intergroup_attention_has_two_real_attention_blocks_and_every_coordinate():
    encoder=GroupedProfileEncoder({"a":[0,1,2],"b":[3,4]},16,2,4)
    assert len(encoder.intergroup.layers)==2
    x=torch.randn(4,5,requires_grad=True)
    encoder(x).square().sum().backward()
    assert (x.grad.abs().sum(0)>0).all()
    for block in encoder.intergroup.layers:
        assert block.self_attn.in_proj_weight.grad.abs().sum()>0


def test_chemical_missing_mask_and_true_zero_context_gaussian_prior():
    model=new_model().eval()
    batch=make_batch(c=0)
    batch["chem_mask"]=torch.tensor([True,False,False])
    before=model(batch)
    _,pm,pv=model.chemical_prior(batch["chem"],batch["chem_mask"])
    details=model._state_details(batch)
    assert torch.equal(pm,details["posterior_mean"])
    assert torch.allclose(pv,details["posterior_var"])
    batch["chem"][1:]=float("nan")
    after=model(batch)
    assert torch.equal(before.mean,after.mean)
    assert torch.equal(before.factors,after.factors)
    assert torch.isfinite(after.sample_joint(5)).all()


def test_conditional_prior_precision_and_training_posterior_have_correct_kl():
    model=new_model()
    batch=make_batch()
    details=model._state_details(batch)
    assert (details["posterior_var"]<=details["prior_var"]).all()
    q=torch.distributions.Normal(details["posterior_mean"],details["posterior_var"].sqrt())
    p=torch.distributions.Normal(details["prior_mean"],details["prior_var"].sqrt())
    expected=torch.distributions.kl_divergence(q,p).sum(-1)
    actual=model.gaussian_kl(q.loc,q.scale.square(),p.loc,p.scale.square())
    assert torch.allclose(actual,expected,atol=1e-6)
    target=torch.randn(3,3,7)
    metrics=model.loss(batch,target)
    # Predictive NLL is exact and separate from the Monte Carlo ELBO.
    expected_nll=-model(batch).joint_log_prob(target)/target.numel()
    assert torch.allclose(metrics["nll"],expected_nll)
    assert metrics["latent_kl"]>=0
    metrics["loss"].backward()
    assert model.chemical_prior.mean.weight.grad.abs().sum()>0
    assert model.chemical_prior.log_variance.weight.grad.abs().sum()>0
    assert model.evidence_head.weight.grad.abs().sum()>0


@pytest.mark.parametrize("mode",["measurement","generic","mlp"])
def test_real_panel_library_and_law_inputs_reach_loss_and_gradients(mode):
    model=new_model(kernel_mode=mode)
    batch=enrich(make_batch())
    metrics=model.loss(batch,torch.randn(3,3,7))
    assert metrics["reference_reconstruction_count"]>0
    assert metrics["reference_reconstruction"]>0
    metrics["loss"].backward()
    for parameter in (model.panel_encoder.token[0].weight,model.panel_encoder.reconstruct[-1].weight,
                      model.library_attention.in_proj_weight,model.library_global[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum()>0
    descriptors=model.descriptors(batch)
    assert descriptors.shape==(3,3,2,9)
    assert torch.equal(descriptors[...,3],torch.full((3,3,2),300.))
    assert torch.equal(descriptors[...,4],torch.ones(3,3,2))
    assert model.operator.descriptor_kind=="laws"


def test_measured_law_basis_equals_formula_and_has_inverted_u_gain():
    rho=torch.logspace(-4,4,1000)
    d=torch.zeros(1000,9)
    d[:,0]=rho; d[:,1]=.2; d[:,2]=1; d[:,3]=100; d[:,4]=1
    basis=MeasurementLawBasis()(d)
    eff=3/(1+2*.2)
    expected=.5*(rho/torch.sqrt((rho+1/eff)*(rho+1))-rho/(rho+1))
    assert torch.allclose(basis[:,12],expected,atol=1e-7)
    maximum=int(basis[:,12].argmax())
    assert 0<maximum<999
    assert basis[maximum,12]>10*basis[0,12]
    assert basis[maximum,12]>10*basis[-1,12]
    assert torch.allclose(basis[:,14],torch.full((1000,),.01))
    d[:,4]=0
    assert torch.equal(MeasurementLawBasis()(d)[:,14:17],torch.zeros(1000,3))


def test_law_geometry_uses_fixed_outcome_affine_transform():
    model=new_model().eval(); batch=make_batch()
    first=model._state_details(batch)["rho"]
    model.set_outcome_transform(torch.full((7,),4.),torch.ones(7))
    second=model._state_details(batch)["rho"]
    assert not torch.allclose(first,second)
    # Pure overall units rescale both signal and noise equally.
    model.set_outcome_transform(torch.zeros(7),torch.full((7,),3.))
    assert torch.allclose(model._state_details(batch)["rho"],first,atol=1e-6)


def test_no_information_leakage_through_masked_library_reference_or_chemistry():
    model=new_model().eval(); batch=enrich(make_batch())
    batch["library_mask"][:]=False
    batch["context_panel_mask"][:]=False
    batch["target_panel_mask"][:]=False
    batch["chem_mask"][:]=False
    first=model(batch)
    batch["library_y"][:]=float("nan");batch["library_cond"][:]=float("nan")
    for key in ("library_global_mean","library_global_variance","library_density","library_count"):
        batch[key][:]=float("nan")
    batch["context_panel_y"][:]=float("nan");batch["target_panel_y"][:]=float("nan")
    batch["chem"][:]=float("nan")
    second=model(batch)
    assert torch.equal(first.mean,second.mean)
    assert torch.equal(first.factors,second.factors)


def test_compact_panel_matches_dense_without_duplicate_loss_weighting():
    torch.manual_seed(83)
    encoder=IdentityMatchedReferenceEncoder(7,16)
    y,template=torch.randn(6,7),torch.randn(6,7)
    index=torch.tensor([[[[0,1],[2,3],[4,5]]]]).expand(3,2,-1,-1).clone()
    mask=torch.ones_like(index,dtype=torch.bool)
    direct=encoder(y[index],template[index],mask,mask)
    compact=encoder.forward_compact(y,template,torch.ones(6,dtype=torch.bool),index,mask)
    for k in (0,1,2,3):
        assert torch.allclose(direct[k],compact[k],atol=2e-6)
    assert compact[4]==6  # Six actual control occurrences, not six compounds times six.


def test_single_reference_identity_cannot_identify_affine_slope():
    encoder=IdentityMatchedReferenceEncoder(7,16)
    template=torch.randn(1,3,4,7)
    y=3*template+2
    mask=torch.ones(1,3,4,dtype=torch.bool)
    same=torch.zeros(1,3,4,dtype=torch.int64)
    result=encoder(y,template,mask,mask,same)
    assert torch.equal(result[2],torch.ones(1,7))
    different=torch.arange(4).expand(1,3,4)
    multi=encoder(y,template,mask,mask,different)
    assert (multi[2]>1).all()


def test_reference_reconstruction_excludes_own_measurement_via_other_templates():
    encoder=IdentityMatchedReferenceEncoder(7,16)
    y=torch.randn(1,3,4,7,requires_grad=True)
    # Other controls' leave-one-out templates contain the held measurement.
    template=(y.sum(-2,keepdim=True)-y)/3
    mask=torch.ones(1,3,4,dtype=torch.bool)
    captured=[]
    hook=encoder.reconstruct.register_forward_hook(lambda module,args,output: captured.append(output))
    encoder(y,template,mask,mask)
    hook.remove()
    prediction=captured[0][0,0,0]+template[0,0,0]
    grad=torch.autograd.grad(prediction.sum(),y)[0]
    assert torch.allclose(grad[0,0,0],torch.zeros(7),atol=1e-7)
    assert grad[0,0,1:].abs().sum()>0


def test_selective_probe_can_retain_bought_coordinates_as_exact_point_masses():
    torch.manual_seed(42)
    distribution=new_model().double().eval()({k:v.double() if v.is_floating_point() else v for k,v in make_batch(b=2).items()})
    observed=torch.zeros_like(distribution.mean,dtype=torch.bool);observed[0,0]=True
    y=distribution.mean.detach().clone();y[0,0]+=2
    posterior=distribution.condition(y,observed,(0,1,2),retain_observed=True)
    assert torch.equal(posterior.mean[0,0],y[0,0])
    assert torch.equal(posterior.diag_var[0,0],torch.zeros(7,dtype=torch.float64))
    assert posterior.factors[0,0].abs().sum()==0
    assert torch.equal(posterior.sample_joint(4)[:,0,0],y[0,0].expand(4,7))
    assert posterior.marginal_variance[1,0].sum()>0
    # Exact dense conditioning for unobserved coordinates verifies mixed roles.
    sigma=dense_full_covariance(distribution);obs=observed.flatten();keep=~obs
    sigma_oo=sigma[obs][:,obs]
    expected=sigma[keep][:,keep]-sigma[keep][:,obs]@torch.linalg.solve(sigma_oo,sigma[obs][:,keep])
    assert torch.allclose(dense_full_covariance(posterior)[keep][:,keep],expected,atol=1e-9)


def test_high_rank_is_real_economical_and_variance_decomposition_adds_up():
    groups={"a":list(range(0,64)),"b":list(range(64,128))}
    model=MeasurementWorldModel(groups,4,5,6,hidden_dim=32,latent_rank=32,residual_rank=8)
    batch=make_batch(d=128)
    dist=model(batch)
    assert model.factor_heads[0].weight.shape==(128,32)
    assert model.factor_heads[0].weight.numel()<32*128*32
    components=model.variance_components(batch)
    assert torch.allclose(sum(components.values()),dist.marginal_variance,atol=1e-6)


def test_exact_library_dedup_keeps_outputs_and_parameter_gradients():
    model=new_model().eval()
    y=torch.randn(3,7);cond=torch.randn(3,4)
    index=torch.tensor([[0,1,2],[2,1,0],[1,2,0]])
    mask=torch.ones_like(index,dtype=torch.bool)
    direct=encode_library_profiles(model.profile_encoder,model.condition_encoder,y[index],cond[index],mask)
    fast=encode_library_profiles(model.profile_encoder,model.condition_encoder,y[index],cond[index],mask,index)
    assert torch.allclose(direct,fast,atol=1e-6)
    parameters=list(model.profile_encoder.parameters())
    ga=torch.autograd.grad(direct.square().sum(),parameters,retain_graph=True)
    gb=torch.autograd.grad(fast.square().sum(),parameters)
    for a,b in zip(ga,gb): assert torch.allclose(a,b,atol=1e-4,rtol=1e-4)


def test_lazy_posterior_mean_density_and_sampling_equal_materialized_gaussian():
    model=new_model().double().eval()
    batch={k:v.double() if v.is_floating_point() else v for k,v in make_batch(b=2).items()}
    prior=model(batch)
    mask=torch.zeros_like(prior.mean,dtype=torch.bool);mask[0,0]=True;mask[1,0,:3]=True
    y=prior.mean.detach().clone();y[mask]+=.6
    full=prior.condition(y,mask,(1,2),lazy=False)
    lazy=prior.condition(y,mask,(1,2),lazy=True)
    assert isinstance(lazy,LazyConditionalGaussian)
    assert torch.allclose(lazy.mean,full.mean,atol=1e-10)
    target=torch.randn_like(lazy.mean)
    assert torch.allclose(lazy.joint_log_prob(target),full.joint_log_prob(target),atol=1e-9)
    draws=lazy.sample_joint(30000,torch.Generator().manual_seed(892)).flatten(1)
    assert torch.allclose(draws.mean(0),full.mean.flatten(),atol=.02)
    actual=torch.cov(draws.T)
    assert torch.allclose(actual,dense_full_covariance(full),atol=.035,rtol=.04)


def test_lazy_repeated_partial_probes_keep_original_law_and_point_masses():
    model=new_model().double().eval()
    batch={k:v.double() if v.is_floating_point() else v for k,v in make_batch(b=2).items()}
    prior=model(batch);y=prior.mean.detach().clone()+.3
    first_mask=torch.zeros_like(y,dtype=torch.bool);first_mask[0,0]=True
    first=prior.condition(y,first_mask,(0,1,2),retain_observed=True,lazy=True)
    second_mask=torch.zeros_like(y,dtype=torch.bool);second_mask[1,0]=True
    second=first.condition(y,second_mask,(0,1,2),retain_observed=True)
    direct=prior.condition(y,first_mask|second_mask,(0,1,2),retain_observed=True,lazy=False)
    assert second.parent is prior
    assert torch.allclose(second.mean,direct.mean,atol=1e-10)
    draws=second.sample_joint(10)
    assert torch.equal(draws[:,:,0],y[:,0].unsqueeze(0).expand(10,-1,-1))


def test_conditional_jepa_supports_zero_context_and_real_reference_library_inputs():
    batch=enrich(make_batch(c=0))
    encoder=GroupedProfileEncoder({"a":[0,1,2],"b":[3,4,5,6]},16)
    model=ConditionalJEPA(encoder,4,5,6)
    target=torch.randn(3,3,7)
    metrics=model.loss(batch,target)
    metrics["loss"].backward()
    assert model.library_attention.in_proj_weight.grad.abs().sum()>0
    assert model.panel_encoder.token[0].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.teacher_encoder.parameters())
