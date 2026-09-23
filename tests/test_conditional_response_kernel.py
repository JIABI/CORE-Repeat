"""Synthetic architecture/scaling/gradient tests, not real assay experiments."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2.conditional_response_kernel import LocalResponseBank,ConditionalResponseKernelMean,MODES
from opal2.geometry_kernel_replacement import ReplacementBasisBank
from opal2.hierarchical_geometry import RidgeResidualMean


def example(mode='conditional_structured'):
    torch.manual_seed(31);rng=np.random.default_rng(17)
    x=rng.normal(size=(12,7))
    bits=np.asarray([[(i>>j)&1 for j in range(8)] for i in range(1,13)],float)
    chem=np.column_stack((bits,np.ones(12)))
    mask=np.ones(12,bool);ids=np.asarray([f'synthetic_{i}' for i in range(12)])
    metadata=dict(fingerprint_indices=list(range(8)),validity_index=8,kind='synthetic binary')
    bank=LocalResponseBank.fit(x,chem,mask,ids,ids[:9],metadata,max_anchors=4)
    hr=RidgeResidualMean(7,rng.normal(size=(7,9))*.05,np.zeros(9),dropout=.4)
    with torch.no_grad():
        hr.network[-1].weight.normal_(0,.1);hr.network[-1].bias.normal_(0,.03)
    model=ConditionalResponseKernelMean(hr,bank,mode=mode)
    return model,hr,bank,torch.tensor(x),torch.tensor(chem),torch.tensor(mask),ids,metadata


@pytest.mark.parametrize('mode',MODES)
def test_exact_initial_hr_successive_nonzero_gradients_and_frozen_state(mode):
    model,hr,_,x,chem,mask,_,_=example(mode)
    hr.eval();reference=hr(x).detach();model.train()
    assert torch.equal(model(x,chem,mask),reference)
    initial=model.diagnostics(x,chem,mask)
    assert torch.equal(initial['gate'],torch.ones_like(initial['gate']))
    assert torch.equal(model.local_coefficients,torch.full_like(model.local_coefficients,1/3))
    assert model.output.bias is None and not model.base_hr.training
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('base_hr.','bank.'))}
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.01,weight_decay=0.)
    target=reference+.15+torch.linspace(-.05,.05,len(x))[:,None]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        result=model.loss(x,chem,mask,target)
        torch.testing.assert_close(result['loss'],result['mean_mse']+.1*result['incremental_mse'])
        result['loss'].backward()
        assert model.output.weight.grad.abs().sum()>0
        if step>=1:
            assert model.local_coefficients.grad.abs().sum()>0
            if mode=='static_structured':assert model.condition_logits.grad.abs().sum()>0
            else:assert model.conditioner[-1].weight.grad.abs().sum()>0
        if step>=2 and mode!='static_structured':
            assert model.conditioner[0].weight.grad.abs().sum()>0
        optimizer.step()
    assert not torch.equal(model(x,chem,mask),reference)
    for key,value in frozen.items():assert torch.equal(model.state_dict()[key],value)
    assert all(p.grad is None and not p.requires_grad for p in model.base_hr.parameters())
    d=model.diagnostics(x,chem,mask)
    assert (d['gate']>=.5).all() and (d['gate']<=1.5).all()
    assert d['total_correction_max']<=.5+1e-15
    torch.testing.assert_close(d['increment'],d['mean']-reference)


def test_basis_shapes_signed_responses_and_amplitude_are_preserved():
    _,_,bank,x,chem,mask,_,_=example()
    d=bank(x,chem,mask);k=bank.anchor_count
    raw=bank._unscaled_blocks(d,'structured')
    assert bank.basis(d,'structured').shape==(12,2*k+3,3)
    t=d['tanimoto'];q=(d['descriptors']*bank.descriptor_bank.descriptor_scale+bank.descriptor_bank.descriptor_center)[:,k:2*k]
    torch.testing.assert_close(raw['chemical'],torch.stack((t,t*t,t**4),-1))
    torch.testing.assert_close(raw['morphology'],torch.stack((q,torch.tanh(2*q),q*q.abs()),-1))
    assert (q<0).any() and (q>0).any()
    inverse={key:value.clone() for key,value in d.items()}
    original=d['descriptors']*bank.descriptor_bank.descriptor_scale+bank.descriptor_bank.descriptor_center
    original[:,k:2*k]*=-1
    inverse['descriptors']=(original-bank.descriptor_bank.descriptor_center)/bank.descriptor_bank.descriptor_scale
    inverse['descriptors'][:,-3:]=-d['descriptors'][:,-3:]
    reverse=bank._unscaled_blocks(inverse,'structured')
    torch.testing.assert_close(reverse['morphology'],-raw['morphology'])
    torch.testing.assert_close(reverse['scalar'],-raw['scalar'])
    assert bank.basis(d,'structured').square().mean((1,2)).std()>.01
    scalar=d['descriptors'][:,-3:]
    torch.testing.assert_close(raw['scalar'][...,2],scalar/torch.hypot(scalar,torch.ones_like(scalar)))


def test_fit_only_block_rms_and_existing_descriptor_chemical_scales_match():
    _,_,bank,x,chem,mask,ids,metadata=example()
    old=ReplacementBasisBank.fit(x.numpy(),chem.numpy(),mask.numpy(),ids,ids[:9],metadata,max_anchors=4)
    for key,value in old.descriptor_bank.state_dict().items():
        assert torch.equal(bank.descriptor_bank.state_dict()[key],value)
    assert torch.equal(bank.chemical_generic_scale,old.chemical_generic_scale)
    assert torch.equal(bank.chemical_structured_scale,old.chemical_structured_scale)
    altered_x=x.numpy().copy();altered_chem=chem.numpy().copy()
    altered_x[9:]=np.nan;altered_chem[9:]=np.nan
    other=LocalResponseBank.fit(altered_x,altered_chem,mask.numpy(),ids,ids[:9],metadata,max_anchors=4)
    assert json.dumps(bank.config,sort_keys=True)==json.dumps(other.config,sort_keys=True)
    for key,value in bank.state_dict().items():assert torch.equal(other.state_dict()[key],value)
    fit=bank(x[:9],chem[:9],mask[:9])
    for mode in ('generic','structured'):
        for value in bank.basis_blocks(fit,mode).values():
            torch.testing.assert_close(value.square().mean().sqrt(),torch.tensor(1.,dtype=x.dtype))


def test_linear_readout_closure_no_nonlinear_bypass_and_parameter_matching():
    models=[example(mode)[0] for mode in MODES]
    _,_,_,x,chem,mask,_,_=example()
    count=lambda m:sum(p.numel() for p in m.trainable_parameters())
    assert count(models[1])==count(models[2]) and count(models[0])<count(models[1])
    assert models[0].conditioner is None
    for model in models:
        assert not any(isinstance(module,torch.nn.LayerNorm) for name,module in model.named_modules() if not name.startswith('base_hr.'))
        with torch.no_grad():model.output.weight.normal_(0,.1)
        d=model.diagnostics(x,chem,mask)
        torch.testing.assert_close(d['kernel_raw'],torch.nn.functional.linear(d['local_activation'],model.output.weight))
        torch.testing.assert_close(d['block_contributions'].sum(1),d['kernel_raw'])
        assert torch.equal(d['path_output'],d['kernel_raw'])
        assert torch.equal(d['output_bias'],torch.zeros_like(d['kernel_raw']))
        original=model.bank.basis_blocks
        model.bank.basis_blocks=lambda description,mode:{key:torch.zeros_like(value)
            for key,value in original(description,mode).items()}
        assert torch.equal(model(x,chem,mask),model.base_hr(x))


@pytest.mark.parametrize('mode',MODES)
def test_roundtrip_and_missing_chemistry_are_exact(mode,tmp_path):
    model,_,bank,x,chem,mask,_,_=example(mode)
    with torch.no_grad():model.output.weight.normal_(0,.15)
    expected=model(x,chem,mask).detach()
    restored=ConditionalResponseKernelMean.from_config(json.loads(json.dumps(model.config)))
    restored.load_state_dict(model.state_dict());restored.train()
    assert torch.equal(restored(x,chem,mask),expected)
    model.save(tmp_path/'model.pt');copied=ConditionalResponseKernelMean.load(tmp_path/'model.pt')
    assert torch.equal(copied(x,chem,mask),expected)
    bank.save(tmp_path/'bank.pt');saved_bank=LocalResponseBank.load(tmp_path/'bank.pt')
    assert torch.equal(saved_bank.basis(saved_bank(x,chem,mask),mode),bank.basis(bank(x,chem,mask),mode))
    unknown=chem.clone();unknown[0]=torch.nan
    available=mask.clone();available[0]=False
    d=model.diagnostics(x,unknown,available)
    assert torch.equal(d['mean'][0],model.base_hr(x)[0])
    assert torch.count_nonzero(d['kernel_raw'][0])==0


def test_invalid_mode_config_scale_target_and_input_are_rejected():
    model,hr,bank,x,chem,mask,_,_=example()
    with pytest.raises(ValueError):ConditionalResponseKernelMean(hr,bank,mode='unknown')
    with pytest.raises(ValueError):ConditionalResponseKernelMean(hr,bank,incremental_penalty=float('nan'))
    with pytest.raises(ValueError):ConditionalResponseKernelMean(hr,bank,hidden_dim=0)
    bad=deepcopy(model.config);bad['output_bias']=True
    with pytest.raises(ValueError,match='architecture'):ConditionalResponseKernelMean.from_config(bad)
    bad=deepcopy(bank.config);bad['morphology']='unsigned RBF'
    with pytest.raises(ValueError,match='definition'):LocalResponseBank.from_config(bad)
    with pytest.raises(ValueError):model.loss(x,chem,mask,torch.full((12,9),torch.nan,dtype=x.dtype))
    xx=x.clone();xx[0,0]=torch.nan
    with pytest.raises(ValueError):model(xx,chem,mask)
    bank.scalar_scale.zero_()
    with pytest.raises(ValueError,match='positive'):bank.basis(bank(x,chem,mask),'structured')
