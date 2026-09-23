"""Synthetic engineering checks only; no real experimental data or training."""
import inspect
import json

import numpy as np
import pytest
import torch

from opal2.geometry_kernel_replacement import ReplacementBasisBank,GeometryKernelReplacementMean
from opal2.hierarchical_geometry import RidgeResidualMean


def example(mode='structured',seed=23):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(12,7))
    bits=np.asarray([[(i>>j)&1 for j in range(8)] for i in range(1,13)],dtype=float)
    chem=np.column_stack((bits,np.ones(len(bits))))
    mask=np.ones(len(x),dtype=bool)
    ids=np.asarray([f'unit_{i}' for i in range(len(x))])
    metadata=dict(fingerprint_indices=list(range(8)),validity_index=8,kind='synthetic binary')
    bank=ReplacementBasisBank.fit(x,chem,mask,ids,ids[:9],metadata,max_anchors=5)
    hr=RidgeResidualMean(7,rng.normal(size=(7,9))*.1,np.zeros(9),dropout=.4)
    with torch.no_grad():
        hr.network[-1].weight.normal_(0,.2)
        hr.network[-1].bias.normal_(0,.1)
    model=GeometryKernelReplacementMean(hr,bank,mode=mode)
    return model,hr,bank,torch.tensor(x),torch.tensor(chem),torch.tensor(mask),ids,metadata


@pytest.mark.parametrize('mode',['mlp','generic','structured'])
def test_exact_initial_hr_freezing_real_path_gradients_and_bound(mode):
    model,hr,_,x,chem,mask,_,_=example(mode)
    hr.eval();reference=hr(x).detach()
    model.train()
    assert not model.base_hr.training
    assert all(not p.requires_grad for p in model.base_hr.parameters())
    assert torch.equal(model(x,chem,mask),reference)
    frozen={k:v.clone() for k,v in model.base_hr.state_dict().items()}
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.02,weight_decay=0.)
    target=reference+.15
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result=model.loss(x,chem,mask,target)
        torch.testing.assert_close(result['loss'],result['mean_mse']+.1*result['incremental_mse'])
        result['loss'].backward()
        assert model.output.weight.grad.abs().sum()>0
        if step:
            first=model.path[0]
            gradient=first.weight.grad if mode=='mlp' else first.spline_weight.grad
            assert gradient.abs().sum()>0
            assert all(p.grad is None for p in model.base_hr.parameters())
        optimizer.step()
    assert not torch.equal(model(x,chem,mask),reference)
    for key,value in frozen.items():
        assert torch.equal(model.base_hr.state_dict()[key],value)
    result=model.loss(x,chem,mask,target)
    torch.testing.assert_close(result['incremental_mse'],(model(x,chem,mask)-reference).square().mean())
    with torch.no_grad():
        model.output.bias.copy_(torch.linspace(-100,100,9))
    details=model.diagnostics(x,chem,mask)
    assert details['total_correction_max']<=.5+1e-15
    torch.testing.assert_close(details['kernel_raw'],details['path_output']+details['output_bias'])


def test_explicit_basis_only_path_common_blocks_and_matched_kernel_parameters():
    structured,_,bank,x,chem,mask,_,_=example('structured')
    generic,*_=example('generic')
    mlp,*_=example('mlp')
    description=bank(x,chem,mask)
    a=bank.basis_blocks(description,'generic');b=bank.basis_blocks(description,'structured')
    assert torch.equal(a['morphology'],b['morphology'])
    assert torch.equal(a['scalar'],b['scalar'])
    assert not torch.equal(a['chemical'],b['chemical'])
    k=bank.anchor_count
    assert a['chemical'].shape==(len(x),3*k)
    assert a['morphology'].shape==(len(x),3*k)
    assert a['scalar'].shape==(len(x),9)
    t=description['tanimoto']
    torch.testing.assert_close(b['chemical']*bank.chemical_structured_scale,
                               torch.cat((t,t.square(),t.pow(4)),-1))
    count=lambda m:sum(p.numel() for p in m.trainable_parameters())
    assert count(generic)==count(structured)
    assert count(mlp)!=count(structured)
    for model in (generic,structured):
        captured=[]
        handle=model.path.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
        details=model.diagnostics(x,chem,mask)
        handle.remove()
        assert torch.equal(captured[0],model.bank.basis(model.bank(x,chem,mask),model.mode))
        assert torch.equal(details['basis_values'],captured[0])
        # Registered trainable modules have exactly one path and one output head.
        names={name.split('.')[0] for name,p in model.named_parameters() if p.requires_grad}
        assert names=={'path','output'}
        assert model.config['descriptor_bypass'] is False
        assert model.config['coefficient_l1_scaling'] is False
        with torch.no_grad():
            model.output.weight.normal_(0,.1)
        baseline=model(x,chem,mask).detach()
        original=model.bank.basis_blocks
        model.bank.basis_blocks=lambda description,mode:{key:torch.zeros_like(value)
            for key,value in original(description,mode).items()}
        assert not torch.equal(model(x,chem,mask),baseline)


def test_entire_block_training_rms_and_no_held_out_fit_information():
    _,_,bank,x,chem,mask,ids,metadata=example()
    altered_x=x.numpy().copy();altered_chem=chem.numpy().copy()
    altered_x[9:]=np.nan;altered_chem[9:]=np.nan
    other=ReplacementBasisBank.fit(altered_x,altered_chem,mask.numpy(),ids,ids[:9],metadata,max_anchors=5)
    assert json.dumps(bank.config,sort_keys=True)==json.dumps(other.config,sort_keys=True)
    for key,value in bank.state_dict().items():
        assert torch.equal(value,other.state_dict()[key])
    d=bank(x[:9],chem[:9],mask[:9])
    for mode in ('generic','structured'):
        for value in bank.basis_blocks(d,mode).values():
            torch.testing.assert_close(value.square().mean().sqrt(),torch.tensor(1.,dtype=x.dtype))
    # The rule preserves between-object magnitude instead of normalizing each row.
    scalar=bank.basis_blocks(d,'structured')['scalar']
    assert scalar.square().mean(-1).std()>.01
    assert bank.config['fitting_ids']==list(ids[:9])


@pytest.mark.parametrize('mode',['mlp','generic','structured'])
def test_complete_roundtrip_and_missing_chemistry_fallback(mode,tmp_path):
    model,_,bank,x,chem,mask,ids,_=example(mode)
    with torch.no_grad():
        model.output.weight.normal_(0,.1);model.output.bias.fill_(.3)
    expected=model(x,chem,mask).detach()
    copied=GeometryKernelReplacementMean.from_config(json.loads(json.dumps(model.config)))
    copied.load_state_dict(model.state_dict());copied.train()
    assert torch.equal(copied(x,chem,mask),expected)
    model.save(tmp_path/'model.pt')
    loaded=GeometryKernelReplacementMean.load(tmp_path/'model.pt')
    assert torch.equal(loaded(x,chem,mask),expected)
    assert all(not p.requires_grad for p in loaded.base_hr.parameters())
    bank.save(tmp_path/'bank.pt');restored=ReplacementBasisBank.load(tmp_path/'bank.pt')
    for kernel_mode in ('generic','structured'):
        assert torch.equal(restored.basis(restored(x,chem,mask),kernel_mode),
                           bank.basis(bank(x,chem,mask),kernel_mode))
    unknown=chem.clone();unknown[0]=torch.nan
    available=mask.clone();available[0]=False
    output=model(x,unknown,available)
    assert torch.equal(output[0],model.base_hr(x)[0])
    assert not {'target','future','target_y'}.intersection(inspect.signature(model.forward).parameters)


def test_no_available_training_chemistry_keeps_complete_hr():
    _,hr,_,x,chem,mask,ids,metadata=example()
    absent=np.zeros(len(x),dtype=bool)
    bank=ReplacementBasisBank.fit(x.numpy(),chem.numpy(),absent,ids,ids[:9],metadata,max_anchors=5)
    assert bank.anchor_count==0 and bank.basis_size==9
    for mode in ('mlp','generic','structured'):
        model=GeometryKernelReplacementMean(hr,bank,mode=mode)
        with torch.no_grad():
            model.output.bias.fill_(.8)
        hr.eval()
        assert torch.equal(model(x,chem,mask),hr(x))
