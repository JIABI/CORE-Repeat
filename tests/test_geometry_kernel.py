"""Synthetic engineering tests; no biological data or experiment fitting."""
import inspect
import json

import numpy as np
import pytest
import torch

from opal2.geometry_kernel import DescriptorBank,GeometryKernelMean
from opal2.hierarchical_geometry import RidgeResidualMean


def example(mode='structured',seed=23):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(12,7))
    bits=np.vstack([np.array([(i>>j)&1 for j in range(8)]) for i in range(1,13)]).astype(float)
    chem=np.column_stack((bits,np.ones(len(bits))))
    mask=np.ones(len(x),dtype=bool)
    ids=np.asarray([f'unit_{i}' for i in range(len(x))])
    metadata=dict(fingerprint_indices=list(range(8)),validity_index=8,kind='synthetic binary')
    bank=DescriptorBank.fit(x,chem,mask,ids,ids[:9],metadata,max_anchors=5)
    hr=RidgeResidualMean(7,rng.normal(size=(7,9))*.1,np.zeros(9),dropout=.4)
    with torch.no_grad():
        hr.network[-1].weight.normal_(0,.2)
        hr.network[-1].bias.normal_(0,.1)
    model=GeometryKernelMean(hr,bank,mode=mode)
    return model,hr,bank,torch.tensor(x),torch.tensor(chem),torch.tensor(mask),ids,metadata


def test_zero_output_is_exact_hr_with_frozen_dropout_and_original_total_bound():
    model,hr,_,x,chem,mask,_,_=example()
    hr.eval();reference=hr(x).detach()
    for training in (True,False,True):
        model.train(training)
        assert model.base_hr.training is False
        assert all(not p.requires_grad for p in model.base_hr.parameters())
        assert torch.equal(model(x,chem,mask),reference)
        result=model.loss(x,chem,mask,reference+.1)
        assert result['incremental_mse'].item()==0
        assert result['incremental_penalty'].item()==0
    with torch.no_grad():
        model.output.bias.copy_(torch.linspace(-100,100,9))
    d=model.diagnostics(x,chem,mask)
    assert (d['mean']-d['ridge_mean']).abs().max()<=.5+1e-15
    loss=model.loss(x,chem,mask,reference+.1)
    torch.testing.assert_close(loss['incremental_mse'],(model(x,chem,mask)-reference).square().mean())
    torch.testing.assert_close(loss['loss'],loss['mean_mse']+.1*loss['incremental_mse'])


@pytest.mark.parametrize('mode',['structured','generic'])
def test_basis_and_kan_receive_real_gradients_after_zero_initialized_first_step(mode):
    model,_,_,x,chem,mask,_,_=example(mode)
    frozen={k:v.clone() for k,v in model.base_hr.state_dict().items()}
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.02,weight_decay=0.)
    target=model(x,chem,mask).detach()+.15
    model.loss(x,chem,mask,target)['loss'].backward()
    assert model.output.weight.grad.abs().sum()>0
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    model.loss(x,chem,mask,target)['loss'].backward()
    assert model.coefficients[0].spline_weight.grad.abs().sum()>0
    assert model.coefficients[-1].spline_weight.grad.abs().sum()>0
    assert model.basis_lift.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.base_hr.parameters())
    for key,value in frozen.items():
        assert torch.equal(model.base_hr.state_dict()[key],value)
    d=model.bank(x,chem,mask)
    assert model.bank.basis(d,mode).shape==(len(x),16)
    # The explicit basis path must affect the trained output, not just diagnostics.
    original=model(x,chem,mask).detach()
    with torch.no_grad():
        model.basis_lift.weight.zero_()
    assert not torch.equal(model(x,chem,mask),original)


def test_bases_have_same_inputs_and_parameter_count_but_distinct_functions():
    structured,*rest=example('structured')
    generic,*_=example('generic')
    x,chem,mask=rest[2:5]
    assert sum(p.numel() for p in structured.trainable_parameters())==sum(p.numel() for p in generic.trainable_parameters())
    a=structured.bank(x,chem,mask);b=generic.bank(x,chem,mask)
    assert torch.equal(a['descriptors'],b['descriptors'])
    basis=structured.bank.basis(a,'structured');t=a['tanimoto']
    assert torch.equal(basis,torch.cat((torch.ones_like(t[:,:1]),t,t.square(),t.pow(4)),-1))
    assert not torch.equal(basis,generic.bank.basis(b,'generic'))
    assert not any('center' in name or 'width' in name for name,_ in generic.named_parameters())


def test_fit_only_geometry_and_scales_ignore_all_held_out_payloads():
    _,_,bank,x,chem,mask,ids,metadata=example()
    altered_x=x.numpy().copy();altered_chem=chem.numpy().copy()
    altered_x[9:]=np.nan;altered_chem[9:]=np.nan
    other=DescriptorBank.fit(altered_x,altered_chem,mask.numpy(),ids,ids[:9],metadata,max_anchors=5)
    assert json.dumps(bank.config,sort_keys=True)==json.dumps(other.config,sort_keys=True)
    for key,value in bank.state_dict().items():
        assert torch.equal(value,other.state_dict()[key])
    assert set(bank.config['anchor_data']['anchor_ids']).issubset(set(ids[:9]))
    changed=altered_x.copy();changed[0,0]+=4
    modified=DescriptorBank.fit(changed,altered_chem,mask.numpy(),ids,ids[:9],metadata,max_anchors=5)
    assert not torch.equal(bank.descriptor_center,modified.descriptor_center)


def test_missing_chemistry_exact_fallback_and_support_excludes_self_without_gating():
    model,_,bank,x,chem,mask,ids,_=example()
    with torch.no_grad():
        model.output.bias.fill_(.3)
    unknown=chem.clone();unknown[0]=torch.nan
    available=mask.clone();available[0]=False
    predicted=model(x,unknown,available)
    assert torch.equal(predicted[0],model.base_hr(x)[0])
    assert not torch.equal(predicted[1:],model.base_hr(x)[1:])
    unexcluded=bank(x,chem,mask)
    excluded=bank(x,chem,mask,ids=ids)
    assert excluded['support_self_excluded'][:9].all()
    assert not excluded['support_self_excluded'][9:].any()
    torch.testing.assert_close(unexcluded['max_similarity'][:9],torch.ones(9,dtype=torch.float64))
    assert torch.all(excluded['max_similarity'][:9]<1.)
    assert torch.equal(unexcluded['descriptors'],excluded['descriptors'])
    assert torch.equal(unexcluded['availability'],excluded['availability'])
    assert bank.config['training_support_summary']['self_identity_excluded'] is True
    bad=chem.clone();bad[1,0]=.2
    with pytest.raises(ValueError,match='binary'):
        model(x,bad,mask)


def test_full_config_state_and_checkpoint_roundtrip(tmp_path):
    model,_,bank,x,chem,mask,ids,_=example('generic')
    with torch.no_grad():
        model.output.weight.normal_(0,.1)
    expected=model(x,chem,mask).detach()
    config=json.loads(json.dumps(model.config))
    restored=GeometryKernelMean.from_config(config)
    restored.load_state_dict(model.state_dict())
    restored.train()
    assert torch.equal(restored(x,chem,mask),expected)
    model.save(tmp_path/'model.pt')
    loaded=GeometryKernelMean.load(tmp_path/'model.pt')
    assert torch.equal(loaded(x,chem,mask),expected)
    assert all(not p.requires_grad for p in loaded.base_hr.parameters())
    bank.save(tmp_path/'bank.pt')
    copied=DescriptorBank.load(tmp_path/'bank.pt')
    for key,value in bank(x,chem,mask,ids=ids).items():
        assert torch.equal(value,copied(x,chem,mask,ids=ids)[key])
    assert not {'target','future','target_y'}.intersection(inspect.signature(model.forward).parameters)


def test_no_available_training_structures_is_exact_inactive_model():
    _,hr,_,x,chem,mask,ids,metadata=example()
    missing=torch.zeros_like(mask)
    bank=DescriptorBank.fit(x.numpy(),chem.numpy(),missing.numpy(),ids,ids[:9],metadata,max_anchors=5)
    assert bank.anchor_count==0
    model=GeometryKernelMean(hr,bank,mode='generic')
    with torch.no_grad():
        model.output.bias.fill_(.8)
    model.eval();hr.eval()
    assert torch.equal(model(x,chem,mask),hr(x))
    assert not model.bank(x,chem,mask)['availability'].any()
