"""Synthetic interface and scientific-isolation tests; no assay results."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from opal2.conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from opal2.hierarchical_geometry import RidgeResidualMean
from opal2.mechanism_response_kernel import MechanismResponseBank, MechanismResponseKernelMean, MODES


def fixture(mode='bio_structured', *, n=14, fit_n=10, anchors=4):
    torch.manual_seed(310);rng=np.random.default_rng(913)
    x=torch.tensor(rng.normal(size=(n,7)))
    bits=np.asarray([[(i>>j)&1 for j in range(9)] for i in range(1,n+1)],float)
    chem=torch.tensor(np.column_stack((bits,np.ones(n))))
    mask=torch.ones(n,dtype=torch.bool);ids=np.asarray([f'unit_{i:03d}' for i in range(n)])
    metadata=dict(fingerprint_indices=list(range(9)),validity_index=9,kind='synthetic')
    old=LocalResponseBank.fit(x.numpy(),chem.numpy(),mask.numpy(),ids,ids[:fit_n],metadata,max_anchors=anchors)
    target=torch.tensor(rng.integers(0,2,size=(n,5)),dtype=torch.float64);target[:,0]=1
    moa=torch.tensor(rng.integers(0,2,size=(n,4)),dtype=torch.float64);moa[:,0]=1
    bio=dict(target=target,target_mask=torch.ones(n,dtype=torch.bool),
             moa=moa,moa_mask=torch.ones(n,dtype=torch.bool))
    bank=MechanismResponseBank.fit(old,bio,ids,ids[:fit_n],
        target_names=[f'T{i}' for i in range(5)],moa_names=[f'M{i}' for i in range(4)])
    hr=RidgeResidualMean(7,rng.normal(size=(7,9))*.025,np.zeros(9),dropout=.2)
    with torch.no_grad():hr.network[-1].weight.normal_(0,.05)
    model=MechanismResponseKernelMean(hr,bank,mode=mode)
    return model,old,bank,hr,x,chem,mask,bio,ids


@pytest.mark.parametrize('mode',MODES)
def test_initial_identity_all_paths_gradient_frozen_and_packed_interface(mode):
    model,_,bank,hr,x,chem,mask,bio,_=fixture(mode)
    packed=bank.pack_information(chem,bio)
    hr.eval();reference=hr(x).detach();model.train()
    assert isinstance(model,ConditionalResponseKernelMean)
    assert torch.equal(model(x,packed,mask),reference)
    assert torch.equal(model(x,chem,mask,bio=bio),reference)
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('base_hr.','bank.'))}
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=.005,weight_decay=0.)
    target=reference+torch.linspace(-.1,.2,len(x))[:,None]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        result=model.loss(x,packed,mask,target)
        torch.testing.assert_close(result['loss'],result['mean_mse']+.1*result['incremental_mse'])
        result['loss'].backward()
        assert model.output.weight.grad.abs().sum()>0
        if step>=1:
            assert model.local_coefficients.grad.abs().sum()>0
            assert model.conditioner[-1].weight.grad.abs().sum()>0
        if step>=2:assert model.conditioner[0].weight.grad.abs().sum()>0
        optimizer.step()
    for key,value in frozen.items():assert torch.equal(model.state_dict()[key],value)
    d=model.diagnostics(x,packed,mask)
    torch.testing.assert_close(d['block_contributions'].sum(1),d['kernel_raw'])
    torch.testing.assert_close(d['old_information_readout']+d['biological_readout'],d['kernel_raw'])
    assert d['total_correction_max']<=.5+1e-15
    assert torch.equal(model(x,packed,mask),model(x,chem,mask,bio=bio))


def test_exact_active_capacity_and_old_models_are_not_changed():
    model,old,bank,hr,x,chem,mask,bio,_=fixture(n=70,fit_n=66,anchors=64)
    assert bank.descriptor_dim==131
    counts=[]
    for mode in MODES:
        torch.manual_seed(112)
        new=MechanismResponseKernelMean(hr,bank,mode)
        counts.append(sum(p.numel() for p in new.trainable_parameters()))
        assert not list(new.bank.parameters())
        assert new.output.bias is None
    assert counts==[3837]*4
    for mode,oldmode in [('old_generic','conditional_generic'),('old_structured','conditional_structured')]:
        torch.manual_seed(98);new=MechanismResponseKernelMean(hr,bank,mode)
        torch.manual_seed(98);previous=ConditionalResponseKernelMean(hr,old,oldmode)
        with torch.no_grad():
            previous.output.weight.normal_(0,.08);new.output.weight.copy_(previous.output.weight)
        assert torch.equal(new(x,bank.pack_information(chem,bio),mask),previous(x,chem,mask))


def test_cd_same_information_only_response_changes_and_missing_is_exact_old_generic():
    _,old,bank,hr,x,chem,mask,bio,_=fixture()
    torch.manual_seed(77);c=MechanismResponseKernelMean(hr,bank,'bio_generic')
    torch.manual_seed(77);d=MechanismResponseKernelMean(hr,bank,'bio_structured')
    with torch.no_grad():
        c.output.weight.normal_(0,.1);d.output.weight.copy_(c.output.weight)
    dc=c.diagnostics(x,chem,mask,bio=bio);dd=d.diagnostics(x,chem,mask,bio=bio)
    for key in ('descriptors','gate','biological_values','biological_support','morphology_basis','scalar_basis'):
        assert torch.equal(dc[key],dd[key])
    assert not torch.equal(dc['chemical_basis'],dd['chemical_basis'])
    assert not torch.equal(dc['mean'],dd['mean'])
    missing={k:v.clone() for k,v in bio.items()}
    missing['target_mask'].fill_(False);missing['moa_mask'].fill_(False)
    missing['target'].fill_(torch.nan);missing['moa'].fill_(torch.nan)
    torch.manual_seed(77);previous=ConditionalResponseKernelMean(hr,old,'conditional_generic')
    previous.output.weight.data.copy_(c.output.weight)
    for model in (c,d):
        assert torch.equal(model(x,chem,mask,bio=missing),previous(x,chem,mask))
        assert model.diagnostics(x,chem,mask,bio=missing)['biological_readout'].count_nonzero()==0
    # A/B do not consume known biological similarities in the prediction.
    for mode in ('old_generic','old_structured'):
        m=MechanismResponseKernelMean(hr,bank,mode)
        m.output.weight.data.normal_(0,.1)
        assert torch.equal(m(x,chem,mask,bio=bio),m(x,chem,mask,bio=missing))


def test_overlap_masks_product_and_zero_anchored_response():
    _,_,bank,_,_,_,_,bio,_=fixture()
    changed={k:v.clone() for k,v in bio.items()}
    changed['target_mask'][0]=False;changed['moa_mask'][1]=False
    values,support=bank.biological_descriptors(changed)
    assert not support[0,:,0].any() and support[0,:,1].all() and not support[0,:,2].any()
    assert support[1,:,0].all() and not support[1,:,1].any() and not support[1,:,2].any()
    torch.testing.assert_close(values[...,2],values[...,0]*values[...,1])
    for mode in ('generic','structured'):
        response=bank.biological_response(torch.zeros_like(values),support,mode)
        assert response.count_nonzero()==0
    expected=(bio['target']/torch.linalg.vector_norm(bio['target'],dim=1)[:,None]) @ (
        bank.anchor_target/torch.linalg.vector_norm(bank.anchor_target,dim=1)[:,None]).T
    torch.testing.assert_close(bank.biological_descriptors(bio)[0][...,0],expected)


def test_biology_scalers_are_fit_only_and_old_buffers_preserved():
    _,old,bank,_,_,_,_,bio,ids=fixture()
    for key,value in old.state_dict().items():assert torch.equal(bank.state_dict()[key],value)
    corrupted={k:v.clone() for k,v in bio.items()}
    corrupted['target'][10:]=torch.nan;corrupted['moa'][10:]=torch.nan
    other=MechanismResponseBank.fit(old,corrupted,ids,ids[:10],target_names=bank.target_names,moa_names=bank.moa_names)
    assert json.dumps(bank.config,sort_keys=True)==json.dumps(other.config,sort_keys=True)
    for key,value in bank.state_dict().items():assert torch.equal(other.state_dict()[key],value)
    selected={k:v[:10] for k,v in bio.items()}
    values,support=bank.biological_descriptors(selected)
    for mode in ('generic','structured'):
        normalized=bank.biological_response(values,support,mode)/getattr(bank,'biology_'+mode+'_scale')
        torch.testing.assert_close(normalized[support].square().mean().sqrt(),torch.tensor(1.,dtype=torch.float64))


@pytest.mark.parametrize('mode',MODES)
def test_full_checkpoint_and_bank_roundtrip(mode,tmp_path):
    model,_,bank,_,x,chem,mask,bio,_=fixture(mode)
    model.output.weight.data.normal_(0,.1)
    packed=bank.pack_information(chem,bio);expected=model(x,packed,mask)
    restored=MechanismResponseKernelMean.from_config(json.loads(json.dumps(model.config)))
    restored.load_state_dict(model.state_dict())
    assert torch.equal(expected,restored(x,packed,mask))
    model.save(tmp_path/'model.pt');loaded=MechanismResponseKernelMean.load(tmp_path/'model.pt')
    assert torch.equal(expected,loaded(x,packed,mask))
    bank.save(tmp_path/'bank.pt');newbank=MechanismResponseBank.load(tmp_path/'bank.pt')
    assert torch.equal(newbank.pack_information(chem,bio),packed)
    for key,value in bank.state_dict().items():assert torch.equal(newbank.state_dict()[key],value)


def test_bad_masks_configs_nonfinite_and_vocabulary_are_rejected():
    model,old,bank,hr,x,chem,mask,bio,ids=fixture()
    with pytest.raises(ValueError):MechanismResponseKernelMean(hr,bank,'none')
    bad={k:v.clone() for k,v in bio.items()};bad['target'][0,0]=torch.nan
    with pytest.raises(ValueError):bank.pack_information(chem,bad)
    bad={k:v.clone() for k,v in bio.items()};bad['target'][0]=0
    with pytest.raises(ValueError):bank.biological_descriptors(bad)
    packed=bank.pack_information(chem,bio);packed[0,-1]=.5
    with pytest.raises(ValueError):model(x,packed,mask)
    with pytest.raises(ValueError):MechanismResponseBank.fit(old,bio,ids,ids[:9])
    changed=deepcopy(model.config);changed['biology_parameter_count']=1
    with pytest.raises(ValueError):MechanismResponseKernelMean.from_config(changed)
    changed=deepcopy(bank.config);changed['biological_structured']='Hill'
    with pytest.raises(ValueError):MechanismResponseBank.from_config(changed)
