"""Synthetic API/isolation checks; these do not train an assay model."""
from copy import deepcopy
import inspect

import numpy as np
import pytest
import torch

from opal2 import eu_core_training as module
from opal2.gram_oof_ridge import fit_preprocessing, transform_input
from opal2.gram_geometry import gram_to_coordinates, profiles_to_gram
from opal2.hierarchical_geometry import RidgeResidualMean
from opal2.kernel_final_reference import fit_reference_bank
from opal2.mechanism_response_kernel import MechanismResponseBank, MechanismResponseKernelMean
from opal2.state_biology_kernel import StateBiologyKernelMean


def fixture():
    rng=np.random.default_rng(448)
    def role(name,n,reference=False):
        chem=np.column_stack((rng.integers(0,2,size=(n,512)),np.ones(n))).astype(float)
        r=dict(ids=np.asarray([f'{name}{i:03}' for i in range(n)]),
               groups=np.asarray([f'{name}group{i:03}' for i in range(n)]),
               chem=chem,chem_mask=np.ones(n,bool))
        r['X' if reference else 'Y']=rng.normal(size=(n,12) if reference else (n,4,12))
        return r
    return role('fit',20),role('val',8),role('ref',70,True),dict(
        chemical=dict(kind='Morgan binary fingerprint',radius=2,bits=512,final_coordinate='valid_SMILES_indicator'))


def test_guard_rejects_reference_future_and_group_or_identity_crossing(tmp_path):
    fit,val,ref,meta=fixture()
    module.validate_roles(fit,val,ref)
    bad=deepcopy(ref);bad['Y']=np.ones((len(ref['ids']),4,12))
    with pytest.raises(ValueError,match='reference roles cannot supply future'):
        module.fit_complete_eu_core(fit,val,bad,meta,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    for field in ('ids','groups'):
        bad=deepcopy(ref);bad[field][0]=fit[field][0]
        with pytest.raises(ValueError,match='crosses'):
            module.validate_roles(fit,val,bad)
    bad=deepcopy(val);bad['groups'][0]=fit['groups'][0]
    with pytest.raises(ValueError,match='crosses'):
        module.validate_roles(fit,bad,ref)


def test_chemistry_schema_is_bits_plus_validity_not_molecular_weight():
    fit,val,ref,_=fixture()
    bad=deepcopy(fit);bad['chem'][:,512]=400.
    with pytest.raises(ValueError,match='validity schema'):module.validate_roles(bad,val,ref)
    bad=deepcopy(fit);bad['chem'][0,1]=.1
    with pytest.raises(ValueError,match='binary fingerprint'):module.validate_roles(bad,val,ref)
    bad=deepcopy(fit);bad['chem_mask'][0]=False;bad['chem'][0]=0.
    with pytest.raises(ValueError,match='available MODEL_TRAIN chemistry'):module.validate_roles(bad,val,ref)


def test_float32_storage_is_converted_to_unchanged_float64_training_dtype():
    fit,val,ref,_=fixture()
    for part in (fit,val,ref):
        for key in ('Y','X','chem'):
            if key in part:part[key]=part[key].astype(np.float32)
    checked=module.validate_roles(fit,val,ref)
    for part in checked:
        assert part['chem'].dtype==np.float64
        assert part.get('X',part.get('Y')).dtype==np.float64


def test_anchor_selection_only_REF_ids_and_chemistry_not_X():
    _,_,ref,meta=fixture()
    a=module.choose_reference_ids(ref,meta['chemical'],81)
    changed=deepcopy(ref);changed['X']*=1000
    assert module.choose_reference_ids(changed,meta['chemical'],81)==a
    assert len(a)==64 and set(a)<=set(ref['ids'])
    lookup={v:i for i,v in enumerate(ref['ids'])}
    assert len({np.packbits(ref['chem'][lookup[v],:512].astype(np.uint8)).tobytes() for v in a})==64
    too_small={k:v[:63] for k,v in ref.items()}
    with pytest.raises(ValueError,match='64 distinct'):module.choose_reference_ids(too_small,meta['chemical'],81)


def test_genuine_empty_biology_full_models_forward_and_state_only_invariance():
    fit,val,ref,meta=fixture()
    torch.manual_seed(72)
    ids=np.r_[fit['ids'],val['ids'],ref['ids']]
    first=np.concatenate((fit['Y'][:,0],val['Y'][:,0],ref['X']))
    chem=np.concatenate((fit['chem'],val['chem'],ref['chem']))
    mask=np.concatenate((fit['chem_mask'],val['chem_mask'],ref['chem_mask']))
    raw=gram_to_coordinates(profiles_to_gram(torch.tensor(fit['Y']))).numpy()
    stats=fit_preprocessing(fit['Y'],raw)
    x=transform_input(first,stats);anchors=module.choose_reference_ids(ref,meta['chemical'],81)
    local=fit_reference_bank(x,chem,mask,ids,np.r_[fit['ids'],ref['ids']].tolist(),
        fit['ids'].tolist(),anchors,meta['chemical'])
    bio=module.unknown_biology(len(ids))
    bank=MechanismResponseBank.fit(local,bio,ids,fit['ids'],target_names=[],moa_names=[])
    assert bank.target_dim==bank.moa_dim==0
    assert bank.config['fitting_supported_pairs_by_channel']==[0,0,0]
    hr=RidgeResidualMean(13,np.zeros((13,9)),np.zeros(9)).double()
    base=MechanismResponseKernelMean(hr,bank,mode='old_generic',hidden_dim=16).double()
    with torch.no_grad():base.output.weight.normal_(0,.01)
    model=StateBiologyKernelMean(base,**module.STATE_CONFIG).double()
    lookup={v:i for i,v in enumerate(ids)};ar=[lookup[v] for v in anchors]
    packed=bank.pack_information(torch.tensor(chem),module._tensor_biology(bio))
    model.fit_input_state(torch.tensor(x[:20]),fit['ids'],torch.tensor(x[ar]),anchors)
    model.fit_aggregation_scale(torch.tensor(x[:20]),packed[:20],torch.tensor(mask[:20]),ids=fit['ids'])
    assert sum(p.numel() for p in base.trainable_parameters())==3837
    assert sum(p.numel() for p in model.trainable_parameters())==6770
    with torch.no_grad():
        for ch in model.channels:ch.output.weight.normal_(0,.01)
    model.eval()
    out=module.predict_eu_core(model,stats,first,chem,mask)
    with torch.no_grad():
        expected=model(torch.tensor(x),packed,torch.tensor(mask))
        different=model(torch.tensor(x),packed,torch.tensor(mask),bio=dict(any_annotation='ignored in state_only'))
    np.testing.assert_array_equal(out['mean_u'],expected.numpy())
    assert torch.equal(expected,different)
    assert out['mean_u'].shape==(98,9) and np.isfinite(out['raw_mean']).all()
    assert tuple(inspect.signature(module.predict_eu_core).parameters)==(
        'model','stats','first_well','chemistry','chemistry_mask')


def test_original_fixed_training_recipe_is_retained():
    cfg=module.ORIGINAL_A_CONFIG
    assert cfg['stage_epochs']==30 and cfg['max_epochs']==100
    assert cfg['train_pairs']==64 and cfg['validation_pairs']==128
    assert cfg['gamma_weight']==1. and cfg['learning_rate']==.0003
    assert module.STATE_CONFIG['mode']=='state_only'
    assert module.STATE_CONFIG['hidden_dim']==16 and module.REFERENCE_COUNT==64
