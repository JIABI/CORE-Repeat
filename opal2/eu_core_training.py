"""Full fresh CORE mean fitting with separately declared EU reference roles.

No files are loaded, no data are downloaded and no role allocation is inferred.
The caller supplies prepared MODEL_TRAIN, MODEL_VALIDATION and REF_FIT arrays.
Only MODEL_TRAIN/VALIDATION accept four-well outcomes. REF_FIT accepts X only.
The distribution law is fitted separately after this mean has been frozen.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .biology_kernel import fit_chemical_anchors
from .biology_kernel_evaluation import write_json
from .gamma_supervised_loss import JointGammaCRPS
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_oof_ridge import transform_input, transform_target
from .hierarchical_geometry_experiment import train_hr, restore_target
from .hierarchical_stability_ridge import fit_validation_ridge
from .kernel_final_reference import fit_reference_bank
from .kernel_final_training import train_branch as train_a
from .lincs_biology_experiment import CONFIG as ORIGINAL_A_CONFIG
from .mechanism_response_kernel import MechanismResponseBank, MechanismResponseKernelMean
from .state_biology_kernel import StateBiologyKernelMean
from .state_biology_training import train_branch as train_state


REFERENCE_COUNT = 64
BASE_KEYS = {'ids', 'groups', 'chem', 'chem_mask'}
STATE_CONFIG = dict(mode='state_only', hidden_dim=16, support_shrinkage=2.,
    raw_increment_bound=1., incremental_penalty=.1, aggregation_scaling='train_fixed',
    scale_max_gain=32.)


def _role(data, name, *, reference=False):
    required=BASE_KEYS | ({'X'} if reference else {'Y'})
    if set(data) != required:
        raise ValueError(f'{name} requires exactly {sorted(required)}; reference roles cannot supply future outcomes')
    out={k:np.asarray(v) for k,v in data.items()}
    ids,groups=out['ids'].astype(str),out['groups'].astype(str)
    n=len(ids)
    if (ids.shape!=(n,) or groups.shape!=(n,) or not n or len(set(ids))!=n or
        any(not str(v).strip() or str(v).lower() in {'nan','none'} for v in np.r_[ids,groups])):
        raise ValueError(f'{name}: unique IDs and known chemistry groups are required')
    out.update(ids=ids,groups=groups)
    profile=out['X' if reference else 'Y']
    shape_ok=(profile.ndim==2 and profile.shape[0]==n) if reference else (
        profile.ndim==3 and profile.shape[:2]==(n,4))
    if not shape_ok or profile.shape[-1]<8 or not np.isfinite(profile).all():
        raise ValueError(f'{name}: finite prepared profiles in declared role order are required')
    chem,mask=out['chem'],out['chem_mask']
    if chem.shape!=(n,513) or mask.shape!=(n,) or mask.dtype!=bool or not np.isfinite(chem).all():
        raise ValueError(f'{name}: 512 Morgan bits plus validity and a Boolean mask are required')
    if np.any((chem[:,:512]!=0)&(chem[:,:512]!=1)) or not np.array_equal(chem[:,512],mask.astype(float)):
        raise ValueError(f'{name}: invalid binary fingerprint/validity schema')
    if np.any(mask & (chem[:,:512].sum(1)==0)) or np.any(chem[~mask,:512]):
        raise ValueError(f'{name}: fingerprint payload and availability disagree')
    # Stored assay arrays may be float32, but the unchanged full CORE fits and
    # its bank require float64. Conversion is numerical, not a new transform.
    out['X' if reference else 'Y']=np.asarray(profile,dtype=np.float64)
    out['chem']=np.asarray(chem,dtype=np.float64)
    return out


def validate_roles(model_train, model_validation, reference_fit):
    """Validate separation before creating output or computing any targets."""
    parts=[_role(model_train,'MODEL_TRAIN'),_role(model_validation,'MODEL_VALIDATION'),
           _role(reference_fit,'REF_FIT',reference=True)]
    if len({p.get('X',p.get('Y')).shape[-1] for p in parts})!=1:
        raise ValueError('Prepared profile dimensions must agree across declared roles')
    for i,j in ((0,1),(0,2),(1,2)):
        if set(parts[i]['ids'])&set(parts[j]['ids']) or set(parts[i]['groups'])&set(parts[j]['groups']):
            raise ValueError('An identity or chemical group crosses MODEL_TRAIN, MODEL_VALIDATION or REF_FIT')
    if len(parts[0]['ids'])<10 or len(set(parts[0]['groups']))<5:
        raise ValueError('Full CORE requires at least ten MODEL_TRAIN rows and five training groups')
    if not parts[0]['chem_mask'].all():
        raise ValueError('The unchanged A reference-bank fit requires available MODEL_TRAIN chemistry')
    return tuple(parts)


def choose_reference_ids(reference_fit, chemical_metadata, seed):
    """Original fixed-seed distinct-fingerprint rule, restricted to REF_FIT."""
    ref=_role(reference_fit,'REF_FIT',reference=True)
    anchor=fit_chemical_anchors(ref['chem'],ref['chem_mask'],ref['ids'],ref['ids'].tolist(),
                               chemical_metadata,max_anchors=REFERENCE_COUNT)
    fp=np.asarray(anchor['training_fingerprints'],dtype=np.uint8)
    available=np.asarray(anchor['training_available'],bool)
    lookup={v:i for i,v in enumerate(anchor['train_ids'])}
    ordered=sorted(lookup)
    order=np.random.default_rng(seed).permutation(len(ordered))
    chosen,seen=[],set()
    for position in order:
        oid=ordered[position];row=lookup[oid];key=np.packbits(fp[row]).tobytes()
        if available[row] and key not in seen:
            chosen.append(oid);seen.add(key)
        if len(chosen)==REFERENCE_COUNT:break
    if len(chosen)!=REFERENCE_COUNT:
        raise ValueError('REF_FIT needs 64 distinct available Morgan fingerprints; do not borrow validation/query anchors')
    return chosen


def unknown_biology(n):
    """True empty annotation vocabularies; no invented biological input."""
    return dict(target=np.empty((n,0),float),moa=np.empty((n,0),float),
                target_mask=np.zeros(n,bool),moa_mask=np.zeros(n,bool))


def _tensor_biology(data):
    return {k:torch.as_tensor(v,dtype=torch.bool if k.endswith('_mask') else torch.float64)
            for k,v in data.items()}


def fit_complete_eu_core(model_train, model_validation, reference_fit, metadata, output, *, seed=20260917):
    """Train RIDGE -> validation-best HR -> A_OLD_GENERIC30 -> STATE50.

    Returns the complete mean model and base joint RIDGE error scatter. It does
    not fit AMP_EMP_LOCAL, select a policy, load a prior domain model or accept
    distribution-calibration/query outcomes. Epoch budgets are not shortened.
    ``metadata['chemical']`` declares the existing Morgan-512+validity schema.
    """
    train,valid,ref=validate_roles(model_train,model_validation,reference_fit)
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('Use a declared nonnegative integer seed')
    chemical=deepcopy(metadata['chemical'])
    config=deepcopy(ORIGINAL_A_CONFIG)
    if config['stage_epochs']!=30 or config['max_anchors']!=64 or config['hidden_dim']!=16:
        raise ValueError('Original full A recipe changed')
    anchors=choose_reference_ids(ref,chemical,int(seed)+config['reference_seed_offset'])
    folder=Path(output)
    if folder.exists():raise FileExistsError('Preserve previous CORE fits; use a fresh output directory')
    folder.mkdir(parents=True)
    ids=np.concatenate((train['ids'],valid['ids'],ref['ids']))
    fit=np.arange(len(train['ids']));vi=np.arange(len(fit),len(fit)+len(valid['ids']))
    joined=np.r_[fit,vi]
    y=np.concatenate((train['Y'],valid['Y']))
    first=np.concatenate((train['Y'][:,0],valid['Y'][:,0],ref['X']))
    chem=np.concatenate((train['chem'],valid['chem'],ref['chem']))
    mask=np.concatenate((train['chem_mask'],valid['chem_mask'],ref['chem_mask']))
    grams=profiles_to_gram(torch.as_tensor(y,dtype=torch.float64)).numpy()
    raw=gram_to_coordinates(torch.as_tensor(grams)).numpy()
    gains=gram_gains(torch.as_tensor(grams)).numpy()
    ridge,stats=fit_validation_ridge(train['Y'],raw[fit],valid['Y'],raw[vi],int(seed),groups=train['groups'])
    ridge.save(folder/'ridge.npz');write_json(folder/'preprocessing.json',stats)
    x,target=transform_input(first,stats),transform_target(raw,stats)
    hr,hr_epoch=train_hr(folder/'HR_fit',ridge.coefficient,ridge.intercept,x[joined],target,
                        fit,vi,int(seed)+401,ids[joined])
    hr.eval().requires_grad_(False)
    # This is a bank-legal metadata pool, NOT the backbone outcome fitting pool.
    bank_pool=np.concatenate((train['ids'],ref['ids'])).tolist()
    local=fit_reference_bank(x,chem,mask,ids,bank_pool,train['ids'].tolist(),anchors,chemical)
    bio=unknown_biology(len(ids))
    bank=MechanismResponseBank.fit(local,bio,ids,train['ids'].tolist(),target_names=[],moa_names=[])
    bank.save(folder/'bank.pt')
    packed=bank.pack_information(torch.as_tensor(chem),_tensor_biology(bio)).numpy()
    gamma_scale=float(np.std(gains[fit,2],ddof=1))
    objective=JointGammaCRPS(ridge.covariance,stats['u_center'],stats['u_scale'],gamma_scale)
    torch.save(objective.state_dict(),folder/'gamma_objective_state.pt')
    branch_seed=int(seed)+config['branch_seed_offset']
    torch.manual_seed(branch_seed)
    base=MechanismResponseKernelMean(hr,bank,mode='old_generic',
        incremental_penalty=config['incremental_penalty'],hidden_dim=config['hidden_dim']).double()
    train_a(folder/'A_OLD_GENERIC',base,objective,x[joined],packed[joined],mask[joined],
            target,gains[:,2],fit,vi,branch_seed,ids[joined],config,'weighted')
    torch.manual_seed(branch_seed)
    state=StateBiologyKernelMean(base,**STATE_CONFIG).double()
    lookup={v:i for i,v in enumerate(ids)}
    ar=np.asarray([lookup[v] for v in anchors],int)
    state_state=state.fit_input_state(torch.as_tensor(x[fit]),ids=train['ids'],
        reference_x=torch.as_tensor(x[ar]),reference_ids=anchors)
    agg=state.fit_aggregation_scale(torch.as_tensor(x[fit]),torch.as_tensor(packed[fit]),
        torch.as_tensor(mask[fit]),ids=train['ids'])
    state_cfg=dict(config,stage_epochs=50)
    train_state(folder/'STATE50',state,objective,x[joined],packed[joined],mask[joined],
        target,gains[:,2],fit,vi,branch_seed,ids[joined],state_cfg)
    state.eval().requires_grad_(False)
    write_json(folder/'input_state.json',state_state);write_json(folder/'aggregation_scale.json',agg)
    report=dict(recipe='full RIDGE + validation-selected HR + A_OLD_GENERIC30 + STATE50',
        seed=int(seed),branch_seed=branch_seed,torch_threads=torch.get_num_threads(),
        model_train_ids=train['ids'].tolist(),model_validation_ids=valid['ids'].tolist(),
        reference_fit_ids=ref['ids'].tolist(),reference_ids=anchors,
        backbone_fit_ids=train['ids'].tolist(),branch_fit_ids=train['ids'].tolist(),
        bank_legal_metadata_pool_ids=bank_pool,reference_group_exclusion=True,
        references_also_in_backbone_fit=False,reference_future_outcomes_supplied=False,
        distribution_fit_outcomes_supplied=False,query_outcomes_supplied=False,
        prior_fold_models_loaded=False,biology_active=False,jepa_active=False,
        target_dimension=0,moa_dimension=0,chemical_dimension=513,
        reference_selection='ID-sorted seeded permutation; first64 distinct available Morgan fingerprints within REF_FIT',
        mean_feature_dimension=first.shape[1],hr_best_epoch=int(hr_epoch),a_checkpoint_epoch=30,state_checkpoint_epoch=50,
        mean_validation_used_for_preprocessing=False,mean_validation_used_for_gradient=False,
        base_covariance_frame='MODEL_TRAIN-standardized nine-dimensional Gram coordinates',
        base_scatter='RIDGE grouped-OOF prediction-error second moment; AMP_EMP_LOCAL fitted separately',
        a_config=config,state_config=state_cfg,state_architecture=STATE_CONFIG,
        role_adaptation='same full architecture and stage budgets; distinct REF_FIT no longer participates in RIDGE/HR fitting',
        formal_certificate=False)
    write_json(folder/'complete.json',report)
    covariance=np.asarray(ridge.covariance).copy()
    scale=np.asarray(stats['u_scale'])
    return dict(model=state,stats=stats,base_covariance=covariance,
                raw_base_covariance=covariance*scale[:,None]*scale[None,:],ridge=ridge,report=report)


def predict_eu_core(model, stats, first_well, chemistry, chemistry_mask):
    """X/chemistry-only predictor; no future role, realized Gamma or biology API."""
    if not isinstance(model,StateBiologyKernelMean) or model.mode!='state_only' or model.base_a.mode!='old_generic':
        raise TypeError('Expected the complete state-only CORE mean')
    x=np.asarray(first_well,float);chem=np.asarray(chemistry,float);mask=np.asarray(chemistry_mask)
    if x.ndim!=2 or chem.shape!=(len(x),513) or mask.shape!=(len(x),) or mask.dtype!=bool:
        raise ValueError('Aligned first-well profiles and Morgan chemistry are required')
    if not np.isfinite(x).all() or not np.isfinite(chem).all():raise ValueError('Finite inference inputs required')
    if np.any((chem[:,:512]!=0)&(chem[:,:512]!=1)) or not np.array_equal(chem[:,512],mask.astype(float)):
        raise ValueError('Inference chemistry violates the Morgan+validity schema')
    if np.any(mask & (chem[:,:512].sum(1)==0)) or np.any(chem[~mask,:512]):
        raise ValueError('Inference fingerprint payload and availability disagree')
    tx=transform_input(x,stats)
    model.eval()
    with torch.no_grad():
        mean=model(torch.as_tensor(tx),torch.as_tensor(chem),torch.as_tensor(mask)).numpy()
    if mean.shape!=(len(x),9) or not np.isfinite(mean).all():raise FloatingPointError('Invalid complete CORE mean')
    return dict(mean_u=mean,raw_mean=restore_target(mean,stats),input_x=tx,
                log_amplitude=np.log(np.linalg.norm(x,axis=1)),norm2_per_feature=np.square(x).mean(1))
