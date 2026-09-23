"""Matched supervised identities with overlapping or disjoint reference banks."""
from copy import deepcopy
import numpy as np
import torch
from .biology_kernel import fit_chemical_anchors, validate_anchor_data
from .geometry_kernel import DescriptorBank
from .conditional_response_kernel import LocalResponseBank


def reference_allocation(ids, fit, chem, mask, metadata, seed, count=64):
    """Assign reference roles with identities/chemistry only, never outcomes."""
    ids = np.asarray(ids)
    fit = np.asarray(fit, dtype=int)
    if len(np.unique(fit)) != len(fit) or len(fit) < 2*count:
        raise ValueError('A unique fitting pool supporting two reference sets is required')
    anchor = fit_chemical_anchors(chem, mask, ids, ids[fit].tolist(), metadata, max_anchors=count)
    names = anchor['train_ids']
    fingerprints = np.asarray(anchor['training_fingerprints'], dtype=np.uint8)
    usable = np.asarray(anchor['training_available'], dtype=bool)
    local = {unit:i for i,unit in enumerate(names)}
    ordered = sorted(names)
    rng = np.random.default_rng(seed)
    ordered = [ordered[i] for i in rng.permutation(len(ordered))]
    chosen, seen = [], set()
    for unit in ordered:
        i = local[unit]
        key = np.packbits(fingerprints[i]).tobytes()
        if usable[i] and key not in seen:
            seen.add(key); chosen.append(unit)
        if len(chosen) == 2*count:
            break
    if len(chosen) != 2*count:
        raise ValueError('Not enough unique available reference fingerprints')
    disjoint, overlap = chosen[:count], chosen[count:]
    refs = set(disjoint)
    supervised = fit[np.asarray([unit not in refs for unit in ids[fit]])]
    return dict(commonbranchfit=supervised.tolist(), commonbranchfit_ids=ids[supervised].tolist(),
        reference_ids=disjoint, anchor_ids_by_mode=dict(O=overlap, D=disjoint),
        originalfit_ids=ids[fit].tolist(), allocation_seed=int(seed),
        allocation_rule='ID-sorted fixed-seed permutation, first 128 distinct usable fingerprints; no outcomes')


def fit_reference_bank(x, chem, mask, ids, originalfit_ids, supervised_ids, anchor_ids, metadata=None):
    """Fixed reference geometry, scalers fitted on the same supervised X rows."""
    ids = np.asarray(ids); lookup = {str(unit):i for i,unit in enumerate(ids)}
    if (len(set(supervised_ids)) != len(supervised_ids) or not supervised_ids
            or not set(supervised_ids).issubset(originalfit_ids)
            or not set(anchor_ids).issubset(originalfit_ids)):
        raise ValueError('Reference/scaler identities must lie in the original fitting pool')
    anchors = fit_chemical_anchors(chem, mask, ids, originalfit_ids, metadata, max_anchors=len(anchor_ids))
    fp = dict(zip(anchors['train_ids'], anchors['training_fingerprints']))
    anchors['anchor_ids'] = list(anchor_ids)
    anchors['anchor_fingerprints'] = [fp[unit] for unit in anchor_ids]
    anchors['descriptor_names'] = ['tanimoto_to:'+unit for unit in anchor_ids]+['fingerprint_bit_density']
    anchors['selection'] = 'fixed_seed_random_distinct_references_from_declared_role_pool'
    validate_anchor_data(anchors)
    config = dict(schema_version=1, input_dim=np.asarray(x).shape[1], anchor_data=anchors,
        descriptor_names=([f'chemical_T:{unit}' for unit in anchor_ids]
            +[f'initial_well_cosine:{unit}' for unit in anchor_ids]
            +['input_log_norm_coordinate','log1p_initial_input_norm','fingerprint_bit_density']),
        fitting_ids=list(supervised_ids), scalar_scale_floor=1e-6,
        reference_originalfit_ids=list(originalfit_ids),
        scope='same supervised X scalers with declared reference roles; no future-reference measurements',
        support_gate='availability only; support diagnostics do not change predictions')
    descriptor = DescriptorBank(config)
    rows = [lookup[unit] for unit in supervised_ids]
    ar = [lookup[unit] for unit in anchor_ids]
    tx = torch.as_tensor(np.asarray(x)[rows], dtype=torch.float64)
    tc = torch.as_tensor(np.asarray(chem)[rows], dtype=torch.float64)
    tm = torch.as_tensor(np.asarray(mask)[rows], dtype=torch.bool)
    ax = torch.as_tensor(np.asarray(x)[ar], dtype=torch.float64)
    with torch.no_grad():
        norms = torch.linalg.vector_norm(ax[:,:-1], dim=-1)
        descriptor.anchor_directions.copy_(ax[:,:-1]/torch.where(norms>0,norms,torch.ones_like(norms))[:,None])
        raw, _, active, _ = descriptor._raw(tx,tc,tm)
        if not active.all():
            raise ValueError('This declared experiment expects available supervised chemistry')
        descriptor.descriptor_center.copy_(raw[active].mean(0))
        scale = raw[active].std(0, unbiased=False)
        descriptor.descriptor_scale.copy_(torch.where(scale<1e-6,torch.ones_like(scale),scale))
        ac = torch.as_tensor(np.asarray(chem)[ar], dtype=torch.float64)
        am = torch.as_tensor(np.asarray(mask)[ar], dtype=torch.bool)
        anchor_raw,_,_,_ = descriptor._raw(ax,ac,am)
        descriptor.generic_centers.copy_((anchor_raw-descriptor.descriptor_center)/descriptor.descriptor_scale)
    bank = LocalResponseBank(descriptor)
    with torch.no_grad():
        description = bank(tx,tc,tm,ids=supervised_ids)
        rms = {}
        for mode in ('generic','structured'):
            blocks = bank._unscaled_blocks(description,mode)
            for name, values in blocks.items():
                key = f'chemical_{mode}' if name=='chemical' else name
                if key in rms: continue
                value = float(values[active].square().mean().sqrt())
                if not np.isfinite(value): raise ValueError('Reference basis has nonfinite fitting RMS')
                rms[key] = value
                getattr(bank,key+'_scale').fill_(max(value,bank.SCALE_FLOOR))
        bank.config.update(fitting_ids=list(supervised_ids), available_fitting_rows=int(active.sum()),
            fitting_block_rms=rms, applied_block_scales={key:max(v,bank.SCALE_FLOOR) for key,v in rms.items()},
            reference_ids=list(anchor_ids), reference_role_explicit=True)
    return bank
