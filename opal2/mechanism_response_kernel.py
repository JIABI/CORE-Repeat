"""Parameter-matched curated target/MoA profile responses on a frozen HR mean.

The biological quantities are annotation-profile overlaps, not binding,
occupancy or causal-response laws. Unknown annotation sets retain separate
masks. Their absence contributes exactly zero rather than evidence of inactivity.
Both biological arms retain the old generic chemistry/morphology/scalar path;
only the fixed response to the same three overlap quantities differs.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from .hierarchical_geometry import RidgeResidualMean


MODES = ('old_generic', 'old_structured', 'bio_generic', 'bio_structured')
BIO_KEYS = ('target', 'target_mask', 'moa', 'moa_mask')


def _names(values, size, label):
    result = [f'{label}_{i}' for i in range(size)] if values is None else list(map(str, values))
    if len(result) != size or len(set(result)) != size or any(not v for v in result):
        raise ValueError('Unique annotation vocabulary names must match '+label)
    return result


class MechanismResponseBank(LocalResponseBank):
    """A fitted old bank plus immutable reference annotation sets and two RMSs.

    ``fit`` accepts only decision-time annotations. It indexes fitting/reference
    identities before reading numeric values. Reference identities come from the
    existing bank and must lie within that bank's declared original TRAIN pool.
    Target and MoA vocabularies are separate; no MoA is assigned to every target.
    """
    def __init__(self, local_bank, target_names, moa_names):
        if not isinstance(local_bank, LocalResponseBank) or isinstance(local_bank, MechanismResponseBank):
            raise TypeError('Provide the complete fitted original LocalResponseBank')
        super().__init__(local_bank.descriptor_bank)
        for name, value in local_bank.state_dict().items():
            self.state_dict()[name].copy_(value)
        self.target_names = _names(target_names, len(target_names), 'target')
        self.moa_names = _names(moa_names, len(moa_names), 'moa')
        self.target_dim, self.moa_dim = len(target_names), len(moa_names)
        ref = self.scalar_scale
        self.register_buffer('anchor_target', ref.new_zeros(self.anchor_count, self.target_dim))
        self.register_buffer('anchor_moa', ref.new_zeros(self.anchor_count, self.moa_dim))
        self.register_buffer('anchor_target_mask', torch.zeros(self.anchor_count, dtype=torch.bool))
        self.register_buffer('anchor_moa_mask', torch.zeros(self.anchor_count, dtype=torch.bool))
        self.register_buffer('biology_generic_scale', ref.new_ones(()))
        self.register_buffer('biology_structured_scale', ref.new_ones(()))
        self.config = dict(schema_version=1, bank_type='mechanism_response',
            local_bank_config=deepcopy(local_bank.config), target_names=self.target_names,
            moa_names=self.moa_names, target_dim=self.target_dim, moa_dim=self.moa_dim,
            biological_descriptors=['target_profile_cosine', 'moa_profile_cosine', 'target_cosine*moa_cosine'],
            biological_generic='exp(-0.5*((v-1)/0.5)^2)-exp(-2), independently for each channel',
            biological_structured='identity response to [target_cosine,moa_cosine,their product]',
            biology_normalization='one supported-TRAIN uncentered RMS per entire three-channel biological block',
            scale_floor=self.SCALE_FLOOR, missing_policy='separate target/MoA masks; product requires both',
            parameter_sharing='add biological bases to existing chemical-reference positions; shared coefficients/gates/readout',
            old_basis_for_biological_arms='conditional_generic', biology_enters_conditioner=False,
            biological_claim='curated profile-overlap prior, not potency, occupancy or causal mechanism',
            packed_order=['old_chemistry', 'target_profile', 'moa_profile', 'target_mask', 'moa_mask'])

    @classmethod
    def fit(cls, local_bank, bio_all, ids, fit_ids, *, target_names=None, moa_names=None):
        if set(bio_all) != set(BIO_KEYS):
            raise ValueError('Biology requires exactly target/target_mask/moa/moa_mask')
        names, requested = list(map(str, ids)), list(map(str, fit_ids))
        if len(set(names)) != len(names) or not requested or len(set(requested)) != len(requested):
            raise ValueError('Source and fitting identities must be unique')
        if requested != list(local_bank.descriptor_bank.config['fitting_ids']):
            raise ValueError('Biology RMS must use exactly the original local-bank fitting identities')
        arrays = {k: np.asarray(v) for k, v in bio_all.items()}
        for key in ('target', 'moa'):
            if arrays[key].ndim != 2 or len(arrays[key]) != len(names):
                raise ValueError('Annotation profiles must align with source identities')
            if arrays[key+'_mask'].shape != (len(names),) or arrays[key+'_mask'].dtype != bool:
                raise ValueError('Annotation availability requires a separate boolean row mask')
        bank = cls(local_bank, _names(target_names, arrays['target'].shape[1], 'target'),
                   _names(moa_names, arrays['moa'].shape[1], 'moa'))
        lookup = {unit: i for i, unit in enumerate(names)}
        anchor_ids = list(local_bank.descriptor_bank.config['anchor_data']['anchor_ids'])
        original_fit = set(local_bank.descriptor_bank.config['anchor_data']['train_ids'])
        if not set(requested+anchor_ids).issubset(lookup) or not set(anchor_ids).issubset(original_fit):
            raise ValueError('Fitting/reference annotations must come from the declared TRAIN identities')
        def take(selected):
            rows = [lookup[v] for v in selected]
            return {k: torch.as_tensor(v[rows], dtype=torch.bool if k.endswith('_mask') else bank.scalar_scale.dtype)
                    for k, v in arrays.items()}
        references = bank._validate_bio(take(anchor_ids), len(anchor_ids))
        fitting = bank._validate_bio(take(requested), len(requested))
        with torch.no_grad():
            for key in ('target', 'moa'):
                getattr(bank, 'anchor_'+key).copy_(references[key])
                getattr(bank, 'anchor_'+key+'_mask').copy_(references[key+'_mask'])
            quantities, support = bank.biological_descriptors(fitting)
            # Invalid chemistry is already gated out of the original path.
            train_lookup = local_bank.descriptor_bank._train_lookup
            active = local_bank.descriptor_bank.training_available[
                torch.tensor([train_lookup[v] for v in requested], dtype=torch.long)]
            support = support & active[:, None, None]
            rms = {}
            for mode in ('generic', 'structured'):
                raw = bank.biological_response(quantities, support, mode)
                values = raw[support]
                value = float(values.square().mean().sqrt()) if values.numel() else 0.
                if not math.isfinite(value):
                    raise ValueError('Nonfinite fitting biological RMS')
                getattr(bank, 'biology_'+mode+'_scale').fill_(max(value, bank.SCALE_FLOOR))
                rms[mode] = value
        bank.config.update(fitting_ids=requested, anchor_ids=anchor_ids,
            fitting_supported_pairs_by_channel=support.sum((0, 1)).tolist(),
            fitting_biological_rms=rms,
            applied_biological_scales={m: float(getattr(bank, 'biology_'+m+'_scale')) for m in rms})
        return bank

    from_local = fit

    def _validate_bio(self, bio, n):
        if not isinstance(bio, dict) or set(bio) != set(BIO_KEYS):
            raise ValueError('Explicit biology requires target/target_mask/moa/moa_mask')
        clean = {}
        for key, dim in (('target', self.target_dim), ('moa', self.moa_dim)):
            value, mask = bio[key], bio[key+'_mask']
            if (not isinstance(value, torch.Tensor) or value.shape != (n, dim)
                    or value.dtype != self.scalar_scale.dtype or value.device != self.scalar_scale.device
                    or not isinstance(mask, torch.Tensor) or mask.shape != (n,) or mask.dtype != torch.bool
                    or mask.device != value.device):
                raise ValueError('Biological profile/mask shapes, dtype and device must match the bank')
            if not torch.isfinite(value[mask]).all() or torch.any(value[mask] < 0):
                raise ValueError('Known annotation profile weights must be finite and nonnegative')
            known = torch.where(mask[:, None], value, torch.zeros_like(value))
            norm = torch.linalg.vector_norm(known, dim=-1)
            if not torch.isfinite(norm).all() or torch.any(mask & (norm <= 0)):
                raise ValueError('An available annotation set must have positive finite profile norm')
            clean[key], clean[key+'_mask'] = known, mask
        return clean

    def biological_descriptors(self, bio):
        bio = self._validate_bio(bio, len(bio['target']))
        similarities, supports = [], []
        for key in ('target', 'moa'):
            query, anchor = bio[key], getattr(self, 'anchor_'+key)
            qn, an = torch.linalg.vector_norm(query, dim=-1), torch.linalg.vector_norm(anchor, dim=-1)
            q = query/torch.where(qn > 0, qn, torch.ones_like(qn))[:, None]
            a = anchor/torch.where(an > 0, an, torch.ones_like(an))[:, None]
            sim = q @ a.T
            support = bio[key+'_mask'][:, None] & getattr(self, 'anchor_'+key+'_mask')[None]
            if not torch.isfinite(sim).all() or torch.any(sim < -1e-12) or torch.any(sim > 1+1e-12):
                raise FloatingPointError('Annotation cosine left its finite nonnegative unit range')
            # Round-off only, not a biological threshold or support cutoff.
            similarities.append(torch.where(support, sim.clamp(0., 1.), torch.zeros_like(sim)))
            supports.append(support)
        t, m = similarities
        return (torch.stack((t, m, t*m), -1),
                torch.stack((supports[0], supports[1], supports[0] & supports[1]), -1))

    @staticmethod
    def biological_response(values, support, mode):
        if mode == 'generic':
            response = torch.exp(-.5*((values-1.)/.5).square())-torch.exp(values.new_tensor(-2.))
        elif mode == 'structured':
            response = values
        else:
            raise ValueError('Unknown biological response mode')
        return torch.where(support, response, torch.zeros_like(response))

    def pack_information(self, chem, bio):
        if not isinstance(chem, torch.Tensor) or chem.ndim != 2 or chem.shape[1] != self.chemical_dim:
            raise ValueError('Packing requires the original chemical input tensor')
        if chem.dtype != self.scalar_scale.dtype or chem.device != self.scalar_scale.device:
            raise ValueError('Packed chemistry dtype/device must match the bank')
        clean = self._validate_bio(bio, len(chem))
        return torch.cat((chem, clean['target'], clean['moa'], clean['target_mask'][:, None].to(chem.dtype),
                          clean['moa_mask'][:, None].to(chem.dtype)), -1)

    def unpack_information(self, packed):
        total = self.chemical_dim+self.target_dim+self.moa_dim+2
        if (not isinstance(packed, torch.Tensor) or packed.ndim != 2 or packed.shape[1] != total
                or packed.dtype != self.scalar_scale.dtype or packed.device != self.scalar_scale.device):
            raise ValueError('Packed decision information has the wrong dimension/dtype/device')
        flags = packed[:, -2:]
        if not torch.isfinite(flags).all() or torch.any((flags != 0) & (flags != 1)):
            raise ValueError('Packed annotation masks must be exact binary flags')
        c, t = self.chemical_dim, self.target_dim
        return packed[:, :c], self._validate_bio(dict(target=packed[:, c:c+t],
            moa=packed[:, c+t:-2], target_mask=flags[:, 0].bool(), moa_mask=flags[:, 1].bool()), len(packed))

    def forward(self, x, chem, mask, ids=None, bio=None):
        description = super().forward(x, chem, mask, ids=ids)
        if bio is not None:
            values, support = self.biological_descriptors(bio)
            support = support & description['availability'][:, None, None]
            description.update(biological_values=values, biological_support=support)
        return description

    def basis_blocks(self, description, mode):
        if mode not in MODES:
            raise ValueError('Unknown mechanism-response mode')
        old = 'conditional_structured' if mode == 'old_structured' else 'conditional_generic'
        blocks = super().basis_blocks(description, old)
        if mode.startswith('bio_'):
            if 'biological_values' not in description:
                raise ValueError('Biological arms require explicit decision-time annotation profiles')
            response_mode = mode.removeprefix('bio_')
            scale = getattr(self, 'biology_'+response_mode+'_scale')
            if not torch.isfinite(scale) or scale <= 0:
                raise ValueError('Biological RMS scale must be positive and finite')
            addition = self.biological_response(description['biological_values'],
                description['biological_support'], response_mode)/scale
            blocks['chemical'] = blocks['chemical']+addition
        if any(not torch.isfinite(v).all() for v in blocks.values()):
            raise FloatingPointError('Mechanism-response basis became nonfinite')
        return blocks

    @classmethod
    def from_config(cls, config):
        config = deepcopy(dict(config))
        if config.get('schema_version') != 1 or config.get('bank_type') != 'mechanism_response':
            raise ValueError('Unsupported mechanism-response bank configuration')
        bank = cls(LocalResponseBank.from_config(config['local_bank_config']), config['target_names'], config['moa_names'])
        for key in ('target_dim', 'moa_dim', 'biological_generic', 'biological_structured',
                    'biological_descriptors', 'missing_policy', 'biology_normalization', 'packed_order',
                    'old_basis_for_biological_arms', 'biology_enters_conditioner', 'scale_floor'):
            if config.get(key) != bank.config[key]:
                raise ValueError('Saved biological response definition changed: '+key)
        bank.config = config
        return bank


class MechanismResponseKernelMean(ConditionalResponseKernelMean):
    """Exactly the old active parameters; fixed biological responses add no weights.

    All four modes take the same packed information interface. A/B ignore the
    biological values in prediction; C/D use exactly the same values and masks.
    ``forward/loss`` also accept original chemistry plus ``bio=`` explicitly.
    """
    def __init__(self, base_hr, bank, mode='bio_structured', incremental_penalty=.1, hidden_dim=16):
        if not isinstance(bank, MechanismResponseBank) or mode not in MODES:
            raise ValueError('Use a fitted mechanism-response bank and one of the four declared modes')
        old = 'conditional_structured' if mode == 'old_structured' else 'conditional_generic'
        super().__init__(base_hr, bank, old, incremental_penalty, hidden_dim)
        self.mode = mode
        self.config.update(model_type='mechanism_response', mode=mode,
            parameter_count_scope='same active old local coefficients, conditioner and linear readout in all four modes',
            biology_enters_conditioner=False, biology_parameter_count=0,
            input_interface='original chemistry plus explicit bio, or bank.pack_information')

    def _details(self, x, chem, mask, ids=None, bio=None):
        if bio is None:
            chem, bio = self.bank.unpack_information(chem)
        self.base_hr._check_inputs(x)
        with torch.no_grad():
            ridge = self.base_hr.base_mean(x)
            hr_raw = self.base_hr.network(x)
            baseline = ridge+self.base_hr.correction_bound*torch.tanh(hr_raw)
        description = self.bank(x, chem, mask, ids=ids, bio=bio if self.mode.startswith('bio_') else None)
        blocks = self.bank.basis_blocks(description, self.mode)
        basis = torch.cat((blocks['chemical'], blocks['morphology'], blocks['scalar']), 1)
        logits = self.conditioner(description['descriptors']).reshape(len(x), 3, 3)
        gate = 1+.5*torch.tanh(logits)
        descriptor_gate = gate[:, self.descriptor_block, :]
        weighted = self.local_coefficients[None]*descriptor_gate
        local = (basis*weighted).sum(-1)
        raw = self.output(local)
        gated = description['availability'][:, None]*raw
        total = hr_raw+gated
        mean = ridge+self.base_hr.correction_bound*torch.tanh(total)
        if not torch.isfinite(raw).all() or not torch.isfinite(mean).all():
            raise FloatingPointError('Mechanism-response mean became nonfinite')
        contributions = torch.stack([F.linear(local*(self.descriptor_block == block)[None], self.output.weight)
                                     for block in range(3)], 1)
        biology_raw = torch.zeros_like(raw)
        if self.mode.startswith('bio_'):
            old_blocks = LocalResponseBank.basis_blocks(self.bank, description, 'conditional_generic')
            extra = blocks['chemical']-old_blocks['chemical']
            biology_local = (extra*weighted[:, :self.bank.anchor_count]).sum(-1)
            biology_raw = F.linear(biology_local, self.output.weight[:, :self.bank.anchor_count])
        return dict(mean=mean, base_hr_mean=baseline, ridge_mean=ridge, increment=mean-baseline,
            kernel_raw=raw, path_output=raw, output_bias=torch.zeros_like(raw), gated_kernel_raw=gated,
            hr_raw=hr_raw, total_raw=total, basis_values=basis.flatten(1), local_basis=basis,
            local_coefficients=self.local_coefficients, block_gate=gate, gate=gate,
            descriptor_gate=descriptor_gate, condition_logits=logits, local_activation=local,
            block_readout_contributions=contributions, block_contributions=contributions,
            chemical_readout=contributions[:, 0], morphology_readout=contributions[:, 1],
            scalar_readout=contributions[:, 2], chemical_basis=blocks['chemical'].flatten(1),
            morphology_basis=blocks['morphology'].flatten(1), scalar_basis=blocks['scalar'].flatten(1),
            biological_readout=biology_raw, old_information_readout=raw-biology_raw, **description)

    def forward(self, x, chem, mask, bio=None):
        return self._details(x, chem, mask, bio=bio)['mean']

    def loss(self, x, chem, mask, target, bio=None):
        details = self._details(x, chem, mask, bio=bio)
        mean = details['mean']
        if (not isinstance(target, torch.Tensor) or target.shape != mean.shape or target.dtype != mean.dtype
                or target.device != mean.device or not torch.isfinite(target).all()):
            raise ValueError('Finite standardized nine-coordinate target required')
        mse, increment = F.mse_loss(mean, target), details['increment'].square().mean()
        penalty = self.incremental_penalty_weight*increment
        return dict(loss=mse+penalty, mean_mse=mse, incremental_mse=increment, incremental_penalty=penalty)

    @torch.no_grad()
    def diagnostics(self, x, chem, mask, ids=None, bio=None):
        details = self._details(x, chem, mask, ids=ids, bio=bio)
        details.update(saturation_fraction=(torch.tanh(details['total_raw']).abs() >= .95).to(x.dtype).mean(),
            inherited_hr_saturation_fraction=(torch.tanh(details['hr_raw']).abs() >= .95).to(x.dtype).mean(),
            incremental_mse=details['increment'].square().mean(),
            total_correction_max=(details['mean']-details['ridge_mean']).abs().max())
        return details

    @classmethod
    def from_config(cls, config):
        config = deepcopy(dict(config))
        if (config.get('schema_version') != 1 or config.get('model_type') != 'mechanism_response'
                or config.get('dtype') not in ('float32', 'float64')):
            raise ValueError('Unsupported mechanism-response model configuration')
        bc, dtype = config['base_hr_config'], getattr(torch, config['dtype'])
        hr = RidgeResidualMean.from_config(bc, coefficient=torch.zeros(bc['input_dim'], 9, dtype=dtype),
                                           intercept=torch.zeros(9, dtype=dtype))
        model = cls(hr, MechanismResponseBank.from_config(config['bank_config']),
                    **{key: config[key] for key in ('mode', 'incremental_penalty', 'hidden_dim')})
        for key in ('gate_formula', 'coefficient_shape', 'raw_descriptor_bypass', 'output_bias',
                    'nonlinear_downstream_readout', 'object_normalization', 'biology_enters_conditioner',
                    'biology_parameter_count', 'input_interface'):
            if config.get(key) != model.config[key]:
                raise ValueError('Saved mechanism-response architecture changed: '+key)
        return model
