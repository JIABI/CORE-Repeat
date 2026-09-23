"""Independent, support-aware relation corrections to a complete frozen A.

The two modes differ only in decision-time relations.  ``old_information``
uses chemical Tanimoto and positive initial-well morphology similarity;
``biology`` uses target-profile and MoA-profile cosine separately.  They have
the same active parameters, response functions and support shrinkage.  These
are bounded relation-response functions, not binding or pharmacological laws.

Neither mode updates the inherited conditional covariance.  No response or
future-well value enters the reference bank or the support gate.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .hierarchical_geometry import _positive_integer
from .mechanism_response_kernel import MechanismResponseKernelMean


MODES = ('old_information', 'biology')
AGGREGATION_SCALINGS = ('none', 'train_fixed')
SUPPORT_FEATURES = ('known', 'has_match', 'mean_weight', 'max_weight',
                    'effective_neighbors_over_K', 'positive_neighbors_over_K')


class _IndependentRelationChannel(nn.Module):
    """A channel owns its coefficients, bias-free readout and support gate."""
    def __init__(self, anchors, hidden_dim):
        super().__init__()
        self.local_coefficients = nn.Parameter(torch.full((anchors, 3), 1/3))
        self.gate = nn.Sequential(nn.Linear(len(SUPPORT_FEATURES), hidden_dim),
                                  nn.GELU(), nn.Linear(hidden_dim, 1))
        self.output = nn.Linear(anchors, 9, bias=False)
        # Positive support gates allow the zero-initialized readout to learn.
        # Upstream gradients begin after its first nonzero update, not before.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.output.weight)


class IndependentBiologyKernelMean(nn.Module):
    """Frozen full-A predictor plus separately selectable bounded channels.

The raw sum is bounded by ``raw_increment_bound``.  It is inserted inside
the original HR tanh, preserving the original total correction bound relative
to RIDGE.  Disabling the branch, or having no matching reference in either
channel, returns the immutable A output exactly, including after training.
"""
    def __init__(self, base_a, mode='biology', incremental_penalty=.1,
                 hidden_dim=16, support_shrinkage=2., raw_increment_bound=1.,
                 aggregation_scaling='none', scale_max_gain=32.):
        super().__init__()
        if not isinstance(base_a, MechanismResponseKernelMean) or base_a.mode != 'old_generic':
            raise TypeError('Provide the complete fitted old_generic A model')
        if mode not in MODES:
            raise ValueError('Independent mode must be old_information or biology')
        if aggregation_scaling not in AGGREGATION_SCALINGS:
            raise ValueError('Aggregation scaling must be none or train_fixed')
        if not math.isfinite(scale_max_gain) or scale_max_gain < 1.:
            raise ValueError('scale_max_gain must be finite and at least one')
        for name, value, positive in (
            ('incremental_penalty', incremental_penalty, False),
            ('support_shrinkage', support_shrinkage, True),
            ('raw_increment_bound', raw_increment_bound, True)):
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(name+' must be finite and '+('positive' if positive else 'nonnegative'))
        hidden_dim = _positive_integer(hidden_dim, 'hidden_dim')
        if base_a.bank.anchor_count < 1:
            raise ValueError('At least one fixed TRAIN reference is required')
        self.base_a = deepcopy(base_a).eval().requires_grad_(False)
        self.mode = mode
        self.incremental_penalty_weight = float(incremental_penalty)
        self.support_shrinkage = float(support_shrinkage)
        self.raw_increment_bound = float(raw_increment_bound)
        self.aggregation_scaling = aggregation_scaling
        self.scale_max_gain = float(scale_max_gain)
        self.channels = nn.ModuleList([
            _IndependentRelationChannel(self.bank.anchor_count, hidden_dim) for _ in range(2)])
        self.to(dtype=self.base_a.base_hr.coefficient.dtype,
                device=self.base_a.base_hr.coefficient.device)
        if aggregation_scaling == 'train_fixed':
            ref = self.base_a.base_hr.coefficient
            # Only opted-in models own these buffers. Schema-1 state dicts
            # therefore retain their original keys and strict-load behavior.
            self.register_buffer('aggregation_scale_gain', ref.new_ones(2))
            self.register_buffer('aggregation_scale_raw_s', ref.new_zeros(2))
            self.register_buffer('aggregation_scale_count', torch.zeros(2, dtype=torch.long, device=ref.device))
            self.register_buffer('aggregation_scale_capped', torch.zeros(2, dtype=torch.bool, device=ref.device))
            self.register_buffer('aggregation_scale_fitted', torch.zeros((), dtype=torch.bool, device=ref.device))
            self._initial_local_coefficient = float(self.channels[0].local_coefficients[0, 0].detach())
        self.config = dict(schema_version=1, model_type='independent_biology_kernel',
            base_a_config=deepcopy(base_a.config), mode=mode,
            incremental_penalty=float(incremental_penalty), hidden_dim=hidden_dim,
            support_shrinkage=float(support_shrinkage), raw_increment_bound=float(raw_increment_bound),
            dtype=str(self.base_a.base_hr.coefficient.dtype).removeprefix('torch.'),
            channels=['chemical', 'positive_morphology'] if mode == 'old_information' else ['target', 'moa'],
            fixed_reference_count=self.bank.anchor_count,
            support_features=list(SUPPORT_FEATURES),
            response_functions='[w,w^2,(exp(-2*(w-1)^2)-exp(-2))/(1-exp(-2))]',
            aggregation='basis_k=positive_weight_k/sum_positive_weight * response(weight_k); zero if no match',
            support_gate='has_match * neff/(neff+support_shrinkage) * sigmoid(independent channel MLP)',
            channel_formula='raw_increment_bound/2 * support_gate * tanh(bias_free_channel_readout)',
            mean_formula='ridge + inherited_bound*tanh(frozen_A_total_raw + sum_channel_increment)',
            penalty_reference='complete frozen A mean', output_bias=False,
            base_trainable=False, biological_chemistry_availability_gate=False,
            covariance_updated=False, parameter_sharing_between_channels=False,
            missing_policy='unknown distinct from known-no-overlap; unsupported rows exactly recover A',
            input_interface='same packed chemistry, target, MoA and separate masks as original A')
        if aggregation_scaling == 'train_fixed':
            self.config.update(schema_version=2, aggregation_scaling=aggregation_scaling,
                scale_max_gain=self.scale_max_gain, scale_floor=1./self.scale_max_gain,
                aggregation_scale_estimator='sqrt(mean_supported_TRAIN(sum_anchor((sum_response(raw_basis/3))^2)))',
                aggregation_scale_application='one fixed positive gain per channel, before local coefficients and readout; no centering',
                aggregation_scale_zero_policy='all-zero or unsupported TRAIN channel: count=0, raw_s=0, gain=1, capped=false')

    @property
    def bank(self):
        return self.base_a.bank

    def train(self, mode=True):
        super().train(mode)
        self.base_a.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def aggregation_scale_metadata(self):
        """Serializable input-only calibration provenance and fixed values."""
        if self.aggregation_scaling == 'none':
            return dict(aggregation_scaling='none', fitted=False, fit_ids=[],
                        count=[0, 0], raw_s=[0., 0.], gain=[1., 1.], capped=[False, False])
        return dict(aggregation_scaling=self.aggregation_scaling,
            fitted=bool(self.aggregation_scale_fitted),
            fit_ids=list(self.config.get('aggregation_scale_fit_ids', [])),
            channels=list(self.config['channels']), scale_max_gain=self.scale_max_gain,
            scale_floor=1./self.scale_max_gain,
            count=self.aggregation_scale_count.tolist(), raw_s=self.aggregation_scale_raw_s.tolist(),
            gain=self.aggregation_scale_gain.tolist(), capped=self.aggregation_scale_capped.tolist())

    @torch.no_grad()
    def fit_aggregation_scale(self, x_fit, packed_fit, mask_fit, *, ids):
        """Fit once from supported TRAIN query inputs, before optimization.

        The raw initial vector is ``sum_response(raw_basis / 3)`` at each
        anchor. Its supported-row RMS vector norm gives one channel scale.
        Multiplication by ``min(1 / scale, scale_max_gain)`` preserves zeros
        and relative query/anchor amplitudes. No response labels are accepted.
        """
        if self.aggregation_scaling != 'train_fixed':
            raise ValueError('Scale fitting requires aggregation_scaling=train_fixed')
        if bool(self.aggregation_scale_fitted):
            raise RuntimeError('The TRAIN aggregation scale can only be fitted once')
        for channel in self.channels:
            initial = torch.full_like(channel.local_coefficients, self._initial_local_coefficient)
            if (torch.count_nonzero(channel.output.weight)
                    or not torch.equal(channel.local_coefficients, initial)
                    or any(parameter.grad is not None for parameter in channel.parameters())):
                raise RuntimeError('Fit aggregation scale before training, at zero readout and initial local coefficients')
        names = list(map(str, ids))
        allowed = set(self.bank.descriptor_bank.config['anchor_data']['train_ids'])
        if not names or len(set(names)) != len(names) or not set(names).issubset(allowed):
            raise ValueError('Scale-fitting identities must be unique members of the declared TRAIN pool')
        ref = self.base_a.base_hr.coefficient
        x = torch.as_tensor(x_fit, dtype=ref.dtype, device=ref.device)
        packed = torch.as_tensor(packed_fit, dtype=ref.dtype, device=ref.device)
        mask = torch.as_tensor(mask_fit, device=ref.device)
        if x.ndim != 2 or len(names) != len(x):
            raise ValueError('Scale-fitting identities must align with the supplied TRAIN queries')
        chem, clean = self._inputs(packed)
        # Validate full inputs even in biology mode, where relations alone do
        # not consume morphology or chemical availability.
        self.bank.descriptor_bank(x, chem, mask, ids=names)
        weights, known = self._relations(x, chem, mask, clean)
        support = self._support(weights, known)
        raw_basis = self.response_functions(weights)*support['normalized_weights'][..., None]
        initial_local = (raw_basis*(1/3)).sum(-1)
        count = support['support'].sum(0)
        mean_square = initial_local.square().sum(-1).sum(0)/count.clamp_min(1)
        raw_s = mean_square.sqrt()
        if not torch.isfinite(raw_s).all():
            raise ValueError('Nonfinite TRAIN aggregation scale')
        positive = (count > 0) & (raw_s > 0)
        gain = torch.where(positive, raw_s.clamp_min(1./self.scale_max_gain).reciprocal(),
                           torch.ones_like(raw_s))
        self.aggregation_scale_count.copy_(torch.where(positive, count, torch.zeros_like(count)))
        self.aggregation_scale_raw_s.copy_(raw_s)
        self.aggregation_scale_gain.copy_(gain)
        self.aggregation_scale_capped.copy_(positive & (raw_s < 1./self.scale_max_gain))
        self.config['aggregation_scale_fit_ids'] = names
        self.aggregation_scale_fitted.fill_(True)
        return self.aggregation_scale_metadata()

    def _inputs(self, packed, bio=None):
        if bio is None:
            return self.bank.unpack_information(packed)
        return packed, self.bank._validate_bio(bio, len(packed))

    @torch.no_grad()
    def baseline_mean(self, x, packed, mask, bio=None):
        chem, clean = self._inputs(packed, bio)
        return self.base_a._details(x, chem, mask, bio=clean)['mean']

    @staticmethod
    def response_functions(weights):
        if (not torch.isfinite(weights).all() or torch.any(weights < 0)
                or torch.any(weights > 1)):
            raise ValueError('Relation weights must be finite in [0,1]')
        anchor = weights.new_tensor(math.exp(-2.))
        rbf = (torch.exp(-2.*(weights-1.).square())-anchor)/(1.-anchor)
        # At zero use exact zero, independent of exp-library rounding.
        rbf = torch.where(weights > 0, rbf.clamp(0., 1.), torch.zeros_like(rbf))
        return torch.stack((weights, weights.square(), rbf), -1)

    def _relations(self, x, chem, mask, bio):
        if self.mode == 'biology':
            values, _ = self.bank.biological_descriptors(bio)
            weights = values[..., :2].transpose(1, 2)
            known = torch.stack((bio['target_mask'], bio['moa_mask']), -1)
        else:
            descriptors = self.bank.descriptor_bank
            description = descriptors(x, chem, mask)
            norm = torch.linalg.vector_norm(x[:, :-1], dim=-1)
            direction = x[:, :-1]/torch.where(norm > 0, norm, torch.ones_like(norm))[:, None]
            morphology = (direction @ descriptors.anchor_directions.T).clamp(0., 1.)
            weights = torch.stack((description['tanimoto'], morphology), 1)
            known = torch.stack((description['availability'], norm > 0), -1)
        weights = torch.where(known[:, :, None], weights, torch.zeros_like(weights))
        return weights, known

    def _support(self, weights, known):
        total = weights.sum(-1)
        squared = weights.square().sum(-1)
        support = total > 0
        effective = total.square()/torch.where(squared > 0, squared, torch.ones_like(squared))
        count = (weights > 0).sum(-1)
        k = weights.shape[-1]
        features = torch.stack((known.to(weights.dtype), support.to(weights.dtype),
            total/k, weights.max(-1).values, effective/k, count.to(weights.dtype)/k), -1)
        normalized = weights/torch.where(support, total, torch.ones_like(total))[:, :, None]
        return dict(support=support, channel_support=support, known=known, effective_neighbors=effective,
                    positive_neighbors=count, relation_mass=total,
                    support_features=features, normalized_weights=normalized)

    def _details(self, x, packed, mask, ids=None, bio=None, branch_enabled=True):
        if not isinstance(branch_enabled, bool):
            raise ValueError('branch_enabled is an explicit boolean ablation switch')
        if (branch_enabled and self.aggregation_scaling == 'train_fixed'
                and not bool(self.aggregation_scale_fitted)):
            raise RuntimeError('Fit the TRAIN aggregation scale before enabling the branch')
        chem, clean = self._inputs(packed, bio)
        # The whole fitted A, including its bank, is immutable and has no
        # training-mode state updates or gradient connection to new channels.
        with torch.no_grad():
            base = self.base_a._details(x, chem, mask, ids=ids, bio=clean)
            weights, known = self._relations(x, chem, mask, clean)
            support = self._support(weights, known)
            raw_bases = self.response_functions(weights)*support['normalized_weights'][..., None]
            bases = (raw_bases*self.aggregation_scale_gain[None, :, None, None]
                     if self.aggregation_scaling == 'train_fixed' else raw_bases)
        raw_outputs, gates, logits, local_values, contributions = [], [], [], [], []
        raw_local_values = []
        for index, channel in enumerate(self.channels):
            local = (bases[:, index]*channel.local_coefficients[None]).sum(-1)
            raw_local = ((raw_bases[:, index]*channel.local_coefficients[None]).sum(-1)
                         if self.aggregation_scaling == 'train_fixed' else local)
            raw = channel.output(local)
            logit = channel.gate(support['support_features'][:, index]).squeeze(-1)
            neff = support['effective_neighbors'][:, index]
            gate = support['support'][:, index].to(x.dtype)*(neff/(neff+self.support_shrinkage))*torch.sigmoid(logit)
            contribution = self.raw_increment_bound/2*gate[:, None]*torch.tanh(raw)
            raw_outputs.append(raw); gates.append(gate); logits.append(logit)
            local_values.append(local); contributions.append(contribution)
            raw_local_values.append(raw_local)
        blocks = torch.stack(contributions, 1)
        if not branch_enabled:
            blocks = blocks*0.
        raw_increment = blocks.sum(1)
        total = base['total_raw']+raw_increment
        proposed = base['ridge_mean']+self.base_a.base_hr.correction_bound*torch.tanh(total)
        active = support['support'].any(-1) & branch_enabled
        mean = torch.where(active[:, None], proposed, base['mean'])
        if not torch.isfinite(mean).all() or not torch.isfinite(raw_increment).all():
            raise FloatingPointError('Independent relation correction became nonfinite')
        return dict(mean=mean, baseline_mean=base['mean'], baseline_A_mean=base['mean'],
            base_hr_mean=base['base_hr_mean'], ridge_mean=base['ridge_mean'],
            increment=mean-base['mean'], total_raw=total, frozen_A_total_raw=base['total_raw'],
            hr_raw=base['hr_raw'], raw_increment=raw_increment, kernel_raw=raw_increment,
            block_contributions=blocks, block_readout_contributions=blocks,
            channel_readout=torch.stack(raw_outputs, 1), channel_gate=torch.stack(gates, 1),
            gate=torch.stack(gates, 1), condition_logits=torch.stack(logits, 1),
            local_activation=torch.stack(local_values, 1), local_basis=bases,
            raw_local_activation=torch.stack(raw_local_values, 1),
            raw_local_basis=raw_bases, scaled_local_basis=bases,
            relation_weights=weights, branch_active=active,
            basis_values=bases.flatten(1), **support)

    def forward(self, x, packed, mask, bio=None, branch_enabled=True):
        if not isinstance(branch_enabled, bool):
            raise ValueError('branch_enabled is an explicit boolean ablation switch')
        if not branch_enabled:
            return self.baseline_mean(x, packed, mask, bio=bio)
        return self._details(x, packed, mask, bio=bio, branch_enabled=branch_enabled)['mean']

    def loss(self, x, packed, mask, target, bio=None):
        details = self._details(x, packed, mask, bio=bio)
        mean = details['mean']
        if (not isinstance(target, torch.Tensor) or target.shape != mean.shape
                or target.dtype != mean.dtype or target.device != mean.device
                or not torch.isfinite(target).all()):
            raise ValueError('Target must be finite and aligned to the standardized nine-coordinate mean')
        mse, increment = F.mse_loss(mean, target), details['increment'].square().mean()
        penalty = self.incremental_penalty_weight*increment
        return dict(loss=mse+penalty, mean_mse=mse, incremental_mse=increment, incremental_penalty=penalty)

    @torch.no_grad()
    def diagnostics(self, x, packed, mask, ids=None, bio=None, branch_enabled=True):
        details = self._details(x, packed, mask, ids=ids, bio=bio, branch_enabled=branch_enabled)
        count = details['support'].sum(0).clamp_min(1)
        def supported_vector_rms(value):
            energy = value.flatten(2).square().sum(-1)
            return ((energy*details['support']).sum(0)/count).sqrt()
        details.update(saturation_fraction=(torch.tanh(details['total_raw']).abs() >= .95).to(x.dtype).mean(),
            inherited_hr_saturation_fraction=(torch.tanh(details['hr_raw']).abs() >= .95).to(x.dtype).mean(),
            incremental_mse=details['increment'].square().mean(),
            total_correction_max=(details['mean']-details['ridge_mean']).abs().max(),
            raw_increment_max=details['raw_increment'].abs().max(),
            raw_basis_rms=supported_vector_rms(details['raw_local_basis']),
            scaled_basis_rms=supported_vector_rms(details['scaled_local_basis']),
            raw_local_rms=supported_vector_rms(details['raw_local_activation']),
            scaled_local_rms=supported_vector_rms(details['local_activation']),
            raw_initial_local_rms=supported_vector_rms((details['raw_local_basis']*(1/3)).sum(-1)),
            scaled_initial_local_rms=supported_vector_rms((details['scaled_local_basis']*(1/3)).sum(-1)),
            aggregation_scale=self.aggregation_scale_metadata())
        return details

    @classmethod
    def from_config(cls, config):
        config = deepcopy(dict(config))
        if (config.get('schema_version') not in (1, 2) or config.get('model_type') != 'independent_biology_kernel'
                or config.get('dtype') not in ('float32', 'float64')):
            raise ValueError('Unsupported independent biological-kernel configuration')
        scaling = {}
        if config['schema_version'] == 2:
            if config.get('aggregation_scaling') != 'train_fixed':
                raise ValueError('Schema 2 requires fixed TRAIN aggregation scaling')
            scaling = dict(aggregation_scaling=config['aggregation_scaling'], scale_max_gain=config['scale_max_gain'])
        elif config.get('aggregation_scaling', 'none') != 'none':
            raise ValueError('Schema 1 does not contain TRAIN aggregation scale buffers')
        base = MechanismResponseKernelMean.from_config(config['base_a_config'])
        model = cls(base, **{key: config[key] for key in ('mode', 'incremental_penalty',
            'hidden_dim', 'support_shrinkage', 'raw_increment_bound')}, **scaling).to(dtype=getattr(torch, config['dtype']))
        for key in ('channels', 'fixed_reference_count', 'support_features', 'response_functions',
                    'aggregation', 'support_gate', 'channel_formula', 'mean_formula', 'penalty_reference',
                    'output_bias', 'base_trainable', 'biological_chemistry_availability_gate',
                    'covariance_updated', 'parameter_sharing_between_channels', 'missing_policy', 'input_interface'):
            if config.get(key) != model.config[key]:
                raise ValueError('Saved independent relation architecture changed: '+key)
        if config['schema_version'] == 2:
            for key in ('scale_floor', 'aggregation_scale_estimator', 'aggregation_scale_application',
                        'aggregation_scale_zero_policy'):
                if config.get(key) != model.config[key]:
                    raise ValueError('Saved aggregation scale definition changed: '+key)
        model.config = config
        return model

    def save(self, path):
        config = deepcopy(self.config)
        config['dtype'] = str(self.base_a.base_hr.coefficient.dtype).removeprefix('torch.')
        torch.save(dict(config=config, state_dict=self.state_dict()), Path(path))

    @classmethod
    def load(cls, path):
        saved = torch.load(Path(path), map_location='cpu', weights_only=True)
        model = cls.from_config(saved['config'])
        model.load_state_dict(saved['state_dict'])
        return model.eval()
