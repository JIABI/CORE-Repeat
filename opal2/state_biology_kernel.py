"""Matched state-conditioned relation kernels around the complete frozen A.

The new branch uses only first-well state in ``state_only`` and target/MoA
relations modulated by that same state in ``state_biology``. Each channel has
an independent signed state interaction with its anchor readout. All state
preprocessing and aggregation gains are fixed from the original common FIT.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn

from .hierarchical_geometry import _positive_integer
from .independent_biology_kernel import IndependentBiologyKernelMean, SUPPORT_FEATURES
from .mechanism_response_kernel import MechanismResponseKernelMean


MODES = ('state_only', 'state_biology')
PCA_COMPONENTS = 8
STATE_DIM = 9
STATE_LATENT_DIM = 4
STATE_SCALE_FLOOR = 1e-6
STATE_FEATURES = tuple(f'PCA_{index+1}' for index in range(PCA_COMPONENTS)) + ('log_norm',)


class _StateRelationChannel(nn.Module):
    def __init__(self, anchors, hidden_dim):
        super().__init__()
        self.local_coefficients = nn.Parameter(torch.full((anchors, 3), 1/3))
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, STATE_LATENT_DIM), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(len(SUPPORT_FEATURES)+STATE_DIM, hidden_dim),
                                  nn.GELU(), nn.Linear(hidden_dim, 1))
        self.output = nn.Linear(anchors, (1+STATE_LATENT_DIM)*9, bias=False)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.output.weight)


class StateBiologyKernelMean(IndependentBiologyKernelMean):
    """Two independently state-modulated, support-aware bounded channels.

    Inherits the complete-A baseline, loss, TRAIN aggregation estimator and
    save/load interface. Initialization creates only this model's active
    parameters, with identical draws and capacity in both relation modes.
    """
    def __init__(self, base_a, mode='state_biology', incremental_penalty=.1,
                 hidden_dim=16, support_shrinkage=2., raw_increment_bound=1.,
                 aggregation_scaling='train_fixed', scale_max_gain=32.):
        nn.Module.__init__(self)
        if not isinstance(base_a, MechanismResponseKernelMean) or base_a.mode != 'old_generic':
            raise TypeError('Provide the complete fitted old_generic A model')
        if mode not in MODES:
            raise ValueError('State mode must be state_only or state_biology')
        if aggregation_scaling != 'train_fixed':
            raise ValueError('State kernels require fixed TRAIN aggregation scaling')
        if not math.isfinite(scale_max_gain) or scale_max_gain < 1.:
            raise ValueError('scale_max_gain must be finite and at least one')
        for name, value, positive in (
            ('incremental_penalty', incremental_penalty, False),
            ('support_shrinkage', support_shrinkage, True),
            ('raw_increment_bound', raw_increment_bound, True)):
            if not math.isfinite(value) or (value <= 0 if positive else value < 0):
                raise ValueError(name+' must be finite and '+('positive' if positive else 'nonnegative'))
        hidden_dim = _positive_integer(hidden_dim, 'hidden_dim')
        if base_a.bank.anchor_count < 1 or base_a.bank.input_dim-1 < PCA_COMPONENTS:
            raise ValueError('State kernels require fixed TRAIN references and at least eight first-well features')
        self.base_a = deepcopy(base_a).eval().requires_grad_(False)
        self.mode = mode
        self.incremental_penalty_weight = float(incremental_penalty)
        self.support_shrinkage = float(support_shrinkage)
        self.raw_increment_bound = float(raw_increment_bound)
        self.aggregation_scaling = aggregation_scaling
        self.scale_max_gain = float(scale_max_gain)
        self.channels = nn.ModuleList([_StateRelationChannel(self.bank.anchor_count, hidden_dim) for _ in range(2)])
        ref = self.base_a.base_hr.coefficient
        self.to(dtype=ref.dtype, device=ref.device)
        self._initial_local_coefficient = float(self.channels[0].local_coefficients[0, 0].detach())
        for name, value in (
            ('aggregation_scale_gain', ref.new_ones(2)),
            ('aggregation_scale_raw_s', ref.new_zeros(2)),
            ('aggregation_scale_count', torch.zeros(2, dtype=torch.long, device=ref.device)),
            ('aggregation_scale_capped', torch.zeros(2, dtype=torch.bool, device=ref.device)),
            ('aggregation_scale_fitted', torch.zeros((), dtype=torch.bool, device=ref.device)),
            ('state_feature_center', ref.new_zeros(self.bank.input_dim-1)),
            ('state_pca_components', ref.new_zeros(PCA_COMPONENTS, self.bank.input_dim-1)),
            ('state_score_center', ref.new_zeros(PCA_COMPONENTS)),
            ('state_score_scale', ref.new_ones(PCA_COMPONENTS)),
            ('state_explained_variance_ratio', ref.new_zeros(PCA_COMPONENTS)),
            ('state_log_norm_center', ref.new_zeros(())),
            ('state_log_norm_scale', ref.new_ones(())),
            ('state_reference_log_norm', ref.new_zeros(self.bank.anchor_count)),
            ('state_fit_train_indices', torch.full((len(self.bank.config['fitting_ids']),), -1, dtype=torch.long, device=ref.device)),
            ('state_reference_train_indices', torch.full((self.bank.anchor_count,), -1, dtype=torch.long, device=ref.device)),
            ('input_state_fitted', torch.zeros((), dtype=torch.bool, device=ref.device))):
            self.register_buffer(name, value)
        self.config = dict(schema_version=1, model_type='state_biology_kernel',
            base_a_config=deepcopy(base_a.config), mode=mode,
            incremental_penalty=float(incremental_penalty), hidden_dim=hidden_dim,
            support_shrinkage=float(support_shrinkage), raw_increment_bound=float(raw_increment_bound),
            dtype=str(ref.dtype).removeprefix('torch.'),
            channels=['positive_morphology', 'log_norm_rbf'] if mode == 'state_only' else ['target', 'moa'],
            fixed_reference_count=self.bank.anchor_count, support_features=list(SUPPORT_FEATURES),
            state_features=list(STATE_FEATURES), pca_components=PCA_COMPONENTS,
            state_dim=STATE_DIM, state_latent_dim=STATE_LATENT_DIM,
            state_estimator='NumPy full SVD on centered commonbranchFIT x[:,:-1]; each component largest-absolute loading made positive',
            state_normalization='commonbranchFIT population mean/std for eight PCA scores and x[:,-1]; scales below 1e-6 replaced by one',
            state_scale_floor=STATE_SCALE_FLOOR,
            state_scope='exact original commonbranchFIT; references disjoint; no validation/test or response labels',
            state_only_relations='positive original anchor-direction cosine; exp(-0.5*(FIT-standardized query log_norm - reference log_norm)^2)',
            state_encoder='independent Linear(9,4) then tanh per channel',
            state_readout='bias-free Linear(K,45) reshaped to [5,9], contracted with [1,tanh(state_encoder_linear(state))]',
            support_gate='has_match * neff/(neff+support_shrinkage) * sigmoid(independent MLP(support6,state9))',
            response_functions='[w,w^2,(exp(-2*(w-1)^2)-exp(-2))/(1-exp(-2))]',
            aggregation='basis_k=positive_weight_k/sum_positive_weight * response(weight_k); zero if no match',
            aggregation_scaling=aggregation_scaling, scale_max_gain=self.scale_max_gain, scale_floor=1./self.scale_max_gain,
            aggregation_scale_estimator='sqrt(mean_supported_TRAIN(sum_anchor((sum_response(raw_basis/3))^2)))',
            aggregation_scale_application='one fixed positive gain per channel, before local coefficients and readout; no centering',
            aggregation_scale_zero_policy='all-zero or unsupported TRAIN channel: count=0, raw_s=0, gain=1, capped=false',
            channel_formula='raw_increment_bound/2 * support_gate * tanh(state_contracted_readout)',
            mean_formula='ridge + inherited_bound*tanh(frozen_A_total_raw + sum_channel_increment)',
            penalty_reference='complete frozen A mean', output_bias=False, base_trainable=False,
            covariance_updated=False, parameter_sharing_between_channels=False,
            new_branch_uses_chemical_features=False, biological_chemistry_availability_gate=False,
            reference_response_borrowing=False, jepa_active=False,
            missing_policy='unknown distinct from known-no-overlap; unsupported rows exactly recover A',
            input_interface='shared packed decision inputs; state_only reads only chemistry for frozen A and X for the new branch')

    def _check_initial_branch(self):
        for channel in self.channels:
            if (torch.count_nonzero(channel.output.weight)
                    or not torch.equal(channel.local_coefficients,
                                       torch.full_like(channel.local_coefficients, self._initial_local_coefficient))
                    or any(parameter.grad is not None for parameter in channel.parameters())):
                raise RuntimeError('Fit fixed input state before training, at zero readout and initial local coefficients')

    def input_state_metadata(self):
        return dict(fitted=bool(self.input_state_fitted), fit_ids=list(self.config.get('state_fit_ids', [])),
            reference_ids=list(self.config.get('state_reference_ids', [])),
            estimator=self.config['state_estimator'], normalization=self.config['state_normalization'],
            pca_components=PCA_COMPONENTS, state_dim=STATE_DIM, state_latent_dim=STATE_LATENT_DIM,
            explained_variance_ratio=self.state_explained_variance_ratio.tolist(),
            explained_variance_ratio_sum=float(self.state_explained_variance_ratio.sum()),
            score_scale=self.state_score_scale.tolist(), log_norm_center=float(self.state_log_norm_center),
            log_norm_scale=float(self.state_log_norm_scale), labels_used=False, torch_rng_consumed=False)

    @torch.no_grad()
    def fit_input_state(self, x_fit, ids, reference_x, reference_ids):
        if bool(self.input_state_fitted):
            raise RuntimeError('The fixed TRAIN input state can only be fitted once')
        self._check_initial_branch()
        names, references = list(map(str, ids)), list(map(str, reference_ids))
        expected = list(self.bank.config['fitting_ids'])
        anchors = list(self.bank.descriptor_bank.config['anchor_data']['anchor_ids'])
        train_ids = list(self.bank.descriptor_bank.config['anchor_data']['train_ids'])
        if (len(names) != len(expected) or len(set(names)) != len(names) or set(names) != set(expected)
                or not set(names).issubset(train_ids) or set(names) & set(anchors)):
            raise ValueError('State fitting requires exactly the commonbranch FIT identities, disjoint from TRAIN references')
        if references != anchors or not set(references).issubset(train_ids):
            raise ValueError('Reference identities must be the original TRAIN anchor IDs in their saved order')
        ref = self.state_feature_center
        x = torch.as_tensor(x_fit, dtype=ref.dtype, device=ref.device)
        ax = torch.as_tensor(reference_x, dtype=ref.dtype, device=ref.device)
        if (x.shape != (len(names), self.bank.input_dim) or ax.shape != (len(anchors), self.bank.input_dim)
                or len(x) <= PCA_COMPONENTS or not torch.isfinite(x).all() or not torch.isfinite(ax).all()):
            raise ValueError('Finite aligned full X and reference X with at least nine fitting rows are required')
        norms = torch.linalg.vector_norm(ax[:, :-1], dim=-1)
        directions = ax[:, :-1]/torch.where(norms > 0, norms, torch.ones_like(norms))[:, None]
        if not torch.allclose(directions, self.bank.descriptor_bank.anchor_directions, rtol=1e-10, atol=1e-12):
            raise ValueError('Reference X does not reproduce the saved anchor directions')
        descriptor = self.bank.descriptor_bank
        anchor_raw = descriptor.generic_centers*descriptor.descriptor_scale+descriptor.descriptor_center
        if not torch.allclose(ax[:, -1], anchor_raw[:, -3], rtol=1e-10, atol=1e-12):
            raise ValueError('Reference log-norm values do not reproduce the saved TRAIN anchors')
        values = x.detach().cpu().double().numpy()
        center = values[:, :-1].mean(0)
        centered = values[:, :-1]-center
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        components = right[:PCA_COMPONENTS].copy()
        pivots = np.argmax(np.abs(components), axis=1)
        signs = np.where(components[np.arange(PCA_COMPONENTS), pivots] < 0, -1., 1.)
        components *= signs[:, None]
        scores = centered@components.T
        score_center, score_scale = scores.mean(0), scores.std(0)
        score_scale = np.where(score_scale < STATE_SCALE_FLOOR, 1., score_scale)
        norm_center, norm_scale = values[:, -1].mean(), values[:, -1].std()
        norm_scale = 1. if norm_scale < STATE_SCALE_FLOOR else norm_scale
        total_variance = np.square(singular).sum()
        variance_ratio = np.square(singular[:PCA_COMPONENTS])/total_variance if total_variance > 0 else np.zeros(PCA_COMPONENTS)
        computed = dict(state_feature_center=center, state_pca_components=components,
            state_score_center=score_center, state_score_scale=score_scale,
            state_explained_variance_ratio=variance_ratio,
            state_log_norm_center=norm_center, state_log_norm_scale=norm_scale)
        if any(not np.isfinite(value).all() for value in computed.values()):
            raise ValueError('Nonfinite TRAIN input-state preprocessing')
        for key, value in computed.items():
            buffer = getattr(self, key)
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device))
        self.state_reference_log_norm.copy_((ax[:, -1]-self.state_log_norm_center)/self.state_log_norm_scale)
        lookup = {unit: index for index, unit in enumerate(train_ids)}
        self.state_fit_train_indices.copy_(torch.tensor([lookup[unit] for unit in names], device=ref.device))
        self.state_reference_train_indices.copy_(torch.tensor([lookup[unit] for unit in references], device=ref.device))
        self.config.update(state_fit_ids=names, state_reference_ids=references)
        self.input_state_fitted.fill_(True)
        return self.input_state_metadata()

    @torch.no_grad()
    def input_state(self, x):
        if not bool(self.input_state_fitted):
            raise RuntimeError('Fit the TRAIN input state before enabling the branch')
        self.base_a.base_hr._check_inputs(x)
        scores = (x[:, :-1]-self.state_feature_center)@self.state_pca_components.T
        scores = (scores-self.state_score_center)/self.state_score_scale
        log_norm = (x[:, -1]-self.state_log_norm_center)/self.state_log_norm_scale
        state = torch.cat((scores, log_norm[:, None]), -1)
        if not torch.isfinite(state).all():
            raise FloatingPointError('Fixed input state became nonfinite')
        return state

    def fit_aggregation_scale(self, x_fit, packed_fit, mask_fit, *, ids):
        if not bool(self.input_state_fitted):
            raise RuntimeError('Fit the TRAIN input state before the aggregation scale')
        if list(map(str, ids)) != self.config['state_fit_ids']:
            raise ValueError('Aggregation scale must use the identical ordered input-state FIT queries')
        return super().fit_aggregation_scale(x_fit, packed_fit, mask_fit, ids=ids)

    def _inputs(self, packed, bio=None):
        if self.mode != 'state_only':
            return super()._inputs(packed, bio)
        total = self.bank.chemical_dim+self.bank.target_dim+self.bank.moa_dim+2
        if (not isinstance(packed, torch.Tensor) or packed.ndim != 2
                or packed.shape[1] not in (self.bank.chemical_dim, total)
                or packed.dtype != self.state_feature_center.dtype or packed.device != self.state_feature_center.device):
            raise ValueError('State-only input must contain the original chemistry with matching dtype/device')
        # old_generic A consumes chemistry, not biology. Passing an explicit
        # empty biological mapping also prevents its packed-input unpacker.
        return packed[:, :self.bank.chemical_dim], {}

    def _relations(self, x, chem, mask, bio):
        if self.mode == 'state_biology':
            values, _ = self.bank.biological_descriptors(bio)
            weights = values[..., :2].transpose(1, 2)
            known = torch.stack((bio['target_mask'], bio['moa_mask']), -1)
        else:
            state = self.input_state(x)
            norm = torch.linalg.vector_norm(x[:, :-1], dim=-1)
            direction = x[:, :-1]/torch.where(norm > 0, norm, torch.ones_like(norm))[:, None]
            morphology = (direction@self.bank.descriptor_bank.anchor_directions.T).clamp(0., 1.)
            norm_relation = torch.exp(-.5*(state[:, -1, None]-self.state_reference_log_norm[None]).square())
            weights = torch.stack((morphology, norm_relation), 1)
            known = torch.stack((norm > 0, torch.ones_like(norm, dtype=torch.bool)), -1)
        return torch.where(known[:, :, None], weights, torch.zeros_like(weights)), known

    def _details(self, x, packed, mask, ids=None, bio=None, branch_enabled=True):
        if not isinstance(branch_enabled, bool):
            raise ValueError('branch_enabled is an explicit boolean ablation switch')
        if branch_enabled and not bool(self.input_state_fitted):
            raise RuntimeError('Fit the TRAIN input state before enabling the branch')
        if branch_enabled and not bool(self.aggregation_scale_fitted):
            raise RuntimeError('Fit the TRAIN aggregation scale before enabling the branch')
        chem, clean = self._inputs(packed, bio)
        with torch.no_grad():
            base = self.base_a._details(x, chem, mask, ids=ids, bio=clean)
            if bool(self.input_state_fitted):
                state = self.input_state(x)
                weights, known = self._relations(x, chem, mask, clean)
            else:
                state = x.new_zeros(len(x), STATE_DIM)
                weights = x.new_zeros(len(x), 2, self.bank.anchor_count)
                known = torch.zeros((len(x), 2), dtype=torch.bool, device=x.device)
            support = self._support(weights, known)
            raw_bases = self.response_functions(weights)*support['normalized_weights'][..., None]
            bases = raw_bases*self.aggregation_scale_gain[None, :, None, None]
        raw_outputs, gates, logits, local_values, raw_local_values, contributions = [], [], [], [], [], []
        latent_values, modulation_values, readout_blocks, readout_terms = [], [], [], []
        for index, channel in enumerate(self.channels):
            local = (bases[:, index]*channel.local_coefficients[None]).sum(-1)
            raw_local = (raw_bases[:, index]*channel.local_coefficients[None]).sum(-1)
            latent = channel.state_encoder(state)
            modulation = torch.cat((torch.ones_like(latent[:, :1]), latent), -1)
            readouts = channel.output(local).reshape(len(x), 1+STATE_LATENT_DIM, 9)
            terms = readouts*modulation[..., None]
            raw = terms.sum(1)
            gate_features = torch.cat((support['support_features'][:, index], state), -1)
            logit = channel.gate(gate_features).squeeze(-1)
            neff = support['effective_neighbors'][:, index]
            gate = support['support'][:, index].to(x.dtype)*(neff/(neff+self.support_shrinkage))*torch.sigmoid(logit)
            contribution = self.raw_increment_bound/2*gate[:, None]*torch.tanh(raw)
            local_values.append(local); raw_local_values.append(raw_local)
            raw_outputs.append(raw); gates.append(gate); logits.append(logit); contributions.append(contribution)
            latent_values.append(latent); modulation_values.append(modulation)
            readout_blocks.append(readouts); readout_terms.append(terms)
        blocks = torch.stack(contributions, 1)
        if not branch_enabled:
            blocks = blocks*0.
        raw_increment = blocks.sum(1)
        total = base['total_raw']+raw_increment
        proposed = base['ridge_mean']+self.base_a.base_hr.correction_bound*torch.tanh(total)
        active = support['support'].any(-1) & branch_enabled
        mean = torch.where(active[:, None], proposed, base['mean'])
        if not torch.isfinite(mean).all() or not torch.isfinite(raw_increment).all():
            raise FloatingPointError('State-conditioned relation correction became nonfinite')
        return dict(mean=mean, baseline_mean=base['mean'], baseline_A_mean=base['mean'],
            base_hr_mean=base['base_hr_mean'], ridge_mean=base['ridge_mean'],
            increment=mean-base['mean'], total_raw=total, frozen_A_total_raw=base['total_raw'],
            hr_raw=base['hr_raw'], raw_increment=raw_increment, kernel_raw=raw_increment,
            block_contributions=blocks, block_readout_contributions=blocks,
            channel_readout=torch.stack(raw_outputs, 1), channel_gate=torch.stack(gates, 1),
            gate=torch.stack(gates, 1), condition_logits=torch.stack(logits, 1),
            local_activation=torch.stack(local_values, 1), raw_local_activation=torch.stack(raw_local_values, 1),
            local_basis=bases, raw_local_basis=raw_bases, scaled_local_basis=bases,
            relation_weights=weights, branch_active=active, basis_values=bases.flatten(1),
            input_state=state, state_latent=torch.stack(latent_values, 1),
            state_modulation=torch.stack(modulation_values, 1),
            state_readout_blocks=torch.stack(readout_blocks, 1),
            state_readout_contributions=torch.stack(readout_terms, 1), **support)

    @torch.no_grad()
    def diagnostics(self, *args, **kwargs):
        result = super().diagnostics(*args, **kwargs)
        result['input_state_metadata'] = self.input_state_metadata()
        return result

    @classmethod
    def from_config(cls, config):
        config = deepcopy(dict(config))
        if (config.get('schema_version') != 1 or config.get('model_type') != 'state_biology_kernel'
                or config.get('dtype') not in ('float32', 'float64')):
            raise ValueError('Unsupported state-conditioned kernel configuration')
        base = MechanismResponseKernelMean.from_config(config['base_a_config'])
        keys = ('mode', 'incremental_penalty', 'hidden_dim', 'support_shrinkage', 'raw_increment_bound',
                'aggregation_scaling', 'scale_max_gain')
        model = cls(base, **{key: config[key] for key in keys}).to(dtype=getattr(torch, config['dtype']))
        for key, expected in model.config.items():
            if key not in ('base_a_config', 'dtype') and config.get(key) != expected:
                raise ValueError('Saved state-conditioned architecture changed: '+key)
        model.config = config
        return model
